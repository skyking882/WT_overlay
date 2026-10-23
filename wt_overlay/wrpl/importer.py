"""One .wrpl import: replay -> backend subprocess -> match_export.json -> scripts/wt_replay_import.py ->
outputs/engagements/wt_real/<name>.jsonl (docs/wrpl_import_spec.md §1.2, §5, §6), and the sandbox's import queue.

The converter runs as its own process with its normal command line, so its output is exactly what
``scripts/wt_replay_import.py EXPORT --name NAME`` writes; the importer then adds the replay's game version to the
header's ``source`` (``wt_version``) and moves the file into place. Imports are recognised by the replay's SHA-256,
which every converted file records in ``source.replay_sha256``: a replay already in the output folder is not imported
again unless forced.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time

from . import backend as backends
from . import gamefiles

ROOT = Path(__file__).resolve().parents[2]
CONVERTER = ROOT/"scripts"/"wt_replay_import.py"
OUTPUTS = ROOT/"outputs"
DEFAULT_OUT = OUTPUTS/"engagements"/"wt_real"
SETTINGS = OUTPUTS/"wrpl_settings.json"
WORK = OUTPUTS/"wrpl_work"
INBOX = OUTPUTS/"wrpl_inbox"
KEPT_EXPORTS = OUTPUTS/"wt_replay_exports"
BACKEND_TIMEOUT_S = 20*60        # D: the 2026-10-08 replays take ~35 s on the Mac x86 backend
CONVERT_TIMEOUT_S = 15*60
MAX_UPLOAD = 1 << 30             # D: 1 GiB; client replays are 5-40 MB
LOG_LINES = 12


class ImportFailed(RuntimeError):
    """A failed step; ``log`` holds the backend / converter output tail."""
    def __init__(self, message, log=()):
        super().__init__(message)
        self.log = list(log)


class UploadTooLarge(ValueError):
    pass


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def safe_name(stem):
    """Output name rule of scripts/wt_replay_import.py (replay_name): '#2026.10.08 10.47.02' -> '2026.10.08_10.47.02'."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_.") or "wt_replay"


def _header(path):
    try:
        with open(path, "rb") as handle:
            line = handle.readline()
        row = json.loads(line)
    except (OSError, ValueError):
        return None
    return row if isinstance(row, dict) and row.get("type") == "header" else None


_SCAN_CACHE = {}
_SCAN_LOCK = threading.Lock()


def imported(out_dir):
    """{replay sha256: [jsonl paths]} of the converted replays in ``out_dir`` (headers only, cached by size/mtime)."""
    found = {}
    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        return found
    with _SCAN_LOCK:
        for path in sorted(out_dir.glob("*.jsonl")):
            try:
                st = path.stat()
            except OSError:
                continue
            key = (str(path), st.st_size, st.st_mtime_ns)
            if key not in _SCAN_CACHE:
                head = _header(path)
                _SCAN_CACHE[key] = ((head or {}).get("source") or {}).get("replay_sha256")
            sha = _SCAN_CACHE[key]
            if sha:
                found.setdefault(sha, []).append(path)
    return found


def output_name(base, sha, out_dir):
    """``base``, or base_2, base_3, ... when another replay already has that name; the same replay keeps its name."""
    out_dir = Path(out_dir)
    for i in range(1, 10000):
        name = base if i == 1 else f"{base}_{i}"
        path = out_dir/f"{name}.jsonl"
        if not path.exists():
            return name
        if ((_header(path) or {}).get("source") or {}).get("replay_sha256") == sha:
            return name
    raise ImportFailed(f"输出名 {base} 已被占用")


def _duration(seconds):
    return f"{seconds/60:g} 分钟" if seconds >= 60 else f"{seconds:g} 秒"


def _tail(text, n=LOG_LINES):
    return [line for line in (text or "").splitlines() if line.strip()][-n:]


def _run(cmd, timeout, on_process=None, cwd=None, env=None):
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8",
                            errors="replace", cwd=cwd, env=env)
    if on_process:
        on_process(proc)
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()
        return None, out, err
    finally:
        if on_process:
            on_process(None)
    return proc.returncode, out, err


def run_backend(backend, game_dir, replay, out, *, timeout=BACKEND_TIMEOUT_S, on_process=None, version_hint=None):
    """Run ``wrpl_export``; returns its stdout summary (dict). ``out`` must not exist."""
    cmd = backend.command+["--game", str(game_dir), "--replay", str(replay), "--output", str(out)]
    pkg = gamefiles.local_package(game_dir)
    if pkg is not None:
        cmd += ["--local-package", str(pkg)]
    try:
        code, stdout, stderr = _run(cmd, timeout, on_process)
    except OSError as exc:
        if getattr(exc, "errno", None) == 86 or "Bad CPU type" in str(exc):
            raise ImportFailed(backends.ROSETTA_HINT) from None
        raise ImportFailed(f"无法运行解析后端：{exc}") from None
    log = _tail(stderr)
    parser_log = _tail(_read(Path(out)/"parser.log"), 6)
    log += [line for line in parser_log if line not in log]
    if code is None:
        raise ImportFailed(f"解析超时（{_duration(timeout)}）", log)
    if code != 0:
        hint = f"；{version_hint}" if version_hint else ""
        raise ImportFailed(f"解析失败（退出码 {code}）{hint}", log)
    if not (Path(out)/"match_export.json").is_file():
        raise ImportFailed("解析后端没有写出 match_export.json", log)
    try:
        return json.loads(stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {}


def _read(path):
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def convert(export_json, out_dir, name, *, team=(), all_units=False, timeout=CONVERT_TIMEOUT_S, on_process=None):
    """scripts/wt_replay_import.py EXPORT --out-dir OUT --name NAME [...]; returns the written .jsonl."""
    cmd = [sys.executable, str(CONVERTER), str(export_json), "--out-dir", str(out_dir), "--name", name]
    for spec in team:
        cmd += ["--team", spec]
    if all_units:
        cmd.append("--all-units")
    code, stdout, stderr = _run(cmd, timeout, on_process, cwd=str(ROOT))
    if code is None:
        raise ImportFailed(f"转换超时（{_duration(timeout)}）", _tail(stderr))
    if code != 0:
        raise ImportFailed(f"转换失败（退出码 {code}）", _tail(stderr) or _tail(stdout))
    out = Path(out_dir)/f"{name}.jsonl"
    if not out.is_file():
        raise ImportFailed("转换没有写出文件", _tail(stdout))
    return out


def place(converted, final, wt_version):
    """Copy ``converted`` to ``final`` (atomic within its folder) with source.wt_version added to the header."""
    final = Path(final)
    final.parent.mkdir(parents=True, exist_ok=True)
    part = final.with_name(f".{final.name}.{secrets.token_hex(4)}.part")
    try:
        with open(converted, "rb") as src, open(part, "wb") as dst:
            header = json.loads(src.readline())
            header.setdefault("source", {})["wt_version"] = wt_version
            dst.write((json.dumps(header, separators=(",", ":"), allow_nan=False)+"\n").encode("utf-8"))
            shutil.copyfileobj(src, dst, 1 << 20)
        os.replace(part, final)
    finally:
        if part.exists():
            part.unlink()
    return final


def import_replay(replay, *, game=None, out_dir=None, name=None, backend=None, keep_export=False, force=False,
                  team=(), all_units=False, ignore_version=False, settings=None, work_root=None, kept_root=None,
                  timeout_s=BACKEND_TIMEOUT_S, sha256=None, progress=None, on_process=None, cache=None, manifest=None):
    """Import one replay. Returns dict(status="imported" | "duplicate", path, file, name, sha256, ...).

    Raises gamefiles.GameFilesError (no / wrong game files), backend.BackendMissing (backend not downloaded yet),
    backend.BackendError, ImportFailed. ``backend`` is a Backend, a path, or None (WT_WRPL_BACKEND, cache).
    out_dir / settings / work_root / kept_root default to the outputs/ folders above; settings=False remembers
    nothing."""
    replay = Path(replay).resolve()
    out_dir = Path(out_dir or DEFAULT_OUT)
    settings = SETTINGS if settings is None else settings
    work_root, kept_root = Path(work_root or WORK), Path(kept_root or KEPT_EXPORTS)
    step = progress or (lambda phase: None)
    if not replay.is_file():
        raise ImportFailed(f"找不到回放：{replay}")
    if not gamefiles.is_replay(replay):
        raise ImportFailed(f"不是 War Thunder 回放（.wrpl）：{replay.name}")
    sha = sha256 or file_sha256(replay)
    if not force:
        existing = imported(out_dir).get(sha)
        if existing:
            return dict(status="duplicate", path=existing[0], file=f"wt_real/{existing[0].name}",
                        name=existing[0].stem, sha256=sha)
    remembered = gamefiles.load_settings(settings).get("game_dir") if settings else None
    game_dir, ginfo = gamefiles.find_game_dir(replay, explicit=game, remembered=remembered,
                                              check_version=not ignore_version)
    if not isinstance(backend, backends.Backend):     # after the replay and game checks: those come first
        backend = backends.find_backend(backend, cache=cache, manifest=manifest)
    base = safe_name(name) if name else safe_name(replay.stem)
    final_name = output_name(base, sha, out_dir)
    work = Path(work_root)/f"{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}"
    work.mkdir(parents=True)
    try:
        if keep_export:
            export_dir = Path(kept_root)/final_name
            i = 2
            while export_dir.exists():
                export_dir, i = Path(kept_root)/f"{final_name}_{i}", i+1
            export_dir.parent.mkdir(parents=True, exist_ok=True)
        else:
            export_dir = work/"export"
        hint = ginfo["version_note"]          # a failure may come from different game files
        if not (ginfo["replay_version"] and ginfo["game_version"]):
            hint = "回放和游戏文件版本须一致" + (f"（游戏文件 {ginfo['game_version']}）" if ginfo["game_version"] else "")
        step("parse")
        summary = run_backend(backend, game_dir, replay, export_dir, timeout=timeout_s, on_process=on_process,
                              version_hint=hint)
        step("convert")
        converted = convert(export_dir/"match_export.json", work/"out", final_name, team=team,
                            all_units=all_units, on_process=on_process)
        path = place(converted, out_dir/f"{final_name}.jsonl", ginfo["replay_version"])
        if settings:
            gamefiles.remember_game_dir(settings, game_dir)
        return dict(status="imported", path=path, file=f"wt_real/{path.name}", name=final_name, sha256=sha,
                    game_dir=str(game_dir), game_reason=ginfo["reason"], replay_version=ginfo["replay_version"],
                    game_version=ginfo["game_version"], version_note=ginfo["version_note"],
                    backend=str(backend.path), summary=summary,
                    export=str(export_dir/"match_export.json") if keep_export else None)
    except BaseException:
        if keep_export and "export_dir" in locals() and export_dir.exists():
            shutil.rmtree(export_dir, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(work, ignore_errors=True)


# -- sandbox queue -------------------------------------------------------------------------------------------------

PHASES = ("saving", "queued", "parse", "convert", "done", "error", "needs_game", "needs_backend", "downloading",
          "cancelled")
FINAL = ("done", "error", "cancelled")
ID_RE = re.compile(r"^[0-9a-f]{12}$")


class ImportQueue:
    """Uploads from POST /api/wrpl, imported one at a time in arrival order on a worker thread. An item waits in
    needs_game / needs_backend until the user gives a game folder or confirms the backend download."""
    def __init__(self, *, inbox=None, out_dir=None, settings=None, work_root=None, backend=None, cache=None,
                 manifest=None, timeout_s=BACKEND_TIMEOUT_S, max_bytes=MAX_UPLOAD, keep=24):
        self.inbox, self.out_dir = Path(inbox or INBOX), Path(out_dir or DEFAULT_OUT)
        self.settings = SETTINGS if settings is None else settings
        self.work_root = Path(work_root or WORK)
        self.backend, self.cache, self.manifest = backend, cache, manifest
        self.timeout_s, self.max_bytes, self.keep = timeout_s, max_bytes, keep
        self.items = OrderedDict()
        self.cond = threading.Condition()
        self.game = None                 # folder given in the sandbox for the waiting item
        self.download_requested = False
        self.download_info = None
        self.proc = None
        self.current = None
        self.stopped = False
        self._clean_inbox()
        self.thread = threading.Thread(target=self._loop, name="wrpl-import", daemon=True)
        self.thread.start()

    def _clean_inbox(self, age_s=6*3600):
        """Leftovers of a stopped server (only our own names, older than ``age_s``)."""
        if not self.inbox.is_dir():
            return
        now = time.time()
        for p in self.inbox.iterdir():
            try:
                old = now-p.stat().st_mtime > age_s
            except OSError:
                continue
            if old and p.is_file() and re.match(r"^[0-9a-f]{12}\.part$", p.name):
                p.unlink(missing_ok=True)
            elif old and p.is_dir() and ID_RE.match(p.name):
                shutil.rmtree(p, ignore_errors=True)

    # -- HTTP side ---------------------------------------------------------------------------------------------

    def receive(self, name, length, stream, *, force=False, chunk=1 << 20):
        """Stream ``length`` bytes of ``stream`` into the inbox while hashing; returns the item id."""
        base = re.split(r"[\\/]", name or "")[-1].strip()
        if not base.lower().endswith(".wrpl") or len(base) <= 5:
            raise ValueError("只接受 .wrpl 文件")
        if length <= 0:
            raise ValueError("文件为空")
        if length > self.max_bytes:
            raise UploadTooLarge(f"文件超过 {self.max_bytes >> 20} MB 上限")
        jid = secrets.token_hex(6)
        item = dict(id=jid, name=base, size=length, phase="saving", message=None, log=[], file=None,
                    duplicate=False, created=time.time(), received=0)
        with self.cond:
            self.items[jid] = item
            self._prune()
        self.inbox.mkdir(parents=True, exist_ok=True)
        part = self.inbox/f"{jid}.part"
        h, left = hashlib.sha256(), length
        try:
            with open(part, "wb") as out:
                while left:
                    block = stream.read(min(chunk, left))
                    if not block:
                        raise ValueError("上传中断")
                    out.write(block)
                    h.update(block)
                    left -= len(block)
                    item["received"] = length-left
        except (OSError, ValueError) as exc:
            part.unlink(missing_ok=True)
            self._finish(jid, "error", message=f"保存失败：{exc}")
            raise ValueError(f"保存失败：{exc}") from None
        sha = h.hexdigest()
        existing = None if force else imported(self.out_dir).get(sha)
        if existing:
            part.unlink(missing_ok=True)
            self._finish(jid, "done", file=f"wt_real/{existing[0].name}", duplicate=True,
                         message=f"已导入过：{existing[0].stem}")
            return jid
        folder = self.inbox/jid
        folder.mkdir()
        local = folder/re.sub(r'[<>:"|?*\x00-\x1f]', "_", base)   # the name is kept: it names the output
        os.replace(part, local)
        with self.cond:
            item.update(phase="queued", path=str(local), sha256=sha, force=force)
            self.cond.notify_all()
        return jid

    def command(self, command):
        action = command.get("action")
        if action == "wrpl_game":
            path = command.get("path")
            if not isinstance(path, str) or not path.strip():
                raise ValueError("请填写游戏目录")
            folder = Path(path.strip().strip('"')).expanduser()
            missing = gamefiles.missing_files(folder)
            if missing:
                raise ValueError(f"该目录缺少 {'、'.join(missing)}")
            with self.cond:
                self.game = folder
                for item in self.items.values():
                    if item["phase"] == "needs_game":
                        item.update(phase="queued", message=None)
                self.cond.notify_all()
            return dict(game_dir=str(folder), version=gamefiles.game_version(folder))
        if action == "wrpl_download":
            with self.cond:
                if not any(i["phase"] == "needs_backend" for i in self.items.values()):
                    raise ValueError("没有等待下载的导入")
                self.download_requested = True
                self.cond.notify_all()
            return dict(download=True)
        if action == "wrpl_cancel":
            jid = command.get("id")
            with self.cond:
                item = self.items.get(jid)
                if item is None or item["phase"] in FINAL:
                    raise ValueError("导入不存在或已结束")
                if item["phase"] in ("parse", "convert") and self.proc is not None:
                    self.proc.kill()
                item["cancel"] = True
                if item["phase"] in ("queued", "needs_game", "needs_backend"):
                    self._finish_locked(item, "cancelled")
                self.cond.notify_all()
            return dict(cancelled=jid)
        raise ValueError(f"未知命令 {action}")

    def item(self, jid):
        with self.cond:
            item = self.items.get(jid)
            return None if item is None else self._public(item)

    def status(self):
        with self.cond:
            items = [self._public(i) for i in self.items.values()]
            waiting = next((i for i in self.items.values() if i["phase"] not in FINAL+("saving",)), None)
            game_dir = None
            if waiting is not None and waiting["phase"] == "needs_game" and self.settings:   # prefill of the field
                game_dir = gamefiles.load_settings(self.settings).get("game_dir")
            return dict(items=items, busy=waiting is not None, game_dir=game_dir, backend=self.public_download(),
                        max_bytes=self.max_bytes)

    def close(self):
        with self.cond:
            self.stopped = True
            if self.proc is not None:
                self.proc.kill()
            self.cond.notify_all()
        self.thread.join(timeout=5)

    # -- worker ------------------------------------------------------------------------------------------------

    @staticmethod
    def _public(item):
        out = {k: item[k] for k in ("id", "name", "size", "phase", "message", "log", "file", "duplicate", "received")}
        out["total"] = item.get("total") or item["size"]          # bytes expected for "received" (upload / download)
        out["done"] = item["phase"] in FINAL
        return out

    def _prune(self):
        done = [k for k, v in self.items.items() if v["phase"] in FINAL]
        for k in done[:max(0, len(self.items)-self.keep)]:
            del self.items[k]

    def _finish(self, jid, phase, **fields):
        with self.cond:
            item = self.items.get(jid)
            if item is not None:
                self._finish_locked(item, phase, **fields)

    def _finish_locked(self, item, phase, **fields):
        item.update(phase=phase, **fields)
        if item.get("path"):
            shutil.rmtree(Path(item["path"]).parent, ignore_errors=True)
        self.cond.notify_all()

    def _set_phase(self, item, phase):
        with self.cond:
            if item["phase"] not in FINAL:
                item["phase"] = phase

    def _on_process(self, proc):
        with self.cond:
            self.proc = proc
            if proc is not None and (self.current or {}).get("cancel"):   # cancelled before the process started
                proc.kill()

    def _next(self):
        """The first unfinished item (strict arrival order), waiting while it cannot run."""
        while not self.stopped:
            head = next((i for i in self.items.values() if i["phase"] not in FINAL+("saving",)), None)
            if head is not None and head["phase"] == "queued":
                return head
            if head is not None and head["phase"] == "needs_backend" and self.download_requested:
                head["phase"] = "downloading"
                return head
            self.cond.wait(timeout=1.)
        return None

    def _loop(self):
        while True:
            with self.cond:
                item = self._next()
                if item is None:
                    return
            if item["phase"] == "downloading":
                self._download(item)
            else:
                self._import(item)

    def _download(self, item):
        asset = (self.download_info or {}).get("asset")
        try:
            if asset is None:
                raise backends.BackendError("没有可下载的解析后端")

            def progress(done, total):
                with self.cond:
                    item["received"], item["total"] = done, total or None
            backends.download(asset, self.cache, progress)
            with self.cond:
                self.download_requested, self.download_info = False, None
                item.update(phase="queued", message=None, received=item["size"], total=None)
        except (backends.BackendError, OSError) as exc:
            with self.cond:
                self.download_requested = False
                item.update(phase="needs_backend", message=f"下载失败：{exc}")

    def _import(self, item):
        def progress(phase):
            self._set_phase(item, phase)
        self._set_phase(item, "parse")
        with self.cond:
            game, self.current = self.game, item
        try:
            result = import_replay(item["path"], game=game, out_dir=self.out_dir, backend=self.backend,
                                   force=item.get("force", False), settings=self.settings, work_root=self.work_root,
                                   timeout_s=self.timeout_s, sha256=item.get("sha256"), progress=progress,
                                   on_process=self._on_process, cache=self.cache, manifest=self.manifest)
        except gamefiles.GameFilesError as exc:
            with self.cond:
                self.game = None
                item.update(phase="needs_game", message=str(exc))
            return
        except backends.BackendMissing as exc:
            with self.cond:
                if exc.asset is None:
                    self._finish_locked(item, "error", message=str(exc))
                else:
                    self.download_info = dict(asset=exc.asset, size_mb=exc.asset.size_mb, message=str(exc))
                    item.update(phase="needs_backend", message=str(exc))
            return
        except ImportFailed as exc:
            with self.cond:
                if item.get("cancel"):
                    self._finish_locked(item, "cancelled")
                else:
                    self._finish_locked(item, "error", message=str(exc), log=exc.log)
            return
        except Exception as exc:  # unexpected: reported on the item, the queue goes on
            with self.cond:
                self._finish_locked(item, "error", message=f"{type(exc).__name__}: {exc}")
            return
        finally:
            with self.cond:
                self.current = None
        with self.cond:
            self.game = None
            msg = f"已导入过：{result['name']}" if result["status"] == "duplicate" else f"已导入：{result['name']}"
            if result.get("version_note"):
                msg += f"（{result['version_note']}）"
            self._finish_locked(item, "done", file=result["file"], duplicate=result["status"] == "duplicate",
                                message=msg)

    def public_download(self):
        info = self.download_info
        return None if info is None else dict(size_mb=info["size_mb"], message=info["message"])
