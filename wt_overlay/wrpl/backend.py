"""Locate, download and run the wrpl_export backend (docs/wrpl_import_spec.md §2, §3).

Order: an explicit path (--backend), the WT_WRPL_BACKEND environment variable, the user cache
(<cache>/<version>/<platform>/wrpl_export/), else a download of this platform's Release asset listed in backend.json,
which happens only after the user confirms it. A downloaded zip is checked against the SHA-256 in backend.json before
it is unpacked; a mismatch deletes it. A path may be the executable, the folder holding it, or (development) a .py
file run with the current Python.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import platform as _platform
import shutil
import stat
import subprocess
import sys
import tempfile
from urllib.request import urlopen
import zipfile

MANIFEST = Path(__file__).with_name("backend.json")
EXE = "wrpl_export"
ENV = "WT_WRPL_BACKEND"
CACHE_ENV = "WT_WRPL_CACHE"          # overrides the cache root (tests, portable setups)
VERIFIED = ".sha256"                 # marker in an unpacked cache folder: the zip's verified hash
ROSETTA_HINT = "需要 Rosetta 才能运行 x86_64 版解析后端：softwareupdate --install-rosetta"


class BackendError(RuntimeError):
    pass


class BackendMissing(BackendError):
    """No backend yet; ``asset`` is the download that would provide it (None: nothing published for this platform)."""
    def __init__(self, message, asset=None):
        super().__init__(message)
        self.asset = asset


@dataclass
class Backend:
    path: Path
    source: str                          # explicit / env / cache
    command: list = field(default_factory=list)

    def run(self, args, **kwargs):
        return subprocess.run(self.command+list(args), **kwargs)


@dataclass
class Asset:
    platform: str
    version: str
    file: str
    url: str
    sha256: str
    size: int

    @property
    def size_mb(self):
        return max(1, round(self.size/1e6)) if self.size else None


def load_manifest(path=None):
    return json.loads(Path(path or MANIFEST).read_text(encoding="utf-8"))


def platform_keys(system=None, machine=None):
    """Asset names to try, best first. macOS: arm64 when it exists, else x64 (Rosetta on Apple Silicon)."""
    system = system or sys.platform
    machine = (machine or _platform.machine()).lower()
    if system.startswith("win"):
        return ["windows-x64"]
    if system == "darwin":
        return ["macos-arm64", "macos-x64"] if machine in ("arm64", "aarch64") else ["macos-x64"]
    return ["linux-x64"] if machine in ("x86_64", "amd64") else []


def cache_root(system=None, env=None):
    env = os.environ if env is None else env
    if env.get(CACHE_ENV):
        return Path(env[CACHE_ENV])
    system = system or sys.platform
    if system.startswith("win"):
        base = Path(env.get("LOCALAPPDATA") or Path.home()/"AppData"/"Local")
        return base/"wt_overlay"/"wrpl"
    if system == "darwin":
        return Path.home()/"Library"/"Caches"/"wt_overlay"/"wrpl"
    return Path(env.get("XDG_CACHE_HOME") or Path.home()/".cache")/"wt_overlay"/"wrpl"


def published_asset(manifest=None, keys=None):
    """The first published asset among ``keys`` (platform_keys()), or None."""
    manifest = manifest or load_manifest()
    for key in keys if keys is not None else platform_keys():
        entry = (manifest.get("assets") or {}).get(key) or {}
        if entry.get("url") and entry.get("sha256"):
            return Asset(key, manifest["version"], entry.get("file") or f"{EXE}-{key}.zip", entry["url"],
                         entry["sha256"].lower(), int(entry.get("size") or 0))
    return None


def _executable_in(path):
    path = Path(path)
    if path.is_file():
        return path
    for name in (EXE+".exe", EXE):
        for candidate in (path/name, path/EXE/name):
            if candidate.is_file():
                return candidate
    return None


def _command(path):
    if path.suffix == ".py":
        return [sys.executable, str(path)]
    return [str(path)]


def cached_backend(cache=None, manifest=None, keys=None):
    manifest = manifest or load_manifest()
    root = Path(cache or cache_root())/manifest["version"]
    for key in keys if keys is not None else platform_keys():
        folder = root/key
        exe = _executable_in(folder)
        if exe is not None and (folder/VERIFIED).is_file():
            return Backend(exe, "cache", _command(exe))
    return None


def find_backend(explicit=None, *, env=None, cache=None, manifest=None, keys=None):
    """Backend to run, or BackendMissing (with the asset to download, if one is published)."""
    env = os.environ if env is None else env
    for value, source in ((explicit, "explicit"), (env.get(ENV), "env")):
        if value:
            exe = _executable_in(value)
            if exe is None:
                where = "--backend" if source == "explicit" else ENV
                raise BackendError(f"{where} 指向的解析后端不存在：{value}")
            return Backend(exe, source, _command(exe))
    manifest = manifest or load_manifest()
    found = cached_backend(cache, manifest, keys)
    if found is not None:
        return found
    asset = published_asset(manifest, keys)
    if asset is None:
        raise BackendMissing("此平台还没有发布的解析后端；本地构建（tools/wrpl_backend）后用 --backend 或 "
                             f"{ENV} 指定")
    size = f"约 {asset.size_mb} MB，" if asset.size_mb else ""
    raise BackendMissing(f"需要下载解析后端（{size}来自本项目 GitHub Release）", asset)


def _safe_members(z, dest):
    dest = dest.resolve()
    for info in z.infolist():
        target = (dest/info.filename).resolve()
        if target != dest and dest not in target.parents:
            raise BackendError(f"压缩包里有越界路径：{info.filename}")
        yield info


def _extract(zip_path, dest):
    """Unpack with file modes and symlinks (the macOS bundle has framework symlinks)."""
    with zipfile.ZipFile(zip_path) as z:
        for info in _safe_members(z, dest):
            mode = info.external_attr >> 16
            target = dest/info.filename
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            elif stat.S_ISLNK(mode):
                link = z.read(info).decode("utf-8")
                points_to = os.path.normpath(os.path.join(os.path.abspath(target.parent), link))
                if os.path.isabs(link) or os.path.commonpath([points_to, os.path.abspath(dest)]) != \
                        os.path.abspath(dest):
                    raise BackendError(f"压缩包里有越界链接：{info.filename}")
                target.parent.mkdir(parents=True, exist_ok=True)
                os.symlink(link, target)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst, 1 << 20)
                if mode & 0o777:
                    os.chmod(target, mode & 0o777)


def download(asset, cache=None, progress=None, chunk=1 << 20):
    """Download, verify and unpack ``asset``; returns the cached Backend. Nothing is kept unless the hash matches."""
    folder = Path(cache or cache_root())/asset.version/asset.platform
    folder.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=asset.platform+"-", suffix=".zip.part", dir=folder.parent)
    tmp = Path(tmp_name)
    unpack = None
    try:
        h, done = hashlib.sha256(), 0
        with os.fdopen(fd, "wb") as out, urlopen(asset.url, timeout=60) as response:
            while True:
                block = response.read(chunk)
                if not block:
                    break
                out.write(block)
                h.update(block)
                done += len(block)
                if progress:
                    progress(done, asset.size)
        if h.hexdigest() != asset.sha256:
            raise BackendError(f"解析后端校验失败（SHA-256 不符），已删除下载的文件：{asset.file}")
        unpack = Path(tempfile.mkdtemp(prefix=asset.platform+"-", dir=folder.parent))
        _extract(tmp, unpack)
        if _executable_in(unpack) is None:
            raise BackendError(f"压缩包里没有 {EXE}：{asset.file}")
        (unpack/VERIFIED).write_text(asset.sha256+"\n", encoding="utf-8")
        if sys.platform == "darwin":   # urllib sets no quarantine flag; a copied-in folder may have one
            subprocess.run(["xattr", "-dr", "com.apple.quarantine", str(unpack)], capture_output=True, check=False)
        if folder.exists():
            shutil.rmtree(folder)
        os.replace(unpack, folder)
        unpack = None
    finally:
        tmp.unlink(missing_ok=True)
        if unpack is not None:
            shutil.rmtree(unpack, ignore_errors=True)
    exe = _executable_in(folder)
    return Backend(exe, "cache", _command(exe))


def version(backend, timeout=120):
    """The backend's --version JSON. Raises BackendError with a Chinese reason (Rosetta, not executable, ...)."""
    try:
        out = backend.run(["--version"], capture_output=True, text=True, timeout=timeout)
    except OSError as exc:
        if getattr(exc, "errno", None) == 86 or "Bad CPU type" in str(exc):
            raise BackendError(ROSETTA_HINT) from None
        raise BackendError(f"无法运行解析后端 {backend.path}：{exc}") from None
    except subprocess.TimeoutExpired:
        raise BackendError("解析后端 --version 超时") from None
    if out.returncode != 0:
        tail = "\n".join((out.stderr or "").strip().splitlines()[-6:])
        raise BackendError(f"解析后端无法启动（退出码 {out.returncode}）\n{tail}".rstrip())
    try:
        return json.loads(out.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        raise BackendError("解析后端 --version 输出无法识别") from None
