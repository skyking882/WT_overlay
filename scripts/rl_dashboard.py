#!/usr/bin/env python3
"""Local web dashboard for RL training runs and simulated-match replays (stdlib only, binds 127.0.0.1).

    python3 scripts/rl_dashboard.py --port 8765 \\
        [--run DIR]... [--remote HOST:PATH]... [--replays DIR]...

Training tab: reads <run>/metrics.jsonl (one JSON object per PPO round; a half-written or corrupt line is skipped),
bc_report.json, config.json, ppo/ckpt_*.pt (listed only, torch is never imported) and train.log.
``--run DIR`` is a run directory, or a folder of run directories (every sub-folder holding metrics.jsonl,
bc_report.json, config.json or ppo/ is a run). Without ``--run`` the folder outputs/rl_runs is scanned when it exists.

Remote runs: ``--remote workstation:rl_runs/smoke_lb`` pulls the small files (metrics.jsonl, bc_report.json,
config.json, train.log and a replays/ sub-folder) with rsync over ssh (BatchMode, nothing but rsync's own read runs on the
remote host) every 30 s into outputs/rl_dashboard_cache/<host>__<path>/ and shows the copy as a run. The checkpoint names
under ppo/ are listed with ``rsync --list-only`` (no file content is transferred).

Replay tab: the JSONL written by wt_overlay.engagement.ReplayWriter. Files under ``--replays DIR`` (default
outputs/engagements) and every run's replays/ folder are listed; a replay is converted on the server to per-aircraft
and per-missile tracks, downsampled (default 1 Hz, the file has 4 Hz) with the first and last sample of every track kept.

JSON endpoints (all GET unless noted):
    /api/info                      server info, remotes, replay folders
    /api/runs                      the run list with a one-line status each
    /api/run/<id>/metrics          ?window=20&target=5000000   rounds + status (rate, ETA, live/stale)
    /api/run/<id>/bc               bc_report.json
    /api/run/<id>/config           config.json
    /api/run/<id>/ckpts            checkpoint files
    /api/run/<id>/log              ?lines=200   tail of train.log
    /api/run/<id>/stage            stage (collect / bc / ppo_pending / ppo) and what train.log says about BC
    /api/replays                   replay list with a summary each
    /api/replay/<id>               ?hz=1   one replay as tracks + events
    /api/sync                      remote sync state; POST triggers a sync now
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

VERSION = "1.0"
ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().with_name("rl_dashboard")
DEFAULT_CACHE = ROOT / "outputs" / "rl_dashboard_cache"
DEFAULT_REPLAYS = ROOT / "outputs" / "engagements"
DEFAULT_RUN_ROOT = ROOT / "outputs" / "rl_runs"

SYNC_FILES = ("metrics.jsonl", "bc_report.json", "config.json", "train.log")
LOG_NAMES = ("train.log", "train.out", "train.txt", "run.log", "stdout.log", "nohup.out")
RUN_MARKERS = ("metrics.jsonl", "bc_report.json", "config.json", "ppo")
HEAVY_KEYS = ("head_active_frac",)           # per-round keys the page never needs
MAX_REPLAY_FILES = 2000
STATIC_TYPES = {".html": "text/html; charset=utf-8", ".js": "application/javascript; charset=utf-8",
                ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml", ".png": "image/png",
                ".ico": "image/x-icon", ".json": "application/json; charset=utf-8"}


# -- small helpers -------------------------------------------------------------------------------------------

def clean(obj):
    """Make ``obj`` strict-JSON safe: NaN / Infinity become None (Python's json writes them as bare NaN)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    return obj


def dumps(obj) -> bytes:
    return json.dumps(clean(obj), separators=(",", ":"), allow_nan=False).encode("utf-8")


def short_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def resolve_path(p) -> Path:
    """A path from the command line: relative paths are taken from the cwd, else from the repository root."""
    path = Path(p).expanduser()
    if path.is_absolute():
        return path
    cand = Path.cwd() / path
    if not cand.exists() and (ROOT / path).exists():
        return ROOT / path
    return cand


def file_sig(path: Path):
    try:
        st = path.stat()
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size)


def read_json_file(path: Path):
    """A JSON file or None (missing / half written / not JSON)."""
    try:
        return json.loads(path.read_bytes())
    except (OSError, ValueError):
        return None


def parse_jsonl_bytes(data: bytes):
    """-> (dict records, number of bad lines, whether the last non-empty line was cut off).

    A line that is not valid JSON (corrupt, or the tail of a file that is being written) is skipped, never fatal."""
    records, bad, last_bad = [], 0, False
    lines = [ln for ln in data.split(b"\n") if ln.strip()]
    for i, raw in enumerate(lines):
        try:
            o = json.loads(raw)
        except ValueError:
            bad += 1
            last_bad = i == len(lines) - 1
            continue
        if isinstance(o, dict):
            records.append(o)
        else:
            bad += 1
    partial_tail = last_bad and not data.endswith(b"\n")
    return records, bad, partial_tail


class Cache:
    """Tiny signature-keyed memo: a value is recomputed only when the file's (mtime, size) changed."""

    def __init__(self, limit=64):
        self.limit, self.items, self.lock = limit, {}, threading.Lock()

    def get(self, key, sig, build):
        with self.lock:
            hit = self.items.get(key)
            if hit is not None and hit[0] == sig:
                return hit[1]
        value = build()
        with self.lock:
            if len(self.items) >= self.limit and key not in self.items:
                self.items.pop(next(iter(self.items)))
            self.items[key] = (sig, value)
        return value


# -- training metrics ----------------------------------------------------------------------------------------

def load_metrics(path: Path):
    """metrics.jsonl -> dict(records, bad, partial_tail). Records are ordered by round; if a round shows up again (a
    resumed run), the earlier records from that round on are dropped, as MetricsLog.truncate_after does."""
    try:
        data = path.read_bytes()
    except OSError:
        return dict(records=[], bad=0, partial_tail=False)
    raw, bad, partial = parse_jsonl_bytes(data)
    records = []
    for r in raw:
        rnd = r.get("round")
        if not isinstance(rnd, (int, float)) or isinstance(rnd, bool):
            bad += 1
            continue
        while records and records[-1]["round"] >= rnd:
            records.pop()
        records.append(r)
    return dict(records=records, bad=bad, partial_tail=partial)


def _median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else None


def rate_per_hour(records, window=20):
    """Decisions per hour over the last ``window`` rounds -> (rate or None, rounds used).

    Wall clock from the record timestamps; a gap much longer than a round (the run was stopped and resumed) is left
    out together with the decisions of the round that ended it. Without timestamps the per-round time split is used."""
    pts = records[-(max(int(window), 1) + 1):]
    if len(pts) >= 2:
        dec = el = 0.0
        pairs = list(zip(pts, pts[1:]))
        dts = [b.get("timestamp") - a.get("timestamp") for a, b in pairs
               if isinstance(a.get("timestamp"), (int, float)) and isinstance(b.get("timestamp"), (int, float))]
        cap = max(300.0, 10.0 * (_median([d for d in dts if d > 0]) or 0.0))
        used = 0
        for a, b in pairs:
            ta, tb = a.get("timestamp"), b.get("timestamp")
            da, db = a.get("decisions_total"), b.get("decisions_total")
            if not all(isinstance(x, (int, float)) for x in (ta, tb, da, db)):
                continue
            if 0 < tb - ta <= cap:
                dec += db - da
                el += tb - ta
                used += 1
        if used and el > 0:
            return dec / el * 3600.0, used
    # fallback: the time split recorded by the trainer
    last = records[-max(int(window), 1):]
    dec = sum(r.get("decisions_in_round") or 0 for r in last)
    el = sum(((r.get("time") or {}).get("round") or 0) for r in last)
    if dec > 0 and el > 0:
        return dec / el * 3600.0, len(last)
    return None, 0


def run_status(records, window=20, target=None, now=None, cfg=None):
    """The status line of a run: round, decisions, rate, ETA to ``target`` decisions, live / stale."""
    now = time.time() if now is None else now
    if not records:
        return dict(state="waiting", round=0, decisions_total=0, last_timestamp=None, age_s=None, rate_per_hour=None,
                    rate_rounds=0, eta_s=None, progress=None, median_round_s=None, target=target, suggested_target=None,
                    rounds_total=None)
    last = records[-1]
    ts = last.get("timestamp") if isinstance(last.get("timestamp"), (int, float)) else None
    age = max(0.0, now - ts) if ts is not None else None
    rate, used = rate_per_hour(records, window)
    dec = last.get("decisions_total") or 0
    round_times = [(r.get("time") or {}).get("round") for r in records[-20:]]
    round_times = [x for x in round_times if isinstance(x, (int, float)) and x > 0]
    med = _median(round_times)
    eta = progress = None
    if isinstance(target, (int, float)) and target > 0:
        progress = min(dec / target, 1.0)
        if dec >= target:
            eta = 0.0
        elif rate:
            eta = (target - dec) / rate * 3600.0
    rounds_total = ((cfg or {}).get("run") or {}).get("rounds")
    per_round = [r.get("decisions_in_round") for r in records[-max(int(window), 1):]
                 if isinstance(r.get("decisions_in_round"), (int, float))]
    suggested = None
    if isinstance(rounds_total, (int, float)) and rounds_total > 0 and per_round:
        suggested = int(round(rounds_total * sum(per_round) / len(per_round), -2))
    live = age is not None and age < max(60.0, 2.5 * (med or 0.0))
    return dict(state="live" if live else "stale", round=last.get("round"), decisions_total=dec, last_timestamp=ts,
                age_s=age, rate_per_hour=rate, rate_rounds=used, eta_s=eta, progress=progress, median_round_s=med,
                target=target, suggested_target=suggested, rounds_total=rounds_total)


def slim_records(records):
    """Records for the page: no head_active_frac; per_aircraft only on the last round (the table shows the latest)."""
    out = []
    for i, r in enumerate(records):
        d = {k: v for k, v in r.items() if k not in HEAVY_KEYS}
        if i != len(records) - 1:
            d.pop("per_aircraft", None)
        out.append(d)
    return out


def config_brief(cfg):
    if not isinstance(cfg, dict):
        return None
    ppo, run = cfg.get("ppo") or {}, cfg.get("run") or {}
    return dict(name=cfg.get("name"), target_kl=ppo.get("target_kl"), clip=ppo.get("clip"), rounds=run.get("rounds"),
                device=(cfg.get("resources") or {}).get("device"))


# -- log tail ------------------------------------------------------------------------------------------------

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def tail_lines(path: Path, n: int, window=262144):
    """Last ``n`` lines of a text file without reading all of it -> (lines, file size, truncated at the front)."""
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            f.seek(max(0, size - window))
            data = f.read()
    except OSError:
        return [], 0, False
    text = data.decode("utf-8", errors="replace")
    lines = text.split("\n")
    cut = size > window
    if cut:
        lines = lines[1:]               # the first line is probably cut in the middle
    lines = [ANSI.sub("", ln.rsplit("\r", 1)[-1]).rstrip() for ln in lines]
    while lines and not lines[-1]:
        lines.pop()
    return lines[-n:], size, cut or len(lines) > n


# -- train.log: the stage a run is in (before metrics.jsonl exists) ---------------------------------------------

LOG_LINE = re.compile(r"^\[(\d\d):(\d\d):(\d\d)\]\s?(.*)$")
RE_SHARD = re.compile(r"^bc data: shard (\d+), (\d+) decisions \(total (\d+)\)")
RE_COLLECTED = re.compile(r"^bc data: (\d+) decisions already collected")
RE_STOPPED = re.compile(r"^bc data: stopped on request after (\d+)")
RE_DATASET = re.compile(r"^bc dataset: (\{.*\})\s*$")
RE_EPOCH = re.compile(r"^bc epoch (\d+)\s+train (\S+)\s+val (\S+)\s+bad (\d+)")
RE_RESUMED = re.compile(r"^bc: resumed after epoch (\d+)")
RE_DONE = re.compile(r"^bc done: best epoch (\d+) val loss (\S+?);")
RE_PPO_ROUND = re.compile(r"^round (\d+)\s+dec (\d+)")
RE_RUN = re.compile(r"^run dir (.*?) \| config (\S+)")
STAGE_LABELS = {"collect": "采集 BC 数据", "bc": "BC 阶段", "ppo_pending": "PPO 启动中", "ppo": "PPO 训练",
                "waiting": "等待数据"}


def _float(text):
    try:
        v = float(text)
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def parse_train_log(text: str) -> dict:
    """What train.log says about the BC stages: collected shards, dataset statistics, BC epochs, 'bc done', the PPO
    round counter. Times are seconds since the first line, wrapped across midnight (the log has clock times only)."""
    shards, epochs, dataset, done, workers, run = [], {}, None, None, None, None
    collected = ppo_round = ppo_started = resumed_from = None
    t_start = t_last = None
    t_dataset = None
    day, prev, first = 0, None, None
    clock = None
    for line in text.splitlines():
        m = LOG_LINE.match(line.strip())
        if not m:
            continue
        sec = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
        clock = "%s:%s:%s" % m.group(1, 2, 3)
        if prev is not None and sec + day * 86400 < prev:
            day += 1
        t = sec + day * 86400
        prev = t
        first = t if first is None else first
        t_last = t
        body = m.group(4)
        if (g := RE_SHARD.match(body)):
            shards.append(dict(shard=int(g[1]), decisions=int(g[2]), total=int(g[3]), t=t))
        elif (g := RE_COLLECTED.match(body)):
            collected = int(g[1])
        elif (g := RE_STOPPED.match(body)):
            collected = int(g[1])
        elif (g := RE_DATASET.match(body)):
            try:
                dataset = json.loads(g[1])
            except ValueError:
                dataset = None
            t_dataset = t
        elif (g := RE_EPOCH.match(body)):
            epochs[int(g[1])] = dict(epoch=int(g[1]), train=_float(g[2]), val=_float(g[3]), bad=int(g[4]), t=t)
        elif (g := RE_RESUMED.match(body)):
            resumed_from = int(g[1])
        elif (g := RE_DONE.match(body)):
            done = dict(best_epoch=int(g[1]), best_val_loss=_float(g[2]))
        elif body.startswith("ppo: ") and "WARNING" not in body:
            ppo_started = True
        elif (g := RE_PPO_ROUND.match(body)):
            ppo_round = int(g[1])
            ppo_started = True
        elif body.startswith("workers: "):
            workers = body[len("workers: "):]
        elif (g := RE_RUN.match(body)):
            run = dict(run_dir=g[1], config=g[2])
            t_start = t
    ep = [epochs[k] for k in sorted(epochs)]
    total = shards[-1]["total"] if shards else collected
    rate = None
    if len(shards) >= 2 and shards[-1]["t"] > shards[0]["t"]:
        rate = (shards[-1]["total"] - shards[0]["total"]) / (shards[-1]["t"] - shards[0]["t"]) * 3600.0
    elif shards and t_start is not None and shards[0]["t"] > t_start:
        rate = shards[0]["total"] / (shards[0]["t"] - t_start) * 3600.0
    epoch_s = None
    if len(ep) >= 2 and ep[-1]["t"] > ep[0]["t"]:
        epoch_s = (ep[-1]["t"] - ep[0]["t"]) / (len(ep) - 1)
    elif ep and t_dataset is not None and ep[0]["t"] >= t_dataset:
        epoch_s = max(ep[0]["t"] - t_dataset, 0.0) or None
    return dict(shards=shards[-40:], n_shards=len(shards), collected=total, collect_rate_per_hour=rate,
                dataset=dataset, epochs=ep, epoch_s=epoch_s, resumed_from=resumed_from, done=done,
                ppo_started=bool(ppo_started), ppo_round=ppo_round, workers=workers, run=run, last_clock=clock,
                any_line=first is not None)


def log_stage(parsed: dict, has_report=False, has_metrics=False, has_bc_data=False) -> str:
    """collect -> bc -> ppo_pending (BC done, no PPO round yet) -> ppo."""
    if has_metrics or parsed["ppo_round"]:
        return "ppo"
    if has_report or parsed["done"] or parsed["ppo_started"]:
        return "ppo_pending"
    if parsed["epochs"] or parsed["dataset"]:
        return "bc"
    if parsed["shards"] or parsed["collected"] or has_bc_data or parsed["any_line"]:
        return "collect"
    return "waiting"


def stage_steps(stage: str):
    order = ["collect", "bc", "ppo"]
    cur = {"collect": 0, "bc": 1, "ppo_pending": 2, "ppo": 2}.get(stage)
    out = []
    for i, k in enumerate(order):
        state = "todo" if cur is None else "done" if i < cur else "active" if i == cur else "todo"
        if stage == "ppo_pending" and k == "ppo":
            state = "pending"
        out.append(dict(id=k, state=state))
    return out


# -- replay files --------------------------------------------------------------------------------------------

PLANE_COLUMNS = ["id", "x", "y", "z", "vx", "vy", "vz", "heading_deg", "missiles", "chaff", "phase"]
MISSILE_COLUMNS = ["uid", "owner", "target", "x", "y", "z", "vx", "vy", "vz", "heading_deg", "age_s", "seeker", "datalink"]


def _lines(path: Path):
    with path.open("rb") as f:
        for raw in f:
            raw = raw.strip()
            if raw:
                yield raw


def _json_or_none(raw):
    try:
        o = json.loads(raw)
    except ValueError:
        return None
    return o if isinstance(o, dict) else None


def read_replay_header(path: Path):
    """The header object of a replay file, or None when the file is not a replay."""
    try:
        with path.open("rb") as f:
            first = f.readline()
    except OSError:
        return None
    o = _json_or_none(first)
    return o if o and o.get("type") == "header" and isinstance(o.get("planes"), list) else None


def replay_summary(path: Path):
    """Header + end record (read from the tail of the file) -> a short summary for the list, or None if not a replay."""
    header = read_replay_header(path)
    if header is None:
        return None
    planes = header["planes"]
    teams = [sum(1 for p in planes if p.get("team") == k) for k in (0, 1)]
    kinds = {}
    for p in planes:
        kinds.setdefault(p.get("team"), set()).add(p.get("aircraft"))
    end = last_t = None
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            f.seek(max(0, size - 196608))
            tail = f.read().split(b"\n")
        if size > 196608:
            tail = tail[1:]
        for raw in reversed(tail):
            if not raw.strip():
                continue
            o = _json_or_none(raw)
            if o is None:
                continue
            if o.get("type") == "end":
                end = o
                break
            if last_t is None and o.get("type") in ("frame", "event") and isinstance(o.get("t"), (int, float)):
                last_t = o["t"]
    except OSError:
        pass
    res = (end or {}).get("result") or {}
    return dict(planes=len(planes), teams=teams, seed=header.get("seed"), map_half_m=header.get("map_half_m"),
                aircraft=[sorted(a for a in kinds.get(k, ()) if a) for k in (0, 1)],
                complete=end is not None, reason=(end or {}).get("reason"),
                duration_s=(end or {}).get("t", last_t), teams_alive=res.get("teams_alive"),
                kills=len(res.get("kills") or []) if end else None, launches=res.get("launches"))


def convert_replay(path: Path, hz=1.0):
    """A replay file -> tracks and events for the page.

    Planes and missiles become column arrays (one track each). Frames are thinned to ``hz`` samples per second, but the
    first and the last sample of every plane and missile are always kept (a plane that dies, a missile that ends),
    and events get the last known position (x, y, z) of the aircraft or missile they are about."""
    header = None
    end = None
    events = []
    bad = 0
    frames_total = frames_kept = 0
    ptracks, mtracks = {}, {}
    pend_p, pend_m = {}, {}
    last_pos, last_mpos = {}, {}
    phases, phase_ix = [], {}
    step = 1
    pcol = {n: i for i, n in enumerate(PLANE_COLUMNS)}
    mcol = {n: i for i, n in enumerate(MISSILE_COLUMNS)}

    def phase_index(name):
        name = name or ""
        if name not in phase_ix:
            phase_ix[name] = len(phases)
            phases.append(name)
        return phase_ix[name]

    def add_plane(t, row):
        pid = row[pcol["id"]]
        tr = ptracks.get(pid)
        if tr is None:
            tr = ptracks[pid] = {k: [] for k in ("t", "x", "y", "z", "vx", "vy", "vz", "h", "m", "c", "p")}
        tr["t"].append(t)
        tr["x"].append(round(row[pcol["x"]]))
        tr["y"].append(round(row[pcol["y"]]))
        tr["z"].append(round(row[pcol["z"]]))
        tr["vx"].append(round(row[pcol["vx"]]))
        tr["vy"].append(round(row[pcol["vy"]]))
        tr["vz"].append(round(row[pcol["vz"]]))
        tr["h"].append(round(row[pcol["heading_deg"]]) % 360)
        tr["m"].append(row[pcol["missiles"]])
        tr["c"].append(row[pcol["chaff"]])
        tr["p"].append(phase_index(row[pcol["phase"]]))

    def add_missile(t, row):
        uid = row[mcol["uid"]]
        tr = mtracks.get(uid)
        if tr is None:
            tr = mtracks[uid] = dict(uid=uid, owner=row[mcol["owner"]], target=row[mcol["target"]],
                                     t=[], x=[], y=[], z=[], h=[], s=[], d=[])
        tr["t"].append(t)
        tr["x"].append(round(row[mcol["x"]]))
        tr["y"].append(round(row[mcol["y"]]))
        tr["z"].append(round(row[mcol["z"]]))
        tr["h"].append(round(row[mcol["heading_deg"]]) % 360)
        tr["s"].append(int(row[mcol["seeker"]]))
        tr["d"].append(int(row[mcol["datalink"]]))

    for raw in _lines(path):
        o = _json_or_none(raw)
        if o is None:
            bad += 1
            continue
        typ = o.get("type")
        if typ == "header":
            header = o
            cols = o.get("plane_columns")
            if isinstance(cols, list) and all(c in cols for c in PLANE_COLUMNS):
                pcol = {n: cols.index(n) for n in PLANE_COLUMNS}
            cols = o.get("missile_columns")
            if isinstance(cols, list) and all(c in cols for c in MISSILE_COLUMNS):
                mcol = {n: cols.index(n) for n in MISSILE_COLUMNS}
            dt = o.get("frame_dt_s") or 0.25
            step = max(1, int(round(1.0 / (max(hz, 1e-3) * dt))))
        elif typ == "frame":
            t = o.get("t")
            if not isinstance(t, (int, float)):
                bad += 1
                continue
            keep = frames_total % step == 0
            frames_total += 1
            frames_kept += keep
            seen = set()
            for row in o.get("planes") or ():
                pid = row[pcol["id"]]
                seen.add(pid)
                last_pos[pid] = (row[pcol["x"]], row[pcol["y"]], row[pcol["z"]])
                if keep or pid not in ptracks:
                    add_plane(t, row)
                    pend_p.pop(pid, None)
                else:
                    pend_p[pid] = (t, row)
            for pid in [k for k in pend_p if k not in seen]:      # the plane is gone: close its track exactly
                add_plane(*pend_p.pop(pid))
            seen = set()
            for row in o.get("missiles") or ():
                uid = row[mcol["uid"]]
                seen.add(uid)
                last_mpos[uid] = (row[mcol["x"]], row[mcol["y"]], row[mcol["z"]])
                if keep or uid not in mtracks:
                    add_missile(t, row)
                    pend_m.pop(uid, None)
                else:
                    pend_m[uid] = (t, row)
            for uid in [k for k in pend_m if k not in seen]:
                add_missile(*pend_m.pop(uid))
        elif typ == "event":
            ref = None
            kind = o.get("kind")
            if kind == "launch":
                ref = last_pos.get(o.get("shooter"))
            elif kind == "kill":
                ref = last_pos.get(o.get("victim"))
            elif kind in ("death", "rwr", "chaff"):
                ref = last_pos.get(o.get("plane"))
            elif kind == "missile_end":
                ref = last_mpos.get(o.get("uid"))
            if ref is not None:
                o = dict(o, x=round(ref[0]), y=round(ref[1]), z=round(ref[2]))
            events.append(o)
        elif typ == "end":
            end = o
    for pid, item in pend_p.items():                       # the final frame of every track that is still running
        add_plane(*item)
    for uid, item in pend_m.items():
        add_missile(*item)
    if header is None:
        raise ValueError("not a replay file (no header line)")
    t_end = (end or {}).get("t")
    if t_end is None:
        times = [tr["t"][-1] for tr in ptracks.values() if tr["t"]]
        t_end = max(times) if times else 0.0
    keep_keys = ("version", "seed", "map_half_m", "frame_dt_s", "time_limit_s", "planes")
    st = path.stat()
    return dict(header={k: header.get(k) for k in keep_keys}, phases=phases, t_end=t_end,
                planes={str(k): v for k, v in sorted(ptracks.items())},
                missiles=[mtracks[k] for k in sorted(mtracks)], events=events, end=end,
                complete=end is not None, sample_hz=1.0 / (step * (header.get("frame_dt_s") or 0.25)),
                frames_total=frames_total, frames_kept=frames_kept, bad_lines=bad,
                source=dict(name=path.name, size=st.st_size, mtime=st.st_mtime))


# -- remote sync ---------------------------------------------------------------------------------------------

HOST_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.@-]*$")
PATH_RE = re.compile(r"^[A-Za-z0-9_./~@+=,:-]+$")
SSH_CMD = "ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=10 -o ServerAliveCountMax=3"
LIST_LINE = re.compile(r"^[-dl][rwxsStT-]{9}\s+([\d,]*)\s+(\d{4}/\d\d/\d\d \d\d:\d\d:\d\d)\s+(\S.*)$")


def parse_remote_spec(spec: str):
    """'host:path' -> (host, path). ``host`` may be user@host; the characters allowed in both are restricted so that
    neither can turn into an ssh/rsync option or a shell fragment on the remote side."""
    host, sep, path = spec.partition(":")
    if not sep or not host or not path:
        raise ValueError("remote must look like HOST:PATH, got %r" % spec)
    if not HOST_RE.match(host):
        raise ValueError("bad host name in %r" % spec)
    if not PATH_RE.match(path) or path.startswith("-"):
        raise ValueError("bad path in %r (allowed: letters, digits and _ . / ~ @ + = , : -)" % spec)
    return host, path


def parse_list_only(text: str):
    """Lines of ``rsync --list-only`` -> [{name, size, mtime}] for ppo/ckpt_*.pt."""
    out = []
    for line in text.splitlines():
        m = LIST_LINE.match(line.strip())
        if not m:
            continue
        name = m.group(3).strip()
        if not re.search(r"(^|/)ckpt_\d+\.pt$", name):
            continue
        try:
            mtime = time.mktime(time.strptime(m.group(2), "%Y/%m/%d %H:%M:%S"))
        except ValueError:
            mtime = None
        out.append(dict(name=name.rsplit("/", 1)[-1], size=int(m.group(1).replace(",", "") or 0), mtime=mtime))
    return sorted(out, key=lambda c: c["name"])


class RemoteSync:
    """Pulls the small files of one remote run into a local cache folder, every ``interval`` seconds.

    Read-only on the remote: rsync is the only thing that runs there (``rsync --server --sender``); nothing is
    written, started or deleted. ``runner`` is injectable for tests (default: subprocess.run)."""

    def __init__(self, spec, cache_root, interval=30.0, runner=None):
        self.spec = spec
        self.host, self.path = parse_remote_spec(spec)
        self.interval = max(float(interval), 5.0)
        self.runner = runner or subprocess.run
        base = self.path.rstrip("/") or "/"
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", "%s__%s" % (self.host, base)).strip("_.")
        self.local = Path(cache_root) / slug
        self.label = "%s @ %s" % (base.rstrip("/").rsplit("/", 1)[-1] or base, self.host)
        self.state = dict(spec=spec, syncing=False, last_attempt=None, last_ok=None, error=None, duration_s=None,
                          count=0, interval_s=self.interval)
        self.ckpts = []
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.stop_flag = threading.Event()
        self.thread = None
        listing = read_json_file(self.local / ".ckpt_listing.json")
        if isinstance(listing, list):
            self.ckpts = listing

    def rsync_cmd(self):
        cmd = ["rsync", "-rtz", "--timeout=30", "--max-size=64m", "-e", SSH_CMD]
        cmd += ["--include=%s" % f for f in SYNC_FILES]
        cmd += ["--include=replays/", "--include=replays/**", "--exclude=*"]
        cmd += ["%s:%s/" % (self.host, self.path.rstrip("/")), str(self.local) + "/"]
        return cmd

    def list_cmd(self):
        cmd = ["rsync", "--list-only", "-r", "--timeout=30", "-e", SSH_CMD,
               "--include=ppo/", "--include=ppo/ckpt_*.pt", "--exclude=*"]
        cmd += ["%s:%s/" % (self.host, self.path.rstrip("/"))]
        return cmd

    def sync_once(self):
        if shutil.which("rsync") is None and self.runner is subprocess.run:
            self._finish(False, "rsync wasn't found on this machine")
            return False
        t0 = time.time()
        with self.lock:
            self.state.update(syncing=True, last_attempt=t0)
        try:
            self.local.mkdir(parents=True, exist_ok=True)
            r = self.runner(self.rsync_cmd(), capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL)
            if r.returncode not in (0, 24):
                msg = (r.stderr or r.stdout or "").strip().splitlines()
                self._finish(False, "rsync exit %s: %s" % (r.returncode, " | ".join(msg[-3:])[-300:]), t0)
                return False
            try:
                r2 = self.runner(self.list_cmd(), capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL)
                if r2.returncode in (0, 24):
                    ck = parse_list_only(r2.stdout or "")
                    with self.lock:
                        self.ckpts = ck
                    (self.local / ".ckpt_listing.json").write_text(json.dumps(ck))
            except (OSError, subprocess.SubprocessError):
                pass                                   # the checkpoint names are a nicety
            self._finish(True, None, t0)
            return True
        except subprocess.TimeoutExpired:
            self._finish(False, "rsync timed out", t0)
        except OSError as exc:
            self._finish(False, "%s: %s" % (type(exc).__name__, exc), t0)
        return False

    def _finish(self, ok, error, t0=None):
        now = time.time()
        with self.lock:
            self.state.update(syncing=False, error=error, duration_s=(now - t0) if t0 else None,
                              count=self.state["count"] + 1, last_attempt=now if t0 is None else self.state["last_attempt"])
            if ok:
                self.state["last_ok"] = now

    def status(self):
        with self.lock:
            return dict(self.state)

    def _loop(self):
        while not self.stop_flag.is_set():
            self.sync_once()
            self.wake.wait(self.interval)
            self.wake.clear()

    def start(self):
        self.thread = threading.Thread(target=self._loop, name="sync-" + self.host, daemon=True)
        self.thread.start()

    def trigger(self):
        self.wake.set()

    def stop(self):
        self.stop_flag.set()
        self.wake.set()


# -- the application -----------------------------------------------------------------------------------------

class Run:
    def __init__(self, path: Path, label: str, kind="local", remote: RemoteSync = None):
        self.path, self.label, self.kind, self.remote = Path(path), label, kind, remote
        self.id = short_id(("remote:%s" % remote.spec) if remote else "local:%s" % self.path.resolve())

    def file(self, name):
        return self.path / name


def looks_like_run(path: Path):
    return path.is_dir() and any((path / m).exists() for m in RUN_MARKERS)


class App:
    def __init__(self, runs=(), remotes=(), replay_dirs=None, cache_dir=None, sync_interval=30.0, auto_sync=True,
                 runner=None, default_run_root=None):
        self.run_dirs = [resolve_path(p) for p in runs]
        self.default_run_root = None if self.run_dirs else (default_run_root or DEFAULT_RUN_ROOT)
        self.replay_dirs = [resolve_path(p) for p in (replay_dirs if replay_dirs else [DEFAULT_REPLAYS])]
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE
        self.remotes = [RemoteSync(s, self.cache_dir, sync_interval, runner) for s in remotes]
        self.auto_sync = auto_sync
        self.started = time.time()
        self.metrics_cache = Cache(32)
        self.summary_cache = Cache(512)
        self.replay_cache = Cache(4)
        self.replay_paths = {}
        self.lock = threading.Lock()

    def start(self):
        if self.auto_sync:
            for r in self.remotes:
                r.start()

    def stop(self):
        for r in self.remotes:
            r.stop()

    # -- runs
    def runs(self):
        found, seen = [], set()

        def add(run):
            if run.id not in seen:
                seen.add(run.id)
                found.append(run)

        roots = list(self.run_dirs)
        if self.default_run_root is not None and self.default_run_root.is_dir():
            roots.append(self.default_run_root)
        for d in roots:
            if looks_like_run(d) or d in self.run_dirs and not d.is_dir():
                add(Run(d, d.name or str(d)))
            elif d.is_dir():
                for sub in sorted(d.iterdir()):
                    if looks_like_run(sub):
                        add(Run(sub, sub.name))
        for r in self.remotes:
            add(Run(r.local, r.label, "remote", r))
        return found

    def default_run_id(self, runs=None):
        """The run the page selects first: the first --remote (the run you are watching), else the newest local one."""
        runs = self.runs() if runs is None else runs
        for r in runs:
            if r.remote is not None:
                return r.id
        best = None
        for r in runs:
            sig = file_sig(r.file("metrics.jsonl")) or file_sig(r.file("train.log")) or file_sig(r.file("config.json"))
            if sig and (best is None or sig[0] > best[0]):
                best = (sig[0], r.id)
        return best[1] if best else (runs[0].id if runs else None)

    def run_by_id(self, rid):
        for r in self.runs():
            if r.id == rid:
                return r
        return None

    def metrics(self, run):
        path = run.file("metrics.jsonl")
        return self.metrics_cache.get(str(path), file_sig(path), lambda: load_metrics(path))

    def config(self, run):
        return read_json_file(run.file("config.json"))

    def ckpts(self, run):
        d = run.file("ppo")
        out = []
        if d.is_dir():
            for f in sorted(d.glob("ckpt_*.pt")):
                try:
                    st = f.stat()
                except OSError:
                    continue
                m = re.search(r"ckpt_(\d+)\.pt$", f.name)
                out.append(dict(name=f.name, round=int(m.group(1)) if m else None, size=st.st_size, mtime=st.st_mtime))
        elif run.remote is not None:
            for c in run.remote.ckpts:
                m = re.search(r"ckpt_(\d+)\.pt$", c["name"])
                out.append(dict(c, round=int(m.group(1)) if m else None))
        return out

    def log_file(self, run):
        for name in LOG_NAMES:
            if run.file(name).is_file():
                return run.file(name)
        logs = sorted(run.path.glob("*.log")) if run.path.is_dir() else []
        return logs[0] if logs else None

    def log_text(self, run, limit=2 << 20):
        f = self.log_file(run)
        if f is None:
            return "", None
        try:
            with f.open("rb") as fh:
                data = fh.read(limit)
        except OSError:
            return "", None
        return data.decode("utf-8", errors="replace"), f

    def parsed_log(self, run):
        f = self.log_file(run)
        sig = file_sig(f) if f else None
        return self.summary_cache.get("log:" + str(f), sig, lambda: parse_train_log(self.log_text(run)[0])) if f \
            else parse_train_log("")

    def stage_of(self, run):
        """The stage of a run: 'ppo' as soon as metrics.jsonl has a round, else read from train.log."""
        m = self.metrics(run)
        if m["records"]:
            return "ppo"
        return log_stage(self.parsed_log(run), run.file("bc_report.json").exists(), False, run.file("bc_data").exists())

    def stage_payload(self, run):
        cfg = self.config(run) or {}
        parsed = self.parsed_log(run)
        stage = self.stage_of(run)
        bc, rc = cfg.get("bc") or {}, cfg.get("run") or {}
        lf = self.log_file(run)
        sig = file_sig(lf) if lf else None
        eta = None
        target = bc.get("collect_decisions")
        if parsed["collect_rate_per_hour"] and target and parsed["collected"] is not None and parsed["collected"] < target:
            eta = (target - parsed["collected"]) / parsed["collect_rate_per_hour"] * 3600.0
        n_ep = len(parsed["epochs"])
        max_ep = bc.get("max_epochs")
        eta_bc = None
        if parsed["epoch_s"] and max_ep and n_ep < max_ep and not parsed["done"]:
            eta_bc = (max_ep - n_ep) * parsed["epoch_s"]
        return dict(id=run.id, stage=stage, label=STAGE_LABELS.get(stage, stage), steps=stage_steps(stage),
                    parsed=parsed, now=time.time(), log_mtime=sig[0] / 1e9 if sig else None,
                    targets=dict(collect_decisions=target, max_epochs=max_ep, patience=bc.get("patience"),
                                 rounds=rc.get("rounds"), collect_eta_s=eta, bc_eta_max_s=eta_bc))

    def run_info(self, run, now=None):
        now = time.time() if now is None else now
        m = self.metrics(run)
        cfg = self.config(run)
        st = run_status(m["records"], 20, None, now, cfg)
        last = m["records"][-1] if m["records"] else {}
        stage = self.stage_of(run)
        lf = self.log_file(run)
        lsig = file_sig(lf) if lf else None
        sync = run.remote.status() if run.remote else None
        mtime = None
        for name in ("metrics.jsonl", "train.log", "bc_report.json", "config.json"):
            sig = file_sig(run.file(name))
            if sig:
                mtime = max(mtime or 0, sig[0] / 1e9)
        return dict(id=run.id, label=run.label, kind=run.kind, path=run.remote.spec if run.remote else str(run.path),
                    stage=stage, stage_label=STAGE_LABELS.get(stage, stage), state=st["state"],
                    rounds=len(m["records"]), round=last.get("round"),
                    decisions_total=st["decisions_total"], last_timestamp=st["last_timestamp"], age_s=st["age_s"],
                    log_mtime=lsig[0] / 1e9 if lsig else None,
                    reward_per_decision=last.get("reward_per_decision"), has_bc=run.file("bc_report.json").exists(),
                    n_ckpts=len(self.ckpts(run)), config_name=(cfg or {}).get("name"),
                    rounds_total=st["rounds_total"], mtime=mtime, sync=sync, bad_lines=m["bad"])

    def metrics_payload(self, run, window=20, target=None):
        m = self.metrics(run)
        cfg = self.config(run)
        recs = m["records"]
        return dict(id=run.id, label=run.label, rounds=slim_records(recs), bad_lines=m["bad"],
                    partial_tail=m["partial_tail"], config=config_brief(cfg),
                    status=run_status(recs, window, target, time.time(), cfg), now=time.time())

    def log_payload(self, run, lines=200):
        f = self.log_file(run)
        if f is None:
            return dict(file=None, lines=[], size=0, mtime=None, truncated=False)
        text, size, cut = tail_lines(f, lines)
        sig = file_sig(f)
        return dict(file=f.name, lines=text, size=size, mtime=sig[0] / 1e9 if sig else None, truncated=cut)

    # -- replays
    def replay_roots(self):
        roots = [(d, d.name or str(d)) for d in self.replay_dirs]
        for r in self.runs():
            roots.append((r.file("replays"), "%s / replays" % r.label))
        return roots

    def list_replays(self):
        items, seen = [], set()
        for root, group in self.replay_roots():
            files = []
            if root.is_file():
                files = [root]
            elif root.is_dir():
                files = sorted(root.rglob("*.jsonl"))
            for f in files:
                if len(items) >= MAX_REPLAY_FILES:
                    break
                try:
                    rel = f.relative_to(root) if root.is_dir() else Path(f.name)
                    if len(rel.parts) > 4:
                        continue
                    key = str(f.resolve())
                except (OSError, ValueError):
                    continue
                if key in seen:
                    continue
                sig = file_sig(f)
                if sig is None:
                    continue
                summary = self.summary_cache.get(key, sig, lambda f=f: replay_summary(f))
                if summary is None:
                    continue
                seen.add(key)
                rid = short_id(key)
                items.append(dict(id=rid, name=str(rel), group=group, size=sig[1], mtime=sig[0] / 1e9, summary=summary,
                                  _path=f))
        items.sort(key=lambda i: (i["group"], -i["mtime"]))
        with self.lock:
            self.replay_paths = {i["id"]: i["_path"] for i in items}
        return items

    def replay_path(self, rid):
        with self.lock:
            p = self.replay_paths.get(rid)
        if p is None or not p.is_file():
            self.list_replays()
            with self.lock:
                p = self.replay_paths.get(rid)
        return p

    def replay_payload(self, rid, hz=1.0):
        path = self.replay_path(rid)
        if path is None:
            return None
        hz = min(max(hz, 0.25), 4.0)
        sig = file_sig(path)
        data = self.replay_cache.get("%s@%.3f" % (path, hz), sig, lambda: convert_replay(path, hz))
        return dict(data, id=rid)

    def info(self):
        return dict(version=VERSION, now=time.time(), started=self.started, sync_interval_s=self.remotes[0].interval
                    if self.remotes else None, remotes=[r.status() | dict(label=r.label, cache=str(r.local))
                                                        for r in self.remotes],
                    run_dirs=[str(d) for d in self.run_dirs] + ([str(self.default_run_root)]
                                                                if self.default_run_root else []),
                    replay_dirs=[str(d) for d in self.replay_dirs], cache_dir=str(self.cache_dir))

    def trigger_sync(self):
        for r in self.remotes:
            r.trigger()


# -- http ----------------------------------------------------------------------------------------------------

ALLOWED_HOSTS = ("127.0.0.1", "localhost", "[::1]")


def make_handler(app: App, verbose=False):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "RLDashboard/" + VERSION

        def log_message(self, fmt, *args):
            if verbose:
                sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

        # -- response helpers
        def send_bytes(self, status, body: bytes, ctype, extra=None):
            headers = {"Content-Type": ctype, "Cache-Control": "no-cache", "X-Content-Type-Options": "nosniff"}
            headers.update(extra or {})
            if len(body) > 1024 and "gzip" in self.headers.get("Accept-Encoding", ""):
                body = gzip.compress(body, 4)
                headers["Content-Encoding"] = "gzip"
            self.send_response(status)
            for k, v in headers.items():
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def send_json(self, obj, status=200):
            self.send_bytes(status, dumps(obj), "application/json; charset=utf-8")

        def fail(self, status, message):
            self.send_json(dict(error=message), status)

        def host_ok(self):
            host = self.headers.get("Host", "")
            name = host.rsplit(":", 1)[0] if not host.endswith("]") else host
            return name in ALLOWED_HOSTS

        # -- routing
        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            try:
                self.route("GET")
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as exc:                         # never let one bad file take the thread down
                try:
                    self.fail(500, "%s: %s" % (type(exc).__name__, exc))
                except OSError:
                    pass

        def do_POST(self):
            try:
                n = int(self.headers.get("Content-Length") or 0)
                if n:
                    self.rfile.read(min(n, 1 << 20))
                self.route("POST")
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as exc:
                try:
                    self.fail(500, "%s: %s" % (type(exc).__name__, exc))
                except OSError:
                    pass

        def route(self, method):
            if not self.host_ok():
                return self.fail(403, "bad Host header")
            url = urlparse(self.path)
            path = unquote(url.path)
            q = {k: v[-1] for k, v in parse_qs(url.query).items()}
            if path in ("/", "/index.html"):
                return self.static("index.html")
            if path.startswith("/static/"):
                return self.static(path[len("/static/"):])
            if not path.startswith("/api/"):
                return self.fail(404, "not found")
            parts = path[len("/api/"):].strip("/").split("/")
            if method == "POST":
                if parts == ["sync"]:
                    app.trigger_sync()
                    return self.send_json(dict(ok=True, remotes=[r.status() for r in app.remotes]), 202)
                return self.fail(405, "POST is only for /api/sync")
            if parts == ["info"]:
                return self.send_json(app.info())
            if parts == ["sync"]:
                return self.send_json(dict(remotes=[dict(r.status(), label=r.label) for r in app.remotes]))
            if parts == ["runs"]:
                now = time.time()
                runs = app.runs()
                return self.send_json(dict(now=now, runs=[app.run_info(r, now) for r in runs],
                                           default=app.default_run_id(runs)))
            if len(parts) == 3 and parts[0] == "run":
                run = app.run_by_id(parts[1])
                if run is None:
                    return self.fail(404, "unknown run")
                what = parts[2]
                if what == "metrics":
                    return self.send_json(app.metrics_payload(run, num(q.get("window"), 20, 1, 5000, int),
                                                              num(q.get("target"), None, 1, 1e15, float)))
                if what == "bc":
                    return self.send_json(dict(report=read_json_file(run.file("bc_report.json"))))
                if what == "config":
                    return self.send_json(dict(config=app.config(run)))
                if what == "stage":
                    return self.send_json(app.stage_payload(run))
                if what == "ckpts":
                    return self.send_json(dict(ckpts=app.ckpts(run), remote=run.remote is not None))
                if what == "log":
                    return self.send_json(app.log_payload(run, num(q.get("lines"), 200, 1, 2000, int)))
                return self.fail(404, "unknown run resource")
            if parts == ["replays"]:
                items = app.list_replays()
                return self.send_json(dict(replays=[{k: v for k, v in i.items() if k != "_path"} for i in items],
                                           dirs=[str(d) for d in app.replay_dirs]))
            if len(parts) == 2 and parts[0] == "replay":
                data = app.replay_payload(parts[1], num(q.get("hz"), 1.0, 0.25, 4.0, float))
                if data is None:
                    return self.fail(404, "unknown replay")
                return self.send_json(data)
            return self.fail(404, "unknown endpoint")

        def static(self, name):
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name.startswith("."):
                return self.fail(404, "not found")
            f = STATIC_DIR / name
            if not f.is_file():
                return self.fail(404, "not found")
            ctype = STATIC_TYPES.get(f.suffix.lower(), "application/octet-stream")
            self.send_bytes(200, f.read_bytes(), ctype)

    return Handler


def num(text, default, lo, hi, cast):
    """A query-string number clamped to [lo, hi]; ``default`` when missing or not a number."""
    if text is None or text == "":
        return default
    try:
        v = cast(float(text)) if cast is int else cast(text)
    except ValueError:
        return default
    if isinstance(v, float) and not math.isfinite(v):
        return default
    return min(max(v, lo), hi)


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def make_server(app: App, port=8765, verbose=False):
    """A server bound to 127.0.0.1 only (port 0 picks a free port)."""
    return Server(("127.0.0.1", port), make_handler(app, verbose))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--run", action="append", default=[], metavar="DIR",
                    help="run directory, or a folder of run directories (repeatable)")
    ap.add_argument("--remote", action="append", default=[], metavar="HOST:PATH",
                    help="remote run directory to rsync into the local cache (repeatable)")
    ap.add_argument("--replays", action="append", default=[], metavar="DIR",
                    help="folder with replay .jsonl files (repeatable; default outputs/engagements)")
    ap.add_argument("--cache-dir", default=None, help="local cache for remote runs (default outputs/rl_dashboard_cache)")
    ap.add_argument("--sync-interval", type=float, default=30.0, help="seconds between remote syncs (default 30)")
    ap.add_argument("--no-sync", action="store_true", help="do not start the remote sync threads")
    ap.add_argument("--verbose", action="store_true", help="log every request")
    args = ap.parse_args(argv)
    for spec in args.remote:
        try:
            parse_remote_spec(spec)
        except ValueError as exc:
            ap.error(str(exc))
    app = App(args.run, args.remote, args.replays or None, args.cache_dir, args.sync_interval, not args.no_sync)
    server = make_server(app, args.port, args.verbose)
    port = server.server_address[1]
    print("RL dashboard: http://127.0.0.1:%d/   (Ctrl-C to stop)" % port, flush=True)
    print("  runs:    %s" % (", ".join(str(r.path) for r in app.runs()) or "(none yet)"), flush=True)
    for r in app.remotes:
        print("  remote:  %s -> %s every %.0f s%s" % (r.spec, r.local, r.interval, "" if app.auto_sync else " (off)"),
              flush=True)
    print("  replays: %s" % ", ".join(str(d) for d in app.replay_dirs), flush=True)
    app.start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.stop()
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
