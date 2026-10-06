"""Process-level plumbing: core cap, threads, device, signals, atomic checkpoints, JSONL metrics.

No torch import at module level: `early_setup()` must run before torch is imported so that the
CPU-affinity cap and OMP/MKL thread defaults are in place when torch creates its thread pool.
"""
from __future__ import annotations

import glob
import json
import os
import signal
import threading
import time
from typing import Optional

EXIT_DONE = 0
EXIT_RESUME_ME = 75      # EX_TEMPFAIL: stopped early (wall limit / signal) after checkpointing; run again


def apply_core_cap(max_cores: int, offset: int = 0):
    """Restrict this process (and every child it starts) to max_cores CPUs. Linux only.

    Returns the list of allowed cores, or None when no cap was applied (max_cores<=0 or the OS
    has no sched_setaffinity, e.g. macOS).
    """
    if max_cores <= 0 or not hasattr(os, "sched_setaffinity"):
        return None
    allowed = sorted(os.sched_getaffinity(0))
    chosen = allowed[offset:offset + max_cores]
    if len(chosen) < max_cores and offset:
        chosen = allowed[:max_cores]
    chosen = chosen[:max_cores]
    os.sched_setaffinity(0, set(chosen))
    return chosen


def early_setup(cfg):
    """Call before importing torch: core cap + thread-pool defaults. Returns the allowed cores or None."""
    cores = apply_core_cap(cfg.resources.max_cores, cfg.resources.core_offset)
    n = str(max(cfg.resources.threads_infer, 1))
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, n)
    return cores


def select_device(name: str):
    import torch
    if name in ("auto", ""):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def set_threads(n: int):
    import torch
    torch.set_num_threads(max(1, n))


def init_torch(cfg):
    import torch
    try:
        torch.set_num_interop_threads(cfg.resources.interop_threads)
    except RuntimeError:
        pass
    set_threads(cfg.resources.threads_infer)
    return select_device(cfg.resources.device)


class StopFlag:
    """Set by SIGTERM / SIGINT / SIGUSR1 (Slurm: --signal=B:USR1@600); checked between rounds."""

    def __init__(self):
        self.ev = threading.Event()
        self.reason = ""
        for name in ("SIGTERM", "SIGINT", "SIGUSR1"):
            sig = getattr(signal, name, None)
            if sig is not None:
                try:
                    signal.signal(sig, self._handle)
                except ValueError:       # not the main thread
                    pass

    def _handle(self, signum, frame):
        self.reason = signal.Signals(signum).name
        self.ev.set()

    def is_set(self):
        return self.ev.is_set()


def atomic_torch_save(obj, path):
    import torch
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


def checkpoint_path(ckpt_dir, round_no):
    return os.path.join(ckpt_dir, "ckpt_%06d.pt" % round_no)


def latest_checkpoint(ckpt_dir) -> Optional[str]:
    files = sorted(glob.glob(os.path.join(ckpt_dir, "ckpt_*.pt")))
    return files[-1] if files else None


def prune_checkpoints(ckpt_dir, keep):
    files = sorted(glob.glob(os.path.join(ckpt_dir, "ckpt_*.pt")))
    for f in files[:-keep] if keep > 0 else []:
        try:
            os.remove(f)
        except OSError:
            pass


class MetricsLog:
    """Append-only JSONL, one record per PPO round; resume drops records newer than the checkpoint."""

    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def truncate_after(self, round_no):
        if not os.path.exists(self.path):
            return
        keep = []
        with open(self.path) as f:
            for line in f:
                try:
                    if json.loads(line).get("round", 0) <= round_no:
                        keep.append(line)
                except ValueError:
                    pass
        with open(self.path, "w") as f:
            f.writelines(keep)

    def write(self, rec: dict):
        rec = dict(rec)
        rec["timestamp"] = time.time()
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, sort_keys=True, default=_json_default) + "\n")


def _json_default(o):
    try:
        return float(o)
    except Exception:
        return str(o)


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)
