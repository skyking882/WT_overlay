"""Shared helpers for the rl tests (torch is available in the venv that runs them)."""
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rl import config as C  # noqa: E402
from rl import spec, wire  # noqa: E402
from rl.encode import Decoded  # noqa: E402

# RL_PYPY overrides (e.g. workstation: ~/lb_bundle/pypy/bin/pypy3); otherwise common install paths, then PATH.
PYPY = next((p for p in (os.path.expanduser(os.environ.get("RL_PYPY", "")), "/opt/homebrew/bin/pypy3.11",
                         "/opt/homebrew/bin/pypy3", "/usr/local/bin/pypy3", shutil.which("pypy3") or "")
             if p and os.path.exists(p)), None)


def smoke_cfg(inproc=True, **env_config):
    cfg = C.preset("smoke")
    cfg.workers.inproc = inproc
    cfg.env.config = dict({"n_agents": 2, "max_steps": 40, "target_entities_max": 50, "p_incoming": 0.08}, **env_config)
    return cfg


def rand_nested(rng, shape, p=0.7, p_empty=0.0):
    """Random bool nested list of `shape`; every 1-D row has a True unless p_empty fires."""
    if len(shape) == 1:
        row = [rng.random() < p for _ in range(shape[0])]
        if not any(row) and rng.random() >= p_empty:
            row[rng.randrange(shape[0])] = True
        return row
    return [rand_nested(rng, shape[1:], p, p_empty) for _ in range(shape[0])]


def random_obs(rng, n, p_empty=0.0, p=0.7):
    """A contract-shaped AgentObs dict with random section-13.1 masks."""
    masks = {h: rand_nested(rng, spec.mask_shape(h, n), p, p_empty) for h in spec.HEAD_NAMES}
    return {
        "own": [rng.uniform(-1, 1) for _ in range(spec.OWN_DIM)],
        "entities": [[rng.uniform(-1, 1) for _ in range(spec.ENT_DIM)] for _ in range(n)],
        "prev_intent": [rng.uniform(0, 1) for _ in range(spec.INTENT_DIM)],
        "masks": masks,
        "truth": [[rng.uniform(-1, 1) for _ in range(spec.TRUTH_DIM)] for _ in range(rng.randint(0, 20))],
        "aircraft": "t", "dt": spec.DT_STEP,
    }


def batch_from_obs(obs_list, first=None):
    dec = Decoded([wire.pack_obs(o) for o in obs_list])
    return dec.to_batch(first=first)
