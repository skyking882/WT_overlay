"""Training configuration: dataclasses, presets (smoke / workstation / gpu_cluster) and CLI overrides.

Pure Python (no torch import) so it can be inspected anywhere.
"""
from __future__ import annotations

import ast
import copy
import json
import os
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Dict, List


@dataclass
class EnvCfg:
    cls: str = "rl.fake_env:FakeMatchEnv"      # "module:Class"; real env: "wt_overlay.rl_env:MatchEnv"
    config: Dict[str, Any] = field(default_factory=dict)   # passed to MatchEnv(config, seed)
    streams_per_env: int = 1       # max number of policy-controlled agents per episode of one env
    validate_steps: int = 25       # deep contract check (finite values etc.) on the first N steps per env
    seed: int = 1000


@dataclass
class WorkersCfg:
    n_local: int = 2               # worker processes started by the learner
    python: List[str] = field(default_factory=list)   # worker interpreters, cycled; [] = this python
    n_remote: int = 0              # extra workers that connect from other nodes (python -m rl.worker)
    listen: str = "127.0.0.1:0"    # learner listen address ("0.0.0.0:5555" for remote workers)
    authkey: str = ""              # "" -> $RL_AUTHKEY or random (written to <run_dir>/authkey if n_remote > 0)
    inproc: bool = False           # host the envs inside the learner process (tests / benchmarks)
    startup_timeout_s: float = 180.0
    max_envs_per_worker: int = 0   # 0 = spread evenly


@dataclass
class ResourcesCfg:
    device: str = "auto"           # auto | cpu | cuda | cuda:N
    max_cores: int = 0             # 0 = no cap. >0: CPU-affinity cap (Linux) + budget check
    core_offset: int = 0           # first allowed core index inside the allowed set when capping
    threads_infer: int = 4         # torch threads while sampling (workers run in the meantime)
    threads_update: int = 4        # torch threads during the update (workers idle then)
    interop_threads: int = 1


@dataclass
class RolloutCfg:
    n_streams: int = 64
    steps: int = 320
    seg_len: int = 80
    burn_in: int = 16
    greedy: bool = False


@dataclass
class PPOCfg:
    gamma_base: float = 0.995
    lambda_base: float = 0.95
    clip: float = 0.15
    lr_actor: float = 1e-4
    lr_critic: float = 3e-4
    adam_eps: float = 1e-5
    max_grad_norm: float = 0.5
    epochs: int = 3
    minibatch_segments: int = 16
    target_kl: float = 0.02
    ent_coef: float = 0.01
    ent_coef_end: float = 0.003
    ent_hold: float = 1.0e6        # decisions
    ent_end: float = 5.0e6
    ent_pause_below: float = 0.05  # pause entropy decay when mean normalised entropy falls below this
    # Per-head multiplier on the entropy bonus (head name -> scale, default 1). Rare per-step decisions such as
    # 'weapon' should not be pushed toward random: s1_v2 (ent_coef 0.05) drove weapon entropy 0.06 -> 0.23 and
    # doubled launches per episode, wasting missiles.
    ent_head_scale: dict = field(default_factory=dict)
    kl_beta: float = 0.05
    kl_beta_hold: float = 2.0e5
    kl_beta_end: float = 2.0e6
    value_std_floor: float = 0.05
    critic_warmup_rounds: int = 0  # rounds that update only the critic


@dataclass
class BCCfg:
    collect_decisions: int = 1_000_000
    episodes_per_request: int = 2
    shard_decisions: int = 100_000
    val_frac: float = 0.2
    seg_len: int = 80
    burn_in: int = 16
    batch: int = 32
    lr: float = 3e-4
    grad_clip: float = 0.5
    max_epochs: int = 10
    patience: int = 3
    w_fire: float = 5.0
    w_switch: float = 3.0
    seed: int = 0
    eval_rounds: int = 2           # rollout rounds of the BC policy flying by itself after training
    max_batches_per_epoch: int = 0  # 0 = full pass (smoke tests cap this)


@dataclass
class RunCfg:
    run_dir: str = "runs/smoke"
    seed: int = 0
    rounds: int = 3                # PPO rounds to run in total (resume continues up to this number)
    ckpt_every: int = 1
    keep_ckpts: int = 3
    max_wall_s: float = 0.0        # 0 = unlimited; else stop cleanly (checkpointed) before this wall time
    wall_margin_s: float = 120.0   # stop starting new rounds when remaining wall time < margin + last round
    log_workers: bool = True


@dataclass
class Config:
    name: str = "smoke"
    env: EnvCfg = field(default_factory=EnvCfg)
    workers: WorkersCfg = field(default_factory=WorkersCfg)
    resources: ResourcesCfg = field(default_factory=ResourcesCfg)
    rollout: RolloutCfg = field(default_factory=RolloutCfg)
    ppo: PPOCfg = field(default_factory=PPOCfg)
    bc: BCCfg = field(default_factory=BCCfg)
    run: RunCfg = field(default_factory=RunCfg)

    # ------------------------------------------------------------------ io
    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Config":
        cfg = Config()
        _merge(cfg, d)
        return cfg

    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=1, sort_keys=True)

    @staticmethod
    def load(path: str) -> "Config":
        with open(path) as f:
            return Config.from_dict(json.load(f))

    def validate(self):
        r = self.rollout
        if r.steps % r.seg_len:
            raise ValueError("rollout.steps (%d) must be a multiple of rollout.seg_len (%d)" % (r.steps, r.seg_len))
        if r.burn_in > r.seg_len:
            raise ValueError("rollout.burn_in must not exceed rollout.seg_len")
        if r.n_streams % self.env.streams_per_env:
            raise ValueError("rollout.n_streams must be a multiple of env.streams_per_env")
        e = self.env
        if not isinstance(e.config, dict):
            raise ValueError("env.config must be a dict, got %r (a --set value is parsed with ast.literal_eval: "
                             "write None / True / False, not null / true / false, or use a JSON config file)"
                             % (e.config,))
        p_self = e.config.get("self_play_prob", 0)
        if isinstance(p_self, (int, float)) and p_self > 0:
            # a self-play episode controls every aircraft slot; a bigger episode would stop the worker at some
            # random later reset, so refuse here
            teams = e.config.get("teams")
            slots = sum(len(t) for t in teams) if teams else 2 * e.config.get("team_size", 1)
            if e.streams_per_env < slots:
                raise ValueError("env.config.self_play_prob > 0 controls all %d aircraft slots per episode: "
                                 "env.streams_per_env must be >= %d (is %d)" % (slots, slots, e.streams_per_env))
        check_core_budget(self)


def _merge(obj, d):
    for k, v in d.items():
        if not hasattr(obj, k):
            raise KeyError("unknown config key %r for %s" % (k, type(obj).__name__))
        cur = getattr(obj, k)
        if is_dataclass(cur) and isinstance(v, dict):
            _merge(cur, v)
        else:
            setattr(obj, k, copy.deepcopy(v))


def apply_overrides(cfg: Config, items: List[str]) -> Config:
    """items like ["ppo.clip=0.1", "workers.n_local=8", "env.config={'n_agents': 3}"]."""
    for it in items:
        key, _, val = it.partition("=")
        if not _:
            raise ValueError("override must look like a.b=value, got %r" % it)
        try:
            parsed = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            parsed = val
        parts = key.split(".")
        obj = cfg
        for p in parts[:-1]:
            obj = getattr(obj, p)
        if not hasattr(obj, parts[-1]):
            raise KeyError("unknown config key %r" % key)
        setattr(obj, parts[-1], parsed)
    return cfg


def check_core_budget(cfg: Config):
    """Hard budget: local workers + inference threads must fit in max_cores (if set).

    The update phase has the workers blocked (synchronous rounds), so only threads_update
    must fit on its own. On Linux the cap is enforced with CPU affinity at start-up
    (resources.apply_core_cap), the check here just refuses an inconsistent config.
    """
    r = cfg.resources
    if r.max_cores <= 0:
        return
    w = cfg.workers
    if not w.inproc and w.n_local + r.threads_infer > r.max_cores:
        raise ValueError(
            "core budget exceeded: %d local workers + %d inference threads > max_cores=%d"
            % (w.n_local, r.threads_infer, r.max_cores))
    if r.threads_update > r.max_cores:
        raise ValueError("resources.threads_update=%d > max_cores=%d" % (r.threads_update, r.max_cores))


# ---------------------------------------------------------------------------
# presets
# ---------------------------------------------------------------------------

def preset(name: str) -> Config:
    if name == "smoke":
        c = Config(name="smoke")
        c.workers.n_local = 2
        c.rollout.n_streams = 8
        c.rollout.steps = 24
        c.rollout.seg_len = 8
        c.rollout.burn_in = 4
        c.env.streams_per_env = 2
        c.env.config = {"n_agents": 2, "max_steps": 60}
        c.ppo.minibatch_segments = 8
        c.ppo.ent_hold, c.ppo.ent_end = 400.0, 2000.0
        c.ppo.kl_beta_hold, c.ppo.kl_beta_end = 200.0, 1000.0
        c.bc.collect_decisions = 3000
        c.bc.shard_decisions = 1500
        c.bc.seg_len, c.bc.burn_in, c.bc.batch = 16, 4, 16
        c.bc.max_epochs = 4
        c.bc.eval_rounds = 1
        c.resources.threads_infer = 2
        c.resources.threads_update = 4
        c.run.rounds = 3
        return c
    if name == "workstation":
        # 96-core CPU box, no GPU. HARD CAP: 48 cores in total (learner + workers).
        # 64 streams x 1 agent/env = 64 envs. With W workers the slowest worker steps
        # ceil(64/W) envs per tick: W=32 and W=44 both give 2, so 32 workers (PyPy) are used,
        # leaving 16 cores for the learner. Sampling uses 8 torch threads (inference batches of
        # 64 stop scaling early); during the update the workers are blocked, so 16 threads fit.
        c = Config(name="workstation")
        c.resources.device = "cpu"
        c.resources.max_cores = 48
        c.resources.threads_infer = 8
        c.resources.threads_update = 16
        c.workers.n_local = 32
        c.workers.python = ["~/lb_bundle/pypy/bin/pypy3"]
        c.workers.listen = "127.0.0.1:0"
        c.env.streams_per_env = 1
        c.run.run_dir = "~/rl_runs/workstation"
        c.run.rounds = 250
        c.run.ckpt_every = 2
        return c
    if name == "gpu_cluster":
        # 1 GPU + up to 60 CPU cores; envs on the CPU cores, networks on the GPU.
        c = Config(name="gpu_cluster")
        c.resources.device = "cuda"
        c.resources.max_cores = 60
        c.resources.threads_infer = 4
        c.resources.threads_update = 4
        c.workers.n_local = 56
        c.workers.python = []      # set e.g. ["~/pypy/bin/pypy3"] once PyPy is installed on the node
        c.env.streams_per_env = 1
        c.run.run_dir = "~/rl_runs/gpu_cluster"
        c.run.rounds = 250
        c.run.ckpt_every = 2
        c.run.wall_margin_s = 300.0
        return c
    raise KeyError("unknown preset %r (smoke | workstation | gpu_cluster)" % name)


def expand(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))
