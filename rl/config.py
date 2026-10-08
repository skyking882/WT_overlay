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

from rl.spec import HEAD_NAMES

# Aircraft kept out of training for the generalisation test (docs/rl_design.md section 5: JAS 39E, MiG-35). The one
# place of this list: env.held_out = True (training) and rl.eval_replay --held-out (testing) default to it.
HELD_OUT_AIRCRAFT = ("saab_jas39e", "mig_35")


@dataclass
class EnvCfg:
    cls: str = "rl.fake_env:FakeMatchEnv"      # "module:Class"; real env: "wt_overlay.rl_env:MatchEnv"
    config: Dict[str, Any] = field(default_factory=dict)   # passed to MatchEnv(config, seed)
    streams_per_env: int = 1       # max number of policy-controlled agents per episode of one env
    validate_steps: int = 25       # deep contract check (finite values etc.) on the first N steps per env
    seed: int = 1000
    # Held-out aircraft (opt-in; False = no change). True holds out HELD_OUT_AIRCRAFT, a list those ids instead
    # (--set env.held_out=True, --set "env.held_out=['mig_35']"). rl.train resolves it when it loads the config
    # (resolve_held_out): env.config aircraft_pool becomes the match model's pool (or the aircraft_pool already there)
    # without the held-out ids, so config.json and every checkpoint store the explicit pool. MatchEnv draws both
    # teams from aircraft_pool: the policy never meets a held-out aircraft in training, neither as its own nor as an
    # enemy. Test on them with rl.eval_replay --stats --held-out.
    held_out: Any = False


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
    # "stop" (default): the first minibatch whose KL to the behaviour policy exceeds target_kl stops the actor for
    # the rest of the round. "skip": a minibatch above target_kl_skip is left out and the actor goes on; it stops
    # once the mean KL of the applied minibatches would exceed target_kl. In s1_ego "stop" ended most rounds after a
    # few minibatches, on spikes of single heads (vertical, antenna).
    kl_mode: str = "stop"
    target_kl_skip: float = 0.08
    ent_coef: float = 0.01
    ent_coef_end: float = 0.003
    ent_hold: float = 1.0e6        # decisions
    ent_end: float = 5.0e6
    ent_pause_below: float = 0.05  # pause entropy decay when mean normalised entropy falls below this
    # Per-head multiplier on the entropy bonus (head name -> scale, default 1). Rare per-step decisions such as
    # 'weapon' should not be pushed toward random: s1_v2 (ent_coef 0.05) drove weapon entropy 0.06 -> 0.23 and
    # doubled launches per episode, wasting missiles.
    ent_head_scale: dict = field(default_factory=dict)
    # Per-head KL constraint to the behaviour policy (opt-in; {} = off): {head: {"target": t, "coef": c0[, "coef_min",
    # "coef_max"]}}. The actor loss gains coef * KL_head (k3 estimate per valid step, from buf.logp_heads and the
    # current log-probs); after each round coef is multiplied by 1.5 if the round's mean KL_head > 1.5 * target and
    # divided by 1.5 if < target / 1.5 (clamped, default [0.01, 100]); the adapted coef is in the trainer state.
    # Suggested for 'weapon': target 2e-4 (s1_ego2's per-minibatch weapon k3 ran ~2e-4..6e-4, the joint KL ~6e-3).
    head_kl: dict = field(default_factory=dict)
    # Per-head entropy floor (opt-in; {} = off): {head: {"floor": f[, "up": 1.5, "down": 1.5, "max_scale": 30.0]}}.
    # The head's entropy-bonus scale becomes ent_head_scale x m_h, m_h adapted between rounds (start 1, fixed within
    # a round, kept in the trainer state): if the round's entropy_head[head] (mean normalised entropy over steps with
    # > 1 legal option) < f, m_h *= up (at most max_scale); if > 2 f, m_h /= down (at least 1). For heads that collapse
    # and never come back: 4v4 'vertical' fell to ~0.001 (1v1 ~0.02) and a fixed 10x scale did nothing in 7 rounds.
    ent_floor: dict = field(default_factory=dict)
    # Teacher imitation, "kickstart" (opt-in; None = off, docs/kickstart_spec.md): {"coef": 0.5, "decay_rounds": 40,
    # "start_round": None}. The actor loss gains coef_t x the mean over labelled steps of -log pi(label) (env.config
    # teacher labels, only where the label is legal under the stored masks); coef_t falls linearly from coef to 0 over
    # decay_rounds rounds from start_round (None: the first round with it configured, kept in the trainer state).
    # Opt-in inside it: "tiers" [[coef, share], ...] instead of "coef" (each actor minibatch draws its tier), "adapt"
    # {"skip_target", "step", "min_scale", "max_scale"} (a scale on the tier coefs and the shares follow the KL skips).
    kickstart: Any = None
    kl_beta: float = 0.05
    kl_beta_hold: float = 2.0e5
    kl_beta_end: float = 2.0e6
    value_std_floor: float = 0.05
    critic_warmup_rounds: int = 0  # rounds that update only the critic


@dataclass
class LeagueCfg:
    # Opponent pool of history episodes (env.config history_prob > 0; unused otherwise), rl.league.
    snapshot_every: int = 10       # rounds between snapshots of the current actor (0: references only)
    keep: int = 8                  # newest snapshots kept in the pool
    references: List[str] = field(default_factory=list)   # fixed checkpoints, always in the pool


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
    league: LeagueCfg = field(default_factory=LeagueCfg)
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
        if self.ppo.kl_mode not in ("stop", "skip"):
            raise ValueError("ppo.kl_mode must be 'stop' or 'skip', got %r" % (self.ppo.kl_mode,))
        if self.ppo.kl_mode == "skip" and self.ppo.target_kl_skip < self.ppo.target_kl:
            raise ValueError("ppo.target_kl_skip must not be below ppo.target_kl")
        e = self.env
        if not isinstance(e.config, dict):
            raise ValueError("env.config must be a dict, got %r (a --set value is parsed with ast.literal_eval: "
                             "write None / True / False, not null / true / false, or use a JSON config file)"
                             % (e.config,))
        ids = held_out_ids(e.held_out)          # ValueError on a malformed value
        if ids:
            clash = sorted({m.get("aircraft") if isinstance(m, dict) else m
                            for t in e.config.get("teams") or () for m in t} & set(ids))
            if clash:
                raise ValueError("env.held_out: env.config.teams fly held-out aircraft %s" % clash)
            pool = e.config.get("aircraft_pool")
            if pool is None:
                raise ValueError("env.held_out is set but env.config has no aircraft_pool: resolve it with "
                                 "rl.config.resolve_held_out (rl.train does when it loads the config)")
            if set(pool) & set(ids):
                raise ValueError("env.config.aircraft_pool contains held-out aircraft %s"
                                 % sorted(set(pool) & set(ids)))
        for key in ("self_play_prob", "history_prob"):
            p_all = e.config.get(key, 0)
            if isinstance(p_all, (int, float)) and p_all > 0:
                # self-play and history episodes control every aircraft slot; a bigger episode would stop the worker
                # at some random later reset, so refuse here
                teams = e.config.get("teams")
                slots = sum(len(t) for t in teams) if teams else 2 * e.config.get("team_size", 1)
                if e.streams_per_env < slots:
                    raise ValueError("env.config.%s > 0 controls all %d aircraft slots per episode: "
                                     "env.streams_per_env must be >= %d (is %d)"
                                     % (key, slots, slots, e.streams_per_env))
        if sum(e.config.get(k, 0) for k in ("self_play_prob", "history_prob")
               if isinstance(e.config.get(k, 0), (int, float))) > 1 + 1e-9:
            raise ValueError("env.config: self_play_prob + history_prob must not exceed 1")
        lg = self.league
        if lg.snapshot_every < 0 or lg.keep < 1 or not isinstance(lg.references, list):
            raise ValueError("league: snapshot_every >= 0, keep >= 1 and references a list are required")
        if history_on(self) and lg.snapshot_every == 0 and not lg.references:
            raise ValueError("env.config.history_prob > 0 needs opponents: league.snapshot_every > 0 or "
                             "league.references")
        for h, v in self.ppo.head_kl.items():
            if h not in HEAD_NAMES:
                raise ValueError("ppo.head_kl: unknown head %r" % (h,))
            if not isinstance(v, dict) or set(v) - {"target", "coef", "coef_min", "coef_max"} or \
                    not v.get("target", 0) > 0 or v.get("coef", 1.0) < 0 or \
                    not 0 < v.get("coef_min", 0.01) <= v.get("coef_max", 100.0):
                raise ValueError("ppo.head_kl[%r] must look like {'target': t > 0, 'coef': c >= 0[, 'coef_min', "
                                 "'coef_max']}, got %r" % (h, v))
        ent_floor_spec(self.ppo.ent_floor)
        kickstart_spec(self.ppo.kickstart)
        check_core_budget(self)


def ent_floor_spec(d) -> Dict[str, tuple]:
    """ppo.ent_floor -> {head: (floor, up, down, max_scale)}. ValueError for an unknown head or a bad entry: floor in
    (0, 1), up and down > 1, max_scale >= 1 (defaults 1.5, 1.5, 30)."""
    if not isinstance(d, dict):
        raise ValueError("ppo.ent_floor must be a dict {head: {'floor': f, ...}}, got %r" % (d,))
    unknown = set(d) - set(HEAD_NAMES)
    if unknown:
        raise ValueError("ppo.ent_floor has unknown heads: %s" % sorted(unknown))
    out = {}
    for h, v in d.items():
        ok = isinstance(v, dict) and "floor" in v and not set(v) - {"floor", "up", "down", "max_scale"}
        if ok:
            try:
                f, up, down, mx = (float(v["floor"]), float(v.get("up", 1.5)), float(v.get("down", 1.5)),
                                   float(v.get("max_scale", 30.0)))
                ok = 0.0 < f < 1.0 and up > 1.0 and down > 1.0 and mx >= 1.0
            except (TypeError, ValueError):
                ok = False
        if not ok:
            raise ValueError("ppo.ent_floor[%r] must look like {'floor': 0 < f < 1[, 'up': > 1, 'down': > 1, "
                             "'max_scale': >= 1]}, got %r" % (h, v))
        out[h] = (f, up, down, mx)
    return out


KICKSTART_DEFAULTS = {"coef": 0.5, "decay_rounds": 40, "start_round": None}
KICKSTART_ADAPT_DEFAULTS = {"skip_target": 0.1, "step": 0.8, "min_scale": 0.2, "max_scale": 1.0}


def _num(x):
    return not isinstance(x, bool) and isinstance(x, (int, float)) and x == x and abs(x) < float("inf")


def kickstart_spec(d):
    """ppo.kickstart -> {"coef", "decay_rounds", "start_round", "tiers", "adapt"} with the defaults filled in; None when
    off. ValueError for an unknown key or a bad value.

    coef: float >= 0 (None with tiers); decay_rounds: int >= 1; start_round: None or int >= 0.
    tiers (opt-in, replaces coef, which must then be left out): [[coef >= 0, share >= 0], ...], shares summing to 1
    (+-1e-6) -> a list of (coef, share) tuples; None when absent.
    adapt (opt-in; {} = KICKSTART_ADAPT_DEFAULTS): {"skip_target": in (0, 1), "step": in (0, 1), "min_scale" > 0,
    "max_scale" >= min_scale}; None when absent."""
    if d is None:
        return None
    keys = set(KICKSTART_DEFAULTS) | {"tiers", "adapt"}
    ok = isinstance(d, dict) and not set(d) - keys and not (d.get("tiers") is not None and "coef" in d)
    tiers = adapt = None
    if ok:
        v = dict(KICKSTART_DEFAULTS, **d)
        c, n, s0 = v["coef"], v["decay_rounds"], v["start_round"]
        ok = _num(c) and c >= 0.0 and type(n) is int and n >= 1 and (s0 is None or (type(s0) is int and s0 >= 0))
    if ok and d.get("tiers") is not None:
        t = d["tiers"]
        ok = isinstance(t, (list, tuple)) and len(t) > 0 and all(
            isinstance(x, (list, tuple)) and len(x) == 2 and _num(x[0]) and _num(x[1]) and x[0] >= 0 and x[1] >= 0
            for x in t)
        if ok:
            tiers = [(float(x[0]), float(x[1])) for x in t]
            ok = abs(sum(sh for _, sh in tiers) - 1.0) <= 1e-6
    if ok and d.get("adapt") is not None:
        a = d["adapt"]
        ok = isinstance(a, dict) and not set(a) - set(KICKSTART_ADAPT_DEFAULTS)
        if ok:
            adapt = dict(KICKSTART_ADAPT_DEFAULTS, **a)
            ok = all(_num(x) for x in adapt.values())
            if ok:
                adapt = {k: float(x) for k, x in adapt.items()}
                ok = (0.0 < adapt["skip_target"] < 1.0 and 0.0 < adapt["step"] < 1.0 and adapt["min_scale"] > 0.0
                      and adapt["max_scale"] >= adapt["min_scale"])
    if not ok:
        raise ValueError("ppo.kickstart must be None or look like {'coef': c >= 0 | 'tiers': [[coef >= 0, share >= 0], "
                         "...] with shares summing to 1 (not both), 'decay_rounds': n >= 1, 'start_round': None or a "
                         "round >= 0[, 'adapt': {'skip_target': (0, 1), 'step': (0, 1), 'min_scale': > 0, "
                         "'max_scale': >= min_scale}]}, got %r" % (d,))
    return {"coef": None if tiers is not None else float(c), "decay_rounds": n, "start_round": s0, "tiers": tiers,
            "adapt": adapt}


def held_out_ids(value) -> List[str]:
    """The aircraft ids an env.held_out value holds out: True -> HELD_OUT_AIRCRAFT, a list / tuple -> its ids,
    False / None / [] -> none."""
    if value is None or value is False:
        return []
    if value is True:
        return list(HELD_OUT_AIRCRAFT)
    if isinstance(value, (list, tuple)) and all(isinstance(a, str) for a in value):
        return list(value)
    raise ValueError("env.held_out must be True / False or a list of aircraft ids, got %r" % (value,))


def model_pool(env_config) -> List[str]:
    """Aircraft of the match model an env config uses (inline "model", else model_path / the default file), in the
    model's order. Imports wt_overlay.match only when there is no inline model."""
    model = env_config.get("model")
    if model is None:
        from wt_overlay import match
        model = match.load_model(env_config.get("model_path"))
    return list(model["aircraft_frequency"]["weights"])


def resolve_held_out(cfg: "Config", pool=None) -> "Config":
    """env.held_out -> env.config aircraft_pool: the aircraft_pool already in the config, else the match model's
    aircraft (``pool``, default model_pool), without the held-out ids. Idempotent (a resumed run's config.json
    already holds the pool); nothing changes when no aircraft is held out. ValueError for a held-out id the model does
    not know (a typo would otherwise hold out nothing) and when no aircraft would be left."""
    ids = held_out_ids(cfg.env.held_out)
    if not ids or not isinstance(cfg.env.config, dict):
        return cfg
    pool = list(pool) if pool is not None else model_pool(cfg.env.config)
    unknown = sorted(set(ids) - set(pool))
    if unknown:
        raise ValueError("env.held_out: aircraft %s are not in the match model" % unknown)
    base = cfg.env.config.get("aircraft_pool")
    kept = [a for a in (pool if base is None else base) if a not in ids]
    if not kept:
        raise ValueError("env.held_out leaves no aircraft in the pool")
    cfg.env.config = dict(cfg.env.config, aircraft_pool=kept)
    return cfg


def history_on(cfg) -> bool:
    """History episodes (frozen past opponents from the league) are switched on by env.config history_prob > 0."""
    p = cfg.env.config.get("history_prob", 0) if isinstance(cfg.env.config, dict) else 0
    return isinstance(p, (int, float)) and not isinstance(p, bool) and p > 0


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
