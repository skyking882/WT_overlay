"""Contract checker for an environment (pure Python; runs under CPython or PyPy).

    python -m rl.check_env --env rl.fake_env:FakeMatchEnv [--config '{"n_agents": 3}'] [--episodes 3]
    python -m rl.check_env --env wt_overlay.rl_env:MatchEnv --config '{...}'

Checks docs/rl_training_spec.md section 13 / 13.1 on real rollouts: field shapes and finiteness, mask
structure (table shapes, non-empty selectable rows, legality of the sampled and the scripted actions),
reward / done / info semantics (dead agents disappear, timeouts keep a final observation for the
survivors), dt, scripted_actions() coverage, and snapshot()/restore() determinism.
Exit code 1 if any error was found.
"""
from __future__ import annotations

import argparse
import json
import random
import sys

from rl import spec, wire
from rl.worker import load_class


class Report:
    def __init__(self):
        self.errors, self.warnings, self.stats = [], [], {}

    def err(self, msg):
        if msg not in self.errors:
            self.errors.append(msg)

    def warn(self, msg):
        if msg not in self.warnings:
            self.warnings.append(msg)

    def count(self, key, n=1):
        self.stats[key] = self.stats.get(key, 0) + n


def _masks(o):
    return o["masks"] if isinstance(o, dict) else o.masks


def _field(o, name):
    return o[name] if isinstance(o, dict) else getattr(o, name)


def check_obs(rep, aid, o, where):
    try:
        w = wire.pack_obs(o, check_finite=True)
    except wire.ContractError as e:
        rep.err("%s agent %r: %s" % (where, aid, e))
        return None
    n = w[1]
    masks = _masks(o)
    for bad in spec.empty_selectable_rows(n, masks):
        rep.err("%s: selectable mask row without a True option: %s (13.1 guarantees at least one)" % (where, bad))
    if not masks["target"][n]:
        rep.warn("%s: the 'none' option of target is masked" % where)
    dt = _field(o, "dt")
    if abs(dt - spec.DT_STEP) > 1e-3:
        rep.warn("%s: dt=%.4f differs from 20/48=%.4f" % (where, dt, spec.DT_STEP))
    rep.count("entity_rows", n)
    rep.stats["max_entities"] = max(rep.stats.get("max_entities", 0), n)
    rep.stats["max_truth"] = max(rep.stats.get("max_truth", 0), w[4])
    return w


def random_legal_action(rng, o):
    masks = _masks(o)
    n = len(_field(o, "entities"))
    chosen = {}
    for h in spec.SAMPLE_ORDER:
        m = spec.effective_mask(h, n, masks, chosen)
        chosen[h] = rng.choice([i for i, ok in enumerate(m) if ok])
    return {h: chosen[h] for h in spec.HEAD_NAMES}, n, masks


def run_episode(rep, env, rng, mode, max_steps, where):
    obs = env.reset()
    if not obs:
        rep.err("%s: reset() returned no agent" % where)
        return None
    alive = set(obs)
    steps = 0
    while obs and steps < max_steps:
        steps += 1
        for aid, o in obs.items():
            check_obs(rep, aid, o, "%s step %d" % (where, steps))
        if mode == "script":
            acts = env.scripted_actions()
            if set(acts) != set(obs):
                rep.err("%s: scripted_actions() keys %s != alive agents %s" % (where, sorted(map(str, acts)), sorted(map(str, obs))))
                return None
            for aid, o in obs.items():
                masks = _masks(o)
                n = len(_field(o, "entities"))
                _, forced, illegal = spec.canonicalize_action(n, masks, acts[aid])
                rep.count("scripted_forced_heads", len(forced))
                if illegal:
                    rep.count("scripted_illegal_heads", len(illegal))
                    rep.err("%s: scripted action of %r is illegal at heads %s although several options exist" % (where, aid, illegal))
        else:
            acts = {aid: random_legal_action(rng, o)[0] for aid, o in obs.items()}
        nobs, rew, done, info = env.step(acts)
        rep.count("decisions", len(acts))
        if set(rew) != set(acts) or set(done) != set(acts):
            rep.err("%s: rewards/dones must be keyed exactly by the agents that acted" % where)
        timeout = bool((info or {}).get("timeout", False))
        kind = (info or {}).get("episode_kind")     # MatchEnv with self_play_prob: "self_play" / "vs_script"
        if kind is not None and steps == 1:
            rep.count("episodes_" + str(kind))
            rep.count("controlled_agents_" + str(kind), len(acts))
        ev = (info or {}).get("events")
        if ev is None:
            rep.warn("info['events'] missing (launch/kill/assist/death/dropped_entities counts)")
        else:
            for k in ("launch", "kill", "assist", "death", "dropped_entities"):
                if k not in ev:
                    rep.warn("info['events'] has no %r" % k)
        for aid in acts:
            if done.get(aid) and aid in nobs and not timeout:
                rep.err("%s: agent %r is done (not a timeout) but is still in obs" % (where, aid))
            if (not done.get(aid)) and aid not in nobs:
                rep.err("%s: agent %r not done but missing from obs" % (where, aid))
            if timeout and (not done.get(aid)):
                rep.err("%s: timeout episode but agent %r has done=False" % (where, aid))
            if timeout and done.get(aid) and aid not in nobs and rew.get(aid, 0.0) >= 0:
                rep.warn("%s: timeout without a final observation for surviving agent %r "
                         "(the trainer then bootstraps from the previous step's value)" % (where, aid))
            if done.get(aid) and rew.get(aid, 0.0) <= -1.5:
                rep.count("deaths")
        if timeout:
            rep.count("timeouts")
        obs = {aid: o for aid, o in nobs.items() if not done.get(aid, False)}
        if timeout:
            for aid, o in nobs.items():
                check_obs(rep, aid, o, "%s final obs" % where)
            break
    rep.count("episodes")
    return steps


def check_snapshot(rep, cls, config, seed, rng):
    env = cls(dict(config), seed)
    obs = env.reset()
    for _ in range(5):
        if not obs:
            break
        obs, rew, done, info = env.step(env.scripted_actions())
        obs = {a: o for a, o in obs.items() if not done.get(a)}
    if not obs:
        rep.warn("snapshot check skipped: the episode ended within 5 steps")
        return
    snap = env.snapshot()
    acts = env.scripted_actions()
    a1 = env.step(acts)
    env.restore(snap)
    acts2 = env.scripted_actions()
    a2 = env.step(acts)
    if acts != acts2:
        rep.err("snapshot/restore: scripted_actions() differs after restore()")
    def sig(r):
        obs, rew, done, info = r
        return ({a: wire.pack_obs(o)[:8] for a, o in obs.items()}, rew, done)
    if sig(a1) != sig(a2):
        rep.err("snapshot/restore: stepping the same actions after restore() gives a different result")
    else:
        rep.count("snapshot_checks")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--env", default="rl.fake_env:FakeMatchEnv")
    ap.add_argument("--config", default="{}", help="JSON dict passed to the env")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    config = json.loads(a.config)
    cls = load_class(a.env)
    rep = Report()
    rng = random.Random(a.seed)
    for mode in ("random", "script"):
        env = cls(dict(config), a.seed + 1)
        for ep in range(a.episodes):
            try:
                run_episode(rep, env, rng, mode, a.max_steps, "%s ep%d" % (mode, ep))
            except Exception as e:      # report instead of crashing, the env author wants the message
                rep.err("%s ep%d: exception %s: %s" % (mode, ep, type(e).__name__, e))
                break
    try:
        check_snapshot(rep, cls, config, a.seed + 2, rng)
    except Exception as e:
        rep.err("snapshot check: exception %s: %s" % (type(e).__name__, e))
    print(json.dumps({"stats": rep.stats, "warnings": rep.warnings, "errors": rep.errors}, indent=1, default=str))
    return 1 if rep.errors else 0


if __name__ == "__main__":
    sys.exit(main())
