"""Environment worker: pure Python (no numpy / torch), runs under CPython or PyPy.

    python -m rl.worker --connect HOST:PORT [--name w0]     (authkey from $RL_AUTHKEY)

The learner process (CPython + torch) owns the networks; workers only step environments.
Messages are length-prefixed pickles (protocol 4, plain types only) over
multiprocessing.connection, so workers can run on other nodes (TCP) or other interpreters.

Ops (request payload -> reply):
  init       {env_cls, env_config, env_ids, seeds, streams_per_env, validate_steps, host_id}
             -> {env_id: {"obs": {aid: wire}, "scenario": ..., "kind": ...[, "frozen": [aid, ...]]}}
             (creates and resets the envs)
  step       {env_id: {aid: action tuple}} -> ({env_id: StepResult}, seconds)
  collect_bc {n_episodes, seed, check_legality} -> list of episode records (scripted pilots)
  ping / close
StepResult: {"rew","done": {aid:..}, "timeout": bool, "obs": {aid: wire} of agents that continue
  (or of the fresh episode if "new_episode"), "final": {aid: wire} final observations of agents
  that survived a timeout (bootstrap), "new_episode": bool, "events": {name: int}, "scenario",
  "kind"} where "kind" is env.episode_kind ("self_play" / "history" / "vs_script" for MatchEnv with self_play_prob or
  history_prob, else None) of the episode "obs" belongs to, like "scenario". Optional keys: "late" {aid: reward} owed
  to agents that finished in an earlier step (an env's info["late_rewards"]: a missile of a downed aircraft scored),
  "tallies" {aid: [kills, deaths]} (info["tallies"]), "time_limit" (info["time_limit"]: the match ran out of time,
  also when the env makes that a terminal step rather than a bootstrapped timeout), "pending": True if the episode
  goes on and env.pending_credit() is still True (the sampler keeps stepping such an env before its PPO update:
  Sampler._settle), "frozen" (with "new_episode", and in init) the env's frozen_ids when there are any: agents of a
  history episode flown by a frozen past policy (the sampler acts for them but trains nothing on them), "teacher"
  (also in init / reset results) {aid: {head: option, "name": teacher}} teacher labels of the decisions on "obs"
  (MatchEnv config teacher: info["teacher"] after a step, env.teacher_labels after a reset), only for agents that
  have a label (docs/kickstart_spec.md); absent when there is none.
The env is reset automatically when no policy-controlled agent is left, or (history episodes) when only frozen agents
are left and nothing is owed to a downed trained agent any more (pending_credit() False). If the env has
pending_credit() (missiles of downed policy aircraft still flying) it is first played on without actions until they
end, and what they score is added to the reward of the agents that finished in this step ("late" for agents that
finished earlier).
"""
from __future__ import annotations

import importlib
import os
import pickle
import sys
import time
import traceback

from rl import spec, wire


def load_class(path):
    mod, _, name = path.partition(":")
    if not name:
        raise ValueError("env class must look like 'package.module:Class', got %r" % path)
    return getattr(importlib.import_module(mod), name)


def _labels(labels, obs):
    """Teacher labels (env info["teacher"] / env.teacher_labels) of the agents in ``obs``, as plain dicts."""
    if not labels:
        return {}
    return {aid: dict(lab) for aid, lab in labels.items() if aid in obs and lab}


class EnvHost:
    def __init__(self, env_cls, env_config, env_ids, seeds, streams_per_env, validate_steps=25, host_id=0):
        self.cls = load_class(env_cls)
        self.config = env_config
        self.streams_per_env = streams_per_env
        self.validate_steps = validate_steps
        self.host_id = host_id
        self.envs = {}
        self.checked = {}
        for j, seed in zip(env_ids, seeds):
            self.envs[j] = self.cls(dict(env_config), seed)
            self.checked[j] = 0
        self.bc_env = None
        self.bc_counter = 0

    # ------------------------------------------------------------------ helpers
    def _pack(self, j, obs):
        deep = self.checked.get(j, 1 << 30) < self.validate_steps
        return {aid: wire.pack_obs(o, check_finite=deep) for aid, o in obs.items()}

    def _reset(self, j):
        env = self.envs[j]
        for _ in range(8):
            obs = env.reset()
            if obs:
                break
        else:
            raise RuntimeError("env %d: reset() returned no policy-controlled agent 8 times" % j)
        if len(obs) > self.streams_per_env:
            raise RuntimeError(
                "env %d: reset() returned %d controlled agents but streams_per_env=%d; raise "
                "env.streams_per_env or lower the number of controlled agents" % (j, len(obs), self.streams_per_env))
        out = {"obs": self._pack(j, obs), "scenario": getattr(env, "scenario", None),
               "kind": getattr(env, "episode_kind", None)}
        frozen = getattr(env, "frozen_ids", None)
        if frozen:
            out["frozen"] = list(frozen)
        labels = _labels(getattr(env, "teacher_labels", None), obs)
        if labels:
            out["teacher"] = labels
        return out

    # ------------------------------------------------------------------ ops
    def reset_all(self):
        return {j: self._reset(j) for j in self.envs}

    def step(self, actions):
        out = {}
        t0 = time.time()
        for j, acts in actions.items():
            env = self.envs[j]
            obs, rew, done, info = env.step({aid: wire.unpack_action(t) for aid, t in acts.items()})
            info = info or {}
            timeout = bool(info.get("timeout", False))
            nxt, final = {}, {}
            for aid, o in obs.items():
                if done.get(aid, False):
                    if timeout:
                        final[aid] = o
                else:
                    nxt[aid] = o
            rew = {aid: float(r) for aid, r in rew.items()}
            events = {k: int(v) for k, v in (info.get("events") or {}).items()}
            late = {aid: float(v) for aid, v in (info.get("late_rewards") or {}).items()}
            tallies = {aid: list(v) for aid, v in (info.get("tallies") or {}).items()}
            time_limit = bool(info.get("time_limit", False))
            pending = getattr(env, "pending_credit", None)
            frozen = getattr(env, "frozen_ids", None)
            if nxt and frozen and all(aid in frozen for aid in nxt) and not (pending is not None and pending()):
                nxt = {}    # history episode: only the frozen side is left and nothing more can reach a trained agent
            if not nxt and pending is not None and not getattr(env, "over", True):
                # Every controlled aircraft is down but some of their missiles still fly: finish them before the
                # reset, so a kill after death (a trade) reaches the shooter.
                while not env.over and pending():
                    _, _, _, i2 = env.step({})
                    i2 = i2 or {}
                    for aid, v in (i2.get("late_rewards") or {}).items():
                        late[aid] = late.get(aid, 0.) + float(v)
                    for aid, (k, d) in (i2.get("tallies") or {}).items():
                        t = tallies.setdefault(aid, [0, 0])
                        t[0] += k
                        t[1] += d
                    for k, v in (i2.get("events") or {}).items():
                        events[k] = events.get(k, 0) + int(v)
                for aid in [a for a in late if a in rew]:   # finished in this very step: credit its final reward
                    rew[aid] += late.pop(aid)
            res = {
                "rew": rew,
                "done": {aid: bool(d) for aid, d in done.items()},
                "timeout": timeout,
                "final": self._pack(j, final),
                "events": events,
                "new_episode": False,
                "scenario": getattr(env, "scenario", None),
                "kind": getattr(env, "episode_kind", None),
            }
            if late:
                res["late"] = late
            if tallies:
                res["tallies"] = tallies
            if time_limit:
                res["time_limit"] = True
            self.checked[j] += 1
            if nxt:
                res["obs"] = self._pack(j, nxt)
                if pending is not None and pending():
                    res["pending"] = True
                labels = _labels(info.get("teacher"), nxt)
                if labels:
                    res["teacher"] = labels
            else:
                r = self._reset(j)
                res["obs"] = r["obs"]
                res["new_episode"] = True
                res["scenario"] = r["scenario"]
                res["kind"] = r["kind"]
                if "frozen" in r:
                    res["frozen"] = r["frozen"]
                if "teacher" in r:
                    res["teacher"] = r["teacher"]
            out[j] = res
        return out, time.time() - t0

    def collect_bc(self, n_episodes, seed, check_legality=True):
        if self.bc_env is None:
            self.bc_env = self.cls(dict(self.config), seed)
        env = self.bc_env
        episodes = []
        bad_counts, forced_counts = {}, {}
        for _ in range(n_episodes):
            self.bc_counter += 1
            obs = env.reset()
            if not obs:
                continue
            scenario = getattr(env, "scenario", None)
            agents = {}
            events = {}
            timeout = False
            steps = 0
            while obs:
                acts = env.scripted_actions()
                for aid, o in obs.items():
                    if aid not in acts:
                        raise RuntimeError("scripted_actions() has no entry for alive agent %r" % (aid,))
                    canon = acts[aid]
                    if check_legality:
                        masks = o["masks"] if isinstance(o, dict) else o.masks
                        n = len(o["entities"] if isinstance(o, dict) else o.entities)
                        canon, forced, illegal = spec.canonicalize_action(n, masks, acts[aid])
                        for h in forced:
                            forced_counts[h] = forced_counts.get(h, 0) + 1
                        for h in illegal:
                            bad_counts[h] = bad_counts.get(h, 0) + 1
                    ag = agents.setdefault(aid, {"aircraft": str(o["aircraft"] if isinstance(o, dict) else o.aircraft),
                                                  "steps": [], "ret": 0.0})
                    ag["steps"].append((wire.pack_obs(o), wire.pack_action(canon)))
                steps += 1
                obs, rew, done, info = env.step(acts)
                info = info or {}
                timeout = bool(info.get("timeout", False))
                for k, v in (info.get("events") or {}).items():
                    events[k] = events.get(k, 0) + int(v)
                for aid, r in rew.items():
                    agents[aid]["ret"] += float(r)
                obs = {aid: o for aid, o in obs.items() if not done.get(aid, False)}
            episodes.append({"episode_id": "%d-%d" % (self.host_id, self.bc_counter), "scenario": scenario,
                             "agents": agents, "events": events, "timeout": timeout, "steps": steps})
        return {"episodes": episodes, "illegal": bad_counts, "forced": forced_counts}


def dispatch(state, op, payload):
    if op == "init":
        state["host"] = EnvHost(**payload)
        return state["host"].reset_all()
    if op == "step":
        return state["host"].step(payload)
    if op == "collect_bc":
        return state["host"].collect_bc(**payload)
    if op == "ping":
        return {"pid": os.getpid()}
    raise ValueError("unknown op %r" % op)


def hello_info(name):
    import platform
    return {"name": name, "pid": os.getpid(), "impl": sys.implementation.name,
            "version": platform.python_version(), "host": platform.node(), "byteorder": sys.byteorder}


def serve(conn, name="w"):
    conn.send_bytes(pickle.dumps(("hello", hello_info(name)), wire.PICKLE_PROTOCOL))
    state = {}
    while True:
        try:
            raw = conn.recv_bytes()
        except (EOFError, ConnectionError, OSError):
            break
        op, payload = pickle.loads(raw)
        if op == "close":
            break
        try:
            reply = ("ok", dispatch(state, op, payload))
        except Exception:
            reply = ("err", traceback.format_exc())
        try:
            conn.send_bytes(pickle.dumps(reply, wire.PICKLE_PROTOCOL))
        except (ConnectionError, OSError):
            break


def main(argv=None):
    import argparse
    from multiprocessing.connection import Client
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--connect", required=True, help="HOST:PORT of the learner")
    ap.add_argument("--name", default="w")
    ap.add_argument("--authkey", default="", help="default: $RL_AUTHKEY")
    ap.add_argument("--retries", type=int, default=60, help="connection attempts (1 s apart)")
    a = ap.parse_args(argv)
    host, _, port = a.connect.rpartition(":")
    key = (a.authkey or os.environ.get("RL_AUTHKEY", "")).encode()
    if not key:
        sys.exit("no authkey: set RL_AUTHKEY or pass --authkey")
    conn = None
    for i in range(a.retries):
        try:
            conn = Client((host, int(port)), authkey=key)
            break
        except (ConnectionRefusedError, OSError):
            time.sleep(1.0)
    if conn is None:
        sys.exit("could not connect to %s" % a.connect)
    try:
        serve(conn, a.name)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
