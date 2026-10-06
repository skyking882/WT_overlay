"""Synchronous rollouts: n_streams logical trajectory streams x `steps` ticks per round.

A *stream* is a fixed slot bound to (env instance, controlled-agent slot). The number of
streams is a config constant and does not depend on the number of worker processes. Env
instances are spread over the workers; each env hosts `streams_per_env` streams. When an
agent dies its stream idles (padding) until its env starts a new episode; envs reset
automatically when no controlled agent is left. Streams never mix across envs. The number of controlled agents
may differ per episode (self-play: all slots, else the configured ones): streams_per_env must cover the largest
episode and the unused streams idle (padding) until the env's next reset; valid_fraction shows it.

The behaviour policy is fixed for the whole round: the actor lives in this process, workers
only step envs and receive action tuples.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

import torch

from rl import spec
from rl.buffer import RoundBuffer, finish_round, ref_postpass
from rl.encode import Decoded, StepStore

H = 256


class RoundAborted(Exception):
    """Raised inside collect() when the abort callback fires (SIGTERM during sampling)."""


class Tail:
    """Last P ticks of a finished round, kept as burn-in history for the next round."""

    def __init__(self, buf: RoundBuffer):
        S, P, T = buf.S, buf.P, buf.T
        self.S, self.P = S, P
        self.store = StepStore(P * S)
        if P > 0:
            self.store.copy_from(buf.store, slice(T * S, (T + P) * S), 0)
        self.store.finalize()
        self.h_actor = buf.h_actor[buf.K].clone()       # state entering tick T-B
        self.h_critic = buf.h_critic[buf.K].clone()


class Sampler:
    def __init__(self, cfg, pool, device, seed=0):
        self.cfg = cfg
        self.pool = pool
        self.device = device
        r = cfg.rollout
        self.S, self.T, self.L, self.B = r.n_streams, r.steps, r.seg_len, r.burn_in
        K = cfg.env.streams_per_env
        self.n_envs = self.S // K
        self.env_streams = [list(range(j * K, (j + 1) * K)) for j in range(self.n_envs)]
        self.stream_env = [s // K for s in range(self.S)]
        self.pending: List[Optional[tuple]] = [None] * self.S
        self.pending_first = [False] * self.S
        self.slot_agent = [None] * self.S
        self.ep_ret = [0.0] * self.S
        self.ep_len = [0] * self.S
        self.ep_kill = [0] * self.S     # steps with a kill (reward +1, or -1 for a kill and death together)
        self.ep_death = [0] * self.S
        self.ep_air = [""] * self.S
        self.ep_kind = [spec.KIND_SCRIPT] * self.S
        self.h_actor = torch.zeros(self.S, H, device=device)
        self.h_critic_end = torch.zeros(self.S, H)
        self.h_ref_end = torch.zeros(self.S, H)
        self.tail: Optional[Tail] = None
        self.aircraft_names: List[str] = []
        self._air_id: Dict[str, int] = {}
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed)
        self.started = False

    # ------------------------------------------------------------------ setup
    def start(self, seed_base):
        resets = self.pool.init_envs(self.n_envs, seed_base)
        for j, r in resets.items():
            self._bind(j, r["obs"], r.get("kind"))
        self.started = True

    def _air(self, name):
        i = self._air_id.get(name)
        if i is None:
            i = self._air_id[name] = len(self.aircraft_names)
            self.aircraft_names.append(name)
        return i

    def _bind(self, j, obs, kind=None):
        for s in self.env_streams[j]:
            self.slot_agent[s] = None
            self.pending[s] = None
        for s, a in zip(self.env_streams[j], sorted(obs, key=str)):
            self.slot_agent[s] = a
            self.pending[s] = obs[a]
            self.pending_first[s] = True
            self.ep_ret[s] = 0.0
            self.ep_len[s] = 0
            self.ep_kill[s] = 0
            self.ep_death[s] = 0
            self.ep_air[s] = obs[a][7]
            self.ep_kind[s] = kind or spec.KIND_SCRIPT

    # ------------------------------------------------------------------ one round
    def collect(self, actor, greedy=None, abort=None):
        """Run T ticks with the (fixed) actor. Returns (RoundBuffer, stats).

        abort: optional callable checked every tick; if it returns True RoundAborted is raised and
        the partial round is dropped (the networks have not been touched, so the last checkpoint is
        still current).
        """
        assert self.started
        greedy = self.cfg.rollout.greedy if greedy is None else greedy
        S, T, L, B, P = self.S, self.T, self.L, self.B, self.B
        K = T // L
        dev = self.device
        buf = RoundBuffer(S, T, L, B)
        if self.tail is not None:
            if P > 0:
                buf.store.copy_from(self.tail.store, slice(0, P * S), 0)
            buf.h_actor[0] = self.tail.h_actor.clone().to(dev)   # state entering tick -B (= tick 0 if B == 0)
        else:
            buf.h_actor[0] = torch.zeros(S, H, device=dev)
        actor.eval()
        st = {"t_infer": 0.0, "t_env": 0.0, "t_worker": 0.0, "episodes": [], "events": {}, "lost": 0,
              "mask_fallbacks": 0, "decisions_by_kind": {}}
        for t in range(T):
            if abort is not None and abort():
                raise RoundAborted()
            if (t + B) % L == 0 and t + B >= L:
                buf.h_actor[(t + B) // L] = self.h_actor.clone()
            active = [s for s in range(S) if self.pending[s] is not None]
            if not active:
                raise RuntimeError("no active stream at tick %d (all envs returned empty resets?)" % t)
            for s in active:
                st["decisions_by_kind"][self.ep_kind[s]] = st["decisions_by_kind"].get(self.ep_kind[s], 0) + 1
            t0 = time.time()
            act_t = torch.tensor(active)
            dec = Decoded([self.pending[s] for s in active])
            first = torch.tensor([self.pending_first[s] for s in active])
            batch = dec.to_batch(first=first).to(dev)
            with torch.no_grad():
                out, h_new = actor.act(batch, self.h_actor[act_t.to(dev)], self.gen, greedy)
            self.h_actor[act_t.to(dev)] = h_new
            actions = out.actions.cpu()
            st["mask_fallbacks"] += int(out.fallback.sum())
            flat = (t + P) * S + act_t
            buf.store.put(flat, dec, first=first, act=actions)
            buf.logp[flat] = out.logp.sum(-1).cpu()
            buf.aircraft[flat] = torch.tensor([self._air(n) for n in dec.aircraft])
            by_env: Dict[int, dict] = {}
            acts_l = actions.tolist()
            for i, s in enumerate(active):
                by_env.setdefault(self.stream_env[s], {})[self.slot_agent[s]] = tuple(acts_l[i])
                self.pending_first[s] = False
            st["t_infer"] += time.time() - t0
            t0 = time.time()
            res = self.pool.step(by_env)
            st["t_env"] += time.time() - t0
            st["t_worker"] += self.pool.last_worker_time
            self._process(res, t, buf, st)
        if B == 0:
            buf.h_actor[K] = self.h_actor.clone()
        for s in range(S):
            if self.pending[s] is not None:
                buf.end_obs[s] = self.pending[s]
                buf.end_first[s] = self.pending_first[s]
        buf.store.finalize()
        st["n_valid"] = buf.n_valid()
        return buf, st

    def _process(self, res, t, buf, st):
        S, P = self.S, self.B
        for j, r in res.items():
            for k, v in r["events"].items():
                st["events"][k] = st["events"].get(k, 0) + v
            for s in self.env_streams[j]:
                a = self.slot_agent[s]
                if a is None:
                    continue
                flat = (t + P) * S + s
                rew = r["rew"].get(a)
                if rew is None:
                    raise RuntimeError("env %d returned no reward for agent %r" % (j, a))
                buf.reward[flat] = rew
                self.ep_ret[s] += rew
                self.ep_len[s] += 1
                # kill +1, death -2, assist +0.3: a death step is <= -0.6; a kill step is >= 0.9 or a kill+death -1(-0.7)
                if rew <= -0.6:
                    self.ep_death[s] += 1
                if rew >= 0.9 or -1.1 <= rew <= -0.6:
                    self.ep_kill[s] += 1
                d = r["done"].get(a, False)
                if d:
                    if r["timeout"] and a in r["final"]:
                        buf.trunc[flat] = True
                        buf.boot_final[flat] = True
                        buf.boot_obs.append((flat, r["final"][a]))
                        kind = "timeout"
                    else:
                        buf.done[flat] = True
                        kind = "terminal"
                    self._end_episode(s, kind, st)
                elif (not r["new_episode"]) and a in r["obs"]:
                    self.pending[s] = r["obs"][a]
                else:                           # agent vanished without done: truncate, bootstrap from own value
                    buf.trunc[flat] = True
                    st["lost"] += 1
                    self._end_episode(s, "lost", st)
            if r["new_episode"]:
                self._bind(j, r["obs"], r.get("kind"))

    def _end_episode(self, s, kind, st):
        st["episodes"].append((self.ep_air[s], self.ep_ret[s], self.ep_len[s], kind))
        k, d = self.ep_kill[s], self.ep_death[s]
        outcome = ("trade" if k else "loss") if d else ("win" if k else "none")
        st.setdefault("outcomes", []).append((self.ep_air[s], outcome, k, d, self.ep_kind[s]))
        self.slot_agent[s] = None
        self.pending[s] = None

    # ------------------------------------------------------------------ post-pass
    def finish(self, buf, critic, ref, gamma_base, lambda_base):
        """Critic values, GAE and reference distributions for a collected round."""
        prefix = self.tail.h_critic if self.tail is not None else torch.zeros(self.S, H)
        self.h_critic_end = finish_round(buf, critic, self.h_critic_end, prefix, self.device, gamma_base, lambda_base)
        if ref is not None:
            self.h_ref_end = ref_postpass(buf, ref, self.h_ref_end, self.device)
        self.tail = Tail(buf)
