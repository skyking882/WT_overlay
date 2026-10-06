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

Settle ticks: when an env still owes late credit after the T ticks (a downed policy aircraft's missile is still
flying while another controlled agent of the env lives on: self-play), the round goes on for those envs only, with
the same actor, L ticks at a time, until none of them owes credit (at most settle_cap ticks). Their agents' steps are
stored in the extended round like any other, so the late reward reaches the downed agent's final step before the
update that trains on it. Streams of the other envs idle there; their running episodes bootstrap at tick T-1.

History episodes (MatchEnv history_prob; the worker reports the env's frozen_ids as "frozen"): every slot is bound to
a stream as in self-play, but the agents of the frozen side get their actions from an opponent of the league (a frozen
past actor, drawn per episode by league.pick()), with their own recurrent state (h_frozen) and generator. Their steps
are never written to the round buffer: those slots stay padding (valid False, no reward, done or bootstrap), so they
reach no loss, GAE, value target or statistic; their rewards and late rewards are dropped. The current agent of the
episode trains as in any other episode, late credit and settle ticks included.
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
    def __init__(self, cfg, pool, device, seed=0, league=None):
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
        self.frozen = [False] * self.S      # stream bound to an agent of the frozen side of a history episode
        self.env_frozen = [frozenset()] * self.n_envs
        self.env_opp = [None] * self.n_envs     # (name, actor) flying the frozen side of env j's history episode
        self.league = league                # opponent pool with pick() -> (name, actor); needed for history episodes
        self.h_frozen = torch.zeros(self.S, H, device=device)
        self.gen_frozen = torch.Generator(device=device)
        self.gen_frozen.manual_seed(seed + 7919)
        self.finished = {}                  # (env, agent) -> final step of an agent whose env episode still runs
        self.env_pending = [False] * self.n_envs    # env still owes late credit (worker "pending")
        self.settle_cap = self.T            # max settle ticks per round; 0 = off (late credit after the round dropped)
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
            self._bind(j, r["obs"], r.get("kind"), r.get("frozen"))
        self.started = True

    def _air(self, name):
        i = self._air_id.get(name)
        if i is None:
            i = self._air_id[name] = len(self.aircraft_names)
            self.aircraft_names.append(name)
        return i

    def _bind(self, j, obs, kind=None, frozen=None):
        fz = self.env_frozen[j] = frozenset(frozen or ())
        if fz and self.league is None:
            raise RuntimeError("env %d reports frozen agents (a history episode) but the sampler has no league: "
                               "history_prob > 0 needs an opponent pool (rl.league)" % j)
        self.env_opp[j] = self.league.pick() if fz else None
        for s in self.env_streams[j]:
            self.slot_agent[s] = None
            self.pending[s] = None
            self.frozen[s] = False
        for s, a in zip(self.env_streams[j], sorted(obs, key=str)):
            self.frozen[s] = a in fz
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
        """Run T ticks (plus settle ticks, see _settle) with the (fixed) actor. Returns (RoundBuffer, stats).

        abort: optional callable checked every tick; if it returns True RoundAborted is raised and
        the partial round is dropped (the networks have not been touched, so the last checkpoint is
        still current).
        """
        assert self.started
        greedy = self.cfg.rollout.greedy if greedy is None else greedy
        S, T, L, B, P = self.S, self.T, self.L, self.B, self.B
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
            if self._tick(actor, greedy, abort, t, buf, st, range(S)) is None:
                raise RuntimeError("no active stream at tick %d (all envs returned empty resets?)" % t)
        self._settle(actor, greedy, abort, buf, st)
        if B == 0:
            buf.h_actor[buf.K] = self.h_actor.clone()
        for s in range(S):
            if self.pending[s] is not None and not self.frozen[s]:
                buf.end_obs[s] = self.pending[s]
                buf.end_first[s] = self.pending_first[s]
        buf.store.finalize()
        st["n_valid"] = buf.n_valid()
        return buf, st

    def _tick(self, actor, greedy, abort, t, buf, st, streams):
        """One decision of every stream in `streams` that has an observation; returns the step results (None if no
        stream acted)."""
        if abort is not None and abort():
            raise RoundAborted()
        S, L, B, P, dev = self.S, self.L, self.B, self.B, self.device
        if (t + B) % L == 0 and t + B >= L:
            buf.h_actor[(t + B) // L] = self.h_actor.clone()
        active = [s for s in streams if self.pending[s] is not None]
        if not active:
            return None
        learn = [s for s in active if not self.frozen[s]]
        for s in learn:
            st["decisions_by_kind"][self.ep_kind[s]] = st["decisions_by_kind"].get(self.ep_kind[s], 0) + 1
        t0 = time.time()
        by_env: Dict[int, dict] = {}
        if learn:
            act_t = torch.tensor(learn)
            dec = Decoded([self.pending[s] for s in learn])
            first = torch.tensor([self.pending_first[s] for s in learn])
            batch = dec.to_batch(first=first).to(dev)
            with torch.no_grad():
                out, h_new = actor.act(batch, self.h_actor[act_t.to(dev)], self.gen, greedy)
            self.h_actor[act_t.to(dev)] = h_new
            actions = out.actions.cpu()
            st["mask_fallbacks"] += int(out.fallback.sum())
            flat = (t + P) * S + act_t
            buf.store.put(flat, dec, first=first, act=actions)
            buf.logp[flat] = out.logp.sum(-1).cpu()
            buf.logp_heads[flat] = out.logp.cpu()
            buf.aircraft[flat] = torch.tensor([self._air(n) for n in dec.aircraft])
            acts_l = actions.tolist()
            for i, s in enumerate(learn):
                by_env.setdefault(self.stream_env[s], {})[self.slot_agent[s]] = tuple(acts_l[i])
                self.pending_first[s] = False
        if len(learn) < len(active):
            self._act_frozen([s for s in active if self.frozen[s]], greedy, by_env, st)
        st["t_infer"] += time.time() - t0
        t0 = time.time()
        res = self.pool.step(by_env)
        st["t_env"] += time.time() - t0
        st["t_worker"] += self.pool.last_worker_time
        self._process(res, t, buf, st)
        return res

    def _act_frozen(self, streams, greedy, by_env, st):
        """Actions of the frozen side of history episodes: one batch per opponent actor, own state and generator;
        nothing is stored."""
        dev = self.device
        groups: Dict[int, list] = {}
        for s in streams:
            groups.setdefault(id(self.env_opp[self.stream_env[s]][1]), []).append(s)
        for ss in groups.values():
            opp = self.env_opp[self.stream_env[ss[0]]][1]
            idx = torch.tensor(ss).to(dev)
            batch = Decoded([self.pending[s] for s in ss]).to_batch(
                first=torch.tensor([self.pending_first[s] for s in ss])).to(dev)
            with torch.no_grad():
                out, h_new = opp.act(batch, self.h_frozen[idx], self.gen_frozen, greedy)
            self.h_frozen[idx] = h_new
            for i, a in enumerate(out.actions.cpu().tolist()):
                by_env.setdefault(self.stream_env[ss[i]], {})[self.slot_agent[ss[i]]] = tuple(a)
                self.pending_first[ss[i]] = False
        st["frozen_decisions"] = st.get("frozen_decisions", 0) + len(streams)

    def _settle(self, actor, greedy, abort, buf, st):
        """Settle ticks (module docstring): extend the round by L ticks at a time and step only the envs that owe
        late credit, until none does at the end of a block. An env that resets meanwhile (the worker settled it)
        stops; its new episode starts next round."""
        envs = {j for j in range(self.n_envs) if self.env_pending[j]}
        if not envs or self.settle_cap <= 0:
            return
        S, L, T0 = self.S, self.L, buf.T
        for s in range(S):
            # Idle from tick T0 on with the episode going on: bootstrap at T0-1 from the next observation, the value
            # the end of the round would have given (GAE stops there as it does at the round end).
            if self.stream_env[s] not in envs and self.pending[s] is not None and not self.pending_first[s] \
                    and not self.frozen[s]:
                f = buf.flat(T0 - 1, s)
                buf.trunc[f] = buf.boot_final[f] = True
                buf.boot_obs.append((f, self.pending[s]))
        st["settle_envs"] = len(envs)
        while envs and buf.T - T0 < self.settle_cap:
            buf.extend(L)
            for t in range(buf.T - L, buf.T):
                res = self._tick(actor, greedy, abort, t, buf, st, [s for j in sorted(envs) for s in self.env_streams[j]])
                envs -= {j for j, r in (res or {}).items() if r["new_episode"]}
            envs = {j for j in envs if self.env_pending[j]}
        st["settle_ticks"] = buf.T - T0
        st["settle_decisions"] = int(buf.store.valid[(T0 + self.B) * S:].sum())

    def _process(self, res, t, buf, st):
        S, P = self.S, self.B
        for j, r in res.items():
            for k, v in r["events"].items():
                st["events"][k] = st["events"].get(k, 0) + v
            self.env_pending[j] = bool(r.get("pending"))
            tallies = r.get("tallies")
            # Rewards owed to agents that finished earlier in this env's episode (a missile of a downed aircraft
            # scored): added to their final step if that step is in this round's buffer (settle ticks keep it there;
            # still dropped past settle_cap, or for credit the env does not report as pending).
            for a, v in (r.get("late") or {}).items():
                if a in self.env_frozen[j]:             # the frozen side of a history episode trains nothing
                    st["late_frozen"] = st.get("late_frozen", 0) + 1
                    continue
                f = self.finished.get((j, a))
                if f is None or f["buf"] is not buf:
                    st["late_dropped"] = st.get("late_dropped", 0) + 1
                    continue
                buf.reward[f["flat"]] += v
                air, ret, ln, kind, ek = st["episodes"][f["ep"]]
                st["episodes"][f["ep"]] = (air, ret + v, ln, kind, ek)
                kills = tallies.get(a, (0, 0))[0] if tallies is not None else int(v >= 0.9)
                if kills:
                    o = st["outcomes"][f["out"]]
                    k2 = o[2] + kills
                    st["outcomes"][f["out"]] = (o[0], "trade" if o[3] else "win", k2, o[3]) + tuple(o[4:])
                st["late_credited"] = st.get("late_credited", 0) + 1
            for s in self.env_streams[j]:
                a = self.slot_agent[s]
                if a is None:
                    continue
                if self.frozen[s]:                      # acted for the frozen side: nothing stored, nothing counted
                    if (not r["new_episode"]) and a in r["obs"] and not r["done"].get(a, False):
                        self.pending[s] = r["obs"][a]
                    else:
                        self.slot_agent[s] = None
                        self.pending[s] = None
                    continue
                flat = (t + P) * S + s
                rew = r["rew"].get(a)
                if rew is None:
                    raise RuntimeError("env %d returned no reward for agent %r" % (j, a))
                buf.reward[flat] = rew
                self.ep_ret[s] += rew
                self.ep_len[s] += 1
                if tallies is not None:                 # the env counts kills and deaths itself
                    k, dd = tallies.get(a, (0, 0))
                    self.ep_kill[s] += k
                    self.ep_death[s] += dd
                else:
                    # kill +1, death -2, assist +0.3: a death step is <= -0.6; a kill step is >= 0.9 or a kill+death
                    # -1(-0.7)
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
                        kind = "timeout" if r.get("time_limit") else "terminal"   # time limit as a terminal step
                    self._end_episode(s, kind, st, (j, a, flat, buf))
                elif (not r["new_episode"]) and a in r["obs"]:
                    self.pending[s] = r["obs"][a]
                else:                           # agent vanished without done: truncate, bootstrap from own value
                    buf.trunc[flat] = True
                    st["lost"] += 1
                    self._end_episode(s, "lost", st)
            if r["new_episode"]:
                for key in [key for key in self.finished if key[0] == j]:
                    del self.finished[key]
                self._bind(j, r["obs"], r.get("kind"), r.get("frozen"))

    def _end_episode(self, s, kind, st, where=None):
        st["episodes"].append((self.ep_air[s], self.ep_ret[s], self.ep_len[s], kind, self.ep_kind[s]))
        k, d = self.ep_kill[s], self.ep_death[s]
        outcome = ("trade" if k else "loss") if d else ("win" if k else "none")
        st.setdefault("outcomes", []).append((self.ep_air[s], outcome, k, d, self.ep_kind[s]))
        opp = self.env_opp[self.stream_env[s]]
        if opp is not None:                     # history episode: which frozen opponent this result was against
            st.setdefault("opponent_of", {})[len(st["outcomes"]) - 1] = opp[0]
        if where is not None:                   # (env, agent, final flat step, buffer): target of late rewards
            j, a, flat, buf = where
            self.finished[(j, a)] = {"buf": buf, "flat": flat, "ep": len(st["episodes"])-1,
                                     "out": len(st["outcomes"])-1}
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
