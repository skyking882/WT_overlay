"""Rewards that arrive after an agent is down (its missile scores later): the worker plays the match on before a
reset, and the sampler adds what arrives later to the agent's final step; settle ticks keep that step in the round.

LateEnv stands in for MatchEnv's reporting: info["late_rewards"], info["tallies"], pending_credit(). DuelEnv is a
scripted 1v1 for exact timing across the round boundary.
"""
import unittest

import common  # noqa: F401
import torch

from rl.fake_env import FakeMatchEnv
from rl.ppo import actor_window
from test_rollout import make


class LateEnv(FakeMatchEnv):
    """Every agent that dies leaves a missile that kills two steps later (+1 owed to it)."""

    def reset(self):
        self.owed = {}
        return super().reset()

    def pending_credit(self):
        return bool(self.owed)

    @property
    def over(self):
        return self.S["over"] and not self.owed

    def _pay(self):
        late = {}
        for aid in list(self.owed):
            self.owed[aid] -= 1
            if self.owed[aid] <= 0:
                del self.owed[aid]
                late[aid] = 1.0
        return late

    def step(self, actions):
        out = self._step(actions)
        self.ledger = getattr(self, "ledger", 0.0) + sum(out[1].values()) + sum(out[3].get("late_rewards", {}).values())
        return out

    def _step(self, actions):
        late = self._pay()
        if not actions:
            info = {"timeout": False, "events": {"kill": len(late)}}
        else:
            obs, rew, done, info = super().step(actions)
        tallies = {aid: [1, 0] for aid in late}
        if actions:
            for aid, r in rew.items():
                k, d = int(r >= 0.9 or -1.1 <= r <= -0.6), int(r <= -0.6)
                if d:
                    self.owed[aid] = 2
                if k or d:
                    t = tallies.setdefault(aid, [0, 0])
                    t[0] += k
                    t[1] += d
        if late:
            info["late_rewards"] = late
        if tallies:
            info["tallies"] = tallies
        if not actions:
            return {}, {}, {}, info
        return obs, rew, done, info


class DuelEnv(FakeMatchEnv):
    """Scripted 1v1 self-play shape where nothing else scores: p0 goes down at step die_at and its missile kills p1 at
    step hit_at (+1 owed to p0, the match ends), or with hit False ends there without a kill and p1 flies on to the
    timeout. Envs with an even seed (every other env: seeds differ by 7919) stay quiet until the timeout. ledger: all
    reward the env handed out."""

    def reset(self):
        obs = super().reset()
        for a in self.S["agents"].values():
            a["missiles"] = 0
        self.quiet = self.seed % 2 == 0
        self.ledger = getattr(self, "ledger", 0.0)
        return obs

    def pending_credit(self):
        return not self.quiet and not self.S["over"] and self.cfg["die_at"] <= self.S["t"] < self.cfg["hit_at"]

    @property
    def over(self):
        return self.S["over"]

    def _down(self, aid, obs, rew, done, info):
        self._kill(aid, rew, done, info["events"])
        obs.pop(aid, None)
        self._cache.pop(aid, None)
        info.setdefault("tallies", {}).setdefault(aid, [0, 0])[1] += 1

    def step(self, actions):
        obs, rew, done, info = super().step(actions)
        t, c = self.S["t"], self.cfg
        if not self.quiet and t == c["die_at"]:
            self._down("p0", obs, rew, done, info)
        if not self.quiet and t == c["hit_at"] and c["hit"]:
            self._down("p1", obs, rew, done, info)
            info["late_rewards"] = {"p0": 1.0}
            info["tallies"]["p0"] = [1, 0]
            self.S["over"] = True
            self._cache = {}
        self.ledger += sum(rew.values()) + sum(info.get("late_rewards", {}).values())
        return obs, rew, done, info


def duel_cfg(die_at, hit_at, hit=True, max_steps=400):
    def fn(cfg):
        cfg.env.cls = "test_late_credit:DuelEnv"
        cfg.env.streams_per_env = 2
        cfg.env.config.update(n_agents=2, min_agents=2, max_steps=max_steps, p_incoming=0.0, p_background_death=0.0,
                              p_match_end=0.0, die_at=die_at, hit_at=hit_at, hit=hit)
    return fn


def envs_of(sm):
    return sm.pool.handles[0].state["host"].envs


def logp_gap(actor, buf):
    """Largest |logp(update windows from the stored states) - logp(rollout)| over the valid loss steps."""
    worst, n = 0.0, 0
    for s in range(buf.S):
        for k in range(buf.K):
            widx = buf.window_index(torch.tensor([s]), torch.tensor([k]))
            b = buf.store.gather(widx)
            out, seg = actor_window(actor, b, buf.h_actor[k][s].unsqueeze(0), buf.B, b.act[:, buf.B:], grad=False)
            m = seg.valid.reshape(-1)
            if m.any():
                worst = max(worst, (out.logp.sum(-1) - buf.logp[widx[:, buf.B:]].reshape(-1)).abs()[m].max().item())
                n += int(m.sum())
    return worst, n


def snapshot(buf, sm):
    d = {"T": buf.T, "boot_obs": list(buf.boot_obs), "end_obs": dict(buf.end_obs), "end_first": dict(buf.end_first),
         "h_critic_end": sm.h_critic_end.clone()}
    for f in ("logp", "logp_heads", "reward", "done", "trunc", "boot_final", "aircraft", "value", "boot_value",
              "end_value", "end_valid", "adv", "ret"):
        d[f] = getattr(buf, f).clone()
    for f in ("own", "prev", "mask", "ent_start", "ent_n", "tr_start", "tr_n", "act", "valid", "first", "dt", "ent_rows",
              "truth_rows"):
        d["store." + f] = getattr(buf.store, f).clone()
    for k in buf.h_actor:
        d["h_actor%d" % k], d["h_critic%d" % k] = buf.h_actor[k].clone(), buf.h_critic[k].clone()
    return d


def late_cfg(n_agents):
    def fn(cfg):
        cfg.env.cls = "test_late_credit:LateEnv"
        cfg.env.streams_per_env = n_agents
        cfg.env.config.update(n_agents=n_agents, min_agents=n_agents, max_steps=40, p_background_death=0.05,
                              p_match_end=0.0)
    return fn


class SamplerCredits(unittest.TestCase):
    def test_one_agent_env_is_played_on_and_the_kill_lands_in_the_death_step(self):
        # vs-script shape: the only controlled agent dies, the worker finishes its missile before the reset.
        cfg, actor, critic, sm = make(streams=4, steps=40, seg=8, burn=4, cfg_fn=late_cfg(1))
        trades, deaths = 0, 0
        for _ in range(3):
            buf, st = sm.collect(actor)
            self.assertEqual(st.get("late_credited", 0), 0)       # nothing left over for the late path
            for o in st["outcomes"]:
                deaths += o[3]
                trades += o[1] == "trade"
                if o[3]:
                    self.assertGreaterEqual(o[2], 1)              # every death left a missile that scored
            for e in st["episodes"]:
                if e[3] == "terminal":
                    self.assertGreater(e[1], -2.0)
        self.assertGreater(deaths, 0)
        self.assertEqual(trades, deaths)

    def test_late_kills_reach_the_final_step_and_turn_losses_into_trades(self):
        cfg, actor, critic, sm = make(streams=8, steps=40, seg=8, burn=4, cfg_fn=late_cfg(2))
        credited, trades = 0, 0
        for _ in range(3):
            buf, st = sm.collect(actor)
            credited += st.get("late_credited", 0)
            trades += sum(1 for o in st["outcomes"] if o[1] == "trade")
            for e in st["episodes"]:
                self.assertEqual(len(e), 5)
        self.assertGreater(credited, 0)
        self.assertGreater(trades, 0)

    def test_random_deaths_lose_no_late_credit_and_count_none_twice(self):
        cfg, actor, critic, sm = make(streams=8, steps=24, seg=8, burn=4, cfg_fn=late_cfg(2))
        total, settled = 0.0, 0
        for _ in range(6):
            buf, st = sm.collect(actor)
            self.assertEqual(st.get("late_dropped", 0), 0)
            settled += st.get("settle_ticks", 0) > 0
            total += float(buf.reward.sum())
        self.assertGreater(settled, 0)
        self.assertAlmostEqual(total, sum(e.ledger for e in envs_of(sm).values()), 4)


class SettleTicks(unittest.TestCase):
    """Late credit owed across the round boundary is settled before the update (Sampler._settle)."""

    def test_death_then_kill_across_the_boundary_is_a_trade_worth_minus_one(self):
        # T = 8: p0 goes down at step 6 (tick 5), its missile kills p1 at step 11 -> settle ticks 8..15 of round 1.
        cfg, actor, critic, sm = make(streams=4, steps=8, seg=8, burn=4, cfg_fn=duel_cfg(6, 11))
        buf, st = sm.collect(actor)
        self.assertEqual((buf.T, st["settle_ticks"], st["settle_envs"]), (16, 8, 1))
        self.assertEqual((st.get("late_dropped", 0), st["late_credited"]), (0, 1))
        lv = buf.loss_view
        rew, done, valid = lv(buf.reward), lv(buf.done), lv(buf.store.valid)
        # p0 (stream 0): last real action at tick 5 carries death -2 and the kill +1
        self.assertEqual(float(rew[5, 0]), -1.0)
        self.assertTrue(bool(done[5, 0]))
        self.assertEqual(float(rew[:, 0].sum()), -1.0)
        self.assertEqual(valid[:, 0].nonzero().flatten().tolist(), list(range(6)))
        # p1 (stream 1) flew on in the settle ticks until the missile hit at tick 10
        self.assertEqual(valid[:, 1].nonzero().flatten().tolist(), list(range(11)))
        self.assertEqual((float(rew[10, 1]), bool(done[10, 1])), (-2.0, True))
        self.assertEqual(st["settle_decisions"], 3)
        eps = sorted((e[1], e[3]) for e in st["episodes"])
        self.assertEqual(eps, [(-2.0, "terminal"), (-1.0, "terminal")])
        outs = sorted(o[1:4] for o in st["outcomes"])
        self.assertEqual(outs, [("loss", 0, 1), ("trade", 1, 1)])
        # the update's return target for p0's last action is the trade (terminal step: return = reward)
        sm.finish(buf, critic, None, 0.995, 0.95)
        self.assertAlmostEqual(float(buf.ret.view(buf.T, buf.S)[5, 0]), -1.0, 6)
        # quiet env (streams 2, 3) idled in the settle ticks and bootstraps at tick 7
        self.assertTrue(bool(lv(buf.trunc)[7, 2]) and bool(lv(buf.trunc)[7, 3]))
        self.assertFalse(bool(valid[8:, 2:].any()))
        # every reward the envs handed out is in the buffers exactly once, over several rounds
        total = float(buf.reward.sum())
        for _ in range(3):
            b, s2 = sm.collect(actor)
            self.assertEqual(s2.get("late_dropped", 0), 0)
            self.assertTrue(all(e[1] == -1.0 for e, o in zip(s2["episodes"], s2["outcomes"]) if o[1] == "trade"))
            total += float(b.reward.sum())
        self.assertAlmostEqual(total, sum(e.ledger for e in envs_of(sm).values()), 5)

    def test_without_settle_ticks_the_kill_is_dropped(self):
        # the old behaviour (settle_cap 0), and a cap shorter than the missile's flight: credit after it is dropped
        for cap, die, hit in ((0, 6, 11), (8, 6, 20)):
            cfg, actor, critic, sm = make(streams=4, steps=8, seg=8, burn=4, cfg_fn=duel_cfg(die, hit))
            sm.settle_cap = cap
            b1, s1 = sm.collect(actor)
            b2, s2 = sm.collect(actor)
            b3, s3 = sm.collect(actor)
            self.assertEqual(b1.T, 8 + cap)
            self.assertEqual(sorted((e[1], o[1]) for e, o in zip(s1["episodes"], s1["outcomes"])), [(-2.0, "loss")])
            self.assertEqual(s1.get("late_dropped", 0) + s2.get("late_dropped", 0) + s3.get("late_dropped", 0), 1)

    def test_a_kill_in_the_same_round_is_counted_once(self):
        # 5-step matches (p0 down at 2, kill at 5) fit 8 times into T = 40; none is open at the end -> no settling
        cfg, actor, critic, sm = make(streams=4, steps=40, seg=8, burn=4, cfg_fn=duel_cfg(2, 5))
        total = 0.0
        for r in range(2):
            buf, st = sm.collect(actor)
            self.assertEqual((buf.T, st.get("settle_ticks", 0), st.get("late_dropped", 0)), (40, 0, 0))
            self.assertEqual(st["late_credited"], 8)
            rew = buf.loss_view(buf.reward)
            self.assertEqual(rew[:, 0].tolist(), [-1.0 if t % 5 == 1 else 0.0 for t in range(40)])
            trades = [e[1] for e, o in zip(st["episodes"], st["outcomes"]) if o[1] == "trade"]
            self.assertEqual(trades, [-1.0] * 8)
            total += float(buf.reward.sum())
        self.assertAlmostEqual(total, sum(e.ledger for e in envs_of(sm).values()), 5)
        self.assertEqual(total, -3.0 * 16)

    def test_trajectories_stay_consistent_across_settle_ticks_and_into_the_next_round(self):
        # p0 down at step 6, its missile misses at step 11: p1 flies through the settle ticks into round 2
        cfgf = duel_cfg(6, 11, hit=False, max_steps=40)
        cfg, actor, critic, sm = make(streams=4, steps=8, seg=8, burn=4, cfg_fn=cfgf)
        _, _, _, ref = make(streams=4, steps=8, seg=8, burn=4, cfg_fn=cfgf)
        ref.settle_cap = 0
        b1, s1 = sm.collect(actor)
        r1, _ = ref.collect(actor)
        sm.finish(b1, critic, None, 0.995, 0.95)
        ref.finish(r1, critic, None, 0.995, 0.95)
        S, P = b1.S, b1.P
        self.assertEqual((b1.T, s1["settle_ticks"]), (16, 8))
        lv = b1.loss_view
        valid, first = lv(b1.store.valid), lv(b1.store.first)
        self.assertTrue(bool(valid[:, 1].all()))                        # p1: every tick, settle ticks included
        self.assertEqual(first[:, 1].nonzero().flatten().tolist(), [0])
        self.assertFalse(bool(lv(b1.done)[:, 1].any() or lv(b1.trunc)[:, 1].any()))
        self.assertTrue(bool(b1.end_valid[1]))                           # p1 bootstraps at the end of the extended round
        # streams of the quiet env: same values, advantages and critic carry as a round without settle ticks
        for name in ("value", "adv", "ret"):
            x, y = getattr(b1, name).view(b1.T, S)[:8, 2:], getattr(r1, name).view(8, S)[:, 2:]
            self.assertLess((x - y).abs().max().item(), 1e-5, name)
        self.assertLess((sm.h_critic_end[2:] - ref.h_critic_end[2:]).abs().max().item(), 1e-5)
        self.assertTrue(torch.equal(sm.h_actor[2:], ref.h_actor[2:]))
        b2, s2 = sm.collect(actor)
        sm.finish(b2, critic, None, 0.995, 0.95)
        T1 = b1.T
        for f in ("own", "valid", "first", "act"):                      # burn-in tail = end of the extended round
            self.assertTrue(torch.equal(getattr(b2.store, f)[:P * S], getattr(b1.store, f)[T1 * S:(T1 + P) * S]), f)
        self.assertTrue(torch.equal(b2.h_actor[0], b1.h_actor[b1.K]))
        self.assertTrue(torch.equal(b2.h_critic[0], b1.h_critic[b1.K]))
        v2, f2 = b2.loss_view(b2.store.valid), b2.loss_view(b2.store.first)
        self.assertEqual(v2[0].tolist(), [False, True, True, True])     # p0 stays down; p1 and the quiet env go on
        self.assertFalse(bool(f2[0].any()))
        for buf in (b1, b2):
            worst, n = logp_gap(actor, buf)
            self.assertLess(worst, 1e-4)
            self.assertGreater(n, 20)
            lv = buf.loss_view
            valid, first, done, trunc = lv(buf.store.valid), lv(buf.store.first), lv(buf.done), lv(buf.trunc)
            for s in range(S):
                for t in range(1, buf.T):
                    if valid[t, s]:
                        expect = (not valid[t - 1, s]) or bool(done[t - 1, s] or trunc[t - 1, s])
                        self.assertEqual(bool(first[t, s]), expect, (t, s))

    def test_rounds_without_pending_credit_are_unchanged(self):
        # settle_cap 0 is the old code path: every round up to the first one that ends with credit owed (and every
        # round of an env without pending_credit) must give identical buffers
        for cfg_fn, steps, rounds, same in ((None, 16, 4, 4), (duel_cfg(2, 5), 40, 3, 3), (late_cfg(2), 24, 3, 2)):
            runs = []
            for cap in (None, 0):
                cfg, actor, critic, sm = make(steps=steps, seg=8, burn=4, cfg_fn=cfg_fn)
                if cap is not None:
                    sm.settle_cap = cap
                out = []
                for _ in range(rounds):
                    buf, st = sm.collect(actor)
                    sm.finish(buf, critic, None, 0.995, 0.95)
                    out.append((snapshot(buf, sm), st.get("settle_ticks", 0)))
                runs.append(out)
            self.assertEqual([sa > 0 for _, sa in runs[0]], [r >= same for r in range(rounds)])
            for r, ((a, sa), (b, sb)) in enumerate(list(zip(*runs))[:same]):
                self.assertEqual(set(a), set(b))
                for k in a:
                    same = torch.equal(a[k], b[k]) if isinstance(a[k], torch.Tensor) else a[k] == b[k]
                    self.assertTrue(same, (cfg_fn, r, k))


if __name__ == "__main__":
    unittest.main()
