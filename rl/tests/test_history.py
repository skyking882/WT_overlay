"""History episodes: the current policy against frozen past policies from the league (rl.league).

HistEnv stands in for MatchEnv with history_prob (same reporting: env.episode_kind "history", env.frozen_ids; every
slot policy-controlled). HistDuelEnv is test_late_credit's scripted 1v1 as a history episode, for exact late credit.
"""
import contextlib
import io
import json
import os
import random
import tempfile
import unittest

import common  # noqa: F401
import torch

from rl import config as C
from rl import eval_replay, spec
from rl import train as T
from rl.fake_env import FakeMatchEnv
from rl.league import League
from rl.model import Actor, Critic
from rl.pool import WorkerPool
from rl.ppo import PPOTrainer, actor_window, clipped_surrogate
from rl.rollout import Sampler
from test_late_credit import DuelEnv, logp_gap
from test_rollout import DEV


class HistEnv(FakeMatchEnv):
    """FakeMatchEnv with MatchEnv's episode kinds: history and self-play episodes control 2 agents (in a history episode
    one of them, drawn per episode, is frozen), vs-script episodes 1. Logs every action and reward it hands out."""

    def __init__(self, config=None, seed=0):
        cfg = dict(config or {})
        self.p_self = cfg.pop("self_play_prob", 0.0)
        self.p_hist = cfg.pop("history_prob", 0.5)
        super().__init__(cfg, seed)
        self.kind_rng = random.Random("%s:kind" % seed)
        self.episode_kind, self.frozen_ids = None, ()
        self.frozen_by_ep, self.acted, self.ledger = {}, [], {}

    def reset(self):
        u = self.kind_rng.random()
        self.episode_kind = (spec.KIND_SELF if u < self.p_self else spec.KIND_HIST if u < self.p_self + self.p_hist
                             else spec.KIND_SCRIPT)
        self.cfg["min_agents"] = self.cfg["n_agents"] = 1 if self.episode_kind == spec.KIND_SCRIPT else 2
        obs = super().reset()
        self.frozen_ids = (self.kind_rng.choice(sorted(obs)),) if self.episode_kind == spec.KIND_HIST else ()
        self.frozen_by_ep[self.episode_count] = self.frozen_ids
        return obs

    def step(self, actions):
        for aid in actions:
            self.acted.append((self.episode_count, aid))
        obs, rew, done, info = super().step(actions)
        for aid, r in rew.items():
            self.ledger[(self.episode_count, aid)] = self.ledger.get((self.episode_count, aid), 0.0) + r
        info["episode_kind"] = self.episode_kind
        return obs, rew, done, info


class HistDuelEnv(DuelEnv):
    """DuelEnv as a history episode: p0 goes down at die_at and its missile kills p1 at hit_at (or misses). cfg frozen:
    the frozen agent; with p0 frozen nothing is owed after its death (pending_credit False) and its kill is dropped."""

    def reset(self):
        obs = super().reset()
        self.episode_kind, self.frozen_ids = spec.KIND_HIST, (self.cfg["frozen"],)
        return obs

    def pending_credit(self):
        return super().pending_credit() and "p0" not in self.frozen_ids

    def step(self, actions):
        out = super().step(actions)
        out[3]["episode_kind"] = self.episode_kind
        return out


def hist_cfg(p_hist=0.5, p_self=0.0, **kw):
    def fn(cfg):
        cfg.env.cls = "test_history:HistEnv"
        cfg.env.streams_per_env = 2
        cfg.env.config.update(dict(dict(history_prob=p_hist, self_play_prob=p_self, max_steps=30,
                                        p_background_death=0.03), **kw))
    return fn


def duel_hist_cfg(die_at, hit_at, frozen="p1", hit=True, max_steps=400):
    def fn(cfg):
        cfg.env.cls = "test_history:HistDuelEnv"
        cfg.env.streams_per_env = 2
        cfg.env.config.update(n_agents=2, min_agents=2, max_steps=max_steps, p_incoming=0.0, p_background_death=0.0,
                              p_match_end=0.0, die_at=die_at, hit_at=hit_at, hit=hit, frozen=frozen)
    return fn


class RecSampler(Sampler):
    """Sampler that logs (tick, stream, agent, opponent) of every frozen action."""

    def _tick(self, actor, greedy, abort, t, buf, st, streams):
        self.t_now = t
        return super()._tick(actor, greedy, abort, t, buf, st, streams)

    def _act_frozen(self, streams, greedy, by_env, st):
        self.frozen_log += [(self.t_now, s, self.slot_agent[s], self.env_opp[self.stream_env[s]][0]) for s in streams]
        return super()._act_frozen(streams, greedy, by_env, st)


def make_hist(cfg_fn, seed=1, streams=8, steps=40, seg=8, burn=4, n_opp=1):
    cfg = common.smoke_cfg()
    cfg.rollout.burn_in, cfg.rollout.seg_len, cfg.rollout.steps, cfg.rollout.n_streams = burn, seg, steps, streams
    cfg.ppo.minibatch_segments = 6
    cfg_fn(cfg)
    torch.manual_seed(seed)
    actor, critic = Actor(), Critic()
    for p in actor.cat.parameters():
        if p.dim() > 1:
            torch.nn.init.normal_(p, std=0.5)
    league = League(cfg.league, DEV, seed=seed)
    for i in range(n_opp):
        torch.manual_seed(99 + i)
        league.add("opp%d" % i, Actor())
    pool = WorkerPool(cfg.workers, cfg.env).start()
    sm = RecSampler(cfg, pool, DEV, seed, league=league)
    sm.frozen_log = []
    sm.start(100 + seed)
    return cfg, actor, critic, sm, league


def envs_of(sm):
    return sm.pool.handles[0].state["host"].envs


class FrozenSide(unittest.TestCase):
    def test_frozen_steps_never_reach_the_buffer_and_the_current_side_is_complete(self):
        cfg, actor, critic, sm, league = make_hist(hist_cfg(0.5, 0.25), streams=16, n_opp=2)
        envs = envs_of(sm)
        total_rew, kinds, opps = 0.0, set(), set()
        for _ in range(3):
            n_acted = sum(len(e.acted) for e in envs.values())
            sm.frozen_log = []
            buf, st = sm.collect(actor)
            sm.finish(buf, critic, None, 0.995, 0.95)
            frozen = {(t, s) for t, s, _, _ in sm.frozen_log}
            self.assertGreater(len(frozen), 0)
            self.assertEqual(st["frozen_decisions"], len(sm.frozen_log))
            for t, s in frozen:
                f = buf.flat(t, s)
                self.assertFalse(bool(buf.store.valid[f]), (t, s))
                self.assertEqual((float(buf.reward[f]), float(buf.logp[f]), bool(buf.done[f]), bool(buf.trunc[f]),
                                  bool(buf.store.first[f]), int(buf.store.ent_n[f])), (0.0, 0.0, False, False, False, 0))
                self.assertFalse(bool(buf.logp_heads[f].any() or buf.store.own[f].any() or buf.store.act[f].any()))
            lt = buf.loss_view(buf.store.valid).nonzero().tolist()
            self.assertFalse(frozen & {(t, s) for t, s in lt})
            # every action an env took is either a stored current-policy decision or a logged frozen one
            self.assertEqual(sum(len(e.acted) for e in envs.values()) - n_acted, st["n_valid"] + st["frozen_decisions"])
            self.assertEqual(sum(st["decisions_by_kind"].values()), st["n_valid"])
            self.assertEqual(buf.store.ent_rows.shape[0], int(buf.store.ent_n.sum()))
            total_rew += float(buf.reward.sum())
            # results: one per current agent, history ones tagged with their opponent
            for i, o in enumerate(st["outcomes"]):
                kinds.add(o[4])
                self.assertEqual(o[4] == spec.KIND_HIST, i in st.get("opponent_of", {}))
            opps |= set(st.get("opponent_of", {}).values())
            worst, n = logp_gap(actor, buf)
            self.assertLess(worst, 1e-4)
            lv = buf.loss_view
            valid, first, done, trunc = lv(buf.store.valid), lv(buf.store.first), lv(buf.done), lv(buf.trunc)
            for s in range(buf.S):
                for t in range(1, buf.T):
                    if valid[t, s]:
                        self.assertEqual(bool(first[t, s]), (not valid[t - 1, s]) or bool(done[t - 1, s] or trunc[t - 1, s]))
        self.assertEqual(kinds, {spec.KIND_SCRIPT, spec.KIND_SELF, spec.KIND_HIST})
        self.assertEqual(opps, {"opp0", "opp1"})
        # the buffers hold exactly the rewards of the trained agents (the frozen side's are dropped)
        trained = sum(r for e in envs.values() for (ep, aid), r in e.ledger.items() if aid not in e.frozen_by_ep[ep])
        self.assertAlmostEqual(total_rew, trained, 4)
        frozen_rew = sum(abs(r) for e in envs.values() for (ep, aid), r in e.ledger.items() if aid in e.frozen_by_ep[ep])
        self.assertGreater(frozen_rew, 0.0)

    def test_the_frozen_actor_flies_with_its_own_state_and_is_never_trained(self):
        cfg, actor, critic, sm, league = make_hist(hist_cfg(1.0), streams=8)
        opp = league.members[0]["actor"]
        calls = []
        inner = opp.act

        def act(b, h, gen, greedy):
            calls.append((b.own.shape[0], gen is sm.gen_frozen, gen is sm.gen))
            return inner(b, h, gen, greedy)
        opp.act = act
        p0 = [p.clone() for p in opp.parameters()]
        trainer = PPOTrainer(cfg, actor, critic, None, DEV)
        buf, st = sm.collect(actor)
        sm.finish(buf, critic, None, 0.995, 0.95)
        m = trainer.update(buf)
        self.assertGreater(m["actor_steps"], 0)
        self.assertEqual(sum(c[0] for c in calls), st["frozen_decisions"])
        self.assertTrue(all(c[1] and not c[2] for c in calls))
        self.assertTrue(all(torch.equal(p, q) for p, q in zip(opp.parameters(), p0)))
        self.assertFalse(any(p.requires_grad for p in opp.parameters()))
        trained = {id(p) for g in trainer.opt_a.param_groups for p in g["params"]}
        self.assertFalse(trained & {id(p) for p in opp.parameters()})
        self.assertTrue(sm.h_frozen.abs().sum() > 0)
        # history only: every episode has a frozen side, every result is the current policy's
        self.assertEqual(set(st["decisions_by_kind"]), {spec.KIND_HIST})
        self.assertEqual(len(st["outcomes"]), len(st["opponent_of"]))

    def test_history_against_a_copy_of_the_actor_is_self_play_without_the_frozen_steps(self):
        """Greedy, no deaths (episodes end together): a history round whose opponent is a copy of the current actor
        must play exactly the self-play episodes, and store exactly the current side's steps of them."""
        quiet = dict(p_incoming=0.0, p_background_death=0.0)
        bufs = {}
        for name, fn in (("sp", hist_cfg(0.0, 1.0, **quiet)), ("hist", hist_cfg(1.0, 0.0, **quiet))):
            cfg, actor, critic, sm, league = make_hist(fn, streams=8, steps=40)
            league.members[0]["actor"] = League._frozen(Actor())
            league.members[0]["actor"].load_state_dict(actor.state_dict())
            for j in range(sm.n_envs):         # the opponent drawn at start() was the placeholder: rebind the copy
                if sm.env_opp[j] is not None:
                    sm.env_opp[j] = ("opp0", league.members[0]["actor"])
            out = []
            for _ in range(2):
                sm.frozen_log = []
                out.append((sm.collect(actor, greedy=True)[0], {(t, s) for t, s, _, _ in sm.frozen_log}))
            bufs[name] = out
        n_frozen = 0
        for (a, _), (b, fz) in zip(bufs["sp"], bufs["hist"]):
            self.assertEqual(a.T, b.T)
            va, vb = a.loss_view(a.store.valid), b.loss_view(b.store.valid)
            self.assertEqual(int(va.sum()), int(vb.sum()) + len(fz))
            n_frozen += len(fz)
            for t, s in fz:
                self.assertTrue(bool(va[t, s]) and not bool(vb[t, s]))
            m = vb.reshape(-1)
            idx = torch.arange(a.P * a.S, a.n_steps)[m]
            for f in ("own", "prev", "act", "first", "dt"):
                self.assertTrue(torch.equal(getattr(a.store, f)[idx], getattr(b.store, f)[idx]), f)
            for f in ("reward", "done", "trunc"):
                self.assertTrue(torch.equal(getattr(a, f)[idx], getattr(b, f)[idx]), f)
            # another batch composition (the frozen streams are not in the current actor's batch): rounding only
            self.assertTrue(torch.allclose(a.logp[idx], b.logp[idx], atol=1e-5))
        self.assertGreater(n_frozen, 100)

    def test_segments_with_only_frozen_steps_give_no_gradient(self):
        """A frozen-only segment is never sampled (no valid step), and even forced through the loss it adds nothing."""
        cfg, actor, critic, sm, league = make_hist(hist_cfg(1.0), streams=8)
        buf, st = sm.collect(actor)
        sm.finish(buf, critic, None, 0.995, 0.95)
        vm = buf.loss_view(buf.store.valid)
        L, B = buf.L, buf.B
        fz = {(s, t // L) for t, s, _, _ in sm.frozen_log}
        segs = [(s, k) for s, k in sorted(fz) if not bool(vm[k * L:(k + 1) * L, s].any())]
        self.assertGreater(len(segs), 0)
        s_b, k_b = torch.tensor([a for a, _ in segs]), torch.tensor([b for _, b in segs])
        widx, lidx = buf.window_index(s_b, k_b), buf.loss_index(s_b, k_b)
        batch = buf.store.gather(widx)
        h0 = torch.stack([buf.h_actor[int(k)][int(s)] for s, k in segs])
        actor.zero_grad(set_to_none=True)
        out, seg = actor_window(actor, batch, h0, B, batch.act[:, B:], keep_dists=True)
        lmf = seg.valid.reshape(-1).to(torch.float32)
        self.assertEqual(float(lmf.sum()), 0.0)
        ratio = (out.logp.sum(-1) - buf.logp[widx[:, B:]].reshape(-1)).exp()
        old_w = buf.logp_heads[widx[:, B:]].reshape(-1, spec.N_HEADS)[:, spec.HEAD_INDEX["weapon"]]
        lr_w = out.logp[:, spec.HEAD_INDEX["weapon"]] - old_w
        n = lmf.sum().clamp(min=1)
        loss = ((-clipped_surrogate(ratio, torch.ones_like(ratio), 0.15) - out.ent.mean(-1)
                 + 10.0 * ((lr_w.exp() - 1) - lr_w)) * lmf).sum() / n
        loss.backward()
        self.assertTrue(all(p.grad is None or not p.grad.any() for p in actor.parameters()))
        self.assertTrue(bool((buf.adv.view(buf.T, buf.S)[lidx // buf.S, lidx % buf.S] == 0).all()))


class HistoryLateCredit(unittest.TestCase):
    def test_current_agent_trades_with_a_kill_after_its_death_across_the_round_boundary(self):
        # p1 frozen: p0 (current) is down at step 6, its missile kills the frozen p1 at step 11 -> settle ticks 8..15
        cfg, actor, critic, sm, league = make_hist(duel_hist_cfg(6, 11), streams=4, steps=8, seg=8, burn=4)
        buf, st = sm.collect(actor)
        self.assertEqual((buf.T, st["settle_ticks"], st["settle_envs"], st["settle_decisions"]), (16, 8, 1, 0))
        self.assertEqual((st.get("late_dropped", 0), st["late_credited"], st.get("late_frozen", 0)), (0, 1, 0))
        lv = buf.loss_view
        rew, done, valid = lv(buf.reward), lv(buf.done), lv(buf.store.valid)
        self.assertEqual((float(rew[5, 0]), bool(done[5, 0])), (-1.0, True))
        self.assertEqual(valid[:, 0].nonzero().flatten().tolist(), list(range(6)))
        self.assertFalse(bool(valid[:, 1].any() or rew[:, 1].any()))              # the frozen p1: nothing stored
        self.assertEqual(sorted(t for t, s, _, _ in sm.frozen_log if s == 1), list(range(11)))   # it flew to the hit
        self.assertEqual([o[1:] for o in st["outcomes"]], [("trade", 1, 1, spec.KIND_HIST)])
        self.assertEqual(st["opponent_of"], {0: "opp0"})
        self.assertEqual([(e[1], e[3], e[4]) for e in st["episodes"]], [(-1.0, "terminal", spec.KIND_HIST)])
        sm.finish(buf, critic, None, 0.995, 0.95)
        self.assertAlmostEqual(float(buf.ret.view(buf.T, buf.S)[5, 0]), -1.0, 6)
        # quiet env (streams 2, 3; frozen 3): the current agent bootstraps at tick 7, the frozen one is never stored
        self.assertTrue(bool(lv(buf.trunc)[7, 2]) and not bool(lv(buf.trunc)[7, 3]))
        self.assertFalse(bool(valid[:, 3].any()))
        self.assertNotIn(3, buf.end_obs)

    def test_nothing_is_owed_to_the_frozen_side(self):
        # p0 frozen: it is down at step 6 (no settle: pending_credit leaves it out), its missile kills the current p1
        # at step 11: a plain loss for p1, the frozen side's late kill is dropped
        cfg, actor, critic, sm, league = make_hist(duel_hist_cfg(6, 11, frozen="p0"), streams=4, steps=8, seg=8, burn=4)
        b1, s1 = sm.collect(actor)
        b2, s2 = sm.collect(actor)
        self.assertEqual((b1.T, s1.get("settle_ticks", 0), s1.get("outcomes", [])), (8, 0, []))
        self.assertEqual([o[1:] for o in s2["outcomes"]], [("loss", 0, 1, spec.KIND_HIST)])
        self.assertEqual((s2.get("late_frozen", 0), s2.get("late_dropped", 0), s2.get("late_credited", 0)), (1, 0, 0))
        self.assertEqual(float(b2.loss_view(b2.reward)[2, 1]), -2.0)              # p1 hit at step 11 = round 2 tick 2
        self.assertFalse(bool(b1.loss_view(b1.store.valid)[:, 0].any()))

    def test_an_episode_ends_once_only_the_frozen_side_is_left_and_nothing_is_owed(self):
        # p0 (current) down at 6, its missile misses at 11: then only the frozen p1 is left -> reset (no flying on to
        # the time limit as in self-play)
        cfg, actor, critic, sm, league = make_hist(duel_hist_cfg(6, 11, hit=False, max_steps=40), streams=4, steps=8)
        buf, st = sm.collect(actor)
        env = envs_of(sm)[0]
        self.assertEqual(env.episode_count, 2)
        self.assertEqual(sorted(t for t, s, _, _ in sm.frozen_log if s == 1), list(range(11)))
        self.assertEqual([o[1:4] for o in st["outcomes"]], [("loss", 0, 1)])
        self.assertEqual(st.get("lost", 0), 0)
        # the next round starts the new episode (first step) on the streams of env 0
        b2, s2 = sm.collect(actor)
        self.assertTrue(bool(b2.loss_view(b2.store.first)[0, 0]))


class Pool(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.logs = []

    def league(self, every=2, keep=2, refs=(), seed=0):
        return League(C.LeagueCfg(snapshot_every=every, keep=keep, references=list(refs)), DEV, self.tmp.name, seed,
                      log=self.logs.append)

    def test_snapshots_every_n_rounds_keep_the_newest_and_are_frozen_copies(self):
        torch.manual_seed(0)
        actor = Actor()
        lg = self.league()
        lg.snapshot(actor, 0)
        added = [lg.after_round(actor, r) for r in range(1, 7)]
        self.assertEqual(added, [None, "r000002", None, "r000004", None, "r000006"])
        self.assertEqual(lg.names(), ["r000004", "r000006"])
        self.assertIsNone(lg.after_round(actor, 6))                     # no second snapshot of the same round
        snap = lg.members[-1]["actor"]
        self.assertFalse(snap.training or any(p.requires_grad for p in snap.parameters()))
        with torch.no_grad():
            next(actor.parameters()).add_(1.0)
        self.assertFalse(torch.equal(next(actor.parameters()), next(snap.parameters())))
        # the files are usable as --opponent-checkpoint
        path = os.path.join(self.tmp.name, "league", "snap_r000006.pt")
        self.assertTrue(torch.equal(next(eval_replay.load_actor(path).parameters()), next(snap.parameters())))
        lg.prune_files()
        self.assertEqual(sorted(os.listdir(os.path.join(self.tmp.name, "league"))),
                         ["snap_r000004.pt", "snap_r000006.pt"])

    def test_restart_restores_the_pool_and_references_stay(self):
        torch.manual_seed(1)
        ref_path = os.path.join(self.tmp.name, "best.pt")
        torch.save({"actor": Actor().state_dict(), "trainer": {}}, ref_path)
        lg = self.league(every=1, keep=3, refs=[ref_path])
        lg.load_references([ref_path])
        torch.manual_seed(2)
        actor = Actor()
        for r in range(5):
            lg.after_round(actor, r + 1)
        self.assertEqual(lg.names(), ["ref0:best.pt", "r000003", "r000004", "r000005"])
        state = json.loads(json.dumps(lg.state_dict()))                 # plain data in the checkpoint
        lg.prune_files()
        os.remove(ref_path)                                               # the reference's original is gone
        lg2 = self.league(every=1, keep=3, refs=[ref_path], seed=5)
        lg2.load_state_dict(state)
        lg2.load_references([ref_path])
        self.assertEqual(sorted(lg2.names()), sorted(lg.names()))
        for m1 in lg.members:
            m2 = next(m for m in lg2.members if m["name"] == m1["name"])
            self.assertTrue(all(torch.equal(p, q) for p, q in zip(m1["actor"].parameters(), m2["actor"].parameters())))
        self.assertTrue(any("is gone" in x for x in self.logs))
        self.assertEqual(lg2.last_snapshot, 5)
        # a snapshot file deleted behind our back is left out, not fatal
        os.remove(os.path.join(self.tmp.name, "league", "snap_r000003.pt"))
        lg3 = self.league(every=1, keep=3)
        lg3.load_state_dict(state)
        self.assertEqual(lg3.names(), ["r000004", "r000005"])
        with self.assertRaises(FileNotFoundError):
            self.league().load_references([os.path.join(self.tmp.name, "nothing.pt")])

    def test_pick_is_uniform_and_reproducible(self):
        lg = self.league(keep=4)
        for i in range(4):
            lg.add("m%d" % i, Actor())
        picks = [lg.pick()[0] for _ in range(4000)]
        for i in range(4):
            self.assertLess(abs(picks.count("m%d" % i) - 1000), 150)
        lg2 = self.league(keep=4)
        for i in range(4):
            lg2.add("m%d" % i, Actor())
        self.assertEqual([lg2.pick()[0] for _ in range(50)], picks[:50])
        with self.assertRaisesRegex(RuntimeError, "no opponent"):
            self.league().pick()


class ConfigChecks(unittest.TestCase):
    def cfg(self, streams_per_env=2, **env_config):
        c = C.preset("smoke")
        c.env.cls = "wt_overlay.rl_env:MatchEnv"
        c.env.streams_per_env = streams_per_env
        c.env.config = env_config
        c.rollout.n_streams = 8
        return c

    def test_history_needs_streams_for_every_slot_and_opponents(self):
        with self.assertRaisesRegex(ValueError, "streams_per_env"):
            self.cfg(1, team_size=1, history_prob=0.25).validate()
        self.cfg(2, team_size=1, history_prob=0.25, self_play_prob=0.25).validate()
        with self.assertRaisesRegex(ValueError, "streams_per_env"):
            self.cfg(2, teams=[[{}, {}], [{}, {}]], history_prob=0.25).validate()
        self.cfg(4, teams=[[{}, {}], [{}, {}]], history_prob=0.25).validate()
        c = self.cfg(2, history_prob=0.25)
        c.league.snapshot_every = 0
        with self.assertRaisesRegex(ValueError, "needs opponents"):
            c.validate()
        c.league.references = ["/some/ckpt.pt"]
        c.validate()
        with self.assertRaisesRegex(ValueError, "must not exceed 1"):
            self.cfg(2, history_prob=0.6, self_play_prob=0.5).validate()
        c = self.cfg(2, history_prob=0.25)
        c.league.keep = 0
        with self.assertRaisesRegex(ValueError, "keep"):
            c.validate()
        self.assertTrue(C.history_on(self.cfg(2, history_prob=0.25)))
        for off in ({}, {"history_prob": 0}, {"history_prob": 0.0, "self_play_prob": 0.5}):
            self.assertFalse(C.history_on(self.cfg(2, **off)))
            self.cfg(2, **off).validate()
        # old configs (no league section) still load
        d = self.cfg(2).to_dict()
        d.pop("league")
        self.assertEqual(C.Config.from_dict(d).league.keep, C.LeagueCfg().keep)

    def test_head_kl_config_is_checked(self):
        c = self.cfg(1)
        c.ppo.head_kl = {"weapon": {"target": 2e-4, "coef": 1.0}}
        c.validate()
        for bad in ({"wepon": {"target": 1e-3}}, {"weapon": {"coef": 1.0}}, {"weapon": {"target": -1.0}},
                    {"weapon": {"target": 1e-3, "coef": -1}}, {"weapon": {"target": 1e-3, "beta": 1}},
                    {"weapon": {"target": 1e-3, "coef_min": 2, "coef_max": 1}}, {"weapon": 0.002}):
            c.ppo.head_kl = bad
            with self.assertRaises(ValueError, msg=repr(bad)):
                c.validate()


class RecordAndRestart(unittest.TestCase):
    def test_round_record_has_history_results_by_opponent(self):
        cfg, actor, critic, sm, league = make_hist(hist_cfg(0.5, 0.25), streams=16, n_opp=2)
        trainer = PPOTrainer(cfg, actor, critic, None, DEV)
        seen = 0
        for _ in range(3):
            buf, st = sm.collect(actor)
            sm.finish(buf, critic, None, 0.995, 0.95)
            rec = T.round_record(cfg, trainer, buf, st, trainer.update(buf), sm)
            json.dumps(rec)
            hist = [o for o in st["outcomes"] if o[4] == spec.KIND_HIST]
            by = rec["league"]["by_opponent"]
            self.assertEqual(rec["league"]["members"], ["opp0", "opp1"])
            self.assertEqual(sum(v["episodes"] for v in by.values()), len(hist))
            if hist:
                seen += 1
                h = rec["outcomes_by_kind"][spec.KIND_HIST]
                self.assertEqual({k: h[k] for k in ("win", "loss", "trade", "none")},
                                 {k: sum(v[k] for v in by.values()) for k in ("win", "loss", "trade", "none")})
                self.assertIn("history: eps", T.summary_line(dict(rec, time=dict(sample=1., inference=1., env_wait=1.,
                                                                                 postpass=1., update=1.))))
            # the long-standing keys stay vs-script only
            vs = [o for o in st["outcomes"] if o[4] == spec.KIND_SCRIPT]
            self.assertEqual(rec["outcomes"], T.outcome_counts(vs) if vs else T.outcome_counts(st["outcomes"]))
            self.assertIn("frozen_decisions", rec["sampler_stats"])
            self.assertEqual(rec["outcomes_by_kind"][spec.KIND_HIST]["decisions"],
                             st["decisions_by_kind"].get(spec.KIND_HIST, 0))
        self.assertGreater(seen, 0)

    def test_stage_ppo_snapshots_persist_over_a_restart(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = C.preset("smoke")
            cfg.workers.inproc = True
            cfg.env.cls = "test_history:HistEnv"
            cfg.env.config = {"history_prob": 0.5, "self_play_prob": 0.25, "max_steps": 25, "p_background_death": 0.03}
            cfg.league.snapshot_every, cfg.league.keep = 1, 2
            cfg.ppo.head_kl = {"weapon": {"target": 1e-9, "coef": 1.0}}       # always above target: coef grows
            cfg.run.keep_ckpts = 5
            path = os.path.join(d, "cfg.json")
            cfg.save(path)
            args = ["ppo", "--config", path, "--run-dir", d]
            out = io.StringIO()
            self.addCleanup(torch.set_num_threads, torch.get_num_threads())
            with contextlib.redirect_stdout(out):
                self.assertEqual(T.main(args + ["--rounds", "2"]), 0)
            ck = torch.load(os.path.join(d, "ppo", "ckpt_000002.pt"), map_location="cpu", weights_only=False)
            self.assertEqual([s["name"] for s in ck["league"]["snapshots"]], ["r000001", "r000002"])
            self.assertEqual(sorted(os.listdir(os.path.join(d, "league"))), ["snap_r000001.pt", "snap_r000002.pt"])
            self.assertAlmostEqual(ck["trainer"]["head_kl_coef"]["weapon"], 1.5 ** 2)
            with open(os.path.join(d, "metrics.jsonl")) as f:
                recs = [json.loads(x) for x in f]
            self.assertEqual(recs[0]["league"]["members"], ["r000000", "r000001"])    # the start actor, then round 1
            self.assertEqual(recs[0]["league"]["added"], "r000001")
            self.assertEqual([r["head_kl"]["weapon"]["coef"] for r in recs], [1.0, 1.5])
            self.assertTrue(all(r["head_kl"]["weapon"]["kl"] > 0 for r in recs))
            self.assertIn(spec.KIND_HIST, recs[0]["outcomes_by_kind"])
            with contextlib.redirect_stdout(out):
                self.assertEqual(T.main(args + ["--rounds", "3"]), 0)
            self.assertIn("league: r000001, r000002", out.getvalue())
            ck3 = torch.load(os.path.join(d, "ppo", "ckpt_000003.pt"), map_location="cpu", weights_only=False)
            self.assertEqual([s["name"] for s in ck3["league"]["snapshots"]], ["r000002", "r000003"])
            self.assertEqual(sorted(os.listdir(os.path.join(d, "league"))), ["snap_r000002.pt", "snap_r000003.pt"])
            with open(os.path.join(d, "metrics.jsonl")) as f:
                recs = [json.loads(x) for x in f]
            self.assertEqual(recs[-1]["head_kl"]["weapon"]["coef"], 1.5 ** 2)       # the adapted coef was restored


if __name__ == "__main__":
    unittest.main()
