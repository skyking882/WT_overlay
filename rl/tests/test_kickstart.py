"""Teacher labels (env teacher -> worker -> sampler -> buffer) and ppo.kickstart, the decaying imitation term
(docs/kickstart_spec.md)."""
import copy
import math
import os
import tempfile
import unittest
from types import SimpleNamespace

import common  # noqa: F401
import torch

from rl import config as C
from rl import spec
from rl import train as T
from rl.buffer import RoundBuffer
from rl.encode import masks_from_flat
from rl.fake_env import FakeMatchEnv
from rl.ppo import PPOTrainer, actor_window, kickstart_ce
from test_history import HistEnv, hist_cfg, make_hist
from test_rollout import DEV, make

V, SP = spec.HEAD_INDEX["vertical"], spec.HEAD_INDEX["speed"]
KS = {"coef": 1.0, "decay_rounds": 10}


def fake_label(t, free):
    """The label the test envs give a decision at episode step t; free: the vertical head is free on view_mode 0."""
    if t % 5 == 4:
        return {"speed": 2, "name": "push"}
    if free and t % 2 == 0:
        return {"vertical": 1, "name": "climb"}
    return None


def teach(env, obs):
    t = env.S["t"]
    env.teacher_labels = {}
    for aid, o in obs.items():
        lab = fake_label(t, all(o.masks["vertical"][0][0]))
        if lab:
            env.teacher_labels[aid] = lab
    return obs


class TeacherEnv(FakeMatchEnv):
    """FakeMatchEnv with MatchEnv's teacher reporting: env.teacher_labels after reset, info["teacher"] after a step."""

    def reset(self):
        return teach(self, super().reset())

    def step(self, actions):
        obs, rew, done, info = super().step(actions)
        teach(self, obs)
        return obs, rew, done, dict(info or {}, teacher=dict(self.teacher_labels))


class TeacherHistEnv(HistEnv):
    """HistEnv that (unlike MatchEnv) labels the frozen side too: the sampler must drop those labels."""

    def reset(self):
        return teach(self, super().reset())

    def step(self, actions):
        obs, rew, done, info = super().step(actions)
        teach(self, obs)
        info["teacher"] = dict(self.teacher_labels)
        return obs, rew, done, info


def teacher_cfg(cfg):
    cfg.env.cls = "test_kickstart:TeacherEnv"


def expected(buf, f, max_steps):
    """The label fake_label gave the stored decision f, from its stored observation (own[4] = t / max_steps)."""
    t = int(round(float(buf.store.own[f, 4]) * max_steps))
    m = masks_from_flat(buf.store.mask[f].unsqueeze(0), int(buf.store.ent_n[f]) + 1)
    return fake_label(t, bool(m["vertical"][0, 0, 0].all()))


def labels_of(buf, f):
    row = buf.teacher[f].tolist()
    lab = {h: v for h, v in zip(spec.HEAD_NAMES, row) if v >= 0}
    if lab:
        lab["name"] = buf.teacher_names[int(buf.teacher_name[f])]
    else:
        assert int(buf.teacher_name[f]) == -1
    return lab or None


def mean_ce(actor, buf):
    """Mean -log pi(label) over the labelled valid loss steps whose label is legal, under ``actor``."""
    tot, n = 0.0, 0
    for s in range(buf.S):
        for k in range(buf.K):
            widx = buf.window_index(torch.tensor([s]), torch.tensor([k]))
            b = buf.store.gather(widx)
            out, seg = actor_window(actor, b, buf.h_actor[k][s].unsqueeze(0), buf.B, b.act[:, buf.B:], keep_dists=True,
                                    grad=False)
            lab = buf.teacher[widx[:, buf.B:]].reshape(-1, spec.N_HEADS).long()
            lab = torch.where(seg.valid.reshape(-1, 1), lab, torch.full_like(lab, -1))
            ce, ok = kickstart_ce(out, lab)
            tot += float(ce.sum())
            n += int(ok.any(-1).sum())
    return tot / n, n


class Plumbing(unittest.TestCase):
    def check(self, buf, max_steps, frozen=()):
        n_lab = {"climb": 0, "push": 0}
        P, S = buf.P, buf.S
        for f in range(P * S, buf.store.N):
            if (f // S - P, f % S) in frozen or not bool(buf.store.valid[f]):
                self.assertTrue(bool((buf.teacher[f] == -1).all()) and int(buf.teacher_name[f]) == -1, f)
                continue
            want = expected(buf, f, max_steps)
            self.assertEqual(labels_of(buf, f), want, f)
            if want:
                n_lab[want["name"]] += 1
        return n_lab

    def test_labels_reach_the_buffer_at_their_decision(self):
        cfg, actor, critic, sm = make(steps=40, streams=8, cfg_fn=teacher_cfg)
        tot = {"climb": 0, "push": 0}
        for _ in range(2):
            buf, st = sm.collect(actor)
            for k, v in self.check(buf, cfg.env.config["max_steps"]).items():
                tot[k] += v
            self.assertEqual(buf.teacher_names, sm.teacher_names)
        self.assertGreater(tot["climb"], 20)
        self.assertGreater(tot["push"], 20)

    def test_the_labels_change_no_rollout(self):
        _, actor, _, sm = make(steps=40, streams=8)
        _, _, _, sm_t = make(steps=40, streams=8, cfg_fn=teacher_cfg)
        for _ in range(2):
            a, _ = sm.collect(actor)
            b, _ = sm_t.collect(actor)
            for x, y in ((a.store.act, b.store.act), (a.logp, b.logp), (a.reward, b.reward), (a.store.own, b.store.own),
                         (a.store.valid, b.store.valid)):
                self.assertTrue(torch.equal(x, y))
            self.assertTrue(bool((a.teacher == -1).all()))
            self.assertTrue(bool((b.teacher >= 0).any()))

    def test_frozen_side_labels_are_never_stored(self):
        def fn(cfg):
            hist_cfg(0.6, 0.2)(cfg)
            cfg.env.cls = "test_kickstart:TeacherHistEnv"
        cfg, actor, critic, sm, league = make_hist(fn, streams=16)
        for _ in range(2):
            sm.frozen_log = []
            buf, st = sm.collect(actor)
            frozen = {(t, s) for t, s, _, _ in sm.frozen_log}
            self.assertGreater(len(frozen), 0)
            n = self.check(buf, cfg.env.config["max_steps"], frozen)
            self.assertGreater(n["climb"] + n["push"], 0)

    def test_extend_pads_without_labels(self):
        buf = RoundBuffer(2, 8, 4, 2)
        buf.teacher[:] = 3
        buf.teacher_name[:] = 0
        n0 = buf.store.N
        buf.extend(4)
        self.assertEqual(buf.teacher.shape, (buf.store.N, spec.N_HEADS))
        self.assertTrue(bool((buf.teacher[:n0] == 3).all() and (buf.teacher[n0:] == -1).all()))
        self.assertTrue(bool((buf.teacher_name[n0:] == -1).all()) and buf.teacher_name.shape == (buf.store.N,))


class CrossEntropy(unittest.TestCase):
    def test_only_legal_labels_count(self):
        lp = torch.log_softmax(torch.tensor([[1.0, 2.0, 0.0, -1.0, 0.5]] * 4), -1)
        eff = torch.tensor([[True] * 5, [True] * 5, [True, False, True, True, True], [True] * 5])
        out = SimpleNamespace(logp_all={"vertical": lp, "speed": lp[:, :3]}, eff={"vertical": eff, "speed": eff[:, :3]})
        lab = torch.full((4, spec.N_HEADS), -1, dtype=torch.long)
        lab[0, V], lab[2, V], lab[3, V] = 1, 1, 7       # legal / masked out / out of range
        lab[1, SP], lab[0, SP] = 2, 0
        ce, ok = kickstart_ce(out, lab)
        self.assertAlmostEqual(float(ce[0]), float(-lp[0, 1] - lp[0, 0]), places=5)    # two labelled heads add up
        self.assertAlmostEqual(float(ce[1]), float(-lp[1, 2]), places=5)
        self.assertEqual(ce[2:].tolist(), [0.0, 0.0])
        self.assertEqual(ok.any(-1).tolist(), [True, True, False, False])
        self.assertEqual(int(ok.sum()), 3)


class Kickstart(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg, cls.actor, cls.critic, sm = make(steps=40, streams=8, peaked=False, cfg_fn=teacher_cfg)
        cls.cfg.ppo.target_kl = 1e9                 # every minibatch updates the actor
        cls.buf, cls.st = sm.collect(cls.actor)
        sm.finish(cls.buf, cls.critic, None, 0.995, 0.95)
        cls.sm = sm

    def run_update(self, kickstart=None, buf=None, lr=None, state=None, **ppo):
        cfg = copy.deepcopy(self.cfg)
        cfg.ppo.kickstart = kickstart
        if lr:
            cfg.ppo.lr_actor = lr
        for k, v in ppo.items():
            setattr(cfg.ppo, k, v)
        a, c = copy.deepcopy(self.actor), copy.deepcopy(self.critic)
        tr = PPOTrainer(cfg, a, c, None, DEV)
        if state is not None:
            tr.load_state_dict(state)
        m = tr.update(buf or self.buf)
        return tr, a, m

    def unlabelled(self):
        b = copy.copy(self.buf)
        b.teacher = torch.full_like(self.buf.teacher, -1)
        b.teacher_name = torch.full_like(self.buf.teacher_name, -1)
        return b

    @staticmethod
    def same(a, b):
        return all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))

    def test_absent_config_is_an_identical_update(self):
        _, a0, m0 = self.run_update(buf=self.unlabelled())
        self.assertNotIn("kickstart", m0)
        _, a1, m1 = self.run_update()                       # labels in the buffer, kickstart off: metrics only
        self.assertTrue(self.same(a0, a1))
        self.assertIsNone(m1["kickstart"]["coef"])
        self.assertGreater(m1["kickstart"]["labelled_steps"], 0)
        for k in ("pg_loss", "kl_target", "entropy", "value_loss", "actor_steps"):
            self.assertEqual(m0[k], m1[k], k)
        _, a2, m2 = self.run_update({"coef": 0.0})          # configured with coef 0: the same update
        self.assertTrue(self.same(a0, a2))
        self.assertEqual(m2["kickstart"]["coef"], 0.0)
        _, a3, _ = self.run_update(KS)
        self.assertFalse(self.same(a0, a3))
        self.assertNotIn("kickstart_start", PPOTrainer(self.cfg, copy.deepcopy(self.actor), copy.deepcopy(self.critic),
                                                       None, DEV).state_dict())

    def test_the_loss_pushes_the_labelled_heads_toward_the_label(self):
        ce0, n = mean_ce(self.actor, self.buf)
        self.assertGreater(n, 20)
        _, a_off, m_off = self.run_update(lr=3e-3)
        _, a_on, m_on = self.run_update(KS, lr=3e-3)
        ce_off, _ = mean_ce(a_off, self.buf)
        ce_on, _ = mean_ce(a_on, self.buf)
        self.assertLess(ce_on, ce0 - 0.3)
        self.assertLess(ce_on, ce_off - 0.3)
        ks = m_on["kickstart"]
        valid = self.buf.loss_view(self.buf.store.valid).reshape(-1)
        lab = self.buf.teacher[self.buf.P * self.buf.S:]
        n_lab = int(((lab >= 0).any(-1) & valid).sum())
        self.assertEqual(ks["labelled_steps"], n_lab)
        self.assertAlmostEqual(ks["label_share"], n_lab / int(valid.sum()))
        self.assertEqual(ks["coef"], 1.0)
        self.assertEqual(ks["start_round"], 0)
        self.assertAlmostEqual(ks["applied_share"], 1.0)    # no KL control left anything out
        self.assertEqual(ks["kl_skipped_minibatches"], 0)
        self.assertTrue(0.0 < ks["legal_share"] <= 1.0 and 0.0 <= ks["agree"] <= 1.0 and ks["ce"] > 0.0)
        self.assertEqual(set(ks["by_teacher"]), {"climb", "push"})
        self.assertEqual(sum(v["labelled_steps"] for v in ks["by_teacher"].values()), n_lab)
        # agree: the sampled option equals the label on every labelled head
        act = self.buf.store.act[self.buf.P * self.buf.S:]
        sel = (lab >= 0).any(-1) & valid
        agree = (((act == lab.long()) | (lab < 0)).all(-1) & sel).sum()
        self.assertAlmostEqual(ks["agree"], float(agree) / n_lab)

    def test_kl_skip_can_block_it_and_is_counted(self):
        # kl_mode "skip" with a tiny skip threshold: after the first applied minibatch the KL to the behaviour policy
        # exceeds it, so the later minibatches (and their kickstart terms) are left out
        _, _, m = self.run_update(KS, lr=3e-3, kl_mode="skip", target_kl=1e9, target_kl_skip=1e-4)
        ks = m["kickstart"]
        self.assertGreater(m["actor_steps"], 0)
        self.assertGreater(m["kl_skipped_minibatches"], 0)
        self.assertGreater(ks["kl_skipped_minibatches"], 0)
        self.assertLessEqual(ks["kl_skipped_minibatches"], m["kl_skipped_minibatches"])
        self.assertLess(ks["applied_share"], 1.0)
        self.assertGreater(ks["applied_share"], 0.0)
        # the KL stop ("stop" mode) drops the rest of the round the same way
        _, _, m = self.run_update(KS, lr=3e-3, kl_mode="stop", target_kl=1e-4)
        self.assertGreater(m["actor_stopped_at_minibatch"], 0)
        self.assertLess(m["kickstart"]["applied_share"], 1.0)

    def test_schedule_decays_and_survives_a_checkpoint(self):
        tr = PPOTrainer(dict_cfg(self.cfg, {"coef": 0.5, "decay_rounds": 4}), copy.deepcopy(self.actor),
                        copy.deepcopy(self.critic), None, DEV)
        tr.round = 7                                         # resumed run: the schedule starts where it is switched on
        coefs = []
        for r in range(7, 13):
            tr.round = r
            coefs.append(tr.kickstart_coef())
        self.assertEqual(tr.ks_start, 7)
        self.assertEqual(coefs, [0.5, 0.375, 0.25, 0.125, 0.0, 0.0])
        tr.round = 9
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ck.pt")
            torch.save({"trainer": tr.state_dict()}, path)
            st = torch.load(path, map_location="cpu", weights_only=False)["trainer"]
        self.assertEqual(st["kickstart_start"], 7)
        tr2 = PPOTrainer(dict_cfg(self.cfg, {"coef": 0.5, "decay_rounds": 4}), copy.deepcopy(self.actor),
                         copy.deepcopy(self.critic), None, DEV)
        tr2.load_state_dict(st)
        self.assertEqual((tr2.round, tr2.ks_start, tr2.kickstart_coef()), (9, 7, 0.25))
        # a configured start_round wins over the checkpoint; before it the term is off
        tr3 = PPOTrainer(dict_cfg(self.cfg, {"coef": 0.5, "decay_rounds": 4, "start_round": 10}),
                         copy.deepcopy(self.actor), copy.deepcopy(self.critic), None, DEV)
        tr3.load_state_dict(st)
        self.assertEqual((tr3.ks_start, tr3.kickstart_coef()), (10, 0.0))
        tr3.round = 11
        self.assertEqual(tr3.kickstart_coef(), 0.375)
        # switched off: the state has no kickstart entry and an old one is ignored
        tr4 = PPOTrainer(self.cfg, copy.deepcopy(self.actor), copy.deepcopy(self.critic), None, DEV)
        tr4.load_state_dict(st)
        self.assertEqual((tr4.kickstart_coef(), tr4.ks_start), (0.0, None))
        self.assertNotIn("kickstart_start", tr4.state_dict())

    def test_updates_follow_the_schedule(self):
        tr, _, m1 = self.run_update({"coef": 0.5, "decay_rounds": 2})
        m2 = tr.update(self.buf)
        m3 = tr.update(self.buf)
        self.assertEqual([m["kickstart"]["coef"] for m in (m1, m2, m3)], [0.5, 0.25, 0.0])
        self.assertEqual(tr.state_dict()["kickstart_start"], 0)
        line = T.summary_line(dict(m2, round=2, decisions_total=1, valid_fraction=1.0, reward_per_decision=0.0,
                                   time={k: 0.0 for k in ("sample", "inference", "env_wait", "postpass", "update")}))
        self.assertIn("  ks 0.25 lab ", line)
        _, _, off = self.run_update()
        line = T.summary_line(dict(off, round=1, decisions_total=1, valid_fraction=1.0, reward_per_decision=0.0,
                                   time={k: 0.0 for k in ("sample", "inference", "env_wait", "postpass", "update")}))
        self.assertNotIn(" ks ", line)

    def test_config_values(self):
        self.assertIsNone(C.PPOCfg().kickstart)
        self.assertIsNone(C.kickstart_spec(None))
        self.assertEqual(C.kickstart_spec({}), {"coef": 0.5, "decay_rounds": 40, "start_round": None, "tiers": None,
                                                "adapt": None})
        self.assertEqual(C.kickstart_spec({"coef": 1, "start_round": 3}), {"coef": 1.0, "decay_rounds": 40,
                                                                           "start_round": 3, "tiers": None,
                                                                           "adapt": None})
        c = C.preset("smoke")
        c.ppo.kickstart = {"coef": 0.5, "decay_rounds": 40, "start_round": None}
        c.validate()
        C.apply_overrides(c, ["ppo.kickstart={'coef': 0.2}"])
        self.assertEqual(c.ppo.kickstart, {"coef": 0.2})
        for bad in ({"coeff": 0.5}, {"coef": -0.1}, {"coef": True}, {"coef": math.inf}, {"decay_rounds": 0},
                    {"decay_rounds": 2.5}, {"start_round": -1}, {"start_round": 1.0}, 0.5, [0.5]):
            c.ppo.kickstart = bad
            with self.assertRaises(ValueError, msg=repr(bad)):
                c.validate()
            with self.assertRaises(ValueError, msg=repr(bad)):
                PPOTrainer(c, copy.deepcopy(self.actor), copy.deepcopy(self.critic), None, DEV)


class RealEnv(unittest.TestCase):
    def test_match_env_climb_labels_reach_the_buffer_and_the_update(self):
        low = [[dict(aircraft="f_15c_golden_eagle", archetype="middle", skill="top", altitude_m=3000., mach=.9)] * 2,
               [dict(aircraft="su_30sm2", archetype="middle", skill="top", altitude_m=3000., mach=.9)] * 2]

        def fn(cfg):
            cfg.env.cls = "wt_overlay.rl_env:MatchEnv"
            cfg.env.streams_per_env = 4
            cfg.env.config = dict(teams=low, range_km=60., controlled="all", execution=dict(vertical_mode="angle"),
                                  time_limit_s=60., teacher={"climb": {}})
        cfg, actor, critic, sm = make(steps=24, streams=4, seg=8, burn=4, cfg_fn=fn, peaked=False)
        buf, st = sm.collect(actor)
        sm.finish(buf, critic, None, 0.995, 0.95)
        off = buf.P * buf.S
        valid = buf.store.valid[off:]
        lab = buf.teacher[off:]
        sel = (lab >= 0).any(-1) & valid
        self.assertGreater(int(sel.sum()), 10)
        self.assertEqual(buf.teacher_names, ["climb"])
        self.assertTrue(bool((buf.teacher_name[off:][sel] == 0).all()))
        others = torch.cat([lab[:, :V], lab[:, V + 1:]], 1)
        self.assertTrue(bool((others == -1).all()))
        self.assertTrue(bool((lab[sel, V] == 1).all()))          # 5 km below 8 km: the steep climb
        first = buf.store.first[off:] & valid
        self.assertTrue(bool(sel[first].all()))                 # every opening decision is free and labelled
        m = masks_from_flat(buf.store.mask[off:][sel], 65)["vertical"]
        self.assertTrue(bool(m[:, 0].all()))                    # view_mode 0: every vertical option legal
        cfg.ppo.kickstart = dict(KS)
        mk = PPOTrainer(cfg, actor, critic, None, DEV).update(buf)["kickstart"]
        self.assertEqual(mk["labelled_steps"], int(sel.sum()))
        self.assertGreater(mk["legal_share"], 0.0)
        self.assertEqual(set(mk["by_teacher"]), {"climb"})


TIERS = [[0.1, 0.10], [0.25, 0.25], [0.4, 0.65]]


class Tiers(unittest.TestCase):
    """ppo.kickstart.tiers (a coef drawn per actor minibatch) and ppo.kickstart.adapt (scale s and shares)."""

    @classmethod
    def setUpClass(cls):
        cls.cfg, cls.actor, cls.critic, sm = make(steps=40, streams=8, peaked=False, cfg_fn=teacher_cfg)
        cls.cfg.ppo.target_kl = 1e9
        cls.bufs = []
        for _ in range(2):
            b, _ = sm.collect(cls.actor)
            sm.finish(b, cls.critic, None, 0.995, 0.95)
            cls.bufs.append(b)
        cls.buf = cls.bufs[0]

    def trainer(self, kickstart, seed=None, actor=None, critic=None, **ppo):
        c = dict_cfg(self.cfg, kickstart)
        if seed is not None:
            c.run.seed = seed
        for k, v in ppo.items():
            setattr(c.ppo, k, v)
        return PPOTrainer(c, actor or copy.deepcopy(self.actor), critic or copy.deepcopy(self.critic), None, DEV)

    @staticmethod
    def roundtrip(state):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ck.pt")
            torch.save({"trainer": state}, path)
            return torch.load(path, map_location="cpu", weights_only=False)["trainer"]

    def test_validation(self):
        good = ({"tiers": TIERS}, {"tiers": [[0.5, 1]]}, {"tiers": TIERS, "adapt": {}},
                {"tiers": [[0.0, 0.5], [1, 0.5]], "decay_rounds": 3, "start_round": 504, "adapt": {"skip_target": 0.2}},
                {"coef": 0.3, "adapt": {"step": 0.5, "min_scale": 0.1, "max_scale": 2.0}},
                {"tiers": [[0.1, 0.3333333], [0.2, 0.3333333], [0.3, 0.3333334]]}, {"coef": 0.5, "tiers": None})
        for k in good:
            C.kickstart_spec(k)
        sp = C.kickstart_spec({"tiers": TIERS, "adapt": {"skip_target": 0.2}})
        self.assertEqual((sp["coef"], sp["tiers"]), (None, [(0.1, 0.1), (0.25, 0.25), (0.4, 0.65)]))
        self.assertEqual(sp["adapt"], {"skip_target": 0.2, "step": 0.8, "min_scale": 0.2, "max_scale": 1.0})
        c = C.preset("smoke")
        for bad in ({"coef": 0.5, "tiers": TIERS}, {"tiers": [[0.1, 0.5], [0.2, 0.4]]}, {"tiers": [[0.1, 0.5], [0.2, 0.6]]},
                    {"tiers": [[-0.1, 0.5], [0.2, 0.5]]}, {"tiers": [[0.1, -0.5], [0.2, 1.5]]}, {"tiers": []},
                    {"tiers": [[0.1, 0.5, 1], [0.2, 0.5]]}, {"tiers": [0.1, 0.9]}, {"tiers": [[True, 1.0]]},
                    {"tiers": [[math.nan, 1.0]]}, {"tiers": [[math.inf, 1.0]]}, {"tiers": "[[0.1, 1.0]]"},
                    {"tiers": TIERS, "adapt": {"target": 0.1}}, {"tiers": TIERS, "adapt": {"skip_target": 0.0}},
                    {"tiers": TIERS, "adapt": {"skip_target": 1.0}}, {"tiers": TIERS, "adapt": {"step": 1.0}},
                    {"tiers": TIERS, "adapt": {"step": 0.0}}, {"tiers": TIERS, "adapt": {"min_scale": 0.0}},
                    {"tiers": TIERS, "adapt": {"min_scale": 0.5, "max_scale": 0.4}}, {"tiers": TIERS, "adapt": True},
                    {"tiers": TIERS, "adapt": {"step": "0.8"}}):
            c.ppo.kickstart = bad
            with self.assertRaises(ValueError, msg=repr(bad)):
                c.validate()
            with self.assertRaises(ValueError, msg=repr(bad)):
                self.trainer(bad)

    def test_draws_are_deterministic_honour_the_shares_and_survive_a_checkpoint(self):
        a, b = self.trainer({"tiers": TIERS}), self.trainer({"tiers": TIERS})
        seq = [a._draw_tier() for _ in range(20000)]
        self.assertEqual(seq, [b._draw_tier() for _ in range(20000)])
        for i, (_, share) in enumerate(TIERS):
            self.assertAlmostEqual(seq.count(i) / len(seq), share, delta=0.015)
        self.assertNotEqual(seq[:200], [self.trainer({"tiers": TIERS}, seed=99)._draw_tier() for _ in range(200)])
        st = self.roundtrip(a.state_dict())
        c = self.trainer({"tiers": TIERS})
        c.load_state_dict(st)
        self.assertEqual([a._draw_tier() for _ in range(500)], [c._draw_tier() for _ in range(500)])
        # a zero share is never drawn
        z = self.trainer({"tiers": [[0.1, 0.0], [0.2, 1.0]]})
        self.assertEqual({z._draw_tier() for _ in range(2000)}, {1})
        # off, or without tiers: no generator and nothing extra in the state
        self.assertIsNone(self.trainer({"coef": 0.5}).ks_gen)
        self.assertFalse({"kickstart_gen", "kickstart_scale", "kickstart_shares"} & set(
            self.trainer({"coef": 0.5}).state_dict()))

    def test_tiers_in_the_update(self):
        tr = self.trainer({"tiers": TIERS})
        m = tr.update(self.buf)
        ks = m["kickstart"]
        self.assertEqual(sum(t["drawn"] for t in ks["tiers"]), m["minibatches"])
        for t in ks["tiers"]:
            self.assertEqual(t["drawn"], t["applied"] + t["skipped"] + t["stopped"])
        self.assertEqual(sum(t["applied"] for t in ks["tiers"]), m["actor_steps"])
        self.assertEqual([(t["coef"], t["share"]) for t in ks["tiers"]], [tuple(x) for x in TIERS])
        self.assertTrue(any(t["ce"] is not None for t in ks["tiers"]))
        self.assertAlmostEqual(ks["coef"], 0.1 * 0.1 + 0.25 * 0.25 + 0.4 * 0.65)    # expected coef this round
        self.assertEqual((ks["decay"], ks["shares"]), (1.0, [0.1, 0.25, 0.65]))
        line = T.summary_line(dict(m, round=1, decisions_total=1, valid_fraction=1.0, reward_per_decision=0.0,
                                   time={k: 0.0 for k in ("sample", "inference", "env_wait", "postpass", "update")}))
        self.assertIn("  ks t[0.1:10% 0.25:25% 0.4:65%] s1.00 d1.00 lab ", line)
        # deterministic: the same trainer twice gives the same draws and parameters
        tr2 = self.trainer({"tiers": TIERS})
        m2 = tr2.update(self.buf)
        self.assertEqual(ks["tiers"], m2["kickstart"]["tiers"])
        self.assertTrue(Kickstart.same(tr.actor, tr2.actor))
        # one tier of share 1 is the single coef exactly; tier draws leave the minibatch order (self.gen) alone
        one, single = self.trainer({"tiers": [[0.5, 1.0]]}), self.trainer({"coef": 0.5})
        one.update(self.buf)
        single.update(self.buf)
        self.assertTrue(Kickstart.same(one.actor, single.actor) and Kickstart.same(one.critic, single.critic))
        self.assertTrue(torch.equal(one.gen.get_state(), single.gen.get_state()))
        zero, off = self.trainer({"tiers": [[0.0, 1.0]]}), self.trainer(None)
        zero.update(self.buf)
        off.update(self.buf)
        self.assertTrue(Kickstart.same(zero.actor, off.actor))
        # the tier coefs matter: all minibatches on a large coef move the actor differently from mixed tiers
        big = self.trainer({"tiers": [[0.4, 1.0]]})
        big.update(self.buf)
        self.assertFalse(Kickstart.same(big.actor, tr.actor))

    def test_a_restart_continues_the_draws(self):
        ks = {"tiers": TIERS, "decay_rounds": 4}
        a = self.trainer(ks)
        a.update(self.bufs[0])
        actor, critic = copy.deepcopy(a.actor), copy.deepcopy(a.critic)
        st = self.roundtrip(a.state_dict())
        m_a = a.update(self.bufs[1])
        b = self.trainer(ks, actor=actor, critic=critic)
        b.load_state_dict(st)
        m_b = b.update(self.bufs[1])
        self.assertEqual(m_a["kickstart"]["tiers"], m_b["kickstart"]["tiers"])
        self.assertEqual((m_a["kickstart"]["decay"], m_b["kickstart"]["decay"]), (0.75, 0.75))
        self.assertTrue(Kickstart.same(a.actor, b.actor))

    def test_adapt_rule(self):
        tr = self.trainer({"tiers": TIERS, "adapt": {}})
        self.assertEqual((tr.ks_scale, tr.ks_shares), (1.0, [0.1, 0.25, 0.65]))
        steps = [(0.5, 0.8, [0.15, 0.25, 0.6]), (0.5, 0.64, [0.2, 0.25, 0.55]), (0.07, 0.64, [0.2, 0.25, 0.55]),
                 (0.1, 0.64, [0.2, 0.25, 0.55]), (0.04, 0.8, [0.15, 0.25, 0.6]), (0.0, 1.0, [0.1, 0.25, 0.65]),
                 (0.0, 1.0, [0.05, 0.25, 0.7]), (0.0, 1.0, [0.05, 0.25, 0.7])]
        for skip, s, shares in steps:
            tr._kickstart_adapt(skip)
            self.assertAlmostEqual(tr.ks_scale, s)
            for x, y in zip(tr.ks_shares, shares):
                self.assertAlmostEqual(x, y)
        for _ in range(20):
            tr._kickstart_adapt(0.9)
            self.assertAlmostEqual(sum(tr.ks_shares), 1.0, places=9)
            self.assertGreaterEqual(min(tr.ks_shares), 0.05 - 1e-12)
        self.assertAlmostEqual(tr.ks_scale, 0.2)                     # min_scale
        self.assertEqual([round(x, 9) for x in tr.ks_shares], [0.7, 0.25, 0.05])
        # unsorted tiers: highest and lowest by coef; a share already below 0.05 gives nothing
        u = self.trainer({"tiers": [[0.4, 0.65], [0.1, 0.32], [0.25, 0.03]], "adapt": {"step": 0.5, "max_scale": 2.0}})
        u._kickstart_adapt(0.0)
        self.assertEqual([round(x, 9) for x in u.ks_shares], [0.7, 0.27, 0.03])
        self.assertEqual(u.ks_scale, 2.0)
        u._kickstart_adapt(0.5)
        self.assertEqual([round(x, 9) for x in u.ks_shares], [0.65, 0.32, 0.03])
        self.assertEqual(u.ks_scale, 1.0)
        # a single coef: only s
        one = self.trainer({"coef": 0.5, "adapt": {}})
        one._kickstart_adapt(0.5)
        self.assertAlmostEqual(one.ks_scale, 0.8)
        self.assertAlmostEqual(one.kickstart_coef(), 0.4)

    def test_adapt_in_the_update_and_through_a_checkpoint(self):
        ks = {"tiers": TIERS, "adapt": {"skip_target": 0.1}}
        # many KL skips: s down, share from the 0.4 tier to the 0.1 tier
        tr = self.trainer(ks, lr_actor=3e-3, kl_mode="skip", target_kl_skip=1e-4)
        m = tr.update(self.buf)["kickstart"]
        self.assertGreater(m["adapt"]["skip_share"], 0.1)
        self.assertEqual((m["scale"], m["shares"]), (1.0, [0.1, 0.25, 0.65]))
        self.assertAlmostEqual(m["adapt"]["scale_next"], 0.8)
        self.assertEqual([round(x, 9) for x in m["adapt"]["shares_next"]], [0.15, 0.25, 0.6])
        st = self.roundtrip(tr.state_dict())
        self.assertAlmostEqual(st["kickstart_scale"], 0.8)
        back = self.trainer(ks)
        back.load_state_dict(st)
        self.assertAlmostEqual(back.ks_scale, 0.8)
        self.assertEqual([round(x, 9) for x in back.ks_shares], [0.15, 0.25, 0.6])
        m2 = back.update(self.bufs[1])["kickstart"]
        self.assertAlmostEqual(m2["scale"], 0.8)
        self.assertEqual([t["share"] for t in m2["tiers"]], m2["shares"])
        # a changed tier list starts from its configured shares (s is kept, within the new bounds)
        other = self.trainer({"tiers": [[0.1, 0.5], [0.4, 0.5]], "adapt": {"min_scale": 0.9}})
        other.load_state_dict(st)
        self.assertEqual((other.ks_shares, other.ks_scale), ([0.5, 0.5], 0.9))
        # no KL skips: s back up (capped at max_scale 1) and share back to the 0.4 tier
        calm = self.trainer(ks)
        calm.load_state_dict(st)
        m3 = calm.update(self.buf)["kickstart"]
        self.assertEqual(m3["adapt"]["skip_share"], 0.0)
        self.assertEqual((calm.ks_scale, [round(x, 9) for x in calm.ks_shares]), (1.0, [0.1, 0.25, 0.65]))
        # not active (no labels, or decayed): nothing moves
        idle = self.trainer(ks, lr_actor=3e-3, kl_mode="skip", target_kl_skip=1e-4)
        b = copy.copy(self.buf)
        b.teacher = torch.full_like(self.buf.teacher, -1)
        b.teacher_name = torch.full_like(self.buf.teacher_name, -1)
        m4 = idle.update(b)["kickstart"]
        self.assertIsNone(m4["adapt"]["skip_share"])
        self.assertEqual((idle.ks_scale, idle.ks_shares), (1.0, [0.1, 0.25, 0.65]))
        done = self.trainer(dict(ks, start_round=0, decay_rounds=1), lr_actor=3e-3, kl_mode="skip",
                            target_kl_skip=1e-4)
        done.round = 5
        self.assertIsNone(done.update(self.buf)["kickstart"]["adapt"]["skip_share"])
        self.assertEqual(done.ks_scale, 1.0)

    def test_the_start_round_of_a_running_job_is_kept(self):
        # a checkpoint of the single-coef version (kickstart_start 504) resumed with tiers and adapt
        old = self.trainer({"coef": 0.5, "decay_rounds": 40})
        old.round = 504
        old.kickstart_coef()
        old.round = 510
        st = self.roundtrip(old.state_dict())
        self.assertEqual(st["kickstart_start"], 504)
        new = self.trainer({"tiers": TIERS, "decay_rounds": 40, "adapt": {"skip_target": 0.1}})
        new.load_state_dict(st)
        self.assertEqual((new.ks_start, new.kickstart_decay(), new.ks_scale), (504, 1.0 - 6 / 40, 1.0))
        self.assertEqual(new.ks_shares, [0.1, 0.25, 0.65])


def dict_cfg(cfg, kickstart):
    c = copy.deepcopy(cfg)
    c.ppo.kickstart = kickstart
    return c


if __name__ == "__main__":
    unittest.main()
