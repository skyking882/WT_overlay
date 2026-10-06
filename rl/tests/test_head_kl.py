"""ppo.head_kl: a per-head KL constraint to the behaviour policy (here 'weapon') with a per-round adapted coefficient."""
import copy
import os
import tempfile
import unittest

import common  # noqa: F401
import torch

from rl import spec
from rl import train as T
from rl.ppo import PPOTrainer
from test_late_credit import duel_cfg
from test_rollout import DEV, make

W = "weapon"


class HeadKL(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg, cls.actor, cls.critic, sm = make(steps=40, streams=8, peaked=False)
        cls.buf, cls.st = sm.collect(cls.actor)
        sm.finish(cls.buf, cls.critic, None, 0.995, 0.95)
        cls.sm = sm

    def run_update(self, head_kl=None, lr=None, state=None):
        cfg = copy.deepcopy(self.cfg)
        cfg.ppo.head_kl = head_kl or {}
        if lr:
            cfg.ppo.lr_actor, cfg.ppo.target_kl = lr, 1e9      # no early stop: every minibatch moves the actor
        a, c = copy.deepcopy(self.actor), copy.deepcopy(self.critic)
        tr = PPOTrainer(cfg, a, c, None, DEV)
        if state is not None:
            tr.load_state_dict(state)
        m = tr.update(self.buf)
        return tr, a, m

    def test_off_adds_nothing_and_a_zero_coefficient_changes_no_parameter(self):
        _, a0, m0 = self.run_update()
        self.assertNotIn("head_kl", m0)
        self.assertGreater(m0["kl_target_head"].get(W, 0.0), 0.0)        # the per-head KL is logged regardless
        tr, a1, m1 = self.run_update({W: {"target": 1e-3, "coef": 0.0, "coef_min": 0.0}})
        self.assertTrue(all(torch.equal(p, q) for p, q in zip(a0.parameters(), a1.parameters())))
        self.assertEqual(m0["kl_target"], m1["kl_target"])
        self.assertEqual(m1["head_kl"][W]["coef"], 0.0)

    def test_on_the_kl_is_positive_logged_and_held_down_by_a_large_coefficient(self):
        _, _, free = self.run_update({W: {"target": 1e-3, "coef": 0.0, "coef_min": 0.0}}, lr=3e-3)
        _, _, held = self.run_update({W: {"target": 1e-3, "coef": 1000.0, "coef_max": 1e4}}, lr=3e-3)
        kf, kh = free["head_kl"][W]["kl"], held["head_kl"][W]["kl"]
        self.assertGreater(kf, 0.0)
        self.assertGreater(kh, 0.0)
        self.assertLess(kh, 0.5 * kf)
        # the round mean is the same quantity as the per-head diagnostic
        self.assertAlmostEqual(free["head_kl"][W]["kl"], free["kl_target_head"][W], places=6)
        self.assertEqual(set(held["head_kl"][W]), {"kl", "coef", "coef_next", "target"})

    def test_the_coefficient_adapts_toward_the_target_within_its_bounds(self):
        _, _, m = self.run_update()
        kl = m["kl_target_head"][W]
        for target, coef, lo, hi, expect in ((kl / 10, 1.0, 0.01, 100.0, 1.5), (kl * 10, 1.0, 0.01, 100.0, 1 / 1.5),
                                             (kl, 1.0, 0.01, 100.0, 1.0), (kl / 1e6, 90.0, 0.01, 100.0, 100.0),
                                             (kl * 1e6, 0.012, 0.01, 100.0, 0.01)):
            tr, _, m = self.run_update({W: {"target": target, "coef": coef, "coef_min": lo, "coef_max": hi}})
            self.assertAlmostEqual(m["head_kl"][W]["coef"], coef)
            self.assertAlmostEqual(m["head_kl"][W]["coef_next"], expect)
            self.assertAlmostEqual(tr.head_kl_coef[W], expect)
            if coef == 1.0:                         # a unit coef only nudges this round's weapon KL
                self.assertAlmostEqual(m["head_kl"][W]["kl"], kl, delta=0.2 * kl)

    def test_the_adapted_coefficient_persists_in_checkpoints(self):
        hk = {W: {"target": 1e-9, "coef": 2.0}}
        tr, a, m = self.run_update(hk)
        self.assertAlmostEqual(tr.head_kl_coef[W], 3.0)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ck.pt")
            torch.save({"trainer": tr.state_dict()}, path)
            st = torch.load(path, map_location="cpu", weights_only=False)["trainer"]
        tr2, _, m2 = self.run_update(hk, state=st)
        self.assertAlmostEqual(m2["head_kl"][W]["coef"], 3.0)
        self.assertAlmostEqual(tr2.head_kl_coef[W], 4.5)
        # an older checkpoint without the key keeps the configured coef; a head no longer configured is ignored
        old = {k: v for k, v in st.items() if k != "head_kl_coef"}
        tr3, _, m3 = self.run_update(hk, state=old)
        self.assertAlmostEqual(m3["head_kl"][W]["coef"], 2.0)
        tr4, _, m4 = self.run_update(None, state=st)
        self.assertEqual(tr4.head_kl_coef, {})
        self.assertNotIn("head_kl", m4)

    def test_summary_line_and_record_carry_the_head_kl(self):
        tr, a, m = self.run_update({W: {"target": 2e-4, "coef": 1.0}})
        rec = T.round_record(self.cfg, tr, self.buf, self.st, m, self.sm)
        self.assertEqual(rec["head_kl"], m["head_kl"])
        line = T.summary_line(dict(rec, time=dict(sample=1., inference=1., env_wait=1., postpass=1., update=1.)))
        self.assertIn("kl[weapon] %.2e coef 1" % m["head_kl"][W]["kl"], line)
        self.assertEqual(spec.HEAD_INDEX[W], spec.HEAD_NAMES.index(W))


class ValidFraction(unittest.TestCase):
    def test_settle_ticks_do_not_lower_the_valid_fraction(self):
        # duel_cfg(6, 11), T = 8: p0 down at step 6, its missile kills p1 at step 11 -> every round is extended to 16
        # ticks; duel_cfg(2, 5), T = 40: 5-step matches end with the round, nothing owed -> no settle ticks (as before)
        for cfg_fn, T0, settles in ((duel_cfg(6, 11), 8, True), (duel_cfg(2, 5), 40, False)):
            cfg, actor, critic, sm = make(streams=4, steps=T0, seg=8, burn=4, cfg_fn=cfg_fn)
            tr = PPOTrainer(cfg, actor, critic, None, DEV)
            for r in range(2):
                buf, st = sm.collect(actor)
                sm.finish(buf, critic, None, 0.995, 0.95)
                rec = T.round_record(cfg, tr, buf, st, tr.update(buf), sm)
                v = buf.loss_view(buf.store.valid)
                self.assertAlmostEqual(rec["valid_fraction"], float(v[:T0].sum()) / (4 * T0))
                if settles:
                    self.assertEqual((buf.T, rec["sampler_stats"]["settle_ticks"]), (16, 8))
                    self.assertGreater(rec["valid_fraction"], rec["decisions_in_round"] / float(4 * 16))
                else:
                    self.assertEqual(buf.T, T0)
                    self.assertAlmostEqual(rec["valid_fraction"], rec["decisions_in_round"] / float(4 * T0))


if __name__ == "__main__":
    unittest.main()
