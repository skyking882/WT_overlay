"""ppo.ent_floor: a per-head entropy floor, an adaptive multiplier on the head's entropy-bonus scale (cf. test_head_kl)."""
import copy
import math
import os
import tempfile
import unittest

import common  # noqa: F401
import torch

from rl import config as C
from rl import train as T
from rl.ppo import PPOTrainer
from test_rollout import DEV, make

H = "speed"         # more than one legal option on every step of the fake env
H2 = "vertical"     # ... on very few
STATE_KEYS = {"opt_a", "opt_c", "ent", "decisions", "round", "gen", "head_kl_coef"}   # the trainer state without it


class EntFloor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg, cls.actor, cls.critic, sm = make(steps=40, streams=8, peaked=True)
        cls.cfg.ppo.target_kl = 1e9                 # every minibatch updates the actor (peaked: else 1 of 15)
        cls.buf, cls.st = sm.collect(cls.actor)
        sm.finish(cls.buf, cls.critic, None, 0.995, 0.95)
        cls.sm = sm
        _, _, cls.m0 = cls.run_update(cls)
        cls.e = cls.m0["entropy_head"][H]

    def run_update(self, ent_floor=None, state=None, head_scale=None, lr=None):
        cfg = copy.deepcopy(self.cfg)
        cfg.ppo.ent_floor = ent_floor or {}
        if head_scale is not None:
            cfg.ppo.ent_head_scale = head_scale
        if lr:
            cfg.ppo.lr_actor, cfg.ppo.target_kl = lr, 1e9      # no early stop: every minibatch moves the actor
        a, c = copy.deepcopy(self.actor), copy.deepcopy(self.critic)
        tr = PPOTrainer(cfg, a, c, None, DEV)
        if state is not None:
            tr.load_state_dict(state)
        m = tr.update(self.buf)
        return tr, a, m

    def state_with(self, mult, head=H):
        """A fresh trainer's state (no update yet) with m_head = mult."""
        st = PPOTrainer(dict_cfg(self.cfg, {head: {"floor": 0.5, "max_scale": 1e9}}), copy.deepcopy(self.actor),
                        copy.deepcopy(self.critic), None, DEV).state_dict()
        st["ent_floor_mult"] = {head: mult}
        return st

    def test_bad_configs_are_refused(self):
        good = {H: {"floor": 0.01, "up": 2.0, "down": 1.2, "max_scale": 1.0}}
        PPOTrainer(dict_cfg(self.cfg, good), copy.deepcopy(self.actor), copy.deepcopy(self.critic), None, DEV)
        c = C.preset("smoke")
        c.ppo.ent_floor = good
        c.validate()
        for bad in ({"vertcal": {"floor": 0.01}}, {H: {"floor": 0.0}}, {H: {"floor": 1.0}}, {H: {"floor": -0.1}},
                    {H: {}}, {H: {"up": 2.0}}, {H: 0.01}, {H: {"floor": 0.01, "up": 1.0}},
                    {H: {"floor": 0.01, "down": 0.9}}, {H: {"floor": 0.01, "max_scale": 0.5}},
                    {H: {"floor": 0.01, "beta": 1}}, {H: {"floor": "x"}}, {H: {"floor": float("nan")}}, [H]):
            with self.assertRaises(ValueError, msg=repr(bad)):
                PPOTrainer(dict_cfg(self.cfg, bad), copy.deepcopy(self.actor), copy.deepcopy(self.critic), None, DEV)
            c.ppo.ent_floor = bad
            with self.assertRaises(ValueError, msg=repr(bad)):
                c.validate()

    def test_off_adds_nothing_and_a_unit_multiplier_changes_nothing(self):
        tr0, a0, m0 = self.run_update()
        self.assertNotIn("ent_floor", m0)
        self.assertEqual(set(tr0.state_dict()), STATE_KEYS)
        self.assertEqual(tr0.ent_floor_mult, {})
        line = T.summary_line(dict(T.round_record(self.cfg, tr0, self.buf, self.st, m0, self.sm), time=TIMES))
        self.assertNotIn("entfloor", line)
        # first round with the floor on: m_h = 1, so the update is the same to the bit
        tr1, a1, m1 = self.run_update({H: {"floor": 0.5}, H2: {"floor": 0.01}})
        self.assertTrue(all(torch.equal(p, q) for p, q in zip(a0.parameters(), a1.parameters())))
        self.assertEqual(nan_free({k: v for k, v in m1.items() if k not in ("ent_floor", "t_update")}),
                         nan_free({k: v for k, v in m0.items() if k != "t_update"}))
        self.assertEqual(set(tr1.state_dict()), STATE_KEYS | {"ent_floor_mult"})
        self.assertEqual(set(m1["ent_floor"][H]), {"entropy", "floor", "mult", "mult_next", "scale"})
        # the round statistic is the logged per-head entropy
        self.assertEqual(m1["ent_floor"][H]["entropy"], m0["entropy_head"][H])
        self.assertEqual(m1["ent_floor"][H]["mult"], 1.0)

    def test_a_larger_multiplier_changes_the_update_and_raises_the_head_entropy(self):
        _, a1, m1 = self.run_update({H: {"floor": 0.5}}, state=self.state_with(1.0), lr=3e-3)
        _, a2, m2 = self.run_update({H: {"floor": 0.5, "max_scale": 1e4}}, state=self.state_with(1e3), lr=3e-3)
        self.assertFalse(all(torch.equal(p, q) for p, q in zip(a1.parameters(), a2.parameters())))
        self.assertEqual(m2["ent_floor"][H]["mult"], 1e3)
        self.assertGreater(m2["entropy_head"][H], m1["entropy_head"][H])
        # the effective scale is ent_head_scale x m_h
        _, _, m3 = self.run_update({H: {"floor": 0.5}}, state=self.state_with(4.0), head_scale={H: 0.5})
        self.assertAlmostEqual(m3["ent_floor"][H]["scale"], 2.0)

    def test_the_multiplier_goes_up_below_the_floor_and_down_above_twice_the_floor(self):
        e = self.e
        self.assertTrue(0.05 < e < 0.9, e)          # room for floors on both sides
        for floor, kw, start, expect in (
                (min(2 * e, 0.99), {}, None, 1.5),                       # below the floor: x up
                (min(2 * e, 0.99), {"up": 2.0}, None, 2.0),
                (min(2 * e, 0.99), {}, 4.0, 6.0),
                (min(2 * e, 0.99), {"max_scale": 5.0}, 4.0, 5.0),       # capped at max_scale
                (0.75 * e, {}, 4.0, 4.0),                                # between floor and 2 floor: kept
                (e / 10, {}, None, 1.0),                                 # above 2 floor: / down, not below 1
                (e / 10, {}, 1.2, 1.0),
                (e / 10, {}, 4.0, 4.0 / 1.5),
                (e / 10, {"down": 2.0}, 4.0, 2.0)):
            cfg = {H: dict({"floor": floor}, **kw)}
            tr, _, m = self.run_update(cfg, state=None if start is None else self.state_with(start))
            d = m["ent_floor"][H]
            self.assertAlmostEqual(d["mult"], 1.0 if start is None else start)
            self.assertAlmostEqual(d["mult_next"], expect, msg=repr((floor, kw, start)))
            self.assertAlmostEqual(tr.ent_floor_mult[H], expect)
            self.assertEqual(d["entropy"], m["entropy_head"][H])
            self.assertAlmostEqual(d["scale"], d["mult"])                # ent_head_scale 1

    def test_no_actor_step_keeps_the_multiplier(self):
        cfg = copy.deepcopy(self.cfg)
        cfg.ppo.critic_warmup_rounds = 1
        cfg.ppo.ent_floor = {H: {"floor": 0.99}}
        tr = PPOTrainer(cfg, copy.deepcopy(self.actor), copy.deepcopy(self.critic), None, DEV)
        m = tr.update(self.buf)
        self.assertEqual(m["ent_floor"][H], {"entropy": None, "floor": 0.99, "mult": 1.0, "mult_next": 1.0,
                                             "scale": 1.0})
        self.assertEqual(tr.ent_floor_mult[H], 1.0)

    def test_the_multiplier_persists_in_checkpoints(self):
        ef = {H: {"floor": 0.99}}                       # always below: x1.5 per round
        tr, _, m = self.run_update(ef)
        self.assertAlmostEqual(tr.ent_floor_mult[H], 1.5)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ck.pt")
            torch.save({"trainer": tr.state_dict()}, path)
            st = torch.load(path, map_location="cpu", weights_only=False)["trainer"]
        tr2, _, m2 = self.run_update(ef, state=st)
        self.assertAlmostEqual(m2["ent_floor"][H]["mult"], 1.5)
        self.assertAlmostEqual(tr2.ent_floor_mult[H], 2.25)
        # an older checkpoint without the key starts at 1; a head no longer configured is dropped, a new one starts at 1
        old = {k: v for k, v in st.items() if k != "ent_floor_mult"}
        _, _, m3 = self.run_update(ef, state=old)
        self.assertEqual(m3["ent_floor"][H]["mult"], 1.0)
        tr4, _, m4 = self.run_update(None, state=st)
        self.assertEqual(tr4.ent_floor_mult, {})
        self.assertNotIn("ent_floor", m4)
        self.assertNotIn("ent_floor_mult", tr4.state_dict())
        tr5, _, m5 = self.run_update({H2: {"floor": 0.99}}, state=st)
        self.assertEqual(set(tr5.ent_floor_mult), {H2})
        self.assertEqual(m5["ent_floor"][H2]["mult"], 1.0)
        # a restored multiplier is kept inside [1, max_scale]
        _, _, m6 = self.run_update({H: {"floor": 0.99, "max_scale": 1.2}}, state=st)
        self.assertAlmostEqual(m6["ent_floor"][H]["mult"], 1.2)

    def test_summary_line_and_record_carry_the_floor(self):
        tr, a, m = self.run_update({H: {"floor": 0.99}})
        rec = T.round_record(self.cfg, tr, self.buf, self.st, m, self.sm)
        self.assertEqual(rec["ent_floor"], m["ent_floor"])
        line = T.summary_line(dict(rec, time=TIMES))
        self.assertIn("entfloor[%s] %.2e x1" % (H, m["ent_floor"][H]["entropy"]), line)


TIMES = dict(sample=1., inference=1., env_wait=1., postpass=1., update=1.)


def nan_free(x):
    """NaN -> "nan" (kl_ref is NaN without a reference policy and NaN != NaN)."""
    if isinstance(x, dict):
        return {k: nan_free(v) for k, v in x.items()}
    return "nan" if isinstance(x, float) and math.isnan(x) else x


def dict_cfg(cfg, ent_floor):
    cfg = copy.deepcopy(cfg)
    cfg.ppo.ent_floor = ent_floor
    return cfg


if __name__ == "__main__":
    unittest.main()
