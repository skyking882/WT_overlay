import unittest

import common  # noqa: F401
import torch

from rl.buffer import compute_gae


def t(x, dtype=torch.float32):
    return torch.tensor(x, dtype=dtype)


class GaeHandComputed(unittest.TestCase):
    """gamma_base = 0.25 and lambda_base = 0.16 make the per-step factors easy to do by hand:
    dt=0.5 -> gamma 0.5, lambda 0.4;  dt=1 -> gamma 0.25, lambda 0.16."""

    def run_toy(self, v_garbage=0.0):
        # columns: stream A (plain, bootstrapped at the end of the round), stream B (timeout at t=1
        # then padding), stream C (death at t=1 then a new episode)
        r = t([[1.0, 1.0, 0.0], [0.0, 1.0, -2.0], [2.0, 9.0, 1.0]])
        v = t([[1.0, 2.0, 1.0], [2.0, 3.0, 2.0], [3.0, 0.0, 3.0]])
        dt = t([[0.5, 1.0, 1.0], [1.0, 0.5, 1.0], [0.5, 1.0, 1.0]])
        valid = torch.tensor([[1, 1, 1], [1, 1, 1], [1, 0, 1]], dtype=torch.bool)
        done = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 0, 0]], dtype=torch.bool)
        trunc = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.bool)
        boot = t([[0, 0, 0], [0, 10.0, 0], [0, 0, 0]])
        end_v = t([4.0, 0.0, 2.0])
        end_valid = torch.tensor([True, False, True])
        if v_garbage:
            r = torch.where(valid, r, torch.full_like(r, v_garbage))
            v = torch.where(valid, v, torch.full_like(v, v_garbage))
            dt = torch.where(valid, dt, torch.full_like(dt, 2.0))
            end_v = torch.where(end_valid, end_v, torch.full_like(end_v, v_garbage))
        return compute_gae(r, v, done, trunc, valid, dt, boot, end_v, end_valid, 0.25, 0.16), valid

    def test_values_match_hand_calculation(self):
        (adv, ret), valid = self.run_toy()
        # stream A: delta2 = 2 + 0.5*4 - 3 = 1;  delta1 = 0 + 0.25*3 - 2 = -1.25;  delta0 = 1 + 0.5*2 - 1 = 1
        #           A2 = 1; A1 = -1.25 + 0.25*0.16*1 = -1.21; A0 = 1 + 0.5*0.4*(-1.21) = 0.758
        a = adv[:, 0]
        self.assertAlmostEqual(float(a[2]), 1.0, 6)
        self.assertAlmostEqual(float(a[1]), -1.21, 6)
        self.assertAlmostEqual(float(a[0]), 0.758, 6)
        # stream B: timeout at t=1 bootstraps from 10 (gamma 0.5): delta1 = 1 + 0.5*10 - 3 = 3, no carry
        #           delta0 = 1 + 0.25*3 - 2 = -0.25;  A0 = -0.25 + 0.25*0.16*3 = -0.13; padding step is 0
        b = adv[:, 1]
        self.assertAlmostEqual(float(b[1]), 3.0, 6)
        self.assertAlmostEqual(float(b[0]), -0.13, 6)
        self.assertEqual(float(b[2]), 0.0)
        # stream C: death at t=1 is terminal (next value 0): delta1 = -2 - 2 = -4, nothing flows back
        #           from the new episode at t=2 (delta2 = 1 + 0.25*2 - 3 = -1.5); delta0 = 0 + 0.25*2 - 1 = -0.5
        #           A0 = -0.5 + 0.25*0.16*(-4) = -0.66
        c = adv[:, 2]
        self.assertAlmostEqual(float(c[2]), -1.5, 6)
        self.assertAlmostEqual(float(c[1]), -4.0, 6)
        self.assertAlmostEqual(float(c[0]), -0.66, 6)
        # returns = advantage + value, zero on padding
        self.assertAlmostEqual(float(ret[0, 0]), 0.758 + 1.0, 6)
        self.assertEqual(float(ret[2, 1]), 0.0)

    def test_padding_garbage_does_not_leak(self):
        (a0, r0), _ = self.run_toy()
        (a1, r1), _ = self.run_toy(v_garbage=123.0)
        self.assertTrue(torch.allclose(a0, a1))
        self.assertTrue(torch.allclose(r0, r1))

    def test_timeout_is_bootstrapped_but_death_is_not(self):
        z = torch.zeros(1, 1)
        one = torch.ones(1, 1, dtype=torch.bool)
        no = torch.zeros(1, 1, dtype=torch.bool)
        dt = torch.full((1, 1), 1.0)
        boot = torch.full((1, 1), 8.0)
        args = (z, z, None, None, one, dt, boot, torch.zeros(1), torch.zeros(1, dtype=torch.bool), 0.5, 0.5)
        adv_to, _ = compute_gae(z, z, no, one, one, dt, boot, torch.zeros(1), torch.zeros(1, dtype=torch.bool), 0.5, 0.5)
        adv_dead, _ = compute_gae(z, z, one, no, one, dt, boot, torch.zeros(1), torch.zeros(1, dtype=torch.bool), 0.5, 0.5)
        self.assertAlmostEqual(float(adv_to), 0.5 * 8.0, 6)          # gamma * V(final obs)
        self.assertEqual(float(adv_dead), 0.0)

    def test_variable_dt_changes_discounting(self):
        # constant value 0, one reward at t=1 and nothing else: A0 = gamma0*lambda0*A1 with A1 = r1
        r = t([[0.0], [1.0]])
        v = torch.zeros(2, 1)
        valid = torch.ones(2, 1, dtype=torch.bool)
        no = torch.zeros(2, 1, dtype=torch.bool)
        for dt0 in (0.5, 1.0, 2.0):
            adv, _ = compute_gae(r, v, no, no, valid, t([[dt0], [1.0]]), torch.zeros(2, 1), torch.zeros(1),
                                 torch.zeros(1, dtype=torch.bool), 0.25, 0.16)
            want = (0.25 ** dt0) * (0.16 ** dt0) * 1.0 + 0.25 ** dt0 * 0.0
            self.assertAlmostEqual(float(adv[0, 0]), want, 6)

    def test_default_per_second_factors(self):
        # the training default: gamma = 0.995^dt, lambda = 0.95^dt with the real step dt = 20/48 s
        # (section 13). The 0.997997 / 0.979692 of the section-7 table are the same formula at 0.4 s.
        dt = 20 / 48
        self.assertAlmostEqual(0.995 ** 0.4, 0.997997, 6)
        self.assertAlmostEqual(0.95 ** 0.4, 0.979692, 6)
        self.assertAlmostEqual(0.995 ** dt, 0.997914, 6)
        r = t([[0.0], [1.0]])
        z = torch.zeros(2, 1)
        valid = torch.ones(2, 1, dtype=torch.bool)
        no = torch.zeros(2, 1, dtype=torch.bool)
        adv, _ = compute_gae(r, z, no, no, valid, torch.full((2, 1), dt), z, torch.zeros(1), torch.zeros(1, dtype=torch.bool))
        self.assertAlmostEqual(float(adv[0, 0]), (0.995 ** dt) * (0.95 ** dt) * 1.0, 6)


if __name__ == "__main__":
    unittest.main()
