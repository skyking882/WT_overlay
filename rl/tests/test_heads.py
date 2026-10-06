import random
import unittest

import common
import torch

from rl import spec
from rl.model import Actor


def run_actor(actor, obs_list, seed, actions=None, keep=False, greedy=False):
    b = common.batch_from_obs(obs_list)
    g = torch.Generator().manual_seed(seed)
    B = len(obs_list)
    x, e = actor.encode(b)
    h0 = torch.zeros(B, 256)
    hs, _ = actor.unroll(x, h0, b.first, b.valid)
    out = actor.heads_from_batch(b, hs, e, actions, gen=g, greedy=greedy, keep_dists=keep)
    return b, out


class Masks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.actor = Actor().eval()
        for p in self.actor.cat.parameters():       # make the policy peaked so masking matters
            if p.dim() > 1:
                torch.nn.init.normal_(p, std=1.0)

    def make_obs(self, seed, B=12):
        rng = random.Random(seed)
        return [common.random_obs(rng, rng.randint(0, 9)) for _ in range(B)]

    def test_masked_options_are_never_sampled(self):
        illegal = 0
        total = 0
        with torch.no_grad():
            for seed in range(40):
                obs = self.make_obs(seed)
                _, out = run_actor(self.actor, obs, seed)
                acts = out.actions.view(len(obs), -1).tolist()
                for o, a in zip(obs, acts):
                    n = len(o["entities"])
                    bad = spec.illegal_heads(n, o["masks"], {h: a[i] for i, h in enumerate(spec.HEAD_NAMES)})
                    illegal += len(bad)
                    total += 1
                self.assertTrue(out.legal.all())
        self.assertEqual(illegal, 0)
        self.assertGreater(total, 400)

    def test_effective_masks_equal_python_reference(self):
        with torch.no_grad():
            for seed in range(15):
                obs = self.make_obs(100 + seed)
                b, out = run_actor(self.actor, obs, seed, keep=True)
                acts = out.actions.view(len(obs), -1).tolist()
                for i, (o, a) in enumerate(zip(obs, acts)):
                    n = len(o["entities"])
                    chosen = {}
                    for h in spec.SAMPLE_ORDER:
                        ref = spec.effective_mask(h, n, o["masks"], chosen)
                        got = out.eff[h][i].tolist()
                        self.assertEqual(got[:len(ref)], ref, (h, i))
                        self.assertFalse(any(got[len(ref):]), "padding positions must be masked")
                        chosen[h] = a[spec.HEAD_INDEX[h]]

    def test_single_option_heads_contribute_zero(self):
        with torch.no_grad():
            obs = self.make_obs(7, B=30)
            _, out = run_actor(self.actor, obs, 1)
        one = out.k == 1
        self.assertTrue(one.any())
        self.assertEqual(out.logp[one].abs().max().item(), 0.0)
        self.assertEqual(out.ent[one].abs().max().item(), 0.0)

    def test_entropy_is_normalised_by_log_k(self):
        with torch.no_grad():
            obs = self.make_obs(9, B=30)
            _, out = run_actor(self.actor, obs, 1, keep=True)
        self.assertTrue(((out.ent >= -1e-6) & (out.ent <= 1 + 1e-5)).all())
        # zero logits -> uniform over the legal options -> normalised entropy exactly 1 for K > 1
        a2 = Actor().eval()
        for p in a2.cat.parameters():
            torch.nn.init.zeros_(p)
        with torch.no_grad():
            _, out = run_actor(a2, obs, 1)
        for h in spec.CAT_HEADS:
            col = spec.HEAD_INDEX[h]
            multi = out.k[:, col] > 1
            if multi.any():
                self.assertLess((out.ent[multi, col] - 1).abs().max().item(), 1e-5, h)

    def test_probabilities_sum_to_one_on_legal_options_only(self):
        with torch.no_grad():
            obs = self.make_obs(11, B=10)
            _, out = run_actor(self.actor, obs, 3, keep=True)
        for h, lp in out.logp_all.items():
            p = lp.exp()
            self.assertLess((p.sum(-1) - 1).abs().max().item(), 1e-5, h)
            self.assertEqual(p[~out.eff[h]].abs().max().item() if (~out.eff[h]).any() else 0.0, 0.0)

    def test_joint_logp_is_sum_over_heads_and_teacher_forcing_reproduces_it(self):
        with torch.no_grad():
            obs = self.make_obs(13, B=16)
            b, out = run_actor(self.actor, obs, 5)
            _, out2 = run_actor(self.actor, obs, 99, actions=out.actions.view(len(obs), 1, -1))
        self.assertLess((out.logp - out2.logp).abs().max().item(), 1e-6)
        self.assertEqual(out2.actions.tolist(), out.actions.tolist())
        joint = out.logp.sum(-1)
        self.assertLess((joint - out2.logp.sum(-1)).abs().max().item(), 1e-5)

    def test_rows_are_selected_by_the_sampled_conditioning_heads(self):
        """Every table kind of 13.1: with one True per row the sampled action is fully determined by
        the row that the earlier choices select (view_mode; view_mode + ref is none; target)."""
        rng = random.Random(21)
        obs = []
        for _ in range(60):
            n = rng.randint(1, 6)
            o = common.random_obs(rng, n)
            for h in spec.HEAD_NAMES:                     # collapse every row to a single True option
                shape = spec.mask_shape(h, n)
                def collapse(m, depth):
                    if depth == len(shape) - 1:
                        i = rng.randrange(len(m))
                        return [j == i for j in range(len(m))]
                    return [collapse(r, depth + 1) for r in m]
                o["masks"][h] = collapse(o["masks"][h], 0)
            obs.append(o)
        with torch.no_grad():
            _, out = run_actor(self.actor, obs, 2)
        acts = out.actions.tolist()
        seen = set()
        for o, a in zip(obs, acts):
            n = len(o["entities"])
            chosen = {}
            for h in spec.SAMPLE_ORDER:
                row = spec.select_row(h, n, o["masks"], chosen)
                self.assertEqual(sum(row), 1)
                self.assertEqual(a[spec.HEAD_INDEX[h]], row.index(True), h)
                chosen[h] = a[spec.HEAD_INDEX[h]]
            seen.add(("ref_null", chosen["maneuver_ref"] == n))
            seen.add(("vm", chosen["view_mode"]))
        self.assertGreaterEqual(len(seen), 4)              # both ref branches and several view modes were hit

    def test_empty_selected_row_falls_back_to_the_default_option_and_is_reported(self):
        rng = random.Random(5)
        o = common.random_obs(rng, 4)
        o["masks"]["speed"] = [False, False, False]
        with torch.no_grad():
            _, out = run_actor(self.actor, [o], 1)
        col = spec.HEAD_INDEX["speed"]
        self.assertTrue(out.fallback[0, col])
        self.assertEqual(int(out.actions[0, col]), spec.DEFAULT_OPTION["speed"])
        self.assertEqual(float(out.logp[0, col]), 0.0)


if __name__ == "__main__":
    unittest.main()
