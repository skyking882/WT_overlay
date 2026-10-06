import copy
import unittest

import common  # noqa: F401
import torch

from rl.ppo import BetaSchedule, EntropySchedule, PPOTrainer, clipped_surrogate, ref_kl, actor_window
from test_rollout import DEV, make


class Surrogate(unittest.TestCase):
    def test_clipping_of_the_joint_ratio_hand_values(self):
        r = torch.tensor([1.3, 1.3, 0.7, 0.7, 1.05, 1.05])
        a = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
        got = clipped_surrogate(r, a, 0.15)
        want = torch.tensor([1.15, -1.3, 0.7, -0.85, 1.05, -1.05])
        self.assertTrue(torch.allclose(got, want, atol=1e-6))

    def test_joint_ratio_is_exp_of_summed_head_logprob_differences(self):
        torch.manual_seed(0)
        cfg, actor, critic, sm = make(burn=0, seg=8)
        buf, _ = sm.collect(actor)
        idx = buf.window_index(torch.tensor([0]), torch.tensor([0]))
        b = buf.store.gather(idx)
        for p in actor.parameters():
            p.data += 0.01 * torch.randn_like(p)
        out, seg = actor_window(actor, b, buf.h_actor[0][:1], 0, b.act, grad=False)
        m = seg.valid.reshape(-1)
        joint = (out.logp.sum(-1) - buf.logp[idx].reshape(-1))[m]
        per_head_sum = out.logp.sum(-1)[m] - buf.logp[idx].reshape(-1)[m]
        self.assertLess((joint - per_head_sum).abs().max().item(), 1e-6)
        self.assertGreater(joint.abs().max().item(), 1e-4)       # the perturbation actually moved the policy


class Schedules(unittest.TestCase):
    def test_entropy_schedule(self):
        s = EntropySchedule(0.01, 0.003, 1e6, 5e6, 0.05)
        self.assertAlmostEqual(s.coef(0), 0.01)
        self.assertAlmostEqual(s.coef(1e6), 0.01)
        self.assertAlmostEqual(s.coef(3e6), 0.0065)
        self.assertAlmostEqual(s.coef(5e6), 0.003)
        self.assertAlmostEqual(s.coef(9e6), 0.003)

    def test_entropy_decay_pauses_on_collapse(self):
        s = EntropySchedule(0.01, 0.003, 1e6, 5e6, 0.05)
        s.observe(0.4, 1e6)                    # healthy
        self.assertFalse(s.paused)
        self.assertAlmostEqual(s.coef(3e6), 0.0065)
        s.observe(0.01, 1e6)                   # collapsed: this round's decisions do not advance the schedule
        self.assertTrue(s.paused)
        self.assertAlmostEqual(s.coef(3e6), 0.01 + (0.003 - 0.01) * (2e6 - 1e6) / 4e6)
        st = s.state_dict()
        s2 = EntropySchedule(0.01, 0.003, 1e6, 5e6, 0.05)
        s2.load_state_dict(st)
        self.assertAlmostEqual(s2.coef(3e6), s.coef(3e6))

    def test_bc_reference_beta_schedule(self):
        b = BetaSchedule(0.05, 2e5, 2e6)
        self.assertAlmostEqual(b.beta(0), 0.05)
        self.assertAlmostEqual(b.beta(2e5), 0.05)
        self.assertAlmostEqual(b.beta(1.1e6), 0.025)
        self.assertEqual(b.beta(2e6), 0.0)
        self.assertEqual(b.beta(5e6), 0.0)


class Update(unittest.TestCase):
    def setUp(self):
        self.cfg, self.actor, self.critic, self.sm = make(peaked=False)
        self.ref = copy.deepcopy(self.actor).eval()
        for p in self.ref.parameters():
            p.requires_grad_(False)
        self.buf, self.st = self.sm.collect(self.actor)
        self.sm.finish(self.buf, self.critic, self.ref, 0.995, 0.95)

    def trainer(self, ref=True):
        a, c = copy.deepcopy(self.actor), copy.deepcopy(self.critic)
        return PPOTrainer(self.cfg, a, c, self.ref if ref else None, DEV), a, c

    def test_first_minibatch_ratio_is_one_and_reference_kl_is_zero_at_the_start(self):
        self.cfg.ppo.lr_actor = 1e-12
        self.cfg.ppo.lr_critic = 1e-12
        tr, a, c = self.trainer()
        m = tr.update(self.buf)
        self.assertLess(m["kl_target"], 1e-6)
        self.assertEqual(m["clip_frac"], 0.0)
        self.assertLess(m["kl_ref"], 1e-6)
        self.assertEqual(m["actor_stopped_at_minibatch"], -1)
        self.assertEqual(m["illegal_actions_taken"], 0)
        self.assertEqual(m["mask_fallbacks"], 0)

    def test_target_kl_stops_the_remaining_actor_updates_but_not_the_critic(self):
        self.cfg.ppo.target_kl = 1e-9
        tr, a, c = self.trainer()
        a0 = [p.clone() for p in a.parameters()]
        c0 = [p.clone() for p in c.parameters()]
        m = tr.update(self.buf)
        self.assertEqual(m["actor_steps"], 1)
        self.assertEqual(m["actor_stopped_at_minibatch"], 2)
        self.assertGreater(m["kl_target_max"], 1e-9)             # the KL that triggered the stop is logged
        self.assertGreater(m["minibatches"], 2)                  # the critic kept training
        self.assertTrue(any((p - q).abs().max() > 0 for p, q in zip(a.parameters(), a0)))
        self.assertTrue(any((p - q).abs().max() > 0 for p, q in zip(c.parameters(), c0)))

    def test_critic_warmup_leaves_the_actor_untouched(self):
        self.cfg.ppo.critic_warmup_rounds = 1
        tr, a, c = self.trainer()
        a0 = [p.clone() for p in a.parameters()]
        c0 = [p.clone() for p in c.parameters()]
        m = tr.update(self.buf)
        self.assertEqual(m["actor_steps"], 0)
        self.assertTrue(all(torch.equal(p, q) for p, q in zip(a.parameters(), a0)))
        self.assertTrue(any((p - q).abs().max() > 0 for p, q in zip(c.parameters(), c0)))

    def test_normal_update_moves_policy_and_reference_kl_becomes_positive(self):
        tr, a, c = self.trainer()
        m = tr.update(self.buf)
        self.assertGreater(m["actor_steps"], 1)
        self.assertGreater(m["kl_target"], 0.0)
        self.assertEqual(m["decisions_in_round"], self.buf.n_valid())
        self.assertEqual(tr.decisions, self.buf.n_valid())
        self.assertEqual(tr.round, 1)
        self.assertAlmostEqual(m["value_scale"], max(m["return_std"], self.cfg.ppo.value_std_floor))
        # reference KL on a perturbed copy is positive and per-head values exist for the active heads
        idx = self.buf.window_index(torch.tensor([0, 1]), torch.tensor([0, 0]))
        b = self.buf.store.gather(idx)
        out, seg = actor_window(a, b, self.buf.h_actor[0][:2], self.buf.B, b.act[:, self.buf.B:], keep_dists=True, grad=False)
        lidx = self.buf.loss_index(torch.tensor([0, 1]), torch.tensor([0, 0])).reshape(-1)
        kl, heads, active = ref_kl(out, self.buf.ref_cat[lidx], self.buf.ref_ptr[lidx][:, :, :out.logp_all["target"].shape[-1]])
        self.assertGreater(float(kl[seg.valid.reshape(-1)].mean()), 0.0)
        self.assertTrue((heads >= -1e-6).all())

    def test_entropy_and_kl_statistics_cover_only_multi_option_heads(self):
        tr, a, c = self.trainer()
        m = tr.update(self.buf)
        for h, v in m["entropy_head"].items():
            self.assertTrue(0.0 <= v <= 1.0 + 1e-6, (h, v))
        self.assertTrue(0.0 <= m["entropy"] <= 1.0)
        self.assertIn("maneuver_ref", m["head_active_frac"])

    def test_trainer_state_round_trip(self):
        tr, a, c = self.trainer()
        tr.update(self.buf)
        st = tr.state_dict()
        a2, c2 = copy.deepcopy(a), copy.deepcopy(c)
        tr2 = PPOTrainer(self.cfg, a2, c2, self.ref, DEV)
        tr2.load_state_dict(st)
        self.assertEqual(tr2.decisions, tr.decisions)
        self.assertEqual(tr2.round, tr.round)
        for (k, v1), (_, v2) in zip(tr.opt_a.state_dict()["state"].items(), tr2.opt_a.state_dict()["state"].items()):
            self.assertTrue(torch.equal(v1["exp_avg"], v2["exp_avg"]))
        self.assertTrue(torch.equal(tr.gen.get_state(), tr2.gen.get_state()))


if __name__ == "__main__":
    unittest.main()
