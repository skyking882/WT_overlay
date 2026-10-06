"""Rollout / update consistency: log-probs, stored hidden states, burn-in, tail, padding."""
import copy
import unittest

import common
import torch

from rl import spec
from rl.model import Actor, Critic
from rl.pool import WorkerPool
from rl.ppo import PPOTrainer, actor_window
from rl.rollout import Sampler

DEV = torch.device("cpu")


def make(seed=1, burn=4, seg=8, steps=24, streams=8, cfg_fn=None, peaked=True):
    cfg = common.smoke_cfg()
    cfg.rollout.burn_in, cfg.rollout.seg_len, cfg.rollout.steps, cfg.rollout.n_streams = burn, seg, steps, streams
    cfg.ppo.minibatch_segments = 6
    if cfg_fn:
        cfg_fn(cfg)
    torch.manual_seed(seed)
    actor, critic = Actor(), Critic()
    if peaked:
        for p in actor.cat.parameters():
            if p.dim() > 1:
                torch.nn.init.normal_(p, std=0.5)
    pool = WorkerPool(cfg.workers, cfg.env).start()
    sm = Sampler(cfg, pool, DEV, seed)
    sm.start(100 + seed)
    return cfg, actor, critic, sm


class LogProbConsistency(unittest.TestCase):
    def check_rounds(self, burn, rounds=3):
        cfg, actor, critic, sm = make(burn=burn)
        worst = 0.0
        for _ in range(rounds):
            buf, st = sm.collect(actor)
            sm.finish(buf, critic, None, 0.995, 0.95)
            B = buf.B
            n = 0
            for s in range(buf.S):
                for k in range(buf.K):
                    widx = buf.window_index(torch.tensor([s]), torch.tensor([k]))
                    b = buf.store.gather(widx)
                    out, seg = actor_window(actor, b, buf.h_actor[k][s].unsqueeze(0), B, b.act[:, B:], grad=False)
                    m = seg.valid.reshape(-1)
                    if m.any():
                        old = buf.logp[widx[:, B:]].reshape(-1)
                        worst = max(worst, (out.logp.sum(-1) - old).abs()[m].max().item())
                        self.assertTrue(out.legal[m].all())
                        self.assertFalse(out.fallback[m].any())
                        n += int(m.sum())
            self.assertGreater(n, 50)
        return worst

    def test_update_logp_equals_rollout_logp_with_burn_in(self):
        self.assertLess(self.check_rounds(burn=4), 1e-4)

    def test_update_logp_equals_rollout_logp_without_burn_in(self):
        self.assertLess(self.check_rounds(burn=0), 1e-4)

    def test_stepwise_equals_sequence_evaluation(self):
        """T=1 evaluation chained with the stored states reproduces the whole-sequence forward."""
        cfg, actor, critic, sm = make(burn=0, seg=24, steps=24, streams=4)
        buf, _ = sm.collect(actor)
        s = 0
        idx = torch.arange(24).unsqueeze(0) * buf.S + s + buf.P * buf.S
        b = buf.store.gather(idx)
        x, e = actor.encode(b)
        hs, _ = actor.unroll(x, buf.h_actor[0][s:s + 1], b.first, b.valid)
        h = buf.h_actor[0][s:s + 1]
        for t in range(24):
            bt = b.time_slice(t, t + 1)
            xt, et = actor.encode(bt)
            hst, h = actor.unroll(xt, h, bt.first, bt.valid)
            self.assertLess((hst[:, 0] - hs[:, t]).abs().max().item(), 1e-5)


class Abort(unittest.TestCase):
    def test_collect_can_be_aborted_between_ticks(self):
        from rl.rollout import RoundAborted
        cfg, actor, critic, sm = make()
        ticks = []
        def abort():
            ticks.append(1)
            return len(ticks) > 5
        with self.assertRaises(RoundAborted):
            sm.collect(actor, abort=abort)
        self.assertEqual(len(ticks), 6)


class HiddenStateAndBuffers(unittest.TestCase):
    def test_tail_and_stored_states_carry_across_rounds(self):
        cfg, actor, critic, sm = make()
        b1, _ = sm.collect(actor)
        sm.finish(b1, critic, None, 0.995, 0.95)
        K, S, T, P = b1.K, b1.S, b1.T, b1.P
        carried_actor = b1.h_actor[K].clone()
        carried_critic = b1.h_critic[K].clone()
        self.assertGreater(float(carried_actor.abs().sum()), 0.0)
        b2, _ = sm.collect(actor)
        self.assertTrue(torch.equal(b2.store.own[:P * S], b1.store.own[T * S:(T + P) * S]))
        self.assertTrue(torch.equal(b2.store.valid[:P * S], b1.store.valid[T * S:(T + P) * S]))
        self.assertTrue(torch.equal(b2.store.first[:P * S], b1.store.first[T * S:(T + P) * S]))
        g1 = b1.store.gather(torch.arange(T * S, (T + P) * S).view(P, S))
        g2 = b2.store.gather(torch.arange(0, P * S).view(P, S))
        self.assertTrue(torch.equal(g1.ent, g2.ent))
        self.assertTrue(torch.equal(g1.ent_n, g2.ent_n))
        self.assertTrue(torch.equal(g1.masks["maneuver"], g2.masks["maneuver"]))
        self.assertTrue(torch.equal(b2.h_actor[0], carried_actor))
        sm.finish(b2, critic, None, 0.995, 0.95)
        self.assertTrue(torch.equal(b2.h_critic[0], carried_critic))

    def test_first_flag_marks_exactly_the_episode_starts(self):
        cfg, actor, critic, sm = make(cfg_fn=lambda c: c.env.config.update({"max_steps": 12, "p_background_death": 0.05}))
        for _ in range(3):
            buf, st = sm.collect(actor)
            lv = buf.loss_view
            valid, first, done, trunc = lv(buf.store.valid), lv(buf.store.first), lv(buf.done), lv(buf.trunc)
            T, S = valid.shape
            starts = 0
            for s in range(S):
                for t in range(1, T):
                    if valid[t, s]:
                        expect = (not valid[t - 1, s]) or bool(done[t - 1, s] or trunc[t - 1, s])
                        self.assertEqual(bool(first[t, s]), expect, (t, s))
                        starts += int(first[t, s])
            self.assertGreater(starts, 0)

    def test_burn_in_steps_carry_no_gradient_and_no_loss(self):
        cfg, actor, critic, sm = make(burn=4, seg=8)
        buf, _ = sm.collect(actor)
        s, k = torch.tensor([0, 1]), torch.tensor([1, 2])
        widx = buf.window_index(s, k)
        b = buf.store.gather(widx)
        b.own.requires_grad_(True)
        out, seg = actor_window(actor, b, buf.h_actor[1][:2], buf.B, b.act[:, buf.B:])
        m = seg.valid.reshape(-1)
        out.logp.sum(-1)[m].sum().backward()
        g = b.own.grad
        self.assertEqual(float(g[:, :buf.B].abs().sum()), 0.0, "burn-in is gradient free")
        self.assertGreater(float(g[:, buf.B:].abs().sum()), 0.0)

    def test_streams_idle_when_agent_dies_and_loss_steps_are_only_valid_ones(self):
        cfg, actor, critic, sm = make(cfg_fn=lambda c: c.env.config.update({"p_background_death": 0.08, "max_steps": 200}))
        buf, st = sm.collect(actor)
        lv = buf.loss_view
        valid = lv(buf.store.valid)
        self.assertLess(int(valid.sum()), valid.numel())
        # invalid steps carry no data at all
        inv = ~valid
        self.assertEqual(float(lv(buf.reward)[inv].abs().sum()), 0.0)
        self.assertEqual(int(lv(buf.store.ent_n)[inv].sum()), 0)
        self.assertEqual(st["n_valid"], int(valid.sum()))


class PaddingIsExcluded(unittest.TestCase):
    def test_update_ignores_whatever_sits_in_invalid_slots(self):
        cfg, actor, critic, sm = make(cfg_fn=lambda c: c.env.config.update({"p_background_death": 0.08}))
        buf, st = sm.collect(actor)
        sm.finish(buf, critic, None, 0.995, 0.95)
        self.assertLess(buf.n_valid(), buf.T * buf.S)
        bad = copy.deepcopy(buf)
        inv_loss = ~bad.loss_view(bad.store.valid).reshape(-1)
        g = torch.Generator().manual_seed(5)
        for name in ("adv", "ret", "value", "boot_value"):
            x = getattr(bad, name)
            x[inv_loss] = torch.randn(int(inv_loss.sum()), generator=g) * 3
        bad.logp[bad.P * bad.S:][inv_loss] = torch.randn(int(inv_loss.sum()), generator=g)
        inv_all = ~bad.store.valid
        bad.reward[inv_all] = torch.randn(int(inv_all.sum()), generator=g)
        a1, c1 = copy.deepcopy(actor), copy.deepcopy(critic)
        a2, c2 = copy.deepcopy(actor), copy.deepcopy(critic)
        t1 = PPOTrainer(cfg, a1, c1, None, DEV)
        t2 = PPOTrainer(cfg, a2, c2, None, DEV)
        m1 = t1.update(buf)
        m2 = t2.update(bad)
        for k in ("pg_loss", "value_loss", "entropy", "kl_target", "clip_frac", "return_std", "adv_mean_raw", "explained_variance"):
            self.assertAlmostEqual(m1[k], m2[k], 5, k)
        self.assertEqual(m1["decisions_in_round"], buf.n_valid())
        for p, q in zip(a1.parameters(), a2.parameters()):
            self.assertLess((p - q).abs().max().item(), 1e-6)
        for p, q in zip(c1.parameters(), c2.parameters()):
            self.assertLess((p - q).abs().max().item(), 1e-6)

    def test_loss_and_grads_do_not_change_when_padding_is_added(self):
        """Extra invalid steps (in the middle and at the end) and an extra invalid row, filled with garbage."""
        from rl import bc as bcmod
        rng = __import__("random").Random(3)
        torch.manual_seed(0)
        actor = Actor()
        cfg = common.smoke_cfg()
        T = 6
        obs = [[common.random_obs(rng, rng.randint(0, 6)) for _ in range(T)] for _ in range(2)]
        def batch(rows, extra_mid=False, extra_tail=0, extra_row=False):
            from rl.encode import Decoded
            from rl import wire
            flat = []
            for r in rows:
                steps = list(r)
                if extra_mid:
                    steps.insert(3, common.random_obs(rng, 9))
                steps += [common.random_obs(rng, 5) for _ in range(extra_tail)]
                flat.append(steps)
            if extra_row:
                flat.append([common.random_obs(rng, 7) for _ in range(len(flat[0]))])
            W = len(flat[0])
            dec = Decoded([wire.pack_obs(o) for st in flat for o in st])
            b = dec.to_batch()
            def re(x):
                return x.view(len(flat), W, *x.shape[2:])
            from rl.encode import Batch
            valid = torch.ones(len(flat), W, dtype=torch.bool)
            if extra_mid:
                valid[:, 3] = False
            if extra_tail:
                valid[:, W - extra_tail:] = False
            if extra_row:
                valid[-1] = False
            masks = {k: v.view(len(flat), W, *v.shape[2:]) for k, v in b.masks.items()}
            # actions: first legal option per head for every step (deterministic, legal by construction)
            act = torch.zeros(len(flat), W, spec.N_HEADS, dtype=torch.long)
            for i, st in enumerate(flat):
                for j, o in enumerate(st):
                    n = len(o["entities"])
                    chosen = {}
                    for h in spec.SAMPLE_ORDER:
                        m = spec.effective_mask(h, n, o["masks"], chosen)
                        chosen[h] = m.index(True)
                    act[i, j] = torch.tensor([chosen[h] for h in spec.HEAD_NAMES])
            return Batch(re(b.own), re(b.ent), re(b.ent_n), re(b.prev), re(b.truth), re(b.truth_n), masks,
                         valid, torch.zeros(len(flat), W, dtype=torch.bool), re(b.dt), act)
        def loss_grads(b):
            actor.zero_grad()
            out, seg, lm, active, w = bcmod.bc_forward(actor, b, 0, cfg.bc, DEV)
            loss, _, _ = bcmod.bc_loss(out, active, w)
            loss.backward()
            return loss.item(), torch.cat([p.grad.flatten() for p in actor.parameters() if p.grad is not None])
        l0, g0 = loss_grads(batch(obs))
        for kw in ({"extra_tail": 3}, {"extra_mid": True}, {"extra_row": True}, {"extra_mid": True, "extra_tail": 2, "extra_row": True}):
            l1, g1 = loss_grads(batch(obs, **kw))
            self.assertAlmostEqual(l0, l1, 4, kw)
            self.assertLess((g0 - g1).abs().max().item(), 1e-4, kw)

    def test_garbage_in_padded_entity_slots_changes_nothing(self):
        torch.manual_seed(0)
        rng = __import__("random").Random(4)
        actor, critic = Actor().eval(), Critic().eval()
        obs = [common.random_obs(rng, n) for n in (1, 4, 0, 6)]
        b = common.batch_from_obs(obs)
        b2 = copy.copy(b)
        ent = b.ent.clone()
        tr = b.truth.clone()
        for i in range(ent.shape[0]):
            n = int(b.ent_n[i, 0])
            ent[i, 0, n:] = torch.randn_like(ent[i, 0, n:]) * 5
            m = int(b.truth_n[i, 0])
            tr[i, 0, m:] = torch.randn_like(tr[i, 0, m:]) * 5
        b2.ent, b2.truth = ent, tr
        with torch.no_grad():
            for net in (actor,):
                x1, e1 = net.encode(b); x2, e2 = net.encode(b2)
                self.assertLess((x1 - x2).abs().max().item(), 1e-5)
                hs, _ = net.unroll(x1, torch.zeros(4, 256), b.first, b.valid)
                o1 = net.heads_from_batch(b, hs, e1, None, gen=torch.Generator().manual_seed(1))
                o2 = net.heads_from_batch(b2, hs, e2, None, gen=torch.Generator().manual_seed(1))
                self.assertEqual(o1.actions.tolist(), o2.actions.tolist())
                self.assertLess((o1.logp - o2.logp).abs().max().item(), 1e-5)
            v1, _ = critic(b, torch.zeros(4, 256)); v2, _ = critic(b2, torch.zeros(4, 256))
            self.assertLess((v1 - v2).abs().max().item(), 1e-5)


if __name__ == "__main__":
    unittest.main()
