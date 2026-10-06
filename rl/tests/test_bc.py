import random
import types
import unittest

import common
import torch

from rl import bc, spec
from rl.model import Actor
from rl.pool import WorkerPool
from rl.worker import EnvHost

DEV = torch.device("cpu")


def scripted_episodes(n_ep=24, seed=0, cfg=None):
    cfg = cfg or common.smoke_cfg()
    host = EnvHost(cfg.env.cls, cfg.env.config, [], [], cfg.env.streams_per_env, host_id=0)
    r = host.collect_bc(n_ep, seed)
    return r["episodes"], r


class Dataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.eps, cls.raw = scripted_episodes()
        cls.shard = bc.build_shard(cls.eps)
        cls.ds = bc.BCDataset([cls.shard], 0.2, 0)

    def test_split_is_grouped_by_episode(self):
        ds = self.ds
        tr = {ds.trajs[i]["episode"] for i in ds.split["train"]}
        va = {ds.trajs[i]["episode"] for i in ds.split["val"]}
        self.assertTrue(tr and va)
        self.assertFalse(tr & va)
        self.assertAlmostEqual(len(va) / (len(va) + len(tr)), 0.2, delta=0.1)

    def test_stats_report_independent_cases_not_just_steps(self):
        st = self.ds.stats()
        self.assertEqual(st["decisions"], int(self.shard["own"].shape[0]))
        self.assertGreater(st["episodes"], 10)
        self.assertGreater(st["scenarios"], 5)
        self.assertGreater(st["fire_steps"], 0)
        self.assertLess(st["maneuver_switch_events"], st["decisions"])      # holds are not independent cases
        self.assertIn("kill", st["env_events"])

    def test_labels_are_canonical_and_legal(self):
        shard = self.shard
        st = self.ds.store
        idx = torch.arange(st.N).view(1, -1)
        b = st.gather(idx)
        from rl.model import Actor
        torch.manual_seed(0)
        actor = Actor().eval()
        with torch.no_grad():
            x, e = actor.encode(b)
            hs, _ = actor.unroll(x, torch.zeros(1, 256), b.first, b.valid)
            out = actor.heads_from_batch(b, hs, e, b.act, keep_dists=False)
        self.assertTrue(out.legal.all(), "every stored label must be legal under its own masks")
        self.assertEqual(self.raw["illegal"], {})

    def test_windows_pad_before_the_start_and_after_the_end(self):
        ds = self.ds
        i = ds.split["train"][0]
        t = ds.trajs[i]
        w = ds.window_index([(i, 0), (i, ((t["length"] - 1) // 16) * 16)], 4, 16)
        self.assertTrue((w[0, :4] == -1).all())
        self.assertEqual(int(w[0, 4]), t["offset"])
        last = w[1]
        self.assertTrue((last[last >= 0] < t["offset"] + t["length"]).all())

    def test_class_weights_and_per_head_weight_normalisation(self):
        act = torch.zeros(4, spec.N_HEADS, dtype=torch.long)
        act[:, spec.HEAD_INDEX["weapon"]] = torch.tensor([1, 0, 1, 0])
        act[:, spec.HEAD_INDEX["maneuver"]] = torch.tensor([0, 3, 0, 5])
        w = bc.head_weights(act, 5.0, 3.0)
        self.assertEqual(w[:, spec.HEAD_INDEX["weapon"]].tolist(), [5.0, 1.0, 5.0, 1.0])
        self.assertEqual(w[:, spec.HEAD_INDEX["maneuver"]].tolist(), [1.0, 3.0, 1.0, 3.0])
        self.assertEqual(w[:, spec.HEAD_INDEX["speed"]].tolist(), [1.0] * 4)
        # loss of a head = sum(w * nll) / sum(w) over its active steps; heads are summed
        logp = torch.zeros(4, spec.N_HEADS)
        logp[:, spec.HEAD_INDEX["weapon"]] = -torch.tensor([1.0, 2.0, 3.0, 4.0])
        active = torch.zeros(4, spec.N_HEADS, dtype=torch.bool)
        active[:, spec.HEAD_INDEX["weapon"]] = torch.tensor([True, True, True, False])
        out = types.SimpleNamespace(logp=logp)
        loss, num, den = bc.bc_loss(out, active, w)
        want = (5 * 1 + 1 * 2 + 5 * 3) / (5 + 1 + 5)
        self.assertAlmostEqual(float(loss), want, 5)
        self.assertEqual(float(den[spec.HEAD_INDEX["speed"]]), 0.0)


class Training(unittest.TestCase):
    def test_bc_learns_the_script_and_reports_per_head_metrics(self):
        eps, _ = scripted_episodes(n_ep=60, seed=3)
        ds = bc.BCDataset([bc.build_shard(eps)], 0.2, 0)
        cfg = common.smoke_cfg()
        cfg.bc.seg_len, cfg.bc.burn_in, cfg.bc.batch, cfg.bc.max_epochs = 16, 4, 8, 10
        cfg.bc.lr, cfg.bc.patience = 1e-3, 100
        torch.manual_seed(0)
        actor = Actor()
        rep = bc.train_bc(cfg, actor, ds, DEV, log=lambda *_: None)
        h = rep["history"]
        self.assertGreaterEqual(len(h), 3)
        self.assertLess(h[-1]["val"]["loss"], h[0]["val"]["loss"] * 0.8)
        v = rep["final_val"]
        for name in ("loss", "head_loss", "head_acc", "fire", "head_samples"):
            self.assertIn(name, v)
        self.assertIn("speed", v["head_loss"])
        self.assertIn("precision", v["fire"])
        self.assertIn("recall", v["fire"])
        self.assertEqual(v["illegal_labels"], {})
        self.assertGreater(v["head_acc"]["speed"], 0.95)          # speed / chaff / vertical are deterministic
        self.assertGreater(v["head_acc"]["chaff"], 0.95)          # functions of the visible threat flag
        self.assertGreater(v["head_acc"]["target"], 0.75)         # nearest radar track: a pointer-head task

    def test_early_stopping_after_patience_epochs_without_improvement(self):
        eps, _ = scripted_episodes(n_ep=20, seed=4)
        ds = bc.BCDataset([bc.build_shard(eps)], 0.2, 0)
        cfg = common.smoke_cfg()
        cfg.bc.lr = 0.0                       # nothing can improve after the first epoch
        cfg.bc.patience, cfg.bc.max_epochs, cfg.bc.batch, cfg.bc.seg_len, cfg.bc.burn_in = 2, 10, 16, 16, 4
        rep = bc.train_bc(cfg, Actor(), ds, DEV, log=lambda *_: None)
        self.assertEqual(len(rep["history"]), 3)                   # epoch 1 sets the best, 2 and 3 do not beat it
        self.assertEqual(rep["best_epoch"], 1)

    def test_training_resumes_from_its_state_file(self):
        import os, tempfile
        eps, _ = scripted_episodes(n_ep=20, seed=6)
        ds = bc.BCDataset([bc.build_shard(eps)], 0.2, 0)
        cfg = common.smoke_cfg()
        cfg.bc.seg_len, cfg.bc.burn_in, cfg.bc.batch, cfg.bc.patience = 16, 4, 8, 100
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "bc_state.pt")
            cfg.bc.max_epochs = 2
            torch.manual_seed(0)
            r1 = bc.train_bc(cfg, Actor(), ds, DEV, log=lambda *_: None, state_path=path)
            self.assertEqual(len(r1["history"]), 2)
            cfg.bc.max_epochs = 4
            msgs = []
            r2 = bc.train_bc(cfg, Actor(), ds, DEV, log=msgs.append, state_path=path)
            self.assertEqual(len(r2["history"]), 4)
            self.assertEqual(r2["history"][0]["train_loss"], r1["history"][0]["train_loss"])
            self.assertTrue(any("resumed" in m for m in msgs))

    def test_shards_round_trip_and_resume_counts(self):
        import os, tempfile
        cfg = common.smoke_cfg()
        cfg.bc.collect_decisions = 600
        cfg.bc.shard_decisions = 300
        with tempfile.TemporaryDirectory() as d:
            pool = WorkerPool(cfg.workers, cfg.env).start()
            pool.init_envs(0, 1)
            meta = bc.collect_to_shards(pool, cfg, d, 11, log=lambda *_: None)
            self.assertGreaterEqual(meta["decisions"], 600)
            n_files = len([f for f in os.listdir(d) if f.startswith("shard_")])
            self.assertEqual(n_files, meta["shards"])
            meta2 = bc.collect_to_shards(pool, cfg, d, 12, log=lambda *_: None)      # nothing more to collect
            self.assertEqual(meta2["shards"], meta["shards"])
            ds = bc.BCDataset(bc.load_shards(d), 0.2, 0)
            self.assertEqual(ds.stats()["decisions"], meta["decisions"])
            pool.close()


class Canonicalisation(unittest.TestCase):
    def test_forced_and_illegal_heads(self):
        rng = random.Random(0)
        o = common.random_obs(rng, 4)
        n = 4
        masks = o["masks"]
        act = {h: 0 for h in spec.HEAD_NAMES}
        act["view_mode"] = 1
        act["maneuver_ref"] = 2           # whatever the row says, the canonical form must be legal
        canon, forced, illegal = spec.canonicalize_action(n, masks, act)
        self.assertEqual(spec.illegal_heads(n, masks, canon), [])
        for h in forced:
            self.assertEqual(sum(spec.effective_mask(h, n, masks, {k: canon[k] for k in spec.SAMPLE_ORDER[:spec.SAMPLE_ORDER.index(h)]})), 1)


if __name__ == "__main__":
    unittest.main()
