import os
import subprocess
import sys
import unittest

import common  # noqa: F401

from rl import config as C
from rl import runtime as R


class Config(unittest.TestCase):
    def test_presets_validate(self):
        for name in ("smoke", "workstation", "gpu_cluster"):
            C.preset(name).validate()

    def test_workstation_fits_in_48_cores_and_uses_pypy_workers(self):
        c = C.preset("workstation")
        self.assertEqual(c.resources.max_cores, 48)
        self.assertEqual(c.resources.device, "cpu")
        self.assertLessEqual(c.workers.n_local + c.resources.threads_infer, 48)
        self.assertLessEqual(c.resources.threads_update, 48)
        self.assertTrue(c.workers.python[0].endswith("lb_bundle/pypy/bin/pypy3"))
        self.assertTrue(c.run.run_dir.startswith("~"))

    def test_core_budget_is_enforced(self):
        c = C.preset("workstation")
        c.workers.n_local = 45
        with self.assertRaisesRegex(ValueError, "core budget"):
            c.validate()
        c = C.preset("workstation")
        c.resources.threads_update = 64
        with self.assertRaises(ValueError):
            c.validate()
        c = C.preset("workstation")
        c.resources.max_cores = 0           # uncapped: no check
        c.workers.n_local = 90
        c.validate()

    def test_overrides_and_json_round_trip(self):
        c = C.preset("smoke")
        C.apply_overrides(c, ["ppo.clip=0.1", "workers.n_local=3", "env.config={'n_agents': 5}", "run.run_dir=/x/y",
                              "workers.python=['/a/pypy3']"])
        self.assertEqual(c.ppo.clip, 0.1)
        self.assertEqual(c.workers.n_local, 3)
        self.assertEqual(c.env.config, {"n_agents": 5})
        self.assertEqual(c.run.run_dir, "/x/y")
        c2 = C.Config.from_dict(c.to_dict())
        self.assertEqual(c2.to_dict(), c.to_dict())
        with self.assertRaises(KeyError):
            C.apply_overrides(c, ["ppo.nonexistent=1"])
        with self.assertRaises(ValueError):
            C.preset("smoke").rollout.__class__  # noqa
            c3 = C.preset("smoke"); c3.rollout.steps = 25; c3.validate()

    def test_spec_defaults(self):
        p = C.PPOCfg()
        self.assertEqual((p.clip, p.lr_actor, p.lr_critic, p.adam_eps, p.max_grad_norm, p.epochs, p.target_kl),
                         (0.15, 1e-4, 3e-4, 1e-5, 0.5, 3, 0.02))
        r = C.RolloutCfg()
        self.assertEqual((r.n_streams, r.steps, r.seg_len, r.burn_in), (64, 320, 80, 16))
        self.assertEqual(p.minibatch_segments, 16)
        self.assertEqual(r.n_streams * (r.steps // r.seg_len) // p.minibatch_segments, 16)    # minibatches per epoch
        b = C.BCCfg()
        self.assertEqual((b.seg_len, b.burn_in, b.batch, b.lr, b.max_epochs, b.patience, b.w_fire, b.w_switch),
                         (80, 16, 32, 3e-4, 10, 3, 5.0, 3.0))


class CoreCap(unittest.TestCase):
    def test_no_cap_requested(self):
        self.assertIsNone(R.apply_core_cap(0))

    def test_cap_picks_the_first_allowed_cores_and_honours_the_offset(self):
        from unittest import mock
        calls = []
        with mock.patch.object(os, "sched_getaffinity", create=True, return_value=set(range(96))), \
                mock.patch.object(os, "sched_setaffinity", create=True, side_effect=lambda pid, cores: calls.append(cores)):
            got = R.apply_core_cap(48)
            self.assertEqual(got, list(range(48)))
            self.assertEqual(calls[-1], set(range(48)))
            got = R.apply_core_cap(48, offset=8)
            self.assertEqual(got, list(range(8, 56)))
        with mock.patch.object(os, "sched_getaffinity", create=True, return_value={4, 5, 6}), \
                mock.patch.object(os, "sched_setaffinity", create=True, side_effect=lambda pid, cores: calls.append(cores)):
            self.assertEqual(R.apply_core_cap(48), [4, 5, 6])          # never more than what is allowed

    def test_early_setup_applies_the_cap_from_the_config(self):
        from unittest import mock
        cfg = C.preset("workstation")
        with mock.patch.object(os, "sched_getaffinity", create=True, return_value=set(range(96))), \
                mock.patch.object(os, "sched_setaffinity", create=True) as setaff:
            cores = R.early_setup(cfg)
        self.assertEqual(len(cores), 48)
        setaff.assert_called_once()

    @unittest.skipUnless(hasattr(os, "sched_setaffinity"), "Linux only")
    def test_cap_restricts_this_process_and_its_children(self):
        code = ("import os,sys,subprocess; sys.path.insert(0, %r)\n"
                "from rl import runtime as R\n"
                "n = min(2, len(os.sched_getaffinity(0)))\n"
                "cores = R.apply_core_cap(n)\n"
                "child = subprocess.run([sys.executable, '-c', 'import os; print(len(os.sched_getaffinity(0)))'], capture_output=True, text=True)\n"
                "print(len(cores), len(os.sched_getaffinity(0)), child.stdout.strip(), n)\n" % common.ROOT)
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).stdout.split()
        self.assertEqual(out[0], out[3])
        self.assertEqual(out[1], out[3])
        self.assertEqual(out[2], out[3])


if __name__ == "__main__":
    unittest.main()
