"""Processes: worker transparency (in-process vs subprocess vs PyPy), the CLI end to end, resume,
wall-time stop, signals."""
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest

import common
import torch

from rl import runtime as R
from rl.model import Actor
from rl.pool import WorkerPool, WorkerError
from rl.rollout import Sampler

PY = sys.executable
PYS = [common.PYPY, PY] if common.PYPY else [PY, PY]


def run_cli(args, timeout=600):
    return subprocess.run([PY, "-m", "rl.train"] + args, cwd=common.ROOT, capture_output=True, text=True, timeout=timeout)


def smoke_args(run_dir, rounds=2, extra=()):
    a = ["run", "--config", "smoke", "--run-dir", run_dir, "--rounds", str(rounds),
         "--set", "workers.python=%r" % PYS]
    for e in extra:
        a += ["--set", e]
    return a


def read_metrics(run_dir):
    with open(os.path.join(run_dir, "metrics.jsonl")) as f:
        return [json.loads(l) for l in f]


class WorkerTransparency(unittest.TestCase):
    def collect(self, inproc, pythons=None):
        torch.set_num_threads(1)
        cfg = common.smoke_cfg(inproc=inproc)
        cfg.rollout.burn_in = 4
        if pythons:
            cfg.workers.python = pythons
            cfg.workers.n_local = len(pythons)
        torch.manual_seed(3)
        actor = Actor()
        pool = WorkerPool(cfg.workers, cfg.env).start()
        try:
            sm = Sampler(cfg, pool, torch.device("cpu"), 5)
            sm.start(77)
            bufs = [sm.collect(actor)[0] for _ in range(2)]
        finally:
            pool.close()
        return bufs

    def test_subprocess_and_pypy_workers_reproduce_the_in_process_rollout_exactly(self):
        ref = self.collect(True)
        got = self.collect(False, PYS)
        for b1, b2 in zip(ref, got):
            for name in ("own", "ent_n", "act", "valid", "first", "mask"):
                self.assertTrue(torch.equal(getattr(b1.store, name), getattr(b2.store, name)), name)
            self.assertTrue(torch.equal(b1.store.ent_rows, b2.store.ent_rows))
            self.assertTrue(torch.equal(b1.reward, b2.reward))
            self.assertTrue(torch.equal(b1.done, b2.done))
            self.assertTrue(torch.allclose(b1.logp, b2.logp, atol=1e-6))

    def test_a_crashing_worker_is_reported_with_its_log(self):
        cfg = common.smoke_cfg(inproc=False)
        cfg.workers.n_local = 1
        cfg.env.cls = "no.such.module:Env"
        with tempfile.TemporaryDirectory() as d:
            pool = WorkerPool(cfg.workers, cfg.env, d).start()
            try:
                with self.assertRaisesRegex(WorkerError, "no.such.module|ModuleNotFoundError"):
                    pool.init_envs(1, 0)
            finally:
                pool.close()

    def test_worker_that_cannot_start_is_reported(self):
        cfg = common.smoke_cfg(inproc=False)
        cfg.workers.n_local = 1
        cfg.workers.python = ["/nonexistent/python"]
        cfg.workers.startup_timeout_s = 20
        with self.assertRaises((WorkerError, FileNotFoundError, OSError)):
            WorkerPool(cfg.workers, cfg.env).start()


class RemoteWorkers(unittest.TestCase):
    def test_a_worker_started_by_hand_joins_over_tcp(self):
        """Simulates another node: `RL_AUTHKEY=... python -m rl.worker --connect HOST:PORT`."""
        import threading
        cfg = common.smoke_cfg(inproc=False)
        cfg.workers.n_local = 0
        cfg.workers.n_remote = 1
        cfg.workers.startup_timeout_s = 60
        with tempfile.TemporaryDirectory() as d:
            pool = WorkerPool(cfg.workers, cfg.env, d)
            th = threading.Thread(target=pool.start, daemon=True)
            th.start()
            addr_f, key_f = os.path.join(d, "worker_address.txt"), os.path.join(d, "authkey")
            t0 = time.time()
            while time.time() - t0 < 30 and not (os.path.exists(addr_f) and os.path.exists(key_f)):
                time.sleep(0.1)
            time.sleep(0.2)
            self.assertEqual(oct(os.stat(key_f).st_mode & 0o777), oct(0o600))
            with open(addr_f) as f:
                addr = f.read().strip()
            with open(key_f) as f:
                key = f.read().strip()
            env = dict(os.environ, RL_AUTHKEY=key, PYTHONPATH=common.ROOT)
            proc = subprocess.Popen([common.PYPY or PY, "-m", "rl.worker", "--connect", addr, "--name", "remote0"],
                                    cwd=common.ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            try:
                th.join(timeout=60)
                self.assertFalse(th.is_alive())
                self.assertEqual(pool.describe()[0]["name"], "remote0")
                resets = pool.init_envs(2, 1)
                self.assertEqual(set(resets), {0, 1})
                res = pool.collect_bc(300, 2, 5)
                self.assertGreaterEqual(res[0], 300)
            finally:
                pool.close()
                proc.wait(timeout=30)

    def test_wrong_authkey_is_rejected(self):
        from multiprocessing import AuthenticationError
        from multiprocessing.connection import Client
        cfg = common.smoke_cfg(inproc=False)
        cfg.workers.n_local = 0
        cfg.workers.n_remote = 1
        cfg.workers.authkey = "right"
        cfg.workers.startup_timeout_s = 3
        import threading
        pool = WorkerPool(cfg.workers, cfg.env)
        errs = []
        def go():
            try:
                pool.start()
            except WorkerError as e:
                errs.append(e)
        th = threading.Thread(target=go, daemon=True)
        th.start()
        t0 = time.time()
        while time.time() - t0 < 10 and pool.listener is None:
            time.sleep(0.05)
        with self.assertRaises(AuthenticationError):
            Client(pool.listener.address, authkey=b"wrong")
        th.join(timeout=30)
        self.assertTrue(errs)


class CLI(unittest.TestCase):
    def test_run_resume_and_artifacts(self):
        with tempfile.TemporaryDirectory() as d:
            r = run_cli(smoke_args(d, rounds=2))
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            for f in ("config.json", "bc_actor.pt", "bc_report.json", "metrics.jsonl", "bc_data/meta.json", "ppo/ckpt_000002.pt"):
                self.assertTrue(os.path.exists(os.path.join(d, f)), f)
            with open(os.path.join(d, "bc_report.json")) as f:
                rep = json.load(f)
            self.assertIn("head_loss", rep["final_val"])
            self.assertIn("precision", rep["final_val"]["fire"])
            self.assertIn("flying", rep)
            self.assertGreater(rep["dataset"]["decisions"], 1000)
            m = read_metrics(d)
            self.assertEqual([x["round"] for x in m], [1, 2])
            need = ("kl_target", "kl_ref", "entropy", "entropy_head", "clip_frac", "value_loss", "explained_variance",
                    "reward_per_decision", "events", "time", "valid_fraction", "kl_ref_head", "ent_coef", "kl_beta",
                    "per_aircraft", "episodes_finished", "decisions_total")
            for k in need:
                self.assertIn(k, m[0], k)
            for k in ("sample", "inference", "update", "round"):
                self.assertIn(k, m[0]["time"])
            self.assertIn("maneuver_ref", m[0]["entropy_head"])
            # the process used both interpreters (when a PyPy is available)
            if common.PYPY:
                self.assertIn("pypy", r.stdout)
            # resume: ask for one more round, the rounds continue numbering and nothing is redone
            r2 = run_cli(smoke_args(d, rounds=3))
            self.assertEqual(r2.returncode, 0, r2.stdout + r2.stderr)
            self.assertIn("resumed from", r2.stdout)
            self.assertNotIn("bc data: shard", r2.stdout)
            m2 = read_metrics(d)
            self.assertEqual([x["round"] for x in m2], [1, 2, 3])
            self.assertEqual(m2[2]["decisions_total"] > m2[1]["decisions_total"], True)
            ck = torch.load(os.path.join(d, "ppo", "ckpt_000003.pt"), map_location="cpu", weights_only=False)
            self.assertEqual(ck["trainer"]["round"], 3)
            self.assertEqual(ck["starts"], 1)

    def test_wall_limit_stops_with_a_checkpoint_and_the_rerun_finishes(self):
        with tempfile.TemporaryDirectory() as d:
            r = run_cli(smoke_args(d, rounds=2, extra=["run.max_wall_s=60", "run.wall_margin_s=3600"]))
            self.assertEqual(r.returncode, R.EXIT_RESUME_ME, r.stdout + r.stderr)
            self.assertIn("checkpoint", r.stdout)
            self.assertTrue(os.path.exists(os.path.join(d, "ppo", "ckpt_000000.pt")))
            r2 = run_cli(smoke_args(d, rounds=2))
            self.assertEqual(r2.returncode, 0, r2.stdout + r2.stderr)
            self.assertEqual([x["round"] for x in read_metrics(d)], [1, 2])

    def test_sigterm_checkpoints_and_exits_resumable(self):
        with tempfile.TemporaryDirectory() as d:
            p = subprocess.Popen([PY, "-m", "rl.train"] + smoke_args(d, rounds=500), cwd=common.ROOT,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            t0 = time.time()
            mpath = os.path.join(d, "metrics.jsonl")
            while time.time() - t0 < 120 and not (os.path.exists(mpath) and os.path.getsize(mpath) > 0):
                time.sleep(0.2)
            self.assertTrue(os.path.exists(mpath), "no round finished in time")
            p.send_signal(signal.SIGTERM)
            out, _ = p.communicate(timeout=120)
            self.assertEqual(p.returncode, R.EXIT_RESUME_ME, out)
            self.assertIn("stop requested", out)
            self.assertTrue(any(f.startswith("ckpt_") for f in os.listdir(os.path.join(d, "ppo"))))
            n = len(read_metrics(d))
            r = run_cli(smoke_args(d, rounds=n + 1))
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertEqual([x["round"] for x in read_metrics(d)], list(range(1, n + 2)))


if __name__ == "__main__":
    unittest.main()
