"""Learner-side handle to the environment workers.

* WorkerPool(inproc=True)  : envs hosted in this process (tests, benchmarks, debugging)
* WorkerPool(n_local=k)    : k subprocesses started with `<python> -m rl.worker` (each may use a
                             different interpreter, e.g. PyPy); they connect back over
                             multiprocessing.connection (TCP on `listen`)
* WorkerPool(n_remote=m)   : additionally wait for m workers started elsewhere with
                             `python -m rl.worker --connect HOST:PORT` (needs RL_AUTHKEY)

Synchronous protocol: send to every busy worker, then receive from every busy worker. The
number of envs is independent of the number of workers (round-robin assignment).
"""
from __future__ import annotations

import os
import pickle
import secrets
import subprocess
import sys
import threading
import time
from multiprocessing.connection import Listener
from typing import Dict, List

from rl import wire
from rl.worker import dispatch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class WorkerError(RuntimeError):
    pass


class _LocalHandle:
    info = {"name": "inproc", "impl": "cpython-inproc"}

    def __init__(self):
        self.state = {}
        self._reply = None

    def send(self, op, payload):
        self._reply = dispatch(self.state, op, payload)

    def recv(self):
        r, self._reply = self._reply, None
        return r

    def close(self):
        pass


class _RemoteHandle:
    def __init__(self, conn, info, proc=None):
        self.conn, self.info, self.proc = conn, info, proc

    def send(self, op, payload):
        try:
            self.conn.send_bytes(pickle.dumps((op, payload), wire.PICKLE_PROTOCOL))
        except (OSError, ConnectionError) as e:
            raise WorkerError("worker %s: send failed (%s)" % (self.info.get("name"), e))

    def recv(self):
        try:
            status, val = pickle.loads(self.conn.recv_bytes())
        except (EOFError, OSError, ConnectionError) as e:
            raise WorkerError("worker %s died (%s)%s" % (self.info.get("name"), type(e).__name__, self._tail()))
        if status == "err":
            raise WorkerError("worker %s raised:\n%s" % (self.info.get("name"), val))
        return val

    def _tail(self):
        path = self.info.get("log")
        if path and os.path.exists(path):
            with open(path, errors="replace") as f:
                return "\n--- worker log tail ---\n" + "".join(f.readlines()[-15:])
        return ""

    def close(self):
        try:
            self.conn.send_bytes(pickle.dumps(("close", None), wire.PICKLE_PROTOCOL))
        except Exception:
            pass
        try:
            self.conn.close()
        except Exception:
            pass
        if self.proc is not None:
            try:
                self.proc.wait(timeout=10)
            except Exception:
                self.proc.kill()


class WorkerPool:
    def __init__(self, wcfg, env_cfg, run_dir=None):
        self.w = wcfg
        self.env = env_cfg
        self.run_dir = run_dir
        self.handles: List = []
        self.listener = None
        self.authkey = ""
        self.env_handle: Dict[int, int] = {}
        self.n_envs = 0
        self.last_worker_time = 0.0

    # ------------------------------------------------------------------ start / stop
    def start(self):
        try:
            return self._start()
        except BaseException:
            self.close()
            raise

    def _start(self):
        w = self.w
        if w.inproc:
            self.handles = [_LocalHandle()]
            return self
        key = w.authkey or os.environ.get("RL_AUTHKEY", "") or secrets.token_hex(16)
        self.authkey = key
        host, _, port = w.listen.rpartition(":")
        # Listener defaults to backlog=1: on Linux a full accept queue drops SYNs, so dozens of workers
        # connecting at once stall in TCP retransmission back-off (seen: 12 of 32 connected in 180 s).
        backlog = max(128, w.n_local + w.n_remote)
        self.listener = Listener((host or "127.0.0.1", int(port or 0)), authkey=key.encode(), backlog=backlog)
        addr = self.listener.address
        if self.run_dir:
            os.makedirs(self.run_dir, exist_ok=True)
            with open(os.path.join(self.run_dir, "worker_address.txt"), "w") as f:
                f.write("%s:%d\n" % (addr[0], addr[1]))
            if w.n_remote > 0:               # remote nodes need the key: export RL_AUTHKEY=$(cat <run_dir>/authkey)
                kp = os.path.join(self.run_dir, "authkey")
                fd = os.open(kp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                with os.fdopen(fd, "w") as f:
                    f.write(key)
        procs = []
        pys = [os.path.expanduser(p) for p in (w.python or [sys.executable])]
        logs = {}
        for i in range(w.n_local):
            py = pys[i % len(pys)]
            env = dict(os.environ)
            env["RL_AUTHKEY"] = key
            env["PYTHONPATH"] = REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
            env["PYTHONUNBUFFERED"] = "1"
            log_path = None
            out = subprocess.DEVNULL
            if self.run_dir:
                os.makedirs(os.path.join(self.run_dir, "workers"), exist_ok=True)
                log_path = os.path.join(self.run_dir, "workers", "w%d.log" % i)
                out = open(log_path, "ab")
            connect_host = addr[0] if addr[0] not in ("0.0.0.0", "::") else "127.0.0.1"
            cmd = [py, "-m", "rl.worker", "--connect", "%s:%d" % (connect_host, addr[1]), "--name", "w%d" % i]
            p = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env, stdin=subprocess.DEVNULL, stdout=out,
                                 stderr=subprocess.STDOUT)
            if out is not subprocess.DEVNULL:
                out.close()                      # the child keeps its own descriptor
            procs.append((p, log_path))
        expected = w.n_local + w.n_remote
        conns = []
        err = []

        def acceptor():
            try:
                for _ in range(expected):
                    conns.append(self.listener.accept())
            except Exception as e:      # listener closed or auth failure
                err.append(e)

        th = threading.Thread(target=acceptor, daemon=True)
        th.start()
        t0 = time.time()
        while len(conns) < expected:
            if err:
                self.close()
                raise WorkerError("accepting workers failed: %r" % (err[0],))
            for p, lp in procs:
                if p.poll() is not None and len(conns) < expected:
                    # a local worker exited before connecting
                    tail = ""
                    if lp and os.path.exists(lp):
                        with open(lp, errors="replace") as f:
                            tail = "".join(f.readlines()[-20:])
                    if p.returncode != 0:
                        self.close()
                        raise WorkerError("local worker exited with code %s before connecting:\n%s" % (p.returncode, tail))
            if time.time() - t0 > w.startup_timeout_s:
                self.close()
                raise WorkerError("timed out waiting for %d workers (%d connected) on %s:%d"
                                  % (expected, len(conns), addr[0], addr[1]))
            time.sleep(0.05)
        th.join(timeout=5)
        by_pid = {p.pid: (p, lp) for p, lp in procs}
        for c in conns:
            status, info = pickle.loads(c.recv_bytes())
            p, lp = by_pid.get(info.get("pid"), (None, None))
            if lp:
                info["log"] = lp
            self.handles.append(_RemoteHandle(c, info, p))
        self.handles.sort(key=lambda h: h.info.get("name", ""))
        return self

    def close(self):
        for h in self.handles:
            h.close()
        self.handles = []
        if self.listener is not None:
            try:
                self.listener.close()
            except Exception:
                pass
            self.listener = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.close()

    def describe(self):
        return [dict(h.info) for h in self.handles]

    # ------------------------------------------------------------------ envs
    def init_envs(self, n_envs, seed_base):
        """Create and reset n_envs environments spread over the workers. Returns {env_id: reset}."""
        self.n_envs = n_envs
        e = self.env
        per = [[] for _ in self.handles]
        self.env_handle = {}
        for j in range(n_envs):
            hi = j % len(self.handles)
            per[hi].append(j)
            self.env_handle[j] = hi
        for hi, h in enumerate(self.handles):
            ids = per[hi]
            h.send("init", dict(env_cls=e.cls, env_config=e.config, env_ids=ids,
                                seeds=[seed_base + 7919 * j for j in ids], streams_per_env=e.streams_per_env,
                                validate_steps=e.validate_steps, host_id=hi))
        out = {}
        for h in self.handles:
            out.update(h.recv())
        return out

    def step(self, actions_by_env):
        """{env_id: {aid: action tuple}} -> {env_id: StepResult}."""
        per = {}
        for j, a in actions_by_env.items():
            per.setdefault(self.env_handle[j], {})[j] = a
        for hi, acts in per.items():
            self.handles[hi].send("step", acts)
        out = {}
        tmax = 0.0
        for hi in per:
            res, t = self.handles[hi].recv()
            out.update(res)
            tmax = max(tmax, t)
        self.last_worker_time = tmax
        return out

    # ------------------------------------------------------------------ scripted data
    def collect_bc(self, target_decisions, episodes_per_request, seed, on_episodes=None, check_legality=True):
        """Ask every worker for scripted episodes until target_decisions decisions were collected.

        Call init_envs(0, ...) first (it creates the host on every worker, with or without envs).
        """
        from multiprocessing.connection import wait
        total = 0
        illegal, forced = {}, {}
        busy = {}
        req = 0

        def submit(i):
            nonlocal req
            req += 1
            self.handles[i].send("collect_bc", dict(n_episodes=episodes_per_request,
                                                    seed=seed + 104729 * (i + 1) + req * 31,
                                                    check_legality=check_legality))
            busy[i] = True

        n = len(self.handles)
        inproc = isinstance(self.handles[0], _LocalHandle)
        for i in range(n):
            submit(i)
            if inproc:
                break
        while busy:
            if inproc:
                ready = [0]
            else:
                conns = {self.handles[i].conn: i for i in busy}
                ready = [conns[c] for c in wait(list(conns), timeout=600)]
                if not ready:
                    raise WorkerError("collect_bc: no worker answered for 600 s")
            for i in ready:
                r = self.handles[i].recv()
                busy.pop(i, None)
                for k, v in r["illegal"].items():
                    illegal[k] = illegal.get(k, 0) + v
                for k, v in r["forced"].items():
                    forced[k] = forced.get(k, 0) + v
                n_dec = sum(len(a["steps"]) for ep in r["episodes"] for a in ep["agents"].values())
                total += n_dec
                if on_episodes is not None:
                    on_episodes(r["episodes"])
                if total < target_decisions:
                    submit(i)
        return total, illegal, forced
