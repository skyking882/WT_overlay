"""Tests for scripts/rl_dashboard.py: metrics parsing, replay conversion, train.log stages, remote sync, HTTP endpoints."""
from __future__ import annotations

import gzip
import http.client
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import rl_dashboard as dash  # noqa: E402

PLANE_COLUMNS = ["id", "x", "y", "z", "vx", "vy", "vz", "heading_deg", "missiles", "chaff", "phase"]
MISSILE_COLUMNS = ["uid", "owner", "target", "x", "y", "z", "vx", "vy", "vz", "heading_deg", "age_s", "seeker", "datalink"]


def jl(*objs):
    return "".join(json.dumps(o, separators=(",", ":")) + "\n" for o in objs)


def round_record(n, ts, decisions, **extra):
    rec = dict(round=n, decisions_total=decisions, decisions_in_round=100, reward_per_decision=0.01 * n,
               episode_return_mean=None if n == 2 else 1.5, entropy=0.5, kl_target=0.01, kl_ref=0.02, clip_frac=0.1,
               value_loss=0.3, explained_variance=0.5, valid_fraction=0.8, events=dict(launch=3, kill=1, death=1, assist=0),
               time=dict(sample=1.0, inference=0.4, update=0.5, postpass=0.1, round=2.0), timestamp=ts,
               per_aircraft=dict(f16=dict(decisions=60, reward_per_decision=0.1)), head_active_frac=dict(a=1.0),
               entropy_head=dict(maneuver=0.3), kl_ref_head=dict(maneuver=0.01))
    rec.update(extra)
    return rec


def write_replay(path, frames=41, die_at=5.0, with_end=True, junk=False):
    """2 aircraft (plane 1 dies at 5.0 s, absent from frames from then on), one missile in the air 2.5-4.25 s."""
    header = dict(type="header", version=1, seed=7, map_half_m=64000.0, tick_s=0.0208, frame_dt_s=0.25, time_limit_s=900.0,
                  planes=[dict(id=0, team=0, aircraft="su_30sm2", name="su_30sm2#0", archetype="left", skill="normal",
                               missile="su_r_77_1", missiles=10, chaff=74, script=dict(archetype="left")),
                          dict(id=1, team=1, aircraft="f_15c", name="f_15c#1", archetype="middle", skill="expert",
                               missile="us_aim_120c_5", missiles=8, chaff=60, script=None)],
                  plane_columns=PLANE_COLUMNS, missile_columns=MISSILE_COLUMNS)
    lines = [header]
    for i in range(frames):
        t = i * 0.25
        planes = [[0, 1000 + 8 * t, 2000, 8000, 250, 0, 0, 90.0, 10, 74, "climb"]]
        if t < die_at:
            planes.append([1, -1000, 3000, 7000, -200, 0, 0, 270.0, 8, 60, "evade"])
        missiles = [[0, 0, 1, 1000 + 20 * t, 2000, 8000, 700, 0, 0, 90.0, t - 2.5, 1, 0]] if 2.5 <= t <= 4.25 else []
        lines.append(dict(type="frame", t=t, planes=planes, missiles=missiles))
        if abs(t - 2.25) < 1e-9:
            lines.append(dict(type="event", t=2.25, kind="launch", uid=0, shooter=0, target=1, missile="su_r_77_1", mode="tws",
                              range_m=38000, left=9, flame_seen=True))
        if abs(t - 4.5) < 1e-9:
            lines.append(dict(type="event", t=4.5, kind="missile_end", uid=0, shooter=0, target=1, result="fuse", miss_m=3.0,
                              flight_s=2.0))
        if abs(t - 4.75) < 1e-9:
            lines.append(dict(type="event", t=4.75, kind="kill", victim=1, cause="missile", killer=0, time_s=4.75, uid=0))
            lines.append(dict(type="event", t=4.75, kind="death", plane=1, cause="missile", killer=0))
            lines.append(dict(type="event", t=4.75, kind="chaff", plane=0, n=2, left=72))
    if with_end:
        lines.append(dict(type="event", t=(frames - 1) * 0.25, kind="end", reason="annihilation"))
        lines.append(dict(type="end", reason="annihilation", t=(frames - 1) * 0.25,
                          result=dict(teams_alive=[1, 0], kills=[dict(victim=1, killer=0)], deaths=[], launches=1, planes=[])))
    text = jl(*lines)
    if junk:
        parts = text.splitlines(keepends=True)
        parts.insert(5, "{this is not json\n")
        text = "".join(parts)
    Path(path).write_text(text)


class MetricsParsing(unittest.TestCase):
    def test_partial_last_line_is_dropped_and_flagged(self):
        data = (jl(round_record(1, 100, 100), round_record(2, 110, 200)) + '{"round": 3, "decisions_tot').encode()
        records, bad, partial = dash.parse_jsonl_bytes(data)
        self.assertEqual([r["round"] for r in records], [1, 2])
        self.assertEqual(bad, 1)
        self.assertTrue(partial)

    def test_corrupt_middle_line_is_skipped_but_not_called_partial(self):
        data = (jl(round_record(1, 100, 100)) + "garbage\n" + "[1, 2]\n" + jl(round_record(3, 130, 300))).encode()
        records, bad, partial = dash.parse_jsonl_bytes(data)
        self.assertEqual([r["round"] for r in records], [1, 3])
        self.assertEqual(bad, 2)
        self.assertFalse(partial)

    def test_load_metrics_drops_rewound_rounds_and_records_without_round(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "metrics.jsonl"
            p.write_text(jl(round_record(1, 1, 10), round_record(2, 2, 20), round_record(3, 3, 30), dict(note="x"),
                            round_record(2, 4, 21), round_record(3, 5, 31)))
            m = dash.load_metrics(p)
            self.assertEqual([r["round"] for r in m["records"]], [1, 2, 3])
            self.assertEqual([r["decisions_total"] for r in m["records"]], [10, 21, 31])      # the later records win
            self.assertEqual(m["bad"], 1)

    def test_load_metrics_missing_and_empty(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(dash.load_metrics(Path(d) / "nope.jsonl")["records"], [])
            (Path(d) / "metrics.jsonl").write_text("")
            self.assertEqual(dash.load_metrics(Path(d) / "metrics.jsonl")["records"], [])

    def test_nan_becomes_null_in_the_json_we_serve(self):
        raw = '{"round": 1, "kl_ref": NaN, "explained_variance": Infinity}\n'.encode()
        records, bad, _ = dash.parse_jsonl_bytes(raw)
        self.assertEqual(bad, 0)
        text = dash.dumps(records[0]).decode()
        self.assertNotIn("NaN", text)
        self.assertNotIn("Infinity", text)
        self.assertEqual(json.loads(text)["kl_ref"], None)

    def test_slim_records_keeps_per_aircraft_only_on_the_last_round(self):
        recs = [round_record(i, i, i * 100) for i in range(1, 4)]
        slim = dash.slim_records(recs)
        self.assertNotIn("per_aircraft", slim[0])
        self.assertIn("per_aircraft", slim[-1])
        self.assertTrue(all("head_active_frac" not in r for r in slim))
        self.assertIn("head_active_frac", recs[0])      # the originals are untouched


class RunStatus(unittest.TestCase):
    def records(self, n=6, dt=60.0, per=1000):
        return [round_record(i + 1, 1000.0 + i * dt, (i + 1) * per, decisions_in_round=per) for i in range(n)]

    def test_rate_eta_and_progress(self):
        st = dash.run_status(self.records(), window=3, target=100000, now=1000.0 + 5 * 60.0 + 10)
        self.assertAlmostEqual(st["rate_per_hour"], 60000.0)         # 1000 decisions per 60 s
        self.assertEqual(st["rate_rounds"], 3)
        self.assertAlmostEqual(st["eta_s"], (100000 - 6000) / 60000.0 * 3600.0)
        self.assertAlmostEqual(st["progress"], 0.06)
        self.assertEqual(st["state"], "live")
        self.assertAlmostEqual(st["age_s"], 10.0)

    def test_stale_when_old(self):
        st = dash.run_status(self.records(), now=1000.0 + 5 * 60.0 + 3600)
        self.assertEqual(st["state"], "stale")

    def test_target_reached_gives_zero_eta(self):
        st = dash.run_status(self.records(), target=5000, now=2000.0)
        self.assertEqual(st["eta_s"], 0.0)
        self.assertEqual(st["progress"], 1.0)

    def test_gap_from_a_resume_is_left_out(self):
        recs = self.records(6, 60.0)
        for r in recs[3:]:
            r["timestamp"] += 7200.0                                  # run was stopped for two hours, then resumed
        rate, used = dash.rate_per_hour(recs, 20)
        self.assertEqual(used, 4)                                     # 5 intervals, 1 of them is the gap
        self.assertAlmostEqual(rate, 60000.0)

    def test_without_timestamps_the_time_split_is_used(self):
        recs = self.records()
        for r in recs:
            del r["timestamp"]
        rate, used = dash.rate_per_hour(recs, 4)
        self.assertEqual(used, 4)
        self.assertAlmostEqual(rate, 1000 / 2.0 * 3600.0)             # time.round = 2 s

    def test_no_records(self):
        st = dash.run_status([])
        self.assertEqual(st["state"], "waiting")
        self.assertIsNone(st["rate_per_hour"])

    def test_suggested_target_from_config(self):
        st = dash.run_status(self.records(), cfg=dict(run=dict(rounds=250)))
        self.assertEqual(st["suggested_target"], 250000)
        self.assertEqual(st["rounds_total"], 250)


class ReplayFiles(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "m.jsonl"
        write_replay(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_header_and_summary(self):
        h = dash.read_replay_header(self.path)
        self.assertEqual(h["seed"], 7)
        self.assertEqual(len(h["planes"]), 2)
        s = dash.replay_summary(self.path)
        self.assertEqual(s["teams"], [1, 1])
        self.assertTrue(s["complete"])
        self.assertEqual(s["reason"], "annihilation")
        self.assertAlmostEqual(s["duration_s"], 10.0)
        self.assertEqual(s["kills"], 1)

    def test_summary_of_an_unfinished_replay(self):
        p = Path(self.tmp.name) / "partial.jsonl"
        write_replay(p, frames=21, with_end=False)
        text = p.read_text() + '{"type":"frame","t":5.25,"pla'            # cut in the middle of a line
        p.write_text(text)
        s = dash.replay_summary(p)
        self.assertFalse(s["complete"])
        self.assertAlmostEqual(s["duration_s"], 5.0)

    def test_not_a_replay(self):
        p = Path(self.tmp.name) / "metrics.jsonl"
        p.write_text(jl(round_record(1, 1, 1)))
        self.assertIsNone(dash.read_replay_header(p))
        self.assertIsNone(dash.replay_summary(p))

    def test_downsample_keeps_first_and_last_sample_of_every_track(self):
        r = dash.convert_replay(self.path, hz=1.0)
        self.assertEqual(r["frames_total"], 41)
        self.assertEqual(r["frames_kept"], 11)
        self.assertEqual(r["sample_hz"], 1.0)
        self.assertEqual(r["planes"]["0"]["t"], [float(i) for i in range(11)])
        self.assertEqual(r["planes"]["1"]["t"], [0.0, 1.0, 2.0, 3.0, 4.0, 4.75])       # last frame before it died
        m = r["missiles"][0]
        self.assertEqual(m["t"], [2.5, 3.0, 4.0, 4.25])                                # first and last frame in the air
        self.assertEqual((m["uid"], m["owner"], m["target"]), (0, 0, 1))
        self.assertAlmostEqual(r["t_end"], 10.0)
        self.assertTrue(r["complete"])

    def test_full_rate_keeps_every_frame(self):
        r = dash.convert_replay(self.path, hz=4.0)
        self.assertEqual(len(r["planes"]["0"]["t"]), 41)
        self.assertEqual(r["frames_kept"], 41)
        self.assertEqual(r["sample_hz"], 4.0)

    def test_columns_and_rounding(self):
        r = dash.convert_replay(self.path, hz=1.0)
        p0 = r["planes"]["0"]
        self.assertEqual(p0["x"][2], 1016)                  # 1000 + 8 * 2.0
        self.assertEqual(p0["z"][0], 8000)
        self.assertEqual(p0["h"][0], 90)
        self.assertEqual(p0["m"][0], 10)
        self.assertEqual(p0["c"][0], 74)
        self.assertEqual(r["phases"][p0["p"][0]], "climb")
        self.assertEqual(r["header"]["map_half_m"], 64000.0)
        self.assertEqual([p["id"] for p in r["header"]["planes"]], [0, 1])
        self.assertEqual(r["header"]["planes"][0]["archetype"], "left")

    def test_events_get_the_last_known_position(self):
        r = dash.convert_replay(self.path, hz=1.0)
        by = {e["kind"]: e for e in r["events"]}
        self.assertEqual((by["launch"]["x"], by["launch"]["y"], by["launch"]["z"]), (1018, 2000, 8000))    # shooter at 2.25 s
        self.assertEqual((by["kill"]["x"], by["kill"]["y"]), (-1000, 3000))                                 # the victim's last position
        self.assertEqual((by["death"]["x"], by["death"]["z"]), (-1000, 7000))
        self.assertIn("x", by["chaff"])
        self.assertNotIn("x", by["end"])
        self.assertIn("x", by["missile_end"])                                                               # where the missile ended

    def test_corrupt_lines_are_skipped_and_counted(self):
        p = Path(self.tmp.name) / "junk.jsonl"
        write_replay(p, junk=True)
        r = dash.convert_replay(p, hz=1.0)
        self.assertEqual(r["bad_lines"], 1)
        self.assertEqual(r["frames_total"], 41)

    def test_missiles_found_by_radar(self):
        self.assertNotIn("r", dash.convert_replay(self.path, hz=1.0)["missiles"][0])     # an older replay: no column
        rows = [json.loads(line) for line in self.path.read_text().splitlines()]
        rows[0]["missile_columns"] = MISSILE_COLUMNS + ["tracked_by"]
        for row in rows:
            for m in row.get("missiles") or ():
                m.append([1] if m[10] >= 1. else [])
        p = Path(self.tmp.name) / "radar.jsonl"
        p.write_text(jl(*rows))
        m = dash.convert_replay(p, hz=1.0)["missiles"][0]
        self.assertEqual(m["r"], [[], [], [1], [1]])                                    # samples at 2.5, 3, 4, 4.25 s

    def test_convert_rejects_a_file_without_header(self):
        p = Path(self.tmp.name) / "x.jsonl"
        p.write_text(jl(dict(type="frame", t=0, planes=[], missiles=[])))
        with self.assertRaises(ValueError):
            dash.convert_replay(p)

    def test_real_replay_if_present(self):
        real = ROOT / "outputs" / "engagements" / "skirmish_2v2.jsonl"
        if not real.exists():
            self.skipTest("no outputs/engagements/skirmish_2v2.jsonl")
        r = dash.convert_replay(real, hz=1.0)
        self.assertEqual(len(r["header"]["planes"]), 4)
        self.assertGreater(r["t_end"], 10)
        self.assertLess(r["frames_kept"], r["frames_total"])
        json.loads(dash.dumps(r))


class TrainLog(unittest.TestCase):
    COLLECT = ("[09:10:01] run dir /x/s1 | config workstation | cores 48 allowed (0..47)\n"
               "[09:10:03] workers: {'pypy 7.3': 32}\n"
               "[09:11:00] bc data: shard 1, 100000 decisions (total 100000)\n"
               "[09:12:00] bc data: shard 2, 100500 decisions (total 200500)\n")
    BC = ('[09:20:00] bc dataset: {"decisions": 1000000, "trajectories": 100}\n'
          "[09:22:00] bc epoch 1  train 20.5000  val 18.0000  bad 0\n"
          "[09:24:00] bc epoch 2  train 15.0000  val -  bad 1\n")

    def test_collect_stage(self):
        p = dash.parse_train_log(self.COLLECT)
        self.assertEqual(dash.log_stage(p), "collect")
        self.assertEqual(p["collected"], 200500)
        self.assertEqual(p["n_shards"], 2)
        self.assertAlmostEqual(p["collect_rate_per_hour"], 100500 / 60.0 * 3600.0)
        self.assertEqual(p["workers"], "{'pypy 7.3': 32}")
        self.assertEqual(p["run"]["config"], "workstation")

    def test_bc_stage_with_epochs(self):
        p = dash.parse_train_log(self.COLLECT + self.BC)
        self.assertEqual(dash.log_stage(p), "bc")
        self.assertEqual(p["dataset"]["decisions"], 1000000)
        self.assertEqual([e["epoch"] for e in p["epochs"]], [1, 2])
        self.assertEqual(p["epochs"][0]["val"], 18.0)
        self.assertIsNone(p["epochs"][1]["val"])
        self.assertAlmostEqual(p["epoch_s"], 120.0)

    def test_bc_done_then_ppo(self):
        done = "[09:30:00] bc done: best epoch 2 val loss 17.5000; flying {}\n[09:30:05] ppo: starting from the BC actor\n"
        p = dash.parse_train_log(self.COLLECT + self.BC + done)
        self.assertEqual(dash.log_stage(p), "ppo_pending")
        self.assertEqual(p["done"], dict(best_epoch=2, best_val_loss=17.5))
        p = dash.parse_train_log(self.COLLECT + self.BC + done + "[09:31:00] round 1  dec 5000  valid 0.7  rew/dec 0.1\n")
        self.assertEqual(dash.log_stage(p), "ppo")
        self.assertEqual(p["ppo_round"], 1)
        self.assertEqual(dash.log_stage(dash.parse_train_log(self.COLLECT), has_metrics=True), "ppo")

    def test_empty_log_is_waiting_and_report_alone_means_pending(self):
        p = dash.parse_train_log("")
        self.assertEqual(dash.log_stage(p), "waiting")
        self.assertEqual(dash.log_stage(p, has_report=True), "ppo_pending")

    def test_clock_wraps_over_midnight(self):
        text = ("[23:58:00] bc data: shard 1, 1000 decisions (total 1000)\n"
                "[00:02:00] bc data: shard 2, 1000 decisions (total 2000)\n")
        p = dash.parse_train_log(text)
        self.assertAlmostEqual(p["collect_rate_per_hour"], 1000 / 240.0 * 3600.0)

    def test_stepper(self):
        self.assertEqual([s["state"] for s in dash.stage_steps("collect")], ["active", "todo", "todo"])
        self.assertEqual([s["state"] for s in dash.stage_steps("bc")], ["done", "active", "todo"])
        self.assertEqual([s["state"] for s in dash.stage_steps("ppo_pending")], ["done", "done", "pending"])
        self.assertEqual([s["state"] for s in dash.stage_steps("ppo")], ["done", "done", "active"])


class TailAndSpecs(unittest.TestCase):
    def test_tail_lines_strips_ansi_and_carriage_returns(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "train.log"
            p.write_text("one\n\x1b[32mtwo\x1b[0m\nprogress 10%\rprogress 100%\nlast\n")
            lines, size, cut = dash.tail_lines(p, 3)
            self.assertEqual(lines, ["two", "progress 100%", "last"])
            self.assertTrue(cut)
            self.assertEqual(size, p.stat().st_size)

    def test_remote_spec_validation(self):
        self.assertEqual(dash.parse_remote_spec("workstation:rl_runs/smoke_lb"), ("workstation", "rl_runs/smoke_lb"))
        self.assertEqual(dash.parse_remote_spec("me@host.example:~/runs/a-1"), ("me@host.example", "~/runs/a-1"))
        for bad in ("nohost", ":path", "host:", "-oProxyCommand=x:path", "host:-rf", "host:a b", "host:a;rm -rf /", "host:$(id)"):
            with self.assertRaises(ValueError, msg=bad):
                dash.parse_remote_spec(bad)

    def test_num_clamps(self):
        self.assertEqual(dash.num("5", 20, 1, 100, int), 5)
        self.assertEqual(dash.num("500", 20, 1, 100, int), 100)
        self.assertEqual(dash.num("abc", 20, 1, 100, int), 20)
        self.assertEqual(dash.num(None, 1.0, 0.25, 4.0, float), 1.0)
        self.assertEqual(dash.num("nan", 1.0, 0.25, 4.0, float), 1.0)


class RemoteSyncTests(unittest.TestCase):
    def test_command_is_read_only_and_batch_mode(self):
        with tempfile.TemporaryDirectory() as d:
            r = dash.RemoteSync("workstation:rl_runs/s1_v1", d)
            cmd = r.rsync_cmd()
            self.assertEqual(cmd[0], "rsync")
            self.assertIn("BatchMode=yes", cmd[cmd.index("-e") + 1])
            self.assertEqual(cmd[-2], "workstation:rl_runs/s1_v1/")           # the remote is the source ...
            self.assertEqual(cmd[-1], str(r.local) + "/")                        # ... the local cache the destination
            self.assertTrue(str(r.local).startswith(d))
            self.assertIn("--exclude=*", cmd)
            for f in ("metrics.jsonl", "bc_report.json", "config.json", "train.log"):
                self.assertIn("--include=" + f, cmd)
            self.assertIn("--include=replays/**", cmd)
            joined = " ".join(cmd)
            for forbidden in ("--delete", "--remove-source-files", "--rsync-path", "--files-from"):
                self.assertNotIn(forbidden, joined)
            lc = r.list_cmd()
            self.assertIn("--list-only", lc)
            self.assertEqual(lc[-1], "workstation:rl_runs/s1_v1/")
            self.assertNotIn("ckpt_000", " ".join(cmd))                         # checkpoints are never transferred

    def test_sync_once_success_and_error_status(self):
        calls = []

        def fake(cmd, **kw):
            calls.append(cmd)
            if "--list-only" in cmd:
                return subprocess.CompletedProcess(cmd, 0, "-rw-r--r--   1,000 2026/10/05 22:17:16 ppo/ckpt_000012.pt\n", "")
            Path(cmd[-1], "train.log").write_text("[10:00:00] workers: {}\n")
            return subprocess.CompletedProcess(cmd, 0, "", "")

        with tempfile.TemporaryDirectory() as d:
            r = dash.RemoteSync("host:runs/a", d, runner=fake)
            self.assertTrue(r.sync_once())
            self.assertEqual(len(calls), 2)
            st = r.status()
            self.assertIsNone(st["error"])
            self.assertIsNotNone(st["last_ok"])
            self.assertEqual([c["name"] for c in r.ckpts], ["ckpt_000012.pt"])
            self.assertTrue((r.local / "train.log").exists())

            def failing(cmd, **kw):
                return subprocess.CompletedProcess(cmd, 255, "", "ssh: connect to host host port 22: Connection refused\n")

            r2 = dash.RemoteSync("host:runs/b", d, runner=failing)
            self.assertFalse(r2.sync_once())
            self.assertIn("Connection refused", r2.status()["error"])
            self.assertIsNone(r2.status()["last_ok"])

            def timing_out(cmd, **kw):
                raise subprocess.TimeoutExpired(cmd, 1)

            r3 = dash.RemoteSync("host:runs/c", d, runner=timing_out)
            self.assertFalse(r3.sync_once())
            self.assertIn("timed out", r3.status()["error"])

    def test_vanished_files_exit_code_is_fine(self):
        def fake(cmd, **kw):
            return subprocess.CompletedProcess(cmd, 24, "", "file has vanished")

        with tempfile.TemporaryDirectory() as d:
            self.assertTrue(dash.RemoteSync("host:runs/a", d, runner=fake).sync_once())

    def test_parse_list_only(self):
        text = ("drwxr-xr-x          352 2026/10/05 22:47:41 .\n"
                "drwxr-xr-x          128 2026/10/05 22:47:41 ppo\n"
                "-rw-r--r--   18,412,031 2026/10/05 22:47:41 ppo/ckpt_000004.pt\n"
                "-rw-r--r--      123,456 2026/10/05 22:47:41 ppo/other.pt\n"
                "-rw-r--r--              2026/10/05 22:47:41 ppo/ckpt_000002.pt\n")
        got = dash.parse_list_only(text)
        self.assertEqual([c["name"] for c in got], ["ckpt_000002.pt", "ckpt_000004.pt"])
        self.assertEqual(got[1]["size"], 18412031)
        self.assertEqual(got[0]["size"], 0)


class HttpEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.run_dir = root / "runs" / "run_a"
        (cls.run_dir / "ppo").mkdir(parents=True)
        recs = [round_record(i, 1000.0 + 60 * i, 1000 * i) for i in range(1, 5)]
        recs[1]["kl_ref"] = float("nan")
        (cls.run_dir / "metrics.jsonl").write_text(jl(*recs) + '{"round": 5, "decisions_to')
        (cls.run_dir / "config.json").write_text(json.dumps(dict(name="smoke", run=dict(rounds=40), ppo=dict(target_kl=0.02, clip=0.15),
                                                             bc=dict(collect_decisions=1000, max_epochs=4, patience=3))))
        (cls.run_dir / "bc_report.json").write_text(json.dumps(dict(best_epoch=2, best_val_loss=1.5, history=[])))
        (cls.run_dir / "train.log").write_text("[10:00:00] run dir x | config smoke | cores uncapped\n[10:00:05] round 4  dec 4000  valid 0.7\n")
        (cls.run_dir / "ppo" / "ckpt_000002.pt").write_bytes(b"x" * 10)
        (cls.run_dir / "ppo" / "ckpt_000004.pt").write_bytes(b"x" * 20)
        cls.bc_run = root / "runs" / "run_bc"                      # a run that has no metrics yet: BC stage
        cls.bc_run.mkdir(parents=True)
        (cls.bc_run / "config.json").write_text(json.dumps(dict(name="lb", bc=dict(collect_decisions=1000000, max_epochs=10, patience=3))))
        (cls.bc_run / "train.log").write_text(TrainLog.COLLECT + TrainLog.BC)
        cls.replays = root / "replays"
        cls.replays.mkdir()
        write_replay(cls.replays / "match.jsonl")
        (cls.replays / "not_a_replay.jsonl").write_text(jl(round_record(1, 1, 1)))
        cls.app = dash.App(runs=[str(root / "runs")], replay_dirs=[str(cls.replays)], cache_dir=str(root / "cache"), auto_sync=False)
        cls.server = dash.make_server(cls.app, 0)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.tmp.cleanup()

    def get(self, path, headers=None, raw=False):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.request("GET", path, headers=headers or {})
        resp = conn.getresponse()
        body = resp.read()
        conn.close()
        if raw:
            return resp, body
        return resp.status, json.loads(body)

    def run_id(self, label):
        _, d = self.get("/api/runs")
        return next(r["id"] for r in d["runs"] if r["label"] == label)

    def test_binds_to_loopback_only(self):
        self.assertEqual(self.server.server_address[0], "127.0.0.1")

    def test_page_and_static_files(self):
        resp, body = self.get("/", raw=True)
        self.assertEqual(resp.status, 200)
        self.assertIn("text/html", resp.getheader("Content-Type"))
        self.assertIn("WT 控制台".encode(), body)
        for name, ctype in (("app.js", "javascript"), ("metrics.js", "javascript"), ("training.js", "javascript"),
                            ("replay.js", "javascript"), ("style.css", "css")):
            resp, body = self.get("/static/" + name, raw=True)
            self.assertEqual(resp.status, 200, name)
            self.assertIn(ctype, resp.getheader("Content-Type"))
            self.assertGreater(len(body), 500)
        for bad in ("/static/../rl_dashboard.py", "/static/nope.js", "/static/.hidden", "/static/%2e%2e/rl_dashboard.py"):
            self.assertEqual(self.get(bad, raw=True)[0].status, 404, bad)

    def test_bad_host_header_is_refused(self):
        status, d = self.get("/api/info", headers={"Host": "evil.example.com"})
        self.assertEqual(status, 403)

    def test_runs_listing(self):
        status, d = self.get("/api/runs")
        self.assertEqual(status, 200)
        labels = {r["label"]: r for r in d["runs"]}
        self.assertEqual(set(labels), {"run_a", "run_bc"})
        a = labels["run_a"]
        self.assertEqual((a["rounds"], a["round"], a["stage"], a["kind"]), (4, 4, "ppo", "local"))
        self.assertEqual(a["n_ckpts"], 2)
        self.assertTrue(a["has_bc"])
        self.assertEqual(a["rounds_total"], 40)
        bc = labels["run_bc"]
        self.assertEqual((bc["rounds"], bc["round"], bc["stage"], bc["stage_label"]), (0, None, "bc", "BC 阶段"))
        self.assertIsNotNone(bc["log_mtime"])
        self.assertIn(d["default"], {r["id"] for r in d["runs"]})

    def test_metrics_endpoint(self):
        rid = self.run_id("run_a")
        resp, body = self.get("/api/run/%s/metrics?window=2&target=100000" % rid, raw=True)
        self.assertEqual(resp.status, 200)
        self.assertNotIn(b"NaN", body)                                  # strict JSON for the browser
        d = json.loads(body)
        self.assertEqual([r["round"] for r in d["rounds"]], [1, 2, 3, 4])      # the cut-off 5th line is ignored
        self.assertTrue(d["partial_tail"])
        self.assertEqual(d["bad_lines"], 1)
        self.assertIsNone(d["rounds"][1]["kl_ref"])
        self.assertIn("per_aircraft", d["rounds"][-1])
        self.assertNotIn("per_aircraft", d["rounds"][0])
        st = d["status"]
        self.assertEqual(st["round"], 4)
        self.assertEqual(st["rate_rounds"], 2)
        self.assertAlmostEqual(st["rate_per_hour"], 60000.0)
        self.assertAlmostEqual(st["eta_s"], 96000 / 60000.0 * 3600.0)
        self.assertEqual(d["config"]["target_kl"], 0.02)

    def test_bc_config_ckpts_log_endpoints(self):
        rid = self.run_id("run_a")
        _, d = self.get("/api/run/%s/bc" % rid)
        self.assertEqual(d["report"]["best_epoch"], 2)
        _, d = self.get("/api/run/%s/config" % rid)
        self.assertEqual(d["config"]["name"], "smoke")
        _, d = self.get("/api/run/%s/ckpts" % rid)
        self.assertEqual([(c["name"], c["round"], c["size"]) for c in d["ckpts"]], [("ckpt_000002.pt", 2, 10), ("ckpt_000004.pt", 4, 20)])
        _, d = self.get("/api/run/%s/log?lines=1" % rid)
        self.assertEqual(d["file"], "train.log")
        self.assertEqual(len(d["lines"]), 1)
        self.assertIn("round 4", d["lines"][0])

    def test_stage_endpoint_for_a_run_in_bc(self):
        rid = self.run_id("run_bc")
        status, d = self.get("/api/run/%s/stage" % rid)
        self.assertEqual(status, 200)
        self.assertEqual(d["stage"], "bc")
        self.assertEqual([s["state"] for s in d["steps"]], ["done", "active", "todo"])
        self.assertEqual(d["parsed"]["collected"], 200500)
        self.assertEqual(len(d["parsed"]["epochs"]), 2)
        self.assertEqual(d["targets"]["collect_decisions"], 1000000)
        self.assertIsNotNone(d["targets"]["bc_eta_max_s"])
        _, m = self.get("/api/run/%s/metrics" % rid)
        self.assertEqual(m["rounds"], [])
        self.assertEqual(m["status"]["state"], "waiting")

    def test_unknown_run_and_resource(self):
        self.assertEqual(self.get("/api/run/doesnotexist/metrics")[0], 404)
        self.assertEqual(self.get("/api/run/%s/nothing" % self.run_id("run_a"))[0], 404)
        self.assertEqual(self.get("/api/nothing")[0], 404)

    def test_replay_list_and_download(self):
        status, d = self.get("/api/replays")
        self.assertEqual(status, 200)
        self.assertEqual([r["name"] for r in d["replays"]], ["match.jsonl"])        # the metrics-like file is not a replay
        item = d["replays"][0]
        self.assertEqual(item["summary"]["teams"], [1, 1])
        self.assertTrue(item["summary"]["complete"])
        status, r = self.get("/api/replay/%s?hz=1" % item["id"])
        self.assertEqual(status, 200)
        self.assertEqual(r["frames_total"], 41)
        self.assertEqual(r["frames_kept"], 11)
        self.assertEqual(r["planes"]["1"]["t"][-1], 4.75)
        status, r = self.get("/api/replay/%s?hz=4" % item["id"])
        self.assertEqual(len(r["planes"]["0"]["t"]), 41)
        self.assertEqual(self.get("/api/replay/ffffffffff")[0], 404)
        status, r = self.get("/api/replay/%s?hz=garbage" % item["id"])              # falls back to the default rate
        self.assertEqual(status, 200)
        self.assertEqual(r["sample_hz"], 1.0)

    def test_gzip_when_asked(self):
        _, d = self.get("/api/replays")
        rid = d["replays"][0]["id"]
        resp, body = self.get("/api/replay/%s?hz=4" % rid, headers={"Accept-Encoding": "gzip"}, raw=True)
        self.assertEqual(resp.getheader("Content-Encoding"), "gzip")
        self.assertEqual(json.loads(gzip.decompress(body))["frames_total"], 41)

    def test_sync_endpoints_without_remotes(self):
        status, d = self.get("/api/sync")
        self.assertEqual((status, d["remotes"]), (200, []))
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.request("POST", "/api/sync", body=b"")
        resp = conn.getresponse()
        self.assertEqual(resp.status, 202)
        resp.read()
        conn.request("POST", "/api/runs", body=b"")
        resp = conn.getresponse()
        self.assertEqual(resp.status, 405)
        resp.read()
        conn.close()

    def test_info(self):
        status, d = self.get("/api/info")
        self.assertEqual(status, 200)
        self.assertEqual(d["replay_dirs"], [str(self.replays)])


class RemoteRunsInTheApp(unittest.TestCase):
    def test_remote_run_is_listed_and_is_the_default(self):
        def fake(cmd, **kw):
            if "--list-only" not in cmd:
                dest = Path(cmd[-1])
                (dest / "replays").mkdir(parents=True, exist_ok=True)
                (dest / "train.log").write_text(TrainLog.COLLECT)
                (dest / "config.json").write_text(json.dumps(dict(name="workstation", bc=dict(collect_decisions=1000000))))
                write_replay(dest / "replays" / "remote_match.jsonl")
            return subprocess.CompletedProcess(cmd, 0, "", "")

        with tempfile.TemporaryDirectory() as d:
            local = Path(d) / "runs" / "old_local"
            local.mkdir(parents=True)
            (local / "metrics.jsonl").write_text(jl(round_record(1, 1.0, 100)))
            app = dash.App(runs=[str(Path(d) / "runs")], remotes=["workstation:rl_runs/s1_v1", "workstation:rl_runs/smoke_lb"],
                           replay_dirs=[str(Path(d) / "none")], cache_dir=str(Path(d) / "cache"), auto_sync=False, runner=fake)
            self.assertTrue(app.remotes[0].sync_once())
            runs = app.runs()
            self.assertEqual([r.kind for r in runs], ["local", "remote", "remote"])
            self.assertEqual(app.default_run_id(runs), runs[1].id)                 # the first --remote
            info = app.run_info(runs[1])
            self.assertEqual((info["label"], info["stage"], info["kind"]), ("s1_v1 @ workstation", "collect", "remote"))
            self.assertIsNone(info["sync"]["error"])
            self.assertEqual(app.run_info(runs[2])["sync"]["last_ok"], None)       # not synced yet
            names = [i["name"] for i in app.list_replays()]
            self.assertEqual(names, ["remote_match.jsonl"])                        # a replays/ folder of a synced run is listed


class StaticFiles(unittest.TestCase):
    ALLOWED_HOSTS = {"cdnjs.cloudflare.com", "fonts.googleapis.com", "fonts.gstatic.com", "www.w3.org"}

    def test_only_allowed_external_hosts(self):
        files = list((ROOT / "scripts" / "rl_dashboard").glob("*"))
        self.assertTrue(any(f.name == "index.html" for f in files))
        for f in files:
            if f.suffix not in (".html", ".js", ".css"):
                continue
            for host in re.findall(r"https?://([A-Za-z0-9.-]+)", f.read_text()):
                self.assertIn(host, self.ALLOWED_HOSTS, "%s references %s" % (f.name, host))

    def test_chartjs_comes_from_cdnjs(self):
        html = (ROOT / "scripts" / "rl_dashboard" / "index.html").read_text()
        self.assertRegex(html, r'<script src="https://cdnjs\.cloudflare\.com/ajax/libs/Chart\.js/[\d.]+/chart\.umd\.min\.js"')
        for src in re.findall(r'<script[^>]+src="([^"]+)"', html):
            self.assertTrue(src.startswith("/static/") or src.startswith("https://cdnjs.cloudflare.com/"), src)


STATIC = ROOT / "scripts" / "rl_dashboard"
NODE = shutil.which("node")


def run_node(script, payload):
    """Run a node script that reads JSON from stdin and prints JSON; returns the parsed output."""
    out = subprocess.run([NODE, "-e", script], input=json.dumps(payload), capture_output=True, text=True, timeout=60,
                         cwd=str(STATIC))
    if out.returncode:
        raise AssertionError(out.stderr)
    return json.loads(out.stdout)


class TrainingTabByKind(unittest.TestCase):
    """The training tab separates vs-script, self-play and mixed numbers (metrics.js) and still reads older runs."""

    # an old run (no outcomes at all), a pre-self-play run (top-level outcomes), a run with outcomes_by_kind but no
    # per-kind return, and a record with every new key
    OLD = dict(round=1, decisions_total=100, episode_return_mean=0.2, episodes_finished=29, value_loss=0.25, value_scale=0.4)
    TOP = dict(round=2, episode_return_mean=0.5, episodes_finished=10, outcomes=dict(win=6, loss=2, trade=1, none=1,
               kills=7, deaths=3), win_rate=0.6, exchange=7 / 3)
    MID = dict(round=3, episode_return_mean=-0.3, outcomes_by_kind=dict(
        vs_script=dict(win=3, loss=1, trade=0, none=1, kills=3, deaths=1, episodes=5, win_rate=0.6, exchange=3.0),
        self_play=dict(win=4, loss=4, trade=1, none=1, kills=5, deaths=5, episodes=10, win_rate=0.4, exchange=1.0)))
    NEW = dict(round=4, episode_return_mean=-0.25, value_loss=0.25, value_scale=0.4, value_rmse=0.2,
               episode_return_by_kind=dict(vs_script=0.4, self_play=-0.6),
               sampler_stats=dict(settle_ticks=80, by_env={"0": 2}, flag=True),
               outcomes_by_kind=dict(
                   vs_script=dict(win=5, loss=2, trade=1, none=2, kills=6, deaths=3, episodes=10, win_rate=0.5,
                                  exchange=2.0, trade_rate=0.1, timeouts=2, timeout_rate=0.2, none_timeout=1, none_other=1),
                   self_play=dict(win=6, loss=6, trade=2, none=2, kills=8, deaths=8, episodes=16, win_rate=0.375,
                                  exchange=1.0, trade_rate=0.125, timeouts=2, timeout_rate=0.125, none_timeout=2,
                                  none_other=0)))

    @unittest.skipUnless(NODE, "node not installed")
    def test_js_files_parse(self):
        for f in sorted(STATIC.glob("*.js")):
            out = subprocess.run([NODE, "--check", str(f)], capture_output=True, text=True, timeout=60)
            self.assertEqual(out.returncode, 0, "%s: %s" % (f.name, out.stderr))

    @unittest.skipUnless(NODE, "node not installed")
    def test_per_kind_numbers_from_old_and_new_records(self):
        script = """
const M = require('./metrics.js');
const recs = JSON.parse(require('fs').readFileSync(0, 'utf8'));
const pick = (s) => s && { win_rate: s.win_rate, exchange: s.exchange, ret: s.ret, trade_rate: s.trade_rate,
  timeout_rate: s.timeout_rate, split: s.split, episodes: s.episodes };
const out = {};
for (const [name, r] of Object.entries(recs)) {
  out[name] = { vs: pick(M.kindStats(r, 'vs_script')), sp: pick(M.kindStats(r, 'self_play')), rmse: M.valueRmse(r),
    shares: ['win', 'none', 'none_timeout', 'none_other'].map((k) => M.outcomeShare(r, 'vs_script', k)) };
}
out.pooled = M.pooled([recs.MID, recs.NEW], 'vs_script', 10);
out.pooled_new = M.pooled([recs.NEW], 'vs_script', 10);
out.has_sp = [M.hasKind([recs.OLD, recs.TOP], 'self_play'), M.hasKind([recs.OLD, recs.MID], 'self_play')];
out.extras = M.samplerExtras(recs.NEW);
console.log(JSON.stringify(out));
"""
        o = run_node(script, dict(OLD=self.OLD, TOP=self.TOP, MID=self.MID, NEW=self.NEW))
        # no outcome keys at all: only the return (all vs-script) and the value error from value_loss * value_scale
        self.assertEqual(o["OLD"]["vs"], dict(win_rate=None, exchange=None, ret=0.2, trade_rate=None, timeout_rate=None,
                                              split=False, episodes=29))
        self.assertIsNone(o["OLD"]["sp"])
        self.assertAlmostEqual(o["OLD"]["rmse"], 0.5 * 0.4)
        # before outcomes_by_kind: the top-level keys are the vs-script numbers; "none" is shown unsplit
        self.assertEqual((o["TOP"]["vs"]["win_rate"], o["TOP"]["vs"]["ret"], o["TOP"]["vs"]["trade_rate"]), (0.6, 0.5, 0.1))
        self.assertEqual(o["TOP"]["shares"], [0.6, 0.1, None, None])
        # both kinds but no per-kind return: the mixed mean belongs to neither
        self.assertEqual((o["MID"]["vs"]["ret"], o["MID"]["sp"]["ret"], o["MID"]["sp"]["win_rate"]), (None, None, 0.4))
        # new records: per-kind return, rates and the none split; the unsplit share is gone
        self.assertEqual(o["NEW"]["vs"], dict(win_rate=0.5, exchange=2.0, ret=0.4, trade_rate=0.1, timeout_rate=0.2,
                                              split=True, episodes=10))
        self.assertEqual(o["NEW"]["sp"]["ret"], -0.6)
        self.assertEqual(o["NEW"]["shares"], [0.5, None, 0.1, 0.1])
        self.assertEqual(o["NEW"]["rmse"], 0.2)
        # pooling: counts summed (8 wins of 15), the none split only when every pooled round has it
        self.assertAlmostEqual(o["pooled"]["win_rate"], 8 / 15)
        self.assertIsNone(o["pooled"]["none_timeout_rate"])
        self.assertAlmostEqual(o["pooled"]["ret"], 0.4)                        # MID has no vs-script return
        self.assertEqual((o["pooled_new"]["none_timeout_rate"], o["pooled_new"]["timeout_rate"]), (0.1, 0.2))
        self.assertEqual(o["has_sp"], [False, True])
        self.assertEqual(o["extras"], [["by_env.0", 2], ["flag", True], ["settle_ticks", 80]])

    @unittest.skipUnless(NODE, "node not installed")
    def test_history_kind_and_weapon_head_kl(self):
        rec = dict(round=5, episode_return_by_kind=dict(vs_script=0.3, history=-0.2),
                   head_kl=dict(weapon=dict(kl=3e-4, coef=1.5, coef_next=2.25, target=2e-4)),
                   outcomes_by_kind=dict(
                       vs_script=dict(win=5, loss=2, trade=1, none=2, kills=6, deaths=3, episodes=10),
                       history=dict(win=4, loss=3, trade=1, none=0, kills=5, deaths=4, episodes=8, win_rate=0.5,
                                    exchange=1.25)))
        script = """
const M = require('./metrics.js');
const r = JSON.parse(require('fs').readFileSync(0, 'utf8'));
const s = M.kindStats(r, M.KIND_HIST);
console.log(JSON.stringify({ kind: M.KIND_HIST, win_rate: s.win_rate, exchange: s.exchange, ret: s.ret,
  episodes: s.episodes, has: [M.hasKind([r], 'history'), M.hasKind([{ round: 1 }], 'history')],
  kl: M.headKl(r, 'weapon', 'kl'), coef: M.headKl(r, 'weapon', 'coef'), none: M.headKl({ round: 1 }, 'weapon', 'kl'),
  pooled: M.pooled([r], 'history', 10).win_rate }));
"""
        o = run_node(script, rec)
        self.assertEqual(o, dict(kind="history", win_rate=0.5, exchange=1.25, ret=-0.2, episodes=8, has=[True, False],
                                 kl=3e-4, coef=1.5, none=None, pooled=0.5))
        js = (STATIC / "training.js").read_text()
        for s in ("对历史对手", "kindCharts(M.KIND_HIST)", "id: 'headkl'", "id: 'headcoef'"):
            self.assertIn(s, js)
        self.assertIn(".cgroup.g-hist", (STATIC / "style.css").read_text())

    def test_labels_and_chart_order(self):
        html = (STATIC / "index.html").read_text()
        order = [html.index('src="/static/%s"' % n) for n in ("app.js", "metrics.js", "training.js")]
        self.assertEqual(order, sorted(order))                                # metrics.js hangs off RLD from app.js
        js = (STATIC / "training.js").read_text()
        self.assertIn("全部训练轨迹 (混合)", js)
        self.assertIn("对脚本局", js)
        self.assertIn("自博弈局", js)
        # "none" (survived without a kill) is never called a timeout; timeouts have their own series
        for m in re.finditer(r"\{ key: 'none[^']*', label: '([^']*)'", js):
            self.assertNotEqual(m.group(1), "超时")
        self.assertIn("key: 'none_timeout'", js)
        self.assertIn("key: 'none_other'", js)
        # the value RMS chart sits between the value loss and the explained variance
        pos = [js.index("id: '%s'" % c) for c in ("vloss", "vrmse", "ev")]
        self.assertEqual(pos, sorted(pos))

    def test_new_record_keys_reach_the_page(self):
        rec = round_record(1, 1, 100, value_rmse=0.2, episode_ends_by_kind=dict(vs_script=dict(terminal=3)),
                           sampler_stats=dict(settle_ticks=80), outcomes_by_kind=self.NEW["outcomes_by_kind"])
        slim = dash.slim_records([rec, round_record(2, 2, 200)])[0]
        for k in ("value_rmse", "episode_ends_by_kind", "sampler_stats", "outcomes_by_kind"):
            self.assertEqual(slim[k], rec[k])


if __name__ == "__main__":
    unittest.main()
