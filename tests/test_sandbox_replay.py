"""WT replay playback and forks in the sandbox (docs/sandbox_replay_spec.md)."""
import json
import math
from pathlib import Path
import tempfile
import threading
import time
import unittest
from urllib.request import urlopen

from wt_overlay.flight import FlightCommand, KeyboardCommand, SUBSTEP_S
from wt_overlay.sandbox import SandboxServer, SandboxSession, set_order
from wt_overlay.sandbox_replay import MAX_GAP_S, REFLY_MAX_AGE_S, ReplayTrack, build_fork, fork_t0

ROOT = Path(__file__).resolve().parents[1]
HAVE_MODELS = (ROOT.parent/"missle_sim"/"src"/"aim120_model").exists()
WT_REAL = ROOT/"outputs"/"engagements"/"wt_real"

PLANE_COLUMNS = ["id", "x", "y", "z", "vx", "vy", "vz", "heading_deg", "missiles", "chaff", "phase"]
MISSILE_COLUMNS = ["uid", "owner", "target", "x", "y", "z", "vx", "vy", "vz", "heading_deg", "age_s", "seeker",
                   "datalink"]
HOLE = (170., 173.)     # plane 0 has no frames strictly inside this span (a 3 s hole)
KILL_T = 150.           # plane 1 is shot down here (kill feed + death)
SHOT_T, SHOT_END = 110., 140.   # modelled AIM-120C-5 from plane 0 at plane 1
IR_T = 115.             # an unmodelled missile from plane 1 at plane 0


def plane0(t):
    return (2000.*math.sin(t/50.), -30000.+300.*t, 8000.), (40.*math.cos(t/50.), 300., 0.)


def plane1(t):
    return (1500., 30000.-280.*t, 7000.+200.*math.sin(t/40.)), (0., -280., 5.*math.cos(t/40.))


def write_replay(path):
    """2 aircraft, 1 modelled missile track, 1 unmodelled launch, a hole, a kill and a death."""
    rows = [dict(type="header", version=1, seed=None, map_half_m=60000., tick_s=None, frame_dt_s=.25, time_limit_s=200.,
                 planes=[dict(id=0, team=0, aircraft="f_16c_block_52_aesa", name="Alpha:f16", archetype="Alpha",
                              skill=None, missile="us_aim_120c_5_default", missiles=2, chaff="?",
                              wt=dict(player="Alpha", samples=800)),
                         dict(id=1, team=1, aircraft="su_30sm2", name="Bravo:su30", archetype="Bravo", skill=None,
                              missile="su_r_73_default", missiles=1, chaff="?", wt=dict(player="Bravo", samples=600))],
                 plane_columns=PLANE_COLUMNS, missile_columns=MISSILE_COLUMNS, source=dict(kind="wt_replay"))]
    events = [dict(type="event", t=SHOT_T, kind="launch", uid=0, shooter=0, target=1, missile="us_aim_120c_5_default",
                   target_basis=dict(source="seeker")),
              dict(type="event", t=IR_T, kind="launch", uid=1, shooter=1, target=0, missile="su_r_73_default",
                   target_basis=dict(source="geometry")),
              dict(type="event", t=SHOT_END, kind="missile_end", uid=0, shooter=0, target=1, result="fuse",
                   outcome="hit", outcome_basis="kill_feed"),
              dict(type="event", t=120., kind="chaff", plane=1, n=2, left="?"),
              dict(type="event", t=KILL_T, kind="kill", victim=1, killer=0, cause="missile", uid=0, source="kill_feed"),
              dict(type="event", t=KILL_T, kind="death", plane=1, cause="missile", killer=0)]
    frames = []
    for k in range(0, 801):
        t = k*.25
        planes = []
        if not HOLE[0] < t < HOLE[1]:
            p, v = plane0(t)
            planes.append([0, *p, *v, math.degrees(math.atan2(v[0], v[1])) % 360., 2 if t < SHOT_T else 1, "?", ""])
        if t <= KILL_T:
            p, v = plane1(t)
            planes.append([1, *p, *v, 180., 1 if t < IR_T else 0, "?", ""])
        missiles = []
        if SHOT_T <= t <= SHOT_END:
            a, _ = plane0(SHOT_T)
            vm = (0., 900., 0.)
            missiles.append([0, 0, 1, a[0], a[1]+900.*(t-SHOT_T), a[2], *vm, 0., t-SHOT_T, 0, 0])
        frames.append(dict(type="frame", t=t, planes=planes, missiles=missiles))
    out = rows+sorted(events+frames, key=lambda r: (r["t"], r["type"] == "frame"))
    out.append(dict(type="end", reason="replay_end", t=200.))
    path.write_text("".join(json.dumps(r)+"\n" for r in out), encoding="utf-8")
    return path


class TrackTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.track = ReplayTrack(write_replay(Path(self.tmp.name)/"synthetic.jsonl"))

    def tearDown(self):
        self.tmp.cleanup()

    def test_exact_at_samples_continuous_between_and_none_outside(self):
        series = self.track.planes[0]
        for i in range(0, len(series.t), 37):
            st = self.track.state_at(0, series.t[i])
            self.assertEqual(st["position"], series.p[i])
            self.assertEqual(st["velocity"], series.v[i])
        previous = None
        t = 20.
        while t < 60.:
            st = self.track.state_at(0, t)
            if previous is not None:
                self.assertLess(math.dist(st["position"], previous), 310.*.01*1.5)
            previous = st["position"]
            t += .01
        # Hermite between frames follows the true curve far better than a frame spacing of travel
        p, _ = plane0(30.1)
        self.assertLess(math.dist(self.track.state_at(0, 30.1)["position"], p), 1.)
        self.assertGreater(HOLE[1]-HOLE[0], MAX_GAP_S)
        self.assertIsNone(self.track.state_at(0, sum(HOLE)/2))
        self.assertIsNotNone(self.track.state_at(0, HOLE[0]))
        self.assertIsNone(self.track.state_at(0, -1.))
        self.assertIsNone(self.track.state_at(0, 200.5))
        self.assertIsNone(self.track.state_at(1, KILL_T+1.))
        self.assertIsNone(self.track.state_at(7, 10.))
        # pitch from the velocity, roll estimated (plane 0 turns gently: small bank)
        st = self.track.state_at(1, 10.)
        self.assertAlmostEqual(st["pitch_deg"], math.degrees(math.asin(st["velocity"][2]/st["speed"])), places=9)
        self.assertLess(abs(self.track.state_at(0, 50.)["roll_deg"]), 10.)


@unittest.skipUnless(HAVE_MODELS, "sibling missile models not installed")
class PlaybackTests(unittest.TestCase):
    def test_load_seek_positions_kill_and_listing(self):
        with tempfile.TemporaryDirectory() as tmp:
            wt = Path(tmp)/"wt"
            wt.mkdir()
            track = ReplayTrack(write_replay(wt/"synthetic.jsonl"))
            session = SandboxSession(output_root=Path(tmp)/"out", replay_root=wt)
            server = SandboxServer(("127.0.0.1", 0), session)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                with urlopen(f"http://127.0.0.1:{server.server_port}/api/replays") as response:
                    listing = json.load(response)
                self.assertEqual([(e["file"], e["kind"], e["players"], e["units"], e["duration_s"]) for e in listing],
                                 [("wt_real/synthetic.jsonl", "wt", 2, 2, 200.)])
                send(self, session, "load_replay", file="../wt/synthetic.jsonl", error=True)
                send(self, session, "load_replay", file="wt_real/synthetic.jsonl")
                send(self, session, "seek", t=100.3)
                state = session.state()
                self.assertEqual(state["status"], "playback")
                self.assertAlmostEqual(state["time_s"], 100.3)
                planes = {p["id"]: p for p in state["planes"]}
                for ident in (0, 1):
                    self.assertEqual(planes[ident]["position_m"], list(track.state_at(ident, 100.3)["position"]))
                    self.assertTrue(planes[ident]["alive"])
                self.assertEqual(planes[0]["player"], "Alpha")
                self.assertFalse(planes[0]["ai"])
                self.assertEqual(planes[0]["attitude_source"], "estimated")
                self.assertEqual(state["replay"]["duration_s"], 200.)
                self.assertEqual([m["kind"] for m in state["replay"]["markers"]], ["launch", "launch", "kill", "death"])
                send(self, session, "seek", t=120.)
                state = session.state()
                self.assertEqual([(m["uid"], m["shooter"], m["target"], m["target_basis"]) for m in state["missiles"]],
                                 [(0, 0, 1, "seeker")])
                send(self, session, "seek", t=KILL_T+2.)
                planes = {p["id"]: p for p in session.state()["planes"]}
                self.assertFalse(planes[1]["alive"])
                self.assertEqual(planes[1]["death"]["cause"], "missile")
                self.assertTrue(planes[0]["alive"])
                self.assertEqual(session.state()["teams_alive"], [1, 0])
                self.assertIn("kill", [e["kind"] for e in session.state()["events"]])
                send(self, session, "seek", t=KILL_T+11.)
                self.assertNotIn(1, [p["id"] for p in session.state()["planes"]])
                send(self, session, "seek", t=195.)
                send(self, session, "speed", speed=64)
                send(self, session, "play")
                wait(self, session, lambda s: not s["replay"]["playing"])
                self.assertEqual(session.state()["time_s"], 200.)
                send(self, session, "unload")
                self.assertEqual(session.state()["status"], "setup")
            finally:
                server.shutdown()
                server.server_close()
                session.close()


@unittest.skipUnless(HAVE_MODELS, "sibling missile models not installed")
class ForkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.track = ReplayTrack(write_replay(Path(cls.tmp.name)/"synthetic.jsonl"))
        from wt_overlay.engagement import default_library
        cls.library = default_library()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_to(self, eng, replay_t):
        while eng.offset+eng.time < replay_t-1e-9:
            eng.step()

    def test_pinned_aircraft_follow_the_replay(self):
        eng = build_fork(self.track, 100., {}, library=self.library)
        self.assertEqual(eng.offset, 100.)
        worst = 0.
        while eng.offset+eng.time < 110.-1e-9:
            eng.step()
            for p in eng.planes:
                want = self.track.kinematics(eng.rids[p.ident], eng.offset+eng.time)[0]
                worst = max(worst, math.dist(want, p.flight.state.position))
        self.assertLess(worst, 1.)
        self.assertTrue(all(p.flight.tracked and p.controller.mode == "track" for p in eng.planes))
        # the ring buffer answers missile queries as for a flying aircraft
        position, _ = eng.planes[0].state_at(eng.time-.5*SUBSTEP_S)
        self.assertLess(math.dist(position, self.track.kinematics(0, eng.offset+eng.time-.5*SUBSTEP_S)[0]), 1.)

    def test_release_is_continuous_then_commands_fly_it(self):
        eng = build_fork(self.track, 120., {0: "manual"}, library=self.library, me=0)
        self.assertEqual(eng.offset, SHOT_T)      # the AIM-120 launched at 110 s is still flying at 120 s
        p = eng.planes[0]
        while p.flight.tracked:
            before = p.flight.state.velocity
            eng.step()
        released_at = eng.offset+eng.time
        self.assertLess(abs(released_at-120.), SUBSTEP_S)
        self.assertLess(math.dist(before, p.flight.state.velocity), 5.)
        after_release = p.flight.state.velocity
        eng.step()
        self.assertLess(math.dist(after_release, p.flight.state.velocity), 5.)
        self.assertEqual(p.controller.mode, "manual")
        self.assertIsInstance(p.flight.command, FlightCommand)
        self.assertEqual(len(eng.ghosts[0]), 79)   # 120 .. 200 s every 1 s, less 171 and 172 s (in the hole)
        set_order(eng, 0, dict(mode="manual", heading_deg=90., altitude_m=9000., speed_mps=300.))
        for _ in range(240):
            eng.step()
        self.assertGreater(p.own.heading_deg, 10.)
        set_order(eng, 0, dict(mode="pilot"))
        p.controller.keys = dict(roll=1, pitch=0, throttle=0, airbrake=False)
        p.controller.keys_until = time.monotonic()+10
        eng.step()
        self.assertIsInstance(p.flight.command, KeyboardCommand)
        with self.assertRaises(ValueError):
            build_fork(self.track, 120., {1: "warp"}, library=self.library)

    def test_replayed_launch_and_skipped_unmodelled_missile(self):
        eng = build_fork(self.track, 100., {}, library=self.library)
        self.run_to(eng, SHOT_T+SUBSTEP_S)
        launches = [e for e in eng.log if e["kind"] == "launch"]
        self.assertEqual([(e["shooter"], e["target"], e["missile"]) for e in launches], [(0, 1, "us_aim_120c_5")])
        self.assertLess(abs(eng.offset+launches[0]["t"]-SHOT_T), SUBSTEP_S)
        self.assertEqual(eng.missiles[0].shooter.ident, 0)
        self.assertEqual(eng.planes[0].missile_id, "us_aim_120c_5")
        self.assertEqual(eng.planes[0].missiles, 1)
        self.run_to(eng, IR_T+SUBSTEP_S)
        skipped = [e for e in eng.log if e["kind"] == "replay_shot_skipped"]
        self.assertEqual([(e["replay_uid"], e["missile"]) for e in skipped], [(1, "su_r_73_default")])
        self.assertIn("不在模型库", skipped[0]["reason"])
        self.assertEqual(eng.planes[1].missiles, 0)

    def test_refly_window_cpa_and_age(self):
        # live until closest approach when the import gives t_cpa; older launches than the window are not re-flown
        end = self.track.end_by_uid[0]
        self.assertEqual(self.track.missile_live_until(0), SHOT_END)
        end["t_cpa"] = 125.
        try:
            self.assertEqual(self.track.missile_live_until(0), 125.)
            self.assertEqual(fork_t0(self.track, 124., self.library), SHOT_T)
            self.assertEqual(fork_t0(self.track, 126., self.library), 126.)        # past its closest approach
            self.assertEqual(fork_t0(self.track, 124., self.library, max_age_s=10.), 124.)   # older than the window
            eng = build_fork(self.track, 124., {}, library=self.library)
            self.assertEqual(eng.offset, SHOT_T)
        finally:
            del end["t_cpa"]


@unittest.skipUnless(HAVE_MODELS and any(WT_REAL.glob("*.jsonl")), "no real WT replay or models")
class RealReplayTests(unittest.TestCase):
    def test_real_file_loads_and_a_released_fork_runs_10_s(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = SandboxSession(output_root=tmp, replay_root=WT_REAL)
            try:
                entry = next(e for e in session.replays() if e["kind"] == "wt")
                send(self, session, "load_replay", file=entry["file"])
                send(self, session, "seek", t=120.)
                state = session.state()
                rid = next(p["id"] for p in state["planes"] if p["alive"] and p["controllable"] and not p["ai"])
                send(self, session, "speed", speed=64)
                send(self, session, "fork", t=120., control={str(rid): "auto"}, me=rid, timeout=120)
                wait(self, session, lambda s: s["time_s"] >= 10. or s["status"] in ("error", "completed"), 180)
                send(self, session, "pause")
                state = session.state()
                self.assertIsNone(state["error"])
                self.assertGreaterEqual(state["time_s"], 10.)
                self.assertEqual(state["fork"]["t_fork"], 120.)
                send(self, session, "save")
                scenario = json.loads((Path(state["output"]["path"]) if state["output"] else
                                       Path(session.state()["output"]["path"]))
                                      .joinpath("scenario.json").read_text())
                self.assertEqual(scenario["replay_source"]["t_fork"], 120.)
                self.assertEqual(len(scenario["replay_source"]["sha256"]), 64)
                send(self, session, "reset")
                self.assertEqual(session.state()["status"], "playback")
                self.assertEqual(session.state()["time_s"], 120.)
            finally:
                session.close()


def send(case, session, action, error=False, timeout=30, **fields):
    jid = session.submit(dict(action=action, **fields))
    until = time.monotonic()+timeout
    while time.monotonic() < until:
        receipt = session.job(jid)
        if receipt["done"]:
            case.assertEqual("error" in receipt, error, receipt)
            return receipt
        time.sleep(.01)
    case.fail("command did not complete")


def wait(case, session, condition, timeout=30):
    until = time.monotonic()+timeout
    while time.monotonic() < until:
        if condition(session.state()):
            return
        time.sleep(.02)
    case.fail(f"session did not reach the state: {session.state()['status']}")


if __name__ == "__main__":
    unittest.main()
