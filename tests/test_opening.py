"""Opening-shot aid (docs/opening_shot_spec.md): start detection, dead reckoning, the pitch cue and its phases, the
plan cache, the controller wiring with a stubbed planner, and (missile_sim present) one planner run against spec
section 7.1."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import math
import os
from pathlib import Path
import time
import unittest
from unittest.mock import patch

from wt_overlay.app import OverlayController
from wt_overlay.contracts import FlightState
from wt_overlay.opening import (DeadReckoning, OpeningPlan, OpeningSettings, PlanCache, PlanRequest, Spawn,
                                SpawnLatch, cue, loft_rule, plan_opening, validate_settings, velocity_frame_direction,
                                worker_init)

MISSILE_SIM = Path(__file__).resolve().parents[2]/"missle_sim"
HAVE_MISSILE_SIM = (MISSILE_SIM/"src"/"aim120_model").exists()


def state(t, altitude=2500., tas=290., vy=0., heading=30., aircraft="f_16c_block_52_aesa", valid=True, **kw):
    return FlightState(t, valid, altitude_m=altitude, tas_mps=tas, vertical_speed_mps=vy, heading_deg=heading,
                       aircraft_id=aircraft, **kw)


PLAN = OpeningPlan(True, climb_deg=15., pull_s=45., pitch_deg=24., launch_s=46.5, window_s=(35., 51.5),
                   launch_range_m=65500., hit_s=95.1, flight_s=48.6, launch_altitude_m=6987., launch_tas_mps=426.,
                   target_altitude_m=9998., profile=tuple((float(t), 2500.+100.*t, 290.+3.*t, 15.) for t in range(47)))
SPAWN = Spawn(100., 30., 2500., 290., "f_16c_block_52_aesa")


class SettingsTests(unittest.TestCase):
    def test_defaults_are_the_user_scenario(self):
        s = OpeningSettings()
        validate_settings(s)
        self.assertEqual((s.distance_m, s.enemy_aircraft, s.enemy_climb_deg, s.enemy_level_m, s.enemy_fuel_fraction),
                         (100000., "ef_2000_aesa", 15., 10000., 1.))

    def test_rejects_out_of_range_values(self):
        for change in (dict(distance_m=10000.), dict(enemy_fuel_fraction=0.), dict(enemy_climb_deg=60.),
                       dict(enemy_level_m=20000.), dict(workers=0), dict(enemy_missiles=1.5), dict(enabled=1)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_settings(replace(OpeningSettings(), **change))

    def test_loft_rule_steepens_with_range(self):
        self.assertEqual([loft_rule(r) for r in (50000., 60000., 75000., 90000.)], [None, 25., 45., 55.])


def climb(t, rate=30., **kw):
    return state(t, vy=rate, **kw)


class StartTests(unittest.TestCase):
    def test_starts_on_the_first_clear_climb(self):
        latch = SpawnLatch()
        self.assertIsNone(latch.update(state(10.)))          # level at spawn speed: armed, not started
        self.assertIn("加速或爬升", latch.note)
        start = latch.update(climb(10.5, altitude=2510.))
        self.assertEqual((start.time_s, start.altitude_m, start.manual), (10.5, 2510., False))
        self.assertIsNone(latch.update(climb(11.)))            # once per life
        self.assertIs(latch.spawn, start)

    def test_starts_on_clear_acceleration(self):
        latch = SpawnLatch()
        self.assertIsNone(latch.update(state(0., tas=270.)))
        self.assertIsNone(latch.update(state(1., tas=273.)))
        self.assertIsNotNone(latch.update(state(2., tas=277.)))

    def test_any_speed_or_altitude_in_the_air_counts(self):
        # Spawns are 0.8 Mach and up (user): no speed or altitude window.
        self.assertIsNotNone(SpawnLatch().update(climb(0., altitude=5200., tas=420.)))
        self.assertIsNone(SpawnLatch().update(climb(0., altitude=100., tas=60.)))    # on the runway

    def test_new_life_rearms(self):
        for event in (dict(t=20.),                                   # telemetry gap
                      dict(t=11., fuel_kg=4000.),                     # fuel went up: respawned
                      dict(t=11., altitude=4000.),                    # teleported
                      dict(t=11., aircraft="j_16")):                  # another aircraft
            with self.subTest(event=event):
                latch = SpawnLatch()
                latch.update(state(9.5, fuel_kg=1500.))
                first = latch.update(climb(10., fuel_kg=1499.))
                self.assertIsNotNone(first)
                t = event.pop("t")
                second = latch.update(climb(t, **{"fuel_kg": 1498., **event}))
                self.assertIsNotNone(second)
                self.assertIsNot(second, first)

    def test_a_lost_poll_or_a_type_appearing_is_not_a_new_life(self):
        latch = SpawnLatch()
        first = latch.update(climb(0., aircraft=None))
        self.assertIsNone(latch.update(climb(2., aircraft="j_16")))       # type identified later
        self.assertIsNone(latch.update(climb(4.5, aircraft="j_16")))      # 2.5 s gap
        self.assertIs(latch.spawn, first)

    def test_manual_start(self):
        latch = SpawnLatch()
        self.assertIsNone(latch.update(state(0., altitude=8000., tas=450.)))
        manual = latch.latch(state(1., altitude=8000., tas=450.))
        self.assertTrue(manual.manual)
        self.assertIsNone(latch.latch(FlightState(2., False)))

    def test_dead_reckoning_along_the_line(self):
        reckoning = DeadReckoning(SPAWN)
        for k in range(1, 101):
            reckoning.update(state(100.+.1*k, tas=300.))
        self.assertAlmostEqual(reckoning.x, 3000., delta=1.)
        reckoning.update(state(130., tas=300.))   # a 20 s gap is not integrated
        self.assertAlmostEqual(reckoning.x, 3000., delta=1.)


class CueTests(unittest.TestCase):
    def test_direction_in_the_velocity_frame(self):
        self.assertEqual(tuple(round(v, 6) for v in velocity_frame_direction(30., 10., 30., 10.)), (0., 0., 1.))
        up = velocity_frame_direction(30., 0., 30., 10., gain=1.)
        self.assertAlmostEqual(up[1], math.sin(math.radians(10.)), places=6)
        self.assertGreater(velocity_frame_direction(30., 0., 30., 5.)[1], 3*math.sin(math.radians(5.)))

    def test_phases_follow_the_plan(self):
        at = lambda t, **kw: cue(PLAN, SPAWN, state(100.+t, **kw))
        level = at(20.)
        self.assertEqual((level.phase, level.target_path_deg, level.yaw_error_deg), ("爬升", 15., None))
        self.assertEqual(level.lines[0], "全加力 爬升 15°（现在 0°）")
        self.assertIn("25 s 后拉起到 24° 发射", level.lines[1])
        self.assertIn("66 km", level.lines[1])
        self.assertAlmostEqual(level.direction[0], 0.)
        self.assertGreater(level.direction[1], 0.)
        on = at(20., vy=290.*math.sin(math.radians(15.)))
        self.assertTrue(on.on_course)
        pull = at(45.5)
        self.assertEqual((pull.phase, pull.target_path_deg, pull.urgent), ("拉起", 24., False))
        self.assertEqual(pull.lines[1], "1 s 后发射")
        fire = at(47.)
        self.assertEqual((fire.phase, fire.urgent, fire.launch_range_m), ("发射", True, 65500.))
        self.assertEqual(at(60.).phase, "结束")
        self.assertFalse(at(60.).active)

    def test_countdown_changes_once_a_second(self):
        lines = {cue(PLAN, SPAWN, state(100.+20.+.1*k)).lines for k in range(10)}
        self.assertLessEqual(len(lines), 2)

    def test_before_a_plan_the_default_climb_without_any_computing_text(self):
        waiting = cue(None, SPAWN, state(101., vy=10.))
        self.assertEqual((waiting.phase, waiting.target_path_deg), ("爬升", 15.))
        self.assertEqual(len(waiting.lines), 1)
        self.assertNotIn("计算", waiting.lines[0])
        self.assertTrue(waiting.active)
        self.assertEqual(cue(None, None, state(1.)).phase, "等待开始")
        failed = cue(OpeningPlan(False, "打不到"), SPAWN, state(110.))
        self.assertEqual((failed.phase, failed.target_path_deg), ("爬升", 15.))
        self.assertIn("打不到", failed.lines[1])

    def test_fire_along_the_climb_without_pull_up(self):
        plan = replace(PLAN, pitch_deg=None, pull_s=46.5)
        climb_cue = cue(plan, SPAWN, state(130.))
        self.assertIn("17 s 后发射", climb_cue.lines[1])
        self.assertEqual(cue(plan, SPAWN, state(147.)).target_path_deg, 15.)

    def test_slow_or_on_the_ground(self):
        parked = cue(PLAN, SPAWN, state(110., tas=0.))
        self.assertIsNone(parked.direction)
        self.assertIn("速度太低", parked.lines[-1])


class CacheTests(unittest.TestCase):
    def request(self, altitude=2500., tas=290., mass=11000., aircraft="f_16c_block_52_aesa"):
        return PlanRequest(aircraft, mass, "us_aim_120c_5", OpeningSettings(True, "us_aim_120c_5"), altitude, tas,
                           0., 0., altitude, tas, 0.)

    def test_exact_bins_nearest_and_disk(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "sub", "plans.json")
            cache = PlanCache(path)
            self.assertIsNone(cache.get(self.request()))
            self.assertIsNone(cache.nearest(self.request()))
            cache.put(self.request(), PLAN)
            self.assertEqual(cache.get(self.request(altitude=2600., tas=295., mass=11300.)).hit_s, PLAN.hit_s)
            self.assertIsNone(cache.get(self.request(altitude=3500.)))
            self.assertEqual(cache.nearest(self.request(altitude=3500.)).hit_s, PLAN.hit_s)
            self.assertIsNone(cache.nearest(self.request(aircraft="j_16")))
            cache.put(self.request(altitude=4000.), OpeningPlan(False, "打不到"))   # failures are not kept
            reloaded = PlanCache(path)
            self.assertEqual(reloaded.get(self.request()).window_s, PLAN.window_s)
            self.assertEqual(len(reloaded.plans), 1)
            self.assertEqual(PlanCache(os.path.join(folder, "missing.json")).plans, {})


class MapDirectionTests(unittest.TestCase):
    def test_player_icon_direction(self):
        from wt_overlay.telemetry import player_track_deg
        other = {"type": "aircraft", "icon": "Fighter", "dx": 1., "dy": 0.}
        self.assertAlmostEqual(player_track_deg([other, {"icon": "Player", "dx": 0., "dy": -1.}]), 0.)
        self.assertAlmostEqual(player_track_deg([{"icon": "player", "dx": 1., "dy": 0.}]), 90.)
        self.assertAlmostEqual(player_track_deg([{"icon": "Player", "dx": -.7071, "dy": .7071}]), 225., places=3)
        self.assertIsNone(player_track_deg([other]))
        self.assertIsNone(player_track_deg({"icon": "Player"}))
        self.assertIsNone(player_track_deg([{"icon": "Player", "dx": 0., "dy": 0.}]))

    def test_client_reads_the_map_only_when_asked_and_without_a_compass(self):
        from wt_overlay.telemetry import TelemetryClient
        payloads = {"/state": {"valid": True, "H, m": 2500, "TAS, km/h": 1044, "Vy, m/s": 0},
                    "/indicators": {"valid": True, "type": "f_16c_block_52_aesa"},
                    "/map_obj.json": [{"type": "aircraft", "icon": "Player", "x": .5, "y": .5, "dx": 1., "dy": 0.}]}
        client = TelemetryClient()
        calls = []
        client._load = lambda endpoint, opener=None: calls.append(endpoint) or payloads[endpoint]

        def settle():
            if client._map_thread is not None:
                client._map_thread.join(1.)

        self.assertIsNone(client.poll(time_s=1.).track_deg)   # nobody asked for the map
        self.assertNotIn("/map_obj.json", calls)
        client.want_track = True
        client.poll(time_s=2.)
        settle()
        first = client.poll(time_s=2.1)
        self.assertEqual((first.heading_deg, first.track_deg), (None, 90.))
        self.assertEqual(calls.count("/map_obj.json"), 1)   # at most every MAP_INTERVAL_S
        payloads["/indicators"]["compass"] = 45.
        with_compass = client.poll(time_s=3.)
        self.assertEqual((with_compass.heading_deg, with_compass.track_deg), (45., None))
        self.assertEqual(calls.count("/map_obj.json"), 1)
        payloads["/indicators"].pop("compass")
        payloads["/map_obj.json"] = "not a list"
        client.poll(time_s=4.)
        settle()
        self.assertIsNone(client.poll(time_s=4.1).track_deg)
        self.assertTrue(client.poll(time_s=4.2).valid)

    def test_a_slow_or_failing_map_never_delays_or_breaks_the_state(self):
        import threading
        from wt_overlay.telemetry import TelemetryClient
        release = threading.Event()
        payloads = {"/state": {"valid": True, "H, m": 2500, "TAS, km/h": 1044, "Vy, m/s": 0},
                    "/indicators": {"valid": True, "type": "f_16c_block_52_aesa"}}

        def load(endpoint, opener=None):
            if endpoint == "/map_obj.json":
                release.wait(2.)
                raise KeyError("odd map payload")
            return payloads[endpoint]
        client = TelemetryClient()
        client._load = load
        client.want_track = True
        started = time.monotonic()
        states = [client.poll(time_s=float(t)) for t in range(5)]
        self.assertLess(time.monotonic()-started, 1.)
        self.assertTrue(all(s.valid for s in states))
        release.set()
        client._map_thread.join(1.)
        self.assertTrue(client.poll(time_s=10.).valid)


class FakeClient:
    def __init__(self):
        self.state = FlightState(0., False)

    def poll(self, time_s=None):
        return replace(self.state, time_s=time_s)


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.client = FakeClient()
        self.controller = OverlayController(client=self.client)
        self.requests = []

    def tearDown(self):
        self.controller.stop()

    def fake_plan(self, request, run, cancel):
        self.requests.append(request)
        return replace(PLAN, base_s=request.t_s)

    def enable(self, **kw):
        self.controller.submit({"action": "opening",
                                "settings": asdict(OpeningSettings(True, "us_aim_120c_5", **kw))})

    def tick(self, t, **kw):
        if kw.get("valid", True):
            self.client.state = state(t, mass_kg=11000., **kw)
        else:
            self.client.state = FlightState(t, False)
        return self.controller.tick(t)

    def settle(self, t, count=1, **kw):
        for _ in range(200):
            snapshot = self.tick(t, **kw)
            if len(self.requests) >= count and self.controller._opening_job is None:
                return self.tick(t, **kw)
            time.sleep(.005)
        self.fail("no plan")

    def test_off_by_default(self):
        self.assertIsNone(self.tick(1.).opening)

    def test_computes_ahead_then_the_start_finds_the_plan(self):
        self.enable()
        with patch("wt_overlay.app.plan_opening", side_effect=self.fake_plan):
            self.assertEqual(self.tick(0.).opening.phase, "等待开始")
            self.settle(.1)                                     # computed ahead from where the aircraft is
            self.assertEqual(self.requests[0].t_s, 0.)
            snapshot = self.tick(.2, vy=30.)                     # the climb starts the clock
            self.assertEqual(snapshot.opening.phase, "爬升")
            self.assertIsNotNone(snapshot.opening.plan)          # straight from the cache
            self.assertEqual(len(self.requests), 1)
            self.assertIn("拉起到 24°", snapshot.opening.lines[1])

    def test_plans_once_and_never_again(self):
        self.enable()
        with patch("wt_overlay.app.plan_opening", side_effect=self.fake_plan):
            snapshot = self.tick(0., vy=30.)                     # starts at once
            self.assertEqual((snapshot.opening.phase, len(snapshot.opening.lines)), ("爬升", 1))
            self.settle(.1)
            for t in range(1, 60):
                self.tick(float(t), vy=0. if t % 7 else 40., tas=290.+t)
                time.sleep(.001)
            self.assertEqual(len(self.requests), 1)

    def test_nearest_cached_plan_at_once_then_the_exact_one(self):
        self.enable()
        other = PlanRequest("f_16c_block_52_aesa", 11000., "us_aim_120c_5", OpeningSettings(True, "us_aim_120c_5"),
                            4000., 290., 0., 0., 4000., 290., 0.)
        self.controller._opening_cache.put(other, replace(PLAN, hit_s=99.))
        with patch("wt_overlay.app.plan_opening", side_effect=self.fake_plan):
            snapshot = self.tick(0., vy=30.)
            self.assertEqual(snapshot.opening.plan.hit_s, 99.)    # nearest, shown at once
            snapshot = self.settle(.1)
            self.assertEqual(snapshot.opening.plan.hit_s, PLAN.hit_s)

    def test_missing_input_keeps_the_default_climb_and_plans_once_it_is_set(self):
        self.controller.submit({"action": "opening", "settings": asdict(OpeningSettings(True, ""))})
        with patch("wt_overlay.app.plan_opening", side_effect=self.fake_plan):
            snapshot = self.tick(0., vy=30.)
            self.assertEqual((snapshot.opening.phase, snapshot.opening.target_path_deg), ("爬升", 15.))
            self.assertIn("选择导弹", snapshot.opening.lines[1])
            for t in (1., 2., 3., 4.):
                self.tick(t)
            self.enable()
            self.settle(5.)
            self.assertEqual(self.requests[0].t_s, 5.)          # from where the aircraft is now
            self.assertEqual(len(self.requests), 1)

    def test_no_type_from_8111_then_the_aircraft_is_picked(self):
        from wt_overlay.telemetry import INDICATORS_MISSING
        self.enable()
        with patch("wt_overlay.app.plan_opening", side_effect=self.fake_plan):
            self.client.state = state(0., vy=30., aircraft=None, heading=None, mass_kg=11000.,
                                      notes=(INDICATORS_MISSING,))
            snapshot = self.controller.tick(0.)
            self.assertEqual(snapshot.opening.phase, "爬升")
            self.assertIn("8111 没给机型", snapshot.opening.lines[1])
            self.controller.submit({"action": "aircraft", "id": "f_16c_block_52_aesa"})
            for k in range(1, 100):
                self.client.state = replace(self.client.state, time_s=.1*k)
                self.controller.tick(.1*k)
                if self.requests:
                    break
                time.sleep(.005)
            self.assertEqual(self.requests[0].aircraft_id, "f_16c_block_52_aesa")
            self.assertEqual(self.controller._latch.spawn.time_s, 0.)

    def test_new_life_starts_again(self):
        self.enable()
        with patch("wt_overlay.app.plan_opening", side_effect=self.fake_plan):
            first = self.tick(0., vy=30.).opening
            self.tick(10., valid=False)
            self.assertEqual(self.tick(30.).opening.phase, "等待开始")
            again = self.tick(30.5, vy=30.).opening
            self.assertEqual((again.phase, again.t_s), ("爬升", 0.))
            self.assertIsNot(self.controller._latch.spawn, first)

    def test_manual_start(self):
        self.enable()
        with patch("wt_overlay.app.plan_opening", side_effect=self.fake_plan):
            self.tick(0., altitude=8000., tas=450.)
            self.controller.submit({"action": "opening_restart"})
            self.assertEqual(self.tick(1., altitude=8000., tas=450.).opening.phase, "爬升")

    def test_disabling_drops_the_start(self):
        self.enable()
        with patch("wt_overlay.app.plan_opening", side_effect=self.fake_plan):
            self.tick(0., vy=30.)
            self.controller.submit({"action": "opening", "settings": asdict(OpeningSettings(False, "us_aim_120c_5"))})
            self.assertIsNone(self.tick(1.).opening)
            self.assertIsNone(self.controller._latch.spawn)

    def test_invalid_settings_are_refused(self):
        with self.assertRaises(ValueError):
            self.controller.submit({"action": "opening", "settings": {"distance_m": 1.}})
        with self.assertRaises(ValueError):
            self.controller.submit({"action": "opening",
                                    "settings": asdict(replace(OpeningSettings(), workers=0))})


@unittest.skipUnless(HAVE_MISSILE_SIM and os.environ.get("WT_SLOW"), "set WT_SLOW=1 (missile_sim, ~15 s, 4 processes)")
class PlannerAcceptanceTests(unittest.TestCase):
    def test_f16_against_the_typhoon_matches_spec_7_1(self):
        from wt_overlay.engagement import default_library
        from wt_overlay.flight import fuel_data
        from wt_overlay.opening import _missile_mass
        fuel = fuel_data("f_16c_block_52_aesa")
        mass = fuel.empty_kg+.3*fuel.internal_kg+6*_missile_mass(default_library(), "us_aim_120c_5")
        settings = OpeningSettings(True, "us_aim_120c_5")
        request = PlanRequest("f_16c_block_52_aesa", mass, "us_aim_120c_5", settings, 2500., 290., 0., 0., 2500.,
                              290., 0.)
        with ProcessPoolExecutor(4, initializer=worker_init) as pool:
            plan = plan_opening(request, lambda f, jobs: list(pool.map(f, jobs)))
        self.assertTrue(plan.available)
        self.assertAlmostEqual(plan.hit_s, 95.1, delta=.5)
        self.assertTrue(35. <= plan.launch_s <= 56.)
        self.assertEqual(plan.climb_deg, 15.)


if __name__ == "__main__":
    unittest.main()
