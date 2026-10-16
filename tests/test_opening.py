"""Opening-shot aid (docs/opening_shot_spec.md): spawn detection, dead reckoning, cue geometry and phases, the
controller wiring with a stubbed planner, and (missile_sim present) one planner run against spec section 7.1."""

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
from wt_overlay.opening import (DeadReckoning, OpeningPlan, OpeningSettings, PlanRequest, Spawn, SpawnLatch, cue,
                                deviates, loft_rule, plan_opening, validate_settings, velocity_frame_direction,
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


class SpawnTests(unittest.TestCase):
    def test_first_valid_air_sample_is_a_spawn(self):
        latch = SpawnLatch()
        spawn = latch.update(state(10.))
        self.assertEqual((spawn.time_s, spawn.heading_deg, spawn.altitude_m, spawn.manual), (10., 30., 2500., False))
        self.assertIsNone(latch.update(state(10.1, altitude=2510., vy=30.)))
        self.assertIs(latch.spawn, spawn)

    def test_a_lost_poll_is_not_a_respawn_but_a_long_gap_is(self):
        latch = SpawnLatch()
        latch.update(state(10.))
        latch.update(FlightState(11., False))
        self.assertIsNone(latch.update(state(12., altitude=2600.)))
        self.assertIsNotNone(latch.update(state(30., heading=200.)))
        self.assertEqual(latch.spawn.heading_deg, 200.)

    def test_aircraft_change_is_a_respawn(self):
        latch = SpawnLatch()
        latch.update(state(10.))
        self.assertEqual(latch.update(state(10.5, aircraft="j_16")).aircraft_id, "j_16")

    def test_airfield_or_mid_flight_start_is_not_a_spawn_until_latched(self):
        latch = SpawnLatch()
        self.assertIsNone(latch.update(state(10., altitude=200., tas=60.)))
        self.assertIsNone(latch.spawn)
        self.assertIn("不像空中出生", latch.note)
        self.assertIsNone(latch.update(state(20., altitude=8000., tas=450., vy=50.)))
        manual = latch.latch(state(21., altitude=8000., tas=450.))
        self.assertTrue(manual.manual)
        self.assertEqual(latch.spawn.altitude_m, 8000.)

    def test_needs_a_heading(self):
        self.assertIsNone(SpawnLatch().update(state(1., heading=None)))

    def test_dead_reckoning_along_and_across_the_spawn_line(self):
        reckoning = DeadReckoning(SPAWN)
        for k in range(1, 101):
            reckoning.update(state(100.+.1*k, tas=300., heading=30.))
        self.assertAlmostEqual(reckoning.x, 3000., delta=1.)
        self.assertAlmostEqual(reckoning.y, 0., delta=1e-6)
        reckoning.update(state(110.1, tas=300., heading=120.))
        self.assertAlmostEqual(reckoning.y, 30., delta=.1)
        reckoning.update(state(130., tas=300.))   # a 20 s gap is not integrated
        self.assertAlmostEqual(reckoning.x, 3000., delta=1.)


class CueTests(unittest.TestCase):
    def test_direction_in_the_velocity_frame(self):
        self.assertEqual(tuple(round(v, 6) for v in velocity_frame_direction(30., 10., 30., 10.)), (0., 0., 1.))
        right = velocity_frame_direction(30., 0., 40., 0., gain=1.)
        self.assertAlmostEqual(right[0], math.sin(math.radians(10.)), places=6)
        up = velocity_frame_direction(30., 0., 30., 10., gain=1.)
        self.assertAlmostEqual(up[1], math.sin(math.radians(10.)), places=6)
        self.assertGreater(velocity_frame_direction(30., 0., 30., 5.)[1], 3*math.sin(math.radians(5.)))
        behind = velocity_frame_direction(0., 0., 180., 0.)
        self.assertLess(behind[2], -.99)
        # Across north: wanted 350 deg from 10 deg is a left turn.
        self.assertLess(velocity_frame_direction(10., 0., 350., 0.)[0], 0.)

    def test_phases_follow_the_plan(self):
        at = lambda t, **kw: cue(PLAN, SPAWN, state(100.+t, **kw))
        climb = at(20., vy=290.*math.sin(math.radians(15.)))
        self.assertEqual((climb.phase, climb.target_path_deg), ("爬升", 15.))
        self.assertTrue(climb.on_course)
        self.assertIn("25 s 后拉起到 24°", climb.lines[0])
        self.assertIn("66 km", climb.lines[1])
        pull = at(45.5)
        self.assertEqual((pull.phase, pull.target_path_deg, pull.urgent), ("拉起", 24., False))
        self.assertAlmostEqual(pull.path_error_deg, 24.)
        fire = at(47.)
        self.assertEqual((fire.phase, fire.urgent, fire.launch_range_m), ("发射", True, 65500.))
        self.assertEqual(at(60.).phase, "结束")
        self.assertFalse(at(60.).active)

    def test_yaw_error_sign(self):
        left_of_line = cue(PLAN, SPAWN, state(110., heading=20.))
        self.assertAlmostEqual(left_of_line.yaw_error_deg, 10.)
        self.assertGreater(left_of_line.direction[0], 0.)

    def test_without_a_plan_the_default_climb_is_shown(self):
        waiting = cue(None, SPAWN, state(101.))
        self.assertEqual((waiting.phase, waiting.target_path_deg), ("计算中", 15.))
        self.assertTrue(waiting.active)
        self.assertEqual(cue(None, None, state(1.)).phase, "等待出生")
        failed = cue(OpeningPlan(False, "打不到"), SPAWN, state(110.))
        self.assertEqual(failed.phase, "无方案")
        self.assertFalse(failed.active)

    def test_fire_along_the_climb_without_pull_up(self):
        plan = replace(PLAN, pitch_deg=None, pull_s=46.5)
        climb = cue(plan, SPAWN, state(130.))
        self.assertIn("16 s 后发射", climb.lines[0])
        self.assertEqual(cue(plan, SPAWN, state(147.)).target_path_deg, 15.)

    def test_deviation_from_the_predicted_climb(self):
        on = state(110., tas=320., vy=320.*math.sin(math.radians(15.)))
        self.assertFalse(deviates(PLAN, on, 10.))
        self.assertTrue(deviates(PLAN, replace(on, vertical_speed_mps=320.*math.sin(math.radians(20.))), 10.))
        self.assertTrue(deviates(PLAN, replace(on, tas_mps=360.), 10.))


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

    def wait_plan(self, t):
        for _ in range(100):
            snapshot = self.tick(t)
            if snapshot.opening.plan is not None:
                return snapshot
            time.sleep(.01)
        self.fail("no plan")

    def test_off_by_default(self):
        self.assertIsNone(self.tick(1.).opening)

    def test_early_replan_when_the_flown_climb_leaves_the_plan(self):
        self.enable()
        with patch("wt_overlay.app.plan_opening", side_effect=self.fake_plan):
            self.tick(0.)
            self.wait_plan(.1)
            for t in (1., 2., 3., 4., 5., 6.):   # level instead of 15 deg for more than DEVIATION_HOLD_S
                self.tick(t, vy=0., tas=290.+3.*t)
                time.sleep(.01)
            for _ in range(100):
                if len(self.requests) == 2:
                    break
                time.sleep(.01)
            self.assertEqual(len(self.requests), 2)
            self.assertEqual(self.requests[1].t_s, 5.)   # MIN_REPLAN_S after the first plan

    def test_spawn_plan_and_cue(self):
        self.enable()
        with patch("wt_overlay.app.plan_opening", side_effect=self.fake_plan):
            self.assertEqual(self.tick(1., valid=False).opening.phase, "等待出生")
            self.assertEqual(self.tick(2.).opening.phase, "计算中")
            snapshot = self.wait_plan(2.1)
            self.assertEqual(snapshot.opening.phase, "爬升")
            request = self.requests[0]
            self.assertEqual((request.aircraft_id, request.mass_kg, request.missile, request.hint),
                             ("f_16c_block_52_aesa", 11000., "us_aim_120c_5", None))
            self.assertEqual((request.spawn_altitude_m, request.spawn_tas_mps), (2500., 290.))
            self.assertEqual(snapshot.opening.launch_range_m, 65500.)

    def test_replans_on_schedule_with_a_hint_and_freezes_before_the_pull(self):
        self.enable()
        with patch("wt_overlay.app.plan_opening", side_effect=self.fake_plan):
            self.tick(0.)
            self.wait_plan(.1)
            for t in range(1, 15):   # on the predicted climb: no early re-plan
                self.tick(float(t), vy=(290.+3.*t)*math.sin(math.radians(15.)), tas=290.+3.*t)
            time.sleep(.05)
            self.tick(14.5, vy=332.*math.sin(math.radians(15.)), tas=333.5)
            self.assertEqual(len(self.requests), 1)
            self.tick(15.5, vy=336.*math.sin(math.radians(15.)), tas=336.5)
            for _ in range(100):
                if len(self.requests) == 2:
                    break
                time.sleep(.01)
            self.assertEqual(self.requests[1].hint, (15., 45.))
            self.assertAlmostEqual(self.requests[1].x_m, 4600., delta=300.)
            for t in range(16, 51):
                self.tick(float(t), vy=(290.+3.*t)*math.sin(math.radians(15.)), tas=290.+3.*t)
                time.sleep(.002)
            self.assertEqual([round(r.t_s) for r in self.requests], [0, 16, 31])   # nothing from 42 s on

    def test_respawn_resets_the_plan(self):
        self.enable()
        with patch("wt_overlay.app.plan_opening", side_effect=self.fake_plan):
            self.tick(0.)
            self.wait_plan(.1)
            self.tick(10., valid=False)
            self.assertEqual(self.tick(30., heading=210.).opening.phase, "计算中")
            self.wait_plan(30.1)
            self.assertEqual(len(self.requests), 2)

    def test_reasons_instead_of_a_plan(self):
        self.controller.submit({"action": "opening", "settings": asdict(OpeningSettings(True, ""))})
        self.tick(0.)
        snapshot = self.tick(.1)
        self.assertEqual(snapshot.opening.phase, "无方案")
        self.assertIn("选择导弹", snapshot.opening.lines[0])

    def test_manual_restart_latches_now(self):
        self.enable()
        with patch("wt_overlay.app.plan_opening", side_effect=self.fake_plan):
            self.tick(0., altitude=8000., tas=450.)
            self.assertEqual(self.tick(5., altitude=8000., tas=450.).opening.phase, "等待出生")
            self.controller.submit({"action": "opening_restart"})
            self.assertEqual(self.tick(6., altitude=8000., tas=450.).opening.phase, "计算中")

    def test_disabling_drops_the_spawn(self):
        self.enable()
        with patch("wt_overlay.app.plan_opening", side_effect=self.fake_plan):
            self.tick(0.)
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
