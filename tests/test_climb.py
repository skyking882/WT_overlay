from dataclasses import replace
import math
from pathlib import Path
from threading import Event
import unittest
from unittest.mock import patch

from wt_overlay.app import OverlayController
from wt_overlay.climb import (ClimbDirector, PlanningCancelled, PlanningUnavailable,
                              build_climb_plan, maximum_at_energy, validate_request)
from wt_overlay.contracts import (G, ClimbRequest, EnergyMetrics, FlightState, ModelInfo,
                                  PerformanceCondition, PerformancePoint)
from wt_overlay.fm import load_model


class AnalyticModel:
    info = ModelInfo("test", "test")

    def evaluate(self, c):
        valid = 0 <= c.altitude_m <= 20000 and 80 <= c.tas_mps <= 650
        return PerformancePoint(c, valid, sep_mps=150-.001*(c.tas_mps-300)**2-.004*c.altitude_m)


class ClimbTests(unittest.TestCase):
    def setUp(self):
        self.model = AnalyticModel()
        self.base = PerformanceCondition(0, 150, 23000)

    def test_optimizer_obeys_energy_constraint_and_differs_from_fixed_height_optimum(self):
        energy = 12000
        point = maximum_at_energy(self.model, self.base, energy, 8000)
        optimum = .6 / (.002-.004/G)
        self.assertAlmostEqual(point.tas_mps, optimum, delta=.3)
        self.assertGreater(point.tas_mps, 350)  # Fixed-height maximum is exactly 300 m/s.
        self.assertAlmostEqual(point.altitude_m+point.tas_mps**2/(2*G), energy)
        # Low energy restricts the maximum to the sea-level boundary.
        low = maximum_at_energy(self.model, self.base, 2000, 8000)
        self.assertAlmostEqual(low.altitude_m, 0, delta=.001)

    def test_cancel_invalid_targets_and_unavailable_path(self):
        for request in (ClimbRequest(math.nan), ClimbRequest(True), ClimbRequest(8000, math.inf),
                        ClimbRequest(25000), ClimbRequest(8000, 30)):
            with self.subTest(request=request), self.assertRaises(ValueError):
                validate_request(request)
        event = Event()
        event.set()
        with self.assertRaises(PlanningCancelled):
            build_climb_plan(self.model, self.base, ClimbRequest(), event)
        with patch.object(self.model, "evaluate", side_effect=lambda c: PerformancePoint(c, False)):
            with self.assertRaises(PlanningUnavailable):
                build_climb_plan(self.model, self.base, ClimbRequest())

    def test_closed_loop_acceleration_climb_capture_and_terminal_speed(self):
        plan = build_climb_plan(self.model, self.base, ClimbRequest(8000, 300))
        director = ClimbDirector()
        h, speed, gamma, last_command = 0., 150., 0., 0.
        phases = set()
        for i in range(4000):
            t = i*.1
            power = self.model.evaluate(replace(self.base, altitude_m=h, tas_mps=speed)).sep_mps
            state = FlightState(t, True, h, speed, vertical_speed_mps=speed*math.sin(math.radians(gamma)), roll_deg=0)
            cue = director.update(plan, state, EnergyMetrics(t, sep_mps=power, ready=True), power)
            self.assertTrue(cue.available, (h, speed, cue))
            phases.add(cue.phase)
            if cue.phase == "到达":
                break
            if i == 0:
                self.assertEqual(cue.phase, "加速")
                self.assertGreater(cue.target_tas_mps, speed+100)
            self.assertLessEqual(abs(cue.target_path_deg-last_command), .30001)
            self.assertTrue(-5 <= cue.target_path_deg <= 45)
            last_command = cue.target_path_deg
            # Pilot tracks the cue with a one-second lag; integrate point-mass physics.
            gamma += .1*(cue.target_path_deg-gamma)
            vertical = speed*math.sin(math.radians(gamma))
            h += vertical*.1
            speed += G*(power-vertical)/speed*.1
        self.assertEqual(phases, {"加速", "爬升", "收平", "到达"})
        self.assertLessEqual(abs(h-8000), 25)
        self.assertGreaterEqual(speed, 300)
        self.assertLess(abs(gamma), 1.5)
        self.assertIsNone(cue.path_error_deg)

    def test_cue_uses_flight_path_not_pitch_and_withdraws_on_missing_or_banked_state(self):
        plan = build_climb_plan(self.model, self.base, ClimbRequest(8000))
        def run(speed, pitch=0):
            director = ClimbDirector()
            for t in (0, 1):
                state = FlightState(t, True, 3000, speed, vertical_speed_mps=0, pitch_deg=pitch, roll_deg=0)
                cue = director.update(plan, state, EnergyMetrics(t), 100)
            return cue
        self.assertEqual(run(390), run(390, 20))
        self.assertGreater(run(390).path_error_deg, run(260).path_error_deg)
        director = ClimbDirector()
        for state in (FlightState(0, False), FlightState(0, True, 3000, 300),
                      FlightState(0, True, 3000, 300, vertical_speed_mps=0, roll_deg=45)):
            cue = director.update(plan, state, EnergyMetrics(0), 100)
            self.assertFalse(cue.available)
            self.assertIsNone(cue.path_error_deg)

    def test_su27_model_produces_feasible_ridge_and_terminal_capture(self):
        model = load_model(str(Path(__file__).resolve().parents[1]/"data/fm/su_27sm.blkx"))
        plan = build_climb_plan(model, PerformanceCondition(5000, 300, 23000), ClimbRequest(8000))
        for point in plan.points:
            self.assertAlmostEqual(point.altitude_m+point.tas_mps**2/(2*G), point.energy_m)
            self.assertGreater(point.sep_mps, 0)
        speed, slope = plan.reference(plan.end_energy_m+5000)
        self.assertAlmostEqual(plan.end_energy_m+5000-speed**2/(2*G), 8000)
        self.assertGreater(slope, 0)


class ClimbControllerTests(unittest.TestCase):
    def setUp(self):
        self.state = FlightState(0, True, 3000, 300, mass_kg=23000,
                                 vertical_speed_mps=0, roll_deg=0, aircraft_id="test")
        self.controller = OverlayController(client=self)
        self.controller.model = AnalyticModel()
        self.addCleanup(self.controller.stop)

    def poll(self, time_s=None):
        return replace(self.state, time_s=time_s)

    def test_default_off_async_plan_and_immediate_cancel_without_obsolete_publication(self):
        c = self.controller
        self.assertFalse(c.tick(0).climb_enabled)
        self.assertIsNone(c._planner)
        entered, exited = Event(), Event()
        def slow(model, base, request, cancel):
            entered.set()
            cancel.wait(2)
            exited.set()
            raise PlanningCancelled
        with patch("wt_overlay.app.build_climb_plan", side_effect=slow):
            c.submit({"action": "climb_enabled", "enabled": True})
            self.assertEqual(c.tick(.1).climb.phase, "计算")
            self.assertTrue(entered.wait(1))
            c.submit({"action": "climb_enabled", "enabled": False})
            snapshot = c.tick(.2)
            self.assertFalse(snapshot.climb_enabled)
            self.assertIsNone(snapshot.climb)
            self.assertTrue(exited.wait(1))
            self.assertIsNone(c._plan)
            self.assertIsNone(c._plan_future)
        c.submit({"action": "climb_target", "altitude_m": 9000, "minimum_tas_mps": 320})
        c.submit({"action": "climb_enabled", "enabled": True})
        c.tick(.3)
        c._plan_future.result(timeout=2)
        snapshot = c.tick(.4)
        self.assertTrue(snapshot.climb.available)
        self.assertEqual(c._plan.request, ClimbRequest(9000, 320))

    def test_disconnect_mismatch_mass_change_and_stale_data_withdraw_cue(self):
        c = self.controller
        c.submit({"action": "climb_enabled", "enabled": True})
        c.tick(0)
        c._plan_future.result(timeout=2)
        self.assertTrue(c.tick(.1).climb.available)
        with patch("wt_overlay.app.time.monotonic", return_value=c._published_at+3):
            self.assertIsNone(c.get_snapshot().climb.path_error_deg)
        self.state = replace(self.state, mass_kg=24000)
        self.assertEqual(c.tick(.2).climb.phase, "计算")
        self.state = replace(self.state, aircraft_id="other")
        self.assertEqual(c.tick(.3).climb.phase, "核对机型")
        self.assertIsNone(c._plan_future)
        self.state = replace(self.state, valid=False)
        self.assertEqual(c.tick(.4).climb.phase, "等待数据")

    def test_target_validation_and_model_failure_remove_old_guidance(self):
        c = self.controller
        for command in ({"action": "climb_enabled", "enabled": 1},
                        {"action": "climb_target", "altitude_m": None},
                        {"action": "climb_target", "altitude_m": 8000, "minimum_tas_mps": math.nan}):
            with self.assertRaises(ValueError):
                c.submit(command)
        c.submit({"action": "climb_enabled", "enabled": True})
        c.tick(0)
        c._plan_future.result(timeout=2)
        self.assertTrue(c.tick(.1).climb.available)
        c.submit({"action": "model", "path": "/missing/fm.blkx"})
        self.assertEqual(c.tick(.2).climb.phase, "选择 FM")
        self.assertIsNone(c._plan)


if __name__ == "__main__":
    unittest.main()
