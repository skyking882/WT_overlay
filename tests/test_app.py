from dataclasses import replace
from pathlib import Path
import math
import unittest
from unittest.mock import patch

from wt_overlay.app import OverlayController
from wt_overlay.contracts import FlightState, G, PerformanceCondition
from wt_overlay.fm import load_model
from wt_overlay.__main__ import snapshot_json


SAMPLE = str(Path(__file__).resolve().parents[1] / "data/fm/su_27sm.blkx")


class FakeClient:
    def __init__(self):
        self.state = FlightState(0, True, altitude_m=5000, tas_mps=300,
                                 aircraft_id="su_27sm")
        self.calls = 0

    def poll(self, time_s=None):
        self.calls += 1
        return replace(self.state, time_s=time_s)


class AppIntegrationTests(unittest.TestCase):
    def controller(self, **kwargs):
        self.client = FakeClient()
        return OverlayController(client=self.client, **kwargs)

    def test_demo_never_polls_game_and_has_measured_energy(self):
        controller = self.controller(mode="demo")
        start = controller._started_at
        for dt in (0, .1, .3):
            snapshot = controller.tick(start+dt)
        self.assertEqual(self.client.calls, 0)
        self.assertEqual(snapshot.state.source, "demo")
        self.assertTrue(snapshot.energy.ready)
        self.assertAlmostEqual(snapshot.energy.sep_mps,
                               snapshot.energy.climb_mps+snapshot.energy.kinetic_sep_mps)
        self.assertIsNone(snapshot.advice)

    def test_live_failure_clears_energy_and_prediction_without_demo(self):
        controller = self.controller(model_path=SAMPLE, mass_kg=23000)
        start = controller._started_at
        controller.tick(start)
        controller.tick(start+.1)
        self.assertTrue(controller.tick(start+.3).energy.ready)
        self.client.state = FlightState(0, False, notes=("connection lost",))
        snapshot = controller.tick(start+.4)
        self.assertEqual(snapshot.mode, "live")
        self.assertFalse(snapshot.state.valid)
        self.assertIsNone(snapshot.energy.sep_mps)
        self.assertIsNone(snapshot.advice)
        self.assertIsNone(snapshot_json(snapshot)["reference_sep_mps"])

    def test_aircraft_mismatch_and_unknown_id_do_not_apply_model(self):
        controller = self.controller(model_path=SAMPLE, mass_kg=23000)
        start = controller._started_at
        self.assertTrue(controller.tick(start).advice.available)
        for i, aircraft in enumerate(("f_15c", None), 1):
            self.client.state = replace(self.client.state, aircraft_id=aircraft)
            snapshot = controller.tick(start+i*.1)
            self.assertFalse(snapshot.advice.available)
            self.assertIsNone(snapshot.advice.current)
            self.assertIsNone(snapshot.advice.best)

    def test_prediction_needs_total_mass_and_mass_change_invalidates_cache(self):
        controller = self.controller(model_path=SAMPLE)
        start = controller._started_at
        self.assertFalse(controller.tick(start).advice.available)
        controller.submit({"action": "mass", "kg": 23000})
        first = controller.tick(start+.1)
        self.assertTrue(first.advice.available)
        controller.submit({"action": "mass", "kg": 25000})
        second = controller.tick(start+.2)
        self.assertEqual(second.advice.current.condition.mass_kg, 25000)
        self.assertNotEqual(first.advice.current.sep_mps, second.advice.current.sep_mps)
        self.assertEqual(second.advice.current.condition.aoa_deg, None)
        self.assertEqual(second.advice.current.condition.load_factor, 1)

    def test_mode_and_afterburner_changes_recompute(self):
        controller = self.controller(mode="demo", model_path=SAMPLE, mass_kg=23000)
        start = controller._started_at
        controller.tick(start)
        controller.tick(start+.1)
        first = controller.tick(start+.3)
        self.assertTrue(first.energy.ready)
        controller.submit({"action": "mode", "value": "live"})
        controller.submit({"action": "afterburner", "enabled": False})
        second = controller.tick(start+.4)
        self.assertEqual(second.mode, "live")
        self.assertFalse(second.energy.ready)
        self.assertFalse(second.advice.current.condition.afterburner)
        self.assertFalse(second.afterburner)

    def test_failed_model_switch_does_not_retain_old_prediction(self):
        controller = self.controller(model_path=SAMPLE, mass_kg=23000)
        start = controller._started_at
        self.assertTrue(controller.tick(start).advice.available)
        controller.submit({"action": "model", "path": SAMPLE+".missing"})
        snapshot = controller.tick(start+.1)
        self.assertIsNone(snapshot.advice)
        self.assertIn("未加载", snapshot.status)
        self.assertEqual(snapshot.model_name, "未加载 FM")

    def test_stale_snapshot_masks_values(self):
        controller = self.controller(model_path=SAMPLE, mass_kg=23000)
        controller.tick()
        with patch("wt_overlay.app.time.monotonic", return_value=controller._published_at+3):
            snapshot = controller.get_snapshot()
        self.assertFalse(snapshot.state.valid)
        self.assertIsNone(snapshot.advice)
        self.assertIsNone(snapshot.energy.sep_mps)
        self.assertIsNone(snapshot_json(snapshot)["tas_mps"])

    def test_command_validation_rejects_bad_mass_and_state(self):
        controller = self.controller()
        for command in ({"action": "mass", "kg": math.nan},
                        {"action": "mass", "kg": True},
                        {"action": "mass", "kg": -1},
                        {"action": "afterburner", "enabled": 1},
                        {"action": "mode", "value": "automatic-demo"}):
            with self.subTest(command=command), self.assertRaises(ValueError):
                controller.submit(command)

    def test_load_factor_means_lift_over_weight_at_any_path_angle(self):
        model = load_model(SAMPLE)
        condition = PerformanceCondition(5000, 300, 23000, load_factor=1.2,
                                         flight_path_deg=30)
        point = model.evaluate(condition)
        self.assertTrue(point.valid, point.reason)
        self.assertAlmostEqual(point.lift_n/(condition.mass_kg*G), 1.2)


if __name__ == "__main__":
    unittest.main()
