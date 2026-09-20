from dataclasses import replace
import unittest

from wt_overlay.contracts import (ClimbGuidance, EnergyMetrics, FlightState, OverlaySnapshot,
                                  PerformanceCondition, PerformancePoint, SEPAdvice)
from wt_overlay.hud import INDICATORS, contents, details
from wt_overlay.telemetry import parse_telemetry


def sample_snapshot():
    point = PerformancePoint(PerformanceCondition(5000, 300, 23000), True, sep_mps=42)
    return OverlaySnapshot("live", "8111 已连接", FlightState(1, True, altitude_m=5000, tas_mps=300),
                           EnergyMetrics(1, 9600, 35, 10, climb_mps=25, ready=True),
                           SEPAdvice(True, point, point))


class HudDataTests(unittest.TestCase):
    def test_climb_shows_indicated_speed_and_both_flight_path_angles(self):
        guidance = ClimbGuidance(True, "爬升", 300, 12, 2, 3000,
                                 actual_path_deg=10, target_ias_mps=210)
        snapshot = replace(sample_snapshot(), climb_enabled=True, climb=guidance)
        rows = {row.label: row.value for row in contents(snapshot)["climb"].rows}
        self.assertEqual(rows["目标 IAS"], "756")
        self.assertNotIn("目标 TAS", rows)
        self.assertEqual(rows["当前航迹角"], "+10.0")
        self.assertEqual(rows["目标航迹角"], "+12.0")
        missing = replace(snapshot, climb=replace(guidance, target_ias_mps=None))
        self.assertEqual(contents(missing)["climb"].rows[1].value, "—")
        stale = replace(snapshot, state=replace(snapshot.state, valid=False))
        self.assertTrue(all(row.value == "—" for row in contents(stale)["climb"].rows[1:]))

    def test_invalid_state_clears_even_previous_energy_and_model_values(self):
        snapshot = sample_snapshot()
        stale = replace(snapshot, state=replace(snapshot.state, valid=False))
        for group in contents(stale).values():
            self.assertTrue(all(row.value == "—" for row in group.rows))

    def test_every_detachable_group_identifies_synthetic_data(self):
        snapshot = sample_snapshot()
        for group in contents(replace(snapshot, mode="demo")).values():
            self.assertTrue(group.demo)
        for group in contents(replace(snapshot, state=replace(snapshot.state, source="demo"))).values():
            self.assertTrue(group.demo)

    def test_partial_data_does_not_invent_values_or_display_nonfinite_numbers(self):
        snapshot = sample_snapshot()
        snapshot = replace(snapshot, energy=replace(snapshot.energy, sep_mps=float("nan")))
        groups = contents(snapshot)
        self.assertEqual(groups["flight"].rows[1].value, "—")
        self.assertEqual(groups["energy"].rows[0].value, "—")
        self.assertIn("未配平", details(snapshot))

    def test_8111_maneuver_data_reaches_hud_without_becoming_model_load(self):
        state = parse_telemetry(
            {"valid": True, "H, m": 5000, "TAS, km/h": 1080,
             "Ny": 5.3, "AoA, deg": 12.4, "AoS, deg": -0.6},
            {"valid": True, "aviahorizon_pitch": 8, "aviahorizon_roll": -45}, 1)
        rows = {row.key: row for group in contents(OverlaySnapshot("live", "ready", state)).values()
                for row in group.rows}
        self.assertEqual(rows["g"].value, "+5.3")
        self.assertEqual(rows["aoa"].value, "+12.4")
        self.assertEqual(rows["aos"].value, "-0.6")
        self.assertEqual(rows["pitch"].value, "+8.0")
        self.assertEqual(rows["roll"].value, "-45.0")
        self.assertIsNone(state.load_factor)

    def test_indicator_filter_is_exact_and_current_tas_is_not_a_hud_metric(self):
        groups = contents(sample_snapshot(), {"aoa", "g"})
        keys = [row.key for group in groups.values() for row in group.rows]
        self.assertEqual(set(keys), {"aoa", "g"})
        self.assertNotIn("tas", {item.key for item in INDICATORS})
        self.assertTrue(all(not group.rows for group in contents(sample_snapshot(), set()).values()))


if __name__ == "__main__":
    unittest.main()
