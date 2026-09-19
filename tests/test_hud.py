from dataclasses import replace
import unittest

from wt_overlay.contracts import (EnergyMetrics, FlightState, OverlaySnapshot,
                                  PerformanceCondition, PerformancePoint, SEPAdvice)
from wt_overlay.hud import contents


def sample_snapshot():
    point = PerformancePoint(PerformanceCondition(5000, 300, 23000), True, sep_mps=42)
    return OverlaySnapshot("live", "8111 已连接", FlightState(1, True, altitude_m=5000, tas_mps=300),
                           EnergyMetrics(1, 9600, 35, 10, climb_mps=25, ready=True),
                           SEPAdvice(True, point, point))


class HudDataTests(unittest.TestCase):
    def test_invalid_state_clears_even_previous_energy_and_model_values(self):
        snapshot = sample_snapshot()
        stale = replace(snapshot, state=replace(snapshot.state, valid=False))
        for group in contents(stale).values():
            self.assertTrue(all(row.value == "—" for row in group.rows))
            self.assertIn("无有效数据", group.title)

    def test_every_detachable_group_identifies_synthetic_data(self):
        snapshot = sample_snapshot()
        for group in contents(replace(snapshot, mode="demo")).values():
            self.assertIn("合成演示", group.title)
        for group in contents(replace(snapshot, state=replace(snapshot.state, source="demo"))).values():
            self.assertIn("合成演示", group.title)

    def test_partial_data_does_not_invent_values_or_display_nonfinite_numbers(self):
        snapshot = sample_snapshot()
        snapshot = replace(snapshot, energy=replace(snapshot.energy, sep_mps=float("nan")))
        groups = contents(snapshot)
        self.assertEqual(groups["flight"].rows[1].value, "—")
        self.assertEqual(groups["energy"].rows[0].value, "—")
        self.assertIn("未配平", groups["reference"].footer)
        self.assertIn("非全程最优", groups["reference"].footer)


if __name__ == "__main__":
    unittest.main()
