import math
import unittest
from dataclasses import replace

from wt_overlay.contracts import (FlightState, ModelInfo, PerformanceCondition,
                                  PerformancePoint, TurnRequest)
from wt_overlay.planning import scan_sep, sample_sep_grid, evaluate_turn


class Model:
    info = ModelInfo('test', 'test', limitations=('Not game validated.',))
    def evaluate(self, c):
        if c.tas_mps > 250:
            raise ValueError('Outside envelope')
        return PerformancePoint(c, True, sep_mps=10 - ((c.tas_mps - 180) / 10)**2,
                                notes=('No trim.',))


class PlanningTests(unittest.TestCase):
    def setUp(self):
        self.c = PerformanceCondition(1000, 150, 10000, load_factor=2, aoa_deg=8)

    def test_fixed_load_scan_preserves_current_and_condition(self):
        result = scan_sep(Model(), self.c, [150, 180, 210])
        self.assertEqual(result.best.condition.tas_mps, 180)
        self.assertEqual(result.current.condition.aoa_deg, 8)
        self.assertEqual(self.c.aoa_deg, 8)
        for point in result.sampled_points:
            self.assertEqual(point.condition, replace(self.c, tas_mps=point.condition.tas_mps, aoa_deg=None))
        self.assertIn('No trim.', result.notes)
        self.assertIn('Not game validated.', result.notes)

    def test_negative_best_is_retained(self):
        result = scan_sep(Model(), self.c, [100, 110])
        self.assertTrue(result.available)
        self.assertEqual(result.best.condition.tas_mps, 110)
        self.assertLess(result.best.sep_mps, 0)
        self.assertTrue(any('negative' in n for n in result.notes))
        self.assertTrue(any('boundary' in n for n in result.notes))

    def test_bad_samples_do_not_hide_good_samples(self):
        result = scan_sep(Model(), self.c, [math.nan, -1, 180, 300])
        self.assertEqual([p.valid for p in result.sampled_points], [False, False, True, False])
        self.assertEqual(result.best.sep_mps, 10)
        self.assertIn('Outside envelope', result.sampled_points[-1].reason)

    def test_unknown_and_invalid_conditions(self):
        self.assertFalse(scan_sep(Model(), self.c, []).available)
        self.assertFalse(scan_sep(Model(), replace(self.c, mass_kg=0)).available)
        self.assertFalse(scan_sep(Model(), replace(self.c, altitude_m=math.inf)).available)
        class Unknown(Model):
            def evaluate(self, c):
                return PerformancePoint(c, True, sep_mps=math.nan)
        self.assertFalse(scan_sep(Unknown(), self.c).available)

    def test_grid_reuses_speed_generator_and_keeps_configuration(self):
        points = list(sample_sep_grid(Model(), self.c, [0, 2000], iter([150, 180])))
        self.assertEqual(len(points), 4)
        self.assertEqual([p.condition.altitude_m for p in points], [0, 0, 2000, 2000])
        self.assertTrue(all(p.condition.load_factor == 2 and p.condition.aoa_deg is None for p in points))

    def test_turn_is_never_fabricated(self):
        state = FlightState(1, True, tas_mps=200)
        for angle in [30, 45, 90, 120]:
            result = evaluate_turn(TurnRequest(angle), state)
            self.assertFalse(result.available)
            self.assertIsNone(result.duration_s)
            self.assertIsNone(result.energy_change_m)
        result = evaluate_turn(TurnRequest(90, objective='minimum_energy_loss'), state)
        self.assertIn('time bound', result.reason)
        result = evaluate_turn(TurnRequest(90, angle_basis='heading'), state)
        self.assertIn('Angle basis', result.reason)
        result = evaluate_turn(TurnRequest(90, endpoint='anything'), state)
        self.assertIn('Endpoint', result.reason)
        result = evaluate_turn(TurnRequest(90, time_limit_s=math.nan), state)
        self.assertIn('finite', result.reason)


if __name__ == '__main__':
    unittest.main()
