"""Precomputed L/q, D/q tables vs live polar assembly."""
import math
import unittest

from wt_overlay.fm import atmosphere, load_aircraft
from wt_overlay.fm.aero_table import AeroForceTable, QUERY_AOA_MAX, QUERY_AOA_MIN, polar_forces_over_q
from wt_overlay.turn import ManeuverModel


def _q(altitude, speed):
    rho, sound = atmosphere(altitude)
    return .5*rho*speed*speed, speed/sound


class AeroTableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fm = load_aircraft("su_27sm")
        cls.table = AeroForceTable.from_aircraft(cls.fm)
        cls.parts = cls.fm.components_at_sweep(0.)

    def test_knots_match_direct_polar_assembly(self):
        aoas = (-40., -22., 0., 8., 16., 21.6, 28., 35., 40., 50.)
        for mach in self.table.mach_knots:
            if abs(mach*5-round(mach*5)) > 1e-9:
                continue
            for aoa in aoas:
                with self.subTest(mach=mach, aoa=aoa):
                    got = self.table.lookup(mach, aoa)
                    expected = polar_forces_over_q(self.parts, mach, aoa)
                    self.assertAlmostEqual(got[0], expected[0], places=9)
                    self.assertAlmostEqual(got[1], expected[1], places=9)

    def test_interpolation_stays_close_to_direct_evaluation(self):
        max_lift = max_drag = 0.
        samples = 0
        for mach in (.35, .55, .725, .785, .905, 1.025, 1.175, 1.55):
            for aoa in (-30.25, -8.25, 4.25, 12.25, 21.6, 27.25, 33.25, 47.25):
                got = self.table.lookup(mach, aoa)
                expected = polar_forces_over_q(self.parts, mach, aoa)
                max_lift = max(max_lift, abs(got[0]-expected[0]))
                max_drag = max(max_drag, abs(got[1]-expected[1]))
                samples += 1
        self.assertGreater(samples, 20)
        # ~0.002 m² observed; q·Δ(L/q) stays well under 1e-3 g.
        self.assertLess(max_lift, .05)
        self.assertLess(max_drag, .05)

    def test_postcritical_peak_is_not_flattened(self):
        curve = self.table.lift_curve(.8)
        positive = [(a, lift) for a, lift in curve if a >= 0]
        peak_aoa, peak = max(positive, key=lambda item: item[1])
        end = positive[-1][1]
        self.assertGreater(peak_aoa, 15)
        self.assertLess(peak_aoa, 45)
        self.assertGreater(peak, end)
        diffs = [positive[i+1][1]-positive[i][1] for i in range(len(positive)-1)]
        sign_changes = sum(a*b < 0 for a, b in zip(diffs, diffs[1:]))
        self.assertGreaterEqual(sign_changes, 1)

    def test_query_bounds(self):
        self.table.lookup(0., QUERY_AOA_MIN)
        self.table.lookup(self.table.mach_knots[-1], QUERY_AOA_MAX)
        with self.assertRaises(ValueError):
            self.table.lookup(-.01, 0.)
        with self.assertRaises(ValueError):
            self.table.lookup(self.table.mach_knots[-1]+.01, 0.)
        with self.assertRaises(ValueError):
            self.table.lookup(.8, QUERY_AOA_MIN-.01)
        with self.assertRaises(ValueError):
            self.table.lookup(.8, QUERY_AOA_MAX+.01)

    def test_other_layouts_and_sweep_build(self):
        for identity in ("j_16", "j_11b", "f_15c_golden_eagle", "f_16c_block_50"):
            fm = load_aircraft(identity)
            table = AeroForceTable.from_aircraft(fm)
            parts = fm.components_at_sweep(0.)
            lift_q, drag_q = table.lookup(.8, 10.)
            expected = polar_forces_over_q(parts, .8, 10.)
            self.assertAlmostEqual(lift_q, expected[0], places=9)
            self.assertAlmostEqual(drag_q, expected[1], places=9)
        swing = load_aircraft("f_14b")
        low = swing.wings[0][0]
        high = swing.wings[-1][0]
        AeroForceTable.from_aircraft(swing, low)
        AeroForceTable.from_aircraft(swing, high)
        mid = AeroForceTable.from_aircraft(swing, .5)
        self.assertGreater(len(mid.mach_knots), 50)

    def test_maneuver_forces_track_table_and_static_sep_path_unchanged(self):
        model = ManeuverModel(self.fm, 23000)
        q, mach = _q(5000, 250)
        lift_q, drag_q = self.table.lookup(mach, 8.)
        thrust, drag, lift, alpha = model.forces_at_aoa(5000, 250, 8.)
        self.assertAlmostEqual(math.degrees(alpha), 8.)
        self.assertAlmostEqual(lift, lift_q*q, places=6)
        self.assertAlmostEqual(drag, drag_q*q, places=6)
        self.assertGreater(thrust, 0)
        from wt_overlay.contracts import PerformanceCondition
        static = self.fm.evaluate(PerformanceCondition(5000, 250, 23000, aoa_deg=8.))
        self.assertTrue(static.valid)
        self.assertAlmostEqual(static.thrust_n, thrust)
        self.assertAlmostEqual(static.lift_n, lift, delta=max(20., abs(static.lift_n)*1e-3))
        self.assertAlmostEqual(static.drag_n, drag, delta=max(20., abs(static.drag_n)*1e-3))


if __name__ == "__main__":
    unittest.main()
