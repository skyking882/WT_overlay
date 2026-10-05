"""Helpers of scripts/build_lowalt_table.py: multipath strength, RWR warning, range bands."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from build_lowalt_table import bands, elevation_deg, ladder, strength, warning_s  # noqa: E402


def sample(t, missile, target=(0., 2., 0.), state="track"):
    return dict(time_s=t, position_m=missile, target_position_m=target, seeker_state=state)


class LowAltTests(unittest.TestCase):
    def test_strength_is_linear_and_held_at_the_ends(self):
        nodes = [(0., 1.), (60., 0.)]
        self.assertEqual(strength(nodes, -5.), 1.)
        self.assertAlmostEqual(strength(nodes, 30.), .5)
        self.assertEqual(strength(nodes, 300.), 0.)

    def test_elevation_is_above_the_target_horizon(self):
        self.assertAlmostEqual(elevation_deg((1000., 1002., 0.), (0., 2., 0.)), 45.)
        self.assertAlmostEqual(elevation_deg((0., 2., 500.), (0., 2., 0.)), 0.)

    def test_warning_starts_when_the_tracking_seeker_enters_the_receiver_cone(self):
        samples = [sample(0., (0., 8000., 4000.), state="search"),    # Not emitting at the target yet.
                   sample(1., (0., 5000., 4000.)),                    # 51 deg up: outside a 45 deg cone.
                   sample(3., (0., 1000., 2000.)),                    # 27 deg up: inside.
                   sample(5., (0., 10., 10.))]
        self.assertEqual(warning_s(samples, [30., 45., 60., 90.]), {"30": 2., "45": 2., "60": 4., "90": 4.})
        steep = [sample(0., (0., 8000., 1000.)), sample(4., (0., 100., 10.))]
        self.assertEqual(warning_s(steep, [45.]), {"45": None})  # Stays in the blind zone.

    def test_ladder_and_bands_allow_holes(self):
        self.assertEqual(ladder("2:3:0.5,5"), [2., 2.5, 3., 5.])
        cells = [dict(range_m=r*1000, hit=h) for r, h in ((2, True), (3, True), (4, False), (5, True), (6, False))]
        self.assertEqual(bands(cells), "2-3, 5")
        self.assertEqual(bands([dict(range_m=2000., hit=False)]), "none")


if __name__ == "__main__":
    unittest.main()
