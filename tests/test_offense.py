"""Kill-rose table lookup: grid loading, bilinear interpolation, clamping."""
import json
import math
from pathlib import Path
import tempfile
import unittest

from wt_overlay.offense import RoseLibrary, available, course_for_blip_direction


def write_table(folder, alt, kmh, chaff, value, miss_at=None):
    cells = []
    for r in (5000., 10000.):
        for c in (0., 90., 180.):
            for g in (-6., 0., 6.):
                reaction = None if miss_at == (r, c) else value+r/1000+c/90+(100 if g else 0)
                cells.append(dict(range_m=r, course_deg=c, turn_g=g, reaction_s=reaction))
    meta = dict(missile="m", evader="e", launch_altitude_m=alt, launch_speed_kmh=kmh, chaff_rcs_ratio=chaff)
    path = Path(folder)/f"m__e__{int(alt)}m_{int(kmh)}kmh__chaff{chaff:g}.json"
    path.write_text(json.dumps(dict(meta=meta, cells=cells)))


class RoseLibraryTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        for alt, base in ((4000., 0.), (8000., 10.)):
            for kmh, extra in ((900., 0.), (1300., 2.)):
                write_table(self.dir.name, alt, kmh, 1., base+extra, miss_at=(10000., 180.) if alt == 8000. else None)
        write_table(self.dir.name, 4000., 900., 0., 50.)  # Other chaff assumption is kept apart.

    def tearDown(self):
        self.dir.cleanup()

    def test_bilinear_in_altitude_and_tas_using_straight_flight_cells(self):
        lib = RoseLibrary("m", "e", 1., Path(self.dir.name))
        rose = lib.rose(6000., 1100./3.6)
        # Mid altitude and mid speed: average of 0, 2, 10, 12 plus the cell term 5 + 0.
        self.assertAlmostEqual(rose.at(5000., 0.), 6.+5.)
        self.assertFalse(rose.altitude_clamped or rose.tas_clamped)

    def test_clamps_outside_grid_and_flags_it(self):
        rose = RoseLibrary("m", "e", 1., Path(self.dir.name)).rose(15000., 2000./3.6)
        self.assertAlmostEqual(rose.at(5000., 90.), 12.+5.+1.)
        self.assertTrue(rose.altitude_clamped and rose.tas_clamped)

    def test_slightly_outside_grid_is_clamped_without_warning(self):
        rose = RoseLibrary("m", "e", 1., Path(self.dir.name)).rose(3800., 880./3.6)
        self.assertFalse(rose.altitude_clamped or rose.tas_clamped)
        self.assertAlmostEqual(rose.at(5000., 0.), 5.)

    def test_unreachable_corner_stays_infinite(self):
        rose = RoseLibrary("m", "e", 1., Path(self.dir.name)).rose(6000., 900./3.6)
        self.assertEqual(rose.at(10000., 180.), math.inf)
        self.assertTrue(math.isfinite(rose.at(10000., 90.)))

    def test_chaff_assumptions_and_availability(self):
        self.assertAlmostEqual(RoseLibrary("m", "e", 0., Path(self.dir.name)).rose(4000., 250.).at(5000., 0.), 55.)
        self.assertEqual(available(Path(self.dir.name)), [("m", "e", 0.), ("m", "e", 1.)])
        with self.assertRaises(FileNotFoundError):
            RoseLibrary("x", "e", 1., Path(self.dir.name))

    def test_incomplete_grid_falls_back_to_nearest_table(self):
        Path(self.dir.name, "m__e__8000m_1300kmh__chaff1.json").unlink()
        lib = RoseLibrary("m", "e", 1., Path(self.dir.name))
        rose = lib.rose(7900., 1100./3.6)  # Nearest present table: 8000 m, 900 km/h.
        self.assertTrue(rose.nearest_only)
        self.assertAlmostEqual(rose.at(5000., 0.), 10.+5.)

    def test_blip_direction_maps_to_target_course(self):
        self.assertEqual(course_for_blip_direction(0.), 0.)
        self.assertEqual(course_for_blip_direction(90.), 90.)
        self.assertEqual(course_for_blip_direction(-90.), 90.)
        self.assertEqual(course_for_blip_direction(180.), 180.)
        self.assertEqual(course_for_blip_direction(-150.), 150.)


if __name__ == "__main__":
    unittest.main()


class EnvelopeLibraryTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        for alt, base in ((4000., 0.), (8000., 10000.)):
            for kmh in (900., 1300.):
                rows = [dict(azimuth_deg=az, alt_diff_m=0., rmax_hot=30000.+base-az*100, rmax_cold=12000.,
                             r3_hot=10000., rne_hot=None if alt == 8000. and az == 60. else 6000.)
                        for az in (0., 30., 60.)]
                meta = dict(missile="m", evader="e", launch_altitude_m=alt, launch_speed_kmh=kmh, chaff_rcs_ratio=1.)
                Path(self.dir.name, f"m__e__{int(alt)}m_{int(kmh)}kmh__chaff1.json").write_text(
                    json.dumps(dict(meta=meta, rows=rows)))

    def tearDown(self):
        self.dir.cleanup()

    def test_interpolates_and_mirrors_azimuth(self):
        from wt_overlay.offense import EnvelopeLibrary
        env = EnvelopeLibrary("m", "e", 1., Path(self.dir.name)).envelope(6000., 1100./3.6)
        line = env.line("rmax_hot")
        self.assertEqual([az for az, _ in line], [-60., -30., 0., 30., 60.])
        self.assertAlmostEqual(dict(line)[0.], 35000.)
        self.assertAlmostEqual(dict(line)[-30.], dict(line)[30.])
        self.assertIsNone(dict(env.line("rne_hot"))[60.])  # A missing corner removes the point.
