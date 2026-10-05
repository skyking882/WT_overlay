"""Low-target window lookup and the window rule of scripts/build_lowalt_window.py."""
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from build_lowalt_window import window  # noqa: E402

from wt_overlay.lowalt import LowAltLibrary, available  # noqa: E402


def write_table(folder, alt, kmh, shift, missing=None):
    """Windows (near, far) km = (2+shift+pitch/30, 10+shift+pitch/15) at azimuth 0, 1 km shorter at 60."""
    rows = []
    for pitch in (0., 30.):
        for az in (0., 60.):
            near, far = 2+shift+pitch/30, 10+shift+pitch/15-(1 if az else 0)
            worst = None if missing == (pitch, az) else [near*1000, far*1000]
            rows.append(dict(pitch_deg=pitch, azimuth_deg=az, worst=worst, reference=[near*1000, (far+5)*1000]))
    meta = dict(missile="m", launch_altitude_m=alt, launch_speed_kmh=kmh, reference_height_m=20.)
    Path(folder, f"m__{int(alt)}m_{int(kmh)}kmh.json").write_text(json.dumps(dict(meta=meta, rows=rows)))


class LowAltTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        for alt, base in ((4000., 0.), (8000., 4.)):
            for kmh, extra in ((900., 0.), (1300., 2.)):
                write_table(self.dir.name, alt, kmh, base+extra, missing=(30., 60.) if alt == 8000. else None)

    def tearDown(self):
        self.dir.cleanup()

    def test_interpolates_altitude_tas_and_pitch(self):
        w = LowAltLibrary("m", Path(self.dir.name)).window(6000., 1100/3.6, 15.)
        # Mid altitude and TAS: shift 3; pitch 15: near + 0.5, far + 1.
        near, far = w.center()
        self.assertAlmostEqual(near, 5500.)
        self.assertAlmostEqual(far, 14000.)
        self.assertAlmostEqual(w.center("reference")[1], 19000.)
        self.assertFalse(w.clamped)

    def test_band_is_mirrored_and_a_missing_corner_drops_that_azimuth(self):
        lib = LowAltLibrary("m", Path(self.dir.name))
        self.assertEqual([az for az, *_ in lib.window(4000., 250., 0.).band()], [-60., 0., 60.])
        w = lib.window(6000., 250., 15.)  # Touches the 8000 m, 30 deg, 60 deg hole.
        self.assertIsNone(w.worst[60.])
        self.assertEqual([az for az, *_ in w.band()], [0.])
        self.assertIsNotNone(w.reference[60.])

    def test_clamps_outside_the_grid_and_flags_it(self):
        w = LowAltLibrary("m", Path(self.dir.name)).window(4000., 250., 60.)
        self.assertEqual(w.center(), (3000., 12000.))
        self.assertTrue(w.clamped)
        self.assertFalse(LowAltLibrary("m", Path(self.dir.name)).window(3700., 880/3.6, -3.).clamped)

    def test_requires_finite_state_and_existing_tables(self):
        with self.assertRaises(ValueError):
            LowAltLibrary("m", Path(self.dir.name)).window(6000., 250., math.nan)
        with self.assertRaises(FileNotFoundError):
            LowAltLibrary("x", Path(self.dir.name))
        self.assertEqual(available(Path(self.dir.name)), ["m"])

    def test_incomplete_grid_uses_the_nearest_table(self):
        Path(self.dir.name, "m__8000m_1300kmh.json").unlink()
        w = LowAltLibrary("m", Path(self.dir.name)).window(7900., 1250/3.6, 0.)
        self.assertEqual(w.center(), (6000., 14000.))  # 8000 m, 900 km/h.

    def test_window_is_the_longest_run_of_hits(self):
        ranges = [1, 2, 3, 4, 5, 6, 7]
        self.assertEqual(window(ranges, [True, False, False, True, True, True, False]), (4, 6))
        self.assertEqual(window(ranges, [True, True, False, True, True, False, False]), (4, 5))  # Tie: farther.
        self.assertIsNone(window(ranges, [False]*7))


if __name__ == "__main__":
    unittest.main()
