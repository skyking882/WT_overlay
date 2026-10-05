import json
import math
from pathlib import Path
import tempfile
import unittest

from wt_overlay import pk
from wt_overlay.offense import REACH_P, OffenseAdvisor, available, course_for_blip_direction


def write_pk_model(folder, name="m", reach_bias=5.2):
    """Synthetic linear model (no hidden layer) with known lines for a head-on, co-altitude shot:
    P(reach) = REACH_P at 30 km hot and 12 km cold (+1 km per km of target altitude above);
    P(hit) = 0.5 at 10 km and 0.25 at 15 km hot, for every skill and launch mode."""
    n = len(pk.FEATURES)
    index = {f: i for i, f in enumerate(pk.FEATURES)}
    def row(**weights):
        r = [0.]*n
        for k, v in weights.items():
            r[index[k]] = v
        return r
    reach = row(cos_course=1.8, range_m=-0.0002, alt_diff_m=0.0002)
    hit = row(cos_course=1.0, range_m=-0.00021972)
    data = dict(missile=name, features=list(pk.FEATURES), outputs=list(pk.OUTPUTS), activation="silu",
                mean=[0.]*n, std=[1.]*n, layers=[dict(w=[reach]+[hit]*4, b=[reach_bias]+[1.1972]*4)])
    Path(folder).mkdir(parents=True, exist_ok=True)
    (Path(folder)/f"{name}.json").write_text(json.dumps(data))


class PkModelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        write_pk_model(self.temp.name)
        self.advisor = OffenseAdvisor("m", pk.Assumption(), data_dir=Path(self.temp.name))

    def tearDown(self):
        self.temp.cleanup()

    def test_rose_cells_follow_range_and_course(self):
        rose = self.advisor.rose(8000., 1200/3.6)
        self.assertEqual(rose.ranges_m[0], 3000.)
        p_hit, p_reach = rose.at(5000., 0.)
        self.assertGreater(p_hit, .7)  # 0.75 at 5 km head-on
        self.assertGreater(p_reach, REACH_P)
        cold_far = rose.at(30000., 180.)
        self.assertLess(cold_far[1], REACH_P)
        self.assertLess(rose.at(20000., 0.)[0], rose.at(10000., 0.)[0])

    def test_envelope_lines_land_where_the_model_puts_them(self):
        env = self.advisor.envelope(8000., 1200/3.6)
        lines = env.lines[(0., 0.)]
        for name, km in (("rmax_hot", 30.), ("rmax_cold", 12.), ("pk50_hot", 10.), ("pk25_hot", 15.)):
            self.assertAlmostEqual(lines[name]/1000, km, delta=.3, msg=name)
        self.assertFalse(env.capped)
        points = env.line("rmax_hot")
        self.assertEqual([az for az, _ in points], [-60., -45., -30., -15., 0., 15., 30., 45., 60.])
        profile = env.profile("rmax_hot")
        self.assertEqual([dh for dh, _ in profile], sorted(dh for dh, _ in profile))
        by_dh = dict(profile)
        self.assertAlmostEqual(by_dh[3000.]/1000, 33., delta=.3)  # Higher target: farther reach.

    def test_lines_beyond_the_training_range_are_capped_not_extrapolated(self):
        write_pk_model(self.temp.name, "far", reach_bias=20.)
        env = OffenseAdvisor("far", pk.Assumption(), data_dir=Path(self.temp.name)).envelope(8000., 1200/3.6)
        self.assertEqual(env.lines[(0., 0.)]["rmax_hot"], pk.MAX_RANGE_M)
        self.assertTrue(env.is_capped("rmax_hot"))
        self.assertIsNone(self.advisor.model.evaluate(8000., 1200., pk.Assumption(), 50000.))

    def test_aliases_share_a_network_and_layout_is_checked(self):
        write_pk_model(self.temp.name, "cn_pl12")
        self.assertIn("cn_sd10a", available(Path(self.temp.name)))
        self.assertEqual(pk.PkNet.load("cn_sd10a", Path(self.temp.name)).missile, "cn_pl12")
        bad = json.loads((Path(self.temp.name)/"m.json").read_text())
        bad["features"] = bad["features"][:-1]
        with self.assertRaises(ValueError):
            pk.PkNet(bad)

    def test_blip_direction_maps_to_target_course(self):
        self.assertEqual(course_for_blip_direction(0.), 0.)
        self.assertEqual(course_for_blip_direction(90.), 90.)
        self.assertEqual(course_for_blip_direction(-90.), 90.)
        self.assertEqual(course_for_blip_direction(180.), 180.)
        self.assertEqual(course_for_blip_direction(150.), 150.)


class ShippedModelTests(unittest.TestCase):
    @unittest.skipUnless((pk.DATA_DIR/"cn_pl12.json").exists(), "no distilled PL-12 model")
    def test_pl12_head_on_close_shot_beats_a_long_cold_one(self):
        advisor = OffenseAdvisor("cn_pl12", pk.Assumption())
        close = advisor.model.evaluate(8000., 1200., advisor.assumption, 5000., 0.)
        far_cold = advisor.model.evaluate(8000., 1200., advisor.assumption, 40000., 180.)
        self.assertGreater(close["p_reach"], .9)
        self.assertGreater(close["normal_tws"], far_cold["normal_tws"])
        self.assertTrue(all(0. <= v <= 1. and math.isfinite(v) for v in close.values()))


if __name__ == "__main__":
    unittest.main()
