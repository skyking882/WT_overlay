import dataclasses
import json
import math
from pathlib import Path
import tempfile
import unittest

from wt_overlay.contracts import G, PerformanceCondition
from wt_overlay.fm import atmosphere, load_model
from wt_overlay.fm.engine import JetEngine
from wt_overlay.fm.polar import MachCurve, PolarProperties


SAMPLE = Path(__file__).resolve().parents[1]/"data/fm/su_27sm.blkx"


class PolarSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = json.loads(SAMPLE.read_text())
        cls.wing = cls.raw["Aerodynamics"]["WingPlane"]
        cls.props = PolarProperties.from_mapping(cls.wing["FlapsPolar0"], 14.7, 61.98)

    def test_linear_lift_and_induced_drag_anchor(self):
        p = self.props.at_mach(0)
        self.assertAlmostEqual(p.cl(0), .03)
        self.assertAlmostEqual(p.cl(5), .03+.065*5)
        self.assertAlmostEqual(p.cd(5), .0071+(.03+.065*5)**2/(math.pi*.53*14.7**2/61.98))

    def test_source_drag_uses_linear_lift_after_stall(self):
        p = self.props.at_mach(0)
        aoa = 35
        expected = min(.0071+(.03+.065*aoa)**2*p.induced+.01*(aoa-28),
                       .15+1.55*abs(math.sin(math.radians(aoa))))
        wrong = min(.0071+p.cl(aoa)**2*p.induced+.01*(aoa-28),
                    .15+1.55*abs(math.sin(math.radians(aoa))))
        self.assertAlmostEqual(p.cd(aoa), expected)
        self.assertGreater(abs(p.cd(aoa)-wrong), .01)

    def test_critical_lift_and_join(self):
        p = self.props.at_mach(0)
        self.assertAlmostEqual(p.cl(28), 1.55)
        self.assertAlmostEqual(p.cl(-20), -.7)
        for aoa in (p.linear_low, p.linear_high, 28, -20):
            self.assertAlmostEqual(p.cl(aoa-1e-7), p.cl(aoa+1e-7), places=6)

    def test_high_aoa_source_branches(self):
        p = self.props.at_mach(0)
        self.assertAlmostEqual(p.cl(180), 0.)
        self.assertAlmostEqual(p.cl(-180), 0.)
        self.assertAlmostEqual(p.cl(150), -1.42*math.sin(math.pi*.0125*30))
        self.assertAlmostEqual(p.cl(-150), .62*math.sin(math.pi*.0125*30))

    def test_mach_curve_hermite_conditions(self):
        c = MachCurve(.9, 1.1, 2.75, -.2, .01)
        self.assertEqual(c.evaluate(.8), 1.)
        self.assertAlmostEqual(c.evaluate(.9), 1.)
        self.assertAlmostEqual(c.evaluate(1.1), 2.75)
        self.assertAlmostEqual(c.evaluate(1.), 1.88)
        self.assertAlmostEqual(c.evaluate(2.), 2.57)
        self.assertAlmostEqual(c.evaluate(100.), .01)

    def test_mach_adjustments_preserve_source_formula(self):
        p = self.props.at_mach(1.1)
        self.assertAlmostEqual(p.cd0, .0071*2.75)
        self.assertAlmostEqual(p.slope, .065*1.4)
        self.assertAlmostEqual(p.cl0, .03*.7)
        cy_mult = self.props.curves[2].evaluate(1.1)
        # Source uses base Cl0 outside and Mach-dependent Cl0 inside parentheses.
        self.assertAlmostEqual(p.critical_cl_high, .03+(1.55-.021)*cy_mult)

    def test_coefficient_rotation(self):
        p = self.props.at_mach(0)
        x, y = p.coefficients(5, 90)
        self.assertAlmostEqual(x, -p.cl(5))
        self.assertAlmostEqual(y, p.cd(5))

    def test_simple_mach_modes(self):
        source = dict(self.wing["FlapsPolar0"], MachFactor=1)
        p = PolarProperties.from_mapping(source, 14.7, 61.98)
        self.assertEqual(p.at_mach(.9).kq, p.at_mach(2).kq)
        source["MachFactor"] = 2
        p = PolarProperties.from_mapping(source, 14.7, 61.98).at_mach(1)
        self.assertAlmostEqual(p.kq, 5.)

    def test_invalid_polar_inputs(self):
        for patch in ({"OswaldsEfficiencyNumber": 0}, {"MachFactor": 8},
                      {"MachMax1": .9}, {"lineClCoeff": float("nan")}):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                PolarProperties.from_mapping(dict(self.wing["FlapsPolar0"], **patch), 14.7, 61.98)


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = json.loads(SAMPLE.read_text())["EngineType0"]["Main"]
        cls.engine = JetEngine(cls.main)

    def test_source_military_wep_grid_anchor(self):
        self.assertAlmostEqual(self.engine.thrust_n(0, 0, False), 7900*.96*G)
        self.assertAlmostEqual(self.engine.thrust_n(0, 0, True), 7900*.96*1.38*1.04*1.15*G)

    def test_bilinear_midpoint(self):
        # Height 0/2000 m, speed 0/200 km/h: .96,.92,.72,.71.
        self.assertAlmostEqual(self.engine.thrust_n(1000, 100/3.6, False),
                               7900*(.96+.92+.72+.71)/4*G)

    def test_no_extrapolation_or_missing_coefficients(self):
        with self.assertRaises(ValueError):
            self.engine.thrust_n(26000, 200, False)
        bad = json.loads(json.dumps(self.main))
        del bad["ThrustMax"]["ThrustMaxCoeff_0_0"]
        with self.assertRaises(ValueError):
            JetEngine(bad)


class StaticPerformanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_model(SAMPLE)
        cls.condition = PerformanceCondition(5000, 300, 23000)

    def test_lift_demand_and_power_identity(self):
        c = self.condition
        p = self.model.evaluate(c)
        self.assertTrue(p.valid, p.reason)
        self.assertAlmostEqual(p.lift_n, c.mass_kg*G, places=4)
        self.assertAlmostEqual(p.sep_mps, c.tas_mps*(p.thrust_n*math.cos(math.radians(p.aoa_deg))-p.drag_n)/(c.mass_kg*G))
        self.assertFalse(self.model.info.validated_in_game)
        self.assertTrue(p.notes)

    def test_supplied_aoa_does_not_enforce_lift(self):
        p = self.model.evaluate(dataclasses.replace(self.condition, aoa_deg=0))
        self.assertTrue(p.valid, p.reason)
        self.assertEqual(p.aoa_deg, 0)
        self.assertGreater(abs(p.lift_n-self.condition.mass_kg*G), 10000)

    def test_afterburner_increases_sep_without_changing_aero(self):
        mil = self.model.evaluate(dataclasses.replace(self.condition, afterburner=False))
        wep = self.model.evaluate(self.condition)
        self.assertTrue(mil.valid and wep.valid)
        self.assertGreater(wep.sep_mps, mil.sep_mps)
        self.assertAlmostEqual(wep.drag_n, mil.drag_n)

    def test_infeasible_lift_returns_invalid(self):
        p = self.model.evaluate(dataclasses.replace(self.condition, tas_mps=20, load_factor=9))
        self.assertFalse(p.valid)
        self.assertIn("升力需求", p.reason)

    def test_unsupported_and_nonfinite_conditions(self):
        for patch in ({"throttle": .5}, {"flap_fraction": .1}, {"gear_fraction": 1},
                      {"airbrake_fraction": 1}, {"mass_kg": 0}, {"tas_mps": float("nan")},
                      {"altitude_m": float("inf")}, {"altitude_m": -1},
                      {"aoa_deg": float("nan")}, {"aoa_deg": 60}, {"afterburner": 1}):
            with self.subTest(patch=patch):
                p = self.model.evaluate(dataclasses.replace(self.condition, **patch))
                self.assertFalse(p.valid)
                self.assertTrue(p.reason)
                self.assertIsNone(p.sep_mps)

    def test_unknown_profile_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"other.blkx"
            path.write_text('{"Aerodynamics": {}}')
            with self.assertRaisesRegex(ValueError, "尚未支持"):
                load_model(path)

    def test_standard_atmosphere_anchor(self):
        rho, sound = atmosphere(0)
        self.assertAlmostEqual(rho, 1.225, places=5)
        self.assertAlmostEqual(sound, 340.294, places=3)
        self.assertAlmostEqual(atmosphere(11000-1e-6)[0], atmosphere(11000+1e-6)[0], places=7)


if __name__ == "__main__":
    unittest.main()
