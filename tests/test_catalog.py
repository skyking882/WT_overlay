"""Fleet schema/identity coverage; numerical checks are not in-game validation."""
from dataclasses import replace
import hashlib
import json
import math
import unittest
from unittest.mock import patch

from wt_overlay.app import OverlayController
from wt_overlay.climb import build_climb_plan
from wt_overlay.contracts import ClimbRequest, FlightState, G, PerformanceCondition
from wt_overlay.fm import load_aircraft, load_model
from wt_overlay.fm.catalog import aircraft_catalog, aircraft_key, find_aircraft
from wt_overlay.fm.polar import PolarProperties


def reference_mass(profile):
    mass = json.loads(profile.path.read_text())["Mass"]
    # Test fixture only. Runtime still requires explicit total mass.
    return mass["EmptyMass"] + .5*mass["MaxFuelMass0"]


class FleetTests(unittest.TestCase):
    def test_catalog_coverage_identity_and_provenance(self):
        profiles = aircraft_catalog()
        self.assertEqual(len(profiles), 138)
        self.assertEqual(len({p.path for p in profiles}), 113)
        self.assertEqual(len({p.country for p in profiles}), 10)
        self.assertEqual(len({aircraft_key(p.id) for p in profiles}), len(profiles))
        self.assertTrue(all(p.br >= 12 and p.name and p.name_en for p in profiles))
        for p in profiles:
            with self.subTest(aircraft=p.id):
                self.assertEqual(hashlib.sha256(p.path.read_bytes()).hexdigest(), p.sha256)
                self.assertIn(p.revision, p.source_url)
                self.assertEqual(find_aircraft(p.id.upper().replace("_", "-")), p)
        for identity in ("f_15c_golden_eagle", "su_27sm", "j_11b"):
            self.assertIsNotNone(find_aircraft(identity))
        self.assertIsNone(find_aircraft("f_15c"))  # Never guess among different variants.
        self.assertIsNone(find_aircraft(None))

    def test_every_aircraft_loads_and_solves_common_reference_conditions(self):
        for profile in aircraft_catalog():
            model = load_aircraft(profile.id)
            mass = reference_mass(profile)
            self.assertEqual(model.info.aircraft_id, profile.id)
            self.assertTrue(model.matches_aircraft(profile.id))
            for altitude, speed in ((0, 200), (5000, 250), (8000, 280)):
                with self.subTest(aircraft=profile.id, altitude=altitude):
                    c = PerformanceCondition(altitude, speed, mass)
                    point = model.evaluate(c)
                    self.assertTrue(point.valid, point.reason)
                    self.assertAlmostEqual(point.lift_n, mass*G, places=4)
                    self.assertGreater(point.thrust_n, 0)
                    self.assertGreaterEqual(point.drag_n, 0)
                    self.assertAlmostEqual(point.sep_mps,
                        (point.thrust_n*math.cos(math.radians(point.aoa_deg))-point.drag_n)*speed/(mass*G))

    def test_engine_instances_dry_engine_and_lift_engine_exclusion(self):
        for identity, count in (("f_16c_block_50", 1), ("f_15c_golden_eagle", 2),
                                ("yak_141", 1), ("m_346fa", 2), ("av_8b_plus", 1)):
            model = load_aircraft(identity)
            with self.subTest(aircraft=identity):
                self.assertEqual(model.engine_count, count)
                c = PerformanceCondition(5000, 250, reference_mass(find_aircraft(identity)))
                mil, maximum = model.evaluate(replace(c, afterburner=False)), model.evaluate(c)
                self.assertEqual(mil.drag_n, maximum.drag_n)
                if identity == "m_346fa":
                    self.assertEqual(mil.thrust_n, maximum.thrust_n)
                else:
                    self.assertGreater(maximum.thrust_n, mil.thrust_n)
        yak = load_aircraft("yak_141")
        self.assertEqual(yak.engine.base_kgf, 10960.)

    def test_sparse_engine_cells_are_invalid_only_when_used(self):
        model = load_aircraft("f_16c_block_40_barak_2")
        engine = model.engine
        self.assertGreater(engine.thrust_n(5000, 250, True), 0)
        with self.assertRaisesRegex(ValueError, "缺少"):
            engine.thrust_n(engine.altitudes[-1], 0, True)

    def test_shared_fm_keeps_selected_variant_and_file_accepts_known_aliases(self):
        for identity in ("saab_ja37di", "saab_ja37di_f21", "saab_ja37d"):
            profile = find_aircraft(identity)
            selected = load_aircraft(identity)
            self.assertEqual(selected.info.aircraft_id, identity)
            self.assertTrue(load_model(profile.path).matches_aircraft(identity))
        self.assertFalse(load_aircraft("saab_ja37di").matches_aircraft("saab_ja37d"))

    def test_sweep_force_interpolation_uses_original_endpoint_polars(self):
        model = load_aircraft("f_14b")
        c = PerformanceCondition(5000, 300, 24000, aoa_deg=3)
        left = model.evaluate(c)
        middle = model.evaluate(replace(c, sweep_fraction=.25))
        right = model.evaluate(replace(c, sweep_fraction=.5))
        self.assertTrue(left.valid and middle.valid and right.valid)
        self.assertAlmostEqual(middle.drag_n, (left.drag_n+right.drag_n)/2)
        self.assertAlmostEqual(middle.lift_n, (left.lift_n+right.lift_n)/2)
        self.assertEqual(left.thrust_n, right.thrust_n)
        self.assertNotEqual(left.drag_n, right.drag_n)
        self.assertFalse(model.evaluate(replace(c, sweep_fraction=1.1)).valid)

    def test_legacy_schema_and_identical_duplicate_scalar(self):
        model = load_aircraft("f-4j")
        self.assertTrue(model.legacy_geometry)
        self.assertAlmostEqual(model.components[0].area_m2, 49.2)
        self.assertAlmostEqual(model.components[2].area_m2, 7.)
        source = {"OswaldsEfficiencyNumber": .7, "ClAfterCritHigh": [1.2, 1.2]}
        self.assertEqual(PolarProperties.from_mapping(source, 5, 7).values["ClAfterCritHigh"], 1.2)
        with self.assertRaises(ValueError):
            PolarProperties.from_mapping(dict(source, ClAfterCritHigh=[1.2, 2.]), 5, 7)

    def test_climb_planner_with_different_fm_layouts(self):
        for identity, sweep in (("f_15c_golden_eagle", 0), ("j_11b", 0), ("f-4j", 0),
                                 ("f_14b", .5), ("yak_141", 0), ("m_346fa", 0),
                                 ("av_8b_plus", 0), ("j_8f", 0)):
            with self.subTest(aircraft=identity):
                model = load_aircraft(identity)
                c = PerformanceCondition(3000, 250, reference_mass(find_aircraft(identity)),
                                         sweep_fraction=sweep)
                plan = build_climb_plan(model, c, ClimbRequest(4000, 250))
                self.assertTrue(plan.points)


class AutomaticSelectionTests(unittest.TestCase):
    def setUp(self):
        self.state = FlightState(0, True, altitude_m=5000, tas_mps=250, aircraft_id="su_27sm")
        self.controller = OverlayController(client=self, mass_kg=18000)
        self.addCleanup(self.controller.stop)

    def poll(self, time_s=None):
        return replace(self.state, time_s=time_s)

    def test_auto_switch_unknown_and_disconnect_keep_only_matching_model(self):
        c = self.controller
        first = c.tick(0)
        self.assertTrue(first.advice.available)
        self.assertEqual(first.model_selection, "auto")
        self.state = replace(self.state, aircraft_id="F-15C Golden Eagle")
        next_snapshot = c.tick(.1)
        self.assertEqual(c.model.info.aircraft_id, "f_15c_golden_eagle")
        self.assertNotEqual(next_snapshot.model_name, first.model_name)
        self.assertTrue(next_snapshot.advice.available)
        self.state = replace(self.state, aircraft_id="unknown")
        self.assertIsNone(c.tick(.2).advice)
        self.assertIsNone(c.model)
        self.state = replace(self.state, aircraft_id="j_11b")
        self.assertTrue(c.tick(.3).advice.available)
        self.state = replace(self.state, valid=False)
        self.assertIsNone(c.tick(.4).advice)
        self.assertEqual(c.model.info.aircraft_id, "j_11b")

    def test_j16_auto_selection_recovers_after_missing_poll_and_normalized_identity(self):
        c = self.controller
        self.state = replace(self.state, aircraft_id="J-16")
        snapshot = c.tick(0)
        self.assertEqual(snapshot.model_selection, "auto")
        self.assertEqual(snapshot.model_name, "歼-16")
        self.assertTrue(snapshot.advice.available)
        model = c.model
        self.state = FlightState(0, False)
        self.assertIsNone(c.tick(.1).advice)
        self.assertIs(c.model, model)
        self.state = FlightState(0, True, altitude_m=5000, tas_mps=250, aircraft_id="j_16")
        self.assertTrue(c.tick(.2).advice.available)
        self.assertIs(c.model, model)

    def test_failed_automatic_load_retries_without_aircraft_switch(self):
        c = self.controller
        self.state = replace(self.state, aircraft_id="j_16")
        with patch("wt_overlay.app.load_aircraft", side_effect=OSError("temporary")) as load:
            self.assertIsNone(c.tick(0).advice)
            self.assertIn("temporary", c.tick(1).status)
            self.assertEqual(load.call_count, 1)
        result = c.tick(2.1)
        self.assertTrue(result.advice.available)
        self.assertEqual(result.model_name, "歼-16")

    def test_manual_selection_gates_mismatch_then_returns_to_auto(self):
        c = self.controller
        c.submit({"action": "aircraft", "id": "f_15c_golden_eagle"})
        self.assertFalse(c.tick(0).advice.available)
        self.assertEqual(c.model.info.aircraft_id, "f_15c_golden_eagle")
        c.submit({"action": "aircraft", "id": "auto"})
        self.assertTrue(c.tick(.1).advice.available)
        self.assertEqual(c.model.info.aircraft_id, "su_27sm")
        with self.assertRaises(ValueError):
            c.submit({"action": "aircraft", "id": "missing"})

    def test_sweep_is_applied_and_change_cancels_climb_plan(self):
        c = self.controller
        self.state = replace(self.state, aircraft_id="f_14b")
        c.submit({"action": "climb_enabled", "enabled": True})
        c.tick(0)
        old_future = c._plan_future
        old_cancel = c._plan_cancel
        c.submit({"action": "sweep", "fraction": .5})
        result = c.tick(.1)
        self.assertTrue(old_cancel.is_set())
        self.assertIsNot(old_future, c._plan_future)
        self.assertEqual(c._plan_base.sweep_fraction, .5)
        self.assertEqual(result.advice.current.condition.sweep_fraction, .5)
        self.assertTrue(result.variable_sweep)

    def test_switching_aircraft_cancels_previous_climb_plan(self):
        c = self.controller
        c.submit({"action": "climb_enabled", "enabled": True})
        c.tick(0)
        cancel = c._plan_cancel
        self.state = replace(self.state, aircraft_id="f_15c_golden_eagle")
        c.tick(.1)
        self.assertTrue(cancel.is_set())
        self.assertEqual(c.model.info.aircraft_id, "f_15c_golden_eagle")

    def test_demo_does_not_auto_select_from_synthetic_identity(self):
        c = self.controller
        c.tick(0)
        c.submit({"action": "mode", "value": "demo"})
        self.assertIsNone(c.tick(.1).advice)
        self.assertIsNone(c.model)


if __name__ == "__main__":
    unittest.main()
