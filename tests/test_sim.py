"""Standalone simulator input, observation and replay contracts."""
import copy
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout

from wt_overlay.engagement import Engagement, ReplayWriter
from wt_overlay.flight import SUBSTEP_S
from wt_overlay.rl_env import ScriptController
from wt_overlay.sim import build_simulation, load_scenario, main, result_document, validate_scenario

HAVE_MISSILE_SIM = (Path(__file__).resolve().parents[2]/"missle_sim"/"src"/"aim120_model").exists()


class ScenarioTests(unittest.TestCase):
    def test_normalization_preserves_input_and_defaults(self):
        value = load_scenario()
        original = copy.deepcopy(value)
        del value["seed"]
        normalized = validate_scenario(value)
        self.assertEqual(normalized["seed"], 1)
        self.assertNotIn("seed", value)
        self.assertEqual(normalized["teams"], original["teams"])

    def test_rejects_bad_shape_types_and_nonfinite_inputs(self):
        edits = [lambda s: s.update(extra=True), lambda s: s.update(seed=True),
                 lambda s: s.update(time_limit_s=-1), lambda s: s.update(map_half_m=float("nan")),
                 lambda s: s.update(teams=[[], []]),
                 lambda s: s["teams"][0][0].update(velocity_mps=[0, 0, 0]),
                 lambda s: s["teams"][0][0].update(position_m=[0, 0, 0]),
                 lambda s: s["teams"][0][0].update(position_m=[0, 0, float("inf")]),
                 lambda s: s["teams"][0][0].update(missiles=1.5),
                 lambda s: s["teams"][0][0].update(skill="ace"),
                 lambda s: s["teams"][0][0].update(flame_probability=2),
                 lambda s: s["teams"][0][0].update(missile=None, missiles=1)]
        for edit in edits:
            value = load_scenario()
            edit(value)
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_scenario(value)


@unittest.skipUnless(HAVE_MISSILE_SIM, "sibling missile models not installed")
class SimulationTests(unittest.TestCase):
    def duel(self, duration=60.):
        value = load_scenario()
        value["time_limit_s"] = duration
        for i, t in enumerate(value["teams"]):
            t[0].update(position_m=[0, -20000 if i == 0 else 20000, 8000], missiles=2)
        return value

    def test_equipment_checks(self):
        for fields in (dict(aircraft="missing"), dict(missile="su_r_77_1"), dict(missiles=13), dict(chaff=10000)):
            value = load_scenario()
            value["teams"][0][0].update(fields)
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                build_simulation(value)
        value = load_scenario()
        value["teams"][0][0].update(aircraft="rafale_c_f3", missiles=1)
        del value["teams"][0][0]["missile"]
        with self.assertRaisesRegex(ValueError, "no supported active missile"):
            build_simulation(value)

    def test_exact_initial_conditions_and_view_boundaries(self):
        value = self.duel(1.)
        value["teams"][0][0].update(position_m=[0, 0, 8000], velocity_mps=[0, 300, 0],
                                     missile=None, missiles=0, chaff=7, mass_factor=1.3)
        value["teams"][1][0].update(position_m=[0, -4000, 8000])
        sim = build_simulation(value)
        p = sim.planes[0]
        self.assertEqual(p.flight.state.position, (0, 0, 8000))
        self.assertEqual(p.flight.state.velocity, (0, 300, 0))
        self.assertEqual(p.controller.pilot.home_xy, (0, 0))
        self.assertEqual(p.controller.pilot.enemy_xy, (0, -4000))
        self.assertEqual(p.chaff, 7)
        self.assertIsNone(p.missile_id)
        self.assertFalse(sim.truth_debug)
        self.assertFalse(p.controller.pilot.debug)
        self.assertIsInstance(p.controller.controller, ScriptController)
        self.assertTrue(90 <= p.camera.fov_deg <= 120)
        obs = sim.observe(p)
        self.assertIsNone(obs.truth)
        self.assertFalse(obs.visual)
        p.camera.point(180, 0)
        p.camera.advance(p.own, 1.)
        self.assertTrue(sim.observe(p).visual)

        rotated = self.duel(1.)
        for i, team in enumerate(rotated["teams"]):
            team[0].update(position_m=[-20000 if i == 0 else 20000, 0, 8000],
                           velocity_mps=[300 if i == 0 else -300, 0, 0])
        other = build_simulation(rotated)
        self.assertEqual([p.controller.pilot.forward_deg for p in other.planes], [90, 270])

    def test_dense_replay_determinism_and_unchanged_world(self):
        value = self.duel()
        replay_a, replay_b = ReplayWriter(None), ReplayWriter(None)
        a = build_simulation(value, replay=replay_a)
        a.run()
        b = build_simulation(value, replay=replay_b)
        b.run()
        self.assertEqual(replay_a.lines, replay_b.lines)
        self.assertEqual(result_document(a), result_document(b))
        rows = [json.loads(s) for s in replay_a.lines]
        self.assertEqual(rows[0]["type"], "header")
        self.assertEqual(rows[-1]["type"], "end")
        self.assertEqual(rows[0]["resolved"]["planes"][0]["position_m"], [0, -20000, 8000])
        dense = [r for r in rows if r["type"] == "missile_tick"]
        self.assertGreater(len(dense), 48)
        self.assertTrue(all(len(m["runtime_state"]) == len(rows[0]["missile_runtime_columns"])
                            for r in dense for m in r["missiles"]))
        self.assertAlmostEqual(dense[1]["t"]-dense[0]["t"], SUBSTEP_S)
        self.assertGreater(a.launches, 0)
        self.assertTrue(any(e["kind"] == "observed_contact" and e["source"] == "radar" for e in a.log))
        self.assertTrue(all(p.controller.observation.truth is None for p in a.planes))
        self.assertEqual(result_document(a)["outcome"], "draw")
        self.assertEqual(a.missile_errors, 0)
        self.assertEqual(sum(p.flight.faults for p in a.planes), 0)
        tick, state = a.tick, a.planes[0].flight.state
        a.step()
        self.assertEqual(a.tick, tick)
        self.assertEqual(a.planes[0].flight.state, state)

        # Remove only the new recorder/subclass hooks. The legacy world must
        # produce identical states and pre-existing events under the same inputs.
        baseline = build_simulation(value)
        baseline.__class__ = Engagement
        for p in baseline.planes:
            p.controller = p.controller.controller
        baseline.run()
        self.assertEqual([p.flight.state for p in a.planes], [p.flight.state for p in baseline.planes])
        self.assertEqual([e for e in a.log if e["kind"] != "observed_contact"], baseline.log)

    def test_cli_outputs_and_pending_missiles_are_explicit(self):
        with tempfile.TemporaryDirectory() as td, redirect_stdout(io.StringIO()):
            code = main(["--time-limit-s", "1", "--out", td])
            self.assertEqual(code, 0)
            out = Path(td)
            resolved = json.loads((out/"scenario.json").read_text())
            result = json.loads((out/"result.json").read_text())
            rows = [json.loads(s) for s in (out/"replay.jsonl").read_text().splitlines()]
        self.assertEqual(result["reason"], "time_limit")
        self.assertTrue(result["terminal"])
        self.assertEqual(result["outcome"], "draw")
        self.assertEqual(result["time_s"], 1.)
        self.assertEqual(resolved["requested"]["time_limit_s"], 1.)
        self.assertEqual(rows[-1]["type"], "end")
        self.assertIn("performance", result)
        self.assertTrue(all("performance" not in row for row in rows))
        self.assertEqual(len(result["replay_sha256"]), 64)
        for p in result["final_states"]:
            self.assertTrue(all(math.isfinite(x) for x in p["state"]["position"]))


if __name__ == "__main__":
    unittest.main()
