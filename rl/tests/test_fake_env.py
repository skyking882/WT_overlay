import contextlib
import io
import json
import random
import subprocess
import unittest

import common

from rl import spec, wire
from rl.fake_env import FakeMatchEnv, IllegalAction
from rl import check_env


def random_actions(rng, env, obs):
    out = {}
    for aid, o in obs.items():
        n = len(o.entities)
        chosen = {}
        for h in spec.SAMPLE_ORDER:
            m = spec.effective_mask(h, n, o.masks, chosen)
            chosen[h] = rng.choice([i for i, ok in enumerate(m) if ok])
        out[aid] = chosen
    return out


class Contract(unittest.TestCase):
    def episodes(self, cfg=None, n_ep=4, seed=0, script=False):
        env = FakeMatchEnv(cfg or {}, seed)
        rng = random.Random(seed)
        for _ in range(n_ep):
            obs = env.reset()
            while obs:
                acts = env.scripted_actions() if script else random_actions(rng, env, obs)
                yield env, obs, acts
                obs, rew, done, info = env.step(acts)
                obs = {a: o for a, o in obs.items() if not done[a]}

    def test_observation_shapes_dt_and_masks(self):
        seen = 0
        for env, obs, _ in self.episodes():
            for aid, o in obs.items():
                n = len(o.entities)
                self.assertEqual(len(o.own), 96)
                self.assertEqual(len(o.prev_intent), 48)
                self.assertTrue(all(len(e) == 48 for e in o.entities))
                self.assertTrue(all(len(t) == 40 for t in o.truth))
                self.assertLessEqual(n, 64)
                self.assertLessEqual(len(o.truth), 128)
                self.assertIsInstance(o.aircraft, str)
                self.assertAlmostEqual(o.dt, 20 / 48)
                self.assertEqual(spec.check_mask_shapes(n, o.masks), [])
                self.assertEqual(spec.empty_selectable_rows(n, o.masks), [])
                wire.pack_obs(o, check_finite=True)
                seen += 1
        self.assertGreater(seen, 200)

    def test_entity_counts_vary_widely(self):
        counts = {len(o.entities) for _, obs, _ in self.episodes(n_ep=6) for o in obs.values()}
        self.assertGreaterEqual(len(counts), 10)
        self.assertIn(0, {len(o.entities) for _, obs, _ in self.episodes({"min_entities": 0, "target_entities_max": 4}, 6) for o in obs.values()})

    def test_overflow_is_truncated_to_64_and_counted(self):
        cfg = {"overflow": True, "min_entities": 70, "target_entities_max": 90, "p_incoming": 0.0, "max_steps": 15}
        env = FakeMatchEnv(cfg, 3)
        obs = env.reset()
        dropped = 0
        rng = random.Random(0)
        steps = 0
        while obs and steps < 10:
            self.assertTrue(all(len(o.entities) <= 64 for o in obs.values()))
            obs, rew, done, info = env.step(random_actions(rng, env, obs))
            dropped += info["events"]["dropped_entities"]
            obs = {a: o for a, o in obs.items() if not done[a]}
            steps += 1
        self.assertGreater(dropped, 0)

    def test_hold_rule_and_free_look_rows(self):
        held_seen = rows_seen = release_seen = 0
        for env, obs, acts in self.episodes(script=True, n_ep=6):
            for aid, o in obs.items():
                m = o.masks
                a = env.S["agents"][aid]
                # rows of the inactive view modes: exactly one True for the mouse-flight heads
                for vm in (1, 2):
                    self.assertEqual(sum(m["maneuver_ref"][vm]), 1)
                    self.assertTrue(m["maneuver_ref"][vm][len(o.entities)])
                    self.assertEqual(sum(m["maneuver"][vm][0]), 1)
                    self.assertEqual(sum(m["vertical"][vm][1]), 1)
                self.assertEqual(sum(m["look_az"][0]), 1)
                self.assertEqual(sum(m["kb_roll"][0]), 1)
                rows_seen += 1
                if a["hold"] > 0 and a["view_mode"] == 0:
                    held_seen += 1
                    self.assertEqual(sum(m["maneuver_ref"][0]), 1)
                    self.assertEqual(sum(m["maneuver"][0][0]), 1)
                    self.assertEqual(sum(m["vertical"][0][0]), 1)
                    self.assertTrue(m["maneuver"][0][0][a["man"]])
                else:
                    self.assertGreater(sum(m["maneuver_ref"][0]), 1 if len(o.entities) else 0)
        self.assertGreater(held_seen, 30)

    def test_emergency_release_on_new_visible_warning(self):
        n = 0
        env = FakeMatchEnv({"p_incoming": 0.3, "p_background_death": 0.0}, 4)
        obs = env.reset()
        for _ in range(60):
            if not obs:
                break
            obs, rew, done, info = env.step(env.scripted_actions())
            for aid, a in env.S["agents"].items():
                if a["alive"] and a.get("warn_new"):
                    self.assertEqual(a["hold"], 0)
                    n += 1
            obs = {a: o for a, o in obs.items() if not done[a]}
        self.assertGreater(n, 0)

    def test_target_indexed_tables(self):
        rows = 0
        for env, obs, _ in self.episodes(script=True):
            for aid, o in obs.items():
                n = len(o.entities)
                types = [e.index(1.0) if 1.0 in e[:10] else -1 for e in o.entities]
                for j in range(n + 1):
                    radar = j < n and types[j] == 0
                    self.assertEqual(o.masks["radar_mode"][j][2], radar)
                    self.assertEqual(o.masks["weapon"][j][1], bool(radar and env.S["agents"][aid]["missiles"] > 0))
                    self.assertTrue(o.masks["weapon"][j][0])
                rows += 1
        self.assertGreater(rows, 100)

    def test_ref_is_none_row_forbids_aim_maneuvers(self):
        for env, obs, _ in self.episodes(script=True):
            for aid, o in obs.items():
                a = env.S["agents"][aid]
                if not (a["hold"] > 0 and a["view_mode"] == 0):
                    for m in (1, 2, 3):
                        self.assertFalse(o.masks["maneuver"][0][1][m])
                        self.assertTrue(o.masks["maneuver"][0][0][m])
                    return
        self.fail("no unheld step found")

    def test_illegal_and_missing_actions_are_rejected(self):
        env = FakeMatchEnv({}, 1)
        obs = env.reset()
        acts = env.scripted_actions()
        aid = next(iter(acts))
        bad = dict(acts[aid])
        bad["view_mode"] = 7
        with self.assertRaises(IllegalAction):
            env.step(dict(acts, **{aid: bad}))
        with self.assertRaises(KeyError):
            env.step({})

    def test_done_semantics_timeout_and_deaths(self):
        cfg = {"max_steps": 6, "p_incoming": 0.0, "p_background_death": 0.0, "p_match_end": 0.0, "n_agents": 3, "min_agents": 3}
        env = FakeMatchEnv(cfg, 2)
        obs = env.reset()
        n_agents = len(obs)
        for t in range(6):
            obs, rew, done, info = env.step(env.scripted_actions())
            if t < 5:
                self.assertFalse(info["timeout"])
                self.assertTrue(not any(done.values()))
                self.assertEqual(len(obs), n_agents)
        self.assertTrue(info["timeout"])
        self.assertTrue(all(done.values()))
        self.assertEqual(set(obs), set(done), "survivors keep a final observation for bootstrapping")
        with self.assertRaises(RuntimeError):
            env.step({})
        # deaths: reward -2, done, agent gone from obs
        env = FakeMatchEnv({"p_background_death": 0.4, "p_incoming": 0.0, "max_steps": 200, "n_agents": 3}, 5)
        obs = env.reset()
        died = False
        for _ in range(100):
            if not obs:
                break
            prev = set(obs)
            obs, rew, done, info = env.step(env.scripted_actions())
            for aid in prev:
                if done[aid] and not info["timeout"]:
                    self.assertLessEqual(rew[aid], -1.0)
                    died = True
            obs = {a: o for a, o in obs.items() if not done[a]}
        self.assertTrue(died)

    def test_scripted_actions_are_legal_and_cover_alive_agents(self):
        for env, obs, acts in self.episodes(script=True):
            self.assertEqual(set(acts), set(obs))
            for aid, a in acts.items():
                self.assertEqual(spec.illegal_heads(len(obs[aid].entities), obs[aid].masks, a), [])

    def test_snapshot_restore_is_exact(self):
        env = FakeMatchEnv({}, 9)
        obs = env.reset()
        for _ in range(7):
            obs, rew, done, info = env.step(env.scripted_actions())
        snap = env.snapshot()
        a1 = env.scripted_actions()
        r1 = env.step(a1)
        env.restore(snap)
        a2 = env.scripted_actions()
        r2 = env.step(a2)
        self.assertEqual(a1, a2)
        sig = lambda r: ({k: (o.own, o.entities, o.masks, o.truth) for k, o in r[0].items()}, r[1], r[2], r[3])
        self.assertEqual(sig(r1), sig(r2))

    def test_seeds_are_deterministic_and_different(self):
        def run(seed):
            env = FakeMatchEnv({}, seed)
            obs = env.reset()
            out = []
            for _ in range(10):
                obs, rew, done, info = env.step(env.scripted_actions())
                out.append((sorted(rew.items()), [len(o.entities) for o in obs.values()]))
                obs = {a: o for a, o in obs.items() if not done[a]}
                if not obs:
                    break
            return out
        self.assertEqual(run(1), run(1))
        self.assertNotEqual(run(1), run(2))

    def test_contract_checker_passes_under_cpython_and_pypy(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(check_env.main(["--episodes", "2"]), 0)
        if common.PYPY:
            r = subprocess.run([common.PYPY, "-m", "rl.check_env", "--episodes", "2"], cwd=common.ROOT,
                               capture_output=True, text=True, timeout=300)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertEqual(json.loads(r.stdout)["errors"], [])

    def _check_broken(self, mutate, expect):
        import sys as _sys
        import types

        class Broken(FakeMatchEnv):
            def observe(self):
                obs = super().observe()
                for o in obs.values():
                    mutate(o)
                return obs

        mod = types.ModuleType("broken_env_mod")
        mod.Broken = Broken
        _sys.modules["broken_env_mod"] = mod
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = check_env.main(["--env", "broken_env_mod:Broken", "--episodes", "1", "--max-steps", "5"])
        self.assertEqual(rc, 1)
        self.assertIn(expect, out.getvalue())

    def test_contract_checker_flags_an_empty_selectable_mask_row(self):
        def empty_speed(o):
            o.masks["speed"] = [False] * 3
        self._check_broken(empty_speed, "without a True option")

    def test_contract_checker_flags_wrong_field_widths(self):
        def short_own(o):
            o.own = o.own[:-1]
        self._check_broken(short_own, "own: expected 96 values")

    def test_contract_checker_flags_old_style_conditional_tables(self):
        def old_key(o):
            o.masks["weapon|target"] = [[True, True]]
        self._check_broken(old_key, "unknown key")


if __name__ == "__main__":
    unittest.main()
