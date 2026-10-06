"""Self-play, step 1: per-episode kinds in the trainer metrics and the new-versus-old --stats path.

The kind-labelling env below stands in for MatchEnv with self_play_prob (same reporting: env.episode_kind, one
controlled agent in a vs-script episode, two in a self-play episode) so the sampler / metrics tests stay fast.
"""
import contextlib
import io
import json
import math
import os
import random
import tempfile
import unittest

import common  # noqa: F401
import torch

from rl import config as C
from rl import eval_replay, spec
from rl import train as T
from rl.fake_env import FakeMatchEnv
from rl.model import Actor, Critic
from rl.ppo import PPOTrainer
from test_rollout import DEV, make


class KindEnv(FakeMatchEnv):
    """FakeMatchEnv with MatchEnv's episode kinds: self-play episodes control 2 agents, the others 1."""

    def __init__(self, config=None, seed=0):
        cfg = dict(config or {})
        self.p_self = cfg.pop("self_play_prob", 0.5)
        super().__init__(cfg, seed)
        self.kind_rng = random.Random("%s:kind" % seed)
        self.episode_kind = None

    def reset(self):
        self.episode_kind = spec.KIND_SELF if self.kind_rng.random() < self.p_self else spec.KIND_SCRIPT
        self.cfg["min_agents"] = self.cfg["n_agents"] = 2 if self.episode_kind == spec.KIND_SELF else 1
        return super().reset()

    def step(self, actions):
        obs, rew, done, info = super().step(actions)
        info["episode_kind"] = self.episode_kind
        return obs, rew, done, info


def kind_cfg(p_self=0.5, streams_per_env=2, **kw):
    def fn(cfg):
        cfg.env.cls = "test_self_play:KindEnv"
        cfg.env.streams_per_env = streams_per_env
        cfg.env.config.update(self_play_prob=p_self, max_steps=30, p_background_death=0.02, **kw)
    return fn


def envs_of(sm):
    return sm.pool.handles[0].state["host"].envs


class Metrics(unittest.TestCase):
    def test_outcome_counts_accepts_old_and_new_tuples(self):
        old = [("a", "win", 1, 0), ("a", "loss", 0, 1)]
        new = [("a", "win", 1, 0, "vs_script"), ("a", "trade", 1, 1, "self_play")]
        self.assertEqual(T.outcome_counts(old), dict(win=1, loss=1, trade=0, none=0, kills=1, deaths=1))
        self.assertEqual(T.outcome_counts(new), dict(win=1, loss=0, trade=1, none=0, kills=2, deaths=1))
        self.assertEqual(T.outcome_rates(T.outcome_counts(old)), dict(win_rate=0.5, exchange=1.0))
        self.assertEqual(T.outcome_rates(T.outcome_counts([])), dict(win_rate=None, exchange=None))

    def test_outcomes_by_kind_splits_counts_and_rates(self):
        rows = [("a", "win", 1, 0, "vs_script"), ("b", "win", 2, 0, "vs_script"), ("a", "loss", 0, 1, "vs_script"),
                ("a", "none", 0, 0, "vs_script"),
                ("a", "win", 1, 0, "self_play"), ("b", "loss", 0, 1, "self_play")]
        by = T.outcomes_by_kind(rows, {"vs_script": 300, "self_play": 120})
        vs, sp = by["vs_script"], by["self_play"]
        self.assertEqual((vs["win"], vs["loss"], vs["none"], vs["kills"], vs["deaths"], vs["episodes"]), (2, 1, 1, 3, 1, 4))
        self.assertEqual((vs["win_rate"], vs["exchange"], vs["decisions"]), (0.5, 3.0, 300))
        self.assertEqual((sp["win"], sp["loss"], sp["episodes"], sp["win_rate"], sp["exchange"], sp["decisions"]),
                         (1, 1, 2, 0.5, 1.0, 120))
        # a kind that acted but finished no episode is still listed
        by = T.outcomes_by_kind(rows[:4], {"vs_script": 10, "self_play": 6})
        self.assertEqual((by["self_play"]["episodes"], by["self_play"]["win_rate"], by["self_play"]["decisions"]),
                         (0, None, 6))
        self.assertEqual(T.outcomes_by_kind([]), {})
        self.assertNotIn("timeout_rate", T.outcomes_by_kind(rows)["vs_script"])       # without episodes: as before

    # (aircraft, outcome, kills, deaths, kind) and the matching (aircraft, return, length, end, kind), in sampler order
    ROWS = [("a", "win", 1, 0, "vs_script"), ("a", "none", 0, 0, "vs_script"), ("b", "none", 0, 0, "vs_script"),
            ("a", "trade", 1, 1, "vs_script"), ("b", "win", 1, 0, "vs_script"),
            ("a", "win", 1, 0, "self_play"), ("b", "none", 0, 0, "self_play"), ("a", "loss", 0, 1, "self_play")]
    EPS = [("a", 1.0, 10, "terminal", "vs_script"), ("a", -0.5, 900, "timeout", "vs_script"),
           ("b", 0.0, 40, "terminal", "vs_script"), ("a", -1.0, 30, "terminal", "vs_script"),
           ("b", 0.5, 1008, "timeout", "vs_script"),           # a kill, then alive at the time limit: win AND timeout
           ("a", 1.0, 20, "terminal", "self_play"), ("b", -0.5, 1008, "timeout", "self_play"),
           ("a", -2.0, 20, "lost", "self_play")]

    def test_outcomes_by_kind_with_episodes_adds_trade_timeout_and_the_none_split(self):
        by = T.outcomes_by_kind(self.ROWS, {"vs_script": 50, "self_play": 30}, self.EPS)
        vs, sp = by["vs_script"], by["self_play"]
        self.assertEqual((vs["win"], vs["none"], vs["trade"], vs["episodes"]), (2, 2, 1, 5))   # old keys unchanged
        self.assertEqual((vs["trade_rate"], vs["timeouts"], vs["timeout_rate"]), (0.2, 2, 0.4))
        # survived without a kill: one ran out of time, one ended otherwise (e.g. the opponent crashed)
        self.assertEqual((vs["none_timeout"], vs["none_other"]), (1, 1))
        self.assertEqual((sp["none_timeout"], sp["none_other"], sp["timeouts"]), (1, 0, 1))
        self.assertAlmostEqual(sp["timeout_rate"], 1 / 3)
        self.assertEqual(sp["trade_rate"], 0.0)
        # lists that do not line up: no split rather than a wrong one
        for eps in (self.EPS[:-1], [("x",) + e[1:] for e in self.EPS]):
            by = T.outcomes_by_kind(self.ROWS, None, eps)
            self.assertEqual((by["vs_script"]["none_timeout"], by["vs_script"]["none_other"]), (None, None))
        # a kind that acted but finished nothing
        by = T.outcomes_by_kind(self.ROWS[:5], {"self_play": 7}, self.EPS[:5])
        self.assertEqual((by["self_play"]["trade_rate"], by["self_play"]["timeout_rate"], by["self_play"]["timeouts"]),
                         (None, None, 0))

    def test_episode_ends_by_kind_and_sampler_extras(self):
        self.assertEqual(T.episode_ends_by_kind(self.EPS + [("c", 0.0, 5, "terminal")]),      # an old 4-tuple
                         {"vs_script": {"terminal": 4, "timeout": 2},
                          "self_play": {"terminal": 1, "timeout": 1, "lost": 1}})
        st = {"t_infer": 1.0, "episodes": [], "outcomes": [], "events": {}, "late_credited": 2, "n_valid": 9,
              "settle_ticks": 160, "settle_envs": 3, "by_env": {"0": 1, "1": 2.5}, "flag": True,
              "nested": {"a": {"b": 1}}, "names": ["x"]}
        self.assertEqual(T.sampler_extras(st), {"settle_ticks": 160, "settle_envs": 3, "by_env": {"0": 1, "1": 2.5},
                                                "flag": True})


def run_round(p_self, streams_per_env=2, streams=8, rounds=1, seed=1):
    """collect + finish + update + round_record for a KindEnv sampler; returns the list of (record, stats)."""
    cfg, actor, critic, sm = make(seed=seed, streams=streams, steps=40, seg=8, burn=4,
                                  cfg_fn=kind_cfg(p_self, streams_per_env))
    trainer = PPOTrainer(cfg, actor, critic, None, DEV)
    out = []
    for _ in range(rounds):
        buf, st = sm.collect(actor)
        sm.finish(buf, critic, None, 0.995, 0.95)
        m = trainer.update(buf)
        out.append((T.round_record(cfg, trainer, buf, st, m, sm), st, buf))
    return out, sm


class SamplerAndRecord(unittest.TestCase):
    def test_streams_follow_the_episode_kind_and_nothing_breaks_with_varying_agent_counts(self):
        cfg, actor, critic, sm = make(streams=8, steps=40, seg=8, burn=4, cfg_fn=kind_cfg(0.5))
        seen = set()
        for _ in range(3):
            buf, st = sm.collect(actor)
            sm.finish(buf, critic, None, 0.995, 0.95)
            for j, env in envs_of(sm).items():
                bound = [s for s in sm.env_streams[j] if sm.slot_agent[s] is not None]
                for s in bound:
                    self.assertEqual(sm.ep_kind[s], env.episode_kind)
                self.assertLessEqual(len(bound), 2 if env.episode_kind == spec.KIND_SELF else 1)
                seen.add(env.episode_kind)
            # every decision belongs to exactly one kind
            self.assertEqual(sum(st["decisions_by_kind"].values()), st["n_valid"])
            for o in st["outcomes"]:
                self.assertEqual(len(o), 5)
                self.assertIn(o[4], (spec.KIND_SCRIPT, spec.KIND_SELF))
        self.assertEqual(seen, {spec.KIND_SCRIPT, spec.KIND_SELF})

    def test_valid_fraction_is_between_all_vs_script_and_all_self_play(self):
        vf = {}
        for p in (0.0, 0.5, 1.0):
            recs, _ = run_round(p, streams=16)
            vf[p] = recs[0][0]["valid_fraction"]
        self.assertGreater(vf[0.5], vf[0.0] + 0.05)       # the second stream of an env is idle in vs-script episodes
        self.assertGreater(vf[1.0], vf[0.5] + 0.05)
        self.assertLess(vf[0.0], 0.55)                    # 1 of 2 streams per env, deaths only lower it
        self.assertLessEqual(vf[1.0], 1.0)

    def test_self_play_with_too_few_streams_stops_with_a_clear_message(self):
        with self.assertRaisesRegex(RuntimeError, "streams_per_env"):
            make(streams=4, cfg_fn=kind_cfg(1.0, streams_per_env=1))

    def test_record_keeps_the_old_keys_over_vs_script_and_adds_outcomes_by_kind(self):
        recs, sm = run_round(0.5, streams=16, rounds=3)
        checked = 0
        for rec, st, _ in recs:
            outs = st["outcomes"]
            vs = [o for o in outs if o[4] == spec.KIND_SCRIPT]
            sp = [o for o in outs if o[4] == spec.KIND_SELF]
            by = rec["outcomes_by_kind"]
            self.assertEqual({k: by[spec.KIND_SCRIPT][k] for k in rec["outcomes"]}, T.outcome_counts(vs))
            self.assertEqual(rec["outcomes"], T.outcome_counts(vs) if vs else T.outcome_counts(outs))
            self.assertEqual((rec["win_rate"], rec["exchange"]),
                             tuple(T.outcome_rates(rec["outcomes"]).values()))
            if sp:
                self.assertEqual({k: by[spec.KIND_SELF][k] for k in rec["outcomes"]}, T.outcome_counts(sp))
                checked += 1
            self.assertEqual(by[spec.KIND_SCRIPT]["decisions"], st["decisions_by_kind"].get(spec.KIND_SCRIPT, 0))
            # per-aircraft win/loss tallies describe the same episodes as `outcomes`
            tally = sum(v.get(k, 0) for v in rec["per_aircraft"].values() for k in ("win", "loss", "trade", "none"))
            self.assertEqual(tally, sum(rec["outcomes"][k] for k in ("win", "loss", "trade", "none")))
            json.dumps(rec["outcomes_by_kind"])
        self.assertGreater(checked, 0, "no self-play episode finished in 3 rounds")

    def test_record_splits_ends_by_kind_and_reports_the_value_error_in_reward_units(self):
        recs, _ = run_round(0.5, streams=16, rounds=2)
        for rec, st, _ in recs:
            eps = st["episodes"]
            ends = rec["episode_ends_by_kind"]
            self.assertEqual(sum(sum(d.values()) for d in ends.values()), rec["episodes_finished"])
            merged = {}
            for d in ends.values():
                for k, n in d.items():
                    merged[k] = merged.get(k, 0) + n
            self.assertEqual(merged, rec["episode_kinds"])            # the old mixed key is the sum over kinds
            for kind, c in rec["outcomes_by_kind"].items():
                mine = [e for e in eps if e[4] == kind]
                self.assertEqual(c["timeouts"], sum(1 for e in mine if e[3] == "timeout"))
                self.assertEqual(c["none_timeout"] + c["none_other"], c["none"])
                if c["episodes"]:
                    self.assertAlmostEqual(c["trade_rate"], c["trade"] / c["episodes"])
                    self.assertAlmostEqual(c["timeout_rate"], c["timeouts"] / len(mine))
                    self.assertAlmostEqual(rec["episode_return_by_kind"][kind], sum(e[1] for e in mine) / len(mine))
            self.assertAlmostEqual(rec["value_rmse"], math.sqrt(rec["value_loss"]) * rec["value_scale"], places=6)
            for k in set(st) - T.ST_KNOWN:                             # extra sampler numbers are kept
                if isinstance(st[k], (int, float)):
                    self.assertEqual(rec["sampler_stats"][k], st[k])
            json.dumps(rec)
            line = T.summary_line(dict(rec, time=dict(sample=1., inference=1., env_wait=1., postpass=1., update=1.)))
            self.assertIn("vrms %.3f" % rec["value_rmse"], line)

    def test_only_self_play_episodes_fall_back_to_all_episodes_for_the_old_keys(self):
        recs, _ = run_round(1.0, streams=8)
        rec, st, _ = recs[0]
        self.assertGreater(len(st["outcomes"]), 0)
        self.assertEqual(set(rec["outcomes_by_kind"]), {spec.KIND_SELF})
        self.assertEqual(rec["outcomes"], T.outcome_counts(st["outcomes"]))

    def test_env_without_kind_counts_as_vs_script_and_leaves_the_old_keys_unchanged(self):
        cfg, actor, critic, sm = make(streams=8, steps=40, seg=8, burn=4,
                                      cfg_fn=lambda c: c.env.config.update(max_steps=30, p_background_death=0.02))
        trainer = PPOTrainer(cfg, actor, critic, None, DEV)
        buf, st = sm.collect(actor)
        sm.finish(buf, critic, None, 0.995, 0.95)
        rec = T.round_record(cfg, trainer, buf, st, trainer.update(buf), sm)
        self.assertEqual(set(rec["outcomes_by_kind"]), {spec.KIND_SCRIPT})
        self.assertEqual(rec["outcomes"], T.outcome_counts([o[:4] for o in st["outcomes"]]))
        self.assertEqual({k: rec["outcomes_by_kind"][spec.KIND_SCRIPT][k] for k in rec["outcomes"]}, rec["outcomes"])
        self.assertEqual(rec["outcomes_by_kind"][spec.KIND_SCRIPT]["decisions"], st["n_valid"])

    def test_summary_line_shows_both_kinds_only_with_self_play(self):
        recs, _ = run_round(0.5, streams=16)
        rec = recs[0][0]
        rec["time"] = dict(sample=1., inference=1., env_wait=1., postpass=1., update=1., round=4.)
        line = T.summary_line(rec)
        self.assertTrue(line.startswith("round %d  dec" % rec["round"]))
        self.assertIn("vs-script: eps", line)
        self.assertIn("self-play: eps", line)
        rec["outcomes_by_kind"].pop(spec.KIND_SELF)
        self.assertNotIn("self-play", T.summary_line(rec))


class ConfigCheck(unittest.TestCase):
    def cfg(self, streams_per_env, **env_config):
        c = C.preset("smoke")
        c.env.cls = "wt_overlay.rl_env:MatchEnv"
        c.env.streams_per_env = streams_per_env
        c.env.config = env_config
        c.rollout.n_streams = 8
        return c

    def test_self_play_needs_a_stream_for_every_aircraft_slot(self):
        with self.assertRaisesRegex(ValueError, "streams_per_env"):
            self.cfg(1, team_size=1, self_play_prob=0.5).validate()
        self.cfg(2, team_size=1, self_play_prob=0.5).validate()
        self.cfg(2, self_play_prob=0.5).validate()                      # 1v1 by default
        with self.assertRaisesRegex(ValueError, "streams_per_env"):
            self.cfg(2, teams=[[{}, {}], [{}, {}]], self_play_prob=0.1).validate()
        self.cfg(4, teams=[[{}, {}], [{}, {}]], self_play_prob=0.1).validate()
        # without the option nothing changes
        self.cfg(1, team_size=1).validate()
        self.cfg(1, team_size=1, self_play_prob=0).validate()

    def test_a_failed_set_parse_is_reported_instead_of_crashing_a_worker_later(self):
        c = self.cfg(2)
        C.apply_overrides(c, ['env.config={"self_play_prob": 0.5, "multipath_gain": null}'])   # JSON null: not a literal
        with self.assertRaisesRegex(ValueError, "env.config must be a dict"):
            c.validate()


# ---------------------------------------------------------------------------
# --stats with a frozen opponent policy
# ---------------------------------------------------------------------------

def save_actor(path, seed):
    torch.manual_seed(seed)
    torch.save({"actor": Actor().state_dict()}, path)


class OpponentCheckpointStats(unittest.TestCase):
    ENV = dict(team_size=1, time_limit_s=20)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.new = os.path.join(self.tmp.name, "new.pt")
        self.old = os.path.join(self.tmp.name, "old.pt")
        save_actor(self.new, 1)
        save_actor(self.old, 2)
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())     # _stats_worker sets 1 thread

    def counted(self, calls):
        real = eval_replay.load_actor

        def load(path):
            actor = real(path)
            inner = actor.act

            def act(*a, **k):
                calls[os.path.basename(path)] = calls.get(os.path.basename(path), 0) + 1
                return inner(*a, **k)
            actor.act = act
            return actor
        eval_replay.load_actor = load
        self.addCleanup(setattr, eval_replay, "load_actor", real)

    def test_slot_1_is_flown_by_the_opponent_policy_and_runs_are_reproducible(self):
        calls = {}
        self.counted(calls)
        job = (self.new, self.ENV, "mix", 0, False, self.old)
        r1 = eval_replay._stats_worker(job)
        self.assertGreater(calls["new.pt"], 0)
        self.assertGreater(calls["old.pt"], 0)
        # both slots decide every step while both live: roughly the same number of decisions
        self.assertLess(abs(calls["new.pt"] - calls["old.pt"]), 0.5 * calls["new.pt"] + 5)
        self.assertEqual(eval_replay._stats_worker(job), r1)            # own seeded generators
        self.assertIn(r1["result"], ("win", "loss", "trade", "timeout"))
        # default: a script flies slot 1, the opponent checkpoint is not touched
        calls.clear()
        eval_replay._stats_worker((self.new, self.ENV, "mix", 0, False))
        self.assertEqual(set(calls), {"new.pt"})

    def test_checkpoints_are_copied_before_the_workers_load_them(self):
        """A live trainer prunes old checkpoints: the workers must read private copies, which vanish afterwards."""
        import multiprocessing
        seen = {}

        class Pool:
            def __init__(self, procs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def map(self, fn, jobs):
                seen["jobs"] = list(jobs)
                for j in jobs:
                    seen["exist"] = os.path.exists(j[0]) and os.path.exists(j[5])
                os.remove(self.new_path)           # the trainer prunes the originals while the workers run
                os.remove(self.old_path)
                return [fn(j) for j in jobs[:1]] * len(jobs)

        class Ctx:
            def Pool(_, procs):
                p = Pool(procs)
                p.new_path, p.old_path = self.new, self.old
                return p

        real = multiprocessing.get_context
        multiprocessing.get_context = lambda method: Ctx()
        self.addCleanup(setattr, multiprocessing, "get_context", real)
        res = eval_replay.stats(self.new, self.ENV, 2, "mix", 1, False, self.old)
        job = seen["jobs"][0]
        self.assertTrue(seen["exist"])
        self.assertNotIn(job[0], (self.new, self.old))
        self.assertNotIn(job[5], (self.new, self.old))
        self.assertFalse(os.path.exists(job[0]))                        # temporary copies are cleaned up
        self.assertEqual((res["policy"], res["opponent_checkpoint"], res["opponent_skill"]), ("new.pt", "old.pt", None))
        self.assertEqual(res["episodes"], 2)

    def test_cli_new_versus_old_and_option_checks(self):
        base = ["--run-dir", self.tmp.name, "--procs", "1", "--checkpoint", self.new, "--env-config", json.dumps(self.ENV)]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(eval_replay.main(base + ["--stats", "2", "--opponent-checkpoint", self.old]), 0)
        res = json.loads(out.getvalue())
        self.assertEqual((res["policy"], res["opponent_checkpoint"], res["episodes"]), ("new.pt", "old.pt", 2))
        self.assertEqual(res["win"] + res["loss"] + res["trade"] + res["timeout"], 2)
        # the default (script opponent) output keeps its shape
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(eval_replay.main(base + ["--stats", "1"]), 0)
        res = json.loads(out.getvalue())
        self.assertNotIn("opponent_checkpoint", res)
        self.assertEqual((res["opponent_skill"], res["policy"]), ("mix", "new.pt"))
        for bad in (["--stats", "1", "--opponent-checkpoint", self.old, "--scripted"],
                    ["--stats", "1", "--opponent-checkpoint", os.path.join(self.tmp.name, "missing.pt")],
                    ["--opponent-checkpoint", self.old]):                  # without --stats
            with self.assertRaises(SystemExit):
                eval_replay.main(base + bad)


if __name__ == "__main__":
    unittest.main()
