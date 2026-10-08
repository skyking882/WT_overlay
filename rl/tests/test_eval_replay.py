"""rl.eval_replay: the env config taken from the checkpoint, the fixed / train exam sets, the --stats return as the env
hands it out (timeout reward and late rewards included) and the paired reference evaluation.

ScriptedMatch below stands in for MatchEnv in the --stats worker: FakeMatchEnv observations (so real actors can act)
and a fixed match script, so returns and outcomes are known exactly. A few tests use the real MatchEnv with a short
time limit.
"""
import contextlib
import io
import json
import multiprocessing
import os
import tempfile
import types
import unittest

import common  # noqa: F401
import torch

from rl import eval_replay as E
from rl.fake_env import FakeMatchEnv
from rl.model import Actor

STORED = dict(team_size=1, time_limit_s=300, timeout_reward=-0.5, self_play_prob=0.5, policy_ids=[0],
              observation_frame="egocentric", script_perturbation={"p": 1.0}, spawn_layout={"rotate": True})


def save_ckpt(path, seed, env_config=None):
    torch.manual_seed(seed)
    payload = {"actor": Actor().state_dict()}
    if env_config is not None:                     # a PPO checkpoint stores the whole training config
        payload["cfg"] = {"name": "test", "env": {"cls": "wt_overlay.rl_env:MatchEnv", "config": env_config}}
    torch.save(payload, path)


class ScriptedMatch:
    """MatchEnv stand-in for _stats_worker; config["script"] decides the match:

    late     slot 0 launches at step 1 and dies at step 2; the match goes on while its missile flies and kills slot 1 at
             step 4 (a late reward for slot 0)
    crash    slot 1 crashes at step 3, nobody scores a kill
    timeout  slot 0 gets an assist at step 2; nobody dies; time limit at step 5 with config["timeout_reward"]
    """

    def __init__(self, config, seed):
        self.cfg = config
        self.ids = list(config["controlled"])
        fake = FakeMatchEnv(dict(n_agents=2, min_agents=2, p_incoming=0., p_background_death=0., p_match_end=0.), seed)
        obs = fake.reset()
        self.obs0 = {slot: obs[k] for slot, k in enumerate(sorted(obs)[:2])}
        planes = [types.SimpleNamespace(team=k, alive=True, launches=0) for k in (0, 1)]
        self.engagement = types.SimpleNamespace(planes=planes, log=[], time=0., reason=None)
        self.over = False
        self.t = 0

    def alive_ids(self):
        return [s for s in self.ids if self.engagement.planes[s].alive]

    def reset(self):
        return {s: self.obs0[s] for s in self.alive_ids()}

    def die(self, slot, rewards):
        self.engagement.planes[slot].alive = False
        self.engagement.log.append(dict(kind="death", plane=slot))
        if slot in rewards:
            rewards[slot] -= 2.

    def step(self, actions):
        assert set(actions) == set(self.alive_ids()), (actions, self.alive_ids())
        eng, script = self.engagement, self.cfg["script"]
        self.t += 1
        eng.time = self.t * 20 / 48.
        rewards = {s: 0. for s in self.alive_ids()}
        late = {}
        if script == "late":
            if self.t == 1:
                eng.planes[0].launches += 1
            if self.t == 2:
                self.die(0, rewards)
            if self.t == 4:
                eng.log.append(dict(kind="kill", killer=0))
                if 0 in self.ids:
                    late[0] = 1.
                self.die(1, rewards)
                eng.reason = "annihilation"
        elif script == "crash" and self.t == 3:
            self.die(1, rewards)
            eng.reason = "annihilation"
        elif script == "timeout":
            if self.t == 2 and 0 in rewards:
                rewards[0] += .3
            if self.t == 5:
                eng.reason = "time_limit"
                for s in rewards:
                    rewards[s] += self.cfg["timeout_reward"]
        self.over = eng.reason is not None
        info = {"timeout": False, "events": {}}
        if late:
            info["late_rewards"] = late
        obs = {} if self.over else {s: self.obs0[s] for s in self.alive_ids()}
        return obs, rewards, {s: self.over or not eng.planes[s].alive for s in rewards}, info

    def scripted_actions(self):
        return {s: None for s in self.alive_ids()}


class TeamMatch(ScriptedMatch):
    """MatchEnv stand-in for team games of 2 per team (slots 0, 1 against 2, 3), the same match whatever is decided:
    step 1 slot 0 launches, step 2 it kills slot 2, step 3 slot 1 shoots down its teammate 0 (friendly fire), step 4
    slot 3 crashes: team 0 wins with slot 1 left. Every instance is kept in ``made`` (config, policy slots)."""
    made = []

    def __init__(self, config, seed):
        self.cfg = config
        self.ids = list(config["controlled"])
        n = sum(len(t) for t in config["teams"])
        fake = FakeMatchEnv(dict(n_agents=n, min_agents=n, p_incoming=0., p_background_death=0., p_match_end=0.), seed)
        obs = fake.reset()
        self.obs0 = {slot: obs[k] for slot, k in enumerate(sorted(obs)[:n])}
        planes = [types.SimpleNamespace(team=t, alive=True, launches=0)
                  for t, m in enumerate(config["teams"]) for _ in m]
        self.engagement = types.SimpleNamespace(planes=planes, log=[], time=0., reason=None)
        self.over = False
        self.t = 0
        TeamMatch.made.append(self)

    def die(self, slot, rewards, cause="missile"):
        self.engagement.planes[slot].alive = False
        self.engagement.log.append(dict(kind="death", plane=slot, cause=cause))
        if slot in rewards:
            rewards[slot] -= 2.

    def step(self, actions):
        assert set(actions) == set(self.alive_ids()), (actions, self.alive_ids())
        eng = self.engagement
        self.t += 1
        eng.time = self.t * 20 / 48.
        rewards = {s: 0. for s in self.alive_ids()}
        if self.t == 1:
            eng.planes[0].launches += 1
        elif self.t == 2:
            eng.log.append(dict(kind="kill", killer=0, victim=2))
            if 0 in rewards:
                rewards[0] += 1.
            self.die(2, rewards)
        elif self.t == 3:
            eng.log.append(dict(kind="friendly_fire", killer=1, victim=0))
            self.die(0, rewards)
        elif self.t == 4:
            self.die(3, rewards, "crash")
            eng.reason = "annihilation"
        self.over = eng.reason is not None
        obs = {} if self.over else {s: self.obs0[s] for s in self.alive_ids()}
        return obs, rewards, {s: self.over or not eng.planes[s].alive for s in rewards}, {"timeout": False}


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())     # _stats_worker sets 1 thread
        self.new = os.path.join(self.tmp.name, "new.pt")
        self.old = os.path.join(self.tmp.name, "old.pt")
        save_ckpt(self.new, 1)
        save_ckpt(self.old, 2)

    def patch(self, obj, name, value):
        real = getattr(obj, name)
        setattr(obj, name, value)
        self.addCleanup(setattr, obj, name, real)

    def scripted_env(self):
        import wt_overlay.rl_env as rl_env
        self.patch(rl_env, "MatchEnv", ScriptedMatch)

    def in_process_pool(self):
        class Pool:
            def __init__(self, procs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def map(self, fn, jobs):
                return [fn(j) for j in jobs]

        class Ctx:
            def Pool(self, procs):
                return Pool(procs)
        self.patch(multiprocessing, "get_context", lambda method: Ctx())


class EnvConfig(Base):
    def test_the_stored_config_is_the_base_and_overrides_only_touch_their_keys(self):
        path = os.path.join(self.tmp.name, "ckpt_000010.pt")
        save_ckpt(path, 3, STORED)
        cfg, source = E.env_config_for(path, {"time_limit_s": 30, "spawn_layout": None})
        self.assertEqual(source, "checkpoint")
        self.assertEqual(cfg, dict(team_size=1, time_limit_s=30, timeout_reward=-0.5, observation_frame="egocentric",
                                   script_perturbation={"p": 1.0}, spawn_layout=None))
        # the evaluation decides itself which slots a policy flies
        self.assertNotIn("self_play_prob", cfg)
        self.assertNotIn("policy_ids", cfg)

    def test_a_checkpoint_without_a_stored_config_keeps_the_old_default(self):
        cfg, source = E.env_config_for(self.new, {"reach_dir": self.tmp.name})
        self.assertEqual((cfg, source), (dict(team_size=1, time_limit_s=420, reach_dir=self.tmp.name), "default"))
        self.assertEqual(E.env_config_for(None, {})[0], E.DEFAULT_ENV)        # --scripted without a checkpoint

    def test_a_training_machine_path_that_is_missing_here_is_reported(self):
        path = os.path.join(self.tmp.name, "ckpt_000010.pt")
        save_ckpt(path, 3, dict(STORED, reach_dir="/no/such/reach_dir"))
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            E.env_config_for(path)
        self.assertIn("reach_dir=/no/such/reach_dir", err.getvalue())
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            E.env_config_for(path, {"reach_dir": self.tmp.name})
        self.assertEqual(err.getvalue(), "")

    def test_exam_sets(self):
        configs, skipped = E.exam_configs(dict(STORED))
        self.assertEqual(list(configs), ["fixed", "train"])
        self.assertNotIn("script_perturbation", configs["fixed"])
        self.assertEqual(configs["train"]["script_perturbation"], {"p": 1.0})
        self.assertEqual({k: v for k, v in configs["train"].items() if k != "script_perturbation"}, configs["fixed"])
        self.assertEqual(skipped, {})
        # {} is a perturbation with the default settings: still a train set
        self.assertIn("train", E.exam_configs(dict(STORED, script_perturbation={}))[0])
        for cfg in ({k: v for k, v in STORED.items() if k != "script_perturbation"}, dict(STORED, script_perturbation=None)):
            configs, skipped = E.exam_configs(cfg)
            self.assertEqual((list(configs), list(skipped)), (["fixed"], ["train"]))
        self.assertEqual(list(E.exam_configs(STORED, ["train"])[0]), ["train"])
        with self.assertRaises(ValueError):
            E.exam_configs(STORED, ["fixed", "nope"])


class Exams(Base):
    """exam() with play() recorded instead of run: which matches, under which names and env configs."""

    def setUp(self):
        super().setUp()
        self.run_dir = os.path.join(self.tmp.name, "run")
        os.makedirs(os.path.join(self.run_dir, "ppo"))
        self.calls = []

        def play(actor, name, scenario, env_config, out_path, range_km=100., greedy=False, opponent=None, exam_set=None):
            self.calls.append(dict(name=name, cfg=env_config, file=os.path.basename(out_path), sp=opponent is not None,
                                   set=exam_set))
            open(out_path, "w").close()
            return dict(scenario=name, set=exam_set, replay=os.path.basename(out_path))
        self.patch(E, "play", play)

    def test_exam_uses_the_checkpoint_config_and_names_the_sets(self):
        save_ckpt(os.path.join(self.run_dir, "ppo", "ckpt_000003.pt"), 3, STORED)
        line = E.exam(self.run_dir, {"time_limit_s": 30}, ["sm2_vs_ge", "ge_vs_sm2"], self_play=True)
        self.assertEqual([c["file"] for c in self.calls],
                         ["r0003_fixed_sm2_vs_ge.jsonl", "r0003_fixed_ge_vs_sm2.jsonl",
                          "r0003_train_sm2_vs_ge.jsonl", "r0003_train_ge_vs_sm2.jsonl",
                          "r0003_sp_sm2_vs_ge.jsonl", "r0003_sp_ge_vs_sm2.jsonl"])
        self.assertEqual([c["set"] for c in self.calls], ["fixed"] * 2 + ["train"] * 2 + ["sp"] * 2)
        self.assertEqual([c["sp"] for c in self.calls], [False] * 4 + [True] * 2)
        for c in self.calls:
            self.assertEqual((c["cfg"]["timeout_reward"], c["cfg"]["time_limit_s"]), (-0.5, 30))   # stored + override
            self.assertNotIn("self_play_prob", c["cfg"])
            self.assertEqual("script_perturbation" in c["cfg"], c["set"] == "train")
        self.assertEqual((line["round"], line["env_source"], line["sets"], line["skipped_sets"]),
                         (3, "checkpoint", ["fixed", "train", "sp"], {}))
        with open(os.path.join(self.run_dir, "replays", "exams.jsonl")) as f:
            self.assertEqual(json.loads(f.readline())["script_perturbation"], {"p": 1.0})

    def test_without_perturbation_only_the_fixed_set_is_played(self):
        stored = {k: v for k, v in STORED.items() if k != "script_perturbation"}
        save_ckpt(os.path.join(self.run_dir, "ppo", "ckpt_000004.pt"), 3, stored)
        line = E.exam(self.run_dir, {}, ["sm2_vs_ge"])
        self.assertEqual([c["file"] for c in self.calls], ["r0004_fixed_sm2_vs_ge.jsonl"])
        self.assertEqual((line["sets"], list(line["skipped_sets"])), (["fixed"], ["train"]))

    def test_bc_actor_keeps_the_old_hand_written_config(self):
        save_ckpt(os.path.join(self.run_dir, "bc_actor.pt"), 3)
        line = E.exam(self.run_dir, {"reach_dir": self.tmp.name}, ["sm2_vs_ge"])
        self.assertEqual(self.calls[0]["cfg"], dict(team_size=1, time_limit_s=420, reach_dir=self.tmp.name))
        self.assertEqual((line["round"], line["env_source"]), (0, "default"))

    def test_cli_exam_and_prune_with_the_new_names(self):
        save_ckpt(os.path.join(self.run_dir, "ppo", "ckpt_000005.pt"), 3, STORED)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(E.main(["--run-dir", self.run_dir, "--scenarios", "sm2_vs_ge", "--sets", "fixed",
                                     "--env-config", '{"time_limit_s": 30}']), 0)
        self.assertEqual(json.loads(out.getvalue())["sets"], ["fixed"])
        self.assertEqual(self.calls[0]["cfg"]["time_limit_s"], 30)
        rep = os.path.join(self.run_dir, "replays")
        for rnd in range(1, 5):
            for tag in ("fixed", "train", "sp"):
                open(os.path.join(rep, "r%04d_%s_sm2_vs_ge.jsonl" % (rnd, tag)), "w").close()
        E.prune(self.run_dir, keep=2, milestone=2)
        rounds = sorted({int(f[1:5]) for f in os.listdir(rep) if f.endswith(".jsonl") and f.startswith("r")})
        self.assertEqual(rounds, [2, 4, 5])
        self.assertEqual(len([f for f in os.listdir(rep) if f.startswith("r0004_")]), 3)
        for bad in (["--sets", "fixed,nope"], ["--env-config", "[1]"], ["--paired"], ["--stats", "1", "--paired"]):
            with self.assertRaises(SystemExit):
                E.main(["--run-dir", self.run_dir] + bad)


class StatsReturn(Base):
    def test_the_return_is_what_the_env_hands_out_incl_late_and_timeout_rewards(self):
        self.scripted_env()
        late = E._stats_worker((self.new, dict(script="late"), "mix", 0, False))
        self.assertEqual((late["result"], late["return"], late["kills"], late["launches"], late["slot"]),
                         ("trade", -1.0, 1, 1, 0))                     # -2 at death, +1 from its missile afterwards
        to = E._stats_worker((self.new, dict(script="timeout", timeout_reward=-0.5), "mix", 0, False))
        self.assertEqual(to["result"], "timeout")
        self.assertAlmostEqual(to["return"], 0.3 - 0.5)                 # (kills - 2 deaths) would say 0
        scripted = E._stats_worker((None, dict(script="timeout", timeout_reward=-0.5), "mix", 0, True))
        self.assertAlmostEqual(scripted["return"], -0.2)

    def test_stats_mean_return_and_win_without_a_kill(self):
        self.scripted_env()
        self.in_process_pool()
        res = E.stats(self.new, dict(script="late"), 2, "mix", 1)
        self.assertEqual((res["episodes"], res["trade"], res["mean_return"], res["paired"]), (2, 2, -1.0, False))
        self.assertNotIn("by_slot", res)
        res = E.stats(self.new, dict(script="crash"), 3, "mix", 1)
        self.assertEqual((res["win"], res["win_no_kill"], res["mean_return"], res["exchange"]), (3, 3, 0.0, None))

    def test_real_env_timeout_reward_reaches_the_return(self):
        env = dict(team_size=1, time_limit_s=20, timeout_reward=-0.5)          # nobody gets near anybody in 20 s
        r = E._stats_worker((self.new, env, "mix", 0, False))
        self.assertEqual((r["result"], r["return"]), ("timeout", -0.5))
        r = E._stats_worker((self.new, dict(env, timeout_reward=None), "mix", 0, False))
        self.assertEqual((r["result"], r["return"]), ("timeout", 0.0))


class Paired(Base):
    def tag_actors(self):
        """load_actor tags actors with their file; _fly records which file flew which slot."""
        real_load, real_fly = E.load_actor, E._fly
        seen = []

        def load(path):
            actor = real_load(path)
            actor.tag = os.path.basename(path)
            return actor

        def fly(actors, gens, env, obs):
            seen.append({slot: a.tag for slot, a in actors.items()})
            return real_fly(actors, gens, env, obs)
        self.patch(E, "load_actor", load)
        self.patch(E, "_fly", fly)
        return seen

    def test_paired_games_swap_the_slots_and_report_both_sides(self):
        self.scripted_env()
        self.in_process_pool()
        seen = self.tag_actors()
        res = E.stats(self.new, dict(script="crash"), 2, "mix", 1, opponent_path=self.old, paired=True)
        # each seed twice: the policy in slot 0 (slot 1 crashes: a win without a kill), then in slot 1 (a loss)
        self.assertEqual(seen, [{0: "policy.pt", 1: "opponent.pt"}, {1: "policy.pt", 0: "opponent.pt"}] * 2)
        self.assertEqual((res["episodes"], res["pairs"], res["paired"]), (4, 2, True))
        self.assertEqual((res["win"], res["loss"], res["trade"], res["timeout"]), (2, 2, 0, 0))
        self.assertEqual((res["win_rate"], res["exchange"], res["mean_return"], res["win_no_kill"]), (0.5, 1.0, -1.0, 2))
        s0, s1 = res["by_slot"]["slot0"], res["by_slot"]["slot1"]
        self.assertEqual((s0["episodes"], s0["win"], s0["loss"], s0["mean_return"]), (2, 2, 0, 0.0))
        self.assertEqual((s1["episodes"], s1["win"], s1["loss"], s1["mean_return"]), (2, 0, 2, -2.0))
        self.assertEqual((res["policy"], res["opponent_checkpoint"]), ("new.pt", "old.pt"))
        with self.assertRaises(ValueError):
            E.stats(self.new, dict(script="crash"), 1, "mix", 1, paired=True)        # needs an opponent

    def test_a_swapped_game_is_the_same_match_seen_from_the_other_slot(self):
        self.scripted_env()
        a = E._stats_worker((self.new, dict(script="late"), "mix", 7, False, self.old, False))
        b = E._stats_worker((self.new, dict(script="late"), "mix", 7, False, self.old, True))
        self.assertEqual((a["slot"], b["slot"]), (0, 1))
        self.assertEqual((a["aircraft"], a["opponent"]), (b["opponent"], b["aircraft"]))
        self.assertEqual((a["return"], b["return"]), (-1.0, -2.0))
        with self.assertRaises(ValueError):
            E._stats_worker((self.new, dict(script="late"), "mix", 7, False, None, True))

    def test_cli_paired_with_the_real_env(self):
        base = ["--run-dir", self.tmp.name, "--procs", "1", "--checkpoint", self.new,
                "--env-config", json.dumps(dict(team_size=1, time_limit_s=20, timeout_reward=-0.5))]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(E.main(base + ["--stats", "1", "--paired", "--opponent-checkpoint", self.old]), 0)
        res = json.loads(out.getvalue())
        self.assertEqual((res["episodes"], res["pairs"], res["paired"], res["env_source"]), (2, 1, True, "default"))
        self.assertEqual(res["win"] + res["loss"] + res["trade"] + res["timeout"], 2)
        self.assertEqual(res["by_slot"]["slot0"]["episodes"] + res["by_slot"]["slot1"]["episodes"], 2)
        self.assertEqual(res["mean_return"], -0.5)                     # two timeouts at 20 s
        self.assertEqual(res["env_overrides"]["timeout_reward"], -0.5)


TEAM2 = dict(team_size=2)


class Teams(Base):
    """--stats with team_size > 1: who flies which slot, the team counts, paired team swaps, and the real env."""

    def team_env(self):
        import wt_overlay.rl_env as rl_env
        TeamMatch.made = []
        self.patch(rl_env, "MatchEnv", TeamMatch)

    def tag_actors(self):
        """_fly records {slot: (checkpoint file, generator seed)} of every game."""
        real_load, real_fly = E.load_actor, E._fly
        seen = []

        def load(path):
            actor = real_load(path)
            actor.tag = os.path.basename(path)
            return actor

        def fly(actors, gens, env, obs):
            self.assertEqual(set(actors), set(gens))
            seen.append({s: (a.tag, gens[s].initial_seed()) for s, a in actors.items()})
            return real_fly(actors, gens, env, obs)
        self.patch(E, "load_actor", load)
        self.patch(E, "_fly", fly)
        return seen

    def test_team_rows_count_the_policy_slots(self):
        self.team_env()
        seen = self.tag_actors()
        r = E._stats_worker((self.new, TEAM2, "top", 7, False))
        self.assertEqual((r["team"], r["slots"], len(r["aircraft"]), len(r["opponent"])), (0, [0, 1], 2, 2))
        self.assertEqual({k: r[k] for k in ("result", "own_deaths", "enemy_deaths", "kills", "launches",
                                            "policy_deaths", "crashes", "friendly_fire", "return")},
                         dict(result="win", own_deaths=1, enemy_deaths=2, kills=1, launches=1, policy_deaths=1,
                              crashes=0, friendly_fire=1, **{"return": -1.0}))     # slot 0: +1 kill, -2 death
        # one generator per policy slot; slots 0 and 1 keep the 1v1 seeds
        self.assertEqual(seen[-1], {0: ("new.pt", 7), 1: ("new.pt", E.OPPONENT_SEED_BASE + 7)})
        teams = TeamMatch.made[-1].cfg["teams"]
        self.assertEqual([m.get("skill") for m in teams[0] + teams[1]], [None, None, "top", "top"])  # enemy scripts
        # the scripted baseline: the same slots on the policy path, the scripts deciding
        b = E._stats_worker((None, TEAM2, "top", 7, True))
        self.assertEqual((TeamMatch.made[-1].ids, len(seen)), ([0, 1], 1))
        self.assertEqual({k: b[k] for k in r if k != "i"}, {k: r[k] for k in r if k != "i"})

    def test_team_control_leaves_the_other_slots_to_the_scripts(self):
        self.team_env()
        seen = self.tag_actors()
        r = E._stats_worker((self.new, TEAM2, "mix", 3, False, None, False, 1))
        self.assertEqual((TeamMatch.made[-1].ids, r["slots"], seen[-1]), ([0], [0], {0: ("new.pt", 3)}))
        # slot 1's friendly-fire kill is a script's; the policy slot's own counts and return remain
        self.assertEqual((r["kills"], r["policy_deaths"], r["friendly_fire"], r["return"]), (1, 1, 0, -1.0))
        b = E._stats_worker((None, TEAM2, "mix", 3, True, None, False, 1))
        self.assertEqual((TeamMatch.made[-1].ids, b["slots"], b["return"]), ([0], [0], -1.0))
        with self.assertRaises(ValueError):
            E._stats_worker((self.new, TEAM2, "mix", 3, False, None, False, 3))

    def test_paired_team_games_swap_the_teams(self):
        self.team_env()
        self.in_process_pool()
        seen = self.tag_actors()
        res = E.stats(self.new, TEAM2, 1, "top", 1, opponent_path=self.old, paired=True, team_control=1)
        o = E.OPPONENT_SEED_BASE
        self.assertEqual(seen, [{0: ("policy.pt", 0), 2: ("opponent.pt", 2 * o), 3: ("opponent.pt", 3 * o)},
                                {2: ("policy.pt", 2 * o), 0: ("opponent.pt", 0), 1: ("opponent.pt", o)}])
        self.assertEqual([m.ids for m in TeamMatch.made], [[0, 2, 3], [0, 1, 2]])
        teams = TeamMatch.made[0].cfg["teams"]
        self.assertEqual(teams, TeamMatch.made[1].cfg["teams"])                        # the same match
        self.assertNotIn("skill", teams[1][0])                                         # no scripted enemies
        res = E.stats(self.new, TEAM2, 2, "top", 1, opponent_path=self.old, paired=True)
        # team 0 (slots 0, 1): a win, 1 own death, 2 enemy; team 1 (slots 2, 3): a loss, both dead, one crashed
        self.assertEqual({k: res[k] for k in ("episodes", "pairs", "team_size", "team_control", "paired", "policy",
                                              "opponent_checkpoint", "opponent_skill")},
                         dict(episodes=4, pairs=2, team_size=2, team_control=2, paired=True, policy="new.pt",
                              opponent_checkpoint="old.pt", opponent_skill=None))
        self.assertEqual({k: res[k] for k in E.RESULTS}, dict(win=2, loss=2, trade=0, timeout=0))
        self.assertEqual((res["win_rate"], res["exchange"], res["own_deaths"], res["enemy_deaths"]), (0.5, 1.0, 6, 6))
        self.assertEqual([res[k] for k in ("policy_aircraft", "survival", "kills_per_aircraft", "deaths_per_aircraft",
                                           "launches_per_aircraft", "crashes", "friendly_fire", "mean_return")],
                         [8, 0.25, 0.25, 0.75, 0.25, 2, 2, -2.5])
        t0, t1 = res["by_team"]["team0"], res["by_team"]["team1"]
        self.assertEqual((t0["episodes"], t0["win"], t0["exchange"], t0["mean_return"]), (2, 2, 2.0, -1.0))
        self.assertEqual((t1["episodes"], t1["loss"], t1["exchange"], t1["mean_return"]), (2, 2, 0.5, -4.0))
        self.assertNotIn("by_slot", res)
        with self.assertRaises(ValueError):
            E.stats(self.new, TEAM2, 1, "top", 1, team_control=3)
        with self.assertRaises(ValueError):
            E.stats(self.new, dict(script="late"), 1, "mix", 1, team_control=2)        # 1v1: k is 1

    def test_cli_team_games_in_the_real_env(self):
        self.in_process_pool()                    # a spawned worker would load the env's tables again (seconds)
        env = dict(team_size=2, time_limit_s=20, timeout_reward=-0.5)               # nobody meets in 20 s
        base = ["--run-dir", self.tmp.name, "--procs", "1", "--checkpoint", self.new, "--env-config", json.dumps(env),
                "--stats", "1", "--team-control", "1"]
        for extra, policy in (([], "new.pt"), (["--scripted"], "scripts")):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(E.main(base + extra), 0)
            res = json.loads(out.getvalue())
            self.assertEqual((res["policy"], res["team_size"], res["team_control"], res["episodes"]),
                             (policy, 2, 1, 1))
            self.assertEqual((res["timeout"], res["own_deaths"], res["enemy_deaths"], res["policy_aircraft"],
                              res["survival"], res["mean_return"]), (1, 0, 0, 1, 1.0, -0.5))
        r = E._stats_worker((self.new, env, "mix", 0, False))                        # the policy flies both
        self.assertEqual((r["slots"], r["result"], r["return"]), ([0, 1], "timeout", -1.0))
        for bad in (["--team-control", "3"],
                    ["--env-config", json.dumps(dict(env, team_size=1)), "--team-control", "2"]):
            with self.assertRaises(SystemExit):
                E.main(base + bad)
        with self.assertRaises(SystemExit):                                          # without --stats
            E.main(["--run-dir", self.tmp.name, "--team-control", "1"])



class AirfieldTeams(Base):
    """With the env's airfield option the team rows also count the policy slots' landings, rearms and the aircraft
    parked at the end (alive for the result); without it, and in 1v1s, the rows are unchanged."""

    def test_team_rows_count_landings_and_a_parked_aircraft_is_alive(self):
        class Parked(TeamMatch):
            def __init__(self, config, seed):
                super().__init__(config, seed)
                self.engagement.airfield = {}
                for p in self.engagement.planes:
                    p.landings = p.rearms = 0
                    p.grounded = False

            def step(self, actions):
                out = super().step(actions)
                if self.t == 2:                          # slot 1 lands and is rearmed; it is parked at the end
                    p = self.engagement.planes[1]
                    p.landings, p.rearms, p.grounded = 1, 1, True
                return out
        import wt_overlay.rl_env as rl_env
        self.patch(rl_env, "MatchEnv", Parked)
        r = E._stats_worker((self.new, TEAM2, "top", 7, False))
        self.assertEqual((r["result"], r["policy_deaths"], r["landings"], r["rearms"], r["grounded_at_end"]),
                         ("win", 1, 1, 1, 1))
        s = E.summarise_teams([r, r])
        self.assertEqual((s["landings"], s["rearms"], s["grounded_at_end"], s["win"]), (2, 2, 2, 2))
        self.patch(rl_env, "MatchEnv", TeamMatch)
        plain = E._stats_worker((self.new, TEAM2, "top", 7, False))
        self.assertEqual({k: v for k, v in r.items() if k not in ("landings", "rearms", "grounded_at_end")}, plain)
        self.assertNotIn("landings", E.summarise_teams([plain]))

    def test_real_env_rows(self):
        env = dict(team_size=2, time_limit_s=20, timeout_reward=-0.5)
        r = E._stats_worker((self.new, dict(env, airfield={}), "mix", 0, False))
        self.assertEqual((r["result"], r["landings"], r["rearms"], r["grounded_at_end"]), ("timeout", 0, 0, 0))
        one = dict(team_size=1, time_limit_s=20, timeout_reward=-0.5)
        self.assertEqual(E._stats_worker((self.new, dict(one, airfield={}), "mix", 0, False)),
                         E._stats_worker((self.new, one, "mix", 0, False)))


class TeamExams(Base):
    def setUp(self):
        super().setUp()
        self.run_dir = os.path.join(self.tmp.name, "run")
        os.makedirs(os.path.join(self.run_dir, "ppo"))

    def test_exams_pick_the_scenarios_of_the_team_size(self):
        calls = []

        def play(actor, name, scenario, env_config, out_path, range_km=100., greedy=False, opponent=None,
                 exam_set=None):
            calls.append((name, len(scenario["teams"][0]), len(scenario["teams"][1]), opponent is not None))
            return dict(scenario=name)
        self.patch(E, "play", play)
        four = [n for n in E.SCENARIOS if n.startswith("4v4_")]
        self.assertEqual(len(four), 3)
        save_ckpt(os.path.join(self.run_dir, "ppo", "ckpt_000003.pt"), 3, dict(STORED, team_size=4))
        line = E.exam(self.run_dir, {}, None, self_play=True)
        self.assertEqual(calls, [("%s_%s" % (s, n), 4, 4, s == "sp") for s in ("fixed", "train", "sp") for n in four])
        self.assertEqual(len(line["results"]), 9)
        calls.clear()
        E.exam(self.run_dir, {"team_size": 1}, None, sets=["fixed"])                  # 1v1: the four 1v1 scenarios
        self.assertEqual([c[0] for c in calls], ["fixed_" + n for n in E.SCENARIOS if not n.startswith("4v4_")])
        for bad_size, names in ((3, None), (4, ["sm2_vs_ge"])):
            with self.assertRaises(ValueError):
                E.exam(self.run_dir, {"team_size": bad_size}, names)
        with self.assertRaises(SystemExit):
            E.main(["--run-dir", self.run_dir, "--env-config", '{"team_size": 3}'])

    def test_a_4v4_scenario_plays_with_every_policy_plane_marked_ai(self):
        actor = E.load_actor(self.new)
        env = dict(team_size=4, time_limit_s=3)
        for opponent, ai_teams in ((None, {0}), (actor, {0, 1})):
            out = os.path.join(self.tmp.name, "4v4.jsonl")
            res = E.play(actor, "4v4_a", E.SCENARIOS["4v4_a"], env, out, opponent=opponent, exam_set="fixed")
            with open(out) as f:
                header = json.loads(f.readline())
            planes = header["planes"]
            self.assertEqual([p["team"] for p in planes], [0] * 4 + [1] * 4)
            self.assertEqual([p["aircraft"] for p in planes],
                             [m["aircraft"] for t in E.SCENARIOS["4v4_a"]["teams"] for m in t])
            self.assertEqual([p["archetype"] == "AI" for p in planes], [p["team"] in ai_teams for p in planes])
            self.assertEqual((res["team_size"], res["result"], res["own_deaths"], res["policy_deaths"]),
                             (4, "timeout", 0, 0))
            self.assertEqual(res["reward"], 0.0)
            self.assertEqual("opponent_kills" in res, opponent is not None)
        self.assertEqual(planes[4]["skill"], "policy")


if __name__ == "__main__":
    unittest.main()
