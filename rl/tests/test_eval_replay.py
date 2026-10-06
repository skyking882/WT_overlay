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


if __name__ == "__main__":
    unittest.main()
