"""Held-out evaluation: league-opponent detection (--require-held-out-opponent), held-out aircraft (env.held_out in
training, --held-out in rl.eval_replay --stats, the *_heldout exam scenarios), the --stats default and the per-head
action frequencies of the training metrics (rl.rollout.action_stats).

The --stats games use the stand-ins of test_eval_replay (ScriptedMatch, TeamMatch), so they run in milliseconds; two
exam replays use the real MatchEnv for 3 s of match time.
"""
import contextlib
import io
import json
import os
import random
import types
import unittest

import common  # noqa: F401
import torch

from rl import config as C
from rl import eval_replay as E
from rl import spec
from rl import train as T
from rl.buffer import RoundBuffer
from rl.model import Actor
from rl.ppo import actor_window
from rl.rollout import action_stats
from test_eval_replay import TEAM2, Base, ScriptedMatch, TeamMatch, save_ckpt
from test_late_credit import duel_cfg
from test_rollout import make

HELD = list(C.HELD_OUT_AIRCRAFT)


def model_weights():
    from wt_overlay import match as M
    return M.load_model()["aircraft_frequency"]["weights"]


def save_full(path, seed, cfg=None, round_no=None, **extra):
    """A checkpoint with an actor (weights from ``seed``), optionally the whole training cfg and the trainer round."""
    torch.manual_seed(seed)
    payload = dict({"actor": Actor().state_dict()}, **extra)
    if cfg is not None:
        payload["cfg"] = cfg
    if round_no is not None:
        payload["trainer"] = {"round": round_no}
    torch.save(payload, path)


def run_cfg(references=(), history_prob=0.25, snapshot_every=10, env=None):
    return {"name": "t", "env": {"config": dict(env or {"team_size": 1, "time_limit_s": 300},
                                                 **({"history_prob": history_prob} if history_prob else {}))},
            "league": {"snapshot_every": snapshot_every, "keep": 8, "references": list(references)}}


class Recorded(ScriptedMatch):
    """ScriptedMatch that keeps every config it was built with."""
    made = []

    def __init__(self, config, seed):
        super().__init__(config, seed)
        Recorded.made.append(config)


# ---------------------------------------------------------------------------------------------------------------
# league opponents
# ---------------------------------------------------------------------------------------------------------------

class LeagueOpponent(Base):
    def setUp(self):
        super().setUp()
        self.run = os.path.join(self.tmp.name, "run")
        os.makedirs(os.path.join(self.run, "ppo"))
        os.makedirs(os.path.join(self.run, "league"))
        self.ref = os.path.join(self.tmp.name, "refs", "s1_r304.pt")
        os.makedirs(os.path.dirname(self.ref))
        save_full(self.ref, 11, round_no=304)
        self.ckpt = os.path.join(self.run, "ppo", "ckpt_000040.pt")
        self.write_cfg([self.ref])

    def write_cfg(self, refs, **kw):
        cfg = run_cfg(refs, **kw)
        with open(os.path.join(self.run, "config.json"), "w") as f:
            json.dump(cfg, f)
        save_full(self.ckpt, 40, cfg, 40)

    def matches(self, opp):
        return E.league_opponent_matches(opp, self.run, self.ckpt)

    def test_a_reference_is_found_by_path_by_weights_and_by_name(self):
        self.assertEqual(self.matches(self.ref), [dict(member=self.ref, by="path")])
        # the same weights packed differently (rl.league's copy, a renamed file)
        twin = os.path.join(self.tmp.name, "elsewhere.pt")
        torch.save({"actor": torch.load(self.ref, weights_only=False)["actor"], "source": "x"}, twin)
        self.assertEqual(self.matches(twin), [dict(member=self.ref, by="weights")])
        # a relative entry from the training machine: here neither it nor a copy -> the file name decides
        self.write_cfg(["refs_on_cluster/s1_r304.pt"])
        other = os.path.join(self.tmp.name, "other", "s1_r304.pt")
        os.makedirs(os.path.dirname(other))
        save_full(other, 12)
        self.assertEqual(self.matches(other), [dict(member="refs_on_cluster/s1_r304.pt", by="basename")])
        # a copy in league/ with other weights settles it: not that reference
        save_full(os.path.join(self.run, "league", "ref0_s1_r304.pt"), 13)
        self.assertEqual(self.matches(other), [])
        save_full(os.path.join(self.run, "league", "ref0_s1_r304.pt"), 12)
        self.assertEqual(self.matches(other), [dict(member="refs_on_cluster/s1_r304.pt", by="weights")])
        # the checkpoint's stored cfg counts too (config.json gone)
        os.remove(os.path.join(self.run, "config.json"))
        self.assertEqual(self.matches(other), [dict(member="refs_on_cluster/s1_r304.pt", by="weights")])

    def test_snapshots_by_weights_and_by_round(self):
        old = os.path.join(self.run, "ppo", "ckpt_000030.pt")
        save_full(old, 30, round_no=30)
        torch.save({"actor": torch.load(old, weights_only=False)["actor"], "round": 30},
                   os.path.join(self.run, "league", "snap_r000030.pt"))
        self.assertEqual(self.matches(old), [dict(member="league/snap_r000030.pt", by="weights"),
                                             dict(member="snapshot r000030", by="snapshot")])
        # pruned from the pool long ago: a checkpoint at a snapshot round is still a past training opponent
        older = os.path.join(self.run, "ppo", "ckpt_000020.pt")
        save_full(older, 20, round_no=20)
        self.assertEqual(self.matches(older), [dict(member="snapshot r000020", by="snapshot")])
        save_full(older, 20, round_no=25)                                       # no snapshot round
        self.assertEqual(self.matches(older), [])
        save_full(older, 20, round_no=50)                                       # after the evaluated round
        self.assertEqual(self.matches(older), [])
        self.write_cfg([self.ref], history_prob=0)                              # no league episodes at all
        save_full(older, 20, round_no=20)
        self.assertEqual(self.matches(older), [])

    def test_an_unrelated_checkpoint_is_not_in_the_league(self):
        self.assertEqual(self.matches(self.old), [])
        self.assertEqual(E.league_opponent_matches(self.old, self.tmp.name, None), [])     # no config anywhere

    def cli(self, opp, *extra):
        import wt_overlay.rl_env as rl_env
        self.patch(rl_env, "MatchEnv", ScriptedMatch)
        self.in_process_pool()
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = E.main(["--run-dir", self.run, "--checkpoint", self.ckpt, "--procs", "1", "--stats", "2",
                           "--opponent-checkpoint", opp, "--env-config", '{"script": "crash"}'] + list(extra))
        return code, (json.loads(out.getvalue()) if out.getvalue() else None), err.getvalue()

    def test_cli_tags_warns_and_refuses(self):
        code, res, err = self.cli(self.ref)
        self.assertEqual(code, 0)
        self.assertIn("training opponent", err)
        self.assertEqual((res["opponent_in_league"], res["opponent_league_matches"]),
                         (True, [dict(member=self.ref, by="path")]))
        played = []
        self.patch(E, "stats", lambda *a, **k: played.append(a) or {})
        with self.assertRaises(SystemExit):
            self.cli(self.ref, "--require-held-out-opponent")
        self.assertEqual(played, [])                                            # refused before any game

    def test_cli_with_an_opponent_outside_the_league(self):
        code, res, err = self.cli(self.old)
        self.assertNotIn("opponent_in_league", res)                             # output as before
        self.assertNotIn("training opponent", err)
        code, res, _ = self.cli(self.old, "--require-held-out-opponent")
        self.assertEqual((code, res["opponent_in_league"], res["episodes"]), (0, False, 2))
        with self.assertRaises(SystemExit):                                     # needs an opponent
            E.main(["--run-dir", self.run, "--stats", "2", "--require-held-out-opponent"])


# ---------------------------------------------------------------------------------------------------------------
# held-out aircraft: training config
# ---------------------------------------------------------------------------------------------------------------

class HeldOutConfig(unittest.TestCase):
    def test_ids_and_resolution(self):
        self.assertEqual(C.held_out_ids(True), HELD)
        self.assertEqual([C.held_out_ids(v) for v in (False, None, [])], [[], [], []])
        self.assertEqual(C.held_out_ids(["mig_35"]), ["mig_35"])
        with self.assertRaises(ValueError):
            C.held_out_ids("mig_35")
        pool = ["su_30sm2", "saab_jas39e", "j_16", "mig_35"]
        cfg = C.preset("smoke")
        C.resolve_held_out(cfg, pool)                                           # off: nothing changes
        self.assertNotIn("aircraft_pool", cfg.env.config)
        cfg.env.held_out = True
        with self.assertRaises(ValueError):                                     # not resolved yet
            cfg.validate()
        C.resolve_held_out(cfg, pool)
        self.assertEqual(cfg.env.config["aircraft_pool"], ["su_30sm2", "j_16"])
        cfg.validate()
        C.resolve_held_out(cfg, pool)                                           # idempotent (resume)
        self.assertEqual(cfg.env.config["aircraft_pool"], ["su_30sm2", "j_16"])
        cfg.env.config["aircraft_pool"] = ["j_16", "mig_35"]                    # an explicit pool keeps its order
        C.resolve_held_out(cfg, pool)
        self.assertEqual(cfg.env.config["aircraft_pool"], ["j_16"])
        cfg.env.config["aircraft_pool"].append("mig_35")
        with self.assertRaises(ValueError):
            cfg.validate()
        cfg.env.config = {"teams": [[{"aircraft": "saab_jas39e"}], ["j_16"]], "aircraft_pool": ["j_16"]}
        with self.assertRaises(ValueError):
            cfg.validate()
        for bad in (["nope"], ["saab_jas39e", "mig_35", "su_30sm2", "j_16"]):    # unknown id / nothing left
            cfg = C.preset("smoke")
            cfg.env.held_out = bad
            with self.assertRaises(ValueError):
                C.resolve_held_out(cfg, pool)

    def test_set_flag_on_the_training_cli_stores_the_explicit_pool(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(T.main(["show-config", "--set", "env.held_out=True"]), 0)
        env = json.loads(out.getvalue())["env"]
        self.assertEqual(env["held_out"], True)
        self.assertEqual(env["config"]["aircraft_pool"], [a for a in model_weights() if a not in HELD])


# ---------------------------------------------------------------------------------------------------------------
# held-out aircraft: --stats
# ---------------------------------------------------------------------------------------------------------------

class HeldOutStats(Base):
    def setUp(self):
        super().setUp()
        import wt_overlay.rl_env as rl_env
        Recorded.made = []
        TeamMatch.made = []
        self.rl_env = rl_env
        self.patch(rl_env, "MatchEnv", Recorded)

    def test_1v1_policy_flies_a_held_out_aircraft_the_rest_is_the_in_pool_game(self):
        w = model_weights()
        flown = set()
        for i in range(12):
            base = E._stats_worker((self.new, dict(script="crash"), "top", i, False))
            # the in-pool draw is the one from before held-out existed
            self.assertEqual((base["aircraft"], base["opponent"]),
                             tuple(random.Random("stats:%d" % i).choices(list(w), list(w.values()), k=2)))
            r = E._stats_worker((self.new, dict(script="crash"), "top", i, False, None, False, None, tuple(HELD)))
            teams = Recorded.made[-1]["teams"]
            self.assertIn(r["aircraft"], HELD)
            self.assertEqual(teams[0], [{"aircraft": r["aircraft"]}])
            self.assertEqual((r["opponent"], teams[1]), (base["opponent"], Recorded.made[-2]["teams"][1]))
            flown.add(r["aircraft"])
        self.assertEqual(flown, set(HELD))                                      # uniform over the held-out ids
        with self.assertRaises(ValueError):                                     # never paired
            E._stats_worker((self.new, dict(script="crash"), "top", 0, False, self.old, True, None, tuple(HELD)))

    def test_aircraft_pool_is_drawn_from_and_dropped_only_for_held_out_games(self):
        pool = [a for a in model_weights() if a not in HELD]
        env = dict(script="crash", aircraft_pool=pool)
        for i in range(20):
            base = E._stats_worker((self.new, env, "top", i, False))
            self.assertEqual(Recorded.made[-1]["aircraft_pool"], pool)          # kept: the training env config
            self.assertTrue({base["aircraft"], base["opponent"]} <= set(pool))
            r = E._stats_worker((self.new, env, "top", i, False, None, False, None, tuple(HELD)))
            self.assertNotIn("aircraft_pool", Recorded.made[-1])
            self.assertEqual(r["opponent"], base["opponent"])                   # from the training pool
        cfg, _ = E.env_config_for(self.new, {"aircraft_pool": None})            # null switches it off
        self.assertNotIn("aircraft_pool", cfg)

    def test_team_games_policy_slots_fly_held_out_aircraft(self):
        self.patch(self.rl_env, "MatchEnv", TeamMatch)
        for i in range(6):
            for control in (None, 1):
                job = (self.new, TEAM2, "top", i, False, None, False, control)
                base = E._stats_worker(job)
                held = E._stats_worker(job + (tuple(HELD),))
                t_base, t_held = TeamMatch.made[-2].cfg["teams"], TeamMatch.made[-1].cfg["teams"]
                k = 2 if control is None else 1
                self.assertTrue(all(m["aircraft"] in HELD for m in t_held[0][:k]))
                self.assertEqual(held["aircraft"][:k], [m["aircraft"] for m in t_held[0][:k]])
                self.assertEqual(t_held[0][k:], t_base[0][k:])                  # script teammate keeps its draw
                self.assertEqual((t_held[1], held["opponent"]), (t_base[1], base["opponent"]))
                self.assertEqual(held["slots"], list(range(k)))
        self.in_process_pool()
        res = E.stats(self.new, TEAM2, 3, "top", 1, held_out=HELD)
        self.assertEqual((res["episodes"], res["held_out"], res["team_size"]), (3, HELD, 2))
        self.assertNotIn("by_aircraft", res)

    def test_stats_summary_by_aircraft_and_paired_refused(self):
        self.in_process_pool()
        res = E.stats(self.new, dict(script="crash"), 6, "top", 1, held_out=HELD)
        self.assertEqual(res["held_out"], HELD)
        self.assertTrue(set(res["by_aircraft"]) <= set(HELD))
        self.assertEqual(sum(v["episodes"] for v in res["by_aircraft"].values()), 6)
        self.assertEqual(sum(v["win"] for v in res["by_aircraft"].values()), res["win"])
        plain = E.stats(self.new, dict(script="crash"), 6, "top", 1)
        self.assertFalse({"held_out", "by_aircraft"} & set(plain))
        with self.assertRaises(ValueError):
            E.stats(self.new, dict(script="crash"), 1, "top", 1, opponent_path=self.old, paired=True, held_out=HELD)

    def cli(self, ckpt, *extra):
        self.in_process_pool()
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = E.main(["--run-dir", self.tmp.name, "--checkpoint", ckpt, "--procs", "1", "--stats", "4",
                           "--env-config", '{"script": "crash"}'] + list(extra))
        return code, json.loads(out.getvalue()), err.getvalue()

    def test_cli_held_out_and_seen_in_training(self):
        code, res, err = self.cli(self.new, "--held-out")                       # no stored config: all seen
        self.assertEqual((code, res["held_out"], res["held_out_seen_in_training"]), (0, HELD, HELD))
        self.assertIn("no held-out test", err)
        trained = os.path.join(self.tmp.name, "ckpt_000010.pt")
        pool = [a for a in model_weights() if a not in HELD]
        save_full(trained, 3, {"env": {"config": {"team_size": 1, "aircraft_pool": pool}, "held_out": ["mig_35"]}})
        code, res, err = self.cli(trained, "--held-out")                        # the checkpoint's own held-out ids
        self.assertEqual((res["held_out"], res["held_out_seen_in_training"], err), (["mig_35"], [], ""))
        self.assertEqual(set(res["by_aircraft"]), {"mig_35"})
        code, res, _ = self.cli(trained, "--held-out", "saab_jas39e,mig_35")
        self.assertEqual((res["held_out"], res["held_out_seen_in_training"]), (HELD, []))
        for bad in (["--held-out", "nope"], ["--held-out", "--paired", "--opponent-checkpoint", self.old]):
            with self.assertRaises(SystemExit):
                self.cli(trained, *bad)
        with self.assertRaises(SystemExit):                                     # only with --stats
            E.main(["--run-dir", self.tmp.name, "--held-out"])

    def test_stats_without_n_plays_200_games(self):
        seen = []
        self.patch(E, "stats", lambda path, env, n, *a, **k: seen.append(n) or {})
        base = ["--run-dir", self.tmp.name, "--checkpoint", self.new, "--env-config", '{"script": "crash"}']
        with contextlib.redirect_stdout(io.StringIO()):
            E.main(base + ["--stats"])
            E.main(base + ["--stats", "--paired", "--opponent-checkpoint", self.old])
            E.main(base + ["--stats", "7"])
        self.assertEqual(seen, [200, 100, 7])


# ---------------------------------------------------------------------------------------------------------------
# held-out exam scenarios
# ---------------------------------------------------------------------------------------------------------------

class HeldOutExams(Base):
    def test_scenarios_are_labelled(self):
        self.assertNotIn("4v4_c", E.SCENARIOS)
        self.assertEqual(E.held_out_of(E.SCENARIOS["4v4_c_heldout"]), ["saab_jas39e"])
        self.assertEqual(E.exam_scenarios(4, ["4v4_c"]), ["4v4_c_heldout"])     # the old name still works
        self.assertEqual({n: E.held_out_of(s) for n, s in E.SCENARIOS.items() if E.held_out_of(s)},
                         {"jas39e_vs_sm2_heldout": ["saab_jas39e"], "mig35_vs_ge_heldout": ["mig_35"],
                          "4v4_c_heldout": ["saab_jas39e"]})
        for name, s in E.SCENARIOS.items():
            self.assertEqual(name.endswith("_heldout"), bool(E.held_out_of(s)), name)
            self.assertFalse({m["aircraft"] for m in s["teams"][1]} & set(HELD), name)   # opponents trained-on
        for name in ("jas39e_vs_sm2_heldout", "mig35_vs_ge_heldout"):
            self.assertEqual((E.team_size_of(E.SCENARIOS[name]), E.SCENARIOS[name]["teams"][1][0]["skill"]), (1, "top"))
        self.assertEqual(len({s["seed"] for s in E.SCENARIOS.values()}), len(E.SCENARIOS))

    def test_replay_and_row_carry_the_label(self):
        actor = E.load_actor(self.new)
        env = dict(team_size=1, time_limit_s=3)
        for name, held in (("jas39e_vs_sm2_heldout", ["saab_jas39e"]), ("sm2_vs_ge", None)):
            out = os.path.join(self.tmp.name, name + ".jsonl")
            res = E.play(actor, "fixed_" + name, E.SCENARIOS[name], env, out, exam_set="fixed")
            with open(out) as f:
                header = json.loads(f.readline())
            self.assertEqual((res.get("held_out"), header["exam"].get("held_out")), (held, held))
            self.assertEqual(header["planes"][0]["aircraft"], E.SCENARIOS[name]["teams"][0][0]["aircraft"])

    def test_cli_accepts_the_old_name(self):
        run = os.path.join(self.tmp.name, "run")
        os.makedirs(os.path.join(run, "ppo"))
        save_ckpt(os.path.join(run, "ppo", "ckpt_000002.pt"), 3, dict(team_size=4, time_limit_s=3))
        names = []
        self.patch(E, "play", lambda actor, name, *a, **k: names.append(name) or dict(scenario=name))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(E.main(["--run-dir", run, "--scenarios", "4v4_c", "--sets", "fixed"]), 0)
        self.assertEqual(names, ["fixed_4v4_c_heldout"])


# ---------------------------------------------------------------------------------------------------------------
# per-head action frequencies
# ---------------------------------------------------------------------------------------------------------------

def brute_force(buf):
    """action_freq / action_freq_pointer by plain loops over the valid loss-region steps."""
    S, P = buf.S, buf.P
    steps = [f for f in range(P * S, buf.store.N) if bool(buf.store.valid[f])]
    cat = {}
    for h in spec.CAT_HEADS:
        c = [0] * spec.CAT_SIZES[h]
        for f in steps:
            c[int(buf.store.act[f, spec.HEAD_INDEX[h]])] += 1
        cat[h] = [x / len(steps) for x in c]
    ptr = {}
    for h in spec.POINTER_HEADS:
        none = sum(1 for f in steps if int(buf.store.act[f, spec.HEAD_INDEX[h]]) == int(buf.store.ent_n[f]))
        ptr[h] = {"entity": 1 - none / len(steps), "none": none / len(steps)}
    return cat, ptr


class ActionFreq(unittest.TestCase):
    def test_hand_computed_round(self):
        buf = RoundBuffer(2, 2, 2, 1)                     # S 2, T 2, L 2, B 1: flat 0-1 burn-in history, 2-5 loss
        buf.n_legal = torch.zeros(buf.n_steps, spec.N_HEADS, dtype=torch.uint8)
        st = buf.store
        M, W, TG = (spec.HEAD_INDEX[h] for h in ("maneuver", "weapon", "target"))
        rows = {0: (9, 1, 0, 2), 1: (9, 1, 0, 2),        # burn-in history of the previous round: not counted
                2: (0, 1, 3, 3), 3: (0, 0, 1, 2), 4: (7, 1, 0, 5), 5: (4, 0, 0, 0)}   # (maneuver, weapon, target, n)
        legal = {2: (11, 2, 4), 3: (1, 2, 1), 4: (11, 2, 6), 5: (3, 1, 1)}           # (k maneuver, weapon, target)
        for f, (m, w, t, n) in rows.items():
            st.act[f, M], st.act[f, W], st.act[f, TG], st.ent_n[f] = m, w, t, n
            st.valid[f] = f != 4                          # 4: padding (e.g. the frozen side of a history episode)
        for f, (km, kw, kt) in legal.items():
            buf.n_legal[f, M], buf.n_legal[f, W], buf.n_legal[f, TG] = km, kw, kt
        s = action_stats(buf)
        third = round(1 / 3, 5)
        self.assertEqual(s["action_freq"]["maneuver"], [round(2 / 3, 5), 0, 0, 0, third] + [0] * 6)
        self.assertEqual(s["action_freq"]["weapon"], [round(2 / 3, 5), third])
        self.assertEqual(s["action_freq"]["speed"], [1.0, 0.0, 0.0])               # never moved off option 0
        self.assertEqual(set(s["action_freq"]), set(spec.CAT_HEADS))
        # target: none at flat 2 (3 of 3) and 5 (0 of 0 entities), an entity at 3
        self.assertEqual(s["action_freq_pointer"]["target"], {"entity": third, "none": round(2 / 3, 5)})
        self.assertEqual(s["action_freq_pointer"]["maneuver_ref"], {"entity": round(2 / 3, 5), "none": third})
        # only the steps with a choice: maneuver at 2 and 5, weapon at 2 and 3, target at 2
        self.assertEqual(s["action_freq_active"], {"maneuver": [0.5, 0, 0, 0, 0.5] + [0] * 6, "weapon": [0.5, 0.5]})
        self.assertEqual(s["action_freq_pointer_active"], {"target": {"entity": 0.0, "none": 1.0}})
        frac = s["head_active_frac_round"]
        self.assertEqual((frac["maneuver"], frac["weapon"], frac["target"], frac["speed"]),
                         (round(2 / 3, 5), round(2 / 3, 5), third, 0.0))
        for d in (s["action_freq"], s["action_freq_active"]):
            for h, p in d.items():
                self.assertEqual(len(p), spec.CAT_SIZES[h])
                self.assertAlmostEqual(sum(p), 1.0, places=4)
        del buf.n_legal                                   # a buffer without the sampler's table: no *_active keys
        self.assertEqual(set(action_stats(buf)), {"action_freq", "action_freq_pointer"})
        st.valid[:] = False
        self.assertEqual(action_stats(buf), {})

    def test_sampled_round_of_the_fake_env(self):
        cfg, actor, critic, sm = make(steps=24, streams=8, peaked=True)
        for _ in range(2):                                # the second round has burn-in history in front
            buf, st = sm.collect(actor)
            sm.finish(buf, critic, None, 0.995, 0.95)
        self.assertLess(buf.n_valid(), buf.S * buf.T)     # some padding in the loss region
        s = action_stats(buf, places=12)
        cat, ptr = brute_force(buf)
        for h in spec.CAT_HEADS:
            self.assertEqual(len(s["action_freq"][h]), spec.CAT_SIZES[h])
            self.assertAlmostEqual(sum(s["action_freq"][h]), 1.0, places=9)
            for a, b in zip(s["action_freq"][h], cat[h]):
                self.assertAlmostEqual(a, b, places=9)
        for h in spec.POINTER_HEADS:
            for key in ("entity", "none"):
                self.assertAlmostEqual(s["action_freq_pointer"][h][key], ptr[h][key], places=9)
        # padding ignored: garbage in the invalid steps changes nothing
        inv = ~buf.store.valid
        buf.store.act[inv] = 1
        buf.n_legal[inv] = 9
        self.assertEqual(action_stats(buf, places=12), s)
        # n_legal is the sampling-time number of legal options (the actor's HeadOut.k on the stored steps)
        for k in range(buf.K):
            for strm in range(buf.S):
                widx = buf.window_index(torch.tensor([strm]), torch.tensor([k]))
                b = buf.store.gather(widx)
                out, seg = actor_window(actor, b, buf.h_actor[k][strm].unsqueeze(0), buf.B, b.act[:, buf.B:],
                                        grad=False)
                m = seg.valid.reshape(-1)
                self.assertTrue(torch.equal(out.k[m].to(torch.uint8), buf.n_legal[widx[0, buf.B:]][m]))
        # and the training record carries it
        rec = T.round_record(cfg, types.SimpleNamespace(round=1, decisions=10), buf, st, {}, sm)
        for key in ("action_freq", "action_freq_pointer", "action_freq_active", "head_active_frac_round"):
            self.assertIn(key, rec)

    def test_settle_ticks_extend_the_legal_option_table(self):
        cfg, actor, critic, sm = make(streams=4, steps=8, seg=8, burn=4, cfg_fn=duel_cfg(6, 11))
        buf, st = sm.collect(actor)
        self.assertEqual(buf.T, 16)                       # settle ticks appended
        self.assertEqual(buf.n_legal.shape[0], buf.store.N)
        settle = buf.store.valid.clone()
        settle[:(buf.P + 8) * buf.S] = False
        self.assertTrue(settle.any())
        self.assertTrue(bool((buf.n_legal[settle] > 0).all()))
        self.assertIn("action_freq", T.round_record(cfg, types.SimpleNamespace(round=1, decisions=0), buf, st, {}, sm))


if __name__ == "__main__":
    unittest.main()
