"""Exam replays: the current policy plays fixed matchups, recorded for the dashboard.

Runs beside training and never touches the run's checkpoints or metrics. It loads the newest
ppo/ckpt_*.pt (or bc_actor.pt before PPO), plays every scenario once (actions sampled with a seeded
generator, as in training) in the real MatchEnv, records each match with wt_overlay.engagement.ReplayWriter as
<run_dir>/replays/r<round>_<set>_<scenario>.jsonl and appends a summary line to
<run_dir>/replays/exams.jsonl. The same seed per scenario every time, so one matchup can be
compared across rounds. Policy aircraft show archetype "AI" in the replay header.

Environment: exams and --stats use the env config stored in the checkpoint (cfg.env.config, the one training used:
timeout_reward, observation_frame, spawn_layout, ...); --env-config '{...}' overrides only the keys it names (null
switches an option off). A checkpoint without a stored config (bc_actor.pt) starts from {team_size 1, time_limit_s
420} as before. self_play_prob, history_prob, policy_ids and policy_count are dropped: the evaluation decides which
slots a policy flies. The config's team_size (default 1) sets the match size of exams and --stats.

Exam sets (--sets, default both); the scenarios (--scenarios, default all of the config's team size: six 1v1s, three
4v4s with mixed top-tier aircraft) have the policy flying all of team 0. Scenarios whose team 0 flies a held-out
aircraft (rl.config.HELD_OUT_AIRCRAFT: jas39e_vs_sm2_heldout, mig35_vs_ge_heldout, 4v4_c_heldout; the old name 4v4_c
still selects the last) are named *_heldout and their result rows / replay headers carry "held_out": [aircraft]; they
test generalisation only for a checkpoint trained without those aircraft (env.held_out):
  fixed  r<round>_fixed_<scenario>.jsonl  the scripts without script_perturbation: comparable over a whole run and
                                          across runs (long-term trend)
  train  r<round>_train_<scenario>.jsonl  the scripts with the stored script_perturbation, as in training; skipped
                                          when the config has none (it would replay the fixed matches)
  sp     r<round>_sp_<scenario>.jsonl     with --self-play: every scenario once more with the same checkpoint flying
                                          the other team as well (every aircraft marked "AI")
--watch repeats whenever the newest checkpoint is at least --every rounds past the last exam; replays of the latest
--keep exams and of every --milestone-th round are kept.

    python -m rl.eval_replay --run-dir ~/rl_runs/s1_v1 --watch --every 10

--stats N plays N random 1v1s and reports win / loss / trade / timeout of the policy, win rate, exchange and
mean_return (the rewards the env hands the policy's slot over the whole match, timeout_reward and late rewards
included: the return training sees). --stats without N: 200 games (with --paired 100 pairs, again 200 games). The
standard error of a win rate p over n independent games is sqrt(p (1 - p) / n), at p = 0.5: 6.5 pp at n = 60 (95 %
interval +-12.7 pp), 4.6 pp at 120 (+-8.9 pp), 3.5 pp at 200 (+-6.9 pp); the two games of a pair are not
independent, so count paired results by pairs to be safe. The aircraft are drawn by match frequency from the model's
pool, restricted to the env config's aircraft_pool when the checkpoint was trained with one (env.held_out): in-pool
numbers are measured on the training pool.
Slot 1 is a script by default; with --opponent-checkpoint PATH both slots are policy-controlled and the other one is
flown by that frozen policy (own seeded generator), i.e. new versus old. Recommended reference evaluation: --paired
plays every seed twice with the two policies swapping slots (and so spawn sides and aircraft); the counts are the new
policy's over all 2N games, by_slot splits them by the slot it flew. About 100 pairs (200 games):

    python -m rl.eval_replay --run-dir RUN --stats 100 --paired --opponent-checkpoint OLD/ppo/ckpt_000100.pt

The reference must not be a training opponent: an --opponent-checkpoint that is (or was) in the run's league (listed
in league.references of RUN/config.json or of the checkpoint's stored cfg, the same weights as a file in
RUN/league/, or a checkpoint of this run at a league snapshot round) prints a warning and tags the JSON with
"opponent_in_league": true plus "opponent_league_matches"; --require-held-out-opponent refuses to play then (and
writes "opponent_in_league": false otherwise). Keep the reference outside league.references, e.g. an older
checkpoint copied away before the league ran, or one of another run.

Held-out aircraft (--held-out [IDS], comma separated; default the checkpoint's env.held_out ids, else
rl.config.HELD_OUT_AIRCRAFT): the aircraft the policy flies are drawn uniformly from those ids (own generator per game),
everything else exactly as in the in-pool game of the same index (opponent / teammate aircraft by frequency from the
training pool, env seed, script skills), so game i differs from in-pool game i only in the policy's aircraft. The env
gets the config without aircraft_pool (it would not list the aircraft flown); the draw of the other aircraft still
uses it (--env-config '{"aircraft_pool": null}' draws them from the whole model pool instead). Adds "held_out",
"held_out_seen_in_training" (held-out ids the checkpoint's training pool contained: no aircraft_pool stored, or listed
in it; warned on stderr: then it is no held-out test) and, in 1v1, "by_aircraft". Team games: the K policy-flown slots
fly held-out aircraft, the script teammates keep their draw. Excludes --paired (its swap would hand the held-out
aircraft to the opponent).

    python -m rl.eval_replay --run-dir RUN --stats 200 --held-out --opp-skill top

Teams (team_size n > 1): every game draws n aircraft per team from the BR pool (the same frequency weights). The policy
flies its team, all n aircraft by default; --team-control K: only the first K, the env's scripts fly the rest (the
execution path training gives them). The other team is scripts (--opp-skill) or, with --opponent-checkpoint, flown
entirely by that policy; --paired swaps the teams (by_team splits the counts). --scripted: the scripts decide for the
slots the policy would fly. A game is a win when only the policy's team has survivors, a loss when only the enemy's,
a trade when neither, else a timeout; exchange = enemy deaths / own deaths (any cause). Kills, launches, deaths,
survival, crashes, friendly-fire kills and the return count only the policy-flown aircraft (mean_return: their summed
return per game). Every policy-flown slot samples with its own generator (seed: game index + slot x
OPPONENT_SEED_BASE, the 1v1 seeds for slots 0 and 1).

    python -m rl.eval_replay --run-dir RUN --stats 200 --opp-skill top                         # policy flies all 4
    python -m rl.eval_replay --run-dir RUN --stats 200 --opp-skill top --team-control 1 [--scripted]
    python -m rl.eval_replay --run-dir RUN --stats 100 --paired --opponent-checkpoint OLD/ppo/ckpt_000100.pt
    python -m rl.eval_replay --run-dir RUN --stats 200 --opp-skill top --held-out              # held-out team

A periodic stats loop (remote) should run per evaluated checkpoint: the in-pool games (--stats 200 --opp-skill top),
the same with --held-out (only meaningful for a run trained with env.held_out), and the paired reference
(--stats 100 --paired --opponent-checkpoint REF --require-held-out-opponent, REF outside league.references), all with
--checkpoint pinned to the same file (docs/rl_training_spec.md, evaluation).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from rl import spec, wire  # noqa: E402
from rl.config import HELD_OUT_AIRCRAFT, held_out_ids  # noqa: E402
from rl.encode import Decoded  # noqa: E402
from rl.model import Actor, H  # noqa: E402

DEFAULT_STATS_GAMES = 200           # --stats without N (with --paired: half as many pairs)
OPPONENT_SEED_BASE = 1_000_003      # --stats: generator seed of slot 1 is this + episode index (slot 0: the index)
DEFAULT_ENV = dict(team_size=1, time_limit_s=420)   # env config of a checkpoint that stores none (bc_actor.pt)
EVAL_DROP = ("self_play_prob", "history_prob", "policy_ids", "policy_count", "team_size_mix")   # the evaluation picks the policy slots
PATH_KEYS = ("model_path", "reach_dir")              # training-machine paths a stored config may carry
EXAM_SETS = ("fixed", "train")
RESULTS = ("win", "loss", "trade", "timeout")

# Stage S1: the policy flies slot 0 against a scripted opponent (spawn separation from --range-km).
SCENARIOS = {
    "sm2_vs_ge": dict(seed=11, teams=[[{"aircraft": "su_30sm2"}],
                                      [{"aircraft": "f_15c_golden_eagle", "archetype": "left", "skill": "top"}]]),
    "ge_vs_sm2": dict(seed=12, teams=[[{"aircraft": "f_15c_golden_eagle"}],
                                      [{"aircraft": "su_30sm2", "archetype": "middle", "skill": "top"}]]),
    "typhoon_vs_j16": dict(seed=13, teams=[[{"aircraft": "ef_2000_typhoon_aesa"}],
                                           [{"aircraft": "j_16", "archetype": "left", "skill": "top"}]]),
    "f16cm_vs_j10c": dict(seed=14, teams=[[{"aircraft": "f_16c_block_52_aesa"}],
                                          [{"aircraft": "j_10c", "archetype": "middle", "skill": "normal"}]]),
    # held-out aircraft (rl.config.HELD_OUT_AIRCRAFT) on the policy side against top scripts of training aircraft
    "jas39e_vs_sm2_heldout": dict(seed=15, teams=[[{"aircraft": "saab_jas39e"}],
                                                  [{"aircraft": "su_30sm2", "archetype": "left", "skill": "top"}]]),
    "mig35_vs_ge_heldout": dict(seed=16, teams=[[{"aircraft": "mig_35"}],
                                                [{"aircraft": "f_15c_golden_eagle", "archetype": "middle",
                                                  "skill": "top"}]]),
}
# Team stage: the policy flies all of team 0 (mixed top tier) against four scripts of mixed archetype and skill.
SCENARIOS.update({
    "4v4_a": dict(seed=21, teams=[
        [{"aircraft": a} for a in ("su_30sm2", "f_15c_golden_eagle", "ef_2000_typhoon_aesa", "j_10c")],
        [{"aircraft": "j_16", "archetype": "left", "skill": "top"},
         {"aircraft": "f_15c_golden_eagle", "archetype": "left", "skill": "normal"},
         {"aircraft": "ef_2000_typhoon_aesa", "archetype": "middle", "skill": "top"},
         {"aircraft": "f_16c_block_52_aesa", "archetype": "middle", "skill": "normal"}]]),
    "4v4_b": dict(seed=22, teams=[
        [{"aircraft": a} for a in ("f_15c_golden_eagle", "j_16", "f_16c_block_52_aesa", "ef_2000_aesa")],
        [{"aircraft": "su_30sm2", "archetype": "left", "skill": "top"},
         {"aircraft": "su_30sm2", "archetype": "middle", "skill": "normal"},
         {"aircraft": "j_10c", "archetype": "middle", "skill": "top"},
         {"aircraft": "ef_2000_typhoon_aesa", "archetype": "right", "skill": "normal"}]]),
    # policy team with saab_jas39e (held out); called 4v4_c before 2026-10-07 (same seed and teams)
    "4v4_c_heldout": dict(seed=23, teams=[
        [{"aircraft": a} for a in ("ef_2000_typhoon_aesa", "su_30sm2", "saab_jas39e", "f_16c_block_52_aesa")],
        [{"aircraft": "f_15c_golden_eagle", "archetype": "left", "skill": "top"},
         {"aircraft": "j_16", "archetype": "left", "skill": "normal"},
         {"aircraft": "fa_18e_block_2", "archetype": "middle", "skill": "top"},
         {"aircraft": "j_10c", "archetype": "crawler", "skill": "normal"}]]),
})
SCENARIO_ALIASES = {"4v4_c": "4v4_c_heldout"}     # old names still accepted by --scenarios


def team_size_of(scenario):
    return len(scenario["teams"][0])


def held_out_of(scenario):
    """The held-out aircraft (rl.config.HELD_OUT_AIRCRAFT) team 0, the policy's team, flies in ``scenario``
    (sorted, each once); [] for an ordinary scenario."""
    return sorted({m["aircraft"] for m in scenario["teams"][0]} & set(HELD_OUT_AIRCRAFT))


def slot_seed(i, slot):
    """Generator seed of policy-flown ``slot`` in game / scenario seed ``i``: i, OPPONENT_SEED_BASE + i, ... (slots 0
    and 1 keep the 1v1 seeds)."""
    return i + slot * OPPONENT_SEED_BASE


def latest(run_dir):
    """(round, path) of the newest PPO checkpoint, (0, bc_actor.pt) before PPO, else None."""
    ckpts = sorted(glob.glob(os.path.join(run_dir, "ppo", "ckpt_*.pt")))
    if ckpts:
        return int(re.search(r"ckpt_(\d+)\.pt$", ckpts[-1]).group(1)), ckpts[-1]
    bc = os.path.join(run_dir, "bc_actor.pt")
    return (0, bc) if os.path.exists(bc) else None


def load_actor(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    actor = Actor()
    actor.load_state_dict(ck["actor"])
    return actor.eval()


def stored_env_config(path):
    """The env config a PPO checkpoint was trained with (cfg.env.config), or None (bc_actor.pt, older files)."""
    if not path:
        return None
    ck = torch.load(path, map_location="cpu", weights_only=False)
    try:
        env = ck["cfg"]["env"]["config"]
    except (KeyError, TypeError):
        return None
    return dict(env) if isinstance(env, dict) else None


def stored_cfg(path):
    """The whole training config a PPO checkpoint stores (cfg, a dict), or None."""
    if not path:
        return None
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck.get("cfg") if isinstance(ck, dict) else None
    return cfg if isinstance(cfg, dict) else None


def env_config_for(path, overrides=None):
    """(env config, source) for evaluating the checkpoint ``path``: its stored training config with ``overrides``
    (--env-config) on top, source "checkpoint"; without a stored config DEFAULT_ENV plus overrides, source "default".
    The keys of EVAL_DROP are removed. A path key that does not exist on this machine is reported on stderr.
    aircraft_pool (a run trained with env.held_out) is kept: the stats draws follow it (draw_pool); null removes it
    (MatchEnv tests the key's presence, not its value)."""
    stored = stored_env_config(path)
    cfg = dict(stored if stored is not None else DEFAULT_ENV)
    cfg.update(overrides or {})
    for k in EVAL_DROP:
        cfg.pop(k, None)
    if "aircraft_pool" in cfg and cfg["aircraft_pool"] is None:
        del cfg["aircraft_pool"]
    for k in PATH_KEYS:
        p = cfg.get(k)
        if isinstance(p, str) and not os.path.exists(os.path.expanduser(p)):
            print("warning: env config %s=%s does not exist here (override it with --env-config)" % (k, p),
                  file=sys.stderr)
    return cfg, ("checkpoint" if stored is not None else "default")


def exam_configs(env_config, sets=EXAM_SETS):
    """({set: env config} of the script exams, {skipped set: reason}). "fixed" plays the scenarios without
    script_perturbation, "train" with the stored one (the scripts the policy trains against); "train" is skipped
    when the config has no perturbation, since it would replay the fixed matches exactly."""
    configs, skipped = {}, {}
    for s in sets:
        if s == "fixed":
            configs[s] = {k: v for k, v in env_config.items() if k != "script_perturbation"}
        elif s == "train":
            if env_config.get("script_perturbation") is None:
                skipped[s] = "no script_perturbation in the env config: same matches as fixed"
            else:
                configs[s] = dict(env_config)
        else:
            raise ValueError("unknown exam set %r (sets: %s)" % (s, ", ".join(EXAM_SETS)))
    return configs, skipped


def add_rewards(acc, rewards, info):
    """Add one step's rewards per policy slot to ``acc``, with what a downed aircraft's missiles scored this step
    (info late_rewards): over a whole match this is the episode return training sees for that slot."""
    for aid, r in (rewards or {}).items():
        acc[aid] = acc.get(aid, 0.) + r
    for aid, r in ((info or {}).get("late_rewards") or {}).items():
        acc[aid] = acc.get(aid, 0.) + r
    return acc


def play(actor, name, scenario, env_config, out_path, range_km=100., greedy=False, opponent=None, exam_set=None):
    """One match recorded to out_path; returns its summary. Actions are sampled with a seeded generator (as in
    training): rare heads such as fire have a small per-step probability, so a greedy argmax never fires.
    ``actor`` flies all of team 0. With ``opponent`` (an Actor, possibly ``actor`` itself) team 1 is policy-flown too:
    a self-play match as in training. Every policy slot samples with its own generator (slot_seed). A scenario with
    held-out aircraft on team 0 (held_out_of) adds "held_out" to the summary and the replay header's exam record."""
    from wt_overlay.engagement import ReplayWriter
    from wt_overlay.rl_env import MatchEnv
    held = held_out_of(scenario)
    n0, total = team_size_of(scenario), sum(len(t) for t in scenario["teams"])
    own, other = list(range(n0)), list(range(n0, total))
    cfg = dict(env_config, teams=scenario["teams"], range_km=scenario.get("range_km", range_km),
               controlled=own + other if opponent is not None else own)
    env = MatchEnv(cfg, scenario["seed"])
    obs = env.reset()
    eng = env.engagement
    writer = ReplayWriter(out_path + ".part", keep=False)
    header = eng._header()
    for p in header["planes"]:
        if p["id"] in env.policy_ids:
            p.update(archetype="AI", skill="policy", controller="policy")
    header["exam"] = dict(scenario=name, seed=scenario["seed"], self_play=opponent is not None, set=exam_set)
    if held:
        header["exam"]["held_out"] = held
    writer.write(header)
    eng.replay = writer
    eng._frame()
    actors = {s: actor if s < n0 else opponent for s in own + other}
    gens = {s: torch.Generator().manual_seed(slot_seed(scenario["seed"], s)) for s in own + other}
    h = {aid: torch.zeros(1, H) for aid in obs}
    first = {aid: True for aid in obs}
    ret, steps = {}, 0
    t0 = time.time()
    while not env.over:
        acts = {}
        for aid, o in obs.items():
            batch = Decoded([wire.pack_obs(o)]).to_batch(first=torch.tensor([first[aid]]))
            out, h[aid] = actors[aid].act(batch, h[aid], gens[aid], greedy=greedy)
            acts[aid] = wire.unpack_action(tuple(out.actions[0].tolist()))
            first[aid] = False
        obs, rewards, dones, info = env.step(acts)
        add_rewards(ret, rewards, info)     # incl. what a slot's missiles score after it is down
        steps += 1
    writer.close()                      # the engagement already closed it at the end; close() is idempotent
    os.replace(out_path + ".part", out_path)
    if total > 2:
        t = team_tally(eng, 0, own)
        res = dict(scenario=name, set=exam_set, reason=eng.reason, time_s=round(eng.time, 1), team_size=n0,
                   result=team_outcome(eng, 0), own_deaths=t["own_deaths"], enemy_deaths=t["enemy_deaths"],
                   policy_kills=t["kills"], policy_launches=t["launches"], policy_deaths=t["policy_deaths"],
                   policy_crashes=t["crashes"], friendly_fire=t["friendly_fire"],
                   reward=round(sum(ret.get(s, 0.) for s in own), 2), decisions=steps,
                   wall_s=round(time.time()-t0, 1), replay=os.path.basename(out_path))
        if "landings" in t:             # airfield option
            res.update(policy_landings=t["landings"], policy_rearms=t["rearms"], grounded_at_end=t["grounded_at_end"])
        if opponent is not None:
            o = team_tally(eng, 1, other)
            res.update(self_play=True, opponent_reward=round(sum(ret.get(s, 0.) for s in other), 2),
                       opponent_kills=o["kills"], opponent_launches=o["launches"], opponent_deaths=o["policy_deaths"])
        if held:
            res["held_out"] = held
        return res
    policy = eng.planes[0]
    kills = sum(1 for e in eng.log if e["kind"] == "kill" and e.get("killer") == 0)
    launches = sum(1 for e in eng.log if e["kind"] == "launch" and e.get("shooter") == 0)
    res = dict(scenario=name, set=exam_set, reason=eng.reason, time_s=round(eng.time, 1), policy_alive=policy.alive,
               policy_kills=kills, policy_launches=launches, reward=round(ret.get(0, 0.), 2),
               decisions=steps, wall_s=round(time.time()-t0, 1), replay=os.path.basename(out_path))
    if opponent is not None:
        res.update(self_play=True, opponent_alive=eng.planes[1].alive, opponent_reward=round(ret.get(1, 0.), 2),
                   opponent_kills=sum(1 for e in eng.log if e["kind"] == "kill" and e.get("killer") == 1),
                   opponent_launches=sum(1 for e in eng.log if e["kind"] == "launch" and e.get("shooter") == 1))
    if held:
        res["held_out"] = held
    return res


def exam_scenarios(team_size, names=None):
    """The scenarios of ``names`` (default all; old names via SCENARIO_ALIASES) with ``team_size`` aircraft per team;
    explicitly named ones of another size are skipped with a note on stderr. ValueError when none is left."""
    pick = list(SCENARIOS) if names is None else [SCENARIO_ALIASES.get(n, n) for n in names]
    unknown = [n for n in pick if n not in SCENARIOS]
    if unknown:
        raise ValueError("unknown scenarios: %s" % unknown)
    fit = [n for n in pick if team_size_of(SCENARIOS[n]) == team_size]
    if not fit:
        raise ValueError("no exam scenario with team_size %d among %s (scenarios of that size: %s)" % (
            team_size, ", ".join(pick), ", ".join(n for n in SCENARIOS if team_size_of(SCENARIOS[n]) == team_size)
            or "none"))
    if names is not None and len(fit) < len(pick):
        print("note: skipping scenarios of another team size than %d: %s"
              % (team_size, ", ".join(n for n in pick if n not in fit)), file=sys.stderr)
    return fit


def exam(run_dir, env_overrides, names, range_km=100., self_play=False, sets=EXAM_SETS):
    """The script exam sets (exam_configs) of every scenario (``names``, None: all) of the env config's team size
    (exam_scenarios), in the checkpoint's own env config with ``env_overrides`` on top (env_config_for); with
    ``self_play`` also once against the same checkpoint flying the other team (set "sp", replay
    r<round>_sp_<scenario>.jsonl), the kind of match a self-play run trains on."""
    found = latest(run_dir)
    if found is None:
        return None
    rnd, path = found
    env_config, source = env_config_for(path, env_overrides)
    names = exam_scenarios(env_config.get("team_size", 1), names)
    actor = load_actor(path)
    configs, skipped = exam_configs(env_config, sets)
    out_dir = os.path.join(run_dir, "replays")
    os.makedirs(out_dir, exist_ok=True)
    results = []
    for set_name, cfg in configs.items():
        for name in names:
            tag = "%s_%s" % (set_name, name)
            out = os.path.join(out_dir, "r%04d_%s.jsonl" % (rnd, tag))
            results.append(play(actor, tag, SCENARIOS[name], cfg, out, range_km, exam_set=set_name))
    if self_play:
        cfg = {k: v for k, v in env_config.items() if k != "script_perturbation"}   # no script flies
        for name in names:
            out = os.path.join(out_dir, "r%04d_sp_%s.jsonl" % (rnd, name))
            results.append(play(actor, "sp_" + name, SCENARIOS[name], cfg, out, range_km, opponent=actor,
                                exam_set="sp"))
    line = dict(round=rnd, checkpoint=os.path.basename(path), time=time.strftime("%Y-%m-%d %H:%M:%S"),
                env_source=source, sets=list(configs) + (["sp"] if self_play else []), skipped_sets=skipped,
                script_perturbation=env_config.get("script_perturbation"), results=results)
    with open(os.path.join(out_dir, "exams.jsonl"), "a") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")
    return line


def outcome(eng, me=0):
    mine = eng.planes[me]
    others = [p for p in eng.planes if p.team != mine.team]
    enemy_alive = any(p.alive for p in others)
    if mine.alive and not enemy_alive:
        return "win"
    if not mine.alive and enemy_alive:
        return "loss"
    if not mine.alive:
        return "trade"
    return "timeout"


def team_outcome(eng, team):
    """win: only ``team`` has survivors, loss: only the enemy, trade: neither, timeout: both."""
    own = any(p.alive for p in eng.planes if p.team == team)
    enemy = any(p.alive for p in eng.planes if p.team != team)
    return "win" if own and not enemy else "loss" if enemy and not own else "trade" if not own else "timeout"


def team_tally(eng, team, slots):
    """A team match seen from ``team``: own / enemy aircraft dead at the end (any cause), and of the policy-flown
    ``slots``: kills, launches, deaths, crashes (death cause "crash") and friendly-fire kills. With the env's airfield
    option also their landings, rearms and grounded_at_end (parked on the airfield when the match ended; alive)."""
    slots, log = set(slots), eng.log
    out = dict(own_deaths=sum(1 for p in eng.planes if p.team == team and not p.alive),
               enemy_deaths=sum(1 for p in eng.planes if p.team != team and not p.alive),
               kills=sum(1 for e in log if e["kind"] == "kill" and e.get("killer") in slots),
               launches=sum(eng.planes[s].launches for s in slots),
               policy_deaths=sum(1 for s in slots if not eng.planes[s].alive),
               crashes=sum(1 for e in log if e["kind"] == "death" and e.get("plane") in slots
                           and e.get("cause") == "crash"),
               friendly_fire=sum(1 for e in log if e["kind"] == "friendly_fire" and e.get("killer") in slots))
    if getattr(eng, "airfield", None) is not None:
        out.update(landings=sum(eng.planes[s].landings for s in slots), rearms=sum(eng.planes[s].rearms for s in slots),
                   grounded_at_end=sum(1 for s in slots if eng.planes[s].alive and eng.planes[s].grounded))
    return out


def _fly(actors, gens, env, obs):
    """Play the env to the end. ``obs`` (from reset) holds exactly the policy-controlled slots; each slot's action
    is sampled from its own actor with its own generator, as in training. Returns {slot: return} (add_rewards)."""
    h = {aid: torch.zeros(1, H) for aid in obs}
    first = {aid: True for aid in obs}
    ret = {}
    while not env.over:
        acts = {}
        for aid, o in obs.items():
            batch = Decoded([wire.pack_obs(o)]).to_batch(first=torch.tensor([first[aid]]))
            out, h[aid] = actors[aid].act(batch, h[aid], gens[aid], greedy=False)
            acts[aid] = wire.unpack_action(tuple(out.actions[0].tolist()))
            first[aid] = False
        obs, rewards, _, info = env.step(acts)
        add_rewards(ret, rewards, info)
    return ret


def draw_pool(env_config):
    """{aircraft: frequency weight} the stats games draw from: the match model's (model_path), restricted to the env
    config's aircraft_pool when it has one, as MatchEnv restricts its own draws in training (model order kept, so a
    config without aircraft_pool draws exactly as before)."""
    from wt_overlay import match as M
    w = M.load_model(env_config.get("model_path"))["aircraft_frequency"]["weights"]
    pool = env_config.get("aircraft_pool")
    if pool is not None:
        allowed = set(pool)
        w = {a: x for a, x in w.items() if a in allowed}
        if not w:
            raise ValueError("aircraft_pool has no aircraft in the match model")
    return w


def held_out_draw(ids, i, k):
    """``k`` aircraft drawn uniformly from the held-out ``ids`` for game ``i``, with a generator of their own: the
    game's other draws stay those of the in-pool game i."""
    import random
    rng = random.Random("held_out:%d" % i)
    return [rng.choice(list(ids)) for _ in range(k)]


def _held_out_env(env_config):
    """The env config of a held-out game: without aircraft_pool, which does not list the aircraft the policy flies
    (MatchEnv only uses it for random draws, none with explicit teams; dropped so no env check can trip on it)."""
    return {k: v for k, v in env_config.items() if k != "aircraft_pool"}


def _stats_worker(job):
    """One 1v1 between random BR-pool aircraft (frequency-weighted, draw_pool), opponent skill per ``opp_skill``. With
    an opponent checkpoint the other slot is flown by that policy instead of a script; ``swap`` (paired games, needs
    the opponent) puts the evaluated policy in slot 1 and the opponent in slot 0 of the same match (seed, aircraft).
    The row describes the evaluated policy's slot: result, kills, launches and its return (add_rewards).
    With env config team_size > 1 a team game instead (_team_game; job[7]: the team control k, None: all).
    job[8] (optional): held-out aircraft ids; the policy's aircraft is drawn from them (held_out_draw), the rest of the
    game is the in-pool game of the same index (not with ``swap``)."""
    import random
    from wt_overlay.rl_env import MatchEnv
    path, env_config, opp_skill, i, scripted = job[:5]
    opp_path = job[5] if len(job) > 5 else None
    swap = bool(job[6]) if len(job) > 6 else False
    held = list(job[8]) if len(job) > 8 and job[8] else None
    if swap and not opp_path:
        raise ValueError("a swapped game needs an opponent checkpoint")
    if swap and held:
        raise ValueError("held-out games are not paired: the swap would hand the held-out aircraft to the opponent")
    if env_config.get("team_size", 1) > 1:
        return _team_game(path, env_config, opp_skill, i, scripted, opp_path, swap, job[7] if len(job) > 7 else None,
                          held)
    me = 1 if swap else 0
    torch.set_num_threads(1)
    rng = random.Random("stats:%d" % i)
    w = draw_pool(env_config)
    pool, weights = list(w), list(w.values())
    a, b = rng.choices(pool, weights, k=2)
    if held:
        a = held_out_draw(held, i, 1)[0]
        env_config = _held_out_env(env_config)
    skill = opp_skill if opp_skill != "mix" else None
    teams = [[{"aircraft": a}], [dict(aircraft=b, **({"skill": skill} if skill else {}))]]
    # Slot 0 always takes the policy execution path (follow-advice delays), so the scripted baseline differs from
    # the policy only in who decides. With an opponent checkpoint slot 1 takes the same path.
    cfg = dict(env_config, teams=teams, controlled=[0, 1] if opp_path else [0])
    env = MatchEnv(cfg, 50000+i)
    obs = env.reset()
    if scripted:                      # baseline: the script decides for slot 0
        ret = {}
        while not env.over:
            labels = env.scripted_actions()
            obs, rewards, _, info = env.step({aid: labels[aid] for aid in obs})
            add_rewards(ret, rewards, info)
    else:
        actors = {me: load_actor(path)}
        if opp_path:
            actors[1 - me] = load_actor(opp_path)
        # generators belong to the slot, so a paired game differs from its partner only in who flies which side
        gens = {0: torch.Generator().manual_seed(i), 1: torch.Generator().manual_seed(OPPONENT_SEED_BASE + i)}
        ret = _fly(actors, gens, env, obs)
    eng = env.engagement
    return dict(i=i, slot=me, aircraft=(a, b)[me], opponent=(a, b)[1 - me], result=outcome(eng, me),
                time_s=round(eng.time, 1), launches=eng.planes[me].launches,
                kills=sum(1 for e in eng.log if e["kind"] == "kill" and e.get("killer") == me),
                **{"return": round(ret.get(me, 0.), 4)})


def _team_game(path, env_config, opp_skill, i, scripted, opp_path=None, swap=False, control=None, held_out=None):
    """One team game: team_size n aircraft per team drawn from the BR pool (frequency-weighted, as the 1v1 draw). The
    evaluated policy flies the first ``control`` (None: all n) slots of its team (team 1 when ``swap``), the env's
    scripts the rest of it (their execution path in training). The other team is scripts (skill per ``opp_skill``) or,
    with ``opp_path``, flown entirely by that policy; a swapped game is the same match (seed, aircraft) with the two
    policies exchanging teams. ``scripted``: the scripts decide for the slots the policy would fly. ``held_out``
    (aircraft ids, not with ``swap``): the policy-flown slots fly aircraft drawn from them (held_out_draw), every
    other aircraft is that of the in-pool game. The row: team result, own / enemy deaths, kills, launches, deaths,
    crashes and friendly-fire kills of the policy slots and their summed return."""
    import random
    from wt_overlay.rl_env import MatchEnv
    n = env_config["team_size"]
    k = n if control is None else control
    if not 1 <= k <= n:
        raise ValueError("team control must be 1..%d, got %r" % (n, control))
    if swap and held_out:
        raise ValueError("held-out games are not paired: the swap would hand the held-out aircraft to the opponent")
    me = 1 if swap else 0
    torch.set_num_threads(1)
    rng = random.Random("stats:%d" % i)
    w = draw_pool(env_config)
    drawn = rng.choices(list(w), list(w.values()), k=2 * n)          # team 0, then team 1
    if held_out:                                                      # the policy-flown slots of team 0
        drawn[:k] = held_out_draw(held_out, i, k)
        env_config = _held_out_env(env_config)
    # the opponent skill only goes to a scripted enemy team: both games of a pair stay the same match
    skill = {"skill": opp_skill} if opp_skill != "mix" and not opp_path else {}
    teams = [[{"aircraft": a} for a in drawn[:n]], [dict(aircraft=b, **skill) for b in drawn[n:]]]
    own = [me * n + j for j in range(k)]
    enemy = [(1 - me) * n + j for j in range(n)] if opp_path else []
    # the policy slots always take the policy execution path, so the scripted baseline differs only in who decides
    env = MatchEnv(dict(env_config, teams=teams, controlled=sorted(own + enemy)), 50000+i)
    obs = env.reset()
    if scripted:                      # baseline: the scripts decide for the policy slots
        ret = {}
        while not env.over:
            labels = env.scripted_actions()
            obs, rewards, _, info = env.step({aid: labels[aid] for aid in obs})
            add_rewards(ret, rewards, info)
    else:
        actor = load_actor(path)
        actors = {s: actor for s in own}
        if opp_path:
            opp = load_actor(opp_path)
            actors.update({s: opp for s in enemy})
        gens = {s: torch.Generator().manual_seed(slot_seed(i, s)) for s in actors}   # per slot, as in the 1v1
        ret = _fly(actors, gens, env, obs)
    eng = env.engagement
    return dict(i=i, team=me, slots=own, aircraft=drawn[me*n:(me+1)*n], opponent=drawn[(1-me)*n:(2-me)*n],
                result=team_outcome(eng, me), time_s=round(eng.time, 1), **team_tally(eng, me, own),
                **{"return": round(sum(ret.get(s, 0.) for s in own), 4)})


def summarise(rows):
    """Counts (RESULTS) and rates over stats rows. exchange counts a win or trade as one kill and a loss or trade
    as one death (1v1); win_no_kill: wins without a kill of the policy's own (the opponent crashed, flew out of
    bounds, ...), which training's outcomes count as "none"; mean_return: mean of the rows' returns."""
    n = len(rows)
    counts = {k: sum(1 for r in rows if r["result"] == k) for k in RESULTS}
    deaths = counts["loss"] + counts["trade"]
    kills = counts["win"] + counts["trade"]
    return dict(counts, win_rate=round(counts["win"]/n, 3) if n else None,
                exchange=round(kills/deaths, 2) if deaths else None,
                mean_return=round(sum(r["return"] for r in rows)/n, 3) if n else None,
                win_no_kill=sum(1 for r in rows if r["result"] == "win" and not r.get("kills")))


def summarise_teams(rows):
    """Counts (RESULTS) and rates over team stats rows: exchange = enemy deaths / own deaths (all aircraft, any
    cause); survival and kills / deaths / launches per aircraft over the policy-flown aircraft; crashes and
    friendly_fire: their totals; mean_return: mean of the rows' (policy slots' summed) returns."""
    n = len(rows)
    counts = {k: sum(1 for r in rows if r["result"] == k) for k in RESULTS}
    own, enemy = sum(r["own_deaths"] for r in rows), sum(r["enemy_deaths"] for r in rows)
    planes = sum(len(r["slots"]) for r in rows)

    def per(key):
        return round(sum(r[key] for r in rows)/planes, 3) if planes else None
    res = dict(counts, win_rate=round(counts["win"]/n, 3) if n else None,
               exchange=round(enemy/own, 2) if own else None, own_deaths=own, enemy_deaths=enemy,
               mean_return=round(sum(r["return"] for r in rows)/n, 3) if n else None, policy_aircraft=planes,
               survival=round(1. - sum(r["policy_deaths"] for r in rows)/planes, 3) if planes else None,
               kills_per_aircraft=per("kills"), deaths_per_aircraft=per("policy_deaths"),
               launches_per_aircraft=per("launches"), crashes=sum(r["crashes"] for r in rows),
               friendly_fire=sum(r["friendly_fire"] for r in rows))
    if rows and all("landings" in r for r in rows):     # airfield: totals over the policy-flown aircraft
        res.update({k: sum(r[k] for r in rows) for k in ("landings", "rearms", "grounded_at_end")})
    return res


def stats(path, env_config, n, opp_skill, procs, scripted=False, opponent_path=None, paired=False, team_control=None,
          held_out=None):
    """Win / loss / trade / timeout of the policy over n random 1v1s (win rate, exchange from its kills/deaths, mean
    return as handed out by the env). ``paired`` (needs ``opponent_path``) plays each of the n matches twice, the
    second time with the two policies swapping slots: the counts cover all 2n games, ``by_slot`` splits them by the
    slot the policy flew.

    With env config team_size > 1: n team games (_team_game) with the policy flying the first ``team_control`` (None:
    all) slots of its team, summarised by summarise_teams; paired games swap the teams, ``by_team`` splits them.

    ``held_out`` (aircraft ids; not with ``paired``): the policy-flown aircraft are drawn from them, the rest of each
    game is the in-pool game of the same index; the result adds "held_out" and, in 1v1, "by_aircraft" (the counts
    per aircraft the policy flew).

    The checkpoints are copied to a temporary directory first: every episode loads them again, and a live trainer
    prunes old checkpoints in its run directory."""
    from multiprocessing import get_context
    if scripted and opponent_path:
        raise ValueError("--scripted and --opponent-checkpoint exclude each other")
    if paired and not opponent_path:
        raise ValueError("--paired needs --opponent-checkpoint")
    if paired and held_out:
        raise ValueError("--held-out and --paired exclude each other (the swap would hand the held-out aircraft to "
                         "the opponent)")
    size = env_config.get("team_size", 1)
    if team_control is not None and not 1 <= team_control <= size:
        raise ValueError("--team-control must be 1..%d (the env config's team_size)" % size)
    held_out = list(held_out) if held_out else None
    # job[7] team control (1v1: unused), job[8] held-out ids; job tuples without held-out stay as they were
    tail = (((team_control,) if size > 1 else ()) if not held_out else
            (team_control if size > 1 else None, tuple(held_out)))
    with tempfile.TemporaryDirectory(prefix="rl_stats_") as tmp:
        copies = {}
        for tag, src in (("policy", None if scripted else path), ("opponent", opponent_path)):
            if src:
                copies[tag] = os.path.join(tmp, tag + ".pt")
                shutil.copyfile(src, copies[tag])
        jobs = [(copies.get("policy"), env_config, opp_skill, i, scripted, copies.get("opponent"), swap) + tail
                for i in range(n) for swap in ((False, True) if paired else (False,))]
        with get_context("spawn").Pool(procs) as pool:
            rows = pool.map(_stats_worker, jobs)
    m = len(rows)
    if size > 1:
        res = dict(episodes=m, opponent_skill=None if opponent_path else opp_skill,
                   policy="scripts" if scripted else os.path.basename(path), team_size=size,
                   team_control=team_control or size, **summarise_teams(rows), paired=bool(paired))
        if opponent_path:
            res["opponent_checkpoint"] = os.path.basename(opponent_path)
        if paired:
            res["pairs"] = n
            res["by_team"] = {}
            for team in (0, 1):
                part = [r for r in rows if r["team"] == team]
                res["by_team"]["team%d" % team] = dict(episodes=len(part), **summarise_teams(part))
        if held_out:
            res["held_out"] = held_out
        return res
    s = summarise(rows)
    res = dict(episodes=m, opponent_skill=None if opponent_path else opp_skill,
               policy="scripts" if scripted else os.path.basename(path), **{k: s[k] for k in RESULTS},
               win_rate=s["win_rate"], exchange=s["exchange"], mean_return=s["mean_return"],
               launches_per_episode=round(sum(r["launches"] for r in rows)/m, 2) if m else None,
               win_no_kill=s["win_no_kill"], paired=bool(paired))
    if opponent_path:
        res["opponent_checkpoint"] = os.path.basename(opponent_path)
    if paired:
        res["pairs"] = n
        res["by_slot"] = {}
        for slot in (0, 1):
            part = [r for r in rows if r["slot"] == slot]
            res["by_slot"]["slot%d" % slot] = dict(episodes=len(part), **summarise(part))
    if held_out:
        res["held_out"] = held_out
        res["by_aircraft"] = {}
        for a in sorted({r["aircraft"] for r in rows}):
            part = [r for r in rows if r["aircraft"] == a]
            res["by_aircraft"][a] = dict(episodes=len(part), **summarise(part))
    return res


def actor_file(path):
    """(sha256 of the actor weights, training round) of a file with an "actor" state dict (PPO checkpoint, league
    snapshot or reference copy, bc_actor.pt): equal digests are the same policy however the file packs it. The round
    is the trainer's (checkpoint) or the snapshot's "round", else None; (None, None) for a file without an actor."""
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:                   # noqa: BLE001 - unreadable or not a torch file: nothing to compare
        return None, None
    sd = ck.get("actor") if isinstance(ck, dict) else None
    if not isinstance(sd, dict):
        return None, None
    h = hashlib.sha256()
    for name in sorted(sd):
        t = sd[name].detach().cpu().contiguous()
        h.update(("%s %s %s;" % (name, t.dtype, tuple(t.shape))).encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    trainer = ck.get("trainer")
    rnd = trainer.get("round") if isinstance(trainer, dict) else ck.get("round")
    return h.hexdigest(), rnd


def _run_dirs(run_dir, checkpoint=None):
    """Real paths of ``run_dir`` and of the run dir the checkpoint lives in (<dir>/ppo/ or <dir>/league/), once each."""
    dirs = [run_dir] if run_dir else []
    if checkpoint:
        d = os.path.dirname(os.path.abspath(os.path.expanduser(checkpoint)))
        if os.path.basename(d) in ("ppo", "league"):
            dirs.append(os.path.dirname(d))
    out = []
    for d in dirs:
        r = os.path.realpath(os.path.expanduser(d))
        if r not in out:
            out.append(r)
    return out


def league_opponent_matches(opp_path, run_dir, checkpoint=None):
    """How the opponent checkpoint ``opp_path`` is, or was, a training opponent of the evaluated run: a list of
    {"member", "by"}, empty when nothing matches. Looked at: league.references of config.json in the run dir(s)
    (``run_dir`` and the evaluated ``checkpoint``'s own, _run_dirs) and of the checkpoint's stored cfg, the files in
    <run dir>/league/ (snapshots, reference copies), and the snapshot rounds. "by":
      path      the same file as a references entry (relative entries: to the working directory or a run dir) or a
                league file
      weights   the same actor weights (actor_file) as a references entry, its copy in league/ or a league file
      basename  a references entry with the same file name when neither it nor its copy is here to compare
      snapshot  a checkpoint / snapshot of this run whose round is a league snapshot round (history_prob > 0,
                round % snapshot_every == 0) before the evaluated checkpoint's round: the league took exactly those
                weights, possibly pruned from the pool since"""
    run_dirs = _run_dirs(run_dir, checkpoint)
    cfgs = []
    for d in run_dirs:
        p = os.path.join(d, "config.json")
        if os.path.isfile(p):
            try:
                with open(p) as f:
                    cfgs.append(json.load(f))
            except (OSError, ValueError):
                pass
    if checkpoint and os.path.isfile(checkpoint):
        c = stored_cfg(checkpoint)
        if c:
            cfgs.append(c)
    files = {}

    def digest(p):
        r = os.path.realpath(p)
        if r not in files:
            files[r] = actor_file(r)
        return files[r][0]
    opp = os.path.realpath(os.path.expanduser(opp_path))
    opp_digest = digest(opp)
    opp_round = files[opp][1]
    hits, matched = [], set()

    def hit(member, by):
        if all(h["member"] != member for h in hits):
            hits.append(dict(member=member, by=by))
    league_files = sorted({os.path.realpath(f) for d in run_dirs for f in glob.glob(os.path.join(d, "league", "*.pt"))})
    refs = []
    for c in cfgs:
        for r in (c.get("league") or {}).get("references") or []:
            if isinstance(r, str) and r not in refs:
                refs.append(r)
    for r in refs:
        p = os.path.expanduser(r)
        base = os.path.basename(p)
        cands = [p] if os.path.isabs(p) else [os.path.abspath(p)] + [os.path.join(d, p) for d in run_dirs]
        cands += [f for f in league_files if os.path.basename(f).startswith("ref") and
                  os.path.basename(f).endswith("_" + base)]                # rl.league's copy ref<i>_<name>
        found = sorted({os.path.realpath(c) for c in cands if os.path.isfile(c)})
        same = [f for f in found if f == opp]
        twins = [f for f in found if opp_digest is not None and digest(f) == opp_digest] if not same else []
        if same or twins:
            hit(r, "path" if same else "weights")
            matched.update(same + twins)
        elif not any(digest(f) for f in found) and base == os.path.basename(opp):
            hit(r, "basename")
    for f in league_files:
        if f in matched:
            continue
        if f == opp:
            hit("league/" + os.path.basename(f), "path")
        elif opp_digest is not None and digest(f) == opp_digest:
            hit("league/" + os.path.basename(f), "weights")
    own = {os.path.join(d, sub) for d in run_dirs for sub in ("ppo", "league")}
    eval_round = actor_file(checkpoint)[1] if checkpoint and os.path.isfile(checkpoint) else None
    if isinstance(opp_round, int) and os.path.dirname(opp) in own and (eval_round is None or opp_round < eval_round):
        for c in cfgs:
            every = (c.get("league") or {}).get("snapshot_every", 0)
            p_hist = ((c.get("env") or {}).get("config") or {}).get("history_prob", 0)
            if isinstance(p_hist, (int, float)) and p_hist > 0 and isinstance(every, int) and every > 0 \
                    and opp_round % every == 0:
                hit("snapshot r%06d" % opp_round, "snapshot")
                break
    return hits


def held_out_setup(arg, path, env_config):
    """(held-out ids, the ones the checkpoint's training pool contained) for --held-out ``arg``: comma separated ids,
    "" for the checkpoint's own env.held_out ids or else rl.config.HELD_OUT_AIRCRAFT. The training pool is the stored
    env config's aircraft_pool; without one (or without a stored config) every aircraft was in it. A seen id is
    reported on stderr: then the numbers are no held-out test. ValueError for an id the match model does not know."""
    from wt_overlay import match as M
    if arg:
        ids = [x for x in arg.split(",") if x]
    else:
        cfg = stored_cfg(path) or {}
        ids = held_out_ids((cfg.get("env") or {}).get("held_out")) or list(HELD_OUT_AIRCRAFT)
    known = M.load_model(env_config.get("model_path"))["aircraft_frequency"]["weights"]
    unknown = [x for x in ids if x not in known]
    if unknown or not ids:
        raise ValueError("--held-out: aircraft %s are not in the match model" % (unknown or ids))
    pool = (stored_env_config(path) or {}).get("aircraft_pool")
    seen = [x for x in ids if pool is None or x in pool]
    if seen:
        print("warning: --held-out aircraft %s were in the training pool of this checkpoint (%s): this is no held-out "
              "test; train with --set env.held_out=True" % (", ".join(seen), "no aircraft_pool in its env config"
                                                            if pool is None else "listed in its aircraft_pool"),
              file=sys.stderr)
    return ids, seen


def prune(run_dir, keep, milestone):
    """Keep replays of the newest ``keep`` exam rounds and of every ``milestone``-th round."""
    files = glob.glob(os.path.join(run_dir, "replays", "r[0-9][0-9][0-9][0-9]_*.jsonl"))
    rounds = sorted({int(os.path.basename(f)[1:5]) for f in files})
    keep_rounds = set(rounds[-keep:]) | {r for r in rounds if milestone and r % milestone == 0}
    for f in files:
        if int(os.path.basename(f)[1:5]) not in keep_rounds:
            os.remove(f)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--scenarios", help="exams, comma separated (default: all of the env config's team_size): "
                                        + ", ".join(SCENARIOS))
    ap.add_argument("--env-config", default="{}",
                    help="JSON overriding keys of the env config stored in the checkpoint (e.g. reach_dir; null "
                         "switches an option off); a checkpoint without one (bc_actor.pt) starts from %s"
                         % json.dumps(DEFAULT_ENV))
    ap.add_argument("--sets", default=",".join(EXAM_SETS),
                    help="exams: script exam sets, comma separated: fixed (no script perturbation), train (the "
                         "stored script_perturbation)")
    ap.add_argument("--range-km", type=float, default=100., help="spawn line separation of every scenario")
    ap.add_argument("--watch", action="store_true", help="keep running, examining new checkpoints")
    ap.add_argument("--self-play", action="store_true",
                    help="exams: also play every scenario policy against policy (same checkpoint on both teams)")
    ap.add_argument("--every", type=int, default=10, help="--watch: rounds between exams")
    ap.add_argument("--poll-s", type=float, default=60.)
    ap.add_argument("--keep", type=int, default=10)
    ap.add_argument("--milestone", type=int, default=50)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--stats", type=int, nargs="?", const=None, default=0, metavar="N",
                    help="instead of exams: N random games (1v1, or team_size per side of the env config), report "
                         "win/loss rates; without N: %d games (--paired: %d pairs). Standard error of a win rate "
                         "near 0.5: 4.6 pp at 120 games, 3.5 pp at 200" % (DEFAULT_STATS_GAMES, DEFAULT_STATS_GAMES // 2))
    ap.add_argument("--opp-skill", default="mix", choices=("mix", "normal", "top"))
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--scripted", action="store_true",
                    help="--stats baseline: the scripts decide for the slots the policy would fly (slot 0 in a 1v1)")
    ap.add_argument("--checkpoint", help="--stats: checkpoint file (default: newest in --run-dir)")
    ap.add_argument("--opponent-checkpoint", help="--stats: the other slot is flown by this frozen policy instead of a "
                                                  "script (both slots policy-controlled): new versus old")
    ap.add_argument("--paired", action="store_true",
                    help="--stats with --opponent-checkpoint (recommended, ~100 pairs): every match twice, the two "
                         "policies swapping slots (teams); counts over all 2N games plus by_slot (by_team)")
    ap.add_argument("--team-control", type=int, metavar="K",
                    help="--stats with team_size n > 1: the policy flies only the first K (1..n) slots of its team, "
                         "the env's scripts the rest (default: all n)")
    ap.add_argument("--held-out", nargs="?", const="", default=None, metavar="IDS",
                    help="--stats: the policy-flown aircraft are drawn uniformly from held-out aircraft (comma "
                         "separated; without IDS the checkpoint's env.held_out, else %s), the rest of each game as in "
                         "the in-pool game of the same index; not with --paired" % ",".join(HELD_OUT_AIRCRAFT))
    ap.add_argument("--require-held-out-opponent", action="store_true",
                    help="--stats with --opponent-checkpoint: refuse when that checkpoint is or was a training "
                         "opponent (league.references, a league file, a snapshot round of this run)")
    a = ap.parse_args(argv)
    if a.stats is None:                 # --stats without N
        a.stats = DEFAULT_STATS_GAMES // 2 if a.paired else DEFAULT_STATS_GAMES
    torch.set_num_threads(a.threads)
    run_dir = os.path.expanduser(a.run_dir)
    overrides = json.loads(a.env_config)
    if not isinstance(overrides, dict):
        sys.exit("--env-config must be a JSON object")
    names = [SCENARIO_ALIASES.get(n, n) for n in a.scenarios.split(",") if n] if a.scenarios else None
    unknown = [n for n in names or () if n not in SCENARIOS]
    if unknown:
        sys.exit("unknown scenarios: %s" % unknown)
    sets = [s for s in a.sets.split(",") if s]
    if not sets or any(s not in EXAM_SETS for s in sets):
        sys.exit("--sets: comma separated from %s" % ", ".join(EXAM_SETS))
    if a.opponent_checkpoint and not a.stats:
        sys.exit("--opponent-checkpoint needs --stats")
    if a.opponent_checkpoint and a.scripted:
        sys.exit("--opponent-checkpoint and --scripted exclude each other")
    if a.paired and not a.opponent_checkpoint:
        sys.exit("--paired needs --stats and --opponent-checkpoint")
    if a.team_control is not None and not a.stats:
        sys.exit("--team-control needs --stats")
    if a.held_out is not None and not a.stats:
        sys.exit("--held-out needs --stats (the exams have their own *_heldout scenarios)")
    if a.held_out is not None and a.paired:
        sys.exit("--held-out and --paired exclude each other (the swap would hand the held-out aircraft to the "
                 "opponent)")
    if a.require_held_out_opponent and not a.opponent_checkpoint:
        sys.exit("--require-held-out-opponent needs --stats and --opponent-checkpoint")
    if a.stats:
        path = a.checkpoint or (latest(run_dir) or (None, None))[1]
        if path is None and not a.scripted:
            sys.exit("no checkpoint found in %s (use --checkpoint)" % run_dir)
        opp = os.path.expanduser(a.opponent_checkpoint) if a.opponent_checkpoint else None
        if opp and not os.path.exists(opp):
            sys.exit("no such opponent checkpoint: %s" % opp)
        league_hits = league_opponent_matches(opp, run_dir, path) if opp else []
        if league_hits:
            print("warning: the opponent checkpoint %s is a training opponent of this run (%s): the result is not "
                  "measured against a held-out policy" % (opp, "; ".join("%s by %s" % (h["member"], h["by"])
                                                                         for h in league_hits)), file=sys.stderr)
            if a.require_held_out_opponent:
                sys.exit("--require-held-out-opponent: %s is in the league; use a checkpoint outside "
                         "league.references" % opp)
        env_config, source = env_config_for(path, overrides)
        held = seen = None
        if a.held_out is not None:
            try:
                held, seen = held_out_setup(a.held_out, path, env_config)
            except ValueError as e:
                sys.exit(str(e))
        ref = stored_env_config(opp)
        if ref is not None:
            diff = sorted(k for k in set(ref) | set(env_config)
                          if k not in EVAL_DROP and k not in overrides and ref.get(k) != env_config.get(k))
            if diff:
                print("warning: the opponent checkpoint was trained with other env settings (%s); both slots play in "
                      "the evaluated checkpoint's config" % ", ".join(diff), file=sys.stderr)
        size = env_config.get("team_size", 1)
        if a.team_control is not None and not 1 <= a.team_control <= size:
            sys.exit("--team-control must be 1..%d (team_size of the env config)" % size)
        res = stats(path, env_config, a.stats, a.opp_skill, a.procs, a.scripted, opp, a.paired,
                    a.team_control if size > 1 else None, held)
        res.update(env_source=source, env_overrides=overrides)
        if held:
            res["held_out_seen_in_training"] = seen
        if league_hits:
            res.update(opponent_in_league=True, opponent_league_matches=league_hits)
        elif a.require_held_out_opponent:
            res["opponent_in_league"] = False
        print(json.dumps(res, ensure_ascii=False))
        return 0
    last = None
    while True:
        found = latest(run_dir)
        if found is not None and (last is None or found[0] >= last + a.every or (found[0] == 0 and last != 0)):
            try:
                line = exam(run_dir, overrides, names, a.range_km, a.self_play, sets)
            except ValueError as e:         # no scenario of the config's team size
                sys.exit(str(e))
            last = line["round"]
            print(json.dumps(line, ensure_ascii=False), flush=True)
            prune(run_dir, a.keep, a.milestone)
        if not a.watch:
            return 0 if found is not None else 1
        time.sleep(a.poll_s)


if __name__ == "__main__":
    sys.exit(main())
