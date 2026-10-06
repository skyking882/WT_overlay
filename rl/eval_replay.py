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
420} as before. self_play_prob, history_prob and policy_ids are dropped: the evaluation decides which slots a policy
flies.

Exam sets (--sets, default both):
  fixed  r<round>_fixed_<scenario>.jsonl  the scripts without script_perturbation: comparable over a whole run and
                                          across runs (long-term trend)
  train  r<round>_train_<scenario>.jsonl  the scripts with the stored script_perturbation, as in training; skipped
                                          when the config has none (it would replay the fixed matches)
  sp     r<round>_sp_<scenario>.jsonl     with --self-play: every scenario once more with the same checkpoint flying
                                          slot 1 as well (both aircraft marked "AI")
--watch repeats whenever the newest checkpoint is at least --every rounds past the last exam; replays of the latest
--keep exams and of every --milestone-th round are kept.

    python -m rl.eval_replay --run-dir ~/rl_runs/s1_v1 --watch --every 10

--stats N plays N random 1v1s and reports win / loss / trade / timeout of the policy, win rate, exchange and
mean_return (the rewards the env hands the policy's slot over the whole match, timeout_reward and late rewards
included: the return training sees). Slot 1 is a script by default; with --opponent-checkpoint PATH both slots are
policy-controlled and the other one is flown by that frozen policy (own seeded generator), i.e. new versus old.
Recommended reference evaluation: --paired plays every seed twice with the two policies swapping slots (and so spawn
sides and aircraft); the counts are the new policy's over all 2N games, by_slot splits them by the slot it flew.
About 100 pairs (200 games):

    python -m rl.eval_replay --run-dir RUN --stats 100 --paired --opponent-checkpoint OLD/ppo/ckpt_000100.pt
"""
from __future__ import annotations

import argparse
import glob
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
from rl.encode import Decoded  # noqa: E402
from rl.model import Actor, H  # noqa: E402

OPPONENT_SEED_BASE = 1_000_003      # --stats: generator seed of slot 1 is this + episode index (slot 0: the index)
DEFAULT_ENV = dict(team_size=1, time_limit_s=420)   # env config of a checkpoint that stores none (bc_actor.pt)
EVAL_DROP = ("self_play_prob", "history_prob", "policy_ids")   # the evaluation decides which slots a policy flies
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
}


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


def env_config_for(path, overrides=None):
    """(env config, source) for evaluating the checkpoint ``path``: its stored training config with ``overrides``
    (--env-config) on top, source "checkpoint"; without a stored config DEFAULT_ENV plus overrides, source "default".
    The keys of EVAL_DROP are removed. A path key that does not exist on this machine is reported on stderr."""
    stored = stored_env_config(path)
    cfg = dict(stored if stored is not None else DEFAULT_ENV)
    cfg.update(overrides or {})
    for k in EVAL_DROP:
        cfg.pop(k, None)
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
    With ``opponent`` (an Actor, possibly ``actor`` itself) slot 1 is policy-flown too, with its own generator:
    a self-play match as in training."""
    from wt_overlay.engagement import ReplayWriter
    from wt_overlay.rl_env import MatchEnv
    cfg = dict(env_config, teams=scenario["teams"], range_km=scenario.get("range_km", range_km),
               controlled=[0, 1] if opponent is not None else [0])
    env = MatchEnv(cfg, scenario["seed"])
    obs = env.reset()
    eng = env.engagement
    writer = ReplayWriter(out_path + ".part", keep=False)
    header = eng._header()
    for p in header["planes"]:
        if p["id"] in env.policy_ids:
            p.update(archetype="AI", skill="policy", controller="policy")
    header["exam"] = dict(scenario=name, seed=scenario["seed"], self_play=opponent is not None, set=exam_set)
    writer.write(header)
    eng.replay = writer
    eng._frame()
    actors = {0: actor, 1: opponent}
    gens = {0: torch.Generator().manual_seed(scenario["seed"]),
            1: torch.Generator().manual_seed(OPPONENT_SEED_BASE + scenario["seed"])}
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
    return res


def exam(run_dir, env_overrides, names, range_km=100., self_play=False, sets=EXAM_SETS):
    """The script exam sets (exam_configs) of every scenario, in the checkpoint's own env config with
    ``env_overrides`` on top (env_config_for); with ``self_play`` also once against the same checkpoint in slot 1
    (set "sp", replay r<round>_sp_<scenario>.jsonl), the kind of match a self-play run trains on."""
    found = latest(run_dir)
    if found is None:
        return None
    rnd, path = found
    actor = load_actor(path)
    env_config, source = env_config_for(path, env_overrides)
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


def _stats_worker(job):
    """One 1v1 between random BR-pool aircraft (frequency-weighted), opponent skill per ``opp_skill``. With an
    opponent checkpoint the other slot is flown by that policy instead of a script; ``swap`` (paired games, needs the
    opponent) puts the evaluated policy in slot 1 and the opponent in slot 0 of the same match (seed, aircraft).
    The row describes the evaluated policy's slot: result, kills, launches and its return (add_rewards)."""
    import random
    from wt_overlay import match as M
    from wt_overlay.rl_env import MatchEnv
    path, env_config, opp_skill, i, scripted = job[:5]
    opp_path = job[5] if len(job) > 5 else None
    swap = bool(job[6]) if len(job) > 6 else False
    if swap and not opp_path:
        raise ValueError("a swapped game needs an opponent checkpoint")
    me = 1 if swap else 0
    torch.set_num_threads(1)
    rng = random.Random("stats:%d" % i)
    model = M.load_model(env_config.get("model_path"))
    w = model["aircraft_frequency"]["weights"]
    pool, weights = list(w), list(w.values())
    a, b = rng.choices(pool, weights, k=2)
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


def stats(path, env_config, n, opp_skill, procs, scripted=False, opponent_path=None, paired=False):
    """Win / loss / trade / timeout of the policy over n random 1v1s (win rate, exchange from its kills/deaths, mean
    return as handed out by the env). ``paired`` (needs ``opponent_path``) plays each of the n matches twice, the
    second time with the two policies swapping slots: the counts cover all 2n games, ``by_slot`` splits them by the
    slot the policy flew.

    The checkpoints are copied to a temporary directory first: every episode loads them again, and a live trainer
    prunes old checkpoints in its run directory."""
    from multiprocessing import get_context
    if scripted and opponent_path:
        raise ValueError("--scripted and --opponent-checkpoint exclude each other")
    if paired and not opponent_path:
        raise ValueError("--paired needs --opponent-checkpoint")
    with tempfile.TemporaryDirectory(prefix="rl_stats_") as tmp:
        copies = {}
        for tag, src in (("policy", None if scripted else path), ("opponent", opponent_path)):
            if src:
                copies[tag] = os.path.join(tmp, tag + ".pt")
                shutil.copyfile(src, copies[tag])
        jobs = [(copies.get("policy"), env_config, opp_skill, i, scripted, copies.get("opponent"), swap)
                for i in range(n) for swap in ((False, True) if paired else (False,))]
        with get_context("spawn").Pool(procs) as pool:
            rows = pool.map(_stats_worker, jobs)
    m = len(rows)
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
    return res


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
    ap.add_argument("--scenarios", default=",".join(SCENARIOS), help="comma separated: " + ", ".join(SCENARIOS))
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
                    help="exams: also play every scenario policy against policy (same checkpoint in both slots)")
    ap.add_argument("--every", type=int, default=10, help="--watch: rounds between exams")
    ap.add_argument("--poll-s", type=float, default=60.)
    ap.add_argument("--keep", type=int, default=10)
    ap.add_argument("--milestone", type=int, default=50)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--stats", type=int, default=0, help="instead of exams: N random 1v1s, report win/loss rates")
    ap.add_argument("--opp-skill", default="mix", choices=("mix", "normal", "top"))
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--scripted", action="store_true", help="--stats baseline: a script flies slot 0")
    ap.add_argument("--checkpoint", help="--stats: checkpoint file (default: newest in --run-dir)")
    ap.add_argument("--opponent-checkpoint", help="--stats: the other slot is flown by this frozen policy instead of a "
                                                  "script (both slots policy-controlled): new versus old")
    ap.add_argument("--paired", action="store_true",
                    help="--stats with --opponent-checkpoint (recommended, ~100 pairs): every match twice, the two "
                         "policies swapping slots; counts over all 2N games plus by_slot")
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    run_dir = os.path.expanduser(a.run_dir)
    overrides = json.loads(a.env_config)
    if not isinstance(overrides, dict):
        sys.exit("--env-config must be a JSON object")
    names = [n for n in a.scenarios.split(",") if n]
    unknown = [n for n in names if n not in SCENARIOS]
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
    if a.stats:
        path = a.checkpoint or (latest(run_dir) or (None, None))[1]
        if path is None and not a.scripted:
            sys.exit("no checkpoint found in %s (use --checkpoint)" % run_dir)
        opp = os.path.expanduser(a.opponent_checkpoint) if a.opponent_checkpoint else None
        if opp and not os.path.exists(opp):
            sys.exit("no such opponent checkpoint: %s" % opp)
        env_config, source = env_config_for(path, overrides)
        ref = stored_env_config(opp)
        if ref is not None:
            diff = sorted(k for k in set(ref) | set(env_config)
                          if k not in EVAL_DROP and k not in overrides and ref.get(k) != env_config.get(k))
            if diff:
                print("warning: the opponent checkpoint was trained with other env settings (%s); both slots play in "
                      "the evaluated checkpoint's config" % ", ".join(diff), file=sys.stderr)
        res = stats(path, env_config, a.stats, a.opp_skill, a.procs, a.scripted, opp, a.paired)
        res.update(env_source=source, env_overrides=overrides)
        print(json.dumps(res, ensure_ascii=False))
        return 0
    last = None
    while True:
        found = latest(run_dir)
        if found is not None and (last is None or found[0] >= last + a.every or (found[0] == 0 and last != 0)):
            line = exam(run_dir, overrides, names, a.range_km, a.self_play, sets)
            last = line["round"]
            print(json.dumps(line, ensure_ascii=False), flush=True)
            prune(run_dir, a.keep, a.milestone)
        if not a.watch:
            return 0 if found is not None else 1
        time.sleep(a.poll_s)


if __name__ == "__main__":
    sys.exit(main())
