"""Exam replays: the current policy plays fixed matchups, recorded for the dashboard.

Runs beside training and never touches the run's checkpoints or metrics. It loads the newest
ppo/ckpt_*.pt (or bc_actor.pt before PPO), plays every scenario once greedily in the real
MatchEnv, records each match with wt_overlay.engagement.ReplayWriter as
<run_dir>/replays/r<round>_<scenario>.jsonl and appends a summary line to
<run_dir>/replays/exams.jsonl. The same seed per scenario every time, so one matchup can be
compared across rounds. --self-play adds every scenario once more with the same checkpoint flying
slot 1 as well (r<round>_sp_<scenario>.jsonl, both aircraft marked "AI"). --watch repeats whenever the newest checkpoint is at least --every
rounds past the last exam; replays of the latest --keep exams and of every --milestone-th
round are kept. Policy aircraft show archetype "AI" in the replay header.

    python -m rl.eval_replay --run-dir ~/rl_runs/s1_v1 --watch --every 10

--stats N plays N random 1v1s and reports win rate / exchange of slot 0 (the policy). Slot 1 is a script by default;
with --opponent-checkpoint PATH both slots are policy-controlled and slot 1 is flown by that frozen policy (own
seeded generator), i.e. new versus old:

    python -m rl.eval_replay --run-dir RUN --stats 200 --opponent-checkpoint OLD/ppo/ckpt_000100.pt
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

OPPONENT_SEED_BASE = 1_000_003      # --stats: generator seed of the opponent checkpoint is this + episode index

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


def play(actor, name, scenario, env_config, out_path, range_km=100., greedy=False, opponent=None):
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
    header["exam"] = dict(scenario=name, seed=scenario["seed"], self_play=opponent is not None)
    writer.write(header)
    eng.replay = writer
    eng._frame()
    actors = {0: actor, 1: opponent}
    gens = {0: torch.Generator().manual_seed(scenario["seed"]),
            1: torch.Generator().manual_seed(OPPONENT_SEED_BASE + scenario["seed"])}
    h = {aid: torch.zeros(1, H) for aid in obs}
    first = {aid: True for aid in obs}
    totals = dict(reward=0., steps=0)
    t0 = time.time()
    while not env.over:
        acts = {}
        for aid, o in obs.items():
            batch = Decoded([wire.pack_obs(o)]).to_batch(first=torch.tensor([first[aid]]))
            out, h[aid] = actors[aid].act(batch, h[aid], gens[aid], greedy=greedy)
            acts[aid] = wire.unpack_action(tuple(out.actions[0].tolist()))
            first[aid] = False
        obs, rewards, dones, info = env.step(acts)
        # Slot 0's reward, including what its missiles score after it is down (info late_rewards).
        totals["reward"] += rewards.get(0, 0.) + ((info or {}).get("late_rewards") or {}).get(0, 0.)
        totals["steps"] += 1
    writer.close()                      # the engagement already closed it at the end; close() is idempotent
    os.replace(out_path + ".part", out_path)
    policy = eng.planes[0]
    kills = sum(1 for e in eng.log if e["kind"] == "kill" and e.get("killer") == 0)
    launches = sum(1 for e in eng.log if e["kind"] == "launch" and e.get("shooter") == 0)
    res = dict(scenario=name, reason=eng.reason, time_s=round(eng.time, 1), policy_alive=policy.alive,
               policy_kills=kills, policy_launches=launches, reward=round(totals["reward"], 2),
               decisions=totals["steps"], wall_s=round(time.time()-t0, 1), replay=os.path.basename(out_path))
    if opponent is not None:
        res.update(self_play=True, opponent_alive=eng.planes[1].alive,
                   opponent_kills=sum(1 for e in eng.log if e["kind"] == "kill" and e.get("killer") == 1),
                   opponent_launches=sum(1 for e in eng.log if e["kind"] == "launch" and e.get("shooter") == 1))
    return res


def exam(run_dir, env_config, names, range_km=100., self_play=False):
    """Every scenario against its script; with ``self_play`` also once against the same checkpoint in slot 1
    (replay r<round>_sp_<scenario>.jsonl), the kind of match a self-play run trains on."""
    found = latest(run_dir)
    if found is None:
        return None
    rnd, path = found
    actor = load_actor(path)
    out_dir = os.path.join(run_dir, "replays")
    os.makedirs(out_dir, exist_ok=True)
    results = []
    for name in names:
        out = os.path.join(out_dir, "r%04d_%s.jsonl" % (rnd, name))
        results.append(play(actor, name, SCENARIOS[name], env_config, out, range_km))
    if self_play:
        for name in names:
            out = os.path.join(out_dir, "r%04d_sp_%s.jsonl" % (rnd, name))
            results.append(play(actor, "sp_" + name, SCENARIOS[name], env_config, out, range_km, opponent=actor))
    line = dict(round=rnd, checkpoint=os.path.basename(path), time=time.strftime("%Y-%m-%d %H:%M:%S"), results=results)
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
    is sampled from its own actor with its own generator, as in training."""
    h = {aid: torch.zeros(1, H) for aid in obs}
    first = {aid: True for aid in obs}
    while not env.over:
        acts = {}
        for aid, o in obs.items():
            batch = Decoded([wire.pack_obs(o)]).to_batch(first=torch.tensor([first[aid]]))
            out, h[aid] = actors[aid].act(batch, h[aid], gens[aid], greedy=False)
            acts[aid] = wire.unpack_action(tuple(out.actions[0].tolist()))
            first[aid] = False
        obs, _, _, _ = env.step(acts)


def _stats_worker(job):
    """One 1v1 between random BR-pool aircraft (frequency-weighted), opponent skill per ``opp_skill``. With an
    opponent checkpoint slot 1 is flown by that policy instead of a script."""
    import random
    from wt_overlay import match as M
    from wt_overlay.rl_env import MatchEnv
    path, env_config, opp_skill, i, scripted = job[:5]
    opp_path = job[5] if len(job) > 5 else None
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
        while not env.over:
            labels = env.scripted_actions()
            obs, _, _, _ = env.step({aid: labels[aid] for aid in obs})
    else:
        actors = {0: load_actor(path)}
        gens = {0: torch.Generator().manual_seed(i)}
        if opp_path:
            actors[1] = load_actor(opp_path)
            gens[1] = torch.Generator().manual_seed(OPPONENT_SEED_BASE + i)
        _fly(actors, gens, env, obs)
    eng = env.engagement
    return dict(i=i, aircraft=a, opponent=b, result=outcome(eng, 0), time_s=round(eng.time, 1),
                launches=eng.planes[0].launches)


def stats(path, env_config, n, opp_skill, procs, scripted=False, opponent_path=None):
    """Win / loss / trade / timeout of slot 0 over n random 1v1s (win rate, exchange from slot 0's kills/deaths).

    The checkpoints are copied to a temporary directory first: every episode loads them again, and a live trainer
    prunes old checkpoints in its run directory."""
    from multiprocessing import get_context
    if scripted and opponent_path:
        raise ValueError("--scripted and --opponent-checkpoint exclude each other")
    with tempfile.TemporaryDirectory(prefix="rl_stats_") as tmp:
        copies = {}
        for tag, src in (("policy", None if scripted else path), ("opponent", opponent_path)):
            if src:
                copies[tag] = os.path.join(tmp, tag + ".pt")
                shutil.copyfile(src, copies[tag])
        jobs = [(copies.get("policy"), env_config, opp_skill, i, scripted, copies.get("opponent")) for i in range(n)]
        with get_context("spawn").Pool(procs) as pool:
            rows = pool.map(_stats_worker, jobs)
    counts = {k: sum(1 for r in rows if r["result"] == k) for k in ("win", "loss", "trade", "timeout")}
    deaths = counts["loss"] + counts["trade"]
    kills = counts["win"] + counts["trade"]
    res = dict(episodes=n, opponent_skill=None if opponent_path else opp_skill,
               policy="scripts" if scripted else os.path.basename(path), **counts,
               win_rate=round(counts["win"]/n, 3), exchange=round(kills/deaths, 2) if deaths else None,
               mean_return=round((kills - 2*deaths)/n, 3), launches_per_episode=round(sum(r["launches"] for r in rows)/n, 2))
    if opponent_path:
        res["opponent_checkpoint"] = os.path.basename(opponent_path)
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
    ap.add_argument("--env-config", default="{}", help="JSON merged into the MatchEnv config (e.g. reach_dir)")
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
    ap.add_argument("--opponent-checkpoint", help="--stats: slot 1 is flown by this frozen policy instead of a script "
                                                  "(both slots policy-controlled): new versus old")
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    run_dir = os.path.expanduser(a.run_dir)
    env_config = dict(dict(team_size=1, time_limit_s=420), **json.loads(a.env_config))
    names = [n for n in a.scenarios.split(",") if n]
    unknown = [n for n in names if n not in SCENARIOS]
    if unknown:
        sys.exit("unknown scenarios: %s" % unknown)
    if a.opponent_checkpoint and not a.stats:
        sys.exit("--opponent-checkpoint needs --stats")
    if a.opponent_checkpoint and a.scripted:
        sys.exit("--opponent-checkpoint and --scripted exclude each other")
    if a.stats:
        path = a.checkpoint or (latest(run_dir) or (None, None))[1]
        if path is None and not a.scripted:
            sys.exit("no checkpoint found in %s (use --checkpoint)" % run_dir)
        opp = os.path.expanduser(a.opponent_checkpoint) if a.opponent_checkpoint else None
        if opp and not os.path.exists(opp):
            sys.exit("no such opponent checkpoint: %s" % opp)
        print(json.dumps(stats(path, env_config, a.stats, a.opp_skill, a.procs, a.scripted, opp), ensure_ascii=False))
        return 0
    last = None
    while True:
        found = latest(run_dir)
        if found is not None and (last is None or found[0] >= last + a.every or (found[0] == 0 and last != 0)):
            line = exam(run_dir, env_config, names, a.range_km, a.self_play)
            last = line["round"]
            print(json.dumps(line, ensure_ascii=False), flush=True)
            prune(run_dir, a.keep, a.milestone)
        if not a.watch:
            return 0 if found is not None else 1
        time.sleep(a.poll_s)


if __name__ == "__main__":
    sys.exit(main())
