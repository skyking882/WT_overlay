#!/usr/bin/env python3
"""Run one simulated air match (1v1, 2v2 or a generated 16v16, or any NvN) and write its replay.

    python3 scripts/run_engagement.py --mode 1v1 --aircraft su_30sm2,f_15c_golden_eagle --archetypes middle,left \\
        --seed 1 --out outputs/engagements/duel.jsonl
    pypy3 scripts/run_engagement.py --mode 16v16 --seed 3 --until-s 300 --out outputs/engagements/m16.jsonl

With ``--aircraft`` (team 0 first, then team 1; one id per aircraft) and optional ``--archetypes``, ``--skills`` (same
order; unspecified ones follow the group priors / the 80-20 skill split) and ``--range-km`` (spawn line separation,
default 100) the match is exactly that; without it both teams are drawn from data/match/top_tier.json (aircraft by match
frequency, archetypes by the group priors, spawn 90-110 km apart). The replay is JSONL (a header, a frame
every 0.25 s, events and an end line); scripts/plot_engagement.py draws it. The printed statistics are kills, deaths,
launches, time per script phase and the wall-clock breakdown (aircraft, missiles, radar, RWR, scripts); ``--profile`` also
runs cProfile (CPython) and prints the 25 most expensive functions.
"""
from __future__ import annotations

import argparse
import collections
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wt_overlay.engagement import MAP_HALF_M, ReplayWriter, default_library  # noqa: E402
from wt_overlay.match import random_match, scenario  # noqa: E402


def team_size(mode):
    left, _, right = mode.partition("v")
    if not (left.isdigit() and left == right and int(left) >= 1):
        raise SystemExit(f"--mode must be NvN (1v1, 2v2, 16v16, ...), not {mode!r}")
    return int(left)


def build(args):
    library = default_library(args.missile_sim)
    half = args.map_half_km*1000.
    n = team_size(args.mode)
    aircraft = [a for a in (args.aircraft or "").split(",") if a]
    if not aircraft:
        if args.archetypes or args.skills:
            raise SystemExit("--archetypes / --skills need --aircraft")
        return random_match(args.seed, team_size=n, library=library, map_half_m=half, debug=args.truth_debug)
    if len(aircraft) != 2*n:
        raise SystemExit(f"--mode {args.mode} needs --aircraft with {2*n} comma-separated ids")
    archetypes = [a for a in (args.archetypes or "").split(",")] if args.archetypes else []
    skills = [a for a in (args.skills or "").split(",")] if args.skills else []
    members = []
    for i, a in enumerate(aircraft):
        members.append(dict(aircraft=a, archetype=(archetypes[i] if i < len(archetypes) and archetypes[i] else None),
                            skill=(skills[i] if i < len(skills) and skills[i] else None)))
    return scenario([members[:n], members[n:]], args.seed, range_km=args.range_km, library=library, map_half_m=half,
                    debug=args.truth_debug)


def report(match, engagement, result, wall_s, out):
    print(f"{result.reason} at {result.time_s:.1f} s ({result.ticks} ticks) | alive team 0 / 1: {result.teams_alive[0]} / "
          f"{result.teams_alive[1]} | launches {result.launches} | kills {len(result.kills)} | missile errors {result.missile_errors}")
    label = lambda i: f"{i}:{match.specs[i].aircraft}/{match.specs[i].archetype}/{match.specs[i].skill}"  # noqa: E731
    for k in result.kills:
        print(f"  kill  t={k['time_s']:7.2f}  {label(k['killer'])} -> {label(k['victim'])} (missile {k['uid']})")
    for d in result.deaths:
        if d["cause"] != "missile":
            print(f"  death t={d['time_s']:7.2f}  {label(d['victim'])} by {d['cause']}")
    faults = sum(p["fm_faults"] for p in result.planes)
    if faults:
        print(f"warning: {faults} aircraft ticks outside the FM tables (flown ballistically): "
              + ", ".join(f"{p['id']}:{p['fm_faults']}" for p in result.planes if p["fm_faults"]))
    print("per aircraft (id aircraft archetype skill | launches kills assists chaff used | missiles left | fate):")
    for p in result.planes:
        s = match.specs[p["id"]]
        fate = "alive" if p["alive"] else f"{p['death']['cause']} t={p['death']['time_s']:.0f}"
        print(f"  {p['id']:2d} {s.aircraft:22s} {s.archetype:8s} {s.skill:6s} | {p['launches']:2d} {p['kills']:2d} {p['assists']:2d} "
              f"{p['chaff_used']:3d} | {p['missiles_left']:2d} | {fate}")
    total = sum(result.phase_time_s.values()) or 1.
    print("time in script phases (aircraft-seconds, share):", ", ".join(
        f"{k or '-'} {v:.0f} ({100*v/total:.0f}%)" for k, v in sorted(result.phase_time_s.items(), key=lambda kv: -kv[1])))
    kinds = collections.Counter(e["kind"] for e in engagement.log)
    print("events:", dict(sorted(kinds.items())))
    w = result.wall
    sim = result.time_s
    print(f"wall {wall_s:.2f} s for {sim:.1f} simulated s = {sim/max(wall_s, 1e-9):.2f} x real time")
    print("  breakdown (s, share of stepping): " + ", ".join(
        f"{k} {w[k]:.2f} ({100*w[k]/max(w['total'], 1e-9):.0f}%)" for k in ("planes", "missiles", "radar", "rwr", "script", "other")))
    print(f"  missile steps {engagement.missile_steps} ({1e6*w['missiles']/max(engagement.missile_steps, 1):.0f} us each); "
          f"aircraft steps {int(result.ticks*len(match.specs))} (<= {1e6*w['planes']/max(result.ticks*len(match.specs), 1):.0f} us each incl. dead)")
    if result.radar_wall:
        print("  radar by aircraft/mode (us per update, updates):")
        for key, (secs, calls) in sorted(result.radar_wall.items(), key=lambda kv: -kv[1][0]):
            print(f"    {key:40s} {1e6*secs/calls:8.1f} us  x{calls}")
    if engagement.rwr_timing:
        print("  RWR update by aircraft (us per update, updates; illuminated-beam checks are in the radar/rwr total):")
        for key, (secs, calls) in sorted(engagement.rwr_timing.items(), key=lambda kv: -kv[1][0])[:8]:
            print(f"    {key:40s} {1e6*secs/calls:8.1f} us  x{calls}")
    if out:
        print(f"replay -> {out}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", default="16v16", help="NvN: 1v1, 2v2, 4v4, 16v16 ...")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--aircraft", help="comma-separated catalog ids, team 0 first (1v1: 2, 2v2: 4)")
    parser.add_argument("--archetypes", help="comma-separated: left, right, middle, crawler, rusher (empty = from the prior)")
    parser.add_argument("--skills", help="comma-separated: normal, top (empty = 80/20 split)")
    parser.add_argument("--range-km", type=float, default=100., help="spawn line separation of 1v1 / 2v2")
    parser.add_argument("--until-s", type=float, default=900., help="time limit")
    parser.add_argument("--map-half-km", type=float, default=MAP_HALF_M/1000., help="map half-width (default 64 = 128 km square)")
    parser.add_argument("--out", type=Path, help="replay JSONL path")
    parser.add_argument("--missile-sim", type=Path, default=None)
    parser.add_argument("--truth-debug", action="store_true", help="give the scripts the true enemy position (debug only)")
    parser.add_argument("--profile", action="store_true", help="run under cProfile and print the hottest functions")
    args = parser.parse_args(argv)

    match = build(args)
    replay = ReplayWriter(args.out, keep=False) if args.out else None
    engagement = match.engagement(replay=replay, time_limit_s=args.until_s, truth_debug=args.truth_debug)
    t0 = time.perf_counter()
    if args.profile:
        import cProfile
        import pstats
        profiler = cProfile.Profile()
        result = profiler.runcall(engagement.run)
        stats = pstats.Stats(profiler).sort_stats("cumulative")
    else:
        result = engagement.run()
        stats = None
    wall = time.perf_counter()-t0
    if replay is not None:
        replay.close()
    report(match, engagement, result, wall, args.out)
    if stats is not None:
        stats.print_stats(25)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
