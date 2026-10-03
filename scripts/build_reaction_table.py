#!/usr/bin/env python3
"""Offline table: how long a target has to react if you fire now (kill rose data).

For each launch range, target course relative to the line of sight (0 = hot,
180 = cold; left/right assumed symmetric) and target pre-launch turn, runs the
FM evader (beam and drag) at every start time on a fixed grid and records
``reaction_s``: the largest t such that every start in [0, t] escapes with the
better of beam and drag. 0 means even an immediate reaction is hit. Positive
turn g turns a crossing target further from hot; negative turns it back hot.

Writes JSON under data/offense/. All modelling limits of escape_window.py and
missile_sim apply; one ownship state per table.

    python scripts/build_reaction_table.py --missile cn_pl12 --aircraft f_16c_block_50 --mass-kg 12000
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import escape_window as ew  # noqa: E402

from wt_overlay.escape import KINDS, EvasionPilot  # noqa: E402


def _cell_args(args, course, turn_g):
    return SimpleNamespace(launch_speed_kmh=args.launch_speed_kmh, launch_altitude_m=args.launch_altitude_m,
                           target_speed_kmh=args.target_speed_kmh, target_altitude_m=args.target_altitude_m,
                           target_course_deg=course, target_turn_g=turn_g, max_time_s=args.max_time_s,
                           observation_mode="sensor_track", loft=True, azimuth_deg=0.)


def _reaction(rows, step, max_gap_s=0.):
    """(reaction_s, reliability) from per-start outcomes, best of all kinds.

    reaction_s is the end of the escaping run that starts at 0, bridging hit
    gaps no longer than ``max_gap_s`` (single-tick hits from chaff phase are
    noise a pilot cannot time). reliability is the escaped fraction of grid
    starts within [0, reaction_s]. An immediate-reaction hit gives (0, 0).
    """
    escaped = {}
    for r in rows:
        escaped[r["start_s"]] = escaped.get(r["start_s"], False) or r["escaped"]
    if not escaped.get(0.):
        return 0., 0.
    starts = sorted(escaped)
    reaction, gap = 0., 0.
    for t in starts:
        if escaped[t]:
            reaction, gap = t, 0.
        else:
            gap += step
            if gap > max_gap_s+1e-9:
                break
    inside = [escaped[t] for t in starts if t <= reaction+1e-9]
    return reaction, sum(inside)/len(inside)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--missile", required=True)
    parser.add_argument("--aircraft", required=True, help="evader FM catalog id")
    parser.add_argument("--mass-kg", type=float, required=True)
    parser.add_argument("--missile-sim", type=Path, default=ew.ROOT.parent/"missle_sim")
    parser.add_argument("--ranges-km", default="5,7,10,13,16")
    parser.add_argument("--courses-deg", default="0,30,60,90,120,150,180")
    parser.add_argument("--turns-g", default="-6,0,6")
    parser.add_argument("--step-s", type=float, default=.25)
    parser.add_argument("--max-gap-s", type=float, default=.5,
                        help="hit gaps up to this long inside an escape run are treated as noise")
    parser.add_argument("--launch-altitude-m", type=float, default=8000.)
    parser.add_argument("--launch-speed-kmh", type=float, default=1100.)
    parser.add_argument("--target-altitude-m", type=float, default=8000.)
    parser.add_argument("--target-speed-kmh", type=float, default=1000.)
    parser.add_argument("--chaff-rcs-ratio", type=float, default=1.)
    parser.add_argument("--max-time-s", type=float, default=60.)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    ranges = [float(x)*1000 for x in args.ranges_km.split(",")]
    courses = [float(x) for x in args.courses_deg.split(",")]
    turns = [float(x) for x in args.turns_g.split(",")]
    chaff = (args.chaff_rcs_ratio, 1., 1, 30) if args.chaff_rcs_ratio > 0 else None
    pilot = EvasionPilot("beam", 0.)
    init = (str(args.missile_sim), args.missile, args.aircraft, args.mass_kg, True, chaff, "rwr",
            (0., .1, 1.), "geometric_mainlobe")
    began = time.perf_counter()
    cells = [(rng, course, turn) for rng in ranges for course in courses for turn in turns]
    with ProcessPoolExecutor(args.workers, initializer=ew._init, initargs=init) as pool:
        bases = list(pool.map(ew._run, [(ew._scenario(_cell_args(args, c, g), r), None, 2.) for r, c, g in cells]))
        jobs, owners = [], []
        for cell, base in zip(cells, bases):
            r, c, g = cell
            if base["escaped"]:
                continue  # Misses without evading; nothing to sweep.
            n = int(base["flight_time_s"]/args.step_s)
            for kind in KINDS:
                for k in range(n+1):
                    jobs.append((ew._scenario(_cell_args(args, c, g), r), replace(pilot, kind=kind, start_s=k*args.step_s), 2.))
                    owners.append(cell)
        results = list(pool.map(ew._run, jobs, chunksize=8))
    elapsed = time.perf_counter()-began

    by_cell = {cell: [] for cell in cells}
    for cell, row in zip(owners, results):
        by_cell[cell].append(row)
    table = []
    for cell, base in zip(cells, bases):
        r, c, g = cell
        rows = by_cell[cell]
        table.append(dict(range_m=r, course_deg=c, turn_g=g, unevaded_hit=not base["escaped"],
                          time_of_flight_s=base["flight_time_s"],
                          **(dict(reaction_s=None, reliability=None) if base["escaped"] else
                             dict(zip(("reaction_s", "reliability"), _reaction(rows, args.step_s, args.max_gap_s)))),
                          escaped_starts=sorted({row["start_s"] for row in rows if row["escaped"]}),
                          faults=sum(1 for row in rows if row["fault"])))
    meta = dict(missile=args.missile, evader_aircraft=args.aircraft, evader_mass_kg=args.mass_kg,
                launch_altitude_m=args.launch_altitude_m, launch_speed_kmh=args.launch_speed_kmh,
                target_altitude_m=args.target_altitude_m, target_speed_kmh=args.target_speed_kmh,
                chaff_rcs_ratio=args.chaff_rcs_ratio, clutter="geometric_mainlobe", perception="rwr",
                evader=dict(max_load=pilot.max_load, alpha_max_deg=pilot.alpha_max_deg,
                            roll_rate_deg_s=pilot.roll_rate_deg_s, dive_deg=pilot.dive_deg),
                step_s=args.step_s, max_gap_s=args.max_gap_s, turn_sign="+ turns a crossing target further from hot, - back toward hot",
                created=time.strftime("%Y-%m-%d %H:%M:%S"), runs=len(jobs)+len(cells), elapsed_s=round(elapsed, 1))
    out = args.out or ew.ROOT/"data"/"offense"/(
        f"{args.missile}__{args.aircraft}__{int(args.launch_altitude_m)}m_{int(args.launch_speed_kmh)}kmh"
        f"__chaff{args.chaff_rcs_ratio:g}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(meta=meta, cells=table), ensure_ascii=False, indent=1))

    print(f"{len(jobs)+len(cells)} runs in {elapsed:.0f} s -> {out}")
    print("reaction_s (s)   course: " + "  ".join(f"{c:>5g}" for c in courses))
    for g in turns:
        for r in ranges:
            vals = {cell["course_deg"]: cell for cell in table if cell["range_m"] == r and cell["turn_g"] == g}
            print(f"turn {g:+3g}g {r/1000:4g} km      " + "  ".join(
                "  miss" if vals[c]["reaction_s"] is None else f"{vals[c]['reaction_s']:5.2f}" for c in courses))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
