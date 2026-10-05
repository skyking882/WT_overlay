#!/usr/bin/env python3
"""Offline table: how long a target has to react if you fire now (kill rose data).

For each launch range, target course relative to the line of sight (0 = hot,
180 = cold; left/right assumed symmetric) and target pre-launch turn, runs the
FM evader at every start time on a fixed grid and records ``reaction_s``: the
largest t such that every start in [0, t] escapes with at least one plan.
``--evasion library`` (default) tries the maneuver building-block library of
optimize_evasion.py (heading x turn plane x dive); ``beamdrag`` only level
beam and level drag. 0 means even an immediate reaction is hit. Positive
turn g turns a crossing target further from hot; negative turns it back hot.

Writes JSON under data/offense/. All modelling limits of escape_window.py and
missile_sim apply; one ownship state per table.

    python scripts/build_reaction_table.py --missile cn_pl12 --aircraft f_16c_block_50 --mass-kg 12000
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import escape_window as ew  # noqa: E402

from optimize_evasion import LIBRARY, _pilot  # noqa: E402

from wt_overlay.escape import EvasionPilot  # noqa: E402

# Level beam and drag first: they escape most often, and a start stops at its first escape.
FIRST = [(90., 0., 0.), (0., 0., 0.)]
PLANS = {"library": FIRST+[p for p in LIBRARY if p not in FIRST], "beamdrag": FIRST}


def _cell_args(args, course, turn_g):
    return SimpleNamespace(launch_speed_kmh=args.launch_speed_kmh, launch_altitude_m=args.launch_altitude_m,
                           target_speed_kmh=args.target_speed_kmh, target_altitude_m=args.target_altitude_m,
                           target_course_deg=course, target_turn_g=turn_g, max_time_s=args.max_time_s,
                           observation_mode="sensor_track", loft=True, azimuth_deg=0.)


def _reaction(rows, step, max_gap_s=0.):
    """(reaction_s, reliability) from per-start outcomes, best of all plans.

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


def _scan_cell(payload):
    """Evasion starts from 0 until the reaction window provably ends (same answer as a full sweep).

    Plans are tried in order and a start stops at its first escape; the scan
    stops at an immediate-reaction hit or once the hit gap exceeds ``max_gap_s``.
    """
    scenario, horizon, step, max_gap_s, plans = payload
    rows, gap, k = [], 0., 0
    while k*step <= horizon+1e-9:
        start = round(k*step, 6)
        escaped = False
        for plan in plans:
            row = ew._run((scenario, _pilot(start, plan), 2.))
            rows.append(row)
            if row["escaped"]:
                escaped = True
                break
        if start == 0. and not escaped:
            break
        gap = 0. if escaped else gap+step
        if gap > max_gap_s+1e-9:
            break
        k += 1
    return rows


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
    parser.add_argument("--scan", choices=("early", "full"), default="early",
                        help="early: stop each cell once its reaction window ends; full: sweep every start")
    parser.add_argument("--evasion", choices=tuple(PLANS), default="library")
    parser.add_argument("--clutter", choices=("look_down_angle", "geometric_mainlobe", "look_down"),
                        default="look_down_angle")
    parser.add_argument("--clutter-depression-deg", type=float, default=2.)
    parser.add_argument("--no-cw", dest="cw", action="store_false")
    args = parser.parse_args(argv)

    ranges = [float(x)*1000 for x in args.ranges_km.split(",")]
    courses = [float(x) for x in args.courses_deg.split(",")]
    turns = [float(x) for x in args.turns_g.split(",")]
    chaff = (args.chaff_rcs_ratio, 1., 1, 30) if args.chaff_rcs_ratio > 0 else None
    pilot = EvasionPilot("beam", 0.)
    plans = PLANS[args.evasion]
    depression = args.clutter_depression_deg if args.clutter == "look_down_angle" else None
    init = (str(args.missile_sim), args.missile, args.aircraft, args.mass_kg, True, chaff, "rwr",
            (0., .1, 1.), args.clutter, depression, args.cw)
    began = time.perf_counter()
    cells = [(rng, course, turn) for rng in ranges for course in courses for turn in turns]
    with ProcessPoolExecutor(args.workers, initializer=ew._init, initargs=init) as pool:
        bases = list(pool.map(ew._run, [(ew._scenario(_cell_args(args, c, g), r), None, 2.) for r, c, g in cells]))
        by_cell = {cell: [] for cell in cells}
        swept = [(cell, base) for cell, base in zip(cells, bases) if not base["escaped"]]  # Misses need no sweep.
        if args.scan == "early":
            payloads = [(ew._scenario(_cell_args(args, c, g), r), base["flight_time_s"], args.step_s,
                         args.max_gap_s, plans) for (r, c, g), base in swept]
            for (cell, _), rows in zip(swept, pool.map(_scan_cell, payloads)):
                by_cell[cell] = rows
        else:
            jobs, owners = [], []
            for (r, c, g), base in swept:
                for plan in plans:
                    for k in range(int(base["flight_time_s"]/args.step_s)+1):
                        jobs.append((ew._scenario(_cell_args(args, c, g), r), _pilot(k*args.step_s, plan), 2.))
                        owners.append((r, c, g))
            for cell, row in zip(owners, pool.map(ew._run, jobs, chunksize=8)):
                by_cell[cell].append(row)
    elapsed = time.perf_counter()-began
    runs = len(cells)+sum(len(rows) for rows in by_cell.values())
    table = []
    for cell, base in zip(cells, bases):
        r, c, g = cell
        rows = by_cell[cell]
        reaction = None if base["escaped"] else _reaction(rows, args.step_s, args.max_gap_s)
        last = [row for row in rows if row["escaped"] and reaction and abs(row["start_s"]-reaction[0]) < 1e-9]
        table.append(dict(range_m=r, course_deg=c, turn_g=g, unevaded_hit=not base["escaped"],
                          time_of_flight_s=base["flight_time_s"],
                          **(dict(reaction_s=None, reliability=None) if base["escaped"] else
                             dict(zip(("reaction_s", "reliability"), reaction))),
                          plan_at_reaction=last[0]["plan"] if last else None,
                          escaped_starts=sorted({row["start_s"] for row in rows if row["escaped"]}),
                          faults=sum(1 for row in rows if row["fault"])))
    meta = dict(missile=args.missile, evader_aircraft=args.aircraft, evader_mass_kg=args.mass_kg,
                launch_altitude_m=args.launch_altitude_m, launch_speed_kmh=args.launch_speed_kmh,
                target_altitude_m=args.target_altitude_m, target_speed_kmh=args.target_speed_kmh,
                chaff_rcs_ratio=args.chaff_rcs_ratio, clutter=args.clutter, clutter_min_depression_deg=depression,
                cw_on_clear_beam=args.cw, perception="rwr", evasion=args.evasion, plans=plans,
                evader=dict(max_load=pilot.max_load, alpha_max_deg=pilot.alpha_max_deg,
                            roll_rate_deg_s=pilot.roll_rate_deg_s),
                step_s=args.step_s, max_gap_s=args.max_gap_s, turn_sign="+ turns a crossing target further from hot, - back toward hot",
                scan=args.scan, created=time.strftime("%Y-%m-%d %H:%M:%S"), runs=runs, elapsed_s=round(elapsed, 1))
    out = args.out or ew.ROOT/"data"/"offense"/(
        f"{args.missile}__{args.aircraft}__{int(args.launch_altitude_m)}m_{int(args.launch_speed_kmh)}kmh"
        f"__chaff{args.chaff_rcs_ratio:g}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(meta=meta, cells=table), ensure_ascii=False, indent=1))

    print(f"{runs} runs in {elapsed:.0f} s -> {out}")
    print("reaction_s (s)   course: " + "  ".join(f"{c:>5g}" for c in courses))
    for g in turns:
        for r in ranges:
            vals = {cell["course_deg"]: cell for cell in table if cell["range_m"] == r and cell["turn_g"] == g}
            print(f"turn {g:+3g}g {r/1000:4g} km      " + "  ".join(
                "  miss" if vals[c]["reaction_s"] is None else f"{vals[c]['reaction_s']:5.2f}" for c in courses))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
