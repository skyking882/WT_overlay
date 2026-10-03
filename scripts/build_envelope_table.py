#!/usr/bin/env python3
"""Offline launch-envelope table for the B-scope overlay (and later the side view).

For each target off-boresight angle and altitude difference, bisects launch
range for four lines (target straight at 1000 km/h by default):

    rmax_hot   farthest hit on a non-evading target flying at you
    rmax_cold  farthest hit on a non-evading target flying away
    r3_hot     farthest range where a hot target has under 3 s to react
    rne_hot    farthest range where a hot target cannot escape at all

Reaction uses the same rule as build_reaction_table.py (beam or drag, chaff,
gaps up to --max-gap-s bridged). Each predicate is assumed monotone in range
inside its first band, found on a coarse range ladder; a line is None when the
property never holds on the ladder (e.g. no escape even at 2 km). All modelling limits of escape_window.py apply.

    python scripts/build_envelope_table.py --missile cn_pl12 --aircraft f_16c_block_50 --mass-kg 12000
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
from build_reaction_table import _reaction, _scan_cell  # noqa: E402

from wt_overlay.escape import EvasionPilot  # noqa: E402

LINES = ("rmax_hot", "rmax_cold", "r3_hot", "rne_hot")
COARSE_KM = (2, 3, 4, 5, 7, 10, 14, 20, 28, 40, 56, 80, 120)


def _args(base, azimuth, alt_diff, course):
    return SimpleNamespace(launch_speed_kmh=base.launch_speed_kmh, launch_altitude_m=base.launch_altitude_m,
                           target_speed_kmh=base.target_speed_kmh,
                           target_altitude_m=max(300., base.launch_altitude_m+alt_diff),
                           target_course_deg=course, target_turn_g=0., max_time_s=base.max_time_s,
                           observation_mode="sensor_track", loft=True, azimuth_deg=azimuth)


def _predicate(task, range_m):
    """True while range_m is inside the line (the property holds)."""
    line, base, azimuth, alt_diff = task
    course = 180. if line == "rmax_cold" else 0.
    scenario = ew._scenario(_args(base, azimuth, alt_diff, course), range_m)
    unevaded = ew._run((scenario, None, 2.))
    if line in ("rmax_hot", "rmax_cold"):
        return not unevaded["escaped"]
    if unevaded["escaped"]:
        return False
    if line == "rne_hot":
        # No escape: an immediate beam or drag is still hit.
        rows = _scan_cell((scenario, 0., base.step_s, base.max_gap_s, EvasionPilot("beam", 0.)))
        return not any(row["escaped"] for row in rows)
    horizon = min(unevaded["flight_time_s"], base.reaction_limit_s+base.max_gap_s)
    rows = _scan_cell((scenario, horizon, base.step_s, base.max_gap_s, EvasionPilot("beam", 0.)))
    reaction, _ = _reaction(rows, base.step_s, base.max_gap_s)
    return reaction < base.reaction_limit_s


def _bisect(task):
    """Outer edge of the first range band where the line's property holds.

    A coarse ladder finds where the band starts (large off-boresight shots
    can fail at the shortest ranges), then bisection refines its far edge.
    """
    _, base, _, _ = task
    ladder = [r for r in COARSE_KM if base.min_range_m <= r*1000 <= base.max_range_m] or [base.min_range_m/1000]
    lo = hi = None
    for km in ladder:
        inside = _predicate(task, km*1000)
        if inside:
            lo = km*1000
        elif lo is not None:
            hi = km*1000
            break
    if lo is None:
        return None
    if hi is None:
        return lo
    while hi-lo > base.tolerance_m:
        mid = (lo+hi)/2
        lo, hi = (mid, hi) if _predicate(task, mid) else (lo, mid)
    return lo


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--missile", required=True)
    parser.add_argument("--aircraft", required=True, help="evader FM catalog id")
    parser.add_argument("--mass-kg", type=float, required=True)
    parser.add_argument("--missile-sim", type=Path, default=ew.ROOT.parent/"missle_sim")
    parser.add_argument("--azimuths-deg", default="0,15,30,45,60")
    parser.add_argument("--alt-diffs-m", default="0")
    parser.add_argument("--launch-altitude-m", type=float, default=8000.)
    parser.add_argument("--launch-speed-kmh", type=float, default=1200.)
    parser.add_argument("--target-speed-kmh", type=float, default=1000.)
    parser.add_argument("--chaff-rcs-ratio", type=float, default=1.)
    parser.add_argument("--reaction-limit-s", type=float, default=3.)
    parser.add_argument("--step-s", type=float, default=.25)
    parser.add_argument("--max-gap-s", type=float, default=.5)
    parser.add_argument("--min-range-km", type=float, default=2.)
    parser.add_argument("--max-range-km", type=float, default=120.)
    parser.add_argument("--tolerance-km", type=float, default=.25)
    parser.add_argument("--max-time-s", type=float, default=150.)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    base = SimpleNamespace(launch_speed_kmh=args.launch_speed_kmh, launch_altitude_m=args.launch_altitude_m,
                           target_speed_kmh=args.target_speed_kmh, max_time_s=args.max_time_s,
                           reaction_limit_s=args.reaction_limit_s, step_s=args.step_s, max_gap_s=args.max_gap_s,
                           min_range_m=args.min_range_km*1000, max_range_m=args.max_range_km*1000,
                           tolerance_m=args.tolerance_km*1000)
    azimuths = [float(x) for x in args.azimuths_deg.split(",")]
    diffs = [float(x) for x in args.alt_diffs_m.split(",")]
    tasks = [(line, base, az, dh) for az in azimuths for dh in diffs for line in LINES]
    chaff = (args.chaff_rcs_ratio, 1., 1, 30) if args.chaff_rcs_ratio > 0 else None
    init = (str(args.missile_sim), args.missile, args.aircraft, args.mass_kg, True, chaff, "rwr",
            (0., .1, 1.), "geometric_mainlobe")
    began = time.perf_counter()
    with ProcessPoolExecutor(args.workers, initializer=ew._init, initargs=init) as pool:
        results = list(pool.map(_bisect, tasks, chunksize=1))
    elapsed = time.perf_counter()-began
    rows = {}
    for (line, _, az, dh), value in zip(tasks, results):
        rows.setdefault((az, dh), dict(azimuth_deg=az, alt_diff_m=dh))[line] = value
    meta = dict(missile=args.missile, evader_aircraft=args.aircraft, evader_mass_kg=args.mass_kg,
                launch_altitude_m=args.launch_altitude_m, launch_speed_kmh=args.launch_speed_kmh,
                target_speed_kmh=args.target_speed_kmh, chaff_rcs_ratio=args.chaff_rcs_ratio,
                reaction_limit_s=args.reaction_limit_s, max_gap_s=args.max_gap_s, step_s=args.step_s,
                range_bracket_km=[args.min_range_km, args.max_range_km], tolerance_km=args.tolerance_km,
                clutter="geometric_mainlobe", perception="rwr", target="straight, course 0 hot / 180 cold",
                created=time.strftime("%Y-%m-%d %H:%M:%S"), elapsed_s=round(elapsed, 1))
    out = args.out or ew.ROOT/"data"/"envelope"/(
        f"{args.missile}__{args.aircraft}__{int(args.launch_altitude_m)}m_{int(args.launch_speed_kmh)}kmh"
        f"__chaff{args.chaff_rcs_ratio:g}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(meta=meta, rows=list(rows.values())), ensure_ascii=False, indent=1))
    km = lambda v: "  none" if v is None else f"{v/1000:6.2f}"  # noqa: E731
    print(f"{len(tasks)} bisections in {elapsed:.0f} s -> {out}")
    print("azimuth  alt diff   " + "  ".join(f"{line:>9s}" for line in LINES))
    for row in rows.values():
        print(f"{row['azimuth_deg']:6g}°  {row['alt_diff_m']:7g} m  " + "  ".join(f"{km(row[l]):>9s}" for l in LINES))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
