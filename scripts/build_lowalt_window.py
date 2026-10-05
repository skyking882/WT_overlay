#!/usr/bin/env python3
"""Offline low-target window table for the B-scope low-target mode.

For one ownship state (altitude, TAS) and missile, for each launch pitch and
target off-boresight azimuth, finds the launch-range window in which the
missile hits a straight, level target at every tabulated height near the
surface (default 25-35 m, where multipath is worst) on both a hot and a cold
course, with multipath gain 0.5 (calibrated on one user recollection: ~20 m
target hit at ~17 km from 7000 m, M1.5). The same window is also stored for a
reference height alone (20 m). A window is the longest contiguous run of hits
on the range ladder (``None`` when nothing hits); the run does not extend past
a ladder gap, so the edges carry the ladder's resolution. Only the
multipath-limited far edge and the look-down-limited near edge are covered:
the target never reacts.

Writes JSON under data/lowalt_window/; all limits of build_lowalt_table.py and
missile_sim apply (flat sea-level surface, units-only proximity fuse, datalink
target truth).

    python scripts/build_lowalt_window.py --missile us_aim_120c_5 --launch-altitude-m 7000 --launch-speed-kmh 1700
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_lowalt_table import ROOT, _init, _run, ladder  # noqa: E402


def window(ranges, hits):
    """Longest contiguous run of hits -> (first, last) range; ties go to the farther run."""
    best, start = None, None
    for i, hit in enumerate(list(hits)+[False]):
        if hit and start is None:
            start = i
        elif not hit and start is not None:
            if best is None or i-start >= best[1]-best[0]+1:
                best = (start, i-1)
            start = None
    return None if best is None else (ranges[best[0]], ranges[best[1]])


def windows(cells, ranges, heights, courses):
    """Window where every (height, course) case hits."""
    hit = {(c["height_m"], c["course_deg"], c["range_m"]): c["hit"] for c in cells}
    return window(ranges, [all(hit[(h, c, r)] for h in heights for c in courses) for r in ranges])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--missile", required=True)
    parser.add_argument("--missile-sim", type=Path, default=ROOT.parent/"missle_sim")
    parser.add_argument("--launch-altitude-m", type=float, required=True)
    parser.add_argument("--launch-speed-kmh", type=float, required=True)
    parser.add_argument("--target-speed-kmh", type=float, default=900.)
    parser.add_argument("--pitches-deg", default="-30,-15,0,15,30,45")
    parser.add_argument("--azimuths-deg", default="0,30,60")
    parser.add_argument("--heights-m", default="25,30,35", help="worst-case band (all must hit)")
    parser.add_argument("--reference-height-m", type=float, default=20.)
    parser.add_argument("--courses-deg", default="0,180")
    parser.add_argument("--ranges-km", default="2:10:0.5,11:40:1")
    parser.add_argument("--gain", type=float, default=.5)
    parser.add_argument("--max-time-s", type=float, default=150.)
    parser.add_argument("--early-miss-s", type=float, default=2.)
    parser.add_argument("--clutter-depression-deg", type=float, default=2.)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    pitches = [float(x) for x in args.pitches_deg.split(",")]
    azimuths = [float(x) for x in args.azimuths_deg.split(",")]
    heights = [float(x) for x in args.heights_m.split(",")]
    courses = [float(x) for x in args.courses_deg.split(",")]
    ranges = [km*1000 for km in ladder(args.ranges_km)]
    every = sorted(set(heights) | {args.reference_height_m})
    bases = {(p, az): dict(launch_altitude_m=args.launch_altitude_m, launch_speed_kmh=args.launch_speed_kmh,
                           launch_pitch_deg=p, target_speed_kmh=args.target_speed_kmh, azimuth_deg=az,
                           max_time_s=args.max_time_s, early_miss_s=args.early_miss_s)
             for p in pitches for az in azimuths}
    keys = [(p, az, h, c, r) for p in pitches for az in azimuths for h in every for c in courses for r in ranges]
    jobs = [(bases[(p, az)], h, c, r, args.gain, []) for p, az, h, c, r in keys]
    began = time.perf_counter()
    with ProcessPoolExecutor(args.workers, initializer=_init,
                             initargs=(str(args.missile_sim), args.missile, "look_down_angle",
                                       args.clutter_depression_deg)) as pool:
        cells = list(pool.map(_run, jobs, chunksize=8))
    elapsed = time.perf_counter()-began
    for (p, az, *_), cell in zip(keys, cells):
        cell.update(pitch_deg=p, azimuth_deg=az)
    rows = []
    for p in pitches:
        for az in azimuths:
            group = [c for c in cells if c["pitch_deg"] == p and c["azimuth_deg"] == az]
            rows.append(dict(pitch_deg=p, azimuth_deg=az,
                             worst=windows(group, ranges, heights, courses),
                             reference=windows(group, ranges, [args.reference_height_m], courses)))
    meta = dict(missile=args.missile, launch_altitude_m=args.launch_altitude_m,
                launch_speed_kmh=args.launch_speed_kmh, target_speed_kmh=args.target_speed_kmh,
                gain=args.gain, heights_m=heights, reference_height_m=args.reference_height_m,
                courses_deg=courses, ranges_m=ranges, pitches_deg=pitches, azimuths_deg=azimuths,
                clutter="look_down_angle", clutter_min_depression_deg=args.clutter_depression_deg,
                early_miss_s=args.early_miss_s, target="straight and level, no reaction",
                surface="flat, sea level", created=time.strftime("%Y-%m-%d %H:%M:%S"),
                elapsed_s=round(elapsed, 1), runs=len(jobs))
    out = args.out or ROOT/"data"/"lowalt_window"/(
        f"{args.missile}__{int(args.launch_altitude_m)}m_{int(args.launch_speed_kmh)}kmh.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(meta=meta, rows=rows), ensure_ascii=False, indent=1))
    km = lambda w: "none" if w is None else f"{w[0]/1000:g}-{w[1]/1000:g}"  # noqa: E731
    print(f"{len(jobs)} runs in {elapsed:.0f} s -> {out}")
    for row in rows:
        print(f"  pitch {row['pitch_deg']:4g}°  az {row['azimuth_deg']:3g}°  worst {km(row['worst']):>9s}  "
              f"{args.reference_height_m:g} m {km(row['reference']):>9s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
