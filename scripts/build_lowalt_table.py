#!/usr/bin/env python3
"""Offline low-altitude table: does the missile hit a straight, level target near the surface?

For one ownship state (altitude, TAS, launch pitch) and missile, runs missile_sim
against a non-evading target flying straight and level at each height above a
flat surface, each course relative to the line of sight (0 = hot, 180 = cold)
and each range on a ladder, once per multipath gain (0 = multipath off; see
missile_sim ``simulate(multipath_gain=...)``, an uncalibrated assumption).
Heights where the missile's ``multipathEffect`` strength is 0 reuse the gain-0
run, which is identical by construction.

Per cell it records the outcome (hit = proximity fuse), closest approach,
terminal dive angle, and for each RWR elevation limit how long before the end
the target's RWR could first see the missile's seeker tracking it from within
that elevation of its horizon (None = never: the missile stays in the blind
zone). Receiver coverage is a per-aircraft assumption (e.g. AN/ALR-56M
receivers are 90 deg wide in elevation); the azimuth is taken as all-round.

Flat surface at sea level, target height = altitude, proximity fuse on units
only. Writes JSON under data/lowalt/.

    python scripts/build_lowalt_table.py --missile cn_pl12 --launch-altitude-m 8000 --launch-speed-kmh 1200
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
HIT_EVENTS = ("proximity_fuse", "hit")
_CTX = {}


def _init(missile_sim, missile_id, clutter, depression):
    sys.path.insert(0, str(Path(missile_sim)/"src"))
    from aim120_model.profile_catalog import load_profile_catalog
    from aim120_model.public_api import simulate
    from missile_gui.library import scan_library
    _, catalog = load_profile_catalog(Path(missile_sim))
    profiles, errors = scan_library(catalog["profiles_dir"], Path(missile_sim))
    found = {p["missile_id"]: p for p in profiles}
    if missile_id not in found:
        raise SystemExit(f"unknown missile {missile_id}; library errors: {errors[:3]}")
    _CTX.update(simulate=simulate, profile=found[missile_id], clutter=clutter,
                clutter_kwargs={} if depression is None else {"clutter_min_depression_deg": depression})


def _strength_nodes(profile):
    """multipathEffect as (height m, strength) nodes, or None when the seeker has none."""
    table = ((profile["guidance"].get("sensor_model") or {}).get("radar_seeker") or {}).get("multipath_effect")
    return None if not table else list(zip(table[::2], table[1::2]))


def strength(nodes, height_m):
    """Piecewise linear, held at the end nodes (same reading as missile_sim)."""
    if height_m <= nodes[0][0]:
        return nodes[0][1]
    for (x0, y0), (x1, y1) in zip(nodes, nodes[1:]):
        if height_m <= x1:
            return y0+(y1-y0)*(height_m-x0)/(x1-x0)
    return nodes[-1][1]


def _scenario(base, height_m, course_deg, range_m):
    return dict(launch_speed_kmh=base["launch_speed_kmh"], launch_altitude_m=base["launch_altitude_m"],
                launch_pitch_deg=base["launch_pitch_deg"], launch_heading_deg=0,
                target_speed_kmh=base["target_speed_kmh"], target_altitude_m=height_m,
                initial_distance_m=range_m, target_azimuth_deg=base["azimuth_deg"],
                target_heading_deg=course_deg, target_course_reference="relative_to_los",
                target_vertical_heading_deg=0, target_constant_turn_g=0,
                max_simulation_time_s=base["max_time_s"], observation_mode="sensor_track", loft_enabled=True)


def elevation_deg(missile, target):
    """Missile elevation above the target's horizon (missile_sim axes: y up)."""
    dx, dy, dz = (m-t for m, t in zip(missile, target))
    return math.degrees(math.atan2(dy, math.hypot(dx, dz)))


def warning_s(samples, limits_deg):
    """Per elevation limit: time before the end at which the seeker first tracks from inside it."""
    end = samples[-1]["time_s"]
    out = {}
    for limit in limits_deg:
        first = next((s["time_s"] for s in samples if s["seeker_state"] == "track"
                      and elevation_deg(s["position_m"], s["target_position_m"]) <= limit), None)
        out[f"{limit:g}"] = None if first is None else round(end-first, 2)
    return out


def _run(job):
    base, height_m, course_deg, range_m, gain, limits = job
    result = _CTX["simulate"](_CTX["profile"], _scenario(base, height_m, course_deg, range_m),
                              early_miss_s=base["early_miss_s"], clutter_model=_CTX["clutter"],
                              multipath_gain=gain or None, **_CTX["clutter_kwargs"])
    summary, samples = result["summary"], result["samples"]
    v = samples[-1]["velocity_mps"]
    acquired = next((s["time_s"] for s in samples if s["seeker_state"] == "track"), None)
    return dict(height_m=height_m, course_deg=course_deg, range_m=range_m, gain=gain,
                event=summary["termination_event"], hit=summary["termination_event"] in HIT_EVENTS,
                min_distance_m=round(summary["minimum_distance_m"], 1),
                flight_time_s=round(summary["flight_time_s"], 2),
                dive_deg=round(math.degrees(math.atan2(-v[1], math.hypot(v[0], v[2]))), 1),
                acquired_s=None if acquired is None else round(acquired, 2),
                rwr_warning_s=warning_s(samples, limits))


def ladder(text):
    """'2:10:0.5,12,15' -> sorted unique km values (start:stop:step inclusive)."""
    values = set()
    for part in text.split(","):
        if ":" in part:
            start, stop, step = (float(x) for x in part.split(":"))
            n = int(round((stop-start)/step))
            values.update(round(start+i*step, 6) for i in range(n+1))
        elif part:
            values.add(float(part))
    return sorted(values)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--missile", required=True)
    parser.add_argument("--missile-sim", type=Path, default=ROOT.parent/"missle_sim")
    parser.add_argument("--launch-altitude-m", type=float, default=8000.)
    parser.add_argument("--launch-speed-kmh", type=float, default=1200.)
    parser.add_argument("--launch-pitch-deg", type=float, default=0.)
    parser.add_argument("--target-speed-kmh", type=float, default=900.)
    parser.add_argument("--azimuth-deg", type=float, default=0., help="target off-boresight angle at launch")
    parser.add_argument("--heights-m", default="2,5,10,15,20,25,30,35,40,45,50,55,60,80,100,150,300")
    parser.add_argument("--courses-deg", default="0,45,90,135,180")
    parser.add_argument("--ranges-km", default="2:10:0.5,11:60:1")
    parser.add_argument("--gains", default="0,0.5,1", help="multipath gains; 0 = off")
    parser.add_argument("--rwr-elevations-deg", default="30,45,60,90")
    parser.add_argument("--max-time-s", type=float, default=150.)
    parser.add_argument("--early-miss-s", type=float, default=2.)
    parser.add_argument("--clutter", choices=("look_down_angle", "geometric_mainlobe", "look_down"),
                        default="look_down_angle")
    parser.add_argument("--clutter-depression-deg", type=float, default=2.)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    heights = [float(x) for x in args.heights_m.split(",")]
    courses = [float(x) for x in args.courses_deg.split(",")]
    ranges = [km*1000 for km in ladder(args.ranges_km)]
    gains = sorted({float(x) for x in args.gains.split(",")} | {0.})
    limits = [float(x) for x in args.rwr_elevations_deg.split(",")]
    base = dict(launch_altitude_m=args.launch_altitude_m, launch_speed_kmh=args.launch_speed_kmh,
                launch_pitch_deg=args.launch_pitch_deg, target_speed_kmh=args.target_speed_kmh,
                azimuth_deg=args.azimuth_deg, max_time_s=args.max_time_s, early_miss_s=args.early_miss_s)
    depression = args.clutter_depression_deg if args.clutter == "look_down_angle" else None
    _init(str(args.missile_sim), args.missile, args.clutter, depression)
    nodes = _strength_nodes(_CTX["profile"])
    if nodes is None and gains != [0.]:
        raise SystemExit(f"{args.missile}: no multipathEffect table; use --gains 0")
    active = [h for h in heights if nodes is not None and strength(nodes, h) > 0]
    jobs = [(base, h, c, r, g, limits) for h in heights for c in courses for r in ranges for g in gains
            if g == 0. or h in active]
    began = time.perf_counter()
    with ProcessPoolExecutor(args.workers, initializer=_init,
                             initargs=(str(args.missile_sim), args.missile, args.clutter, depression)) as pool:
        cells = list(pool.map(_run, jobs, chunksize=8))
    elapsed = time.perf_counter()-began
    off = {(c["height_m"], c["course_deg"], c["range_m"]): c for c in cells if c["gain"] == 0.}
    for g in gains[1:]:
        cells += [dict(cell, gain=g) for (h, _, _), cell in off.items() if h not in active]
    meta = dict(missile=args.missile, launch_altitude_m=args.launch_altitude_m,
                launch_speed_kmh=args.launch_speed_kmh, launch_pitch_deg=args.launch_pitch_deg,
                target_speed_kmh=args.target_speed_kmh, azimuth_deg=args.azimuth_deg, heights_m=heights,
                courses_deg=courses, ranges_m=ranges, gains=gains, rwr_elevations_deg=limits,
                multipath_effect=nodes, clutter=args.clutter, clutter_min_depression_deg=depression,
                early_miss_s=args.early_miss_s, max_time_s=args.max_time_s,
                target="straight and level, no reaction", surface="flat, sea level",
                created=time.strftime("%Y-%m-%d %H:%M:%S"), elapsed_s=round(elapsed, 1), runs=len(jobs))
    out = args.out or ROOT/"data"/"lowalt"/(
        f"{args.missile}__{int(args.launch_altitude_m)}m_{int(args.launch_speed_kmh)}kmh"
        f"_pitch{args.launch_pitch_deg:g}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    cells.sort(key=lambda c: (c["gain"], c["course_deg"], c["height_m"], c["range_m"]))
    out.write_text(json.dumps(dict(meta=meta, cells=cells), ensure_ascii=False, separators=(",", ":")))
    print(f"{len(jobs)} runs in {elapsed:.0f} s -> {out}")
    for g in gains:
        print(f"\ngain {g:g}: hit ranges (km) by height, course")
        for c in courses:
            for h in heights:
                row = [cell for cell in cells if cell["gain"] == g and cell["course_deg"] == c and cell["height_m"] == h]
                print(f"  {c:5g}°  {h:5g} m  {bands(row)}")
    return 0


def bands(cells):
    """Contiguous hit intervals on the range ladder, e.g. '2-9.5, 14-31'."""
    out, start, last = [], None, None
    for cell in sorted(cells, key=lambda c: c["range_m"]):
        km = cell["range_m"]/1000
        if cell["hit"]:
            start = km if start is None else start
            last = km
        elif start is not None:
            out.append((start, last))
            start = None
    if start is not None:
        out.append((start, last))
    return ", ".join(f"{a:g}-{b:g}" if a != b else f"{a:g}" for a, b in out) or "none"


if __name__ == "__main__":
    raise SystemExit(main())
