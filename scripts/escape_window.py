#!/usr/bin/env python3
"""Offline escape-window sweep: a WT_overlay FM evader against a missile_sim missile.

For each launch range and evasion kind, starts the FM pilot at each grid time
and runs missile_sim's general runtime. An escape is any termination other than
the proximity fuse, however narrow the miss; each window reports its closest
miss. Windows are contiguous escaping start times at the grid resolution.
With --early-miss-s (default 2 s) a burnt-out missile that has passed, is
slower and opening, or cannot arrive in its remaining time is stopped early;
near misses keep their exact closest approach, but a slow chase cut this way
reports the distance at the cut. Results inherit every limitation of both
models (see wt_overlay/escape.py and the missile_sim README). Not used by the
HUD.

    python scripts/escape_window.py --aircraft f_16c_block_50 --mass-kg 12000 --missile us_aim_120c_5
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
import sys
import time
from concurrent.futures import ProcessPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wt_overlay.escape import KINDS, EvasionPilot, Perception, fm_evader_factory  # noqa: E402
from wt_overlay.fm import load_aircraft  # noqa: E402
from wt_overlay.turn import ManeuverModel  # noqa: E402

_CTX = {}


def _init(missile_sim, missile_id, aircraft, mass, afterburner, chaff, perception, crank, clutter, depression=None,
          cw=False, launcher_gimbal_deg=None, require_lock=False):
    sys.path.insert(0, str(Path(missile_sim) / "src"))
    from aim120_model.chaff import ChaffProgram, ChaffSpec, chaffing_factory
    from aim120_model.profile_catalog import load_profile_catalog
    from aim120_model.public_api import simulate
    from aim120_model.target import TargetState
    from missile_gui.library import scan_library
    _, catalog = load_profile_catalog(Path(missile_sim))
    profiles, errors = scan_library(catalog["profiles_dir"], Path(missile_sim))
    found = {p["missile_id"]: p for p in profiles}
    if missile_id not in found:
        raise SystemExit(f"unknown missile {missile_id}; library errors: {errors[:3]}")
    profile = found[missile_id]
    _CTX.update(simulate=simulate, state=TargetState, profile=profile, clutter=clutter,
                clutter_kwargs={**({} if depression is None else {"clutter_min_depression_deg": depression}),
                                **({"cw_on_clear_beam": True} if cw else {}),
                                **({} if launcher_gimbal_deg is None else
                                   {"launcher_radar_gimbal_deg": launcher_gimbal_deg}),
                                **({"require_seeker_lock": True} if require_lock else {})},
                model=ManeuverModel(load_aircraft(aircraft), mass, afterburner))
    if perception == "rwr":
        # Marker visible while the motor burns; RWR sees the active seeker.
        burn = sum(float(s.get("fire_delay_s") or 0.)+s["duration_s"] for s in profile["propulsion"]["stages"])
        seeker = ((profile["guidance"].get("sensor_model") or {}).get("radar_seeker") or {}).get("receiver") or {}
        if not seeker.get("range_m"):
            raise SystemExit(f"{missile_id}: no active seeker range for --perception rwr")
        _CTX["perception"] = Perception(burn, float(seeker["range_m"]), *crank)
    if chaff is not None:
        ratio, interval, salvo, total = chaff
        spec = ChaffSpec(rcs_ratio=ratio)
        # Chaff starts with the evasion; the aircraft RCS is the unit.
        _CTX["chaff"] = lambda inner, start: chaffing_factory(
            inner, spec, ChaffProgram(start, interval_s=interval, per_salvo=salvo, total=total))


def _scenario(args, range_m):
    return dict(launch_speed_kmh=args.launch_speed_kmh, launch_altitude_m=args.launch_altitude_m,
                launch_pitch_deg=0, launch_heading_deg=0, target_speed_kmh=args.target_speed_kmh,
                target_altitude_m=args.target_altitude_m, initial_distance_m=range_m,
                target_azimuth_deg=args.azimuth_deg, target_heading_deg=args.target_course_deg,
                target_course_reference="relative_to_los", target_vertical_heading_deg=0,
                target_constant_turn_g=args.target_turn_g, max_simulation_time_s=args.max_time_s,
                observation_mode=args.observation_mode, loft_enabled=args.loft)


def _run(job):
    scenario, pilot, early_miss_s = job
    factory = None if pilot is None else fm_evader_factory(pilot, _CTX["model"], _CTX["state"],
                                                           _CTX.get("perception"))
    if pilot is not None and "chaff" in _CTX:
        factory = _CTX["chaff"](factory, pilot.start_s)
    result = _CTX["simulate"](_CTX["profile"], scenario, target_factory=factory, early_miss_s=early_miss_s,
                              clutter_model=_CTX["clutter"], **_CTX["clutter_kwargs"])
    summary, evader = result["summary"], result["model"].get("target_model") or {}
    return dict(range_m=scenario["initial_distance_m"], kind=pilot.kind if pilot else "none",
                plan=pilot.describe() if pilot else None,
                start_s=pilot.start_s if pilot else 0.0, event=summary["termination_event"],
                flight_time_s=round(summary["flight_time_s"], 3),
                min_distance_m=round(summary["minimum_distance_m"], 1),
                # A faulted evader (ground impact, FM table range) is never an escape.
                escaped=summary["termination_event"] not in ("proximity_fuse", "hit") and not evader.get("fault"),
                radar_reject=summary.get("last_radar_reject_reason"),
                min_speed_kmh=round(evader["min_speed_mps"]*3.6) if evader else None,
                min_altitude_m=round(evader["min_altitude_m"]) if evader else None,
                fault=evader.get("fault"),
                decoy_track_s=round(summary.get("decoy_track_time_s") or 0., 2),
                launcher_steer_s=round(evader.get("launcher_steer_s") or 0., 2))


def _windows(rows, key="escaped"):
    out, current = [], None
    for row in sorted(rows, key=lambda r: r["start_s"]):
        if row[key]:
            current = [row["start_s"], row["start_s"]] if current is None else [current[0], row["start_s"]]
        elif current is not None:
            out.append(tuple(current))
            current = None
    if current is not None:
        out.append(tuple(current))
    return out


def _refine(pool, rows, args, pilot, early):
    """Bisect every coarse interval whose outcome flips down to --fine-step-s.

    Intervals whose two ends agree are not searched, so a window narrower than
    the coarse step that falls between two samples can still be missed.
    """
    fine = args.fine_step_s
    snap = lambda x: round(round(x/fine)*fine, 6)  # noqa: E731
    rounds = 0
    while True:
        groups = {}
        for row in rows:
            groups.setdefault((row["range_m"], row["kind"]), []).append(row)
        jobs = []
        for (rng, kind), group in groups.items():
            group.sort(key=lambda r: r["start_s"])
            for a, b in zip(group, group[1:]):
                if a["escaped"] != b["escaped"] and b["start_s"]-a["start_s"] > fine+1e-9:
                    mid = snap((a["start_s"]+b["start_s"])/2)
                    if a["start_s"] < mid < b["start_s"]:
                        jobs.append((_scenario(args, rng), pilot(kind, mid), early))
        if not jobs:
            return rounds
        rounds += 1
        for row in pool.map(_run, jobs):
            row["stage"] = "refine"
            rows.append(row)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--aircraft", required=True, help="catalog id, e.g. f_16c_block_50")
    parser.add_argument("--mass-kg", type=float, required=True, help="total mass; not inferred")
    parser.add_argument("--no-afterburner", dest="afterburner", action="store_false")
    parser.add_argument("--missile", default="us_aim_120c_5")
    parser.add_argument("--missile-sim", type=Path, default=ROOT.parent / "missle_sim")
    parser.add_argument("--ranges-km", default="15,20,25,30,40")
    parser.add_argument("--kinds", default=",".join(KINDS))
    parser.add_argument("--start-max-s", type=float, default=40.)
    parser.add_argument("--start-step-s", type=float, default=1.,
                        help="coarse grid; a window narrower than this can be missed")
    parser.add_argument("--fine-step-s", type=float, default=.25,
                        help="bisect outcome flips down to this; 0 disables refinement")
    parser.add_argument("--max-load", type=float, default=9.)
    parser.add_argument("--alpha-max-deg", type=float, default=20.)
    parser.add_argument("--roll-rate-deg-s", type=float, default=120.)
    parser.add_argument("--dive-deg", type=float, default=0.)
    parser.add_argument("--launch-altitude-m", type=float, default=8000.)
    parser.add_argument("--launch-speed-kmh", type=float, default=1100.)
    parser.add_argument("--target-altitude-m", type=float, default=8000.)
    parser.add_argument("--target-speed-kmh", type=float, default=1000.)
    parser.add_argument("--target-course-deg", type=float, default=0., help="relative to LOS; 0 = head-on")
    parser.add_argument("--target-turn-g", type=float, default=0.,
                        help="target's level turn before it reacts (missile_sim target_constant_turn_g; sign = side)")
    parser.add_argument("--observation-mode", default="sensor_track", choices=("sensor_track", "ideal_truth"))
    parser.add_argument("--loft", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-time-s", type=float, default=120.)
    parser.add_argument("--early-miss-s", type=float, default=2.,
                        help="stop a burnt-out missile opening from a faster target this long; 0 runs to lifetime")
    parser.add_argument("--perception", choices=("rwr", "truth"), default="rwr",
                        help="rwr: evade the missile only while its motor burns or its seeker is active, "
                             "else the launcher; truth: always the true missile")
    parser.add_argument("--clutter", choices=("look_down_angle", "geometric_mainlobe", "look_down"),
                        default="look_down_angle",
                        help="missile_sim clutter notch condition; look_down flips on centimetres at co-altitude")
    parser.add_argument("--clutter-depression-deg", type=float, default=2.,
                        help="look_down_angle: minimum line-of-sight depression for ground clutter")
    parser.add_argument("--no-cw", dest="cw", action="store_false",
                        help="disable: chaff captures a beaming target with no clutter behind it (CW mode)")
    parser.add_argument("--azimuth-deg", type=float, default=0.,
                        help="target off-boresight angle at launch (missile_sim target_azimuth_deg)")
    parser.add_argument("--crank-deg", type=float, default=0.,
                        help="after launch the shooter turns away until the initial line of sight is this "
                             "far off its nose (only changes what the target's RWR shows; needs rwr)")
    parser.add_argument("--crank-g", type=float, default=4.)
    parser.add_argument("--crank-start-s", type=float, default=1.)
    parser.add_argument("--chaff-rcs-ratio", type=float, default=0.,
                        help="peak bundle RCS / aircraft RCS (an assumption); 0 = no chaff")
    parser.add_argument("--chaff-interval-s", type=float, default=1.)
    parser.add_argument("--chaff-salvo", type=int, default=1)
    parser.add_argument("--chaff-total", type=int, default=30)
    parser.add_argument("--robust-miss-m", type=float, default=0.,
                        help="optionally report escapes closer than this separately as marginal")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    ranges = [float(x)*1000 for x in args.ranges_km.split(",")]
    kinds = [k for k in args.kinds.split(",") if k]
    starts = [round(i*args.start_step_s, 6) for i in range(int(round(args.start_max_s/args.start_step_s))+1)]

    def pilot(kind, start):
        return EvasionPilot(kind, start, max_load=args.max_load, alpha_max_deg=args.alpha_max_deg,
                            roll_rate_deg_s=args.roll_rate_deg_s, dive_deg=args.dive_deg,
                            throttle_percent=110. if args.afterburner else 100.)

    for kind in kinds:
        pilot(kind, 0.)  # Validate before spawning workers.
    early = args.early_miss_s if args.early_miss_s > 0 else None
    began = time.perf_counter()
    chaff = ((args.chaff_rcs_ratio, args.chaff_interval_s, args.chaff_salvo, args.chaff_total)
             if args.chaff_rcs_ratio > 0 else None)
    # ENU bearing of the target is -azimuth (missile_sim z = -north); launch heading is 0.
    bearing = -math.radians(args.azimuth_deg)
    side = 1. if bearing >= 0 else -1.
    crank_rad = bearing-side*math.radians(args.crank_deg) if math.radians(args.crank_deg) > abs(bearing) else 0.
    if args.crank_deg and args.perception != "rwr":
        raise SystemExit("--crank-deg only changes the target's perception; use --perception rwr")
    crank = (crank_rad, args.crank_g*9.80665/(args.launch_speed_kmh/3.6), args.crank_start_s)
    init = (str(args.missile_sim), args.missile, args.aircraft, args.mass_kg, args.afterburner, chaff,
            args.perception, crank, args.clutter,
            args.clutter_depression_deg if args.clutter == "look_down_angle" else None, args.cw)
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=init) as pool:
        baselines = {r["range_m"]: r for r in pool.map(_run, [(_scenario(args, rng), None, early) for rng in ranges])}
        jobs = [(_scenario(args, rng), pilot(kind, start), early) for rng in ranges for kind in kinds for start in starts
                # A start after the non-evading intercept cannot change the outcome.
                if baselines[rng]["escaped"] or start < baselines[rng]["flight_time_s"]]
        rows = list(pool.map(_run, jobs, chunksize=4))
        coarse_runs = len(rows)
        for row in rows:
            row["stage"] = "coarse"
        rounds = _refine(pool, rows, args, pilot, early) if 0 < args.fine_step_s < args.start_step_s else 0
    elapsed = time.perf_counter()-began
    for row in baselines.values():
        row["stage"] = "baseline"
    for row in list(baselines.values())+rows:
        # A seeker that loses lock late can still pass just outside the fuse
        # radius; in game that may be a hit. Only wide misses count as robust.
        row["robust"] = row["escaped"] and row["min_distance_m"] >= args.robust_miss_m
        row["marginal"] = row["escaped"] and not row["robust"]

    out = args.out or ROOT/"outputs"/"escape_window"/(
        f"{args.aircraft}_vs_{args.missile}_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0] if rows else baselines[ranges[0]]))
        writer.writeheader()
        writer.writerows(list(baselines.values())+rows)

    print(f"{args.aircraft} {args.mass_kg:.0f} kg vs {args.missile} ({args.observation_mode}) | "
          f"launch {args.launch_altitude_m:.0f} m {args.launch_speed_kmh:.0f} km/h | target "
          f"{args.target_altitude_m:.0f} m {args.target_speed_kmh:.0f} km/h course {args.target_course_deg:g} | "
          f"{args.max_load:g} g, AoA ≤ {args.alpha_max_deg:g}°, dive {args.dive_deg:g}° | grid {args.start_step_s:g} s"
          f"{f' refined to {args.fine_step_s:g} s' if rounds else ''} | "
          f"clutter {args.clutter} | perception {args.perception} | off-boresight {args.azimuth_deg:g}°, crank "
          f"{f'{args.crank_deg:g}° at {args.crank_g:g} g' if crank_rad else 'none'} | early miss {f'{early:g} s' if early else 'off'} | chaff "
          f"{f'ratio {args.chaff_rcs_ratio:g}, {args.chaff_salvo}/{args.chaff_interval_s:g} s x{args.chaff_total}' if chaff else 'off'}")
    print(f"{len(rows)+len(baselines)} runs ({coarse_runs} coarse, {len(rows)-coarse_runs} refine in {rounds} rounds) "
          f"in {elapsed:.1f} s on {args.workers} workers -> {out}")
    faults = sum(1 for r in rows if r["fault"])
    if faults:
        kinds_of = sorted({r["fault"] for r in rows if r["fault"]})
        print(f"warning: {faults} evader faults counted as not escaped: {'; '.join(kinds_of)}")
    for rng in ranges:
        base = baselines[rng]
        note = "no evasion: miss" if base["escaped"] else f"no evasion: hit at {base['flight_time_s']:.1f} s"
        print(f"\n{rng/1000:g} km ({note})")
        for kind in kinds:
            subset = [r for r in rows if r["range_m"] == rng and r["kind"] == kind]
            fmt = lambda spans: ", ".join(f"{a:g}-{b:g} s" if a != b else f"{a:g} s" for a, b in spans) or "none"  # noqa: E731
            marginal = [r for r in subset if r["marginal"]]
            robust = [r for r in subset if r["robust"]]
            speed = (f" | min speed {min(r['min_speed_kmh'] for r in robust)} km/h, "
                     f"closest miss {min(r['min_distance_m'] for r in robust):g} m") if robust else ""
            label = "robust" if args.robust_miss_m > 0 else "escape"
            line = f"  {kind:5s} {label}: {fmt(_windows(subset, 'robust'))}{speed}"
            if marginal:
                line += (f" | marginal (<{args.robust_miss_m:g} m): {fmt(_windows(subset, 'marginal'))}, "
                         f"closest {min(r['min_distance_m'] for r in marginal):g} m")
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
