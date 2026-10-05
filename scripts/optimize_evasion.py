#!/usr/bin/env python3
"""Explore the target's best evasive maneuvers against one missile.

For each engagement (launch range x target course at launch), step the
evasion start time T upward on a grid and ask whether ANY maneuver escapes:
first a library of single-segment building blocks (heading from "away" x turn
plane x dive), then, only if none escapes, a cross-entropy search over
two-segment combinations (both segments' heading/plane/dive and the switch
time). Reports the latest escaping start for level-only maneuvers, for the
library, and for the optimizer, with the maneuvers that achieve it.

The optimizer scores hits by how hard the missile had to work (longer flight,
lower terminal speed) so it has a gradient before the first escape. This is
exploration: a search that finds nothing is not proof that nothing exists.

    python scripts/optimize_evasion.py --missile cn_pl12 --aircraft f_16c_block_50 --mass-kg 12000
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import escape_window as ew  # noqa: E402

from wt_overlay.escape import EvasionPilot, fm_evader_factory  # noqa: E402

LIBRARY = [(target, plane, dive) for target in (0., 45., 90.) for plane in (0., 45., 70., 90.)
           for dive in (0., 20., 40.)]
LEVEL = [(0., 0., 0.), (90., 0., 0.)]
# Finer dives for the minimum-altitude-loss question.
LIBRARY_FINE = [(target, plane, dive) for target in (0., 45., 90.) for plane in (0., 45., 70., 90.)
                for dive in (0., 5., 10., 20., 30., 40.)]
# Cross-entropy search space: target1, plane1, dive1, switch_s, target2, plane2, dive2
# [, speed1_kmh, speed2_kmh when speed control is searched; >= FULL_KMH means hold afterburner].
BOUNDS = [(0., 150.), (0., 90.), (0., 55.), (1., 8.), (0., 150.), (0., 90.), (0., 55.)]
SPEED_BOUNDS = [(450., 1300.), (450., 1300.)]
FULL_KMH = 1250.
# Speed targets of the escape-surrogate library: hold full throttle (any value >= FULL_KMH), or
# idle + airbrake down to 800 / 600 km/h before and during the turn (user practice: slow, then beam).
FULL_SPEED_KMH = 1500.
SPEEDS_KMH = (FULL_SPEED_KMH, 800., 600.)
LIBRARY_SPEED = [p+(v,) for p in LIBRARY_FINE for v in SPEEDS_KMH]


def _speed(value):
    return None if value is None or value >= FULL_KMH else round(value)


def _pilot(start, params):
    """params: (target, plane, dive[, speed]) or the 7/9-element two-segment vector."""
    target, plane, dive = params[:3]
    speed1 = params[3] if len(params) == 4 else params[7] if len(params) == 9 else None
    then = ()
    if len(params) in (7, 9):
        speed2 = params[8] if len(params) == 9 else None
        then = ((round(params[3], 2), params[4], params[5], params[6], _speed(speed2)),)
    return EvasionPilot("beam", start, target_deg=target, plane_deg=plane, dive_deg=dive, then=then,
                        speed_kmh=_speed(speed1))


def _evaluate(job):
    """Run one evasion; returns (escaped, score, miss_m, altitude_lost_m, recommit).

    ``recommit`` (escaped runs only, else None): dict(defeat_s, back_s, nose_on_s, speed_kmh,
    altitude_m) - closest approach, seconds of afterburner level turn from there until the nose
    is within 30 deg of the launcher, the launch-relative time that happens, and the state then.
    """
    scenario, pilot = job
    chaff = ew._CTX.get("chaff")
    evaders = []
    fm = fm_evader_factory(pilot, ew._CTX["model"], ew._CTX["state"], ew._CTX.get("perception"))
    factory = lambda base: evaders.append(fm(base)) or evaders[-1]  # noqa: E731
    if chaff is not None:
        factory = chaff(factory, pilot.start_s)
    result = ew._CTX["simulate"](ew._CTX["profile"], scenario, target_factory=factory, early_miss_s=2.,
                                 clutter_model=ew._CTX["clutter"], **ew._CTX["clutter_kwargs"])
    summary, evader = result["summary"], result["model"].get("target_model") or {}
    escaped = summary["termination_event"] not in ("proximity_fuse", "hit") and not evader.get("fault")
    miss = summary["minimum_distance_m"]
    lost = scenario["target_altitude_m"]-evader.get("min_altitude_m", scenario["target_altitude_m"])
    if escaped:
        score = 10. + math.log10(max(miss, 1.))
    else:
        score = summary["flight_time_s"]/60. - summary["terminal_speed_kmh"]/6000.
    recommit = None
    if escaped and evaders:
        closest = min(result["samples"], key=lambda x: x["distance_to_target_m"])
        back = evaders[-1].recommit(float(closest["time_s"]))
        if back is not None:
            recommit = dict(defeat_s=round(float(closest["time_s"]), 2), back_s=round(back["seconds"], 2),
                            nose_on_s=round(float(closest["time_s"])+back["seconds"], 2),
                            speed_kmh=round(back["speed_mps"]*3.6), altitude_m=round(back["altitude_m"]))
    return escaped, score, miss, lost, recommit


def _cem(pool, scenario, start, seed_params, rng, population, elites, iterations, bounds):
    """Cross-entropy search over two-segment plans; returns (escaped, best params, best score, evaluations)."""
    mean = list(seed_params[:3])+[3., seed_params[0], seed_params[1], seed_params[2]]
    if len(bounds) == 9:
        seed_speed = seed_params[3] if len(seed_params) == 4 and seed_params[3] is not None else FULL_KMH
        mean += [seed_speed, seed_speed]
    sigma = [(hi-lo)/3 for lo, hi in bounds]
    best, evaluations = (False, None, -math.inf), 0
    for _ in range(iterations):
        candidates = [[min(hi, max(lo, rng.gauss(m, s))) for m, s, (lo, hi) in zip(mean, sigma, bounds)]
                      for _ in range(population)]
        results = list(pool.map(_evaluate, [(scenario, _pilot(start, c)) for c in candidates]))
        evaluations += len(candidates)
        ranked = sorted(zip(results, candidates), key=lambda rc: rc[0][1], reverse=True)
        (escaped, score, *_), params = ranked[0]
        if score > best[2]:
            best = (escaped, params, score)
        if best[0]:
            break
        top = [c for _, c in ranked[:elites]]
        mean = [sum(c[i] for c in top)/elites for i in range(len(mean))]
        sigma = [max(.05*(hi-lo), math.sqrt(sum((c[i]-mean[i])**2 for c in top)/elites))
                 for i, (lo, hi) in enumerate(bounds)]
    return best[0], best[1], best[2], evaluations


def _escaping(pool, scenario, plans, start, cache):
    """{plan: altitude_lost_m} of the plans that escape when started at ``start`` (memoised)."""
    if start not in cache:
        results = list(pool.map(_evaluate, [(scenario, _pilot(start, p)) for p in plans]))
        cache[start] = {p: (r[1], r[3], r[4]) for p, r in zip(plans, results) if r[0]}
    return cache[start]


def _latest(pool, scenario, plans, horizon_s, step_s, cache=None):
    """Latest start (on the step grid) at which some plan escapes, by bisection, with its best plan.

    Assumes escaping at T implies escaping at earlier starts; isolated windows
    after a gap are not searched (the full scan in non-fast mode does that).
    """
    cache = {} if cache is None else cache
    def best(start):
        good = _escaping(pool, scenario, plans, start, cache)
        return max(good, key=lambda p: good[p][0]) if good else None
    first = best(0.)
    if first is None:
        return None, None
    lo, hi, plan = 0, max(1, int(horizon_s/step_s)), first
    while hi-lo > 1:
        mid = (lo+hi)//2
        found = best(mid*step_s)
        if found is not None:
            lo, plan = mid, found
        else:
            hi = mid
    return lo*step_s, _pilot(lo*step_s, plan).describe()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--missile", required=True)
    parser.add_argument("--aircraft", required=True)
    parser.add_argument("--mass-kg", type=float, required=True)
    parser.add_argument("--missile-sim", type=Path, default=ew.ROOT.parent/"missle_sim")
    parser.add_argument("--ranges-km", default="7,10,13,16")
    parser.add_argument("--courses-deg", default="0,90,180")
    parser.add_argument("--launch-altitude-m", type=float, default=8000.)
    parser.add_argument("--launch-speed-kmh", type=float, default=1200.)
    parser.add_argument("--target-speed-kmh", type=float, default=1000.)
    parser.add_argument("--chaff-rcs-ratio", type=float, default=1.)
    parser.add_argument("--start-step-s", type=float, default=1.)
    parser.add_argument("--population", type=int, default=30)
    parser.add_argument("--elites", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--clutter", choices=("look_down_angle", "geometric_mainlobe", "look_down"),
                        default="look_down_angle")
    parser.add_argument("--clutter-depression-deg", type=float, default=2.)
    parser.add_argument("--no-cw", dest="cw", action="store_false")
    parser.add_argument("--min-altitude", action="store_true",
                        help="with --fast: finer dive library, and report the least altitude lost among escaping "
                             "plans at the latest start and 2 s earlier")
    parser.add_argument("--fast", action="store_true",
                        help="bisection on the start time with the building-block library only (no optimizer)")
    parser.add_argument("--speed-targets", default="",
                        help="comma list of library speed targets in km/h ('full' = afterburner); "
                             "when set, the optimizer also searches each segment's speed")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    rng = random.Random(args.seed)
    speeds = [None if x.strip() == "full" else float(x) for x in args.speed_targets.split(",") if x.strip()]
    base_library = LIBRARY_FINE if args.min_altitude else LIBRARY
    library = [p+(v,) for p in base_library for v in speeds] if speeds else base_library
    level = [p+(v,) for p in LEVEL for v in speeds] if speeds else LEVEL
    bounds = BOUNDS+SPEED_BOUNDS if speeds else BOUNDS
    chaff = (args.chaff_rcs_ratio, 1., 1, 30) if args.chaff_rcs_ratio > 0 else None
    init = (str(args.missile_sim), args.missile, args.aircraft, args.mass_kg, True, chaff, "rwr",
            (0., .1, 1.), args.clutter, args.clutter_depression_deg if args.clutter == "look_down_angle" else None,
            args.cw)
    cases = []
    began = time.perf_counter()
    with ProcessPoolExecutor(args.workers, initializer=ew._init, initargs=init) as pool:
        for rng_km in (float(x) for x in args.ranges_km.split(",")):
            for course in (float(x) for x in args.courses_deg.split(",")):
                cell = SimpleNamespace(launch_speed_kmh=args.launch_speed_kmh, launch_altitude_m=args.launch_altitude_m,
                                       target_speed_kmh=args.target_speed_kmh, target_altitude_m=args.launch_altitude_m,
                                       target_course_deg=course, target_turn_g=0., max_time_s=90.,
                                       observation_mode="sensor_track", loft=True, azimuth_deg=0.)
                scenario = ew._scenario(cell, rng_km*1000)
                base = pool.submit(ew._run, (scenario, None, 2.)).result()
                case = dict(range_km=rng_km, course_deg=course, unevaded_hit=not base["escaped"],
                            time_of_flight_s=base["flight_time_s"], steps=[])
                if base["escaped"]:
                    cases.append(case)
                    continue
                if args.fast:
                    level_s, _ = _latest(pool, scenario, level, base["flight_time_s"], args.start_step_s)
                    cache = {}
                    library_s, plan = _latest(pool, scenario, library, base["flight_time_s"], args.start_step_s, cache)
                    case.update(latest_level_s=level_s, latest_library_s=library_s, latest_any_s=library_s,
                                best_plan=plan)
                    if args.min_altitude and library_s is not None:
                        for key, start in (("min_loss_latest", library_s), ("min_loss_2s_earlier", library_s-2.)):
                            if start < 0:
                                continue
                            good = _escaping(pool, scenario, library, start, cache)
                            if good:
                                cheapest = min(good, key=lambda p: good[p][1])
                                nose_on = lambda p: (good[p][2] or {}).get("nose_on_s", math.inf)  # noqa: E731
                                fastest = min(good, key=nose_on)
                                case[key] = dict(start_s=start,
                                                 least_loss=dict(plan=_pilot(start, cheapest).describe(),
                                                                 altitude_lost_m=round(good[cheapest][1]),
                                                                 recommit=good[cheapest][2]),
                                                 fastest_back=dict(plan=_pilot(start, fastest).describe(),
                                                                   altitude_lost_m=round(good[fastest][1]),
                                                                   recommit=good[fastest][2]))
                    print(f"{rng_km:g} km course {course:g}: level {level_s} library {library_s} best {plan}", flush=True)
                    cases.append(case)
                    continue
                misses = 0
                k = 0
                while k*args.start_step_s < base["flight_time_s"] and misses < 2:
                    start = k*args.start_step_s
                    results = list(pool.map(_evaluate, [(scenario, _pilot(start, p)) for p in library]))
                    escaping = [(p, r) for p, r in zip(library, results) if r[0]]
                    full_level = [p for p in level if len(p) == 3 or p[3] is None]
                    step = dict(start_s=start, level=any(r[0] for p, r in zip(library, results) if p in full_level),
                                library=[dict(plan=_pilot(start, p).describe(), miss_m=round(r[2]))
                                         for p, r in sorted(escaping, key=lambda pr: -pr[1][1])[:5]],
                                optimized=None, evaluations=len(library))
                    if not escaping:
                        seed = max(zip(library, results), key=lambda pr: pr[1][1])[0]
                        found, params, score, n = _cem(pool, scenario, start, seed, rng, args.population,
                                                       args.elites, args.iterations, bounds)
                        step["evaluations"] += n
                        if found:
                            step["optimized"] = dict(plan=_pilot(start, params).describe(), score=round(score, 2))
                    step["escape"] = bool(escaping) or step["optimized"] is not None
                    misses = 0 if step["escape"] else misses+1
                    case["steps"].append(step)
                    print(f"{rng_km:g} km course {course:g}: start {start:g}s level={step['level']} "
                          f"library={len(escaping)} optimized={'yes' if step['optimized'] else 'no'}"
                          + (f" best={step['library'][0]['plan']}" if step["library"] else
                             f" opt={step['optimized']['plan']}" if step["optimized"] else ""), flush=True)
                    k += 1
                def latest(key):
                    good = [s["start_s"] for s in case["steps"] if s[key]]
                    return max(good) if good else None
                case.update(latest_level_s=latest("level"),
                            latest_library_s=max([s["start_s"] for s in case["steps"] if s["library"]], default=None),
                            latest_any_s=latest("escape"))
                cases.append(case)
    meta = dict(missile=args.missile, evader=args.aircraft, launch_altitude_m=args.launch_altitude_m,
                launch_speed_kmh=args.launch_speed_kmh, target_speed_kmh=args.target_speed_kmh,
                chaff_rcs_ratio=args.chaff_rcs_ratio, start_step_s=args.start_step_s, library=library,
                speed_targets=args.speed_targets, clutter=args.clutter, fast=args.fast,
                clutter_depression_deg=args.clutter_depression_deg, cw_on_clear_beam=args.cw,
                cem=dict(population=args.population, elites=args.elites, iterations=args.iterations, seed=args.seed),
                elapsed_s=round(time.perf_counter()-began, 1))
    out = args.out or ew.ROOT/"outputs"/"evasion_opt"/(
        f"{args.missile}__{int(args.launch_altitude_m)}m_{int(args.launch_speed_kmh)}kmh__chaff{args.chaff_rcs_ratio:g}"
        f"{'__speed' if speeds else ''}{'__fast' if args.fast else ''}{'__minalt' if args.min_altitude else ''}__{args.clutter}{args.clutter_depression_deg:g}{'_cw' if args.cw else ''}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(meta=meta, cases=cases), ensure_ascii=False, indent=1))
    print(f"\n{out}  ({meta['elapsed_s']} s)")
    print("range  course  TOF   latest escaping start (s): level / library / with optimizer")
    for c in cases:
        if not c["unevaded_hit"]:
            print(f"{c['range_km']:4g}  {c['course_deg']:5g}   misses without evading")
            continue
        fmt = lambda v: "  -  " if v is None else f"{v:5.1f}"  # noqa: E731
        def pick(x):
            r = x["recommit"]
            back = "never back" if r is None else (f"defeat {r['defeat_s']:4.1f}s +{r['back_s']:4.1f}s back "
                                                   f"@{r['speed_kmh']}kmh {r['altitude_m']}m")
            return f"{x['plan']:12s} -{x['altitude_lost_m']:4d}m {back}"
        loss = "".join(f"\n      {k[9:]:>11} @{c[k]['start_s']:g}s  least-loss {pick(c[k]['least_loss'])}"
                       f"\n      {'':>11}         fastest-back {pick(c[k]['fastest_back'])}"
                       for k in ("min_loss_latest", "min_loss_2s_earlier") if k in c)
        print(f"{c['range_km']:4g}  {c['course_deg']:5g}  {c['time_of_flight_s']:4.1f}  "
              f"{fmt(c['latest_level_s'])} / {fmt(c['latest_library_s'])} / {fmt(c['latest_any_s'])}{loss}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
