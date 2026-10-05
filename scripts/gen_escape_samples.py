#!/usr/bin/env python3
"""Random-scenario escape samples for training a reaction-time surrogate.

Each scenario draws a random launcher and target state (launcher altitude and
speed, altitude difference, target speed, course, off-boresight, pre-launch
turn, range, chaff RCS ratio) and an evader aircraft from --aircraft with a
random load (empty mass x MASS_FACTOR; or --mass-kg). The aircraft enters the
samples as FM descriptors at the target's altitude and speed (available load
factor and thrust/drag per weight), so a surrogate can generalise across
types. It runs once without evasion, then, if that
hits, ``--evasions`` times with a random evasion start in [0, time of flight]
and a random plan from the speed library (fine building blocks x speed target). One JSON line per
scenario. Escaped runs also carry the recommit leg (closest approach to nose
back within 30 deg of the launcher, see FMEvader.recommit).

Scenario i always uses Random(f"{seed}:{i}"), so shards are disjoint index
ranges and a restart skips indices already in --out:

    pypy3 scripts/gen_escape_samples.py --missile cn_pl12 --aircraft all --indices 0:20000 \\
        --out samples/shard0.jsonl
"""
from __future__ import annotations

import argparse
from collections import deque
import copy
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
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
from optimize_evasion import LIBRARY_SPEED, _pilot  # noqa: E402

from wt_overlay.contracts import G  # noqa: E402
from wt_overlay.escape import fm_evader_factory  # noqa: E402
from wt_overlay.fm import load_aircraft  # noqa: E402
from wt_overlay.fm.catalog import aircraft_catalog  # noqa: E402
from wt_overlay.pk import descriptors  # noqa: E402,F401  (re-exported for the other scripts)
from wt_overlay.turn import ManeuverModel  # noqa: E402

# Sampling ranges; chaff ratio is log-uniform, with no chaff P_NO_CHAFF of the time.
SPACE = dict(launch_altitude_m=(1000., 13000.), launch_speed_kmh=(700., 1450.), alt_diff_m=(-5000., 5000.),
             target_speed_kmh=(700., 1400.), course_deg=(0., 180.), azimuth_deg=(0., 60.),
             turn_g=(-6., 6.), range_km=(2., 45.), chaff_rcs_ratio=(.25, 4.))
TARGET_ALTITUDE_M = (500., 14000.)
P_LEVEL = .5     # Share of targets flying straight before launch.
P_NO_CHAFF = .3
MAX_TIME_S = 150.
MASS_FACTOR = (1.15, 1.45)  # Total / empty mass: fuel plus missiles.


def draw(rng, aircraft=("f_16c_block_50",), range_km=None):
    """One random engagement; range_km=(lo, hi) narrows the launch range (e.g. a short-range top-up)."""
    space = SPACE if range_km is None else dict(SPACE, range_km=tuple(range_km))
    u = lambda key: rng.uniform(*space[key])  # noqa: E731
    s = dict(launch_altitude_m=u("launch_altitude_m"), launch_speed_kmh=u("launch_speed_kmh"),
             alt_diff_m=u("alt_diff_m"), target_speed_kmh=u("target_speed_kmh"), course_deg=u("course_deg"),
             azimuth_deg=u("azimuth_deg"), turn_g=0. if rng.random() < P_LEVEL else u("turn_g"),
             range_m=1000*u("range_km"))
    lo, hi = space["chaff_rcs_ratio"]
    s["chaff_rcs_ratio"] = 0. if rng.random() < P_NO_CHAFF else math.exp(rng.uniform(math.log(lo), math.log(hi)))
    s["target_altitude_m"] = min(TARGET_ALTITUDE_M[1], max(TARGET_ALTITUDE_M[0], s["launch_altitude_m"]+s["alt_diff_m"]))
    s["aircraft"] = rng.choice(aircraft)
    s["mass_factor"] = rng.uniform(*MASS_FACTOR)
    return s


MODEL_CACHE = 8  # Each aircraft's tables take ~15 MB and rebuild in ~0.1 s; 138 cached would be ~2 GB per worker.


def _model(aircraft, mass):
    """ManeuverModel for this aircraft at this mass (small LRU of built models per worker)."""
    cache = ew._CTX.setdefault("models", {})
    if aircraft in cache:
        cache[aircraft] = cache.pop(aircraft)  # Most recently used last.
    else:
        while len(cache) >= MODEL_CACHE:
            cache.pop(next(iter(cache)))
        fm = load_aircraft(aircraft)
        cache[aircraft] = ManeuverModel(fm, fm.empty_mass_kg*1.3, True)
    model = copy.copy(cache[aircraft])
    model.mass, model._aoa_cache = mass, {}
    return model


def _scenario(s):
    cell = SimpleNamespace(launch_speed_kmh=s["launch_speed_kmh"], launch_altitude_m=s["launch_altitude_m"],
                           target_speed_kmh=s["target_speed_kmh"], target_altitude_m=s["target_altitude_m"],
                           target_course_deg=s["course_deg"], target_turn_g=s["turn_g"], max_time_s=MAX_TIME_S,
                           observation_mode="sensor_track", loft=True, azimuth_deg=s["azimuth_deg"])
    return ew._scenario(cell, s["range_m"])


def _init(*args):
    ew._init(*args)
    from aim120_model.chaff import ChaffProgram, ChaffSpec, chaffing_factory
    ew._CTX.update(ChaffProgram=ChaffProgram, ChaffSpec=ChaffSpec, chaffing_factory=chaffing_factory)


def _simulate(scenario, factory):
    c = ew._CTX
    return c["simulate"](c["profile"], scenario, target_factory=factory, early_miss_s=2.,
                         clutter_model=c["clutter"], **c["clutter_kwargs"])


def detection_times(result):
    """(t_active, burn_s) of an unevaded run: when the missile's seeker comes within its
    active range of the target (the RWR warning; None if it never does) and when its motor
    burns out (the end of the visible-marker window), per the 'rwr' perception model."""
    p = ew._CTX.get("perception")
    if p is None:
        return None, None
    t_active = next((float(x["time_s"]) for x in result["samples"] if x["distance_to_target_m"] <= p.active_range_m),
                    None)
    return (None if t_active is None else round(t_active, 3)), round(p.burn_s, 3)


def _escaped(summary, evader):
    return summary["termination_event"] not in ("proximity_fuse", "hit") and not evader.get("fault")


def _miss_exact(summary, samples):
    """True when miss_m is the true closest approach: a fuse event, or the range was opening
    again when the run stopped. An early-miss stop while still closing leaves only an upper bound."""
    if summary["termination_event"] in ("proximity_fuse", "hit"):
        return True
    distances = [x["distance_to_target_m"] for x in samples]
    return distances[-1] > min(distances)+1.


def _evade(scenario, pilot, ratio, model):
    c = ew._CTX
    evaders = []
    fm = fm_evader_factory(pilot, model, c["state"], c.get("perception"))
    factory = lambda base: evaders.append(fm(base)) or evaders[-1]  # noqa: E731
    if ratio > 0:
        factory = c["chaffing_factory"](factory, c["ChaffSpec"](rcs_ratio=ratio),
                                        c["ChaffProgram"](pilot.start_s, interval_s=1., per_salvo=1, total=30))
    result = _simulate(scenario, factory)
    summary, evader = result["summary"], result["model"].get("target_model") or {}
    escaped = _escaped(summary, evader)
    row = dict(start_s=round(pilot.start_s, 3), escaped=escaped, event=summary["termination_event"],
               miss_m=round(summary["minimum_distance_m"], 1), flight_s=round(summary["flight_time_s"], 2),
               altitude_lost_m=round(scenario["target_altitude_m"]-evader.get("min_altitude_m",
                                                                              scenario["target_altitude_m"])),
               min_speed_kmh=round(evader.get("min_speed_mps", 0.)*3.6), fault=evader.get("fault"),
               cw_s=round(summary.get("cw_mode_time_s") or 0., 2),
               decoy_s=round(summary.get("decoy_track_time_s") or 0., 2), recommit=None,
               miss_exact=_miss_exact(summary, result["samples"]))
    if escaped and evaders:
        closest = min(result["samples"], key=lambda x: x["distance_to_target_m"])
        back = evaders[-1].recommit(float(closest["time_s"]))
        if back is not None:
            row["recommit"] = dict(defeat_s=round(float(closest["time_s"]), 2), back_s=round(back["seconds"], 2),
                                   speed_kmh=round(back["speed_mps"]*3.6), altitude_m=round(back["altitude_m"]))
    return row


def run_scenario(job):
    """One scenario row; a scenario the simulator rejects (e.g. a launch state outside the
    missile's thrust table) comes back as {index, error} so it is skipped, not retried."""
    try:
        return _run_scenario(job)
    except Exception as exc:  # noqa: BLE001 - must not reach the pool: some errors do not unpickle.
        return dict(index=job[1], seed=job[0], error=f"{type(exc).__name__}: {exc}"[:300])


def _run_scenario(job):
    seed, index, evasions, aircraft, mass_kg = job[:5]
    range_km = job[5] if len(job) > 5 else None
    rng = random.Random(f"{seed}:{index}")
    s = draw(rng, aircraft, range_km)
    scenario = _scenario(s)
    began = time.perf_counter()
    fm = load_aircraft(s["aircraft"]) if mass_kg is None else None
    s["mass_kg"] = mass_kg if mass_kg is not None else fm.empty_mass_kg*s["mass_factor"]
    model = _model(s["aircraft"], s["mass_kg"])
    unevaded = _simulate(scenario, None)
    base = unevaded["summary"]
    t_active, burn_s = detection_times(unevaded)
    out = dict(index=index, seed=seed, **{k: round(v, 3) if isinstance(v, float) else v for k, v in s.items()},
               evader=descriptors(model, s["target_altitude_m"], s["target_speed_kmh"]/3.6),
               t_active=t_active, burn_s=burn_s,
               unevaded_hit=base["termination_event"] in ("proximity_fuse", "hit"),
               unevaded_miss_m=round(base["minimum_distance_m"], 1), time_of_flight_s=round(base["flight_time_s"], 2),
               evasions=[])
    if out["unevaded_hit"]:
        for _ in range(evasions):
            plan = rng.choice(LIBRARY_SPEED)
            start = rng.uniform(0., base["flight_time_s"])
            row = _evade(scenario, _pilot(start, plan), s["chaff_rcs_ratio"], model)
            row["plan"] = list(plan)
            out["evasions"].append(row)
    out["cpu_s"] = round(time.perf_counter()-began, 2)
    return out


def add_datalink_args(parser):
    """Launcher-radar datalink and seeker-lock rules (missile_sim opt-in, user-reported, D); on by default."""
    parser.add_argument("--launcher-gimbal-deg", type=float, default=60.,
                        help="launcher radar cone; the datalink drops for good outside it or in the launcher's notch "
                             "(0 = legacy truth datalink)")
    parser.add_argument("--no-require-lock", dest="require_lock", action="store_false",
                        help="let a seeker that never locked still fuse (legacy)")


def datalink_init(args):
    return (args.launcher_gimbal_deg or None), args.require_lock


def done_indices(out):
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            try:
                done.add(json.loads(line)["index"])
            except (ValueError, KeyError):
                pass  # A line cut short by a kill; its index is simply rerun.
    return done


def _guarded(worker, job, index, seed):
    """Run in the worker process: any exception becomes an {index, error} row here, because
    some exceptions (e.g. the simulator's input errors) cannot be unpickled by the parent and
    would otherwise break the whole pool."""
    try:
        return worker(job)
    except Exception as exc:  # noqa: BLE001
        return dict(index=index, seed=seed, error=f"{type(exc).__name__}: {exc}"[:300])


def run_resumable(out, todo, n_done, workers, init, job_for, worker, seed):
    """Run worker(job_for(i)) for every index in ``todo`` over a process pool, appending one JSON
    line per index to ``out`` as each finishes. A broken pool restarts with a quarter fewer
    workers; an exception for one index is written as {index, error} and not retried."""
    print(f"{len(todo)} scenarios to run ({n_done} already in {out}), {workers} workers", flush=True)
    began, n = time.perf_counter(), 0
    with out.open("a") as f:
        while todo:
            # Bounded in-flight work, written as it completes. A killed worker (e.g. a
            # memory limit) breaks the pool: restart it with a quarter fewer workers.
            pending = {}
            try:
                with ProcessPoolExecutor(workers, initializer=_init, initargs=init) as pool:
                    while todo or pending:
                        while todo and len(pending) < 4*workers:
                            i = todo.popleft()
                            pending[pool.submit(_guarded, worker, job_for(i), i, seed)] = i
                        finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                        for future in finished:
                            i = pending.pop(future)
                            try:
                                row = future.result()
                            except BrokenProcessPool:
                                pending[future] = i
                                raise
                            except Exception as exc:  # noqa: BLE001
                                row = dict(index=i, seed=seed, error=f"{type(exc).__name__}: {exc}"[:300])
                            f.write(json.dumps(row, separators=(",", ":"))+"\n")
                            f.flush()
                            n += 1
                            if n % 50 == 0:
                                rate = n/(time.perf_counter()-began)
                                print(f"{n} done, {len(todo)+len(pending)} left  {rate:.2f} scenarios/s  "
                                      f"eta {(len(todo)+len(pending))/rate/3600:.1f} h  ({workers} workers)", flush=True)
            except BrokenProcessPool:
                todo = deque(sorted(set(todo) | set(pending.values())))
                workers = max(1, workers*3//4)
                print(f"process pool broke (a worker died); restarting with {workers} workers", flush=True)
    print(f"all done: {n} scenarios in {(time.perf_counter()-began)/3600:.2f} h", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--missile", required=True)
    parser.add_argument("--aircraft", required=True,
                        help="evader FM catalog ids, comma separated, or 'all' for the whole catalog")
    parser.add_argument("--mass-kg", type=float, help="fixed evader mass (default: empty mass x random MASS_FACTOR)")
    parser.add_argument("--missile-sim", type=Path, default=ew.ROOT.parent/"missle_sim")
    parser.add_argument("--indices", required=True, help="first:end scenario indices (end exclusive)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--evasions", type=int, default=8, help="evasion runs per scenario that is hit unevaded")
    parser.add_argument("--clutter", choices=("look_down_angle", "geometric_mainlobe", "look_down"),
                        default="look_down_angle")
    parser.add_argument("--clutter-depression-deg", type=float, default=2.)
    parser.add_argument("--no-cw", dest="cw", action="store_false")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--range-km", help="lo,hi: draw launch ranges only from this band (default the full space)")
    add_datalink_args(parser)
    args = parser.parse_args(argv)
    first, end = (int(x) for x in args.indices.split(":"))
    range_km = tuple(float(x) for x in args.range_km.split(",")) if args.range_km else None
    done = done_indices(args.out)
    todo = deque(i for i in range(first, end) if i not in done)
    pool_ids = tuple(a.id for a in aircraft_catalog()) if args.aircraft == "all" else tuple(args.aircraft.split(","))
    depression = args.clutter_depression_deg if args.clutter == "look_down_angle" else None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(missile=args.missile, evader_aircraft=pool_ids, evader_mass_kg=args.mass_kg, mass_factor=MASS_FACTOR,
                seed=args.seed,
                indices=args.indices, evasions=args.evasions, clutter=args.clutter,
                clutter_min_depression_deg=depression, cw_on_clear_beam=args.cw, perception="rwr",
                chaff_program="from evasion start, 1 bundle/s, 30 total", space=SPACE,
                target_altitude_m=TARGET_ALTITUDE_M, p_level=P_LEVEL, p_no_chaff=P_NO_CHAFF,
                max_time_s=MAX_TIME_S, plans=LIBRARY_SPEED,
                plan_fields="target_deg, plane_deg, dive_deg, speed_kmh (>= 1250: full throttle)",
                launcher_radar_gimbal_deg=datalink_init(args)[0], require_seeker_lock=args.require_lock,
                range_km=range_km or SPACE["range_km"])
    Path(str(args.out)+".meta.json").write_text(json.dumps(meta, indent=1))
    init = (str(args.missile_sim), args.missile, pool_ids[0], args.mass_kg or 10000., True, None, "rwr",
            (0., .1, 1.), args.clutter, depression, args.cw, *datalink_init(args))
    run_resumable(args.out, todo, len(done), args.workers, init,
                  lambda i: (args.seed, i, args.evasions, pool_ids, args.mass_kg, range_km), run_scenario, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
