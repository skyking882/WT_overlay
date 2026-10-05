#!/usr/bin/env python3
"""Reference reaction times on random engagements, for choosing and testing surrogates.

Engagements are drawn like gen_escape_samples.py (same ranges, aircraft from
--aircraft with a random load) but from their own seed. For each one that is hit
without evasion, evasion starts are scanned on a --step-s grid from 0 and every
plan of the fine library is tried at each start (in a fixed order, stopping at
the first that escapes). The scan ends once the hit gap exceeds --max-gap-s.

reaction_s is then the end of the escaping run that starts at 0, bridging hit
gaps up to --max-gap-s (build_reaction_table._reaction): the time within which
starting an evasion at any moment still has an escaping plan. It is exact for
this simulator, plan set and grid; isolated escape windows after a longer gap
are deliberately not part of it. Each row records every scanned start with its
first escaping plan, and a split (select / calibrate / test, by index) so model
choice, threshold calibration and the final report use disjoint engagements.

    pypy3 scripts/gen_reference_set.py --missile cn_pl12 --aircraft all --indices 0:3000 \\
        --out reference/cn_pl12.jsonl
"""
from __future__ import annotations

import argparse
from collections import deque
import json
import os
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import escape_window as ew  # noqa: E402
import gen_escape_samples as gen  # noqa: E402
from build_reaction_table import _reaction  # noqa: E402
from optimize_evasion import FULL_SPEED_KMH, LIBRARY_SPEED, _pilot  # noqa: E402

from wt_overlay.fm import load_aircraft  # noqa: E402
from wt_overlay.fm.catalog import aircraft_catalog  # noqa: E402

FIRST = [(90., 0., 0., FULL_SPEED_KMH), (0., 0., 0., FULL_SPEED_KMH)]  # Level beam and drag escape most often.
PLANS = FIRST+[p for p in LIBRARY_SPEED if p not in FIRST]
SPLITS = ("select", "calibrate", "test")


def reference(job):
    seed, index, aircraft, step, max_gap_s = job
    rng = random.Random(f"ref{seed}:{index}")
    s = gen.draw(rng, aircraft)
    scenario = gen._scenario(s)
    began = time.perf_counter()
    s["mass_kg"] = load_aircraft(s["aircraft"]).empty_mass_kg*s["mass_factor"]
    model = gen._model(s["aircraft"], s["mass_kg"])
    unevaded = gen._simulate(scenario, None)
    base = unevaded["summary"]
    t_active, burn_s = gen.detection_times(unevaded)
    out = dict(index=index, seed=seed, split=SPLITS[index % 3], t_active=t_active, burn_s=burn_s,
               **{k: round(v, 3) if isinstance(v, float) else v for k, v in s.items()},
               evader=gen.descriptors(model, s["target_altitude_m"], s["target_speed_kmh"]/3.6),
               unevaded_hit=base["termination_event"] in ("proximity_fuse", "hit"),
               time_of_flight_s=round(base["flight_time_s"], 2), starts=[], runs=1)
    if out["unevaded_hit"]:
        rows, gap, k = [], 0., 0
        while k*step <= base["flight_time_s"]+1e-9:
            start = round(k*step, 6)
            plan = None
            for p in PLANS:
                out["runs"] += 1
                if gen._evade(scenario, _pilot(start, p), s["chaff_rcs_ratio"], model)["escaped"]:
                    plan = p
                    break
            rows.append(dict(start_s=start, escaped=plan is not None))
            out["starts"].append([start, list(plan) if plan else None])
            if start == 0. and plan is None:
                break
            gap = 0. if plan else gap+step
            if gap > max_gap_s+1e-9:
                break
            k += 1
        reaction, reliability = _reaction(rows, step, max_gap_s)
        out.update(reaction_s=reaction, reliability=round(reliability, 3),
                   plan_at_reaction=next((p for t, p in out["starts"] if abs(t-reaction) < 1e-9), None))
    out["cpu_s"] = round(time.perf_counter()-began, 1)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--missile", required=True)
    parser.add_argument("--aircraft", default="all", help="evader FM catalog ids, comma separated, or 'all'")
    parser.add_argument("--missile-sim", type=Path, default=ew.ROOT.parent/"missle_sim")
    parser.add_argument("--indices", required=True, help="first:end engagement indices (end exclusive)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--step-s", type=float, default=.25)
    parser.add_argument("--max-gap-s", type=float, default=.5)
    parser.add_argument("--clutter", choices=("look_down_angle", "geometric_mainlobe", "look_down"),
                        default="look_down_angle")
    parser.add_argument("--clutter-depression-deg", type=float, default=2.)
    parser.add_argument("--no-cw", dest="cw", action="store_false")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--out", type=Path, required=True)
    gen.add_datalink_args(parser)
    args = parser.parse_args(argv)
    first, end = (int(x) for x in args.indices.split(":"))
    pool_ids = tuple(a.id for a in aircraft_catalog()) if args.aircraft == "all" else tuple(args.aircraft.split(","))
    done = gen.done_indices(args.out)
    todo = deque(i for i in range(first, end) if i not in done)
    depression = args.clutter_depression_deg if args.clutter == "look_down_angle" else None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(missile=args.missile, evader_aircraft=pool_ids, seed=args.seed, indices=args.indices,
                step_s=args.step_s, max_gap_s=args.max_gap_s, plans=PLANS, splits=SPLITS, clutter=args.clutter,
                clutter_min_depression_deg=depression, cw_on_clear_beam=args.cw, perception="rwr",
                chaff_program="from evasion start, 1 bundle/s, 30 total", space=gen.SPACE,
                mass_factor=gen.MASS_FACTOR, reaction="end of the escaping run from 0, gaps <= max_gap_s bridged",
                launcher_radar_gimbal_deg=gen.datalink_init(args)[0], require_seeker_lock=args.require_lock)
    Path(str(args.out)+".meta.json").write_text(json.dumps(meta, indent=1))
    init = (str(args.missile_sim), args.missile, pool_ids[0], 10000., True, None, "rwr", (0., .1, 1.),
            args.clutter, depression, args.cw, *gen.datalink_init(args))
    gen.run_resumable(args.out, todo, len(done), args.workers, init,
                      lambda i: (args.seed, i, pool_ids, args.step_s, args.max_gap_s), reference, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
