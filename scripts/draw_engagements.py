#!/usr/bin/env python3
"""Random engagements for distill_pk.py, as feature rows (run under PyPy: the FM work is the slow part).

Each line: [teacher features (train_surrogate.state_features + evader + mass), student features
(wt_overlay.pk.FEATURES)]. Drawn like the training data, with SHORT_SHARE from 2-12 km.

    pypy3 scripts/draw_engagements.py --engagements 300000 --out outputs/pk_distill/engagements.jsonl
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_escape_samples as gen  # noqa: E402

from wt_overlay import pk  # noqa: E402
from wt_overlay.fm.catalog import aircraft_catalog  # noqa: E402

SHORT_SHARE, SHORT_KM = .3, (2., 12.)
EVADER_KEYS = ("load_cap", "thrust_w", "drag_cap_w", "drag_1g_w")
_EMPTY = {}


def state_features(s):
    # Same as train_surrogate.state_features (that module needs torch).
    la, ta, chaff = s["launch_altitude_m"], s["target_altitude_m"], s["chaff_rcs_ratio"]
    course = math.radians(s["course_deg"])
    return [la, s["launch_speed_kmh"], ta-la, ta, s["target_speed_kmh"], math.cos(course), math.sin(course),
            s["azimuth_deg"], s["turn_g"], s["range_m"], math.log(s["range_m"]), 1. if chaff > 0 else 0.,
            math.log2(chaff) if chaff > 0 else 0.]


def engagement(job):
    seed, index, aircraft = job
    gen.MODEL_CACHE = 200
    rng = random.Random(f"distill{seed}:{index}")
    s = gen.draw(rng, aircraft, SHORT_KM if rng.random() < SHORT_SHARE else None)
    if s["aircraft"] not in _EMPTY:
        _EMPTY[s["aircraft"]] = gen.load_aircraft(s["aircraft"]).empty_mass_kg
    mass = _EMPTY[s["aircraft"]]*s["mass_factor"]
    ev = pk.descriptors(gen._model(s["aircraft"], mass), s["target_altitude_m"], s["target_speed_kmh"]/3.6)
    if ev is None:
        return None
    s["chaff_rcs_ratio"] = 1.  # The behaviour model sets the defender's chaff itself.
    teacher = state_features(s)+[ev[k] for k in EVADER_KEYS]+[mass]
    student = pk.features(s["launch_altitude_m"], s["launch_speed_kmh"], s["target_altitude_m"], s["target_speed_kmh"],
                          s["course_deg"], s["azimuth_deg"], s["turn_g"], s["range_m"], ev, mass)
    return [teacher, student]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--engagements", type=int, default=300000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    ids = tuple(a.id for a in aircraft_catalog())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(args.workers) as pool, args.out.open("w") as f:
        for row in pool.map(engagement, [(args.seed, i, ids) for i in range(args.engagements)], chunksize=512):
            if row is not None:
                f.write(json.dumps(row, separators=(",", ":"))+"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
