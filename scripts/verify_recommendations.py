#!/usr/bin/env python3
"""Fly a surrogate's recommendations in the simulator.

Reads <model dir>/recommendations_<split>.json (eval_surrogate.py) and the
reference set it was made from. For every engagement where the surrogate
claimed a reaction time T > 0 with plan P, the evasion P is started at T and
simulated. A hit is a false escape claim. Reports the false-claim rate (overall,
seen / unseen aircraft, chaff / no chaff) and the coverage: the share of
engagements with any reference escape window for which the surrogate claimed one.

    pypy3 scripts/verify_recommendations.py outputs/surrogate_runs/pl12_mlp --missile cn_pl12
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import escape_window as ew  # noqa: E402
import gen_escape_samples as gen  # noqa: E402
from optimize_evasion import _pilot  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def rec_plans():
    from optimize_evasion import LIBRARY_SPEED
    return LIBRARY_SPEED


def fly(job):
    row, start, plan = job
    try:
        scenario = gen._scenario(row)
        model = gen._model(row["aircraft"], row["mass_kg"])
        return gen._evade(scenario, _pilot(start, tuple(plan)), row["chaff_rcs_ratio"], model)["escaped"]
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"[:200]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", type=Path)
    parser.add_argument("--missile", required=True)
    parser.add_argument("--split", default="select")
    parser.add_argument("--reference", nargs="+", type=Path,
                        default=sorted((ROOT/"outputs"/"reference").glob("*.jsonl.gz")))
    parser.add_argument("--missile-sim", type=Path, default=ew.ROOT.parent/"missle_sim")
    parser.add_argument("--deltas", type=float, nargs="+", default=[0., 1., 2., 3.],
                        help="start the evasion this many seconds before the claimed reaction time")
    parser.add_argument("--choosers", nargs="+", choices=("plain", "robust", "beam"), default=["plain", "robust"])
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    gen.add_datalink_args(parser)
    args = parser.parse_args(argv)
    rec = json.loads((args.model/f"recommendations_{args.split}.json").read_text())
    ref = {}
    for path in args.reference:
        for line in gzip.decompress(path.read_bytes()).decode().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            ref.setdefault(row["index"], row)
    plans = rec_plans()
    jobs, keys = [], []
    for r in rec["records"]:
        if not r["pred"]:
            continue
        for delta in args.deltas:
            start = r["pred"]-delta
            if start < -1e-9:
                continue
            k = int(round(start/.25))
            for chooser in args.choosers:
                plan = (90., 0., 0.) if chooser == "beam" else plans[r[chooser][k]]  # beam: fixed level-beam baseline
                jobs.append((ref[r["index"]], start, plan))
                keys.append((r["index"], delta, chooser))
    claims = [r for r in rec["records"] if r["pred"]]
    init = (str(args.missile_sim), args.missile, "f_16c_block_50", 10000., True,
            None, "rwr", (0., .1, 1.), "look_down_angle", 2., True, *gen.datalink_init(args))
    with ProcessPoolExecutor(args.workers, initializer=gen._init, initargs=init) as pool:
        outcome = list(pool.map(fly, jobs, chunksize=4))
    errors = [o for o in outcome if isinstance(o, str)]
    by_index = {r["index"]: r for r in rec["records"]}
    for (index, delta, chooser), o in zip(keys, outcome):
        by_index[index].setdefault("verified", {})[f"{delta:g}/{chooser}"] = o if not isinstance(o, str) else None

    def line(name, sel, mode):
        """sel: every record of the group; coverage counts reference windows the surrogate also claimed."""
        flown = [r for r in sel if r.get("verified", {}).get(mode) is not None]
        false = sum(1 for r in flown if not r["verified"][mode])
        windows = [r for r in sel if r["ref"]]
        return (f"  {name:16s} flown {len(flown):5d}  false {false:4d} ({false/max(1, len(flown)):5.1%})"
                f"  coverage {sum(1 for r in windows if r['pred'])/max(1, len(windows)):.0%}")
    print(f"{args.model.name} ({args.split}): {len(claims)} claims, {len(jobs)} flights, {len(errors)} simulator errors")
    records = rec["records"]
    groups = {"all": records, "seen aircraft": [r for r in records if not r["unseen"]],
              "unseen aircraft": [r for r in records if r["unseen"]],
              "chaff": [r for r in records if r["chaff"]], "no chaff": [r for r in records if not r["chaff"]]}
    summary = {}
    for delta in args.deltas:
        for chooser in args.choosers:
            mode = f"{delta:g}/{chooser}"
            print(f" start = reaction - {delta:g} s, plan {chooser}:")
            for name, sel in groups.items():
                print(line(name, sel, mode))
            flown = [r for r in records if r.get("verified", {}).get(mode) is not None]
            summary[mode] = dict(flown=len(flown), false=sum(1 for r in flown if not r["verified"][mode]))
    rec["verification"] = dict(claims=len(claims), errors=len(errors), modes=summary)
    (args.model/f"recommendations_{args.split}.json").write_text(json.dumps(rec))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
