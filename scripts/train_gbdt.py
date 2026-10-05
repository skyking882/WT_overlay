#!/usr/bin/env python3
"""Gradient-boosted tree baseline for the escape surrogate (scikit-learn histogram GBDT).

Same data, features, held-out scenarios and unseen aircraft as train_surrogate.py;
writes a directory eval_surrogate.py can score like a network.

    outputs/.mlenv/bin/python scripts/train_gbdt.py outputs/samples/cn_pl12__all.jsonl.gz --out outputs/surrogate_runs/pl12_gbdt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_surrogate as ts  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("samples", nargs="+", type=Path)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--leaves", type=int, default=127)
    parser.add_argument("--learning-rate", type=float, default=.1)
    parser.add_argument("--holdout-aircraft", type=float, default=.15)
    parser.add_argument("--legacy-aircraft", default="f_16c_block_50")
    parser.add_argument("--legacy-mass-kg", type=float, default=12000.)
    parser.add_argument("--tables", type=Path, default=ts.ROOT/"data"/"offense")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    began = time.perf_counter()
    rows, skipped = ts.load(args.samples)
    aircraft = lambda r: r.get("aircraft", args.legacy_aircraft)  # noqa: E731
    unseen = [r for r in rows if ts.is_holdout(aircraft(r), args.holdout_aircraft)]
    rows = [r for r in rows if not ts.is_holdout(aircraft(r), args.holdout_aircraft)]
    held = [r for r in rows if r["index"] % 10 == 0]
    rows = [r for r in rows if r["index"] % 10 != 0]
    _, s, hit, tof, ex, ys = ts.arrays(rows, args.legacy_aircraft, args.legacy_mass_kg)
    _, hs, hhit, _, hex_, hys = ts.arrays(held, args.legacy_aircraft, args.legacy_mass_kg)
    _, us, uhit, _, uex, uys = ts.arrays(unseen, args.legacy_aircraft, args.legacy_mass_kg)
    kw = dict(max_iter=args.max_iter, max_leaf_nodes=args.leaves, learning_rate=args.learning_rate,
              early_stopping=True, validation_fraction=.05, n_iter_no_change=30, random_state=0)
    print(f"{len(ex)} evasion runs; fitting", flush=True)
    hit_clf = HistGradientBoostingClassifier(**kw).fit(s, hit)
    tof_reg = HistGradientBoostingRegressor(**kw).fit(s[hit > .5], np.log(np.maximum(tof[hit > .5], .1)))
    esc_clf = HistGradientBoostingClassifier(**kw).fit(ex, ys[:, 0])
    print(f"  iterations: hit {hit_clf.n_iter_}, tof {tof_reg.n_iter_}, escape {esc_clf.n_iter_}", flush=True)
    launch, escape = ts.TreeLaunch(hit_clf, tof_reg), ts.TreeEscape(esc_clf)
    acc = lambda clf, x, y: float((clf.predict(x) == (y > .5)).mean()) if len(x) else None  # noqa: E731
    metrics = dict(arch="gbdt", width=args.leaves, depth=args.max_iter, heads=["escape"],
                   launch_holdout_acc=acc(hit_clf, hs, hhit), escape_holdout_acc=acc(esc_clf, hex_, hys[:, 0]),
                   launch_unseen_acc=acc(hit_clf, us, uhit), escape_unseen_acc=acc(esc_clf, uex, uys[:, 0]),
                   unseen_aircraft=sorted({aircraft(r) for r in unseen}))
    print(f"accuracy held-out: hit {metrics['launch_holdout_acc']:.1%} escape {metrics['escape_holdout_acc']:.1%}; "
          f"unseen aircraft: hit {metrics['launch_unseen_acc']:.1%} escape {metrics['escape_unseen_acc']:.1%}")
    metrics["tables"] = ts.check_tables(ts.Surrogate(launch, escape), args.tables)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/"gbdt.pkl").write_bytes(pickle.dumps(dict(launch=launch, escape=escape)))
    metrics.update(samples=[str(p) for p in args.samples], evasion_runs=len(ex),
                   elapsed_s=round(time.perf_counter()-began, 1))
    (args.out/"metrics.json").write_text(json.dumps(metrics, indent=1))
    print(f"-> {args.out} ({metrics['elapsed_s']} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
