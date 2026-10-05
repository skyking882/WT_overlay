#!/usr/bin/env python3
"""Pick the offensive HUD's decision thresholds on the calibration split, then report them on another.

Two logits decide what the kill rose draws:
    launch threshold   a cell is reachable when the hit logit is above it
    escape threshold   an evasion start counts as escaping when its best plan's logit is above it

For the shooter the costly errors are a target drawn reachable / red / no-escape
when it is not (a missile spent on a target that defeats it). Lowering the escape
threshold lengthens predicted reaction times (fewer false red and false no-escape,
more missed opportunities); raising the launch threshold draws fewer false reachable
cells. The sweep reports every pair; the chosen pair is the one closest to (0, 0)
that meets the targets on the calibration split, and it is then scored once on
--report-split (default test) with no further tuning.

Inputs come from eval_surrogate.py run on both splits (raw logits are stored).

    outputs/.mlenv/bin/python scripts/calibrate_offense.py outputs/surrogate_runs/pl12_mlp_e80x5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_offense import colour  # noqa: E402
from eval_surrogate import reaction  # noqa: E402


def predict(rec, launch_t, escape_t):
    if rec["launch_logit"] <= launch_t:
        return None
    return reaction(np.array(rec["maxlogit"]) > escape_t)


def score(records, launch_t, escape_t):
    pred = [predict(r, launch_t, escape_t) for r in records]
    ref = [r["ref"] for r in records]
    pc, rc = [colour(p) for p in pred], [colour(x) for x in ref]
    n = len(records)
    pred_reach = [i for i in range(n) if pred[i] is not None]
    pred_red = [i for i in range(n) if pc[i] == "red"]
    pred_ne = [i for i in range(n) if pred[i] == 0.]
    ref_red = [i for i in range(n) if rc[i] == "red"]
    ref_ne = [i for i in range(n) if ref[i] == 0.]
    err = np.array([pred[i]-ref[i] for i in range(n) if pred[i] is not None and ref[i] is not None])
    rate = lambda bad, of: bad/max(1, of)  # noqa: E731
    return dict(launch_t=launch_t, escape_t=escape_t, n=n, agree=sum(p == r for p, r in zip(pc, rc))/n,
                false_reach=rate(sum(1 for i in pred_reach if ref[i] is None), len(pred_reach)),
                false_red=rate(sum(1 for i in pred_red if rc[i] != "red"), len(pred_red)), pred_red=len(pred_red),
                false_ne=rate(sum(1 for i in pred_ne if ref[i] != 0.), len(pred_ne)), pred_ne=len(pred_ne),
                missed_red=rate(sum(1 for i in ref_red if pc[i] != "red"), len(ref_red)),
                missed_ne=rate(sum(1 for i in ref_ne if pred[i] != 0.), len(ref_ne)),
                mae=float(np.abs(err).mean()) if len(err) else None, bias=float(err.mean()) if len(err) else None)


def line(s):
    return (f"  launch>{s['launch_t']:+.2f} escape>{s['escape_t']:+.2f}  colour {s['agree']:.1%}  "
            f"false reach {s['false_reach']:.1%}  false red {s['false_red']:.1%} of {s['pred_red']}  "
            f"false no-escape {s['false_ne']:.1%} of {s['pred_ne']}  missed red {s['missed_red']:.1%}  "
            f"missed no-escape {s['missed_ne']:.1%}  MAE {s['mae']:.2f}  bias {s['bias']:+.2f}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", type=Path)
    parser.add_argument("--calibrate-split", default="calibrate")
    parser.add_argument("--report-split", default="test")
    parser.add_argument("--max-false-red", type=float, default=.02)
    parser.add_argument("--max-false-no-escape", type=float, default=.02)
    parser.add_argument("--max-false-reach", type=float, default=.02)
    args = parser.parse_args(argv)
    cal = json.loads((args.model/f"recommendations_{args.calibrate_split}.json").read_text())["records"]
    launch_grid = [0., .5, 1., 1.5, 2., 3.]
    escape_grid = [round(x, 2) for x in np.arange(-4., 1.01, .25)]
    sweep = [score(cal, lt, et) for lt in launch_grid for et in escape_grid]
    print(f"{args.model.name}: {len(cal)} engagements in '{args.calibrate_split}'\n\nescape threshold sweep (launch > 0):")
    for s in sweep:
        if s["launch_t"] == 0.:
            print(line(s))
    print("\nlaunch threshold sweep (escape > 0):")
    for s in sweep:
        if s["escape_t"] == 0.:
            print(line(s))
    ok = [s for s in sweep if s["false_red"] <= args.max_false_red and s["false_ne"] <= args.max_false_no_escape
          and s["false_reach"] <= args.max_false_reach]
    chosen = min(ok, key=lambda s: (s["launch_t"]**2+s["escape_t"]**2)) if ok else None
    out = dict(targets=dict(false_red=args.max_false_red, false_no_escape=args.max_false_no_escape,
                            false_reach=args.max_false_reach), calibrate=chosen)
    if chosen is None:
        print("\nno threshold pair meets the targets on the calibration split")
    else:
        print(f"\nchosen on '{args.calibrate_split}':\n{line(chosen)}")
        report = json.loads((args.model/f"recommendations_{args.report_split}.json").read_text())["records"]
        base, final = score(report, 0., 0.), score(report, chosen["launch_t"], chosen["escape_t"])
        print(f"\non '{args.report_split}' ({len(report)} engagements), untouched thresholds vs chosen:")
        print(line(base))
        print(line(final))
        out.update(report_split=args.report_split, report_default=base, report_chosen=final)
    (args.model/"offense_thresholds.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
