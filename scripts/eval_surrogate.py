#!/usr/bin/env python3
"""Score trained surrogates on the reference set (gen_reference_set.py).

For every reference engagement of the chosen split, each model predicts
whether the target is hit without evasion, the reaction time (same definition
and 72-plan library as the reference) and the plan it would recommend at that
time. Reports, overall and by seen / unseen aircraft and chaff / no chaff:

    hit/miss flips, reaction MAE and bias, share over-estimated by > 1 s / > 2 s

and writes the recommendations to <model dir>/recommendations_<split>.json so
verify_recommendations.py can fly them in the simulator (does the recommended
plan, started at the predicted time, actually escape?).

    outputs/.mlenv/bin/python scripts/eval_surrogate.py outputs/surrogate_runs/* --split select
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_surrogate as ts  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def load_reference(paths, split):
    rows = {}
    for path in paths:
        text = gzip.decompress(path.read_bytes()).decode() if path.suffix == ".gz" else path.read_text()
        for line in text.splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if "error" not in row and (split == "all" or row["split"] == split):
                rows.setdefault(row["index"], row)
    return [rows[i] for i in sorted(rows)]


def load_model(model_dir, member=None):
    """Surrogate of a run directory; member k picks one network of an ensemble."""
    metrics = json.loads((model_dir/"metrics.json").read_text())
    if (model_dir/"gbdt.pkl").exists():
        import pickle
        trees = pickle.loads((model_dir/"gbdt.pkl").read_bytes())
        return ts.Surrogate(trees["launch"], trees["escape"]), metrics
    ck = torch.load(model_dir/"surrogate.pt", weights_only=False)

    def build(states, n_in, n_out):
        states = states if isinstance(states, list) else [states]
        if member is not None:
            states = [states[member]]
        nets = []
        for sd in states:
            net = ts.make_model(ck["arch"], np.zeros((2, n_in), np.float32), n_out, ck["width"], ck["depth"])
            net.load_state_dict(sd)
            nets.append(net.eval())
        return nets[0] if len(nets) == 1 else ts.Ensemble(nets).eval()
    first = ck["launch"][0] if isinstance(ck["launch"], list) else ck["launch"]
    n_launch = next(v.shape[0] for k, v in reversed(list(first.items())) if k.endswith("weight"))
    launch = build(ck["launch"], ck["n_state"], n_launch)
    escape = build(ck["escape"], ck["n_escape"], len(ck["heads"]))
    return ts.Surrogate(launch.eval(), escape.eval()), metrics


STEP, MAX_GAP_S, ROBUST_S = .25, .5, 1.


def reaction(escaped):
    """End of the escaping run from start 0 on the STEP grid, bridging gaps up to MAX_GAP_S."""
    if not escaped[0]:
        return 0.
    best, gap = 0., 0.
    for k, ok in enumerate(escaped):
        if ok:
            best, gap = k*STEP, 0.
        else:
            gap += STEP
            if gap > MAX_GAP_S+1e-9:
                break
    return best


def summarize(records):
    flips = sum(1 for r in records if (r["pred"] is None) != (r["ref"] is None))
    err = np.array([r["pred"]-r["ref"] for r in records if r["pred"] is not None and r["ref"] is not None])
    if not len(err):
        return dict(n=len(records), flips=flips)
    return dict(n=len(records), flips=flips, mae_s=round(float(np.abs(err).mean()), 2),
                bias_s=round(float(err.mean()), 2), over_1s=round(float(np.mean(err > 1)), 3),
                over_2s=round(float(np.mean(err > 2)), 3), within_1s=round(float(np.mean(np.abs(err) <= 1)), 3))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("models", nargs="+", type=Path, help="train_surrogate.py output directories")
    parser.add_argument("--reference", nargs="+", type=Path,
                        default=sorted((ROOT/"outputs"/"reference").glob("*.jsonl.gz")))
    parser.add_argument("--split", choices=("select", "calibrate", "test", "all"), default="select")
    parser.add_argument("--threshold", type=float, default=0., help="escape logit above which a start counts")
    parser.add_argument("--launch-threshold", type=float, default=0., help="hit logit above which a shot reaches")
    parser.add_argument("--member", type=int, help="score only this member of an ensemble")
    args = parser.parse_args(argv)
    ref = load_reference(args.reference, args.split)
    print(f"{len(ref)} reference engagements ({args.split})")
    for model_dir in args.models:
        sur, metrics = load_model(model_dir, args.member)
        unseen = set(metrics.get("unseen_aircraft", []))
        began = time.perf_counter()
        records = []
        for r in ref:
            ev = ts.evader_features(r, r["aircraft"], r["mass_kg"])
            if ev is None:
                continue
            starts, logits = sur.scan(ts.state_features(r)+ev, ts.LIBRARY_SPEED, force=True)
            rec = dict(index=r["index"], aircraft=r["aircraft"], unseen=r["aircraft"] in unseen,
                       chaff=r["chaff_rcs_ratio"] > 0, ref=r.get("reaction_s") if r["unevaded_hit"] else None,
                       pred=None, plan=None, launch_logit=round(sur.last_launch_logit, 3),
                       maxlogit=np.round(logits.max(axis=1), 3).tolist())
            if sur.last_launch_logit > args.launch_threshold:
                # plain: best plan at that start; robust: best worst-case over the next ROBUST_S.
                k = int(round(ROBUST_S/STEP))
                window_min = np.stack([logits[i:i+k+1].min(axis=0) for i in range(len(starts))])
                rec.update(pred=reaction(logits.max(axis=1) > args.threshold),
                           plain=logits.argmax(axis=1).tolist(), robust=window_min.argmax(axis=1).tolist())
                rec["plan"] = list(ts.LIBRARY_SPEED[rec["plain"][int(round(rec["pred"]/STEP))]]) if rec["pred"] else None
            records.append(rec)
        per_query_ms = (time.perf_counter()-began)/max(1, len(records))*1000
        groups = {"all": records, "seen aircraft": [x for x in records if not x["unseen"]],
                  "unseen aircraft": [x for x in records if x["unseen"]],
                  "chaff": [x for x in records if x["chaff"]], "no chaff": [x for x in records if not x["chaff"]]}
        report = {name: summarize(g) for name, g in groups.items()}
        print(f"\n{model_dir.name}  ({metrics.get('arch')} {metrics.get('width')}x{metrics.get('depth')}, "
              f"heads {metrics.get('heads')}; {per_query_ms:.1f} ms per engagement)")
        for name, s in report.items():
            if "mae_s" in s:
                print(f"  {name:16s} n {s['n']:5d}  flips {s['flips']:3d}  MAE {s['mae_s']:5.2f}  bias {s['bias_s']:+5.2f}"
                      f"  within 1 s {s['within_1s']:.0%}  over >1 s {s['over_1s']:.1%}  over >2 s {s['over_2s']:.1%}")
        out = dict(split=args.split, model=str(model_dir), per_query_ms=per_query_ms, report=report, records=records)
        tag = "" if args.member is None else f"_m{args.member}"
        (model_dir/f"recommendations_{args.split}{tag}.json").write_text(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
