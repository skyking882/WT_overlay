#!/usr/bin/env python3
"""Hit probability: surrogate (pk_model.py) against simulator ground truth (sim_pk.py).

Both use the same behaviour model and quadrature, so a difference is the surrogate's
error alone. Reports, for TWS and STT shots, per split: mean absolute error, bias,
calibration by predicted-probability bin, and the shooter's costly error: shots
predicted at >= 70 % that the simulator puts below 50 %.

    outputs/.mlenv/bin/python scripts/pk_compare.py outputs/surrogate_runs/pl12_mlp_e80x5_pk --sim outputs/pk/cn_pl12.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_surrogate as es  # noqa: E402
import pk_model  # noqa: E402
import train_surrogate as ts  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def sim_pk(rec, mode, b: pk_model.Behaviour):
    """Hit probability from simulated outcomes with the behaviour's weights."""
    weights = {}  # detection time -> probability (both branches collapse when t_active == 0)
    for p, t in b.detections(mode, rec["t_active"]):
        weights[t] = weights.get(t, 0.)+p
    correct = wrong = 0.
    seen = set()
    for br in rec["branches"]:
        key = (br["t_det"], br["k"], br["ratio"])
        if key in seen:  # t_active == 0: both detection branches flew the same starts
            continue
        seen.add(key)
        w = weights.get(br["t_det"], 0.)/b.delay_points/len(b.chaff_ratios)
        correct += w*any(br["escaped"])
        wrong += w*sum(br["escaped"])/len(br["escaped"])
    return (1-b.p_react)+b.p_react*(b.p_correct*(1-correct)+(1-b.p_correct)*(1-wrong))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", type=Path)
    parser.add_argument("--sim", type=Path, required=True)
    parser.add_argument("--reference", nargs="+", type=Path,
                        default=sorted((ROOT/"outputs"/"reference").glob("*.jsonl.gz")))
    args = parser.parse_args(argv)
    sur, _ = es.load_model(args.model)
    ref = {r["index"]: r for r in es.load_reference(args.reference, "all")}
    sims = {}
    for line in args.sim.read_text().splitlines():
        r = json.loads(line)
        if "error" not in r:
            sims[r["index"]] = r
    b = pk_model.Behaviour()
    report = {}
    for mode in ("tws", "stt"):
        rows = []
        for i, r in ref.items():
            ev = ts.evader_features(r, r["aircraft"], r["mass_kg"])
            if ev is None or (r["unevaded_hit"] and i not in sims):
                continue
            truth = sim_pk(sims[i], mode, b) if r["unevaded_hit"] else 0.
            got = pk_model.hit_probability(sur, ts.state_features(r)+ev, mode, b)["p_hit"]
            rows.append((r["split"], truth, got, r["chaff_rcs_ratio"] > 0))
        report[mode] = {}
        print(f"\n{mode.upper()} shots")
        for split in ("select", "calibrate", "test"):
            sel = [x for x in rows if x[0] == split]
            t, g = np.array([x[1] for x in sel]), np.array([x[2] for x in sel])
            hi = g >= .7
            rep = dict(n=len(sel), mae=float(np.abs(g-t).mean()), bias=float((g-t).mean()),
                       brier_vs_truth=float(((g-t)**2).mean()),
                       high_pred=int(hi.sum()), high_pred_truth_below_50=int((hi & (t < .5)).sum()))
            bins = []
            for lo, up in ((0, .1), (.1, .3), (.3, .5), (.5, .7), (.7, .9), (.9, 1.01)):
                m = (g >= lo) & (g < up)
                if m.any():
                    bins.append(dict(bin=f"{lo:.1f}-{min(up, 1):.1f}", n=int(m.sum()), pred=float(g[m].mean()),
                                     sim=float(t[m].mean())))
            rep["bins"] = bins
            report[mode][split] = rep
            print(f"  {split:9s} n {rep['n']:4d}  MAE {rep['mae']:.3f}  bias {rep['bias']:+.3f}  "
                  f">=70% predicted {rep['high_pred']} of which sim <50%: {rep['high_pred_truth_below_50']}")
            print("     calibration (predicted -> simulated): " + "  ".join(
                f"[{x['bin']}] n{x['n']} {x['pred']:.2f}->{x['sim']:.2f}" for x in bins))
    (args.model/"pk_compare.json").write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
