#!/usr/bin/env python3
"""Score surrogates the way the offensive HUD uses them.

1. Kill-rose colours on the reference set (recommendations_<split>.json from
   eval_surrogate.py): unreachable (no hit without evasion), red (< 3 s), amber
   (3-6 s), grey (>= 6 s), with the no-escape (0 s) cells counted separately.
   The costly mistake for the shooter is a cell drawn red / no-escape that is
   not: a missile spent on a target that can still defeat it.
2. Residuals near the 0, 3 and 6 s colour boundaries.
3. B-scope lines (rmax_hot, rmax_cold, r3_hot, rne_hot) recomputed from the
   surrogate with the envelope tables' own procedure (36-plan library, coarse
   range ladder then bisection to 0.25 km) and compared, in km, to data/envelope.

    outputs/.mlenv/bin/python scripts/eval_offense.py outputs/surrogate_runs/pl12_mlp_noaux
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_surrogate as es  # noqa: E402
import train_surrogate as ts  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RED_S, AMBER_S = 3., 6.  # wt_overlay/ui.py ROSE_BANDS
CLASSES = ("unreachable", "red", "amber", "grey")
COARSE_KM = (2, 3, 4, 5, 7, 10, 14, 20, 28, 40, 56, 80, 120)  # build_envelope_table.py
LINES = ("rmax_hot", "rmax_cold", "r3_hot", "rne_hot")


def colour(reaction):
    if reaction is None:
        return "unreachable"
    return "red" if reaction < RED_S else "amber" if reaction < AMBER_S else "grey"


def rose_report(records):
    n = len(records)
    conf = {(a, b): 0 for a in CLASSES for b in CLASSES}
    for r in records:
        conf[(colour(r["ref"]), colour(r["pred"]))] += 1
    agree = sum(conf[(c, c)] for c in CLASSES)/n
    pred_red = [r for r in records if colour(r["pred"]) == "red"]
    ref_red = [r for r in records if colour(r["ref"]) == "red"]
    pred_ne = [r for r in records if r["pred"] == 0.]
    ref_ne = [r for r in records if r["ref"] == 0.]
    out = dict(n=n, agree=agree,
               false_red=sum(1 for r in pred_red if colour(r["ref"]) != "red"), pred_red=len(pred_red),
               missed_red=sum(1 for r in ref_red if colour(r["pred"]) != "red"), ref_red=len(ref_red),
               false_no_escape=sum(1 for r in pred_ne if r["ref"] != 0.), pred_no_escape=len(pred_ne),
               missed_no_escape=sum(1 for r in ref_ne if r["pred"] != 0.), ref_no_escape=len(ref_ne),
               confusion={f"{a}->{b}": v for (a, b), v in conf.items() if v})
    near = {}
    for centre, lo, hi in ((0., 0., 1.), (3., 2., 4.), (6., 5., 7.)):
        err = np.array([r["pred"]-r["ref"] for r in records
                        if r["ref"] is not None and r["pred"] is not None and lo <= r["ref"] <= hi])
        if len(err):
            near[f"{centre:g}s"] = dict(n=len(err), mae=float(np.abs(err).mean()), bias=float(err.mean()),
                                        under_1s=float(np.mean(err < -1)), over_1s=float(np.mean(err > 1)))
    out["near"] = near
    return out


def print_rose(name, rep):
    print(f"  {name:16s} n {rep['n']:5d}  colour agrees {rep['agree']:.1%}  "
          f"false red {rep['false_red']}/{rep['pred_red']} ({rep['false_red']/max(1, rep['pred_red']):.1%})  "
          f"missed red {rep['missed_red']}/{rep['ref_red']}  "
          f"false no-escape {rep['false_no_escape']}/{rep['pred_no_escape']} "
          f"({rep['false_no_escape']/max(1, rep['pred_no_escape']):.1%})  missed no-escape "
          f"{rep['missed_no_escape']}/{rep['ref_no_escape']}")
    for k, v in rep["near"].items():
        print(f"      reference near {k:3s}: n {v['n']:4d}  MAE {v['mae']:.2f}  bias {v['bias']:+.2f}  "
              f"pred >1 s short {v['under_1s']:.0%}  >1 s long {v['over_1s']:.0%}")


def envelope_lines(sur, meta, azimuth, alt_diff, tolerance_m=250.):
    """The four B-scope lines (m or None) from the surrogate, like build_envelope_table._bisect."""
    base = dict(launch_altitude_m=meta["launch_altitude_m"], launch_speed_kmh=meta["launch_speed_kmh"],
                target_altitude_m=max(300., meta["launch_altitude_m"]+alt_diff),
                target_speed_kmh=meta["target_speed_kmh"], azimuth_deg=azimuth, turn_g=0.,
                chaff_rcs_ratio=meta["chaff_rcs_ratio"])
    aircraft, mass = meta["evader_aircraft"], meta["evader_mass_kg"]

    def predicate(line, range_m):
        s = dict(base, course_deg=180. if line == "rmax_cold" else 0., range_m=range_m)
        vec = ts.state_features(s)+ts.evader_features(s, aircraft, mass)
        reaction, _ = sur.reaction(vec, ts.LIBRARY)
        if line in ("rmax_hot", "rmax_cold"):
            return reaction is not None
        if reaction is None:
            return False
        return reaction == 0. if line == "rne_hot" else reaction < RED_S

    out = {}
    for line in LINES:
        lo = hi = None
        for km in COARSE_KM:
            if predicate(line, km*1000):
                lo = km*1000
            elif lo is not None:
                hi = km*1000
                break
        if lo is not None and hi is not None:
            while hi-lo > tolerance_m:
                mid = (lo+hi)/2
                lo, hi = (mid, hi) if predicate(line, mid) else (lo, mid)
        out[line] = lo
    return out


def envelope_report(sur, envelope_dir):
    rows = []
    for path in sorted(Path(envelope_dir).glob("*.json")):
        d = json.loads(path.read_text())
        meta = dict(d["meta"])
        meta.setdefault("evader_aircraft", "f_16c_block_50")
        meta.setdefault("evader_mass_kg", 12000.)
        if meta.get("clutter") != "look_down_angle":
            continue
        for row in d["rows"]:
            got = envelope_lines(sur, meta, row["azimuth_deg"], row["alt_diff_m"])
            rows.append(dict(table=path.stem, azimuth=row["azimuth_deg"], alt_diff=row["alt_diff_m"],
                             ref={k: row[k] for k in LINES}, got=got))
    print(f"\n  B-scope lines vs data/envelope ({len(rows)} azimuth x altitude rows), km  [reference -> surrogate]")
    print("  az   dh    " + "".join(f"{l:>17s}" for l in LINES))
    km = lambda v: " none" if v is None else f"{v/1000:5.1f}"  # noqa: E731
    errs = {l: [] for l in LINES}
    mism = {l: 0 for l in LINES}
    for r in rows:
        print(f"  {r['azimuth']:3g} {r['alt_diff']:6g}  " + "".join(
            f"      {km(r['ref'][l])}->{km(r['got'][l])}" for l in LINES))
        for l in LINES:
            if (r["ref"][l] is None) != (r["got"][l] is None):
                mism[l] += 1
            elif r["ref"][l] is not None:
                errs[l].append((r["got"][l]-r["ref"][l])/1000)
    summary = {}
    for l in LINES:
        e = np.array(errs[l])
        summary[l] = dict(n=len(e), mae_km=float(np.abs(e).mean()) if len(e) else None,
                          bias_km=float(e.mean()) if len(e) else None, none_mismatch=mism[l])
        if len(e):
            print(f"  {l:10s} MAE {np.abs(e).mean():5.2f} km  bias {e.mean():+5.2f} km  max |err| "
                  f"{np.abs(e).max():5.2f} km  present/absent mismatches {mism[l]}")
    return dict(rows=rows, summary=summary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("models", nargs="+", type=Path)
    parser.add_argument("--split", default="select")
    parser.add_argument("--envelope", type=Path, default=ROOT/"data"/"envelope")
    parser.add_argument("--member", type=int, help="score only this member of an ensemble")
    parser.add_argument("--no-envelope", action="store_true", help="skip the data/envelope comparison (tables made with other rules)")
    args = parser.parse_args(argv)
    for model_dir in args.models:
        tag = "" if args.member is None else f"_m{args.member}"
        rec = json.loads((model_dir/f"recommendations_{args.split}{tag}.json").read_text())["records"]
        print(f"\n{model_dir.name}{tag} ({args.split})")
        groups = {"all": rec, "unseen aircraft": [r for r in rec if r["unseen"]],
                  "chaff": [r for r in rec if r["chaff"]], "no chaff": [r for r in rec if not r["chaff"]]}
        report = {}
        for name, sel in groups.items():
            report[name] = rose_report(sel)
            print_rose(name, report[name])
        sur, _ = es.load_model(model_dir, args.member) if not args.no_envelope else (None, None)
        if not args.no_envelope:
            report["envelope"] = envelope_report(sur, args.envelope)
        (model_dir/f"offense_{args.split}{tag}.json").write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
