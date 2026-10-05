#!/usr/bin/env python3
"""Distil a missile's escape-surrogate ensemble into the HUD's small hit-probability network.

Teacher: train_surrogate.py ensemble + pk_model.hit_probability_batch (detection x
human delay x chaff x repertoire, 'normal' and 'top' opponents, TWS and STT).
Student: a small SiLU MLP on wt_overlay.pk.FEATURES with the five wt_overlay.pk.OUTPUTS
as logits, fitted with cross-entropy to the teacher's probabilities on random
engagements drawn like the training data (plus a short-range share). Reports the
student's error against the teacher on held-out engagements and, with --reference
and --sim, against simulator ground truth; writes data/pk_models/<missile>.json,
which the HUD evaluates in pure Python (wt_overlay.pk.PkNet).

    outputs/.mlenv/bin/python scripts/distill_pk.py --missile cn_pl12 --teacher outputs/models_v3/v2_cn_pl12
"""
from __future__ import annotations

import argparse
import json
import math
from multiprocessing import Pool
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import eval_surrogate as es  # noqa: E402
import gen_escape_samples as gen  # noqa: E402
import pk_model  # noqa: E402
import train_surrogate as ts  # noqa: E402

from wt_overlay import pk  # noqa: E402
from wt_overlay.fm.catalog import aircraft_catalog  # noqa: E402

BEHAVIOURS = {"normal": pk_model.Behaviour(), "top": pk_model.Behaviour(p_react=1., p_correct=1.)}
SHORT_SHARE, SHORT_KM = .3, (2., 12.)


def _engagement(job):
    seed, index, aircraft = job
    rng = random.Random(f"distill{seed}:{index}")
    short = rng.random() < SHORT_SHARE
    s = gen.draw(rng, aircraft, SHORT_KM if short else None)
    fm_mass = gen.load_aircraft(s["aircraft"]).empty_mass_kg*s["mass_factor"]
    model = gen._model(s["aircraft"], fm_mass)
    ev = pk.descriptors(model, s["target_altitude_m"], s["target_speed_kmh"]/3.6)
    if ev is None:
        return None
    s["mass_kg"], s["chaff_rcs_ratio"] = fm_mass, 1.  # The behaviour sets the defender's chaff itself.
    teacher = ts.state_features(s)+[ev[k] for k in ts.EVADER_KEYS]+[fm_mass]
    student = pk.features(s["launch_altitude_m"], s["launch_speed_kmh"], s["target_altitude_m"], s["target_speed_kmh"],
                          s["course_deg"], s["azimuth_deg"], s["turn_g"], s["range_m"], ev, fm_mass)
    return teacher, student


def _init():
    gen.MODEL_CACHE = 200  # One process sees every aircraft; ~15 MB each.


def labels(sur, teacher_x, chunk=2048):
    out = pk_model.hit_probability_batch(sur, teacher_x, modes=("tws", "stt"), behaviours=BEHAVIOURS, chunk=chunk)
    return np.stack([out["p_reach"]]+[out[(b, m)] for b, m in (("normal", "tws"), ("normal", "stt"),
                                                                 ("top", "tws"), ("top", "stt"))], axis=1)


def student_model(n_in, width, depth, x):
    mean, std = x.mean(0), x.std(0).clip(1e-6)
    layers, n = [], n_in
    for _ in range(depth):
        layers += [nn.Linear(n, width), nn.SiLU()]
        n = width
    return nn.Sequential(*layers, nn.Linear(n, len(pk.OUTPUTS))), mean, std


def export(net, mean, std, missile, meta, path):
    linears = [m for m in net if isinstance(m, nn.Linear)]
    data = dict(missile=missile, features=list(pk.FEATURES), outputs=list(pk.OUTPUTS), activation="silu",
                mean=[float(v) for v in mean], std=[float(v) for v in std],
                layers=[dict(w=[[round(float(v), 7) for v in row] for row in m.weight.detach().cpu().numpy()],
                             b=[round(float(v), 7) for v in m.bias.detach().cpu().numpy()]) for m in linears],
                meta=meta)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, separators=(",", ":")))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--missile", required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--engagements", type=int, default=300000)
    parser.add_argument("--engagements-file", type=Path,
                        help="draw_engagements.py output (much faster under PyPy) instead of drawing here")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--labels-cache", type=Path, help="save / reuse the teacher's labels (npz) for these engagements")
    parser.add_argument("--reference", nargs="*", type=Path, default=[])
    parser.add_argument("--sim", type=Path, help="sim_pk.py output for --reference (simulator ground truth)")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    began = time.perf_counter()
    if args.engagements_file:
        rows = [json.loads(l) for l in args.engagements_file.read_text().splitlines() if l.strip()]
    else:
        pool_ids = tuple(a.id for a in aircraft_catalog())
        with Pool(args.workers, initializer=_init) as pool:
            rows = [r for r in pool.map(_engagement, [(args.seed, i, pool_ids) for i in range(args.engagements)],
                                        chunksize=256) if r is not None]
    tx = np.array([r[0] for r in rows], dtype=np.float32)
    sx = np.array([r[1] for r in rows], dtype=np.float32)
    print(f"{len(rows)} engagements in {time.perf_counter()-began:.0f} s", flush=True)
    cache = args.labels_cache
    if cache is not None and cache.exists():
        y = np.load(cache)["y"]
        if len(y) != len(tx):
            raise SystemExit(f"{cache}: {len(y)} labels for {len(tx)} engagements")
        print(f"teacher labels from {cache}; mean " +
              ", ".join(f"{n} {v:.3f}" for n, v in zip(pk.OUTPUTS, y.mean(0))), flush=True)
    else:
        sur, _ = es.load_model(args.teacher)
        t = time.perf_counter()
        y = labels(sur, tx).astype(np.float32)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, y=y)
        print(f"teacher labels in {time.perf_counter()-t:.0f} s; mean " +
          ", ".join(f"{n} {v:.3f}" for n, v in zip(pk.OUTPUTS, y.mean(0))), flush=True)

    held = np.arange(len(sx)) % 10 == 0
    torch.manual_seed(0)
    net, mean, std = student_model(sx.shape[1], args.width, args.depth, sx[~held])
    xs = torch.tensor((sx-mean)/std, device=args.device)
    ys = torch.tensor(y, device=args.device)
    tr, te = torch.tensor(np.where(~held)[0], device=args.device), torch.tensor(np.where(held)[0], device=args.device)
    net.to(args.device)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-5)
    batch = 4096
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=args.epochs*math.ceil(len(tr)/batch))
    for epoch in range(args.epochs):
        net.train()
        perm = tr[torch.randperm(len(tr), device=args.device)]
        for k in range(0, len(perm), batch):
            b = perm[k:k+batch]
            loss = nn.functional.binary_cross_entropy_with_logits(net(xs[b]), ys[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        if epoch % 10 == 9 or epoch == args.epochs-1:
            net.eval()
            with torch.no_grad():
                err = (torch.sigmoid(net(xs[te]))-ys[te]).abs()
            print(f"  epoch {epoch+1}/{args.epochs}  held-out |student-teacher| mean " +
                  ", ".join(f"{n} {v:.4f}" for n, v in zip(pk.OUTPUTS, err.mean(0).tolist())), flush=True)
    net = net.cpu().eval()
    with torch.no_grad():
        pred = torch.sigmoid(net(torch.tensor((sx[held]-mean)/std))).numpy()
    err = np.abs(pred-y[held])
    report = {n: dict(mae=float(err[:, i].mean()), p95=float(np.quantile(err[:, i], .95)),
                      max=float(err[:, i].max())) for i, n in enumerate(pk.OUTPUTS)}
    print("student vs teacher (held-out): " + "; ".join(
        f"{n} MAE {r['mae']:.4f} p95 {r['p95']:.3f} max {r['max']:.3f}" for n, r in report.items()))

    out = args.out or ROOT/"data"/"pk_models"/f"{args.missile}.json"
    meta = dict(teacher=str(args.teacher), engagements=len(rows), width=args.width, depth=args.depth,
                epochs=args.epochs, student_vs_teacher=report, behaviours={k: vars(v) if hasattr(v, "__dict__") else str(v)
                                                                           for k, v in BEHAVIOURS.items()},
                range_m=[pk.MIN_RANGE_M, pk.MAX_RANGE_M], created=time.strftime("%Y-%m-%d %H:%M:%S"))
    export(net, mean, std, args.missile, meta, out)
    check = pk.PkNet(json.loads(out.read_text()))
    worst = max(abs(check(list(map(float, sx[held][i])))[n]-float(pred[i, j]))
                for i in range(200) for j, n in enumerate(pk.OUTPUTS))
    print(f"exported {out} ({out.stat().st_size/1024:.0f} KB); pure-Python vs torch max diff {worst:.2e}")
    t = time.perf_counter()
    for i in range(500):
        check(list(map(float, sx[i])))
    print(f"pure-Python evaluation: {(time.perf_counter()-t)/500*1000:.2f} ms per engagement")

    if args.reference and args.sim:
        import pk_compare
        sims = {json.loads(l)["index"]: json.loads(l) for l in args.sim.read_text().splitlines() if l.strip()}
        for skill, b in BEHAVIOURS.items():
            for mode in ("tws", "stt"):
                truth, got = [], []
                for r in es.load_reference(args.reference, "test"):
                    ev = ts.evader_features(r, r["aircraft"], r["mass_kg"])
                    if ev is None or (r["unevaded_hit"] and r["index"] not in sims):
                        continue
                    evd = dict(zip(ts.EVADER_KEYS, ev[:4]))
                    x = pk.features(r["launch_altitude_m"], r["launch_speed_kmh"], r["target_altitude_m"],
                                    r["target_speed_kmh"], r["course_deg"], r["azimuth_deg"], r["turn_g"], r["range_m"],
                                    evd, r["mass_kg"])
                    truth.append(pk_compare.sim_pk(sims[r["index"]], mode, b) if r["unevaded_hit"] else 0.)
                    got.append(check(x)[f"{skill}_{mode}"])
                e = np.array(got)-np.array(truth)
                print(f"student vs simulator ({skill}, {mode.upper()}, test split n {len(e)}): MAE {np.abs(e).mean():.3f} "
                      f"bias {e.mean():+.3f} p95 {np.quantile(np.abs(e), .95):.3f}; predicted >= 70% but sim < 50%: "
                      f"{int(np.sum((np.array(got) >= .7) & (np.array(truth) < .5)))}")
    print(f"done in {time.perf_counter()-began:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
