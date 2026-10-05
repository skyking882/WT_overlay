#!/usr/bin/env python3
"""Train the escape surrogate on gen_escape_samples.py output and check it against the grid tables.

Two networks:
    launch net   state, evader -> P(unevaded hit), log time of flight
    escape net   state, evader, time of flight, evasion start, plan -> escape logit
                 [, log1p miss distance (auxiliary), altitude lost, recommit seconds]

"evader" is the target aircraft as FM descriptors at the target's altitude and
speed (load at the AoA cap, thrust/drag per weight) plus its mass, so types not
seen in training can be queried. Inputs are standardised with training-set
statistics stored in the model.

Reaction time follows build_reaction_table.py: the end of the escaping run that
starts at 0 on a 0.25 s grid, bridging hit gaps up to 0.5 s, where a start
escapes if any plan of the chosen set does. The table check uses the tables' own
36-plan library so the comparison is like for like.

Miss-distance supervision skips evader faults (a ground impact is no escape at
any distance); a miss recorded while the missile was still closing
(miss_exact false) is only an upper bound, so it is penalised one-sided.

Needs numpy and torch (offline only; the HUD never imports this):

    outputs/.mlenv/bin/python scripts/train_surrogate.py outputs/samples/*.jsonl.gz --arch resmlp
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import gen_escape_samples as gen  # noqa: E402
from optimize_evasion import FULL_SPEED_KMH, LIBRARY, LIBRARY_FINE, LIBRARY_SPEED  # noqa: E402

PLAN_SETS = {"speed": LIBRARY_SPEED, "fine": LIBRARY_FINE, "coarse": LIBRARY}
EVADER_KEYS = ("load_cap", "thrust_w", "drag_cap_w", "drag_1g_w")
HEADS = ("escape", "miss", "alt", "back")


# ---------------------------------------------------------------- features

def state_features(s):
    la, ta, chaff = s["launch_altitude_m"], s["target_altitude_m"], s["chaff_rcs_ratio"]
    course = math.radians(s["course_deg"])
    return [la, s["launch_speed_kmh"], ta-la, ta, s["target_speed_kmh"], math.cos(course), math.sin(course),
            s["azimuth_deg"], s["turn_g"], s["range_m"], math.log(s["range_m"]), 1. if chaff > 0 else 0.,
            math.log2(chaff) if chaff > 0 else 0.]


_EVADER_CACHE = {}


def evader_features(row, legacy_aircraft, legacy_mass_kg):
    """FM descriptors + mass. Rows from the single-aircraft generator carry none: rebuild them."""
    ev = row.get("evader")
    mass = row.get("mass_kg", legacy_mass_kg)
    if ev is None:
        aircraft = row.get("aircraft", legacy_aircraft)
        key = (aircraft, round(mass), round(row["target_altitude_m"], -1), round(row["target_speed_kmh"]))
        if key not in _EVADER_CACHE:
            _EVADER_CACHE[key] = gen.descriptors(gen._model(aircraft, mass), row["target_altitude_m"],
                                                 row["target_speed_kmh"]/3.6)
        ev = _EVADER_CACHE[key]
    if ev is None:  # Outside the FM tables at this condition.
        return None
    return [ev[k] for k in EVADER_KEYS]+[mass]


def plan_features(plans):
    """[target, plane, dive, speed target km/h]; plans without a speed hold full throttle."""
    return np.array([tuple(p)+((FULL_SPEED_KMH,) if len(p) == 3 else ()) for p in plans], dtype=np.float32)


def is_holdout(aircraft, fraction):
    if not fraction:
        return False
    return int(hashlib.md5(aircraft.encode()).hexdigest(), 16) % 1000 < fraction*1000


# ---------------------------------------------------------------- models

class Standardize(nn.Module):
    def __init__(self, x):
        super().__init__()
        x = torch.as_tensor(x)
        self.register_buffer("mean", x.mean(0))
        self.register_buffer("std", x.std(0).clamp_min(1e-6))

    def forward(self, x):
        return (x-self.mean)/self.std


class ResBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.body = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))

    def forward(self, x):
        return x+self.body(x)


def make_model(arch, x, n_out, width, depth):
    n_in = x.shape[1]
    if arch == "mlp":
        layers, n = [], n_in
        for _ in range(depth):
            layers += [nn.Linear(n, width), nn.SiLU()]
            n = width
        body = nn.Sequential(*layers, nn.Linear(n, n_out))
    else:  # resmlp: depth = number of residual blocks (two linear layers each).
        body = nn.Sequential(nn.Linear(n_in, width), *[ResBlock(width) for _ in range(depth)],
                             nn.LayerNorm(width), nn.SiLU(), nn.Linear(width, n_out))
    return nn.Sequential(Standardize(x), body)


# ---------------------------------------------------------------- data

def load(paths):
    """Rows of all files; rejected scenarios dropped, and an index written twice in one file
    (a scenario rerun after a pool restart) kept once."""
    rows, skipped = [], 0
    for path in paths:
        text = gzip.decompress(path.read_bytes()).decode() if path.suffix == ".gz" else path.read_text()
        timing = {}
        sidecar = Path(str(path).removesuffix(".gz").removesuffix(".jsonl")+".timing.jsonl")
        if sidecar.exists():  # add_timing.py back-fill for files generated before t_active was recorded.
            for line in sidecar.read_text().splitlines():
                try:
                    t = json.loads(line)
                    timing[t["index"]] = t
                except (ValueError, KeyError):
                    pass
        seen = set()
        for line in text.splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if "error" in row:  # Scenarios the simulator rejected carry no outcome.
                skipped += 1
                continue
            if row["index"] in seen:
                continue
            seen.add(row["index"])
            if "t_active" not in row and row["index"] in timing:
                row["t_active"] = timing[row["index"]].get("t_active")
            rows.append(row)
    return rows, skipped


def arrays(rows, legacy_aircraft, legacy_mass_kg):
    feats, keep = [], []
    for r in rows:
        ev = evader_features(r, legacy_aircraft, legacy_mass_kg)
        if ev is not None:
            feats.append(state_features(r)+ev)
            keep.append(r)
    s = np.array(feats, dtype=np.float32)
    hit = np.array([r["unevaded_hit"] for r in keep], dtype=np.float32)
    tof = np.array([r["time_of_flight_s"] for r in keep], dtype=np.float32)
    t_active = np.array([math.nan if r.get("t_active") is None else r["t_active"] for r in keep], dtype=np.float32)
    xs, ys = [], []
    for i, r in enumerate(keep):
        for e in r["evasions"]:
            rec = e["recommit"]
            miss = math.nan if e["fault"] else math.log1p(e["miss_m"])
            plan = list(e["plan"])+([FULL_SPEED_KMH] if len(e["plan"]) == 3 else [])
            xs.append((i, e["start_s"], *(FULL_SPEED_KMH if v is None else v for v in plan)))
            ys.append((e["escaped"], miss, 0. if e.get("miss_exact", True) else 1., e["altitude_lost_m"]/1000,
                       rec["back_s"]/10 if rec else math.nan))
    xs = np.array(xs, dtype=np.float32).reshape(-1, 6)
    ys = np.array(ys, dtype=np.float32).reshape(-1, 5)
    idx = xs[:, 0].astype(np.int64)
    ex = np.concatenate([s[idx], tof[idx, None], xs[:, 1:2], plan_features(xs[:, 2:6])], axis=1)
    return keep, s, hit, (tof, t_active), ex, ys


# ---------------------------------------------------------------- training

def _loss(out, tt, b):
    loss = 0.
    for j, kind, y, c in tt:
        t, o = y[b], out[:, j]
        if kind == "bce":
            loss = loss+nn.functional.binary_cross_entropy_with_logits(o, t)
            continue
        m = ~torch.isnan(t)
        if not m.any():
            continue
        err = o[m]-t[m]
        if kind == "censored":
            err = torch.where(c[b][m] > .5, err.clamp_min(0.), err)
        loss = loss+.5*(err*err).mean()
    return loss


def _tensors(targets, device):
    return [(j, kind, torch.tensor(y, device=device), None if c is None else torch.tensor(c, device=device))
            for j, kind, y, c in targets]


def train(model, x, targets, epochs, device, batch=4096, lr=2e-3, label="", val=None):
    """targets: list of (column index in the output, kind, y, censored flags or None).

    kind 'bce': logistic; 'mse': squared error, NaN masked; 'censored': squared
    error where exact, one-sided (only predictions above y) where flagged.
    val: (x, targets) of held-out data, scored after every epoch (loss and the
    accuracy of output column 0, which is always a 'bce' head)."""
    model.to(device)
    x = torch.tensor(x, device=device)
    tt = _tensors(targets, device)
    if val is not None:
        xv, tv = torch.tensor(val[0], device=device), _tensors(val[1], device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs*math.ceil(len(x)/batch))
    history = []
    for epoch in range(epochs):
        model.train()
        lr_now = opt.param_groups[0]["lr"]
        perm = torch.randperm(len(x), device=device)
        total = 0.
        for k in range(0, len(x), batch):
            b = perm[k:k+batch]
            loss = _loss(model(x[b]), tt, b)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += float(loss.detach())*len(b)
        row = dict(epoch=epoch+1, train_loss=total/len(x), lr=lr_now)
        if val is not None:
            model.eval()
            with torch.no_grad():
                vl, correct = 0., 0
                for k in range(0, len(xv), 65536):
                    b = torch.arange(k, min(k+65536, len(xv)), device=device)
                    out = model(xv[b])
                    vl += float(_loss(out, tv, b))*len(b)
                    correct += int(((out[:, 0] > 0) == (tv[0][2][b] > .5)).sum())
            row.update(val_loss=vl/len(xv), val_acc=correct/len(xv))
        history.append(row)
        print(f"  {label} epoch {epoch+1:3d}/{epochs}  train loss {row['train_loss']:.4f}"
              + (f"  val loss {row['val_loss']:.4f}  val acc {row['val_acc']:.2%}" if val is not None else "")
              + f"  lr {row['lr']:.2e}", flush=True)
    return model.cpu().eval(), history


class Ensemble(nn.Module):
    """Mean output of several members (logits and regressions alike)."""
    def __init__(self, members):
        super().__init__()
        self.members = nn.ModuleList(members)

    def forward(self, x):
        return torch.stack([m(x) for m in self.members]).mean(0)


def accuracy(model, x, y):
    if len(x) == 0:
        return None
    with torch.no_grad():
        p = model(torch.tensor(x))[:, 0].numpy() > 0
    return float((p == (y > .5)).mean())


# ---------------------------------------------------------------- reaction time

class Surrogate:
    def __init__(self, launch, escape):
        self.launch, self.escape = launch, escape

    @torch.no_grad()
    def scan(self, state_vec, plans, step=.25, force=False):
        """(None, None) for a predicted unevaded miss, else (starts, escape logits [start, plan]).
        force: scan even a predicted miss (the hit logit is in self.last_launch_logit)."""
        f = np.array([state_vec], dtype=np.float32)
        out = self.launch(torch.tensor(f))[0]
        self.last_launch_logit = float(out[0])
        if out[0] <= 0 and not force:
            return None, None
        tof = float(math.exp(out[1]))
        starts = np.arange(0., tof+1e-9, step, dtype=np.float32)
        n, m = len(starts), len(plans)
        x = np.concatenate([np.repeat(f, n*m, 0), np.full((n*m, 1), tof, np.float32),
                            np.repeat(starts, m)[:, None], np.tile(plan_features(plans), (n, 1))], axis=1)
        return starts, self.escape(torch.tensor(x))[:, 0].numpy().reshape(n, m)

    @torch.no_grad()
    def reaction(self, state_vec, plans, step=.25, max_gap_s=.5):
        """(reaction_s or None for an unevaded miss, plan at the reaction start).

        state_vec: state_features + evader_features of one engagement."""
        f = np.array([state_vec], dtype=np.float32)
        out = self.launch(torch.tensor(f))[0]
        if out[0] <= 0:
            return None, None
        tof = float(math.exp(out[1]))
        starts = np.arange(0., tof+1e-9, step, dtype=np.float32)
        n, m = len(starts), len(plans)
        x = np.concatenate([np.repeat(f, n*m, 0), np.full((n*m, 1), tof, np.float32),
                            np.repeat(starts, m)[:, None], np.tile(plan_features(plans), (n, 1))], axis=1)
        p = self.escape(torch.tensor(x))[:, 0].numpy().reshape(n, m)
        escaped = p.max(axis=1) > 0
        if not escaped[0]:
            return 0., None
        reaction, gap, best = 0., 0., None
        for k, t in enumerate(starts):
            if escaped[k]:
                reaction, gap, best = float(t), 0., tuple(plans[int(p[k].argmax())])
            else:
                gap += step
                if gap > max_gap_s+1e-9:
                    break
        return reaction, best


def check_tables(sur, table_dir):
    """Replay every look_down_angle grid-table cell with the tables' 36-plan library."""
    errors, flips, n, per = [], 0, 0, {}
    for path in sorted(Path(table_dir).glob("*.json")):
        d = json.loads(path.read_text())
        meta = d["meta"]
        if meta.get("clutter") != "look_down_angle":
            continue
        for c in d["cells"]:
            state = dict(launch_altitude_m=meta["launch_altitude_m"], launch_speed_kmh=meta["launch_speed_kmh"],
                         target_altitude_m=meta["target_altitude_m"], target_speed_kmh=meta["target_speed_kmh"],
                         course_deg=c["course_deg"], azimuth_deg=0., turn_g=c["turn_g"], range_m=c["range_m"],
                         chaff_rcs_ratio=meta["chaff_rcs_ratio"], aircraft=meta["evader_aircraft"],
                         mass_kg=meta["evader_mass_kg"])
            ev = evader_features(state, meta["evader_aircraft"], meta["evader_mass_kg"])
            got, _ = sur.reaction(state_features(state)+ev, LIBRARY)
            want = c["reaction_s"]
            n += 1
            if (got is None) != (want is None):
                flips += 1
                continue
            if want is not None:
                errors.append(got-want)
                per.setdefault(path.stem.split("__", 2)[2], []).append(got-want)
    err = np.array(errors)
    e = np.abs(err)
    print(f"\ntable check (36 plans): {n} cells, hit/miss flips {flips}, reaction MAE {e.mean():.2f} s, "
          f"bias {err.mean():+.2f} s, within 1 s {np.mean(e <= 1):.0%}, within 2 s {np.mean(e <= 2):.0%}, "
          f"over by >1 s {np.mean(err > 1):.0%}, over by >2 s {np.mean(err > 2):.0%}")
    for name, v in per.items():
        v = np.array(v)
        print(f"  {name:28s} MAE {np.abs(v).mean():5.2f}  bias {v.mean():+5.2f}")
    return dict(cells=n, flips=flips, mae_s=float(e.mean()), bias_s=float(err.mean()),
                within_1s=float(np.mean(e <= 1)), within_2s=float(np.mean(e <= 2)),
                over_1s=float(np.mean(err > 1)), over_2s=float(np.mean(err > 2)))


class TreeLaunch:
    """Gradient-boosted trees behind the launch-net interface: [hit logit, log time of flight]."""
    def __init__(self, classifier, regressor):
        self.classifier, self.regressor = classifier, regressor

    def __call__(self, x):
        x = x.numpy()
        return torch.tensor(np.stack([self.classifier.decision_function(x), self.regressor.predict(x)], axis=1))


class TreeEscape:
    """Gradient-boosted trees behind the escape-net interface: [escape logit]."""
    def __init__(self, classifier):
        self.classifier = classifier

    def __call__(self, x):
        return torch.tensor(self.classifier.decision_function(x.numpy())[:, None])


# ---------------------------------------------------------------- main

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("samples", nargs="+", type=Path)
    parser.add_argument("--arch", choices=("mlp", "resmlp"), default="mlp")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4, help="hidden layers (mlp) or residual blocks (resmlp)")
    parser.add_argument("--heads", default="escape",
                        help=f"escape-net outputs, comma separated from {','.join(HEADS)}; escape is required")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seeds", default="0", help="comma separated; several train an ensemble (mean logits)")
    parser.add_argument("--escape-from", type=Path, help="reuse this run's escape nets; train only the launch nets")
    parser.add_argument("--holdout-aircraft", type=float, default=.15,
                        help="fraction of aircraft types (by name hash) kept out of training entirely")
    parser.add_argument("--legacy-aircraft", default="f_16c_block_50", help="evader of rows without one")
    parser.add_argument("--legacy-mass-kg", type=float, default=12000.)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--threads", type=int, help="CPU threads for torch (several trainings side by side)")
    parser.add_argument("--tables", type=Path, default=ROOT/"data"/"offense",
                        help="grid tables to replay (PL-12 vs F-16C); skipped when absent - pass a missing path for other missiles")
    parser.add_argument("--out", type=Path, default=ROOT/"outputs"/"surrogate")
    args = parser.parse_args(argv)
    heads = args.heads.split(",")
    if heads[0] != "escape" or any(h not in HEADS for h in heads):
        raise SystemExit(f"--heads must start with escape and use {HEADS}")
    seeds = [int(x) for x in args.seeds.split(",")]
    if args.threads:
        torch.set_num_threads(args.threads)
    began = time.perf_counter()
    rows, skipped = load(args.samples)
    aircraft = lambda r: r.get("aircraft", args.legacy_aircraft)  # noqa: E731
    unseen = [r for r in rows if is_holdout(aircraft(r), args.holdout_aircraft)]
    rows = [r for r in rows if not is_holdout(aircraft(r), args.holdout_aircraft)]
    held = [r for r in rows if r["index"] % 10 == 0]
    rows = [r for r in rows if r["index"] % 10 != 0]
    unseen_types = sorted({aircraft(r) for r in unseen})
    print(f"{len(rows)} train / {len(held)} held-out scenarios / {len(unseen)} on {len(unseen_types)} unseen "
          f"aircraft; {skipped} rejected scenarios skipped", flush=True)
    rows, s, hit, tof, ex, ys = arrays(rows, args.legacy_aircraft, args.legacy_mass_kg)
    _, hs, hhit, htof, hex_, hys = arrays(held, args.legacy_aircraft, args.legacy_mass_kg)
    _, us, uhit, _, uex, uys = arrays(unseen, args.legacy_aircraft, args.legacy_mass_kg)
    print(f"{len(ex)} evasion runs, escaped {ys[:, 0].mean():.0%}, miss exact {np.mean(ys[:, 2] < .5):.0%}",
          flush=True)
    (tof, t_act), (htof, ht_act) = tof, htof
    log_tof = lambda h, t: np.where(h > .5, np.log(np.maximum(t, .1)), np.nan).astype(np.float32)  # noqa: E731
    active = lambda h, t: np.where(h > .5, t/10, np.nan).astype(np.float32)  # noqa: E731
    launch_t = [(0, "bce", hit, None), (1, "mse", log_tof(hit, tof), None), (2, "mse", active(hit, t_act), None)]
    launch_v = (hs, [(0, "bce", hhit, None), (1, "mse", log_tof(hhit, htof), None), (2, "mse", active(hhit, ht_act), None)])
    reuse = None
    if args.escape_from:
        ck = torch.load(args.escape_from/"surrogate.pt", weights_only=False)
        reuse = ck["escape"] if isinstance(ck["escape"], list) else [ck["escape"]]
        if ck["n_escape"] != ex.shape[1] or ck["arch"] != args.arch or len(reuse) != len(seeds):
            raise SystemExit("--escape-from: escape nets do not match these features / arch / seeds")
    columns = lambda y: {"escape": (0, "bce", y[:, 0], None), "miss": (1, "censored", y[:, 1], y[:, 2]),  # noqa: E731
                         "alt": (3, "mse", y[:, 3], None), "back": (4, "mse", y[:, 4], None)}
    pick = lambda y: [(j, kind, t, c) for j, (_, kind, t, c) in enumerate(columns(y)[h] for h in heads)]  # noqa: E731
    launches, escapes, history = [], [], {}
    for seed in seeds:
        torch.manual_seed(seed)
        print(f"\n== seed {seed}", flush=True)
        net, history[f"launch_{seed}"] = train(make_model(args.arch, s, 3, args.width, args.depth), s, launch_t,
                                               args.epochs*2, args.device, label=f"launch[{seed}]", val=launch_v)
        launches.append(net)
        if reuse is not None:
            net = make_model(args.arch, ex, len(heads), args.width, args.depth)
            net.load_state_dict(reuse[len(escapes)])
            escapes.append(net.eval())
            continue
        net, history[f"escape_{seed}"] = train(make_model(args.arch, ex, len(heads), args.width, args.depth), ex,
                                               pick(ys), args.epochs, args.device, label=f"escape[{seed}]",
                                               val=(hex_, pick(hys)))
        escapes.append(net)
    launch = launches[0] if len(seeds) == 1 else Ensemble(launches).eval()
    escape = escapes[0] if len(seeds) == 1 else Ensemble(escapes).eval()
    metrics = dict(launch_holdout_acc=accuracy(launch, hs, hhit), escape_holdout_acc=accuracy(escape, hex_, hys[:, 0]),
                   launch_unseen_acc=accuracy(launch, us, uhit), escape_unseen_acc=accuracy(escape, uex, uys[:, 0]),
                   unseen_aircraft=unseen_types)
    fmt = lambda v: "  -  " if v is None else f"{v:.1%}"  # noqa: E731
    print(f"\naccuracy  held-out scenarios: unevaded hit {fmt(metrics['launch_holdout_acc'])}, "
          f"escape {fmt(metrics['escape_holdout_acc'])}\n          unseen aircraft:    unevaded hit "
          f"{fmt(metrics['launch_unseen_acc'])}, escape {fmt(metrics['escape_unseen_acc'])}")
    if args.tables and Path(args.tables).is_dir() and any(Path(args.tables).glob("*.json")):
        metrics["tables"] = check_tables(Surrogate(launch, escape), args.tables)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(dict(launch=[m.state_dict() for m in launches], escape=[m.state_dict() for m in escapes],
                    arch=args.arch, width=args.width,
                    depth=args.depth, heads=heads, n_state=s.shape[1], n_escape=ex.shape[1]), args.out/"surrogate.pt")
    metrics.update(samples=[str(p) for p in args.samples], scenarios=len(rows)+len(held)+len(unseen),
                   evasion_runs=len(ex), arch=args.arch, width=args.width, depth=args.depth, heads=heads,
                   epochs=args.epochs, seeds=seeds, history=history, elapsed_s=round(time.perf_counter()-began, 1))
    (args.out/"metrics.json").write_text(json.dumps(metrics, indent=1))
    print(f"\n-> {args.out}  ({metrics['elapsed_s']} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

