#!/usr/bin/env python3
"""Simulator ground truth for the hit-probability model (pk_behaviour.Behaviour).

For each reference engagement that hits without evasion, flies every combination
the behaviour model averages over: detection at launch or at t_active (RWR), each
human-delay quadrature point, each chaff ratio, each repertoire plan. A start at or
after the time of flight is not flown (the missile has arrived). One JSON line per
engagement holds the escape outcomes, so the hit probability for any behaviour
weights (p_react, p_correct, TWS or STT detection mix) is computed afterwards
without rerunning anything (pk_compare.py).

    pypy3 scripts/sim_pk.py --missile cn_pl12 --reference ref/cn_pl12__0.jsonl.gz --split calibrate --out pk/cn_pl12.jsonl
"""
from __future__ import annotations

import argparse
from collections import deque
import gzip
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import escape_window as ew  # noqa: E402
import gen_escape_samples as gen  # noqa: E402
from optimize_evasion import _pilot  # noqa: E402
from pk_behaviour import Behaviour  # noqa: E402

BEHAVIOUR = Behaviour()


def fly(job):
    row, t_active = job
    scenario = gen._scenario(row)
    model = gen._model(row["aircraft"], row["mass_kg"])
    tof = row["time_of_flight_s"]
    branches = []
    for t_det in (0., t_active):
        for k, delay in enumerate(BEHAVIOUR.delays()):
            if t_det is None:
                continue
            start = t_det+delay
            for ratio in BEHAVIOUR.chaff_ratios:
                if start >= tof:
                    escaped = [False]*len(BEHAVIOUR.repertoire)
                else:
                    escaped = [gen._evade(scenario, _pilot(start, plan), ratio, model)["escaped"]
                               for plan in BEHAVIOUR.repertoire]
                branches.append(dict(t_det=t_det, k=k, start=round(start, 3), ratio=ratio, escaped=escaped))
    return dict(index=row["index"], split=row["split"], t_active=t_active, tof=tof, branches=branches)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--missile", required=True)
    parser.add_argument("--reference", nargs="+", type=Path, required=True)
    parser.add_argument("--split", default="calibrate", help="select / calibrate / test / all")
    parser.add_argument("--missile-sim", type=Path, default=ew.ROOT.parent/"missle_sim")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--out", type=Path, required=True)
    gen.add_datalink_args(parser)
    args = parser.parse_args(argv)
    rows, timing = {}, {}
    for path in args.reference:
        text = gzip.decompress(path.read_bytes()).decode() if path.suffix == ".gz" else path.read_text()
        for line in text.splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if "error" not in r and r["unevaded_hit"] and (args.split == "all" or r["split"] == args.split):
                rows.setdefault(r["index"], r)
        sidecar = Path(str(path).removesuffix(".gz").removesuffix(".jsonl")+".timing.jsonl")
        if sidecar.exists():
            for line in sidecar.read_text().splitlines():
                t = json.loads(line)
                timing[t["index"]] = t.get("t_active")
    for i, r in rows.items():
        if r.get("t_active") is None and i in timing:
            r["t_active"] = timing[i]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = gen.done_indices(args.out)
    todo = deque(i for i in sorted(rows) if i not in done)
    init = (str(args.missile_sim), args.missile, "f_16c_block_50", 10000., True, None, "rwr", (0., .1, 1.),
            "look_down_angle", 2., True, *gen.datalink_init(args))
    gen.run_resumable(args.out, todo, len(done), args.workers, init,
                      lambda i: (rows[i], rows[i].get("t_active")), fly, 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
