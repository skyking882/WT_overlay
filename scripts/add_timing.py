#!/usr/bin/env python3
"""Add detection timing (t_active, burn_s; see gen_escape_samples.detection_times) to existing
sample or reference files by rerunning only their unevaded flight. Writes <file>.timing.jsonl
with {index, t_active, burn_s, tof_check}; tof_check is the rerun's time of flight, which must
match the stored one (the stored state is rounded to 1e-3, so a tiny difference is possible).

    pypy3 scripts/add_timing.py --missile cn_pl12 outputs/samples/cn_pl12__all.jsonl.gz
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

ROWS = {}


def timing(row):
    result = gen._simulate(gen._scenario(row), None)
    t_active, burn_s = gen.detection_times(result)
    return dict(index=row["index"], t_active=t_active, burn_s=burn_s,
                tof_check=round(result["summary"]["flight_time_s"], 2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--missile", required=True)
    parser.add_argument("--missile-sim", type=Path, default=ew.ROOT.parent/"missle_sim")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    gen.add_datalink_args(parser)
    args = parser.parse_args(argv)
    init = (str(args.missile_sim), args.missile, "f_16c_block_50", 10000., True, None, "rwr", (0., .1, 1.),
            "look_down_angle", 2., True, *gen.datalink_init(args))
    for path in args.files:
        text = gzip.decompress(path.read_bytes()).decode() if path.suffix == ".gz" else path.read_text()
        rows = {}
        for line in text.splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if "error" not in r:
                rows.setdefault(r["index"], {k: r[k] for k in (
                    "index", "launch_altitude_m", "launch_speed_kmh", "target_altitude_m", "target_speed_kmh",
                    "course_deg", "azimuth_deg", "turn_g", "range_m")})
        out = Path(str(path).removesuffix(".gz").removesuffix(".jsonl")+".timing.jsonl")
        done = gen.done_indices(out)
        todo = deque(i for i in sorted(rows) if i not in done)
        gen.run_resumable(out, todo, len(done), args.workers, init, lambda i: rows[i], timing, 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
