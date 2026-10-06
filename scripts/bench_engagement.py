#!/usr/bin/env python3
"""Throughput of the engagement simulator in real generated matches: simulated seconds per wall-clock second.

    python3 scripts/bench_engagement.py --modes 1v1,4v4,16v16 --seeds 1,2,3 --until-s 300
    pypy3 scripts/bench_engagement.py --modes 1v1,4v4,16v16 --seeds 1,2,3 --until-s 900 --out outputs/engagements/bench_pypy.json

Each run is a ``match.random_match`` with the given seed (aircraft drawn by match frequency, so radar types mix as in a
real match) stepped until the match ends or ``--until-s``. Building the engagement (flight tables, ~0.4 s per aircraft
type on CPython the first time) is excluded. Reported per run: simulated / wall seconds, and the split into aircraft,
missiles, radar, RWR (beam-coverage checks and receiver updates) and scripts (observation building and decisions); then,
over all runs, the radar update cost per aircraft type and radar mode, which is what the earlier scripts/bench_sensors.py
could not show (it ran one radar type with every target in range).

The matches differ by interpreter (float summation, random streams), so equal seeds are not equal matches across CPython
and PyPy; compare the rates, not the outcomes. PyPy's first run includes JIT warm-up (the rows are in run order).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wt_overlay.engagement import RadarCommand, default_library  # noqa: E402
from wt_overlay.match import random_match  # noqa: E402

PARTS = ("planes", "missiles", "radar", "rwr", "script", "other")


class ForceRadarMode:
    """Wraps a pilot and replaces its radar command's mode (and pattern index 0): to price the radar modes the scripts do
    not use. The pilot's flying and shooting are untouched, so launches need a TWS track and stop under 'search'."""

    def __init__(self, inner, mode):
        self.inner, self.mode = inner, mode

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def decide(self, obs):
        action = self.inner.decide(obs)
        if action is not None and action.radar is not None:
            action = dataclasses.replace(action, radar=RadarCommand(self.mode, 0, action.radar.azimuth_deg,
                                                                     action.radar.elevation_deg))
        return action


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--modes", default="1v1,4v4,16v16")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--until-s", type=float, default=300.)
    parser.add_argument("--radar-mode", choices=("tws", "search"), default="tws",
                        help="tws: as the scripts fly (the real-match cost); search: every radar scans in search mode")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    library = default_library()
    interpreter = f"{platform.python_implementation()} {platform.python_version()}"
    print(f"{interpreter}, until {args.until_s:g} s, radar mode {args.radar_mode}")
    print(f"{'mode':>6} {'seed':>4} {'sim s':>7} {'wall s':>8} {'sim/wall':>9} | " + " ".join(f"{p:>8}" for p in PARTS)
          + " | launches kills missile-steps")
    rows, radar, rwr = [], {}, {}
    for mode in args.modes.split(","):
        n = int(mode.split("v")[0])
        for seed in (int(x) for x in args.seeds.split(",")):
            match = random_match(seed, team_size=n, library=library)
            if args.radar_mode != "tws":
                for spec in match.specs:
                    spec.controller = ForceRadarMode(spec.controller, args.radar_mode)
            built = time.perf_counter()
            engagement = match.engagement()
            build_s = time.perf_counter()-built
            t0 = time.perf_counter()
            result = engagement.run(until_s=args.until_s)
            wall = time.perf_counter()-t0
            w = result.wall
            total = max(w["total"], 1e-9)
            rows.append(dict(mode=mode, seed=seed, sim_s=result.time_s, wall_s=wall, rate=result.time_s/wall, build_s=build_s,
                             parts={p: w[p] for p in PARTS}, launches=result.launches, kills=len(result.kills),
                             missile_steps=engagement.missile_steps, ticks=result.ticks, aircraft=len(match.specs),
                             reason=result.reason))
            print(f"{mode:>6} {seed:>4} {result.time_s:7.1f} {wall:8.2f} {result.time_s/wall:9.2f} | "
                  + " ".join(f"{100*w[p]/total:7.0f}%" for p in PARTS)
                  + f" | {result.launches:8d} {len(result.kills):5d} {engagement.missile_steps:13d}")
            for key, (secs, calls) in result.radar_wall.items():
                slot = radar.setdefault(key, [0., 0])
                slot[0] += secs
                slot[1] += calls
            for key, (secs, calls) in engagement.rwr_timing.items():
                slot = rwr.setdefault(key, [0., 0])
                slot[0] += secs
                slot[1] += calls
    print("\nper mode (all seeds): simulated s per wall s, and microseconds per tick of each part (all aircraft together)")
    summary = {}
    for mode in args.modes.split(","):
        runs = [r for r in rows if r["mode"] == mode]
        sim, wall = sum(r["sim_s"] for r in runs), sum(r["wall_s"] for r in runs)
        ticks = sum(r["ticks"] for r in runs)
        parts = {p: sum(r["parts"][p] for r in runs)/ticks*1e6 for p in PARTS}
        steps = sum(r["missile_steps"] for r in runs)/ticks
        summary[mode] = dict(rate=sim/wall, us_per_tick=parts, missile_steps_per_tick=steps)
        print(f"{mode:>6}: {sim/wall:7.2f} x real time | " + " ".join(f"{p} {parts[p]:.0f}" for p in PARTS)
              + f" | {sum(parts.values()):.0f} us/tick, {steps:.1f} missile steps/tick")
    print("\nradar update cost by aircraft and mode over all runs (us per update, updates, share of radar time):")
    total = sum(v[0] for v in radar.values()) or 1.
    for key, (secs, calls) in sorted(radar.items(), key=lambda kv: -kv[1][0]):
        print(f"  {key:38s} {1e6*secs/calls:7.1f} us  x{calls:<9d} {100*secs/total:5.1f}%")
    print("\nRWR receiver update by aircraft (us per update where it ran, updates):")
    for key, (secs, calls) in sorted(rwr.items(), key=lambda kv: -kv[1][0]):
        print(f"  {key:38s} {1e6*secs/calls:7.1f} us  x{calls}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(dict(interpreter=interpreter, until_s=args.until_s, runs=rows, summary=summary,
                                            radar={k: v for k, v in radar.items()}, rwr=rwr), indent=1))
        print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
