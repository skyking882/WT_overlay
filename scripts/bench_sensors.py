#!/usr/bin/env python3
"""Cost of the radar / RWR sensor model for a 16 v 16 match: 32 radars, each seeing 31 other aircraft per tick.

    python3 scripts/bench_sensors.py [--ticks 480] [--radar f_16c_block_50] [--spread-km 50]
    pypy3 scripts/bench_sensors.py

All aircraft are within ``--spread-km`` of each other (every target passes the range gate: the worst case) and fly
straight at 250 m/s; every radar tracks (TWS, first pattern) and every aircraft has an RWR.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wt_overlay import units  # noqa: E402
from wt_overlay.sensors import OwnState, RadarSensor, RwrSensor, TargetTruth  # noqa: E402

DT = 1/48


def world(n, spread_m, rng):
    return [dict(id=i, position=[rng.uniform(-spread_m, spread_m), rng.uniform(-spread_m, spread_m), rng.uniform(2000., 8000.)],
                 heading=rng.uniform(0., 360.)) for i in range(n)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ticks", type=int, default=480)
    parser.add_argument("--radar", default="f_16c_block_50")
    parser.add_argument("--rwr", default="f_16c_block_50")
    parser.add_argument("--aircraft", type=int, default=32)
    parser.add_argument("--spread-km", type=float, default=50.)
    args = parser.parse_args(argv)
    u = units.load()
    rng = random.Random(1)
    planes = world(args.aircraft, args.spread_km*1000., rng)
    radars = [RadarSensor(u.radar_of(args.radar), owner=p["id"]) for p in planes]
    rwrs = [RwrSensor(u.rwr_of(args.rwr)) for _ in planes]
    for r in radars:
        r.set_mode("tws", 0, 0., 0., t=0.)
    spent = dict(radar=0., illuminated=0., rwr=0., build=0.)
    hits = contacts = flashes = 0
    for k in range(1, args.ticks+1):
        t = k*DT
        t0 = time.perf_counter()
        owns, truths = [], []
        for p in planes:
            h = math.radians(p["heading"])
            v = (250.*math.sin(h), 250.*math.cos(h), 0.)
            p["position"] = [x+y*DT for x, y in zip(p["position"], v)]
            owns.append(OwnState(tuple(p["position"]), v, p["heading"]))
            truths.append(TargetTruth(p["id"], tuple(p["position"]), v))
        t1 = time.perf_counter()
        pictures = [r.update(t, DT, o, truths) for r, o in zip(radars, owns)]
        t2 = time.perf_counter()
        covered = [r.illuminated(t, o, [(q["id"], tuple(q["position"])) for q in planes if q["id"] != r.owner], DT)
                   for r, o in zip(radars, owns)]
        t3 = time.perf_counter()
        for i, rwr in enumerate(rwrs):
            emissions = [r.emission(j, t, owns[j], truths[i].position, 0.) if i in covered[j] else None
                         for j, r in enumerate(radars) if j != i]
            rwr.update(t, owns[i], [e for e in emissions if e is not None])
        t4 = time.perf_counter()
        if k > args.ticks//4:   # the first quarter is warm-up (JIT)
            spent["build"] += t1-t0
            spent["radar"] += t2-t1
            spent["illuminated"] += t3-t2
            spent["rwr"] += t4-t3
        hits += sum(len(p.hits) for p in pictures)
        contacts += sum(len(p.contacts) for p in pictures)
        flashes += sum(len(c) for c in covered)
    n = args.ticks-args.ticks//4
    per = {k: v/n for k, v in spent.items()}
    radar_us = per["radar"]/len(planes)*1e6
    print(f"{sys.implementation.name} {sys.version.split()[0]}: {len(planes)} aircraft, {args.radar} radars, {args.ticks} ticks "
          f"({hits} detections, {contacts} track-ticks, {flashes} illuminations)")
    print(f"  radar update (31 targets):  {radar_us:7.1f} us per radar, {per['radar']*1e3:6.2f} ms per tick for {len(planes)} radars")
    print(f"  illuminated (31 points):    {per['illuminated']/len(planes)*1e6:7.1f} us per radar, {per['illuminated']*1e3:6.2f} ms per tick")
    print(f"  emissions + RWR update:     {per['rwr']/len(planes)*1e6:7.1f} us per receiver, {per['rwr']*1e3:6.2f} ms per tick")
    print(f"  own/truth state build:      {per['build']*1e3:6.2f} ms per tick")
    print(f"  total sensors:              {sum(per.values())*1e3:6.2f} ms per tick ({1/DT:.0f} Hz real time = {1e3*DT:.1f} ms)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
