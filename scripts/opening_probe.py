#!/usr/bin/env python3
"""Offline run of the opening-shot planner (wt_overlay.opening.plan_opening) from a spawn state.

Same code path as the HUD: worker processes, the same search, the same plan. For checks against
docs/opening_shot_spec.md section 7 and for trying other aircraft, missiles and enemy assumptions:

    python3 scripts/opening_probe.py --aircraft f_16c_block_52_aesa --missile us_aim_120c_5 --fuel 0.3
    python3 scripts/opening_probe.py --aircraft j_16 --missile cn_pl12 --missiles 8 --fuel 0.3 --workers 4

Own mass = FM empty mass + fuel fraction x internal fuel + missiles x the missile's launch mass (pylons and other
stores are not counted; in game the planner uses the 8111 mass instead).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wt_overlay.opening import OpeningSettings, PlanRequest, plan_opening, validate_settings, worker_init  # noqa: E402


def own_mass(aircraft, fuel_fraction, missiles, missile):
    from wt_overlay.flight import fuel_data
    from wt_overlay.opening import _library, _missile_mass
    fuel = fuel_data(aircraft)
    return fuel.empty_kg+fuel_fraction*fuel.internal_kg+missiles*_missile_mass(_library(), missile)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--aircraft", required=True)
    ap.add_argument("--missile", required=True)
    ap.add_argument("--missiles", type=int, default=6, help="missiles carried (mass only)")
    ap.add_argument("--fuel", type=float, default=.3, help="own internal fuel fraction")
    ap.add_argument("--mass-kg", type=float, help="own total mass instead of --fuel / --missiles")
    ap.add_argument("--spawn-altitude", type=float, default=2500.)
    ap.add_argument("--spawn-tas", type=float, default=290.)
    ap.add_argument("--distance-km", type=float, default=100.)
    ap.add_argument("--enemy", default=OpeningSettings.enemy_aircraft)
    ap.add_argument("--enemy-fuel", type=float, default=OpeningSettings.enemy_fuel_fraction)
    ap.add_argument("--enemy-climb", type=float, default=OpeningSettings.enemy_climb_deg)
    ap.add_argument("--enemy-level", type=float, default=OpeningSettings.enemy_level_m)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    settings = OpeningSettings(True, a.missile, a.distance_km*1000., a.enemy, a.enemy_fuel,
                               enemy_climb_deg=a.enemy_climb, enemy_level_m=a.enemy_level, workers=a.workers)
    validate_settings(settings)
    mass = a.mass_kg or own_mass(a.aircraft, a.fuel, a.missiles, a.missile)
    request = PlanRequest(a.aircraft, mass, a.missile, settings, a.spawn_altitude, a.spawn_tas, 0., 0.,
                          a.spawn_altitude, a.spawn_tas, 0.)
    with ProcessPoolExecutor(a.workers, initializer=worker_init) as pool:
        plan = plan_opening(request, lambda f, jobs: list(pool.map(f, jobs)))
    out = asdict(plan)
    out.pop("profile")
    print(json.dumps(dict(mass_kg=round(mass), **out), ensure_ascii=False, indent=1, default=str))
    return 0 if plan.available else 1


if __name__ == "__main__":
    raise SystemExit(main())
