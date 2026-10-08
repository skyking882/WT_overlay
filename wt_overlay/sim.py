"""Standalone, observation-driven 1v1 air combat: ``python -m wt_overlay.sim``.

``build_simulation(scenario, replay=...)`` returns an Engagement with the existing
script/intent/camera route. Its public ``step`` and ``run`` methods advance the
same world used by the legacy match runner and RL adapter. Replay truth is for
analysis; controllers still receive only their normal Observation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

from .archetypes import ARCHETYPES, SKILLS, bearing_of
from .engagement import Engagement, ReplayWriter, default_library, equipment_data
from .flight import MAX_ALTITUDE_M, MIN_STEP_SPEED_MPS, SUBSTEP_S
from .fm import atmosphere
from .match import scenario as make_match
from .rl_env import equip_scripts

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENARIO = ROOT / "data" / "sim" / "duel.json"
SCENARIO_KEYS = {"name", "seed", "time_limit_s", "map_half_m", "teams"}
PLANE_KEYS = {"aircraft", "position_m", "velocity_mps", "archetype", "skill", "missile", "missiles",
              "chaff", "mass_factor", "rcs_ratio", "flame_probability"}


def _number(value, label, *, lo=None, hi=None, inclusive_lo=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    if lo is not None and (value < lo if inclusive_lo else value <= lo):
        raise ValueError(f"{label} must be {'at least' if inclusive_lo else 'greater than'} {lo:g}")
    if hi is not None and value > hi:
        raise ValueError(f"{label} must be at most {hi:g}")
    return float(value)


def _integer(value, label, lo=None):
    if type(value) is not int or (lo is not None and value < lo):
        raise ValueError(f"{label} must be an integer" + (f" >= {lo}" if lo is not None else ""))
    return value


def validate_scenario(value):
    """Validate the compact JSON contract, without loading flight/missile models.

    Positions/velocities are ENU metres and m/s. This release takes two singleton
    teams. Omitted equipment and pilot fields use existing seeded match draws;
    the resolved values are saved alongside the requested scenario.
    """
    if not isinstance(value, dict) or set(value) - SCENARIO_KEYS:
        raise ValueError("scenario must be an object with keys: " + ", ".join(sorted(SCENARIO_KEYS)))
    out = copy.deepcopy(value)
    out.setdefault("name", "duel")
    if not isinstance(out["name"], str) or not out["name"].strip():
        raise ValueError("name must be a nonempty string")
    out["seed"] = _integer(out.get("seed", 1), "seed")
    out["time_limit_s"] = _number(out.get("time_limit_s", 900.), "time_limit_s", lo=0.)
    out["map_half_m"] = _number(out.get("map_half_m", 64000.), "map_half_m", lo=0.)
    teams = out.get("teams")
    if not isinstance(teams, list) or len(teams) != 2 or any(not isinstance(t, list) or len(t) != 1 for t in teams):
        raise ValueError("teams must contain exactly two teams with one aircraft each (1v1)")
    for i, team in enumerate(teams):
        p, label = team[0], f"teams[{i}][0]"
        if not isinstance(p, dict) or set(p) - PLANE_KEYS:
            raise ValueError(label + " must be an object with keys: " + ", ".join(sorted(PLANE_KEYS)))
        if not isinstance(p.get("aircraft"), str) or not p["aircraft"]:
            raise ValueError(label + ".aircraft must be a catalog id")
        for field in ("position_m", "velocity_mps"):
            v = p.get(field)
            if not isinstance(v, (list, tuple)) or len(v) != 3:
                raise ValueError(label + "." + field + " must contain three ENU numbers")
            p[field] = [_number(x, label + "." + field) for x in v]
        x, y, z = p["position_m"]
        if abs(x) > out["map_half_m"] or abs(y) > out["map_half_m"]:
            raise ValueError(label + ".position_m must start inside the map")
        _number(z, label + ".position_m altitude", lo=0., hi=MAX_ALTITUDE_M)
        speed = math.sqrt(sum(v*v for v in p["velocity_mps"]))
        _number(speed, label + ".velocity_mps speed", lo=MIN_STEP_SPEED_MPS, inclusive_lo=True,
                hi=2.*atmosphere(z)[1])
        if math.hypot(*p["velocity_mps"][:2]) < 1e-9:
            raise ValueError(label + ".velocity_mps needs a horizontal heading for the scripted pilot")
        for field, choices in (("archetype", ARCHETYPES), ("skill", SKILLS)):
            if field in p and p[field] not in choices:
                raise ValueError(label + "." + field + " must be one of " + ", ".join(choices))
        if "missile" in p and p["missile"] is not None and not isinstance(p["missile"], str):
            raise ValueError(label + ".missile must be a missile id or null (unarmed)")
        for field in ("missiles", "chaff"):
            if field in p:
                _integer(p[field], label + "." + field, 0)
        if "missile" in p and p["missile"] is None:
            if p.get("missiles", 0) != 0:
                raise ValueError(label + ": a null missile needs zero ammunition")
            p["missiles"] = 0
        for field in ("mass_factor", "rcs_ratio"):
            if field in p:
                p[field] = _number(p[field], label + "." + field, lo=0.)
        if "flame_probability" in p:
            p["flame_probability"] = _number(p["flame_probability"], label + ".flame_probability",
                                              lo=0., hi=1., inclusive_lo=True)
    return out


def load_scenario(path=DEFAULT_SCENARIO):
    return validate_scenario(json.loads(Path(path).read_text(encoding="utf-8")))


class _RecordedController:
    """Observe the exact input of an existing ScriptController without extra sensing."""
    def __init__(self, controller, engagement, ident):
        self.controller, self.engagement, self.ident = controller, engagement, ident
        self.contacts = {}

    def __getattr__(self, name):
        return getattr(self.controller, name)

    def decide(self, obs):
        # Contact refs are observation IDs, not world target IDs. A disappearance
        # means absence from this decision's picture, not a proved physical loss.
        for source in ("radar", "rwr", "maw", "flames", "visual", "missile_marks"):
            current = {}
            for c in getattr(obs, source):
                if source == "radar":
                    key = (c.kind, c.track_id, c.mark_id)
                elif source == "rwr":
                    key = (c.kind, c.contact_id, c.tracking, c.missile_warning)
                else:
                    key = (c.kind, c.ref)
                current[key] = c
            previous = self.contacts.get(source, {})
            for key in previous:
                if key not in current:
                    self.engagement.event("observed_contact", plane=self.ident, source=source,
                                          change="disappeared", contact_key=list(key))
            for key, c in current.items():
                if key not in previous:
                    self.engagement.event("observed_contact", plane=self.ident, source=source,
                                          change="appeared", contact_key=list(key), contact=asdict(c))
            self.contacts[source] = current
        return self.controller.decide(obs)


class Simulation(Engagement):
    """The existing world, with standalone replay instrumentation only.

    ``step()`` advances 1/48 s; ``run()`` completes the engagement. Terminal
    ``step()`` calls are no-ops. Dense missile samples include the public runtime
    state before cleanup, including the final sample of ended missiles.
    """
    def step(self):
        if self.reason is not None:
            return self.time
        return super().step()

    def _settle(self):
        if self.replay is not None and self.missiles:
            self.replay.write(dict(type="missile_tick", t=self.time, missiles=[
                dict(uid=m.uid, shooter=m.shooter.ident, target=m.target.ident, age_s=m.runtime.time_s,
                     runtime_state=list(m.runtime.state), seeker=m.seeker_on, datalink=m.datalink,
                     done=m.done, termination_event=m.event)
                for m in self.missiles if m.runtime is not None]))
        super()._settle()

    def _header(self):
        header = super()._header()
        header.update(simulator="wt_overlay.sim", scenario=self.scenario, resolved=self.resolved_scenario(),
                      observation_contract="ScriptController / autonomous IntentExecutor; truth_debug=False; camera-gated vision",
                      missile_tick_s=SUBSTEP_S,
                      missile_runtime_frame="east, up, south",
                      missile_runtime_columns=["x", "y", "z", "vx", "vy", "vz", "qx", "qy", "qz", "qw",
                                               "omega_x", "omega_y", "omega_z"])
        return header

    def resolved_scenario(self):
        planes = []
        for p in self.planes:
            planes.append(dict(id=p.ident, team=p.team, aircraft=p.aircraft, archetype=p.spec.archetype,
                               skill=p.spec.skill, missile=p.missile_id, profile_id=None if p.missile_id is None else
                               self.library.info(p.missile_id).profile_id, missiles=p.spec.missiles,
                               chaff=p.spec.chaff, position_m=list(p.spec.position), velocity_mps=list(p.spec.velocity),
                               mass_kg=p.flight.model.mass, mass_factor=p.spec.mass_factor, rcs_ratio=p.rcs_ratio,
                               flame_probability=p.flame_p, camera_fov_deg=p.camera.fov_deg,
                               radar=p.radar.radar.id if p.radar else None, rwr=p.rwr.rwr.id if p.rwr else None,
                               script=p.controller.describe()))
        return dict(name=self.scenario["name"], seed=self.seed, time_limit_s=self.time_limit_s,
                    map_half_m=self.map_half_m, planes=planes)


def build_simulation(value=None, *, replay=None, missile_sim=None):
    """Build a fresh simulation, suitable for ``sim.step()`` or ``sim.run()``.

    ``value`` is the compact scenario dict (None loads the included duel).
    ``replay`` is an optional existing ReplayWriter. World physics and equipment
    remain those of Engagement; no truth-enabled or legacy open-camera route is
    exposed here.
    """
    value = load_scenario() if value is None else validate_scenario(value)
    library = default_library(missile_sim)
    equipment = equipment_data().equipment
    teams = []
    for i, team in enumerate(value["teams"]):
        p = team[0]
        eq = equipment.get(p["aircraft"])
        if eq is None:
            raise ValueError(f"unknown aircraft {p['aircraft']!r}")
        mid = p.get("missile")
        if mid is not None:
            if mid not in eq.missiles:
                raise ValueError(f"{p['aircraft']} cannot carry {mid}")
            info = library.info(mid)
            if not library.is_active(info.profile_id):
                raise ValueError(f"{mid} is not an active radar missile supported by this release")
            if p.get("missiles", eq.missiles[mid]) > eq.missiles[mid]:
                raise ValueError(f"{p['aircraft']} carries at most {eq.missiles[mid]} {mid}")
        if p.get("chaff", 0) > eq.countermeasures:
            raise ValueError(f"{p['aircraft']} has at most {eq.countermeasures} countermeasure rounds")
        speed = math.sqrt(sum(v*v for v in p["velocity_mps"]))
        member = {k: p[k] for k in ("aircraft", "archetype", "skill", "missile", "missiles") if k in p}
        member.update(altitude_m=p["position_m"][2], mach=speed/atmosphere(p["position_m"][2])[1])
        teams.append([member])
    m = make_match(teams, value["seed"], range_km=math.dist(value["teams"][0][0]["position_m"][:2],
                   value["teams"][1][0]["position_m"][:2])/1000., library=library, map_half_m=value["map_half_m"])
    for i, s in enumerate(m.specs):
        p, enemy = value["teams"][i][0], value["teams"][1-i][0]
        if s.missile is None and p.get("missiles", 0) > 0:
            raise ValueError(f"{s.aircraft} has no supported active missile for the requested ammunition")
        if s.missile is not None and s.missiles > equipment[s.aircraft].missiles[s.missile]:
            raise ValueError(f"{s.aircraft} carries at most {equipment[s.aircraft].missiles[s.missile]} {s.missile}")
        s.position, s.velocity = tuple(p["position_m"]), tuple(p["velocity_mps"])
        if "missile" in p and p["missile"] is None:
            s.missile = None
        for key in ("chaff", "mass_factor", "rcs_ratio", "flame_probability"):
            if key in p:
                setattr(s, key, p[key])
        s.home_xy, s.enemy_xy = tuple(s.position[:2]), tuple(enemy["position_m"][:2])
        pilot = s.controller
        v = math.hypot(*s.velocity[:2])
        pilot.forward = (s.velocity[0]/v, s.velocity[1]/v)
        pilot.forward_deg = bearing_of(*pilot.forward)
        pilot.left = (-pilot.forward[1], pilot.forward[0])
        pilot.home_xy, pilot.enemy_xy = s.home_xy, s.enemy_xy
        pilot.anc = (*s.enemy_xy, None, "spawn")
    eng = Simulation(m.specs, m.seed, map_half_m=m.map_half_m, time_limit_s=value["time_limit_s"], library=library)
    eng.scenario = value
    equip_scripts(eng)
    for p in eng.planes:
        p.controller = _RecordedController(p.controller, eng, p.ident)
    eng.replay = replay
    if replay is not None:
        replay.write(eng._header())
    return eng


def result_document(sim):
    """Terminal/partial result with recorded outcomes, without guessed miss causes."""
    r = sim.result()
    winner = next((i for i, n in enumerate(r.teams_alive) if n), None) if r.reason == "annihilation" else None
    deaths = {d["uid"]: d for d in r.deaths if d["uid"] is not None}
    missiles = []
    for e in sim.log:
        if e["kind"] == "missile_end":
            death = deaths.get(e["uid"])
            outcome = "kill" if death is not None else "error" if e["result"] == "error" else \
                "fuse_without_new_kill" if e["result"] == "fuse" else "miss"
            missiles.append(dict(uid=e["uid"], shooter=e["shooter"], target=e["target"], outcome=outcome,
                                 termination_event=e["result"], end_time_s=e["t"], flight_s=e["flight_s"],
                                 closest_recorded_distance_m=e["miss_m"] if math.isfinite(e["miss_m"]) else None,
                                 killed_plane=None if death is None else death["victim"],
                                 miss_cause=None if outcome != "miss" else "not separately recorded"))
    return dict(name=sim.scenario["name"], seed=sim.seed, reason=r.reason, terminal=r.reason != "running",
                outcome="running" if r.reason == "running" else "draw" if winner is None else "team_win",
                winner_team=winner, time_s=r.time_s, ticks=r.ticks, teams_alive=list(r.teams_alive),
                launches=r.launches, missile_errors=r.missile_errors, fm_faults=sum(p["fm_faults"] for p in r.planes),
                planes=r.planes, kills=r.kills, deaths=r.deaths, missile_outcomes=missiles,
                missiles_in_flight=[dict(uid=m.uid, shooter=m.shooter.ident, target=m.target.ident, age_s=m.time_s,
                                        outcome="unresolved_at_match_end") for m in sim.missiles],
                event_counts=dict(sorted(Counter(e["kind"] for e in sim.log).items())),
                phase_time_s=r.phase_time_s,
                final_states=[dict(id=p.ident, alive=p.alive, state=asdict(p.flight.state),
                                   camera_bearing_deg=p.camera.bearing_deg, camera_elevation_deg=p.camera.elevation_deg)
                              for p in sim.planes])


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)+"\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run a complete, reproducible 1v1 air-combat simulation without a renderer.")
    parser.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO, help="compact JSON scenario (default: included duel)")
    parser.add_argument("--seed", type=int, help="override the scenario seed")
    parser.add_argument("--time-limit-s", type=float, help="override the scenario time limit")
    parser.add_argument("--out", type=Path, default=ROOT/"outputs"/"headless"/"duel", help="directory for scenario.json, replay.jsonl and result.json")
    parser.add_argument("--missile-sim", type=Path, help="read-only missile-model repository (default: sibling missle_sim)")
    args = parser.parse_args(argv)
    try:
        value = load_scenario(args.scenario)
        if args.seed is not None:
            value["seed"] = args.seed
        if args.time_limit_s is not None:
            value["time_limit_s"] = args.time_limit_s
        sim = build_simulation(value, missile_sim=args.missile_sim)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        parser.error(str(exc))
    args.out.mkdir(parents=True, exist_ok=True)
    _write_json(args.out/"scenario.json", dict(requested=sim.scenario, resolved=sim.resolved_scenario()))
    replay_path = args.out/"replay.jsonl"
    replay = ReplayWriter(replay_path, keep=False)
    sim.replay = replay
    replay.write(sim._header())
    started = time.perf_counter()
    try:
        sim.run()
    finally:
        replay.close()
    elapsed = time.perf_counter()-started
    doc = result_document(sim)
    # All replay bytes are simulation data. Performance and output paths occur
    # only in this explicitly separated result section, never in the replay.
    with replay_path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    doc["replay_sha256"] = digest
    doc["performance"] = dict(wall_s=elapsed, stepping_wall_s=sim.result().wall,
                              radar_wall=sim.result().radar_wall, missile_steps=sim.missile_steps)
    _write_json(args.out/"result.json", doc)
    result = f"team {doc['winner_team']} wins" if doc["winner_team"] is not None else "draw"
    print(f"{result} | {doc['reason']} at {doc['time_s']:.3f} simulated s | alive {doc['teams_alive']}")
    print(f"launches {doc['launches']} | missile errors {doc['missile_errors']} | FM faults {doc['fm_faults']} | "
          f"missiles still in flight {len(doc['missiles_in_flight'])}")
    print("missile outcomes: " + json.dumps(dict(Counter(m["outcome"] for m in doc["missile_outcomes"])), sort_keys=True))
    print(f"wall {elapsed:.2f} s | replay SHA-256 {digest}")
    print(f"outputs: {args.out.resolve()}")
    return 1 if doc["missile_errors"] or doc["fm_faults"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
