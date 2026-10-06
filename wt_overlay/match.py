"""Match generator: aircraft, spawns, loadouts, scripts for a simulated top-tier Air RB match (docs/rl_design.md section 1).

``random_match`` draws two teams from data/match/top_tier.json (aircraft by match frequency, archetypes around the
group priors by a Dirichlet, skill normal 80 % / top 20 %), spawns them on two lines 90-110 km apart,
2-3 km high at Mach 0.8-1.2, heading at the enemy, with +-15 km lateral spread, and fits each with one of the modelled
active missiles at its maximum count. ``scenario`` builds small matches (1v1, 2v2) from chosen aircraft, archetypes and
range. Both return a ``Match``; ``Match.engagement(replay=...)`` builds the ``engagement.Engagement``.

World layout: team 0 spawns south of the centre flying north (forward = +y), team 1 north of it flying south, so each
team's left (forward turned counter-clockwise) is west for team 0 and east for team 1: left flyers of both teams run
past each other and meet the other side's right flank. Spawn parameters other than those in the JSON are assumptions
(grade D): the spawn jitter along the line, the loadout choice, the chaff share of the dispensers (50-100 %),
the chaff RCS ratio, the aircraft mass factor (1.15-1.45) and the skill split.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import random

from . import pk
from .archetypes import ARCHETYPES, SKILLS, Pilot, perturb_params, range_judge, sample_params
from .engagement import MAP_HALF_M, Engagement, MissileLibrary, PlaneSpec, default_library, equipment_data
from .fm import atmosphere

DATA_FILE = Path(__file__).resolve().parents[1]/"data"/"match"/"top_tier.json"
SKILL_SHARE = {"normal": .8, "top": .2}   # D
CHAFF_SHARE = (.5, 1.)                    # share of the countermeasure rounds that are chaff (D)
RCS_RATIOS = (.5, 1., 2.)
MASS_FACTOR = (1.15, 1.45)                # total / empty mass (D, spec)
ALONG_JITTER_M = 1000.                    # spawn scatter along the flight direction (D)
LINE_FILL = (.2, .8)                      # stratified lateral spawn slots: position inside its slot


def load_model(path=None) -> dict:
    return json.loads(Path(path or DATA_FILE).read_text())


def group_of(model: dict, aircraft: str) -> dict | None:
    return next((g for g in model["groups"] if aircraft in g["aircraft"]), None)


def dirichlet(prior: dict, concentration: float, rng: random.Random) -> dict:
    """A draw around ``prior`` (mean = prior, Dirichlet(concentration x prior))."""
    draws = {k: rng.gammavariate(max(1e-3, concentration*v), 1.) for k, v in prior.items()}
    total = sum(draws.values())
    return {k: v/total for k, v in draws.items()}


def choose(weights: dict, rng: random.Random):
    """A key of ``weights`` with probability proportional to its value (insertion order, deterministic)."""
    total = sum(weights.values())
    x = rng.random()*total
    acc = 0.
    last = None
    for key, w in weights.items():
        acc += w
        last = key
        if x < acc:
            return key
    return last


_PK = []


def _pk_available():
    if not _PK:
        _PK.append(frozenset(pk.available()))
    return _PK[0]


def modelled_missiles(aircraft: str, library: MissileLibrary) -> list:
    """Active missiles this aircraft carries that missile_sim and the hit-probability model both have."""
    eq = equipment_data().equipment.get(aircraft)
    if eq is None:
        return []
    have_pk = _pk_available()
    out = []
    for mid in sorted(eq.missiles):
        try:
            info = library.info(mid)
        except (KeyError, ValueError):
            continue
        if mid in have_pk and library.is_active(info.profile_id):
            out.append(mid)
    return out


def spawn_state(team: int, slot: int, n: int, rng: random.Random, separation_m: float, spread_m: float, aircraft: str,
                altitude_m: float, mach: float):
    """Position and velocity (ENU) of aircraft ``slot`` of ``n`` on its team's spawn line."""
    lat = -spread_m+2.*spread_m*((slot+rng.uniform(*LINE_FILL))/n) if n > 1 else 0.
    along = rng.uniform(-ALONG_JITTER_M, ALONG_JITTER_M)
    forward = (0., 1.) if team == 0 else (0., -1.)
    y = (-separation_m/2. if team == 0 else separation_m/2.)+along
    speed = mach*atmosphere(altitude_m)[1]
    return (lat, y, altitude_m), (forward[0]*speed, forward[1]*speed, 0.)


def _rotate(v, theta_deg):
    """East/north vector turned clockwise by compass ``theta_deg`` (a bearing b becomes b+theta)."""
    if not theta_deg:
        return tuple(v)
    c, s = math.cos(math.radians(theta_deg)), math.sin(math.radians(theta_deg))
    return v[0]*c+v[1]*s, v[1]*c-v[0]*s


def spawn_layout(seed, layout, separation_m, spread_m, map_half_m):
    """Opt-in spawn geometry: ``layout`` = {"rotate": bool, "offset_km": km, "margin_km": km (default 8)}.

    The north-south spawn picture (team 0 south, team 1 north) is turned about the map centre by a random compass
    angle (rotate) and shifted by up to offset_km along each map axis, keeping every spawn line at least margin_km
    inside the map. Diagonal and off-centre spawns, as on real maps. Returns (theta_deg, dx_m, dy_m) or None; draws
    come from their own generator, so matches without a layout are unchanged."""
    if not layout:
        return None
    rng = random.Random(f"{seed}:layout")
    margin = float(layout.get("margin_km", 8.))*1000.
    reach = float(layout.get("offset_km", 0.))*1000.
    half_len = separation_m/2.+ALONG_JITTER_M
    corners = [(sx*spread_m, sy*half_len) for sx in (-1., 1.) for sy in (-1., 1.)]
    for _ in range(64):
        theta = rng.uniform(0., 360.) if layout.get("rotate") else 0.
        pts = [_rotate(c, theta) for c in corners]
        shift = []
        for k in (0, 1):
            lo = -map_half_m+margin-min(p[k] for p in pts)
            hi = map_half_m-margin-max(p[k] for p in pts)
            if lo > hi:
                break
            a, b = max(lo, -reach), min(hi, reach)
            shift.append(rng.uniform(a, b) if a <= b else min(max(0., lo), hi))
        if len(shift) == 2:
            return theta, shift[0], shift[1]
    return 0., 0., 0.


@dataclass
class Match:
    seed: int
    specs: list
    separation_m: float
    map_half_m: float
    library: MissileLibrary
    model: dict
    notes: dict = field(default_factory=dict)

    def engagement(self, replay=None, **kwargs) -> Engagement:
        intent_layer=kwargs.pop("intent_layer",True)
        fov_range=kwargs.pop("fov_range",(90.,120.))
        execution=kwargs.pop("execution",None)
        eng=Engagement(self.specs, self.seed, map_half_m=self.map_half_m, library=self.library, replay=replay,**kwargs)
        if intent_layer:
            from .rl_env import equip_scripts
            equip_scripts(eng,fov_range=fov_range,execution=execution)
        return eng

    def describe(self):
        return [dict(id=i, team=s.team, aircraft=s.aircraft, archetype=s.archetype, skill=s.skill, missile=s.missile,
                     missiles=s.missiles, chaff=s.chaff, rcs_ratio=s.rcs_ratio) for i, s in enumerate(self.specs)]


def _spec(i, team, aircraft, archetype, skill, slot, n, rng, model, library, separation_m, spread_m, map_half_m,
          debug, judges, missile=None, altitude_m=None, mach=None, missiles=None, layout=None, perturb=None, seed=0):
    spawn = model["spawn"]
    altitude = altitude_m if altitude_m is not None else rng.uniform(*spawn["altitude_m"])
    mach = mach if mach is not None else rng.uniform(*spawn["speed"]["mach_range"])
    position, velocity = spawn_state(team, slot, n, rng, separation_m, spread_m, aircraft, altitude, mach)
    eq = equipment_data().equipment[aircraft]
    options = modelled_missiles(aircraft, library)
    if missile is None:
        missile = options[rng.randrange(len(options))] if options else None
    count = 0 if missile is None else (eq.missiles.get(missile, 0) if missiles is None else missiles)
    chaff = round(rng.uniform(*CHAFF_SHARE)*eq.countermeasures)
    params = sample_params(model, aircraft, archetype, skill, rng)
    if perturb is not None:
        params = perturb_params(params, perturb, random.Random(f"{seed}:perturb:{i}"))
    forward = (0., 1.) if team == 0 else (0., -1.)
    half = separation_m/2.
    home, enemy = (0., -half if team == 0 else half), (0., half if team == 0 else -half)
    if layout is not None:
        theta, dx, dy = layout
        place = lambda p: tuple(a+b for a, b in zip(_rotate(p, theta), (dx, dy)))
        position = (*place(position[:2]), position[2])
        velocity = (*_rotate(velocity[:2], theta), velocity[2])
        forward, home, enemy = _rotate(forward, theta), place(home), place(enemy)
    pilot = Pilot(params, random.Random(rng.random()), team_forward=forward, home_xy=home, enemy_xy=enemy,
                  missile_id=missile if count else None, map_half_m=map_half_m, debug=debug,
                  judge=range_judge(missile, judges) if count else None)
    pilot.ident = i
    return PlaneSpec(aircraft, team, position, velocity, pilot, missile, count, chaff, rng.choice(RCS_RATIOS), .5,
                     rng.uniform(*MASS_FACTOR), name=f"{aircraft}#{i}", skill=skill, archetype=archetype)


def random_match(seed: int, *, team_size: int | None = None, model: dict | None = None, library: MissileLibrary | None = None,
                 map_half_m: float = MAP_HALF_M, debug: bool = False, layout: dict | None = None,
                 perturb: dict | None = None) -> Match:
    """A full match: ``team_size`` (default the JSON's 16) aircraft per team drawn by match frequency. ``layout``:
    optional rotated / shifted spawn geometry (spawn_layout); ``perturb``: optional script perturbation
    (archetypes.perturb_params, own generator per pilot)."""
    model = model or load_model()
    library = library or default_library()
    rng = random.Random(f"{seed}:match")
    n = team_size or model["team_size"]
    weights = {a: w for a, w in model["aircraft_frequency"]["weights"].items() if modelled_missiles(a, library)}
    separation = rng.uniform(*model["spawn"]["separation_km"])*1000.
    jitter = {}
    for g in model["groups"]:
        jitter[g["name"]] = dirichlet(g["prior"], 20., rng)
    specs = []
    judges = {}
    spread = model["spawn"].get("spread_km", 15.)*1000.
    lay = spawn_layout(seed, layout, separation, spread, map_half_m)
    for team in (0, 1):
        aircraft_list = [choose(weights, rng) for _ in range(n)]
        for slot, aircraft in enumerate(aircraft_list):
            group = group_of(model, aircraft)
            archetype = choose(jitter[group["name"]], rng) if group else "left"
            skill = choose(SKILL_SHARE, rng)
            specs.append(_spec(len(specs), team, aircraft, archetype, skill, slot, n, rng, model, library, separation,
                               spread, map_half_m, debug, judges, layout=lay, perturb=perturb, seed=seed))
    notes = dict(archetype_mix=jitter)
    if lay is not None:
        notes["layout"] = lay
    return Match(seed, specs, separation, map_half_m, library, model, notes)


def scenario(teams, seed=0, *, range_km: float | None = None, model: dict | None = None, library: MissileLibrary | None = None,
             map_half_m: float = MAP_HALF_M, debug: bool = False, spread_km: float = 15.,
             layout: dict | None = None, perturb: dict | None = None) -> Match:
    """A chosen match. ``teams``: two lists, one per team, of aircraft ids or dicts with ``aircraft`` and optional
    ``archetype``, ``skill``, ``missile``, ``missiles``, ``altitude_m``, ``mach``. Unspecified archetypes follow the group
    prior, skills the 80/20 split. ``range_km``: spawn line separation (default random 90-110 km). ``layout``:
    optional rotated / shifted spawn geometry (spawn_layout); ``perturb``: optional script perturbation. A member's
    explicit archetype / skill is kept; perturbation varies only its parameters."""
    model = model or load_model()
    library = library or default_library()
    rng = random.Random(f"{seed}:scenario")
    separation = (range_km if range_km is not None else rng.uniform(*model["spawn"]["separation_km"]))*1000.
    specs, judges = [], {}
    lay = spawn_layout(seed, layout, separation, spread_km*1000., map_half_m)
    for team, members in enumerate(teams):
        for slot, member in enumerate(members):
            member = dict(aircraft=member) if isinstance(member, str) else dict(member)
            aircraft = member["aircraft"]
            group = group_of(model, aircraft)
            archetype = member.get("archetype") or (choose(dirichlet(group["prior"], 20., rng), rng) if group else "left")
            skill = member.get("skill") or choose(SKILL_SHARE, rng)
            if archetype not in ARCHETYPES or skill not in SKILLS:
                raise ValueError(f"unknown archetype {archetype!r} or skill {skill!r}")
            specs.append(_spec(len(specs), team, aircraft, archetype, skill, slot, len(members), rng, model, library,
                               separation, spread_km*1000., map_half_m, debug, judges, missile=member.get("missile"),
                               altitude_m=member.get("altitude_m"), mach=member.get("mach"),
                               missiles=member.get("missiles"), layout=lay, perturb=perturb, seed=seed))
    return Match(seed, specs, separation, map_half_m, library, model, {} if lay is None else dict(layout=lay))
