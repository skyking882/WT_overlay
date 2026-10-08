"""Scripted player archetypes for the engagement simulator (docs/rl_design.md section 2).

Five archetypes (left, right, middle, crawler, rusher) fly one phase machine::

    climb      climb to the level altitude while drifting toward the flank (left / right) or straight (middle)
    suppress   optional: turn toward the enemy and fire one long-range shot at 80-100 % of Rmax, then turn away or
               crank 30-50 degrees and support the missile until its seeker is on
    advance    no suppression shot planned: fly toward the enemy and shoot inside the second-round range
    evade      three-nine (beam) or turn away toward the outer side, maybe diving, chaff when a missile warning shows
    recommit   the RWR has been clear for 3-10 s: turn back toward the enemy
    round2     fire inside 20-30 km, support until the seeker is on, then evade again (evade -> recommit -> round2 ...)

Crawlers fly ``crawl`` (30-200 m) toward the enemy, ``popup`` to shoot when close and then evade; rushers
fly ``rush`` straight at the enemy, shoot at mid range and evade late or not at all; a pilot with no missiles left
goes ``home`` (still defending; with the opt-in airfield he lands there, rearms and comes back through ``advance``,
``crawl`` or ``rush``). Left / right / middle share the shape above with their own flank and
suppression parameters. Coordinates are ENU; a team's forward points at the enemy spawn and its left is forward
turned counter-clockwise by 90 degrees, so "outer side" is left for left flyers and middle pilots, right for right flyers.

A pilot decides from an ``engagement.Observation`` only (own state, radar tracks, RWR, MAW, missile flames, sightings
and map marks), about every 0.5 s; the command holds in between. New threats (RWR missile warning, MAW, flame, an
STT lock warning noticed with ``p_lock_react``) start a reaction that happens after a log-normal delay (median 1.5 s);
"normal" pilots react with probability 0.85 and pick the right manoeuvre with probability 0.60, "top" pilots always
react and are always right; a wrong pick is drawn from the repertoire of scripts/pk_behaviour.py (the same defaults, a
test pins them). Range judgement uses wt_overlay.offense's Rmax line from the distilled hit-probability model, binned by
altitude and speed and cached, with pk.Assumption()'s default target (the enemy type is unknown).

Every parameter that is not in data/match/top_tier.json is an assumption (grade D, listed in docs/rl_design.md
section 8): the flank distance, cruise Mach, dive angle and floor, chaff cadence, support limits, the crawler's
pop-up distance, the rusher's shooting range and reaction probability, the boundary guard, going home with no missiles.

Population v2 (``PilotV2``, docs/population_spec.md; opt-in through match ``population="v2"`` or MatchEnv
``script_perturbation={"population": "v2"}``): the same observation, executor and intent heads, but pilots turn in on
information events instead of altitude and flank distance, draw every evasion afresh from their own propensities, follow
a drawn missile doctrine, and come in six more archetypes (spammer, speedster, deck_notcher, pair, bait, pusher).
Without the option the classes and functions above behave exactly as before.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import math
import random

from . import offense, pk
from .engagement import STT_TRACK, Action, Observation, RadarCommand
from .fm import atmosphere
from .flight import FlightCommand

ARCHETYPES = ("left", "right", "middle", "crawler", "rusher")
SKILLS = ("normal", "top")

# Human behaviour; scripts/pk_behaviour.Behaviour has the same defaults (tests/test_engagement.py compares).
P_REACT = {"normal": .85, "top": 1.}
P_CORRECT = {"normal": .60, "top": 1.}
P_LOCK_REACT = {"normal": .5, "top": 1.}
DELAY_MEDIAN_S, DELAY_SIGMA = 1.5, .4
# (target_deg from away-from-threat, plane_deg, dive_deg, speed_kmh; >= 1250 means full power): see pk_behaviour.Behaviour.
REPERTOIRE = ((90., 0., 0., 1500.), (90., 0., 20., 1500.), (90., 0., 40., 1500.), (90., 90., 20., 1500.),
              (0., 0., 0., 1500.), (0., 0., 20., 1500.), (90., 0., 0., 600.))
FULL_POWER_KMH = 1250.

# D: assumptions of this module.
DENIED_S = 40.            # a pilot who decided not to react ignores new warnings this long
MIN_EVADE_S = 3.
EVADE_DIVE_DEG = (20., 40.)
EVADE_FLOOR_M = 2500.
SPLIT_S_DIVE_DEG = 50.    # plane_deg 90 (split-S into the beam) is flown as a steep dive
P_DIVE = .5
THREAT_RANGE_M = 25000.   # an RWR missile contact farther than this does not count as a threat
CHAFF_RANGE_M = 14000.    # start dropping when a missile warning is closer than this (or gives no range)
CHAFF_CADENCE = {"continuous": 1, "rhythmic": 3}  # decisions (0.5 s) between bundles
CRUISE_MACH = (1.0, 1.25)
CLIMB_DEG = 25.
MIN_CLIMB_SPEED_MPS = 250.
FLANK_DIST_M = (10000., 25000.)
SUPPRESS_TIMEOUT_S = 90.  # give up waiting for a track in range
SUPPORT_MAX_S = 15.       # longest a pilot supports a missile before turning away
SALVO_GAP_S = 6.
ROUND2_SALVO_EXTRA_P = .3
ROUND2_CRANK_P = .5
RUSH_RANGE_M = (25000., 40000.)
RUSH_P_REACT = .3
RUSH_GAP_S = 20.
RUSH_DELAY_MEDIAN_S = 3.
POPUP_RANGE_M = (30000., 45000.)
POPUP_ALT_M = (2500., 4000.)
RUSH_ALT_M = (5000., 8000.)
BOUNDARY_TURN_G = 5.        # the guard assumes a turn this hard
BOUNDARY_MARGIN_M = 3000.
BOUNDARY_FLOOR_M = 1000.
DEFAULT_RMAX_M = 20000.
HOME_ALT_M = 6000.
HOME_ORBIT_M = 12000.       # within this of the spawn point a pilot without missiles loiters
HOME_ORBIT_TURN_DEG = 3.    # heading target ahead of the current heading while loitering (about 6 deg/s)
LAST_KNOWN_S = 90.   # a lost enemy is looked for where he was heading this long (D)
PEAK_STALL_S, PEAK_STALL_M = 30., 100.   # a peak climb that gains under 100 m in 30 s has stalled
ANGLE_LEVEL_TOL_M = 500.  # vertical_mode 'angle': climbed within this of the own target (the executor levels short of it)


def wrap(deg):
    return (deg+180.) % 360.-180.


def bearing_of(dx, dy):
    return math.degrees(math.atan2(dx, dy)) % 360.


def aircraft_contacts(obs):
    """The radar contacts without the missile tracks (radar_sees_missiles; truth, as the action masks): a script never
    anchors on or shoots at a missile."""
    if not obs.radar_missiles:
        return obs.radar
    return tuple(c for c, uid in zip(obs.radar, obs.radar_missiles) if uid is None)


def direction(heading_deg, gamma_deg=0.):
    h, g = math.radians(heading_deg), math.radians(gamma_deg)
    return (math.sin(h)*math.cos(g), math.cos(h)*math.cos(g), math.sin(g))


@dataclass(frozen=True)
class PilotParams:
    archetype: str
    skill: str
    level_alt_m: float            # altitude held after the climb
    peak_alt_m: float | None      # climb first to this, then dive to level_alt (11 km pilots)
    flank_offset_deg: float
    flank_dist_m: float
    suppress: bool
    suppress_fraction: float
    after_shot: str               # 'turn_away' | 'crank'
    crank_offset_deg: float
    go_home: bool
    early_left: bool
    chaff_style: str
    recommit_clear_s: float
    second_round_m: float
    crawler_alt_m: float
    popup_range_m: float
    popup_alt_m: float
    rush_range_m: float
    rush_alt_m: float
    defend_maneuver: str          # 'beam' | 'drag'
    defend_dive: bool
    dive_deg: float
    cruise_mach: float
    p_react: float
    p_correct: float
    p_lock_react: float
    delay_median_s: float
    # Opt-in perturbations (perturb_params); the defaults reproduce the unperturbed pilot exactly.
    commit_alt_m: float | None = None       # altitude that counts as "climbed" (None: 8000 m under managed execution)
    commit_delay_s: float = 0.              # keep flanking this long after the climb before turning in
    early_recommit_s: float | None = None   # end an evasion after this long even if the RWR is not clear
    evade_inner: bool = False               # beam toward the inner side instead of the outer side
    # Population v2 (PilotV2, population_params, docs/population_spec.md). The defaults are those of a v1 pilot, and
    # describe() leaves these fields out for one, so v1 records are unchanged.
    population: str | None = None           # None (v1) | "v2"
    commit_triggers: tuple = ()             # of COMMIT_TRIGGERS, plus "lead" for a wingman; empty without a climb
    commit_never: str | None = None         # None, or "home" / "cold": never turns in
    commit_contact_m: float = 0.            # R_commit: a radar contact this close
    commit_timer_s: float = 0.              # patience after reaching the cruise altitude
    commit_clear_s: float = 0.              # T_clear: the RWR showing nothing this long (speedster: no lock / missile)
    evade_weights: tuple = ()               # ((style, weight), ...) of the correct reactions, drawn per event
    chaff_weights: tuple = ()               # ((rhythm, weight), ...), per event
    recommit_weights: tuple = ()            # ((after_clear | early | never, weight), ...), per event
    clear_wait_s: float = 0.                # centre of the per-event wait after the RWR clears (log-normal around it)
    p_outer: float = 1.                     # evade toward the outer side (else the inner one), per event
    p_switch: float = 0.                    # change the manoeuvre halfway through an evasion, per event
    shots_per_engagement: int = 1
    salvo_gap_s: float = SALVO_GAP_S
    launch_fraction: float = 1.             # opening / spam / deck shots inside this share of Rmax (aspect-adjusted)
    support_weights: tuple = ()             # ((crank | straight | drag, weight), ...), per engagement
    support_max_s: float = SUPPORT_MAX_S
    p_stt: float = 0.                       # lock STT for an engagement (radar_mode head), per engagement
    deck_offset_deg: float = 0.             # deck_notcher: approach this far off the line of sight (outer side)
    pair_offset_m: float = 0.               # pair: station abeam of the lead
    pair_back_m: float = 0.                 # pair: station behind the lead
    bait_max_s: float = 0.                  # bait: longest drag toward a teammate
    push_wait_s: float = 0.                 # pusher: shortest time after his last launch before another ripple
    # (A pusher's flank_offset_deg is signed: + flanks left, - right; left / right flyers keep it positive.)

    def describe(self):
        if self.population is None:
            return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in self.__dict__.items()
                    if k not in V2_FIELDS}
        return {k: _plain(v) for k, v in self.__dict__.items()}


def sample_params(model: dict, aircraft: str, archetype: str, skill: str, rng: random.Random) -> PilotParams:
    """Draw one pilot's parameters from data/match/top_tier.json's ``parameters`` (grades there; unspecified ones D)."""
    if archetype not in ARCHETYPES or skill not in SKILLS:
        raise ValueError(f"unknown archetype {archetype!r} or skill {skill!r}")
    prm = model["parameters"]
    choice = rng.random()
    acc, level, peak = 0., None, None
    for option in prm["climb_altitude_m"]:
        acc += option["weight"]
        if choice < acc or option is prm["climb_altitude_m"][-1]:
            if "range" in option:
                level = rng.uniform(*option["range"])
            else:
                peak = float(option["value"])
                level = rng.uniform(*prm["climb_altitude_m"][0]["range"])  # dive down to the usual level altitude
            break
    group_notes = next((g.get("notes", "") for g in model["groups"] if aircraft in g["aircraft"]), "")
    suppress_p = {"left": prm["suppression_shot_probability"]["left"], "right": prm["suppression_shot_probability"]["left"],
                  "middle": prm["suppression_shot_probability"]["middle"], "crawler": 0., "rusher": 0.}[archetype]
    after = prm["after_suppression_shot"]
    early_left = archetype == "middle" and "middle.early_left" in group_notes and rng.random() < .7
    chaff = "continuous" if rng.random() < prm["chaff"]["continuous"] else "rhythmic"
    lo, hi = prm["recommit_after_rwr_clear_s"]
    r2lo, r2hi = prm["second_round_launch_km"]
    c_lo, c_hi = prm["crawler_altitude_m"]["range"]
    f_lo, f_hi = prm["flank_offset_deg"]["range"]
    s_lo, s_hi = prm["suppression_range_fraction_of_rmax"]
    p_react = P_REACT[skill] if archetype != "rusher" or skill == "top" else RUSH_P_REACT
    return PilotParams(
        archetype, skill, level, peak, rng.uniform(f_lo, f_hi) if archetype in ("left", "right") else rng.uniform(-5., 5.),
        rng.uniform(*FLANK_DIST_M), rng.random() < suppress_p, rng.uniform(s_lo, s_hi),
        "turn_away" if rng.random() < after["turn_away"] else "crank", rng.uniform(*after["crank_offset_deg"]),
        archetype == "middle" and rng.random() < prm["middle_go_home_probability"]["value"], early_left, chaff,
        rng.uniform(lo, hi), 1000.*rng.uniform(r2lo, r2hi), rng.uniform(c_lo, c_hi), rng.uniform(*POPUP_RANGE_M),
        rng.uniform(*POPUP_ALT_M), rng.uniform(*RUSH_RANGE_M),
        rng.uniform(*RUSH_ALT_M),
        rng.choice(prm["defend"]["maneuvers"]), prm["defend"]["dive"] and rng.random() < P_DIVE,
        rng.choice(EVADE_DIVE_DEG), rng.uniform(*CRUISE_MACH), p_react, P_CORRECT[skill], P_LOCK_REACT[skill],
        RUSH_DELAY_MEDIAN_S if archetype == "rusher" and skill != "top" else DELAY_MEDIAN_S)


# Script perturbation (user 2026-10-06): real players vary more than the archetype priors, so a policy must read
# what it sees rather than a fixed rhythm. Each key is a draw range or a probability; all D.
PERTURBATION = dict(
    p=.5,                                   # share of pilots perturbed
    level_alt_m=(5000., 11500.),            # level-off altitude, and the climb counts as done there
    commit_delay_s=(0., 45.), p_commit_delay=.5,
    recommit_clear_s=(1., 25.),
    second_round_km=(12., 40.),
    p_flip_suppress=.3, suppress_fraction=(.6, 1.2),
    delay_factor=(.5, 2.),
    p_early_recommit=.2, early_recommit_s=(5., 15.),
    p_evade_inner=.15,
    # Not a perturbation: which population the scripts come from (match._population reads and removes it). None / "v1":
    # the pilots above, perturbed with the other keys; "v2": PilotV2 (docs/population_spec.md), no other key allowed.
    population=None,
)


def perturb_params(params: PilotParams, config: dict, rng: random.Random) -> PilotParams:
    """A perturbed copy of ``params`` (``config`` overrides PERTURBATION keys), or ``params`` unchanged for the pilots
    the draw leaves alone. ``rng`` is the pilot's own perturbation generator."""
    c = dict(PERTURBATION, **{k: v for k, v in config.items()})
    if rng.random() >= c["p"]:
        return params
    level = rng.uniform(*c["level_alt_m"])
    changes = dict(
        level_alt_m=level, commit_alt_m=level,
        peak_alt_m=None if params.peak_alt_m is None or params.peak_alt_m <= level+500. else params.peak_alt_m,
        commit_delay_s=rng.uniform(*c["commit_delay_s"]) if rng.random() < c["p_commit_delay"] else 0.,
        recommit_clear_s=rng.uniform(*c["recommit_clear_s"]),
        second_round_m=1000.*rng.uniform(*c["second_round_km"]),
        suppress_fraction=rng.uniform(*c["suppress_fraction"]),
        delay_median_s=params.delay_median_s*rng.uniform(*c["delay_factor"]),
        early_recommit_s=rng.uniform(*c["early_recommit_s"]) if rng.random() < c["p_early_recommit"] else None,
        evade_inner=rng.random() < c["p_evade_inner"])
    if params.archetype in ("left", "right", "middle") and rng.random() < c["p_flip_suppress"]:
        changes["suppress"] = not params.suppress
    return replace(params, **changes)


class RangeJudge:
    """Rmax of a missile from the hit-probability model: the range up to which an undefended target is hit, hot and cold.
    The lines are computed on a coarse grid of ownship altitude (every 2000 m) and speed (every 200 km/h), cached, and
    interpolated bilinearly, so a pilot's reading is continuous and costs one model evaluation per grid node ever
    visited. The target is pk.Assumption()'s default (the enemy type is unknown to the pilot). One judge per missile
    type and match."""
    ALT_STEP_M, SPEED_STEP_KMH = 2000., 200.
    ALT_RANGE_M, SPEED_RANGE_KMH = (1000., 15000.), (400., 2000.)

    def __init__(self, missile_id: str, assumption: pk.Assumption | None = None):
        self.missile_id = missile_id
        self.advisor = offense.OffenseAdvisor(missile_id, assumption)
        self._cache = {}

    def _node(self, i, j):
        found = self._cache.get((i, j))
        if found is None:
            hot, cold = self.advisor.reach_lines(i*self.ALT_STEP_M, j*self.SPEED_STEP_KMH/3.6)
            hot = DEFAULT_RMAX_M if hot is None else hot
            found = self._cache[(i, j)] = (hot, 0. if cold is None else min(cold, hot))
        return found

    def lines(self, altitude_m, speed_mps):
        """(hot, cold) Rmax in metres at this altitude and speed (bilinear between grid nodes)."""
        a = min(max(altitude_m, self.ALT_RANGE_M[0]), self.ALT_RANGE_M[1])/self.ALT_STEP_M
        v = min(max(speed_mps*3.6, self.SPEED_RANGE_KMH[0]), self.SPEED_RANGE_KMH[1])/self.SPEED_STEP_KMH
        i, j = int(a), int(v)
        fa, fv = a-i, v-j
        out = [0., 0.]
        for di, wa in ((0, 1.-fa), (1, fa)):
            for dj, wv in ((0, 1.-fv), (1, fv)):
                if wa*wv > 0.:
                    node = self._node(i+di, j+dj)
                    out[0] += wa*wv*node[0]
                    out[1] += wa*wv*node[1]
        return out[0], out[1]

    def rmax(self, altitude_m, speed_mps, aspect_deg=0.):
        """Rmax against a target with aspect ``aspect_deg`` (0 = hot, 180 = cold), interpolated between the hot and cold lines."""
        hot, cold = self.lines(altitude_m, speed_mps)
        return cold+(hot-cold)*(1.+math.cos(math.radians(aspect_deg)))/2.


def range_judge(missile_id: str, cache: dict | None = None) -> RangeJudge:
    """The RangeJudge of a missile, kept in ``cache`` (a dict the caller owns; a match builder passes one per match, so
    a run never depends on what an earlier run in the same process had cached)."""
    cache = {} if cache is None else cache
    judge = cache.get(missile_id)
    if judge is None:
        judge = cache[missile_id] = RangeJudge(missile_id)
    return judge


@dataclass
class _Reaction:
    at: float
    correct: bool


@dataclass
class _Evade:
    start: float
    plan: tuple                   # (target_deg, plane_deg, dive_deg, speed_kmh)
    side: float | None = None     # +1 clockwise (compass), -1 anticlockwise, chosen on the first heading
    clear_since: float | None = None
    last_bearing: float | None = None
    decisions: int = 0
    altitude: float | None = None   # held through the evasion unless the plan dives


class Pilot:
    """One scripted pilot. ``decide(obs)`` returns an ``engagement.Action``; ``phase`` names the current script phase."""

    def __init__(self, params: PilotParams, rng: random.Random, *, team_forward, home_xy, enemy_xy, missile_id=None,
                 map_half_m=64000., judge: RangeJudge | None = None, debug=False):
        self.p, self.rng, self.debug = params, rng, debug
        fx, fy = team_forward
        n = math.hypot(fx, fy)
        self.forward = (fx/n, fy/n)
        self.forward_deg = bearing_of(*self.forward)
        self.left = (-self.forward[1], self.forward[0])   # forward rotated counter-clockwise by 90 degrees
        self.home_xy, self.enemy_xy, self.map_half_m = tuple(home_xy), tuple(enemy_xy), map_half_m
        self.judge = judge if judge is not None else (range_judge(missile_id) if missile_id else None)
        self.outer = -1. if params.archetype == "right" else 1.   # +1 = left
        # Hard floor: crawlers live below it, everyone else keeps above 1000 m unless the script says otherwise.
        self.floor = max(5., params.crawler_alt_m/3.) if params.archetype == "crawler" else BOUNDARY_FLOOR_M
        self.phase = {"crawler": "crawl", "rusher": "rush"}.get(params.archetype, "climb")
        self.peak_done = params.peak_alt_m is None
        self.climbed_at = None
        self.seen, self.pending, self.deny_until = set(), None, -1.
        self.evade: _Evade | None = None
        self.last_threat_bearing, self.last_threat_t, self.last_threat_range = None, -1e9, None
        self.start_xy = None
        self.shot_t, self.shots_in_phase, self.phase_t = -1e9, 0, 0.
        self.support_from = None
        self.alt_history = []
        self.last_known = None   # (x, y, vx, vy, time) of the enemy last tracked or marked
        self.extra_shot = False
        self.ident = None   # set by the match builder; only the debug switch uses it
        self.crank_side = 1.
        self.guard = False
        self.anc = (self.enemy_xy[0], self.enemy_xy[1], None, "spawn")
        self.managed_execution = False
        self.estimates = {}
        self.alt_uncertainty = 6000.
        self.reach = None
        self.candidate_count, self.side_weight, self.threat_weight = 4, 2., 3.
        self.over_shoulder_p = .8 if params.skill == "top" else .3

    # -- helpers ---------------------------------------------------------------------------------------------

    def describe(self):
        return dict(self.p.describe())

    def execution_settings(self, execution=None):
        """Keyword arguments for this pilot's IntentExecutor (rl_env): ``execution`` as configured. PilotV2 puts its
        own reaction-delay median into the executor's autonomous delay path."""
        return dict(execution or {})

    def _set_phase(self, phase, now):
        if phase != self.phase:
            self.phase, self.phase_t, self.shots_in_phase, self.support_from = phase, now, 0, None

    def _anchor(self, obs):
        """(x, y, z or None, source) of the best known enemy: nearest radar track, else nearest map mark, else where he was
        last seen (carried on at his last velocity for LAST_KNOWN_S), else the spawn.
        With ``debug`` and a truth-enabled engagement the nearest real enemy instead (a debugging switch)."""
        own = obs.own
        px, py, _ = own.position
        if self.debug and obs.truth is not None and self.ident is not None:
            me = obs.truth.planes[self.ident]
            near = min(obs.truth.enemies(me), key=lambda q: math.dist(q.own.position, own.position), default=None)
            if near is not None:
                return (near.own.position[0], near.own.position[1], near.own.position[2], "truth")
        now = obs.time_s
        best, best_d, best_v = None, math.inf, (0., 0.)
        for c in aircraft_contacts(obs):
            if c.position is not None:
                d = math.hypot(c.position[0]-px, c.position[1]-py)
                if d < best_d:
                    best, best_d = (c.position[0], c.position[1], c.position[2], "radar"), d
                    best_v = (c.velocity[0], c.velocity[1]) if c.velocity is not None else (0., 0.)
        if best is None:
            for m in obs.marks:
                if not m.friend and now-m.time_s <= 30.:
                    d = math.hypot(m.x-px, m.y-py)
                    if d < best_d:
                        best, best_d = (m.x, m.y, m.z, "mark"), d
        if best is not None:
            self.last_known = (best[0], best[1], best_v[0], best_v[1], now)
            return best
        # Nobody sees him (he flew past, or the radar lost him): where he was, carried on at his last velocity, for a while.
        if self.last_known is not None and now-self.last_known[4] <= LAST_KNOWN_S:
            x, y, vx, vy, t = self.last_known
            return (x+vx*(now-t), y+vy*(now-t), None, "memory")
        return (self.enemy_xy[0], self.enemy_xy[1], None, "spawn")

    def _lateral(self, own):
        return ((own.position[0]-self.start_xy[0])*self.left[0]+(own.position[1]-self.start_xy[1])*self.left[1])

    def _cruise_speed(self, altitude):
        return self.p.cruise_mach*atmosphere(max(0., min(19000., altitude)))[1]

    def _fly(self, heading_deg, *, altitude=None, gamma=None, speed=None, floor=None, brake=False, throttle=None):
        floor = self.floor if floor is None else floor
        if gamma is not None:
            return FlightCommand(direction=direction(heading_deg, gamma), speed_mps=speed, throttle_percent=throttle,
                                 airbrake_allowed=brake, floor_m=floor)
        return FlightCommand(heading_deg=heading_deg, altitude_m=altitude, speed_mps=speed, throttle_percent=throttle,
                             airbrake_allowed=brake, floor_m=floor, min_speed_mps=MIN_CLIMB_SPEED_MPS,
                             max_climb_deg=CLIMB_DEG)

    # -- threats and reaction ---------------------------------------------------------------------------------

    def _threats(self, obs, now):
        """Update the threat picture and schedule a reaction to a new warning. Returns True while any warning is shown."""
        present, bearing, nearest = False, None, math.inf
        new = []
        for c in obs.rwr:
            if c.missile_warning:
                if c.range_m is not None and c.range_m > THREAT_RANGE_M:
                    continue   # far and still shown: lost track or spent, no longer a threat (D)
                present = True
                r = c.range_m if c.range_m is not None else 1e5
                if r < nearest:
                    nearest, bearing = r, c.bearing_deg
                    self.last_threat_range = c.range_m
                if ("r", c.contact_id) not in self.seen:
                    new.append((("r", c.contact_id), "missile"))
            elif c.tracking:
                present = True
                if bearing is None:
                    bearing = c.bearing_deg
                if ("l", c.contact_id) not in self.seen:
                    new.append((("l", c.contact_id), "lock"))
        for s in obs.maw:
            present = True
            if bearing is None:
                bearing = s.bearing_deg
            if ("m", s.ref) not in self.seen:
                new.append((("m", s.ref), "maw"))
        for s in obs.flames:
            present = True
            if bearing is None:
                bearing = s.bearing_deg
            if ("f", s.ref) not in self.seen:
                new.append((("f", s.ref), "flame"))
        if bearing is not None:
            self.last_threat_bearing, self.last_threat_t = bearing, now
        for key, kind in new:
            self.seen.add(key)
            if self.pending is not None or self.phase == "evade" or now < self.deny_until:
                continue
            if kind == "lock" and self.rng.random() >= self.p.p_lock_react:
                continue
            if self.rng.random() >= self.p.p_react:
                self.deny_until = now+DENIED_S
                continue
            delay = 0. if self.managed_execution else self.rng.lognormvariate(math.log(self.p.delay_median_s), DELAY_SIGMA)
            self.pending = _Reaction(now+delay, self.rng.random() < self.p.p_correct)
        return present

    def _plan(self, correct):
        p = self.p
        if correct:
            dive = p.dive_deg if p.defend_dive else 0.
            return (90. if p.defend_maneuver == "beam" else 0., 0., dive, 1500.)
        return REPERTOIRE[self.rng.randrange(len(REPERTOIRE))]

    def _start_evade(self, now, correct, reference=None):
        self.evade = _Evade(now, self._plan(correct), last_bearing=reference)
        self.pending = None
        self._set_phase("evade", now)

    def _evade_command(self, obs, now, present):
        e, own = self.evade, obs.own
        target_deg, plane_deg, dive_deg, speed_kmh = e.plan
        e.decisions += 1
        if e.altitude is None:
            e.altitude = own.altitude_m
        bearing = self.last_threat_bearing if (self.last_threat_bearing is not None and now-self.last_threat_t < 8.) \
            else e.last_bearing
        if bearing is None:
            ax, ay = self.anc[0], self.anc[1]
            bearing = bearing_of(ax-own.position[0], ay-own.position[1])
        e.last_bearing = bearing
        away = (bearing+180.) % 360.
        if target_deg == 0.:
            heading = away
            if self.p.go_home and self.p.archetype == "middle":
                heading = (self.forward_deg+180.) % 360.
        else:
            if e.side is None:
                # Beam to the outer side: of the two three-nine headings take the one more toward the outer direction;
                # when they differ little (the threat is on the outer-inner axis) the one nearer the current heading.
                # The side is kept for the whole evasion, so the heading follows the threat without flipping.
                side_x, side_y = self.left[0]*self.outer, self.left[1]*self.outer
                score = []
                for sign in (1., -1.):
                    h = math.radians(bearing+sign*90.)
                    turn = math.cos(math.radians(wrap(bearing+sign*90.-own.heading_deg)))
                    score.append(math.sin(h)*side_x+math.cos(h)*side_y+.5*turn)
                e.side = 1. if score[0] >= score[1] else -1.
                if self.p.evade_inner:
                    e.side = -e.side
            heading = (bearing+e.side*90.) % 360.
        gamma = -(SPLIT_S_DIVE_DEG if plane_deg >= 90. else dive_deg)
        if self.p.archetype == "crawler":
            gamma = 0.   # a crawler defends on the deck
        floor = EVADE_FLOOR_M if gamma < 0 else self.floor
        if gamma < 0:
            cmd = self._fly(heading, gamma=gamma, floor=floor)
        elif speed_kmh < FULL_POWER_KMH:
            cmd = self._fly(heading, altitude=e.altitude, speed=speed_kmh/3.6, brake=True)
        else:
            cmd = self._fly(heading, altitude=e.altitude)
        chaff = 0
        close = any(c.missile_warning and c.age_s <= 1. and (c.range_m is None or c.range_m <= CHAFF_RANGE_M)
                    for c in obs.rwr)
        if close and obs.own.chaff > 0:
            if e.decisions % CHAFF_CADENCE[self.p.chaff_style] == 0:
                chaff = 1
        # Phase end: warnings clear for the pilot's wait.
        if present:
            e.clear_since = None
        elif e.clear_since is None:
            e.clear_since = now
        done = e.clear_since is not None and now-e.clear_since >= self.p.recommit_clear_s and now-e.start >= MIN_EVADE_S
        if self.p.early_recommit_s is not None and now-e.start >= max(MIN_EVADE_S, self.p.early_recommit_s):
            done = True                    # perturbed pilot: turns back before the RWR is clear
        return cmd, chaff, done

    # -- shooting ---------------------------------------------------------------------------------------------

    def _tracks(self, obs):
        out = []
        for c in aircraft_contacts(obs):
            if c.track_id is not None or c.kind == "stt":
                if c.range_m is not None:
                    out.append(c)
        return sorted(out, key=lambda c: c.range_m)

    def _aspect(self, obs, c):
        """Aspect of the tracked target seen from the pilot: 0 = flying straight at him (hot), 180 = flying away."""
        if c.velocity is None or c.position is None:
            return 0.
        own = obs.own.position
        rx, ry, rz = own[0]-c.position[0], own[1]-c.position[1], own[2]-c.position[2]
        rn = math.sqrt(rx*rx+ry*ry+rz*rz)
        vn = math.sqrt(sum(v*v for v in c.velocity))
        if rn < 1. or vn < 1.:
            return 0.
        cos = (rx*c.velocity[0]+ry*c.velocity[1]+rz*c.velocity[2])/(rn*vn)
        return math.degrees(math.acos(max(-1., min(1., cos))))

    def _shot(self, obs, now, max_range_m=None, fraction=None, gap=SALVO_GAP_S, accept=None):
        """Track id to fire at, or None. ``fraction``: inside that share of Rmax (aspect-adjusted); ``max_range_m``: inside that range.
        ``accept`` (population v2's pusher): a further test of each track."""
        own = obs.own
        if own.missiles <= 0 or now-self.shot_t < gap:
            return None
        eligible = []
        for c in self._tracks(obs):
            from .intent import launch_limit
            limit_angle = launch_limit(own.aircraft, own.missile_id)
            if abs(c.azimuth_deg) > (limit_angle if self.managed_execution else 60.):
                continue
            if accept is not None and not accept(c):
                continue
            if abs(c.azimuth_deg) > 60. and self.rng.random() >= self.over_shoulder_p:
                continue
            limit = max_range_m
            if fraction is not None and self.judge is not None:
                reach = None if self.reach is None else self.reach.rmax(own.altitude_m,own.speed_mps,self._aspect(obs,c),
                             0. if c.position is None else c.position[2]-own.altitude_m)
                limit = fraction*(reach if reach is not None else self.judge.rmax(own.altitude_m, own.speed_mps, self._aspect(obs, c)))
            if limit is not None and c.range_m <= limit:
                eligible.append(c)
        if not eligible:
            return None
        # Pilots do not all pick the nearest track (that would put every missile on one aircraft): by 1 / range.
        if self.managed_execution:
            threatening=[c.bearing_deg for c in obs.rwr if c.tracking or c.missile_warning]
            def score(c):
                side=1. if wrap(c.bearing_deg-self.forward_deg)*(-self.outer)>=0. else 0.
                threat=1. if any(abs(wrap(c.bearing_deg-b))<15. for b in threatening) else 0.
                return (1.+self.side_weight*side+self.threat_weight*threat)/max(c.range_m,1000.)
            eligible=sorted(eligible,key=score,reverse=True)[:max(1,int(self.candidate_count))]
            weights=[score(c) for c in eligible]
        else:
            weights = [1./max(c.range_m, 1000.) for c in eligible]
        x = self.rng.random()*sum(weights)
        for c, w in zip(eligible, weights):
            x -= w
            if x <= 0.:
                break
        return STT_TRACK if c.kind == "stt" else c.track_id

    def _fired(self, now):
        self.shot_t = now
        self.shots_in_phase += 1

    # -- radar ------------------------------------------------------------------------------------------------

    def _radar(self, obs):
        own = obs.own
        ax, ay, az, _ = self.anc
        dx, dy = ax-own.position[0], ay-own.position[1]
        bearing = bearing_of(dx, dy)
        el = 0.
        if az is not None:
            el = max(-25., min(25., math.degrees(math.atan2(az-own.position[2], max(1., math.hypot(dx, dy))))))
        if self.managed_execution and self.alt_uncertainty>500.:
            distance=max(1000.,math.hypot(dx,dy))
            span=min(20.,math.degrees(math.atan2(self.alt_uncertainty,distance)))
            el=max(-20.,min(20.,el+span*math.sin(obs.time_s*math.pi/4.)))
        return RadarCommand("tws", 0, wrap(bearing-own.heading_deg), el)

    # -- the decision -----------------------------------------------------------------------------------------

    def decide(self, obs: Observation) -> Action:
        p, now, own = self.p, obs.time_s, obs.own
        if self.start_xy is None:
            self.start_xy = (own.position[0], own.position[1])
        present = self._threats(obs, now)
        self.anc = ax, ay, az, _ = self._estimated_anchor(obs) if self.managed_execution else self._anchor(obs)
        to_enemy = bearing_of(ax-own.position[0], ay-own.position[1])
        # Out of missiles, or bingo fuel (opt-in fuel model; False without it): go home, stay home.
        spent = own.missiles <= 0 or getattr(own, "bingo", False)
        if self.phase == "home" and not spent:
            # Rearmed at the airfield (opt-in): leaving "home" takes off; back to the fight as from the opening, but
            # straight at the enemy (no second flank climb or suppression shot).
            self._set_phase({"crawler": "crawl", "rusher": "rush"}.get(p.archetype, "advance"), now)
        if spent and self.phase not in ("home", "evade") and self.support_from is None:
            self._set_phase("home", now)
        if self.pending is not None and now >= self.pending.at and self.phase != "evade":
            self._start_evade(now, self.pending.correct)
        fire, chaff = None, 0
        phase = self.phase
        if phase == "evade":
            cmd, chaff, done = self._evade_command(obs, now, present)
            if done:
                self.evade = None
                self.extra_shot = False
                self._set_phase("home" if spent else {"crawler": "crawl", "rusher": "rush"}.get(
                    p.archetype, "recommit"), now)
        elif phase == "climb":
            cmd = self._climb(obs, now)
        elif phase == "suppress":
            cmd, fire = self._suppress(obs, now, to_enemy)
        elif phase in ("advance", "round2", "recommit"):
            cmd, fire = self._engage(obs, now, to_enemy)
        elif phase in ("crawl", "popup"):
            cmd, fire = self._crawler(obs, now, to_enemy)
        elif phase == "rush":
            cmd, fire = self._rush(obs, now, to_enemy)
        else:  # home: no missiles left; fly to the own spawn point and loiter there in a wide turn
            hx, hy = self.home_xy
            dx, dy = hx-own.position[0], hy-own.position[1]
            heading = bearing_of(dx, dy) if math.hypot(dx, dy) > HOME_ORBIT_M else (own.heading_deg+HOME_ORBIT_TURN_DEG) % 360.
            cmd = self._fly(heading, altitude=max(own.altitude_m, HOME_ALT_M), speed=self._cruise_speed(own.altitude_m))
        cmd = self._guard(obs, cmd, phase)
        if self.managed_execution:
            # Unload before a shot; outside defence do not spend sub-corner energy on hard turns.
            if fire is not None:
                cmd=replace(cmd,max_load=min(cmd.max_load,2.),throttle_percent=110.,speed_mps=None)
            elif phase != "evade":
                from .rl_observation import energy_state
                from .flight import aircraft_model
                if not hasattr(self,"performance_model"):
                    self.performance_model=aircraft_model(own.aircraft,mass_factor=1.3)
                model=self.performance_model
                en,ps,ratio,*_=energy_state(own,model,model.structure.limits(model.mass))
                if ratio is not None and ratio<1.:
                    cmd=replace(cmd,max_load=min(cmd.max_load,3.),throttle_percent=110.,speed_mps=None)
        if fire is not None:
            self._fired(now)
        return Action(cmd, self._radar(obs), fire, chaff)

    def propose(self, obs, entities):
        """Public script intent, before delay/noise/rejection; same route as policy actions."""
        from .intent import from_flight_action
        self.managed_execution=True
        # The executor's vertical mode ('altitude' options or 'angle': any altitude target is flown), as
        # from_flight_action reads it from the entity list; _climb judges the climb by it.
        self.vertical_mode=getattr(entities,"vertical_mode","altitude")
        action=self.decide(obs)
        return from_flight_action(action,obs,entities,phase=self.phase,home_xy=self.home_xy,guard=self.guard)

    def _estimated_anchor(self, obs):
        now,own=obs.time_s,obs.own
        # Keys are observed mark ids, never the simulator's aircraft slots.
        for c in aircraft_contacts(obs):
            if c.position is not None:
                key=c.mark_id if getattr(c,"mark_id",None) is not None else ("track",c.track_id)
                self.estimates[key]=(c.position,c.velocity or (0.,0.,0.),now,300.+100.*c.age_s)
        for m in obs.marks:
            if m.friend:
                continue
            old=self.estimates.get(m.mark_id)
            if old:
                pos,v,t,u=old
                age=now-t
                z=pos[2]+v[2]*age
                self.estimates[m.mark_id]=((m.x,m.y,z),v,now,u+100.*age)
            else:
                self.estimates[m.mark_id]=((m.x,m.y,own.altitude_m),(0.,0.,0.),now,6000.)
        for cue in (*obs.visual,*obs.contrails):
            old=self.estimates.get(cue.ref)
            distance=math.hypot(old[0][0]-own.position[0],old[0][1]-own.position[1]) if old else math.dist(own.position[:2],self.enemy_xy)
            a=math.radians(cue.bearing_deg)
            z=own.altitude_m+distance*math.tan(math.radians(cue.elevation_deg))
            pos=(own.position[0]+distance*math.sin(a),own.position[1]+distance*math.cos(a),z)
            self.estimates[cue.ref]=(pos,old[1] if old else (0.,0.,0.),now,500.)
        current=[]
        for key,(pos,v,t,u) in self.estimates.items():
            age=now-t
            if age<=LAST_KNOWN_S:
                projected=tuple(p+x*age for p,x in zip(pos,v))
                current.append((math.dist(projected[:2],own.position[:2]),str(key),projected,u+100.*age))
        if not current:
            if obs.rwr:
                c=min(obs.rwr,key=lambda c:(not c.missile_warning,not c.tracking,c.age_s))
                a=math.radians(c.bearing_deg)
                distance=c.range_m or 50000.
                self.alt_uncertainty=6000.
                return own.position[0]+distance*math.sin(a),own.position[1]+distance*math.cos(a),None,"rwr"
            return self._anchor(obs)
        _,_,pos,u=min(current,key=lambda c:c[:2])
        x,y,z=pos
        bearing=bearing_of(x-own.position[0],y-own.position[1])
        # An unassociated RWR constrains only azimuth; use it only when near the estimate.
        nearby=[c for c in obs.rwr if abs(wrap(c.bearing_deg-bearing))<20.]
        if nearby:
            b=min(nearby,key=lambda c:c.age_s).bearing_deg
            distance=math.hypot(x-own.position[0],y-own.position[1])
            a=math.radians(b)
            x,y=own.position[0]+distance*math.sin(a),own.position[1]+distance*math.cos(a)
        self.alt_uncertainty=u
        return x,y,z,"estimate"

    # -- phases -----------------------------------------------------------------------------------------------

    def _climb(self, obs, now):
        p, own = self.p, obs.own
        side = {"left": 1., "right": -1., "middle": 0.}[p.archetype]
        heading = (self.forward_deg-side*p.flank_offset_deg if side else self.forward_deg+p.flank_offset_deg) % 360.
        target = p.level_alt_m if self.peak_done else p.peak_alt_m
        self.alt_history.append((now, own.altitude_m))
        while self.alt_history[0][0] < now-PEAK_STALL_S:
            self.alt_history.pop(0)
        # vertical_mode 'angle' (opt-in): the executor flies the pilot's own altitude target and levels 150-300 m short
        # of it, so the climb is judged against that target with ANGLE_LEVEL_TOL_M, and a stalled climb counts as done
        # for every pilot. The default altitude options keep the original rule (8000 +- 400 m when managed).
        angle = self.managed_execution and getattr(self, "vertical_mode", "altitude") == "angle"
        if not self.peak_done and (own.altitude_m >= p.peak_alt_m-(ANGLE_LEVEL_TOL_M if angle else 150.) or (
                now-self.alt_history[0][0] >= PEAK_STALL_S-1. and own.altitude_m-self.alt_history[0][1] < PEAK_STALL_M)):
            # At the peak, or the climb has stalled (this aircraft cannot get there): level off at the usual altitude.
            self.peak_done, target = True, p.level_alt_m
        if angle:
            level = p.commit_alt_m if p.commit_alt_m is not None else p.level_alt_m
            reached = self.peak_done and abs(own.altitude_m-level) < ANGLE_LEVEL_TOL_M
        else:
            level = p.commit_alt_m if p.commit_alt_m is not None else (8000. if self.managed_execution else p.level_alt_m)
            reached = self.peak_done and abs(own.altitude_m-level) < 400.
        if (p.commit_alt_m is not None or (angle and own.altitude_m > .5*level)) and self.peak_done and not reached and \
                now-self.alt_history[0][0] >= PEAK_STALL_S-1. and abs(own.altitude_m-self.alt_history[0][1]) < PEAK_STALL_M:
            # Perturbed level between the executor's altitude steps (e.g. 9 km while the climb holds 8 km): the climb
            # has stopped, so it counts as done where it is.
            reached = True
        drifted = side == 0. or abs(self._lateral(own)) >= p.flank_dist_m
        if reached and drifted:
            if self.climbed_at is None:
                self.climbed_at = now
            if now-self.climbed_at >= p.commit_delay_s:
                self._set_phase("suppress" if p.suppress and own.missiles > 0 else "advance", now)
                return self._fly(self.forward_deg, altitude=p.level_alt_m, speed=self._cruise_speed(own.altitude_m))
            target = p.level_alt_m          # perturbed pilot: hold the flank a while longer before turning in
        speed = None if own.altitude_m < target-300. else self._cruise_speed(own.altitude_m)
        return self._fly(heading, altitude=target, speed=speed)

    def _supporting(self, obs, now):
        """True while the pilot keeps supporting his missile (until its seeker is on, the datalink dropped, it ended or the time ran out)."""
        active = any(s.active or not s.datalink for s in obs.shots)
        return not (active or now-self.support_from > SUPPORT_MAX_S or not obs.shots)

    def _suppress(self, obs, now, to_enemy):
        """Turn toward the enemy, fire at the Rmax fraction, then crank (and support) or turn away."""
        p, own = self.p, obs.own
        speed = self._cruise_speed(own.altitude_m)
        straight = self._fly(to_enemy, altitude=p.level_alt_m, speed=speed)
        if self.support_from is None:
            fire = self._shot(obs, now, fraction=p.suppress_fraction)
            if fire is not None:
                if p.after_shot == "turn_away" or p.early_left:
                    self._start_evade(now, correct=True, reference=to_enemy)
                else:
                    self.support_from, self.crank_side = now, self.outer
                return straight, fire
            if now-self.phase_t > SUPPRESS_TIMEOUT_S:
                self._set_phase("advance", now)
            return straight, None
        if not self._supporting(obs, now):
            self._start_evade(now, correct=True, reference=to_enemy)
            return straight, None
        # Crank 30-50 degrees off the line of sight, to the outer side.
        return self._fly((to_enemy-self.crank_side*p.crank_offset_deg) % 360., altitude=p.level_alt_m, speed=speed), None

    def _engage(self, obs, now, to_enemy):
        """advance / recommit / round2: fly at the enemy, fire inside the second-round range, support, then evade."""
        p, own = self.p, obs.own
        speed = self._cruise_speed(own.altitude_m)
        straight = self._fly(to_enemy, altitude=p.level_alt_m, speed=speed)
        if self.support_from is not None:
            fire = None
            if self.extra_shot and now-self.shot_t >= SALVO_GAP_S:
                fire = self._shot(obs, now, max_range_m=p.second_round_m)
                if fire is not None:
                    self.extra_shot, self.support_from = False, now
            if fire is None and not self._supporting(obs, now):
                self._start_evade(now, correct=True, reference=to_enemy)
            return straight, fire
        if self.phase == "recommit" and abs(wrap(to_enemy-own.heading_deg)) < 30.:
            self._set_phase("round2", now)
        fire = self._shot(obs, now, max_range_m=p.second_round_m)
        if fire is not None:
            self.support_from = now
            self.extra_shot = own.missiles > 1 and self.rng.random() < ROUND2_SALVO_EXTRA_P
        return straight, fire

    def _crawler(self, obs, now, to_enemy):
        p, own = self.p, obs.own
        distance = math.hypot(self.anc[0]-own.position[0], self.anc[1]-own.position[1])
        speed = self._cruise_speed(p.crawler_alt_m)
        if self.phase == "crawl":
            if distance <= p.popup_range_m and own.missiles > 0:
                self._set_phase("popup", now)
            return self._fly(to_enemy, altitude=p.crawler_alt_m, speed=speed), None
        pop = self._fly(to_enemy, altitude=p.popup_alt_m, speed=speed)
        if self.support_from is not None:
            if not self._supporting(obs, now):
                self._start_evade(now, correct=True, reference=to_enemy)
            return pop, None
        fire = self._shot(obs, now, max_range_m=p.second_round_m)
        if fire is not None:
            self.support_from = now
        return pop, fire

    def _rush(self, obs, now, to_enemy):
        p, own = self.p, obs.own
        fire = self._shot(obs, now, max_range_m=p.rush_range_m, gap=RUSH_GAP_S)
        speed = None if own.altitude_m < p.rush_alt_m-300. else self._cruise_speed(own.altitude_m)
        return self._fly(to_enemy, altitude=p.rush_alt_m, speed=speed), fire

    # -- map boundary -----------------------------------------------------------------------------------------

    def _guard(self, obs, cmd, phase):
        """Turn back toward the map centre when the heading runs out of the map: on an axis moving outward, closer to
        the edge than the stopping distance of a 5 g turn plus a margin (the boundary is known to the pilot)."""
        own = obs.own
        x, y, _ = own.position
        vx, vy = own.velocity[0], own.velocity[1]
        half = self.map_half_m
        self.guard = False
        for pos, vel in ((x, vx), (y, vy)):
            out = vel if pos > 0 else -vel
            margin=out*out/(2.*BOUNDARY_TURN_G*9.80665)+BOUNDARY_MARGIN_M
            if self.managed_execution:
                margin=max(margin,own.speed_mps**2/(2.*9.80665)+own.speed_mps*7.+BOUNDARY_MARGIN_M)
            if out > 1. and half-abs(pos) < margin:
                self.guard = True
        if not self.guard:
            return cmd
        heading = bearing_of(-x, -y)
        if self.managed_execution:
            edge_x=half-abs(x)<half-abs(y)
            if edge_x:
                dy=cmd.direction[1] if cmd.direction is not None else math.cos(math.radians(cmd.heading_deg or own.heading_deg))
                sy=1. if dy>=0. else -1.
                if half-abs(y)<BOUNDARY_MARGIN_M*2.:
                    sy=-1. if y>0 else 1.
                heading=bearing_of(-(.35 if x>0 else -.35),sy)
            else:
                dx=cmd.direction[0] if cmd.direction is not None else math.sin(math.radians(cmd.heading_deg or own.heading_deg))
                sx=1. if dx>=0. else -1.
                if half-abs(x)<BOUNDARY_MARGIN_M*2.:
                    sx=-1. if x>0 else 1.
                heading=bearing_of(sx,-(.35 if y>0 else -.35))
        return FlightCommand(heading_deg=heading, altitude_m=own.altitude_m, floor_m=self.floor,
                             max_load=cmd.max_load, speed_mps=cmd.speed_mps, throttle_percent=cmd.throttle_percent,
                             airbrake_allowed=cmd.airbrake_allowed, min_speed_mps=cmd.min_speed_mps)


# -- population v2 (docs/population_spec.md) --------------------------------------------------------------------------
#
# Opt-in. Every per-pilot draw lands in PilotParams (so in the replay header); every per-event draw (each commit,
# attack plan and evasion) is appended to PilotV2.events. Values are grade D unless the user's C-grade facts of
# docs/rl_design.md section 2 say otherwise; those are kept: the climb altitudes, the suppression share and range, the
# crank offset, chaff styles, the 3-10 s recommit wait (now the centre of a per-event draw), the 20-30 km second round
# (now 75 % of pilots), the group priors of the five old archetypes.

POPULATIONS = (None, "v1", "v2")
NEW_ARCHETYPES = ("spammer", "speedster", "deck_notcher", "pair", "bait", "pusher")
ARCHETYPES_V2 = ARCHETYPES+NEW_ARCHETYPES
TEAM_ARCHETYPES = ("pair", "bait")          # need a teammate: never drawn for a one-aircraft team
BASE_ARCHETYPE = dict(spammer="middle", speedster="middle", deck_notcher="crawler", pair="left", bait="middle",
                      pusher="left")
DECK_ARCHETYPES = ("crawler", "deck_notcher")
NO_CLIMB = ("crawler", "deck_notcher", "rusher")      # no climb phase, so no commit
ATTACK_PHASES = ("suppress", "advance", "recommit", "round2", "spam")
COMMIT_TRIGGERS = ("contact", "team_launch", "rwr_clear")   # drawn; every climbing pilot also has "timer" (patience)
# (target_deg from away-from-threat, plane_deg, dive_deg, speed_kmh) as in REPERTOIRE. REPERTOIRE_STYLES names the
# REPERTOIRE entries in their order (a normal pilot's wrong pick), so every flown evasion has a name.
STYLE_PLANS = dict(beam=(90., 0., 0., 1500.), drag=(0., 0., 0., 1500.), dive20=(90., 0., 20., 1500.),
                   dive40=(90., 0., 40., 1500.), split_s=(90., 90., 0., 1500.), drag_dive=(0., 0., 20., 1500.),
                   beam_slow=(90., 0., 0., 600.), bait=(0., 0., 0., 1500.))
REPERTOIRE_STYLES = ("beam", "dive20", "dive40", "split_s", "drag", "drag_dive", "beam_slow")
CHAFF_RHYTHMS = ("continuous", "paced", "none", "late")
RECOMMIT_MODES = ("after_clear", "early", "never")
SUPPORT_MODES = ("crank", "straight", "drag")
FLANK_LEVELS_DEG = (-90., -40., 0., 40., 90.)  # heading offsets off a reference the intent heads fly (from_flight_action)
LAUNCH_GRACE_S = 6.       # the executor has not launched the missile yet: support does not end for want of it
LATE_NO_RANGE_S = 6.      # "late" chaff on a warning without range: after this much of the evasion
CLOSE_CONTACT_M = 30000.  # a contact this close commits a pilot with the contact trigger even before his climb is done
NEVER_AFTER_S = 90.       # a never-commit pilot who cannot reach his altitude goes cold after this
COLD_ORBIT_M = 12000.     # a cold pilot circles inside this of his hold point
POPUP_GIVE_UP_S = 60.     # a popped-up crawler with nothing to shoot goes back down
PUSH_LEVELS_DEG = (0., 40., 90.)   # target off the nose the intent heads can hold (abs of FLANK_LEVELS_DEG)
PUSH_SIDE_HYST_DEG = 10.  # a pusher keeps the target on its side unless it is this far off the nose or the tail
PUSH_LEVEL_HYST_DEG = 25. # ... and at his held offset while the target stays within this of it

# Defaults of the match model's "population" -> "v2" section (population_settings merges a file's values over them).
POPULATION_V2 = {
    "extra_prior": {"spammer": .05, "speedster": .04, "deck_notcher": .04, "pair": .06, "bait": .03, "pusher": .05},
    "extra_concentration": 20.,
    "commit": {
        "triggers": {
            "default": {"contact": .5, "team_launch": .4, "rwr_clear": .3},
            "middle": {"contact": .6, "team_launch": .3, "rwr_clear": .25},
            "spammer": {"contact": .8, "team_launch": .2, "rwr_clear": .1},
            "pair": {"contact": .3, "team_launch": .6, "rwr_clear": .1},
        },
        "never": {"default": .04, "middle": .06, "spammer": .02, "speedster": 0., "pair": 0., "bait": 0., "pusher": 0.},
        "never_mode": {"home": .5, "cold": .5},
        "contact_km": [30., 70.],
        "timer_s": {"median": 104., "sigma": .55, "clip": [20., 420.]},
        "clear_s": [5., 30.],
        "speedster_clear_s": [5., 20.],
        "hesitation_s": [0., 10.],
        "min_alt_fraction": .7,
        "team_seen_km": 25.,
        "flank_offset_deg": [20., 60.],
        "flank_cycle_s": [16., 30.],
    },
    "evasion": {
        "styles": {
            "default": {"beam": .3, "drag": .2, "dive20": .15, "dive40": .1, "split_s": .1, "ignore": .05},
            "crawler": {"beam": .7, "drag": .25, "ignore": .05},
            "deck_notcher": {"beam": .75, "drag": .2, "ignore": .05},
            "speedster": {"drag": 1.},
            "rusher": {"beam": .3, "drag": .2, "dive20": .1, "ignore": .4},
            "bait": {"bait": .7, "beam": .15, "drag": .15},
        },
        "concentration": 6.,
        "chaff": {"default": {"continuous": .35, "paced": .35, "none": .1, "late": .2}},
        "chaff_start_km": [8., 20.],
        "chaff_paced_s": [1., 3.],
        "chaff_late_km": [3., 7.],
        "min_evade_s": [3., 20.],
        "recommit": {"default": {"after_clear": .7, "early": .2, "never": .1}, "speedster": {"after_clear": 1.},
                     "bait": {"after_clear": .9, "never": .1}},
        "clear_wait_sigma": .6,
        "clear_wait_clip_s": [.5, 40.],
        "early_s": [4., 15.],
        "p_outer": [.6, .95],
        "p_switch": [0., .2],
        "switch_s": [3., 10.],
        "split_s_alt_m": [200., 1500.],
    },
    "doctrine": {
        "shots": {"default": {"1": .5, "2": .35, "3": .15}, "spammer": {"2": .5, "3": .5},
                  "pusher": {"2": .6, "3": .4}},
        "salvo_gap_s": {"default": [2., 8.], "spammer": [2., 5.], "pusher": [3., 6.]},
        "launch_fraction": {"default": [.5, 1.1], "spammer": [.9, 1.1], "speedster": [.7, 1.], "pusher": [.8, 1.]},
        "support": {"default": {"crank": .4, "straight": .35, "drag": .25}, "spammer": {"drag": 1.},
                    "speedster": {"straight": .5, "drag": .5}, "pusher": {"straight": 1.}},
        "support_max_s": [8., 25.],
        "p_stt": [0., .4],
        "p_second_round_wide": .25,
        "second_round_wide_km": [12., 40.],
    },
    "deck": {"deck_offset_deg": [0., 40.]},
    "speedster": {"level_alt_m": [9500., 11500.]},
    "team": {"pair_offset_km": [3., 8.], "pair_back_km": [0., 2.], "bait_friend_km": 40., "bait_max_s": [60., 120.]},
    # pusher (docs/analysis_offboresight_push.md): re-push no sooner than wait_s (per pilot) after the last launch, on a
    # target that turned back hot or left the beam (beam_deg either side of it); hold the target margin_deg inside the
    # launch limit; beyond wide_km fire only within far_deg of the nose; pursue at most pursue_max_s after the last
    # launch while own missiles fly.
    "push": {"wait_s": [8., 16.], "margin_deg": 10., "beam_deg": 20., "pursue_max_s": 60., "wide_km": 16.,
             "far_deg": 45.},
    "skill": {"delay_median_s": {"normal": [1.5, 2.5], "top": [.8, 1.5]}, "rusher_delay_factor": [1.3, 1.8]},
}
V2_FIELDS = frozenset((
    "population", "commit_triggers", "commit_never", "commit_contact_m", "commit_timer_s", "commit_clear_s",
    "evade_weights", "chaff_weights", "recommit_weights", "clear_wait_s", "p_outer", "p_switch",
    "shots_per_engagement", "salvo_gap_s", "launch_fraction", "support_weights", "support_max_s", "p_stt",
    "deck_offset_deg", "pair_offset_m", "pair_back_m", "bait_max_s", "push_wait_s"))


def _plain(v):
    """A record-friendly PilotParams value: floats rounded, ((key, weight), ...) as a dict, other tuples as lists."""
    if isinstance(v, float):
        return round(v, 3)
    if isinstance(v, tuple):
        if v and all(isinstance(x, tuple) and len(x) == 2 and isinstance(x[0], str) for x in v):
            return {k: round(w, 3) for k, w in v}
        return [_plain(x) for x in v]
    return v


def _uniform(rng, bounds):
    return rng.uniform(float(bounds[0]), float(bounds[1]))


def _pick(rng, weights):
    """A key of ``weights`` (a dict or (key, weight) pairs, in order) with probability proportional to its weight;
    None when there is none."""
    items = list(weights.items()) if isinstance(weights, dict) else list(weights)
    total = sum(w for _, w in items)
    if total <= 0.:
        return items[0][0] if items else None
    x = rng.random()*total
    for key, w in items:
        x -= w
        if x < 0.:
            return key
    return items[-1][0]


def _by_archetype(table, archetype):
    """``table[archetype]``, else the entry of its base archetype (BASE_ARCHETYPE), else ``table["default"]``."""
    if archetype in table:
        return table[archetype]
    return table.get(BASE_ARCHETYPE.get(archetype), table["default"])


def _propensity(rng, base, concentration):
    """One pilot's weights around ``base`` (a Dirichlet with mean ``base``), as ((key, weight), ...) in base order."""
    draws = [(k, rng.gammavariate(concentration*w, 1.) if w > 0. else 0.) for k, w in base.items()]
    total = sum(w for _, w in draws) or 1.
    return tuple((k, w/total) for k, w in draws)


def population_settings(model=None):
    """POPULATION_V2 with the match model's ``population`` -> ``v2`` values merged over it (a section's keys one by one;
    ``extra_prior`` as a whole)."""
    out = copy.deepcopy(POPULATION_V2)
    over = ((model or {}).get("population") or {}).get("v2") or {}
    for key, value in over.items():
        if key not in out:
            raise ValueError(f"unknown population v2 setting {key!r}")
        if isinstance(value, dict) and isinstance(out[key], dict) and key != "extra_prior":
            for k, v in value.items():
                if k not in out[key]:
                    raise ValueError(f"unknown population v2 setting {key}.{k}")
                out[key][k] = copy.deepcopy(v)
        else:
            out[key] = copy.deepcopy(value)
    return out


def population_params(params: PilotParams, archetype: str, model: dict, aircraft: str, rng: random.Random,
                      settings: dict | None = None) -> PilotParams:
    """The population-v2 pilot ``archetype`` built on ``params`` (sample_params' draw for the same aircraft and skill,
    whose archetype-free values are kept: climb and crawler altitudes, pop-up range, rusher values, cruise Mach, crank
    offset, suppression fraction, the 3-10 s recommit wait as the centre of the per-event one) from the pilot's own
    generator ``rng``. The archetype-dependent v1 values are drawn again for ``archetype``, so the match generator can
    keep its own random sequence (and so the aircraft, spawns and loadouts) the same as without population v2."""
    if archetype not in ARCHETYPES_V2:
        raise ValueError(f"unknown archetype {archetype!r}")
    s = settings if settings is not None else population_settings(model)
    cm, ev, dc, team, sk = s["commit"], s["evasion"], s["doctrine"], s["team"], s["skill"]
    prm = model["parameters"]
    skill = params.skill
    notes = next((g.get("notes", "") for g in model["groups"] if aircraft in g["aircraft"]), "")
    flank = _uniform(rng, cm["flank_offset_deg"]) if archetype in ("left", "right", "pair", "pusher") \
        else rng.uniform(-5., 5.)
    if archetype == "pusher" and rng.random() < .5:
        flank = -flank                     # a pusher flanks to a drawn side: the sign (- right)
    left_p, middle_p = prm["suppression_shot_probability"]["left"], prm["suppression_shot_probability"]["middle"]
    suppress = rng.random() < dict(left=left_p, right=left_p, pair=left_p, middle=middle_p).get(archetype, 0.)
    early_left = archetype == "middle" and "middle.early_left" in notes and rng.random() < .7
    go_home = archetype == "middle" and rng.random() < prm["middle_go_home_probability"]["value"]
    triggers, never, contact, timer, clear = (), None, 0., 0., 0.
    if archetype == "speedster":
        clear = _uniform(rng, cm["speedster_clear_s"])
    elif archetype not in NO_CLIMB:
        probs = _by_archetype(cm["triggers"], archetype)
        drawn = tuple(k for k in COMMIT_TRIGGERS if rng.random() < probs.get(k, 0.))
        # a wingman also turns in on his lead; a pusher as soon as an enemy is inside his launch envelope
        triggers = {"pair": ("lead",), "pusher": ("envelope",)}.get(archetype, ())+drawn+("timer",)
        if rng.random() < _by_archetype(cm["never"], archetype):
            never = _pick(rng, cm["never_mode"])
        contact = 1000.*_uniform(rng, cm["contact_km"])
        t = cm["timer_s"]
        timer = min(max(rng.lognormvariate(math.log(t["median"]), t["sigma"]), t["clip"][0]), t["clip"][1])
        clear = _uniform(rng, cm["clear_s"])
    conc = ev["concentration"]
    evade = _propensity(rng, _by_archetype(ev["styles"], archetype), conc)
    chaff = _propensity(rng, _by_archetype(ev["chaff"], archetype), conc)
    recommit = _propensity(rng, _by_archetype(ev["recommit"], archetype), conc)
    p_outer, p_switch = _uniform(rng, ev["p_outer"]), _uniform(rng, ev["p_switch"])
    shots = int(_pick(rng, _by_archetype(dc["shots"], archetype)))
    gap = _uniform(rng, _by_archetype(dc["salvo_gap_s"], archetype))
    fraction = _uniform(rng, _by_archetype(dc["launch_fraction"], archetype))
    support = _propensity(rng, _by_archetype(dc["support"], archetype), conc)
    support_max = _uniform(rng, dc["support_max_s"])
    # a pusher stays in TWS: he fires the moment a shot is legal, and an STT lock asked for with the shot is not
    # there yet when the executor fires (the launch is lost)
    p_stt = 0. if archetype == "pusher" else _uniform(rng, dc["p_stt"])
    second = 1000.*_uniform(rng, dc["second_round_wide_km"]) if rng.random() < dc["p_second_round_wide"] \
        else params.second_round_m
    level, peak = params.level_alt_m, params.peak_alt_m
    if archetype == "speedster":
        level, peak = _uniform(rng, s["speedster"]["level_alt_m"]), None
    deck_offset = _uniform(rng, s["deck"]["deck_offset_deg"]) if archetype == "deck_notcher" else 0.
    pair_offset = 1000.*_uniform(rng, team["pair_offset_km"]) if archetype == "pair" else 0.
    pair_back = 1000.*_uniform(rng, team["pair_back_km"]) if archetype == "pair" else 0.
    bait_max = _uniform(rng, team["bait_max_s"]) if archetype == "bait" else 0.
    push_wait = _uniform(rng, s["push"]["wait_s"]) if archetype == "pusher" else 0.
    delay = _uniform(rng, sk["delay_median_s"][skill])
    if archetype == "rusher" and skill != "top":
        delay *= _uniform(rng, sk["rusher_delay_factor"])
    p_react = RUSH_P_REACT if archetype == "rusher" and skill != "top" else P_REACT[skill]
    p_correct, p_lock = P_CORRECT[skill], P_LOCK_REACT[skill]
    if archetype == "speedster":
        p_react = p_correct = p_lock = 1.     # runs from every lock and missile, always cold
    return replace(
        params, archetype=archetype, level_alt_m=level, peak_alt_m=peak, flank_offset_deg=flank, flank_dist_m=0.,
        suppress=suppress, after_shot="per_engagement", go_home=go_home, early_left=early_left,
        chaff_style="per_event", second_round_m=second, defend_maneuver="per_event", defend_dive=False, dive_deg=0.,
        p_react=p_react, p_correct=p_correct, p_lock_react=p_lock, delay_median_s=delay,
        population="v2", commit_triggers=triggers, commit_never=never, commit_contact_m=contact,
        commit_timer_s=timer, commit_clear_s=clear, evade_weights=evade, chaff_weights=chaff,
        recommit_weights=recommit, clear_wait_s=params.recommit_clear_s, p_outer=p_outer, p_switch=p_switch,
        shots_per_engagement=shots, salvo_gap_s=gap, launch_fraction=fraction, support_weights=support,
        support_max_s=support_max, p_stt=p_stt, deck_offset_deg=deck_offset, pair_offset_m=pair_offset,
        pair_back_m=pair_back, bait_max_s=bait_max, push_wait_s=push_wait)


class TeamRadio:
    """What a team's v2 pilots learn of each other beyond the map marks (D): a teammate's launch, seen as a smoke
    trail when it happens within the commit settings' ``team_seen_km`` (any distance for his wingman, on voice), and a
    lead's "committing" call to his wingman. One per team and match; calls are (time, ident, kind, x, y). A pilot posts
    a launch when his own missile shows in his observation (so the policy's real launches count, never a script's
    unflown wish) and a commit once his heading actually points at the enemy."""

    def __init__(self):
        self.calls = []

    def post(self, t, ident, kind, x, y):
        self.calls.append((t, ident, kind, x, y))


class PilotV2(Pilot):
    """A population-v2 pilot (docs/population_spec.md): the observation, the IntentExecutor and the discrete intent
    heads of ``Pilot``, but

      commit     a climbing pilot turns in on the first of his drawn information triggers (a radar contact inside
                 R_commit, a teammate's launch seen, the RWR showing nothing for T_clear, his patience timer after the
                 climb; a wingman also on his lead's turn-in or launch) after a short hesitation; or never (cold / home);
                 a speedster only while no lock or missile is on his RWR;
      evasion    every reaction draws its manoeuvre from the pilot's propensities (a normal pilot's wrong pick from
                 REPERTOIRE, P_CORRECT per event as before), its chaff rhythm, side, length, a possible change of mind,
                 and how it ends (after a drawn wait once the RWR is clear, early before it clears, or never: cold);
      doctrine   shots per engagement and salvo gap, the launch range as a share of Rmax, the support after the
                 shot (crank, straight, drag) and STT per engagement;
      archetypes spammer, speedster, deck_notcher, pair (wingman of a teammate), bait, pusher, besides the five of
                 ``Pilot``.

    The pusher (``_push``, docs/analysis_offboresight_push.md) fires at the first radar track inside his launch
    envelope and launch-angle limit without turning in first (an aircraft whose limit does not reach the target turns
    toward it only as far as the intent heads' 40 / 90 degree offsets need; beyond 16 km to 40 degrees), ripples 2-3
    missiles a few seconds apart while turning in to pursue, and ripples again when the target turns back hot or
    leaves the beam.

    ``events`` logs every commit, attack plan and evasion with its draws."""

    def __init__(self, params, rng, *, settings=None, **kwargs):
        super().__init__(params, rng, **kwargs)
        a = params.archetype
        self.settings = settings if settings is not None else population_settings()
        if a in DECK_ARCHETYPES:
            self.floor = max(5., params.crawler_alt_m/3.)
        self.phase = {"crawler": "crawl", "rusher": "rush", "deck_notcher": "deck"}.get(a, "climb")
        self.radio, self.radio_read = None, 0
        self.lead_ident = self.lead_spawn = self.lead_mark = None   # pair: set by the match builder
        self.pair_side = self.lead_commit_t = None
        self.commit_call = False
        self.shots_seen = set()
        self.team_launches = []          # (time, ident, distance when it happened) of teammates' launches
        self.rwr_last_t = 0.             # the RWR last showed anything
        self.threat_last_t = -1e9        # ... a lock or a missile warning
        self.commit_t = self.commit_at = self.commit_trigger = None
        self.flank_cycle = None
        self.plan = None                 # the current attack plan (drawn when an attack phase starts)
        self.event = None                # the current evasion's draws
        self.events = []
        self.cold_point, self.gone_cold = None, False
        self.want_stt = False
        self.last_chaff_t = -1e9
        self.push_side = None            # pusher: the side (+1 right) the target is held on
        self.push_level = None           # pusher: the offset (PUSH_LEVELS_DEG) the target is held at before a shot
        self.push_top = None             # pusher: the largest offset the intent heads can hold inside the limit
        if a == "pusher":
            self.outer = -1. if params.flank_offset_deg < 0. else 1.
            self.over_shoulder_p = 1.    # shoots beyond 60 degrees off the nose whenever the limit allows

    def describe(self):
        out = dict(self.p.describe())
        if self.p.archetype == "pair":
            out["pair_lead"] = self.lead_ident
        return out

    def execution_settings(self, execution=None):
        """``execution`` with the autonomous delay path's personal median fixed at this pilot's delay_median_s (the
        executor otherwise draws it from U(1, 2.5) s for everyone); a configured sigma and bounds stay."""
        out = dict(execution or {})
        delay = {k: dict(v) for k, v in (out.get("delay") or {}).items()}
        median = float(self.p.delay_median_s)
        delay.setdefault("autonomous", {})["median_range"] = (median, median)
        out["delay"] = delay
        return out

    def propose(self, obs, entities):
        intent = super().propose(obs, entities)
        if self.want_stt and intent.target is not None:
            e = next((e for e in entities if e.key == intent.target), None)
            if e is not None and e.kind == "radar" and e.track_id is not None and e.missile is None:
                intent = replace(intent, radar_mode=2)     # STT on the radar target, through the radar_mode head
        return intent

    def _set_phase(self, phase, now):
        if phase != self.phase:
            self.plan, self.want_stt = None, False
        super()._set_phase(phase, now)

    # -- information --------------------------------------------------------------------------------------------

    def _post(self, now, kind, own):
        if self.radio is not None:
            self.radio.post(now, self.ident, kind, own.position[0], own.position[1])

    def _listen(self, obs, now):
        own = obs.own
        for s in obs.shots:
            if s.uid not in self.shots_seen:
                self.shots_seen.add(s.uid)
                self._post(now, "launch", own)
                plan = self.plan
                if plan is not None and plan["pending_t"] is not None:   # the proposed shot has left the rail
                    plan["uids"].append(s.uid)
                    plan["launched"] += 1
                    plan["pending_t"], plan["last_t"] = None, now
        radio = self.radio
        if radio is None:
            return
        for t, ident, kind, x, y in radio.calls[self.radio_read:]:
            if ident == self.ident:
                continue
            if kind == "launch":
                self.team_launches.append((t, ident, math.hypot(x-own.position[0], y-own.position[1])))
            if ident == self.lead_ident and self.lead_commit_t is None:
                self.lead_commit_t = t       # the lead's "committing" or his first launch, on voice
        self.radio_read = len(radio.calls)

    def _lead(self, obs):
        """The lead's friend mark: at first the one nearest the lead's spawn point, then by its mark id."""
        friends = [m for m in obs.marks if m.friend]
        if self.lead_mark is None:
            if self.lead_spawn is None or not friends:
                return None
            m = min(friends, key=lambda m: math.hypot(m.x-self.lead_spawn[0], m.y-self.lead_spawn[1]))
            self.lead_mark = m.mark_id
            return m
        return next((m for m in friends if m.mark_id == self.lead_mark), None)

    def _bait_friend(self, obs):
        own = obs.own
        reach = self.settings["team"]["bait_friend_km"]*1000.
        near = [(math.hypot(m.x-own.position[0], m.y-own.position[1]), m) for m in obs.marks if m.friend]
        near = [f for f in near if f[0] <= reach]
        return min(near, key=lambda f: f[0])[1] if near else None

    # -- the decision -------------------------------------------------------------------------------------------

    def decide(self, obs: Observation) -> Action:
        p, now, own = self.p, obs.time_s, obs.own
        if self.start_xy is None:
            self.start_xy = (own.position[0], own.position[1])
        self._listen(obs, now)
        present = self._threats(obs, now)
        if obs.rwr:
            self.rwr_last_t = now
        if obs.maw or any(c.tracking or (c.missile_warning and (c.range_m is None or c.range_m <= THREAT_RANGE_M))
                          for c in obs.rwr):
            self.threat_last_t = now
        self.anc = ax, ay, az, _ = self._estimated_anchor(obs) if self.managed_execution else self._anchor(obs)
        to_enemy = bearing_of(ax-own.position[0], ay-own.position[1])
        if self.commit_call and abs(wrap(to_enemy-own.heading_deg)) < 25.:
            self.commit_call = False
            self._post(now, "commit", own)
        spent = own.missiles <= 0 or getattr(own, "bingo", False)      # as Pilot: no missiles or bingo fuel
        if self.phase == "home" and not spent:              # rearmed at the airfield (opt-in)
            self._set_phase({"crawler": "crawl", "rusher": "rush", "deck_notcher": "deck",
                             "pusher": "push"}.get(p.archetype, "advance"), now)
        if spent and self.phase not in ("home", "evade") and self.support_from is None:
            self._set_phase("home", now)
        if self.pending is not None and now >= self.pending.at and self.phase != "evade":
            self._react(obs, now, self.pending.correct)
        fire, chaff, phase = None, 0, self.phase
        if phase == "evade":
            cmd, chaff, done = self._evade_command(obs, now, present)
            if done:
                self._end_evade(obs, now)
        elif phase == "climb":
            cmd = self._climb(obs, now, to_enemy)
            if self.phase == "push":                         # pusher: the envelope commit shoots at once
                cmd, fire = self._push(obs, now, to_enemy)
        elif phase == "push":
            cmd, fire = self._push(obs, now, to_enemy)
        elif phase in ATTACK_PHASES:
            speed = None if p.archetype == "speedster" else self._cruise_speed(own.altitude_m)
            cmd, fire = self._attack(obs, now, to_enemy, altitude=p.level_alt_m, speed=speed)
        elif phase in ("crawl", "popup"):
            cmd, fire = self._crawler(obs, now, to_enemy)
        elif phase == "deck":
            cmd, fire = self._attack(obs, now, to_enemy, altitude=p.crawler_alt_m,
                                     speed=self._cruise_speed(p.crawler_alt_m),
                                     approach=(to_enemy-self.outer*p.deck_offset_deg) % 360.)
        elif phase == "rush":
            cmd, fire = self._rush(obs, now, to_enemy)
        elif phase == "cold":
            cmd = self._cold(obs)
        else:
            cmd = self._home(obs)
        cmd = self._guard(obs, cmd, phase)
        if self.managed_execution:
            # As Pilot: unload before a shot; outside defence no sub-corner energy on hard turns.
            if fire is not None:
                cmd = replace(cmd, max_load=min(cmd.max_load, 2.), throttle_percent=110., speed_mps=None)
            elif phase != "evade":
                from .rl_observation import energy_state
                from .flight import aircraft_model
                if not hasattr(self, "performance_model"):
                    self.performance_model = aircraft_model(own.aircraft, mass_factor=1.3)
                model = self.performance_model
                en, ps, ratio, *_ = energy_state(own, model, model.structure.limits(model.mass))
                if ratio is not None and ratio < 1.:
                    cmd = replace(cmd, max_load=min(cmd.max_load, 3.), throttle_percent=110., speed_mps=None)
        if fire is not None:
            self._fired(now)
        return Action(cmd, self._radar(obs), fire, chaff)

    def _home(self, obs):
        own = obs.own
        hx, hy = self.home_xy
        dx, dy = hx-own.position[0], hy-own.position[1]
        heading = bearing_of(dx, dy) if math.hypot(dx, dy) > HOME_ORBIT_M else (own.heading_deg+HOME_ORBIT_TURN_DEG) % 360.
        return self._fly(heading, altitude=max(own.altitude_m, HOME_ALT_M), speed=self._cruise_speed(own.altitude_m))

    # -- commit -------------------------------------------------------------------------------------------------

    def _climb(self, obs, now, to_enemy):
        p, own = self.p, obs.own
        target = p.level_alt_m if self.peak_done else p.peak_alt_m
        self.alt_history.append((now, own.altitude_m))
        while self.alt_history[0][0] < now-PEAK_STALL_S:
            self.alt_history.pop(0)
        span = now-self.alt_history[0][0] >= PEAK_STALL_S-1.
        gained = own.altitude_m-self.alt_history[0][1]
        if not self.peak_done and (own.altitude_m >= p.peak_alt_m-ANGLE_LEVEL_TOL_M or (span and gained < PEAK_STALL_M)):
            self.peak_done, target = True, p.level_alt_m
        # Climbed: within ANGLE_LEVEL_TOL_M of the own level altitude (vertical_mode 'angle' levels short of it), or no
        # longer climbing up there (the altitude options hold the nearest altitude they can fly).
        reached = self.peak_done and (abs(own.altitude_m-p.level_alt_m) < ANGLE_LEVEL_TOL_M or
                                      (span and abs(gained) < PEAK_STALL_M and own.altitude_m > .5*p.level_alt_m))
        if reached and self.climbed_at is None:
            self.climbed_at = now
        if p.archetype == "pusher" and p.commit_never is None and self.commit_trigger != "envelope" and \
                self._push_ready(obs):
            self.commit_trigger, self.commit_at = "envelope", now   # no hesitation: shoot when it is legal
        elif self.commit_at is None:
            trigger = self._commit_trigger(obs, now, reached)
            if trigger is not None:
                self.commit_trigger = trigger
                self.commit_at = now+(0. if trigger == "never" else
                                      _uniform(self.rng, self.settings["commit"]["hesitation_s"]))
        if self.commit_at is not None and now >= self.commit_at:
            return self._commit(obs, now, to_enemy)
        if p.archetype == "pair":
            cmd = self._formation(obs)
            if cmd is not None:
                return cmd
        speed = None if own.altitude_m < target-300. or p.archetype == "speedster" else self._cruise_speed(own.altitude_m)
        return self._fly(self._flank_heading(now, to_enemy), altitude=target, speed=speed)

    def _commit_trigger(self, obs, now, reached):
        """The first information event of the pilot's set that has happened, or None."""
        p, own, cm = self.p, obs.own, self.settings["commit"]
        if p.commit_never is not None:
            return "never" if reached or now >= NEVER_AFTER_S else None
        if p.archetype == "speedster":
            return "rwr_quiet" if reached and now-self.threat_last_t >= p.commit_clear_s else None
        high = reached or own.altitude_m >= cm["min_alt_fraction"]*p.level_alt_m
        trig = p.commit_triggers
        if "contact" in trig:
            near = min((c.range_m for c in aircraft_contacts(obs) if c.range_m is not None), default=math.inf)
            if near <= p.commit_contact_m and (high or near <= CLOSE_CONTACT_M):
                return "contact"
        if not high:
            return None
        if "lead" in trig and self.lead_commit_t is not None:
            return "lead"
        seen = cm["team_seen_km"]*1000.
        if "team_launch" in trig and any(d <= seen or i == self.lead_ident for _, i, d in self.team_launches):
            return "team_launch"
        if reached and "rwr_clear" in trig and now-self.rwr_last_t >= p.commit_clear_s:
            return "rwr_clear"
        if self.climbed_at is not None and now-self.climbed_at >= p.commit_timer_s:
            return "timer"
        return None

    def _commit(self, obs, now, to_enemy):
        p, own = self.p, obs.own
        self.commit_t = now
        self.events.append(dict(t=round(now, 2), kind="commit", trigger=self.commit_trigger))
        if p.commit_never is not None:
            self.gone_cold = True
            self._set_phase("cold", now)
            return self._cold(obs)
        self.commit_call = True
        self._set_phase("spam" if p.archetype == "spammer" else "push" if p.archetype == "pusher" else
                        "suppress" if p.suppress and own.missiles > 0 else "advance", now)
        return self._fly(to_enemy, altitude=p.level_alt_m,
                         speed=None if p.archetype == "speedster" else self._cruise_speed(own.altitude_m))

    def _flank_heading(self, now, to_enemy):
        """The climb heading, as Pilot's off the team's forward direction (left / right / an unled wingman
        ``flank_offset_deg`` to their side, the others within +-5 deg of straight ahead). The intent heads fly it only
        as one of FLANK_LEVELS_DEG off the enemy reference, so a flanker flies a time share of the two offsets around
        the wanted one, in cycles of flank_cycle_s (the mean track is the wanted heading); the others the nearest."""
        p = self.p
        flanker = p.archetype in ("left", "right", "pair", "pusher")
        side = (self.outer if p.archetype == "pusher" else -1. if p.archetype == "right" else 1.) if flanker else 0.
        offset = abs(p.flank_offset_deg) if p.archetype == "pusher" else p.flank_offset_deg
        want = (self.forward_deg-side*offset if flanker else self.forward_deg+offset) % 360.
        rel = max(FLANK_LEVELS_DEG[0], min(FLANK_LEVELS_DEG[-1], wrap(want-to_enemy)))
        if not flanker:
            return (to_enemy+min(FLANK_LEVELS_DEG, key=lambda v: abs(v-rel))) % 360.
        lo = max(v for v in FLANK_LEVELS_DEG if v <= rel)
        hi = min(v for v in FLANK_LEVELS_DEG if v >= rel)
        level = lo
        if hi > lo:
            if self.flank_cycle is None or now >= self.flank_cycle[1]:
                cycle = _uniform(self.rng, self.settings["commit"]["flank_cycle_s"])
                self.flank_cycle = (now, now+cycle, (rel-lo)/(hi-lo))
            start, end, share = self.flank_cycle
            level = hi if now < start+share*(end-start) else lo
        return (to_enemy+level) % 360.

    def _formation(self, obs):
        """A wingman's station: pair_offset_m abeam (on his side) and pair_back_m behind the lead's map mark."""
        lead = self._lead(obs)
        if lead is None or lead.heading_deg is None:
            return None
        p, own = self.p, obs.own
        h = math.radians(lead.heading_deg)
        fx, fy = math.sin(h), math.cos(h)
        rx, ry = fy, -fx                       # the lead's right
        x, y = own.position[0], own.position[1]
        if self.pair_side is None:
            self.pair_side = 1. if (x-lead.x)*rx+(y-lead.y)*ry >= 0. else -1.
        sx = lead.x+self.pair_side*p.pair_offset_m*rx-p.pair_back_m*fx
        sy = lead.y+self.pair_side*p.pair_offset_m*ry-p.pair_back_m*fy
        dx, dy = sx-x, sy-y
        along, cross = dx*fx+dy*fy, dx*rx+dy*ry
        if math.hypot(dx, dy) > 6000. and along > 0.:
            heading = bearing_of(dx, dy)
        else:
            heading = (lead.heading_deg+max(-60., min(60., .012*cross))) % 360.
        target = p.level_alt_m if self.peak_done else p.peak_alt_m
        cruise = self._cruise_speed(own.altitude_m)
        if along > 1000.:
            return self._fly(heading, altitude=target)                          # full power: catch up
        if along < -3000.:
            return self._fly(heading, altitude=target, speed=.6*cruise, brake=True)
        return self._fly(heading, altitude=target, speed=cruise)

    # -- attack -------------------------------------------------------------------------------------------------

    def _limit(self):
        p = self.p
        if self.phase == "suppress":
            return dict(fraction=p.suppress_fraction)
        if self.phase in ("advance", "spam", "deck") or p.archetype == "spammer":
            return dict(fraction=p.launch_fraction)
        return dict(max_range_m=p.second_round_m)          # recommit, round2, popup

    def _attack_plan(self, obs, now):
        """The engagement's draws (logged): shots, salvo gap, support mode, STT; and its launch book-keeping (a proposed
        shot is pending until its missile shows in the observation; one the executor lost is proposed again)."""
        if self.plan is None:
            p, rng = self.p, self.rng
            self.plan = dict(t=round(now, 2), kind="attack", phase=self.phase,
                             shots=max(1, min(p.shots_per_engagement, obs.own.missiles)), gap=p.salvo_gap_s,
                             support=_pick(rng, p.support_weights) or "crank", crank=p.crank_offset_deg,
                             stt=rng.random() < p.p_stt, launched=0, lost=0, pending_t=None, last_t=None, uids=[])
            self.events.append(self.plan)
        plan = self.plan
        if plan["pending_t"] is not None and now-plan["pending_t"] > LAUNCH_GRACE_S:
            plan["pending_t"] = None                     # never launched (the executor dropped or refused it)
            plan["lost"] += 1
            if not plan["launched"]:
                self.support_from = None                 # nothing in the air: as before the first shot
        return plan

    def _near_shot(self, obs, limit):
        """An aircraft track within 1.3 times the launch limit (time to lock STT before the shot)."""
        own = obs.own
        if "max_range_m" in limit:
            reach = limit["max_range_m"]
        elif self.judge is not None:
            reach = limit["fraction"]*self.judge.rmax(own.altitude_m, own.speed_mps, 0.)
        else:
            return False
        return any(c.range_m <= 1.3*reach for c in self._tracks(obs))

    def _supporting_v2(self, obs, now, plan):
        """Still supporting this engagement's missiles: one is in flight, none has its seeker on or lost the datalink,
        and the last left the rail no more than support_max_s ago."""
        mine = [s for s in obs.shots if s.uid in plan["uids"]]
        if not mine:
            return False
        active = any(s.active or not s.datalink for s in mine)
        return not (active or now-plan["last_t"] > self.p.support_max_s)

    def _attack(self, obs, now, to_enemy, *, altitude, speed, approach=None):
        """Fly at the enemy (``approach``: this heading until the first shot), fire the plan's salvo inside the phase's
        limit, support it (crank / straight) until the seeker is on or turn away at once (drag), then evade."""
        p, own = self.p, obs.own
        straight = self._fly(to_enemy, altitude=altitude, speed=speed)
        plan, limit = self._attack_plan(obs, now), self._limit()
        if self.support_from is not None:
            if plan["pending_t"] is not None:            # wait for the missile to leave the rail
                return straight, None
            if plan["launched"] < plan["shots"] and own.missiles > 0 and plan["last_t"] is not None and \
                    now-plan["last_t"] >= plan["gap"]:
                fire = self._shot(obs, now, gap=1., **limit)       # the salvo gap counts from the last launch
                if fire is not None:
                    plan["pending_t"] = now
                    return straight, fire
            salvo_over = plan["launched"] >= plan["shots"] or own.missiles <= 0 or \
                (plan["last_t"] is not None and now-plan["last_t"] > plan["gap"]+LAUNCH_GRACE_S)
            if (plan["support"] == "drag" and salvo_over) or (plan["launched"] and not self._supporting_v2(obs, now, plan)):
                self._turn_away(obs, now, to_enemy, forced="drag" if plan["support"] == "drag" else None)
                return straight, None
            if plan["support"] == "crank":
                return self._fly((to_enemy-self.crank_side*plan["crank"]) % 360., altitude=altitude, speed=speed), None
            return straight, None
        if self.phase == "recommit" and abs(wrap(to_enemy-own.heading_deg)) < 30.:
            self._set_phase("round2", now)
            plan, limit = self._attack_plan(obs, now), self._limit()
        self.want_stt = plan["stt"] and self._near_shot(obs, limit)
        fire = self._shot(obs, now, gap=1., **limit)
        if fire is not None:
            plan["pending_t"] = now
            if p.early_left:
                plan["shots"], plan["support"] = 1, "drag"
            self.support_from, self.crank_side = now, self.outer
            return straight, fire
        if self.phase == "suppress" and now-self.phase_t > SUPPRESS_TIMEOUT_S:
            self._set_phase("advance", now)
        elif self.phase == "popup" and now-self.phase_t > POPUP_GIVE_UP_S:
            self._set_phase("crawl", now)
        return (straight if approach is None else self._fly(approach, altitude=altitude, speed=speed)), None

    def _crawler(self, obs, now, to_enemy):
        p, own = self.p, obs.own
        speed = self._cruise_speed(p.crawler_alt_m)
        if self.phase == "crawl":
            distance = math.hypot(self.anc[0]-own.position[0], self.anc[1]-own.position[1])
            if distance > p.popup_range_m or own.missiles <= 0:
                return self._fly(to_enemy, altitude=p.crawler_alt_m, speed=speed), None
            self._set_phase("popup", now)
        return self._attack(obs, now, to_enemy, altitude=p.popup_alt_m, speed=speed)

    # -- pusher -------------------------------------------------------------------------------------------------

    def _push_limit(self, own):
        """The largest of PUSH_LEVELS_DEG at least margin_deg inside the launch-angle limit (intent.launch_limit under
        the executor, 60 degrees otherwise, as _shot): 90 for the Su-30SM2's R-77-1 (120), 40 for 60-70 degree
        limits."""
        if self.push_top is None:
            from .intent import launch_limit
            limit = launch_limit(own.aircraft, own.missile_id) if self.managed_execution else 60.
            room = limit-self.settings["push"]["margin_deg"]
            self.push_top = max((v for v in PUSH_LEVELS_DEG if v <= room), default=0.)
        return self.push_top

    def _push_reach(self, obs, c):
        """launch_fraction of the Rmax against track ``c``, aspect-adjusted as _shot judges it."""
        own = obs.own
        aspect = self._aspect(obs, c)
        reach = None if self.reach is None else self.reach.rmax(
            own.altitude_m, own.speed_mps, aspect, 0. if c.position is None else c.position[2]-own.altitude_m)
        return self.p.launch_fraction*(reach if reach is not None else
                                       self.judge.rmax(own.altitude_m, own.speed_mps, aspect))

    def _push_ready(self, obs):
        """An enemy inside the launch envelope at any angle off the nose: a radar track within launch_fraction of its
        Rmax, or the best estimate (map mark, sighting, memory; not the spawn or a bare RWR bearing) within
        launch_fraction of the hot Rmax. Needs missiles and a range judge."""
        own = obs.own
        if own.missiles <= 0 or self.judge is None:
            return False
        if any(c.range_m <= self._push_reach(obs, c) for c in self._tracks(obs)):
            return True
        ax, ay, _, source = self.anc
        if source in ("spawn", "rwr"):
            return False
        hot = self.judge.rmax(own.altitude_m, own.speed_mps, 0.)
        return math.hypot(ax-own.position[0], ay-own.position[1]) <= self.p.launch_fraction*hot

    def _push_far(self, own, distance):
        """Beyond wide_km a shot far off the nose seldom kills (the analysis: Pk 0.03 at 90 degrees and 22 km):
        the largest offset to hold there and the widest azimuth to fire at."""
        ps = self.settings["push"]
        if distance is None or distance <= 1000.*ps["wide_km"]:
            return None
        return max(v for v in PUSH_LEVELS_DEG if v <= ps["far_deg"]), ps["far_deg"]

    def _push_heading(self, obs, to_enemy, distance):
        """A heading that holds the enemy at one of PUSH_LEVELS_DEG off the nose on the side he is on: the one nearest
        his present offset (ties: the larger; kept while he stays within PUSH_LEVEL_HYST_DEG of it), at most the
        largest inside the limit (beyond wide_km at most 40 degrees, and pointing at him while he is beyond far_deg),
        so no turn toward him unless the limit or the range needs it. The intent heads fly these offsets off the
        reference (from_flight_action)."""
        own = obs.own
        az = wrap(to_enemy-own.heading_deg)
        if self.push_side is None or PUSH_SIDE_HYST_DEG <= abs(az) <= 180.-PUSH_SIDE_HYST_DEG:
            self.push_side = 1. if az >= 0. else -1.
        top = self._push_limit(own)
        far = self._push_far(own, distance)
        if far is not None:
            top = min(top, far[0])
            if abs(az) > far[1]:
                top = 0.            # turn in until he is inside far_deg (holding 40 can settle a few degrees outside)
        if self.push_level is not None and self.push_level <= top and \
                abs(abs(az)-self.push_level) <= PUSH_LEVEL_HYST_DEG:
            level = self.push_level
        else:
            level = min((v for v in PUSH_LEVELS_DEG if v <= top), key=lambda v: (abs(v-abs(az)), -v))
        self.push_level = level
        return (to_enemy-self.push_side*level) % 360.

    def _push_shot(self, obs, now):
        """(track id, contact) of a legal shot inside the envelope (_shot: track, launch-angle limit on the airframe
        azimuth, Rmax share; beyond wide_km within far_deg of the nose, horizontally as the analysis measured it: the
        airframe azimuth shrinks in a banked turn), or (None, None)."""
        def near_or_ahead(c):
            far = self._push_far(obs.own, c.range_m)
            return far is None or abs(wrap(c.bearing_deg-obs.own.heading_deg)) <= far[1]
        fire = self._shot(obs, now, gap=1., fraction=self.p.launch_fraction, accept=near_or_ahead)
        if fire is None:
            return None, None
        contact = next((c for c in self._tracks(obs) if (STT_TRACK if c.kind == "stt" else c.track_id) == fire), None)
        return fire, contact

    @staticmethod
    def _push_key(c):
        """A track's observed identity: its map mark id (stable across track ids), else its track id."""
        mark = getattr(c, "mark_id", None)
        return f"mark:{mark}" if mark is not None else f"track:{c.track_id}"

    def _push_pending(self, plan, obs, now, c):
        """Log a proposed shot: time (the executor's delay follows), off-boresight (horizontal), airframe azimuth (the
        launch limit's), range."""
        plan["pending_t"] = now
        plan["shot_t"].append(round(now, 2))
        plan["oba"].append(None if c is None else round(abs(wrap(c.bearing_deg-obs.own.heading_deg)), 1))
        plan["az"].append(None if c is None else round(abs(c.azimuth_deg), 1))
        plan["range_m"].append(None if c is None else round(c.range_m))

    def _push(self, obs, now, to_enemy):
        """pusher (phase push; docs/analysis_offboresight_push.md), one logged plan per ripple:

          aim     fly at the enemy until he is inside the envelope (_push_ready), then hold him at the offset nearest
                  where he is (turning toward him only as far as the launch limit needs, and beyond wide_km to 40
                  degrees) and fire at the first legal track, at any angle off the nose within wide_km;
          ripple  turn in and pursue; the plan's 2-3 shots salvo_gap_s apart, counted from the previous press once
                  its missile has left the rail, while a track stays legal (over after gap + LAUNCH_GRACE_S without
                  one);
          pursue  straight at the target while own missiles fly (at most pursue_max_s after the last launch); a new
                  ripple when push_wait_s have passed since the last launch and the target turns back toward him
                  (hot, aspect under 90 - beam_deg, after that target was seen off hot since the ripple began) or
                  leaves the beam (past 90 + beam_deg after being seen within beam_deg of it); when no missile is
                  left in flight, a new engagement (aim), still pursuing, or home when spent.

        (The analysis: cranking or holding the target at 90 degrees after the shot killed 38-41 % within 60 s against
        a notch-and-chaff defender, pursuing with a follow-up 4 s later 84-89 %.)"""
        p, own, ps = self.p, obs.own, self.settings["push"]
        plan = self._attack_plan(obs, now)
        if "state" not in plan:
            plan.update(state="aim", repush=False, shot_t=[], oba=[], az=[], range_m=[], seen={})
        alt, speed = p.level_alt_m, self._cruise_speed(own.altitude_m)
        if plan["state"] == "ripple" and not plan["launched"] and plan["pending_t"] is None:
            plan["state"] = "pursue" if plan["repush"] else "aim"    # its first shot never left the rail
        state = plan["state"]
        if state == "aim":
            inside = self._push_ready(obs)
            distance = math.hypot(self.anc[0]-own.position[0], self.anc[1]-own.position[1])
            heading = self._push_heading(obs, to_enemy, distance) if inside else to_enemy
            fire, c = self._push_shot(obs, now) if inside else (None, None)
            if fire is not None:
                plan["state"] = "ripple"
                self._push_pending(plan, obs, now, c)
                self.support_from = now
            return self._fly(heading, altitude=alt, speed=speed), fire
        hold = self._fly(to_enemy, altitude=alt, speed=speed)        # after the first shot: turn in and pursue
        band = ps["beam_deg"]
        for c in self._tracks(obs):                     # what each target did since the ripple began: (off hot, beam)
            aspect, key = self._aspect(obs, c), self._push_key(c)
            off, beam = plan["seen"].get(key, (False, False))
            plan["seen"][key] = (off or aspect >= 90.-band, beam or abs(aspect-90.) <= band)
        if plan["pending_t"] is not None:                           # wait for the missile to leave the rail
            return hold, None
        if state == "ripple":
            if plan["launched"] < plan["shots"] and own.missiles > 0 and \
                    now-plan["last_t"] <= plan["gap"]+LAUNCH_GRACE_S:
                if now-plan["shot_t"][-1] >= plan["gap"]:          # the gap counts from the previous press
                    fire, c = self._push_shot(obs, now)
                    if fire is not None:
                        self._push_pending(plan, obs, now, c)
                        return hold, fire
                return hold, None
            plan.update(state="pursue", pursue_t=round(now, 2))
        last = plan["last_t"] if plan["last_t"] is not None else plan["t"]
        if own.missiles > 0 and now-last >= p.push_wait_s:
            fire, c = self._push_shot(obs, now)
            aspect = None if c is None else self._aspect(obs, c)
            off, beam = (False, False) if c is None else plan["seen"].get(self._push_key(c), (False, False))
            if fire is not None and aspect is not None and ((aspect < 90.-band and off) or (aspect > 90.+band and beam)):
                plan["end"] = "repush"                              # turned back hot, or left the beam: again
                self.plan = None
                new = self._attack_plan(obs, now)
                new.update(state="ripple", repush=True, aspect=round(aspect, 1), shot_t=[], oba=[], az=[], range_m=[],
                           seen={})
                self._push_pending(new, obs, now, c)
                return hold, fire
        if obs.shots and now-last <= ps["pursue_max_s"]:
            return hold, None
        plan["end"] = "new"                                         # nothing left in flight: a new engagement
        self.plan, self.support_from = None, None
        return hold, None

    def _cold(self, obs):
        """Never turns in (again): to a hold point (home: the own spawn; cold: half way back there) and round it."""
        p, own = self.p, obs.own
        x, y = own.position[0], own.position[1]
        if self.cold_point is None:
            hx, hy = self.home_xy
            self.cold_point = (hx, hy) if p.commit_never == "home" else (x+.5*(hx-x), y+.5*(hy-y))
        dx, dy = self.cold_point[0]-x, self.cold_point[1]-y
        bearing = bearing_of(dx, dy)
        heading = bearing if math.hypot(dx, dy) > COLD_ORBIT_M else (bearing+90.*self.outer) % 360.
        altitude = p.crawler_alt_m if p.archetype in DECK_ARCHETYPES else p.level_alt_m
        return self._fly(heading, altitude=altitude, speed=self._cruise_speed(own.altitude_m))

    # -- evasion ------------------------------------------------------------------------------------------------

    def _draw_event(self, obs, now, correct, reason, forced=None, ignore_ok=True):
        """One evasion's draws (logged in ``events``): the manoeuvre (``correct``: from the pilot's propensities;
        otherwise a REPERTOIRE pick), chaff rhythm and its ranges, side, minimum length, how it ends, a change of mind."""
        p, rng, e = self.p, self.rng, self.settings["evasion"]
        if forced is not None:
            style = forced
        elif correct:
            weights = dict(p.evade_weights)
            if not ignore_ok:
                weights.pop("ignore", None)
            if "bait" in weights and self._bait_friend(obs) is None:
                weights["drag"] = weights.get("drag", 0.)+weights.pop("bait")
            style = _pick(rng, weights) or "drag"
        else:
            style = REPERTOIRE_STYLES[rng.randrange(len(REPERTOIRE_STYLES))]
        ev = dict(t=round(now, 2), kind="evade", reason=reason, correct=bool(correct), style=style)
        self.events.append(ev)
        if style == "ignore":
            return ev
        lo, hi = e["clear_wait_clip_s"]
        ev.update(chaff=_pick(rng, p.chaff_weights) or "continuous",
                  chaff_start_m=1000.*_uniform(rng, e["chaff_start_km"]), paced_s=_uniform(rng, e["chaff_paced_s"]),
                  late_m=1000.*_uniform(rng, e["chaff_late_km"]), outer=rng.random() < p.p_outer,
                  min_s=_uniform(rng, e["min_evade_s"]), recommit=_pick(rng, p.recommit_weights) or "after_clear",
                  wait_s=min(max(rng.lognormvariate(math.log(max(.1, p.clear_wait_s)), e["clear_wait_sigma"]), lo), hi),
                  early_s=_uniform(rng, e["early_s"]), deck_m=_uniform(rng, e["split_s_alt_m"]))
        ev["switch_at"] = now+_uniform(rng, e["switch_s"]) if rng.random() < p.p_switch else None
        return ev

    def _react(self, obs, now, correct):
        self.pending = None
        ev = self._draw_event(obs, now, correct, "threat")
        if ev["style"] != "ignore":
            self._begin(now, ev, None)

    def _turn_away(self, obs, now, reference, forced=None):
        """After the shot (support over, or drag at once): an evasion without a threat yet, drawn as any other."""
        self._begin(now, self._draw_event(obs, now, True, "after_shot", forced=forced, ignore_ok=False), reference)

    def _begin(self, now, ev, reference):
        self.evade = _Evade(now, STYLE_PLANS[ev["style"]], last_bearing=reference)
        self.event = ev
        self.pending = None
        self._set_phase("evade", now)

    def _chaff(self, obs, now, ev, e):
        rhythm = ev["chaff"]
        if rhythm == "none" or obs.own.chaff <= 0:
            return 0
        warn = [c for c in obs.rwr if c.missile_warning and c.age_s <= 1.]
        if not warn:
            return 0
        ranges = [c.range_m for c in warn if c.range_m is not None]
        r = min(ranges) if ranges else None
        if rhythm == "late":
            go = r <= ev["late_m"] if r is not None else now-e.start >= LATE_NO_RANGE_S
        else:
            go = r is None or r <= ev["chaff_start_m"]
        if go and now-self.last_chaff_t >= (ev["paced_s"] if rhythm == "paced" else 0.)-1e-6:
            self.last_chaff_t = now
            return 1
        return 0

    def _evade_command(self, obs, now, present):
        e, own, ev, p = self.evade, obs.own, self.event, self.p
        if ev.get("switch_at") is not None and now >= ev["switch_at"]:
            ev["switch_at"] = None
            weights = [(k, w) for k, w in p.evade_weights if k not in ("ignore", "bait", ev.get("flown", ev["style"]))]
            new = _pick(self.rng, weights)
            if new is not None:                              # a change of mind halfway through
                ev.update(switched_to=new, switched_at=round(now, 2), flown=new)
                e.plan, e.side = STYLE_PLANS[new], None
        style = ev.get("flown", ev["style"])
        target_deg, plane_deg, dive_deg, speed_kmh = e.plan
        e.decisions += 1
        if e.altitude is None:
            e.altitude = own.altitude_m
        bearing = self.last_threat_bearing if (self.last_threat_bearing is not None and now-self.last_threat_t < 8.) \
            else e.last_bearing
        if bearing is None:
            bearing = bearing_of(self.anc[0]-own.position[0], self.anc[1]-own.position[1])
        e.last_bearing = bearing
        away = (bearing+180.) % 360.
        if style == "bait":
            heading = away
            friend = self._bait_friend(obs)
            if friend is not None:
                toward = bearing_of(friend.x-own.position[0], friend.y-own.position[1])
                if abs(wrap(toward-bearing)) > 70.:          # not toward the threat: lead the pursuer to him
                    heading = toward
        elif target_deg == 0.:
            heading = away
            if p.go_home and p.archetype == "middle":
                heading = (self.forward_deg+180.) % 360.
        else:
            if e.side is None:
                # As Pilot: of the two three-nine headings the one more toward the outer side (or, when they differ
                # little, nearer the current heading); this event's draw may take the inner one instead.
                side_x, side_y = self.left[0]*self.outer, self.left[1]*self.outer
                score = []
                for sign in (1., -1.):
                    h = math.radians(bearing+sign*90.)
                    turn = math.cos(math.radians(wrap(bearing+sign*90.-own.heading_deg)))
                    score.append(math.sin(h)*side_x+math.cos(h)*side_y+.5*turn)
                e.side = 1. if score[0] >= score[1] else -1.
                if not ev["outer"]:
                    e.side = -e.side
            heading = (bearing+e.side*90.) % 360.
        deck = p.archetype in DECK_ARCHETYPES
        hold = p.crawler_alt_m if deck else e.altitude        # deck pilots defend back down on the deck, level
        if style == "split_s" and not deck:
            cmd = self._fly(heading, altitude=ev["deck_m"], floor=min(self.floor, .5*ev["deck_m"]))
        elif dive_deg > 0. and not deck:
            cmd = self._fly(heading, gamma=-dive_deg, floor=EVADE_FLOOR_M)
        elif speed_kmh < FULL_POWER_KMH:
            cmd = self._fly(heading, altitude=hold, speed=speed_kmh/3.6, brake=True)
        else:
            cmd = self._fly(heading, altitude=hold)
        chaff = self._chaff(obs, now, ev, e)
        if present:
            e.clear_since = None
        elif e.clear_since is None:
            e.clear_since = now
        lasted = now-e.start
        done = e.clear_since is not None and now-e.clear_since >= ev["wait_s"] and lasted >= max(MIN_EVADE_S, ev["min_s"])
        if ev["recommit"] == "early" and lasted >= max(MIN_EVADE_S, ev["early_s"]):
            done = True                                       # back before the RWR is clear
        if ev["style"] == "bait":
            reach = self.settings["team"]["bait_friend_km"]*1000.
            heard = any(t > e.start and d <= reach for t, _, d in self.team_launches)
            if (heard and lasted >= MIN_EVADE_S) or lasted >= p.bait_max_s:
                done = True                                   # the teammate has fired at the pursuer (or gave up)
        if done:
            ev["lasted_s"] = round(lasted, 2)
        return cmd, chaff, done

    def _end_evade(self, obs, now):
        ev = self.event
        self.evade = self.event = None
        if ev is not None and ev.get("recommit") == "never":
            self.gone_cold = True
        a = self.p.archetype
        if obs.own.missiles <= 0 or getattr(obs.own, "bingo", False):
            nxt = "home"
        elif self.gone_cold:
            nxt = "cold"
        elif a in ("crawler", "deck_notcher", "rusher"):
            nxt = {"crawler": "crawl", "deck_notcher": "deck", "rusher": "rush"}[a]
        elif self.commit_t is None:
            nxt = "climb"                                     # attacked before turning in: carry on climbing
        else:
            nxt = {"speedster": "advance", "pusher": "push"}.get(a, "recommit")
        self._set_phase(nxt, now)
