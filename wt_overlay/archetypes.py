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
goes ``home`` (still defending). Left / right / middle share the shape above with their own flank and
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
"""
from __future__ import annotations

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


def wrap(deg):
    return (deg+180.) % 360.-180.


def bearing_of(dx, dy):
    return math.degrees(math.atan2(dx, dy)) % 360.


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

    def describe(self):
        return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


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
        for c in obs.radar:
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
        for c in obs.radar:
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

    def _shot(self, obs, now, max_range_m=None, fraction=None, gap=SALVO_GAP_S):
        """Track id to fire at, or None. ``fraction``: inside that share of Rmax (aspect-adjusted); ``max_range_m``: inside that range."""
        own = obs.own
        if own.missiles <= 0 or now-self.shot_t < gap:
            return None
        eligible = []
        for c in self._tracks(obs):
            from .intent import launch_limit
            limit_angle = launch_limit(own.aircraft, own.missile_id)
            if abs(c.azimuth_deg) > (limit_angle if self.managed_execution else 60.):
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
        if own.missiles <= 0 and self.phase not in ("home", "evade") and self.support_from is None:
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
                self._set_phase("home" if own.missiles <= 0 else {"crawler": "crawl", "rusher": "rush"}.get(
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
        action=self.decide(obs)
        return from_flight_action(action,obs,entities,phase=self.phase,home_xy=self.home_xy,guard=self.guard)

    def _estimated_anchor(self, obs):
        now,own=obs.time_s,obs.own
        # Keys are observed mark ids, never the simulator's aircraft slots.
        for c in obs.radar:
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
        if not self.peak_done and (own.altitude_m >= p.peak_alt_m-150. or (
                now-self.alt_history[0][0] >= PEAK_STALL_S-1. and own.altitude_m-self.alt_history[0][1] < PEAK_STALL_M)):
            # At the peak, or the climb has stalled (this aircraft cannot get there): level off at the usual altitude.
            self.peak_done, target = True, p.level_alt_m
        level = p.commit_alt_m if p.commit_alt_m is not None else (8000. if self.managed_execution else p.level_alt_m)
        reached = self.peak_done and abs(own.altitude_m-level) < 400.
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
