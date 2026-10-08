"""Engagement manager: aircraft, missiles, radars and RWRs advancing on one 1/48 s clock.

An ``Engagement`` owns a flat-earth world (ENU metres, z up, ground z = 0, square map of half-width
``map_half_m``, default 64 km = a 128 km map), a list of ``Plane`` entities and the missiles in flight.
Every tick (1/48 s, the missile control tick) it runs, in this order:

  1. controllers decide (each plane about every 0.5 s, staggered; the command holds in between),
  2. every aircraft advances (wt_overlay.flight),
  3. every missile advances (missile_sim ``create_surface_missile`` runtimes, via the public stepping API;
     aircraft first, so a missile can query the aircraft at both tick ends and in between),
  4. events settle: fuse = kill (D), ground, range / lifetime end, crash, leaving the map,
  5. sensors update: each aircraft's radar, then each RWR.

A controller sees only an ``Observation`` (own state, radar picture, RWR, MAW, missile flames, visual
sightings, map marks of friends and spotted enemies, own missiles in flight). Truth ids stay inside the
manager; ``truth_debug=True`` adds the manager itself to ``Observation.truth`` for debugging.

Missiles are missile_sim runtimes (sibling repository, read-only: its ``src`` goes on ``sys.path``). Each
missile's target is a ``MissileTarget`` proxy that serves the target aircraft's history in missile_sim's
frame (x east, y up, z = -north; time since launch) plus the chaff bundles the target dropped. The shooter's
datalink support is the ``launcher_support`` callback: it holds while the shooter is alive and its radar
still has a track (TWS, extrapolated included, or STT) on the target.

Rules added here that are not game facts (grade D, collected in docs/rl_design.md section 8):
the map size (C), a fuse = a kill, a launch needs a radar track on the target (TWS or STT) and 1 s since the last
launch, the launch state is the aircraft's velocity with the nose along it (no AoA), radars only see enemy
aircraft (friends come from the map), the missile seeker is audible on the RWR once the missile is within the seeker's
receiver range of its target, an MLWS warns during the motor burn within 10 km, a pilot sees the plume of a
missile aimed at him within 30 km with the plane's ``flame_probability`` (drawn once per missile), enemies within 8 km
are seen, a spotted enemy stays on the map 20 s, leaving the map for 15 s is a loss, a missile whose
target died continues and may reacquire a living aircraft, including a friend, and a match with no missile in the air for 60 s while the teams are
more than 120 km apart ends.

Opt-in ``radar_sees_missiles`` (docs/radar_missile_detection_spec.md): the radars also see enemy missiles in flight
(MISSILE_RCS_M2, D; a TWS track slot each), an NCTR radar names them, a missile track is never a launch target and,
unless ``allow_missile_targets``, never locked in STT or selectable as a target by the action masks.

Opt-in ``wreck_s`` (MatchEnv, policy aircraft only): a shot-down aircraft stays a wreck for that many seconds for the
radars and eyes of the planes in ``wreck_viewers`` (keeps its last velocity, falls, stops on the ground; detected and
seen by the usual rules, never a team-map mark). They may track it and fire at it; such a missile is wasted (it never
takes a new target). Missiles already in flight, kills and everyone else's sensors are unchanged.

Opt-in ``airfield`` (docs/airfield_rearm_spec.md, AIRFIELD): a plane whose executed intent is go-home lands at its team's
spawn point when close, low and slow enough, is parked there (``grounded``: invisible to every sensor, not a target, no
crash / out-of-bounds checks; missiles chasing it guide at the touchdown point and may retarget), is rearmed after the
turnaround and takes off again once its intent is no longer go-home. All alive planes parked with no missile in the air
for ``all_grounded_end_s`` ends the match ("all_grounded").
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any

from . import pk, units as units_mod
from .escape import from_enu, to_enu
from .fm import atmosphere
from .flight import SUBSTEP_S, Aircraft, FlightCommand, FlightParams, FuelTank, aircraft_model, fuel_data
from .sensors import Emission, OwnState, RadarSensor, RwrSensor, TargetTruth, world_angles

MAP_HALF_M = 64000.            # 128 km map (C: player reports of top-tier Air RB maps); configurable
TIME_LIMIT_S = 900.
DECISION_TICKS = 24            # 0.5 s
LAUNCH_GAP_S = 1.              # D
FAR_M, IDLE_S = 120000., 60.   # D: stalemate: teams this far apart and no missile in the air this long
OUT_OF_BOUNDS_S = 15.          # D
VISUAL_RANGE_M = 8000.         # D
MAW_RANGE_M = 10000.           # D
FLAME_RANGE_M = 30000.         # D
MAP_HOLD_S = 20.               # D
FRAME_TICKS = 12               # 0.25 s replay frames
TRIM_ROWS_TICKS = 96           # missile_sim keeps every sample; drop old ones so concurrent missiles stay small
CHAFF_LIFETIME_S = 15.         # missile_sim chaff.ChaffSpec.lifetime_s
STT_TRACK = -1                 # Action.fire value meaning "the STT target"
DEFAULT_MISSILE_SIM = Path(__file__).resolve().parents[2]/"missle_sim"
LAUNCH_OPTIONS = dict(observation_mode="sensor_track", loft=True, clutter_model="look_down_angle",
                      clutter_min_depression_deg=2., cw_on_clear_beam=True, require_seeker_lock=True)
KILL_EVENTS = ("fuse",)
# radar_sees_missiles (opt-in): enemy missiles in flight are radar targets with truth id MISSILE_TRUTH+uid.
# MISSILE_RCS_M2 (D) is set so a typical top-tier radar sees a missile at about 70 km (user, C). Reference: the
# APG-63(V)3's MPRF search waveform (mprfSearch, the TWS waveform the simulator flies): 70 km for its 1 m^2 reference
# RCS; range ~ RCS^(1/4), so RCS = 1 m^2 * (70 km / 70 km)^4 = 1 m^2. Over the 19 top-tier aircraft weighted by match
# frequency the median detection range is then 68 km (N011M 46 km ... N035E 114 km). The N035E as the reference would
# give 3 m^2 * (70 / 150)^4 = 0.14 m^2 and a 42 km median.
MISSILE_RCS_M2 = 1.
MISSILE_TRUTH = 1000000        # as the RWR emitter ids of missile seekers
NCTR_RANGE_M = math.inf        # D: an NCTR radar names a missile track at any range, as a tracked aircraft gets its type
# airfield (opt-in, docs/airfield_rearm_spec.md): go home (intent maneuver 10) to the own spawn point, land, rearm after
# the turnaround, take off again. Approach / landing / take-off values are D; the 20 s turnaround is the user's (C).
# approach_m 21 km (was 15 km): a straight-in approach from 8 km high takes about 90 s from the approach radius to the
# landing (user 2026-10-07: ~90 s without the transit). Measured: 76-86 s at full throttle (policy go-home), 90-103 s at
# the scripts' cruise (docs/airfield_rearm_spec.md section 9).
AIRFIELD = dict(approach_m=21000., approach_alt_m=300., approach_ias_kmh=450., land_radius_m=2500., land_max_alt_m=600.,
                land_max_ias_kmh=550., turnaround_s=20., takeoff_alt_m=100., takeoff_ias_kmh=350., all_grounded_end_s=30.)
TAKEOFF_MIN_IAS_KMH = 200.     # the flight model needs 50 m/s (180 km/h) at least


def airfield_settings(config):
    """The ``airfield`` option filled with the AIRFIELD defaults; None (off) stays None. ValueError for unknown keys or
    values that cannot be flown."""
    if config is None:
        return None
    if not isinstance(config, dict) or set(config)-set(AIRFIELD):
        raise ValueError("airfield must be a dict with keys from "+", ".join(AIRFIELD))
    out = dict(AIRFIELD)
    for k, v in config.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 or \
                (v == 0 and k not in ("turnaround_s", "all_grounded_end_s")):
            raise ValueError(f"airfield {k} must be a positive number")
        out[k] = float(v)
    if out["takeoff_ias_kmh"] < TAKEOFF_MIN_IAS_KMH:
        raise ValueError(f"airfield takeoff_ias_kmh must be at least {TAKEOFF_MIN_IAS_KMH:g}")
    return out


# fuel (opt-in, docs/fuel_spec.md): each aircraft takes off with a share ``fraction`` [lo, hi] (drawn per aircraft, own
# generator; the player picks the load in the game, D) of its FM tank capacity (``tanks``: "max" = Mass.MaxFuelMass0, drop
# tanks included; "internal" = the tanks not flagged external). Mass = FM empty mass + fuel + missiles (missile_sim launch
# mass) instead of the spec's mass_factor; flight.FuelTank burns it; empty tanks = no thrust ("flameout" event). The
# airfield turnaround refuels to the initial load. OwnObs.bingo: fuel share of the initial load below ``bingo``.
FUEL = dict(fraction=(.45, 1.), bingo=.15, tanks="max")
MISSILE_MASS_KG = 150.         # D: a missile whose missile_sim profile has no geometry.initial_mass_kg


def fuel_settings(config):
    """The ``fuel`` option filled with the FUEL defaults; None (off) stays None. ValueError for unknown keys or values."""
    if config is None:
        return None
    if not isinstance(config, dict) or set(config)-set(FUEL):
        raise ValueError("fuel must be a dict with keys from "+", ".join(FUEL))
    out = dict(FUEL, **config)
    share = out["fraction"]
    if not isinstance(share, (list, tuple)) or len(share) != 2 or any(
            isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in share) \
            or not 0. < share[0] <= share[1] <= 1.:
        raise ValueError("fuel fraction must be [lo, hi] with 0 < lo <= hi <= 1")
    bingo = out["bingo"]
    if isinstance(bingo, bool) or not isinstance(bingo, (int, float)) or not 0. <= bingo < 1.:
        raise ValueError("fuel bingo must be a number in [0, 1)")
    if out["tanks"] not in ("max", "internal"):
        raise ValueError("fuel tanks must be 'max' or 'internal'")
    out["fraction"], out["bingo"] = (float(share[0]), float(share[1])), float(bingo)
    return out


ASSIST_RULES = ("first_shot",)


def assist_rule_setting(rule):
    """None (an assist to every teammate with a missile at the victim in the last 20 s) or 'first_shot' (_kill)."""
    if rule is not None and rule not in ASSIST_RULES:
        raise ValueError("assist_rule must be None or one of "+", ".join(ASSIST_RULES))
    return rule


# -- missile_sim ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class MissileInfo:
    missile_id: str        # as carried
    profile_id: str        # the missile_sim profile used (alias fallback)
    burn_s: float          # powered phase, from launch
    seeker_range_m: float  # receiver range; the seeker is audible on an RWR inside it (D)
    lifetime_s: float


class MissileLibrary:
    """missile_sim profiles and the few facts the manager needs, loaded read-only from its public API."""

    def __init__(self, root=None):
        root = Path(root or DEFAULT_MISSILE_SIM)
        src = str(root/"src")
        if not (root/"src"/"aim120_model").exists():
            raise FileNotFoundError(f"missile_sim not found at {root}")
        if src not in sys.path:
            sys.path.insert(0, src)
        from aim120_model.chaff import ChaffSpec, Reflector
        from aim120_model.profile_catalog import load_profile_catalog
        from aim120_model.public_api import create_surface_missile
        from aim120_model.target import TargetState
        from missile_gui.library import scan_library
        _, catalog = load_profile_catalog(root)
        profiles, _ = scan_library(catalog["profiles_dir"], root)
        self.profiles = {p["missile_id"]: p for p in profiles}
        self.create, self.TargetState, self.Reflector, self.ChaffSpec = create_surface_missile, TargetState, Reflector, ChaffSpec
        self._info = {}

    def info(self, missile_id) -> MissileInfo:
        info = self._info.get(missile_id)
        if info is None:
            profile_id = missile_id if missile_id in self.profiles else pk.ALIASES.get(missile_id)
            if profile_id not in self.profiles:
                raise KeyError(f"no missile_sim profile for {missile_id}")
            p = self.profiles[profile_id]
            burn = sum(float(s.get("fire_delay_s") or 0.)+s["duration_s"] for s in p["propulsion"]["stages"])
            receiver = ((p["guidance"].get("sensor_model") or {}).get("radar_seeker") or {}).get("receiver") or {}
            if not receiver.get("range_m"):
                raise ValueError(f"{missile_id}: no active seeker range")
            info = self._info[missile_id] = MissileInfo(missile_id, profile_id, burn, float(receiver["range_m"]),
                                                        float(p["performance"]["lifetime_s"]))
        return info

    def profile(self, missile_id):
        return self.profiles[self.info(missile_id).profile_id]

    def is_active(self, profile_id) -> bool:
        """Does the missile_sim profile have an active radar seeker?"""
        p = self.profiles[profile_id]
        return bool(((p["guidance"].get("sensor_model") or {}).get("radar_seeker") or {}).get("active"))


_LIBRARIES = {}


def default_library(root=None) -> MissileLibrary:
    key = str(root or DEFAULT_MISSILE_SIM)
    if key not in _LIBRARIES:
        _LIBRARIES[key] = MissileLibrary(root)
    return _LIBRARIES[key]


_UNITS = []


def equipment_data():
    if not _UNITS:
        _UNITS.append(units_mod.load())
    return _UNITS[0]


# -- commands and observations -------------------------------------------------------------------------------

@dataclass(frozen=True)
class RadarCommand:
    """Radar mode ('off' | 'search' | 'tws' | 'stt'), scan pattern index, scan centre (azimuth from the nose, elevation
    from the horizon) and, for 'stt', the radar track id to lock (a track of the radar picture)."""
    mode: str = "tws"
    pattern: int = 0
    azimuth_deg: float = 0.
    elevation_deg: float = 0.
    stt_track: int | None = None


@dataclass(frozen=True)
class Action:
    """A controller's decision: flight command (held until the next decision), radar command, a launch at a radar
    track id (``STT_TRACK`` for the STT target) and a number of chaff bundles to drop now."""
    flight: FlightCommand | None = None
    radar: RadarCommand | None = None
    fire: int | None = None
    chaff: int = 0


@dataclass(frozen=True)
class OwnObs:
    time_s: float
    team: int
    aircraft: str
    position: tuple
    velocity: tuple
    heading_deg: float
    pitch_deg: float
    roll_deg: float
    speed_mps: float
    altitude_m: float
    aoa_deg: float
    load: float
    engine_percent: float
    missile_id: str | None
    missiles: int
    chaff: int
    radar_mode: str
    stt_state: str | None
    has_maw: bool
    # fuel (opt-in; None / False without it): fuel on board, its share of the initial load, the total mass (empty + fuel
    # + missiles) and whether the share is below the bingo setting. Raw observation only (not in the actor's own vector).
    fuel_kg: float | None = None
    fuel_fraction: float | None = None
    mass_kg: float | None = None
    bingo: bool = False


@dataclass(frozen=True)
class Sighting:
    """A direction (world bearing, compass, and elevation above the horizon) seen by the pilot or a warning system:
    kind 'maw' (missile launch warning), 'flame' (missile plume) or 'aircraft' (visual). ``ref`` is an opaque id
    that stays the same for the same object; ``range_m`` is None where the sense gives none."""
    kind: str
    ref: int
    bearing_deg: float
    elevation_deg: float
    range_m: float | None
    time_s: float


@dataclass(frozen=True)
class MapMark:
    """A map marker: an enemy spotted by a teammate's radar or eyes (``time_s`` = when), or a friend (now)."""
    mark_id: int
    x: float
    y: float
    z: float | None
    time_s: float
    friend: bool = False
    heading_deg: float | None = None


@dataclass(frozen=True)
class OwnShot:
    """One of the pilot's missiles in flight: seconds since launch, which map mark it was fired at (None if that
    enemy has no mark), whether the datalink still feeds it and whether its seeker is on. ``bearing_deg`` (compass),
    ``elevation_deg`` and ``range_m`` place the missile marker; they are None when the pilot's camera does not see it."""
    uid: int
    age_s: float
    target_mark: int | None
    datalink: bool
    active: bool
    bearing_deg: float | None = None
    elevation_deg: float | None = None
    range_m: float | None = None


@dataclass(frozen=True)
class Observation:
    time_s: float
    own: OwnObs
    radar: tuple              # RadarContact: TWS tracks, search blips or the STT contact (enemies only)
    rwr: tuple                # RwrContact
    maw: tuple                # Sighting 'maw'
    flames: tuple             # Sighting 'flame'
    visual: tuple             # Sighting 'aircraft'
    marks: tuple              # MapMark: friends (alive) and spotted enemies
    shots: tuple              # OwnShot
    map_half_m: float
    truth: Any = None
    boxes: tuple = ()
    contrails: tuple = ()
    missile_marks: tuple = ()
    # radar_sees_missiles: per ``radar`` contact the uid of the missile behind it, None for an aircraft. Truth, read only
    # by the action masks and the scripts' choice of targets (never encoded); ``missile_targets``: allow_missile_targets.
    radar_missiles: tuple = ()
    missile_targets: bool = False
    grounded: bool = False    # airfield (opt-in): parked at the own airfield (weapons and chaff masked)


class Camera:
    """Rate-limited third-person camera; compass azimuth and horizon elevation.

    Horizontal FOV and a 16:9 rectilinear projection are D assumptions. ``OPEN_CAMERA``
    is retained only for legacy low-level Engagement callers; MatchEnv uses 90–120°.
    """
    def __init__(self, fov_deg=90., rate_deg_s=180., aspect=16/9):
        self.fov_deg, self.rate_deg_s, self.aspect = fov_deg, rate_deg_s, aspect
        self.bearing_deg = self.elevation_deg = None
        self.want_bearing = self.want_elevation = None
        self.mode = 0

    def point(self, bearing, elevation, mode=0):
        self.want_bearing, self.want_elevation, self.mode = bearing % 360., elevation, mode

    def advance(self, own, dt):
        if self.bearing_deg is None:
            self.bearing_deg, self.elevation_deg = own.heading_deg, own.pitch_deg
        az = own.heading_deg if self.want_bearing is None else self.want_bearing
        el = own.pitch_deg if self.want_elevation is None else self.want_elevation
        da = (az-self.bearing_deg+180.) % 360.-180.
        de = el-self.elevation_deg
        distance = math.hypot(da, de)
        fraction = min(1., self.rate_deg_s*dt/max(distance, 1e-12))
        self.bearing_deg = (self.bearing_deg+da*fraction) % 360.
        self.elevation_deg += de*fraction

    def sees(self, own, bearing_deg, elevation_deg):
        if self.fov_deg >= 360.:
            return True
        az = own.heading_deg if self.bearing_deg is None else self.bearing_deg
        el = own.pitch_deg if self.elevation_deg is None else self.elevation_deg
        # Project the ray onto forward/right/up camera axes (roll stabilised).
        a, e, ce = map(math.radians, (bearing_deg-az, elevation_deg, el))
        forward = math.cos(e)*math.cos(a)*math.cos(ce)+math.sin(e)*math.sin(ce)
        right = math.cos(e)*math.sin(a)
        up = math.sin(e)*math.cos(ce)-math.cos(e)*math.cos(a)*math.sin(ce)
        tangent = math.tan(math.radians(self.fov_deg/2.))
        return forward > 0. and abs(right) <= forward*tangent+1e-12 and abs(up) <= forward*tangent/self.aspect+1e-12


OPEN_CAMERA = Camera(360.)


@dataclass(frozen=True)
class TargetBox:
    ref: int
    bearing_deg: float
    elevation_deg: float
    range_m: float
    closing_speed_mps: float
    aircraft: str
    time_s: float


class LauncherSupport:
    """A copyable callable: snapshot must never retain a closure over the old world."""
    def __init__(self, missile):
        self.missile = missile

    def __call__(self, t, truth):
        m = self.missile
        if not m.shooter.alive:
            return "shooter_dead"
        if not m.target.alive or m.target.ident not in m.shooter.tracked:
            return "track_lost"
        return ""


# -- entities ------------------------------------------------------------------------------------------------

@dataclass
class PlaneSpec:
    aircraft: str
    team: int
    position: tuple
    velocity: tuple
    controller: Any = None            # object with decide(Observation) -> Action | None, optional .phase / .describe()
    missile: str | None = None
    missiles: int = 0
    chaff: int = 0
    rcs_ratio: float = 1.             # chaff bundle RCS / aircraft RCS (0.5, 1 or 2 in the hit-probability model)
    flame_probability: float = .5     # D, as the hit-probability model's P_SEE_BURN
    mass_factor: float = 1.3          # total mass / empty mass
    name: str = ""
    skill: str = ""
    archetype: str = ""
    camera: Camera | None = None      # None: OPEN_CAMERA (no view gating)

    rcs_m2: float = 5.
    home_xy: tuple | None = None      # the team's spawn point (airfield); None: the own spawn position
    enemy_xy: tuple | None = None     # the enemy spawn point (take-off heading); None: the spawn heading


class Plane:
    def __init__(self, ident, spec: PlaneSpec, flight: Aircraft, equipment, radar, rwr, rng):
        self.ident, self.spec, self.team, self.aircraft = ident, spec, spec.team, spec.aircraft
        self.name = spec.name or f"{spec.aircraft}#{ident}"
        self.flight, self.equipment, self.radar, self.rwr, self.rng = flight, equipment, radar, rwr, rng
        self.controller = spec.controller
        self.missile_id, self.missiles, self.chaff, self.rcs_ratio = spec.missile, spec.missiles, spec.chaff, spec.rcs_ratio
        self.rcs_m2 = spec.rcs_m2
        self.flame_p = spec.flame_probability
        self.camera = spec.camera if spec.camera is not None else OPEN_CAMERA
        self.has_maw = bool(equipment and equipment.mlws)
        self.alive = True
        self.death = None                  # dict(cause, time_s, killer)
        self.last_launch = -math.inf
        self.next_decision = 0
        self.own = None
        self.picture = None                # latest RadarPicture (built at the last decision)
        self.rwr_picture = None
        self.rwr_logged = set()
        self.tracked = set()
        self.bundles = []                  # (release time, position, velocity) in missile_sim's frame
        self.oob_s = 0.
        self.phase = ""
        self.kills = self.assists = self.launches = self.chaff_used = 0
        self.phase_time = {}
        self.missile_hist = []             # (shooter ident, t_launch, t_end or None) of missiles aimed at this plane
        # airfield (opt-in): the executor sets want_home (executed maneuver 10); grounded planes do not fly
        self.grounded = self.rearmed = self.want_home = False
        self.ground_t = None
        self.landings = self.rearms = self.takeoffs = 0
        hx, hy = spec.home_xy if spec.home_xy is not None else spec.position[:2]
        self.airfield_xy = (float(hx), float(hy))
        ex, ey = (spec.enemy_xy[0]-hx, spec.enemy_xy[1]-hy) if spec.enemy_xy is not None else spec.velocity[:2]
        self.takeoff_deg = math.degrees(math.atan2(ex, ey)) % 360.

    def state_at(self, time_s):
        """(position, velocity) in ENU; a wreck stays where its last tick left it."""
        f = self.flight
        if time_s <= f.time+1e-9:
            return f.state_at(time_s)
        return f.state.position, (0., 0., 0.)


class MissileFlight:
    def __init__(self, uid, shooter, target, info, t_launch):
        self.uid, self.shooter, self.target, self.info, self.t_launch = uid, shooter, target, info, t_launch
        self.runtime = self.proxy = None
        self.seeker_on = False
        self.done = False
        self.event = None
        self.min_dist2 = math.inf
        self.flame_seen = False
        self.datalink = True
        self.time_s = 0.
        self.pos_enu = (0., 0., 0.)
        self.vel_enu = (0., 0., 0.)
        self.rows_trim = 0
        self.retargeted = False
        self.aware_at = None
        self.lost = None                   # airfield: ENU point where the target landed (guided at like a wreck)
        self.wreck_shot = False            # wreck_s: fired at a wreck (never retargets)


class MissileTarget:
    """What a missile sees of its target: the aircraft's history (missile_sim frame; time = seconds since launch) and
    the chaff the aircraft has dropped. ``rcs_m2`` is 1 so a bundle's RCS is the plain ratio to the aircraft."""
    rcs_m2 = 1.

    def __init__(self, engagement, missile: MissileFlight):
        self.eng, self.m = engagement, missile
        if getattr(engagement, 'seeker_search', None) is not None:
            self.rcs_m2 = missile.target.rcs_m2
        self._state, self._reflector = engagement.library.TargetState, engagement.library.Reflector

    def state_at(self, t):
        if getattr(self.eng, 'seeker_search', None) is not None:
            self.rcs_m2 = self.m.target.rcs_m2
        lost = getattr(self.m, "lost", None)   # airfield: the target landed
        if lost is not None:
            return self._state(from_enu(lost), (0., 0., 0.))
        position, velocity = self.m.target.state_at(self.m.t_launch+t)
        return self._state(from_enu(position), from_enu(velocity))

    def observe_missile(self, t, position, velocity):
        self.m.pos_enu, self.m.vel_enu = to_enu(position), to_enu(velocity)

    def decoys_at(self, t):
        bundles = self.m.target.bundles
        if not bundles:
            return []
        now = self.m.t_launch+t
        spec = self.eng.chaff_specs[self.m.target.ident]
        tau = spec.stop_time_constant_s
        out = []
        release_counts = {}
        for t0, p0, v0 in bundles:
            # Pruning removes complete release times, so each release's ordinal stays stable.
            release_index = release_counts.get(t0, 0)
            release_counts[t0] = release_index+1
            age = now-t0
            rcs = spec.rcs_ratio*spec.envelope(age)
            if getattr(self.eng, 'seeker_search', None) is not None:
                rcs *= self.rcs_m2
            if age < 0 or rcs <= 0:
                continue
            decay = math.exp(-age/tau)
            k = tau*(1-decay)
            out.append(self._reflector("chaff", (p0[0]+v0[0]*k, p0[1]+v0[1]*k, p0[2]+v0[2]*k),
                                       (v0[0]*decay, v0[1]*decay, v0[2]*decay), rcs,
                                       **({'identity': (self.m.target.ident, t0, release_index)}
                                          if getattr(self.eng, 'seeker_search', None) is not None else {})))
        return out


# -- replay --------------------------------------------------------------------------------------------------

class ReplayWriter:
    """JSONL replay: a header line, a frame every 0.25 s, event lines, an end line. With no path the lines are only kept
    in ``lines`` (tests compare them for determinism)."""

    def __init__(self, path=None, keep=True):
        self.path = None if path is None else Path(path)
        self.lines = [] if (keep or path is None) else None
        self._handle = None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.path.open("w")

    def write(self, obj):
        line = json.dumps(obj, separators=(",", ":"))
        if self._handle is not None:
            self._handle.write(line+"\n")
        if self.lines is not None:
            self.lines.append(line)

    def close(self):
        if self._handle is not None:
            self._handle.close()
            self._handle = None


@dataclass
class Result:
    reason: str
    time_s: float
    ticks: int
    teams_alive: tuple
    planes: list
    kills: list
    deaths: list
    launches: int
    missile_errors: int
    phase_time_s: dict
    wall: dict
    radar_wall: dict


# -- observations --------------------------------------------------------------------------------------------

class ObservationBuilder:
    """Builds one aircraft's ``Observation`` from the world: the single place that decides what a pilot can know. Each
    item has its own method, and the view-dependent ones go through ``plane.camera`` (``Camera.sees``; the default
    sees everything, so the observation is not view-gated yet):

      ungated  own state, the radar picture (tracks with range and closure), the RWR picture, the MAW warning, the map
               marks (friends, and enemies spotted by a teammate's radar or eyes);
      gated    visual sightings of aircraft (``_visual``; a gated sighting also does not spot the enemy for the team),
               missile plumes (``_cues``) and own-missile markers (``_shots``).

    Target boxes, contrails and other-missile markers are built by _view_items and gated by the same camera."""

    def __init__(self, engagement):
        self.eng = engagement

    def build(self, plane: Plane) -> Observation:
        eng, t = self.eng, self.eng.time
        own = plane.own
        ground = plane.grounded   # airfield: parked, radar off, eyes and warnings shut; friends, marks and shots stay
        picture = self._radar(plane, t)
        visual = [] if ground else self._visual(plane, t)
        maw, flames = ([], []) if ground else self._cues(plane, t)
        marks = self._marks(plane, t)
        shots = self._shots(plane)
        boxes, contrails, missile_marks = ([], [], []) if ground else self._view_items(plane, visual, t)
        rwr = plane.rwr_picture.contacts if plane.rwr_picture is not None else ()
        extra = {}
        if eng.radar_sees_missiles:
            extra = dict(missile_targets=eng.allow_missile_targets, radar_missiles=() if picture is None else tuple(
                i-MISSILE_TRUTH if i >= MISSILE_TRUTH else None for i in picture.truth_ids))
        if ground:
            extra["grounded"] = True
        return Observation(t, self._own(plane, t), picture.contacts if picture is not None else (), rwr, tuple(maw),
                           tuple(flames), tuple(visual), tuple(marks), tuple(shots), eng.map_half_m,
                           eng if eng.truth_debug else None, tuple(boxes), tuple(contrails), tuple(missile_marks), **extra)

    def _own(self, plane: Plane, t) -> OwnObs:
        own, f, radar = plane.own, plane.flight, plane.radar
        tank = f.fuel
        fuel = {} if tank is None else dict(fuel_kg=tank.kg, fuel_fraction=tank.fraction, mass_kg=tank.mass_kg,
                                            bingo=tank.fraction < (self.eng.fuel or FUEL)["bingo"])
        if plane.grounded:   # airfield: at rest on the ground (the frozen flight state is the touchdown)
            return OwnObs(t, plane.team, plane.aircraft, own.position, own.velocity, own.heading_deg, 0., 0., 0., 0.,
                          0., 1., 0., plane.missile_id, plane.missiles, plane.chaff, "off", None, plane.has_maw, **fuel)
        return OwnObs(t, plane.team, plane.aircraft, own.position, own.velocity, own.heading_deg, own.pitch_deg,
                      own.roll_deg, f.speed, f.altitude, f.state.aoa_deg, f.load, f.state.engine_percent,
                      plane.missile_id, plane.missiles, plane.chaff, radar.mode if radar is not None else "off",
                      radar.stt_state if radar is not None else None, plane.has_maw, **fuel)

    def _radar(self, plane: Plane, t):
        """The radar picture (ungated); its aircraft tracks also put the enemies on the team's map. A missile contact
        (radar_sees_missiles) has no map mark; an NCTR radar names it 'missile' within NCTR_RANGE_M."""
        radar = plane.radar
        picture = radar.picture(t, plane.own) if radar is not None and radar.mode != "off" else None
        plane.picture = picture
        if picture is not None:
            team_marks = self.eng.marks[plane.team]
            for contact, truth in zip(picture.contacts, picture.truth_ids):
                if contact.position is not None and truth < MISSILE_TRUTH and \
                        (self.eng.wreck_s is None or self.eng.planes[truth].alive):   # a wreck is never marked
                    team_marks[truth] = (contact.position[0], contact.position[1], contact.position[2], contact.updated_s)
        if picture is not None:
            if self.eng.radar_sees_missiles:
                nctr = radar.radar.identifies_missiles
                contacts = tuple(replace(c, mark_id=self.eng.mark_ids[ident]) if ident < MISSILE_TRUTH else
                                 replace(c, target_type="missile") if nctr and (c.range_m or 0.) <= NCTR_RANGE_M else c
                                 for c, ident in zip(picture.contacts, picture.truth_ids))
            else:
                contacts = tuple(replace(c, mark_id=self.eng.mark_ids[ident])
                                 for c, ident in zip(picture.contacts, picture.truth_ids))
            picture = replace(picture, contacts=contacts)
            plane.picture = picture
        return picture

    def _visual(self, plane: Plane, t):
        """Enemies within eye range as bearings (camera-gated); each one seen is also marked on the team's map."""
        eng, own = self.eng, plane.own
        px, py, pz = own.position
        team_marks = eng.marks[plane.team]
        out = []
        for q in eng.live:
            if q.team == plane.team or q.grounded:
                continue
            qp = q.own.position
            d = (qp[0]-px, qp[1]-py, qp[2]-pz)
            if d[0]*d[0]+d[1]*d[1]+d[2]*d[2] <= VISUAL_RANGE_M*VISUAL_RANGE_M:
                bearing, elevation = world_angles(d)
                if not plane.camera.sees(own, bearing, elevation):
                    continue
                out.append(Sighting("aircraft", eng.mark_ids[q.ident], bearing, elevation, None, t))
                team_marks[q.ident] = (qp[0], qp[1], qp[2], t)
        for q, qp, _ in eng.wrecks_seen_by(plane):   # wreck_s: seen like an aircraft, never put on the map
            d = (qp[0]-px, qp[1]-py, qp[2]-pz)
            if d[0]*d[0]+d[1]*d[1]+d[2]*d[2] <= VISUAL_RANGE_M*VISUAL_RANGE_M:
                bearing, elevation = world_angles(d)
                if plane.camera.sees(own, bearing, elevation):
                    out.append(Sighting("aircraft", eng.mark_ids[q.ident], bearing, elevation, None, t))
        return out

    def _cues(self, plane: Plane, t):
        """(MAW warnings, missile plumes) of enemy missiles aimed at the plane in their motor burn: the MLWS warning is
        ungated, the plume the pilot happened to see (drawn at launch) is camera-gated."""
        own = plane.own
        px, py, pz = own.position
        maw, flames = [], []
        for m in self.eng.missiles:
            if m.target is not plane or m.done or m.time_s > m.info.burn_s or m.lost is not None:
                continue
            mp = m.pos_enu
            d = (mp[0]-px, mp[1]-py, mp[2]-pz)
            dist = math.sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2])
            bearing, elevation = world_angles(d)
            if plane.has_maw and dist <= MAW_RANGE_M:
                maw.append(Sighting("maw", m.uid, bearing, elevation, None, t))
            if m.flame_seen and dist <= FLAME_RANGE_M and plane.camera.sees(own, bearing, elevation):
                flames.append(Sighting("flame", m.uid, bearing, elevation, None, t))
        for cue in maw+flames:
            m = next(m for m in self.eng.missiles if m.uid == cue.ref)
            if m.aware_at is None:
                m.aware_at = t
        return maw, flames

    def _marks(self, plane: Plane, t):
        """Map markers (ungated): living friends now, enemies spotted by the team within MAP_HOLD_S."""
        eng = self.eng
        marks = []
        for q in eng.live:
            if q.team == plane.team and q is not plane:
                v = q.own.velocity
                marks.append(MapMark(eng.mark_ids[q.ident], q.own.position[0], q.own.position[1], q.own.position[2], t,
                                     True, math.degrees(math.atan2(v[0], v[1])) % 360.))
        team_marks = eng.marks[plane.team]
        for ident in sorted(team_marks):
            x, y, z, seen = team_marks[ident]
            if t-seen <= MAP_HOLD_S and eng.planes[ident].alive and not eng.planes[ident].grounded:
                marks.append(MapMark(eng.mark_ids[ident], x, y, None, seen))
        return marks

    def _shots(self, plane: Plane):
        """The pilot's own missiles in flight; the marker position only when the camera sees the missile."""
        eng, own = self.eng, plane.own
        px, py, pz = own.position
        team_marks = eng.marks[plane.team]
        out = []
        for m in eng.missiles:
            if m.shooter is not plane or m.done:
                continue
            mp = m.pos_enu
            d = (mp[0]-px, mp[1]-py, mp[2]-pz)
            bearing, elevation = world_angles(d)
            seen = plane.camera.sees(own, bearing, elevation)
            out.append(OwnShot(m.uid, m.time_s, eng.mark_ids[m.target.ident] if m.target.ident in team_marks else None,
                               m.datalink, m.seeker_on, bearing if seen else None, elevation if seen else None,
                               math.sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]) if seen else None))
        return out

    def _view_items(self, plane, visual, t):
        eng, own = self.eng, plane.own
        seen = {s.ref for s in visual}
        radar = {c.mark_id: c for c in (plane.picture.contacts if plane.picture else ())
                 if c.kind in ("track", "stt") and c.mark_id is not None}
        boxes, contrails, missiles = [], [], []
        objects = [(q, q.own.position, q.own.velocity) for q in eng.live]
        objects.extend(eng.wrecks_seen_by(plane))   # wreck_s: a wreck is visible like an aircraft
        for q, q_position, q_velocity in objects:
            if q is plane or q.team == plane.team or q.grounded:
                continue
            d = tuple(b-a for a, b in zip(own.position, q_position))
            bearing, elevation = world_angles(d)
            if not plane.camera.sees(own, bearing, elevation):
                continue
            ref = eng.mark_ids[q.ident]
            if q_position[2] >= 9500.:
                contrails.append(Sighting("contrail", ref, bearing, elevation, None, t))
            if ref in seen or ref in radar:
                # A radar box uses the radar estimate; a visual box has HUD range/closure (D).
                c = radar.get(ref)
                if c is not None and c.range_m is not None:
                    boxes.append(TargetBox(ref, c.bearing_deg, c.world_elevation_deg, c.range_m,
                                           c.closing_speed_mps or 0., q.aircraft, t))
                else:
                    distance = math.sqrt(sum(v*v for v in d))
                    closure = -sum((b-a)*r for a,b,r in zip(own.velocity,q_velocity,d))/max(distance,1.)
                    boxes.append(TargetBox(ref,bearing,elevation,distance,closure,q.aircraft,t))
        for m in eng.missiles:
            if m.done or m.shooter is plane:
                continue
            d = tuple(b-a for a,b in zip(own.position,m.pos_enu))
            bearing,elevation = world_angles(d)
            # Marker recognition range is unknown: 10 km is a configurable D limit.
            if sum(v*v for v in d) <= eng.missile_marker_range_m**2 and plane.camera.sees(own,bearing,elevation):
                missiles.append(Sighting("missile_marker",m.uid,bearing,elevation,None,t))
        return boxes, contrails, missiles


# -- the engagement ------------------------------------------------------------------------------------------

class Engagement:
    def __init__(self, specs, seed=0, *, map_half_m=MAP_HALF_M, time_limit_s=TIME_LIMIT_S, library=None, replay=None,
                 truth_debug=False, decision_ticks=DECISION_TICKS, multipath_gain=None,
                 missile_marker_range_m=10000., retarget_dead=True, structural_speed=False, seeker_search=None,
                 radar_sees_missiles=False, allow_missile_targets=False, airfield=None, fuel=None, assist_rule=None,
                 wreck_s=None):
        if not specs:
            raise ValueError("an engagement needs aircraft")
        if wreck_s is not None and (isinstance(wreck_s, bool) or not isinstance(wreck_s, (int, float))
                                    or not math.isfinite(wreck_s) or wreck_s <= 0):
            raise ValueError("wreck_s must be None or a positive number of seconds")
        # opt-in: a shot-down plane stays a wreck this long for the sensors of wreck_viewers (MatchEnv: policy planes)
        self.wreck_s = None if wreck_s is None else float(wreck_s)
        self.wrecks = {}                   # ident -> (death time, ENU position, velocity) when wreck_s is set
        self.wreck_viewers = frozenset()
        self.seed, self.map_half_m, self.time_limit_s = seed, float(map_half_m), float(time_limit_s)
        self.airfield = airfield_settings(airfield)   # opt-in: land, rearm, take off (AIRFIELD)
        self._grounded_since = None
        self.fuel = fuel_settings(fuel)               # opt-in: fuel load, burn, flameout (FUEL, docs/fuel_spec.md)
        self.assist_rule = assist_rule_setting(assist_rule)   # opt-in "first_shot" (_kill); None: the 20 s window rule
        self.assisted = set()                         # (shooter, victim) pairs credited with an assist
        self.retargeted_uids = set()                  # missiles that took a new target after theirs died or landed
        self.library = library or default_library()
        if seeker_search is not None:
            from aim120_model.radar_seeker import validate_seeker_search
            seeker_search = validate_seeker_search(seeker_search)
        self.seeker_search = seeker_search
        self.multipath_gain, self.missile_marker_range_m = multipath_gain, missile_marker_range_m
        self.retarget_dead = retarget_dead
        self.structural_speed = structural_speed   # opt-in: VNE tears the wings off (flight.Aircraft)
        # opt-in: radars also see enemy missiles in flight; a missile track may be locked only with allow_missile_targets
        self.radar_sees_missiles, self.allow_missile_targets = bool(radar_sees_missiles), bool(allow_missile_targets)
        self.replay = replay
        self.truth_debug, self.decision_ticks = truth_debug, decision_ticks
        self.rng = random.Random(f"{seed}:engagement")
        self.tick = 0
        self.planes, self.missiles = [], []
        self.log = []                      # events (dicts), also written to the replay
        self.kills, self.deaths = [], []
        self.missile_errors = 0
        self.launches = 0
        self.retarget_count = 0
        self._uid = 0
        self.last_missile_t = 0.
        self.reason = None
        self.timing = dict(planes=0., missiles=0., radar=0., rwr=0., script=0., other=0.)
        self.radar_timing = {}             # (aircraft, mode) -> [seconds, updates]
        self.rwr_timing = {}               # aircraft -> [seconds, updates]
        self.missile_steps = 0
        self.chaff_specs = {}
        data = equipment_data()
        for ident, spec in enumerate(specs):
            self._add_plane(ident, spec, data)
        order = list(range(len(self.planes)))
        self.rng.shuffle(order)
        self.mark_ids = {p.ident: order[i] for i, p in enumerate(self.planes)}
        self.marks = ({}, {})              # per team: enemy ident -> (x, y, z, t)
        self.live = list(self.planes)
        self.observer = ObservationBuilder(self)
        self._refresh_own()
        if self.replay is not None:
            self.replay.write(self._header())

    # -- construction ----------------------------------------------------------------------------------------

    def _add_plane(self, ident, spec, data):
        tank = self._fuel_tank(ident, spec) if self.fuel is not None else None   # opt-in fuel: mass from the load
        model = aircraft_model(spec.aircraft, mass_factor=spec.mass_factor) if tank is None else \
            aircraft_model(spec.aircraft, mass_kg=tank.mass_kg)
        flight = Aircraft(model, spec.position, spec.velocity, params=FlightParams(), structural_speed=self.structural_speed)
        if tank is not None:
            flight.fuel = tank
        equipment = data.equipment.get(spec.aircraft)
        radar_data = data.radars.get(equipment.radar) if equipment and equipment.radar else None
        rwr_data = data.rwrs.get(equipment.rwr) if equipment and equipment.rwr else None
        rng = random.Random(f"{self.seed}:plane:{ident}")
        radar = RadarSensor(radar_data, owner=ident) if radar_data is not None else None
        rwr = RwrSensor(rwr_data, rng=random.Random(f"{self.seed}:rwr:{ident}")) if rwr_data is not None else None
        plane = Plane(ident, spec, flight, equipment, radar, rwr, rng)
        if spec.missile is not None:
            self.library.info(spec.missile)  # fail early when there is no profile
        if radar is not None:
            radar.set_mode("tws", 0, 0., 0., t=0.)
        plane.next_decision = 0
        plane.stagger = (ident*5) % self.decision_ticks
        self.chaff_specs[ident] = self.library.ChaffSpec(rcs_ratio=spec.rcs_ratio)
        self.planes.append(plane)

    def _fuel_tank(self, ident, spec):
        """fuel (opt-in): the load is ``fraction`` of the FM tank capacity, drawn from the plane's own generator; the
        payload is the missiles carried."""
        data = fuel_data(spec.aircraft)
        capacity = data.max_kg if self.fuel["tanks"] == "max" else data.internal_kg
        share = random.Random(f"{self.seed}:fuel:{ident}").uniform(*self.fuel["fraction"])
        return FuelTank(data, capacity*share, spec.missiles*self.missile_mass(spec.missile))

    def missile_mass(self, missile_id):
        """Launch mass of one missile (missile_sim geometry.initial_mass_kg, A as far as the profile goes), else
        MISSILE_MASS_KG."""
        if missile_id is None:
            return 0.
        mass = (self.library.profile(missile_id).get("geometry") or {}).get("initial_mass_kg")
        return float(mass) if isinstance(mass, (int, float)) and not isinstance(mass, bool) and mass > 0 else MISSILE_MASS_KG

    def _header(self):
        planes = []
        for p in self.planes:
            describe = getattr(p.controller, "describe", None)
            planes.append(dict(id=p.ident, team=p.team, aircraft=p.aircraft, name=p.name, archetype=p.spec.archetype,
                               skill=p.spec.skill, missile=p.missile_id, missiles=p.missiles, chaff=p.chaff,
                               rcs_ratio=p.rcs_ratio, radar=p.radar.radar.id if p.radar else None,
                               rwr=p.rwr.rwr.id if p.rwr else None, mass_kg=round(p.flight.model.mass),
                               # population v2 specs keep the pilot's draws (script_params) for when MatchEnv has
                               # taken the controller off the spec; v1 specs have none, so this stays None there.
                               script=describe() if describe is not None else getattr(p.spec, "script_params", None)))
        header = dict(type="header", version=1, seed=self.seed, map_half_m=self.map_half_m, tick_s=SUBSTEP_S,
                      frame_dt_s=FRAME_TICKS*SUBSTEP_S, time_limit_s=self.time_limit_s, planes=planes,
                      plane_columns=["id", "x", "y", "z", "vx", "vy", "vz", "heading_deg", "missiles", "chaff", "phase",
                                     *(["fuel_kg"] if self.fuel is not None else [])],
                      missile_columns=["uid", "owner", "target", "x", "y", "z", "vx", "vy", "vz", "heading_deg", "age_s",
                                       "seeker", "datalink", *(["tracked_by"] if self.radar_sees_missiles else [])])
        if self.airfield is not None:   # per plane id: airfield x, y and take-off heading
            header["airfield"] = dict(self.airfield, bases=[[round(p.airfield_xy[0], 1), round(p.airfield_xy[1], 1),
                                                             round(p.takeoff_deg, 1)] for p in self.planes])
        if self.fuel is not None:   # per plane id: initial fuel and tank capacity, kg (mass_kg above includes the fuel)
            header["fuel"] = dict(self.fuel, fraction=list(self.fuel["fraction"]), loads=[
                [round(p.flight.fuel.initial_kg, 1), round(fuel_data(p.aircraft).max_kg if self.fuel["tanks"] == "max"
                                                           else fuel_data(p.aircraft).internal_kg, 1)] for p in self.planes])
        return header

    # -- helpers ---------------------------------------------------------------------------------------------

    @property
    def time(self):
        return self.tick*SUBSTEP_S

    def event(self, kind, **fields):
        record = dict(type="event", t=round(self.time, 3), kind=kind, **fields)
        self.log.append(record)
        if self.replay is not None:
            self.replay.write(record)

    def _refresh_own(self):
        for p in self.live:
            f = p.flight
            heading, pitch, roll = f.attitude()
            p.own = OwnState(f.state.position, f.state.velocity, heading, pitch, roll)

    def enemies(self, plane):
        return [q for q in self.live if q.team != plane.team]

    # -- the tick --------------------------------------------------------------------------------------------

    def step(self):
        clock = time.perf_counter
        t0 = clock()
        t = self.time
        self._decide()
        t1 = clock()
        for p in self.live:
            if not p.grounded:
                p.flight.step()
        self.tick += 1   # From here on the world is at the end of the tick: events and sensors carry that time.
        t2 = clock()
        self._advance_missiles()
        t3 = clock()
        self._settle()
        t4 = clock()
        self._sense()
        t5 = clock()
        if self.tick % FRAME_TICKS == 0 and self.replay is not None:
            self._frame()
        self._check_end()
        timing = self.timing
        timing["script"] += t1-t0
        timing["planes"] += t2-t1
        timing["missiles"] += t3-t2
        timing["other"] += (t4-t3)+(clock()-t5)
        return t

    def run(self, until_s=None):
        """Step until the match ends or ``until_s`` (simulated seconds); returns the Result."""
        if self.replay is not None and self.tick == 0:
            self._frame()
        limit = None if until_s is None else round(until_s/SUBSTEP_S)
        while self.reason is None and (limit is None or self.tick < limit):
            self.step()
        return self.result()

    # -- decisions -------------------------------------------------------------------------------------------

    def _decide(self):
        tick = self.tick
        for p in self.live:
            if tick < p.next_decision:
                continue
            p.next_decision = tick+self.decision_ticks if tick else 1+p.stagger
            if p.controller is None:
                continue
            obs = self.observe(p)
            action = p.controller.decide(obs)
            phase = getattr(p.controller, "phase", "")
            if phase != p.phase:
                self.event("phase", plane=p.ident, frm=p.phase, to=phase)
                p.phase = phase
            if action is not None:
                self.apply(p, action)

        for p in self.live:
            advance=getattr(p.controller,"advance",None)
            if advance is not None:
                advance(self,p)

    def apply(self, plane: Plane, action: Action):
        if plane.grounded:   # airfield: on the ground only the go-home rule (want_home) acts
            return
        if action.flight is not None:
            plane.flight.command = action.flight
        if action.radar is not None and plane.radar is not None:
            self._set_radar(plane, action.radar)
        if action.fire is not None:
            self.launch(plane, action.fire)
        if action.chaff:
            self.drop_chaff(plane, action.chaff)

    def _set_radar(self, plane, command: RadarCommand):
        if command.mode == "stt" and self.radar_sees_missiles and not self.allow_missile_targets and \
                (plane.radar.track_truth(command.stt_track) or 0) >= MISSILE_TRUTH:
            command = replace(command, mode="tws", stt_track=None)   # a missile track is not locked: the scan goes on
        try:
            plane.radar.set_mode(command.mode, command.pattern, command.azimuth_deg, command.elevation_deg,
                                 stt_track=command.stt_track, t=self.time)
        except ValueError:
            return
        plane.tracked = plane.radar.tracked_ids()

    # -- observation -----------------------------------------------------------------------------------------

    def observe(self, plane: Plane) -> Observation:
        """What the pilot of ``plane`` can know now (see ObservationBuilder)."""
        return self.observer.build(plane)

    # -- weapons ---------------------------------------------------------------------------------------------

    def launch(self, plane: Plane, track) -> MissileFlight | None:
        """Fire a missile at the target of radar track id ``track`` (``STT_TRACK`` = the STT target) of the picture the
        plane's controller just saw. Needs ammunition, 1 s since the last launch and a TWS track or STT on the target."""
        picture = plane.picture
        if picture is None or plane.missiles <= 0 or plane.missile_id is None or not plane.alive or plane.grounded:
            return None
        if self.time-plane.last_launch < LAUNCH_GAP_S-1e-9:
            return None
        truth = None
        for contact, ident in zip(picture.contacts, picture.truth_ids):
            if (contact.kind == "stt" and track == STT_TRACK) or (contact.track_id is not None and contact.track_id == track):
                truth = ident
                break
        if truth is None or truth not in plane.tracked or truth >= MISSILE_TRUTH:
            return None   # a missile track is never a launch target (no missile-on-missile model)
        from .intent import launch_limit
        contact = next(c for c, ident in zip(picture.contacts, picture.truth_ids) if ident == truth)
        if abs(contact.azimuth_deg) > launch_limit(plane.aircraft, plane.missile_id):
            return None
        target = self.planes[truth]
        if not target.alive and self.wreck_s is not None and plane.ident in self.wreck_viewers and \
                target.team != plane.team and self.wreck_state(truth) is not None:
            return self.fire(plane, target, wreck=True)   # wreck_s: a missile wasted on a wreck
        if not target.alive or target.team == plane.team or target.grounded:
            return None
        return self.fire(plane, target)

    def fire(self, plane: Plane, target: Plane, wreck=False) -> MissileFlight:
        """Create the missile (no track or ammunition checks; ``launch`` makes them). ``wreck``: fired at a wreck
        (wreck_s); it never takes a new target."""
        info = self.library.info(plane.missile_id)
        m = MissileFlight(self._uid, plane, target, info, self.time)
        m.wreck_shot = bool(wreck)
        self._uid += 1
        state = plane.flight.state
        v = state.velocity
        speed = math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2])
        pitch = math.degrees(math.asin(max(-1., min(1., v[2]/speed))))
        heading = math.degrees(math.atan2(-v[1], v[0]))  # missile_sim: heading 0 = +x (east), positive toward +z (south)
        m.proxy = MissileTarget(self, m)
        m.flame_seen = self.rng.random() < target.flame_p
        m.runtime = self.library.create(
            self.library.profile(plane.missile_id), launch_position_m=from_enu(state.position),
            launch_velocity_mps=from_enu(v), launch_pitch_deg=pitch, launch_heading_deg=heading, target=m.proxy,
            launcher_support=LauncherSupport(m), multipath_gain=self.multipath_gain,
            **({"seeker_search": self.seeker_search} if self.seeker_search is not None else {}), **LAUNCH_OPTIONS)
        m.pos_enu, m.vel_enu = state.position, v
        plane.missiles -= 1
        if plane.flight.fuel is not None:   # opt-in fuel: the missile's mass leaves with it
            tank = plane.flight.fuel
            tank.payload_kg = max(0., tank.payload_kg-self.missile_mass(plane.missile_id))
            tank.sync(plane.flight.model)
        plane.launches += 1
        plane.last_launch = self.time
        self.launches += 1
        self.missiles.append(m)
        target.missile_hist.append([plane.ident, self.time, None])
        m.hist = target.missile_hist[-1]
        mode = plane.radar.mode if plane.radar is not None else "off"
        rel = (state.position[0]-target.flight.state.position[0], state.position[1]-target.flight.state.position[1],
               state.position[2]-target.flight.state.position[2])
        self.event("launch", uid=m.uid, shooter=plane.ident, target=target.ident, missile=plane.missile_id,
                   mode=mode, range_m=round(math.sqrt(rel[0]**2+rel[1]**2+rel[2]**2)), left=plane.missiles,
                   flame_seen=m.flame_seen, altitude_m=state.position[2],
                   mach=speed/atmosphere(state.position[2])[1])
        self.last_missile_t = self.time
        return m

    def drop_chaff(self, plane: Plane, n: int):
        n = min(int(n), plane.chaff)
        if n <= 0 or plane.grounded:
            return 0
        t = self.time
        state = plane.flight.state
        p, v = from_enu(state.position), from_enu(state.velocity)
        spec = self.chaff_specs[plane.ident]
        eject = (v[0], v[1]-spec.eject_speed_mps, v[2])
        for _ in range(n):
            plane.bundles.append((t, p, eject))
        plane.chaff -= n
        plane.chaff_used += n
        horizon = t-CHAFF_LIFETIME_S
        if plane.bundles[0][0] < horizon:
            plane.bundles = [b for b in plane.bundles if b[0] >= horizon]
        self.event("chaff", plane=plane.ident, n=n, left=plane.chaff)
        return n

    # -- missiles and events ---------------------------------------------------------------------------------

    def _advance_missiles(self):
        finished = []
        for m in self.missiles:
            if m.done:
                continue
            if self.retarget_dead and (not m.target.alive or m.lost is not None) and not m.wreck_shot:
                self._retarget(m)
            try:
                m.runtime.step()
            except (ValueError, ArithmeticError) as exc:   # missile_sim's numerical or input failures; counted and logged
                self.missile_errors += 1
                m.done, m.event = True, "error"
                self.event("missile_error", uid=m.uid, error=f"{type(exc).__name__}: {exc}")
                finished.append(m)
                continue
            self.missile_steps += 1
            rt = m.runtime
            state = rt.state
            m.time_s = rt.time_s
            m.pos_enu, m.vel_enu = (state[0], -state[2], state[1]), (state[3], -state[5], state[4])
            tp = m.target.flight.state.position if m.lost is None else m.lost
            d = (m.pos_enu[0]-tp[0], m.pos_enu[1]-tp[1], m.pos_enu[2]-tp[2])
            d2 = d[0]*d[0]+d[1]*d[1]+d[2]*d[2]
            if d2 < m.min_dist2:
                m.min_dist2 = d2
            if not m.seeker_on and d2 <= m.info.seeker_range_m**2:
                m.seeker_on = True
                self.event("seeker_on", uid=m.uid, target=m.target.ident, range_m=round(math.sqrt(d2)))
            provider = getattr(rt, "provider", None)
            if m.datalink and provider is not None and not provider.datalink_connected:
                m.datalink = False
                reason = getattr(provider, "datalink_lost_reason", "") or (
                    "seeker_track" if getattr(provider, "radar_has_tracked_once", False) else "other")
                self.event("datalink_lost", uid=m.uid, shooter=m.shooter.ident, target=m.target.ident, reason=reason)
            m.rows_trim += 1
            if m.rows_trim >= TRIM_ROWS_TICKS:
                m.rows_trim = 0
                rows = getattr(rt, "_rows", None)
                if rows is not None and len(rows) > 1:
                    del rows[:-1]  # Only the latest sample is read back; the rest is history nobody uses.
            if rt.done:
                m.done, m.event = True, rt.event
                finished.append(m)
        self._finished = finished

    def _settle(self):
        finished = getattr(self, "_finished", [])
        self._finished = []
        t = self.time
        for m in finished:
            m.hist[2] = t
            self.event("missile_end", uid=m.uid, shooter=m.shooter.ident, target=m.target.ident, result=m.event,
                       miss_m=round(math.sqrt(m.min_dist2), 1), flight_s=round(m.time_s, 2))
            if m.event in KILL_EVENTS and m.target.alive and m.lost is None:
                self._kill(m.target, m.shooter, "missile", m)
            m.runtime = m.proxy = None  # free the sample history
        if self.fuel is not None:
            self._flameouts()
        for p in self.live:
            if p.grounded:
                continue
            if p.flight.crashed:
                self._kill(p, None, "crash", None)
            elif p.flight.overspeed:
                self._kill(p, None, "overspeed", None)
            elif abs(p.flight.state.position[0]) > self.map_half_m or abs(p.flight.state.position[1]) > self.map_half_m:
                p.oob_s += SUBSTEP_S
                if p.oob_s >= OUT_OF_BOUNDS_S:
                    self._kill(p, None, "out_of_bounds", None)
            else:
                p.oob_s = 0.
        if self.airfield is not None:
            self._airfield()
        dead = [m for m in self.missiles if m.done or (not self.retarget_dead and (not m.target.alive or m.lost is not None))]
        if dead:
            for m in dead:
                if not m.done:
                    m.done, m.event = True, "target_dead" if m.lost is None else "target_lost"
                    self.event("missile_end", uid=m.uid, shooter=m.shooter.ident, target=m.target.ident,
                               result=m.event, miss_m=round(math.sqrt(m.min_dist2), 1), flight_s=round(m.time_s, 2))
                    m.hist[2] = t
                    m.runtime = m.proxy = None
            self.missiles = [m for m in self.missiles if not m.done]

    def _retarget(self, m):
        x,y,z,w=m.runtime.state[6:10]  # public runtime state: quaternion xyzw
        nose=to_enu((1-2*(y*y+z*z),2*(x*y+w*z),2*(x*z-w*y)))
        seeker = ((self.library.profile(m.info.missile_id)["guidance"].get("sensor_model") or {}).get("radar_seeker") or {})
        half = float(seeker.get("angle_max_deg") or 60.)
        candidates = []
        for p in self.live:
            if p.grounded:
                continue
            d = tuple(b-a for a,b in zip(m.pos_enu,p.flight.state.position))
            distance = math.sqrt(sum(x*x for x in d))
            if distance > 1. and sum(a*b for a,b in zip(d,nose))/distance >= math.cos(math.radians(half)):
                candidates.append((distance,p.ident,p))
        if candidates:
            old = m.target
            m.hist[2] = self.time
            m.target = min(candidates, key=lambda item:item[:2])[2]
            m.target.missile_hist.append([m.shooter.ident, self.time, None])
            m.hist = m.target.missile_hist[-1]
            m.retargeted = True
            self.retargeted_uids.add(m.uid)   # rl_env retarget_kill_reward: its kill is credited apart
            m.min_dist2, m.aware_at, m.lost = math.inf, None, None
            self.retarget_count += 1
            self.event("retarget", uid=m.uid, old=old.ident, target=m.target.ident,
                       friendly=m.target.team == m.shooter.team)

    def _kill(self, victim: Plane, killer: Plane | None, cause: str, missile: MissileFlight | None):
        if not victim.alive:
            return
        t = self.time
        victim.alive = False
        victim.death = dict(cause=cause, time_s=round(t, 3), killer=None if killer is None else killer.ident)
        self.live = [p for p in self.live if p.alive]
        if self.wreck_s is not None:
            state = victim.flight.state
            self.wrecks[victim.ident] = (t, tuple(state.position), tuple(state.velocity))
        record = dict(victim=victim.ident, cause=cause, killer=None if killer is None else killer.ident,
                      time_s=round(t, 3), uid=None if missile is None else missile.uid)
        self.deaths.append(record)
        friendly = killer is not None and killer.team == victim.team
        record["friendly_fire"] = friendly
        if friendly:
            self.event("friendly_fire", **record)
        if killer is not None and not friendly:
            killer.kills += 1
            self.kills.append(record)
            self.event("kill", **record)
            # Assist: a teammate of the killer had a missile in flight at the victim in the last 20 s. assist_rule
            # "first_shot" (opt-in): only a missile aimed at the victim before the killing missile was (its launch, or
            # its retarget; without a missile object the kill time), and once per (shooter, victim) in the match.
            first = self.assist_rule == "first_shot"
            shot = (t if missile is None else missile.hist[1]) if first else None
            credited = set()
            for shooter, t_launch, t_end in victim.missile_hist:
                other = self.planes[shooter]
                if other.team == killer.team and other is not killer and shooter not in credited \
                        and (t_end is None or t_end >= t-20.) \
                        and (not first or (t_launch < shot and (shooter, victim.ident) not in self.assisted)):
                    credited.add(shooter)
                    if first:
                        self.assisted.add((shooter, victim.ident))
                    other.assists += 1
                    self.event("assist", plane=shooter, victim=victim.ident)
        f = victim.flight
        self.event("death", plane=victim.ident, cause=cause, killer=record["killer"],
                   altitude_m=f.altitude, speed_mps=f.speed, energy_height_m=f.altitude+f.speed**2/(2*9.80665),
                   load=f.load)
        if missile is not None:
            self.event("reaction_window", uid=missile.uid, victim=victim.ident,
                       aware_at=missile.aware_at, impact_at=t,
                       reaction_s=None if missile.aware_at is None else t-missile.aware_at)

    # -- wrecks (opt-in) -------------------------------------------------------------------------------------

    def wreck_state(self, ident, t=None):
        """wreck_s: (ENU position, velocity) of the wreck of shot-down ``ident`` at ``t`` (default now), None when it
        is not a wreck or has gone (more than wreck_s after the death). It keeps its last velocity under gravity and
        stops on the ground (D)."""
        w = self.wrecks.get(ident)
        if w is None:
            return None
        t0, (x, y, z), (vx, vy, vz) = w
        tau = (self.time if t is None else t)-t0
        if tau < -1e-9 or tau > self.wreck_s+1e-9:
            return None
        tau = max(0., tau)
        if z <= 0.:
            return (x, y, 0.), (0., 0., 0.)
        g = 9.80665
        ground = (vz+math.sqrt(vz*vz+2.*g*z))/g
        if tau >= ground:
            return (x+vx*ground, y+vy*ground, 0.), (0., 0., 0.)
        return (x+vx*tau, y+vy*tau, z+vz*tau-.5*g*tau*tau), (vx, vy, vz-g*tau)

    def _wreck_truths(self):
        """Per team, the TargetTruth of each wreck still there (radar targets of the wreck viewers)."""
        out = ([], [])
        for ident in sorted(self.wrecks):
            state = self.wreck_state(ident)
            if state is not None:
                q = self.planes[ident]
                out[q.team].append(TargetTruth(ident, state[0], state[1], q.rcs_m2))
        return out

    def wrecks_seen_by(self, plane):
        """(plane, position, velocity) of the enemy wrecks ``plane`` may see (it is a wreck viewer); [] otherwise."""
        if self.wreck_s is None or not self.wrecks or plane.ident not in self.wreck_viewers:
            return []
        out = []
        for ident in sorted(self.wrecks):
            q = self.planes[ident]
            if q.team != plane.team:
                state = self.wreck_state(ident)
                if state is not None:
                    out.append((q, state[0], state[1]))
        return out

    # -- fuel (opt-in) ---------------------------------------------------------------------------------------

    def _flameouts(self):
        """A 'flameout' event when a flying plane's tanks run dry (flight.FuelTank); it glides on with no thrust."""
        for p in self.live:
            tank = p.flight.fuel
            if tank is not None and tank.out and not tank.logged and not p.grounded:
                tank.logged = True
                f = p.flight
                self.event("flameout", plane=p.ident, altitude_m=round(f.altitude, 1), speed_mps=round(f.speed, 1),
                           ias_kmh=round(f.indicated()*3.6))

    # -- airfield (opt-in) -----------------------------------------------------------------------------------

    def _airfield(self):
        """Land a plane whose executed intent is go-home (``want_home``) near its airfield, low and slow enough; refill
        missiles and chaff ``turnaround_s`` after the landing; then take off as soon as the intent is no longer go-home."""
        a, t = self.airfield, self.time
        for p in self.live:
            if p.grounded:
                if not p.rearmed and t-p.ground_t >= a["turnaround_s"]-1e-9:
                    p.missiles, p.chaff, p.rearmed = p.spec.missiles, p.spec.chaff, True
                    p.rearms += 1
                    tank = p.flight.fuel
                    if tank is not None:   # opt-in fuel: refuelled to the initial load, payload back to full
                        tank.refuel(p.flight.model, p.missiles*self.missile_mass(p.missile_id))
                    self.event("rearm", plane=p.ident, missiles=p.missiles, chaff=p.chaff,
                               **({} if tank is None else dict(fuel_kg=round(tank.kg, 1))))
                if p.rearmed and not p.want_home:
                    self._takeoff(p)
            elif p.want_home:
                f = p.flight
                x, y, z = f.state.position
                if z <= a["land_max_alt_m"] and math.hypot(x-p.airfield_xy[0], y-p.airfield_xy[1]) <= a["land_radius_m"] \
                        and f.indicated()*3.6 <= a["land_max_ias_kmh"]:
                    self._land(p)

    def _land(self, p):
        """Park ``p`` on its airfield: radar off, gone from every radar, map and seeker (missiles chasing it guide at the
        touchdown point like at a wreck and may retarget); its own missiles fly on without datalink."""
        t, f = self.time, p.flight
        x, y, z = f.state.position
        p.grounded, p.rearmed, p.ground_t, p.oob_s = True, False, t, 0.
        p.landings += 1
        if p.radar is not None:
            p.radar.set_mode("off")
        p.tracked, p.picture, p.rwr_picture = set(), None, None
        for q in self.live:
            if q.team != p.team and q.radar is not None:
                q.radar.forget(p.ident)
                if p.ident in q.tracked:
                    q.tracked = q.tracked-{p.ident}
                    self.event("track_lost", plane=q.ident, target=p.ident, supporting=any(
                        m.shooter is q and m.target is p and not m.done for m in self.missiles))
        self.marks[1-p.team].pop(p.ident, None)
        for m in self.missiles:
            if m.target is p and not m.done:
                m.lost = f.state.position
                m.hist[2] = t
                m.hist = [m.shooter.ident, m.t_launch, t]   # detached: no longer counted as a missile at p
        p.own = OwnState((p.airfield_xy[0], p.airfield_xy[1], 0.), (0., 0., 0.), p.takeoff_deg)
        self.event("landing", plane=p.ident, missiles=p.missiles, chaff=p.chaff, altitude_m=round(z),
                   ias_kmh=round(f.indicated()*3.6), distance_m=round(math.hypot(x-p.airfield_xy[0], y-p.airfield_xy[1])))

    def _takeoff(self, p):
        """A fresh flight model over the airfield at takeoff_alt_m / takeoff_ias_kmh, heading for the enemy spawn."""
        a, old = self.airfield, p.flight
        alt = a["takeoff_alt_m"]
        tas = a["takeoff_ias_kmh"]/3.6/math.sqrt(atmosphere(alt)[0]/1.225)
        h = math.radians(p.takeoff_deg)
        p.flight = Aircraft(old.model, (p.airfield_xy[0], p.airfield_xy[1], alt), (tas*math.sin(h), tas*math.cos(h), 0.),
                            params=old.params, t0=self.time, structural_speed=self.structural_speed)
        p.flight.faults = old.faults
        if old.fuel is not None:   # opt-in fuel: the tank (refuelled at the rearm) stays with the plane
            p.flight.fuel = old.fuel
        p.flight.command = FlightCommand(heading_deg=p.takeoff_deg, altitude_m=alt)
        p.grounded, p.rearmed, p.ground_t = False, False, None
        p.takeoffs += 1
        if p.radar is not None:
            p.radar.set_mode("tws", 0, 0., 0., t=self.time)
        self.event("takeoff", plane=p.ident, missiles=p.missiles, chaff=p.chaff)

    # -- sensors ---------------------------------------------------------------------------------------------

    def _sense(self):
        clock = time.perf_counter
        t = self.time
        live = self.live
        t0 = clock()
        truths = ([], [])
        for p in live:
            if p.grounded:   # airfield: p.own was set at the landing; no sensor sees a parked plane
                continue
            f = p.flight
            heading, pitch, roll = f.attitude()
            p.own = own = OwnState(f.state.position, f.state.velocity, heading, pitch, roll)
            truths[p.team].append(TargetTruth(p.ident, own.position, own.velocity, p.rcs_m2))
        if self.radar_sees_missiles:
            for m in self.missiles:
                if not m.done:
                    truths[m.shooter.team].append(TargetTruth(MISSILE_TRUTH+m.uid, m.pos_enu, m.vel_enu, MISSILE_RCS_M2))
        radar_timing = self.radar_timing
        wrecks = self._wreck_truths() if self.wrecks and self.wreck_viewers else None
        for p in live:
            radar = p.radar
            if radar is None or radar.mode == "off":
                continue
            a = clock()
            targets = truths[1-p.team]
            if wrecks is not None and wrecks[1-p.team] and p.ident in self.wreck_viewers:
                targets = targets+wrecks[1-p.team]   # wreck_s: a wreck is a radar target for the viewers
            radar.update(t, SUBSTEP_S, p.own, targets, report=False)
            tracked = radar.tracked_ids()
            if tracked != p.tracked:
                for gone in sorted(p.tracked-tracked):
                    if gone < MISSILE_TRUTH:
                        self.event("track_lost", plane=p.ident, target=gone, supporting=any(
                            m.shooter is p and m.target.ident == gone and not m.done for m in self.missiles))
                for new in sorted(tracked-p.tracked):
                    if new >= MISSILE_TRUTH:
                        self.event("radar_missile_track", plane=p.ident, uid=new-MISSILE_TRUTH)
                p.tracked = tracked
            key = (p.aircraft, radar.mode)
            slot = radar_timing.get(key)
            if slot is None:
                slot = radar_timing[key] = [0., 0]
            slot[0] += clock()-a
            slot[1] += 1
        t1 = clock()
        emissions = {}
        receivers = ([], [])
        for p in live:
            if p.rwr is not None and not p.grounded:
                receivers[p.team].append((p.ident, p.own.position))
        for p in live:
            radar = p.radar
            if radar is None or radar.mode == "off" or not receivers[1-p.team]:
                continue
            kind = radar.emission_kind
            if kind is None:
                continue
            for rid in radar.illuminated(t, p.own, receivers[1-p.team], SUBSTEP_S):
                emissions.setdefault(rid, []).append(Emission(p.ident, p.own.position, kind, radar.band, True, radar.radar.id))
        for m in self.missiles:
            if m.seeker_on and not m.done and m.target.alive and m.lost is None:
                emissions.setdefault(m.target.ident, []).append(
                    Emission(1000000+m.uid, m.pos_enu, "missile", 8, True, None))
        rwr_timing = self.rwr_timing
        for p in live:
            rwr = p.rwr
            if rwr is None:
                continue
            mine = emissions.get(p.ident)
            last = p.rwr_picture
            if mine is None and (last is None or not last.contacts):
                continue
            a = clock()
            picture = rwr.update(t, p.own, mine or ())
            p.rwr_picture = picture
            for c, emitter in zip(picture.contacts, picture.truth_ids):
                if c.new and c.illuminated and (c.missile_warning or c.tracking) and (p.ident, c.contact_id) not in p.rwr_logged:
                    p.rwr_logged.add((p.ident, c.contact_id))
                    if c.missile_warning and emitter >= 1000000:
                        m = next((m for m in self.missiles if m.uid == emitter-1000000), None)
                        if m is not None and m.aware_at is None:
                            m.aware_at = t
                    self.event("rwr", plane=p.ident, warning="missile" if c.missile_warning else "lock",
                               bearing_deg=round(c.bearing_deg, 1), emitter=emitter if emitter < 1000000 else None,
                               missile_uid=emitter-1000000 if emitter >= 1000000 else None)
            slot = rwr_timing.get(p.aircraft)
            if slot is None:
                slot = rwr_timing[p.aircraft] = [0., 0]
            slot[0] += clock()-a
            slot[1] += 1
        t2 = clock()
        self.timing["radar"] += t1-t0
        self.timing["rwr"] += t2-t1
        for p in live:
            phase = "ground" if p.grounded else p.phase
            p.phase_time[phase] = p.phase_time.get(phase, 0.)+SUBSTEP_S

    # -- replay and end --------------------------------------------------------------------------------------

    def _frame(self):
        r1 = lambda x: round(x, 1)  # noqa: E731
        planes = []
        for p in self.live:
            if p.grounded:
                pos, v, phase = p.own.position, p.own.velocity, "ground"
            else:
                s = p.flight.state
                pos, v, phase = s.position, s.velocity, p.phase
            planes.append([p.ident, r1(pos[0]), r1(pos[1]), r1(pos[2]), r1(v[0]), r1(v[1]), r1(v[2]),
                           r1(p.own.heading_deg) % 360., p.missiles, p.chaff, phase,
                           *([r1(p.flight.fuel.kg)] if self.fuel is not None else [])])   # fuel (opt-in): fuel_kg
        missiles = [[m.uid, m.shooter.ident, m.target.ident, r1(m.pos_enu[0]), r1(m.pos_enu[1]), r1(m.pos_enu[2]),
                     r1(m.vel_enu[0]), r1(m.vel_enu[1]), r1(m.vel_enu[2]),
                     r1(math.degrees(math.atan2(m.vel_enu[0], m.vel_enu[1])) % 360.) % 360., round(m.time_s, 2), int(m.seeker_on),
                     int(m.datalink), *([self.tracked_by(m)] if self.radar_sees_missiles else [])]
                    for m in self.missiles if not m.done]
        self.replay.write(dict(type="frame", t=round(self.time, 3), planes=planes, missiles=missiles))

    def tracked_by(self, m):
        """Idents of the living aircraft whose radar holds a track (TWS or STT) on missile ``m``."""
        return [p.ident for p in self.live if MISSILE_TRUTH+m.uid in p.tracked]

    def alive_counts(self):
        counts = [0, 0]
        for p in self.live:
            counts[p.team] += 1
        return tuple(counts)

    def _check_end(self):
        if self.reason is not None:
            return
        counts = self.alive_counts()
        if min(counts) == 0 and not self.missiles:
            self.reason = "annihilation" if max(counts) > 0 else "mutual_annihilation"
        elif self.time >= self.time_limit_s-1e-9:
            self.reason = "time_limit"
        elif self.tick % 48 == 0:
            if self.missiles:
                self.last_missile_t = self.time
            elif self.time-self.last_missile_t >= IDLE_S:
                nearest = min(math.dist(a.own.position, b.own.position) for a in self.live if a.team == 0
                              for b in self.live if b.team == 1)
                if nearest >= FAR_M:
                    self.reason = "stalemate"
        if self.reason is None and self.airfield is not None:
            # airfield: everyone alive parked and no missile in the air for all_grounded_end_s ends the match early
            if self.missiles or not all(p.grounded for p in self.live):
                self._grounded_since = None
            else:
                if self._grounded_since is None:
                    self._grounded_since = self.time
                if self.time-self._grounded_since >= self.airfield["all_grounded_end_s"]-1e-9:
                    self.reason = "all_grounded"
        if self.reason is not None:
            self.event("end", reason=self.reason)
            if self.replay is not None:
                self._frame()
                self.replay.write(dict(type="end", reason=self.reason, t=round(self.time, 3),
                                       result=self.summary()))
                self.replay.close()

    def summary(self):
        planes = [dict(id=p.ident, alive=p.alive, kills=p.kills, assists=p.assists, launches=p.launches,
                       chaff_used=p.chaff_used, missiles_left=p.missiles, death=p.death, fm_faults=p.flight.faults)
                  for p in self.planes]
        if self.airfield is not None:
            for row, p in zip(planes, self.planes):
                row.update(grounded=p.grounded, landings=p.landings, rearms=p.rearms, takeoffs=p.takeoffs)
        return dict(teams_alive=list(self.alive_counts()), kills=self.kills, deaths=self.deaths, launches=self.launches,
                    planes=planes)

    def result(self) -> Result:
        phase_time = {}
        for p in self.planes:
            for phase, secs in p.phase_time.items():
                phase_time[phase] = phase_time.get(phase, 0.)+secs
        wall = dict(self.timing)
        wall["total"] = sum(wall.values())
        return Result(self.reason or "running", self.time, self.tick, self.alive_counts(), self.summary()["planes"],
                      self.kills, self.deaths, self.launches, self.missile_errors, phase_time, wall,
                      {f"{a}/{mode}": tuple(v) for (a, mode), v in sorted(self.radar_timing.items())})
