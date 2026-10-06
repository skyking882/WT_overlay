"""Command-driven aircraft for the engagement simulator, built on the FM evader's physics.

``advance`` is the force/attitude integrator that ``escape.FMEvader._fly`` used inline (roll the lift
vector toward the demanded acceleration at a limited rate, demand load capped by the crew limit and an
AoA ceiling, first-order AoA / engine / airbrake response, tabulated thrust / drag / lift from
``turn.ManeuverModel``). FMEvader now calls it too and its output is bit-identical to before
(tests/test_flight.py pins that). ``Aircraft`` wraps it with a ``FlightCommand`` interface and a
rolling history so missiles can query it between ticks.

Frame: ENU metres, z up, flat ground at z = 0 (escape.to_enu converts missile_sim's x, up, z frame).
Headings are compass degrees, clockwise from north. One tick is ``SUBSTEP_S`` = 1/48 s, the missile
control tick; AoA / roll time constants are at least 0.35 s.

Pilot rules (project assumptions, grade D, listed in docs/rl_design.md section 8): the control law
below is not Instructor physics. It demands a turn rate of angle error / ``turn_time_constant_s``,
reverses through a horizontal turn when the target is more than 90 degrees off, captures an altitude
with a vertical speed of error / ALT_TAU_S and refuses to descend into the ground floor.

Two command types drive an ``Aircraft``. ``FlightCommand`` is mouse-aim-like: a wanted direction, altitude and speed,
with the load cap the command gives (9 g by default, a crew limit; ``Aircraft(limit_structure=True)`` also applies the
airframe limit below). ``KeyboardCommand`` is the coarse keyboard: roll, pitch, throttle in {-1, 0, +1} and the airbrake
on or off, bang-bang at full authority unless ``authority`` < 1. It runs ``turn.ManeuverModel.step`` with the project's
``KeyboardTurnSettings`` (the same model the keyboard turn guidance plans with), so a plan found there replays here.

Airframe load limits come from each aircraft's FM file (``Structure``): Aerodynamics.WingPlane.Strength.CritOverload
(the wing's critical force in N, [negative, positive]) divided by the current weight, times the lower value of each pair of
Instructor.overloadMult when Instructor.limitOverload is true, capped by Instructor.loadFactorLimit when
Instructor.limitLoadfactor is true. Not used: Passport.IAS.maxRollRate* (the tables look the same across aircraft) and the
FlyByWire presets (they hold control modes, no numbers). The roll rate, response times and throttle rate of the keyboard
model are KeyboardTurnSettings' defaults (D). Without an FM file the defaults ``+9 / -3`` g apply (D).
"""
from __future__ import annotations

from collections import deque
import copy
from dataclasses import dataclass
import json
import math

from .contracts import G, KeyboardTurnSettings
from .fm import atmosphere
from .turn import (Action as KeyAction, ManeuverModel, Motion, add, cross, dot, norm, rotate, scale, transport, unit)

SUBSTEP_S = 1/48          # One missile control tick.
ALT_TAU_S = 6.            # Altitude error -> vertical speed (D).
SPEED_PI = (6., 2.)       # Autothrottle gains: percent per (m/s) and percent per (m/s s) of speed error (D).
BRAKE_EXCESS_MPS = 15.    # Speed above target at which an allowed airbrake opens (D).
FLOOR_PULL_OUT = 1.5      # Margin on the circular pull-out height above the floor (as escape.FMEvader).
CLIMB_OUT_DEG = 10.       # Flight path angle of the climb out when below the floor (D).
PULL_OUT_SINK_MPS = 5.    # Sinking faster than this while low: pull up wings level even if the command is level (D).
MIN_STEP_SPEED_MPS = 50.  # ManeuverModel's lower speed bound; below it the aircraft falls ballistically.
MACH_LIMIT = 2.         # Autothrottle ceiling (D): the FM tables end at Mach 2.35 and fighters do not fly past about 2 in the game.
MACH_BAND_MPS = 20.     # Throttle fades out over this much speed below the limit.
ENGINE_MARGIN_MPS = 8.  # Stay this far below the engine table's top TAS.
MAX_ALTITUDE_M = 19500.  # ManeuverModel's atmosphere ends at 20 km.
# Structural speed (opt-in, Aircraft(structural_speed=True)): the FM file's VNE (indicated, km/h) tears the wings off
# (user, 2026-10-06: in the game exceeding VNE rips the aircraft apart); above VneControl the controls stiffen.
OVERSPEED_TOLERANCE_S = 1.   # time above VNE before the wings fail (D)
VNE_GOVERNOR = .97           # the pilot pulls the throttle back approaching this share of VNE and brakes above it (D)
VNE_GOVERNOR_BAND_KMH = 60.  # throttle fades to idle over this much indicated speed below the governor point (D)
VNE_CONTROL_LOSS = .6        # share of the load factor above 1 g lost between VneControl and VNE (D)
CEILING_BAND_M = 1500.  # Climb angle fades out over this much altitude below the ceiling.


@dataclass(frozen=True)
class FlightState:
    position: tuple[float, float, float]
    velocity: tuple[float, float, float]
    normal: tuple[float, float, float]
    aoa_deg: float
    engine_percent: float = 110.
    airbrake: float = 0.


def alpha_for(model, altitude, speed, load, reference, alpha_max_deg, throttle_percent):
    """AoA that gives ``load`` (L/W), capped at ``alpha_max_deg``."""
    q_lift_ceiling = model.forces_at_aoa(altitude, speed, alpha_max_deg, throttle_percent)[2]
    if load*model.mass*G >= q_lift_ceiling:
        return alpha_max_deg
    return min(alpha_max_deg, model.target_aoa(altitude, speed, load, reference))


def advance(model, s, h, demand, max_load, alpha_max_deg, roll_rate_deg_s, load_response_s, engine_response_s,
            alpha_throttle_percent, throttle, brake):
    """One tick of ``h`` seconds from state ``s`` under the demanded acceleration ``demand`` (ENU, m/s^2,
    gravity compensation included). Returns (new FlightState, lift / weight). Raises ValueError outside
    the FM tables (speed below 50 m/s, above Mach 2.35, altitude above 20 km, AoA beyond +-60 deg)."""
    speed = norm(s.velocity)
    direction = unit(s.velocity)
    load = min(norm(demand)/G, max_load)
    # Roll the lift vector toward the demand at the limited rate.
    want = unit(demand) if norm(demand) > 1e-9 else s.normal
    phi = math.atan2(dot(direction, cross(s.normal, want)), dot(s.normal, want))
    limit = math.radians(roll_rate_deg_s)*h
    normal = rotate(s.normal, direction, max(-limit, min(limit, phi)))
    # Load is only useful once the lift vector points roughly at the demand.
    load *= max(0., dot(normal, want))
    alpha_target = alpha_for(model, s.position[2], speed, load, s.aoa_deg, alpha_max_deg, alpha_throttle_percent)
    alpha_mid = alpha_target+(s.aoa_deg-alpha_target)*math.exp(-h/(2*load_response_s))
    alpha_end = alpha_target+(s.aoa_deg-alpha_target)*math.exp(-h/load_response_s)
    engine = throttle+(s.engine_percent-throttle)*math.exp(-h/engine_response_s)
    step = (model.airbrake_speed or 1.)*h
    airbrake = min(brake, s.airbrake+step) if brake > s.airbrake else max(brake, s.airbrake-step)
    thrust, drag, lift, alpha = model.forces_at_aoa(s.position[2], speed, alpha_mid,
                                                    (s.engine_percent+engine)/2, (s.airbrake+airbrake)/2)
    mass = model.mass
    acceleration = add(add(scale(direction, (thrust*math.cos(alpha)-drag)/mass),
                           scale(normal, (lift+thrust*math.sin(alpha))/mass)), (0., 0., -G))
    velocity = add(s.velocity, scale(acceleration, h))
    normal = transport(normal, direction, unit(velocity))
    position = add(s.position, scale(add(s.velocity, velocity), h/2))
    return FlightState(position, velocity, normal, alpha_end, engine, airbrake), lift/(mass*G)


# -- models --------------------------------------------------------------------------------------------------

_BASE_MODELS = {}


def model_with_mass(base: ManeuverModel, mass_kg: float) -> ManeuverModel:
    """A copy of ``base`` at another mass. The AoA/force table (~15 MB) is shared, being immutable. The copy gets its
    own AoA and thrust caches: both quantise their keys, so a shared cache would return values that depend on which
    state filled a bucket first, and two runs of one seed in a process would differ (the AoA cache also depends on
    mass)."""
    if not math.isfinite(mass_kg) or mass_kg <= 0:
        raise ValueError("mass must be positive")
    model = copy.copy(base)
    model.mass = mass_kg
    model._aoa_cache = {}
    model._thrust_cache = {}
    return model


def aircraft_model(aircraft_id: str, mass_kg: float | None = None, mass_factor: float = 1., afterburner=True):
    """ManeuverModel for a catalog aircraft at ``mass_kg`` (default empty mass x ``mass_factor``). The
    expensive table is built once per process and aircraft. ``model.structure`` holds the FM file's load limits."""
    from .fm import load_aircraft
    from .fm.catalog import find_aircraft
    key = (aircraft_id, afterburner)
    base = _BASE_MODELS.get(key)
    if base is None:
        fm = load_aircraft(aircraft_id)
        model = ManeuverModel(fm, fm.empty_mass_kg, afterburner)
        model.structure = read_structure(json.loads(find_aircraft(aircraft_id).path.read_text()))
        base = _BASE_MODELS[key] = (model, fm.empty_mass_kg)
    model, empty = base
    return model_with_mass(model, mass_kg if mass_kg is not None else empty*mass_factor)


# -- airframe load limits ------------------------------------------------------------------------------------

_KEYBOARD = KeyboardTurnSettings()
DEFAULT_LOAD_LIMITS = (_KEYBOARD.min_load, _KEYBOARD.max_load)   # (negative, positive) g without an FM file (D)
LOAD_FACTOR_CAP = (5., 12.)                                      # Instructor.loadFactorLimit's usual values (A, not used unless limitLoadfactor)


@dataclass(frozen=True)
class Structure:
    """What an FM file says about the airframe's load limits (None = absent)."""
    crit_neg_n: float | None = None       # Aerodynamics.WingPlane.Strength.CritOverload[0] (negative, N)
    crit_pos_n: float | None = None       # CritOverload[1]
    mult_neg: float = 1.                  # lower value of the Instructor.overloadMult pair, if limitOverload
    mult_pos: float = 1.
    cap_neg: float | None = None          # Instructor.loadFactorLimit, if limitLoadfactor is true
    cap_pos: float | None = None
    vne_kmh: float | None = None          # Aerodynamics.WingPlane.Strength.VNE: indicated speed that tears the wings off
    vne_control_kmh: float | None = None  # VneControl: indicated speed above which the controls stiffen

    def limits(self, mass_kg: float):
        """(negative, positive) load factor the airframe allows at ``mass_kg`` (weight = mass x g): the critical wing
        force over the weight, times the Instructor multiplier, capped by loadFactorLimit; the keyboard-turn defaults
        for the side the file does not give. The reading of CritOverload / weight as the g limit is an inference (D):
        it comes to 8-9 g at empty mass for the Eagle and the Su-30, and falls as fuel and missiles are added."""
        weight = mass_kg*G
        neg, pos = DEFAULT_LOAD_LIMITS
        if self.crit_pos_n:
            pos = self.mult_pos*self.crit_pos_n/weight
        if self.crit_neg_n:
            neg = -self.mult_neg*abs(self.crit_neg_n)/weight
        if self.cap_pos is not None:
            pos = min(pos, self.cap_pos)
        if self.cap_neg is not None:
            neg = max(neg, -abs(self.cap_neg))
        return neg, pos


def read_structure(raw: dict) -> Structure:
    """Structure from a parsed FM file (the blkx JSON)."""
    aero = raw.get("Aerodynamics") or {}
    wing = aero.get("WingPlane") or next((v for k, v in aero.items() if k.startswith("WingPlane") and isinstance(v, dict)), {})
    crit = (wing.get("Strength") or {}).get("CritOverload")
    neg = pos = None
    if isinstance(crit, list) and len(crit) == 2 and all(isinstance(x, (int, float)) for x in crit):
        neg, pos = (float(x) for x in crit)
    instructor = raw.get("Instructor") or {}
    mult = instructor.get("overloadMult")
    mult_pos = mult_neg = 1.
    if instructor.get("limitOverload") and isinstance(mult, list) and len(mult) == 4:
        mult_pos, mult_neg = min(mult[0], mult[1]), min(mult[2], mult[3])
    cap = instructor.get("loadFactorLimit")
    cap_neg = cap_pos = None
    if instructor.get("limitLoadfactor") and isinstance(cap, list) and len(cap) == 2:
        cap_neg, cap_pos = float(cap[0]), float(cap[1])
    vne = (wing.get("Strength") or {}).get("VNE")
    vne_control = raw.get("VneControl")
    number = lambda x: float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) and x > 0 else None  # noqa: E731
    return Structure(neg, pos, mult_neg, mult_pos, cap_neg, cap_pos, number(vne), number(vne_control))


# -- commands ------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class FlightCommand:
    """What the pilot wants. Direction: ``direction`` (an ENU vector, vertical part included) or ``heading_deg``
    (compass); with neither, the current horizontal heading is held. Vertical: ``altitude_m`` (capture and hold;
    ``max_climb_deg`` / ``max_dive_deg`` bound the flight path), else ``climb_deg`` (flight path angle), else the
    direction vector's own, else level. Speed: ``speed_mps`` (autothrottle, plus airbrake when
    ``airbrake_allowed``) or ``throttle_percent`` (0-110, held), else full power. ``min_speed_mps`` stops a climb
    from trading the last speed away. ``floor_m``: the pilot will not descend into it."""
    direction: tuple | None = None
    heading_deg: float | None = None
    climb_deg: float | None = None
    altitude_m: float | None = None
    speed_mps: float | None = None
    throttle_percent: float | None = None
    max_load: float = 9.
    airbrake_allowed: bool = False
    max_climb_deg: float = 30.
    max_dive_deg: float = 30.
    min_speed_mps: float | None = None
    floor_m: float = 100.

    def __post_init__(self):
        for name in ("heading_deg", "climb_deg", "altitude_m", "speed_mps", "throttle_percent", "min_speed_mps"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.direction is not None and (len(self.direction) != 3 or not all(math.isfinite(x) for x in self.direction)):
            raise ValueError("direction must be a finite ENU triple")
        if not math.isfinite(self.max_load) or self.max_load <= 0:
            raise ValueError("max_load must be positive")


@dataclass(frozen=True)
class KeyboardCommand:
    """Keyboard flying: ``roll`` -1 left / 0 / +1 right, ``pitch`` -1 push / 0 release / +1 pull, ``throttle`` -1 down /
    0 hold / +1 up, ``airbrake`` on or off. Bang-bang at full authority: full roll rate, and a full pull or push asks for
    the airframe's positive or negative load limit (``Aircraft.load_limits``), still capped by the AoA the tables allow.
    ``authority`` in (0, 1] scales it for players who limit their input: roll rate x authority, pull to 1 + authority x
    (n_max - 1) g, push to 1 - authority x (1 - n_min) g."""
    roll: int = 0
    pitch: int = 0
    throttle: int = 0
    airbrake: bool = False
    authority: float = 1.

    def __post_init__(self):
        for name in ("roll", "pitch", "throttle"):
            if getattr(self, name) not in (-1, 0, 1):
                raise ValueError(f"{name} must be -1, 0 or +1")
        if not math.isfinite(self.authority) or not 0. < self.authority <= 1.:
            raise ValueError("authority must be in (0, 1]")


@dataclass(frozen=True)
class FlightParams:
    """Pilot and airframe response constants shared with escape.EvasionPilot (same defaults, D)."""
    alpha_max_deg: float = 20.
    roll_rate_deg_s: float = 120.
    load_response_s: float = .6
    turn_time_constant_s: float = .5
    engine_response_s: float = 1.

    def __post_init__(self):
        for name in ("alpha_max_deg", "roll_rate_deg_s", "load_response_s", "turn_time_constant_s", "engine_response_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive")


# -- the aircraft --------------------------------------------------------------------------------------------

class Aircraft:
    """One aircraft stepping 1/48 s at a time under ``command``. ``state_at`` answers missile queries."""

    def __init__(self, model: ManeuverModel, position, velocity, *, normal=None, params: FlightParams = FlightParams(),
                 t0=0., engine_percent=None, history=96, keep_history=False, limit_structure=False,
                 structural_speed=False):
        self.model, self.params = model, params
        self.limit_structure = limit_structure    # also cap a FlightCommand's max_load at the airframe limit
        # VNE / VneControl from the FM file: overspeed tears the wings off (self.overspeed), the controls stiffen
        # above VneControl, and the pilot holds the speed below VNE with throttle and airbrake.
        structure = getattr(model, "structure", None)
        self.vne_mps = structure.vne_kmh/3.6 if structural_speed and structure and structure.vne_kmh else None
        self.vne_control_mps = structure.vne_control_kmh/3.6 if self.vne_mps and structure.vne_control_kmh else None
        self.overspeed, self.overspeed_s = False, 0.
        self._roll_rate = 0.                      # keyboard mode: rad/s about the velocity
        self._kb_throttle = None                  # keyboard mode: the throttle setting, percent
        self.t0, self.tick = float(t0), 0
        self.command = FlightCommand()   # or a KeyboardCommand
        self.alive, self.crashed, self.faults, self.load = True, False, 0, 1.
        self._turn_sign = None
        self._thrust_i = None
        self._low = False
        position, velocity = tuple(map(float, position)), tuple(map(float, velocity))
        direction = unit(velocity)
        if normal is None:
            normal = unit(add((0., 0., 1.), scale(direction, -direction[2])))
        # Wings level at the start: lift = weight cos(gamma).
        try:
            alpha = alpha_for(model, position[2], norm(velocity), math.sqrt(1-direction[2]**2), 0.,
                              params.alpha_max_deg, model.max_throttle)
        except ValueError:  # Outside the FM tables at the start; step() falls ballistically until it is inside.
            alpha = 0.
        engine = model.max_throttle if engine_percent is None else engine_percent
        self.state = FlightState(position, velocity, tuple(normal), alpha, engine)
        self._ring = deque([(position, velocity)], maxlen=max(2, history))
        self._first = 0                       # tick index of _ring[0]
        self.states = [self.state] if keep_history else None
        self.min_altitude_m = position[2]
        # The engine thrust tables end at some TAS and altitude and are never extrapolated: stay inside them.
        engines = model.model.engines
        self.v_engine_mps = min(e.velocities_kph[-1] for e in engines)/3.6-ENGINE_MARGIN_MPS
        self.ceiling_m = min(MAX_ALTITUDE_M, min(e.altitudes[-1] for e in engines))

    # -- time and queries ------------------------------------------------------------------------------------

    @property
    def time(self):
        return self.t0+self.tick*SUBSTEP_S

    def state_at(self, time_s):
        """(position, velocity) in ENU at world time ``time_s``: a stored tick exactly, otherwise linear between
        the two ticks around it (as FMEvader). Times past the latest tick are an error (advance the aircraft
        first); times older than the history raise too."""
        x = (time_s-self.t0)/SUBSTEP_S
        i = round(x)
        if abs(x-i) < 1e-6:
            return self._at_tick(i)
        lo = math.floor(x)
        a, b = self._at_tick(lo), self._at_tick(lo+1)
        w = x-lo
        mix = lambda p, q: tuple(u+w*(v-u) for u, v in zip(p, q))  # noqa: E731
        return mix(a[0], b[0]), mix(a[1], b[1])

    def _at_tick(self, i):
        if i > self.tick:
            raise ValueError(f"aircraft state requested {(i-self.tick)*SUBSTEP_S:.3f} s ahead of its latest tick")
        k = i-self._first
        if k < 0:
            raise ValueError("aircraft history no longer holds that time")
        return self._ring[k]

    def attitude(self):
        """(heading deg compass, pitch deg, roll deg right-wing-down) of the nose: the velocity direction raised by the
        angle of attack along the lift axis. Sideslip is zero."""
        s = self.state
        v = s.velocity
        speed = norm(v)
        d = (v[0]/speed, v[1]/speed, v[2]/speed)
        horizontal = math.hypot(d[0], d[1])
        heading = math.degrees(math.atan2(d[0], d[1])) % 360.
        gamma = math.asin(max(-1., min(1., d[2])))
        if horizontal < 1e-6:
            return heading, math.degrees(gamma), 0.
        right = (d[1]/horizontal, -d[0]/horizontal, 0.)
        up = (-d[0]*d[2]/horizontal, -d[1]*d[2]/horizontal, horizontal)  # Level-wing up axis perpendicular to d.
        n = s.normal
        roll = math.atan2(dot(n, right), dot(n, up))
        pitch = gamma+math.radians(s.aoa_deg)*math.cos(roll)
        return heading, math.degrees(pitch), math.degrees(roll)

    @property
    def speed(self):
        return norm(self.state.velocity)

    @property
    def altitude(self):
        return self.state.position[2]

    def indicated(self, s=None):
        """Indicated (equivalent) airspeed, m/s: TAS x sqrt(rho / rho0)."""
        s = s or self.state
        rho = atmosphere(max(0., min(19999., s.position[2])))[0]
        return norm(s.velocity)*math.sqrt(rho/1.225)

    def _control_share(self, ias):
        """Share of the load factor above 1 g still available: 1 below VneControl, falling to 1-VNE_CONTROL_LOSS at VNE."""
        if self.vne_control_mps is None or ias <= self.vne_control_mps or self.vne_mps <= self.vne_control_mps:
            return 1.
        return 1.-VNE_CONTROL_LOSS*min(1., (ias-self.vne_control_mps)/(self.vne_mps-self.vne_control_mps))

    def _governed(self, ias, throttle, brake, top):
        """The pilot keeps clear of VNE: throttle fades to idle approaching the governor point, airbrake above it."""
        limit = VNE_GOVERNOR*self.vne_mps
        band = VNE_GOVERNOR_BAND_KMH/3.6
        if ias > limit-band:
            throttle = min(throttle, top*max(0., (limit-ias)/band))
            if ias > limit:
                brake = 1.
        return throttle, brake

    @property
    def load_limits(self):
        """(negative, positive) load factor the airframe allows at the current mass (see ``Structure.limits``)."""
        structure = getattr(self.model, "structure", None)
        return structure.limits(self.model.mass) if structure is not None else DEFAULT_LOAD_LIMITS

    # -- control law -----------------------------------------------------------------------------------------

    def _desired(self, s, cmd, speed, direction):
        """Unit ENU direction the pilot steers the velocity toward, or None to fly straight at 1 g."""
        if cmd.direction is not None:
            d = cmd.direction
            n = norm(d)
            if n < 1e-9:
                return None
            hx, hy, vz = d[0]/n, d[1]/n, d[2]/n
            horizontal = math.hypot(hx, hy)
            gamma = math.asin(max(-1., min(1., vz)))
            heading = None if horizontal < 1e-6 else (hx/horizontal, hy/horizontal)
        else:
            gamma = 0.
            heading = None
            if cmd.heading_deg is not None:
                a = math.radians(cmd.heading_deg)
                heading = (math.sin(a), math.cos(a))
        if heading is None:
            ch = math.hypot(direction[0], direction[1])
            if ch < 1e-6:
                return None
            heading = (direction[0]/ch, direction[1]/ch)
        if cmd.altitude_m is not None:
            vz = max(-speed, min(speed, (cmd.altitude_m-s.position[2])/ALT_TAU_S))
            gamma = math.asin(vz/speed)
            gamma = max(-math.radians(cmd.max_dive_deg), min(math.radians(cmd.max_climb_deg), gamma))
        elif cmd.direction is None and cmd.climb_deg is not None:
            gamma = math.radians(cmd.climb_deg)
        if gamma > 0. and s.position[2] > self.ceiling_m-CEILING_BAND_M:
            gamma *= max(0., (self.ceiling_m-s.position[2])/CEILING_BAND_M)
        if cmd.min_speed_mps is not None and gamma > 0.:
            gamma *= max(0., min(1., (speed-cmd.min_speed_mps)/30.))
        # Ground floor: level off early enough to arrest the sink rate with the load limit (circular pull-out). The
        # pull-out outranks turning: a descending aircraft that is low pulls up wings level on its current heading,
        # and below the floor it climbs out.
        sink = max(0., -s.velocity[2])
        pull_out = FLOOR_PULL_OUT*sink*sink/(2*max(.5, cmd.max_load-1)*G) if sink else 0.
        self._low = s.position[2] < cmd.floor_m+pull_out
        if self._low and (gamma < 0. or sink > PULL_OUT_SINK_MPS):
            gamma = max(gamma, 0.)
            ch = math.hypot(direction[0], direction[1])
            if sink > 0. and ch > 1e-6:
                heading = (direction[0]/ch, direction[1]/ch)
        if s.position[2] < cmd.floor_m:
            gamma = max(gamma, math.radians(CLIMB_OUT_DEG))
        # Reversals stay a horizontal turn with a locked side; aiming straight behind would degenerate into a loop.
        ch = math.hypot(direction[0], direction[1])
        if ch > 1e-6:
            hc = (direction[0]/ch, direction[1]/ch)
            # Compass-sense (clockwise positive) angle from the current heading to the wanted one.
            error = -math.atan2(hc[0]*heading[1]-hc[1]*heading[0], hc[0]*heading[0]+hc[1]*heading[1])
            if abs(error) > math.pi/2:
                if self._turn_sign is None:
                    self._turn_sign = 1. if error >= 0 else -1.
                a = math.atan2(hc[0], hc[1])+self._turn_sign*math.pi/2  # Compass angle, clockwise.
                heading = (math.sin(a), math.cos(a))
            else:
                self._turn_sign = None
        c = math.cos(gamma)
        return (heading[0]*c, heading[1]*c, math.sin(gamma))

    def _throttle(self, s, cmd, speed):
        """(throttle percent, airbrake 0/1) for the command, with the Mach limit applied."""
        throttle, brake = self._throttle_command(s, cmd, speed)
        limit = min(MACH_LIMIT*atmosphere(max(0., min(19999., s.position[2])))[1], self.v_engine_mps)
        if speed > limit-MACH_BAND_MPS:
            throttle = min(throttle, self.model.max_throttle*max(0., (limit-speed)/MACH_BAND_MPS))
            if speed > limit and cmd.airbrake_allowed:
                brake = 1.
        return throttle, brake

    def _throttle_command(self, s, cmd, speed):
        top = self.model.max_throttle
        if cmd.speed_mps is not None:
            excess = speed-cmd.speed_mps
            if self._thrust_i is None:
                self._thrust_i = s.engine_percent
            kp, ki = SPEED_PI
            self._thrust_i = max(0., min(top, self._thrust_i-ki*excess*SUBSTEP_S))
            throttle = max(0., min(top, self._thrust_i-kp*excess))
            return throttle, (1. if cmd.airbrake_allowed and excess > BRAKE_EXCESS_MPS else 0.)
        self._thrust_i = None
        if cmd.throttle_percent is not None:
            return max(0., min(top, cmd.throttle_percent)), 0.
        return top, 0.

    # -- stepping --------------------------------------------------------------------------------------------

    def step(self, command: FlightCommand | KeyboardCommand | None = None):
        """Advance one tick (1/48 s), optionally replacing the command first. A crashed aircraft does not move."""
        if command is not None:
            self.command = command
        if not self.alive:
            return self.state
        s, cmd, p = self.state, self.command, self.params
        speed = norm(s.velocity)
        if isinstance(cmd, KeyboardCommand):
            return self._keyboard_step(s, cmd, speed)
        self._kb_throttle, self._roll_rate = None, 0.
        try:
            direction = unit(s.velocity)
            demand = add((0., 0., G), scale(direction, -G*direction[2]))
            desired = self._desired(s, cmd, speed, direction)
            if desired is not None:
                along = dot(desired, direction)
                perp = add(desired, scale(direction, -along))
                error = math.atan2(norm(perp), along)
                if norm(perp) < 1e-6:
                    perp = s.normal if along < 0 else (0., 0., 0.)
                if norm(perp) > 1e-9:
                    demand = add(demand, scale(unit(perp), speed*error/p.turn_time_constant_s))
            throttle, brake = self._throttle(s, cmd, speed)
            max_load = min(cmd.max_load, self.load_limits[1]) if self.limit_structure else cmd.max_load
            if self.vne_mps is not None:
                ias = self.indicated(s)
                throttle, brake = self._governed(ias, throttle, brake, self.model.max_throttle)
                max_load = 1.+self._control_share(ias)*(max_load-1.)
            new, self.load = advance(self.model, s, SUBSTEP_S, demand, max_load, p.alpha_max_deg, p.roll_rate_deg_s,
                                     p.load_response_s, p.engine_response_s, self.model.max_throttle, throttle, brake)
        except ValueError:
            # Outside the FM tables (very slow, supersonic beyond Mach 2.35, above 20 km): fall ballistically for
            # this tick; the aircraft re-enters the tables when its speed does.
            self.faults += 1
            velocity = add(s.velocity, (0., 0., -G*SUBSTEP_S))
            new = FlightState(add(s.position, scale(add(s.velocity, velocity), SUBSTEP_S/2)), velocity, s.normal,
                              s.aoa_deg, s.engine_percent, s.airbrake)
            self.load = 0.
        return self._commit(new)

    def _keyboard_step(self, s, cmd, speed):
        """One tick under a KeyboardCommand: ``ManeuverModel.step`` with settings from the airframe's limits."""
        if self._kb_throttle is None:                 # entering keyboard flying: keep the engine where it is
            self._kb_throttle = s.engine_percent
        a = cmd.authority
        n_neg, n_pos = self.load_limits
        if self.vne_mps is not None:
            ias = self.indicated(s)
            share = self._control_share(ias)
            n_neg, n_pos = 1.-share*(1.-n_neg), 1.+share*(n_pos-1.)
        base = KeyboardTurnSettings()
        settings = KeyboardTurnSettings(max_load=1.+a*(n_pos-1.), min_load=1.-a*(1.-n_neg), roll_rate_deg_s=base.roll_rate_deg_s*a)
        top = self.model.max_throttle
        limit = min(MACH_LIMIT*atmosphere(max(0., min(19999., s.position[2])))[1], self.v_engine_mps)
        if speed > limit-MACH_BAND_MPS:               # the tables end: the throttle fades out as in the other mode
            self._kb_throttle = min(self._kb_throttle, top*max(0., (limit-speed)/MACH_BAND_MPS))
        airbrake = cmd.airbrake
        if self.vne_mps is not None:
            self._kb_throttle, brake = self._governed(ias, self._kb_throttle, 1. if airbrake else 0., top)
            airbrake = airbrake or brake > 0.
        try:
            motion = Motion(s.position[2], s.velocity, s.normal, self._roll_rate, self.load, self._kb_throttle,
                            s.engine_percent, s.aoa_deg, s.airbrake)
            m = self.model.step(motion, KeyAction(cmd.roll, cmd.pitch, cmd.throttle, 1 if airbrake else 0), SUBSTEP_S,
                                settings)
            position = add(s.position, scale(add(s.velocity, m.velocity), SUBSTEP_S/2))
            new = FlightState(position, m.velocity, m.normal, m.aoa_deg, m.engine_throttle_percent, m.airbrake_fraction)
            self._roll_rate, self._kb_throttle, self.load = m.roll_rate, m.throttle_percent, m.load
        except ValueError:                            # outside the FM tables: ballistic for this tick, as in the other mode
            self.faults += 1
            velocity = add(s.velocity, (0., 0., -G*SUBSTEP_S))
            new = FlightState(add(s.position, scale(add(s.velocity, velocity), SUBSTEP_S/2)), velocity, s.normal,
                              s.aoa_deg, s.engine_percent, s.airbrake)
            self.load = 0.
        return self._commit(new)

    def _commit(self, new):
        self.state = new
        self.tick += 1
        self._ring.append((new.position, new.velocity))
        self._first = self.tick+1-len(self._ring)
        if self.states is not None:
            self.states.append(new)
        if new.position[2] < self.min_altitude_m:
            self.min_altitude_m = new.position[2]
        if new.position[2] <= 0.:
            self.alive, self.crashed = False, True
        elif self.vne_mps is not None:
            if self.indicated(new) > self.vne_mps:
                self.overspeed_s += SUBSTEP_S
                if self.overspeed_s >= OVERSPEED_TOLERANCE_S:
                    self.alive, self.overspeed = False, True      # wings torn off
            else:
                self.overspeed_s = 0.
        return new
