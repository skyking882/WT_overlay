"""Offline FM-driven evader for missile escape-window tables.

Implements missile_sim's external-target protocol (``state_at`` and
``observe_missile``) without importing missile_sim, so the HUD never depends
on it. Before ``start_s`` the scenario target is reproduced unchanged; from
``start_s`` the selected FM flies toward a beam or drag direction relative to
the missile's last observed position, or, with a ``Perception``, relative to
whatever the pilot could see: the missile while its motor burns or once its
seeker is active, otherwise the launcher.

The pilot is a project rule, not Instructor physics: roll the lift vector
toward the demanded acceleration at a limited rate, then demand load capped by
the crew limit and an AoA ceiling, with first-order AoA response. Forces use
the same tabulated polars and thrust as the keyboard turn model. A segment's
speed target closes the throttle and opens the airbrake when fast, or runs the
plan throttle when slow, with a first-order engine response. Chaff is applied
by the missile side (missile_sim chaffing_factory); no FCS G-limiter is modeled. The per-tick force and attitude integration is
wt_overlay.flight.advance, shared with the command-driven Aircraft of the engagement simulator (tests/test_flight.py pins
that FMEvader's output is unchanged). Coordinates here are east, north, up; missile_sim uses x, up, z with north = -z.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass
import math
from types import SimpleNamespace

from .climb import valid_number
from .contracts import G
from .flight import SUBSTEP_S, FlightState as _State, advance, alpha_for  # SUBSTEP_S: one missile control tick.
from .turn import ManeuverModel, add, cross, dot, norm, rotate, scale, unit

KINDS = ("beam", "drag")
TILT_HANDOFF_DEG = 45.  # Below this direction error a tilted-plane turn hands back to direct tracking.


@dataclass(frozen=True)
class EvasionPilot:
    kind: str
    start_s: float
    max_load: float = 9.      # Crew/G-limit target, L/W.
    alpha_max_deg: float = 20.  # Mouse-aim-like AoA ceiling; an assumption.
    roll_rate_deg_s: float = 120.
    load_response_s: float = .6
    turn_time_constant_s: float = .5  # Direction error -> demanded turn rate.
    dive_deg: float = 0.      # Flight path below horizon after evasion starts.
    throttle_percent: float = 110.
    floor_m: float = 300.     # Below this the dive component is removed.
    # Maneuver building blocks (defaults reproduce the level beam/drag above):
    # target_deg = final heading measured from "away from the threat" (0 drag,
    # 90 beam, side nearest the current heading); plane_deg = tilt of the turn
    # plane below horizontal while far from that heading (0 level turn, 45
    # slice, 90 pull straight down as in a split-S); ``then`` = later segments
    # (seconds after start_s, target_deg, plane_deg, dive_deg), increasing.
    target_deg: float | None = None
    plane_deg: float = 0.
    then: tuple = ()
    # Optional speed target (km/h, None = hold throttle_percent): above it the
    # pilot idles and opens the airbrake, below it selects throttle_percent.
    # ``then`` entries may carry a fifth element with their own speed target.
    speed_kmh: float | None = None
    engine_response_s: float = 1.

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"evasion kind must be one of {KINDS}")
        segments = [(0., self.heading_deg, self.plane_deg, self.dive_deg, self.speed_kmh), *self.then]
        for (t0, *_), (t1, *_) in zip(segments, segments[1:]):
            if not valid_number(t1) or t1 <= t0:
                raise ValueError("segment times must increase")
        for _, target, plane, dive, *speed in segments:
            if not (valid_number(target) and 0 <= target <= 180 and valid_number(plane) and 0 <= plane <= 90
                    and valid_number(dive) and 0 <= dive < 60):
                raise ValueError("segment angles out of range")
            if len(speed) > 1 or (speed and speed[0] is not None and not (valid_number(speed[0]) and speed[0] > 0)):
                raise ValueError("segment speed target must be positive or None")
        if not valid_number(self.engine_response_s) or self.engine_response_s <= 0:
            raise ValueError("engine_response_s must be positive")
        positive = ("max_load", "alpha_max_deg", "roll_rate_deg_s", "load_response_s", "turn_time_constant_s")
        for name in positive:
            if not valid_number(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not valid_number(self.start_s) or self.start_s < 0:
            raise ValueError("start_s must be nonnegative")
        if not valid_number(self.dive_deg) or not 0 <= self.dive_deg < 60:
            raise ValueError("dive_deg must be in [0, 60)")

    @property
    def heading_deg(self):
        return self.target_deg if self.target_deg is not None else (90. if self.kind == "beam" else 0.)

    def segment(self, elapsed_s):
        """(target_deg, plane_deg, dive_deg) active ``elapsed_s`` after the evasion starts."""
        return self.segment_full(elapsed_s)[:3]

    def segment_full(self, elapsed_s):
        """(target_deg, plane_deg, dive_deg, speed_kmh) active ``elapsed_s`` after the evasion starts."""
        active = (self.heading_deg, self.plane_deg, self.dive_deg, self.speed_kmh)
        for after, target, plane, dive, *speed in self.then:
            if elapsed_s >= after:
                active = (target, plane, dive, speed[0] if speed else None)
        return active

    def describe(self):
        def one(target, plane, dive, speed):
            return f"{target:g}/{plane:g}/{dive:g}" + ("" if speed is None else f"@{speed:g}")
        parts = [one(self.heading_deg, self.plane_deg, self.dive_deg, self.speed_kmh)]
        parts += [f"+{after:g}s " + one(target, plane, dive, speed[0] if speed else None)
                  for after, target, plane, dive, *speed in self.then]
        return " ".join(parts)


@dataclass(frozen=True)
class Perception:
    """What the evading pilot can see of the missile (user-reported WT rules, D).

    The missile marker is visible while its motor burns and, latched, once it
    is within its seeker's active range of the target (RWR picks up the active
    seeker). In between only the launcher's direction is known. The launcher
    keeps its launch speed and altitude; from ``crank_start_s`` it turns its
    heading by ``crank_rad`` (ENU, + = counter-clockwise) at
    ``crank_rate_rad_s``, then flies straight. The missile's own guidance is not
    tied to the launcher, so a crank past the radar gimbal is not penalised.
    """
    burn_s: float
    active_range_m: float
    crank_rad: float = 0.
    crank_rate_rad_s: float = .1
    crank_start_s: float = 0.

    def __post_init__(self):
        if not valid_number(self.burn_s) or self.burn_s < 0 or not valid_number(self.active_range_m) \
                or self.active_range_m <= 0 or not valid_number(self.crank_rad) \
                or not valid_number(self.crank_rate_rad_s) or self.crank_rate_rad_s <= 0 \
                or not valid_number(self.crank_start_s) or self.crank_start_s < 0:
            raise ValueError("invalid perception")

    def launcher_offset(self, velocity, elapsed_s):
        """Launcher displacement after ``elapsed_s`` (ENU): straight, turn, straight."""
        speed = math.hypot(velocity[0], velocity[1])
        straight = min(elapsed_s, self.crank_start_s)
        offset = scale(velocity, straight)
        left = elapsed_s-straight
        if left <= 0 or speed <= 0 or self.crank_rad == 0:
            return add(offset, scale(velocity, max(0., left)))
        heading = math.atan2(velocity[1], velocity[0])
        sign = 1. if self.crank_rad > 0 else -1.
        turn_s = min(left, abs(self.crank_rad)/self.crank_rate_rad_s)
        radius = speed/self.crank_rate_rad_s
        end = heading+sign*self.crank_rate_rad_s*turn_s
        arc = (radius*sign*(math.sin(end)-math.sin(heading)), radius*sign*(math.cos(heading)-math.cos(end)),
               velocity[2]*turn_s)
        after = left-turn_s
        return add(add(offset, arc), (speed*math.cos(end)*after, speed*math.sin(end)*after, velocity[2]*after))


def to_enu(v):
    return (v[0], -v[2], v[1])


def from_enu(v):
    return (v[0], v[2], -v[1])


class FMEvader:
    def __init__(self, base, pilot: EvasionPilot, model: ManeuverModel, make_state=SimpleNamespace,
                 perception: Perception | None = None):
        self.base, self.pilot, self.model, self.make_state = base, pilot, model, make_state
        self.perception = perception
        self._missile = None
        self._missile_time = 0.
        self._launcher = None  # (time, position, velocity) of the first observation, ENU.
        self._active_seen = False
        self.launcher_steer_s = 0.
        self._beam_sign = None
        self._turn_sign = None
        self._tilt_segment = None
        self._tilt_normal = None
        self._tilt_error = None
        self._tilt_done = False
        self._plane = 0.
        self._low = False
        self._hot = False  # Recommit phase: turn back toward the launcher.
        self.fault = None
        start = base.state_at(pilot.start_s)
        position, velocity = to_enu(start.position), to_enu(start.velocity)
        direction = unit(velocity)
        normal = unit(add((0., 0., 1.), scale(direction, -direction[2])))
        # Wings level at the scenario's straight flight: lift = weight·cos γ.
        alpha = self._alpha_for(position[2], norm(velocity), math.sqrt(1-direction[2]**2), 0.)
        self._times = [pilot.start_s]
        self._states = [_State(position, velocity, normal, alpha, pilot.throttle_percent)]
        self.min_speed_mps = self.final_speed_mps = norm(velocity)
        self.min_altitude_m = position[2]

    def observe_missile(self, time_s, position, velocity):
        self._missile, self._missile_time = to_enu(position), time_s
        if self._launcher is None:
            # At launch the missile is at the launcher with the launcher's velocity.
            self._launcher = (time_s, self._missile, to_enu(velocity))

    def describe(self):
        p = self.pilot
        return dict(kind=p.kind, start_s=p.start_s, model="wt_overlay_fm_pilot", plan=p.describe(), max_load=p.max_load,
                    alpha_max_deg=p.alpha_max_deg, roll_rate_deg_s=p.roll_rate_deg_s, dive_deg=p.dive_deg,
                    perception="truth" if self.perception is None else "marker_or_rwr",
                    launcher_steer_s=self.launcher_steer_s,
                    min_speed_mps=self.min_speed_mps, final_speed_mps=self.final_speed_mps,
                    min_altitude_m=self.min_altitude_m, fault=self.fault)

    def _reference(self, position):
        """The point the pilot evades: the missile if perceivable, else the launcher."""
        if self._missile is None or self.perception is None:
            return self._missile
        p = self.perception
        gap = add(self._missile, scale(position, -1.))
        if not self._active_seen and norm(gap) <= p.active_range_m:
            self._active_seen = True
        if self._missile_time <= p.burn_s or self._active_seen:
            return self._missile
        t0, launcher, velocity = self._launcher
        self.launcher_steer_s += SUBSTEP_S
        return add(launcher, p.launcher_offset(velocity, self._missile_time-t0))

    def state_at(self, time_s):
        if time_s <= self.pilot.start_s:
            return self.base.state_at(time_s)
        while self._times[-1] < time_s:
            self._advance()
        # times[0] == start_s < time_s <= times[-1], so 1 <= i < len; a query a rounding
        # error away from a substep (e.g. a 1/48 s missile tick on a 0.25 s start) is that substep.
        i = bisect.bisect_left(self._times, time_s)
        for j in (i-1, i):
            if abs(self._times[j]-time_s) <= 1e-9:
                s = self._states[j]
                return self.make_state(position=from_enu(s.position), velocity=from_enu(s.velocity))
        t0, t1 = self._times[i-1], self._times[i]
        a, b = self._states[i-1], self._states[i]
        w = (time_s-t0)/(t1-t0)
        mix = lambda x, y: tuple(p+w*(q-p) for p, q in zip(x, y))  # noqa: E731
        return self.make_state(position=from_enu(mix(a.position, b.position)),
                               velocity=from_enu(mix(a.velocity, b.velocity)))

    def _alpha_for(self, altitude, speed, load, reference):
        return alpha_for(self.model, altitude, speed, load, reference, self.pilot.alpha_max_deg,
                         self.pilot.throttle_percent)

    def _launcher_at(self, time_s):
        """Where the evader believes the launcher is (straight or cranking, as perceived)."""
        t0, launcher, velocity = self._launcher
        offset = (self.perception.launcher_offset(velocity, time_s-t0) if self.perception is not None
                  else scale(velocity, time_s-t0))
        return add(launcher, offset)

    def recommit(self, time_s, within_deg=30., max_s=60.):
        """Seconds from ``time_s`` (missile defeated) until the nose is back on the launcher.

        From ``time_s`` the pilot flies a level, afterburner turn toward the
        launcher's horizontal direction; done once the heading is within
        ``within_deg`` of it. History after ``time_s`` is discarded. Returns
        dict(seconds, speed_mps, altitude_m) or None if never within ``max_s``
        or the FM run faults.
        """
        if self._launcher is None or self.fault is not None:
            return None
        # A missile can pass before a late evasion even starts; the turn back then
        # begins at the evasion start and the straight leg before it still counts.
        begin = max(time_s, self.pilot.start_s)
        self.state_at(begin)
        keep = max(1, bisect.bisect_right(self._times, begin+1e-12))
        del self._times[keep:], self._states[keep:]
        self._hot, self._turn_sign, self._plane = True, None, 0.
        start = time_s
        while self._times[-1]-start <= max_s:
            s = self._states[-1]
            to = add(self._launcher_at(self._times[-1]), scale(s.position, -1.))
            heading = (s.velocity[0], s.velocity[1], 0.)
            if norm(heading) > 1e-9 and norm((to[0], to[1], 0.)) > 1e-9:
                h, t = unit(heading), unit((to[0], to[1], 0.))
                if abs(math.degrees(math.atan2(cross(h, t)[2], dot(h, t)))) <= within_deg:
                    return dict(seconds=self._times[-1]-start, speed_mps=norm(s.velocity), altitude_m=s.position[2])
            self._advance()
            if self.fault is not None:
                return None
        return None

    def _desired(self, position, velocity):
        if self._hot:
            to = add(self._launcher_at(self._times[-1]), scale(position, -1.))
            horizontal = unit((to[0], to[1], 0.))
            return self._level_turn(horizontal, velocity, 0.) if norm(horizontal) > 0 else None
        source = self._reference(position)
        if source is None:
            return None
        los = (position[0]-source[0], position[1]-source[1], 0.)
        if norm(los) < 1e-9:
            return None
        away = unit(los)
        target, plane, dive_deg = self.pilot.segment(self._times[-1]-self.pilot.start_s)
        self._plane = plane
        if target == 0.:
            horizontal = away
        else:
            if self._beam_sign is None:
                # Lock the side nearest the heading at the first evasive step.
                self._beam_sign = 1. if cross(away, velocity)[2] >= 0 else -1.
            horizontal = rotate(away, (0., 0., 1.), self._beam_sign*math.radians(target))
        if plane == 0.:
            horizontal = self._level_turn(horizontal, velocity, None)
        return self._dive_toward(horizontal, position, velocity, dive_deg)

    def _level_turn(self, horizontal, velocity, dive_deg):
        """Keep reversals a horizontal turn; with ``dive_deg`` also apply the floor and dive."""
        heading = (velocity[0], velocity[1], 0.)
        if norm(heading) > 1e-9:
            heading = unit(heading)
            error = math.atan2(cross(heading, horizontal)[2], dot(heading, horizontal))
            if abs(error) > math.pi/2:
                # Reversals stay a horizontal turn with a locked direction;
                # aiming straight behind would degenerate into a vertical loop.
                if self._turn_sign is None:
                    self._turn_sign = 1. if error >= 0 else -1.
                horizontal = rotate(heading, (0., 0., 1.), self._turn_sign*math.pi/2)
        if dive_deg is None:
            return horizontal
        return self._dive_toward(horizontal, self._states[-1].position, velocity, dive_deg)

    def _dive_toward(self, horizontal, position, velocity, dive_deg):
        # Level off early enough to arrest the sink rate at the floor with the
        # load limit (circular pull-out radius V²/((n-1)g), 50% margin).
        sink = max(0., -velocity[2])
        pull_out = 1.5*sink*sink/(2*max(.5, self.pilot.max_load-1)*G) if sink else 0.
        self._low = position[2] < self.pilot.floor_m+pull_out
        dive = 0. if self._low else math.radians(dive_deg)
        return add(scale(horizontal, math.cos(dive)), (0., 0., -math.sin(dive)))

    def _tilted(self, perp, direction, error, speed, altitude):
        """Rotate the velocity in a fixed plane tilted ``plane`` below horizontal.

        The plane is fixed when the turn starts (current velocity and an axis
        tilted between the horizontal side and straight down), so a pull through
        the vertical continues over the top like a split-S. Direct tracking
        resumes once within TILT_HANDOFF_DEG of the target direction or once the
        error starts growing again (the plane's closest approach has passed).
        A half loop of room above the floor is required.
        """
        segment = self.pilot.segment(self._times[-1]-self.pilot.start_s)
        if segment != self._tilt_segment:
            self._tilt_segment, self._tilt_normal, self._tilt_error, self._tilt_done = segment, None, None, False
        plane = math.radians(self._plane)
        half_loop = 2*speed*speed/(max(.5, self.pilot.max_load-1)*G)
        if (plane <= 0. or self._tilt_done or self._low or altitude < self.pilot.floor_m+half_loop
                or error < math.radians(TILT_HANDOFF_DEG)
                or (self._tilt_error is not None and error > self._tilt_error+1e-4)):
            if self._tilt_normal is not None:
                self._tilt_done = True
            return perp
        if self._tilt_normal is None:
            side = (perp[0], perp[1], 0.)
            if norm(side) < 1e-6:
                side = rotate((direction[0], direction[1], 0.), (0., 0., 1.), (self._beam_sign or 1.)*math.pi/2)
            axis = add(scale(unit(side), math.cos(plane)), (0., 0., -math.sin(plane)))
            normal = cross(direction, axis)
            if norm(normal) < 1e-9:
                return perp
            self._tilt_normal = unit(normal)
        self._tilt_error = error
        return cross(self._tilt_normal, direction)

    def _advance(self):
        h = SUBSTEP_S
        s = self._states[-1]
        if self.fault is None:
            try:
                s = self._fly(s, h)
            except ValueError as exc:  # Left the FM table range; continue straight.
                self.fault = str(exc)
            if s.position[2] <= 0:
                self.fault = "ground impact"
        if self.fault is not None:
            s = _State(add(s.position, scale(s.velocity, h)), s.velocity, s.normal, s.aoa_deg,
                       s.engine_percent, s.airbrake)
        self._times.append(self._times[-1]+h)
        self._states.append(s)
        speed = norm(s.velocity)
        self.min_speed_mps = min(self.min_speed_mps, speed)
        self.final_speed_mps = speed
        self.min_altitude_m = min(self.min_altitude_m, s.position[2])

    def _fly(self, s, h):
        p = self.pilot
        speed = norm(s.velocity)
        direction = unit(s.velocity)
        gravity_perp = add((0., 0., G), scale(direction, -G*direction[2]))
        demand = gravity_perp
        desired = self._desired(s.position, s.velocity)
        if desired is not None:
            along = dot(desired, direction)
            perp = add(desired, scale(direction, -along))
            error = math.atan2(norm(perp), along)
            if norm(perp) < 1e-6:
                perp = s.normal if along < 0 else (0., 0., 0.)
            perp = self._tilted(perp, direction, error, speed, s.position[2])
            if norm(perp) > 1e-9:
                demand = add(demand, scale(unit(perp), speed*error/p.turn_time_constant_s))
        # Speed target: idle plus airbrake when fast, throttle_percent when slow (±5 m/s deadband).
        target_kmh = None if self._hot else p.segment_full(self._times[-1]-p.start_s)[3]
        throttle, brake = p.throttle_percent, 0.
        if target_kmh is not None:
            excess = speed-target_kmh/3.6
            throttle, brake = ((0., 1.) if excess > 5. else (p.throttle_percent, 0.) if excess < -5.
                               else (s.engine_percent, 0.))
        return advance(self.model, s, h, demand, p.max_load, p.alpha_max_deg, p.roll_rate_deg_s, p.load_response_s,
                       p.engine_response_s, p.throttle_percent, throttle, brake)[0]


def fm_evader_factory(pilot: EvasionPilot, model: ManeuverModel, make_state=SimpleNamespace,
                      perception: Perception | None = None):
    return lambda base: FMEvader(base, pilot, model, make_state, perception)
