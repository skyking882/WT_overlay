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
the same tabulated polars and thrust as the keyboard turn model. No airbrake,
chaff, FCS G-limiter or engine spool-up is modeled; throttle stays at the plan
value. Coordinates here are east, north, up; missile_sim uses x, up, z with
north = -z.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass
import math
from types import SimpleNamespace

from .climb import valid_number
from .contracts import G
from .turn import ManeuverModel, add, cross, dot, norm, rotate, scale, transport, unit

KINDS = ("beam", "drag")
SUBSTEP_S = 1/48  # One missile control tick; AoA/roll time constants are >= 0.35 s.


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

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"evasion kind must be one of {KINDS}")
        positive = ("max_load", "alpha_max_deg", "roll_rate_deg_s", "load_response_s", "turn_time_constant_s")
        for name in positive:
            if not valid_number(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not valid_number(self.start_s) or self.start_s < 0:
            raise ValueError("start_s must be nonnegative")
        if not valid_number(self.dive_deg) or not 0 <= self.dive_deg < 60:
            raise ValueError("dive_deg must be in [0, 60)")


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


@dataclass(frozen=True)
class _State:
    position: tuple[float, float, float]
    velocity: tuple[float, float, float]
    normal: tuple[float, float, float]
    aoa_deg: float


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
        self.fault = None
        start = base.state_at(pilot.start_s)
        position, velocity = to_enu(start.position), to_enu(start.velocity)
        direction = unit(velocity)
        normal = unit(add((0., 0., 1.), scale(direction, -direction[2])))
        # Wings level at the scenario's straight flight: lift = weight·cos γ.
        alpha = self._alpha_for(position[2], norm(velocity), math.sqrt(1-direction[2]**2), 0.)
        self._times = [pilot.start_s]
        self._states = [_State(position, velocity, normal, alpha)]
        self.min_speed_mps = self.final_speed_mps = norm(velocity)
        self.min_altitude_m = position[2]

    def observe_missile(self, time_s, position, velocity):
        self._missile, self._missile_time = to_enu(position), time_s
        if self._launcher is None:
            # At launch the missile is at the launcher with the launcher's velocity.
            self._launcher = (time_s, self._missile, to_enu(velocity))

    def describe(self):
        p = self.pilot
        return dict(kind=p.kind, start_s=p.start_s, model="wt_overlay_fm_pilot", max_load=p.max_load,
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
        while self._times[-1] < time_s-1e-12:
            self._advance()
        i = bisect.bisect_left(self._times, time_s-1e-12)
        if abs(self._times[i]-time_s) <= 1e-12:
            s = self._states[i]
            return self.make_state(position=from_enu(s.position), velocity=from_enu(s.velocity))
        t0, t1 = self._times[i-1], self._times[i]
        a, b = self._states[i-1], self._states[i]
        w = (time_s-t0)/(t1-t0)
        mix = lambda x, y: tuple(p+w*(q-p) for p, q in zip(x, y))  # noqa: E731
        return self.make_state(position=from_enu(mix(a.position, b.position)),
                               velocity=from_enu(mix(a.velocity, b.velocity)))

    def _alpha_for(self, altitude, speed, load, reference):
        ceiling = self.pilot.alpha_max_deg
        q_lift_ceiling = self.model.forces_at_aoa(altitude, speed, ceiling, self.pilot.throttle_percent)[2]
        if load*self.model.mass*G >= q_lift_ceiling:
            return ceiling
        return min(ceiling, self.model.target_aoa(altitude, speed, load, reference))

    def _desired(self, position, velocity):
        source = self._reference(position)
        if source is None:
            return None
        los = (position[0]-source[0], position[1]-source[1], 0.)
        if norm(los) < 1e-9:
            return None
        away = unit(los)
        if self.pilot.kind == "drag":
            horizontal = away
        else:
            if self._beam_sign is None:
                # Lock the beam side nearest the heading at the first evasive step.
                self._beam_sign = 1. if cross(away, velocity)[2] >= 0 else -1.
            horizontal = (-away[1]*self._beam_sign, away[0]*self._beam_sign, 0.)
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
        # Level off early enough to arrest the sink rate at the floor with the
        # load limit (circular pull-out radius V²/((n-1)g), 50% margin).
        sink = max(0., -velocity[2])
        pull_out = 1.5*sink*sink/(2*max(.5, self.pilot.max_load-1)*G) if sink else 0.
        dive = 0. if position[2] < self.pilot.floor_m+pull_out else math.radians(self.pilot.dive_deg)
        return add(scale(horizontal, math.cos(dive)), (0., 0., -math.sin(dive)))

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
            s = _State(add(s.position, scale(s.velocity, h)), s.velocity, s.normal, s.aoa_deg)
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
            if norm(perp) > 1e-9:
                demand = add(demand, scale(unit(perp), speed*error/p.turn_time_constant_s))
        load = min(norm(demand)/G, p.max_load)
        # Roll the lift vector toward the demand at the limited rate.
        want = unit(demand) if norm(demand) > 1e-9 else s.normal
        phi = math.atan2(dot(direction, cross(s.normal, want)), dot(s.normal, want))
        limit = math.radians(p.roll_rate_deg_s)*h
        normal = rotate(s.normal, direction, max(-limit, min(limit, phi)))
        # Load is only useful once the lift vector points roughly at the demand.
        load *= max(0., dot(normal, want))
        alpha_target = self._alpha_for(s.position[2], speed, load, s.aoa_deg)
        alpha_mid = alpha_target+(s.aoa_deg-alpha_target)*math.exp(-h/(2*p.load_response_s))
        alpha_end = alpha_target+(s.aoa_deg-alpha_target)*math.exp(-h/p.load_response_s)
        thrust, drag, lift, alpha = self.model.forces_at_aoa(s.position[2], speed, alpha_mid, p.throttle_percent)
        mass = self.model.mass
        acceleration = add(add(scale(direction, (thrust*math.cos(alpha)-drag)/mass),
                               scale(normal, (lift+thrust*math.sin(alpha))/mass)), (0., 0., -G))
        velocity = add(s.velocity, scale(acceleration, h))
        normal = transport(normal, direction, unit(velocity))
        position = add(s.position, scale(add(s.velocity, velocity), h/2))
        return _State(position, velocity, normal, alpha_end)


def fm_evader_factory(pilot: EvasionPilot, model: ManeuverModel, make_state=SimpleNamespace,
                      perception: Perception | None = None):
    return lambda base: FMEvader(base, pilot, model, make_state, perception)
