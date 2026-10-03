"""Keyboard maneuver reference: continuous AoA and finite roll/pitch response.

The response constants are explicit user-adjustable assumptions, not recovered
Instructor physics. No input is sent to the game. See README for the model scope.
Coordinates are east, north, up; roll is positive right-wing-down.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from bisect import bisect_left
from collections import deque
from dataclasses import dataclass, replace
import math
from threading import Event
import time

from .climb import valid_number
from .contracts import G, KeyboardTurnGuidance, KeyboardTurnSettings
from .fm import atmosphere
from .fm.aero_table import AeroForceTable, QUERY_AOA_MAX, QUERY_AOA_MIN

MIN_AOA_DEG, MAX_AOA_DEG = QUERY_AOA_MIN, QUERY_AOA_MAX  # Forward-flight query bounds, not stall angles.
COMMIT_LOCK_S = .25
REPLACE_REMAINING_DEG = 2.
STALE_RESULT_S = 2.5
REPLAN_S = .8


def dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def add(a, b):
    return tuple(x+y for x, y in zip(a, b))


def scale(a, k):
    return tuple(x*k for x in a)


def norm(a):
    return math.sqrt(dot(a, a))


def unit(a):
    length = norm(a)
    if length < 1e-9:
        raise ValueError("方向不可用")
    return scale(a, 1/length)


def angle(a, b):
    return math.degrees(math.acos(max(-1., min(1., dot(unit(a), unit(b))))))


def rotate(vector, axis, radians):
    c, s = math.cos(radians), math.sin(radians)
    return add(add(scale(vector, c), scale(cross(axis, vector), s)), scale(axis, dot(axis, vector)*(1-c)))


def transport(normal, old_direction, new_direction):
    axis = cross(old_direction, new_direction)
    sine = norm(axis)
    if sine > 1e-9:
        normal = rotate(normal, scale(axis, 1/sine), math.atan2(sine, dot(old_direction, new_direction)))
    return unit(add(normal, scale(new_direction, -dot(normal, new_direction))))


def validate_settings(s: KeyboardTurnSettings):
    if s.angle_deg not in (30, 45, 90, 120):
        raise ValueError("目标转角须为 30、45、90 或 120 度")
    bounds = {"horizon_s": (3, 30), "max_altitude_loss_m": (0, 5000),
              "minimum_tas_mps": (50, 650), "max_load": (1, 12), "min_load": (-5, 0),
              "roll_rate_deg_s": (10, 360), "roll_response_s": (.1, 2),
              "load_response_s": (.1, 3), "reaction_s": (0, 1.5), "hold_s": (.8, 3),
              "throttle_rate_percent_s": (5, 200), "engine_response_s": (.1, 5)}
    for key, (low, high) in bounds.items():
        value = getattr(s, key)
        if not valid_number(value) or not low <= value <= high:
            raise ValueError(f"{key} 须在 {low}–{high} 之间")


@dataclass(frozen=True)
class Motion:
    altitude: float
    velocity: tuple[float, float, float]
    normal: tuple[float, float, float]
    roll_rate: float  # Radians/s about velocity, not an Euler-angle derivative.
    load: float       # Aerodynamic L/W, never telemetry body Ny.
    throttle_percent: float = 110.
    engine_throttle_percent: float = 110.
    aoa_deg: float | None = None  # Live states always carry AoA, including postcritical branches.
    airbrake_fraction: float = 0.  # Internal opening; 8111 does not report it.

    @property
    def speed(self):
        return norm(self.velocity)

    @property
    def energy(self):
        return self.altitude+dot(self.velocity, self.velocity)/(2*G)


def flight_frame(state, *, enforce_envelope=True):
    if not state.valid:
        raise ValueError("等待有效飞行数据")
    fields = ((state.tas_mps, "真空速 TAS"), (state.altitude_m, "海拔高度"),
              (state.heading_deg, "航向 compass"), (state.pitch_deg, "俯仰 aviahorizon_pitch"),
              (state.roll_deg, "滚转 aviahorizon_roll"), (state.aoa_deg, "迎角 AoA"),
              (state.aos_deg, "侧滑 AoS"), (state.vertical_speed_mps, "垂直速度 Vy"))
    missing = [label for value, label in fields if not valid_number(value)]
    if missing:
        raise ValueError("缺少转向读数："+"、".join(missing))
    if state.tas_mps <= 0 or enforce_envelope and state.tas_mps < 50:
        raise ValueError("当前 TAS 低于转向计算下限 180 km/h")
    if enforce_envelope and (abs(state.aos_deg) > 10 or not MIN_AOA_DEG <= state.aoa_deg <= MAX_AOA_DEG):
        raise ValueError("姿态超出转向参考范围")
    heading, pitch, roll, alpha, beta = map(math.radians,
        (state.heading_deg, state.pitch_deg, state.roll_deg, state.aoa_deg, state.aos_deg))
    forward = (math.cos(pitch)*math.sin(heading), math.cos(pitch)*math.cos(heading), math.sin(pitch))
    right = (math.cos(heading), -math.sin(heading), 0.)
    up = (-math.sin(pitch)*math.sin(heading), -math.sin(pitch)*math.cos(heading), math.cos(pitch))
    body_up = add(scale(up, math.cos(roll)), scale(right, math.sin(roll)))
    body_right = add(scale(right, math.cos(roll)), scale(up, -math.sin(roll)))
    direction = add(scale(add(scale(forward, math.cos(alpha)), scale(body_up, -math.sin(alpha))),
                          math.cos(beta)), scale(body_right, math.sin(beta)))
    # Vy anchors flight path; heading is reconstructed using AoA/AoS, not the nose.
    vertical = state.vertical_speed_mps/state.tas_mps
    if abs(vertical) > 1 or math.hypot(*direction[:2]) < .05 or enforce_envelope and abs(vertical) >= .985:
        raise ValueError("近垂直飞行暂不提供指令")
    horizontal = unit((direction[0], direction[1], 0.))
    direction = add(scale(horizontal, math.sqrt(1-vertical*vertical)), (0., 0., vertical))
    normal = unit(add(body_up, scale(direction, -dot(body_up, direction))))
    return scale(direction, state.tas_mps), normal


@dataclass(frozen=True)
class VelocityGoal:
    """A frozen initial direction, or an externally supplied complete LOS vector.

    Beam mode is a geometric interface only. Its caller must refresh the LOS and
    enforce freshness; the UI does not infer a threat from ownship telemetry.
    """
    direction: tuple[float, float, float]
    angle_deg: float = 90.
    kind: str = "turn"
    tolerance_deg: float = 3.

    def remaining(self, velocity):
        if self.kind == "beam":
            return max(0., abs(90-angle(velocity, self.direction))-self.tolerance_deg)
        if self.kind != "turn":
            raise ValueError("unknown velocity goal")
        return max(0., self.angle_deg-angle(velocity, self.direction))


@dataclass(frozen=True)
class Action:
    roll: int
    pitch: int
    throttle: int = 0
    airbrake: int = 0  # 0 retract / hold in, 1 deploy; player switch, not a polar axis.

    @property
    def label(self):
        parts = [x for x in (("左滚", "停止滚转", "右滚")[self.roll+1],
                            ("推杆", "松杆", "拉杆")[self.pitch+1])]
        return "＋".join(parts)


ACTIONS = tuple(Action(roll, pitch, throttle) for throttle in (0, -1, 1)
                for roll in (0, -1, 1) for pitch in (1, 0, -1))


class ManeuverModel:
    """Full static polars at continuous AoA via a Mach–AoA L/q, D/q table."""
    def __init__(self, model, mass, afterburner=True, sweep=0.):
        if not valid_number(mass) or mass <= 0:
            raise ValueError("需要参考总质量")
        if not hasattr(model, "components_at_sweep") or not hasattr(model, "engines"):
            raise ValueError("机型不支持转向参考")
        self.model, self.mass, self.afterburner, self.sweep = model, mass, afterburner, sweep
        self.max_throttle = 110. if afterburner and any(e.has_wep for e in model.engines) else 100.
        self._aero = AeroForceTable.from_aircraft(model, sweep)
        self._aoa_cache = {}
        self._thrust_cache = {}
        has = bool(getattr(model, "has_airbrake", False))
        cd = getattr(model, "airbrake_cd", 0.) or 0.
        area = getattr(model, "airbrake_ref_area_m2", 0.) or 0.
        self.airbrake_dq = area*cd if has and cd > 0 else 0.
        speed = getattr(model, "airbrake_speed", .5)
        self.airbrake_speed = speed if valid_number(speed) and speed >= 0 else .5

    def _atmosphere(self, altitude, speed):
        rho, sound = atmosphere(altitude)
        if speed < 50 or speed/sound > 2.35:
            raise ValueError("超出模型范围")
        return .5*rho*speed*speed, speed/sound

    def _thrusts(self, altitude, speed):
        key = (int(altitude), int(speed*2))
        cached = self._thrust_cache.get(key)
        if cached is not None:
            return cached
        thrusts = tuple((e, e.thrust_n(altitude, speed, False), e.thrust_n(altitude, speed, True))
                        for e in self.model.engines)
        if len(self._thrust_cache) > 4096:
            self._thrust_cache.clear()
        self._thrust_cache[key] = thrusts
        return thrusts

    def target_aoa(self, altitude, speed, load, reference_deg=0.):
        """Bracket each sampled crossing; never bisect the entire nonmonotone polar."""
        q, mach = self._atmosphere(altitude, speed)
        key = (int(mach*500), int(q), round(load, 2), int(reference_deg*2))
        cached = self._aoa_cache.get(key)
        if cached is not None:
            return cached
        alpha = self._aero.target_aoa(mach, load*self.mass*G/q, reference_deg)
        if len(self._aoa_cache) > 4096:
            self._aoa_cache.clear()
        self._aoa_cache[key] = alpha
        return alpha

    def observed_load(self, altitude, speed, alpha):
        if not valid_number(alpha) or not MIN_AOA_DEG <= alpha <= MAX_AOA_DEG:
            raise ValueError("迎角超出转向计算范围 ±60°")
        q, mach = self._atmosphere(altitude, speed)
        return q*self._aero.lookup(mach, alpha)[0]/(self.mass*G)

    def forces_at_aoa(self, altitude, speed, alpha, throttle_percent=None, airbrake_fraction=0.):
        if not valid_number(alpha) or not MIN_AOA_DEG <= alpha <= MAX_AOA_DEG:
            raise ValueError("迎角超出转向计算范围 ±60°")
        q, mach = self._atmosphere(altitude, speed)
        throttle = self.max_throttle if throttle_percent is None else throttle_percent
        # Same steady throttle law as static SEP; spool lag is applied by the caller.
        thrust = sum(engine.blend(military, maximum, min(110., max(0., throttle)))
                     for engine, military, maximum in self._thrusts(altitude, speed))
        lift_q, drag_q = self._aero.lookup(mach, alpha)
        opening = 0. if not valid_number(airbrake_fraction) else min(1., max(0., airbrake_fraction))
        drag_q += self.airbrake_dq*opening
        return thrust, drag_q*q, lift_q*q, math.radians(alpha)

    def forces(self, altitude, speed, load, throttle_percent=None, *, reference_deg=0.,
               airbrake_fraction=0.):
        alpha = self.target_aoa(altitude, speed, load, reference_deg)
        return self.forces_at_aoa(altitude, speed, alpha, throttle_percent, airbrake_fraction)

    def _airbrake_opening(self, current, action, dt):
        if action is None or self.airbrake_dq <= 0:
            return current
        target = 1. if action.airbrake else 0.
        if self.airbrake_speed <= 0:
            return target
        step = self.airbrake_speed*dt
        if target > current:
            return min(target, current+step)
        return max(target, current-step)

    def step(self, state: Motion, action: Action | None, dt, settings):
        # None is the human response interval: continue the observed roll/AoA.
        p_target = state.roll_rate if action is None else math.radians(settings.roll_rate_deg_s)*action.roll
        p = p_target+(state.roll_rate-p_target)*math.exp(-dt/settings.roll_response_s)
        alpha0 = state.aoa_deg
        if alpha0 is None:  # Compatibility for load-only offline initial states.
            alpha0 = self.target_aoa(state.altitude, state.speed, state.load)
        if not valid_number(alpha0) or not MIN_AOA_DEG <= alpha0 <= MAX_AOA_DEG:
            raise ValueError("迎角超出转向计算范围 ±60°")
        alpha_target = alpha0
        if action is not None:
            n_target = settings.max_load if action.pitch > 0 else settings.min_load if action.pitch < 0 else state.normal[2]
            # Held pitch selects the root nearest current AoA. Releasing pitch
            # requests the low-AoA branch, reached through a finite response.
            alpha_target = self.target_aoa(state.altitude, state.speed, n_target,
                                           alpha0 if action.pitch else 0.)
        alpha_mid = alpha_target+(alpha0-alpha_target)*math.exp(-dt/(2*settings.load_response_s))
        alpha_end = alpha_target+(alpha0-alpha_target)*math.exp(-dt/settings.load_response_s)
        throttle = state.throttle_percent
        if action is not None and action.throttle:
            demanded = throttle+action.throttle*settings.throttle_rate_percent_s*dt
            throttle = max(0., demanded) if action.throttle < 0 else min(self.max_throttle, demanded)
        engine = throttle+(state.engine_throttle_percent-throttle)*math.exp(-dt/settings.engine_response_s)
        opening0 = state.airbrake_fraction
        opening = self._airbrake_opening(opening0, action, dt)
        direction = unit(state.velocity)
        normal = rotate(state.normal, direction, (state.roll_rate+p)*dt/4)
        thrust, drag, lift, alpha = self.forces_at_aoa(state.altitude, state.speed, alpha_mid,
                                                      (state.engine_throttle_percent+engine)/2,
                                                      (opening0+opening)/2)
        acceleration = add(add(scale(direction, (thrust*math.cos(alpha)-drag)/self.mass),
                               scale(normal, (lift+thrust*math.sin(alpha))/self.mass)), (0., 0., -G))
        velocity = add(state.velocity, scale(acceleration, dt))
        normal = rotate(state.normal, direction, (state.roll_rate+p)*dt/2)
        normal = transport(normal, direction, unit(velocity))
        altitude = state.altitude+(state.velocity[2]+velocity[2])*dt/2
        load = self.observed_load(altitude, norm(velocity), alpha_end)
        return Motion(altitude, velocity, normal, p, load, throttle, engine, alpha_end, opening)


@dataclass(frozen=True)
class TurnStep:
    action: Action
    duration_s: float


def append_step(steps, action, duration):
    if steps and steps[-1].action == action:
        return (*steps[:-1], TurnStep(action, steps[-1].duration_s+duration))
    return (*steps, TurnStep(action, duration))


def allowed_motion(s, settings, floor, initial_load=None):
    # A measured initial overload may recover; no trajectory may exceed its
    # starting excess. Commanded targets still use the configured load limits.
    low = min(settings.min_load, initial_load) if initial_load is not None else settings.min_load
    high = max(settings.max_load, initial_load) if initial_load is not None else settings.max_load
    return (floor <= s.altitude <= 20000 and s.speed >= settings.minimum_tas_mps
            and abs(s.velocity[2])/s.speed < .985
            and low-.2 <= s.load <= high+.2)


@dataclass(frozen=True)
class TurnPlan:
    initial: Motion
    action: Action | None
    duration_s: float | None
    energy_change_m: float | None
    reached: bool
    expansions: int = 0
    steps: tuple[TurnStep, ...] = ()


class Cancelled(Exception):
    pass


def _replay_steps(model, initial, start, delay, steps, goal, settings, allowed, expansions):
    state, elapsed = start, delay
    built = ()
    for step in steps:
        t = 0.
        while t < step.duration_s-1e-9:
            dt = min(.15, step.duration_s-t)
            try:
                state = model.step(state, step.action, dt, settings)
            except (ValueError, OverflowError):
                return None
            if not allowed(state):
                return None
            t += dt
            elapsed += dt
            if goal.remaining(state.velocity) == 0:
                done = append_step(built, step.action, t)
                return TurnPlan(initial, done[0].action, elapsed, state.energy-initial.energy,
                                True, expansions, done), 0.
        built = append_step(built, step.action, step.duration_s)
    remaining = goal.remaining(state.velocity)
    return TurnPlan(initial, built[0].action if built else None,
                    elapsed if remaining == 0 else None, state.energy-initial.energy,
                    remaining == 0, expansions, built), remaining


def _airbrake_schedules(steps, lock_first=False):
    if not steps:
        return []
    if lock_first:
        if len(steps) < 2:
            return []
        first, second, *rest = steps
        flipped = replace(second.action, airbrake=0 if second.action.airbrake else 1)
        return [(first, TurnStep(flipped, second.duration_s), *rest)]
    first = steps[0]
    variants = [(TurnStep(replace(first.action, airbrake=1), first.duration_s), *steps[1:])]
    if len(steps) >= 2:
        second = steps[1]
        variants.append((first, TurnStep(replace(second.action, airbrake=1), second.duration_s), *steps[2:]))
        variants.append((TurnStep(replace(first.action, airbrake=1), first.duration_s),
                         TurnStep(replace(second.action, airbrake=1), second.duration_s), *steps[2:]))
    return variants


def _prefer_airbrake(model, initial, start, delay, plan, goal, settings, allowed, expansions,
                     lock_first=False):
    if model.airbrake_dq <= 0 or plan.action is None:
        return plan
    steps = plan.steps or (TurnStep(plan.action, settings.hold_s),)
    baseline = _replay_steps(model, initial, start, delay, steps, goal, settings, allowed, expansions)
    if baseline is None:
        return plan
    best, remaining = baseline
    best_key = (0 if best.reached else 1, best.duration_s if best.reached else remaining,
                -(best.energy_change_m or 0))
    for variant in _airbrake_schedules(steps, lock_first):
        replayed = _replay_steps(model, initial, start, delay, variant, goal, settings, allowed, expansions)
        if replayed is None:
            continue
        cand, rem = replayed
        key = (0 if cand.reached else 1, cand.duration_s if cand.reached else rem,
               -(cand.energy_change_m or 0))
        if key < best_key:
            best, best_key = cand, key
    return best


def search_turn(model, initial, goal, settings, floor, cancel=None, *, budget_s=.8, beam_width=27,
                skip_reaction=False, commit=None):
    """Bounded beam search over held actions; no global-optimality guarantee.

    Keeps different initial actions alive so early rolling/unloading is not
    immediately discarded in favor of instant angular progress.
    """
    validate_settings(settings)
    deadline = time.monotonic()+budget_s
    def allowed(s):
        return allowed_motion(s, settings, floor, initial.load)
    if not allowed(initial):
        raise ValueError("当前状态超出所设机动限制")
    if goal.remaining(initial.velocity) == 0:
        return TurnPlan(initial, None, 0., 0., True)
    state, delay, expansions = initial, 0., 0
    if not skip_reaction:
        while delay < settings.reaction_s-1e-9:
            if cancel is not None and cancel.is_set():
                raise Cancelled
            dt = min(.1, settings.reaction_s-delay)
            state = model.step(state, None, dt, settings)
            if not allowed(state):
                return TurnPlan(initial, None, None, None, False)
            delay += dt
        if goal.remaining(state.velocity) == 0:
            return TurnPlan(initial, Action(0, 0), delay, state.energy-initial.energy, True)
    start = state
    if commit is not None:
        t, trial, held = 0., start, True
        while t < commit.duration_s-1e-9:
            if cancel is not None and cancel.is_set():
                raise Cancelled
            if time.monotonic() >= deadline:
                return TurnPlan(initial, commit.action, None, None, False, expansions, (commit,))
            dt = min(.15, commit.duration_s-t)
            try:
                trial = model.step(trial, commit.action, dt, settings)
            except (ValueError, OverflowError):
                held = False
                break
            expansions += 1
            if not allowed(trial):
                held = False
                break
            t += dt
            if goal.remaining(trial.velocity) == 0:
                done = (TurnStep(commit.action, t),)
                return TurnPlan(initial, commit.action, delay+t, trial.energy-initial.energy,
                                True, expansions, done)
        if held:
            start, delay = trial, delay+commit.duration_s
            beam = [(start, (commit,))]
        else:
            commit = None
            beam = [(start, ())]
    else:
        beam = [(start, ())]
    reacted = start
    best = None
    elapsed = delay
    def finish(plan):
        return _prefer_airbrake(model, initial, reacted, delay, plan, goal, settings, allowed, expansions,
                                lock_first=commit is not None)
    while elapsed < settings.horizon_s-1e-9:
        candidates, reached = [], []
        segment = min(settings.hold_s, settings.horizon_s-elapsed)
        for current, steps in beam:
            for action in ACTIONS:
                if len(steps) >= 4 and action != steps[-1].action:
                    continue
                if (action.throttle < 0 and current.throttle_percent <= 0
                        or action.throttle > 0 and current.throttle_percent >= model.max_throttle):
                    continue
                if cancel is not None and cancel.is_set():
                    raise Cancelled
                if time.monotonic() >= deadline:
                    if reached:
                        return finish(min(reached, key=lambda p: (p.duration_s, -p.energy_change_m)))
                    return finish(best or TurnPlan(initial, None, None, None, False, expansions))
                trial, t = current, 0.
                command = steps[0].action if steps else action
                feasible = True
                while t < segment-1e-9:
                    dt = min(.15, segment-t)
                    try:
                        trial = model.step(trial, action, dt, settings)
                    except (ValueError, OverflowError):
                        feasible = False
                        break
                    expansions += 1
                    if not allowed(trial):
                        feasible = False
                        break
                    t += dt
                    if goal.remaining(trial.velocity) == 0:
                        # First crossing is resolved on the integration grid.
                        reached.append(TurnPlan(initial, command, elapsed+t,
                                                trial.energy-initial.energy, True, expansions,
                                                append_step(steps, action, t)))
                        feasible = False
                        break
                if feasible:
                    candidates.append((trial, append_step(steps, action, segment)))
        if reached:
            return finish(min(reached, key=lambda p: (p.duration_s, -p.energy_change_m)))
        if not candidates:
            break
        candidates.sort(key=lambda x: (goal.remaining(x[0].velocity), len(x[1]), -x[0].energy))
        best_state, best_steps = candidates[0]
        if goal.remaining(best_state.velocity) < goal.remaining(initial.velocity)-.25:
            best = TurnPlan(initial, best_steps[0].action, None, None, False, expansions, best_steps)
        beam, seen = [], set()
        for candidate in candidates:
            first = candidate[1][0].action
            if first not in seen:
                beam.append(candidate)
                seen.add(first)
                if len(beam) >= beam_width:
                    break
        elapsed += segment
    return finish(best or TurnPlan(initial, None, None, None, False, expansions))


def remaining_steps(execution, elapsed):
    if not execution.steps or elapsed >= execution.ends[-1]-1e-9:
        return ()
    i = execution.index(elapsed)
    leftover = execution.ends[i]-elapsed
    rest = execution.steps[i+1:]
    if leftover > 1e-9:
        return (TurnStep(execution.steps[i].action, leftover), *rest)
    return rest


def truncate_steps(steps, horizon):
    if horizon <= 1e-9:
        return ()
    out, t = [], 0.
    for step in steps:
        if t >= horizon-1e-9:
            break
        take = min(step.duration_s, horizon-t)
        if take > 1e-9:
            out.append(TurnStep(step.action, take))
        t += take
    return tuple(out)


def replay_from(model, motion, steps, goal, settings, floor):
    """Integrate steps from the latest observation. None if a constraint fails."""
    if not steps:
        return True, goal.remaining(motion.velocity), None
    allowed = lambda s: allowed_motion(s, settings, floor, motion.load)
    result = _replay_steps(model, motion, motion, 0., steps, goal, settings, allowed, 0)
    if result is None:
        return False, None, None
    plan, remaining = result
    return True, remaining, plan


def nearby(expected, actual, *, coarse=False):
    factor = 2.5 if coarse else 1.
    return (angle(expected.velocity, actual.velocity) <= 12*factor
            and angle(expected.normal, actual.normal) <= 25*factor
            and abs(expected.speed-actual.speed) <= 25*factor
            and abs(expected.altitude-actual.altitude) <= 150*factor
            and abs(expected.load-actual.load) <= 2.5*factor
            and (expected.aoa_deg is None or actual.aoa_deg is None
                 or abs(expected.aoa_deg-actual.aoa_deg) <= 8*factor)
            and abs(expected.throttle_percent-actual.throttle_percent) <= 20*factor
            and abs(expected.roll_rate-actual.roll_rate) <= math.radians(70*factor))


@dataclass(frozen=True)
class Execution:
    steps: tuple[TurnStep, ...]
    ends: tuple[float, ...]
    times: tuple[float, ...]
    states: tuple[Motion, ...]
    reached: bool

    def reference(self, elapsed):
        i = min(bisect_left(self.times, elapsed), len(self.times)-1)
        if i == 0:
            return self.states[0]
        left, right = self.states[i-1], self.states[i]
        f = max(0., min(1., (elapsed-self.times[i-1])/(self.times[i]-self.times[i-1])))
        def blend(a, b):
            return a+(b-a)*f
        velocity = tuple(blend(a, b) for a, b in zip(left.velocity, right.velocity))
        normal = unit(tuple(blend(a, b) for a, b in zip(left.normal, right.normal)))
        normal = unit(add(normal, scale(unit(velocity), -dot(normal, unit(velocity)))))
        return Motion(blend(left.altitude, right.altitude), velocity, normal,
                      blend(left.roll_rate, right.roll_rate), blend(left.load, right.load),
                      blend(left.throttle_percent, right.throttle_percent),
                      blend(left.engine_throttle_percent, right.engine_throttle_percent),
                      blend(left.aoa_deg, right.aoa_deg)
                      if left.aoa_deg is not None and right.aoa_deg is not None else None,
                      blend(left.airbrake_fraction, right.airbrake_fraction))

    def index(self, elapsed):
        return min(bisect_left(self.ends, elapsed), len(self.steps)-1)


def prepare_execution(model, motion, plan, goal, settings, floor, *, include_reaction=True):
    """Rebase the selected sequence on the current measured state before display."""
    requested = plan.steps or (TurnStep(plan.action, settings.hold_s),)
    initial_load = motion.load
    times, states, steps, ends = [0.], [motion], [], []
    elapsed = 0.
    lead = ((None, settings.reaction_s),) if include_reaction else ()
    for action, duration in (*lead, *((x.action, x.duration_s) for x in requested)):
        dt_left = duration
        while dt_left > 1e-9:
            dt = min(.15, dt_left)
            motion = model.step(motion, action, dt, settings)
            if not allowed_motion(motion, settings, floor, initial_load):
                raise ValueError("动作序列超出所设限制，重新规划")
            elapsed += dt
            dt_left -= dt
            times.append(elapsed); states.append(motion)
            if action is not None and goal.remaining(motion.velocity) == 0:
                steps.append(TurnStep(action, duration-dt_left)); ends.append(elapsed)
                return Execution(tuple(steps), tuple(ends), tuple(times), tuple(states), True)
        if action is not None:
            steps.append(TurnStep(action, duration)); ends.append(elapsed)
    return Execution(tuple(steps), tuple(ends), tuple(times), tuple(states), False)


class TurnSession:
    """Commit to a readable sequence; replan the tail from the latest observation."""
    def __init__(self):
        self.executor = None
        self.reset()

    def reset(self, *, require_restart=False, reason=""):
        if getattr(self, "cancel", None) is not None:
            self.cancel.set()
        if getattr(self, "future", None) is not None:
            self.future.cancel()
        self.future = self.cancel = self.goal = self.previous = self.plan = None
        self.model = self.signature = self.execution = None
        self.execution_index = 0
        self.last_submit = self.plan_time = -math.inf
        self.action = self.rate = None
        self.require_restart, self.restart_reason = require_restart, reason
        self.last_sample_time = self.engine_throttle = None
        self.completed = False
        self.floor = None
        self.turned = self.remaining = None
        self.progress_current = False
        self.progress_samples = deque(maxlen=12)
        self.following = self.prepare_release = False
        self.incumbent_ok = False
        self._schedule_mark = None
        self._shown_elapsed = 0.

    def record_progress(self, direction, now):
        self.turned = angle(self.goal.direction, direction)
        self.remaining = max(0., self.goal.angle_deg-self.turned)
        self.progress_current = True
        if self.progress_samples and not 0 < now-self.progress_samples[-1][0] <= .6:
            self.progress_samples.clear()
        self.progress_samples.append((now, self.turned))
        while len(self.progress_samples) > 1 and now-self.progress_samples[0][0] > .6:
            self.progress_samples.popleft()

    def progress_rate(self):
        samples = self.progress_samples
        if len(samples) < 3 or samples[-1][0]-samples[0][0] < .15:
            return None
        times = [t-samples[0][0] for t, _ in samples]
        mean_t = sum(times)/len(times)
        mean_angle = sum(a for _, a in samples)/len(samples)
        return sum((t-mean_t)*(a-mean_angle) for t, (_, a) in zip(times, samples))/sum(
            (t-mean_t)**2 for t in times)

    def follow_guidance(self, state, settings, reason=""):
        """Measured progress cues only; no FM action sequence or arrival-time claim."""
        rate = self.progress_rate()
        lead = max(2., (rate or 0.)*(settings.reaction_s+settings.load_response_s))
        if rate is not None and rate > .5 and self.remaining <= lead:
            self.prepare_release = True
        if self.prepare_release:
            action, next_action = "准备松键", "到达目标后松开机动键"
        elif rate is not None and rate <= .5:
            action, next_action = "检查转向", "转角未增加"
        else:
            action, next_action = "保持机动", "接近目标时准备松键"
        return KeyboardTurnGuidance(True, "机动跟随", action, self.turned, self.remaining,
            reason=reason, throttle_percent=state.throttle_percent, next_action=next_action)

    def follow_without_model(self, state, settings, reason):
        self.pause(reason, keep_previous=True, current=True)
        self.following = True
        return self.follow_guidance(state, settings, reason)

    def invalidate(self, reason="机型或参考设置已改变，请重新开始转向"):
        self.reset(require_restart=self.require_restart or self.goal is not None, reason=reason)

    def status(self, phase, reason="", *, current=False):
        return KeyboardTurnGuidance(phase=phase, reason=reason, turned_deg=self.turned,
                                    remaining_deg=self.remaining, progress_stale=not current)

    def pause(self, reason="", phase="等待数据", *, keep_previous=False, current=False):
        if self.cancel is not None:
            self.cancel.set()
        if self.future is not None:
            self.future.cancel()
        self.future = self.cancel = self.plan = self.action = self.execution = None
        self.last_submit = -math.inf
        self.rate = None
        if not current:
            self.progress_samples.clear()
            self.prepare_release = False
        if not keep_previous:
            self.previous = None
        return self.status(phase, reason, current=current)

    def close(self):
        self.reset()
        if self.executor:
            self.executor.shutdown(wait=False, cancel_futures=True)
            self.executor = None

    def _elapsed(self, now):
        return max(0., now-self.plan_time)

    def _schedule_elapsed(self, now, motion):
        """Advance the displayed step only while the observation still matches the plan."""
        if self.execution is None:
            self._schedule_mark = None
            self._shown_elapsed = 0.
            return 0.
        wall = min(self._elapsed(now), self.execution.ends[-1])
        if self._schedule_mark is not self.execution or wall+1e-9 < self._shown_elapsed:
            self._schedule_mark = self.execution
            self._shown_elapsed = 0.
        if wall > self._shown_elapsed+1e-9 and nearby(self.execution.reference(wall), motion, coarse=True):
            self._shown_elapsed = wall
        return self._shown_elapsed

    def _refresh_incumbent(self, motion, settings, shown):
        if self.execution is None:
            self.incumbent_ok = False
            return
        rem = remaining_steps(self.execution, shown)
        if not rem:
            self.incumbent_ok = False
            return
        ok, end_rem, _ = replay_from(self.model, motion, rem[:1], self.goal, settings, self.floor)
        now_rem = self.goal.remaining(motion.velocity)
        reverse = ok and end_rem is not None and end_rem > now_rem+3.
        self.incumbent_ok = ok and not reverse

    def _commit_step(self, shown):
        if self.execution is None or not self.incumbent_ok:
            return None
        leftover = remaining_steps(self.execution, shown)
        if leftover and leftover[0].duration_s > COMMIT_LOCK_S:
            return leftover[0]
        return None

    def _take_completed_search(self, motion, state, settings, shown):
        future, self.future = self.future, None
        try:
            candidate = future.result()
        except Cancelled:
            return
        if candidate.action is None:
            return
        age = state.time_s-self.last_submit
        if not math.isfinite(self.last_submit) or not 0 <= age <= STALE_RESULT_S:
            return
        rem = remaining_steps(self.execution, shown) if self.execution else ()
        steps = candidate.steps or (TurnStep(candidate.action, settings.hold_s),)
        if rem and steps[0].action == rem[0].action and rem[0].duration_s > 1e-9:
            shrink = max(0., state.time_s-self.last_submit)
            duration = max(rem[0].duration_s, steps[0].duration_s-shrink)
            steps = (TurnStep(steps[0].action, duration), *steps[1:])
            candidate = replace(candidate, steps=steps)
        include_reaction = self.execution is None
        try:
            execution = prepare_execution(self.model, motion, candidate, self.goal, settings, self.floor,
                                          include_reaction=include_reaction)
        except ValueError:
            return
        if not self._should_replace(execution, motion, settings, shown):
            return
        self.execution = execution
        self.plan, self.plan_time = candidate, state.time_s
        self.following = self.prepare_release = False
        self.incumbent_ok = True

    def _should_replace(self, execution, motion, settings, shown):
        if self.execution is None or not self.incumbent_ok:
            return True
        rem = remaining_steps(self.execution, shown)
        new_first = execution.steps[0].action
        old_first = rem[0].action if rem else self.execution.steps[-1].action
        if new_first == old_first:
            return True
        if not rem:
            return True
        if rem[0].duration_s > COMMIT_LOCK_S:
            return False
        horizon = sum(step.duration_s for step in rem)
        ok_old, old_rem, _ = replay_from(self.model, motion, rem, self.goal, settings, self.floor)
        ok_new, new_rem, _ = replay_from(self.model, motion, truncate_steps(execution.steps, horizon),
                                         self.goal, settings, self.floor)
        if not ok_new or new_rem is None:
            return False
        if not ok_old or old_rem is None:
            return True
        if new_rem <= .05 < old_rem:
            return True
        return old_rem-new_rem >= REPLACE_REMAINING_DEG

    def _submit_search(self, motion, settings, now, commit):
        if self.executor is None:
            self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="wt-turn")
        self.cancel = Event()
        self.last_submit = now
        self.future = self.executor.submit(
            search_turn, self.model, motion, self.goal, settings, self.floor, self.cancel,
            skip_reaction=self.execution is not None, commit=commit)

    def execution_guidance(self, motion, now, shown=None):
        execution = self.execution
        elapsed = self._elapsed(now) if shown is None else shown
        i = execution.index(elapsed)
        self.execution_index = i
        action = self.action = execution.steps[i].action
        uses_brake = any(step.action.airbrake for step in execution.steps)
        next_action = (execution.steps[i+1].action.label+"／"+
            ("收油", "保持油门", "加油")[execution.steps[i+1].action.throttle+1]+
            (("／展开减速板" if execution.steps[i+1].action.airbrake else "／收起减速板") if uses_brake else "")
            if i+1 < len(execution.steps) else "达到目标后松键" if execution.reached else "继续规划")
        endpoint = execution.reference(execution.ends[i])
        return KeyboardTurnGuidance(True, "转向", action.label, self.turned, self.remaining,
            max(0., execution.ends[-1]-elapsed) if execution.reached else None,
            execution.states[-1].energy-motion.energy, action.roll, action.pitch,
            throttle_command=action.throttle, throttle_percent=motion.throttle_percent,
            target_throttle_percent=endpoint.throttle_percent, next_action=next_action,
            step_index=i+1, step_count=len(execution.steps),
            step_remaining_s=max(0., execution.ends[i]-elapsed),
            airbrake_command=action.airbrake if uses_brake else None)

    def update(self, state, fm, mass, afterburner, sweep, settings):
        self.progress_current = False
        if self.require_restart:
            return self.status("重新开始", self.restart_reason)
        if not state.valid:
            return self.pause("等待有效飞行数据")
        if self.goal is not None and self.last_sample_time is not None and state.time_s-self.last_sample_time > 2.5:
            self.invalidate("飞行数据中断超过 2.5 秒，请重新开始转向")
            return self.status("重新开始", self.restart_reason)
        previous_time = self.last_sample_time
        self.last_sample_time = state.time_s
        try:
            # Geometry remains measurable when the force model is out of range.
            velocity, normal = flight_frame(state, enforce_envelope=False)
            signature = (id(fm), afterburner, sweep, settings)
            if signature != self.signature:
                self.reset()
                self.signature = signature
                self.model = ManeuverModel(fm, mass, afterburner, sweep)
                self.last_sample_time = state.time_s
            elif abs(mass/self.model.mass-1) > .01:
                self.pause("质量变化，更新计划", keep_previous=True)
                self.model = ManeuverModel(fm, mass, afterburner, sweep)
            direction = unit(velocity)
            if self.goal is not None:
                self.record_progress(direction, state.time_s)
            if not valid_number(state.throttle_percent) or not 0 <= state.throttle_percent <= 110:
                return self.pause("需要有效油门读数", "缺少油门", current=self.progress_current)
            if self.previous is None:
                self.previous = (state.time_s, direction, normal)
                return self.status("读取姿态", current=self.progress_current)
            t0, v0, n0 = self.previous
            dt = state.time_s-t0
            self.previous = (state.time_s, direction, normal)
            if not .02 <= dt <= .6:
                return self.pause("采样间隔变化，正在重新读取姿态", "读取姿态",
                                  keep_previous=True, current=self.progress_current)
            previous_normal = transport(n0, v0, direction)
            rate = math.atan2(dot(direction, cross(previous_normal, normal)), dot(previous_normal, normal))/dt
            if abs(rate) > math.radians(400):
                return self.pause("姿态变化过快", "姿态跳变", current=self.progress_current)
            self.rate = rate if self.rate is None else self.rate+(rate-self.rate)*(1-math.exp(-dt/.25))
            if self.goal is None:
                self.goal = VelocityGoal(direction, settings.angle_deg)
                self.floor = max(0., state.altitude_m-settings.max_altitude_loss_m)
                self.record_progress(direction, state.time_s)
            if self.remaining <= .05:
                self.completed = True
            if self.completed:
                if self.cancel:
                    self.cancel.set()
                self.action = self.execution = None
                return KeyboardTurnGuidance(True, "到达", "松开机动键", self.turned, 0.)
            if state.altitude_m < self.floor:
                return self.pause("已低于本次机动高度下限；恢复高度或重新开始", "高度不足",
                                  keep_previous=True, current=True)
            if state.tas_mps < settings.minimum_tas_mps:
                return self.pause("当前 TAS 低于所设最低速度", "速度不足", keep_previous=True, current=True)
            flight_frame(state)  # Apply the prediction envelope after updating progress.
            load = self.model.observed_load(state.altitude_m, state.tas_mps, state.aoa_deg)
            if self.engine_throttle is None:
                self.engine_throttle = state.throttle_percent
            elif previous_time is not None:
                elapsed = max(0., state.time_s-previous_time)
                self.engine_throttle = state.throttle_percent+(self.engine_throttle-state.throttle_percent)*math.exp(
                    -elapsed/settings.engine_response_s)
            motion = Motion(state.altitude_m, velocity, normal, self.rate, load,
                            state.throttle_percent, self.engine_throttle, state.aoa_deg)
            shown = self._schedule_elapsed(state.time_s, motion)
            self._refresh_incumbent(motion, settings, shown)
            if self.future is not None and self.future.done():
                before = self.execution
                self._take_completed_search(motion, state, settings, shown)
                if self.execution is not before:
                    shown = self._schedule_elapsed(state.time_s, motion)
                    self._refresh_incumbent(motion, settings, shown)
            commit = self._commit_step(shown)
            if self.future is None and (not self.incumbent_ok or state.time_s-self.last_submit >= REPLAN_S):
                self._submit_search(motion, settings, state.time_s, commit)
            if self.execution is not None:
                return self.execution_guidance(motion, state.time_s, shown)
            if self.following:
                return self.follow_guidance(state, settings, "气动预测恢复，正在计算后续动作")
            return self.status("计算" if self.future else "无可用动作", current=True)
        except Cancelled:
            return self.status("计算", current=self.progress_current)
        except (ValueError, OverflowError) as exc:
            if self.progress_current:
                return self.follow_without_model(state, settings, str(exc))
            missing = not valid_number(state.pitch_deg) or not valid_number(state.roll_deg)
            return self.pause(str(exc), "缺少姿态" if missing else "模型范围",
                              current=self.progress_current)
