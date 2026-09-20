"""Keyboard maneuver reference: signed-load point mass with finite roll/load response.

The response constants are explicit user-adjustable assumptions, not recovered
Instructor physics. No input is sent to the game. See README for the model scope.
Coordinates are east, north, up; roll is positive right-wing-down.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from bisect import bisect_left
from dataclasses import dataclass, replace
from functools import lru_cache
import math
from threading import Event
import time

from .climb import valid_number
from .contracts import G, KeyboardTurnGuidance, KeyboardTurnSettings
from .fm import atmosphere


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
    if enforce_envelope and (abs(state.aos_deg) > 10 or not -20 <= state.aoa_deg <= 30):
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

    @property
    def label(self):
        parts = [x for x in (("左滚", "停止滚转", "右滚")[self.roll+1],
                            ("推杆", "松杆", "拉杆")[self.pitch+1])]
        return "＋".join(parts)


ACTIONS = tuple(Action(roll, pitch, throttle) for throttle in (0, -1, 1)
                for roll in (0, -1, 1) for pitch in (1, 0, -1))


class ManeuverModel:
    """Quasi-steady FM forces, signed lift demand and wind-axis roll response."""
    def __init__(self, model, mass, afterburner=True, sweep=0.):
        if not valid_number(mass) or mass <= 0:
            raise ValueError("需要参考总质量")
        if not hasattr(model, "components_at_sweep") or not hasattr(model, "engines"):
            raise ValueError("机型不支持转向参考")
        self.model, self.mass, self.afterburner, self.sweep = model, mass, afterburner, sweep
        self.max_throttle = 110. if afterburner and any(e.has_wep for e in model.engines) else 100.
        self._context = lru_cache(maxsize=128)(self._make_context)

    def _make_context(self, altitude, speed):
        rho, sound = atmosphere(altitude)
        if speed < 50 or speed/sound > 2.35:
            raise ValueError("超出模型范围")
        polars = [(p, p.properties.at_mach(speed/sound)) for p in self.model.components_at_sweep(self.sweep)]
        low = max(-12., *(p.critical_aoa_low-part.incidence_deg+.5 for part, p in polars if not part.vertical))
        high = min(25., *(p.critical_aoa_high-part.incidence_deg-.5 for part, p in polars if not part.vertical))
        if low >= high:
            raise ValueError("无共同失速前迎角区间")
        military = sum(e.thrust_n(altitude, speed, False) for e in self.model.engines)
        maximum = sum(e.thrust_n(altitude, speed, True) for e in self.model.engines)
        return polars, .5*rho*speed*speed, (military, maximum), low, high

    @staticmethod
    def _forces(polars, dynamic_pressure, alpha):
        lift = drag = 0.
        for part, polar in polars:
            effective = part.incidence_deg if part.vertical else alpha+part.incidence_deg
            cd, cl = polar.coefficients(effective)
            drag += dynamic_pressure*part.area_m2*cd
            if not part.vertical:
                lift += dynamic_pressure*part.area_m2*cl
        return lift, drag

    def observed_load(self, altitude, speed, alpha):
        polars, q, _, low, high = self._context(altitude, speed)
        if not low <= alpha <= high:
            raise ValueError("迎角超出共同失速前范围")
        return self._forces(polars, q, alpha)[0]/(self.mass*G)

    def forces(self, altitude, speed, load, throttle_percent=None):
        polars, q, thrusts, low, high = self._context(altitude, speed)
        throttle = self.max_throttle if throttle_percent is None else throttle_percent
        military, maximum = thrusts
        # Reference interpolation only: the raw FM supplies full-thrust tables,
        # not a verified partial-throttle/spool law. Leave static SEP untouched.
        thrust = (military*max(0., throttle)/100 if throttle <= 100 else
                  military+(maximum-military)*min(1., (throttle-100)/10))
        target = load*self.mass*G
        l0, l1 = self._forces(polars, q, low)[0], self._forces(polars, q, high)[0]
        if l0 >= l1:
            raise ValueError("升力区间不可用")
        target = min(l1, max(l0, target))
        for _ in range(12):
            mid = (low+high)/2
            if self._forces(polars, q, mid)[0] < target:
                low = mid
            else:
                high = mid
        alpha = (low+high)/2
        lift, drag = self._forces(polars, q, alpha)
        return thrust, drag, lift, math.radians(alpha)

    def step(self, state: Motion, action: Action | None, dt, settings):
        # None is the human response interval: continue the observed roll/load.
        p_target = state.roll_rate if action is None else math.radians(settings.roll_rate_deg_s)*action.roll
        n_target = state.load if action is None else (
            settings.max_load if action.pitch > 0 else settings.min_load if action.pitch < 0 else state.normal[2])
        p = p_target+(state.roll_rate-p_target)*math.exp(-dt/settings.roll_response_s)
        n = n_target+(state.load-n_target)*math.exp(-dt/settings.load_response_s)
        throttle = state.throttle_percent
        if action is not None and action.throttle:
            demanded = throttle+action.throttle*settings.throttle_rate_percent_s*dt
            throttle = max(0., demanded) if action.throttle < 0 else min(self.max_throttle, demanded)
        engine = throttle+(state.engine_throttle_percent-throttle)*math.exp(-dt/settings.engine_response_s)
        direction = unit(state.velocity)
        normal = rotate(state.normal, direction, (state.roll_rate+p)*dt/4)
        thrust, drag, lift, alpha = self.forces(state.altitude, state.speed, (state.load+n)/2,
                                               (state.engine_throttle_percent+engine)/2)
        acceleration = add(add(scale(direction, (thrust*math.cos(alpha)-drag)/self.mass),
                               scale(normal, (lift+thrust*math.sin(alpha))/self.mass)), (0., 0., -G))
        velocity = add(state.velocity, scale(acceleration, dt))
        normal = rotate(state.normal, direction, (state.roll_rate+p)*dt/2)
        normal = transport(normal, direction, unit(velocity))
        # Prevent unavailable lift accumulating as hidden integrator state.
        achieved = lift/(self.mass*G)
        requested = (state.load+n)/2
        if abs(achieved-requested) > .01:
            n = min(n, achieved) if achieved < requested else max(n, achieved)
        return Motion(state.altitude+(state.velocity[2]+velocity[2])*dt/2, velocity, normal, p, n,
                      throttle, engine)


@dataclass(frozen=True)
class TurnStep:
    action: Action
    duration_s: float


def append_step(steps, action, duration):
    if steps and steps[-1].action == action:
        return (*steps[:-1], TurnStep(action, steps[-1].duration_s+duration))
    return (*steps, TurnStep(action, duration))


def allowed_motion(s, settings, floor):
    return (floor <= s.altitude <= 20000 and s.speed >= settings.minimum_tas_mps
            and abs(s.velocity[2])/s.speed < .985
            and settings.min_load-.2 <= s.load <= settings.max_load+.2)


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


def search_turn(model, initial, goal, settings, floor, cancel=None, *, budget_s=.8, beam_width=27):
    """Bounded beam search over held actions; no global-optimality guarantee.

    Keeps different initial actions alive so early rolling/unloading is not
    immediately discarded in favor of instant angular progress.
    """
    validate_settings(settings)
    deadline = time.monotonic()+budget_s
    def allowed(s):
        return allowed_motion(s, settings, floor)
    if not allowed(initial):
        raise ValueError("当前状态超出所设机动限制")
    if goal.remaining(initial.velocity) == 0:
        return TurnPlan(initial, None, 0., 0., True)
    state, delay = initial, 0.
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
    beam = [(state, ())]
    best = None
    elapsed, expansions = delay, 0
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
                        return min(reached, key=lambda p: (p.duration_s, -p.energy_change_m))
                    return best or TurnPlan(initial, None, None, None, False, expansions)
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
            return min(reached, key=lambda p: (p.duration_s, -p.energy_change_m))
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
    return best or TurnPlan(initial, None, None, None, False, expansions)


def nearby(expected, actual, *, coarse=False):
    factor = 2.5 if coarse else 1.
    return (angle(expected.velocity, actual.velocity) <= 12*factor
            and angle(expected.normal, actual.normal) <= 25*factor
            and abs(expected.speed-actual.speed) <= 25*factor
            and abs(expected.altitude-actual.altitude) <= 150*factor
            and abs(expected.load-actual.load) <= 2.5*factor
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
                      blend(left.engine_throttle_percent, right.engine_throttle_percent))

    def index(self, elapsed):
        return min(bisect_left(self.ends, elapsed), len(self.steps)-1)


def prepare_execution(model, motion, plan, goal, settings, floor):
    """Rebase the selected sequence on the current measured state before display."""
    requested = plan.steps or (TurnStep(plan.action, settings.hold_s),)
    times, states, steps, ends = [0.], [motion], [], []
    elapsed = 0.
    for action, duration in [(None, settings.reaction_s), *((x.action, x.duration_s) for x in requested)]:
        dt_left = duration
        while dt_left > 1e-9:
            dt = min(.15, dt_left)
            motion = model.step(motion, action, dt, settings)
            if not allowed_motion(motion, settings, floor):
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
    """Commit to a readable sequence; validate against its moving trajectory."""
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
        if not keep_previous:
            self.previous = None
        return self.status(phase, reason, current=current)

    def close(self):
        self.reset()
        if self.executor:
            self.executor.shutdown(wait=False, cancel_futures=True)
            self.executor = None

    def execution_guidance(self, motion, now):
        execution = self.execution
        elapsed = max(0., now-self.plan_time)
        i = execution.index(elapsed)
        self.execution_index = i
        action = self.action = execution.steps[i].action
        next_action = (execution.steps[i+1].action.label+"／"+
            ("收油", "保持油门", "加油")[execution.steps[i+1].action.throttle+1]
            if i+1 < len(execution.steps) else "达到目标后松键" if execution.reached else "继续规划")
        endpoint = execution.reference(execution.ends[i])
        return KeyboardTurnGuidance(True, "转向", action.label, self.turned, self.remaining,
            max(0., execution.ends[-1]-elapsed) if execution.reached else None,
            execution.states[-1].energy-motion.energy, action.roll, action.pitch,
            throttle_command=action.throttle, throttle_percent=motion.throttle_percent,
            target_throttle_percent=endpoint.throttle_percent, next_action=next_action,
            step_index=i+1, step_count=len(execution.steps),
            step_remaining_s=max(0., execution.ends[i]-elapsed))

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
                self.turned = angle(self.goal.direction, direction)
                self.remaining = max(0., settings.angle_deg-self.turned)
                self.progress_current = True
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
                self.turned, self.remaining = 0., settings.angle_deg
                self.progress_current = True
            if self.remaining <= .05:
                self.completed = True
            if self.completed:
                if self.cancel:
                    self.cancel.set()
                self.action = self.execution = None
                return KeyboardTurnGuidance(True, "到达", "", self.turned, 0.)
            if state.altitude_m < self.floor:
                return self.pause("已低于本次机动高度下限；恢复高度或重新开始", "高度不足", current=True)
            if state.tas_mps < settings.minimum_tas_mps:
                return self.pause("当前 TAS 低于所设最低速度", "速度不足", current=True)
            flight_frame(state)  # Apply the prediction envelope after updating progress.
            load = self.model.observed_load(state.altitude_m, state.tas_mps, state.aoa_deg)
            if not settings.min_load-.2 <= load <= settings.max_load+.2:
                return self.pause("由迎角估计的载荷超出所设机动限制", "载荷超限", current=True)
            if self.engine_throttle is None:
                self.engine_throttle = state.throttle_percent
            elif previous_time is not None:
                elapsed = max(0., state.time_s-previous_time)
                self.engine_throttle = state.throttle_percent+(self.engine_throttle-state.throttle_percent)*math.exp(
                    -elapsed/settings.engine_response_s)
            motion = Motion(state.altitude_m, velocity, normal, self.rate, load,
                            state.throttle_percent, self.engine_throttle)
            if self.execution is not None:
                elapsed = state.time_s-self.plan_time
                expected = self.execution.reference(elapsed)
                index = self.execution.index(elapsed)
                if (not nearby(expected, motion, coarse=True)
                        or index != self.execution_index and not nearby(expected, motion)):
                    self.pause("执行轨迹偏离预测，正在调整后续动作", "调整动作", keep_previous=True, current=True)
                elif elapsed >= self.execution.ends[-1]:
                    self.pause("本段完成，继续规划", "更新计划", keep_previous=True, current=True)
                else:
                    return self.execution_guidance(motion, state.time_s)
            if self.future is not None and self.future.done():
                future, self.future = self.future, None
                candidate = future.result()
                # During computation no new cue was visible. Project the observed
                # motion through that latency, then rebase the selected sequence.
                expected, elapsed = candidate.initial, 0.
                latency = max(0., state.time_s-self.last_submit)
                while elapsed < min(latency, 2.5)-1e-9:
                    step = min(.15, latency-elapsed)
                    expected = self.model.step(expected, None, step, settings)
                    elapsed += step
                if latency <= 2.5 and nearby(expected, motion) and candidate.action is not None:
                    try:
                        self.execution = prepare_execution(self.model, motion, candidate, self.goal, settings, self.floor)
                    except ValueError as exc:
                        return self.pause(str(exc), "调整动作", keep_previous=True, current=True)
                    self.plan, self.plan_time = candidate, state.time_s
                    return self.execution_guidance(motion, state.time_s)
            if self.future is None and state.time_s-self.last_submit >= .8:
                if self.executor is None:
                    self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="wt-turn")
                self.cancel = Event()
                self.last_submit = state.time_s
                self.future = self.executor.submit(search_turn, self.model, motion, self.goal, settings, self.floor, self.cancel)
            return self.status("计算" if self.future else "无可用动作", current=True)
        except Cancelled:
            return self.status("计算", current=self.progress_current)
        except (ValueError, OverflowError) as exc:
            missing = not valid_number(state.pitch_deg) or not valid_number(state.roll_deg)
            return self.pause(str(exc), "缺少姿态" if missing else "模型范围",
                              current=self.progress_current)
