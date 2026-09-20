"""Keyboard maneuver reference: signed-load point mass with finite roll/load response.

The response constants are explicit user-adjustable assumptions, not recovered
Instructor physics. No input is sent to the game. See README for the model scope.
Coordinates are east, north, up; roll is positive right-wing-down.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
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
              "load_response_s": (.1, 3), "reaction_s": (0, 1.5), "hold_s": (.3, 2),
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


def flight_frame(state):
    fields = (state.tas_mps, state.altitude_m, state.heading_deg, state.pitch_deg,
              state.roll_deg, state.aoa_deg, state.aos_deg, state.vertical_speed_mps)
    if not state.valid or not all(valid_number(x) for x in fields) or state.tas_mps < 50:
        raise ValueError("需要姿态、迎角、侧滑和速度数据")
    if abs(state.aos_deg) > 10 or not -20 <= state.aoa_deg <= 30:
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
    if abs(vertical) >= .985 or math.hypot(*direction[:2]) < .05:
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
class TurnPlan:
    initial: Motion
    action: Action | None
    duration_s: float | None
    energy_change_m: float | None
    reached: bool
    expansions: int = 0


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
        return (floor <= s.altitude <= 20000 and s.speed >= settings.minimum_tas_mps
                and abs(s.velocity[2])/s.speed < .985
                and settings.min_load-.2 <= s.load <= settings.max_load+.2)
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
    beam = [(state, None)]
    best = None
    elapsed, expansions = delay, 0
    while elapsed < settings.horizon_s-1e-9:
        candidates, reached = [], []
        segment = min(settings.hold_s, settings.horizon_s-elapsed)
        for current, first in beam:
            for action in ACTIONS:
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
                command = first or action
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
                                                trial.energy-initial.energy, True, expansions))
                        feasible = False
                        break
                if feasible:
                    candidates.append((trial, command))
        if reached:
            return min(reached, key=lambda p: (p.duration_s, -p.energy_change_m))
        if not candidates:
            break
        candidates.sort(key=lambda x: (goal.remaining(x[0].velocity), -x[0].energy))
        best_state, best_action = candidates[0]
        if goal.remaining(best_state.velocity) < goal.remaining(initial.velocity)-.25:
            best = TurnPlan(initial, best_action, None, None, False, expansions)
        beam, seen = [], set()
        for candidate in candidates:
            if candidate[1] not in seen:
                beam.append(candidate)
                seen.add(candidate[1])
                if len(beam) >= beam_width:
                    break
        elapsed += segment
    return best or TurnPlan(initial, None, None, None, False, expansions)


class TurnSession:
    """Worker-owned session with a frozen target and cancellable background search."""
    def __init__(self):
        self.executor = None
        self.reset()

    def reset(self, *, require_restart=False, reason=""):
        if getattr(self, "cancel", None) is not None:
            self.cancel.set()
        if getattr(self, "future", None) is not None:
            self.future.cancel()
        self.future = self.cancel = self.goal = self.previous = self.plan = None
        self.model = self.signature = None
        self.last_submit = self.last_switch = -math.inf
        self.plan_time = -math.inf
        self.action = None
        self.rate = None
        self.require_restart = require_restart
        self.restart_reason = reason
        self.last_sample_time = None
        self.engine_throttle = None
        self.completed = False
        self.floor = None

    def invalidate(self, reason="机型或参考设置已改变，请重新开始转向"):
        self.reset(require_restart=self.require_restart or self.goal is not None, reason=reason)

    def pause(self, reason="", phase="等待数据", *, keep_previous=False):
        """Withdraw unsafe cues without discarding the original maneuver goal."""
        if self.cancel is not None:
            self.cancel.set()
        if self.future is not None:
            self.future.cancel()
        self.future = self.cancel = self.plan = self.action = None
        self.last_submit = -math.inf
        self.rate = None
        if not keep_previous:
            self.previous = None
        return KeyboardTurnGuidance(phase=phase, reason=reason)

    def close(self):
        self.reset()
        if self.executor:
            self.executor.shutdown(wait=False, cancel_futures=True)
            self.executor = None

    def update(self, state, fm, mass, afterburner, sweep, settings):
        if self.require_restart:
            return KeyboardTurnGuidance(phase="重新开始", reason=self.restart_reason)
        if not state.valid:
            return self.pause("等待有效飞行数据")
        if self.goal is not None and self.last_sample_time is not None and state.time_s-self.last_sample_time > 2.5:
            self.invalidate("飞行数据中断超过 2.5 秒，请重新开始转向")
            return KeyboardTurnGuidance(phase="重新开始", reason=self.restart_reason)
        previous_time = self.last_sample_time
        self.last_sample_time = state.time_s
        try:
            velocity, normal = flight_frame(state)
            if not valid_number(state.throttle_percent) or not 0 <= state.throttle_percent <= 110:
                return self.pause("需要有效油门读数")
            signature = (id(fm), afterburner, sweep, settings)
            if signature != self.signature:
                self.reset()
                self.signature = signature
                self.model = ManeuverModel(fm, mass, afterburner, sweep)
                self.last_sample_time = state.time_s
            elif abs(mass/self.model.mass-1) > .01:
                # A fuel-related mass update must not re-anchor the turn angle.
                if self.cancel:
                    self.cancel.set()
                self.future = self.plan = self.action = None
                self.model = ManeuverModel(fm, mass, afterburner, sweep)
                self.last_submit = -math.inf
            load = self.model.observed_load(state.altitude_m, state.tas_mps, state.aoa_deg)
            if self.engine_throttle is None:
                self.engine_throttle = state.throttle_percent
            elif previous_time is not None:
                elapsed = max(0., state.time_s-previous_time)
                self.engine_throttle = state.throttle_percent+(self.engine_throttle-state.throttle_percent)*math.exp(
                    -elapsed/settings.engine_response_s)
            direction = unit(velocity)
            if self.previous is None:
                self.previous = (state.time_s, direction, normal)
                return KeyboardTurnGuidance(phase="读取姿态")
            t0, v0, n0 = self.previous
            dt = state.time_s-t0
            self.previous = (state.time_s, direction, normal)
            if not .02 <= dt <= .6:
                return self.pause("采样间隔变化，正在重新读取姿态", "读取姿态", keep_previous=True)
            previous_normal = transport(n0, v0, direction)
            rate = math.atan2(dot(direction, cross(previous_normal, normal)), dot(previous_normal, normal))/dt
            if abs(rate) > math.radians(400):
                raise ValueError("姿态变化过快")
            self.rate = rate if self.rate is None else self.rate+(rate-self.rate)*(1-math.exp(-dt/.25))
            rate = self.rate
            motion = Motion(state.altitude_m, velocity, normal, rate, load,
                            state.throttle_percent, self.engine_throttle)
            if self.goal is None:
                self.goal = VelocityGoal(direction, settings.angle_deg)
                self.floor = max(0., state.altitude_m-settings.max_altitude_loss_m)
            turned = angle(self.goal.direction, direction)
            remaining = max(0., settings.angle_deg-turned)
            if motion.altitude < self.floor:
                return self.pause("已低于本次机动高度下限；恢复高度或重新开始", "高度不足")
            if motion.speed < settings.minimum_tas_mps:
                return self.pause("当前 TAS 低于所设最低速度", "速度不足")
            if not settings.min_load-.2 <= load <= settings.max_load+.2:
                return self.pause("由迎角估计的载荷超出所设机动限制", "载荷超限")
            if remaining <= .05:
                self.completed = True
            if self.completed:
                if self.cancel:
                    self.cancel.set()
                self.action = None
                return KeyboardTurnGuidance(True, "到达", "", turned, 0.)
            if self.future is not None and self.future.done():
                future, self.future = self.future, None
                candidate = future.result()
                # No old-aircraft/settings result can survive reset. Reject a
                # result if actual movement has left its starting neighborhood.
                if (angle(candidate.initial.velocity, velocity) <= 10
                        and angle(candidate.initial.normal, normal) <= 20
                        and abs(candidate.initial.speed-motion.speed) <= 25
                        and abs(candidate.initial.altitude-motion.altitude) <= 150
                        and abs(candidate.initial.load-motion.load) <= 2
                        and abs(candidate.initial.throttle_percent-motion.throttle_percent) <= 15
                        and abs(candidate.initial.engine_throttle_percent-motion.engine_throttle_percent) <= 15
                        and abs(candidate.initial.roll_rate-motion.roll_rate) <= math.radians(60)):
                    if (candidate.action == self.action or self.action is None
                            or state.time_s-self.last_switch >= settings.hold_s):
                        if candidate.action != self.action:
                            self.last_switch = state.time_s
                        self.plan, self.action = candidate, candidate.action
                        self.plan_time = state.time_s
                else:
                    self.plan = self.action = None
            if self.future is None and state.time_s-self.last_submit >= .8:
                if self.executor is None:
                    self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="wt-turn")
                self.cancel = Event()
                self.last_submit = state.time_s
                self.future = self.executor.submit(search_turn, self.model, motion, self.goal, settings, self.floor, self.cancel)
            if self.plan is not None and (angle(self.plan.initial.velocity, velocity) > 15
                    or angle(self.plan.initial.normal, normal) > 30
                    or abs(self.plan.initial.throttle_percent-motion.throttle_percent) > 20
                    or state.time_s-self.plan_time > 2.):
                self.plan = self.action = None
            if self.action is None:
                return KeyboardTurnGuidance(False, "计算" if self.future else "无可用动作", "", turned, remaining)
            return KeyboardTurnGuidance(True, "转向", self.action.label, turned, remaining,
                self.plan.duration_s, self.plan.energy_change_m, self.action.roll, self.action.pitch,
                throttle_command=self.action.throttle, throttle_percent=motion.throttle_percent,
                target_throttle_percent=(motion.throttle_percent if self.action.throttle == 0 else
                    max(0., motion.throttle_percent-settings.throttle_rate_percent_s*settings.hold_s)
                    if self.action.throttle < 0 else min(self.model.max_throttle,
                        motion.throttle_percent+settings.throttle_rate_percent_s*settings.hold_s)))
        except Cancelled:
            return KeyboardTurnGuidance(phase="计算")
        except (ValueError, OverflowError) as exc:
            return self.pause(str(exc))
