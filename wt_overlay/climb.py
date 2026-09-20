"""Constant-energy SEP search and a manual longitudinal flight director.

The energy-state approximation, capture law and limits are documented in README.
This module neither sends flight controls nor changes the static FM equations.
"""

from bisect import bisect_right
from dataclasses import dataclass, replace
import math
from threading import Event

from .contracts import (G, ClimbGuidance, ClimbRequest, EnergyMetrics, FlightState,
                        PerformanceCondition, PerformanceModel, PerformancePoint)


MIN_SPEED = 80.0
MAX_SPEED = 650.0
MAX_PATH_DEG = 45.0
PATH_RATE_DEG_S = 1.0
PATH_RESPONSE_S = 2.0
PATH_DISPLAY_S = 0.5
CUE_DEADBAND_DEG = 1.5
CUE_RANGE_DEG = 12.0


def valid_number(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def target_indicated_speed(state: FlightState, target_tas_mps: float) -> float | None:
    """Local IAS estimate from the simultaneous telemetry pair; used only for display."""
    if not state.valid or not all(valid_number(v) and v > 0
                                  for v in (state.ias_mps, state.tas_mps, target_tas_mps)):
        return None
    value = target_tas_mps * (state.ias_mps / state.tas_mps)
    return value if valid_number(value) else None


def validate_request(request: ClimbRequest):
    if not valid_number(request.target_altitude_m) or not 100 <= request.target_altitude_m <= 20000:
        raise ValueError("目标高度须为 100–20000 m")
    if request.minimum_tas_mps is not None and (
            not valid_number(request.minimum_tas_mps) or not MIN_SPEED <= request.minimum_tas_mps <= MAX_SPEED):
        raise ValueError("到达真空速须为 288–2340 km/h，或留空")


class PlanningCancelled(Exception):
    pass


class PlanningUnavailable(ValueError):
    pass


@dataclass(frozen=True)
class EnergyPoint:
    energy_m: float
    altitude_m: float
    tas_mps: float
    sep_mps: float


@dataclass(frozen=True)
class ClimbPlan:
    request: ClimbRequest
    condition: PerformanceCondition
    points: tuple[EnergyPoint, ...]
    end_tas_mps: float
    end_energy_m: float
    capture_span_m: float

    def _speed(self, energy_m: float) -> float:
        if energy_m >= self.end_energy_m:
            speed = math.sqrt(2 * G * (energy_m - self.request.target_altitude_m))
            if speed > MAX_SPEED:
                raise PlanningUnavailable("能量超出规划范围")
            return speed
        energies = [point.energy_m for point in self.points]
        if not energies[0] <= energy_m <= energies[-1]:
            raise PlanningUnavailable("能量超出规划范围")
        i = min(max(bisect_right(energies, energy_m) - 1, 0), len(energies) - 2)
        a, b = self.points[i:i + 2]
        fraction = (energy_m - a.energy_m) / (b.energy_m - a.energy_m)
        speed = math.sqrt(a.tas_mps ** 2 + fraction * (b.tas_mps ** 2 - a.tas_mps ** 2))
        start = self.end_energy_m - self.capture_span_m
        if energy_m > start:
            u = min(1.0, (energy_m - start) / self.capture_span_m)
            blend = u * u * (3 - 2 * u)
            terminal = math.sqrt(max(0.0, 2 * G * (energy_m - self.request.target_altitude_m)))
            speed = (1 - blend) * speed + blend * terminal
        return speed

    def reference(self, energy_m: float) -> tuple[float, float]:
        """Target TAS and d(TAS)/d(energy height), including terminal capture."""
        speed = self._speed(energy_m)
        if energy_m >= self.end_energy_m:
            return speed, G / speed
        low = max(self.points[0].energy_m, energy_m - 5.0)
        high = min(self.points[-1].energy_m, energy_m + 5.0)
        slope = (self._speed(high) - self._speed(low)) / (high - low)
        return speed, slope


def _point(model, condition, cancel):
    if cancel is not None and cancel.is_set():
        raise PlanningCancelled
    point = model.evaluate(condition)
    if (not isinstance(point, PerformancePoint) or not point.valid
            or not valid_number(point.sep_mps) or point.condition != condition):
        return None
    return point


def _maximum(evaluate, lower: float, upper: float) -> PerformancePoint | None:
    """Global coarse sampling followed by local refinement, including endpoints."""
    if upper < lower:
        return None
    best = None
    for count in (49, 9, 9):
        step = (upper - lower) / (count - 1)
        for i in range(count):
            point = evaluate(lower + i * step)
            if point is not None and (best is None or point.sep_mps > best.sep_mps):
                best = point
        if best is None:
            return None
        center = best.condition.tas_mps
        lower, upper = max(lower, center - step), min(upper, center + step)
    return best


def maximum_at_energy(model: PerformanceModel, base: PerformanceCondition, energy_m: float,
                      ceiling_m: float, cancel: Event | None = None) -> EnergyPoint | None:
    lower = max(MIN_SPEED, math.sqrt(max(0.0, 2 * G * (energy_m - ceiling_m))))
    upper = min(MAX_SPEED, math.sqrt(max(0.0, 2 * G * energy_m)))

    def evaluate(speed):
        altitude = energy_m - speed * speed / (2 * G)
        if altitude < -1e-8 or altitude > ceiling_m + 1e-8:
            return None
        condition = replace(base, altitude_m=min(ceiling_m, max(0.0, altitude)),
                            tas_mps=speed, aoa_deg=None, load_factor=1.0)
        return _point(model, condition, cancel)

    best = _maximum(evaluate, lower, upper)
    if best is None or best.sep_mps <= 0:
        return None
    return EnergyPoint(energy_m, best.condition.altitude_m, best.condition.tas_mps, best.sep_mps)


def build_climb_plan(model: PerformanceModel, base: PerformanceCondition,
                     request: ClimbRequest, cancel: Event | None = None) -> ClimbPlan:
    validate_request(request)
    if not valid_number(base.mass_kg) or base.mass_kg <= 0:
        raise PlanningUnavailable("需要总质量")
    current = _point(model, replace(base, aoa_deg=None, load_factor=1.0), cancel)
    if current is None:
        raise PlanningUnavailable("当前状态超出模型范围")
    terminal_base = replace(base, altitude_m=request.target_altitude_m, aoa_deg=None, load_factor=1.0)
    if request.minimum_tas_mps is None:
        terminal = _maximum(lambda v: _point(model, replace(terminal_base, tas_mps=v), cancel), MIN_SPEED, MAX_SPEED)
    else:
        terminal = _point(model, replace(terminal_base, tas_mps=request.minimum_tas_mps), cancel)
    if terminal is None or terminal.sep_mps <= 0:
        raise PlanningUnavailable("目标状态不可达")
    end_speed = terminal.condition.tas_mps
    end_energy = request.target_altitude_m + end_speed * end_speed / (2 * G)
    initial_energy = base.altitude_m + base.tas_mps * base.tas_mps / (2 * G)
    low = max(MIN_SPEED * MIN_SPEED / (2 * G), min(initial_energy, end_energy - 250))
    high = end_energy
    count = max(2, math.ceil((high - low) / 250) + 1)
    points = []
    for i in range(count):
        point = maximum_at_energy(model, base, low + (high - low) * i / (count - 1),
                                  request.target_altitude_m, cancel)
        if point is None:
            raise PlanningUnavailable("规划路径不连续")
        points.append(point)
    # Leave a positive kinetic-energy interval for smooth terminal blending.
    capture_span = min(3000.0, 0.6 * end_speed * end_speed / (2 * G))
    return ClimbPlan(request, base, tuple(points), end_speed, end_energy, capture_span)


class ClimbDirector:
    def __init__(self):
        self.reset()

    def reset(self):
        self._time = None
        self._command = None
        self._filtered_power = None
        self._filtered_path = None
        self._complete = False

    def update(self, plan: ClimbPlan, state: FlightState, energy: EnergyMetrics,
               model_sep_mps: float | None) -> ClimbGuidance:
        if (not state.valid or not all(valid_number(x) for x in (state.time_s, state.altitude_m, state.tas_mps))
                or state.tas_mps < MIN_SPEED):
            self.reset()
            return ClimbGuidance(phase="等待数据")
        if state.roll_deg is not None and abs(state.roll_deg) > 30:
            self.reset()
            return ClimbGuidance(phase="改平")
        vertical = state.vertical_speed_mps
        if vertical is None and energy.ready:
            vertical = energy.climb_mps
        if not valid_number(vertical) or abs(vertical) > state.tas_mps:
            self.reset()
            return ClimbGuidance(phase="等待数据")
        actual_path = math.degrees(math.asin(vertical / state.tas_mps))
        remaining = plan.request.target_altitude_m - state.altitude_m
        if abs(remaining) <= 25 and state.tas_mps >= plan.end_tas_mps and abs(actual_path) <= 1.5:
            self._complete = True
        if self._complete:
            return ClimbGuidance(True, "到达", plan.end_tas_mps, remaining_height_m=max(0.0, remaining),
                                 actual_path_deg=actual_path,
                                 target_ias_mps=target_indicated_speed(state, plan.end_tas_mps))
        energy_height = state.altitude_m + state.tas_mps ** 2 / (2 * G)
        try:
            target_speed, slope = plan.reference(energy_height)
        except PlanningUnavailable:
            self.reset()
            return ClimbGuidance(phase="重新规划")
        power = energy.sep_mps if energy.ready and valid_number(energy.sep_mps) else model_sep_mps
        if not valid_number(power):
            self.reset()
            return ClimbGuidance(phase="等待数据")
        dt = state.time_s - self._time if self._time is not None else 0.0
        if not 0 < dt <= 2:
            self._command = max(-5.0, min(MAX_PATH_DEG, actual_path))
            self._filtered_power = power
            self._filtered_path = actual_path
            dt = 0.0
        else:
            blend = 1 - math.exp(-dt / 1.0)
            self._filtered_power += blend * (power - self._filtered_power)
            self._filtered_path += (1 - math.exp(-dt / PATH_DISPLAY_S)) * (actual_path - self._filtered_path)
        power = self._filtered_power
        acceleration = slope * power + 0.08 * (target_speed - state.tas_mps)
        climb = power - state.tas_mps / G * acceleration
        # Capture uses altitude feedback before the target so flattening is gradual.
        climb = min(climb, max(0.0, remaining) / 12.0)
        climb = max(0.0, climb)
        if remaining < 0:
            climb = max(-state.tas_mps * math.sin(math.radians(5)), remaining / 12.0)
        path = math.degrees(math.asin(max(-1.0, min(1.0, climb / state.tas_mps))))
        path = max(-5.0, min(MAX_PATH_DEG, path))
        change = PATH_RATE_DEG_S * dt
        step = (1 - math.exp(-dt / PATH_RESPONSE_S)) * (path - self._command)
        self._command += max(-change, min(change, step))
        self._time = state.time_s
        phase = ("收平" if abs(remaining) < 250 or energy_height >= plan.end_energy_m else
                 "加速" if path < 1.0 else "爬升")
        display_speed = target_speed
        if phase == "加速":
            # Show the upcoming climb-entry speed while following the local schedule.
            entry = next((p for p in plan.points if p.energy_m > energy_height
                          and p.altitude_m > state.altitude_m + 50), None)
            if entry is not None:
                display_speed = max(target_speed, plan.reference(entry.energy_m)[0])
        return ClimbGuidance(True, phase, display_speed, self._command,
                             self._command - self._filtered_path, remaining, self._filtered_path,
                             target_indicated_speed(state, display_speed))
