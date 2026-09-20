"""Static, untrimmed performance sampling; no inferred dynamic maneuver solver."""
from __future__ import annotations

from dataclasses import replace
from math import isfinite
from typing import Iterable

from .contracts import (FlightState, PerformanceCondition, PerformanceModel,
                        PerformancePoint, SEPAdvice, TurnAdvice, TurnRequest)

STATIC_NOTE = "Sampled static local best only; not a global minimum-time climb or SEP-gradient trajectory."
LOAD_NOTE = "Speed samples solve the same specified load demand (aoa_deg=None); current point retains supplied AoA."


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)


def _condition_error(c: PerformanceCondition) -> str:
    for name in ('altitude_m', 'tas_mps', 'mass_kg', 'throttle', 'load_factor',
                 'flight_path_deg', 'flap_fraction', 'gear_fraction', 'airbrake_fraction', 'sweep_fraction'):
        if not _finite(getattr(c, name)):
            return f"{name} must be finite."
    if c.tas_mps <= 0 or c.mass_kg <= 0:
        return "TAS and mass must be positive."
    if c.load_factor < 0:
        return "Negative load demand is unsupported by this static planner."
    if abs(c.flight_path_deg) > 90:
        return "Flight path angle must lie in [-90, 90] degrees."
    if c.aoa_deg is not None and not _finite(c.aoa_deg):
        return "AoA must be finite when supplied."
    if any(not 0 <= getattr(c, n) <= 1 for n in
           ('throttle', 'flap_fraction', 'gear_fraction', 'airbrake_fraction', 'sweep_fraction')):
        return "Throttle and configuration fractions must lie in [0, 1]."
    return ""


def _evaluate(model: PerformanceModel, condition: PerformanceCondition) -> PerformancePoint:
    error = _condition_error(condition)
    if error:
        return PerformancePoint(condition, False, reason=error)
    try:
        point = model.evaluate(condition)
    except Exception as exc:
        return PerformancePoint(condition, False, reason=f"Model evaluation failed: {type(exc).__name__}: {exc}")
    if not isinstance(point, PerformancePoint):
        return PerformancePoint(condition, False, reason="Model returned an unsupported result type.")
    if point.condition != condition:
        return replace(point, condition=condition, valid=False,
                       reason="Model result condition does not match the requested sample.")
    if point.valid and not _finite(point.sep_mps):
        return replace(point, valid=False, reason="Model did not supply a finite SEP.")
    return point


def scan_sep(model: PerformanceModel, condition: PerformanceCondition,
             speeds_mps: Iterable[float] | None = None) -> SEPAdvice:
    """Compare fixed-load samples at unchanged height, mass and configuration.

    The current point can describe an instantaneous supplied AoA. The scan always
    clears that AoA so the model must solve the requested load at each speed.
    Default samples cover 70–130% of current TAS, not a certified flight envelope.
    Unsupported samples remain visible; there is no extrapolation or optimizer.
    """
    notes = [*model.info.limitations, STATIC_NOTE, LOAD_NOTE]
    error = _condition_error(condition)
    if error:
        return SEPAdvice(False, reason=error, notes=tuple(notes))
    current = _evaluate(model, condition)
    notes.extend(current.notes)
    if speeds_mps is None:
        speeds = [condition.tas_mps * (0.7 + 0.05 * i) for i in range(13)]
        notes.append("Default search: 13 TAS samples from 70% to 130% of current TAS; model validity determines usable samples.")
    else:
        speeds = list(speeds_mps)
    points = tuple(_evaluate(model, replace(condition, tas_mps=v, aoa_deg=None)) for v in speeds)
    for point in points:
        notes.extend(point.notes)
    valid = [p for p in points if p.valid]
    if not valid:
        return SEPAdvice(False, current=current, sampled_points=points,
                         reason="No valid finite SEP samples are available.", notes=tuple(dict.fromkeys(notes)))
    best = max(valid, key=lambda p: p.sep_mps)
    if best.sep_mps < 0:
        notes.append("All valid sampled SEP values are negative; best means least energy loss, not energy gain.")
    if len(valid) < len(points):
        notes.append("Some samples are unavailable; the best is restricted to valid samples and may miss a better state.")
    if best.condition.tas_mps in (min(p.condition.tas_mps for p in valid), max(p.condition.tas_mps for p in valid)):
        notes.append("Best lies at a boundary of the valid sampled speed range; an interior optimum is not established.")
    return SEPAdvice(True, current, best, points, notes=tuple(dict.fromkeys(notes)))


def sample_sep_grid(model: PerformanceModel, base_condition: PerformanceCondition,
                    altitudes_m: Iterable[float], speeds_mps: Iterable[float]) -> Iterable[PerformancePoint]:
    """Yield fixed-load samples, preserving every other base-condition field."""
    speeds = tuple(speeds_mps)
    for altitude in altitudes_m:
        for speed in speeds:
            yield _evaluate(model, replace(base_condition, altitude_m=altitude,
                                           tas_mps=speed, aoa_deg=None))


def evaluate_turn(request: TurnRequest, state: FlightState | None,
                  dynamic_model=None) -> TurnAdvice:
    """Validate a finite 3D turn request without substituting steady-turn rates.

    A dynamic provider is intentionally not accepted until its state, constraint,
    and solution-validation contract is established. Dataclass defaults describe
    a request, not a user-approved optimum.
    """
    reason = ""
    if not _finite(request.angle_deg) or request.angle_deg not in (30, 45, 90, 120):
        reason = "Supported requested turn angles are 30, 45, 90 and 120 degrees."
    elif request.objective not in ('minimum_time', 'minimum_energy_loss'):
        reason = "Objective must be minimum_time or minimum_energy_loss."
    elif request.angle_basis not in ('velocity', 'nose'):
        reason = "Angle basis must explicitly be velocity or nose."
    elif request.endpoint not in ('first_crossing', 'stable_exit'):
        reason = "Endpoint must be first_crossing or stable_exit."
    else:
        for name in ('time_limit_s', 'max_altitude_loss_m', 'minimum_exit_tas_mps'):
            value = getattr(request, name)
            if value is not None and (not _finite(value) or value < 0 or (name == 'time_limit_s' and value == 0)):
                reason = f"{name} must be finite and nonnegative (time limit strictly positive)."
                break
    if not reason and request.objective == 'minimum_energy_loss' and request.time_limit_s is None:
        reason = "Minimum energy loss requires an explicit positive time bound."
    if not reason and (state is None or not state.valid):
        reason = "A valid current flight state is required; no dynamic prediction is available."
    if not reason:
        reason = "No verified dynamic maneuver model is integrated; finite 3D turn time and energy cannot be predicted."
    return TurnAdvice(False, request, reason=reason, notes=(
        "Velocity-direction change and nose-direction change are distinct objectives.",
        "First crossing and stable exit are distinct endpoints; request defaults are not an approved optimum.",
        "Steady level-turn formulas do not model entry, roll, transient load or exit.",
    ))
