"""Shared interfaces. All physical quantities use SI unless named otherwise."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

G = 9.80665


@dataclass(frozen=True)
class FlightState:
    time_s: float
    valid: bool
    altitude_m: float | None = None
    tas_mps: float | None = None
    ias_mps: float | None = None
    mach: float | None = None
    vertical_speed_mps: float | None = None
    aoa_deg: float | None = None
    aos_deg: float | None = None
    pitch_deg: float | None = None
    roll_deg: float | None = None
    heading_deg: float | None = None
    load_factor: float | None = None
    throttle_percent: float | None = None
    fuel_kg: float | None = None
    mass_kg: float | None = None
    thrust_n: float | None = None
    aircraft_id: str | None = None
    source: str = "live"
    notes: tuple[str, ...] = ()
    raw_state: Mapping[str, Any] = field(default_factory=dict, repr=False)
    raw_indicators: Mapping[str, Any] = field(default_factory=dict, repr=False)
    normal_load_g: float | None = None  # Telemetry body Ny; independent of model L/W.


@dataclass(frozen=True)
class EnergyMetrics:
    time_s: float
    energy_height_m: float | None = None
    sep_mps: float | None = None
    kinetic_sep_mps: float | None = None
    acceleration_mps2: float | None = None
    climb_mps: float | None = None
    ready: bool = False
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelInfo:
    aircraft_id: str
    name: str
    source_url: str = ""
    revision: str = ""
    validated_in_game: bool = False
    limitations: tuple[str, ...] = ()


@dataclass(frozen=True)
class PerformanceCondition:
    altitude_m: float
    tas_mps: float
    mass_kg: float
    throttle: float = 1.0
    afterburner: bool = True
    load_factor: float = 1.0  # Aerodynamic L/W, not body Ny or a gravity multiplier.
    aoa_deg: float | None = None
    flight_path_deg: float = 0.0
    flap_fraction: float = 0.0
    gear_fraction: float = 0.0
    airbrake_fraction: float = 0.0


@dataclass(frozen=True)
class PerformancePoint:
    condition: PerformanceCondition
    valid: bool
    thrust_n: float | None = None
    drag_n: float | None = None
    lift_n: float | None = None
    aoa_deg: float | None = None
    sep_mps: float | None = None
    reason: str = ""
    notes: tuple[str, ...] = ()


class PerformanceModel(Protocol):
    info: ModelInfo

    def evaluate(self, condition: PerformanceCondition) -> PerformancePoint: ...


@dataclass(frozen=True)
class SEPAdvice:
    available: bool
    current: PerformancePoint | None = None
    best: PerformancePoint | None = None
    sampled_points: tuple[PerformancePoint, ...] = ()
    reason: str = ""
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class TurnRequest:
    angle_deg: float
    objective: str = "minimum_time"
    angle_basis: str = "velocity"
    endpoint: str = "first_crossing"
    time_limit_s: float | None = None
    max_altitude_loss_m: float | None = None
    minimum_exit_tas_mps: float | None = None


@dataclass(frozen=True)
class TurnAdvice:
    available: bool
    request: TurnRequest
    duration_s: float | None = None
    energy_change_m: float | None = None
    reason: str = ""
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ClimbRequest:
    target_altitude_m: float = 8000.0
    minimum_tas_mps: float | None = None


@dataclass(frozen=True)
class ClimbGuidance:
    available: bool = False
    phase: str = "等待"
    target_tas_mps: float | None = None
    target_path_deg: float | None = None
    path_error_deg: float | None = None
    remaining_height_m: float | None = None


@dataclass(frozen=True)
class OverlaySnapshot:
    mode: str
    status: str
    state: FlightState | None = None
    energy: EnergyMetrics | None = None
    advice: SEPAdvice | None = None
    model_name: str = "未加载 FM"
    notes: tuple[str, ...] = ()
    mass_override_kg: float | None = None
    afterburner: bool = True
    climb_enabled: bool = False
    climb_request: ClimbRequest = field(default_factory=ClimbRequest)
    climb: ClimbGuidance | None = None
