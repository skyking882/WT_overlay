"""Bounded rolling linear estimates of measured energy-height and its rates."""
from __future__ import annotations

from collections import deque
import math

from .contracts import EnergyMetrics, FlightState, G


class EnergyEstimator:
    """Estimate rates over a common history; reset after any break in continuity.

Rates are least-squares slopes over at most window_s and 512 samples. At least
0.2 seconds (or half a smaller configured window) and three samples are needed.
They are window averages, not instantaneous end-of-window derivatives.
"""
    def __init__(self, window_s: float = 1.2, max_gap_s: float = 2.0):
        if not all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0
                   for v in (window_s, max_gap_s)):
            raise ValueError('Energy window and maximum gap must be finite and positive')
        self.window_s, self.max_gap_s = window_s, max_gap_s
        self._samples: deque[tuple[float, float, float, float]] = deque(maxlen=512)
        self._identity: tuple[str, str | None] | None = None

    def reset(self) -> None:
        self._samples.clear()
        self._identity = None

    def update(self, state: FlightState) -> EnergyMetrics:
        values = (state.time_s, state.altitude_m, state.tas_mps)
        if (not state.valid or any(isinstance(v, bool) or not isinstance(v, (int, float))
                                  or not math.isfinite(v) for v in values)
                or state.tas_mps < 0):
            self.reset()
            timestamp = state.time_s if isinstance(state.time_s, (float, int)) and math.isfinite(state.time_s) else 0.0
            return EnergyMetrics(timestamp, notes=('Waiting for valid flight telemetry',))
        t, h, v = values
        kinetic = v * v / (2 * G)
        energy = h + kinetic
        if not math.isfinite(energy):
            self.reset()
            return EnergyMetrics(t, notes=('Non-finite energy',))
        identity = (state.source, state.aircraft_id)
        reset_reason = None
        if self._samples:
            dt = t - self._samples[-1][0]
            if dt <= 0:
                reset_reason = 'Telemetry time restarted'
            elif dt > self.max_gap_s:
                reset_reason = 'Telemetry gap'
            elif identity != self._identity:
                reset_reason = 'Aircraft or telemetry source changed'
        if reset_reason:
            self.reset()
        self._identity = identity
        self._samples.append((t, h, v, kinetic))
        while self._samples and self._samples[0][0] < t - self.window_s:
            self._samples.popleft()
        span = t - self._samples[0][0]
        if len(self._samples) < 3 or span < min(0.2, self.window_s / 2):
            notes = ((reset_reason,) if reset_reason else ()) + ('Energy rate warming up',)
            return EnergyMetrics(t, energy_height_m=energy, notes=notes)
        xs = [row[0] - t for row in self._samples]
        mean = math.fsum(xs) / len(xs)
        centered = [x - mean for x in xs]
        denom = math.fsum(x * x for x in centered)
        def slope(column: int) -> float:
            # Subtract a nearby value to avoid cancellation for large absolute altitude.
            base = self._samples[-1][column]
            return math.fsum(x * (row[column] - base)
                             for x, row in zip(centered, self._samples)) / denom
        try:
            climb, acceleration, kinetic_sep = slope(1), slope(2), slope(3)
            sep = climb + kinetic_sep
        except (OverflowError, ValueError, ZeroDivisionError):
            self.reset()
            return EnergyMetrics(t, energy_height_m=energy, notes=('Invalid energy rate',))
        if not all(math.isfinite(x) for x in (climb, acceleration, kinetic_sep, sep)):
            self.reset()
            return EnergyMetrics(t, energy_height_m=energy, notes=('Invalid energy rate',))
        return EnergyMetrics(t, energy, sep, kinetic_sep, acceleration, climb, True,
                             ('Rates use the same rolling telemetry window.',))
