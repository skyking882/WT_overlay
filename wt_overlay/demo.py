"""Explicit synthetic samples for interface demonstrations, never a real aircraft FM."""
from __future__ import annotations

import math
from .contracts import FlightState
from .fm import atmosphere


def make_demo_sample(time_s: float) -> FlightState:
    if not isinstance(time_s, (int, float)) or not math.isfinite(time_s):
        return FlightState(0.0, False, source='demo', notes=('Invalid demo timestamp',))
    phase = time_s / 8.0
    altitude = 4000.0 + 180.0 * math.sin(phase)
    tas = 280.0 + 22.0 * math.sin(phase * 0.7)
    density, _ = atmosphere(altitude)
    return FlightState(
        time_s, True, altitude_m=altitude, tas_mps=tas,
        ias_mps=tas * math.sqrt(density / 1.225),
        vertical_speed_mps=22.5 * math.cos(phase),
        aoa_deg=4.0 + 2.0 * math.sin(phase), roll_deg=25.0 * math.sin(phase * 0.5),
        normal_load_g=1.0 + 2.0 * math.sin(phase * 0.5),
        aos_deg=0.5 * math.sin(phase * 0.7), pitch_deg=8.0 + 3.0 * math.sin(phase),
        heading_deg=(45.+time_s*4.) % 360.,
        fuel_kg=4600.0, thrust_n=210000.0,
        throttle_percent=100.0, aircraft_id='synthetic-demo', source='demo',
        notes=('SYNTHETIC DEMO: these values do not represent a real aircraft.',),
    )
