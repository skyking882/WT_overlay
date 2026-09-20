"""Read-only, loopback-only 8111 polling; exact unit-bearing state keys are used.

Schema reference: matrixsukhoi/voidmei src/parser/State.java and Indicators.java.
Body Ny has its own display field; it is not the lift/weight ratio used by performance models.
"""
from __future__ import annotations

from http.client import HTTPException
import ipaddress
import json
import math
import re
import time
from collections.abc import Mapping
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, build_opener

from .contracts import FlightState, G

_MAX_BYTES = 256 * 1024


def _number(data: Mapping, key: str) -> float | None:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        return None
    try:
        number = float(value)
    except (OverflowError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _bad_numbers(value: object) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, Mapping):
        return any(_bad_numbers(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_bad_numbers(v) for v in value)
    return False


def parse_telemetry(state: Mapping, indicators: Mapping, time_s: float) -> FlightState:
    """Parse SI state; missing optional data remain None, missing core data invalidate."""
    timestamp = _number({'t': time_s}, 't')
    if timestamp is None:
        return FlightState(0.0, False, notes=('Invalid telemetry timestamp',))
    if not isinstance(state, Mapping) or not isinstance(indicators, Mapping):
        return FlightState(timestamp, False, notes=('Invalid telemetry payload',))
    if state.get('valid') is not True or indicators.get('valid') is not True:
        return FlightState(timestamp, False, notes=('No active flight telemetry',))
    if indicators.get('army') in ('tank', 'ship', 'boat'):
        return FlightState(timestamp, False, notes=('Telemetry is not an aircraft',))
    if _bad_numbers(state) or _bad_numbers(indicators):
        return FlightState(timestamp, False, notes=('Non-finite telemetry payload',))
    altitude, tas = _number(state, 'H, m'), _number(state, 'TAS, km/h')
    if altitude is None or tas is None or tas < 0:
        return FlightState(timestamp, False, notes=('Missing or invalid altitude/TAS',))
    ias = _number(state, 'IAS, km/h')
    fuel = _number(state, 'Mfuel, kg')
    mass = _number(state, 'mass, kg')
    # A unit must be explicit. A bare "thrust" or ambiguous "kG" is not converted.
    thrusts: list[float] = []
    for key in state:
        if not isinstance(key, str):
            continue
        match = re.fullmatch(r'thrust \d+, (N|kgf)', key)
        if match:
            value = _number(state, key)
            if value is None:
                thrusts = []
                break
            thrusts.append(value * (G if match[1] == 'kgf' else 1.0))
    try:
        thrust = math.fsum(thrusts) if thrusts else None
    except OverflowError:
        return FlightState(timestamp, False, notes=('Invalid total thrust',))
    if thrust is not None and not math.isfinite(thrust):
        return FlightState(timestamp, False, notes=('Invalid total thrust',))
    aircraft = indicators.get('type')
    aircraft = aircraft.strip() if isinstance(aircraft, str) else None
    notes = ('Body Ny is retained as raw telemetry, not interpreted as L/W.',) if 'Ny' in state else ()
    return FlightState(
        timestamp, True, altitude_m=altitude, tas_mps=tas / 3.6,
        ias_mps=ias / 3.6 if ias is not None and ias >= 0 else None,
        mach=_number(state, 'M'), vertical_speed_mps=_number(state, 'Vy, m/s'),
        aoa_deg=_number(state, 'AoA, deg'), aos_deg=_number(state, 'AoS, deg'),
        heading_deg=_number(indicators, 'compass'),
        pitch_deg=_number(indicators, 'aviahorizon_pitch'),
        roll_deg=_number(indicators, 'aviahorizon_roll'),
        throttle_percent=_number(state, 'throttle 1, %'),
        fuel_kg=fuel if fuel is not None and fuel >= 0 else None,
        mass_kg=mass if mass is not None and mass > 0 else None,
        thrust_n=thrust, aircraft_id=aircraft or None, notes=notes,
        raw_state=dict(state), raw_indicators=dict(indicators),
        normal_load_g=_number(state, 'Ny'),
    )


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError('8111 redirects are not permitted')


class TelemetryClient:
    """Poll two local endpoints; call from a worker thread, never the GUI thread."""
    def __init__(self, base_url: str = 'http://127.0.0.1:8111', timeout: float = 0.5):
        parts = urlsplit(base_url)
        host = parts.hostname
        try:
            local = host == 'localhost' or ipaddress.ip_address(host or '').is_loopback
            port = parts.port
        except ValueError:
            local = False
            port = None
        if (not local or parts.scheme != 'http' or parts.username is not None
                or parts.password is not None or parts.path not in ('', '/')
                or parts.query or parts.fragment or port == 0):
            raise ValueError('8111 URL must be an HTTP loopback address without credentials or path')
        if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or not 0 < timeout <= 5:
            raise ValueError('timeout must be between 0 and 5 seconds')
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self._opener = build_opener(ProxyHandler({}), _NoRedirect())

    def _read(self, endpoint: str) -> Mapping:
        with self._opener.open(self.base_url + endpoint, timeout=self.timeout) as response:
            payload = response.read(_MAX_BYTES + 1)
        if len(payload) > _MAX_BYTES:
            raise ValueError('Telemetry response exceeds size limit')
        data = json.loads(payload)
        if not isinstance(data, dict):
            raise ValueError('Telemetry response must be a JSON object')
        return data

    def poll(self, time_s: float | None = None) -> FlightState:
        timestamp = time.monotonic() if time_s is None else time_s
        try:
            state = self._read('/state')
            indicators = self._read('/indicators')
            return parse_telemetry(state, indicators, timestamp)
        except (OSError, HTTPException, ValueError, TypeError, OverflowError, RecursionError) as error:
            timestamp = _number({'t': timestamp}, 't')
            return FlightState(timestamp if timestamp is not None else 0.0, False,
                               notes=(f'8111 unavailable: {type(error).__name__}',))
