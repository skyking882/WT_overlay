"""Aircraft type from /state alone, for aircraft whose 8111 /indicators stays invalid (no "type").

Seen in game 2026-10-16: /indicators {"valid": false} with a live /state. /state still carries ``Mfuel0, kg``
(equal to the FM's Mass.MaxFuelMass0, checked on a Typhoon: 7110 kg, and a J-16: 9400 kg) and one
``thrust N, kgs`` per engine. Candidates are the catalog aircraft with that maximum fuel and engine count.
Candidates whose FMs give the same thrust at a few reference conditions form one class (the six Typhoon variants
are one class), and a single class is the answer at once. Otherwise the measured total thrust is compared with
each class's FM thrust at the same altitude, TAS and throttle while the throttle is steady, and the best class
wins when, over SAMPLES samples, its mean |ratio - 1| is within TOLERANCE and the runner-up is at least SEPARATION
worse. Live samples: J-16 at 2.9 km / 1013 km/h 0.995 (J-11B 1.018, Su-30SM2 1.036); Typhoon at 14 km / M1.76
0.961 while the 9400 kg family sat at 0.42-0.46, so the absolute tolerance is loose and the ranking decides.
A manual choice in settings always overrides this.
"""
from __future__ import annotations

from functools import lru_cache
import json
import math
import re

from .contracts import G, PerformanceCondition

TOLERANCE = .08      # mean |ratio - 1| of the best class (FM vs game thrust drifts up to ~4 % at M1.8, 14 km)
SEPARATION = .01     # the runner-up must be at least this much worse (mean |ratio - 1|)
SAME_THRUST = .005   # FMs within 0.5 % of each other at every reference condition are one class
REFERENCES = ((1000., 200., 100.), (3000., 280., 100.), (3000., 300., 110.), (8000., 400., 110.),
              (12000., 520., 110.))   # (altitude m, TAS m/s, throttle %) for the classes
SAMPLES = 3          # steady-throttle samples before deciding
MIN_SPACING_S = .5


@lru_cache(maxsize=1)
def _index():
    """(aircraft id, max fuel kg, engine count) for every catalog aircraft with readable fuel data."""
    from .fm.catalog import aircraft_catalog
    from .flight import read_fuel
    out = []
    for aircraft in aircraft_catalog():
        try:
            raw = json.loads(aircraft.path.read_text(encoding="utf-8"))
            fuel = read_fuel(raw)
        except (OSError, ValueError):
            continue
        engines = sum(1 for k in raw if k.startswith("Engine") and k[6:].isdigit())
        out.append((aircraft.id, fuel.max_kg, engines))
    return tuple(out)


def _number(value):
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) \
        else None


def state_signature(raw):
    """(max fuel kg, engine count, total thrust kgf, throttle %) from a raw /state payload."""
    fuel0 = _number(raw.get("Mfuel0, kg"))
    thrusts = [_number(v) for k, v in raw.items() if isinstance(k, str) and re.fullmatch(r"thrust \d+, kgs", k)]
    throttle = _number(raw.get("throttle 1, %"))
    if fuel0 is None or not thrusts or any(t is None for t in thrusts):
        return None
    return fuel0, len(thrusts), sum(thrusts), throttle


def thrust_classes(candidates, model=None):
    """Candidates grouped by FM thrust at REFERENCES (catalog order kept)."""
    if model is None:
        from .fm import load_aircraft as model
    signatures = {}
    for aircraft in candidates:
        values = []
        for altitude, tas, throttle in REFERENCES:
            try:
                point = model(aircraft).evaluate(PerformanceCondition(altitude, tas, 20000.,
                                                                      throttle_percent=throttle))
                values.append(point.thrust_n if point.valid else None)
            except (OSError, ValueError, KeyError):
                values.append(None)
        signatures[aircraft] = values
    classes = []
    for aircraft in candidates:
        for group in classes:
            other = signatures[group[0]]
            if all((a is None and b is None) or (a and b and abs(a/b-1.) <= SAME_THRUST)
                   for a, b in zip(signatures[aircraft], other)):
                group.append(aircraft)
                break
        else:
            classes.append([aircraft])
    return [tuple(g) for g in classes]


class AircraftGuess:
    def __init__(self, loader=None):
        self._loader = loader
        self.reset()

    def reset(self):
        self._key = None
        self._candidates = ()
        self._classes = []
        self._errors = {}
        self._samples = 0
        self._last = None          # (time, throttle) of the previous sample
        self._last_used = -math.inf
        self.aircraft_id = None
        self.note = ""

    def _model(self, aircraft_id):
        if self._loader is not None:
            return self._loader(aircraft_id)
        from .fm import load_aircraft
        return load_aircraft(aircraft_id)

    def update(self, state) -> str | None:
        """Feed one valid sample; returns the identified aircraft id (or None while undecided)."""
        signature = state_signature(state.raw_state or {})
        if signature is None:
            self.note = "8111 /state 没有 Mfuel0 或推力，无法识别机型"
            return None
        fuel0, engines, thrust, throttle = signature
        key = (round(fuel0), engines)
        if key != self._key:
            self.reset()
            self._key = key
            self._candidates = tuple(a for a, fuel, n in _index() if abs(fuel-fuel0) < 1. and n == engines)
            self._classes = thrust_classes(self._candidates, self._model) if len(self._candidates) > 1 else \
                [self._candidates] if self._candidates else []
            if not self._candidates:
                self.note = f"没有最大油量 {fuel0:.0f} kg、{engines} 台发动机的机型"
            elif len(self._classes) == 1:
                self.aircraft_id = self._classes[0][0]
                if len(self._classes[0]) > 1:
                    self.note = f"同系 {len(self._classes[0])} 个机型推力相同，按 {self.aircraft_id} 计算"
            else:
                self._errors = {c[0]: [] for c in self._classes}
        if self.aircraft_id is not None or len(self._classes) < 2:
            return self.aircraft_id
        steady = (self._last is not None and throttle is not None and self._last[1] == throttle
                  and state.time_s-self._last[0] < 2.)
        self._last = (state.time_s, throttle)
        if not (steady and thrust > 0 and throttle and state.tas_mps and state.tas_mps > 50
                and state.time_s-self._last_used >= MIN_SPACING_S):
            return None
        self._last_used = state.time_s
        values = {}
        for aircraft in self._errors:
            try:
                point = self._model(aircraft).evaluate(PerformanceCondition(
                    state.altitude_m, state.tas_mps, 20000., throttle_percent=throttle))
            except (OSError, ValueError, KeyError):
                continue
            if point.valid and point.thrust_n:
                values[aircraft] = point.thrust_n/G
        if not values:
            return None
        self._samples += 1
        for aircraft, fm_thrust in values.items():
            self._errors[aircraft].append(abs(thrust/fm_thrust-1.))
        if self._samples < SAMPLES:
            self.note = f"按油量和推力识别机型中（{len(self._classes)} 类候选）"
            return None
        mean = {a: sum(e)/len(e) for a, e in self._errors.items() if len(e) == self._samples}
        ranked = sorted(mean, key=mean.get)
        best = ranked[0]
        if mean[best] <= TOLERANCE and (len(ranked) == 1 or mean[ranked[1]]-mean[best] >= SEPARATION):
            self.aircraft_id = best
            group = next(c for c in self._classes if c[0] == best)
            self.note = "" if len(group) == 1 else f"同系 {len(group)} 个机型推力相同，按 {best} 计算"
            return best
        self.note = "无法按油量和推力确定机型，请在设置里手动选择"
        self._errors = {c[0]: [] for c in self._classes}
        self._samples = 0
        return None
