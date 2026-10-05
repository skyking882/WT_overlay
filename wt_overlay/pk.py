"""Hit probability of a shot, from a distilled per-missile network (pure Python, no numpy).

scripts/distill_pk.py fits a small MLP to the offline escape-surrogate ensemble
(scripts/pk_model.py: opponent behaviour averaged over detection, human delay,
chaff and a defensive repertoire) and writes data/pk_models/<missile>.json.
Here it is only evaluated. Outputs per engagement:

    p_reach                        the missile hits a target that does not defend
    normal_tws / normal_stt        P(hit) against the default opponent mix
    top_tws / top_stt              P(hit) against an opponent who always defends correctly

The engagement is the shooter's state (8111 telemetry) plus an assumed target:
course relative to the line of sight (0 = hot), off-boresight azimuth, range,
altitude difference, speed, pre-launch turn and aircraft type (as FM descriptors
at the target's altitude and speed, see ``descriptors``). The training data only
reached 45 km (MAX_RANGE_M); beyond it nothing is extrapolated.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

from .contracts import G

DATA_DIR = Path(__file__).resolve().parents[1]/"data"/"pk_models"
OUTPUTS = ("p_reach", "normal_tws", "normal_stt", "top_tws", "top_stt")
FEATURES = ("launch_altitude_m", "launch_speed_kmh", "alt_diff_m", "target_altitude_m", "target_speed_kmh",
            "cos_course", "sin_course", "azimuth_deg", "turn_g", "range_m", "log_range",
            "load_cap", "thrust_w", "drag_cap_w", "drag_1g_w", "mass_kg")
MIN_RANGE_M, MAX_RANGE_M = 2000., 45000.
MASS_FACTOR = 1.3  # Total / empty mass of the assumed target (fuel and missiles), as in training.
# Game units with identical parameters share one network.
ALIASES = {"cn_sd10a": "cn_pl12", "su_rvv_ae": "su_r_77", "us_aim_120b": "us_aim_120a", "us_aim_120c_7": "us_aim_120c_5"}


def descriptors(model, altitude_m, speed_mps):
    """Target FM at one condition, per unit weight: load at the 20 deg AoA cap, max thrust,
    drag at the cap and at 1 g. None outside the FM tables."""
    weight = model.mass*G
    try:
        thrust, drag_cap, lift, _ = model.forces_at_aoa(altitude_m, speed_mps, 20., model.max_throttle)
        drag_1g = model.forces_at_aoa(altitude_m, speed_mps, model.target_aoa(altitude_m, speed_mps, 1.),
                                      model.max_throttle)[1]
    except ValueError:
        return None
    return dict(load_cap=round(lift/weight, 3), thrust_w=round(thrust/weight, 3), drag_cap_w=round(drag_cap/weight, 3),
                drag_1g_w=round(drag_1g/weight, 4))


def features(launch_altitude_m, launch_speed_kmh, target_altitude_m, target_speed_kmh, course_deg, azimuth_deg,
             turn_g, range_m, evader, mass_kg):
    course = math.radians(course_deg)
    return [launch_altitude_m, launch_speed_kmh, target_altitude_m-launch_altitude_m, target_altitude_m,
            target_speed_kmh, math.cos(course), math.sin(course), azimuth_deg, turn_g, range_m, math.log(range_m),
            evader["load_cap"], evader["thrust_w"], evader["drag_cap_w"], evader["drag_1g_w"], mass_kg]


class PkNet:
    """Standardised inputs -> SiLU MLP -> logits -> probabilities."""

    def __init__(self, data: dict):
        if tuple(data["features"]) != FEATURES or tuple(data["outputs"]) != OUTPUTS:
            raise ValueError("hit-probability model has a different feature or output layout")
        self.missile = data["missile"]
        self.meta = data.get("meta", {})
        self.mean, self.std = data["mean"], data["std"]
        self.layers = [(layer["w"], layer["b"]) for layer in data["layers"]]

    @classmethod
    def load(cls, missile: str, data_dir: Path | None = None) -> "PkNet":
        name = ALIASES.get(missile, missile)
        return cls(json.loads((Path(data_dir or DATA_DIR)/f"{name}.json").read_text()))

    def __call__(self, x):
        h = [(v-m)/s for v, m, s in zip(x, self.mean, self.std)]
        last = len(self.layers)-1
        for i, (w, b) in enumerate(self.layers):
            h = [sum(wij*hj for wij, hj in zip(row, h))+bi for row, bi in zip(w, b)]
            if i < last:
                h = [v/(1.+math.exp(-v)) if v > -60. else 0. for v in h]
        return {name: 1./(1.+math.exp(-max(-60., min(60., v)))) for name, v in zip(OUTPUTS, h)}


def available(data_dir: Path | None = None) -> list[str]:
    """Missiles with a hit-probability model, aliases included."""
    found = {p.stem for p in Path(data_dir or DATA_DIR).glob("*.json")}
    return sorted(found | {alias for alias, base in ALIASES.items() if base in found})


@dataclass(frozen=True)
class Assumption:
    """What the shooter assumes about the target."""
    aircraft: str = "f_16c_block_50"
    speed_kmh: float = 1000.
    turn_g: float = 0.
    skill: str = "normal"   # 'normal' or 'top'
    mode: str = "tws"       # 'tws' or 'stt'

    @property
    def output(self):
        return f"{self.skill}_{self.mode}"


class PkModel:
    """One missile's network plus the assumed target's flight model."""

    def __init__(self, missile: str, data_dir: Path | None = None):
        self.missile = missile
        self.net = PkNet.load(missile, data_dir)
        self._fm = {}
        self._descriptors = {}

    def _target(self, aircraft, altitude_m, speed_kmh):
        if aircraft not in self._fm:
            from .fm import load_aircraft
            from .turn import ManeuverModel
            fm = load_aircraft(aircraft)
            self._fm[aircraft] = ManeuverModel(fm, fm.empty_mass_kg*MASS_FACTOR, True)
        key = (aircraft, round(altitude_m, -2), round(speed_kmh, -1))
        if key not in self._descriptors:
            self._descriptors[key] = descriptors(self._fm[aircraft], key[1], key[2]/3.6)
        return self._descriptors[key], self._fm[aircraft].mass

    def evaluate(self, own_altitude_m, own_speed_kmh, assumption: Assumption, range_m, course_deg=0.,
                 azimuth_deg=0., alt_diff_m=0.):
        """All outputs for one engagement, or None outside the trained range or the target FM tables."""
        if not MIN_RANGE_M <= range_m <= MAX_RANGE_M:
            return None
        target_altitude = max(500., own_altitude_m+alt_diff_m)
        evader, mass = self._target(assumption.aircraft, target_altitude, assumption.speed_kmh)
        if evader is None:
            return None
        return self.net(features(own_altitude_m, own_speed_kmh, target_altitude, assumption.speed_kmh, course_deg,
                                 abs(azimuth_deg), assumption.turn_g, range_m, evader, mass))
