"""Offensive kill-rose data: target reaction time if you fire now.

Tables come from scripts/build_reaction_table.py (one per ownship altitude and
TAS, missile, evader aircraft and chaff assumption) under data/offense/. Here
they are only looked up: bilinear in ownship altitude and TAS, clamped to the
grid. A cell with ``reaction_s`` None (the missile misses even an unevading
target) reads as infinite. The rose is indexed by how the target blip moves on
a B-scope: down = target hot (course 0 relative to the line of sight),
sideways = beaming (90), up = cold (180); left and right are symmetric.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import json
import math
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1]/"data"/"offense"
INF = math.inf


@dataclass(frozen=True)
class RoseTable:
    altitude_m: float
    tas_mps: float
    ranges_m: tuple[float, ...]
    courses_deg: tuple[float, ...]
    reaction: dict  # (range_m, course_deg) -> seconds (inf = unreachable)


@dataclass(frozen=True)
class Rose:
    missile: str
    evader: str
    chaff_rcs_ratio: float
    ranges_m: tuple[float, ...]
    courses_deg: tuple[float, ...]
    reaction: dict  # (range_m, course_deg) -> seconds
    altitude_clamped: bool
    tas_clamped: bool
    nearest_only: bool = False  # Incomplete ownship grid: nearest table, no interpolation.

    def at(self, range_m, course_deg):
        return self.reaction[(range_m, course_deg)]


def _load(path: Path, turn_g: float) -> tuple[dict, RoseTable]:
    data = json.loads(path.read_text())
    meta = data["meta"]
    reaction = {(c["range_m"], c["course_deg"]): INF if c["reaction_s"] is None else float(c["reaction_s"])
                for c in data["cells"] if c["turn_g"] == turn_g}
    if not reaction:
        raise ValueError(f"{path.name}: no cells with turn {turn_g:g} g")
    return meta, RoseTable(float(meta["launch_altitude_m"]), float(meta["launch_speed_kmh"])/3.6,
                           tuple(sorted({k[0] for k in reaction})), tuple(sorted({k[1] for k in reaction})), reaction)


class RoseLibrary:
    """All tables for one missile, evader aircraft and chaff assumption."""

    def __init__(self, missile: str, evader: str, chaff_rcs_ratio: float, data_dir: Path | None = None,
                 turn_g: float = 0.):
        self.missile, self.evader, self.chaff_rcs_ratio = missile, evader, chaff_rcs_ratio
        tables = []
        for path in sorted(Path(data_dir or DATA_DIR).glob(f"{missile}__{evader}__*.json")):
            meta, table = _load(path, turn_g)
            if float(meta["chaff_rcs_ratio"]) == chaff_rcs_ratio:
                tables.append(table)
        if not tables:
            raise FileNotFoundError(f"no reaction tables for {missile} vs {evader}, chaff {chaff_rcs_ratio:g}")
        shapes = {(t.ranges_m, t.courses_deg) for t in tables}
        if len(shapes) != 1:
            raise ValueError("reaction tables use different range/course grids")
        self.ranges_m, self.courses_deg = shapes.pop()
        self._grid = {(t.altitude_m, t.tas_mps): t for t in tables}
        self.altitudes = sorted({t.altitude_m for t in tables})
        self.speeds = sorted({t.tas_mps for t in tables})
        self.complete = all((a, v) in self._grid for a in self.altitudes for v in self.speeds)

    @staticmethod
    def _bracket(values, x):
        if x <= values[0]:
            return values[0], values[0], 0., x < values[0]
        if x >= values[-1]:
            return values[-1], values[-1], 0., x > values[-1]
        i = bisect_right(values, x)
        lo, hi = values[i-1], values[i]
        return lo, hi, (x-lo)/(hi-lo), False

    def rose(self, altitude_m: float, tas_mps: float) -> Rose:
        if not (math.isfinite(altitude_m) and math.isfinite(tas_mps)):
            raise ValueError("ownship altitude and TAS are required")
        if not self.complete:
            # Scale: 1 km of altitude ~ 100 km/h of TAS when choosing the nearest table.
            key = min(self._grid, key=lambda k: ((k[0]-altitude_m)/1000)**2 + ((k[1]-tas_mps)*3.6/100)**2)
            table = self._grid[key]
            return Rose(self.missile, self.evader, self.chaff_rcs_ratio, self.ranges_m, self.courses_deg,
                        dict(table.reaction), not self.altitudes[0] <= altitude_m <= self.altitudes[-1],
                        not self.speeds[0] <= tas_mps <= self.speeds[-1], True)
        a0, a1, wa, a_clamped = self._bracket(self.altitudes, altitude_m)
        v0, v1, wv, v_clamped = self._bracket(self.speeds, tas_mps)
        corners = [((a0, v0), (1-wa)*(1-wv)), ((a1, v0), wa*(1-wv)), ((a0, v1), (1-wa)*wv), ((a1, v1), wa*wv)]
        reaction = {}
        for key in self._grid[(a0, v0)].reaction:
            terms = [(self._grid[corner].reaction[key], w) for corner, w in corners if w > 0]
            reaction[key] = INF if any(v == INF for v, _ in terms) else sum(v*w for v, w in terms)
        return Rose(self.missile, self.evader, self.chaff_rcs_ratio, self.ranges_m, self.courses_deg, reaction,
                    a_clamped, v_clamped)


def available(data_dir: Path | None = None) -> list[tuple[str, str, float]]:
    """(missile, evader, chaff ratio) combinations present on disk."""
    found = set()
    for path in Path(data_dir or DATA_DIR).glob("*__*__*.json"):
        missile, evader = path.name.split("__")[:2]
        found.add((missile, evader, float(json.loads(path.read_text())["meta"]["chaff_rcs_ratio"])))
    return sorted(found)


def course_for_blip_direction(direction_deg: float) -> float:
    """B-scope blip motion direction (0 = straight down, ±180 = up) -> target course (0 = hot)."""
    return abs((direction_deg+180.) % 360.-180.)
