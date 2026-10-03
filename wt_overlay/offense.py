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
# Values are clamped to the grid edge; only clearly outside counts as "outside the table".
ALTITUDE_MARGIN_M = 500.
TAS_MARGIN_MPS = 50/3.6


def _outside(values, x, margin):
    return x < values[0]-margin or x > values[-1]+margin


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
                        dict(table.reaction), _outside(self.altitudes, altitude_m, ALTITUDE_MARGIN_M),
                        _outside(self.speeds, tas_mps, TAS_MARGIN_MPS), True)
        a0, a1, wa, _ = self._bracket(self.altitudes, altitude_m)
        v0, v1, wv, _ = self._bracket(self.speeds, tas_mps)
        a_clamped = _outside(self.altitudes, altitude_m, ALTITUDE_MARGIN_M)
        v_clamped = _outside(self.speeds, tas_mps, TAS_MARGIN_MPS)
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


ENVELOPE_DIR = Path(__file__).resolve().parents[1]/"data"/"envelope"
ENVELOPE_LINES = ("rmax_hot", "rmax_cold", "r3_hot", "rne_hot")


@dataclass(frozen=True)
class Envelope:
    """Launch-envelope lines (metres, None = line absent) by (azimuth_deg, alt_diff_m)."""
    missile: str
    azimuths_deg: tuple[float, ...]
    alt_diffs_m: tuple[float, ...]
    lines: dict  # (azimuth_deg, alt_diff_m) -> {line: metres or None}
    altitude_clamped: bool
    tas_clamped: bool
    nearest_only: bool = False

    def line(self, name: str, alt_diff_m: float = 0.) -> list[tuple[float, float | None]]:
        """(azimuth, range) points for one line, mirrored to negative azimuths."""
        half = [(az, self.lines[(az, alt_diff_m)][name]) for az in self.azimuths_deg]
        return [(-az, r) for az, r in reversed(half) if az > 0] + half


class EnvelopeLibrary(RoseLibrary):
    """Envelope tables share the rose's ownship grid handling."""

    def __init__(self, missile: str, evader: str, chaff_rcs_ratio: float, data_dir: Path | None = None):
        self.missile, self.evader, self.chaff_rcs_ratio = missile, evader, chaff_rcs_ratio
        grid = {}
        for path in sorted(Path(data_dir or ENVELOPE_DIR).glob(f"{missile}__{evader}__*.json")):
            data = json.loads(path.read_text())
            meta = data["meta"]
            if float(meta["chaff_rcs_ratio"]) != chaff_rcs_ratio:
                continue
            lines = {(float(r["azimuth_deg"]), float(r["alt_diff_m"])): {k: r.get(k) for k in ENVELOPE_LINES}
                     for r in data["rows"]}
            grid[(float(meta["launch_altitude_m"]), float(meta["launch_speed_kmh"])/3.6)] = lines
        if not grid:
            raise FileNotFoundError(f"no envelope tables for {missile} vs {evader}, chaff {chaff_rcs_ratio:g}")
        keys = {tuple(sorted(lines)) for lines in grid.values()}
        if len(keys) != 1:
            raise ValueError("envelope tables use different azimuth/altitude grids")
        self._keys = keys.pop()
        self._grid = grid
        self.altitudes = sorted({k[0] for k in grid})
        self.speeds = sorted({k[1] for k in grid})
        self.complete = all((a, v) in grid for a in self.altitudes for v in self.speeds)

    def envelope(self, altitude_m: float, tas_mps: float) -> Envelope:
        if not (math.isfinite(altitude_m) and math.isfinite(tas_mps)):
            raise ValueError("ownship altitude and TAS are required")
        a_clamped = _outside(self.altitudes, altitude_m, ALTITUDE_MARGIN_M)
        v_clamped = _outside(self.speeds, tas_mps, TAS_MARGIN_MPS)
        if self.complete:
            a0, a1, wa, _ = self._bracket(self.altitudes, altitude_m)
            v0, v1, wv, _ = self._bracket(self.speeds, tas_mps)
            corners = [(c, w) for c, w in (((a0, v0), (1-wa)*(1-wv)), ((a1, v0), wa*(1-wv)),
                                           ((a0, v1), (1-wa)*wv), ((a1, v1), wa*wv)) if w > 0]
        else:
            nearest = min(self._grid, key=lambda k: ((k[0]-altitude_m)/1000)**2 + ((k[1]-tas_mps)*3.6/100)**2)
            corners = [(nearest, 1.)]
        lines = {}
        for key in self._keys:
            values = {}
            for name in ENVELOPE_LINES:
                terms = [(self._grid[c][key][name], w) for c, w in corners]
                values[name] = None if any(v is None for v, _ in terms) else sum(v*w for v, w in terms)
            lines[key] = values
        return Envelope(self.missile, tuple(sorted({k[0] for k in self._keys})),
                        tuple(sorted({k[1] for k in self._keys})), lines, a_clamped, v_clamped, not self.complete)
