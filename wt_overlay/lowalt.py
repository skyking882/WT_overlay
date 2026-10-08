"""Low-target mode: the launch-range window against a straight, level target near the surface.

Tables come from scripts/build_lowalt_window.py under data/lowalt_window/, one
per missile, ownship altitude and TAS, each holding a window (near, far) per
launch pitch and target off-boresight azimuth. The window is where every
worst-case height (25-35 m, multipath gain 0.5) is hit on both a hot and a cold
course: the far edge is set by multipath, the near edge by how steeply the
missile can dive. Looked up multilinearly in ownship altitude, TAS and pitch,
clamped to the grid; a window missing at any corner is missing. Aliases share
their base missile's tables (``pk.ALIASES``, e.g. SD-10A uses PL-12).
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import json
import math
from pathlib import Path

from .pk import ALIASES

DATA_DIR = Path(__file__).resolve().parents[1]/"data"/"lowalt_window"
# Values are clamped to the grid edge; only clearly outside counts as "outside the table".
ALTITUDE_MARGIN_M = 500.
TAS_MARGIN_MPS = 50/3.6
PITCH_MARGIN_DEG = 5.


def _bracket(values, x):
    """(lo, hi, weight of hi) on a sorted grid, clamped at the ends."""
    if x <= values[0]:
        return values[0], values[0], 0.
    if x >= values[-1]:
        return values[-1], values[-1], 0.
    i = bisect_right(values, x)
    lo, hi = values[i-1], values[i]
    return lo, hi, (x-lo)/(hi-lo)


def _outside(values, x, margin):
    return x < values[0]-margin or x > values[-1]+margin


def _blend(terms):
    """Weighted mean of (near, far) windows; None if any weighted window is None."""
    terms = [(w, v) for v, w in terms if w > 0]
    if any(v is None for _, v in terms):
        return None
    return tuple(sum(w*v[k] for w, v in terms) for k in (0, 1))


@dataclass(frozen=True)
class LowAltWindow:
    missile: str
    azimuths_deg: tuple[float, ...]
    worst: dict      # azimuth -> (near_m, far_m) or None
    reference: dict  # same, for the reference height alone
    reference_height_m: float
    clamped: bool    # ownship altitude, TAS or pitch outside the tables

    def band(self, which: str = "worst") -> list[tuple[float, float, float]]:
        """(azimuth, near, far) mirrored to negative azimuths; azimuths without a window are left out."""
        windows = getattr(self, which)
        half = [(az, *windows[az]) for az in self.azimuths_deg if windows[az] is not None]
        return [(-az, near, far) for az, near, far in reversed(half) if az > 0] + half

    def center(self, which: str = "worst") -> tuple[float, float] | None:
        return getattr(self, which).get(0.)


def available(data_dir: Path | None = None) -> list[str]:
    """Missiles with at least one low-target table on disk, aliases included."""
    found = {p.name.split("__")[0] for p in Path(data_dir or DATA_DIR).glob("*__*.json")}
    return sorted(found | {alias for alias, base in ALIASES.items() if base in found})


class LowAltLibrary:
    """All low-target tables for one missile."""

    def __init__(self, missile: str, data_dir: Path | None = None):
        self.missile = missile
        self._grid, shapes = {}, set()
        for path in sorted(Path(data_dir or DATA_DIR).glob(f"{ALIASES.get(missile, missile)}__*.json")):
            data = json.loads(path.read_text())
            meta = data["meta"]
            rows = {(float(r["pitch_deg"]), float(r["azimuth_deg"])):
                    tuple(None if r[k] is None else tuple(map(float, r[k])) for k in ("worst", "reference"))
                    for r in data["rows"]}
            self._grid[(float(meta["launch_altitude_m"]), float(meta["launch_speed_kmh"])/3.6)] = rows
            shapes.add((tuple(sorted(rows)), float(meta["reference_height_m"])))
        if not self._grid:
            raise FileNotFoundError(f"no low-target tables for {missile}")
        if len(shapes) != 1:
            raise ValueError("low-target tables use different pitch/azimuth grids")
        keys, self.reference_height_m = shapes.pop()
        self.pitches = sorted({p for p, _ in keys})
        self.azimuths = tuple(sorted({az for _, az in keys}))
        self.altitudes = sorted({a for a, _ in self._grid})
        self.speeds = sorted({v for _, v in self._grid})

    def _state(self, altitude_m, tas_mps):
        """Ownship-grid corners with weights; the nearest table alone when the grid has a hole."""
        a0, a1, wa = _bracket(self.altitudes, altitude_m)
        v0, v1, wv = _bracket(self.speeds, tas_mps)
        corners = [((a0, v0), (1-wa)*(1-wv)), ((a1, v0), wa*(1-wv)), ((a0, v1), (1-wa)*wv), ((a1, v1), wa*wv)]
        if all(c in self._grid for c, w in corners if w > 0):
            return corners
        # Scale: 1 km of altitude ~ 100 km/h of TAS when choosing the nearest table.
        nearest = min(self._grid, key=lambda k: ((k[0]-altitude_m)/1000)**2 + ((k[1]-tas_mps)*3.6/100)**2)
        return [(nearest, 1.)]

    def window(self, altitude_m: float, tas_mps: float, pitch_deg: float) -> LowAltWindow:
        if not all(math.isfinite(x) for x in (altitude_m, tas_mps, pitch_deg)):
            raise ValueError("ownship altitude, TAS and pitch are required")
        p0, p1, wp = _bracket(self.pitches, pitch_deg)
        corners = [((state, p), w*wq) for state, w in self._state(altitude_m, tas_mps)
                   for p, wq in ((p0, 1-wp), (p1, wp))]
        worst, reference = {}, {}
        for az in self.azimuths:
            terms = [(self._grid[state][(p, az)], w) for (state, p), w in corners]
            worst[az] = _blend([(v[0], w) for v, w in terms])
            reference[az] = _blend([(v[1], w) for v, w in terms])
        clamped = (_outside(self.altitudes, altitude_m, ALTITUDE_MARGIN_M) or _outside(self.speeds, tas_mps, TAS_MARGIN_MPS)
                   or _outside(self.pitches, pitch_deg, PITCH_MARGIN_DEG))
        return LowAltWindow(self.missile, self.azimuths, worst, reference, self.reference_height_m, clamped)
