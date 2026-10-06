"""Offensive aids: hit probability if you fire now, from the distilled network (wt_overlay.pk).

Kill rose: rings are launch range, sectors how the target blip moves on the
B-scope (down = target hot, course 0 relative to the line of sight; sideways =
beaming; up = cold; left and right are symmetric), each cell the probability of
a hit against the assumed opponent. B-scope and side-view lines: maximum range
(an undefended target is hit) hot and cold, and where the hit probability of a
hot shot first falls below 50 % and 25 %, by off-boresight azimuth and target
altitude difference. Lines are found on a coarse range ladder then bisected; the
network was trained on 2-45 km, so a line still holding at 45 km is reported as
45 km with ``capped`` set ("> 45 km") instead of extrapolated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path

from . import pk

ROSE_RANGES_M = (3000., 5000., 7000., 10000., 15000., 20000., 30000.)
ROSE_COURSES_DEG = (0., 30., 60., 90., 120., 150., 180.)
ENVELOPE_LINES = ("rmax_hot", "rmax_cold", "pk50_hot", "pk25_hot")
ENVELOPE_AZIMUTHS_DEG = (0., 15., 30., 45., 60.)
ENVELOPE_ALT_DIFFS_M = (-5000., -3000., -1500., 0., 1500., 3000., 5000.)
LADDER_KM = (2, 3, 4, 5, 7, 10, 14, 20, 28, 36, 45)
TOLERANCE_M = 250.
# A shot counts as reaching when P(reach) is above the logit-1 threshold calibrated for the HUD.
REACH_P = 1/(1+math.exp(-1.))
SKILLS = {"normal": "普通", "top": "高手"}
MODES = {"tws": "TWS", "stt": "STT"}


def course_for_blip_direction(direction_deg: float) -> float:
    """B-scope blip motion direction (0 = straight down, ±180 = up) -> target course (0 = hot)."""
    return abs((direction_deg+180.) % 360.-180.)


def available(data_dir: Path | None = None) -> list[str]:
    return pk.available(data_dir)


@dataclass(frozen=True)
class Rose:
    missile: str
    assumption: pk.Assumption
    ranges_m: tuple[float, ...]
    courses_deg: tuple[float, ...]
    cells: dict  # (range_m, course_deg) -> (p_hit, p_reach) or None (outside the model)

    def at(self, range_m, course_deg):
        return self.cells[(range_m, course_deg)]


@dataclass(frozen=True)
class Envelope:
    """Lines (metres, None = absent) by (azimuth_deg, alt_diff_m) for the co-altitude row and the head-on
    column; ``capped`` holds (azimuth, alt_diff, line) that still held at the 45 km training limit."""
    missile: str
    assumption: pk.Assumption
    azimuths_deg: tuple[float, ...]
    alt_diffs_m: tuple[float, ...]
    lines: dict
    capped: frozenset = field(default_factory=frozenset)

    def line(self, name: str, alt_diff_m: float = 0.) -> list[tuple[float, float | None]]:
        """(azimuth, range) points for one line, mirrored to negative azimuths (co-altitude only)."""
        half = [(az, self.lines[(az, alt_diff_m)][name]) for az in self.azimuths_deg]
        return [(-az, r) for az, r in reversed(half) if az > 0] + half

    def profile(self, name: str, azimuth_deg: float = 0.) -> list[tuple[float, float | None]]:
        """(alt_diff_m, range) points for one line at one azimuth, low to high (head-on only)."""
        return [(dh, self.lines[(azimuth_deg, dh)][name]) for dh in self.alt_diffs_m]

    def is_capped(self, name: str, azimuth_deg: float = 0., alt_diff_m: float = 0.) -> bool:
        return (azimuth_deg, alt_diff_m, name) in self.capped


class OffenseAdvisor:
    """Rose and envelope for one missile and target assumption, from ownship altitude and TAS."""

    def __init__(self, missile: str, assumption: pk.Assumption | None = None, data_dir: Path | None = None):
        self.missile = missile
        self.assumption = assumption or pk.Assumption()
        self.model = pk.PkModel(missile, data_dir)

    def _hit(self, own_altitude_m, own_kmh, range_m, course_deg=0., azimuth_deg=0., alt_diff_m=0.):
        out = self.model.evaluate(own_altitude_m, own_kmh, self.assumption, range_m, course_deg, azimuth_deg,
                                  alt_diff_m)
        return None if out is None else (out[self.assumption.output], out["p_reach"])

    def rose(self, own_altitude_m: float, own_tas_mps: float, alt_diff_m: float = 0.) -> Rose:
        if not (math.isfinite(own_altitude_m) and math.isfinite(own_tas_mps)):
            raise ValueError("ownship altitude and TAS are required")
        kmh = own_tas_mps*3.6
        cells = {(r, c): self._hit(own_altitude_m, kmh, r, c, 0., alt_diff_m)
                 for r in ROSE_RANGES_M for c in ROSE_COURSES_DEG}
        return Rose(self.missile, self.assumption, ROSE_RANGES_M, ROSE_COURSES_DEG, cells)

    def _edge(self, holds):
        """(outer edge of the first range band where ``holds``, capped at the training limit), or (None, False)."""
        lo = hi = None
        for km in LADDER_KM:
            if holds(km*1000.):
                lo = km*1000.
            elif lo is not None:
                hi = km*1000.
                break
        if lo is None:
            return None, False
        if hi is None:
            return lo, lo >= pk.MAX_RANGE_M
        while hi-lo > TOLERANCE_M:
            mid = (lo+hi)/2
            lo, hi = (mid, hi) if holds(mid) else (lo, mid)
        return lo, False

    def reach_lines(self, own_altitude_m: float, own_tas_mps: float):
        """(rmax_hot, rmax_cold) in metres, None where absent: the same lines as ``envelope`` at zero off-boresight
        and co-altitude, without the two hit-probability lines (half the work). Hot lines stop at the network's 45 km."""
        if not (math.isfinite(own_altitude_m) and math.isfinite(own_tas_mps)):
            raise ValueError("ownship altitude and TAS are required")
        kmh = own_tas_mps*3.6

        def line(course):
            def holds(range_m):
                got = self._hit(own_altitude_m, kmh, range_m, course)
                return got is not None and got[1] >= REACH_P
            return self._edge(holds)[0]
        return line(0.), line(180.)

    def envelope(self, own_altitude_m: float, own_tas_mps: float, azimuths_deg=ENVELOPE_AZIMUTHS_DEG,
                 alt_diffs_m=ENVELOPE_ALT_DIFFS_M) -> Envelope:
        if not (math.isfinite(own_altitude_m) and math.isfinite(own_tas_mps)):
            raise ValueError("ownship altitude and TAS are required")
        kmh = own_tas_mps*3.6
        lines, capped = {}, set()
        # The B-scope uses the co-altitude row, the side view the head-on column.
        combos = sorted({(az, 0.) for az in azimuths_deg} | {(0., dh) for dh in alt_diffs_m})
        for az, dh in combos:
            def value(range_m, course, which, az=az, dh=dh):
                out = self._hit(own_altitude_m, kmh, range_m, course, az, dh)
                return None if out is None else out[which]
            tests = {"rmax_hot": lambda r: (value(r, 0., 1) or 0.) >= REACH_P,
                     "rmax_cold": lambda r: (value(r, 180., 1) or 0.) >= REACH_P,
                     "pk50_hot": lambda r: (value(r, 0., 0) or 0.) >= .5,
                     "pk25_hot": lambda r: (value(r, 0., 0) or 0.) >= .25}
            row = {}
            for name, holds in tests.items():
                row[name], cap = self._edge(holds)
                if cap:
                    capped.add((az, dh, name))
            lines[(az, dh)] = row
        return Envelope(self.missile, self.assumption, tuple(azimuths_deg), tuple(alt_diffs_m), lines,
                        frozenset(capped))
