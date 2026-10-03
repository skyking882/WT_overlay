"""Precomputed aircraft L/q and D/q vs Mach and body AoA.

Built from the same component polars as static force assembly. Vertical surfaces
contribute drag only. This is a lookup cache, not a new aerodynamic model.
SEP/climb continue to call StaticModel.evaluate directly.
"""
from __future__ import annotations

from bisect import bisect_right

QUERY_AOA_MIN, QUERY_AOA_MAX = -60., 60.
MAX_MACH = 2.35
_AOA_STEP = .1
_MACH_EPS = 1e-12


def _sorted_unique(values, eps=_MACH_EPS):
    out = []
    for x in sorted(values):
        if not out or x-out[-1] > eps:
            out.append(float(x))
    return tuple(out)


def _mach_knots(breakpoints):
    knots = [i/50 for i in range(int(MAX_MACH*50)+1)]
    knots.append(MAX_MACH)
    knots.extend(i/100 for i in range(70, 131))
    knots.extend(b for b in breakpoints if 0 <= b <= MAX_MACH)
    return _sorted_unique(knots)


def _aoa_grid():
    step = int(round(_AOA_STEP*10))
    lo = int(round(QUERY_AOA_MIN*10))
    hi = int(round(QUERY_AOA_MAX*10))
    return [i/10 for i in range(lo, hi+step, step)]


def assemble_over_q(polars, aoa_deg):
    """Aircraft L/q and D/q (m²) at body AoA; same assembly as the turn force model."""
    lift_q = drag_q = 0.
    for part, polar in polars:
        effective = part.incidence_deg if part.vertical else aoa_deg+part.incidence_deg
        cd, cl = polar.coefficients(effective)
        drag_q += part.area_m2*cd
        if not part.vertical:
            lift_q += part.area_m2*cl
    return lift_q, drag_q


def polar_forces_over_q(parts, mach, aoa_deg):
    polars = [(part, part.properties.at_mach(mach)) for part in parts]
    return assemble_over_q(polars, aoa_deg)


def _weight(xs, x, low_msg, high_msg):
    if x < xs[0]:
        raise ValueError(low_msg)
    if x > xs[-1]:
        raise ValueError(high_msg)
    i = bisect_right(xs, x)-1
    if xs[i] == x or i == len(xs)-1:
        return i, 0.
    return i, (x-xs[i])/(xs[i+1]-xs[i])


_TABLE_CACHE = {}
_TABLE_CACHE_KEYS = []
_TABLE_CACHE_LIMIT = 32


class AeroForceTable:
    """Piecewise-linear Mach–AoA table of whole-aircraft L/q and D/q."""

    def __init__(self, machs, aoa, lift, drag):
        if len(machs) < 2:
            raise ValueError("气动表无法覆盖可用马赫范围")
        self._machs = machs
        self._aoa = aoa
        self._lift = lift
        self._drag = drag
        idx = list(range(0, len(aoa), 5))
        if idx[-1] != len(aoa)-1:
            idx.append(len(aoa)-1)
        self._target_idx = tuple(idx)

    @property
    def mach_knots(self):
        return self._machs

    @classmethod
    def from_aircraft(cls, model, sweep=0.):
        ident = getattr(getattr(model, "info", None), "aircraft_id", None) or id(model)
        key = (ident, round(float(sweep), 10))
        cached = _TABLE_CACHE.get(key)
        if cached is not None:
            return cached
        table = cls._build(model, sweep)
        _TABLE_CACHE[key] = table
        _TABLE_CACHE_KEYS.append(key)
        if len(_TABLE_CACHE_KEYS) > _TABLE_CACHE_LIMIT:
            _TABLE_CACHE.pop(_TABLE_CACHE_KEYS.pop(0), None)
        return table

    @classmethod
    def _build(cls, model, sweep):
        parts = model.components_at_sweep(sweep)
        breakpoints = [v for part in parts for curve in part.properties.curves
                       for v in (curve.critical, curve.maximum)]
        aoa = tuple(_aoa_grid())
        machs, lift, drag = [], [], []
        for mach in _mach_knots(breakpoints):
            try:
                polars = [(part, part.properties.at_mach(mach)) for part in parts]
            except (ValueError, OverflowError):
                continue
            row_l, row_d = zip(*(assemble_over_q(polars, a) for a in aoa))
            machs.append(mach)
            lift.append(row_l)
            drag.append(row_d)
        return cls(tuple(machs), aoa, tuple(lift), tuple(drag))

    def lookup(self, mach, aoa_deg):
        i, tm = _weight(self._machs, mach, "超出模型范围", "超出模型范围")
        j, ta = _weight(self._aoa, aoa_deg, "迎角超出转向计算范围 ±60°", "迎角超出转向计算范围 ±60°")
        def cell(ii, jj):
            return self._lift[ii][jj], self._drag[ii][jj]
        if ta == 0:
            l0, d0 = cell(i, j)
            if tm == 0:
                return l0, d0
            l1, d1 = cell(i+1, j)
            return l0+tm*(l1-l0), d0+tm*(d1-d0)
        l00, d00 = cell(i, j)
        l01, d01 = cell(i, j+1)
        l0, d0 = l00+ta*(l01-l00), d00+ta*(d01-d00)
        if tm == 0:
            return l0, d0
        l10, d10 = cell(i+1, j)
        l11, d11 = cell(i+1, j+1)
        l1, d1 = l10+ta*(l11-l10), d10+ta*(d11-d10)
        return l0+tm*(l1-l0), d0+tm*(d1-d0)

    def lift_curve(self, mach):
        """AoA–L/q samples at this Mach on the shared 0.1° grid."""
        i, t = _weight(self._machs, mach, "超出模型范围", "超出模型范围")
        aoa = self._aoa
        if t == 0:
            lift = self._lift[i]
        else:
            a, b = self._lift[i], self._lift[i+1]
            lift = tuple((1-t)*x+t*y for x, y in zip(a, b))
        return tuple(zip(aoa, lift))

    def target_aoa(self, mach, target_lift_q, reference_deg=0.):
        """Nearest-to-reference root of L/q on the interpolant; else nearest sample."""
        i, t = _weight(self._machs, mach, "超出模型范围", "超出模型范围")
        aoa = self._aoa
        lo, hi = self._lift[i], self._lift[i] if t == 0 else self._lift[i+1]
        def sample(k):
            return lo[k] if t == 0 else (1-t)*lo[k]+t*hi[k]
        roots = []
        prev_k = self._target_idx[0]
        prev_a, prev_l = aoa[prev_k], sample(prev_k)
        if prev_l == target_lift_q:
            roots.append(prev_a)
        for k in self._target_idx[1:]:
            a, lift = aoa[k], sample(k)
            fa, fb = prev_l-target_lift_q, lift-target_lift_q
            if fb == 0:
                roots.append(a)
            elif fa*fb < 0:
                a0, l0, k0 = prev_a, prev_l, prev_k
                for j in range(k0+1, k+1):
                    a1, l1 = aoa[j], sample(j)
                    f0, f1 = l0-target_lift_q, l1-target_lift_q
                    if f1 == 0:
                        roots.append(a1)
                        break
                    if f0*f1 < 0:
                        roots.append(a0-f0*(a1-a0)/(l1-l0))
                        break
                    a0, l0 = a1, l1
            prev_a, prev_l, prev_k = a, lift, k
        if roots:
            return min(roots, key=lambda a: (abs(a-reference_deg), abs(a)))
        best = min(self._target_idx, key=lambda k: (abs(sample(k)-target_lift_q), abs(aoa[k]-reference_deg)))
        return aoa[best]


__all__ = ["AeroForceTable", "QUERY_AOA_MIN", "QUERY_AOA_MAX", "MAX_MACH",
           "assemble_over_q", "polar_forces_over_q"]
