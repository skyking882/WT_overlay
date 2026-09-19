"""Static Gaijin polares port (degrees), see THIRD_PARTY_NOTICES.md.

The caller supplies reference span/area and effective AoA. This module does not
infer component flow, control deflection, force reference frames or trim.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import cos, isfinite, pi, radians, sin, sqrt
from typing import Mapping


def finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def div(a: float, b: float) -> float:
    return a / b if b else 0.0


@dataclass(frozen=True)
class MachCurve:
    critical: float
    maximum: float
    peak: float
    slope: float
    limit: float
    initial: float = 1.0

    def evaluate(self, mach: float) -> float:
        if mach < self.critical:
            return self.initial
        if mach > self.maximum:
            value = self.peak + self.slope * (mach - self.maximum)
            return min(value, self.limit) if self.slope >= 0 else max(value, self.limit)
        width = self.maximum - self.critical
        if width <= 0:
            raise ValueError("singular Mach interpolation interval is unsupported")
        t = (mach - self.critical) / width
        # Equivalent to source cubic with f'(critical)=0, f'(maximum)=slope.
        return ((2*t**3 - 3*t**2 + 1)*self.initial
                + (-2*t**3 + 3*t**2)*self.peak
                + (t**3 - t**2)*width*self.slope)


_DEFAULT_CURVES = (
    (.6, 1., 7., -5.2, 1.), (.65, .97, 6.7, -3.7, 1.),
    (.3, 1., .32, -.44, .25), (.3, 1., .4, -.2, .25),
    (.6, 1.5, 2., 1.1, 5.), (0., 0., 0., 0., 0.),
    (0., 1., 1., 0., 1.),
)


@dataclass(frozen=True)
class PolarProperties:
    values: Mapping[str, float]
    curves: tuple[MachCurve, ...]
    mode: int
    combined_cl: bool
    effective_aspect_ratio: float

    @classmethod
    def from_mapping(cls, source: Mapping, span: float, area: float) -> "PolarProperties":
        span, area = finite(span, "span"), finite(area, "area")
        if span <= 0 or area <= 0:
            raise ValueError("positive span and area are required")
        defaults = {"lineClCoeff": .85, "AfterCritParabAngle": 5.,
                    "AfterCritDeclineCoeff": .007, "AfterCritMaxDistanceAngle": 30.,
                    "CxAfterCoeff": .01, "ClAfterCritHigh": source.get("ClAfterCrit", 1.09),
                    "Cl0": 0., "alphaCritHigh": 16., "alphaCritLow": -16.,
                    "ClCritHigh": 1., "ClCritLow": -1., "CdMin": .02}
        values = {key: finite(source.get(key, value), key) for key, value in defaults.items()}
        values["ClAfterCritLow"] = finite(source.get("ClAfterCritLow", -values["ClAfterCritHigh"]), "ClAfterCritLow")
        oswald = finite(source.get("OswaldsEfficiencyNumber"), "OswaldsEfficiencyNumber")
        if oswald <= 0 or values["lineClCoeff"] <= 0 or values["CdMin"] < 0:
            raise ValueError("invalid polar efficiency, lift slope or base drag")
        if values["alphaCritLow"] >= values["alphaCritHigh"]:
            raise ValueError("invalid critical AoA interval")
        mode = source.get("MachFactor", 0)
        if type(mode) is not int or mode not in (0, 1, 2, 3):
            raise ValueError("unsupported MachFactor")
        combined = source.get("CombinedCl", True)
        if type(combined) is not bool:
            raise ValueError("CombinedCl must be boolean")
        curves = []
        for i, default in enumerate(_DEFAULT_CURVES, 1):
            args = [finite(source.get(f"{key}{i}", d), f"{key}{i}") for key, d in zip(
                ("MachCrit", "MachMax", "MultMachMax", "MultLineCoeff", "MultLimit"), default)]
            if mode == 3 and args[1] <= args[0]:
                raise ValueError(f"Mach curve {i}: singular or reversed interval unsupported")
            curves.append(MachCurve(*args, initial=0. if i == 6 else 1.))
        return cls(values, tuple(curves), mode, combined, oswald*span*span/area)

    def at_mach(self, mach: float) -> "Polar":
        mach = finite(mach, "Mach")
        if mach < 0:
            raise ValueError("Mach must be nonnegative")
        v = self.values
        cl0, cd0, slope = v["Cl0"], v["CdMin"], v["lineClCoeff"]
        hi, lo = v["ClCritHigh"], v["ClCritLow"]
        ah, al = v["alphaCritHigh"], v["alphaCritLow"]
        induced, kq, clkq = 1/(pi*self.effective_aspect_ratio), 1., 1.
        if self.mode == 3:
            m = [curve.evaluate(mach) for curve in self.curves]
            cd0 *= m[0]
            slope *= 1 + m[0] - m[1] if self.combined_cl else m[1]
            cl0 *= m[6]
            hi, lo = v["Cl0"]+(hi-cl0)*m[2], v["Cl0"]+(lo-cl0)*m[2]
            ah, al, induced = ah*m[3], al*m[3], induced*m[4]
        elif self.mode in (1, 2):
            local_mach = min(mach, .9) if self.mode == 1 else mach
            base = abs(1-local_mach**2)
            if self.mode == 1:
                base = min(base, 1.)
            kq = 1/max(base**.3, .2)
            clkq = sqrt(kq)
        if slope <= 0 or hi <= lo or ah <= al or induced < 0 or cd0 < 0:
            raise ValueError("Mach produces unsupported/nonphysical polar parameters")
        lh, ll = 2*div(hi-cl0, slope)-ah, 2*div(lo-cl0, slope)-al
        if ll > lh:
            ll = lh = (ll+lh)/2
        ph = div(hi-(cl0+lh*slope), (ah-lh)**2)
        pl = div(-lo+(cl0+ll*slope), (ll-al)**2)
        return Polar(cl0, cd0, slope, induced, hi, lo, ah, al, lh, ll, ph, pl,
                     v["AfterCritParabAngle"], v["AfterCritDeclineCoeff"],
                     v["AfterCritMaxDistanceAngle"], v["CxAfterCoeff"],
                     v["ClAfterCritHigh"], v["ClAfterCritLow"], kq, clkq)


@dataclass(frozen=True)
class Polar:
    cl0: float
    cd0: float
    slope: float
    induced: float
    critical_cl_high: float
    critical_cl_low: float
    critical_aoa_high: float
    critical_aoa_low: float
    linear_high: float
    linear_low: float
    parab_high: float
    parab_low: float
    parab_angle: float
    decline: float
    max_distance: float
    after_drag: float
    after_cl_high: float
    after_cl_low: float
    kq: float = 1.
    cl_kq: float = 1.

    def cl(self, aoa: float) -> float:
        aoa = finite(aoa, "AoA")
        if not -180 <= aoa <= 180:
            raise ValueError("AoA outside [-180, 180]")
        if self.linear_low <= aoa <= self.linear_high:
            return self.cl0 + self.slope*aoa
        sign = 1. if aoa-self.linear_high+.01 >= 0 else -1.
        crit = self.critical_aoa_high if sign > 0 else self.critical_aoa_low
        cy = self.critical_cl_high if sign > 0 else self.critical_cl_low
        after = self.after_cl_high if sign > 0 else self.after_cl_low
        if sign*(aoa-crit) <= 0:
            parab = self.parab_high if sign > 0 else self.parab_low
            return cy-sign*parab*(crit-aoa)**2
        max_ang = max(40., self.max_distance)
        if sign*aoa <= max_ang:
            if sign*aoa <= self.max_distance:
                da = aoa-crit
                if sign*da < self.parab_angle:
                    return cy-sign*self.decline*da**2
                h = after*sin(pi*.0125*self.max_distance)
                max_da = sign*(self.max_distance-self.parab_angle)-crit
                need = cy-sign*self.decline*self.parab_angle**2-h
                return h+div(need, max_da**2)*(sign*self.max_distance-aoa)**2
            return after*sin(pi*.0125*sign*aoa)
        if sign*aoa <= 140:
            local_sign, local_aoa = sign, sign*aoa
            if local_aoa > 90:
                local_sign *= -1
                local_aoa = 180-local_aoa
            offset = sin(pi*.0125*max_ang)-sin(pi*.5+pi*.01*max(max_ang-40, 0))
            return sign*after*(sign*offset*(1-(sign*aoa-40)/100)
                               +sin(pi*.5+pi*.01*(local_aoa-40))*local_sign)
        return -after*sin(pi*.0125*(180-sign*aoa))

    def cd(self, aoa: float) -> float:
        aoa = finite(aoa, "AoA")
        if not -180 <= aoa <= 180:
            raise ValueError("AoA outside [-180, 180]")
        # Deliberately uses the linear expression, NOT self.cl(aoa).
        cy = self.cl0+self.slope*aoa
        drag = self.cd0+cy*cy*self.induced
        sign = 1. if aoa >= 0 else -1.
        critical = self.critical_aoa_high if aoa >= 0 else self.critical_aoa_low
        drag += self.after_drag*max(sign*(aoa-critical), 0.)
        return min(drag, .15+self.critical_cl_high*abs(sin(radians(aoa))))

    def coefficients(self, aoa: float, rotation_deg: float = 0.) -> tuple[float, float]:
        """Source calc_c: drag-like x and lift-like y, before dimensionalization."""
        angle = radians(finite(rotation_deg, "coefficient rotation"))
        cd, cl = self.cd(aoa), self.cl(aoa)
        return ((cd*cos(angle)-cl*sin(angle))*self.kq,
                (cl*cos(angle)+cd*sin(angle))*self.cl_kq)
