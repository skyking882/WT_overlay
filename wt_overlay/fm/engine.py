"""WTAPC jet thrust adaptation; see THIRD_PARTY_NOTICES.md.

Only steady full military / full WEP operation is supported. The upstream
calculator itself warns jet thrust can be incorrect; no game-validation claim.
"""
from __future__ import annotations

from bisect import bisect_right
import re

from wt_overlay.contracts import G
from .polar import finite


def _axis(data: dict, prefix: str) -> tuple[list[int], list[float]]:
    indices = sorted(int(k[len(prefix):]) for k in data if re.fullmatch(prefix+r"\d+", k))
    values = [finite(data[f"{prefix}{i}"], prefix) for i in indices]
    if len(values) < 2 or any(b <= a for a, b in zip(values, values[1:])):
        raise ValueError(f"{prefix} must contain at least two strictly increasing knots")
    return indices, values


def _bracket(axis: list[float], value: float) -> tuple[int, float]:
    if not axis[0] <= value <= axis[-1]:
        raise ValueError("outside engine thrust table; extrapolation disabled")
    i = min(max(bisect_right(axis, value)-1, 0), len(axis)-2)
    return i, (value-axis[i])/(axis[i+1]-axis[i])


class JetEngine:
    def __init__(self, main: dict, *, allow_sparse: bool = False):
        if main.get("Type") != "Jet":
            raise ValueError("only tabulated Jet engines supported")
        table = main.get("ThrustMax", {})
        if table.get("VelocityType", "TAS") != "TAS":
            raise ValueError("only TAS engine thrust tables supported")
        hi, self.altitudes = _axis(table, "Altitude_")
        vi, self.velocities_kph = _axis(table, "Velocity_")
        self.base_kgf = finite(table.get("ThrustMax0"), "ThrustMax0")
        self.boost = finite(main.get("AfterburnerBoost", 1.), "AfterburnerBoost")
        if self.base_kgf <= 0 or self.boost <= 0:
            raise ValueError("invalid base thrust / afterburner boost")
        self.grids = []
        for prefix in ("ThrustMaxCoeff", "ThrAftMaxCoeff"):
            grid = [[None if allow_sparse and f"{prefix}_{h}_{v}" not in table
                     else finite(table.get(f"{prefix}_{h}_{v}"), prefix) for v in vi] for h in hi]
            if any(x is not None and x < 0 for row in grid for x in row):
                raise ValueError("negative thrust coefficient")
            self.grids.append(grid)
        modes = []
        for key, value in main.items():
            if re.fullmatch(r"Mode\d+", key):
                modes.append((finite(value.get("Throttle"), key+" throttle"),
                              finite(value.get("ThrustMult"), key+" multiplier")))
        military = [x for x in modes if x[0] <= 1]
        wep = [x[1] for x in modes if x[0] > 1]
        if not military or max(military)[0] != 1:
            raise ValueError("explicit full military mode required")
        self.military = max(military)[1]
        self.has_wep = bool(wep)
        self.wep = max(wep) if wep else self.military
        if min(self.military, self.wep) <= 0:
            raise ValueError("invalid thrust mode multiplier")

    def thrust_n(self, altitude_m: float, tas_mps: float, afterburner: bool) -> float:
        h = finite(altitude_m, "altitude")
        speed = finite(tas_mps, "TAS")*3.6
        i, u = _bracket(self.altitudes, h)
        j, v = _bracket(self.velocities_kph, speed)

        def interp(grid: list[list[float]]) -> float:
            cells = ((grid[i][j], (1-u)*(1-v)), (grid[i][j+1], (1-u)*v),
                     (grid[i+1][j], u*(1-v)), (grid[i+1][j+1], u*v))
            if any(value is None and weight > 0 for value, weight in cells):
                raise ValueError("缺少此高度/速度的推力表节点；不填补或外推")
            return sum(value*weight for value, weight in cells if weight > 0)

        thrust = self.base_kgf*interp(self.grids[0])
        if afterburner and self.has_wep:
            thrust *= self.wep*interp(self.grids[1])*self.boost
        else:
            thrust *= self.military
        return thrust*G
