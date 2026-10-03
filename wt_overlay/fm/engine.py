"""WTAPC jet thrust adaptation; see THIRD_PARTY_NOTICES.md.

Full military / full WEP follow WTAPC. Partial throttle is a project
interpolation of the FM Mode table (see thrust_at_throttle). The upstream
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
        wep = [x for x in modes if x[0] > 1]
        if not military or max(military)[0] != 1:
            raise ValueError("explicit full military mode required")
        self.military = max(military)[1]
        self.has_wep = bool(wep)
        # Full WEP is the strongest mode; its throttle ends the afterburner ramp.
        self.wep_throttle, self.wep = max(wep, key=lambda x: x[1]) if wep else (1., self.military)
        if min(self.military, self.wep) <= 0:
            raise ValueError("invalid thrust mode multiplier")
        # Reverse-thrust modes (negative throttle) are outside forward flight.
        self.dry_modes = sorted(x for x in military if x[0] >= 0)

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

    def military_fraction(self, throttle_percent: float) -> float:
        """Mode ThrustMult at a dry throttle, relative to full military (piecewise linear)."""
        t = finite(throttle_percent, "throttle")/100
        modes = self.dry_modes
        if not modes[0][0] <= t <= 1:
            raise ValueError("油门超出 FM 油门模式表；不外推")
        i = min(max(bisect_right([x[0] for x in modes], t)-1, 0), len(modes)-2)
        (a, fa), (b, fb) = modes[i], modes[i+1]
        return (fa+(fb-fa)*(t-a)/(b-a))/self.military

    def blend(self, military_n: float, maximum_n: float, throttle_percent: float) -> float:
        """Steady thrust at a game throttle percent from full military / full WEP thrust.

        Dry range follows the Mode table; above 100 % thrust ramps linearly to
        full WEP at its mode throttle. Neither law is validated in game, and
        spool transients are not modelled.
        """
        t = finite(throttle_percent, "throttle")
        if not 0 <= t <= 110:
            raise ValueError("油门须在 0–110% 之间")
        if t <= 100:
            return military_n*self.military_fraction(t)
        if not self.has_wep:
            return military_n
        return military_n+(maximum_n-military_n)*min(1., (t-100)/(100*self.wep_throttle-100))

    def thrust_at_throttle(self, altitude_m: float, tas_mps: float, throttle_percent: float) -> float:
        military = self.thrust_n(altitude_m, tas_mps, False)
        if throttle_percent <= 100 or not self.has_wep:
            return self.blend(military, military, throttle_percent)
        return self.blend(military, self.thrust_n(altitude_m, tas_mps, True), throttle_percent)
