"""Source-backed static FM research model. No complete/game-validated FM claim."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from math import cos, isfinite, radians, sqrt
from pathlib import Path

from wt_overlay.contracts import G, ModelInfo, PerformanceCondition, PerformancePoint
from .engine import JetEngine
from .polar import PolarProperties, finite

FM_REVISION = "60345f1697efeb0fa54256b2f0aab3d83d7e708e"
FM_SOURCE = ("https://github.com/gszabi99/War-Thunder-Datamine/blob/"+FM_REVISION
             +"/aces.vromfs.bin_u/gamedata/flightmodels/fm/su_27sm.blkx")
_SAMPLE_SHA256 = "6dfd1fc07e3ab930352032412db6f0623431b8ece08151f9f7307fb38e9dd2ec"

LIMITATIONS = (
    "研究近似，未经过游戏验证；仅支持所附固定版本苏-27SM，未覆盖全部目标飞机。",
    "各部件共用标准大气来流；安装角直接加至迎角；中立舵面，零侧滑、零转动率；未模拟下洗与流动惯性。",
    "部件面积按各自 Areas 汇总，机翼仅取六个翼段；部件载荷视为同一风轴，整机调用语义尚未验证。",
    "仅干净构型；不计武器/挂架、损伤、改装、起落架、襟翼、减速板及额外整机阻力项。",
    "不求力矩配平；未提供迎角时仅求 L = n·m·g，n 定义为 L/W；未施加完整法向运动平衡。",
    "推力沿机身纵轴近似，忽略喷口安装偏角；WTAPC 推力算法作者提示喷气推力可能不准确。",
    "仅稳定全军推/全加力；无部分油门、发动机瞬态、操纵限制、力矩或三维机动预测。",
)


def atmosphere(altitude_m: float) -> tuple[float, float]:
    """ISA geopotential altitude approximation, 0–20 km; returns density, sound speed.

    WT map weather is not inferred from telemetry. This standard atmosphere is
    an explicit project assumption, not a recovered WT atmosphere model.
    """
    h = finite(altitude_m, "altitude")
    if not 0 <= h <= 20000:
        raise ValueError("static atmosphere supports 0–20000 m only")
    r = 287.05287
    if h <= 11000:
        t = 288.15-.0065*h
        p = 101325*(t/288.15)**(G/(r*.0065))
    else:
        from math import exp
        t = 216.65
        p = 22632.040095*exp(-G*(h-11000)/(r*t))
    return p/(r*t), sqrt(1.4*r*t)


@dataclass(frozen=True)
class Component:
    name: str
    area_m2: float
    incidence_deg: float
    properties: PolarProperties
    vertical: bool = False


class StaticModel:
    def __init__(self, raw: dict):
        self.info = ModelInfo("su_27sm", "苏-27SM · 静态研究近似", FM_SOURCE,
                              FM_REVISION, False, LIMITATIONS)
        aero = raw["Aerodynamics"]
        components = []
        for name in ("WingPlane", "FuselagePlane", "HorStabPlane", "VerStabPlane"):
            block = aero[name]
            areas = block["Areas"]
            if name == "WingPlane":
                area = sum(areas[k] for k in ("LeftIn", "LeftMid", "LeftOut", "RightIn", "RightMid", "RightOut"))
                polar = block["FlapsPolar0"]
            else:
                area = sum(areas.values())
                polar = block["Polar"]
            components.append(Component(name, area, block["Angle"],
                PolarProperties.from_mapping(polar, block["Span"], area), name == "VerStabPlane"))
        self.components = tuple(components)
        self.engine = JetEngine(raw["EngineType0"]["Main"])
        self.engine_count = 2  # Pinned profile contains two Type 0 engine instances.

    def evaluate(self, condition: PerformanceCondition) -> PerformancePoint:
        """Return invalid with a reason for unsupported conditions, never silent fallback."""
        try:
            c = condition
            for name in ("altitude_m", "tas_mps", "mass_kg", "throttle", "load_factor",
                         "flight_path_deg", "flap_fraction", "gear_fraction", "airbrake_fraction"):
                finite(getattr(c, name), name)
            if c.mass_kg <= 0 or c.tas_mps <= 0:
                raise ValueError("mass and TAS must be positive")
            if not 0 < c.load_factor <= 20:
                raise ValueError("load factor must be in (0, 20]; this is a query bound, not an aircraft limit")
            if not -89 <= c.flight_path_deg <= 89:
                raise ValueError("flight path must be within [-89, 89] degrees")
            if type(c.afterburner) is not bool:
                raise ValueError("afterburner must be boolean")
            if c.throttle != 1.:
                raise ValueError("部分油门未实现；仅支持 throttle=1 的全军推/全加力")
            if any(getattr(c, k) != 0 for k in ("flap_fraction", "gear_fraction", "airbrake_fraction")):
                raise ValueError("仅支持干净构型：襟翼、起落架、减速板均为 0")
            density, sound_speed = atmosphere(c.altitude_m)
            mach = c.tas_mps/sound_speed
            if mach > 2.35:
                raise ValueError("超过此研究模型 Mach 2.35 查询边界")
            thrust = self.engine_count*self.engine.thrust_n(c.altitude_m, c.tas_mps, c.afterburner)
            polars = [(part, part.properties.at_mach(mach)) for part in self.components]
            q = .5*density*c.tas_mps**2

            def forces(aoa: float) -> tuple[float, float]:
                drag = lift = 0.
                for part, polar in polars:
                    effective = part.incidence_deg if part.vertical else aoa+part.incidence_deg
                    cd, cl = polar.coefficients(effective)
                    drag += q*part.area_m2*cd
                    if not part.vertical:
                        lift += q*part.area_m2*cl
                return drag, lift

            notes = list(LIMITATIONS)
            if c.aoa_deg is None:
                # Search only the common precritical interval, then bracket a root.
                low = max(p.critical_aoa_low-part.incidence_deg for part, p in polars if not part.vertical)
                high = min(p.critical_aoa_high-part.incidence_deg for part, p in polars if not part.vertical)
                target = c.load_factor*c.mass_kg*G
                samples = [(low+(high-low)*i/128) for i in range(129)]
                bracket = None
                a, fa = samples[0], forces(samples[0])[1]-target
                for b in samples[1:]:
                    fb = forces(b)[1]-target
                    if fa <= 0 <= fb:
                        bracket = (a, b)
                        break
                    a, fa = b, fb
                if bracket is None:
                    raise ValueError("升力需求超出共同失速前求解区间；不外推迎角")
                a, b = bracket
                for _ in range(45):
                    mid = (a+b)/2
                    if forces(mid)[1] < target:
                        a = mid
                    else:
                        b = mid
                aoa = (a+b)/2
                notes.append("迎角由指定升力需求求得；不是纵向配平。")
            else:
                aoa = finite(c.aoa_deg, "AoA")
                if not -30 <= aoa <= 40:
                    raise ValueError("整机静态近似仅接受 [-30, 40] 度迎角")
                notes.append("指定迎角的瞬时受力；未强制满足升力需求。")
            drag, lift = forces(aoa)
            sep = (thrust*cos(radians(aoa))-drag)*c.tas_mps/(c.mass_kg*G)
            if not all(isfinite(x) for x in (drag, lift, thrust, sep)) or drag < 0:
                raise ValueError("invalid calculated force / SEP")
            return PerformancePoint(c, True, thrust, drag, lift, aoa, sep, notes=tuple(notes))
        except (ValueError, OverflowError) as exc:
            return PerformancePoint(condition, False, reason=str(exc), notes=LIMITATIONS)


def load_model(path: str | Path) -> StaticModel:
    """Load only the audited, unmodified sample; other profiles need schema review.

    A strict content gate prevents silently applying Su-27-specific assumptions
    to another aircraft, changed FM, variable-sweep aircraft or mixed engines.
    """
    content = Path(path).read_bytes()
    if hashlib.sha256(content).hexdigest() != _SAMPLE_SHA256:
        raise ValueError("尚未支持此 FM 文件或版本；当前只接受所附固定版本 su_27sm.blkx")
    raw = json.loads(content)
    return StaticModel(raw)


__all__ = ["load_model", "StaticModel", "atmosphere"]
