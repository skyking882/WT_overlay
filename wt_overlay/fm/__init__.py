"""Source-backed static FM research model. No complete/game-validated FM claim."""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import re
from math import cos, isfinite, radians, sqrt
from pathlib import Path

from wt_overlay.contracts import G, ModelInfo, PerformanceCondition, PerformancePoint
from .engine import JetEngine
from .polar import PolarProperties, finite
from .catalog import Aircraft, aircraft_catalog, aircraft_key, find_aircraft

FM_REVISION = "60345f1697efeb0fa54256b2f0aab3d83d7e708e"
FM_SOURCE = ("https://github.com/gszabi99/War-Thunder-Datamine/blob/"+FM_REVISION
             +"/aces.vromfs.bin_u/gamedata/flightmodels/fm/su_27sm.blkx")
_SAMPLE_SHA256 = "6dfd1fc07e3ab930352032412db6f0623431b8ece08151f9f7307fb38e9dd2ec"

LIMITATIONS = (
    "静态研究近似，未经过游戏验证；仅接受机型目录内固定版本的 FM。",
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
    def __init__(self, raw: dict, info: ModelInfo | None = None,
                 aircraft_ids: tuple[str, ...] = ()):
        self.info = info or ModelInfo("su_27sm", "苏-27SM", FM_SOURCE,
                                     FM_REVISION, False, LIMITATIONS)
        self.aircraft_ids = aircraft_ids or (self.info.aircraft_id,)
        aero = raw["Aerodynamics"]
        self.legacy_geometry = "NoFlaps" in aero and "WingPlane" not in aero
        if self.legacy_geometry:
            aero = _legacy_components(raw)
        wings = [(float(block["Sweep"]), _component("WingPlane", block))
                 for name, block in aero.items() if re.fullmatch(r"WingPlaneSweep\d+", name)]
        self.wings = tuple(sorted(wings, key=lambda item: item[0]))
        wing = self.wings[0][1] if self.wings else _component("WingPlane", aero["WingPlane"])
        self.components = (wing, *(_component(name, aero[name]) for name in
                                  ("FuselagePlane", "HorStabPlane", "VerStabPlane")))
        engines = []
        for name, instance in raw.items():
            if not re.fullmatch(r"Engine\d+", name):
                continue
            engine_type = raw[f"EngineType{instance['Type']}"]
            controls = dict(engine_type.get("Controls", {}), **instance.get("Controls", {}))
            # Lift engines whose throttle is zero with VTOL retracted are off
            # in the forward-flight reference (Yak-141 Type 1).
            limit = controls.get("vtolToThrottleLim0")
            if controls.get("hasVtolControl") and limit and limit[1] == 0:
                continue
            main = dict(engine_type["Main"], **instance.get("Main", {}))
            engines.append(JetEngine(main, allow_sparse=True))
        if not engines:
            raise ValueError("no forward-flight jet engines")
        self.engines = tuple(engines)
        self.engine, self.engine_count = engines[0], len(engines)

    def matches_aircraft(self, identity: str | None) -> bool:
        return bool(identity and any(aircraft_key(identity) == aircraft_key(x) for x in self.aircraft_ids))

    def components_at_sweep(self, fraction: float) -> tuple[Component, ...]:
        if not self.wings:
            return self.components
        if not self.wings[0][0] <= fraction <= self.wings[-1][0]:
            raise ValueError("后掠设置超出 FM 范围")
        for (a, left), (b, right) in zip(self.wings, self.wings[1:]):
            if a <= fraction <= b:
                weight = (fraction-a)/(b-a)
                # Interpolate component forces, preserving each endpoint's AR.
                blend = tuple(replace(part, area_m2=part.area_m2*w)
                              for part, w in ((left, 1-weight), (right, weight)) if w > 0)
                return (*blend, *self.components[1:])
        return self.components

    def evaluate(self, condition: PerformanceCondition) -> PerformancePoint:
        """Return invalid with a reason for unsupported conditions, never silent fallback."""
        try:
            c = condition
            for name in ("altitude_m", "tas_mps", "mass_kg", "throttle", "load_factor",
                         "flight_path_deg", "flap_fraction", "gear_fraction", "airbrake_fraction",
                         "sweep_fraction"):
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
            if not 0 <= c.sweep_fraction <= 1:
                raise ValueError("后掠设置必须在 0–1 之间")
            if any(getattr(c, k) != 0 for k in ("flap_fraction", "gear_fraction", "airbrake_fraction")):
                raise ValueError("仅支持干净构型：襟翼、起落架、减速板均为 0")
            density, sound_speed = atmosphere(c.altitude_m)
            mach = c.tas_mps/sound_speed
            if mach > 2.35:
                raise ValueError("超过此研究模型 Mach 2.35 查询边界")
            thrust = sum(engine.thrust_n(c.altitude_m, c.tas_mps, c.afterburner) for engine in self.engines)
            polars = [(part, part.properties.at_mach(mach)) for part in self.components_at_sweep(c.sweep_fraction)]
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
            if self.legacy_geometry:
                notes.append("旧版 FM 机身暂以机翼面积/翼展为参考，尾翼按显式面积/尺寸组装；参考尺度未经游戏验证。")
            if self.wings:
                notes.append(f"固定参考后掠 {c.sweep_fraction:.0%}；端点受力线性插值，未复刻自动后掠控制。")
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


def _component(name: str, block: dict) -> Component:
    areas = block["Areas"]
    wing = name == "WingPlane"
    area = sum(areas[k] for k in ("LeftIn", "LeftMid", "LeftOut", "RightIn", "RightMid", "RightOut")) if wing else sum(areas.values())
    return Component(name, area, block["Angle"], PolarProperties.from_mapping(
        block["FlapsPolar0" if wing else "Polar"], block["Span"], area), name == "VerStabPlane")


def _legacy_components(raw: dict) -> dict:
    """Project reference convention for pre-component FM layouts; see README.

    Old FMs do not state a separate fuselage reference area/span. Using the
    wing reference is an explicit approximation, not recovered game semantics.
    """
    aero, areas = raw["Aerodynamics"], raw["Areas"]
    wing_areas = {k: areas["Wing"+k] for k in
                  ("LeftIn", "LeftMid", "LeftOut", "RightIn", "RightMid", "RightOut")}
    def plane(span, area, angle, polar):
        return {"Span": span, "Areas": {"Main": area}, "Angle": angle, "Polar": polar}
    return {
        "WingPlane": {"Span": raw["Wingspan"], "Areas": wing_areas,
                      "Angle": raw["WingAngle"], "FlapsPolar0": dict(aero, **aero["NoFlaps"])},
        "FuselagePlane": plane(raw["Wingspan"], sum(wing_areas.values()), 0., aero["Fuselage"]),
        "HorStabPlane": plane(raw["StabWidth"], areas["Stabilizer"]+areas["Elevator"], raw["StabAngle"], aero["Stab"]),
        "VerStabPlane": plane(raw["FinHeight"], areas["Keel"]+areas["Rudder"], raw["KeelAngle"], aero["Fin"]),
    }


def _catalog_model(content: bytes, profile: Aircraft, identities: tuple[str, ...]) -> StaticModel:
    if hashlib.sha256(content).hexdigest() != profile.sha256:
        raise ValueError(f"FM 文件校验失败：{profile.id}")
    info = ModelInfo(profile.id, profile.name, profile.source_url, profile.revision, False, LIMITATIONS)
    return StaticModel(json.loads(content), info, identities)


def load_aircraft(identity: str) -> StaticModel:
    profile = find_aircraft(identity)
    if profile is None:
        raise ValueError(f"机型不在目录中：{identity}")
    return _catalog_model(profile.path.read_bytes(), profile, (profile.id,))


def load_model(path: str | Path) -> StaticModel:
    """Content-verified catalog FM or the original Su-27SM regression sample."""
    content = Path(path).read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if digest == _SAMPLE_SHA256:
        return StaticModel(json.loads(content))
    profiles = [p for p in aircraft_catalog() if p.sha256 == digest]
    if not profiles:
        raise ValueError("尚未支持此 FM 文件或版本；请选择附带机型目录中的固定版本")
    profile = next((p for p in profiles if p.id == Path(path).stem), profiles[0])
    return _catalog_model(content, profile, tuple(p.id for p in profiles))


__all__ = ["load_model", "load_aircraft", "StaticModel", "atmosphere"]
