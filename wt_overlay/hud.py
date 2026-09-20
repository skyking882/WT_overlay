"""Presentation of telemetry snapshots; no GUI or changes to physical estimates."""

from dataclasses import dataclass
import math

from .contracts import OverlaySnapshot


@dataclass(frozen=True)
class Indicator:
    key: str
    group: str
    label: str
    unit: str
    description: str
    enabled: bool = True


INDICATORS = (
    Indicator("g", "flight", "过载", "g", "8111 的机体法向过载 Ny；不作为模型的升力/重量。"),
    Indicator("aoa", "flight", "迎角", "°", "8111 AoA。"),
    Indicator("aos", "flight", "侧滑角", "°", "8111 AoS。"),
    Indicator("pitch", "flight", "俯仰角", "°", "8111 地平仪俯仰读数。"),
    Indicator("roll", "flight", "滚转角", "°", "8111 地平仪滚转读数。"),
    Indicator("heading", "flight", "航向", "°", "8111 罗盘读数。", False),
    Indicator("vy", "flight", "垂直速度", "m/s", "8111 Vy 直接读数；与平滑后的爬升率分开。", False),
    Indicator("ias", "flight", "表速", "km/h", "8111 IAS。", False),
    Indicator("altitude", "flight", "高度", "m", "8111 海拔高度。", False),
    Indicator("mach", "flight", "马赫数", "", "8111 马赫数。", False),
    Indicator("sep", "energy", "SEP", "m/s", "遥测总比能变化率，使用最长 1.2 秒窗口。"),
    Indicator("energy_height", "energy", "比能高度", "m", "海拔高度 + TAS²/(2g)。"),
    Indicator("climb", "energy", "爬升率", "m/s", "高度的平滑变化率，与 SEP 使用同一窗口。"),
    Indicator("kinetic", "energy", "动能变化率", "m/s", "TAS²/(2g) 的变化率；与爬升率相加等于 SEP。"),
    Indicator("acceleration", "energy", "加速度", "m/s²", "TAS 的平滑变化率；不是过载或完整三维加速度。"),
    Indicator("fuel", "engine", "燃油", "kg", "8111 燃油质量，不等于飞机总质量。"),
    Indicator("thrust", "engine", "推力", "kN", "8111 明确标记 N 或 kgf 的各发动机推力之和；缺失或其他单位不猜测。"),
    Indicator("throttle", "engine", "油门 1", "%", "8111 第一台发动机油门。", False),
    Indicator("mass", "engine", "总质量", "kg", "只显示明确的 mass, kg 遥测字段；不使用手动参考质量代替。", False),
    Indicator("reference_sep", "reference", "参考 SEP", "m/s", "当前速度下的静态模型值：同高、1g、干净构型，未配平、未经游戏验证。"),
    Indicator("reference_tas", "reference", "参考峰值 TAS", "km/h", "局部采样中最高 SEP 对应的真空速，不是全程最优爬升速度。"),
    Indicator("reference_peak", "reference", "参考峰值 SEP", "m/s", "局部采样的最高静态 SEP，不代表当前机动的可用 SEP。"),
)
DEFAULT_INDICATORS = frozenset(item.key for item in INDICATORS if item.enabled)


@dataclass(frozen=True)
class HudRow:
    label: str
    value: str
    unit: str = ""
    tone: str = "normal"
    key: str = ""


@dataclass(frozen=True)
class HudContent:
    title: str
    rows: tuple[HudRow, ...]
    demo: bool = False


def number(value, digits=0, signed=False, scale=1.0):
    if value is None or not math.isfinite(value):
        return "—"
    return format(value * scale, f"{'+' if signed else ''},.{digits}f")


def contents(snapshot: OverlaySnapshot, enabled=None) -> dict[str, HudContent]:
    state, energy, advice = snapshot.state, snapshot.energy, snapshot.advice
    valid = state is not None and state.valid
    ready = valid and energy is not None and energy.ready
    demo = snapshot.mode == "demo" or (state is not None and state.source == "demo")
    sep = energy.sep_mps if ready else None
    tone = "negative" if sep is not None and sep < 0 else "accent"
    current = advice.current if valid and advice else None
    best = advice.best if valid and advice and advice.available else None
    def flight(field, digits=1, signed=False, scale=1.0):
        return number(getattr(state, field) if valid else None, digits, signed, scale)

    values = {
        "g": flight("normal_load_g", 1, True), "aoa": flight("aoa_deg", 1, True),
        "aos": flight("aos_deg", 1, True), "pitch": flight("pitch_deg", 1, True),
        "roll": flight("roll_deg", 1, True), "heading": flight("heading_deg", 0),
        "vy": flight("vertical_speed_mps", 1, True), "ias": flight("ias_mps", 0, scale=3.6),
        "altitude": flight("altitude_m", 0), "mach": flight("mach", 2),
        "fuel": flight("fuel_kg", 0), "thrust": flight("thrust_n", 1, scale=0.001),
        "throttle": flight("throttle_percent", 0), "mass": flight("mass_kg", 0),
        "sep": number(sep, 1, True),
        "energy_height": number(energy.energy_height_m if valid and energy else None),
        "climb": number(energy.climb_mps if ready else None, 1, True),
        "kinetic": number(energy.kinetic_sep_mps if ready else None, 1, True),
        "acceleration": number(energy.acceleration_mps2 if ready else None, 2, True),
        "reference_sep": number(current.sep_mps if current and current.valid else None, 1, True),
        "reference_tas": number(best.condition.tas_mps if best and best.valid else None, scale=3.6),
        "reference_peak": number(best.sep_mps if best and best.valid else None, 1, True),
    }
    enabled = DEFAULT_INDICATORS if enabled is None else enabled
    groups = {"flight": "飞行状态", "energy": "实际能量", "engine": "动力与燃油", "reference": "静态参考"}
    return {key: HudContent(title, tuple(
        HudRow(item.label, values[item.key], item.unit, tone if item.key == "sep" else "normal", item.key)
        for item in INDICATORS if item.group == key and item.key in enabled), demo)
        for key, title in groups.items()}


def details(snapshot: OverlaySnapshot) -> str:
    notes = [snapshot.status, *snapshot.notes]
    advice = snapshot.advice
    if advice and advice.reason:
        notes.append(advice.reason)
    for item in (snapshot.state, snapshot.energy, advice,
                 advice.current if advice else None, advice.best if advice else None):
        if item:
            notes.extend(item.notes)
            reason = getattr(item, "reason", "")
            if reason:
                notes.append(reason)
    if advice:
        notes.append("模型未求配平，尚未通过游戏验证；采样最优不等于全程最优爬升。")
    notes.append("三维转向建议：动态模型尚未接入。")
    notes.append("指标定义\n" + "\n".join(f"{item.label}：{item.description}" for item in INDICATORS))
    return "\n\n".join(dict.fromkeys(notes))
