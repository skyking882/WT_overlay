"""Presentation of telemetry snapshots; no GUI or changes to physical estimates."""

from dataclasses import dataclass
import math

from .contracts import OverlaySnapshot


@dataclass(frozen=True)
class HudRow:
    label: str
    value: str
    unit: str = ""
    tone: str = "normal"


@dataclass(frozen=True)
class HudContent:
    title: str
    rows: tuple[HudRow, ...]
    footer: str = ""


def number(value, digits=0, signed=False, scale=1.0):
    if value is None or not math.isfinite(value):
        return "—"
    return format(value * scale, f"{'+' if signed else ''},.{digits}f")


def contents(snapshot: OverlaySnapshot) -> dict[str, HudContent]:
    state, energy, advice = snapshot.state, snapshot.energy, snapshot.advice
    valid = state is not None and state.valid
    ready = valid and energy is not None and energy.ready
    demo = snapshot.mode == "demo" or (state is not None and state.source == "demo")
    suffix = " · 合成演示" if demo else (" · LIVE" if valid else " · 无有效数据")
    sep = energy.sep_mps if ready else None
    tone = "negative" if sep is not None and sep < 0 else "accent"
    current = advice.current if valid and advice else None
    best = advice.best if valid and advice and advice.available else None
    reason = advice.reason if advice else "加载 FM 并设置参考总质量"
    if current and not current.valid:
        reason = reason or current.reason
    return {
        "flight": HudContent("飞行状态" + suffix, (
            HudRow("真空速", number(state.tas_mps if valid else None, scale=3.6), "km/h"),
            HudRow("表速", number(state.ias_mps if valid else None, scale=3.6), "km/h"),
            HudRow("高度", number(state.altitude_m if valid else None), "m"),
        ), "合成数据 · 非游戏实测" if demo else snapshot.status),
        "energy": HudContent("实际能量" + suffix, (
            HudRow("SEP", number(sep, 1, True), "m/s", tone),
            HudRow("比能高度", number(energy.energy_height_m if valid and energy else None), "m"),
            HudRow("爬升贡献", number(energy.climb_mps if valid and energy else None, 1, True), "m/s"),
            HudRow("动能贡献", number(energy.kinetic_sep_mps if ready else None, 1, True), "m/s"),
        ), "遥测变化率 · 最长 1.2 秒平滑" if ready else "等待有效采样"),
        "reference": HudContent("静态参考" + suffix, (
            HudRow("当前参考 SEP", number(current.sep_mps if current and current.valid else None, 1, True), "m/s"),
            HudRow("采样最佳真空速", number(best.condition.tas_mps if best and best.valid else None, scale=3.6), "km/h"),
            HudRow("采样最高 SEP", number(best.sep_mps if best and best.valid else None, 1, True), "m/s"),
        ), "同高 · 1g · 干净构型 · 未配平\n" + (reason or "未通过游戏验证 · 非全程最优")),
    }


def details(snapshot: OverlaySnapshot) -> str:
    notes = list(snapshot.notes)
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
    return "\n\n".join(dict.fromkeys(notes))
