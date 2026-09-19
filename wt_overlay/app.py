"""Join telemetry, measured energy and optional static reference performance."""

from __future__ import annotations

from dataclasses import replace
import math
from queue import Empty, Full, Queue
import re
from threading import Event, Lock, Thread
import time

from .contracts import EnergyMetrics, FlightState, OverlaySnapshot, PerformanceCondition, SEPAdvice
from .demo import make_demo_sample
from .energy import EnergyEstimator
from .fm import load_model
from .planning import scan_sep
from .telemetry import TelemetryClient


def positive(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label}必须是正数")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label}必须是有限正数")
    return float(value)


def _aircraft_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


class OverlayController:
    """One worker owns mutable calculation state; the GUI only copies snapshots."""

    def __init__(self, *, mode: str = "live", base_url: str = "http://127.0.0.1:8111",
                 model_path: str | None = None, mass_kg: float | None = None,
                 afterburner: bool = True, interval_s: float = 0.1, client=None):
        if mode not in ("live", "demo"):
            raise ValueError("模式必须为 live 或 demo")
        if type(afterburner) is not bool:
            raise ValueError("加力选项必须为布尔值")
        self.interval = positive(interval_s, "采样间隔")
        if not 0.05 <= self.interval <= 0.5:
            raise ValueError("采样间隔应在 0.05–0.5 秒内")
        self.mode = mode
        self.mass_kg = positive(mass_kg, "总质量") if mass_kg is not None else None
        self.afterburner = afterburner
        self.client = client if client is not None else TelemetryClient(base_url)
        self.model = load_model(model_path) if model_path else None
        self.estimator = EnergyEstimator()
        self._commands: Queue[dict] = Queue(maxsize=32)
        self._lock = Lock()
        self._stop = Event()
        self._thread: Thread | None = None
        self._started_at = time.monotonic()
        self._published_at: float | None = None
        self._last_prediction = -math.inf
        self._advice: SEPAdvice | None = None
        self._identity = None
        self._settings_error = ""
        self._snapshot = OverlaySnapshot(mode, "等待首个数据样本", mass_override_kg=self.mass_kg,
                                         afterburner=afterburner)

    def submit(self, command: dict) -> None:
        """Validate and queue UI intent without filesystem/network work on the GUI thread."""
        if not isinstance(command, dict):
            raise ValueError("设置命令无效")
        action = command.get("action")
        if action == "mode":
            if command.get("value") not in ("live", "demo"):
                raise ValueError("模式必须为 live 或 demo")
        elif action == "mass":
            positive(command.get("kg"), "总质量")
        elif action == "afterburner":
            if type(command.get("enabled")) is not bool:
                raise ValueError("加力选项必须为布尔值")
        elif action == "model":
            if not isinstance(command.get("path"), str) or not command["path"].strip():
                raise ValueError("请选择 FM 文件")
        else:
            raise ValueError("未知设置命令")
        try:
            self._commands.put_nowait(dict(command))
        except Full as exc:
            raise ValueError("设置更新中，请稍后再试") from exc

    def _apply_commands(self) -> None:
        while True:
            try:
                command = self._commands.get_nowait()
            except Empty:
                return
            self._settings_error = ""
            self._advice = None
            self._last_prediction = -math.inf
            action = command["action"]
            if action == "mode":
                self.mode = command["value"]
                self.estimator.reset()
                self._identity = None
            elif action == "mass":
                self.mass_kg = float(command["kg"])
            elif action == "afterburner":
                self.afterburner = command["enabled"]
            elif action == "model":
                self.model = None
                try:
                    self.model = load_model(command["path"])
                except (OSError, ValueError) as exc:
                    self._settings_error = f"FM 未加载：{exc}"

    def _predict(self, state: FlightState) -> SEPAdvice | None:
        if self.model is None:
            return None
        if self.mode == "live":
            if not state.aircraft_id:
                return SEPAdvice(False, reason="未能确认本机机型，暂不套用所选 FM")
            if _aircraft_key(state.aircraft_id) != _aircraft_key(self.model.info.aircraft_id):
                return SEPAdvice(False, reason="当前机型与所选 FM 不符，模型预测已停用")
        mass = self.mass_kg if self.mass_kg is not None else state.mass_kg
        if mass is None:
            return SEPAdvice(False, reason="请在设置中输入参考总质量；不从燃油量推算")
        condition = PerformanceCondition(state.altitude_m, state.tas_mps, mass,
                                         afterburner=self.afterburner)
        advice = scan_sep(self.model, condition)
        scope = (f"静态参考：同高度、1g、干净构型、总质量 {mass:,.0f} kg、"
                 f"{'全加力' if self.afterburner else '全军推'}；不代表当前操纵状态。")
        if self.mode == "demo":
            scope += " 演示状态仅用于驱动界面，不能用于验证所选飞机。"
        return replace(advice, notes=(scope, *advice.notes))

    def tick(self, now: float | None = None) -> OverlaySnapshot:
        """Perform one sampling cycle, on the worker (or synchronously in tests)."""
        self._apply_commands()
        now = time.monotonic() if now is None else now
        state = (make_demo_sample(now-self._started_at) if self.mode == "demo"
                 else self.client.poll(time_s=now))
        energy = self.estimator.update(state)
        identity = (state.source, state.aircraft_id)
        if identity != self._identity:
            self._advice = None
            self._last_prediction = -math.inf
            self._identity = identity
        if not state.valid:
            self._advice = None
            self._last_prediction = -math.inf
        elif now-self._last_prediction >= 1.0:
            self._advice = self._predict(state)
            self._last_prediction = now
        if self.mode == "demo":
            status = "合成演示 · 未连接游戏"
        elif state.valid:
            status = "8111 已连接 · 能量速率为滚动窗口估计"
        else:
            status = "等待游戏飞行数据 · 请检查 8111 与当前飞行状态"
        notes = ["实际能量速率使用同一段最长 1.2 秒的遥测窗口。",
                 "模型区域是同高、1g、干净构型参考，每秒更新；不是当前机动的可用 SEP。"]
        if self.mass_kg is not None:
            notes.append(f"参考总质量由手动指定为 {self.mass_kg:,.0f} kg，不随燃油消耗更新。")
        if self._settings_error:
            status = self._settings_error
            notes.append(self._settings_error)
        snapshot = OverlaySnapshot(
            self.mode, status, state, energy, self._advice,
            self.model.info.name if self.model else "未加载 FM",
            tuple(notes), self.mass_kg, self.afterburner)
        with self._lock:
            self._snapshot = snapshot
            self._published_at = time.monotonic()
        return snapshot

    def get_snapshot(self) -> OverlaySnapshot:
        with self._lock:
            snapshot, published = self._snapshot, self._published_at
        if (published is not None and time.monotonic()-published > 2.5
                and snapshot.state is not None and snapshot.state.valid):
            return replace(snapshot, status="数据已过期，等待重新连接",
                           state=replace(snapshot.state, valid=False), advice=None,
                           energy=EnergyMetrics(snapshot.state.time_s, notes=("样本已过期",)))
        return snapshot

    def _run(self) -> None:
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                self.tick(started)
            except Exception as exc:
                self.estimator.reset()
                self._advice = None
                self._last_prediction = -math.inf
                with self._lock:
                    self._snapshot = OverlaySnapshot(
                        self.mode, f"采样失败：{type(exc).__name__}: {exc}",
                        notes=("当前数据不可用；没有切换到演示数据。",),
                        mass_override_kg=self.mass_kg, afterburner=self.afterburner)
            self._stop.wait(max(0.0, self.interval-(time.monotonic()-started)))

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = Thread(target=self._run, name="wt-telemetry", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
