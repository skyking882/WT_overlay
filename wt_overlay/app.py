"""Join telemetry, measured energy and optional static reference performance."""

from __future__ import annotations

from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import math
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
import time

from .contracts import (G, ClimbGuidance, ClimbRequest, EnergyMetrics, FlightState,
                        KeyboardTurnGuidance, KeyboardTurnSettings,
                        OverlaySnapshot, PerformanceCondition, SEPAdvice)
from .climb import (ClimbDirector, PlanningCancelled, PlanningUnavailable,
                    build_climb_plan, valid_number, validate_request)
from .demo import make_demo_sample
from .energy import EnergyEstimator
from .fm import load_aircraft, load_model
from .fm.catalog import aircraft_key, find_aircraft
from .planning import scan_sep
from .telemetry import TelemetryClient
from .turn import TurnSession, validate_settings
from .attitude import AttitudeEstimator


def positive(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label}必须是正数")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label}必须是有限正数")
    return float(value)


def _matches(model, identity: str | None) -> bool:
    if hasattr(model, "matches_aircraft"):
        return model.matches_aircraft(identity)
    return bool(identity and aircraft_key(identity) == aircraft_key(model.info.aircraft_id))


class OverlayController:
    """One worker owns mutable calculation state; the GUI only copies snapshots."""

    def __init__(self, *, mode: str = "live", base_url: str = "http://127.0.0.1:8111",
                 model_path: str | None = None, mass_kg: float | None = None,
                 afterburner: bool = True, interval_s: float = 0.1, client=None,
                 aircraft: str | None = None, sweep_fraction: float = 0.):
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
        if model_path and aircraft:
            raise ValueError("请选择机型或 FM 文件其中一种")
        self.model_selection = "file" if model_path else aircraft or "auto"
        self.model = (load_model(model_path) if model_path else
                      load_aircraft(aircraft) if aircraft and aircraft != "auto" else None)
        if self.model is not None and self.model_selection != "file":
            self.model_selection = self.model.info.aircraft_id
        if not valid_number(sweep_fraction) or not 0 <= sweep_fraction <= 1:
            raise ValueError("参考后掠必须在 0–100% 之间")
        self.sweep_fraction = float(sweep_fraction)
        self._auto_identity = None
        self._auto_retry_at = 0.
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
        self.climb_enabled = False
        self.climb_request = ClimbRequest()
        self._director = ClimbDirector()
        self._planner = None
        self._plan_future = None
        self._plan_cancel = None
        self._plan = None
        self._plan_base = None
        self._failed_energy = None
        self.turn_enabled = False
        self.turn_settings = KeyboardTurnSettings()
        self._turn_session = TurnSession()
        self._attitude = AttitudeEstimator()
        self._pose_calibration_sign = None
        self._snapshot = OverlaySnapshot(mode, "等待首个数据样本", mass_override_kg=self.mass_kg,
                                         afterburner=afterburner, model_selection=self.model_selection,
                                         sweep_fraction=self.sweep_fraction,
                                         variable_sweep=bool(self.model and self.model.wings))

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
        elif action == "aircraft":
            value = command.get("id")
            if not isinstance(value, str) or (value != "auto" and find_aircraft(value) is None):
                raise ValueError("请选择目录内的机型")
        elif action == "sweep":
            value = command.get("fraction")
            if not valid_number(value) or not 0 <= value <= 1:
                raise ValueError("参考后掠必须在 0–100% 之间")
        elif action == "climb_enabled":
            if type(command.get("enabled")) is not bool:
                raise ValueError("爬升开关须为布尔值")
        elif action == "climb_target":
            validate_request(ClimbRequest(command.get("altitude_m"), command.get("minimum_tas_mps")))
        elif action == "turn_enabled":
            if type(command.get("enabled")) is not bool:
                raise ValueError("转向开关须为布尔值")
        elif action == "turn_target":
            try:
                validate_settings(KeyboardTurnSettings(**command["settings"]))
            except (TypeError, KeyError) as exc:
                raise ValueError("转向设置无效") from exc
        elif action == "turn_restart":
            pass
        elif action == "pose_calibrate":
            if type(command.get("roll_sign")) is not int or command["roll_sign"] not in (-1, 1):
                raise ValueError("请选择滚转率方向")
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
            action = command["action"]
            if command["action"] in ("mode", "mass", "afterburner", "sweep", "model", "aircraft"):
                self._turn_session.invalidate()
            if (not action.startswith("turn_") and action != "pose_calibrate"
                    or action == "turn_enabled" and command["enabled"]):
                self._reset_climb()
                self._advice = None
                self._last_prediction = -math.inf
            if action == "mode":
                self._attitude.reset()
                self.mode = command["value"]
                self.estimator.reset()
                self._identity = None
                self._auto_identity = None
                self._auto_retry_at = 0.
                if self.model_selection == "auto":
                    self.model = None
            elif action == "mass":
                self.mass_kg = float(command["kg"])
            elif action == "afterburner":
                self.afterburner = command["enabled"]
            elif action == "sweep":
                self.sweep_fraction = float(command["fraction"])
            elif action == "climb_enabled":
                self.climb_enabled = command["enabled"]
                if self.climb_enabled:
                    self.turn_enabled = False
                    self._turn_session.reset()
            elif action == "climb_target":
                self.climb_request = ClimbRequest(command["altitude_m"], command.get("minimum_tas_mps"))
            elif action == "turn_target":
                settings = KeyboardTurnSettings(**command["settings"])
                if settings != self.turn_settings:
                    self._turn_session.reset()
                self.turn_settings = settings
            elif action == "turn_restart":
                self._turn_session.reset()
            elif action == "pose_calibrate":
                self._pose_calibration_sign = command["roll_sign"]
                self._turn_session.reset()
            elif action == "turn_enabled":
                self.turn_enabled = command["enabled"]
                self._turn_session.reset()
                if self.turn_enabled:
                    self.climb_enabled = False
            elif action == "model":
                self.model = None
                self.model_selection = "file"
                try:
                    self.model = load_model(command["path"])
                except (OSError, ValueError) as exc:
                    self._settings_error = f"FM 未加载：{exc}"
            elif action == "aircraft":
                self.model = None
                self.model_selection = command["id"]
                self._auto_identity = None
                self._auto_retry_at = 0.
                if self.model_selection != "auto":
                    try:
                        self.model = load_aircraft(self.model_selection)
                        self.model_selection = self.model.info.aircraft_id
                    except (OSError, ValueError) as exc:
                        self._settings_error = f"FM 未加载：{exc}"

    def _select_live_aircraft(self, state: FlightState) -> None:
        if self.model_selection != "auto" or self.mode != "live":
            return
        # An interrupted poll is not an aircraft change. Keep the loaded FM,
        # while prediction/director gates below suppress use of invalid data.
        if not state.valid or not state.aircraft_id:
            return
        identity = aircraft_key(state.aircraft_id)
        profile = find_aircraft(identity)
        if identity == self._auto_identity and (self.model is not None or profile is None
                                               or state.time_s < self._auto_retry_at):
            return
        self._auto_identity = identity
        self._reset_climb()
        self._advice = None
        self._last_prediction = -math.inf
        self.model = None
        self._settings_error = ""
        if profile:
            try:
                self.model = load_aircraft(profile.id)
            except (OSError, ValueError) as exc:
                self._settings_error = f"FM 未加载：{exc}"
                self._auto_retry_at = state.time_s+2.
        else:
            self._settings_error = f"未找到机型：{state.aircraft_id}"

    def _reset_climb(self):
        if self._plan_cancel is not None:
            self._plan_cancel.set()
        if self._plan_future is not None:
            self._plan_future.cancel()
        self._plan_future = self._plan_cancel = self._plan = self._plan_base = None
        self._failed_energy = None
        self._director.reset()

    def _climb_guidance(self, state, energy):
        if not self.climb_enabled:
            return None
        if not state.valid or not all(valid_number(v) for v in (state.altitude_m, state.tas_mps)):
            self._reset_climb()
            return ClimbGuidance(phase="等待数据")
        if self.model is None:
            return ClimbGuidance(phase="选择 FM")
        if self.mode == "live" and not _matches(self.model, state.aircraft_id):
            return ClimbGuidance(phase="核对机型")
        mass = self.mass_kg if self.mass_kg is not None else state.mass_kg
        if not valid_number(mass) or mass <= 0:
            self._reset_climb()
            return ClimbGuidance(phase="设置质量")
        base = PerformanceCondition(state.altitude_m, state.tas_mps, mass, afterburner=self.afterburner,
                                    sweep_fraction=self.sweep_fraction)
        if self._plan_base is not None and abs(mass / self._plan_base.mass_kg - 1) > .01:
            self._reset_climb()
        current = self.model.evaluate(base)
        if not current.valid or not valid_number(current.sep_mps):
            self._director.reset()
            return ClimbGuidance(phase="超出范围")
        es = state.altitude_m + state.tas_mps ** 2 / (2 * G)
        if self._failed_energy is not None:
            if abs(es - self._failed_energy) < 500:
                return ClimbGuidance(phase="无法规划")
            self._reset_climb()
        if self._plan_future is not None and self._plan_future.done():
            future, self._plan_future = self._plan_future, None
            try:
                self._plan = future.result()
            except (PlanningUnavailable, PlanningCancelled, ValueError):
                self._failed_energy = es
                return ClimbGuidance(phase="无法规划")
        if self._plan is None:
            if self._plan_future is None:
                if self._planner is None:
                    self._planner = ThreadPoolExecutor(max_workers=1, thread_name_prefix="wt-climb")
                self._plan_cancel = Event()
                self._plan_base = base
                self._plan_future = self._planner.submit(
                    build_climb_plan, self.model, base, self.climb_request, self._plan_cancel)
            return ClimbGuidance(phase="计算")
        try:
            speed, _ = self._plan.reference(es)
        except PlanningUnavailable:
            self._reset_climb()
            return ClimbGuidance(phase="重新规划")
        reference = self.model.evaluate(replace(base, altitude_m=max(0, es-speed**2/(2*G)), tas_mps=speed))
        if not reference.valid:
            self._director.reset()
            return ClimbGuidance(phase="超出范围")
        return self._director.update(self._plan, state, energy, current.sep_mps)

    def _predict(self, state: FlightState) -> SEPAdvice | None:
        if self.model is None:
            return None
        if self.mode == "live":
            if not state.aircraft_id:
                return SEPAdvice(False, reason="未能确认本机机型，暂不套用所选 FM")
            if not _matches(self.model, state.aircraft_id):
                return SEPAdvice(False, reason="当前机型与所选 FM 不符，模型预测已停用")
        mass = self.mass_kg if self.mass_kg is not None else state.mass_kg
        if mass is None:
            return SEPAdvice(False, reason="请在设置中输入参考总质量；不从燃油量推算")
        condition = PerformanceCondition(state.altitude_m, state.tas_mps, mass,
                                         afterburner=self.afterburner, sweep_fraction=self.sweep_fraction)
        advice = scan_sep(self.model, condition)
        scope = (f"静态参考：同高度、1g、干净构型、总质量 {mass:,.0f} kg、"
                 f"{'全加力' if self.afterburner else '全军推'}；不代表当前操纵状态。")
        if self.mode == "demo":
            scope += " 演示状态仅用于驱动界面，不能用于验证所选飞机。"
        return replace(advice, notes=(scope, *advice.notes))

    def _turn_guidance(self, state):
        if not self.turn_enabled:
            return None
        if not state.valid:
            return self._turn_session.pause("等待有效飞行数据")
        if self.model is None:
            return self._turn_session.pause(self._settings_error or "等待匹配的机型", "选择机型")
        if self.mode == "live" and not _matches(self.model, state.aircraft_id):
            return self._turn_session.pause("等待游戏机型与所选模型匹配", "核对机型")
        mass = self.mass_kg if self.mass_kg is not None else state.mass_kg
        if not valid_number(mass) or mass <= 0:
            return self._turn_session.pause("请设置参考总质量", "设置质量")
        return self._turn_session.update(state, self.model, mass, self.afterburner,
                                         self.sweep_fraction, self.turn_settings)

    def tick(self, now: float | None = None) -> OverlaySnapshot:
        """Perform one sampling cycle, on the worker (or synchronously in tests)."""
        self._apply_commands()
        now = time.monotonic() if now is None else now
        state = (make_demo_sample(now-self._started_at) if self.mode == "demo"
                 else self.client.poll(time_s=now))
        turn_state = self._attitude.update(state, calibrate_sign=self._pose_calibration_sign)
        self._pose_calibration_sign = None
        self._select_live_aircraft(state)
        energy = self.estimator.update(state)
        identity = (state.source, aircraft_key(state.aircraft_id)) if state.valid and state.aircraft_id else None
        if identity is not None and identity != self._identity:
            if self._identity is not None:
                self._turn_session.invalidate()
            self._reset_climb()
            self._advice = None
            self._last_prediction = -math.inf
            self._identity = identity
        if not state.valid:
            self._advice = None
            self._last_prediction = -math.inf
        elif (now-self._last_prediction >= 1.0
              or self.mode == "live" and self.model is not None and not _matches(self.model, state.aircraft_id)):
            self._advice = self._predict(state)
            self._last_prediction = now
        climb = self._climb_guidance(state, energy)
        turn = self._turn_guidance(turn_state)
        if turn is not None and self._attitude.estimated:
            turn = replace(turn, estimated_pitch_deg=turn_state.pitch_deg,
                           estimated_roll_deg=turn_state.roll_deg,
                           reason="\n".join(x for x in (turn.reason, self._attitude.reason) if x))
        elif turn is not None and state.valid and (state.pitch_deg is None or state.roll_deg is None):
            turn = replace(turn, phase="需要校准", reason=self._attitude.reason)
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
            tuple(notes), self.mass_kg, self.afterburner,
            self.climb_enabled, self.climb_request, climb, self.model_selection,
            self.sweep_fraction, bool(self.model and getattr(self.model, "wings", ())),
            self.turn_enabled, self.turn_settings, turn, self._attitude.reason)
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
                           energy=EnergyMetrics(snapshot.state.time_s, notes=("样本已过期",)),
                           climb=ClimbGuidance(phase="等待数据") if snapshot.climb_enabled else None,
                           turn=replace(snapshot.turn, available=False, phase="等待数据", action="",
                                        duration_s=None, next_action="", step_remaining_s=None,
                                        progress_stale=True) if snapshot.turn else None)
        return snapshot

    def _run(self) -> None:
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                self.tick(started)
            except Exception as exc:
                self._turn_session.pause(str(exc))
                self._reset_climb()
                self.estimator.reset()
                self._advice = None
                self._last_prediction = -math.inf
                with self._lock:
                    self._snapshot = OverlaySnapshot(
                        self.mode, f"采样失败：{type(exc).__name__}: {exc}",
                        notes=("当前数据不可用；没有切换到演示数据。",),
                        mass_override_kg=self.mass_kg, afterburner=self.afterburner,
                        model_selection=self.model_selection, sweep_fraction=self.sweep_fraction,
                        variable_sweep=bool(self.model and getattr(self.model, "wings", ())),
                        turn_enabled=self.turn_enabled, turn_settings=self.turn_settings,
                        turn=KeyboardTurnGuidance(phase="等待数据") if self.turn_enabled else None,
                        climb_enabled=self.climb_enabled, climb_request=self.climb_request,
                        climb=ClimbGuidance(phase="等待数据") if self.climb_enabled else None)
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
        self._reset_climb()
        self._turn_session.close()
        if self._planner is not None:
            self._planner.shutdown(wait=False, cancel_futures=True)
            self._planner = None
