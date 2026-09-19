"""Native, dependency-free display. Importing this module never creates a window."""

from __future__ import annotations

import math
from typing import Callable

from .contracts import OverlaySnapshot


class OverlayApp:
    """Render snapshots on the Tk thread; commands are delegated to the owner."""

    BG = "#101820"
    PANEL = "#192630"
    FG = "#edf4f6"
    MUTED = "#a7b8c2"
    ACCENT = "#69d9c5"
    WARNING = "#f4c778"
    NEGATIVE = "#ff9c9c"

    def __init__(self, get_snapshot: Callable[[], OverlaySnapshot],
                 on_command: Callable[[dict], None], *, title: str = "WT Energy"):
        global tk, filedialog
        try:
            import tkinter as tk
            from tkinter import filedialog
        except ImportError as exc:
            raise RuntimeError("图形界面需要带 Tcl/Tk 的 Python；Windows 安装时请启用 Tcl/Tk。") from exc
        self._get_snapshot = get_snapshot
        self._on_command = on_command
        self._closed = False
        self._after_id: str | None = None
        self._notes: tuple[str, ...] = ()
        self.root = tk.Tk()
        self.root.title(title)
        self.root.configure(bg=self.BG)
        self.root.geometry("430x690")
        self.root.minsize(390, 540)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._font = "Microsoft YaHei UI" if self.root.tk.call("tk", "windowingsystem") == "win32" else "Helvetica"
        self._topmost = tk.BooleanVar(value=True)
        self._mode = tk.StringVar(value="live")
        self._afterburner = tk.BooleanVar(value=True)
        self._mass = tk.StringVar()
        self._details_open = False
        self._settings_open = False
        self._build()
        self._set_topmost()
        self._refresh()

    def _label(self, parent, text="", size=10, color=None, bold=False, **kwargs):
        return tk.Label(parent, text=text, font=(self._font, size, "bold" if bold else "normal"),
                        bg=parent.cget("bg"), fg=color or self.FG, **kwargs)

    def _button(self, parent, text, command):
        return tk.Button(parent, text=text, command=command, font=(self._font, 10),
                         bg=self.PANEL, fg=self.FG, activebackground="#2b414f",
                         activeforeground=self.FG, relief="flat", padx=8, pady=5,
                         highlightthickness=0, cursor="hand2")

    def _build(self):
        # The scroll container keeps controls reachable at high display scaling.
        canvas = tk.Canvas(self.root, bg=self.BG, highlightthickness=0)
        scrollbar = tk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        body = tk.Frame(canvas, bg=self.BG, padx=18, pady=16)
        window = canvas.create_window((0, 0), window=body, anchor="nw")
        body.bind("<Configure>", lambda event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(window, width=event.width))

        header = tk.Frame(body, bg=self.BG)
        header.pack(fill="x")
        self._label(header, "WT  /  能量", 15, bold=True).pack(side="left")
        self.badge = self._label(header, "等待数据", 10, self.MUTED, bold=True)
        self.badge.pack(side="right")
        self.aircraft = self._label(body, "等待飞机状态", 12, anchor="w")
        self.aircraft.pack(fill="x", pady=(9, 3))
        self.status = self._label(body, "", 9, self.MUTED, anchor="w", justify="left", wraplength=345)
        self.status.pack(fill="x")
        self.banner = self._label(body, "", 10, self.WARNING, bold=True, anchor="w", wraplength=345)
        self.banner.pack(fill="x", pady=(5, 7))

        states = tk.Frame(body, bg=self.BG)
        states.pack(fill="x", pady=(0, 14))
        self.state_values = {}
        for column, (key, caption) in enumerate((("tas", "真空速 km/h"), ("ias", "表速 km/h"), ("alt", "高度 m"))):
            cell = tk.Frame(states, bg=self.BG)
            cell.grid(row=0, column=column, sticky="ew")
            states.columnconfigure(column, weight=1)
            self._label(cell, caption, 9, self.MUTED).pack(anchor="w")
            value = self._label(cell, "—", 19, bold=True)
            value.pack(anchor="w")
            self.state_values[key] = value

        actual = tk.Frame(body, bg=self.PANEL, padx=15, pady=13)
        actual.pack(fill="x")
        self._label(actual, "实际 SEP", 11, self.MUTED).pack(anchor="w")
        sep_line = tk.Frame(actual, bg=self.PANEL)
        sep_line.pack(fill="x")
        self.sep = self._label(sep_line, "—", 38, self.ACCENT, bold=True)
        self.sep.pack(side="left")
        self._label(sep_line, "m/s", 12, self.MUTED).pack(side="left", padx=8, pady=(20, 0))
        self.sep_hint = self._label(actual, "等待有效采样", 9, self.MUTED, anchor="w", wraplength=315)
        self.sep_hint.pack(fill="x", pady=(0, 9))
        self.energy_height = self._metric_row(actual, "比能高度", "m")
        self.climb = self._metric_row(actual, "爬升贡献", "m/s")
        self.kinetic = self._metric_row(actual, "动能贡献", "m/s")

        predicted = tk.Frame(body, bg=self.BG)
        predicted.pack(fill="x", pady=(17, 0))
        self._label(predicted, "静态参考 · 同高 / 1g / 干净构型", 11, bold=True).pack(anchor="w")
        self.model = self._label(predicted, "未加载 FM", 9, self.MUTED, anchor="w", wraplength=345)
        self.model.pack(fill="x", pady=(3, 5))
        self.pred_sep = self._metric_row(predicted, "当前速度的参考 SEP", "m/s")
        self.best_speed = self._metric_row(predicted, "采样最高 SEP 对应真空速", "km/h")
        self.best_sep = self._metric_row(predicted, "采样最高 SEP", "m/s")
        self.pred_hint = self._label(predicted, "等待模型", 9, self.WARNING, anchor="w", justify="left", wraplength=345)
        self.pred_hint.pack(fill="x", pady=(5, 0))
        self._label(body, "三维转向建议：动态模型尚未接入", 9, self.MUTED, anchor="w").pack(fill="x", pady=(14, 8))

        controls = tk.Frame(body, bg=self.BG)
        controls.pack(fill="x")
        self.settings_button = self._button(controls, "设置 ▸", self._toggle_settings)
        self.settings_button.pack(side="left")
        self.details_button = self._button(controls, "说明 ▸", self._toggle_details)
        self.details_button.pack(side="left", padx=6)
        self._check(controls, "置顶", self._topmost, self._set_topmost).pack(side="right")

        self.settings = tk.Frame(body, bg=self.PANEL, padx=10, pady=10)
        mode_row = tk.Frame(self.settings, bg=self.PANEL)
        mode_row.pack(fill="x")
        for caption, value in (("实时 8111", "live"), ("合成演示", "demo")):
            tk.Radiobutton(mode_row, text=caption, value=value, variable=self._mode,
                           command=lambda: self._command({"action": "mode", "value": self._mode.get()}),
                           bg=self.PANEL, fg=self.FG, selectcolor=self.BG,
                           activebackground=self.PANEL, activeforeground=self.FG,
                           font=(self._font, 10)).pack(side="left")
        self._button(self.settings, "选择 FM 文件…", self._choose_model).pack(fill="x", pady=7)
        mass_row = tk.Frame(self.settings, bg=self.PANEL)
        mass_row.pack(fill="x")
        self._label(mass_row, "总质量 kg", 10).pack(side="left")
        entry = tk.Entry(mass_row, textvariable=self._mass, width=11, font=(self._font, 10),
                         bg=self.BG, fg=self.FG, insertbackground=self.FG, relief="flat")
        entry.pack(side="left", padx=7, ipady=5)
        entry.bind("<Return>", lambda event: self._apply_mass())
        self._button(mass_row, "应用", self._apply_mass).pack(side="right")
        self._check(self.settings, "模型使用加力", self._afterburner,
                    lambda: self._command({"action": "afterburner", "enabled": self._afterburner.get()})).pack(anchor="w", pady=5)
        self._label(self.settings, "窗口不透明度", 9, self.MUTED).pack(anchor="w")
        tk.Scale(self.settings, from_=0.45, to=1.0, resolution=0.05, orient="horizontal",
                 showvalue=False, command=self._set_opacity, bg=self.PANEL, fg=self.FG,
                 highlightthickness=0, troughcolor=self.BG, variable=tk.DoubleVar(value=1.0)).pack(fill="x")
        self.details = self._label(body, "暂无额外说明", 9, self.MUTED, justify="left", anchor="w", wraplength=345)
        self.command_status = self._label(body, "", 9, self.WARNING, anchor="w", justify="left", wraplength=345)
        self.command_status.pack(fill="x", pady=(7, 0))

    def _check(self, parent, text, variable, command):
        return tk.Checkbutton(parent, text=text, variable=variable, command=command,
                              bg=parent.cget("bg"), fg=self.MUTED, selectcolor=self.BG,
                              activebackground=parent.cget("bg"), activeforeground=self.FG,
                              font=(self._font, 9))

    def _metric_row(self, parent, caption, unit):
        row = tk.Frame(parent, bg=parent.cget("bg"))
        row.pack(fill="x", pady=3)
        self._label(row, caption, 10, self.MUTED).pack(side="left")
        value = self._label(row, "— " + unit, 11, bold=True)
        value.pack(side="right")
        return value

    @staticmethod
    def _number(value, digits=0, signed=False, scale=1.0):
        if value is None or not math.isfinite(value):
            return "—"
        return format(value * scale, f"{'+' if signed else ''},.{digits}f")

    def _render(self, snapshot: OverlaySnapshot):
        state, energy, advice = snapshot.state, snapshot.energy, snapshot.advice
        valid = state is not None and state.valid
        demo = snapshot.mode == "demo" or (state is not None and state.source == "demo")
        self._mode.set(snapshot.mode)
        self._afterburner.set(snapshot.afterburner)
        if not self._mass.get() and snapshot.mass_override_kg is not None:
            self._mass.set(f"{snapshot.mass_override_kg:g}")
        self.badge.configure(text="DEMO" if demo else ("LIVE" if valid else "未连接 / 无效"),
                             fg=self.WARNING if demo else (self.ACCENT if valid else self.MUTED))
        self.banner.configure(text="合成演示数据 · 非游戏实测" if demo else "")
        self.aircraft.configure(text=state.aircraft_id if state and state.aircraft_id else "等待飞机状态")
        self.status.configure(text=snapshot.status)
        for key, value, scale in (("tas", state.tas_mps if valid else None, 3.6),
                                  ("ias", state.ias_mps if valid else None, 3.6),
                                  ("alt", state.altitude_m if valid else None, 1.0)):
            self.state_values[key].configure(text=self._number(value, scale=scale))
        ready = valid and energy is not None and energy.ready
        sep = energy.sep_mps if ready else None
        self.sep.configure(text=self._number(sep, 1, True),
                           fg=self.NEGATIVE if sep is not None and sep < 0 else self.ACCENT)
        self.sep_hint.configure(text=("总比能正在增加" if sep > 0 else "总比能正在减少" if sep < 0 else "总比能基本不变")
                                if sep is not None and math.isfinite(sep) else
                                ("等待稳定采样" if valid else "等待有效飞行数据"))
        self.energy_height.configure(text=self._number(energy.energy_height_m if valid and energy else None) + " m")
        self.climb.configure(text=self._number(energy.climb_mps if valid and energy else None, 1, True) + " m/s")
        self.kinetic.configure(text=self._number(energy.kinetic_sep_mps if ready else None, 1, True) + " m/s")
        self.model.configure(text=snapshot.model_name)
        current = advice.current if valid and advice else None
        best = advice.best if valid and advice and advice.available else None
        self.pred_sep.configure(text=self._number(current.sep_mps if current and current.valid else None, 1, True) + " m/s")
        self.best_speed.configure(text=self._number(best.condition.tas_mps if best and best.valid else None, scale=3.6) + " km/h")
        self.best_sep.configure(text=self._number(best.sep_mps if best and best.valid else None, 1, True) + " m/s")
        hint = advice.reason if advice and advice.reason else (current.reason if current and not current.valid else "")
        self.pred_hint.configure(text=hint or ("未求配平 · 尚未通过游戏验证 · 不代表当前机动" if current or best else "加载 FM 并设置总质量后查看参考性能"))
        notes = list(snapshot.notes)
        for item in (state, energy, advice, current, best):
            if item:
                notes.extend(item.notes)
        if current or best:
            notes.insert(0, "模型预测未求配平，尚未通过游戏验证；采样最优不等于全程最优爬升。")
        self._notes = tuple(dict.fromkeys(notes))
        self.details.configure(text="\n\n".join(self._notes) or "暂无额外说明")

    def _refresh(self):
        if self._closed:
            return
        try:
            self._render(self._get_snapshot())
        except Exception as exc:
            self._render(OverlaySnapshot(mode="live", status="暂时无法读取飞行状态"))
            self.badge.configure(text="数据不可用", fg=self.WARNING)
            self.command_status.configure(text=f"读取状态失败：{exc}")
            self.sep.configure(text="—")
        self._after_id = self.root.after(100, self._refresh)

    def _command(self, command):
        try:
            self._on_command(command)
        except Exception as exc:
            self.command_status.configure(text=f"操作未完成：{exc}")
        else:
            self.command_status.configure(text="")

    def _choose_model(self):
        path = filedialog.askopenfilename(parent=self.root, title="选择飞机 FM 文件",
                                          filetypes=(("FM 文件", "*.blkx *.json *.blk"), ("所有文件", "*")))
        if path:
            self._command({"action": "model", "path": path})

    def _apply_mass(self):
        try:
            value = float(self._mass.get().strip())
            if not math.isfinite(value) or value <= 0:
                raise ValueError
        except ValueError:
            self.command_status.configure(text="请输入大于 0 的总质量（kg）。")
            return
        self._command({"action": "mass", "kg": value})

    def _toggle_settings(self):
        self._settings_open = not self._settings_open
        self.settings_button.configure(text="设置 ▾" if self._settings_open else "设置 ▸")
        if self._settings_open:
            self.settings.pack(fill="x", pady=(9, 0), before=self.command_status)
        else:
            self.settings.pack_forget()

    def _toggle_details(self):
        self._details_open = not self._details_open
        self.details_button.configure(text="说明 ▾" if self._details_open else "说明 ▸")
        if self._details_open:
            self.details.pack(fill="x", pady=(10, 0), before=self.command_status)
        else:
            self.details.pack_forget()

    def _set_topmost(self):
        try:
            self.root.attributes("-topmost", self._topmost.get())
        except tk.TclError:
            self.command_status.configure(text="当前窗口系统不支持置顶。")

    def _set_opacity(self, value):
        try:
            self.root.attributes("-alpha", float(value))
        except tk.TclError:
            self.command_status.configure(text="当前窗口系统不支持透明度调整。")

    def run(self):
        self.root.mainloop()

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._after_id is not None:
            self.root.after_cancel(self._after_id)
        self.root.destroy()
