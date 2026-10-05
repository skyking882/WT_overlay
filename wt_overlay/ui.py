"""Transparent Qt HUD groups and an independent settings window.

Interaction follows WTRTI's group/OSD approach. Rendering uses Qt's native
translucency and input-transparent windows, not whole-window opacity.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import ctypes
from ctypes import wintypes
import math
import json
from dataclasses import asdict, replace
import sys
from typing import Callable

try:
    from PySide6.QtCore import QAbstractNativeEventFilter, QPoint, QPointF, QRect, QRectF, QSettings, Qt, QTimer, Signal
    from PySide6.QtGui import QAction, QColor, QFont, QFontMetrics, QIcon, QPainter, QPainterPath, QPen, QPixmap, QPolygonF
    from PySide6.QtWidgets import (QApplication, QCheckBox, QColorDialog, QComboBox, QDoubleSpinBox,
        QFileDialog, QFontComboBox, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
        QLineEdit, QMenu, QPushButton, QScrollArea, QSpinBox, QSystemTrayIcon,
        QTextEdit, QVBoxLayout, QWidget)
except ImportError as exc:
    raise RuntimeError("图形界面需要 PySide6。请运行 start_windows.cmd，或安装 requirements.txt 中的依赖。") from exc

from .contracts import ClimbRequest, OverlaySnapshot, KeyboardTurnSettings
from .climb import CUE_DEADBAND_DEG, CUE_RANGE_DEG, validate_request
from . import pk
from .fm.catalog import aircraft_catalog, find_aircraft
from .hud import INDICATORS, HudContent, contents, details
from .offense import MODES, REACH_P, SKILLS, OffenseAdvisor, available, course_for_blip_direction
from .windows import GameWindow, WindowsDesktop
from .turn import validate_settings


GROUPS = {"flight": "飞行状态", "energy": "实际能量", "engine": "动力与燃油", "reference": "静态参考",
          "climb": "爬升引导", "turn": "转向引导", "rose": "进攻动向图", "scope": "B 显发射区", "side": "高度侧视图"}
DEFAULT_POSITIONS = {"flight": (0.03, 0.22), "energy": (0.03, 0.60),
                     "engine": (0.73, 0.30), "reference": (0.73, 0.60), "climb": (0.42, 0.65),
                     "turn": (0.42, 0.65), "rose": (0.80, 0.78), "scope": (0.80, 0.40), "side": (0.80, 0.15)}
HOTKEYS = {"O": "显示 / 隐藏 HUD", "L": "进入 / 退出布局", "S": "打开设置", "C": "开关爬升引导",
           "T": "开关转向引导", "R": "重新开始转向", "A": "平翼校准", "K": "开关进攻动向图",
           "B": "开关 B 显发射区", "V": "开关高度侧视图"}
# Hit probability if fired now: from 50 % red, from 25 % amber, else grey; dark when the missile cannot reach.
ROSE_BANDS = ((0.5, QColor(226, 75, 74, 190)), (0.25, QColor(239, 159, 39, 170)))
ROSE_SLOW = QColor(95, 94, 90, 140)
ROSE_UNREACHABLE = QColor(44, 44, 42, 120)
# (line, label, colour, dashed): ranges from data/envelope tables.
SCOPE_LINES = (("rmax_hot", "Rmax 迎头", "#EF9F27", False), ("rmax_cold", "Rmax 背离", "#EF9F27", True),
               ("pk25_hot", "命中 25%", "#E24B4A", True), ("pk50_hot", "命中 50%", "#E24B4A", False))
# Enemy types offered as the assumed target (first = default) and their assumed speeds.
ENEMY_AIRCRAFT = ("f_16c_block_50", "su_27sm", "j_11b", "su_30sm", "mig_29_9_13", "f_15c_msip2", "j_10c",
                  "ef_2000_typhoon_aesa", "rafale_c_f3", "saab_jas39c", "fa_18c_late", "mirage_2000_5f")
TARGET_SPEEDS_KMH = (800, 900, 1000, 1100, 1200, 1300)
SCOPE_RANGES_KM = (10, 20, 40, 80, 160)
SCOPE_AZIMUTHS_DEG = (30, 45, 60, 70)
MISSILE_NAMES = {"cn_pl12": "PL-12", "cn_pl12a": "PL-12A", "su_r_77": "R-77", "su_r_77_1": "R-77-1",
                 "cn_pl15": "PL-15", "cn_sd10a": "SD-10A", "il_derby": "Derby", "jp_aam4": "AAM-4",
                 "r_darter": "R-Darter", "swd_rb99": "Rb 99", "su_rvv_ae": "RVV-AE",
                 "us_aim_120a": "AIM-120A", "us_aim_120b": "AIM-120B", "us_aim_120c_5": "AIM-120C-5",
                 "us_aim_120c_7": "AIM-120C-7", "us_aim_120d": "AIM-120D"}


def game_geometry(game: GameWindow, screen) -> QRect:
    """Map native pixels using the monitor's origin, including mixed-DPI desktops."""
    x, y, width, height = game.rect
    mx, my = game.monitor_origin
    area, ratio = screen.geometry(), screen.devicePixelRatio()
    return QRect(area.x() + round((x - mx) / ratio), area.y() + round((y - my) / ratio),
                 round(width / ratio), round(height / ratio))


def position_fraction(position: QPoint, size, viewport: QRect) -> tuple[float, float]:
    return (min(1.0, max(0.0, (position.x() - viewport.x()) / max(1, viewport.width() - size.width()))),
            min(1.0, max(0.0, (position.y() - viewport.y()) / max(1, viewport.height() - size.height()))))


class HudGroup(QWidget):
    moved = Signal(str)

    def __init__(self, key: str, font: QFont, color: str):
        super().__init__()
        self.key = key
        self.content = HudContent(GROUPS[key], ())
        self.editing = False
        self.fraction = DEFAULT_POSITIONS[key]
        self.viewport = QRect(0, 0, 1920, 1080)
        self._drag = None
        self.accent = color
        self.setFont(font)
        self.setWindowTitle("WT Energy HUD · " + GROUPS[key])
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setAutoFillBackground(False)
        self.setWindowFlags(self._flags())
        self._measure()

    def _flags(self):
        flags = (Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint
                 | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.WindowDoesNotAcceptFocus)
        if not self.editing:
            flags |= Qt.WindowType.WindowTransparentForInput
        return flags

    def set_editing(self, enabled: bool):
        if enabled == self.editing:
            return
        self.editing = enabled
        self._drag = None
        position, visible = self.pos(), self.isVisible()
        self.setWindowFlags(self._flags())
        self.move(position)
        self._measure()
        self.setCursor(Qt.CursorShape.SizeAllCursor if enabled else Qt.CursorShape.ArrowCursor)
        if visible:
            self.show()
        self.update()

    def set_content(self, content: HudContent):
        if content != self.content:
            self.content = content
            self._measure()
            self.update()

    def set_style(self, font: QFont, color: str):
        self.setFont(font)
        self.accent = color
        self._measure()
        self.update()

    def _measure(self):
        self.small_font = QFont(self.font())
        self.small_font.setPointSizeF(max(8.0, self.font().pointSizeF() * 0.75))
        self.small_metrics = QFontMetrics(self.small_font)
        self.metrics = QFontMetrics(self.font())
        self.row_height = self.metrics.height() + 3
        self.header_height = self.small_metrics.height() + 8 if self._header() else 0
        labels = max((self.metrics.horizontalAdvance(row.label) for row in self.content.rows), default=120)
        values = max([self.metrics.horizontalAdvance("−12,345.6"),
                      *(self.metrics.horizontalAdvance(row.value) for row in self.content.rows)])
        self.unit_width = max([self.small_metrics.horizontalAdvance("km/h"),
                              *(self.small_metrics.horizontalAdvance(row.unit) for row in self.content.rows)])
        ideal = max(labels + values + self.unit_width + 54,
                    self.small_metrics.horizontalAdvance(self._header()) + 24)
        width = min(max(220, ideal), max(100, self.viewport.width()))
        height = 16 + self.header_height + len(self.content.rows) * self.row_height
        if self.content.cue_active:
            height += self.small_metrics.height() + 25
        self.resize(width, height)

    def _header(self):
        if self.editing:
            return self.content.title + (" · DEMO" if self.content.demo else "")
        return "DEMO" if self.content.demo else ""

    def place(self, viewport: QRect):
        if self.viewport != viewport:
            self.viewport = QRect(viewport)
            self._measure()
        if self._drag is None:
            x, y = self.fraction
            self.move(viewport.x() + round(max(0, viewport.width() - self.width()) * x),
                      viewport.y() + round(max(0, viewport.height() - self.height()) * y))

    @staticmethod
    def _text(painter, x, baseline, text, font, color):
        path = QPainterPath()
        path.addText(x, baseline, font, text)
        outline = QPen(QColor(0, 0, 0, 235), 2.5, Qt.PenStyle.SolidLine,
                       Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        painter.strokePath(path, outline)
        painter.fillPath(path, QColor(color))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
        painter.fillRect(self.rect(), Qt.GlobalColor.transparent)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        if self.editing:
            painter.fillRect(self.rect(), QColor(12, 22, 30, 110))
            painter.setPen(QPen(QColor(self.accent), 1, Qt.PenStyle.DashLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(self.rect().adjusted(1, 1, -2, -2), 5, 5)
        header = self._header()
        header = self.small_metrics.elidedText(header, Qt.TextElideMode.ElideRight, self.width() - 24)
        if header:
            header_color = "#ffce79" if self.content.demo else self.accent
            self._text(painter, 12, 8 + self.small_metrics.ascent(), header, self.small_font, header_color)
        y = 8 + self.header_height
        value_right = self.width() - self.unit_width - 22
        for row in self.content.rows:
            value_width = self.metrics.horizontalAdvance(row.value)
            label = self.metrics.elidedText(row.label, Qt.TextElideMode.ElideRight,
                                            max(1, value_right - value_width - 25))
            baseline = y + self.metrics.ascent()
            self._text(painter, 12, baseline, label, self.font(), "#f1f5f8")
            color = {"accent": self.accent, "negative": "#ff8d86"}.get(row.tone, "#ffffff")
            self._text(painter, value_right - value_width, baseline, row.value, self.font(), color)
            self._text(painter, value_right + 8, baseline, row.unit, self.small_font, "#d2dee5")
            y += self.row_height
        if self.content.cue_active:
            left, right = 20, self.width() - 20
            center, half = (left + right) / 2, (right - left) / 2
            bar_y = y + 9
            painter.setPen(QPen(QColor(0, 0, 0, 220), 5))
            painter.drawLine(QPoint(left, bar_y), QPoint(right, bar_y))
            painter.setPen(QPen(QColor("#d2dee5"), 2))
            painter.drawLine(QPoint(left, bar_y), QPoint(right, bar_y))
            band = max(3, round(half * CUE_DEADBAND_DEG / CUE_RANGE_DEG))
            painter.fillRect(round(center)-band, bar_y-5, 2*band+1, 11, QColor("#71e3a0"))
            error = self.content.cue_error_deg
            if error is not None and math.isfinite(error):
                x = round(center + half * max(-1, min(1, error / CUE_RANGE_DEG)))
                painter.setPen(QPen(QColor("#101820"), 2))
                painter.setBrush(QColor("#ffffff"))
                painter.drawEllipse(QPoint(x, bar_y), 5, 5)
            baseline = bar_y + 12 + self.small_metrics.ascent()
            self._text(painter, left, baseline, "压低", self.small_font, "#d2dee5")
            self._text(painter, right-self.small_metrics.horizontalAdvance("抬高"), baseline,
                       "抬高", self.small_font, "#d2dee5")
        painter.end()

    def mousePressEvent(self, event):
        if self.editing and event.button() == Qt.MouseButton.LeftButton:
            self._drag = event.globalPosition().toPoint() - self.pos()
            event.accept()

    def mouseMoveEvent(self, event):
        if self.editing and self._drag is not None:
            point = event.globalPosition().toPoint() - self._drag
            point.setX(min(max(self.viewport.left(), point.x()),
                           self.viewport.left() + max(0, self.viewport.width() - self.width())))
            point.setY(min(max(self.viewport.top(), point.y()),
                           self.viewport.top() + max(0, self.viewport.height() - self.height())))
            self.move(point)
            event.accept()

    def mouseReleaseEvent(self, event):
        if self._drag is not None:
            self._drag = None
            self.fraction = position_fraction(self.pos(), self.size(), self.viewport)
            self.moved.emit(self.key)
            event.accept()


class RoseGroup(HudGroup):
    """Kill rose: rings are launch range, sectors the blip's motion on the B-scope
    (down = target hot), colour the hit probability if fired now against the
    assumed opponent (offense.OffenseAdvisor).

    Rings are equally wide (not to range scale), labelled with their outer range in km
    along the left axis; the head-on, beam and cold axes carry each ring's hit probability
    in %. An annulus shows its outer ring's value, so a target between two rings is drawn
    with the farther (usually lower) probability.
    """

    def __init__(self, key: str, font: QFont, color: str):
        self.rose = None
        self.caption = ""
        super().__init__(key, font, color)

    def set_rose(self, rose, caption=""):
        if rose != self.rose or caption != self.caption:
            self.rose, self.caption = rose, caption
            self._measure()
            self.update()

    def has_data(self):
        return self.rose is not None

    def _measure(self):
        super()._measure()
        # Rings must fit a two-digit percentage across them.
        self.radius = max(90, round(self.metrics.height() * 8))
        self.label_height = self.small_metrics.height() + 4
        width = max(2 * self.radius + 24, self.small_metrics.horizontalAdvance(self.caption) + 24)
        height = 8 + self.header_height + 2 * self.label_height + 2 * self.radius + 8
        if self.caption:
            height += self.label_height
        self.resize(min(width, max(100, self.viewport.width())), height)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
        painter.fillRect(self.rect(), Qt.GlobalColor.transparent)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        if self.editing:
            painter.fillRect(self.rect(), QColor(12, 22, 30, 110))
            painter.setPen(QPen(QColor(self.accent), 1, Qt.PenStyle.DashLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(self.rect().adjusted(1, 1, -2, -2), 5, 5)
        header = self._header()
        if header:
            self._text(painter, 12, 8 + self.small_metrics.ascent(), header, self.small_font,
                       "#ffce79" if self.content.demo else self.accent)
        top = 8 + self.header_height
        cx, cy = self.width() / 2, top + self.label_height + self.radius
        self._label(painter, cx, top + self.small_metrics.ascent(), "↑ 背离")
        if self.rose is not None:
            ranges = self.rose.ranges_m
            # Equal-width rings: the short ranges, where the colours change, are not squeezed into the centre.
            ring = self.radius / len(ranges)
            painter.setPen(QPen(QColor(0, 0, 0, 170), 1))
            for i, range_m in enumerate(ranges):
                outer, inner = (i + 1) * ring, i * ring
                for k in range(12):
                    cell = self.rose.at(range_m, course_for_blip_direction(30.0 * k))
                    color = (ROSE_UNREACHABLE if cell is None or cell[1] < REACH_P else
                             next((c for limit, c in ROSE_BANDS if cell[0] >= limit), ROSE_SLOW))
                    # Qt angles are counter-clockwise from 3 o'clock; 270° is straight down.
                    start, span = 270.0 + 30.0 * k - 15.0, 30.0
                    path = QPainterPath()
                    outer_rect = QRect(round(cx - outer), round(cy - outer), round(2 * outer), round(2 * outer))
                    if inner > 0:
                        inner_rect = QRect(round(cx - inner), round(cy - inner), round(2 * inner), round(2 * inner))
                        path.arcMoveTo(outer_rect, start)
                        path.arcTo(outer_rect, start, span)
                        path.arcTo(inner_rect, start + span, -span)
                    else:
                        path.moveTo(cx, cy)
                        path.arcTo(outer_rect, start, span)
                    path.closeSubpath()
                    painter.setBrush(color)
                    painter.drawPath(path)
            small = self.small_metrics
            # Range scale along the left half of the horizontal axis (km at each ring's outer edge).
            for i, range_m in enumerate(ranges):
                text = f"{range_m / 1000:g}"
                self._text(painter, cx - (i + 1) * ring + 2, cy - 3, text, self.small_font, "#9fb3bf")
            # Hit probability in % along the head-on (down), beam (right) and cold (up) axes.
            for course, dx, dy in ((0.0, 0, 1), (90.0, 1, 0), (180.0, 0, -1)):
                for i, range_m in enumerate(ranges):
                    cell = self.rose.at(range_m, course)
                    text = "–" if cell is None or cell[1] < REACH_P else f"{round(cell[0] * 100)}"
                    mid = (i + 0.5) * ring
                    x = cx + dx * mid - small.horizontalAdvance(text) / 2
                    y = cy + dy * mid + small.ascent() / 2 - 1
                    self._text(painter, x, y, text, self.small_font, "#ffffff")
        else:
            painter.setPen(QPen(QColor(self.accent), 1, Qt.PenStyle.DashLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QPoint(round(cx), round(cy)), self.radius, self.radius)
        bottom = cy + self.radius + 4
        self._label(painter, cx, bottom + self.small_metrics.ascent(), "↓ 迎头")
        if self.caption:
            self._label(painter, cx, bottom + self.label_height + self.small_metrics.ascent(), self.caption)
        painter.end()

    def _label(self, painter, center_x, baseline, text):
        x = center_x - self.small_metrics.horizontalAdvance(text) / 2
        self._text(painter, x, baseline, text, self.small_font, "#d2dee5")


class ScopeGroup(HudGroup):
    """Launch-envelope lines over the game's B-scope: x = azimuth, y = range up from the bottom.

    The user sizes and places the box on the radar display in layout mode
    (drag to move, bottom-right corner to resize); range scale and azimuth
    limit come from settings and must match the game's radar. Only the
    straight-flight, co-altitude lines are drawn.
    """
    GRIP = 16

    def __init__(self, key: str, font: QFont, color: str):
        self.envelope = None
        self.scale_m, self.half_azimuth_deg = 40000.0, 60.0
        self.box = (260, 260)
        self._resize = None
        super().__init__(key, font, color)

    def set_envelope(self, envelope, scale_m, half_azimuth_deg):
        if (envelope, scale_m, half_azimuth_deg) != (self.envelope, self.scale_m, self.half_azimuth_deg):
            self.envelope, self.scale_m, self.half_azimuth_deg = envelope, scale_m, half_azimuth_deg
            self.update()

    def set_box(self, width, height):
        self.box = (max(80, int(width)), max(80, int(height)))
        self._measure()
        self.update()

    def has_data(self):
        return self.envelope is not None

    def _measure(self):
        super()._measure()
        self.resize(*self.box)

    def _point(self, azimuth_deg, range_m):
        # Unclamped: lines beyond the range scale fall above the box and are clipped away.
        return QPoint(round((azimuth_deg / self.half_azimuth_deg + 1) / 2 * self.width()),
                      round(self.height() - range_m / self.scale_m * self.height()))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
        painter.fillRect(self.rect(), Qt.GlobalColor.transparent)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        if self.editing:
            painter.fillRect(self.rect(), QColor(12, 22, 30, 110))
            painter.setPen(QPen(QColor(self.accent), 1, Qt.PenStyle.DashLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(self.rect().adjusted(0, 0, -1, -1))
            painter.setBrush(QColor(self.accent))
            w, h, g = self.width(), self.height(), self.GRIP
            painter.drawPolygon([QPoint(w - 1, h - g), QPoint(w - 1, h - 1), QPoint(w - g, h - 1)])
            self._text(painter, 8, 8 + self.small_metrics.ascent(),
                       f"{self.content.title} · {self.scale_m / 1000:g} km · ±{self.half_azimuth_deg:g}°",
                       self.small_font, self.accent)
        if self.envelope is not None:
            painter.setClipRect(self.rect())
            labels = []
            for name, label, color, dashed in SCOPE_LINES:
                points = [(az, r) for az, r in self.envelope.line(name) if r is not None]
                if len(points) < 2:
                    continue
                polyline = [self._point(az, r) for az, r in points]
                painter.setPen(QPen(QColor(0, 0, 0, 200), 4))
                painter.drawPolyline(polyline)
                painter.setPen(QPen(QColor(color), 2, Qt.PenStyle.DashLine if dashed else Qt.PenStyle.SolidLine))
                painter.drawPolyline(polyline)
                center = dict(points).get(0.0)
                if center is not None:
                    above = center > self.scale_m
                    value = ">45" if self.envelope.is_capped(name) else f"{center / 1000:.0f}"
                    labels.append((("↑ " if above else "") + f"{label} {value}", color,
                                   self.small_metrics.ascent() + 2 if above else self._point(0.0, center).y() - 3))
            self._draw_labels(painter, labels)
        painter.end()

    def _draw_labels(self, painter, labels):
        """Place each label right of centre, else on the left or right edge, avoiding overlaps."""
        placed, height = [], self.small_metrics.height()
        for text, color, baseline in labels:
            width = self.small_metrics.horizontalAdvance(text)
            baseline = max(self.small_metrics.ascent() + 2, min(self.height() - 4, baseline))
            for x in (self.width() / 2 + 4, 4, self.width() - width - 4):
                box = QRect(round(x), round(baseline - self.small_metrics.ascent()), width, height)
                if not any(box.intersects(other) for other in placed):
                    break
            placed.append(box)
            self._text(painter, box.x(), baseline, text, self.small_font, color)

    def mousePressEvent(self, event):
        position = event.position().toPoint()
        if (self.editing and event.button() == Qt.MouseButton.LeftButton
                and position.x() >= self.width() - self.GRIP and position.y() >= self.height() - self.GRIP):
            self._resize = (event.globalPosition().toPoint(), self.width(), self.height())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._resize is not None:
            origin, width, height = self._resize
            delta = event.globalPosition().toPoint() - origin
            self.set_box(width + delta.x(), height + delta.y())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._resize is not None:
            self._resize = None
            self.fraction = position_fraction(self.pos(), self.size(), self.viewport)
            self.moved.emit(self.key)
            event.accept()
            return
        super().mouseReleaseEvent(event)


class SideGroup(HudGroup):
    """Launch envelope in the vertical plane: x = launch range, y = target altitude minus own altitude.

    Same four lines as the B-scope overlay, taken head-on (azimuth 0) at each tabulated
    altitude difference. Fixed size, scaled with the font; the range scale follows the
    B-scope setting so distances match the radar.
    """
    MIN_EXTENT_M, MARGIN_M, TICK_M = 4000., 1000., 3000.

    def __init__(self, key: str, font: QFont, color: str):
        self.envelope = None
        self.scale_m = 40000.0
        super().__init__(key, font, color)

    def set_envelope(self, envelope, scale_m):
        if (envelope, scale_m) != (self.envelope, self.scale_m):
            self.envelope, self.scale_m = envelope, scale_m
            self.update()

    def has_data(self):
        return self.envelope is not None

    def _measure(self):
        super()._measure()
        unit = self.metrics.height()
        self.resize(min(16 * unit, max(100, self.viewport.width())), 9 * unit + self.header_height)

    def _plot(self):
        small = self.small_metrics
        left = small.horizontalAdvance("−3") + 12
        top = 8 + self.header_height + 4
        return QRectF(left, top, self.width() - left - 10, self.height() - top - small.height() - 8)

    def _profiles(self):
        return [(name, label, color, dashed,
                 [(dh, r) for dh, r in self.envelope.profile(name, 0.) if r is not None])
                for name, label, color, dashed in SCOPE_LINES]

    def _extent_m(self, profiles):
        spread = max((abs(dh) for *_, points in profiles for dh, _ in points), default=0.)
        return max(self.MIN_EXTENT_M, spread + self.MARGIN_M)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
        painter.fillRect(self.rect(), Qt.GlobalColor.transparent)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        if self.editing:
            painter.fillRect(self.rect(), QColor(12, 22, 30, 110))
            painter.setPen(QPen(QColor(self.accent), 1, Qt.PenStyle.DashLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(self.rect().adjusted(1, 1, -2, -2), 5, 5)
        header = self._header()
        if header:
            self._text(painter, 12, 8 + self.small_metrics.ascent(), header, self.small_font,
                       "#ffce79" if self.content.demo else self.accent)
        plot, small = self._plot(), self.small_metrics
        profiles = self._profiles() if self.envelope is not None else []
        extent = self._extent_m(profiles)
        mid = plot.center().y()
        x_of = lambda range_m: plot.left() + range_m / self.scale_m * plot.width()  # noqa: E731
        y_of = lambda dh: mid - dh / extent * plot.height() / 2  # noqa: E731
        grid, light = QColor(210, 222, 229, 55), "#d2dee5"
        painter.setPen(QPen(grid, 1))
        for k in range(1, 5):
            painter.drawLine(QPointF(plot.left() + plot.width() * k / 4, plot.top()),
                             QPointF(plot.left() + plot.width() * k / 4, plot.bottom()))
        for dh in (-self.TICK_M, self.TICK_M):
            painter.drawLine(QPointF(plot.left(), y_of(dh)), QPointF(plot.right(), y_of(dh)))
        painter.setPen(QPen(QColor(210, 222, 229, 150), 1, Qt.PenStyle.DashLine))
        painter.drawLine(QPointF(plot.left(), mid), QPointF(plot.right(), mid))
        painter.setPen(QPen(QColor(210, 222, 229, 90), 1))
        painter.drawLine(QPointF(plot.left(), plot.top()), QPointF(plot.left(), plot.bottom()))
        painter.drawLine(QPointF(plot.left(), plot.bottom()), QPointF(plot.right(), plot.bottom()))
        for text, dh in (("+3", self.TICK_M), ("0", 0.), ("−3", -self.TICK_M)):
            self._text(painter, plot.left() - 4 - small.horizontalAdvance(text),
                       y_of(dh) + small.ascent() / 2 - 1, text, self.small_font, light)
        baseline = plot.bottom() + 4 + small.ascent()
        for k in range(5):
            text = f"{self.scale_m * k / 4000:g}" + (" km" if k == 4 else "")
            width = small.horizontalAdvance(text)
            x = min(plot.left() + plot.width() * k / 4 - width / 2, self.width() - width - 2)
            self._text(painter, max(2., x), baseline, text, self.small_font, light)
        self._text(painter, plot.left() + 4, plot.top() + small.ascent() + 1, "↑ 目标高", self.small_font, light)
        self._text(painter, plot.left() + 4, plot.bottom() - 3, "↓ 目标低", self.small_font, light)
        labels = []
        painter.setClipRect(plot.adjusted(0, -1, 1, 1))
        for name, label, color, dashed, points in profiles:
            if len(points) < 2:
                continue
            polyline = [QPointF(x_of(r), y_of(dh)) for dh, r in points]
            painter.setPen(QPen(QColor(0, 0, 0, 200), 4))
            painter.drawPolyline(polyline)
            painter.setPen(QPen(QColor(color), 2, Qt.PenStyle.DashLine if dashed else Qt.PenStyle.SolidLine))
            painter.drawPolyline(polyline)
            level = dict(points).get(0.0)
            if level is not None:
                off = level > self.scale_m
                value = ">45" if self.envelope.is_capped(name, 0.0, 0.0) else f"{level / 1000:.0f}"
                labels.append((("→ " if off else "") + f"{label} {value}", color,
                               None if off else x_of(level), mid))
        painter.setClipping(False)
        # Own aircraft: a marker on the zero line at the left edge.
        marker = QPolygonF([QPointF(plot.left() - 1, mid - 5), QPointF(plot.left() + 9, mid),
                            QPointF(plot.left() - 1, mid + 5)])
        painter.setPen(QPen(QColor(0, 0, 0, 220), 1))
        painter.setBrush(QColor("#ffffff"))
        painter.drawPolygon(marker)
        self._draw_labels(painter, labels, plot)
        painter.end()

    def _draw_labels(self, painter, labels, plot):
        """Label each line at the zero-difference point, stacking above/below the zero line to avoid overlaps."""
        small = self.small_metrics
        placed, height = [], small.height()
        placed.append(QRect(round(plot.left() + 4), round(plot.top()), 60, height + 1))  # "↑ 目标高"
        for text, color, anchor_x, mid in labels:
            width = small.horizontalAdvance(text)
            if anchor_x is None:
                xs = (plot.right() - width - 3,)
            else:
                xs = (anchor_x + 4, anchor_x - width - 4, plot.right() - width - 3)
            steps = []
            for i in range(5):
                steps += [-2 - i * height, small.ascent() + 2 + i * height]
            box = None
            for x in xs:
                for dy in steps:
                    cx = min(max(plot.left() + 2, x), plot.right() - width - 2)
                    base = mid + dy
                    cand = QRect(round(cx), round(base - small.ascent()), width, height)
                    if (not any(cand.adjusted(-6, 0, 6, 0).intersects(o) for o in placed) and cand.top() >= plot.top()
                            and cand.bottom() <= plot.bottom()):
                        box, baseline = cand, base
                        break
                if box:
                    break
            if box is None:
                continue
            placed.append(box)
            self._text(painter, box.x(), baseline, text, self.small_font, color)


class SettingsWindow(QWidget):
    def __init__(self, owner):
        super().__init__()
        self.owner = owner
        self.setWindowTitle("WT Energy · 设置")
        self.resize(600, 730)
        self.setMinimumSize(450, 400)
        outer = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        outer.addWidget(scroll)
        body = QWidget()
        scroll.setWidget(body)
        layout = QVBoxLayout(body)
        title = QLabel("WT ENERGY  /  游戏叠加显示")
        title.setStyleSheet("font-size: 20px; font-weight: 600")
        layout.addWidget(title)
        self.status = QLabel("等待飞行数据")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        help_text = QLabel("游戏使用「全屏窗口」模式。关闭设置后，指标仍可显示在游戏中。\n"
                           "战斗时鼠标穿透；调整位置时进入布局模式。")
        help_text.setWordWrap(True)
        layout.addWidget(help_text)
        display = QGroupBox("显示与布局")
        form = QFormLayout(display)
        layout.addWidget(display)
        self.visible_box = QCheckBox("显示 HUD")
        self.visible_box.setChecked(owner.hud_visible)
        self.visible_box.toggled.connect(owner.set_visible)
        self.layout_box = QCheckBox("布局模式：拖动指标组")
        self.layout_box.toggled.connect(owner.set_editing)
        form.addRow(self.visible_box)
        form.addRow(self.layout_box)
        self.group_boxes = {}
        group_row = QHBoxLayout()
        for key, caption in GROUPS.items():
            if key in ("climb", "turn", "rose", "scope", "side"):
                continue
            check = QCheckBox(caption)
            check.setChecked(owner.group_enabled[key])
            check.toggled.connect(lambda enabled, key=key: owner.set_group_enabled(key, enabled))
            group_row.addWidget(check)
            self.group_boxes[key] = check
        form.addRow("指标组", group_row)
        self.font_box = QFontComboBox()
        self.font_box.setCurrentFont(owner.hud_font)
        self.font_box.currentFontChanged.connect(owner.change_font)
        form.addRow("字体", self.font_box)
        self.font_size = QSpinBox()
        self.font_size.setRange(9, 28)
        self.font_size.setValue(owner.hud_font.pointSize())
        self.font_size.valueChanged.connect(owner.change_font_size)
        form.addRow("字号", self.font_size)
        self.color_button = QPushButton("选择强调色…")
        self.color_button.clicked.connect(owner.choose_color)
        form.addRow("颜色", self.color_button)
        self.screen_box = QComboBox()
        self.screen_box.currentIndexChanged.connect(owner.change_screen)
        form.addRow("预览 / 备用屏幕", self.screen_box)
        self.follow_box = QCheckBox("自动跟随战雷窗口的位置和尺寸")
        self.follow_box.setChecked(owner.follow_game)
        self.follow_box.toggled.connect(owner.set_follow_game)
        self.hide_box = QCheckBox("切出游戏或最小化时隐藏（演示 / 布局除外）")
        self.hide_box.setChecked(owner.hide_outside)
        self.hide_box.toggled.connect(owner.set_hide_outside)
        form.addRow(self.follow_box)
        form.addRow(self.hide_box)
        self.surface_status = QLabel()
        self.surface_status.setWordWrap(True)
        form.addRow(self.surface_status)
        reset = QPushButton("恢复默认位置")
        reset.clicked.connect(owner.reset_positions)
        form.addRow(reset)
        indicators = QGroupBox("指标（勾选后显示）")
        indicator_layout = QGridLayout(indicators)
        self.indicator_boxes = {}
        for index, item in enumerate(INDICATORS):
            check = QCheckBox(item.label)
            check.setToolTip(item.description)
            check.setChecked(owner.indicator_enabled[item.key])
            check.toggled.connect(lambda enabled, key=item.key: owner.set_indicator_enabled(key, enabled))
            indicator_layout.addWidget(check, index // 2, index % 2)
            self.indicator_boxes[item.key] = check
        layout.addWidget(indicators)
        data = QGroupBox("数据与静态参考")
        form = QFormLayout(data)
        layout.addWidget(data)
        self.mode_box = QComboBox()
        self.mode_box.addItem("实时 8111", "live")
        self.mode_box.addItem("合成演示 · 非游戏实测", "demo")
        self.mode_box.currentIndexChanged.connect(lambda i: owner.command({"action": "mode", "value": self.mode_box.itemData(i)}))
        form.addRow("数据来源", self.mode_box)
        self.aircraft_box = QComboBox()
        self.aircraft_box.setEditable(True)
        self.aircraft_box.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.aircraft_box.addItem("自动识别机型", "auto")
        for aircraft in aircraft_catalog():
            label = aircraft.label
            if aircraft.name_en != aircraft.name:
                label += f" · {aircraft.name_en}"
            self.aircraft_box.addItem(label, aircraft.id)
        self.aircraft_box.addItem("FM 文件", "file")
        self.aircraft_box.model().item(self.aircraft_box.count()-1).setEnabled(False)
        self.aircraft_box.completer().setFilterMode(Qt.MatchFlag.MatchContains)
        self.aircraft_box.completer().setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.aircraft_box.setMinimumContentsLength(28)
        self.aircraft_box.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.aircraft_box.activated.connect(self.select_aircraft)
        form.addRow("机型", self.aircraft_box)
        self.model_label = QLabel("未加载 FM")
        self.model_label.setWordWrap(True)
        form.addRow("当前模型", self.model_label)
        choose_model = QPushButton("手动载入 FM 文件…")
        choose_model.clicked.connect(owner.choose_model)
        form.addRow(choose_model)
        self.sweep = QSpinBox()
        self.sweep.setRange(0, 100)
        self.sweep.setSuffix(" %")
        self.sweep.setSingleStep(5)
        self.sweep.setKeyboardTracking(False)
        self.sweep.valueChanged.connect(lambda value: owner.command({"action": "sweep", "fraction": value/100}))
        self.sweep_label = QLabel("固定参考后掠")
        form.addRow(self.sweep_label, self.sweep)
        row = QHBoxLayout()
        self.mass = QLineEdit()
        self.mass.setPlaceholderText("留空：空重 + 燃油 + 挂载自动估算")
        self.mass.returnPressed.connect(self.apply_mass)
        row.addWidget(self.mass)
        apply_mass = QPushButton("应用")
        apply_mass.clicked.connect(self.apply_mass)
        row.addWidget(apply_mass)
        form.addRow("总质量 / kg", row)
        self.payload = QSpinBox()
        self.payload.setRange(0, 50000)
        self.payload.setSingleStep(100)
        self.payload.setKeyboardTracking(False)
        self.payload.setToolTip("自动估算总质量时加上的武器、吊舱与弹药质量；8111 不提供这些数据。")
        self.payload.valueChanged.connect(lambda value: owner.command({"action": "payload", "kg": float(value)}))
        form.addRow("挂载 / kg", self.payload)
        self.afterburner_box = QCheckBox("允许最大推力（无油门读数时按此档计算；关闭则为全军推）")
        self.afterburner_box.toggled.connect(lambda enabled: owner.command({"action": "afterburner", "enabled": enabled}))
        form.addRow(self.afterburner_box)
        climb = QGroupBox("爬升引导")
        form = QFormLayout(climb)
        layout.addWidget(climb)
        self.climb_box = QCheckBox("启用爬升引导 · Ctrl+Alt+C")
        self.climb_box.toggled.connect(owner.set_climb_enabled)
        form.addRow(self.climb_box)
        self.climb_altitude = QSpinBox()
        self.climb_altitude.setRange(100, 20000)
        self.climb_altitude.setSingleStep(500)
        self.climb_altitude.setValue(round(owner.climb_request.target_altitude_m))
        form.addRow("目标高度 / m", self.climb_altitude)
        self.climb_speed = QLineEdit()
        self.climb_speed.setPlaceholderText("留空自动选择")
        if owner.climb_request.minimum_tas_mps is not None:
            self.climb_speed.setText(f"{owner.climb_request.minimum_tas_mps * 3.6:g}")
        self.climb_speed.returnPressed.connect(self.apply_climb_target)
        form.addRow("到达最低 TAS / km/h", self.climb_speed)
        apply_climb = QPushButton("应用目标")
        apply_climb.clicked.connect(self.apply_climb_target)
        form.addRow(apply_climb)
        turn = QGroupBox("键盘转向引导")
        form = QFormLayout(turn)
        layout.addWidget(turn)
        self.turn_box = QCheckBox("启用转向引导 · Ctrl+Alt+T")
        self.turn_box.toggled.connect(owner.set_turn_enabled)
        form.addRow(self.turn_box)
        self.pose_status = QLabel()
        self.pose_status.setWordWrap(True)
        form.addRow("姿态来源", self.pose_status)
        self.pose_sign = QComboBox()
        self.pose_sign.addItem("正向 Wx", 1)
        self.pose_sign.addItem("反向 Wx", -1)
        self.pose_sign.setCurrentIndex(1 if owner.preferences.value("turn/pose_sign", 1, type=int) == -1 else 0)
        form.addRow("滚转率方向", self.pose_sign)
        self.calibrate_pose = QPushButton("平翼校准 · Ctrl+Alt+A · 机翼水平且停止滚转")
        self.calibrate_pose.clicked.connect(self.apply_pose_calibration)
        form.addRow(self.calibrate_pose)
        self.turn_angle = QComboBox()
        for value in (30, 45, 90, 120):
            self.turn_angle.addItem(f"{value}°", value)
        self.turn_angle.setCurrentIndex(self.turn_angle.findData(owner.turn_settings.angle_deg))
        form.addRow("速度方向转角", self.turn_angle)
        self.turn_fields = {}
        for key, label, low, high, step, factor in (
            ("max_altitude_loss_m", "允许损失高度 / m", 0, 5000, 100, 1),
            ("minimum_tas_mps", "最低 TAS / km/h", 180, 2340, 36, 3.6),
            ("horizon_s", "预测时长上限 / s", 3, 30, 1, 1),
            ("max_load", "正向载荷上限 / g", 1, 12, .5, 1),
            ("min_load", "负向载荷下限 / g", -5, 0, .5, 1),
            ("roll_rate_deg_s", "满滚转参考速度 / °/s", 10, 360, 10, 1),
            ("roll_response_s", "滚转响应时间 / s", .1, 2, .05, 1),
            ("load_response_s", "俯仰响应时间 / s", .1, 3, .1, 1),
            ("reaction_s", "操纵反应时间 / s", 0, 1.5, .05, 1),
            ("hold_s", "每段最短时间 / s", .8, 3, .1, 1),
            ("throttle_rate_percent_s", "油门变化速度 / %/s", 5, 200, 5, 1),
            ("engine_response_s", "推力响应时间 / s", .1, 5, .1, 1),
        ):
            widget = QDoubleSpinBox()
            widget.setRange(low, high)
            widget.setDecimals(2 if high <= 3 else 1)
            widget.setSingleStep(step)
            widget.setValue(getattr(owner.turn_settings, key)*factor)
            self.turn_fields[key] = (widget, factor)
            form.addRow(label, widget)
        apply_turn = QPushButton("应用转向设置")
        apply_turn.clicked.connect(self.apply_turn_settings)
        form.addRow(apply_turn)
        restart_turn = QPushButton("重新开始转向 · Ctrl+Alt+R")
        restart_turn.clicked.connect(owner.restart_turn)
        form.addRow(restart_turn)
        offense = QGroupBox("进攻辅助")
        form = QFormLayout(offense)
        layout.addWidget(offense)
        self.rose_box = QCheckBox("显示进攻动向图 · Ctrl+Alt+K")
        self.rose_box.setChecked(owner.rose_enabled)
        self.rose_box.toggled.connect(owner.set_rose_enabled)
        form.addRow(self.rose_box)
        self.offense_missile = QComboBox()
        for missile in available():
            self.offense_missile.addItem(MISSILE_NAMES.get(missile, missile), missile)
        self.offense_enemy = QComboBox()
        for ident in ENEMY_AIRCRAFT:
            profile = find_aircraft(ident)
            if profile is not None:
                self.offense_enemy.addItem(profile.name, ident)
        self.offense_skill = QComboBox()
        for key in SKILLS:
            self.offense_skill.addItem({"normal": "普通（85% 会躲，其中 60% 做对）",
                                        "top": "高手（总能躲，且动作正确）"}[key], key)
        self.offense_mode = QComboBox()
        for key, label in MODES.items():
            self.offense_mode.addItem(label, key)
        self.offense_speed = QComboBox()
        for kmh in TARGET_SPEEDS_KMH:
            self.offense_speed.addItem(f"{kmh} km/h", kmh)
        for box, key, value in ((self.offense_missile, "missile", owner.offense_missile),
                                (self.offense_enemy, "enemy", owner.offense_enemy),
                                (self.offense_skill, "skill", owner.offense_skill),
                                (self.offense_mode, "mode", owner.offense_mode),
                                (self.offense_speed, "target_kmh", owner.offense_target_kmh)):
            box.setCurrentIndex(max(0, box.findData(value)))
            box.currentIndexChanged.connect(lambda i, box=box, key=key: owner.set_offense_option(key, box.itemData(i)))
        form.addRow("导弹", self.offense_missile)
        form.addRow("假设敌机", self.offense_enemy)
        form.addRow("敌机速度", self.offense_speed)
        form.addRow("对手水平", self.offense_skill)
        form.addRow("发射方式", self.offense_mode)
        offense_help = QLabel("颜色为现在发射的命中概率：≥50% 红、25–50% 黄、<25% 灰、打不到暗色。主动弹无发射告警；"
                              "对手在导弹开机（约 16 km）、看到动力段尾焰或被 STT 锁定时察觉。网络训练范围 2–45 km。")
        offense_help.setWordWrap(True)
        form.addRow(offense_help)
        self.scope_box = QCheckBox("显示 B 显发射区 · Ctrl+Alt+B")
        self.scope_box.setChecked(owner.scope_enabled)
        self.scope_box.toggled.connect(owner.set_scope_enabled)
        form.addRow(self.scope_box)
        self.scope_scale = QComboBox()
        for km in SCOPE_RANGES_KM:
            self.scope_scale.addItem(f"{km} km", km)
        self.scope_scale.setCurrentIndex(max(0, self.scope_scale.findData(owner.scope_scale_km)))
        self.scope_scale.currentIndexChanged.connect(
            lambda i: owner.set_scope_geometry(self.scope_scale.itemData(i), owner.scope_half_azimuth_deg))
        form.addRow("B 显量程（与游戏一致）", self.scope_scale)
        self.scope_azimuth = QComboBox()
        for deg in SCOPE_AZIMUTHS_DEG:
            self.scope_azimuth.addItem(f"±{deg}°", deg)
        self.scope_azimuth.setCurrentIndex(max(0, self.scope_azimuth.findData(owner.scope_half_azimuth_deg)))
        self.scope_azimuth.currentIndexChanged.connect(
            lambda i: owner.set_scope_geometry(owner.scope_scale_km, self.scope_azimuth.itemData(i)))
        form.addRow("B 显方位范围", self.scope_azimuth)
        self.side_box = QCheckBox("显示高度侧视图 · Ctrl+Alt+V")
        self.side_box.setChecked(owner.side_enabled)
        self.side_box.toggled.connect(owner.set_side_enabled)
        form.addRow(self.side_box)
        side_help = QLabel("高度侧视图：横轴 = 发射距离（量程与上面的 B 显量程相同），纵轴 = 目标高度减本机高度"
                           "（目标更高在上）。线型与 B 显发射区一致，为迎头直线飞行的发射区边界。")
        side_help.setWordWrap(True)
        form.addRow(side_help)
        scope_help = QLabel("B 显发射区：进入布局模式，把框拖到游戏 B 显上，拖右下角调整到同样大小。"
                            "橙实线 = 迎头 Rmax，橙虚线 = 背离 Rmax，红虚线 = 迎头对方不足 3 s，红实线 = 迎头无逃逸。"
                            "目标假设为同高度直线飞行；量程切换需在这里同步。")
        scope_help.setWordWrap(True)
        form.addRow(scope_help)
        rose_help = QLabel("方向 = 目标光点在 B 显上的移动方向（往下冲 = 迎头），圈 = 发射距离，"
                           "颜色 = 现在发射时对方能用来反应的秒数：红 < 3 s，黄 3–6 s，灰更久。"
                           "光点正弯向笔直往下时比图上更好打，弯向横向时更难打。"
                           "离线表按本机高度与真空速插值；对方速度 1000 km/h、同高度、9 g、箔条比值均为假设。")
        rose_help.setWordWrap(True)
        form.addRow(rose_help)
        self.error = QLabel()
        self.error.setWordWrap(True)
        self.error.setStyleSheet("color: #ce562e")
        layout.addWidget(self.error)
        self.hotkey_status = QLabel()
        self.hotkey_status.setWordWrap(True)
        layout.addWidget(self.hotkey_status)
        self.notes = QTextEdit()
        self.notes.setReadOnly(True)
        self.notes.setMinimumHeight(120)
        layout.addWidget(self.notes)
        buttons = QHBoxLayout()
        done = QPushButton("完成布局并收起设置")
        done.clicked.connect(self.close)
        quit_button = QPushButton("退出程序")
        quit_button.clicked.connect(owner.close)
        buttons.addWidget(done)
        buttons.addWidget(quit_button)
        outer.addLayout(buttons)

    def select_aircraft(self, index):
        identity = self.aircraft_box.itemData(index)
        if identity and identity != "file":
            self.owner.command({"action": "aircraft", "id": identity})

    def apply_pose_calibration(self):
        sign = self.pose_sign.currentData()
        if self.owner.command({"action": "pose_calibrate", "roll_sign": sign}):
            self.owner.preferences.setValue("turn/pose_sign", sign)

    def apply_mass(self):
        if not self.mass.text().strip():
            self.owner.command({"action": "mass", "kg": None})
            return
        try:
            value = float(self.mass.text().strip())
            if not math.isfinite(value) or value <= 0:
                raise ValueError
        except ValueError:
            self.error.setText("请输入大于 0 的总质量（kg）。")
            return
        self.owner.command({"action": "mass", "kg": value})

    def apply_climb_target(self):
        try:
            text = self.climb_speed.text().strip()
            request = ClimbRequest(self.climb_altitude.value(), float(text)/3.6 if text else None)
            validate_request(request)
        except ValueError as exc:
            self.error.setText(str(exc))
            return False
        return self.owner.set_climb_target(request)

    def apply_turn_settings(self):
        settings = KeyboardTurnSettings(angle_deg=self.turn_angle.currentData(),
            **{key: widget.value()/factor for key, (widget, factor) in self.turn_fields.items()})
        try:
            validate_settings(settings)
        except ValueError as exc:
            self.error.setText(str(exc))
            return False
        return self.owner.set_turn_settings(settings)

    def closeEvent(self, event):
        if self.owner.closed:
            event.accept()
        elif self.owner.can_reopen_settings:
            self.owner.set_editing(False)
            self.hide()
            event.ignore()
        else:
            # Never strand a click-through HUD without a tray or a settings hotkey.
            self.owner.close()
            event.accept()


class HotkeyFilter(QAbstractNativeEventFilter):
    def __init__(self, callbacks):
        super().__init__()
        self.callbacks = callbacks

    def nativeEventFilter(self, event_type, message):
        if bytes(event_type) in (b"windows_generic_MSG", b"windows_dispatcher_MSG"):
            msg = ctypes.cast(int(message), ctypes.POINTER(wintypes.MSG)).contents
            if msg.message == 0x0312 and int(msg.wParam) in self.callbacks:
                QTimer.singleShot(0, self.callbacks[int(msg.wParam)])
                return True, 0
        return False, 0


class OverlayApp:
    def __init__(self, get_snapshot: Callable[[], OverlaySnapshot], on_command: Callable[[dict], None],
                 *, title="WT Energy", settings: QSettings | None = None, show_on_start=True,
                 climb_request: ClimbRequest | None = None,
                 turn_settings: KeyboardTurnSettings | None = None):
        self.app = QApplication.instance() or QApplication([sys.argv[0]])
        self.app.setQuitOnLastWindowClosed(False)
        self.app.setApplicationName(title)
        self._get_snapshot, self._on_command = get_snapshot, on_command
        self.closed = False
        self.preferences = settings if settings is not None else QSettings(
            QSettings.Format.IniFormat, QSettings.Scope.UserScope, "WT Energy", "Overlay")
        self.hud_visible = True
        self.editing = False
        self.climb_enabled = False
        self._pending_climb_enabled = None
        self.turn_enabled = False
        self._pending_turn_enabled = None
        self.turn_settings = turn_settings or KeyboardTurnSettings()
        if turn_settings is None and self.preferences.contains("turn/settings"):
            try:
                saved_turn = KeyboardTurnSettings(**json.loads(self.preferences.value("turn/settings", "{}", type=str)))
                if saved_turn.hold_s < .8:
                    saved_turn = replace(saved_turn, hold_s=1.2)
                validate_settings(saved_turn)
                self.turn_settings = saved_turn
            except (ValueError, TypeError):
                pass
        self.climb_request = climb_request or ClimbRequest()
        if climb_request is None and self.preferences.contains("climb/altitude_m"):
            try:
                speed = self.preferences.value("climb/speed_mps", "", type=str)
                saved = ClimbRequest(self.preferences.value("climb/altitude_m", 8000, type=float),
                                     float(speed) if speed else None)
                validate_request(saved)
                self.climb_request = saved
            except (ValueError, TypeError):
                pass
        self.follow_game = self.preferences.value("follow_game", True, type=bool)
        self.hide_outside = self.preferences.value("hide_outside", True, type=bool)
        self.screen_name = self.preferences.value("screen", "", type=str)
        family = self.preferences.value("font_family", "Microsoft YaHei UI" if sys.platform == "win32" else "PingFang SC", type=str)
        size = max(9, min(28, self.preferences.value("font_size", 11, type=int)))
        if self.preferences.value("hud_format", 1, type=int) < 2:
            # Migrate existing profiles once; subsequent font choices are preserved.
            size = min(size, 11)
            self.preferences.setValue("font_size", size)
            self.preferences.setValue("hud_format", 2)
        self.hud_font = QFont(family, size)
        self.hud_font.setWeight(QFont.Weight.DemiBold)
        self.color = self.preferences.value("accent", "#71e3ce", type=str)
        if not QColor(self.color).isValid():
            self.color = "#71e3ce"
        self.indicator_enabled = {item.key: self.preferences.value(
            f"indicators/{item.key}", item.enabled, type=bool) for item in INDICATORS}
        self.rose_enabled = self.preferences.value("offense/enabled", False, type=bool)
        missiles = available()
        saved = self.preferences.value("offense/missile", "", type=str)
        self.offense_missile = saved if saved in missiles else (missiles[0] if missiles else "")
        enemy = self.preferences.value("offense/enemy", ENEMY_AIRCRAFT[0], type=str)
        self.offense_enemy = enemy if enemy in ENEMY_AIRCRAFT else ENEMY_AIRCRAFT[0]
        skill = self.preferences.value("offense/skill", "normal", type=str)
        self.offense_skill = skill if skill in SKILLS else "normal"
        mode = self.preferences.value("offense/mode", "tws", type=str)
        self.offense_mode = mode if mode in MODES else "tws"
        kmh = self.preferences.value("offense/target_kmh", 1000, type=int)
        self.offense_target_kmh = kmh if kmh in TARGET_SPEEDS_KMH else 1000
        self.advisor = None
        self._offense_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="offense")
        self._envelope_job = None   # (key, future) of the envelope being computed
        self._envelope_cache = (None, None)  # (key, envelope)
        self._rose_cache = (None, None)
        self.scope_enabled = self.preferences.value("offense/scope_enabled", False, type=bool)
        self.side_enabled = self.preferences.value("offense/side_enabled", False, type=bool)
        self.scope_scale_km = self.preferences.value("offense/scope_scale_km", 40, type=int)
        self.scope_scale_km = self.scope_scale_km if self.scope_scale_km in SCOPE_RANGES_KM else 40
        self.scope_half_azimuth_deg = self.preferences.value("offense/scope_half_azimuth_deg", 60, type=int)
        if self.scope_half_azimuth_deg not in SCOPE_AZIMUTHS_DEG:
            self.scope_half_azimuth_deg = 60
        self.groups, self.group_enabled = {}, {}
        for key in GROUPS:
            group = {"rose": RoseGroup, "scope": ScopeGroup, "side": SideGroup}.get(key, HudGroup)(key, self.hud_font, self.color)
            if key == "scope":
                group.set_box(self.preferences.value("groups/scope/w", 260, type=int),
                              self.preferences.value("groups/scope/h", 260, type=int))
            x, y = DEFAULT_POSITIONS[key]
            position = (self.preferences.value(f"groups/{key}/x", x, type=float),
                        self.preferences.value(f"groups/{key}/y", y, type=float))
            group.fraction = tuple(min(1.0, max(0.0, p)) if math.isfinite(p) else d
                                   for p, d in zip(position, (x, y)))
            group.moved.connect(self.save_position)
            self.groups[key] = group
            self.group_enabled[key] = self.preferences.value(f"groups/{key}/enabled", True, type=bool)
        self.desktop = WindowsDesktop() if sys.platform == "win32" else None
        self.game = None
        self.snapshot = OverlaySnapshot("live", "等待飞行数据")
        self.settings_window = SettingsWindow(self)
        self._load_advisor()
        self._setup_tray()
        self._setup_hotkeys()
        if climb_request is None and self.preferences.contains("climb/altitude_m"):
            self.set_climb_target(self.climb_request)
        if turn_settings is None and self.preferences.contains("turn/settings"):
            self.set_turn_settings(self.turn_settings)
        self.can_reopen_settings = self.tray.isVisible() or self.settings_hotkey_available
        self.app.screenAdded.connect(self._screens_changed)
        self.app.screenRemoved.connect(self._screens_changed)
        self._screens_changed()
        self.timer = QTimer()
        self.timer.timeout.connect(self.refresh)
        self.timer.start(100)
        self.surface_timer = QTimer()
        self.surface_timer.timeout.connect(self.refresh_surface)
        self.surface_timer.start(250)
        self.refresh()
        self.refresh_surface()
        self.app.aboutToQuit.connect(self.close)
        if show_on_start:
            self.show_settings()

    def _setup_tray(self):
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor(self.color), 3))
        painter.drawEllipse(3, 3, 26, 26)
        painter.drawLine(9, 21, 16, 10)
        painter.drawLine(16, 10, 23, 21)
        painter.end()
        self.tray = QSystemTrayIcon(QIcon(pixmap), self.settings_window)
        self.tray.setToolTip("WT Energy · 透明 HUD")
        self.tray_menu = QMenu()
        self.visible_action = QAction("显示 HUD", self.tray_menu, checkable=True, checked=True)
        self.visible_action.toggled.connect(self.set_visible)
        self.edit_action = QAction("布局模式", self.tray_menu, checkable=True)
        self.edit_action.toggled.connect(self.set_editing)
        self.tray_menu.addAction(self.visible_action)
        self.tray_menu.addAction(self.edit_action)
        self.climb_action = QAction("爬升引导", self.tray_menu, checkable=True)
        self.climb_action.toggled.connect(self.set_climb_enabled)
        self.tray_menu.addAction(self.climb_action)
        self.turn_action = QAction("转向引导", self.tray_menu, checkable=True)
        self.turn_action.toggled.connect(self.set_turn_enabled)
        self.tray_menu.addAction(self.turn_action)
        self.rose_action = QAction("进攻动向图", self.tray_menu, checkable=True, checked=self.rose_enabled)
        self.rose_action.toggled.connect(self.set_rose_enabled)
        self.tray_menu.addAction(self.rose_action)
        self.scope_action = QAction("B 显发射区", self.tray_menu, checkable=True, checked=self.scope_enabled)
        self.scope_action.toggled.connect(self.set_scope_enabled)
        self.tray_menu.addAction(self.scope_action)
        self.side_action = QAction("高度侧视图", self.tray_menu, checkable=True, checked=self.side_enabled)
        self.side_action.toggled.connect(self.set_side_enabled)
        self.tray_menu.addAction(self.side_action)
        self.tray_menu.addAction("重新开始转向", self.restart_turn)
        self.tray_menu.addAction("设置…", self.show_settings)
        self.tray_menu.addSeparator()
        self.tray_menu.addAction("退出", self.close)
        self.tray.setContextMenu(self.tray_menu)
        self.tray.activated.connect(lambda reason: self.show_settings()
            if reason == QSystemTrayIcon.ActivationReason.DoubleClick else None)
        if QSystemTrayIcon.isSystemTrayAvailable():
            self.tray.show()

    def _setup_hotkeys(self):
        callbacks = {"O": lambda: self.set_visible(not self.hud_visible),
                     "L": lambda: self.set_editing(not self.editing), "S": self.show_settings,
                     "C": lambda: self.set_climb_enabled(not self.climb_enabled),
                     "T": lambda: self.set_turn_enabled(not self.turn_enabled), "R": self.restart_turn,
                     "A": self.settings_window.apply_pose_calibration,
                     "K": lambda: self.set_rose_enabled(not self.rose_enabled),
                     "B": lambda: self.set_scope_enabled(not self.scope_enabled),
                     "V": lambda: self.set_side_enabled(not self.side_enabled)}
        active, failures = {}, []
        self.settings_hotkey_available = False
        if self.desktop:
            for identifier, (letter, callback) in enumerate(callbacks.items(), 0x5740):
                if self.desktop.register_hotkey(identifier, letter):
                    active[identifier] = callback
                    if letter == "S":
                        self.settings_hotkey_available = True
                else:
                    failures.append("Ctrl+Alt+" + letter)
        self.hotkey_filter = HotkeyFilter(active)
        if active:
            self.app.installNativeEventFilter(self.hotkey_filter)
        text = "  ·  ".join(f"Ctrl+Alt+{key}：{value}" for key, value in HOTKEYS.items())
        if not self.desktop:
            text = "全局热键与自动跟随仅在 Windows 提供；当前可用设置界面预览布局。"
        elif failures:
            text += "\n注册失败（可能已被占用）：" + "、".join(failures) + "。可使用托盘或设置。"
        self.settings_window.hotkey_status.setText(text)

    @staticmethod
    def _checked(widget, value):
        blocked = widget.blockSignals(True)
        widget.setChecked(value)
        widget.blockSignals(blocked)

    def set_visible(self, enabled):
        self.hud_visible = enabled
        if not enabled and self.editing:
            self.set_editing(False)
        self._checked(self.settings_window.visible_box, enabled)
        self._checked(self.visible_action, enabled)
        self._sync_surface()

    def set_editing(self, enabled):
        self.editing = enabled
        if enabled:
            self.set_visible(True)
        for group in self.groups.values():
            group.set_editing(enabled)
        self._checked(self.settings_window.layout_box, enabled)
        self._checked(self.edit_action, enabled)
        self._sync_surface()
        self.preferences.sync()

    def set_group_enabled(self, key, enabled):
        self.group_enabled[key] = enabled
        self.preferences.setValue(f"groups/{key}/enabled", enabled)
        self._sync_surface()

    def set_climb_target(self, request):
        if not self.command({"action": "climb_target", "altitude_m": request.target_altitude_m,
                             "minimum_tas_mps": request.minimum_tas_mps}):
            return False
        self.climb_request = request
        self.preferences.setValue("climb/altitude_m", request.target_altitude_m)
        self.preferences.setValue("climb/speed_mps", "" if request.minimum_tas_mps is None else request.minimum_tas_mps)
        return True

    def set_climb_enabled(self, enabled):
        # Apply the visible target before enabling; hiding never waits for the worker.
        if enabled and not self.settings_window.apply_climb_target():
            self._checked(self.settings_window.climb_box, self.climb_enabled)
            self._checked(self.climb_action, self.climb_enabled)
            return
        if enabled and self.turn_enabled:
            self.set_turn_enabled(False)
        if self.command({"action": "climb_enabled", "enabled": enabled}):
            self.climb_enabled = enabled
            self._pending_climb_enabled = enabled
        self._checked(self.settings_window.climb_box, self.climb_enabled)
        self._checked(self.climb_action, self.climb_enabled)
        self._sync_surface()

    def set_turn_settings(self, settings):
        if not self.command({"action": "turn_target", "settings": asdict(settings)}):
            return False
        self.turn_settings = settings
        self.preferences.setValue("turn/settings", json.dumps(asdict(settings)))
        return True

    def set_turn_enabled(self, enabled):
        if enabled and not self.settings_window.apply_turn_settings():
            self._checked(self.settings_window.turn_box, self.turn_enabled)
            self._checked(self.turn_action, self.turn_enabled)
            return
        if enabled and self.climb_enabled:
            self.set_climb_enabled(False)
        if self.command({"action": "turn_enabled", "enabled": enabled}):
            self.turn_enabled = enabled
            self._pending_turn_enabled = enabled
        self._checked(self.settings_window.turn_box, self.turn_enabled)
        self._checked(self.turn_action, self.turn_enabled)
        self._sync_surface()

    def restart_turn(self):
        if self.turn_enabled:
            self.command({"action": "turn_restart"})

    def set_rose_enabled(self, enabled):
        self.rose_enabled = enabled
        self.preferences.setValue("offense/enabled", enabled)
        self._checked(self.settings_window.rose_box, enabled)
        self._checked(self.rose_action, enabled)
        self._sync_surface()

    def set_scope_enabled(self, enabled):
        self.scope_enabled = enabled
        self.preferences.setValue("offense/scope_enabled", enabled)
        self._checked(self.settings_window.scope_box, enabled)
        self._checked(self.scope_action, enabled)
        self._sync_surface()

    def set_side_enabled(self, enabled):
        self.side_enabled = enabled
        self.preferences.setValue("offense/side_enabled", enabled)
        self._checked(self.settings_window.side_box, enabled)
        self._checked(self.side_action, enabled)
        self._sync_surface()

    def set_scope_geometry(self, scale_km, half_azimuth_deg):
        self.scope_scale_km, self.scope_half_azimuth_deg = int(scale_km), int(half_azimuth_deg)
        self.preferences.setValue("offense/scope_scale_km", self.scope_scale_km)
        self.preferences.setValue("offense/scope_half_azimuth_deg", self.scope_half_azimuth_deg)
        self.refresh()

    def set_offense_option(self, key, value):
        """Missile, assumed enemy aircraft and speed, opponent skill and launch mode for the offensive aids."""
        setattr(self, f"offense_{key}", value)
        self.preferences.setValue(f"offense/{key}", value)
        self._load_advisor()
        self.refresh()

    def _load_advisor(self):
        """Hit-probability model for the chosen missile under the current target assumption."""
        self.advisor = None
        self._envelope_job, self._envelope_cache, self._rose_cache = None, (None, None), (None, None)
        if not self.offense_missile:
            return
        assumption = pk.Assumption(aircraft=self.offense_enemy, speed_kmh=float(self.offense_target_kmh),
                                   skill=self.offense_skill, mode=self.offense_mode)
        try:
            self.advisor = OffenseAdvisor(self.offense_missile, assumption)
        except (OSError, ValueError, KeyError) as exc:
            self.settings_window.error.setText(f"命中概率模型不可用：{exc}")

    @staticmethod
    def _offense_key(state):
        # Recompute only when ownship altitude or TAS has moved noticeably.
        return round(state.altitude_m / 200.0), round(state.tas_mps * 3.6 / 20.0)

    def _offense_caption(self):
        enemy = find_aircraft(self.offense_enemy)
        return (f"{MISSILE_NAMES.get(self.offense_missile, self.offense_missile)} · {SKILLS[self.offense_skill]}"
                f" · {MODES[self.offense_mode]} · 假设 {enemy.name if enemy else self.offense_enemy} "
                f"{self.offense_target_kmh} km/h")

    def _envelope_for(self, state):
        """The envelope for this ownship state: cached, computed in the background (the first one inline)."""
        key = self._offense_key(state)
        job = self._envelope_job
        if job is not None and job[1].done():
            try:
                self._envelope_cache = (job[0], job[1].result())
            except (ValueError, OSError):
                self._envelope_cache = (job[0], None)
            self._envelope_job = job = None
        cached_key, cached = self._envelope_cache
        if cached_key == key:
            return cached
        if cached is None:
            envelope = self.advisor.envelope(state.altitude_m, state.tas_mps)
            self._envelope_cache = (key, envelope)
            return envelope
        if job is None:
            self._envelope_job = (key, self._offense_pool.submit(self.advisor.envelope, state.altitude_m,
                                                                 state.tas_mps))
        return cached

    def _update_scope(self, snapshot):
        group, side, library, state = (self.groups["scope"], self.groups["side"], self.advisor, snapshot.state)
        for key in ("scope", "side"):
            self.groups[key].set_content(HudContent(GROUPS[key], (), demo=snapshot.mode == "demo"))
        scale_m = self.scope_scale_km * 1000.
        if (library is None or state is None or not state.valid or state.altitude_m is None
                or state.tas_mps is None or not math.isfinite(state.altitude_m) or not math.isfinite(state.tas_mps)):
            group.set_envelope(None, scale_m, float(self.scope_half_azimuth_deg))
            side.set_envelope(None, scale_m)
            return
        envelope = self._envelope_for(state)
        group.set_envelope(envelope, scale_m, float(self.scope_half_azimuth_deg))
        side.set_envelope(envelope, scale_m)

    def _update_rose(self, snapshot):
        group, library, state = self.groups["rose"], self.advisor, snapshot.state
        group.set_content(HudContent(GROUPS["rose"], (), demo=snapshot.mode == "demo"))
        if (library is None or state is None or not state.valid or state.altitude_m is None
                or state.tas_mps is None or not math.isfinite(state.altitude_m) or not math.isfinite(state.tas_mps)):
            group.set_rose(None)
            return
        key = self._offense_key(state)
        if self._rose_cache[0] != key:
            self._rose_cache = (key, library.rose(state.altitude_m, state.tas_mps))
        group.set_rose(self._rose_cache[1], self._offense_caption())

    def set_indicator_enabled(self, key, enabled):
        self.indicator_enabled[key] = enabled
        self.preferences.setValue(f"indicators/{key}", enabled)
        self.refresh()

    def save_position(self, key):
        x, y = self.groups[key].fraction
        self.preferences.setValue(f"groups/{key}/x", x)
        self.preferences.setValue(f"groups/{key}/y", y)
        if key == "scope":
            width, height = self.groups[key].box
            self.preferences.setValue("groups/scope/w", width)
            self.preferences.setValue("groups/scope/h", height)
        self.preferences.sync()

    def reset_positions(self):
        for key, group in self.groups.items():
            group.fraction = DEFAULT_POSITIONS[key]
            self.save_position(key)
        self._sync_surface()

    def change_font(self, font):
        self.hud_font.setFamily(font.family())
        self._apply_style()

    def change_font_size(self, size):
        self.hud_font.setPointSize(size)
        self._apply_style()

    def choose_color(self):
        color = QColorDialog.getColor(QColor(self.color), self.settings_window, "选择强调色")
        if color.isValid():
            self.color = color.name()
            self._apply_style()

    def _apply_style(self):
        self.preferences.setValue("font_family", self.hud_font.family())
        self.preferences.setValue("font_size", self.hud_font.pointSize())
        self.preferences.setValue("accent", self.color)
        for group in self.groups.values():
            group.set_style(self.hud_font, self.color)
        self._sync_surface()

    def _screens_changed(self, *args):
        box = self.settings_window.screen_box
        box.blockSignals(True)
        box.clear()
        for screen in self.app.screens():
            box.addItem(screen.name() or "主屏幕", screen.name())
        box.setCurrentIndex(max(0, box.findData(self.screen_name)))
        box.blockSignals(False)
        self._sync_surface()

    def change_screen(self, index):
        self.screen_name = self.settings_window.screen_box.itemData(index) or ""
        self.preferences.setValue("screen", self.screen_name)
        self._sync_surface()

    def set_follow_game(self, enabled):
        self.follow_game = enabled
        self.preferences.setValue("follow_game", enabled)
        self._sync_surface()

    def set_hide_outside(self, enabled):
        self.hide_outside = enabled
        self.preferences.setValue("hide_outside", enabled)
        self._sync_surface()

    def refresh_surface(self):
        if self.closed:
            return
        self.game = self.desktop.game_window() if self.desktop else None
        self._sync_surface()

    def _sync_surface(self):
        screens = self.app.screens()
        if not screens:
            return
        screen = next((s for s in screens if s.name() == self.screen_name), self.app.primaryScreen())
        viewport = screen.availableGeometry()
        game = self.game
        demo = self.snapshot.mode == "demo"
        if game and self.follow_game and not game.minimized and not demo:
            game_screen = next((s for s in screens if s.name() == game.monitor_name), screen)
            area = game_geometry(game, game_screen)
            if area.width() > 0 and area.height() > 0:
                viewport = area
        allowed = self.hud_visible and (self.editing or demo or not self.hide_outside
                                        or not self.desktop or bool(game and game.foreground and not game.minimized))
        for key, group in self.groups.items():
            group.place(viewport)
            enabled = (self.climb_enabled if key == "climb" else self.turn_enabled if key == "turn"
                       else self.rose_enabled if key == "rose" else self.scope_enabled if key == "scope"
                       else self.side_enabled if key == "side"
                       else self.group_enabled[key])
            has_data = group.has_data() if key in ("rose", "scope", "side") else bool(group.content.rows)
            visible = allowed and enabled and (has_data or self.editing)
            if group.isVisible() != visible:
                group.setVisible(visible)
        status = ("布局模式 · 拖动边框，Ctrl+Alt+L 完成" if self.editing else
                  "合成演示 · 使用所选屏幕" if demo else
                  "未检测到战雷窗口 · 可进入布局模式预览" if self.desktop and not game else
                  "战雷已最小化" if game and game.minimized else
                  "已识别战雷窗口" if game else "当前为跨平台预览；Windows 游戏覆盖仍需实测")
        self.settings_window.surface_status.setText(status)

    def refresh(self):
        if self.closed:
            return
        try:
            snapshot = self._get_snapshot()
        except Exception as exc:
            snapshot = OverlaySnapshot(self.snapshot.mode, f"读取状态失败：{exc}")
        self.snapshot = snapshot
        if self._pending_climb_enabled == snapshot.climb_enabled:
            self._pending_climb_enabled = None
        if self._pending_climb_enabled is None:
            self.climb_enabled = snapshot.climb_enabled
        self._checked(self.settings_window.climb_box, self.climb_enabled)
        self._checked(self.climb_action, self.climb_enabled)
        if self._pending_turn_enabled == snapshot.turn_enabled:
            self._pending_turn_enabled = None
        if self._pending_turn_enabled is None:
            self.turn_enabled = snapshot.turn_enabled
        self._checked(self.settings_window.turn_box, self.turn_enabled)
        self._checked(self.turn_action, self.turn_enabled)
        enabled = {key for key, value in self.indicator_enabled.items() if value}
        for key, content in contents(snapshot, enabled).items():
            self.groups[key].set_content(content)
        self._update_rose(snapshot)
        self._update_scope(snapshot)
        window = self.settings_window
        window.status.setText(snapshot.status)
        window.pose_status.setText(snapshot.attitude_status or "等待姿态数据")
        if snapshot.model_selection == "auto":
            state = snapshot.state
            if snapshot.mode == "demo":
                model_text = "演示模式：请手动选择机型"
            elif state is None or not state.valid or not state.aircraft_id:
                model_text = "自动识别：等待游戏机型"
            elif snapshot.model_name == "未加载 FM":
                model_text = f"自动识别：未加载模型（游戏机型：{state.aircraft_id}）"
            else:
                model_text = f"自动识别：{snapshot.model_name}（游戏机型：{state.aircraft_id}）"
        else:
            model_text = snapshot.model_name
        window.model_label.setText(model_text)
        if not (window.aircraft_box.hasFocus() or window.aircraft_box.lineEdit().hasFocus()
                or window.aircraft_box.view().isVisible()):
            window.aircraft_box.blockSignals(True)
            window.aircraft_box.setCurrentIndex(window.aircraft_box.findData(snapshot.model_selection))
            window.aircraft_box.blockSignals(False)
        window.sweep.setVisible(snapshot.variable_sweep)
        window.sweep_label.setVisible(snapshot.variable_sweep)
        if not window.sweep.hasFocus():
            window.sweep.blockSignals(True)
            window.sweep.setValue(round(snapshot.sweep_fraction*100))
            window.sweep.blockSignals(False)
        window.mode_box.blockSignals(True)
        window.mode_box.setCurrentIndex(1 if snapshot.mode == "demo" else 0)
        window.mode_box.blockSignals(False)
        self._checked(window.afterburner_box, snapshot.afterburner)
        if not window.mass.text() and snapshot.mass_override_kg is not None:
            window.mass.setText(f"{snapshot.mass_override_kg:g}")
        if not window.payload.hasFocus():
            window.payload.blockSignals(True)
            window.payload.setValue(round(snapshot.payload_kg))
            window.payload.blockSignals(False)
        note_text = details(snapshot)
        if window.notes.toPlainText() != note_text:
            window.notes.setPlainText(note_text)
        self._sync_surface()

    def command(self, command):
        try:
            self._on_command(command)
        except Exception as exc:
            self.settings_window.error.setText(f"操作未完成：{exc}")
            return False
        else:
            self.settings_window.error.clear()
            return True

    def choose_model(self):
        path, _ = QFileDialog.getOpenFileName(self.settings_window, "选择飞机 FM 文件", "", "FM 文件 (*.blkx *.json *.blk);;所有文件 (*)")
        if path:
            self.command({"action": "model", "path": path})

    def show_settings(self):
        self.settings_window.showNormal()
        self.settings_window.raise_()
        self.settings_window.activateWindow()

    def run(self):
        return self.app.exec()

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.timer.stop()
        self.surface_timer.stop()
        self._offense_pool.shutdown(wait=False, cancel_futures=True)
        self.app.removeNativeEventFilter(self.hotkey_filter)
        if self.desktop:
            self.desktop.close()
        self.preferences.sync()
        self.tray.hide()
        for group in self.groups.values():
            group.close()
        self.settings_window.close()
        self.app.quit()
