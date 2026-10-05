"""Qt render/interaction checks; Win32 compositor and game behavior need Windows."""

from dataclasses import replace
import json
import importlib.util
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HAS_QT = importlib.util.find_spec("PySide6") is not None
if HAS_QT:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QPoint, QRect, QSettings, Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    from wt_overlay.ui import OverlayApp, game_geometry
    from wt_overlay.windows import GameWindow

from wt_overlay.contracts import (ClimbGuidance, ClimbRequest, EnergyMetrics, FlightState,
                                  KeyboardTurnGuidance, KeyboardTurnSettings, OverlaySnapshot)


def write_pk_model(folder, name="m"):
    """Synthetic hit-probability model (see tests/test_offense.py): head-on co-altitude Rmax 30 km
    (+1 km per km of target altitude above), cold 12 km; P(hit) 0.5 at 10 km, 0.25 at 15 km."""
    from wt_overlay import pk
    n = len(pk.FEATURES)
    index = {f: i for i, f in enumerate(pk.FEATURES)}
    def row(**weights):
        r = [0.] * n
        for k, v in weights.items():
            r[index[k]] = v
        return r
    reach = row(cos_course=1.8, range_m=-0.0002, alt_diff_m=0.0002)
    hit = row(cos_course=1.0, range_m=-0.00021972)
    data = dict(missile=name, features=list(pk.FEATURES), outputs=list(pk.OUTPUTS), activation="silu",
                mean=[0.] * n, std=[1.] * n, layers=[dict(w=[reach] + [hit] * 4, b=[5.2] + [1.1972] * 4)])
    Path(folder).mkdir(parents=True, exist_ok=True)
    (Path(folder) / f"{name}.json").write_text(json.dumps(data))


@unittest.skipUnless(HAS_QT, "PySide6 is optional for backend-only tests")
class OverlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.settings_path = str(Path(self.temp.name) / "overlay.ini")
        self.snapshot = OverlaySnapshot("demo", "合成演示", FlightState(1, True, altitude_m=5000, tas_mps=300),
                                        EnergyMetrics(1, 9600, 35, 10, climb_mps=25, ready=True))
        self.commands = []
        self.ui = self.make_ui()
        self.app.processEvents()

    def make_ui(self, show_on_start=False):
        ui = OverlayApp(lambda: self.snapshot, self.commands.append,
                        settings=QSettings(self.settings_path, QSettings.Format.IniFormat), show_on_start=show_on_start)
        ui.timer.stop()
        ui.surface_timer.stop()
        return ui

    def tearDown(self):
        self.ui.close()
        self.app.processEvents()
        self.temp.cleanup()

    def make_rose_ui(self, enabled):
        folder = Path(self.temp.name) / "pk_models"
        write_pk_model(folder)
        settings = QSettings(self.settings_path, QSettings.Format.IniFormat)
        settings.setValue("offense/enabled", enabled)
        settings.setValue("offense/scope_enabled", enabled)
        settings.setValue("offense/side_enabled", enabled)
        settings.setValue("offense/missile", "m")
        self.ui.close()
        self._pk_patch = patch("wt_overlay.pk.DATA_DIR", folder)
        self._pk_patch.start()
        self.addCleanup(self._pk_patch.stop)
        self.ui = self.make_ui()
        self.snapshot = replace(self.snapshot, mode="live")
        self.ui.refresh()
        return self.ui.groups["rose"]

    def test_rose_is_off_by_default_and_draws_coloured_sectors_when_enabled(self):
        self.assertFalse(self.ui.rose_enabled)
        self.assertFalse(self.ui.groups["rose"].isVisible())
        group = self.make_rose_ui(True)
        self.assertTrue(group.has_data())
        self.assertTrue(group.isVisible())
        self.assertEqual(group._header(), "")
        image = group.grab().toImage()
        self.assertEqual(image.pixelColor(0, 0).alpha(), 0)
        reds = sum(1 for y in range(0, image.height(), 2) for x in range(0, image.width(), 2)
                   if image.pixelColor(x, y).red() > 150 and image.pixelColor(x, y).green() < 120)
        self.assertGreater(reds, 20)  # The synthetic model's hot sectors are above 50 % inside 10 km.
        self.snapshot = replace(self.snapshot, state=FlightState(2, False))
        self.ui.refresh()
        self.assertFalse(group.has_data())
        self.assertFalse(group.isVisible())

    def test_rose_hotkey_and_choice_persist(self):
        group = self.make_rose_ui(False)
        self.assertFalse(group.isVisible())
        registrations = []
        self.ui.desktop = SimpleNamespace(register_hotkey=lambda identifier, letter:
            registrations.append((identifier, letter)) or True, close=lambda: None)
        self.ui._setup_hotkeys()
        self.assertIn((0x5747, "K"), registrations)
        self.ui.desktop = None  # No game window under the stub would hide every group.
        self.ui.hotkey_filter.callbacks[0x5747]()
        self.assertTrue(self.ui.rose_enabled and group.isVisible())
        self.assertEqual(self.ui.offense_missile, "m")
        self.ui.close()
        self.ui = self.make_ui()
        self.assertTrue(self.ui.rose_enabled)

    def test_scope_overlay_draws_envelope_lines_at_the_chosen_range_scale(self):
        self.assertFalse(self.ui.scope_enabled)
        self.make_rose_ui(True)
        scope = self.ui.groups["scope"]
        self.assertTrue(scope.has_data() and scope.isVisible())
        self.assertEqual(scope.size().width(), 260)
        image = scope.grab().toImage()
        self.assertEqual(image.pixelColor(5, 5).alpha(), 0)
        def colour_rows(img, test):
            return {y for y in range(img.height()) for x in range(0, img.width(), 3) if test(img.pixelColor(x, y))}
        amber = lambda c: c.alpha() > 200 and c.red() > 200 and 120 < c.green() < 190 and c.blue() < 90  # noqa: E731
        rows = colour_rows(image, amber)
        # 40 km scale: Rmax hot (30 km) sits a quarter of the height from the top.
        self.assertTrue(any(abs(y - 65) <= 3 for y in rows), rows)
        self.ui.set_scope_geometry(80, 60)
        rows = colour_rows(scope.grab().toImage(), amber)
        self.assertTrue(any(abs(y - 162) <= 3 for y in rows), rows)  # 30 of 80 km.

    def test_scope_resizes_in_layout_mode_and_size_survives_restart(self):
        self.make_rose_ui(True)
        self.ui.set_editing(True)
        scope = self.ui.groups["scope"]
        grip = QPoint(scope.width() - 4, scope.height() - 4)
        QTest.mousePress(scope, Qt.MouseButton.LeftButton, pos=grip)
        QTest.mouseMove(scope, grip + QPoint(60, 30))
        QTest.mouseRelease(scope, Qt.MouseButton.LeftButton, pos=grip + QPoint(60, 30))
        self.assertEqual(scope.box, (320, 290))
        self.ui.close()
        self.ui = self.make_ui()
        self.assertEqual(self.ui.groups["scope"].box, (320, 290))
        registrations = []
        self.ui.desktop = SimpleNamespace(register_hotkey=lambda identifier, letter:
            registrations.append((identifier, letter)) or True, close=lambda: None)
        self.ui._setup_hotkeys()
        self.assertIn((0x5748, "B"), registrations)

    def test_side_view_is_off_by_default_and_draws_the_envelope_profile(self):
        self.assertFalse(self.ui.side_enabled)
        self.assertFalse(self.ui.groups["side"].isVisible())
        self.make_rose_ui(True)
        side = self.ui.groups["side"]
        self.assertTrue(side.has_data() and side.isVisible())
        image = side.grab().toImage()
        self.assertEqual(image.pixelColor(0, 0).alpha(), 0)
        amber = lambda c: c.alpha() > 200 and c.red() > 200 and 120 < c.green() < 190 and c.blue() < 90  # noqa: E731
        pixels = [(x, y) for y in range(image.height()) for x in range(image.width()) if amber(image.pixelColor(x, y))]
        self.assertGreater(len(pixels), 30)
        # Rmax hot rises with target altitude: its pixels span a slope, so more than one row is lit.
        self.assertGreater(len({y for _, y in pixels}), 10)
        self.ui.set_scope_geometry(80, 60)
        wider = side.grab().toImage()
        self.assertNotEqual(image, wider)
        self.snapshot = replace(self.snapshot, state=FlightState(2, False))
        self.ui.refresh()
        self.assertFalse(side.has_data())
        self.assertFalse(side.isVisible())

    def test_side_view_hotkey_toggles_visibility(self):
        side = self.make_rose_ui(False) and self.ui.groups["side"]
        self.assertFalse(side.isVisible())
        registrations = []
        self.ui.desktop = SimpleNamespace(register_hotkey=lambda identifier, letter:
            registrations.append((identifier, letter)) or True, close=lambda: None)
        self.ui._setup_hotkeys()
        self.assertIn((0x5749, "V"), registrations)
        self.assertIn((0x5748, "B"), registrations)
        self.ui.desktop = None
        self.ui.hotkey_filter.callbacks[0x5749]()
        self.assertTrue(self.ui.side_enabled and side.isVisible())
        self.ui.hotkey_filter.callbacks[0x5749]()
        self.assertFalse(side.isVisible())

    def test_background_is_zero_alpha_while_text_remains_opaque(self):
        group = self.ui.groups["energy"]
        image = group.grab().toImage()
        alphas = [image.pixelColor(x, y).alpha() for y in range(image.height()) for x in range(image.width())]
        self.assertGreater(alphas.count(0), len(alphas) * 0.55)
        self.assertIn(255, alphas)
        self.assertEqual(image.pixelColor(0, 0).alpha(), 0)
        self.assertEqual(group.windowOpacity(), 1.0)
        # A black outline must not paint over the light glyph fill at small font sizes.
        bright = sum(1 for y in range(image.height()) for x in range(image.width())
                     if image.pixelColor(x, y).alpha() > 200 and image.pixelColor(x, y).lightness() > 160)
        self.assertGreater(bright, 200)

    def test_aircraft_selector_search_manual_auto_and_sweep_settings(self):
        window = self.ui.settings_window
        box = window.aircraft_box
        self.assertEqual(box.count(), 140)  # 138 aircraft + auto + file.
        self.assertTrue(box.isEditable())
        self.assertEqual(box.completer().filterMode(), Qt.MatchFlag.MatchContains)
        box.completer().setCompletionPrefix("F-15")
        expected = [box.itemText(i) for i in range(box.count()) if "f-15" in box.itemText(i).casefold()]
        self.assertGreater(len(expected), 0)
        self.assertEqual(box.completer().completionCount(), len(expected))
        self.assertEqual(self.commands, [])
        index = box.findData("f_15c_golden_eagle")
        self.assertGreater(index, 0)
        box.setCurrentIndex(index)
        box.activated.emit(index)
        self.assertEqual(self.commands[-1], {"action": "aircraft", "id": "f_15c_golden_eagle"})
        window.select_aircraft(0)
        self.assertEqual(self.commands[-1], {"action": "aircraft", "id": "auto"})
        self.snapshot = replace(self.snapshot, model_selection="f_14b", variable_sweep=True,
                                sweep_fraction=.5)
        count = len(self.commands)
        self.ui.refresh()
        self.assertFalse(window.sweep.isHidden())
        self.assertEqual(window.sweep.value(), 50)
        self.assertEqual(len(self.commands), count)
        window.sweep.setValue(65)
        self.assertEqual(self.commands[-1], {"action": "sweep", "fraction": .65})
        self.snapshot = replace(self.snapshot, variable_sweep=False)
        self.ui.refresh()
        self.assertTrue(window.sweep.isHidden())

    def test_auto_model_status_distinguishes_j16_missing_identity_and_load_failure(self):
        self.snapshot = replace(self.snapshot, mode="live", model_selection="auto", model_name="歼-16",
                                state=replace(self.snapshot.state, aircraft_id="J-16"))
        self.ui.refresh()
        self.assertIn("自动识别：歼-16", self.ui.settings_window.model_label.text())
        self.assertIn("J-16", self.ui.settings_window.model_label.text())
        self.snapshot = replace(self.snapshot, state=replace(self.snapshot.state, valid=False))
        self.ui.refresh()
        self.assertIn("等待游戏机型", self.ui.settings_window.model_label.text())
        self.snapshot = replace(self.snapshot, model_name="未加载 FM",
                                state=replace(self.snapshot.state, valid=True))
        self.ui.refresh()
        self.assertIn("未加载模型", self.ui.settings_window.model_label.text())
        self.assertIn("J-16", self.ui.settings_window.model_label.text())

    def test_layout_drag_is_saved_and_battle_mode_restores_input_transparency(self):
        group = self.ui.groups["energy"]
        self.assertTrue(group.windowFlags() & Qt.WindowType.WindowTransparentForInput)
        self.ui.set_editing(True)
        self.assertFalse(group.windowFlags() & Qt.WindowType.WindowTransparentForInput)
        self.assertTrue(group.windowFlags() & Qt.WindowType.WindowDoesNotAcceptFocus)
        original = group.pos()
        QTest.mousePress(group, Qt.MouseButton.LeftButton, pos=QPoint(25, 20))
        QTest.mouseMove(group, QPoint(120, 40))
        QTest.mouseRelease(group, Qt.MouseButton.LeftButton, pos=QPoint(25, 20))
        self.assertNotEqual(group.pos(), original)
        saved = group.fraction
        self.ui.set_editing(False)
        self.assertTrue(group.windowFlags() & Qt.WindowType.WindowTransparentForInput)
        self.assertEqual(group.grab().toImage().pixelColor(0, 0).alpha(), 0)
        self.ui.close()
        self.ui = self.make_ui()
        self.assertEqual(self.ui.groups["energy"].fraction, saved)

    def test_game_focus_hides_hud_without_overriding_user_hide_or_demo_mode(self):
        self.snapshot = replace(self.snapshot, mode="live")
        self.ui.desktop = SimpleNamespace(close=lambda: None)
        self.ui.refresh()
        self.assertFalse(any(g.isVisible() for g in self.ui.groups.values()))
        game = GameWindow(True, False, (0, 0, 1280, 720), (0, 0), self.app.primaryScreen().name())
        self.ui.game = game
        self.ui._sync_surface()
        self.assertTrue(all(g.isVisible() for key, g in self.ui.groups.items() if key not in ("climb", "turn", "rose", "scope", "side")))
        self.ui.game = replace(game, foreground=False)
        self.ui._sync_surface()
        self.assertFalse(any(g.isVisible() for g in self.ui.groups.values()))
        self.ui.set_editing(True)
        self.assertTrue(all(g.isVisible() for key, g in self.ui.groups.items() if key not in ("climb", "turn", "rose", "scope", "side")))
        self.ui.set_visible(False)
        self.assertFalse(self.ui.editing)
        self.assertFalse(any(g.isVisible() for g in self.ui.groups.values()))
        self.snapshot = replace(self.snapshot, mode="demo")
        self.ui.refresh()
        self.assertFalse(any(g.isVisible() for g in self.ui.groups.values()))
        self.ui.set_visible(True)
        self.assertTrue(all(g.isVisible() for key, g in self.ui.groups.items() if key not in ("climb", "turn", "rose", "scope", "side")))

    def test_turn_toggle_restart_hotkeys_and_mutual_exclusion(self):
        self.assertFalse(self.ui.groups["turn"].isVisible())
        self.ui.set_turn_enabled(True)
        self.assertEqual(self.commands[-1], {"action": "turn_enabled", "enabled": True})
        self.snapshot = replace(self.snapshot, turn_enabled=True,
            turn=KeyboardTurnGuidance(True, "转向", "右滚＋拉杆", 15, 75, 4))
        self.ui.refresh()
        group = self.ui.groups["turn"]
        self.assertTrue(group.isVisible())
        self.assertEqual(group.grab().toImage().pixelColor(0, 0).alpha(), 0)
        self.ui.restart_turn()
        self.assertEqual(self.commands[-1], {"action": "turn_restart"})
        self.ui.set_climb_enabled(True)
        self.assertFalse(group.isVisible())
        self.assertFalse(self.ui.turn_enabled)
        registrations = []
        self.ui.desktop = SimpleNamespace(register_hotkey=lambda identifier, letter:
            registrations.append((identifier, letter)) or True, close=lambda: None)
        self.ui._setup_hotkeys()
        self.assertIn((0x5744, "T"), registrations)
        self.assertIn((0x5745, "R"), registrations)
        self.assertIn((0x5746, "A"), registrations)
        self.ui.settings_window.pose_sign.setCurrentIndex(1)
        self.ui.hotkey_filter.callbacks[0x5746]()
        self.assertEqual(self.commands[-1], {"action": "pose_calibrate", "roll_sign": -1})
        self.ui.hotkey_filter.callbacks[0x5744]()
        self.assertTrue(self.ui.turn_enabled)
        self.assertFalse(self.ui.climb_enabled)
        self.ui.hotkey_filter.callbacks[0x5745]()
        self.assertEqual(self.commands[-1], {"action": "turn_restart"})
        self.ui.set_turn_enabled(False)
        self.ui.refresh()
        self.assertFalse(group.isVisible())

    def test_turn_preferences_restore_target_but_never_start_automatically(self):
        w = self.ui.settings_window
        w.turn_angle.setCurrentIndex(w.turn_angle.findData(120))
        w.turn_fields["roll_rate_deg_s"][0].setValue(90)
        w.turn_fields["minimum_tas_mps"][0].setValue(720)
        w.turn_fields["throttle_rate_percent_s"][0].setValue(30)
        w.turn_fields["engine_response_s"][0].setValue(1.5)
        self.assertTrue(w.apply_turn_settings())
        self.assertEqual(self.ui.turn_settings.minimum_tas_mps, 200)
        self.ui.close()
        self.ui = self.make_ui()
        self.assertFalse(self.ui.turn_enabled)
        self.assertEqual(self.ui.turn_settings.angle_deg, 120)
        self.assertEqual(self.ui.turn_settings.roll_rate_deg_s, 90)
        self.assertEqual(self.ui.turn_settings.throttle_rate_percent_s, 30)
        self.assertEqual(self.ui.turn_settings.engine_response_s, 1.5)

    def test_legacy_short_hold_migrates_without_losing_turn_target(self):
        self.ui.preferences.setValue("turn/settings", json.dumps({"angle_deg": 120, "hold_s": .6,
                                                                "max_altitude_loss_m": 800}))
        self.ui.close()
        self.ui = self.make_ui()
        self.assertEqual(self.ui.turn_settings.hold_s, 1.2)
        self.assertEqual(self.ui.turn_settings.angle_deg, 120)
        self.assertEqual(self.ui.turn_settings.max_altitude_loss_m, 800)

    def test_climb_is_independent_defaults_off_and_hides_before_worker_acknowledges(self):
        self.assertFalse(self.ui.groups["climb"].isVisible())
        self.ui.set_climb_enabled(True)
        self.assertEqual(self.commands[-1], {"action": "climb_enabled", "enabled": True})
        self.snapshot = replace(self.snapshot, climb_enabled=True,
                                climb=ClimbGuidance(True, "爬升", 320, 12, 2, 3000))
        self.ui.refresh()
        self.assertTrue(self.ui.groups["climb"].isVisible())
        self.ui.set_climb_enabled(False)
        self.assertFalse(self.ui.groups["climb"].isVisible())
        self.ui.refresh()  # The worker's last snapshot still says enabled.
        self.assertFalse(self.ui.groups["climb"].isVisible())
        self.assertTrue(self.ui.groups["energy"].isVisible())
        self.snapshot = replace(self.snapshot, climb_enabled=False, climb=None)
        self.ui.refresh()
        self.assertIsNone(self.ui._pending_climb_enabled)

    def test_climb_target_is_saved_but_active_mode_is_not(self):
        window = self.ui.settings_window
        window.climb_altitude.setValue(9000)
        window.climb_speed.setText("1200")
        self.assertTrue(window.apply_climb_target())
        self.assertEqual(self.commands[-1], {"action": "climb_target", "altitude_m": 9000,
                                             "minimum_tas_mps": 1200/3.6})
        self.ui.set_climb_enabled(True)
        self.ui.close()
        self.ui = self.make_ui()
        self.assertFalse(self.ui.climb_enabled)
        self.assertEqual(self.ui.climb_request, ClimbRequest(9000, 1200/3.6))
        self.assertEqual(self.ui.settings_window.climb_speed.text(), "1200")
        self.ui.settings_window.climb_speed.setText("nan")
        count = len(self.commands)
        self.ui.set_climb_enabled(True)
        self.assertFalse(self.ui.climb_enabled)
        self.assertEqual(len(self.commands), count)

    def test_climb_cue_has_transparent_background_green_band_and_no_stale_marker(self):
        self.snapshot = replace(self.snapshot, mode="live", climb_enabled=True,
                                climb=ClimbGuidance(True, "爬升", 320, 12, 3, 3000,
                                                    actual_path_deg=9, target_ias_mps=224))
        self.ui.refresh()
        group = self.ui.groups["climb"]
        image = group.grab().toImage()
        self.assertEqual(image.pixelColor(0, 0).alpha(), 0)
        self.assertEqual(group._header(), "")
        green = sum(1 for y in range(image.height()) for x in range(image.width())
                    if image.pixelColor(x, y).green() > 190 and image.pixelColor(x, y).red() < 150)
        self.assertGreater(green, 30)
        self.snapshot = replace(self.snapshot, state=replace(self.snapshot.state, valid=False))
        self.ui.refresh()
        self.assertIsNone(group.content.cue_error_deg)
        self.assertTrue(all(row.value == "—" for row in group.content.rows[1:]))

    def test_windows_climb_hotkey_keeps_settings_registration_and_toggles(self):
        registrations = []
        self.ui.desktop = SimpleNamespace(register_hotkey=lambda identifier, letter:
            registrations.append((identifier, letter)) or True, close=lambda: None)
        self.ui._setup_hotkeys()
        self.assertIn((0x5742, "S"), registrations)
        self.assertIn((0x5743, "C"), registrations)
        self.ui.hotkey_filter.callbacks[0x5743]()
        self.assertTrue(self.ui.climb_enabled)
        self.ui.hotkey_filter.callbacks[0x5743]()
        self.assertFalse(self.ui.climb_enabled)

    def test_failed_snapshot_read_clears_rendered_flight_and_energy(self):
        def broken():
            raise RuntimeError("sample unavailable")
        self.ui._get_snapshot = broken
        self.ui.refresh()
        for group in self.ui.groups.values():
            self.assertTrue(all(row.value == "—" for row in group.content.rows))
        self.assertIn("sample unavailable", self.ui.settings_window.status.text())

    def test_battle_hud_has_only_metric_rows_and_settings_keep_explanations(self):
        self.snapshot = replace(self.snapshot, mode="live", status="8111 已连接 · 采样说明")
        self.ui.refresh()
        for group in self.ui.groups.values():
            self.assertEqual(group._header(), "")
            self.assertEqual(group.header_height, 0)
            self.assertFalse(hasattr(group.content, "footer"))
        self.assertIn("采样说明", self.ui.settings_window.notes.toPlainText())
        self.assertEqual(self.ui.hud_font.pointSize(), 11)

    def test_individual_indicator_choices_and_compact_font_survive_restart(self):
        self.ui.settings_window.indicator_boxes["aoa"].setChecked(False)
        self.ui.settings_window.indicator_boxes["heading"].setChecked(True)
        self.ui.change_font_size(13)
        self.ui.close()
        self.ui = self.make_ui()
        keys = {row.key for row in self.ui.groups["flight"].content.rows}
        self.assertNotIn("aoa", keys)
        self.assertIn("heading", keys)
        self.assertEqual(self.ui.hud_font.pointSize(), 13)

    def test_settings_send_validated_mass_without_commands_during_refresh(self):
        self.assertEqual(self.commands, [])
        self.ui.settings_window.mass.setText("nan")
        self.ui.settings_window.apply_mass()
        self.assertEqual(self.commands, [])
        self.ui.settings_window.mass.setText("23000")
        self.ui.settings_window.apply_mass()
        self.assertEqual(self.commands, [{"action": "mass", "kg": 23000.0}])

    def test_repeat_launch_opens_settings_for_existing_profiles_and_demo(self):
        self.ui.preferences.setValue("configured", True)
        self.ui.close()
        for mode in ("live", "demo"):
            with self.subTest(mode=mode), patch("wt_overlay.ui.QSystemTrayIcon.isSystemTrayAvailable", return_value=True):
                self.snapshot = replace(self.snapshot, mode=mode)
                self.ui = self.make_ui(show_on_start=True)
                self.app.processEvents()
                self.assertTrue(self.ui.can_reopen_settings)
                self.assertTrue(self.ui.settings_window.isVisible())
                self.ui.settings_window.close()
                self.assertFalse(self.ui.closed)
                self.assertFalse(self.ui.settings_window.isVisible())
                self.ui.show_settings()
                self.assertTrue(self.ui.settings_window.isVisible())
                self.ui.close()

    def test_open_settings_restores_a_minimized_window(self):
        self.ui.settings_window.showMinimized()
        self.app.processEvents()
        self.assertTrue(self.ui.settings_window.isMinimized())
        self.ui.show_settings()
        self.app.processEvents()
        self.assertTrue(self.ui.settings_window.isVisible())
        self.assertFalse(self.ui.settings_window.isMinimized())

    def test_mixed_dpi_coordinates_use_monitor_origin_not_global_scaling(self):
        screen = SimpleNamespace(geometry=lambda: QRect(-1920, 0, 1280, 720), devicePixelRatio=lambda: 1.5)
        game = GameWindow(True, False, (-1770, 150, 1500, 900), (-1920, 0), "secondary")
        self.assertEqual(game_geometry(game, screen), QRect(-1820, 100, 1000, 600))


if __name__ == "__main__":
    unittest.main()
