"""Qt render/interaction checks; Win32 compositor and game behavior need Windows."""

from dataclasses import replace
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

from wt_overlay.contracts import ClimbGuidance, ClimbRequest, EnergyMetrics, FlightState, OverlaySnapshot


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
        self.assertTrue(all(g.isVisible() for key, g in self.ui.groups.items() if key != "climb"))
        self.ui.game = replace(game, foreground=False)
        self.ui._sync_surface()
        self.assertFalse(any(g.isVisible() for g in self.ui.groups.values()))
        self.ui.set_editing(True)
        self.assertTrue(all(g.isVisible() for key, g in self.ui.groups.items() if key != "climb"))
        self.ui.set_visible(False)
        self.assertFalse(self.ui.editing)
        self.assertFalse(any(g.isVisible() for g in self.ui.groups.values()))
        self.snapshot = replace(self.snapshot, mode="demo")
        self.ui.refresh()
        self.assertFalse(any(g.isVisible() for g in self.ui.groups.values()))
        self.ui.set_visible(True)
        self.assertTrue(all(g.isVisible() for key, g in self.ui.groups.items() if key != "climb"))

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
