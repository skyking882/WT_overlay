"""Qt render/interaction checks; Win32 compositor and game behavior need Windows."""

from dataclasses import replace
import importlib.util
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

HAS_QT = importlib.util.find_spec("PySide6") is not None
if HAS_QT:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QPoint, QRect, QSettings, Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    from wt_overlay.ui import OverlayApp, game_geometry
    from wt_overlay.windows import GameWindow

from wt_overlay.contracts import EnergyMetrics, FlightState, OverlaySnapshot


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

    def make_ui(self):
        ui = OverlayApp(lambda: self.snapshot, self.commands.append,
                        settings=QSettings(self.settings_path, QSettings.Format.IniFormat), show_on_start=False)
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
        self.assertTrue(all(g.isVisible() for g in self.ui.groups.values()))
        self.ui.game = replace(game, foreground=False)
        self.ui._sync_surface()
        self.assertFalse(any(g.isVisible() for g in self.ui.groups.values()))
        self.ui.set_editing(True)
        self.assertTrue(all(g.isVisible() for g in self.ui.groups.values()))
        self.ui.set_visible(False)
        self.assertFalse(self.ui.editing)
        self.assertFalse(any(g.isVisible() for g in self.ui.groups.values()))
        self.snapshot = replace(self.snapshot, mode="demo")
        self.ui.refresh()
        self.assertFalse(any(g.isVisible() for g in self.ui.groups.values()))
        self.ui.set_visible(True)
        self.assertTrue(all(g.isVisible() for g in self.ui.groups.values()))

    def test_failed_snapshot_read_clears_rendered_flight_and_energy(self):
        def broken():
            raise RuntimeError("sample unavailable")
        self.ui._get_snapshot = broken
        self.ui.refresh()
        for group in self.ui.groups.values():
            self.assertTrue(all(row.value == "—" for row in group.content.rows))
        self.assertIn("sample unavailable", self.ui.settings_window.status.text())

    def test_settings_send_validated_mass_without_commands_during_refresh(self):
        self.assertEqual(self.commands, [])
        self.ui.settings_window.mass.setText("nan")
        self.ui.settings_window.apply_mass()
        self.assertEqual(self.commands, [])
        self.ui.settings_window.mass.setText("23000")
        self.ui.settings_window.apply_mass()
        self.assertEqual(self.commands, [{"action": "mass", "kg": 23000.0}])

    def test_mixed_dpi_coordinates_use_monitor_origin_not_global_scaling(self):
        screen = SimpleNamespace(geometry=lambda: QRect(-1920, 0, 1280, 720), devicePixelRatio=lambda: 1.5)
        game = GameWindow(True, False, (-1770, 150, 1500, 900), (-1920, 0), "secondary")
        self.assertEqual(game_geometry(game, screen), QRect(-1820, 100, 1000, 600))


if __name__ == "__main__":
    unittest.main()
