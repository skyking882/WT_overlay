from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from wt_overlay.windowed import main


class WindowedLaunchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.log = Path(self.temp.name)/"WT Energy"/"startup.log"
        env = patch.dict("os.environ", {"LOCALAPPDATA": self.temp.name})
        env.start()
        self.addCleanup(env.stop)
        dialog = patch("wt_overlay.windowed.show_error")
        self.dialog = dialog.start()
        self.addCleanup(dialog.stop)

    def test_success_with_pythonw_streams_keeps_arguments_and_captures_output(self):
        argv = ["--model", r"C:\plane models\su_27sm.blkx", "--mass-kg", "23000"]
        def run(args):
            self.assertEqual(args, argv)
            print("started")
            print("diagnostic", file=sys.stderr)
            return 0
        with patch("wt_overlay.__main__.main", side_effect=run), patch("sys.stdout", None), patch("sys.stderr", None):
            result = main(argv)
            self.assertIsNone(sys.stderr)
        self.assertEqual(result, 0)
        self.assertIn("diagnostic", self.log.read_text())
        self.dialog.assert_not_called()

    def test_application_error_is_visible_and_saved(self):
        def run(args):
            print("Unable to load FM", file=sys.stderr)
            return 2
        with patch("wt_overlay.__main__.main", side_effect=run):
            self.assertEqual(main([]), 2)
        text = self.dialog.call_args.args[0]
        self.assertIn("Unable to load FM", text)
        self.assertIn(str(self.log), text)

    def test_argument_errors_and_unexpected_failures_are_visible(self):
        self.assertEqual(main(["--unknown-option"]), 2)
        self.assertIn("--unknown-option", self.dialog.call_args.args[0])
        with patch("wt_overlay.__main__.main", side_effect=RuntimeError("window failed")):
            self.assertEqual(main([]), 2)
        self.assertIn("RuntimeError: window failed", self.dialog.call_args.args[0])
        self.assertIn("Traceback", self.log.read_text())

    def test_log_directory_failure_is_reported(self):
        (Path(self.temp.name)/"WT Energy").write_text("not a directory")
        self.assertEqual(main([]), 2)
        self.assertIn("无法启动", self.dialog.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
