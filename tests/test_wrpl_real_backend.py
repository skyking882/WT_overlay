"""Real backend against a verified export (docs/wrpl_import_spec.md §8.4). Local and opt-in: runs only with
WT_WRPL_BACKEND pointing at a built wrpl_export and the replay, its game files and the reference export present in
outputs/ (never committed). The export must equal the reference field by field except backend_revision and
capture_script_sha256; the converted .jsonl must be byte-identical after the header, and the header equal except the
export path / hash and backend_revision.

    WT_WRPL_BACKEND=~/.cache/wt_wrpl_build/dist-macos-x64/wrpl_export python3 -m unittest tests.test_wrpl_real_backend
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from wt_overlay.wrpl import backend as backends
from wt_overlay.wrpl import importer

ROOT = Path(__file__).resolve().parents[1]
GAME = ROOT/"outputs"/"wt_replays"
REPLAY = GAME/"#2026.10.08 10.47.02.wrpl"
REFERENCE = ROOT/"outputs"/"wt_replay_exports"/"2026.10.08_10.47.02"/"match_export.json"
RUN_METADATA = ("backend_revision", "capture_script_sha256")
HEADER_METADATA = ("export_file", "export_sha256", "backend_revision")


@unittest.skipUnless(os.environ.get(backends.ENV) and REPLAY.is_file() and REFERENCE.is_file() and
                     (GAME/"aces.vromfs.bin").is_file(), "needs WT_WRPL_BACKEND, the replay, game files and reference")
class RealBackendTests(unittest.TestCase):
    def test_export_and_conversion_match_the_reference(self):
        backend = backends.find_backend()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/"export"
            importer.run_backend(backend, GAME, str(REPLAY), out)
            new = json.loads((out/"match_export.json").read_text(encoding="utf-8"))
            ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
            self.assertEqual(list(new), list(ref))
            for key in ref:
                if key not in RUN_METADATA:
                    self.assertEqual(new[key], ref[key], key)
            lines = []
            for export, name in ((REFERENCE, "ref"), (out/"match_export.json", "new")):
                subprocess.run([sys.executable, str(importer.CONVERTER), str(export), "--out-dir", tmp, "--name", name],
                               check=True, capture_output=True, cwd=ROOT)
                lines.append((Path(tmp)/f"{name}.jsonl").read_bytes().split(b"\n"))
            self.assertEqual(lines[0][1:], lines[1][1:])
            heads = [json.loads(rows[0]) for rows in lines]
            for head in heads:
                for key in HEADER_METADATA:
                    head["source"].pop(key)
            self.assertEqual(heads[0], heads[1])


if __name__ == "__main__":
    unittest.main()
