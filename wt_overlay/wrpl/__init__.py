"""Direct .wrpl import (docs/wrpl_import_spec.md): replay -> backend (wrpl_export) -> match_export.json -> converter
(scripts/wt_replay_import.py) -> outputs/engagements/wt_real/<name>.jsonl. ``python -m wt_overlay.wrpl``.

Only export.py imports PyReplayParser, and only inside the frozen backend; nothing here needs it.
"""
