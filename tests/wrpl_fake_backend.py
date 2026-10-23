#!/usr/bin/env python3
"""Fake wrpl_export for the tests (docs/wrpl_import_spec.md §8.1): the §2 command line, a small fixed
match_export.json (two aircraft flying head-on, one player each). FAKE_WRPL_MODE: ok (default), fail, sleep, crash.
FAKE_WRPL_CALLS: a file that gets one line per run (to count backend calls)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def export(replay):
    def track(x0, vx, z):
        return [{"t_ms": t*250, "xyz": [x0+vx*t*0.25, 6000.0, z], "euler_raw": [0.0, 0.0 if vx > 0 else 3.14159, 0.0]}
                for t in range(0, 241)]
    units = [{"eid": 1001+i, "uid": i, "unit_name": name, "player_internal_name": f"p{i}", "owner_pid": i,
              "unit_type": 2, "army": i+1, "loadout": "", "spawn_matrix_repr": "", "created_at_ms": 0,
              "killed_at_ms": 4294967295, "destroyed_at_ms": 4294967295, "weapons": [], "storage_weapons": [],
              "track": track(x0, vx, 1000.0*i)}
             for i, (name, x0, vx) in enumerate((("f_16c_block_50", -30000.0, 300.0), ("su_27sm", 30000.0, -300.0)))]
    players = [{"index": i, "name": f"Pilot{i}", "account_id": 100+i, "team": i+1, "country": "c", "real_nick": None,
                "owned_unit_eids": [1001+i], "owned_units": 1} for i in range(2)]
    return {"source": str(replay), "source_sha256": hashlib.sha256(Path(replay).read_bytes()).hexdigest(),
            "backend_revision": "fake", "capture_script_sha256": "fake", "export": "match_export v1",
            "probes_missing": [], "replay_length_ms": 60000,
            "header": {"mission": "test", "mission_file": "", "level": "levels/avg_test.bin", "player_no": 0,
                       "player_count": 2, "game_version": 563203356491830, "environment": "", "weather": ""},
            "units": units, "players": players, "battle_messages": [], "rockets": [], "countermeasures": [],
            "untracked_rockets": [], "fm_probe": {}, "unit_mpi": {"rows": [], "counts": [], "columns": []},
            "status": {}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", action="store_true")
    ap.add_argument("--game")
    ap.add_argument("--replay")
    ap.add_argument("--output")
    ap.add_argument("--local-package")
    args = ap.parse_args()
    if args.version:
        print(json.dumps({"backend_version": "fake", "upstream_revision": "fake", "fork_commit": "fake"}))
        return 0
    calls = os.environ.get("FAKE_WRPL_CALLS")
    if calls:
        with open(calls, "a", encoding="utf-8") as handle:
            handle.write(f"{args.replay}\n")
    out = Path(args.output)
    if out.exists():
        print("error: output directory already exists", file=sys.stderr)
        return 2
    out.mkdir(parents=True)
    (out/"parser.log").write_text("0.001 [I] init started\n0.002 [I] init finished\n", encoding="utf-8")
    mode = os.environ.get("FAKE_WRPL_MODE", "ok")
    if mode == "sleep":
        time.sleep(60)
    if mode == "fail":
        with open(out/"parser.log", "a", encoding="utf-8") as handle:
            handle.write("0.003 [E] fake parser failure\n")
        print("error: RuntimeError: replay rejected by the parser", file=sys.stderr)
        print("parser.log (end):", file=sys.stderr)
        print("0.003 [E] fake parser failure", file=sys.stderr)
        return 1
    if mode == "crash":
        os._exit(139)
    (out/"match_export.json").write_text(json.dumps(export(args.replay)), encoding="utf-8")
    print(json.dumps({"units": 2, "players": 2}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
