#!/usr/bin/env python3
"""Fetch the unit (airframe) files of every catalog aircraft and the sensor files they use.

The FM catalog (data/fm/catalog.json) pins a War-Thunder-Datamine revision and records
each unit file's sha256. This downloads those unit files at the same revision, checks
the hashes, then downloads every sensor (radar, RWR, ...) and weapon container (missile
racks) the units use. Raw files go to a cache under outputs/ (reproducible, not tracked);
the parsed equipment, radars and RWRs go to data/units/ (wt_overlay.units).

    python3 scripts/import_datamine_units.py
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wt_overlay import units as un  # noqa: E402

CATALOG = ROOT/"data"/"fm"/"catalog.json"
CACHE = ROOT/"outputs"/"datamine"


def raw_url(repository, revision, path):
    return f"https://raw.githubusercontent.com/{repository}/{revision}/{path}"


def fetch(url, dest: Path):
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_suffix(dest.suffix+".part")
        subprocess.run(["curl", "-sSfL", "--retry", "3", "-o", str(part), url], check=True)
        part.rename(dest)
    return hashlib.sha256(dest.read_bytes()).hexdigest()


as_list = un.as_list


def source_path(blk: str) -> str:
    """'gameData/sensors/us_an_apg_68_v_7.blk' -> 'aces.vromfs.bin_u/gamedata/sensors/us_an_apg_68_v_7.blkx'."""
    return "aces.vromfs.bin_u/"+blk.replace("\\", "/").lower()+"x"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, default=un.DATA_DIR)
    args = parser.parse_args(argv)
    catalog = json.loads(CATALOG.read_text())
    repo, rev = catalog["repository"], catalog["revision"]
    root = CACHE/rev
    units = {a["unit_source_path"]: a["unit_sha256"] for a in catalog["aircraft"]}

    def get(path):
        return path, fetch(raw_url(repo, rev, path), root/path)

    with ThreadPoolExecutor(args.workers) as pool:
        got = dict(pool.map(get, sorted(units)))
    bad = [p for p, h in got.items() if h != units[p]]
    if bad:
        raise SystemExit(f"sha256 mismatch for {len(bad)} unit files, e.g. {bad[:3]}")
    sensors, containers = set(), set()
    unit_data = {}
    for a in catalog["aircraft"]:
        unit = unit_data[a["id"]] = json.loads((root/a["unit_source_path"]).read_text())
        sensors |= {source_path(s["blk"]) for s in as_list((unit.get("sensors") or {}).get("sensor"))}
        for slot in as_list((unit.get("WeaponSlots") or {}).get("WeaponSlot")):
            for p in as_list(slot.get("WeaponPreset")):
                containers |= {source_path(w["blk"]) for w in as_list(p.get("Weapon"))
                               if "/containers/" in w.get("blk", "").lower()}
    with ThreadPoolExecutor(args.workers) as pool:
        got_sensors = dict(pool.map(get, sorted(sensors)))
        got_containers = dict(pool.map(get, sorted(containers)))
    manifest = dict(repository=repo, revision=rev, game_version=catalog.get("game_version"),
                    units=got, sensors=got_sensors, containers=got_containers)
    (root/"manifest.json").write_text(json.dumps(manifest, indent=1))
    size = sum((root/p).stat().st_size for p in [*got, *got_sensors, *got_containers])
    print(f"{len(got)} unit files (hashes match the FM catalog), {len(got_sensors)} sensor files, "
          f"{len(got_containers)} containers, {size/1e6:.1f} MB in {root}")

    sensor_data = {un.stem(p): json.loads((root/p).read_text()) for p in got_sensors}
    types = {k: d.get("type") for k, d in sensor_data.items()}
    racks = {}
    for p in got_containers:
        c = json.loads((root/p).read_text())
        if c.get("blk"):
            racks[un.stem(p)] = (c["blk"], c.get("bullets") or 1)
    equipment = {a: un.parse_unit(a, u, types, racks) for a, u in unit_data.items()}
    radars = {k: un.parse_radar(k, d) for k, d in sensor_data.items() if d.get("type") == "radar"}
    rwrs = {k: un.parse_rwr(k, d) for k, d in sensor_data.items() if d.get("type") == "rwr"}
    un.dump(args.out/"equipment.json", equipment)
    un.dump(args.out/"radars.json", radars)
    un.dump(args.out/"rwrs.json", rwrs)
    (args.out/"source.json").write_text(json.dumps(dict(repository=repo, revision=rev, game_version=catalog.get("game_version"),
                                                        files=len(got)+len(got_sensors)+len(got_containers)), indent=1))
    print(f"wrote {len(equipment)} aircraft, {len(radars)} radars, {len(rwrs)} RWRs to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
