"""Terrain contours for the sandbox map, standard library only.

Source: the per-map height rasters exported from the installed game (``outputs/wt_air_maps/air_maps/<map>/``, local
only, never published): ``height_world_y_m.tif`` is a float32 Deflate TIFF whose pixel [row_z, column_x] is the
world height Y in metres at X = origin_x_m + column * cell_size_m, Z = origin_z_m + row * cell_size_m (metadata.json);
uncovered samples are NaN. The sandbox frame is x = Dagor X - origin_x, y = Dagor Z - origin_z, z = Dagor Y, so the
contours below stay in Dagor X / Z and the page subtracts the replay origin.

The raster is decimated by a whole stride to about ``TARGET`` samples per side, contoured with marching squares
(cells touching a NaN are skipped, so lines stop where coverage stops), chained into polylines and thinned with
Douglas-Peucker. The interval is the smallest of ``INTERVALS`` that keeps the estimated segment count within
``SEGMENT_BUDGET``. One extra line follows the configured water level when the map has one; it is where the terrain
crosses that level, not a surveyed shoreline, and lines below it are marked "under" (often sea floor, but the level
alone does not prove there is water). The result is cached as JSON per map.
"""
from __future__ import annotations

from array import array
import base64
import json
import math
import os
from pathlib import Path
import re
import struct
import sys
import threading
import zlib

from .sim import ROOT

DEFAULT_ROOT = ROOT/"outputs"/"wt_air_maps"/"air_maps"
TARGET = 512
SEGMENT_BUDGET = 240000
# (contour interval, major line interval), metres
INTERVALS = ((10, 50), (20, 100), (25, 100), (50, 250), (100, 500), (200, 1000), (250, 1000), (500, 2500),
             (1000, 5000))
SIMPLIFY_CELLS = .3
VERSION = 2
_STEM = re.compile(r"^[A-Za-z0-9_]+$")


def terrain_root(path=None):
    return Path(path or os.environ.get("WT_TERRAIN_DIR") or DEFAULT_ROOT)


def map_stem(level):
    """'levels/air_afghan.bin' (a replay export's header.level) -> 'air_afghan'; None when not a plain name."""
    if not isinstance(level, str) or not level:
        return None
    stem = level.replace("\\", "/").rsplit("/", 1)[-1]
    stem = stem[:-4] if stem.endswith(".bin") else stem
    return stem if _STEM.match(stem) else None


def _folder(root, stem):
    if not isinstance(stem, str) or not _STEM.match(stem):
        raise ValueError("无效的地图名")
    folder = Path(root)/stem
    if not (folder/"metadata.json").is_file() or not (folder/"height_world_y_m.tif").is_file():
        raise FileNotFoundError(f"没有地图 {stem} 的地形数据")
    return folder


def list_maps(root=None):
    """[{map, width_km, height_min_m, height_max_m}] for every exported map under root, by name."""
    root = terrain_root(root)
    out = []
    if not root.is_dir():
        return out
    for folder in sorted(root.iterdir()):
        try:
            meta = json.loads((folder/"metadata.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not (folder/"height_world_y_m.tif").is_file() or not _STEM.match(folder.name):
            continue
        out.append(dict(map=folder.name, width_km=round(meta["width"]*meta["cell_size_m"]/1000., 1),
                        height_min_m=round(meta["height_min_m"]), height_max_m=round(meta["height_max_m"])))
    return out


def map_center(root, stem):
    meta = json.loads((_folder(terrain_root(root), stem)/"metadata.json").read_text(encoding="utf-8"))
    return [meta["origin_x_m"]+(meta["width"]-1)*meta["cell_size_m"]/2.,
            meta["origin_z_m"]+(meta["height"]-1)*meta["cell_size_m"]/2.]


# -- TIFF ----------------------------------------------------------------------------------------------------------

_TYPES = {3: "H", 4: "I"}


def read_decimated(path, stride):
    """Rows 0, stride, 2 stride, ... of a single-band float32 strip TIFF (no or Deflate compression), each row taken
    every ``stride`` columns. Returns (nx, nz, values) with values a flat array('f'), row-major."""
    with open(path, "rb") as handle:
        head = handle.read(8)
        order = {b"II": "<", b"MM": ">"}.get(head[:2])
        if order is None or struct.unpack(order+"H", head[2:4])[0] != 42:
            raise ValueError("不是 TIFF 文件")
        handle.seek(struct.unpack(order+"I", head[4:8])[0])
        tags = {}
        for _ in range(struct.unpack(order+"H", handle.read(2))[0]):
            tag, kind, count, value = struct.unpack(order+"HHI4s", handle.read(12))
            tags[tag] = (kind, count, value)

        def values(tag, default=None):
            if tag not in tags:
                return default
            kind, count, raw = tags[tag]
            code = _TYPES.get(kind)
            if code is None:
                raise ValueError(f"TIFF 标签 {tag} 类型不支持")
            size = struct.calcsize(code)*count
            if size <= 4:
                return list(struct.unpack(order+code*count, raw[:size]))
            back = handle.tell()
            handle.seek(struct.unpack(order+"I", raw)[0])
            out = list(struct.unpack(order+code*count, handle.read(size)))
            handle.seek(back)
            return out

        width, height = values(256)[0], values(257)[0]
        compression = values(259, [1])[0]
        if values(258, [0])[0] != 32 or values(339, [1])[0] != 3 or values(277, [1])[0] != 1 \
                or values(317, [1])[0] != 1 or compression not in (1, 8, 32946) or 322 in tags:
            raise ValueError("只支持单通道 float32、无预测器的条带 TIFF")
        offsets, counts = values(273), values(279)
        per_strip = values(278, [height])[0]
        row_bytes = width*4
        swap = (order == "<") != (sys.byteorder == "little")
        nx, nz = (width+stride-1)//stride, (height+stride-1)//stride
        grid = array("f")
        for strip, (offset, count) in enumerate(zip(offsets, counts)):
            first = strip*per_strip
            wanted = [r for r in range(first, min(first+per_strip, height)) if r % stride == 0]
            if not wanted:
                continue
            handle.seek(offset)
            raw = handle.read(count)
            if compression != 1:
                raw = zlib.decompress(raw)
            for r in wanted:
                row = array("f")
                row.frombytes(raw[(r-first)*row_bytes:(r-first+1)*row_bytes])
                if swap:
                    row.byteswap()
                grid.extend(row[::stride])
        if len(grid) != nx*nz:
            raise ValueError("TIFF 数据不完整")
        return nx, nz, grid


# -- contours ------------------------------------------------------------------------------------------------------

# Corners a=(i,j) 1, b=(i,j+1) 2, c=(i+1,j+1) 4, d=(i+1,j) 8 at or above the level; edges 0 a-b, 1 b-c, 2 d-c,
# 3 a-d. Saddles 5 and 10 are split by the cell centre.
_CASES = {1: ((3, 0),), 2: ((0, 1),), 3: ((3, 1),), 4: ((1, 2),), 6: ((0, 2),), 7: ((2, 3),), 8: ((2, 3),),
          9: ((0, 2),), 11: ((1, 2),), 12: ((3, 1),), 13: ((0, 1),), 14: ((3, 0),)}
_SADDLE = {(5, True): ((0, 1), (2, 3)), (5, False): ((3, 0), (1, 2)),
           (10, True): ((3, 0), (1, 2)), (10, False): ((0, 1), (2, 3))}


def relief(grid, nx, nz):
    """Sum over complete cells of (highest - lowest corner): the length of contour per metre of interval, in cells."""
    total = 0.
    for i in range(nz-1):
        top, bottom = i*nx, (i+1)*nx
        for j in range(nx-1):
            a, b, c, d = grid[top+j], grid[top+j+1], grid[bottom+j+1], grid[bottom+j]
            if a == a and b == b and c == c and d == d:     # NaN-safe: min/max alone may skip a NaN
                total += max(a, b, c, d)-min(a, b, c, d)
    return total


def pick_interval(total_relief, budget=SEGMENT_BUDGET):
    for minor, major in INTERVALS:
        if total_relief/minor <= budget:
            return minor, major
    return INTERVALS[-1]


def contour(grid, nx, nz, levels):
    """{level: [polyline as [(column, row), ...] in grid units]} by marching squares; a closed line repeats its
    first point."""
    crossings = {h: {} for h in levels}   # level -> edge id -> point
    segments = {h: [] for h in levels}
    ordered = sorted(levels)
    for i in range(nz-1):
        top, bottom = i*nx, (i+1)*nx
        for j in range(nx-1):
            a, b, c, d = grid[top+j], grid[top+j+1], grid[bottom+j+1], grid[bottom+j]
            if not (a == a and b == b and c == c and d == d):
                continue
            lo, hi = min(a, b, c, d), max(a, b, c, d)
            if lo == hi:
                continue
            for h in ordered:
                if h <= lo:
                    continue
                if h > hi:
                    break
                case = (a >= h) | (b >= h) << 1 | (c >= h) << 2 | (d >= h) << 3
                pairs = _CASES.get(case) or _SADDLE[(case, (a+b+c+d)/4. >= h)]
                cross = crossings[h]
                ids = ((i*nx+j)*2, (i*nx+j+1)*2+1, ((i+1)*nx+j)*2, (i*nx+j)*2+1)
                for edge in {e for pair in pairs for e in pair}:
                    eid = ids[edge]
                    if eid in cross:
                        continue
                    if edge == 0:
                        p, q, x0, y0, dx, dy = a, b, j, i, 1, 0
                    elif edge == 1:
                        p, q, x0, y0, dx, dy = b, c, j+1, i, 0, 1
                    elif edge == 2:
                        p, q, x0, y0, dx, dy = d, c, j, i+1, 1, 0
                    else:
                        p, q, x0, y0, dx, dy = a, d, j, i, 0, 1
                    t = (h-p)/(q-p)
                    cross[eid] = (x0+dx*t, y0+dy*t)
                segments[h].extend((ids[e0], ids[e1]) for e0, e1 in pairs)
    return {h: _chain(segments[h], crossings[h]) for h in levels}


def _chain(segments, points):
    ends = {}
    for k, (e0, e1) in enumerate(segments):
        ends.setdefault(e0, []).append(k)
        ends.setdefault(e1, []).append(k)
    used = bytearray(len(segments))
    lines = []

    def walk(eid, k):
        path = []
        while True:
            used[k] = 1
            e0, e1 = segments[k]
            eid = e1 if e0 == eid else e0
            path.append(eid)
            nxt = [s for s in ends[eid] if not used[s]]
            if not nxt:
                return path
            k = nxt[0]

    for k, (e0, e1) in enumerate(segments):
        if used[k]:
            continue
        forward = walk(e0, k)                       # e0 -> e1 -> ...
        backward = [s for s in ends[e0] if not used[s]]
        head = walk(e0, backward[0]) if backward else []
        ids = head[::-1]+[e0]+forward
        lines.append([points[e] for e in ids])
    return lines


def simplify(points, tolerance):
    """Douglas-Peucker; keeps both ends (a closed line keeps its repeated first point)."""
    if len(points) < 3:
        return list(points)
    keep = bytearray(len(points))
    keep[0] = keep[-1] = 1
    stack = [(0, len(points)-1)]
    tol2 = tolerance*tolerance
    while stack:
        s, e = stack.pop()
        (x0, y0), (x1, y1) = points[s], points[e]
        dx, dy = x1-x0, y1-y0
        n2 = dx*dx+dy*dy
        worst, at = -1., -1
        for k in range(s+1, e):
            px, py = points[k][0]-x0, points[k][1]-y0
            if n2 > 0.:
                cr = px*dy-py*dx
                d2 = cr*cr/n2
            else:
                d2 = px*px+py*py
            if d2 > worst:
                worst, at = d2, k
        if at > 0 and worst > tol2:
            keep[at] = 1
            stack.append((s, at))
            stack.append((at, e))
    return [p for p, k in zip(points, keep) if k]


# -- build / cache -------------------------------------------------------------------------------------------------

def build(root, stem, target=TARGET):
    folder = _folder(terrain_root(root), stem)
    meta = json.loads((folder/"metadata.json").read_text(encoding="utf-8"))
    size = max(meta["width"], meta["height"])
    stride = max(1, math.ceil(size/target))
    nx, nz, grid = read_decimated(folder/"height_world_y_m.tif", stride)
    cell = meta["cell_size_m"]*stride
    x0, z0 = meta["origin_x_m"], meta["origin_z_m"]
    finite = [v for v in grid if v == v]
    if not finite:
        raise ValueError(f"地图 {stem} 没有有效地形样点")
    lo, hi = min(finite), max(finite)
    minor, major = pick_interval(relief(grid, nx, nz))
    levels = {float(h): ("major" if h % major == 0 else "minor")
              for h in range(math.ceil(lo/minor)*minor, math.floor(hi/minor)*minor+1, minor)}
    water = meta.get("water_level_m_READ")
    if isinstance(water, (int, float)):
        for h in levels:
            if h < water:
                levels[h] = "under"
        if lo < water <= hi:
            levels[float(water)] = "water"
    out_levels = []
    for h, lines in sorted(contour(grid, nx, nz, list(levels)).items()):
        polys = []
        for line in lines:
            line = simplify(line, SIMPLIFY_CELLS)
            if len(line) < 2:
                continue
            flat = []
            for c, r in line:
                flat.extend((round(x0+c*cell), round(z0+r*cell)))
            polys.append(flat)
        if polys:
            out_levels.append(dict(h=h if h % 1 else int(h), kind=levels[h], lines=polys))
    heights = array("h", (-32768 if v != v else max(-32767, min(32767, round(v))) for v in grid))
    if sys.byteorder != "little":
        heights.byteswap()
    return dict(version=VERSION, map=stem, x0_m=x0, z0_m=z0, cell_m=cell, nx=nx, nz=nz,
                heights_i16le=base64.b64encode(heights.tobytes()).decode("ascii"),
                height_min_m=round(lo, 1), height_max_m=round(hi, 1), interval_m=minor, major_m=major,
                water_m=water if isinstance(water, (int, float)) else None,
                vertices=sum(len(p)//2 for lv in out_levels for p in lv["lines"]), levels=out_levels)


_LOCK = threading.Lock()
_MEMORY = {}


def load(stem, *, root=None, cache_dir=None):
    """Contours for one map, from memory, the JSON cache (keyed by the TIFF's size and mtime), or built."""
    root = terrain_root(root)
    tif = _folder(root, stem)/"height_world_y_m.tif"
    st = tif.stat()
    key = [VERSION, TARGET, SEGMENT_BUDGET, st.st_size, st.st_mtime_ns]
    with _LOCK:
        hit = _MEMORY.get(stem)
        if hit is not None and hit[0] == key:
            return hit[1]
        cached = Path(cache_dir)/f"{stem}.json" if cache_dir else None
        data = None
        if cached is not None and cached.is_file():
            try:
                stored = json.loads(cached.read_text(encoding="utf-8"))
                if stored.get("key") == key:
                    data = stored["data"]
            except (OSError, ValueError, KeyError):
                data = None
        if data is None:
            data = build(root, stem)
            if cached is not None:
                try:
                    cached.parent.mkdir(parents=True, exist_ok=True)
                    tmp = cached.with_suffix(".tmp")
                    tmp.write_text(json.dumps(dict(key=key, data=data), separators=(",", ":")), encoding="utf-8")
                    tmp.replace(cached)
                except OSError:
                    pass
        _MEMORY.clear()          # one map at a time is enough for the page
        _MEMORY[stem] = (key, data)
        return data


def main(argv=None):
    """Build the cache for some or all maps: python -m wt_overlay.terrain [map ...] [--cache DIR]."""
    import argparse
    import time
    parser = argparse.ArgumentParser(description="生成沙盘地形等高线缓存")
    parser.add_argument("maps", nargs="*")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--cache", type=Path, default=ROOT/"outputs"/"sandbox"/"terrain_cache")
    args = parser.parse_args(argv)
    names = args.maps or [m["map"] for m in list_maps(args.root)]
    for name in names:
        start = time.monotonic()
        data = load(name, root=args.root, cache_dir=args.cache)
        print(f"{name:28s} {data['nx']}x{data['nz']} cell {data['cell_m']:g} m  interval {data['interval_m']} / "
              f"{data['major_m']} m  {len(data['levels'])} levels  {data['vertices']} vertices  "
              f"{time.monotonic()-start:.1f} s", flush=True)


if __name__ == "__main__":
    main()
