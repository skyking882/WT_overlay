import json
import math
from pathlib import Path
import struct
import tempfile
import unittest
import zlib

from wt_overlay import terrain


def write_tiff(path, width, height, value, rows_per_strip=4, compress=True):
    """Single-band float32 little-endian strip TIFF, Deflate or uncompressed, as the map export writes it."""
    strips = []
    for first in range(0, height, rows_per_strip):
        raw = b"".join(struct.pack("<f", value(r, c)) for r in range(first, min(first+rows_per_strip, height))
                       for c in range(width))
        strips.append(zlib.compress(raw) if compress else raw)
    n = len(strips)
    data_at = 8
    offsets, at = [], data_at
    for s in strips:
        offsets.append(at)
        at += len(s)
    arrays_at = at
    tags = [(256, 4, 1, width), (257, 4, 1, height), (258, 3, 1, 32), (259, 3, 1, 8 if compress else 1),
            (262, 3, 1, 1), (273, 4, n, arrays_at), (277, 3, 1, 1), (278, 4, 1, rows_per_strip),
            (279, 4, n, arrays_at+4*n), (284, 3, 1, 1), (339, 3, 1, 3)]
    ifd_at = arrays_at+8*n
    with open(path, "wb") as f:
        f.write(b"II"+struct.pack("<HI", 42, ifd_at))
        for s in strips:
            f.write(s)
        f.write(struct.pack(f"<{n}I", *offsets))
        f.write(struct.pack(f"<{n}I", *(len(s) for s in strips)))
        f.write(struct.pack("<H", len(tags)))
        for tag, kind, count, value_ in tags:
            if count == 1 and kind == 3:
                f.write(struct.pack("<HHIHH", tag, kind, count, value_, 0))
            else:
                f.write(struct.pack("<HHII", tag, kind, count, value_))
        f.write(struct.pack("<I", 0))


def cone(r, c, size=64, peak=1000., slope=40.):
    if r < 20 and c < 20:
        return math.nan          # an uncovered corner
    return peak-slope*math.hypot(r-size/2, c-size/2)


def make_map(root, stem="air_test", size=64, cell=100., water=200.):
    folder = Path(root)/stem
    folder.mkdir(parents=True)
    write_tiff(folder/"height_world_y_m.tif", size, size, lambda r, c: cone(r, c, size))
    (folder/"metadata.json").write_text(json.dumps(dict(
        width=size, height=size, cell_size_m=cell, origin_x_m=-size*cell/2, origin_z_m=-size*cell/2,
        height_min_m=-500., height_max_m=1000., water_level_m_READ=water)))
    return folder


class TerrainTests(unittest.TestCase):
    def test_map_stem_from_level(self):
        self.assertEqual(terrain.map_stem("levels/air_afghan.bin"), "air_afghan")
        self.assertEqual(terrain.map_stem("levels\\spain.bin"), "spain")
        self.assertEqual(terrain.map_stem("../maps/korea"), "korea")      # only the last component is used
        self.assertIsNone(terrain.map_stem("levels/air afghan.bin"))
        self.assertIsNone(terrain.map_stem("levels/"))
        self.assertIsNone(terrain.map_stem(None))

    def test_tiff_decimation_matches_full_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            for compress in (True, False):
                path = Path(tmp)/f"h{compress}.tif"
                write_tiff(path, 10, 9, lambda r, c: r*100.+c, rows_per_strip=4, compress=compress)
                nx, nz, full = terrain.read_decimated(path, 1)
                self.assertEqual((nx, nz), (10, 9))
                self.assertEqual(list(full), [r*100.+c for r in range(9) for c in range(10)])
                nx, nz, half = terrain.read_decimated(path, 3)
                self.assertEqual((nx, nz), (4, 3))
                self.assertEqual(list(half), [r*100.+c for r in (0, 3, 6) for c in (0, 3, 6, 9)])

    def test_cone_contours_are_closed_circles_and_stop_at_missing_data(self):
        size = 64
        grid = [cone(r, c, size) for r in range(size) for c in range(size)]
        lines = terrain.contour(grid, size, size, [600., 0.])
        (ring,) = lines[600.]
        self.assertEqual(ring[0], ring[-1])
        for c, r in ring:                                  # radius 10 cells, within the marching-squares error
            self.assertAlmostEqual(math.hypot(r-size/2, c-size/2), 10., delta=.1)
        # The 0 m circle (radius 25) runs into the NaN corner: it is cut there, never closed through it.
        self.assertTrue(lines[0.])
        for line in lines[0.]:
            self.assertFalse(any(r < 20 and c < 20 for c, r in line))
            self.assertNotEqual(line[0], line[-1])

    def test_interval_follows_relief_budget(self):
        self.assertEqual(terrain.pick_interval(1000., budget=100), (10, 50))
        self.assertEqual(terrain.pick_interval(4000., budget=100), (50, 250))
        self.assertEqual(terrain.pick_interval(1e12, budget=100), (1000, 5000))

    def test_simplify_keeps_corners_and_ends(self):
        line = [(0, 0), (1, 0.01), (2, 0), (2, 1), (2, 2)]
        self.assertEqual(terrain.simplify(line, .1), [(0, 0), (2, 0), (2, 2)])

    def test_build_load_and_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, cache = Path(tmp)/"maps", Path(tmp)/"cache"
            make_map(root)
            self.assertEqual([m["map"] for m in terrain.list_maps(root)], ["air_test"])
            self.assertEqual(terrain.map_center(root, "air_test"), [-50., -50.])
            data = terrain.load("air_test", root=root, cache_dir=cache)
            self.assertEqual((data["nx"], data["nz"], data["cell_m"]), (64, 64, 100.))
            kinds = {lv["h"]: lv["kind"] for lv in data["levels"]}
            self.assertEqual(kinds[200], "water")
            self.assertTrue(all(lv["kind"] == "under" for lv in data["levels"] if lv["h"] < 200))
            self.assertTrue(all(lv["kind"] == "major" for lv in data["levels"] if lv["h"] % data["major_m"] == 0
                                and lv["h"] > 200))
            # Contours are in Dagor X/Z metres: the 600 m ring is 1 km around the peak at sample (32, 32) = (0, 0).
            ring = next(lv for lv in data["levels"] if lv["h"] == 600)["lines"][0]
            for x, z in zip(ring[::2], ring[1::2]):
                self.assertAlmostEqual(math.hypot(x, z), 1000., delta=15.)
            self.assertTrue((cache/"air_test.json").is_file())
            terrain._MEMORY.clear()
            self.assertEqual(terrain.load("air_test", root=root, cache_dir=cache), data)
            self.assertEqual(data["center_dagor_xz"], [-50., -50.])
            with self.assertRaises(ValueError):
                terrain.load("../air_test", root=root)
            with self.assertRaises(FileNotFoundError):
                terrain.load("air_missing", root=root)

    def test_the_caches_alone_are_enough(self):
        # the terrain pack: contour caches without the rasters (a release asset unzipped into the cache directory)
        import json, shutil
        with tempfile.TemporaryDirectory() as tmp:
            root, cache = Path(tmp)/"maps", Path(tmp)/"cache"
            make_map(root)
            built = terrain.load("air_test", root=root, cache_dir=cache)
            shutil.rmtree(root)
            terrain._MEMORY.clear()
            self.assertEqual(terrain.list_maps(root), [])
            self.assertEqual(terrain.list_maps(root, cache_dir=cache),
                             [dict(map="air_test", width_km=6.4, height_min_m=round(built["height_min_m"]),
                                   height_max_m=round(built["height_max_m"]))])
            self.assertEqual(terrain.load("air_test", root=root, cache_dir=cache), built)
            self.assertEqual(built["center_dagor_xz"], [-50., -50.])
            with self.assertRaises(FileNotFoundError):
                terrain.load("air_test", root=root)                       # no cache directory given
            with self.assertRaises(ValueError):
                terrain.load("../air_test", root=root, cache_dir=cache)   # names are checked before any file
            # a cache of another version is neither listed nor loaded
            old = json.loads((cache/"air_test.json").read_text())
            old["data"]["version"] = terrain.VERSION-1
            (cache/"air_old.json").write_text(json.dumps(dict(old, data=dict(old["data"], map="air_old"))))
            self.assertEqual([m["map"] for m in terrain.list_maps(root, cache_dir=cache)], ["air_test"])
            with self.assertRaises(FileNotFoundError):
                terrain.load("air_old", root=root, cache_dir=cache)


if __name__ == "__main__":
    unittest.main()
