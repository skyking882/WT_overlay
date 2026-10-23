"""Direct .wrpl import (docs/wrpl_import_spec.md §8.1-§8.3, §8.5) with the fake backend tests/wrpl_fake_backend.py.
No network: downloads use file:// URLs."""
import contextlib
import hashlib
import http.client
import io
import json
import os
from pathlib import Path
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock
import zipfile

from wt_overlay.sandbox import SandboxServer, SandboxSession
from wt_overlay.wrpl import __main__ as cli
from wt_overlay.wrpl import backend as backends
from wt_overlay.wrpl import gamefiles, importer

ROOT = Path(__file__).resolve().parents[1]
FAKE = ROOT/"tests"/"wrpl_fake_backend.py"
VERSION = (2, 59, 0, 54)


def write_game(folder, version=VERSION):
    folder.mkdir(parents=True, exist_ok=True)
    packed = version[0] << 24 | version[1] << 16 | version[2] << 8 | version[3]
    for name in gamefiles.REQUIRED:
        (folder/name).write_bytes(b"VRFx\0\0PC"+bytes(8)+struct.pack("<HHI", 8, 0, packed)+os.urandom(16))
    return folder


def write_replay(path, version=VERSION, payload=None):
    head = bytearray(gamefiles.REPLAY_HEADER_SIZE)
    struct.pack_into("<II", head, 0, gamefiles.REPLAY_HEADER_MAGIC, 0x18C1C)
    struct.pack_into("<Q", head, gamefiles.REPLAY_VERSION_OFFSET,
                     version[0] << 48 | version[1] << 32 | version[2] << 16 | version[3])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(head)+(payload if payload is not None else os.urandom(2048)))
    return path


@contextlib.contextmanager
def env(**values):
    old = {k: os.environ.get(k) for k in values}
    os.environ.update({k: v for k, v in values.items() if v is not None})
    for k, v in values.items():
        if v is None:
            os.environ.pop(k, None)
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="wrpl-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.game = write_game(self.tmp/"game")
        self.out = self.tmp/"out"
        self.calls = self.tmp/"calls.txt"
        patches = [mock.patch.object(importer, "SETTINGS", self.tmp/"settings.json"),
                   mock.patch.object(importer, "WORK", self.tmp/"work"),
                   mock.patch.object(importer, "KEPT_EXPORTS", self.tmp/"kept"),
                   mock.patch.object(importer, "INBOX", self.tmp/"inbox")]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        ctx = env(WT_WRPL_BACKEND=None, WT_WRPL_CACHE=str(self.tmp/"cache"), FAKE_WRPL_MODE="ok",
                  FAKE_WRPL_CALLS=str(self.calls))
        ctx.__enter__()
        self.addCleanup(ctx.__exit__, None, None, None)

    def backend_calls(self):
        return len(self.calls.read_text().splitlines()) if self.calls.exists() else 0

    def header(self, path):
        with open(path, encoding="utf-8") as handle:
            return json.loads(handle.readline())

    def run_cli(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli.main([str(a) for a in args])
        return code, out.getvalue(), err.getvalue()


# -- §8.3 game folder search -----------------------------------------------------------------------------------

class GameFilesTests(Base):
    def test_versions_from_headers(self):
        replay = write_replay(self.tmp/"r.wrpl")
        self.assertEqual(gamefiles.replay_info(replay), dict(version="2.59.0.54", format=0x18C1C))
        self.assertEqual(gamefiles.game_version(self.game), "2.59.0.54")
        self.assertFalse(gamefiles.is_replay(self.game/"aces.vromfs.bin"))
        self.assertEqual(gamefiles.version_match("2.59.0.44", "2.59.0.54"), "build")
        self.assertEqual(gamefiles.version_match("2.57.0.10", "2.59.0.54"), "other")

    def test_libraryfolders_vdf_old_and_new(self):
        new = '"libraryfolders"\n{\n\t"0"\n\t{\n\t\t"path"\t\t"C:\\\\Program Files (x86)\\\\Steam"\n\t\t"apps"\n\t\t{\n' \
              '\t\t\t"236390"\t\t"123"\n\t\t}\n\t}\n\t"1"\n\t{\n\t\t"path"\t\t"D:\\\\SteamLibrary"\n\t}\n}\n'
        old = '"LibraryFolders"\n{\n\t"TimeNextStatsReport"\t\t"1"\n\t"1"\t\t"E:\\\\Games\\\\Steam"\n}\n'
        self.assertEqual(gamefiles.parse_vdf_paths(new), [r"C:\Program Files (x86)\Steam", r"D:\SteamLibrary"])
        self.assertEqual(gamefiles.parse_vdf_paths(old), [r"E:\Games\Steam"])

    def test_steam_library_from_vdf(self):
        steam = self.tmp/"Steam"
        library = self.tmp/"lib2"
        write_game(library/"steamapps"/"common"/"War Thunder")
        (steam/"steamapps").mkdir(parents=True)
        (steam/"steamapps"/"libraryfolders.vdf").write_text(
            '"libraryfolders"\n{\n "0" { "path" "%s" }\n "1" { "path" "%s" }\n}\n'
            % (str(steam).replace("\\", "\\\\"), str(library).replace("\\", "\\\\")), encoding="utf-8")
        replay = write_replay(self.tmp/"elsewhere"/"r.wrpl")
        folder, info = gamefiles.find_game_dir(replay, platform="win32", env={}, steam_roots=[steam])
        self.assertEqual(folder, library/"steamapps"/"common"/"War Thunder")
        self.assertEqual(info["reason"], "steam")

    def test_order_explicit_remembered_replay_parent(self):
        game = write_game(self.tmp/"WarThunder")
        replay = write_replay(game/"Replays"/"#2026.10.20 10.00.00.wrpl")
        other = write_game(self.tmp/"other")
        kw = dict(platform="linux", env={"HOME": str(self.tmp/"home")}, steam_roots=[])
        self.assertEqual(gamefiles.find_game_dir(replay, **kw)[1]["reason"], "replay_parent")
        self.assertEqual(gamefiles.find_game_dir(replay, remembered=other, **kw)[0], other)
        self.assertEqual(gamefiles.find_game_dir(replay, explicit=self.game, remembered=other, **kw)[0], self.game)
        order = [r for _, r in gamefiles.candidates(replay, remembered=other, **kw)]
        self.assertEqual(order[:3], ["remembered", "replay_parent", "replay_folder"])
        self.assertEqual([r for _, r in gamefiles.candidates(replay, explicit=other, **kw)], ["explicit"])
        with self.assertRaises(gamefiles.GameFilesError) as ctx:
            gamefiles.find_game_dir(replay, explicit=self.tmp/"empty", **kw)
        self.assertIn("aces.vromfs.bin", str(ctx.exception))

    def test_version_choice_and_mismatch(self):
        kw = dict(platform="linux", env={"HOME": str(self.tmp/"home")}, steam_roots=[])
        replay = write_replay(self.tmp/"Replays"/"r.wrpl", version=(2, 59, 0, 44))
        build_only = write_game(self.tmp/"g54", (2, 59, 0, 54))
        folder, info = gamefiles.find_game_dir(replay, remembered=build_only, **kw)
        self.assertEqual(folder, build_only)
        self.assertEqual(info["version_note"], "回放版本 2.59.0.44，游戏文件版本 2.59.0.54")
        exact = write_game(self.tmp, (2, 59, 0, 44))            # the replay folder's parent
        self.assertTrue(gamefiles.find_game_dir(replay, remembered=build_only, **kw)[0].samefile(exact))
        old = write_replay(self.tmp/"x"/"Replays"/"old.wrpl", version=(2, 57, 0, 1))
        with self.assertRaises(gamefiles.GameFilesError) as ctx:
            gamefiles.find_game_dir(old, explicit=build_only, **kw)
        self.assertEqual(str(ctx.exception), "回放版本 2.57.0.1，游戏文件版本 2.59.0.54")
        self.assertEqual(gamefiles.find_game_dir(old, explicit=build_only, check_version=False, **kw)[0], build_only)

    def test_remembered_folder_round_trip(self):
        path = self.tmp/"s"/"wrpl_settings.json"
        self.assertEqual(gamefiles.load_settings(path), {})
        gamefiles.remember_game_dir(path, self.game)
        self.assertEqual(gamefiles.load_settings(path)["game_dir"], str(self.game.resolve()))


# -- §8.2 backend lookup and download ---------------------------------------------------------------------------

def make_bundle(folder, script_body=None):
    """A one-folder 'bundle': wrpl_export/wrpl_export (executable) + a symlink, zipped like build.py does."""
    bundle = folder/"wrpl_export"
    (bundle/"_internal").mkdir(parents=True)
    exe = bundle/"wrpl_export"
    exe.write_text(script_body or f"#!{sys.executable}\nimport runpy, sys\nsys.argv[0] = {str(FAKE)!r}\n"
                                   f"runpy.run_path({str(FAKE)!r}, run_name='__main__')\n", encoding="utf-8")
    exe.chmod(0o755)
    (bundle/"_internal"/"data.txt").write_text("x", encoding="utf-8")
    os.symlink("data.txt", bundle/"_internal"/"link.txt")
    import importlib.util
    spec = importlib.util.spec_from_file_location("wrpl_build", ROOT/"tools"/"wrpl_backend"/"build.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    target = folder/"wrpl_export-test.zip"
    build.write_zip(bundle, target)
    return target


def manifest_for(zip_path, platform_key, sha=None):
    data = zip_path.read_bytes()
    return {"version": "9.9.9", "upstream_revision": "u", "fork": "f", "fork_branch": "b", "fork_commit": "c",
            "assets": {platform_key: {"file": zip_path.name, "url": zip_path.resolve().as_uri(),
                                      "sha256": sha or hashlib.sha256(data).hexdigest(), "size": len(data)}}}


@unittest.skipIf(os.name == "nt", "the test bundle is a POSIX script with a symlink")
class BackendTests(Base):
    key = "linux-x64"

    def setUp(self):
        super().setUp()
        self.zip = make_bundle(self.tmp/"bundle")
        self.cache = self.tmp/"cache"

    def test_published_asset_downloads_verifies_and_hits_the_cache(self):
        manifest = manifest_for(self.zip, self.key)
        with self.assertRaises(backends.BackendMissing) as ctx:
            backends.find_backend(cache=self.cache, manifest=manifest, keys=[self.key])
        asset = ctx.exception.asset
        self.assertEqual((asset.platform, asset.size), (self.key, self.zip.stat().st_size))
        self.assertIn("需要下载解析后端", str(ctx.exception))
        found = backends.download(asset, self.cache)
        folder = self.cache/"9.9.9"/self.key
        self.assertTrue(found.path.samefile(folder/"wrpl_export"/"wrpl_export"))
        self.assertTrue(os.access(found.path, os.X_OK))
        self.assertTrue((folder/"wrpl_export"/"_internal"/"link.txt").is_symlink())
        self.assertEqual(backends.version(found)["backend_version"], "fake")
        # cache hit: no download even when the URL is gone
        manifest["assets"][self.key]["url"] = (self.tmp/"gone.zip").as_uri()
        self.assertEqual(backends.find_backend(cache=self.cache, manifest=manifest, keys=[self.key]).source, "cache")
        self.assertEqual(list(folder.parent.glob("*.part")), [])

    def test_sha256_mismatch_is_refused_and_deleted(self):
        manifest = manifest_for(self.zip, self.key, sha="0"*64)
        asset = backends.published_asset(manifest, [self.key])
        with self.assertRaises(backends.BackendError) as ctx:
            backends.download(asset, self.cache)
        self.assertIn("SHA-256", str(ctx.exception))
        self.assertEqual(list((self.cache/"9.9.9").iterdir()), [])
        self.assertIsNone(backends.cached_backend(self.cache, manifest, [self.key]))

    def test_env_beats_cache_and_explicit_beats_env(self):
        manifest = manifest_for(self.zip, self.key)
        backends.download(backends.published_asset(manifest, [self.key]), self.cache)
        self.assertEqual(backends.find_backend(cache=self.cache, manifest=manifest, keys=[self.key],
                                               env={backends.ENV: str(FAKE)}).source, "env")
        own = self.tmp/"own"/"wrpl_export"
        own.parent.mkdir()
        shutil.copy(FAKE, own)
        found = backends.find_backend(own.parent, cache=self.cache, manifest=manifest, keys=[self.key],
                                      env={backends.ENV: str(FAKE)})
        self.assertEqual((found.source, found.path), ("explicit", own))
        with self.assertRaises(backends.BackendError):
            backends.find_backend(self.tmp/"nothing", env={})

    def test_unpublished_platform_and_bad_archives(self):
        with self.assertRaises(backends.BackendMissing) as ctx:
            backends.find_backend(cache=self.cache, manifest=backends.load_manifest(), env={})
        self.assertIsNone(ctx.exception.asset)
        evil = self.tmp/"evil.zip"
        with zipfile.ZipFile(evil, "w") as z:
            z.writestr("../outside.txt", "x")
        asset = backends.published_asset(manifest_for(evil, self.key), [self.key])
        with self.assertRaises(backends.BackendError):
            backends.download(asset, self.cache)
        self.assertFalse((self.cache/"9.9.9"/"outside.txt").exists())

    def test_platform_keys_and_cache_roots(self):
        self.assertEqual(backends.platform_keys("darwin", "arm64"), ["macos-arm64", "macos-x64"])
        self.assertEqual(backends.platform_keys("darwin", "x86_64"), ["macos-x64"])
        self.assertEqual(backends.platform_keys("win32", "AMD64"), ["windows-x64"])
        self.assertEqual(backends.cache_root("win32", {"LOCALAPPDATA": "C:/L"}), Path("C:/L/wt_overlay/wrpl"))
        self.assertEqual(backends.cache_root("linux", {"XDG_CACHE_HOME": "/c"}), Path("/c/wt_overlay/wrpl"))


# -- §8.1 command line -----------------------------------------------------------------------------------------

class CliTests(Base):
    def test_import_duplicate_force_and_names(self):
        replay = write_replay(self.tmp/"rep"/"#2026.10.20 10.00.00.wrpl")
        code, out, err = self.run_cli(replay, "--game", self.game, "--backend", FAKE, "--out-dir", self.out)
        self.assertEqual(code, 0, err)
        path = self.out/"2026.10.20_10.00.00.jsonl"
        head = self.header(path)
        self.assertEqual(head["source"]["replay_sha256"], hashlib.sha256(replay.read_bytes()).hexdigest())
        self.assertEqual(head["source"]["wt_version"], "2.59.0.54")
        self.assertEqual(len(head["planes"]), 2)
        self.assertEqual(json.loads(importer.SETTINGS.read_text())["game_dir"], str(self.game.resolve()))
        self.assertEqual(list((self.tmp/"work").iterdir()), [])
        # again: recognised by SHA-256, the backend is not run
        code, out, _ = self.run_cli(replay, "--backend", FAKE, "--out-dir", self.out)
        self.assertEqual((code, self.backend_calls()), (0, 1))
        self.assertIn("已导入过", out)
        # --force: re-imported under the same name
        code, _, _ = self.run_cli(replay, "--backend", FAKE, "--out-dir", self.out, "--force")
        self.assertEqual((code, self.backend_calls(), sorted(p.name for p in self.out.iterdir())),
                         (0, 2, ["2026.10.20_10.00.00.jsonl"]))
        # another replay with the same file name: suffix
        other = write_replay(self.tmp/"rep2"/"#2026.10.20 10.00.00.wrpl")
        self.assertEqual(self.run_cli(other, "--backend", FAKE, "--out-dir", self.out)[0], 0)
        self.assertTrue((self.out/"2026.10.20_10.00.00_2.jsonl").is_file())

    def test_folder_argument_is_not_recursive(self):
        folder = self.tmp/"Replays"
        write_replay(folder/"a.wrpl")
        write_replay(folder/"b.WRPL")
        write_replay(folder/"sub"/"c.wrpl")
        code, _, err = self.run_cli(folder, "--game", self.game, "--backend", FAKE, "--out-dir", self.out)
        self.assertEqual(code, 0, err)
        self.assertEqual(sorted(p.name for p in self.out.iterdir()), ["a.jsonl", "b.jsonl"])

    def test_backend_failure_and_crash(self):
        replay = write_replay(self.tmp/"r.wrpl")
        with env(FAKE_WRPL_MODE="fail"):
            code, _, err = self.run_cli(replay, "--game", self.game, "--backend", FAKE, "--out-dir", self.out)
        self.assertEqual(code, 1)
        for text in ("解析失败（退出码 1）", "replay rejected by the parser", "fake parser failure"):
            self.assertIn(text, err)
        with env(FAKE_WRPL_MODE="crash"):
            code, _, err = self.run_cli(replay, "--game", self.game, "--backend", FAKE, "--out-dir", self.out)
        self.assertEqual(code, 1)
        self.assertIn("退出码 139", err)
        self.assertFalse(self.out.exists() and any(self.out.iterdir()))
        self.assertEqual(list((self.tmp/"work").iterdir()), [])

    def test_timeout(self):
        replay = write_replay(self.tmp/"r.wrpl")
        with env(FAKE_WRPL_MODE="sleep"):
            with self.assertRaises(importer.ImportFailed) as ctx:
                importer.import_replay(replay, game=self.game, backend=FAKE, out_dir=self.out, timeout_s=1)
        self.assertEqual(str(ctx.exception), "解析超时（1 秒）")

    def test_missing_game_files_and_not_a_replay(self):
        replay = write_replay(self.tmp/"lonely"/"r.wrpl")
        with mock.patch.object(gamefiles, "INSTALL_CANDIDATES", {k: [] for k in gamefiles.INSTALL_CANDIDATES}), \
                mock.patch.object(gamefiles, "STEAM_ROOTS", {k: [] for k in gamefiles.STEAM_ROOTS}), \
                mock.patch.object(gamefiles, "_windows_steam_root", lambda: []):
            code, _, err = self.run_cli(replay, "--backend", FAKE, "--out-dir", self.out)
        self.assertEqual(code, 1)
        self.assertIn("找不到游戏文件", err)
        bogus = self.tmp/"x.wrpl"
        bogus.write_bytes(b"not a replay")
        code, _, err = self.run_cli(bogus, "--game", self.game, "--backend", FAKE, "--out-dir", self.out)
        self.assertIn("不是 War Thunder 回放", err)
        self.assertEqual(self.backend_calls(), 0)

    def test_no_backend_non_interactive(self):
        replay = write_replay(self.tmp/"r.wrpl")
        with mock.patch.object(sys.stdin, "isatty", lambda: False):
            code, _, err = self.run_cli(replay, "--game", self.game, "--out-dir", self.out)
        self.assertEqual(code, 1)
        self.assertIn("还没有发布的解析后端", err)

    @unittest.skipIf(os.name == "nt", "the test bundle is a POSIX script")
    def test_download_needs_the_flag_when_not_interactive(self):
        zip_path = make_bundle(self.tmp/"bundle")
        key = backends.platform_keys()[0]
        manifest = manifest_for(zip_path, key)
        replay = write_replay(self.tmp/"r.wrpl")
        with mock.patch.object(backends, "load_manifest", lambda path=None: manifest), \
                mock.patch.object(sys.stdin, "isatty", lambda: False):
            code, _, err = self.run_cli(replay, "--game", self.game, "--out-dir", self.out)
            self.assertEqual(code, 1)
            self.assertIn("--download-backend", err)
            self.assertFalse((self.tmp/"cache"/"9.9.9").exists() and any((self.tmp/"cache"/"9.9.9").iterdir()))
            code, _, err = self.run_cli(replay, "--game", self.game, "--out-dir", self.out, "--download-backend")
        self.assertEqual(code, 0, err)
        self.assertTrue((self.tmp/"cache"/"9.9.9"/key/backends.VERIFIED).is_file())

    def test_keep_export(self):
        replay = write_replay(self.tmp/"#k.wrpl")
        result = importer.import_replay(replay, game=self.game, backend=FAKE, out_dir=self.out, keep_export=True)
        self.assertEqual(Path(result["export"]), self.tmp/"kept"/"k"/"match_export.json")
        self.assertEqual(self.header(result["path"])["source"]["export_file"], result["export"])


# -- §8.5 the converter is unchanged ---------------------------------------------------------------------------

class ConverterTests(Base):
    def test_importer_output_is_the_converter_output_plus_wt_version(self):
        replay = write_replay(self.tmp/"#c.wrpl")
        result = importer.import_replay(replay, game=self.game, backend=FAKE, out_dir=self.out, keep_export=True)
        direct = self.tmp/"direct"
        subprocess.run([sys.executable, str(importer.CONVERTER), result["export"], "--out-dir", str(direct),
                        "--name", "c"], check=True, capture_output=True, cwd=ROOT)
        a, b = Path(result["path"]).read_bytes().split(b"\n"), (direct/"c.jsonl").read_bytes().split(b"\n")
        self.assertEqual(a[1:], b[1:])
        ha, hb = json.loads(a[0]), json.loads(b[0])
        self.assertEqual(ha["source"].pop("wt_version"), "2.59.0.54")
        self.assertEqual(ha, hb)


def load_module(name, path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ConverterTieTests(Base):
    """scripts/wt_replay_import.py: a plane's header missile when two weapon names are launched equally often."""
    def test_main_weapon_rule(self):
        conv = load_module("wt_replay_import_tie", importer.CONVERTER)
        self.assertIsNone(conv.main_weapon([]))
        self.assertEqual(conv.main_weapon(["us_aim_120c_5_default", "us_aim_120c_5"]), "us_aim_120c_5")
        self.assertEqual(conv.main_weapon(["us_aim_120c_5", "us_aim_120c_5_default"]), "us_aim_120c_5")
        self.assertEqual(conv.main_weapon(["su_r_77_1", "su_r_73"]), "su_r_77_1")          # first launched
        self.assertEqual(conv.main_weapon(["a_default", "b_default"]), "a_default")
        # different missiles: first launched, a *_default name is a real missile too
        self.assertEqual(conv.main_weapon(["su_r_77_1_default", "su_r_27et"]), "su_r_77_1_default")
        self.assertEqual(conv.main_weapon(["us_aim7p_sparrow_fa_18", "us_aim9m_sidewinder_default"]),
                         "us_aim7p_sparrow_fa_18")
        self.assertEqual(conv.main_weapon(["su_r_73", "su_r_77_1_default", "su_r_77_1"]), "su_r_73")
        self.assertEqual(conv.main_weapon(["x_default", "y", "x_default"]), "x_default")   # count first

    def test_header_missile_does_not_depend_on_the_hash_seed(self):
        fake = load_module("wrpl_fake_backend_tie", FAKE)
        replay = write_replay(self.tmp/"tie.wrpl")
        export = fake.export(replay)

        def rocket(eid, name, t0):
            track = [{"t_ms": t0+i*250, "xyz": [-30000.0+0.3*t0+150.0*i, 6000.0, 0.0], "euler_raw": [0.0, 0.0, 0.0]}
                     for i in range(40)]
            return {"eid": eid, "owner_eid": 1001, "eid2": 0, "weapon_name": name, "template": "rocket+arh_shell",
                    "weapon_path": "", "created_at_ms": t0, "destroyed_at_ms": t0+10000, "track": track, "sync": []}
        export["rockets"] = [rocket(5001, "us_aim_120c_5_default", 5000), rocket(5002, "us_aim_120c_5", 9000)]
        path = self.tmp/"tie"/"match_export.json"
        path.parent.mkdir()
        path.write_text(json.dumps(export), encoding="utf-8")
        outputs = set()
        for seed in ("1", "2", "3", "4", "5", "6"):
            out = self.tmp/f"seed{seed}"
            subprocess.run([sys.executable, str(importer.CONVERTER), str(path), "--out-dir", str(out), "--name", "t"],
                           check=True, capture_output=True, cwd=ROOT, env=dict(os.environ, PYTHONHASHSEED=seed))
            outputs.add((out/"t.jsonl").read_bytes())
            self.assertEqual(self.header(out/"t.jsonl")["planes"][0]["missile"], "us_aim_120c_5")
        self.assertEqual(len(outputs), 1)


# -- §8.1 sandbox POST /api/wrpl --------------------------------------------------------------------------------

def light_session(queue):
    """SandboxSession without the physics world (no missile models needed): only what the wrpl routes use."""
    session = SandboxSession.__new__(SandboxSession)
    session.lock, session.jobs, session.snapshot, session.wrpl = threading.Lock(), {}, {"status": "setup"}, queue
    return session


class SandboxImportTests(Base):
    def setUp(self):
        super().setUp()
        self.queue = None

    def serve(self, remember=True, **kw):
        kw.setdefault("backend", FAKE)
        if remember:                     # the game folder found by an earlier import
            gamefiles.remember_game_dir(self.tmp/"settings.json", self.game)
        self.queue = importer.ImportQueue(inbox=self.tmp/"inbox", out_dir=self.out, settings=self.tmp/"settings.json",
                                          work_root=self.tmp/"work", cache=self.tmp/"cache", **kw)
        self.addCleanup(self.queue.close)
        self.server = SandboxServer(("127.0.0.1", 0), light_session(self.queue))
        thread = threading.Thread(target=self.server.serve_forever, kwargs=dict(poll_interval=.05), daemon=True)
        thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.port = self.server.server_port

    def request(self, method, path, body=None, headers=None, token=True, host=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=20)
        h = {"Host": host or f"127.0.0.1:{self.port}"}
        if token:
            h["X-Sandbox-Token"] = self.server.token
        h.update(headers or {})
        conn.request(method, path, body=body, headers=h)
        r = conn.getresponse()
        data = json.loads(r.read() or b"{}")
        conn.close()
        return r.status, data

    def upload(self, data, name="#2026.10.20 10.00.00.wrpl", force=False, **kw):
        from urllib.parse import quote
        return self.request("POST", f"/api/wrpl?name={quote(name)}{'&force=1' if force else ''}", data,
                            {"Content-Type": "application/octet-stream", **kw.pop("headers", {})}, **kw)

    def command(self, **command):
        status, data = self.request("POST", "/api/command", json.dumps(command).encode(),
                                    {"Content-Type": "application/json"})
        self.assertEqual(status, 202)
        return self.request("GET", f"/api/command?id={data['id']}")[1]

    def wait(self, jid, phases=importer.FINAL, timeout=30):
        end = time.time()+timeout
        while time.time() < end:
            items = {i["id"]: i for i in self.request("GET", "/api/state")[1]["wrpl"]["items"]}
            if jid in items and items[jid]["phase"] in phases:
                return items[jid]
            time.sleep(.05)
        self.fail(f"import {jid} did not reach {phases}: {items.get(jid)}")

    def replay_bytes(self):
        return write_replay(self.tmp/"src.wrpl").read_bytes()

    def test_upload_import_duplicate_and_reimport(self):
        self.serve()
        data = self.replay_bytes()
        status, reply = self.upload(data)
        self.assertEqual(status, 202, reply)
        item = self.wait(reply["id"])
        self.assertEqual((item["phase"], item["file"], item["duplicate"]),
                         ("done", "wt_real/2026.10.20_10.00.00.jsonl", False))
        self.assertTrue((self.out/"2026.10.20_10.00.00.jsonl").is_file())
        self.assertEqual(self.request("GET", f"/api/command?id={reply['id']}")[1]["phase"], "done")
        self.assertEqual(list((self.tmp/"inbox").iterdir()), [])
        # the same file again: located, not imported
        status, reply = self.upload(data)
        item = self.wait(reply["id"])
        self.assertEqual((item["phase"], item["duplicate"], self.backend_calls()), ("done", True, 1))
        # 重新导入
        status, reply = self.upload(data, force=True)
        self.assertEqual((self.wait(reply["id"])["duplicate"], self.backend_calls()), (False, 2))
        self.assertEqual(sorted(p.name for p in self.out.iterdir()), ["2026.10.20_10.00.00.jsonl"])

    def test_queue_runs_one_at_a_time_in_order(self):
        self.serve()
        ids = [self.upload(self.replay_bytes(), name=f"q{i}.wrpl")[1]["id"] for i in range(3)]
        items = [self.wait(j) for j in ids]
        self.assertEqual([i["phase"] for i in items], ["done"]*3)
        self.assertEqual([Path(line).name for line in self.calls.read_text().splitlines()],
                         ["q0.wrpl", "q1.wrpl", "q2.wrpl"])

    def test_refusals(self):
        self.serve(max_bytes=4096)
        data = self.replay_bytes()
        self.assertEqual(self.upload(data, token=False)[0], 403)
        self.assertEqual(self.upload(data, headers={"X-Sandbox-Token": "wrong"}, token=False)[0], 403)
        self.assertEqual(self.upload(data, headers={"Origin": "http://evil.example"})[0], 403)
        self.assertEqual(self.upload(data, host="localhost:1")[0], 403)
        self.assertEqual(self.upload(data, headers={"Content-Type": "application/json"})[0], 400)
        self.assertEqual(self.upload(data, name="notes.txt")[0], 400)
        status, reply = self.upload(data+bytes(8192))
        self.assertEqual(status, 413)
        self.assertIn("上限", reply["error"])
        self.assertEqual(self.backend_calls(), 0)
        self.assertEqual([p for p in (self.tmp/"inbox").glob("*") if p.is_file()], [])

    def test_backend_failure_and_timeout(self):
        self.serve(timeout_s=1)
        with env(FAKE_WRPL_MODE="fail"):
            item = self.wait(self.upload(self.replay_bytes())[1]["id"])
        self.assertEqual(item["phase"], "error")
        self.assertIn("解析失败", item["message"])
        self.assertIn("0.003 [E] fake parser failure", item["log"])
        with env(FAKE_WRPL_MODE="sleep"):
            item = self.wait(self.upload(self.replay_bytes())[1]["id"])
        self.assertEqual((item["phase"], item["message"]), ("error", "解析超时（1 秒）"))
        self.assertFalse(self.out.exists() and any(self.out.glob("*.jsonl")))

    def test_missing_game_files_then_a_typed_folder(self):
        self.serve(remember=False)
        with mock.patch.object(gamefiles, "INSTALL_CANDIDATES", {k: [] for k in gamefiles.INSTALL_CANDIDATES}), \
                mock.patch.object(gamefiles, "STEAM_ROOTS", {k: [] for k in gamefiles.STEAM_ROOTS}), \
                mock.patch.object(gamefiles, "_windows_steam_root", lambda: []):
            jid = self.upload(self.replay_bytes())[1]["id"]
            item = self.wait(jid, ("needs_game",))
            self.assertIn("找不到游戏文件", item["message"])
            self.assertIn("缺少", self.command(action="wrpl_game", path=str(self.tmp/"nowhere"))["error"])
            receipt = self.command(action="wrpl_game", path=str(self.game))
            self.assertEqual(receipt["result"]["version"], "2.59.0.54")
            self.assertEqual(self.wait(jid)["phase"], "done")
        self.assertEqual(gamefiles.load_settings(self.tmp/"settings.json")["game_dir"], str(self.game.resolve()))

    @unittest.skipIf(os.name == "nt", "the test bundle is a POSIX script")
    def test_backend_download_waits_for_the_button(self):
        zip_path = make_bundle(self.tmp/"bundle")
        manifest = manifest_for(zip_path, backends.platform_keys()[0])
        self.serve(backend=None, manifest=manifest)
        jid = self.upload(self.replay_bytes())[1]["id"]
        item = self.wait(jid, ("needs_backend",))
        self.assertIn("需要下载解析后端", item["message"])
        self.assertEqual(self.request("GET", "/api/state")[1]["wrpl"]["backend"]["size_mb"], 1)
        time.sleep(.3)
        self.assertEqual(self.wait(jid, ("needs_backend",))["phase"], "needs_backend")   # nothing without the click
        self.assertIn("result", self.command(action="wrpl_download"))
        self.assertEqual(self.wait(jid)["phase"], "done")

    def test_cancel_a_waiting_import(self):
        self.serve()
        with env(FAKE_WRPL_MODE="sleep"):
            first = self.upload(self.replay_bytes())[1]["id"]
            end = time.time()+20
            while self.backend_calls() < 1 and time.time() < end:   # the fake backend is running
                time.sleep(.02)
            second = self.upload(self.replay_bytes())[1]["id"]
            self.assertEqual(self.wait(second, ("queued",))["phase"], "queued")
            self.assertIn("result", self.command(action="wrpl_cancel", id=second))
            self.assertIn("result", self.command(action="wrpl_cancel", id=first))
            self.assertEqual(self.wait(first)["phase"], "cancelled")
        self.assertEqual(self.wait(second)["phase"], "cancelled")
        self.assertEqual(self.backend_calls(), 1)


if __name__ == "__main__":
    unittest.main()
