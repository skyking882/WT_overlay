"""Find the War Thunder game files a .wrpl needs (docs/wrpl_import_spec.md §4) and read the versions of both.

The backend mounts aces.vromfs.bin, game.vromfs.bin and char.vromfs.bin from one folder (read only) and, when present,
content/pkg_local. Search order: the explicit folder (--game / the sandbox field), the remembered one
(outputs/wrpl_settings.json), the replay folder's parent (the game writes replays to <game>/Replays/), the replay's own
folder, then the install locations in INSTALL_CANDIDATES and the Steam libraries listed in libraryfolders.vdf.

Versions: the replay header (1234 bytes, ReplayStructs.h) ends with gameVersion, a u64 of four u16 parts
(0x0002_003b_0000_0036 = 2.59.0.54); a VRFx vromfs header carries the same version as a u32 of four bytes at offset 20
(0x023b0036). Both read 2.59.0.54 for the 2026-10-08 replays and their game files.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import struct
import sys

REQUIRED = ("aces.vromfs.bin", "game.vromfs.bin", "char.vromfs.bin")
LOCAL_PACKAGE = Path("content")/"pkg_local"
REPLAY_HEADER_MAGIC = 0x1000ACE5
REPLAY_HEADER_SIZE = 1234
REPLAY_VERSION_OFFSET = 1226
REPLAY_FORMAT_OFFSET = 4

# Install locations searched after the explicit, remembered and replay-relative folders. NONE OF THESE IS VERIFIED on
# a game PC yet (docs/wrpl_import_spec.md §4, D): they are the usual Steam and Gaijin launcher defaults as far as
# known. Environment variables are expanded; missing ones drop the entry. Keep every guess in this one table.
INSTALL_CANDIDATES = {
    "win32": [
        r"%LOCALAPPDATA%\WarThunder",                 # Gaijin launcher default (guess)
        r"%ProgramFiles%\War Thunder",
        r"%ProgramFiles(x86)%\War Thunder",
        r"C:\Games\War Thunder",
        r"D:\Games\War Thunder",
        r"C:\Gaijin\WarThunder",
    ],
    "darwin": [
        "~/Library/Application Support/WarThunder",  # guess
        "/Applications/WarThunder.app/Contents/Resources/game",  # guess
    ],
    "linux": [
        "~/WarThunder",                                # Gaijin launcher (guess)
        "~/.local/share/WarThunder",                   # guess
    ],
}
# Steam roots whose steamapps/libraryfolders.vdf lists the libraries (also guesses outside Windows).
STEAM_ROOTS = {
    "win32": [r"%ProgramFiles(x86)%\Steam", r"%ProgramFiles%\Steam"],
    "darwin": ["~/Library/Application Support/Steam"],
    "linux": ["~/.steam/steam", "~/.local/share/Steam", "~/.var/app/com.valvesoftware.Steam/.local/share/Steam"],
}
STEAM_GAME = Path("steamapps")/"common"/"War Thunder"


class GameFilesError(ValueError):
    """No usable game folder; ``tried`` lists the folders looked at, ``versions`` the mismatching ones."""
    def __init__(self, message, tried=(), versions=None):
        super().__init__(message)
        self.tried = list(tried)
        self.versions = versions or {}


def is_game_dir(path):
    try:
        return all((Path(path)/name).is_file() for name in REQUIRED)
    except (OSError, TypeError, ValueError):
        return False


def missing_files(path):
    return [name for name in REQUIRED if not (Path(path)/name).is_file()]


def local_package(game_dir):
    p = Path(game_dir)/LOCAL_PACKAGE
    return p if p.is_dir() else None


def _version_text(parts):
    return ".".join(str(int(x)) for x in parts)


def replay_info(path):
    """{"version": "2.59.0.54" or None, "format": 0x18c1c or None} from the replay header; None fields if unreadable."""
    try:
        with open(path, "rb") as handle:
            head = handle.read(REPLAY_HEADER_SIZE)
    except OSError:
        return dict(version=None, format=None)
    if len(head) < REPLAY_HEADER_SIZE or struct.unpack_from("<I", head, 0)[0] != REPLAY_HEADER_MAGIC:
        return dict(version=None, format=None)
    raw = struct.unpack_from("<Q", head, REPLAY_VERSION_OFFSET)[0]
    version = _version_text((raw >> 48 & 0xFFFF, raw >> 32 & 0xFFFF, raw >> 16 & 0xFFFF, raw & 0xFFFF)) if raw else None
    return dict(version=version, format=struct.unpack_from("<I", head, REPLAY_FORMAT_OFFSET)[0])


def is_replay(path):
    """True when the file starts with the .wrpl header magic."""
    try:
        with open(path, "rb") as handle:
            head = handle.read(4)
    except OSError:
        return False
    return len(head) == 4 and struct.unpack("<I", head)[0] == REPLAY_HEADER_MAGIC


def vromfs_version(path):
    """Version of a VRFx vromfs file ("2.59.0.54"), or None (plain VRFs header, unreadable)."""
    try:
        with open(path, "rb") as handle:
            head = handle.read(24)
    except OSError:
        return None
    if len(head) < 24 or head[:4] != b"VRFx":
        return None
    raw = struct.unpack_from("<I", head, 20)[0]
    return _version_text((raw >> 24 & 0xFF, raw >> 16 & 0xFF, raw >> 8 & 0xFF, raw & 0xFF)) if raw else None


def game_version(game_dir):
    """Version of the game folder: aces.vromfs.bin's (the three files of one install carry the same one)."""
    return vromfs_version(Path(game_dir)/"aces.vromfs.bin")


def parse_vdf_paths(text):
    """Library paths ("path" values) from a Steam libraryfolders.vdf, old or new format."""
    paths = []
    for key, value in re.findall(r'"([^"\\]*(?:\\.[^"\\]*)*)"\s+"([^"\\]*(?:\\.[^"\\]*)*)"', text):
        if key == "path" or key.isdigit() and ("/" in value or "\\" in value):
            paths.append(value.replace("\\\\", "\\"))
    return paths


def _expand(entry, env):
    def var(m):
        return env.get(m.group(1), m.group(0))
    text = re.sub(r"%([^%]+)%", var, entry)
    if "%" in text:
        return None
    if text.startswith("~"):
        home = env.get("HOME") or env.get("USERPROFILE") or str(Path.home())
        text = home+text[1:]
    return Path(text)


def _windows_steam_root():
    try:
        import winreg
    except ImportError:
        return []
    found = []
    for hive, key, name in ((winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam", "SteamPath"),
                            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Valve\Steam", "InstallPath")):
        try:
            with winreg.OpenKey(hive, key) as handle:
                found.append(Path(winreg.QueryValueEx(handle, name)[0]))
        except OSError:
            continue
    return found


def steam_libraries(roots):
    """Library folders of the given Steam roots (each root itself plus the ones in libraryfolders.vdf)."""
    libs = []
    for root in roots:
        root = Path(root)
        libs.append(root)
        for vdf in (root/"steamapps"/"libraryfolders.vdf", root/"config"/"libraryfolders.vdf"):
            try:
                text = vdf.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            libs.extend(Path(p) for p in parse_vdf_paths(text))
    out = []
    for lib in libs:
        if lib not in out:
            out.append(lib)
    return out


def candidates(replay=None, *, explicit=None, remembered=None, platform=None, env=None, steam_roots=None):
    """[(folder, reason)] in search order; duplicates removed. ``steam_roots`` overrides the Steam root discovery
    (tests)."""
    platform = platform or sys.platform
    key = "win32" if platform.startswith("win") else "darwin" if platform == "darwin" else "linux"
    env = os.environ if env is None else env
    out = []

    def add(path, reason):
        if path is None:
            return
        path = Path(path)
        if all(path != p for p, _ in out):
            out.append((path, reason))

    if explicit:
        add(explicit, "explicit")
        return out
    if remembered:
        add(remembered, "remembered")
    if replay is not None:
        folder = Path(replay).resolve().parent
        add(folder.parent, "replay_parent")
        add(folder, "replay_folder")
    for entry in INSTALL_CANDIDATES[key]:
        add(_expand(entry, env), "install")
    roots = steam_roots
    if roots is None:
        roots = [p for p in (_expand(e, env) for e in STEAM_ROOTS[key]) if p is not None]
        if key == "win32":
            roots = _windows_steam_root()+roots
    for lib in steam_libraries(roots):
        add(lib/STEAM_GAME, "steam")
    return out


def version_match(replay_version, game_version):
    """"exact", "build" (only the last part differs), "other", or None when either is unknown.

    D: a build-number difference is accepted. The reference exports of the 2026-10-04..07 replays (2.59.0.44-.50)
    were made with the 2.59.0.54 game files, and the backend reproduces them field by field."""
    if not replay_version or not game_version:
        return None
    if replay_version == game_version:
        return "exact"
    return "build" if replay_version.split(".")[:3] == game_version.split(".")[:3] else "other"


def find_game_dir(replay=None, *, explicit=None, remembered=None, check_version=True, **kwargs):
    """(folder, info) for the replay; info = {reason, replay_version, game_version, version_note, tried}.

    An explicit folder is the only one tried. Otherwise, among the folders with the three files, the first whose
    version equals the replay's wins, else the first that differs only in the build number (or whose version cannot
    be read). When all have another version, GameFilesError names both versions (unless check_version is False)."""
    rinfo = replay_info(replay) if replay is not None else dict(version=None, format=None)
    tried, valid = [], []
    for folder, reason in candidates(replay, explicit=explicit, remembered=remembered, **kwargs):
        tried.append(str(folder))
        if not is_game_dir(folder):
            if reason == "explicit":
                raise GameFilesError(f"{folder} 里缺少 {'、'.join(missing_files(folder))}", tried)
            continue
        gv = game_version(folder)
        valid.append((folder, reason, gv, version_match(rinfo["version"], gv)))
        if valid[-1][3] == "exact":
            break
    if not valid:
        raise GameFilesError("找不到游戏文件（aces / game / char.vromfs.bin），请指定游戏目录", tried)
    ranked = sorted(valid, key=lambda v: {"exact": 0, None: 1, "build": 1, "other": 2}[v[3]])
    folder, reason, gv, match = ranked[0]
    if match == "other" and check_version:
        raise GameFilesError(version_message(rinfo["version"], gv), tried, {str(v[0]): v[2] for v in valid})
    note = None if match in ("exact", None) else version_message(rinfo["version"], gv)
    return folder, dict(reason=reason, replay_version=rinfo["version"], game_version=gv, version_note=note,
                        tried=tried, replay_format=rinfo["format"])


def version_message(replay_version, game_version):
    return f"回放版本 {replay_version}，游戏文件版本 {game_version}"


# -- remembered folder (outputs/wrpl_settings.json, git-ignored) ----------------------------------------------------

def load_settings(path):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def remember_game_dir(path, game_dir):
    path = Path(path)
    data = load_settings(path)
    data["game_dir"] = str(Path(game_dir).resolve())
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1)+"\n", encoding="utf-8")
    os.replace(tmp, path)
