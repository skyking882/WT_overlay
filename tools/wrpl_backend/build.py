#!/usr/bin/env python3
"""Freeze the wrpl_export backend with PyInstaller (one-folder) and zip it. Used locally and by
.github/workflows/wrpl_backend.yml on Windows, Linux and macOS.

    python tools/wrpl_backend/build.py --module-dir BUILD/python/main --platform macos-x64 \
        --fork-src third_party/WrplReplayParser [--dist DIR] [--work DIR] [--zip]

Run it with the Python the PyReplayParser module was compiled against (same version and architecture), with
PyInstaller installed. --module-dir holds PyReplayParser*.so / *.pyd and, except on Windows, the signal_tracer
executable (the parser looks for it next to the module). Output: DIST/wrpl_export/ (the frozen folder), with --zip also
DIST/wrpl_export-<platform>.zip and DIST/wrpl_export-<platform>.zip.sha256; THIRD_PARTY_NOTICES.md is copied into the
folder. The build ends with a --version smoke test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import stat
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
EXPORT_PY = ROOT/"wt_overlay"/"wrpl"/"export.py"
MANIFEST = ROOT/"wt_overlay"/"wrpl"/"backend.json"
PLATFORMS = ("windows-x64", "macos-arm64", "macos-x64", "linux-x64")
NAME = "wrpl_export"


def source_sha256(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_head(src):
    return subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], check=True, capture_output=True,
                          text=True).stdout.strip()


def write_zip(folder, target):
    """Zip FOLDER as its own top-level directory; symlinks stay symlinks, file modes are kept."""
    base = folder.parent
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for path in sorted(folder.rglob("*")):
            rel = path.relative_to(base).as_posix()
            if path.is_symlink():
                info = zipfile.ZipInfo(rel)
                info.create_system = 3
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
                z.writestr(info, os.readlink(path))
            elif path.is_dir():
                info = zipfile.ZipInfo(rel+"/")
                info.create_system = 3
                info.external_attr = (stat.S_IFDIR | 0o755) << 16 | 0x10
                z.writestr(info, b"")
            else:
                info = zipfile.ZipInfo.from_file(path, rel)
                info.create_system = 3
                info.external_attr = (stat.S_IFREG | (path.stat().st_mode & 0o777)) << 16
                info.compress_type = zipfile.ZIP_DEFLATED
                with open(path, "rb") as src, z.open(info, "w") as dst:
                    shutil.copyfileobj(src, dst, 1 << 20)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--module-dir", type=Path, required=True, help="folder with the built PyReplayParser module")
    ap.add_argument("--platform", choices=PLATFORMS, required=True)
    ap.add_argument("--fork-src", type=Path, help="WrplReplayParser checkout the module was built from (for its commit)")
    ap.add_argument("--fork-commit", help="fork commit, instead of --fork-src")
    ap.add_argument("--dist", type=Path, default=ROOT/"outputs"/"wrpl_backend"/"dist")
    ap.add_argument("--work", type=Path, default=ROOT/"outputs"/"wrpl_backend"/"work")
    ap.add_argument("--zip", action="store_true", help="also write wrpl_export-<platform>.zip and its .sha256")
    args = ap.parse_args(argv)

    modules = sorted(p for p in args.module_dir.glob("PyReplayParser*") if p.suffix in (".so", ".pyd"))
    if len(modules) != 1:
        ap.error(f"expected one PyReplayParser module in {args.module_dir}, found {[p.name for p in modules]}")
    tracer = next((p for p in (args.module_dir/"signal_tracer", args.module_dir/"signal_tracer.exe") if p.is_file()),
                  None)
    if tracer is None and not args.platform.startswith("windows"):
        ap.error(f"signal_tracer not found in {args.module_dir} (copy it next to the module)")
    commit = args.fork_commit or (git_head(args.fork_src) if args.fork_src else None)
    if not commit:
        ap.error("--fork-src or --fork-commit is needed")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    work = args.work.resolve()
    dist = args.dist.resolve()
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    dist.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(dist/NAME, ignore_errors=True)
    info = {"backend_version": manifest["version"], "upstream_revision": manifest["upstream_revision"],
            "fork": manifest["fork"], "fork_branch": manifest["fork_branch"], "fork_commit": commit,
            "platform": args.platform, "export_py_sha256": source_sha256(EXPORT_PY),
            "python": platform.python_version(), "module": modules[0].name}
    (work/"wrpl_build_info.json").write_text(json.dumps(info, indent=1)+"\n", encoding="utf-8")

    sep = os.pathsep
    cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--onedir", "--console", "--name", NAME,
           "--distpath", str(dist), "--workpath", str(work/"pyinstaller"), "--specpath", str(work),
           "--paths", str(args.module_dir.resolve()), "--hidden-import", "PyReplayParser",
           # imported from C++ (python/ecs: py::module_::import("inspect")), invisible to PyInstaller's analysis
           "--hidden-import", "inspect",
           "--add-data", f"{work/'wrpl_build_info.json'}{sep}.",
           "--exclude-module", "tkinter", "--exclude-module", "unittest", "--exclude-module", "pydoc",
           "--exclude-module", "sqlite3", "--exclude-module", "lib2to3"]
    if tracer is not None:
        cmd += ["--add-binary", f"{tracer.resolve()}{sep}."]
    cmd.append(str(EXPORT_PY))
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

    exe = dist/NAME/(NAME+(".exe" if args.platform.startswith("windows") else ""))
    out = subprocess.run([str(exe), "--version"], check=True, capture_output=True, text=True, timeout=300)
    version = json.loads(out.stdout.strip().splitlines()[-1])
    if version.get("fork_commit") != commit or version.get("backend_version") != manifest["version"]:
        raise SystemExit(f"--version smoke test: unexpected {version}")
    print("smoke test:", json.dumps(version), flush=True)
    shutil.copy(ROOT/"THIRD_PARTY_NOTICES.md", dist/NAME/"THIRD_PARTY_NOTICES.md")   # licenses travel with the zip
    size = sum(p.stat().st_size for p in (dist/NAME).rglob("*") if p.is_file() and not p.is_symlink())
    print(f"bundle: {dist/NAME} ({size/1e6:.1f} MB unpacked)")
    if args.zip:
        target = dist/f"{NAME}-{args.platform}.zip"
        write_zip(dist/NAME, target)
        digest = file_sha256(target)
        (dist/(target.name+".sha256")).write_text(f"{digest}  {target.name}\n", encoding="utf-8")
        print(f"zip: {target} ({target.stat().st_size/1e6:.1f} MB) sha256 {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
