"""python -m wt_overlay.wrpl 回放.wrpl [更多.wrpl 或文件夹 ...] [--game DIR] [--out-dir DIR] [--name NAME]
                           [--backend PATH] [--download-backend] [--keep-export] [--force]

把 War Thunder 回放转成沙盘可播放的 outputs/engagements/wt_real/<名字>.jsonl（docs/wrpl_import_spec.md §6）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import backend as backends
from . import gamefiles
from . import importer


def expand(inputs):
    """Files as given; a folder contributes its .wrpl files (not recursive)."""
    out = []
    for raw in inputs:
        p = Path(raw)
        if p.is_dir():
            out.extend(sorted(q for q in p.iterdir() if q.is_file() and q.suffix.lower() == ".wrpl"))
        else:
            out.append(p)
    return out


def confirm_download(asset, args):
    size = f"约 {asset.size_mb} MB，" if asset.size_mb else ""
    if args.download_backend:
        return True
    if not sys.stdin.isatty():
        print(f"需要下载解析后端（{size}来自本项目 GitHub Release）：加 --download-backend 允许下载，"
              f"或用 --backend / {backends.ENV} 指定本地后端", file=sys.stderr)
        return False
    answer = input(f"需要下载解析后端（{size}来自本项目 GitHub Release）。下载？[y/N] ").strip().lower()
    return answer in ("y", "yes")


def run_one(replay, args, backend):
    return importer.import_replay(replay, game=args.game, out_dir=args.out_dir, name=args.name, backend=backend,
                                  keep_export=args.keep_export, force=args.force, team=args.team,
                                  all_units=args.all_units, ignore_version=args.ignore_version,
                                  timeout_s=int(args.timeout*60),
                                  progress=lambda phase: print(f"{replay.name}: {'解析' if phase == 'parse' else '转换'}"
                                                               "...", flush=True))


def report(replay, exc):
    hint = "（用 --game 指定游戏目录）" if isinstance(exc, gamefiles.GameFilesError) else ""
    print(f"{replay.name}: {exc}{hint}", file=sys.stderr)
    for line in getattr(exc, "log", ()):
        print(f"    {line}", file=sys.stderr)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m wt_overlay.wrpl", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help=".wrpl 文件或文件夹（只看这一层）")
    ap.add_argument("--game", help="含 aces / game / char.vromfs.bin 的游戏目录（与回放同版本）")
    ap.add_argument("--out-dir", default=str(importer.DEFAULT_OUT), help="输出目录，缺省 outputs/engagements/wt_real")
    ap.add_argument("--name", help="输出文件名（只限一个回放）")
    ap.add_argument("--backend", help=f"本地解析后端（wrpl_export 或其目录），也可用环境变量 {backends.ENV}")
    ap.add_argument("--download-backend", action="store_true", help="需要时直接下载解析后端，不再询问")
    ap.add_argument("--keep-export", action="store_true", help="保留中间的 match_export.json（outputs/wt_replay_exports）")
    ap.add_argument("--force", action="store_true", help="已导入过的也重新导入")
    ap.add_argument("--ignore-version", action="store_true", help="回放和游戏文件版本不一致时仍然解析")
    ap.add_argument("--timeout", type=float, default=importer.BACKEND_TIMEOUT_S/60, help="解析超时 / 分钟，缺省 20")
    ap.add_argument("--team", action="append", default=[], metavar="NAME=0|1", help="转给转换脚本：指定队伍")
    ap.add_argument("--all-units", action="store_true", help="转给转换脚本：保留地面单位和不动的单位")
    args = ap.parse_args(argv)
    replays = expand(args.inputs)
    if not replays:
        ap.error("没有找到 .wrpl 文件")
    if args.name and len(replays) > 1:
        ap.error("--name 只能用于一个回放")
    backend = args.backend             # a path or None until found; downloaded at most once, after confirmation
    failed = 0
    for replay in replays:
        try:
            result = run_one(replay, args, backend)
        except backends.BackendMissing as exc:
            if exc.asset is None or not confirm_download(exc.asset, args):
                if exc.asset is None:
                    print(str(exc), file=sys.stderr)
                return 1
            print(f"下载 {exc.asset.file} ...", flush=True)
            try:
                backend = backends.download(exc.asset)
                result = run_one(replay, args, backend)
            except (importer.ImportFailed, backends.BackendError, gamefiles.GameFilesError) as error:
                failed += 1
                report(replay, error)
                continue
        except (importer.ImportFailed, backends.BackendError, gamefiles.GameFilesError) as exc:
            failed += 1
            report(replay, exc)
            continue
        if result["status"] == "duplicate":
            print(f"{replay.name}: 已导入过 -> {result['path']}（--force 重新导入）")
            continue
        if result.get("version_note"):
            print(f"{replay.name}: 注意：{result['version_note']}")
        print(f"{replay.name} -> {result['path']}")
        if result.get("export"):
            print(f"    match_export.json: {result['export']}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
