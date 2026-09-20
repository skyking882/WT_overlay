"""Run with `python -m wt_overlay`; --demo never connects to the game."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import sys
import time

from .app import OverlayController
from .contracts import ClimbRequest


def _positive(text: str) -> float:
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("请输入数值") from exc
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("数值必须为有限正数")
    return value


def snapshot_json(snapshot) -> dict:
    state, energy, advice = snapshot.state, snapshot.energy, snapshot.advice
    current = advice.current if advice else None
    best = advice.best if advice and advice.available else None
    return {
        "mode": snapshot.mode, "status": snapshot.status,
        "valid": bool(state and state.valid),
        "aircraft": state.aircraft_id if state else None,
        "tas_mps": state.tas_mps if state and state.valid else None,
        "altitude_m": state.altitude_m if state and state.valid else None,
        "energy_height_m": energy.energy_height_m if energy else None,
        "energy_ready": bool(energy and energy.ready),
        "sep_mps": energy.sep_mps if energy else None,
        "climb_mps": energy.climb_mps if energy else None,
        "kinetic_sep_mps": energy.kinetic_sep_mps if energy else None,
        "reference_model": snapshot.model_name,
        "reference_sep_mps": current.sep_mps if current and current.valid else None,
        "reference_best_tas_mps": best.condition.tas_mps if best else None,
        "reference_best_sep_mps": best.sep_mps if best else None,
        "reference_mass_kg": current.condition.mass_kg if current else snapshot.mass_override_kg,
        "reference_afterburner": snapshot.afterburner,
        "reference_scope": "same-altitude 1g clean; untrimmed and not game-validated",
        "reference_reason": advice.reason if advice else "FM 未加载",
        "climb_enabled": snapshot.climb_enabled,
        "climb_target": asdict(snapshot.climb_request),
        "climb_guidance": asdict(snapshot.climb) if snapshot.climb else None,
        "notes": list(dict.fromkeys((*snapshot.notes, *(advice.notes if advice else ())))),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="WT 8111 实际能量与静态 SEP 参考面板")
    parser.add_argument("--demo", action="store_true", help="显式合成演示；不读取游戏")
    parser.add_argument("--url", default="http://127.0.0.1:8111", help="本机 8111 HTTP 地址")
    parser.add_argument("--model", help="FM 路径；当前仅支持附带的固定版本苏-27SM")
    parser.add_argument("--mass-kg", type=_positive, help="静态模型参考总质量（kg）")
    parser.add_argument("--military", action="store_true", help="静态模型用全军推；默认全加力")
    parser.add_argument("--headless", action="store_true", help="输出 JSON 行，不创建窗口")
    parser.add_argument("--duration", type=_positive, default=5.0, help="无窗口运行秒数，默认 5")
    parser.add_argument("--climb-altitude", type=_positive, help="开启爬升引导，目标高度（m）")
    parser.add_argument("--arrival-speed-kmh", type=_positive, help="到达最低 TAS（km/h）；留空自动选择")
    args = parser.parse_args(argv)
    if sys.version_info < (3, 11):
        parser.error("需要 Python 3.11 或更新版本")
    if args.arrival_speed_kmh is not None and args.climb_altitude is None:
        parser.error("到达速度需要与 --climb-altitude 一起使用")
    try:
        controller = OverlayController(mode="demo" if args.demo else "live", base_url=args.url,
                                       model_path=args.model, mass_kg=args.mass_kg,
                                       afterburner=not args.military)
        if args.climb_altitude is not None:
            controller.submit({"action": "climb_target", "altitude_m": args.climb_altitude,
                               "minimum_tas_mps": args.arrival_speed_kmh/3.6 if args.arrival_speed_kmh else None})
            controller.submit({"action": "climb_enabled", "enabled": True})
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    try:
        if args.headless:
            controller.start()
            deadline = time.monotonic()+args.duration
            while time.monotonic() < deadline:
                print(json.dumps(snapshot_json(controller.get_snapshot()), ensure_ascii=False,
                                 allow_nan=False), flush=True)
                time.sleep(min(0.5, max(0, deadline-time.monotonic())))
            print(json.dumps(snapshot_json(controller.get_snapshot()), ensure_ascii=False,
                             allow_nan=False), flush=True)
        else:
            from .ui import OverlayApp
            request = ClimbRequest(args.climb_altitude, args.arrival_speed_kmh/3.6 if args.arrival_speed_kmh else None) if args.climb_altitude else None
            window = OverlayApp(controller.get_snapshot, controller.submit, climb_request=request)
            controller.start()
            window.run()
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"无法运行：{exc}\n可用 --demo --headless 检查非图形部分。", file=sys.stderr)
        return 2
    finally:
        controller.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
