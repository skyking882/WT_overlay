"""Loopback-only page for the opening shot (开局抽奖), computed on this computer: python -m wt_overlay.planner_web.

Same layout as missile_sim's local GUI (src/missile_gui/server.py): a stdlib HTTP server on 127.0.0.1, static files
from ./static and a small JSON API. The plan is wt_overlay.opening.plan_opening, the same search the in-game aid
runs, over all cores but one; the chosen shot is flown once more (trace_shot) for the side view.

    GET  /api/health
    GET  /api/options     aircraft with their radar missiles, enemy defaults
    POST /api/plan        {aircraft, missile, altitude_m, mach, fuel_percent, missiles, enemy: {...}}
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import math
import mimetypes
import os
from pathlib import Path
import threading
import time
import traceback
import urllib.error
import urllib.request
import webbrowser
from urllib.parse import urlparse

from ..opening import OpeningSettings, PlanRequest, plan_opening, trace_shot, validate_settings, worker_init

LOGGER = logging.getLogger("wt_planner")
STATIC_DIR = Path(__file__).resolve().parent/"static"
ROOT = Path(__file__).resolve().parents[2]
MAX_REQUEST_BYTES = 100_000
EXCLUDED_MISSILES = {"cn_pl15"}   # user decision 2026-10-05: PL-15 is not modelled (dual pulse) and not offered
MISSILE_NAMES = {"cn_pl12": "PL-12", "cn_pl12a": "PL-12A", "su_r_77": "R-77", "su_r_77_1": "R-77-1",
                 "cn_sd10a": "SD-10A", "il_derby": "Derby", "jp_aam4": "AAM-4", "r_darter": "R-Darter",
                 "swd_rb99": "Rb 99", "su_rvv_ae": "RVV-AE", "su_rvv_sd": "RVV-SD", "us_aim_120a": "AIM-120A",
                 "us_aim_120b": "AIM-120B", "us_aim_120c_5": "AIM-120C-5", "us_aim_120c_7": "AIM-120C-7",
                 "us_aim_120d": "AIM-120D", "us_aim_54a": "AIM-54A", "us_aim_54b": "AIM-54B", "us_aim_54c": "AIM-54C",
                 "us_aim_54c_plus": "AIM-54C+", "us_aim7f_sparrow": "AIM-7F", "us_aim7m_sparrow": "AIM-7M",
                 "us_aim7p_sparrow": "AIM-7P", "su_r_27er": "R-27ER", "su_r_27er1": "R-27ER1", "su_r_27r": "R-27R",
                 "fr_matra_super_530d": "Super 530D", "fr_matra_super_530f": "Super 530F",
                 "il_aspide_1a": "Aspide 1A", "ir_fakour_90": "Fakour-90", "cn_ld10": "LD-10", "cn_pf10": "PF-10",
                 "cn_pl11": "PL-11", "uk_skyflash_aim_7": "Skyflash"}


def missile_name(missile_id):
    return MISSILE_NAMES.get(missile_id, missile_id)


class Planner:
    """One computation at a time over a process pool kept between requests (models stay loaded in the workers)."""

    def __init__(self, workers=None):
        self.workers = workers or max(1, (os.cpu_count() or 2)-1)
        self.pool = None
        self.lock = threading.Lock()
        self._options = None

    def run(self, function, jobs):
        if self.pool is None:
            self.pool = ProcessPoolExecutor(max_workers=self.workers, initializer=worker_init)
        try:
            return list(self.pool.map(function, jobs))
        except BrokenProcessPool:
            self.pool = None
            raise

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=False, cancel_futures=True)

    def options(self):
        """Catalog aircraft (newest first), each with the radar missiles it carries that missile_sim can fly."""
        if self._options is not None:
            return self._options
        from ..fm.catalog import aircraft_catalog
        from ..opening import _library
        library = _library()
        radar, active = {}, {}
        for missile_id in library.profiles:
            if missile_id in EXCLUDED_MISSILES:
                continue
            try:
                info = library.info(missile_id)   # needs a radar seeker with a range
            except (KeyError, ValueError):
                continue
            radar[missile_id] = missile_name(missile_id)
            active[missile_id] = library.is_active(info.profile_id)
        equipment = json.loads((ROOT/"data"/"units"/"equipment.json").read_text(encoding="utf-8"))
        aircraft = []
        for a in sorted(aircraft_catalog(), key=lambda a: (-a.br, a.name)):
            carried = (equipment.get(a.id) or {}).get("missiles") or {}
            missiles = [dict(id=m, name=radar[m], count=n, active=active[m]) for m, n in carried.items() if m in radar]
            # Default first: active radar before semi-active, then the later variant (AIM-120C-5 before -120B).
            missiles.sort(key=lambda m: m["name"], reverse=True)
            missiles.sort(key=lambda m: not m["active"])
            aircraft.append(dict(id=a.id, name=a.name, name_en=a.name_en, br=a.br, missiles=missiles))
        defaults = OpeningSettings()
        self._options = dict(aircraft=aircraft, radar_missiles=[dict(id=k, name=v) for k, v in sorted(
            radar.items(), key=lambda kv: kv[1])], enemy=dict(
            distance_km=defaults.distance_m/1000., aircraft=defaults.enemy_aircraft,
            fuel_percent=round(defaults.enemy_fuel_fraction*100), climb_deg=defaults.enemy_climb_deg,
            level_m=defaults.enemy_level_m))
        return self._options

    def plan(self, payload):
        request, inputs = build_request(payload)
        started = time.perf_counter()
        plan = plan_opening(request, self.run)
        result = dict(inputs=inputs, plan=_plan_json(plan))
        if plan.available:
            trace = self.run(trace_shot, [(request, plan.climb_deg, plan.pull_s, plan.pitch_deg)])[0]
            result["trace"] = trace
            result["steps"] = steps(plan, inputs, trace)
        result["wall_s"] = time.perf_counter()-started
        return result


class InputError(ValueError):
    pass


def _number(payload, key, low, high, label):
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or \
            not low <= value <= high:
        raise InputError(f"{label}须在 {low:g}–{high:g} 之间")
    return float(value)


def build_request(payload):
    """The planner input from the page's form: the start (= spawn) state, own mass, the scenario."""
    from ..fm import atmosphere
    from ..fm.catalog import find_aircraft
    from ..flight import fuel_data
    from ..opening import _library, _missile_mass
    aircraft = find_aircraft(payload.get("aircraft") or "")
    if aircraft is None:
        raise InputError("请选择机型")
    missile = payload.get("missile")
    library = _library()
    if not isinstance(missile, str) or missile in EXCLUDED_MISSILES or missile not in library.profiles:
        raise InputError("请选择导弹")
    altitude = _number(payload, "altitude_m", 300., 12000., "起始高度")
    mach = _number(payload, "mach", .3, 2., "起始马赫数")
    fuel_fraction = _number(payload, "fuel_percent", 5., 100., "内油")/100.
    count = int(_number(payload, "missiles", 0., 12., "导弹数"))
    enemy = payload.get("enemy") or {}
    if not isinstance(enemy, dict):
        raise InputError("敌机设定无效")
    enemy_aircraft = find_aircraft(enemy.get("aircraft") or OpeningSettings.enemy_aircraft)
    if enemy_aircraft is None:
        raise InputError("请选择敌机机型")
    settings = OpeningSettings(
        True, missile, _number(enemy, "distance_km", 30., 200., "出生间距")*1000., enemy_aircraft.id,
        _number(enemy, "fuel_percent", 5., 100., "敌机内油")/100.,
        enemy_climb_deg=_number(enemy, "climb_deg", 3., 45., "敌机爬升角"),
        enemy_level_m=_number(enemy, "level_m", 3000., 14000., "敌机改平高度"))
    try:
        validate_settings(settings)
    except ValueError as exc:
        raise InputError(str(exc)) from exc
    fuel = fuel_data(aircraft.id)
    mass = fuel.empty_kg+fuel_fraction*fuel.internal_kg+count*_missile_mass(library, missile)
    tas = mach*atmosphere(altitude)[1]
    request = PlanRequest(aircraft.id, mass, missile, settings, altitude, tas, 0., 0., altitude, tas, 0.)
    inputs = dict(aircraft=aircraft.id, aircraft_name=aircraft.name, missile=missile, missile_name=missile_name(missile),
                  altitude_m=altitude, mach=mach, tas_mps=tas, fuel_kg=fuel_fraction*fuel.internal_kg,
                  missiles=count, mass_kg=mass, enemy_name=enemy_aircraft.name, distance_m=settings.distance_m,
                  enemy_climb_deg=settings.enemy_climb_deg, enemy_level_m=settings.enemy_level_m,
                  enemy_fuel_percent=settings.enemy_fuel_fraction*100.)
    return request, inputs


def _plan_json(plan):
    out = asdict(plan)
    out["profile"] = [list(row) for row in plan.profile]
    return out


def _mach(altitude, tas):
    from ..fm import atmosphere
    return tas/atmosphere(max(0., min(20000., altitude)))[1]


def steps(plan, inputs, trace):
    """The operation sequence shown on the page."""
    rows = [dict(t=0., title=f"全加力，爬升 {plan.climb_deg:.0f}°",
                 detail=f"保持出生航向。起点 {inputs['altitude_m']:.0f} m，M{inputs['mach']:.2f}"
                        f"（{inputs['tas_mps']*3.6:.0f} km/h）")]
    profile = {round(row[0]): row for row in plan.profile}
    if plan.pitch_deg is not None:
        row = profile.get(round(plan.pull_s))
        state = f"，此时约 {row[1]:.0f} m、M{_mach(row[1], row[2]):.2f}" if row else ""
        rows.append(dict(t=plan.pull_s, title=f"拉起到 {plan.pitch_deg:.0f}°", detail=f"从 {plan.climb_deg:.0f}° 拉到 "
                         f"{plan.pitch_deg:.0f}°{state}"))
    rows.append(dict(t=plan.launch_s, title="发射", detail=f"目标距离 {plan.launch_range_m/1000:.1f} km；本机 "
                     f"{plan.launch_altitude_m:.0f} m、{plan.launch_tas_mps:.0f} m/s"
                     f"（M{_mach(plan.launch_altitude_m, plan.launch_tas_mps):.2f}）"))
    end_x = trace["end"][1]
    rows.append(dict(t=plan.hit_s, title="命中", detail=f"导弹飞行 {plan.flight_s:.1f} s；目标高度 "
                     f"{plan.target_altitude_m:.0f} m；命中点离你的出生点 {end_x/1000:.1f} km"))
    return rows


class PlannerServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, address, planner):
        super().__init__(address, PlannerHandler)
        self.planner = planner


class PlannerHandler(BaseHTTPRequestHandler):
    server_version = "WTOpeningPlanner/1.0"

    def log_message(self, format, *args):
        LOGGER.info("%s - %s", self.client_address[0], format % args)

    def _json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status, code, message):
        self._json(status, {"ok": False, "error": {"code": code, "message": message}})

    def _static(self, name):
        path = STATIC_DIR/name
        if not path.is_file():
            self._error(HTTPStatus.NOT_FOUND, "not_found", "资源不存在。")
            return
        body = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type in ("application/javascript", "application/json"):
            content_type += "; charset=utf-8"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self'; script-src 'self'; "
                                                    "connect-src 'self'; img-src 'self' data:")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 - stdlib handler API
        path = urlparse(self.path).path
        try:
            if path == "/api/health":
                self._json(HTTPStatus.OK, {"ok": True})
            elif path == "/api/options":
                self._json(HTTPStatus.OK, {"ok": True, **self.server.planner.options()})
            elif path in ("/", "/index.html"):
                self._static("index.html")
            elif path.startswith("/static/"):
                name = path[len("/static/"):]
                if "/" in name or "\\" in name or name.startswith("."):
                    self._error(HTTPStatus.NOT_FOUND, "not_found", "资源不存在。")
                else:
                    self._static(name)
            else:
                self._error(HTTPStatus.NOT_FOUND, "not_found", "页面不存在。")
        except FileNotFoundError as exc:   # missile_sim missing: the message says what to do
            self._error(HTTPStatus.SERVICE_UNAVAILABLE, "missile_sim", str(exc))
        except (BrokenPipeError, ConnectionResetError):
            LOGGER.info("客户端在响应完成前断开连接。")
        except Exception:
            LOGGER.exception("处理 GET %s 时发生未预期错误", path)
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "internal_error", "本地服务发生错误，请查看终端日志。")

    def do_POST(self):  # noqa: N802 - stdlib handler API
        path = urlparse(self.path).path
        if path != "/api/plan":
            self._error(HTTPStatus.NOT_FOUND, "not_found", "接口不存在。")
            return
        planner = self.server.planner
        if not planner.lock.acquire(blocking=False):
            self._error(HTTPStatus.CONFLICT, "busy", "上一次计算还没完成，请稍候。")
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > MAX_REQUEST_BYTES:
                self._error(HTTPStatus.BAD_REQUEST, "invalid_request", "请求为空或过大。")
                return
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._error(HTTPStatus.BAD_REQUEST, "invalid_json", "请求不是合法 JSON。")
                return
            if not isinstance(payload, dict):
                self._error(HTTPStatus.BAD_REQUEST, "invalid_request", "请求顶层必须是对象。")
                return
            self._json(HTTPStatus.OK, {"ok": True, **planner.plan(payload)})
        except InputError as exc:
            self._error(HTTPStatus.UNPROCESSABLE_ENTITY, "invalid_input", str(exc))
        except FileNotFoundError as exc:
            self._error(HTTPStatus.SERVICE_UNAVAILABLE, "missile_sim", str(exc))
        except (BrokenPipeError, ConnectionResetError):
            LOGGER.info("客户端在计算完成前断开连接。")
        except Exception as exc:
            LOGGER.exception("计算失败")
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "internal_error", f"计算失败：{type(exc).__name__}: {exc}")
        finally:
            planner.lock.release()


def _open_browser_when_ready(url):
    health = url.rstrip("/")+"/api/health"
    for _ in range(60):
        try:
            with urllib.request.urlopen(health, timeout=.5) as response:
                if response.status == HTTPStatus.OK:
                    webbrowser.open(url, new=2)
                    return
        except (OSError, urllib.error.URLError):
            time.sleep(.1)


def main(argv=None):
    parser = argparse.ArgumentParser(description="WT 开局抽奖：本机计算的离线页面")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("WT_PLANNER_PORT", "8795")))
    parser.add_argument("--workers", type=int, help="计算进程数，默认 CPU 核数减 1")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    if args.host != "127.0.0.1":
        parser.error("只允许绑定 127.0.0.1")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    url = f"http://{args.host}:{args.port}/"
    planner = Planner(args.workers)
    try:
        server = PlannerServer((args.host, args.port), planner)
    except OSError:
        print(f"无法启动：端口 {args.port} 已被占用或不可用。可关闭占用该端口的程序，或设置 WT_PLANNER_PORT 后重试。",
              flush=True)
        traceback.print_exc()
        return 2
    LOGGER.info("开局抽奖页面已启动：%s（只监听 127.0.0.1；按 Ctrl+C 关闭）", url)
    if not args.no_browser:
        threading.Thread(target=_open_browser_when_ready, args=(url,), daemon=True).start()
    try:
        server.serve_forever(poll_interval=.25)
    except KeyboardInterrupt:
        LOGGER.info("收到关闭请求。")
    finally:
        server.server_close()
        planner.close()
    return 0
