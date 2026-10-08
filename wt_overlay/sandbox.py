"""Local interactive air-combat sandbox: ``python -m wt_overlay.sandbox``.

Only the worker thread owns the Engagement. HTTP readers receive cached analysis
snapshots; commands are serialized between public physics ticks. Autonomous
controllers use the same observation/intent/camera route as the standalone sim.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import queue
import secrets
import threading
import time
from urllib.parse import urlsplit, parse_qs
import webbrowser

from .archetypes import ARCHETYPES, SKILLS, bearing_of
from .engagement import Action, RadarCommand, ReplayWriter, default_library, equipment_data, STT_TRACK
from .flight import FlightCommand, KeyboardCommand, MAX_ALTITUDE_M, MIN_STEP_SPEED_MPS, SUBSTEP_S
from .fm import atmosphere
from .match import scenario as make_match, modelled_missiles
from .rl_env import equip_scripts
from .sim import ROOT, Simulation, _RecordedController, _number, _write_json, load_scenario, result_document, validate_scenario

ASSETS = Path(__file__).with_name("sandbox_assets")
MAX_TEAM = 16
SPEEDS = (0.5, 1, 2, 4, 8, 16, 32, 64)


def validate_sandbox(value):
    """Reuse the existing strict aircraft schema, permitting 1..16 per team."""
    if not isinstance(value, dict):
        raise ValueError("场景必须是 JSON 对象")
    teams = value.get("teams")
    if not isinstance(teams, list) or len(teams) != 2 or any(
            not isinstance(t, list) or not 1 <= len(t) <= MAX_TEAM for t in teams):
        raise ValueError("需要两个队伍，每队 1–16 架飞机")
    out = None
    normalized = [[], []]
    for team, members in enumerate(teams):
        for member in members:
            probe = dict(value, teams=[[member], [teams[1-team][0]]])
            checked = validate_scenario(probe)
            normalized[team].append(checked["teams"][0][0])
            if out is None:
                out = checked
    out["teams"] = normalized
    return out


def catalog(library):
    # Every airframe with equipment data and at least one modelled active-radar missile, not only
    # the RL training groups: all of them build and fly under the same FM / missile models.
    data = equipment_data().equipment
    rows = []
    for aircraft in sorted(data):
        eq = data[aircraft]
        missiles = modelled_missiles(aircraft, library)
        if missiles:
            rows.append(dict(id=aircraft, missiles=[dict(id=m, max=eq.missiles[m],
                                                       profile=library.info(m).profile_id) for m in missiles],
                             chaff_max=eq.countermeasures))
    return dict(aircraft=rows, archetypes=list(ARCHETYPES), skills=list(SKILLS), max_team=MAX_TEAM,
                speeds=list(SPEEDS), tick_s=SUBSTEP_S,
                presets={"duel": preset(), "four": preset(True), "mixed": mixed_preset()})


def preset(multi=False):
    scene = load_scenario()
    if multi:
        scene["name"] = "双机编队对抗"
        for team in scene["teams"]:
            other = copy.deepcopy(team[0])
            team[0]["position_m"][0] = -5000
            other["position_m"][0] = 5000
            team.append(other)
    return scene


MIXED = (
    (("f_15c_golden_eagle", "us_aim_120d", "middle"), ("f_16c_block_52_aesa", "us_aim_120c_5", "left"),
     ("saab_jas39e", "swd_rb99", "right"), ("ef_2000_typhoon_aesa", "us_aim_120c_5", "rusher")),
    (("su_30sm2", "su_r_77_1", "middle"), ("j_16", "cn_pl12a", "left"),
     ("mig_29smt_9_19", "su_r_77", "crawler"), ("j_10c", "cn_pl12a", "right")))


def mixed_preset():
    """4 v 4 with different airframes and missiles on each side, 60 km apart, 8 km between wingmen."""
    eq = equipment_data().equipment
    scene = load_scenario()
    scene["name"] = "4 对 4 混编"
    teams = []
    for team, members in enumerate(MIXED):
        side = -1. if team == 0 else 1.
        teams.append([dict(aircraft=a, archetype=arch, skill="top", missile=m, missiles=eq[a].missiles[m],
                           position_m=[(i-1.5)*8000., side*30000., 2500.+500.*(i % 2)],
                           velocity_mps=[0., -side*300., 0.]) for i, (a, m, arch) in enumerate(members)])
    scene["teams"] = teams
    return scene


class SandboxPilot:
    """Explicit autonomous/manual switch; manual weapons need observed tracks."""
    def __init__(self, controller, eng, ident):
        self.auto = _RecordedController(controller, eng, ident)
        self.mode = "auto"
        self.observation = None
        self.order = None
        self.keys = dict(roll=0, pitch=0, throttle=0, airbrake=False)
        self.keys_until = 0.

    @property
    def phase(self):
        return self.auto.phase if self.mode == "auto" else self.mode

    def describe(self):
        return self.auto.describe()

    def decide(self, obs):
        self.observation = obs
        if self.mode == "auto":
            return self.auto.decide(obs)
        return Action(radar=RadarCommand(mode="tws"))

    def advance(self, eng, plane):
        if self.mode == "auto":
            self.auto.advance(eng, plane)
            return
        if self.mode == "pilot":
            keys = dict(self.keys) if time.monotonic() < self.keys_until else dict(roll=0, pitch=0, throttle=0, airbrake=False)
            if keys.pop("level", False) and keys["roll"] == 0:
                # Wings-level assist: bang-bang roll toward 0 deg, aimed at where the current roll rate settles.
                settle = plane.own.roll_deg+math.degrees(plane.flight._roll_rate)*.35
                keys["roll"] = 1 if settle < -4. else -1 if settle > 4. else 0
            eng.apply(plane, Action(flight=KeyboardCommand(**keys, authority=.5)))
            plane.camera.point(plane.own.heading_deg, plane.own.pitch_deg)
            plane.camera.advance(plane.own, SUBSTEP_S)
            return
        order = self.order
        heading = order["heading_deg"]
        xy = order["destination_m"]
        if xy is not None:
            dx, dy = xy[0]-plane.own.position[0], xy[1]-plane.own.position[1]
            if math.hypot(dx, dy) < max(500., math.hypot(*plane.own.velocity)*2):
                order["destination_m"] = None
                order["heading_deg"] = plane.own.heading_deg
                eng.event("waypoint_arrived", plane=plane.ident, destination_m=xy)
                heading = order["heading_deg"]
            else:
                heading = bearing_of(dx, dy)
        eng.apply(plane, Action(flight=FlightCommand(heading_deg=heading, altitude_m=order["altitude_m"],
                                                    speed_mps=order["speed_mps"], airbrake_allowed=True)))
        plane.camera.point(plane.own.heading_deg, plane.own.pitch_deg)
        plane.camera.advance(plane.own, SUBSTEP_S)


class SandboxSimulation(Simulation):
    def _header(self):
        header = super()._header()
        header["simulator"] = "wt_overlay.sandbox"
        header["analysis_view"] = "world truth; never provided to autonomous controllers"
        header["manual_route"] = "FlightCommand navigation / KeyboardCommand piloting; TWS search; observed-track launch; Action chaff"
        return header


def build_sandbox(value, *, library=None, missile_sim=None):
    value = validate_sandbox(value)
    library = library or default_library(missile_sim)
    equipment = equipment_data().equipment
    members = []
    for team in value["teams"]:
        group = []
        for p in team:
            eq = equipment.get(p["aircraft"])
            if eq is None:
                raise ValueError(f"未知机型 {p['aircraft']}")
            mid = p.get("missile")
            if mid is not None:
                if mid not in modelled_missiles(p["aircraft"], library):
                    raise ValueError(f"{p['aircraft']} 不支持主动雷达导弹 {mid}")
                if p.get("missiles", eq.missiles[mid]) > eq.missiles[mid]:
                    raise ValueError(f"{p['aircraft']} 最多挂载 {eq.missiles[mid]} 枚 {mid}")
            if p.get("chaff", 0) > eq.countermeasures:
                raise ValueError(f"{p['aircraft']} 箔条数量上限为 {eq.countermeasures}")
            member = {k: p[k] for k in ("aircraft", "archetype", "skill", "missile", "missiles") if k in p}
            member.update(altitude_m=p["position_m"][2],
                          mach=math.sqrt(sum(v*v for v in p["velocity_mps"]))/atmosphere(p["position_m"][2])[1])
            group.append(member)
        members.append(group)
    match = make_match(members, value["seed"], range_km=60., library=library, map_half_m=value["map_half_m"])
    by_team = [[p for p in match.specs if p.team == i] for i in (0, 1)]
    centers = [tuple(sum(p["position_m"][j] for p in t)/len(t) for j in (0, 1)) for t in value["teams"]]
    for i, group in enumerate(by_team):
        for s, p in zip(group, value["teams"][i]):
            if "missile" in p and p["missile"] is None:
                s.missile, s.missiles = None, 0
            if s.missile is None and p.get("missiles", 0) > 0:
                raise ValueError(f"{s.aircraft} 没有受支持的导弹")
            s.position, s.velocity = tuple(p["position_m"]), tuple(p["velocity_mps"])
            for key in ("chaff", "mass_factor", "rcs_ratio", "flame_probability"):
                if key in p:
                    setattr(s, key, p[key])
            s.home_xy, s.enemy_xy = tuple(s.position[:2]), centers[1-i]
            pilot = s.controller
            v = math.hypot(*s.velocity[:2])
            pilot.forward = (s.velocity[0]/v, s.velocity[1]/v)
            pilot.forward_deg = bearing_of(*pilot.forward)
            pilot.left = (-pilot.forward[1], pilot.forward[0])
            pilot.home_xy, pilot.enemy_xy = s.home_xy, s.enemy_xy
            pilot.anc = (*s.enemy_xy, None, "spawn")
    eng = SandboxSimulation(match.specs, match.seed, map_half_m=match.map_half_m,
                            time_limit_s=value["time_limit_s"], library=library)
    eng.scenario = value
    equip_scripts(eng)
    for p in eng.planes:
        p.controller = SandboxPilot(p.controller, eng, p.ident)
    return eng


def set_order(eng, ident, command):
    """Validated navigation order, applied only by the world-owning thread."""
    if type(ident) is not int or not 0 <= ident < len(eng.planes):
        raise ValueError("请选择一架飞机")
    if eng.reason is not None:
        raise ValueError("对局已结束，请重置")
    p = eng.planes[ident]
    if not p.alive:
        raise ValueError("已损失的飞机无法接收命令")
    if not isinstance(command, dict) or set(command)-{"mode", "heading_deg", "altitude_m", "speed_mps", "destination_m"}:
        raise ValueError("未知导航命令字段")
    mode = command.get("mode", "manual")
    if mode not in ("auto", "manual", "pilot"):
        raise ValueError("控制方式必须为 auto、manual 或 pilot")
    pilot = p.controller
    pilot.keys, pilot.keys_until = dict(roll=0, pitch=0, throttle=0, airbrake=False), 0.
    if mode == "auto":
        pilot.mode, pilot.order = "auto", None
    elif mode == "pilot":
        pilot.mode, pilot.order = "pilot", None
    else:
        old = pilot.order or dict(heading_deg=p.own.heading_deg, altitude_m=p.own.position[2],
                                 speed_mps=math.hypot(*p.own.velocity), destination_m=None)
        order = dict(old)
        for k, v in command.items():
            if k == "mode":
                continue
            if k == "destination_m":
                if v is not None and (not isinstance(v, (list, tuple)) or len(v) != 2):
                    raise ValueError("目的地需要东、北两个坐标")
                order[k] = None if v is None else [_number(x, "目的地", lo=-eng.map_half_m,
                                                            hi=eng.map_half_m, inclusive_lo=True) for x in v]
            else:
                order[k] = _number(v, k, lo=0., hi=360., inclusive_lo=True) if k == "heading_deg" else \
                    _number(v, k, lo=100., hi=MAX_ALTITUDE_M, inclusive_lo=True) if k == "altitude_m" else \
                    _number(v, k, lo=MIN_STEP_SPEED_MPS, inclusive_lo=True)
        # Independently validate speed against the final altitude (dict key order has no effect).
        _number(order["speed_mps"], "目标速度", lo=MIN_STEP_SPEED_MPS, inclusive_lo=True,
                hi=2.*atmosphere(order["altitude_m"])[1])
        pilot.mode, pilot.order = "manual", order
    p.next_decision = eng.tick
    eng.event("human_order", plane=ident, mode=mode, order=copy.deepcopy(pilot.order))


def analysis_snapshot(eng):
    planes = []
    for p in eng.planes:
        pilot = p.controller
        obs = pilot.observation
        tracks = [] if obs is None else [dict(track=STT_TRACK if c.kind == "stt" else c.track_id,
                                              kind=c.kind, range_m=c.range_m, bearing_deg=c.bearing_deg)
                                         for c in obs.radar if c.kind == "stt" or c.track_id is not None]
        planes.append(dict(id=p.ident, team=p.team, aircraft=p.aircraft, alive=p.alive,
                           position_m=list(p.flight.state.position), heading_deg=p.own.heading_deg,
                           pitch_deg=p.own.pitch_deg, roll_deg=p.own.roll_deg,
                           speed_mps=math.sqrt(sum(v*v for v in p.flight.state.velocity)),
                           velocity_mps=list(p.flight.state.velocity), load_g=p.flight.load,
                           engine_percent=p.flight.state.engine_percent, aoa_deg=p.flight.state.aoa_deg,
                           airbrake=p.flight.state.airbrake, missiles=p.missiles,
                           missile=p.missile_id, chaff=p.chaff, phase=p.phase, mode=pilot.mode,
                           order=copy.deepcopy(pilot.order), tracks=tracks, death=p.death,
                           radar_mode=p.radar.mode if p.radar else "off"))
    missiles = [dict(uid=m.uid, team=m.shooter.team, shooter=m.shooter.ident, target=m.target.ident,
                     position_m=list(m.pos_enu), velocity_mps=list(m.vel_enu), heading_deg=bearing_of(*m.vel_enu[:2]),
                     age_s=m.time_s, seeker=m.seeker_on, datalink=m.datalink) for m in eng.missiles if not m.done]
    return dict(time_s=eng.time, tick=eng.tick, planes=planes, missiles=missiles,
                teams_alive=list(eng.alive_counts()), launches=eng.launches,
                missile_errors=eng.missile_errors, fm_faults=sum(p.flight.faults for p in eng.planes),
                events=copy.deepcopy(eng.log[-120:]), event_total=len(eng.log), reason=eng.reason)


class SandboxSession:
    """A single worker world, a responsive cached read API, and bounded commands."""
    def __init__(self, *, output_root=None, missile_sim=None, replay_root=None):
        self.output_root = Path(output_root or ROOT/"outputs"/"sandbox").resolve()
        # WT replays (docs/sandbox_replay_spec.md): playback and forks; sandbox runs under output_root are listed too
        self.replay_root = Path(replay_root or ROOT/"outputs"/"engagements"/"wt_real").resolve()
        self.track = self.track_file = self.track_sha = self.track_caps = None
        self.play_t, self.playing, self._play_wall, self.fork_t = 0., False, 0., None
        self.library = default_library(missile_sim)
        self.catalog = catalog(self.library)
        self.scenario = preset()
        self.eng = None
        self.status = "setup"
        self.speed = 1.
        self.advance_ticks = 0
        self.error = None
        self.run_dir = None
        self.last_output = None
        self.output_dirs = {}
        self.commands = queue.Queue()
        self.jobs = {}
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.snapshot = {}
        self.wall_start = None
        self.actual_speed = 0.
        self._rate_time = time.monotonic()
        self._rate_sim = 0.
        self._publish()
        self.thread = threading.Thread(target=self._work, name="air-combat-physics", daemon=True)
        self.thread.start()

    def submit(self, command):
        if not isinstance(command, dict) or not isinstance(command.get("action"), str):
            raise ValueError("命令必须包含 action")
        jid = secrets.token_hex(6)
        with self.lock:
            self.jobs[jid] = dict(id=jid, done=False)
            # Completed job receipts are ephemeral; active requests are retained.
            for old in list(self.jobs)[:-64]:
                if self.jobs[old]["done"]:
                    del self.jobs[old]
        self.commands.put((jid, copy.deepcopy(command)))
        return jid

    def state(self):
        with self.lock:
            return copy.deepcopy(self.snapshot)

    def job(self, jid):
        with self.lock:
            return copy.deepcopy(self.jobs.get(jid))

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=10.)

    def _start_world(self, eng=None):
        eng = eng or build_sandbox(self.scenario, library=self.library)
        name = datetime.now().strftime("run-%Y%m%d-%H%M%S-")+secrets.token_hex(3)
        run_dir = self.output_root/name
        run_dir.mkdir(parents=True)
        _write_json(run_dir/"scenario.json", self._scenario_doc(eng))
        eng.replay = ReplayWriter(run_dir/"replay.jsonl", keep=False)
        eng.replay.write(eng._header())
        eng._frame()
        self.eng, self.run_dir = eng, run_dir
        self.output_dirs[name] = run_dir
        self.wall_start = time.perf_counter()
        self._rate_sim, self._rate_time = 0., time.monotonic()
        self.error = None
        self.status = "paused"

    def _scenario_doc(self, eng):
        doc = dict(requested=eng.scenario, resolved=eng.resolved_scenario())
        if hasattr(eng, "replay_source"):   # replay fork: provenance, release and skipped shots
            from .sandbox_replay import fork_source
            doc["replay_source"] = fork_source(eng)
        return doc

    def _save(self, disposition):
        if self.eng is None:
            self.output_root.mkdir(parents=True, exist_ok=True)
            _write_json(self.output_root/"scenario.json", self.scenario)
            return dict(scenario="/api/download?file=scenario.json", path=str(self.output_root/"scenario.json"))
        if hasattr(self.eng, "replay_source"):
            _write_json(self.run_dir/"scenario.json", self._scenario_doc(self.eng))
        doc = result_document(self.eng)
        doc["sandbox_status"] = disposition
        doc["performance"] = dict(wall_s=time.perf_counter()-self.wall_start,
                                  stepping_wall_s=self.eng.result().wall, missile_steps=self.eng.missile_steps)
        _write_json(self.run_dir/"result.json", doc)
        if self.eng.replay and self.eng.replay._handle is not None:
            self.eng.replay._handle.flush()
        self.last_output = dict(run=self.run_dir.name, path=str(self.run_dir), status=disposition,
                                terminal=doc["terminal"], reason=doc["reason"], time_s=doc["time_s"],
                                winner_team=doc["winner_team"], result=doc,
                                links={k: f"/api/download?run={self.run_dir.name}&file={k}" for k in
                                       ("scenario.json", "result.json", "replay.jsonl")})
        return self.last_output

    def _cancel(self, disposition="cancelled"):
        if self.eng is not None and self.eng.reason is None:
            self.eng.event("sandbox_cancel", reason=disposition)
            self.eng._frame()
            self._save(disposition)
            self.eng.replay.close()
        elif self.eng is not None:
            self._save("completed")

    def _execute(self, command):
        action = command["action"]
        if action == "configure":
            if self.status != "setup":
                raise ValueError("布置只在准备阶段可修改，请先重置")
            checked = validate_sandbox(command["scenario"])
            # Build with the real models so equipment errors surface before Start.
            build_sandbox(checked, library=self.library)
            self.scenario = checked
        elif action == "start":
            if self.status == "playback":
                raise ValueError("回放模式：请用播放 / 暂停，或从此刻接管")
            if self.status == "setup":
                self._start_world()
            if self.eng.reason is not None or self.status == "error":
                raise ValueError("对局已经结束或发生错误，请重置")
            self.status = "running"
        elif action == "pause":
            self.playing = False
            if self.status in ("running", "stepping"):
                self.status, self.advance_ticks = "paused", 0
            self._release_keys()
        elif action == "step":
            if self.status not in ("setup", "paused"):
                raise ValueError("单步推进前请暂停；结束后需重置")
            seconds = _number(command.get("seconds", 1.), "推进时间", lo=0., hi=10.)
            if self.status == "setup":
                self._start_world()
            self.advance_ticks = max(1, round(seconds/SUBSTEP_S))
            self.status = "stepping"
        elif action == "reset":
            self._cancel()
            self.eng = None
            self.status, self.advance_ticks, self.error, self.actual_speed = "setup", 0, None, 0.
            if self.track is not None:   # a replay stays loaded: back to its playback (at the fork time)
                self.status, self.playing = "playback", False
                if self.fork_t is not None:
                    self.play_t, self.fork_t = self.fork_t, None
        elif action == "speed":
            speed = command.get("speed")
            if isinstance(speed, bool) or speed not in SPEEDS:
                raise ValueError("请选择列表中的推进倍率")
            self.speed = float(speed)
        elif action == "order":
            if self.eng is None or self.status == "error":
                raise ValueError("先启动或单步建立对局，再下达航行命令")
            self._check_released(command.get("plane"))
            set_order(self.eng, command.get("plane"), command.get("order", {}))
        elif action == "keys":
            if self.eng is None or self.eng.reason is not None or self.status == "error":
                raise ValueError("需要尚未结束的对局")
            ident, keys = command.get("plane"), command.get("keys")
            if type(ident) is not int or not 0 <= ident < len(self.eng.planes):
                raise ValueError("请选择飞机")
            if not isinstance(keys, dict) or set(keys)-{"level"} != {"roll", "pitch", "throttle", "airbrake"} or \
                    any(type(keys[k]) is not int or keys[k] not in (-1, 0, 1) for k in ("roll", "pitch", "throttle")) or \
                    any(type(keys.get(k, False)) is not bool for k in ("airbrake", "level")):
                raise ValueError("无效的驾驶输入")
            self._check_released(ident)
            p = self.eng.planes[ident]
            if p.controller.mode != "pilot" or not p.alive:
                raise ValueError("请进入一架存活飞机的驾驶模式")
            if self.status != "running" and any(keys.values()):
                raise ValueError("暂停时不能保持驾驶输入，请先继续")
            if keys != p.controller.keys:
                self.eng.event("human_input", plane=ident, keys=keys)
            p.controller.keys = keys
            p.controller.keys_until = time.monotonic()+.5
        elif action in ("fire", "chaff"):
            if self.eng is None or self.eng.reason is not None or self.status == "error":
                raise ValueError("需要一场尚未结束的对局")
            ident = command.get("plane")
            if type(ident) is not int or not 0 <= ident < len(self.eng.planes):
                raise ValueError("请选择飞机")
            self._check_released(ident)
            p = self.eng.planes[ident]
            if not p.alive or p.controller.mode == "auto":
                raise ValueError("请切换一架存活飞机到人工导航或驾驶模式")
            if action == "fire":
                track = command.get("track")
                obs = p.controller.observation
                observed = [] if obs is None else [STT_TRACK if c.kind == "stt" else c.track_id for c in obs.radar
                                                   if c.kind == "stt" or c.track_id is not None]
                if type(track) is not int or track not in observed:
                    raise ValueError("只能向该飞机实际观测到的雷达航迹发射")
                if self.eng.launch(p, track) is None:
                    raise ValueError("发射被现有检查拒绝：航迹、弹药、间隔或发射角条件未满足")
            else:
                if p.chaff <= 0:
                    raise ValueError("箔条已用完")
                self.eng.apply(p, Action(chaff=1))
        elif action == "save":
            if self.eng is None and self.status == "playback":
                raise ValueError("回放模式没有可保存的推演；接管后才会生成记录")
            return self._save("completed" if self.eng and self.eng.reason else
                              "error" if self.status == "error" else "partial")
        elif action in ("load_replay", "play", "seek", "unload", "fork"):
            return self._replay_command(action, command)
        else:
            raise ValueError(f"未知命令 {action}")

    # -- WT replays (docs/sandbox_replay_spec.md) ------------------------------------------------------------

    def replays(self):
        from .sandbox_replay import list_replays
        return [entry for entry, _ in list_replays(self.replay_root, self.output_root)]

    def _check_released(self, ident):
        if self.eng is not None and type(ident) is int and 0 <= ident < len(self.eng.planes) and \
                getattr(self.eng.planes[ident].controller, "tracked", False):
            raise ValueError("该飞机仍钉在 WT 轨迹上，不能下令")

    def _replay_command(self, action, command):
        from . import sandbox_replay as sr
        if action == "load_replay":
            if self.status not in ("setup", "playback"):
                raise ValueError("请先重置当前推演，再加载回放")
            name = command.get("file")
            files = {entry["file"]: path for entry, path in sr.list_replays(self.replay_root, self.output_root)}
            if not isinstance(name, str) or name not in files:
                raise ValueError("只能加载回放列表中的文件")
            from .fm.catalog import find_aircraft
            track = sr.ReplayTrack(files[name], name=name)
            self.track, self.track_file, self.track_sha = track, name, sr.file_sha256(files[name])
            self.track_caps = sr.capabilities(track, self.library, equipment_data().equipment, find_aircraft)
            self.play_t, self.playing, self.fork_t, self.status = track.start_s, False, None, "playback"
            self._rate_sim, self._rate_time = self.play_t, time.monotonic()
            # Flight models of the aircraft that could be taken over: built in the background (~1.5 s each, cached)
            types = sorted({track.info[r]["aircraft"] for r, c in self.track_caps.items() if c["controllable"]})
            threading.Thread(target=_warm_models, args=(types,), name="fm-warm", daemon=True).start()
            return dict(file=name, duration_s=track.duration_s, start_s=track.start_s)
        if self.track is None or self.status != "playback":
            if action == "unload" and self.track is None:
                return None
            if action != "unload":
                raise ValueError("请先加载回放（接管推演中请先返回回放）")
        if action == "play":
            if self.play_t >= self.track.duration_s-1e-9:
                self.play_t = self.track.start_s
            self.playing, self._play_wall = True, time.monotonic()
        elif action == "seek":
            self.play_t = _number(command.get("t"), "回放时刻", lo=self.track.start_s, hi=self.track.duration_s,
                                  inclusive_lo=True)
            self._play_wall = self._rate_time = time.monotonic()
            self._rate_sim = self.play_t
        elif action == "unload":
            if self.eng is not None:
                self._cancel()
                self.eng = None
            self.track = self.track_file = self.track_sha = self.track_caps = None
            self.playing, self.fork_t = False, None
            self.status, self.advance_ticks, self.error, self.actual_speed = "setup", 0, None, 0.
        elif action == "fork":
            t = _number(command.get("t", self.play_t), "接管时刻", lo=self.track.start_s, hi=self.track.duration_s,
                        inclusive_lo=True)
            control = command.get("control", {})
            if not isinstance(control, dict) or any(not isinstance(v, str) for v in control.values()):
                raise ValueError("控制方式需要 {飞机编号: 方式}")
            try:
                control = {int(k): v for k, v in control.items()}
            except (TypeError, ValueError):
                raise ValueError("飞机编号必须是整数") from None
            include_ai, me = command.get("include_ai", False), command.get("me")
            if type(include_ai) is not bool or (me is not None and type(me) is not int):
                raise ValueError("无效的接管选项")
            eng = sr.build_fork(self.track, t, control, library=self.library, include_ai=include_ai, me=me,
                                file=self.track_file, sha256=self.track_sha)
            self.playing = False
            self._start_world(eng)
            self.fork_t = t
            if "pilot" in control.values():
                self.speed = 1.
            # Up to t_fork everything is pinned to the replay: that stretch runs unthrottled (_work) and the world
            # pauses at t_fork so the user takes the controls from a standstill.
            self.status = "running" if eng.offset < t-1e-9 else "paused"
            return dict(t0=eng.offset, t_fork=t, planes=len(eng.planes), me=eng.fork_me)

    def _pre_fork(self):
        """A replay fork still before its takeover: some chosen aircraft not yet released (or, with none chosen, before
        t_fork). Every aircraft is pinned to the replay until then."""
        if self.fork_t is None or self.eng is None:
            return False
        pending = [p for p in self.eng.planes if p.ident in getattr(self.eng, "release_at", {})]
        if pending:
            return any(p.alive and p.flight.tracked for p in pending)
        return getattr(self.eng, "offset", 0.)+self.eng.time < self.fork_t-1e-9

    def _release_keys(self):
        if self.eng is not None:
            for p in self.eng.planes:
                p.controller.keys = dict(roll=0, pitch=0, throttle=0, airbrake=False)
                p.controller.keys_until = 0.

    def _publish(self):
        if self.eng is None and self.status == "playback" and self.track is not None:
            from .sandbox_replay import playback_snapshot
            snap = playback_snapshot(self.track, self.play_t, caps=self.track_caps, playing=self.playing,
                                     file=self.track_file)
        elif self.eng is not None and hasattr(self.eng, "replay_source"):
            from .sandbox_replay import fork_snapshot
            snap = fork_snapshot(self.eng)
        elif self.eng is not None:
            snap = analysis_snapshot(self.eng)
        else:
            planes = []
            for team, members in enumerate(self.scenario["teams"]):
                for p in members:
                    planes.append(dict(id=len(planes), team=team, aircraft=p["aircraft"], alive=True,
                                       position_m=p["position_m"], heading_deg=bearing_of(*p["velocity_mps"][:2]),
                                       speed_mps=math.sqrt(sum(v*v for v in p["velocity_mps"])),
                                       missiles=p.get("missiles"), missile=p.get("missile"), chaff=p.get("chaff"),
                                       mode="auto", order=None, tracks=[], phase="准备"))
            snap = dict(time_s=0., tick=0, planes=planes, missiles=[], teams_alive=[len(t) for t in self.scenario["teams"]],
                        launches=0, events=[], event_total=0, reason=None, fm_faults=0, missile_errors=0)
        snap.update(status=self.status, speed=self.speed, actual_speed=self.actual_speed,
                    error=self.error, scenario=copy.deepcopy(self.scenario), output=copy.deepcopy(self.last_output),
                    analysis_view=True)
        if self.track is not None:
            snap["replay_file"] = self.track_file
        with self.lock:
            self.snapshot = snap

    def _work(self):
        if self.eng is None and self.status == "setup":
            # Warm the model caches (~7 s cold) so the first Start responds immediately.
            try:
                build_sandbox(self.scenario, library=self.library)
            except ValueError:
                pass
        next_tick = time.monotonic()
        published = next_tick
        try:
            while not self.stop_event.is_set():
                try:
                    jid, command = self.commands.get(timeout=.02 if self.status not in ("running", "stepping") else .001)
                except queue.Empty:
                    jid = None
                if jid:
                    try:
                        result = self._execute(command)
                        receipt = dict(id=jid, done=True, result=result)
                    except (ValueError, KeyError, FileNotFoundError) as exc:
                        receipt = dict(id=jid, done=True, error=str(exc))
                    self._publish()
                    with self.lock:
                        self.jobs[jid] = receipt
                    next_tick = time.monotonic()
                now = time.monotonic()
                if self.status in ("running", "stepping"):
                    deadline = now+.035
                    while time.monotonic() < deadline and (self.status == "stepping" or self._pre_fork() or
                                                           time.monotonic() >= next_tick):
                        pre_fork = self._pre_fork()
                        self.eng.step()
                        if self.eng.reason is not None:
                            self.status, self.advance_ticks = "completed", 0
                            self._save("completed")
                            break
                        if pre_fork:
                            next_tick = time.monotonic()
                            if not self._pre_fork():   # reached t_fork: hand over from a paused world
                                self.status, self.advance_ticks = "paused", 0
                                break
                            continue
                        if self.status == "stepping":
                            self.advance_ticks -= 1
                            if self.advance_ticks <= 0:
                                self.status = "paused"
                                break
                        next_tick += SUBSTEP_S/self.speed
                    # Cap scheduling debt, never skip or enlarge a physics tick.
                    next_tick = max(next_tick, time.monotonic()-.1)
                elif self.status == "paused":
                    next_tick = now
                elif self.status == "playback" and self.playing:
                    # Playback clock: wall time x the chosen speed; pauses itself at the end of the replay.
                    self.play_t = min(self.track.duration_s, self.play_t+(now-self._play_wall)*self.speed)
                    self._play_wall = now
                    if self.play_t >= self.track.duration_s-1e-9:
                        self.playing = False
                if now-self._rate_time >= 1.:
                    sim_t = self.eng.time if self.eng else self.play_t if self.status == "playback" else 0.
                    self.actual_speed = max(0., (sim_t-self._rate_sim)/(now-self._rate_time))
                    self._rate_sim, self._rate_time = sim_t, now
                if now-published >= .05 or self.status == "completed":
                    self._publish()
                    published = now
                    if self.status == "completed":
                        self.stop_event.wait(.02)
        except Exception as exc:
            # Unexpected model errors surface explicitly; no default world is substituted.
            self.error = f"{type(exc).__name__}: {exc}"
            self.status = "error"
            if jid is not None:
                with self.lock:
                    self.jobs[jid] = dict(id=jid, done=True, error=self.error)
            if self.eng is not None:
                self.eng.event("sandbox_error", error=self.error)
                self._save("error")
                self.eng.replay.close()
            self._publish()
            # Keep serving reset/save commands after an explicit runtime failure.
            while not self.stop_event.is_set():
                try:
                    jid, command = self.commands.get(timeout=.1)
                except queue.Empty:
                    continue
                try:
                    if command["action"] not in ("save", "reset"):
                        raise ValueError("运行错误：请重置")
                    result = self._execute(command)
                    receipt = dict(id=jid, done=True, result=result)
                except Exception as error:
                    receipt = dict(id=jid, done=True, error=str(error))
                self._publish()
                with self.lock:
                    self.jobs[jid] = receipt
                if command["action"] == "reset" and "error" not in receipt:
                    return self._work()
        finally:
            if self.stop_event.is_set():
                self._cancel("server_stopped")


def _warm_models(types):
    from .flight import aircraft_model
    for aircraft in types:
        try:
            aircraft_model(aircraft)
        except (ValueError, KeyError, OSError):
            pass


class SandboxServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, session):
        self.session = session
        self.token = secrets.token_urlsafe(24)
        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def _reply(self, value, status=200):
        raw = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(raw)

    def _valid_host(self):
        return self.headers.get("Host") == f"127.0.0.1:{self.server.server_port}"

    def do_GET(self):
        if not self._valid_host():
            return self._reply(dict(error="仅允许本机地址"), 403)
        parsed = urlsplit(self.path)
        if parsed.path == "/api/state":
            return self._reply(self.server.session.state())
        if parsed.path == "/api/replays":
            return self._reply(self.server.session.replays())
        if parsed.path == "/api/catalog":
            return self._reply(dict(self.server.session.catalog, token=self.server.token))
        if parsed.path == "/api/command":
            job = self.server.session.job(parse_qs(parsed.query).get("id", [""])[0])
            return self._reply(job or dict(error="命令不存在"), 200 if job else 404)
        if parsed.path == "/api/download":
            query = parse_qs(parsed.query)
            name = query.get("file", [""])[0]
            run = query.get("run", [None])[0]
            if name not in ("scenario.json", "result.json", "replay.jsonl"):
                return self._reply(dict(error="文件不存在"), 404)
            root = self.server.session.output_root if run is None else self.server.session.output_dirs.get(run)
            if root is None:
                return self._reply(dict(error="对局不存在"), 404)
            path = root/name
            if not path.is_file():
                return self._reply(dict(error="文件尚未保存"), 404)
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Disposition", f'attachment; filename="{name}"')
            size = path.stat().st_size
            self.send_header("Content-Length", str(size))
            self.end_headers()
            with path.open("rb") as handle:
                remaining = size
                while remaining:
                    block = handle.read(min(65536, remaining))
                    if not block:
                        break
                    self.wfile.write(block)
                    remaining -= len(block)
            return
        asset = {"/": ("index.html", "text/html; charset=utf-8"),
                 "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                 "/style.css": ("style.css", "text/css; charset=utf-8")}.get(parsed.path)
        if asset is None:
            return self._reply(dict(error="不存在"), 404)
        raw = (ASSETS/asset[0]).read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", asset[1])
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; object-src 'none'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self):
        if not self._valid_host() or urlsplit(self.path).path != "/api/command":
            return self._reply(dict(error="拒绝请求"), 403)
        origin = self.headers.get("Origin")
        if origin is not None and origin != f"http://127.0.0.1:{self.server.server_port}":
            return self._reply(dict(error="拒绝跨站命令"), 403)
        if self.headers.get("X-Sandbox-Token") != self.server.token:
            return self._reply(dict(error="页面会话已过期，请刷新"), 403)
        try:
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= 131072 or self.headers.get_content_type() != "application/json":
                return self._reply(dict(error="需要有效的 JSON 命令"), 400)
            command = json.loads(self.rfile.read(length))
            jid = self.server.session.submit(command)
        except (ValueError, json.JSONDecodeError) as exc:
            return self._reply(dict(error=str(exc)), 400)
        self._reply(dict(id=jid), 202)


def main(argv=None):
    parser = argparse.ArgumentParser(description="本机空战沙盘：地图布置、实时推演与人工导航")
    parser.add_argument("--port", type=int, default=8765, help="127.0.0.1 端口，默认 8765；0 为自动选择")
    parser.add_argument("--missile-sim", type=Path, help="相邻 missle_sim 模型目录的替代路径")
    parser.add_argument("--no-browser", action="store_true", help="只输出网址，不自动打开浏览器")
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error("端口必须为 0–65535")
    try:
        session = SandboxSession(missile_sim=args.missile_sim)
        server = SandboxServer(("127.0.0.1", args.port), session)
    except (FileNotFoundError, ValueError, OSError) as exc:
        parser.error(str(exc))
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"空战沙盘：{url}\n输出：{session.output_root}\nCtrl+C 关闭；未完成的对局保存为中止。", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        session.close()


if __name__ == "__main__":
    main()
