"""Opening 'lottery' shot: from the air spawn, how to climb and when / at what pitch to fire so the first missile
hits soonest (docs/opening_shot_spec.md). Model-based, no RL policy.

Scenario (user, C): the teams spawn head-on ``distance_m`` apart and climb straight at each other; only the first
shot counts, nothing after it is modelled. The enemy flies its own FM (default a Typhoon with full internal fuel,
full afterburner, a 15 deg climb to 10000 m, then level) straight at our spawn and never reacts. Our aircraft flies
its FM at the 8111 mass from the measured state: full afterburner, a constant-angle climb, a flown pull-up to the
launch pitch, then the missile (missile_sim public API, launcher support always on, grade D).

The search (``plan_opening``) runs its shots in worker processes; ``cue`` turns a plan and the live telemetry into
the arrow and text the HUD shows. Frame: ENU metres, our spawn at the origin, +x = the spawn heading (towards the
enemy spawn).
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field, replace
from functools import lru_cache
import json
import math
import os
import time

from .climb import CUE_DEADBAND_DEG, valid_number

# Air spawn (replays: about 2.5 km and 290 m/s, level): the first valid sample after a gap must look like one.
SPAWN_ALTITUDE_M = (1000., 6000.)
SPAWN_TAS_MPS = (150., 400.)
SPAWN_MAX_VY_MPS = 20.
SPAWN_GAP_S = 3.          # a shorter telemetry gap is a lost poll, not a respawn
NEW_LIFE_FUEL_KG = 100.   # fuel going up this much between samples = a new aircraft (respawn)
NEW_LIFE_ALTITUDE_M = 500.  # an altitude jump this large within 2 s = a new aircraft
AIRBORNE_ALTITUDE_M = 300.
AIRBORNE_TAS_MPS = 100.
START_VY_MPS = 8.         # the clock starts on a clear climb ...
START_DV_MPS = 6.         # ... or this much TAS gained within START_DV_S
START_DV_S = 3.
CLIMB_CAP_M = 12000.      # the climb command's altitude; launches come before it
DEFAULT_CLIMB_DEG = 15.   # shown until the first plan arrives
CLIMBS_DEG = (10., 35.)   # searched climb angles
STAGE1_CLIMBS_DEG = (15., 25.)
STAGE1_TIMES_S = (20., 30., 40., 50., 60., 70.)
LATEST_PULL_S = 85.
PULL_MAX_S = 8.
TRACK_TAIL_S = 2.         # own track beyond the last candidate pull + PULL_MAX_S
ENEMY_TRACK_S = 240.
WINDOW_TOLERANCE_S = 1.   # launch window: launches whose hit is at most this much later than the best
SWITCH_CLIMB_S = .5       # a re-plan changes the climb angle only for a hit at least this much earlier
SWITCH_PULL_S = .2        # ... and the pull-up time only for this much
REPLAN_S = 15.
MIN_REPLAN_S = 5.         # a deviation re-plans at most this often (a pilot ignoring the cue would loop)
FREEZE_BEFORE_PULL_S = 3. # no new plan this close to the pull-up
DEVIATION_PATH_DEG = 3.
DEVIATION_TAS_MPS = 20.
DEVIATION_HOLD_S = 2.
END_AFTER_WINDOW_S = 5.
SLOW_MPS = 50.            # below this the arrow waits (parked, taking off)
ARROW_GAIN = 5.           # small direction errors are drawn this much larger (the arrow stays readable)
FALLBACK_MISSILE_MASS_KG = 160.


def wrap(degrees):
    return (degrees+180.) % 360.-180.


def loft_rule(range_m):
    """Launch pitch to try first at a launch range (probe results, D): farther out, steeper; None = no pull-up."""
    if range_m >= 85000.:
        return 55.
    if range_m >= 72000.:
        return 45.
    if range_m >= 58000.:
        return 25.
    return None


@dataclass(frozen=True)
class OpeningSettings:
    """The scenario. Defaults: user decisions of 2026-10-16 (100 km head-on, Typhoon 15 deg to 10000 m)."""
    enabled: bool = False
    missile: str = ""
    distance_m: float = 100000.
    enemy_aircraft: str = "ef_2000_aesa"
    enemy_fuel_fraction: float = 1.
    enemy_missiles: int = 6
    enemy_missile: str = "us_aim_120c_5"      # only its mass is used
    enemy_climb_deg: float = 15.
    enemy_level_m: float = 10000.
    workers: int = 1


def validate_settings(s: OpeningSettings) -> None:
    if type(s.enabled) is not bool:
        raise ValueError("开局抽奖开关须为布尔值")
    if not isinstance(s.missile, str):
        raise ValueError("导弹须为型号名")
    checks = (("distance_m", 30000., 200000., "出生间距须为 30–200 km"),
              ("enemy_fuel_fraction", .05, 1., "敌机油量须为 5–100 %"),
              ("enemy_climb_deg", 3., 45., "敌机爬升角须为 3–45°"),
              ("enemy_level_m", 3000., 14000., "敌机改平高度须为 3000–14000 m"))
    for name, low, high, message in checks:
        value = getattr(s, name)
        if not valid_number(value) or not low <= value <= high:
            raise ValueError(message)
    if type(s.enemy_missiles) is not int or not 0 <= s.enemy_missiles <= 12:
        raise ValueError("敌机导弹数须为 0–12")
    if type(s.workers) is not int or not 1 <= s.workers <= 8:
        raise ValueError("计算进程数须为 1–8")
    if not isinstance(s.enemy_aircraft, str) or not s.enemy_aircraft:
        raise ValueError("请选择敌机机型")


# -- spawn and own position --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Spawn:
    time_s: float               # telemetry clock at the spawn sample
    heading_deg: float | None   # = the bearing of the enemy spawn (replays: median 3 deg off); None = no heading
    altitude_m: float
    tas_mps: float
    aircraft_id: str | None     # None while 8111 gives no type (the user picks the aircraft in settings)
    manual: bool = False


def path_angle_deg(state) -> float | None:
    if not (valid_number(state.tas_mps) and state.tas_mps > 1 and valid_number(state.vertical_speed_mps)):
        return None
    return math.degrees(math.asin(max(-1., min(1., state.vertical_speed_mps/state.tas_mps))))


def heading_of(state) -> float | None:
    """Compass from /indicators, else the own-aircraft direction from /map_obj.json (aircraft without a compass
    gauge), else None."""
    for value in (state.heading_deg, getattr(state, "track_deg", None)):
        if valid_number(value):
            return float(value) % 360.
    return None


class SpawnLatch:
    """When to start the clock (user, 2026-10-16: "8111 看到明显变化（加速 / 爬升）就开始"). A new life (a telemetry
    gap of SPAWN_GAP_S, fuel jumping up, an altitude jump, or a change of aircraft type) arms the latch; so does
    switching the aid on. Armed, the first airborne sample with a clear climb (Vy >= START_VY_MPS) or acceleration
    (TAS up START_DV_MPS within START_DV_S) starts it: that sample is the 'spawn' the plan counts from. No speed or
    altitude window and no heading. ``latch`` starts it now (Ctrl+Alt+N)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.spawn: Spawn | None = None
        self.note = "开局抽奖 · 加速或爬升时开始"
        self._armed = True
        self._last = None          # previous usable sample: (time, altitude, fuel, identity)
        self._speeds = deque()     # recent (time, TAS) while armed

    @staticmethod
    def _usable(state):
        return state.valid and valid_number(state.altitude_m) and valid_number(state.tas_mps)

    def latch(self, state, manual=True) -> Spawn | None:
        if not self._usable(state):
            self.note = ("8111 没有飞行数据，无法开始" if not state.valid else "8111 缺少高度或真空速，无法开始")
            return None
        self.spawn = Spawn(state.time_s, None, float(state.altitude_m), float(state.tas_mps), state.aircraft_id,
                           manual)
        self.note = ""
        self._armed = False
        self._speeds.clear()
        return self.spawn

    def _new_life(self, state, identity):
        if self._last is None:
            return False
        t, altitude, fuel, previous = self._last
        dt = state.time_s-t
        return (dt > SPAWN_GAP_S
                or (valid_number(fuel) and valid_number(state.fuel_kg) and state.fuel_kg-fuel > NEW_LIFE_FUEL_KG)
                or (dt < 2. and abs(state.altitude_m-altitude) > NEW_LIFE_ALTITUDE_M)
                or (identity is not None and previous is not None and identity != previous))

    def update(self, state) -> Spawn | None:
        """Returns the Spawn on the sample that starts the clock, else None."""
        if not self._usable(state):
            return None
        from .fm.catalog import aircraft_key
        identity = aircraft_key(state.aircraft_id) if state.aircraft_id else None
        if self._new_life(state, identity):
            self.spawn = None
            self._armed = True
            self._speeds.clear()
            self.note = "开局抽奖 · 加速或爬升时开始"
        previous = self._last[3] if self._last else None
        self._last = (state.time_s, state.altitude_m, state.fuel_kg, identity or previous)
        if not self._armed:
            return None
        self._speeds.append((state.time_s, state.tas_mps))
        while self._speeds and state.time_s-self._speeds[0][0] > START_DV_S:
            self._speeds.popleft()
        airborne = state.altitude_m >= AIRBORNE_ALTITUDE_M and state.tas_mps >= AIRBORNE_TAS_MPS
        vy = state.vertical_speed_mps
        climbing = valid_number(vy) and vy >= START_VY_MPS
        accelerating = state.tas_mps-min(v for _, v in self._speeds) >= START_DV_MPS
        if airborne and (climbing or accelerating):
            return self.latch(state, manual=False)
        return None


class DeadReckoning:
    """Own distance flown along the spawn line, integrated from TAS and Vy (no heading: all of it along the line)."""

    def __init__(self, spawn: Spawn):
        self.x = 0.
        self.last_s = spawn.time_s

    def update(self, state, spawn: Spawn | None = None):
        if not (state.valid and valid_number(state.tas_mps)):
            return
        dt = state.time_s-self.last_s
        self.last_s = state.time_s
        if 0 < dt < 2.:
            self.x += state.tas_mps*math.cos(math.radians(path_angle_deg(state) or 0.))*dt


# -- the search ----------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class PlanRequest:
    aircraft_id: str
    mass_kg: float
    missile: str
    settings: OpeningSettings
    spawn_altitude_m: float      # the enemy spawns like us
    spawn_tas_mps: float
    t_s: float                   # the start state's time since spawn
    x_m: float                   # its distance flown along the spawn line
    altitude_m: float
    tas_mps: float
    path_deg: float
    hint: tuple | None = None    # (climb_deg, pull_s) of the previous plan: re-plans search around it


@dataclass(frozen=True)
class OpeningPlan:
    available: bool
    reason: str = ""
    climb_deg: float | None = None
    pull_s: float | None = None          # time since spawn the pull-up starts (= launch_s without a pull-up)
    pitch_deg: float | None = None       # launch flight-path angle; None = fire along the climb
    launch_s: float | None = None
    window_s: tuple | None = None        # launch times whose hit is <= best + WINDOW_TOLERANCE_S (same climb)
    launch_range_m: float | None = None
    hit_s: float | None = None
    flight_s: float | None = None
    launch_altitude_m: float | None = None
    launch_tas_mps: float | None = None
    target_altitude_m: float | None = None   # at the hit
    profile: tuple = ()                  # predicted own (t_s, altitude_m, tas_mps, path_deg) each second to launch
    base_s: float = 0.                   # the time since spawn the plan was computed from
    shots: int = 0
    wall_s: float = 0.
    notes: tuple = ()


MISSILE_SIM_ENV = "WT_MISSILE_SIM"


def missile_sim_root():
    """Where missile_sim is: $WT_MISSILE_SIM, else the in-repo submodule missile_sim/ (git submodule update
    --init), else next to WT_overlay as missle_sim (the local folder name) or missile_sim (the GitHub repository
    name). It must be a version whose public API exports the stepping runtime (create_surface_missile): the
    submodule's branch wt-overlay-runtime does, the older GitHub branches main and dev do not."""
    from pathlib import Path
    here = Path(__file__).resolve()
    candidates = [Path(os.environ[MISSILE_SIM_ENV])] if os.environ.get(MISSILE_SIM_ENV) else []
    candidates += [here.parents[1]/"missile_sim", here.parents[2]/"missle_sim", here.parents[2]/"missile_sim"]
    old = []
    for root in candidates:
        api = root/"src"/"aim120_model"/"public_api.py"
        if api.is_file():
            if "create_surface_missile" in api.read_text(encoding="utf-8", errors="ignore"):
                return root
            old.append(root)
    if old:
        raise FileNotFoundError(f"{old[0]} 里的 missile_sim 版本太旧（没有步进接口 create_surface_missile）；"
                                "在 WT_overlay 目录运行 git submodule update --init missile_sim 用仓库内的版本")
    raise FileNotFoundError("没找到 missile_sim：在 WT_overlay 目录运行 git submodule update --init missile_sim"
                            f"（start_windows.cmd 会自动做），或用环境变量 {MISSILE_SIM_ENV} 指定"
                            f"（找过：{'、'.join(str(c) for c in candidates)}）")


def _library():
    from .engagement import default_library
    return default_library(missile_sim_root())


def _launch_options(library, missile):
    """engagement.LAUNCH_OPTIONS, without the active-radar-only CW option for a semi-active (SARH) missile:
    missile_sim refuses cw_on_clear_beam on SARH profiles. Launcher support (illumination) stays always on."""
    from .engagement import LAUNCH_OPTIONS
    if library.is_active(library.info(missile).profile_id):
        return LAUNCH_OPTIONS
    return {k: v for k, v in LAUNCH_OPTIONS.items() if k != "cw_on_clear_beam"}


def _r(x, step):
    return round(x/step)*step


def _missile_mass(library, missile):
    try:
        mass = (library.profile(missile).get("geometry") or {}).get("initial_mass_kg")
    except KeyError:
        return FALLBACK_MISSILE_MASS_KG
    return float(mass) if valid_number(mass) and mass > 0 else FALLBACK_MISSILE_MASS_KG


@lru_cache(maxsize=4)
def _enemy_track(aircraft, fuel_fraction, n_missiles, missile, climb_deg, level_m, distance_m, altitude_m, tas_mps):
    """Per-tick (position, velocity) of the enemy from its spawn: (D, 0, h0) flying -x, full power."""
    from .flight import Aircraft, FlightCommand, aircraft_model, fuel_data
    fuel = fuel_data(aircraft)
    mass = fuel.empty_kg+fuel_fraction*fuel.internal_kg+n_missiles*_missile_mass(_library(), missile)
    plane = Aircraft(aircraft_model(aircraft, mass_kg=mass), (distance_m, 0., altitude_m), (-tas_mps, 0., 0.),
                     keep_history=True)
    command = FlightCommand(heading_deg=270., altitude_m=level_m, max_climb_deg=climb_deg, floor_m=100.)
    while plane.time < ENEMY_TRACK_S:
        plane.step(command)
    return tuple((s.position, s.velocity) for s in plane.states)


def _enemy(request):
    s = request.settings
    return _enemy_track(s.enemy_aircraft, round(s.enemy_fuel_fraction, 3), s.enemy_missiles, s.enemy_missile,
                        s.enemy_climb_deg, s.enemy_level_m, s.distance_m, _r(request.spawn_altitude_m, 10.),
                        _r(request.spawn_tas_mps, 1.))


@lru_cache(maxsize=16)
def _own_track(aircraft, mass, t0, x0, h0, v0, path0, climb_deg, t_end):
    """Per-tick (position, velocity, normal) from the request state under a full-power climb at climb_deg."""
    from .flight import Aircraft, FlightCommand, aircraft_model
    g = math.radians(path0)
    plane = Aircraft(aircraft_model(aircraft, mass_kg=mass), (x0, 0., h0), (v0*math.cos(g), 0., v0*math.sin(g)),
                     t0=t0, keep_history=True)
    # Never a descent: from above the cap (a manual 'now' high up) the climb holds the current altitude.
    command = FlightCommand(heading_deg=90., altitude_m=max(CLIMB_CAP_M, h0), max_climb_deg=climb_deg, floor_m=100.)
    while plane.time < t_end:
        plane.step(command)
    return tuple((s.position, s.velocity, s.normal) for s in plane.states)


def _own(request, climb_deg):
    t_end = max(request.t_s+5., LATEST_PULL_S)+PULL_MAX_S+TRACK_TAIL_S
    return _own_track(request.aircraft_id, _r(request.mass_kg, 10.), round(request.t_s, 2), _r(request.x_m, 5.),
                      _r(request.altitude_m, 5.), _r(request.tas_mps, .5), round(request.path_deg, 1),
                      climb_deg, t_end)


def _at(track, t0, t):
    from .flight import SUBSTEP_S
    return track[max(0, min(len(track)-1, int(round((t-t0)/SUBSTEP_S))))]


class _EnemyTarget:
    """missile_sim target: the enemy track from launch time t_l on, in the runtime frame."""

    def __init__(self, track, t_l):
        self.track, self.t_l = track, t_l

    def position_velocity(self, t):
        from .flight import SUBSTEP_S
        x = (self.t_l+t)/SUBSTEP_S
        i = max(0, min(int(x), len(self.track)-2))
        f = min(1., x-i)
        (p0, v0), (p1, v1) = self.track[i], self.track[i+1]
        return (tuple(a+f*(b-a) for a, b in zip(p0, p1)), tuple(a+f*(b-a) for a, b in zip(v0, v1)))

    def state_at(self, t):
        from aim120_model.target import TargetState
        from .escape import from_enu
        p, v = self.position_velocity(t)
        return TargetState(from_enu(p), from_enu(v))


class _Support:
    """Launcher support always on (grade D, as scripts/build_reach_table.py)."""

    def __call__(self, t, truth):
        return ''


def _pull_up(aircraft, mass, state, pitch):
    """Fly from a stored state to the launch pitch (at most PULL_MAX_S). (seconds, position, velocity)."""
    from .flight import Aircraft, FlightCommand, aircraft_model
    position, velocity, normal = state
    speed = math.sqrt(sum(v*v for v in velocity))
    path = math.degrees(math.asin(max(-1., min(1., velocity[2]/speed))))
    if pitch is None or pitch <= path+.5:
        return 0., position, velocity
    plane = Aircraft(aircraft_model(aircraft, mass_kg=mass), position, velocity, normal=normal)
    command = FlightCommand(heading_deg=90., climb_deg=pitch, max_climb_deg=max(30., pitch), floor_m=100.)
    while plane.time < PULL_MAX_S:
        plane.step(command)
        v = plane.state.velocity
        if math.degrees(math.asin(max(-1., min(1., v[2]/math.sqrt(sum(x*x for x in v)))))) >= pitch-1.:
            break
    return plane.time, plane.state.position, plane.state.velocity


def shot(job):
    """One candidate: (request, climb_deg, pull_s, pitch). pitch 'auto' = loft_rule at the pull-up range.
    Runs in a worker process."""
    from .escape import from_enu
    request, climb_deg, pull_s, pitch = job
    own = _own(request, climb_deg)
    enemy = _enemy(request)
    state = _at(own, request.t_s, pull_s)
    if pitch == "auto":
        enemy_position = _at(enemy, 0., pull_s)[0]
        pitch = loft_rule(math.dist(state[0], enemy_position))
    pull_time, position, velocity = _pull_up(request.aircraft_id, _r(request.mass_kg, 10.), state, pitch)
    t_l = pull_s+pull_time
    speed = math.sqrt(sum(v*v for v in velocity))
    launch_path = math.degrees(math.asin(max(-1., min(1., velocity[2]/speed))))
    library = _library()
    target = _EnemyTarget(enemy, t_l)
    runtime = library.create(library.profile(request.missile), launch_position_m=from_enu(position),
                             launch_velocity_mps=from_enu(velocity), launch_heading_deg=0.,
                             launch_pitch_deg=launch_path, target=target, launcher_support=_Support(),
                             **_launch_options(library, request.missile))
    while not runtime.done:
        runtime.step()
        if len(runtime._rows) > 96:
            del runtime._rows[:-1]
    target_position = target.position_velocity(0.)[0]
    return dict(climb_deg=climb_deg, pull_s=pull_s, pitch_deg=None if pull_time == 0. else round(launch_path, 1),
                launch_s=t_l, altitude_m=position[2], tas_mps=speed, path_deg=launch_path,
                range_m=math.dist(position, target_position), flight_s=runtime.time_s,
                hit=runtime.event == "fuse", event=runtime.event, hit_s=t_l+runtime.time_s,
                target_altitude_m=target.position_velocity(runtime.time_s)[0][2])


def trace_shot(job):
    """The chosen shot again, recorded for a side view: (t, x, altitude) of the own aircraft every second to the
    launch, of the missile every 0.25 s to its end, and of the enemy every second to the end. x is the distance
    along the spawn line from our spawn (the enemy spawns at distance_m). Runs in a worker process."""
    from .escape import from_enu
    from .flight import SUBSTEP_S
    request, climb_deg, pull_s, pitch = job
    own = _own(request, climb_deg)
    enemy = _enemy(request)
    state = _at(own, request.t_s, pull_s)
    pull_time, position, velocity = _pull_up(request.aircraft_id, _r(request.mass_kg, 10.), state, pitch)
    t_l = pull_s+pull_time
    own_path = []
    t = math.ceil(request.t_s)
    while t < pull_s:
        p = _at(own, request.t_s, t)[0]
        own_path.append((float(t), p[0], p[2]))
        t += 1.
    p = state[0]
    own_path += [(pull_s, p[0], p[2]), (t_l, position[0], position[2])]
    speed = math.sqrt(sum(v*v for v in velocity))
    launch_path = math.degrees(math.asin(max(-1., min(1., velocity[2]/speed))))
    library = _library()
    target = _EnemyTarget(enemy, t_l)
    runtime = library.create(library.profile(request.missile), launch_position_m=from_enu(position),
                             launch_velocity_mps=from_enu(velocity), launch_heading_deg=0.,
                             launch_pitch_deg=launch_path, target=target, launcher_support=_Support(),
                             **_launch_options(library, request.missile))
    missile_path, every, step = [(t_l, position[0], position[2])], max(1, round(.25/SUBSTEP_S)), 0
    while not runtime.done:
        runtime.step()
        step += 1
        if step % every == 0:
            missile_path.append((t_l+runtime.time_s, runtime.state[0], runtime.state[1]))
        if len(runtime._rows) > 96:
            del runtime._rows[:-1]
    end = t_l+runtime.time_s
    missile_path.append((end, runtime.state[0], runtime.state[1]))
    enemy_path, t = [], 0.
    while t <= end:
        p = _at(enemy, 0., t)[0]
        enemy_path.append((t, p[0], p[2]))
        t += 1.
    p = target.position_velocity(runtime.time_s)[0]
    enemy_path.append((end, p[0], p[2]))
    return dict(own=own_path, missile=missile_path, enemy=enemy_path, launch=(t_l, position[0], position[2]),
                end=(end, runtime.state[0], runtime.state[1]), hit=runtime.event == "fuse", event=runtime.event,
                launch_speed_mps=speed, launch_path_deg=launch_path)


def own_profile(job):
    """Predicted own (t_s, altitude, TAS, path) each second from the request time to ``until_s``."""
    request, climb_deg, until_s = job
    own = _own(request, climb_deg)
    out, t = [], math.ceil(request.t_s)
    while t <= until_s:
        position, velocity, _ = _at(own, request.t_s, t)
        speed = math.sqrt(sum(v*v for v in velocity))
        out.append((float(t), position[2], speed, math.degrees(math.asin(max(-1., min(1., velocity[2]/speed))))))
        t += 1.
    return tuple(out)


def worker_init():
    """One worker at normal priority: below normal it could be starved by the game and never finish (a plan is
    computed once per new aircraft / start state and then cached)."""
    return None


class Cancelled(Exception):
    pass


def _times(request):
    """Stage-1 pull-up times: the fixed grid (or around the hint), never earlier than 2 s from now."""
    earliest = request.t_s+2.
    if request.hint is not None:
        center = request.hint[1]
        grid = [center+d for d in (-10., -5., 0., 5., 10.)]
    else:
        grid = list(STAGE1_TIMES_S)
    times = sorted({round(max(earliest, t), 2) for t in grid if t <= LATEST_PULL_S})
    k = 1
    while len(times) < 3 and earliest+5.*k <= LATEST_PULL_S:
        times = sorted(set(times) | {round(earliest+5.*k, 2)})
        k += 1
    return times or [round(earliest, 2)]


def _choose(hits, hint):
    """The earliest hit; a re-plan keeps the previous climb and pull-up time unless switching gains enough
    (near-ties would otherwise move the arrow for nothing)."""
    def score(r):
        value = r["hit_s"]
        if hint is not None:
            value += SWITCH_CLIMB_S*(r["climb_deg"] != hint[0])+SWITCH_PULL_S*(abs(r["pull_s"]-hint[1]) > .01)
        return value
    return min(hits, key=score)


def _window(done, best):
    """Hits with the chosen climb no more than WINDOW_TOLERANCE_S later than that climb's earliest."""
    climb = [r for r in done.values() if r["hit"] and r["climb_deg"] == best["climb_deg"]]
    first = min(r["hit_s"] for r in climb)
    return [r for r in climb if r["hit_s"] <= first+WINDOW_TOLERANCE_S]


def plan_opening(request: PlanRequest, run=None, cancel=None) -> OpeningPlan:
    """Coarse-to-fine search for the earliest hit. ``run(function, jobs)`` maps the jobs (a process pool's map;
    default: in this process). ``cancel``: an Event checked between stages."""
    started = time.perf_counter()
    run = run or (lambda function, jobs: [function(j) for j in jobs])
    done = {}

    def evaluate(jobs):
        jobs = list(dict.fromkeys(j for j in jobs if j not in done))
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        for job, result in zip(jobs, run(shot, [(request, *j) for j in jobs])):
            done[job] = result

    climbs = (request.hint[0],) if request.hint is not None else STAGE1_CLIMBS_DEG
    evaluate([(c, t, "auto") for c in climbs for t in _times(request)])
    hits = [r for r in done.values() if r["hit"]]
    if not hits:
        evaluate([(c, t, p) for c in climbs for t in _times(request) for p in (45., 55.)])
        hits = [r for r in done.values() if r["hit"]]
    if not hits:
        return OpeningPlan(False, "按现在的状态，到 85 s 为止哪个时刻发射都打不到（导弹射程不够）",
                           base_s=request.t_s, shots=len(done), wall_s=time.perf_counter()-started)
    best = min(hits, key=lambda r: r["hit_s"])
    climb, pull = best["climb_deg"], best["pull_s"]
    rule = loft_rule(best["range_m"])
    pitches = {None, *(p for p in ((rule or best["path_deg"])-10., (rule or best["path_deg"])+10., rule)
                       if p is not None and 0. < p <= 60.)}
    refine = [(climb, round(t, 2), p) for t in (pull-5., pull, pull+5.) if request.t_s+2. <= t <= LATEST_PULL_S
              for p in pitches]
    refine += [(c, pull, "auto") for c in (climb-5., climb+5.) if CLIMBS_DEG[0] <= c <= CLIMBS_DEG[1]]
    evaluate(refine)
    best = _choose([r for r in done.values() if r["hit"]], request.hint)
    same = _window(done, best)
    if len({round(r["launch_s"], 1) for r in same}) < 2:
        evaluate([(best["climb_deg"], round(best["pull_s"]+d, 2), "auto") for d in (-10., -5., 5., 10.)
                  if request.t_s+2. <= best["pull_s"]+d <= LATEST_PULL_S])
        same = _window(done, best)
    if cancel is not None and cancel.is_set():
        raise Cancelled()
    profile = run(own_profile, [(request, best["climb_deg"], best["launch_s"])])[0]
    return OpeningPlan(
        True, climb_deg=best["climb_deg"], pull_s=best["pull_s"], pitch_deg=best["pitch_deg"],
        launch_s=best["launch_s"], window_s=(min(r["launch_s"] for r in same), max(r["launch_s"] for r in same)),
        launch_range_m=best["range_m"], hit_s=best["hit_s"], flight_s=best["flight_s"],
        launch_altitude_m=best["altitude_m"], launch_tas_mps=best["tas_mps"],
        target_altitude_m=best["target_altitude_m"], profile=profile, base_s=request.t_s, shots=len(done),
        wall_s=time.perf_counter()-started)


# -- the cue -------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class OpeningCue:
    phase: str                               # 等待出生 / 计算中 / 爬升 / 拉起 / 发射 / 结束 / 无方案 / 无数据
    t_s: float | None = None                 # time since spawn
    target_path_deg: float | None = None
    yaw_error_deg: float | None = None       # + = turn right
    path_error_deg: float | None = None      # + = pull up
    direction: tuple | None = None           # wanted direction in the velocity frame (right, up, forward), unit
    on_course: bool = False
    urgent: bool = False                     # the firing cue
    lines: tuple = ()
    launch_range_m: float | None = None      # for the B-scope line
    plan: OpeningPlan | None = field(default=None, repr=False)

    @property
    def active(self):
        return self.phase in ("计算中", "爬升", "拉起", "发射")


def _unit(e, n, u):
    norm = math.sqrt(e*e+n*n+u*u)
    return e/norm, n/norm, u/norm


def _world_direction(heading_deg, path_deg):
    h, g = math.radians(heading_deg), math.radians(path_deg)
    return math.sin(h)*math.cos(g), math.cos(h)*math.cos(g), math.sin(g)


def velocity_frame_direction(heading_deg, path_deg, wanted_heading_deg, wanted_path_deg, gain=ARROW_GAIN):
    """The wanted direction in the current velocity frame (right, up, forward), with the across-track parts scaled
    by ``gain`` before normalising (small errors drawn larger; 90 deg stays 90 deg). Horizon-stabilised: 'up' is
    the world vertical with the forward part removed."""
    f = _world_direction(heading_deg, path_deg)
    up = (0., 0., 1.)
    dot = sum(a*b for a, b in zip(up, f))
    u = _unit(*(a-dot*b for a, b in zip(up, f)))
    r = (f[1]*u[2]-f[2]*u[1], f[2]*u[0]-f[0]*u[2], f[0]*u[1]-f[1]*u[0])
    w = _world_direction(wanted_heading_deg, wanted_path_deg)
    x, y, z = (sum(a*b for a, b in zip(w, axis)) for axis in (r, u, f))
    x, y = gain*x, gain*y
    norm = math.sqrt(x*x+y*y+z*z) or 1.
    return x/norm, y/norm, z/norm


def cue(plan: OpeningPlan | None, spawn: Spawn | None, state, note="") -> OpeningCue:
    """Pitch arrow and two text lines for the current sample (no heading: the pilot keeps the spawn heading).
    Before the plan arrives the climb is the default DEFAULT_CLIMB_DEG; the launch line only appears with a plan.
    Countdowns are whole seconds so the text changes at most once a second."""
    if spawn is None:
        return OpeningCue("等待开始", lines=(note or "开局抽奖 · 加速或爬升时开始",))
    t = state.time_s-spawn.time_s
    usable = plan is not None and plan.available
    if usable and t > plan.window_s[1]+END_AFTER_WINDOW_S:
        return OpeningCue("结束", t, plan=plan)
    path = path_angle_deg(state) if state.valid else None
    now = "" if path is None else f"（现在 {path:.0f}°）"
    if not usable or t < plan.pull_s:
        phase = "爬升"
        wanted = plan.climb_deg if usable else DEFAULT_CLIMB_DEG
        lines = [f"全加力 爬升 {wanted:.0f}°{now}"]
        if usable:
            wait = math.ceil(plan.pull_s-t)
            launch = (f"{wait} s 后拉起到 {plan.pitch_deg:.0f}° 发射" if plan.pitch_deg is not None else
                      f"{math.ceil(plan.launch_s-t)} s 后发射")
            lines.append(f"{launch} · 目标约 {plan.launch_range_m/1000:.0f} km")
        elif plan is not None:
            lines.append(f"发射时机：{plan.reason}")
    elif t < plan.launch_s:
        phase = "拉起"
        wanted = plan.pitch_deg if plan.pitch_deg is not None else plan.climb_deg
        lines = [f"拉起到 {wanted:.0f}°{now}", f"{math.ceil(plan.launch_s-t)} s 后发射"]
    else:
        phase = "发射"
        wanted = plan.pitch_deg if plan.pitch_deg is not None else plan.climb_deg
        lines = ["发射！", f"目标约 {plan.launch_range_m/1000:.0f} km · 预计 T+{plan.hit_s:.0f} s 命中"]
    if (phase == "爬升" and state.valid and valid_number(state.altitude_m)
            and state.altitude_m >= max(CLIMB_CAP_M, spawn.altitude_m)-200.):
        wanted = 0.
    if path is None or not valid_number(state.tas_mps) or state.tas_mps < SLOW_MPS:
        lines.append("速度太低（在地面？）" if valid_number(state.tas_mps) and state.tas_mps < SLOW_MPS
                     else "缺少垂直速度")
        return OpeningCue(phase, t, wanted, lines=tuple(lines), plan=plan,
                          launch_range_m=plan.launch_range_m if usable else None)
    pitch_error = wanted-path
    return OpeningCue(phase, t, wanted, None, pitch_error, velocity_frame_direction(0., path, 0., wanted),
                      on_course=abs(pitch_error) < CUE_DEADBAND_DEG, urgent=phase == "发射", lines=tuple(lines),
                      plan=plan, launch_range_m=plan.launch_range_m if usable else None)


class PlanCache:
    """Plans already computed, kept on disk: the same aircraft, missile, scenario and a nearby start state
    (altitude, TAS, mass bins) needs no new computation. ``nearest`` gives the closest plan of the same aircraft,
    missile and scenario to fly at once while the exact one is computed."""
    ALTITUDE_BIN_M, TAS_BIN_MPS, MASS_BIN_KG = 500., 20., 1000.

    def __init__(self, path=None):
        self.path = path
        self.plans = {}
        if path is not None:
            try:
                with open(path, encoding="utf-8") as f:
                    for key, value in json.load(f).items():
                        value["window_s"] = tuple(value["window_s"]) if value.get("window_s") else None
                        self.plans[key] = OpeningPlan(**value)
            except (OSError, ValueError, TypeError):
                self.plans = {}

    @staticmethod
    def scenario(request):
        s = request.settings
        return "|".join(str(x) for x in (request.aircraft_id, request.missile, s.distance_m, s.enemy_aircraft,
                                         s.enemy_fuel_fraction, s.enemy_missiles, s.enemy_climb_deg, s.enemy_level_m))

    def key(self, request):
        bins = tuple(math.floor(value/size+.5) for value, size in (
            (request.altitude_m, self.ALTITUDE_BIN_M), (request.tas_mps, self.TAS_BIN_MPS),
            (request.mass_kg, self.MASS_BIN_KG)))
        return self.scenario(request)+"|"+",".join(map(str, bins))

    def get(self, request):
        return self.plans.get(self.key(request))

    def nearest(self, request):
        prefix = self.scenario(request)+"|"
        target = (request.altitude_m/self.ALTITUDE_BIN_M, request.tas_mps/self.TAS_BIN_MPS,
                  request.mass_kg/self.MASS_BIN_KG)
        best, distance = None, math.inf
        for key, plan in self.plans.items():
            if not key.startswith(prefix) or not plan.available:
                continue
            bins = tuple(float(x) for x in key[len(prefix):].split(","))
            d = sum((a-b)**2 for a, b in zip(bins, target))
            if d < distance:
                best, distance = plan, d
        return best

    def put(self, request, plan):
        if not plan.available:
            return
        self.plans[self.key(request)] = replace(plan, profile=())
        if self.path is None:
            return
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            temporary = self.path+".tmp"
            with open(temporary, "w", encoding="utf-8") as f:
                json.dump({k: asdict(v) for k, v in self.plans.items()}, f, ensure_ascii=False)
            os.replace(temporary, self.path)
        except OSError:
            pass


def default_cache_path():
    base = os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "WT Energy", "opening_plans.json")


def deviates(plan: OpeningPlan, state, t_s) -> bool:
    """Is the flown state off the plan's prediction (path angle or TAS)?"""
    if not plan.available or not plan.profile:
        return False
    row = min(plan.profile, key=lambda r: abs(r[0]-t_s))
    if abs(row[0]-t_s) > 1.:
        return False
    path = path_angle_deg(state)
    return ((path is not None and abs(path-row[3]) > DEVIATION_PATH_DEG)
            or abs(state.tas_mps-row[2]) > DEVIATION_TAS_MPS)


def replace_settings(settings: OpeningSettings, **values) -> OpeningSettings:
    updated = replace(settings, **values)
    validate_settings(updated)
    return updated
