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

from dataclasses import dataclass, field, replace
from functools import lru_cache
import math
import os
import sys
import time

from .climb import CUE_DEADBAND_DEG, valid_number

# Air spawn (replays: about 2.5 km and 290 m/s, level): the first valid sample after a gap must look like one.
SPAWN_ALTITUDE_M = (1000., 6000.)
SPAWN_TAS_MPS = (150., 400.)
SPAWN_MAX_VY_MPS = 20.
SPAWN_GAP_S = 3.          # a shorter telemetry gap is a lost poll, not a respawn
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
    workers: int = 2


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
    time_s: float           # telemetry clock at the spawn sample
    heading_deg: float      # = the bearing of the enemy spawn (replays: median 3 deg off)
    altitude_m: float
    tas_mps: float
    aircraft_id: str
    manual: bool = False


def path_angle_deg(state) -> float | None:
    if not (valid_number(state.tas_mps) and state.tas_mps > 1 and valid_number(state.vertical_speed_mps)):
        return None
    return math.degrees(math.asin(max(-1., min(1., state.vertical_speed_mps/state.tas_mps))))


class SpawnLatch:
    """Detects an air spawn: the first valid sample after a telemetry gap of SPAWN_GAP_S (or an aircraft change),
    if it looks like a spawn (altitude, TAS, nearly level, a heading). ``latch`` takes the current state instead
    (manual restart, e.g. when the overlay started mid-flight)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.spawn: Spawn | None = None
        self.note = "等待空中出生"
        self._last_valid_s = None
        self._identity = None

    @staticmethod
    def _usable(state):
        return (state.valid and valid_number(state.altitude_m) and valid_number(state.tas_mps)
                and valid_number(state.heading_deg) and bool(state.aircraft_id))

    def latch(self, state, manual=True) -> Spawn | None:
        if not self._usable(state):
            self.note = "没有可用的飞行数据，无法当作出生"
            return None
        self.spawn = Spawn(state.time_s, float(state.heading_deg) % 360., float(state.altitude_m),
                           float(state.tas_mps), state.aircraft_id, manual)
        self.note = ""
        return self.spawn

    def update(self, state) -> Spawn | None:
        """Returns the new Spawn on the sample where one is detected, else None."""
        if not self._usable(state):
            return None
        from .fm.catalog import aircraft_key
        identity = aircraft_key(state.aircraft_id)
        fresh = (self._last_valid_s is None or state.time_s-self._last_valid_s > SPAWN_GAP_S
                 or identity != self._identity)
        self._last_valid_s, self._identity = state.time_s, identity
        if not fresh:
            return None
        vy = state.vertical_speed_mps
        if (SPAWN_ALTITUDE_M[0] <= state.altitude_m <= SPAWN_ALTITUDE_M[1]
                and SPAWN_TAS_MPS[0] <= state.tas_mps <= SPAWN_TAS_MPS[1]
                and (vy is None or abs(vy) <= SPAWN_MAX_VY_MPS)):
            return self.latch(state, manual=False)
        self.spawn = None
        self.note = (f"不像空中出生（{state.altitude_m:.0f} m，{state.tas_mps*3.6:.0f} km/h）；"
                     "可在设置里“以现在为出生”")
        return None


class DeadReckoning:
    """Own position along (x) and across (y) the spawn line, integrated from TAS, Vy and compass."""

    def __init__(self, spawn: Spawn):
        self.heading = spawn.heading_deg
        self.x = self.y = 0.
        self.last_s = spawn.time_s

    def update(self, state):
        if not (state.valid and valid_number(state.tas_mps) and valid_number(state.heading_deg)):
            return
        dt = state.time_s-self.last_s
        self.last_s = state.time_s
        if not 0 < dt < 2.:
            return
        path = path_angle_deg(state) or 0.
        horizontal = state.tas_mps*math.cos(math.radians(path))
        off = math.radians(wrap(state.heading_deg-self.heading))
        self.x += horizontal*math.cos(off)*dt
        self.y += horizontal*math.sin(off)*dt


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
    from .engagement import default_library
    from .flight import Aircraft, FlightCommand, aircraft_model, fuel_data
    fuel = fuel_data(aircraft)
    mass = fuel.empty_kg+fuel_fraction*fuel.internal_kg+n_missiles*_missile_mass(default_library(), missile)
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
    command = FlightCommand(heading_deg=90., altitude_m=CLIMB_CAP_M, max_climb_deg=climb_deg, floor_m=100.)
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
    from .engagement import LAUNCH_OPTIONS, default_library
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
    library = default_library()
    target = _EnemyTarget(enemy, t_l)
    runtime = library.create(library.profile(request.missile), launch_position_m=from_enu(position),
                             launch_velocity_mps=from_enu(velocity), launch_heading_deg=0.,
                             launch_pitch_deg=launch_path, target=target, launcher_support=_Support(),
                             **LAUNCH_OPTIONS)
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
    """Worker processes run below normal priority so they do not take frames from the game."""
    try:
        if sys.platform == "win32":
            import ctypes
            ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
        else:
            os.nice(10)
    except (OSError, AttributeError):
        pass


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
    """Arrow and text for the current sample. Without a plan yet the arrow shows the default climb."""
    if spawn is None:
        return OpeningCue("等待出生", lines=(note or "开局抽奖 · 等待空中出生",))
    t = state.time_s-spawn.time_s
    if plan is not None and not plan.available:
        return OpeningCue("无方案", t, lines=(f"开局抽奖：{plan.reason}",), plan=plan)
    if plan is not None and t > plan.window_s[1]+END_AFTER_WINDOW_S:
        return OpeningCue("结束", t, plan=plan)
    if plan is None:
        phase, wanted = "计算中", DEFAULT_CLIMB_DEG
        lines = (f"全加力 爬升 {DEFAULT_CLIMB_DEG:.0f}° · 计算中…",)
    elif t < plan.pull_s:
        phase, wanted = "爬升", plan.climb_deg
        first = f"全加力 爬升 {plan.climb_deg:.0f}°"
        first += (f" · {plan.pull_s-t:.0f} s 后拉起到 {plan.pitch_deg:.0f}°" if plan.pitch_deg is not None
                  else f" · {plan.launch_s-t:.0f} s 后发射")
        lines = (first, f"发射时目标约 {plan.launch_range_m/1000:.0f} km · 预计 T+{plan.hit_s:.0f} s 命中")
    elif t < plan.launch_s:
        phase = "拉起"
        wanted = plan.pitch_deg if plan.pitch_deg is not None else plan.climb_deg
        lines = (f"拉起到 {wanted:.0f}° · {plan.launch_s-t:.1f} s 后发射",
                 f"目标约 {plan.launch_range_m/1000:.0f} km")
    else:
        phase = "发射"
        wanted = plan.pitch_deg if plan.pitch_deg is not None else plan.climb_deg
        lines = ("发射！", f"窗口到 T+{plan.window_s[1]:.0f} s · 目标约 {plan.launch_range_m/1000:.0f} km"
                           f" · 预计 T+{plan.hit_s:.0f} s 命中")
    if state.valid and state.altitude_m is not None and state.altitude_m >= CLIMB_CAP_M-200. and phase == "爬升":
        wanted = 0.
    path = path_angle_deg(state) if state.valid else None
    heading = state.heading_deg if state.valid and valid_number(state.heading_deg) else None
    if path is None or heading is None:
        return OpeningCue(phase, t, wanted, lines=(*lines, "缺少航向或垂直速度"), plan=plan,
                          launch_range_m=plan.launch_range_m if plan else None)
    yaw, pitch_error = wrap(spawn.heading_deg-heading), wanted-path
    direction = velocity_frame_direction(heading, path, spawn.heading_deg, wanted)
    return OpeningCue(phase, t, wanted, yaw, pitch_error, direction,
                      on_course=math.hypot(yaw*math.cos(math.radians(path)), pitch_error) < CUE_DEADBAND_DEG,
                      urgent=phase == "发射", lines=lines, plan=plan,
                      launch_range_m=plan.launch_range_m if plan else None)


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
