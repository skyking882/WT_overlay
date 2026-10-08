"""War Thunder (and sandbox) replays in the sandbox: playback as recorded, and forks run by our models.

``ReplayTrack`` reads a replay JSONL (header / frame / event / end; docs/sandbox_replay_spec.md, the format of
scripts/wt_replay_import.py and of the sandbox's own replay.jsonl) into per-unit time series. Positions between
frames are cubic Hermite in position and velocity, velocities linear; a hole longer than ``MAX_GAP_S`` or a time
outside a unit's first / last frame has no state (playback hides the unit). Pitch comes from the velocity, roll is
an estimate: the coordinated-turn bank for the velocity heading's rate of change, ``atan(V w / g)``, limited to
+-85 deg (``attitude_source = "estimated"``).

A fork (``build_fork``) is a normal sandbox world started at ``t0`` = the earlier of ``t_fork`` and the launch of the
oldest modelled missile still live (before its closest approach) at ``t_fork``. Every aircraft starts pinned to its replay track
(``TrackedAircraft`` writes the interpolated state through ``Aircraft._commit``, so ``state_at``, the ring buffer and
``min_altitude_m`` work as for a flying aircraft); the replay's launches of modelled missiles are fired again by
``Engagement.fire`` at their recorded times, recorded chaff releases replayed as ``Action(chaff=1)`` and recorded deaths
applied to the aircraft still pinned. At ``t_fork`` the chosen aircraft are released: the same flight object then
integrates the flight model from the pinned state and a ``SandboxPilot`` (auto / manual / pilot) flies it.
Nothing here changes ``flight.py`` or ``engagement.py``; the fork world is a ``SandboxSimulation`` subclass.
"""
from __future__ import annotations

from bisect import bisect_right
import hashlib
import json
import math
from pathlib import Path

from .archetypes import bearing_of
from .engagement import Action, PlaneSpec, RadarCommand
from .flight import Aircraft, FlightParams, FlightState, MIN_STEP_SPEED_MPS, SUBSTEP_S
from .fm import atmosphere
from .match import scenario as make_match, modelled_missiles
from .rl_env import equip_scripts
from .sandbox import SandboxPilot, SandboxSimulation, analysis_snapshot, set_order

G = 9.80665
MAX_GAP_S = 2.            # frames further apart than this: a hole (no state shown)
ROLL_LIMIT_DEG = 85.
RATE_WINDOW_S = 1.        # heading rate for the roll estimate: central difference over +-this
WRECK_S = 10.             # a dead unit's wreck stays on the map this long
TRACK_AOA_DEG = 3.        # pinned aircraft: constant angle of attack (D) ...
TRACK_ENGINE_PERCENT = 100.   # ... and engine setting (D); neither is in the replay
PARKED_MPS = 30.          # a pinned aircraft slower than this is on the ground in the replay (landed / taxiing / parked)
REPLAY_MAX_TEAM = 24      # fork worlds: per team (the editor's MAX_TEAM stays 16)
GHOST_S, GHOST_STEP_S = 120., 1.
SURROGATE_FM = "su_30sm2" # flight model carried (never flown) by pinned units without one of their own (AI units)
PLAYBACK_EVENTS = ("launch", "missile_end", "kill", "death")
CONTROL_MODES = ("track", "auto", "manual", "pilot")
EPS = 1e-9


def base_missile(mid):
    """Replay missile id -> library id: the ``_default`` suffix removed."""
    return mid[:-len("_default")] if isinstance(mid, str) and mid.endswith("_default") else mid


def missile_modelled(library, mid):
    """Our missile model can fly it: a missile_sim profile with an active radar seeker."""
    if not isinstance(mid, str):
        return False
    try:
        return bool(library.is_active(library.info(mid).profile_id))
    except (KeyError, ValueError):
        return False


def _heading(vx, vy):
    return math.degrees(math.atan2(vx, vy)) % 360.


def _lerp_angle(a, b, w):
    return (a+((b-a+540.) % 360.-180.)*w) % 360.


class _Series:
    """One unit's samples: times, positions, velocities, the raw row, and (planes) the estimated roll."""
    __slots__ = ("t", "p", "v", "rows", "roll")

    def __init__(self):
        self.t, self.p, self.v, self.rows, self.roll = [], [], [], [], None

    def add(self, t, p, v, row):
        if self.t and t <= self.t[-1]+EPS:   # a repeated frame time: keep the first
            return
        self.t.append(t)
        self.p.append(p)
        self.v.append(v)
        self.rows.append(row)

    def estimate_roll(self):
        ts, vs, n = self.t, self.v, len(self.t)
        heads = [_heading(v[0], v[1]) for v in vs]
        out = []
        for i in range(n):
            # widest window of up to +-RATE_WINDOW_S that stays inside this stretch of samples (no hole)
            a = i
            while a > 0 and ts[a]-ts[a-1] <= MAX_GAP_S and ts[i]-ts[a-1] <= RATE_WINDOW_S+EPS:
                a -= 1
            b = i
            while b < n-1 and ts[b+1]-ts[b] <= MAX_GAP_S and ts[b+1]-ts[i] <= RATE_WINDOW_S+EPS:
                b += 1
            v = vs[i]
            speed = math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2])
            if b == a or math.hypot(vs[a][0], vs[a][1]) < 1. or math.hypot(vs[b][0], vs[b][1]) < 1.:
                out.append(0.)
                continue
            rate = math.radians((heads[b]-heads[a]+540.) % 360.-180.)/(ts[b]-ts[a])   # clockwise, rad/s
            roll = math.degrees(math.atan(speed*rate/G))
            out.append(max(-ROLL_LIMIT_DEG, min(ROLL_LIMIT_DEG, roll)))
        self.roll = out

    def at(self, t, bridge=False, extrapolate=False):
        """(position, velocity, sample index at or before t, weight to the next one) or None. ``bridge``: interpolate
        across holes too; ``extrapolate``: past the last sample, straight on at the last velocity."""
        ts = self.t
        n = len(ts)
        if not n or t < ts[0]-EPS:
            return None
        if t > ts[-1]+EPS:
            if not extrapolate:
                return None
            p, v, dt = self.p[-1], self.v[-1], t-ts[-1]
            return (p[0]+v[0]*dt, p[1]+v[1]*dt, p[2]+v[2]*dt), v, n-1, 0.
        i = min(max(bisect_right(ts, t+EPS)-1, 0), n-1)
        if abs(t-ts[i]) <= EPS or i == n-1:
            return self.p[i], self.v[i], i, 0.
        h = ts[i+1]-ts[i]
        if h > MAX_GAP_S and not bridge:
            return None
        s = (t-ts[i])/h
        s2, s3 = s*s, s*s*s
        h00, h10, h01, h11 = 2*s3-3*s2+1, s3-2*s2+s, -2*s3+3*s2, s3-s2
        p0, p1, v0, v1 = self.p[i], self.p[i+1], self.v[i], self.v[i+1]
        pos = tuple(h00*p0[k]+h10*h*v0[k]+h01*p1[k]+h11*h*v1[k] for k in range(3))
        vel = tuple(v0[k]+s*(v1[k]-v0[k]) for k in range(3))
        return pos, vel, i, s


class ReplayTrack:
    """A whole replay in memory: per plane / missile series, events in time order, header."""
    def __init__(self, path, *, name=None, kind=None):
        self.path = Path(path)
        self.name = name or self.path.name
        self.header = None
        self.planes, self.missiles = {}, {}
        self.events, self.end = [], None
        self.first_t = self.last_t = None
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip() or '"missile_tick"' in line[:40]:
                    continue
                row = json.loads(line)
                kind_ = row.get("type")
                if kind_ == "header":
                    self._read_header(row)
                elif kind_ == "frame":
                    self._read_frame(row)
                elif kind_ == "event":
                    self.events.append(row)
                elif kind_ == "end":
                    self.end = row
        if self.header is None:
            raise ValueError("回放文件缺少 header")
        self.kind = kind or ("wt" if (self.header.get("source") or {}).get("kind") == "wt_replay" else "sandbox")
        self.events.sort(key=lambda e: e.get("t", 0.))
        for s in self.planes.values():
            s.estimate_roll()
        self.event_times = [e.get("t", 0.) for e in self.events]
        self.duration_s = max(x for x in (self.last_t, (self.end or {}).get("t")) if x is not None) \
            if (self.last_t is not None or self.end) else 0.
        self.start_s = self.first_t if self.first_t is not None else 0.
        self.deaths, self.launch_by_uid, self.end_by_uid = {}, {}, {}
        for e in self.events:
            k = e.get("kind")
            if k == "death" and e.get("plane") not in self.deaths:
                self.deaths[e.get("plane")] = e
            elif k == "launch":
                self.launch_by_uid.setdefault(e.get("uid"), e)
            elif k == "missile_end":
                self.end_by_uid.setdefault(e.get("uid"), e)
        self.info = {}
        for p in self.header.get("planes", []):
            wt = p.get("wt") or {}
            self.info[p["id"]] = dict(id=p["id"], team=p.get("team", 0), aircraft=p.get("aircraft"),
                                      player=wt.get("player") or p.get("player") or p.get("name") or f"#{p['id']}",
                                      ai=p.get("skill") == "AI", missile=p.get("missile"))
        self.map_half_m = float(self.header.get("map_half_m") or 64000.)
        self.markers = self._markers()

    # -- reading -------------------------------------------------------------------------------------------

    def _read_header(self, row):
        self.header = row
        self.pcol = {c: i for i, c in enumerate(row.get("plane_columns") or [])}
        self.mcol = {c: i for i, c in enumerate(row.get("missile_columns") or [])}

    def _read_frame(self, row):
        t = float(row["t"])
        self.first_t = t if self.first_t is None else min(self.first_t, t)
        self.last_t = t if self.last_t is None else max(self.last_t, t)
        pc, mc = self.pcol, self.mcol
        for r in row.get("planes") or ():
            s = self.planes.get(r[pc["id"]])
            if s is None:
                s = self.planes[r[pc["id"]]] = _Series()
            s.add(t, (float(r[pc["x"]]), float(r[pc["y"]]), float(r[pc["z"]])),
                  (float(r[pc["vx"]]), float(r[pc["vy"]]), float(r[pc["vz"]])), r)
        for r in row.get("missiles") or ():
            s = self.missiles.get(r[mc["uid"]])
            if s is None:
                s = self.missiles[r[mc["uid"]]] = _Series()
            s.add(t, (float(r[mc["x"]]), float(r[mc["y"]]), float(r[mc["z"]])),
                  (float(r[mc["vx"]]), float(r[mc["vy"]]), float(r[mc["vz"]])), r)

    def _markers(self):
        out = []
        for e in self.events:
            k = e.get("kind")
            if k == "launch":
                out.append(dict(t=e["t"], kind="launch", text=f"{self.label(e.get('shooter'))} 发射 "
                                f"{base_missile(e.get('missile'))} → {self.label(e.get('target'))}"))
            elif k == "kill":
                out.append(dict(t=e["t"], kind="kill", text=f"{self.label(e.get('killer'))} 击落 "
                                f"{self.label(e.get('victim'))}"))
            elif k == "death":
                out.append(dict(t=e["t"], kind="death", text=f"{self.label(e.get('plane'))} 损失 · {e.get('cause')}"))
        return out

    def label(self, ident):
        info = self.info.get(ident)
        return "?" if info is None else info["player"][:14]

    # -- queries -------------------------------------------------------------------------------------------

    def state_at(self, ident, t):
        """Plane ``ident`` at replay time ``t``: dict(position, velocity, heading_deg, pitch_deg, roll_deg, missiles,
        chaff, phase) or None outside its first / last frame and in holes longer than MAX_GAP_S."""
        s = self.planes.get(ident)
        got = None if s is None else s.at(t)
        if got is None:
            return None
        return self._plane_state(s, got)

    def _plane_state(self, s, got):
        pos, vel, i, w = got
        pc, row = self.pcol, s.rows[i]
        speed = math.sqrt(sum(v*v for v in vel))
        roll = s.roll[i] if w <= 0. or i+1 >= len(s.roll) else s.roll[i]+w*(s.roll[i+1]-s.roll[i])
        hcol = pc.get("heading_deg")
        if hcol is None:
            heading = _heading(vel[0], vel[1])
        elif w > 0. and i+1 < len(s.rows):
            heading = _lerp_angle(row[hcol], s.rows[i+1][hcol], w)
        else:
            heading = row[hcol]
        missiles = row[pc["missiles"]] if "missiles" in pc and len(row) > pc["missiles"] else None
        chaff = row[pc["chaff"]] if "chaff" in pc and len(row) > pc["chaff"] else None
        phase = row[pc["phase"]] if "phase" in pc and len(row) > pc["phase"] else ""
        return dict(position=pos, velocity=vel, speed=speed, heading_deg=float(heading) % 360.,
                    pitch_deg=math.degrees(math.asin(max(-1., min(1., vel[2]/speed)))) if speed > 1e-6 else 0.,
                    roll_deg=roll, missiles=missiles if isinstance(missiles, int) else None,
                    chaff=chaff if isinstance(chaff, int) and not isinstance(chaff, bool) else None, phase=phase or "")

    def kinematics(self, ident, t):
        """For a pinned aircraft: (position, velocity, roll_deg) at ``t``, interpolated across holes too and carried
        straight on past the last frame; None before the first frame."""
        s = self.planes.get(ident)
        got = None if s is None else s.at(t, bridge=True, extrapolate=True)
        if got is None:
            return None
        pos, vel, i, w = got
        if t > s.t[-1]+EPS:
            return pos, vel, 0.
        roll = s.roll[i] if w <= 0. or i+1 >= len(s.roll) else s.roll[i]+w*(s.roll[i+1]-s.roll[i])
        return pos, vel, roll

    def present(self, ident, t):
        """In the replay's air at ``t``: inside its frames (holes included) and not dead yet."""
        s = self.planes.get(ident)
        if s is None or not s.t or t < s.t[0]-EPS or t > s.t[-1]+EPS:
            return False
        death = self.deaths.get(ident)
        return death is None or death["t"] > t+EPS

    def last_before(self, ident, t):
        s = self.planes.get(ident)
        if s is None or not s.t or t < s.t[0]-EPS:
            return None
        i = min(max(bisect_right(s.t, t+EPS)-1, 0), len(s.t)-1)
        return self._plane_state(s, (s.p[i], s.v[i], i, 0.))

    def missile_at(self, uid, t):
        s = self.missiles.get(uid)
        got = None if s is None else s.at(t)
        if got is None:
            return None
        pos, vel, i, _ = got
        mc, row = self.mcol, s.rows[i]
        get = lambda c: row[mc[c]] if c in mc and len(row) > mc[c] else None  # noqa: E731
        return dict(position=pos, velocity=vel, shooter=get("owner"), target=get("target"),
                    age_s=t-s.t[0], seeker=bool(get("seeker")), datalink=bool(get("datalink")))

    def missile_end_t(self, uid):
        e = self.end_by_uid.get(uid)
        if e is not None:
            return e["t"]
        s = self.missiles.get(uid)
        return s.t[-1] if s is not None and s.t else math.inf

    def missile_live_until(self, uid):
        """When the missile stops being a threat: its closest approach to the target (``t_cpa``, WT imports) when
        known, else its end. A WT missile that missed flies on for up to ~2 min after passing its target."""
        e = self.end_by_uid.get(uid)
        if e is not None and isinstance(e.get("t_cpa"), (int, float)) and math.isfinite(e["t_cpa"]):
            return float(e["t_cpa"])
        return self.missile_end_t(uid)

    def ghost(self, ident, t_from, span=GHOST_S, step=GHOST_STEP_S):
        out = []
        k = 0
        while k*step <= span+EPS:
            st = self.state_at(ident, t_from+k*step)
            if st is not None:
                out.append([round(x, 1) for x in st["position"]])
            k += 1
        return out


# -- listing ---------------------------------------------------------------------------------------------------

def _last_line(path, limit=1 << 23):
    size = path.stat().st_size
    block = 65536
    with path.open("rb") as handle:
        while True:
            start = max(0, size-block)
            handle.seek(start)
            data = handle.read(size-start)
            lines = data.split(b"\n")
            complete = [x for x in (lines[:-1] if not data.endswith(b"\n") else lines) if x.strip()]
            if start > 0 and len(complete) < 2 and block < limit:
                block *= 4          # the last line may be a long one (sandbox missile_tick rows)
                continue
            if not complete:
                return None
            return json.loads(complete[-1])


_LIST_CACHE = {}


def replay_entry(path, rel, kind):
    st = path.stat()
    key = (str(path), st.st_mtime_ns, st.st_size)
    hit = _LIST_CACHE.get(str(path))
    if hit is not None and hit[0] == key:
        return dict(hit[1])
    with path.open("r", encoding="utf-8") as handle:
        header = json.loads(handle.readline())
    if header.get("type") != "header":
        raise ValueError("not a replay")
    try:
        last = _last_line(path)
    except (ValueError, OSError):
        last = None
    planes = header.get("planes") or []
    entry = dict(file=rel, kind=kind, duration_s=None if not last else last.get("t"),
                 players=sum(1 for p in planes if p.get("skill") != "AI"), units=len(planes), mtime=st.st_mtime)
    _LIST_CACHE[str(path)] = (key, entry)
    return dict(entry)


def list_replays(wt_dir, sandbox_dir):
    """[(entry, path)], newest first; entry = {file, kind, duration_s, players, units, mtime}. Reads only the header
    and the last line of each file."""
    found = []
    wt_dir, sandbox_dir = Path(wt_dir), Path(sandbox_dir)
    if wt_dir.is_dir():
        for path in wt_dir.glob("*.jsonl"):
            found.append((path, f"wt_real/{path.name}", "wt"))
    if sandbox_dir.is_dir():
        for path in sandbox_dir.glob("run-*/replay.jsonl"):
            found.append((path, f"sandbox/{path.parent.name}/replay.jsonl", "sandbox"))
    out = []
    for path, rel, kind in found:
        try:
            out.append((replay_entry(path, rel, kind), path))
        except (ValueError, OSError, UnicodeDecodeError):
            continue
    out.sort(key=lambda item: -item[0]["mtime"])
    return out


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# -- playback --------------------------------------------------------------------------------------------------

def capabilities(track, library, equipment, find_fm):
    """Per replay plane: whether it can be released and why not, and the missile it would carry."""
    out = {}
    for rid, info in track.info.items():
        reason = None
        if info["ai"]:
            reason = "AI 单位只能按 WT 轨迹"
        elif info["aircraft"] not in equipment:
            reason = "没有该机型的装备数据"
        elif find_fm(info["aircraft"]) is None:
            reason = "没有该机型的飞行模型"
        missile = None
        if info["aircraft"] in equipment:
            carried = set(modelled_missiles(info["aircraft"], library))
            for e in track.events:
                if e.get("kind") == "launch" and e.get("shooter") == rid and base_missile(e.get("missile")) in carried:
                    missile = base_missile(e.get("missile"))
                    break
        out[rid] = dict(controllable=reason is None, reason=reason, fork_missile=missile)
    return out


def playback_snapshot(track, t, *, caps=None, playing=False, file=None):
    """The analysis snapshot's structure, from the replay at ``t``."""
    caps = caps or {}
    planes, alive = [], [0, 0]
    for rid in sorted(track.planes):
        info = track.info.get(rid) or dict(team=0, aircraft="?", player=f"#{rid}", ai=False, missile=None)
        death = track.deaths.get(rid)
        dead = death is not None and death["t"] <= t+EPS
        if dead and t > death["t"]+WRECK_S:
            continue
        st = track.state_at(rid, t)
        if st is None:
            if not dead:
                continue
            st = track.last_before(rid, t)
            if st is None:
                continue
            st["velocity"], st["speed"] = (0., 0., 0.), 0.
        if not dead:
            alive[info["team"] if info["team"] in (0, 1) else 0] += 1
        cap = caps.get(rid, {})
        planes.append(dict(id=rid, team=info["team"], aircraft=info["aircraft"], alive=not dead,
                           position_m=list(st["position"]), heading_deg=st["heading_deg"], pitch_deg=st["pitch_deg"],
                           roll_deg=st["roll_deg"], speed_mps=st["speed"], velocity_mps=list(st["velocity"]),
                           load_g=None, engine_percent=None, aoa_deg=None, airbrake=0., missiles=st["missiles"],
                           missile=base_missile(info.get("missile")), chaff=st["chaff"], phase=st["phase"],
                           mode="replay", order=None, tracks=[],
                           death=None if not dead else dict(cause=death.get("cause"), time_s=death["t"],
                                                            killer=death.get("killer")),
                           radar_mode="unknown", player=info["player"], ai=info["ai"], attitude_source="estimated",
                           controllable=cap.get("controllable", False), control_reason=cap.get("reason"),
                           fork_missile=cap.get("fork_missile")))
    missiles = []
    for uid in sorted(track.missiles):
        m = track.missile_at(uid, t)
        if m is None:
            continue
        launch = track.launch_by_uid.get(uid) or {}
        shooter = m["shooter"] if m["shooter"] is not None else launch.get("shooter")
        target = m["target"] if m["target"] is not None else launch.get("target")
        basis = launch.get("target_basis")
        missiles.append(dict(uid=uid, team=(track.info.get(shooter) or {}).get("team", 0), shooter=shooter,
                             target=target, position_m=list(m["position"]), velocity_mps=list(m["velocity"]),
                             heading_deg=bearing_of(*m["velocity"][:2]), age_s=m["age_s"], seeker=m["seeker"],
                             datalink=m["datalink"], missile=base_missile(launch.get("missile")),
                             target_basis=basis.get("source") if isinstance(basis, dict) else basis))
    n = bisect_right(track.event_times, t+EPS)
    shown = [e for e in track.events[:n] if e.get("kind") in PLAYBACK_EVENTS]
    roster = {str(rid): dict(player=i["player"], aircraft=i["aircraft"], team=i["team"], ai=i["ai"])
              for rid, i in track.info.items()}
    return dict(time_s=t, tick=0, planes=planes, missiles=missiles, teams_alive=alive,
                launches=sum(1 for e in shown if e.get("kind") == "launch"), missile_errors=0, fm_faults=0,
                events=json.loads(json.dumps(shown[-120:])), event_total=len(shown), reason=None,
                map_half_m=track.map_half_m,
                replay=dict(file=file or track.name, kind=track.kind, start_s=track.start_s, duration_s=track.duration_s,
                            t=t, playing=playing, markers=track.markers, roster=roster))


# -- fork world ------------------------------------------------------------------------------------------------

def _normal(velocity, roll_deg):
    speed = math.sqrt(sum(v*v for v in velocity))
    if speed < 1e-6:
        return (0., 0., 1.)
    d = tuple(v/speed for v in velocity)
    h = math.hypot(d[0], d[1])
    if h < 1e-6:
        return (0., 1., 0.)
    right = (d[1]/h, -d[0]/h, 0.)
    up = (-d[0]*d[2]/h, -d[1]*d[2]/h, h)
    r = math.radians(roll_deg)
    c, s = math.cos(r), math.sin(r)
    return tuple(c*u+s*q for u, q in zip(up, right))


class TrackedAircraft(Aircraft):
    """An aircraft pinned to a replay track until ``release()``; afterwards the ordinary flight model integrates it from
    the last pinned state (same object, ring buffer and history kept)."""
    def __init__(self, model, track, rid, offset, *, params=FlightParams(), structural_speed=False):
        self.track, self.rid, self.offset = track, rid, float(offset)
        self.released = False
        pos, vel, roll = track.kinematics(rid, self.offset)
        super().__init__(model, pos, vel, normal=_normal(vel, roll), params=params, structural_speed=structural_speed)
        self.state = FlightState(self.state.position, self.state.velocity, self.state.normal, TRACK_AOA_DEG,
                                 min(TRACK_ENGINE_PERCENT, model.max_throttle))
        self.load = 1./max(.1, math.cos(math.radians(roll)))

    @property
    def tracked(self):
        return not self.released

    def release(self):
        self.released = True

    def step(self, command=None):
        if self.released:
            return super().step(command)
        if command is not None:
            self.command = command
        if not self.alive:
            return self.state
        pos, vel, roll = self.track.kinematics(self.rid, self.offset+(self.tick+1)*SUBSTEP_S+self.t0)
        if math.hypot(*vel) < PARKED_MPS:   # on the ground in the replay: no attitude to derive from the velocity
            self.load = 1.
            new = FlightState(pos, vel, self.state.normal, 0., self.state.engine_percent, 0.)
        else:
            self.load = 1./max(.1, math.cos(math.radians(roll)))
            new = FlightState(pos, vel, _normal(vel, roll), TRACK_AOA_DEG, self.state.engine_percent, 0.)
        return self._commit(new)

    def _commit(self, new):
        if self.released:
            return super()._commit(new)
        # Pinned: only the replay (or our missiles, via the engagement) ends it -- not our ground-contact rule
        # (some maps lie below the replay's zero datum) nor our overspeed rule.
        alive = self.alive
        out = super()._commit(new)
        self.alive, self.crashed, self.overspeed, self.overspeed_s = alive, False, False, 0.
        return out


class TrackedController:
    """A pinned aircraft's pilot: TWS search (as the sandbox's manual mode), no launches of its own, no chaff of its own.
    ``pilot`` is the SandboxPilot that takes over at the release (None: cannot be released)."""
    tracked = True

    def __init__(self, pilot, rid):
        self.pilot, self.rid = pilot, rid
        self.mode, self.order, self.observation = "track", None, None
        self.keys, self.keys_until = dict(roll=0, pitch=0, throttle=0, airbrake=False), 0.

    phase = "track"

    def describe(self):
        return self.pilot.describe() if self.pilot is not None else dict(kind="wt_track", replay_id=self.rid)

    def decide(self, obs):
        self.observation = obs
        return Action(radar=RadarCommand(mode="tws"))


class ReplayForkSimulation(SandboxSimulation):
    """A sandbox world whose aircraft start pinned to a replay; replay launches / chaff / deaths are applied at their
    recorded times (``offset`` + world time = replay time)."""
    def __init__(self, specs, seed, *, track, offset, rids, surrogates, library, **kwargs):
        self.track, self.offset, self.rids = track, float(offset), list(rids)
        self._surrogates = dict(surrogates)
        self.schedule, self._next = [], 0
        self.skipped_shots, self.released, self.chaff_skipped = [], [], 0
        self.ghosts, self.release_at = {}, {}
        self.replay_source = {}
        super().__init__(specs, seed, library=library, **kwargs)
        self.by_rid = {rid: p for rid, p in zip(self.rids, self.planes)}

    def _add_plane(self, ident, spec, data):
        real, surrogate = spec.aircraft, self._surrogates.get(ident)
        if surrogate is not None:
            spec.aircraft = surrogate
        try:
            super()._add_plane(ident, spec, data)
        finally:
            spec.aircraft = real
        p = self.planes[-1]
        if surrogate is not None:   # flight model only to construct the object; no sensors of another airframe
            p.aircraft, p.equipment, p.radar, p.rwr, p.has_maw = real, None, None, None, False
        f = p.flight
        p.flight = TrackedAircraft(f.model, self.track, self.rids[ident], self.offset, params=f.params,
                                   structural_speed=self.structural_speed)

    def _header(self):
        header = super()._header()
        header["simulator"] = "wt_overlay.sandbox (replay fork)"
        header["replay_source"] = self.replay_source
        for row, p in zip(header["planes"], self.planes):
            info = self.track.info.get(self.rids[p.ident]) or {}
            row.update(player=info.get("player"), replay_id=self.rids[p.ident], skill=row.get("skill") or
                       ("AI" if info.get("ai") else row.get("skill")))
        return header

    def step(self):
        if self.reason is None:
            self._replay_actions()
            # a pinned aircraft on the ground in the replay is parked for the engagement (the airfield ``grounded``
            # state: invisible to sensors, not a target); its pinned track keeps advancing below
            t = self.offset+self.time+SUBSTEP_S
            for p in self.live:
                if p.flight.tracked:
                    kin = self.track.kinematics(self.rids[p.ident], t)
                    parked = kin is not None and math.hypot(*kin[1]) < PARKED_MPS
                    if parked != p.grounded:
                        p.grounded, p.ground_t = parked, (self.time if parked else None)
        out = super().step()
        for p in self.live:
            if p.grounded and p.flight.tracked:   # the engagement does not step grounded aircraft
                p.flight.step()
        return out

    def _replay_actions(self):
        now = self.offset+self.time+SUBSTEP_S/2
        while self._next < len(self.schedule) and self.schedule[self._next][0] <= now:
            t, _, kind, data = self.schedule[self._next]
            self._next += 1
            getattr(self, "_do_"+kind)(t, data)

    def _skip(self, t, e, reason):
        rec = dict(replay_uid=e.get("uid"), replay_t=t, shooter=e.get("shooter"), target=e.get("target"),
                   missile=e.get("missile"), reason=reason)
        self.skipped_shots.append(rec)
        self.event("replay_shot_skipped", **rec)

    def _do_launch(self, t, e):
        shooter, target = self.by_rid.get(e.get("shooter")), self.by_rid.get(e.get("target"))
        mid = base_missile(e.get("missile"))
        reason = None
        if shooter is None:
            reason = "射手不在世界里"
        elif not shooter.alive:
            reason = "射手已在本推演中损失"
        elif not shooter.flight.tracked:
            reason = "射手已接管（由脚本 / 人决定发射）"
        elif not missile_modelled(self.library, mid):
            reason = f"导弹 {e.get('missile')} 不在模型库（非主动雷达或无 missile_sim 档案）"
        elif target is None:
            reason = "目标不在世界里"
        elif not target.alive:
            reason = "目标已在本推演中损失"
        elif target.team == shooter.team:
            reason = "目标为友军"
        if reason is not None:
            if shooter is not None and shooter.alive and shooter.flight.tracked:
                shooter.missiles = max(0, shooter.missiles-1)
            self._skip(t, e, reason)
            return
        old = shooter.missile_id
        shooter.missile_id = mid
        try:
            m = self.fire(shooter, target)
        finally:
            shooter.missile_id = old
        basis = e.get("target_basis")
        self.event("replay_shot", uid=m.uid, replay_uid=e.get("uid"), replay_t=t, missile=mid,
                   target_basis=basis.get("source") if isinstance(basis, dict) else basis)

    def _do_chaff(self, t, e):
        p = self.by_rid.get(e.get("plane"))
        if p is None or not p.alive or not p.flight.tracked:
            return
        if p.chaff <= 0:
            self.chaff_skipped += 1
            return
        self.apply(p, Action(chaff=1))

    def _do_death(self, t, e):
        p = self.by_rid.get(e.get("plane"))
        if p is None or not p.alive or not p.flight.tracked:
            return
        killer = self.by_rid.get(e.get("killer"))
        if killer is p or (killer is not None and not killer.flight.tracked):
            killer = None
        self._kill(p, killer, f"replay_{e.get('cause')}", None)

    def _do_release(self, t, data):
        p, mode = self.planes[data["plane"]], data["mode"]
        if not p.alive:
            self.event("replay_release_skipped", plane=p.ident, replay_id=self.rids[p.ident], reason="已在本推演中损失")
            return
        pilot = p.controller.pilot
        p.flight.release()
        p.controller = pilot
        self.event("replay_release", plane=p.ident, replay_id=self.rids[p.ident], mode=mode, replay_t=t)
        set_order(self, p.ident, dict(mode=mode))
        p.next_decision = self.tick


REFLY_MAX_AGE_S = 60.   # a missile launched longer than this before t_fork is not re-flown (kinematically spent)


def _airborne(track, rid, t):
    kin = track.kinematics(rid, t)
    return kin is not None and math.hypot(*kin[1][:2]) >= MIN_STEP_SPEED_MPS


def fork_t0(track, t_fork, library, max_age_s=REFLY_MAX_AGE_S):
    """The earlier of ``t_fork`` and the launch of the oldest modelled missile still live at ``t_fork`` (not yet past
    its closest approach, ``ReplayTrack.missile_live_until``) and launched at most ``max_age_s`` before ``t_fork``."""
    t0 = t_fork
    for uid, e in track.launch_by_uid.items():
        if t_fork-max_age_s-EPS <= e["t"] <= t_fork+EPS and track.missile_live_until(uid) > t_fork+EPS and \
                missile_modelled(library, base_missile(e.get("missile"))):
            t0 = min(t0, e["t"])
    return t0


def build_fork(track, t_fork, control, *, library, include_ai=False, me=None, seed=0, file=None, sha256=None):
    """The fork world (paused at world time 0 = replay time t0). ``control``: {replay plane id: mode}; unlisted planes
    stay on their WT track. ``me``: the user's plane (replay id) for the per-team cap."""
    from .engagement import equipment_data
    from .fm.catalog import find_aircraft
    equipment = equipment_data().equipment
    t_fork = float(t_fork)
    if not track.start_s-EPS <= t_fork <= track.duration_s+EPS:
        raise ValueError("接管时刻超出回放范围")
    caps = capabilities(track, library, equipment, find_aircraft)
    control = {int(k): v for k, v in (control or {}).items()}
    for rid, mode in control.items():
        if mode not in CONTROL_MODES:
            raise ValueError(f"未知控制方式 {mode}")
        if rid not in track.info:
            raise ValueError(f"回放中没有飞机 {rid}")
        if mode != "track":
            if not caps[rid]["controllable"]:
                raise ValueError(f"{track.label(rid)} 不能接管：{caps[rid]['reason']}")
            if not track.present(rid, t_fork) or not _airborne(track, rid, t_fork):
                raise ValueError(f"{track.label(rid)} 在接管时刻不在空中")
    released = {rid: m for rid, m in control.items() if m != "track"}
    t0 = fork_t0(track, t_fork, library)
    notes = []
    present = [rid for rid in sorted(track.info) if track.present(rid, t0) and track.kinematics(rid, t0) is not None
               and (include_ai or not track.info[rid]["ai"])]
    # an aircraft still on the airfield (parked / taxiing) cannot be built by the flight model: left out like a late one
    members = [rid for rid in present if _airborne(track, rid, t0)]
    if len(members) < len(present):
        notes.append(f"{len(present)-len(members)} 架在世界起点仍在机场（速度低于 {MIN_STEP_SPEED_MPS:.0f} m/s），不在推演里")
    for rid in released:
        if rid not in members:
            raise ValueError(f"{track.label(rid)} 在世界起点 t0={t0:.2f} s 不在空中")
    # per team at most REPLAY_MAX_TEAM: the ones nearest the user's plane (released planes always stay)
    anchor = me if me in members else next(iter(released), None)
    ref = track.kinematics(anchor, t0)[0] if anchor is not None else None
    kept = []
    for team in (0, 1):
        group = [rid for rid in members if track.info[rid]["team"] == team]
        if len(group) > REPLAY_MAX_TEAM:
            key = (lambda r: (r not in released and r != anchor, math.dist(track.kinematics(r, t0)[0], ref))) \
                if ref is not None else (lambda r: (r not in released, r))
            chosen = sorted(group, key=key)[:REPLAY_MAX_TEAM]
            notes.append(f"队伍 {team} 有 {len(group)} 架，只取" + ("离我的飞机最近的" if ref is not None else "编号最小的")
                         + f" {REPLAY_MAX_TEAM} 架")
            group = [r for r in group if r in chosen]
        kept += group
    members = sorted(kept)
    if not members:
        raise ValueError("t0 时刻回放里没有飞机")
    # specs: equipment-known aircraft through the match builder (script pilots for the release), others by hand
    states = {rid: track.kinematics(rid, t0) for rid in members}
    scripted = [[], []]
    for rid in members:
        info = track.info[rid]
        if info["aircraft"] in equipment and not info["ai"] and info["team"] in (0, 1):
            pos, vel, _ = states[rid]
            st = track.state_at(rid, t0) or track.last_before(rid, t0) or {}
            count = st.get("missiles") if isinstance(st.get("missiles"), int) else 0
            member = dict(aircraft=info["aircraft"], skill="top", altitude_m=max(1., pos[2]),
                          mach=max(.1, math.sqrt(sum(v*v for v in vel))/atmosphere(max(0., min(19999., pos[2])))[1]))
            missile = caps[rid]["fork_missile"]
            if missile is not None:
                member.update(missile=missile, missiles=count)
            scripted[info["team"]].append((rid, member, missile))
    match = make_match([[m for _, m, _ in scripted[0]], [m for _, m, _ in scripted[1]]], seed, range_km=60.,
                       library=library, map_half_m=track.map_half_m) if scripted[0] or scripted[1] else None
    made = {}
    if match is not None:
        it = iter(match.specs)
        for team in (0, 1):
            for rid, _, missile in scripted[team]:
                s = next(it)
                if missile is None:
                    s.missile, s.missiles = None, 0
                made[rid] = s
    centers = []
    for team in (0, 1):
        pts = [states[r][0] for r in members if track.info[r]["team"] == team]
        centers.append((sum(p[0] for p in pts)/len(pts), sum(p[1] for p in pts)/len(pts)) if pts else None)
    specs, surrogates = [], {}
    for ident, rid in enumerate(members):
        info = track.info[rid]
        pos, vel, _ = states[rid]
        s = made.get(rid)
        if s is None:
            s = PlaneSpec(info["aircraft"], info["team"] if info["team"] in (0, 1) else 0, pos, vel, None, None, 0, 0,
                          skill="AI" if info["ai"] else "", archetype="")
        s.position, s.velocity = tuple(pos), tuple(vel)
        s.name = info["player"]
        if find_aircraft(info["aircraft"]) is None:
            surrogates[ident] = SURROGATE_FM
        team = s.team
        enemy = centers[1-team] or (pos[0]+vel[0]*100., pos[1]+vel[1]*100.)
        s.home_xy, s.enemy_xy = tuple(pos[:2]), tuple(enemy)
        pilot = s.controller
        if pilot is not None:
            v = math.hypot(vel[0], vel[1]) or 1.
            pilot.forward = (vel[0]/v, vel[1]/v) if math.hypot(vel[0], vel[1]) > 1e-6 else (0., 1.)
            pilot.forward_deg = bearing_of(*pilot.forward)
            pilot.left = (-pilot.forward[1], pilot.forward[0])
            pilot.home_xy, pilot.enemy_xy = s.home_xy, s.enemy_xy
            pilot.anc = (*s.enemy_xy, None, "spawn")
            pilot.ident = ident
        specs.append(s)
    if surrogates:
        notes.append(f"{len(surrogates)} 架没有飞行模型的钉轨飞机借用 {SURROGATE_FM} 的模型构造（不积分、无传感器）")
    notes.append(f"导弹重飞：只重飞接管时尚未过最近点、且发射不早于接管前 {REFLY_MAX_AGE_S:.0f} s 的已建模导弹")
    notes.append("箔条数量回放里未知：按种子抽样（与普通沙盘相同）；回放箔条事件按记录时刻各投 1 束，不足时跳过")
    notes.append("导弹数量 = t0 时帧里的 missiles 列（WT 导入里是此后回放中看到的发射次数，不是真实挂载）")
    notes.append("世界在回放结束时到达时间上限；钉轨飞机在其轨迹结束后沿最后速度直线外推")
    duration = max(1., track.duration_s-t0)
    eng = ReplayForkSimulation(specs, seed, track=track, offset=t0, rids=members, surrogates=surrogates,
                               library=library, map_half_m=track.map_half_m, time_limit_s=duration)
    idents = {rid: i for i, rid in enumerate(members)}
    if me is not None and me not in idents:
        notes.append("我的飞机不在世界里")
    skipped_static = []
    eng.scenario = dict(name=f"WT 回放接管 · {file or track.name} @ {t_fork:.2f} s", seed=seed, time_limit_s=duration,
                        map_half_m=track.map_half_m, notes=notes)
    equip_scripts(eng)
    for p in eng.planes:
        pilot = SandboxPilot(p.controller, eng, p.ident) if p.controller is not None and \
            caps[members[p.ident]]["controllable"] else None
        p.controller = TrackedController(pilot, members[p.ident])
    # the replay's actions from t0 on, in time order: deaths, releases, launches, chaff
    sched = []
    for uid, e in track.launch_by_uid.items():   # live at t_fork but older than the re-fly window
        if e["t"] < t0-EPS and track.missile_live_until(uid) > t_fork+EPS and \
                missile_modelled(library, base_missile(e.get("missile"))):
            skipped_static.append(dict(replay_uid=uid, replay_t=e["t"],
                                       reason=f"发射早于接管 {REFLY_MAX_AGE_S:.0f} s 以上，视为能量耗尽，不重飞"))
    for e in track.events:
        t, k = e.get("t", 0.), e.get("kind")
        if t < t0-EPS:
            continue
        if k == "death" and e.get("plane") in idents:
            sched.append((t, 0, "death", e))
        elif k == "launch":
            sched.append((t, 2, "launch", e))
            reason = None
            if e.get("shooter") not in idents:
                reason = "射手不在世界里"
            elif not missile_modelled(library, base_missile(e.get("missile"))):
                reason = f"导弹 {e.get('missile')} 不在模型库（非主动雷达或无 missile_sim 档案）"
            elif e.get("target") not in idents:
                reason = "目标不在世界里"
            if reason is not None:
                skipped_static.append(dict(replay_uid=e.get("uid"), replay_t=t, reason=reason))
        elif k == "chaff" and e.get("plane") in idents:
            sched.append((t, 3, "chaff", e))
    released_rows = []
    for rid, mode in released.items():
        ident = idents[rid]
        sched.append((t_fork, 1, "release", dict(plane=ident, mode=mode)))
        eng.ghosts[ident] = track.ghost(rid, t_fork)
        eng.release_at[ident] = t_fork
        released_rows.append(dict(replay_id=rid, plane=ident, player=track.info[rid]["player"],
                                  aircraft=track.info[rid]["aircraft"], mode=mode))
    sched.sort(key=lambda x: (x[0], x[1]))
    eng.schedule = sched
    eng.released = released_rows
    eng.replay_source = dict(file=file or track.name, sha256=sha256, t_fork=t_fork, t0=t0, released=released_rows,
                             skipped_shots=skipped_static, include_ai=bool(include_ai), me=me,
                             me_plane=idents.get(me), members=members, notes=notes)
    eng.fork_me = idents.get(me)
    return eng


def fork_source(eng):
    """replay_source for scenario.json: the shots skipped so far (``reached``) and those the build already knew would be
    skipped but whose time has not come (``reached`` false)."""
    src = dict(eng.replay_source)
    seen = {s["replay_uid"] for s in eng.skipped_shots}
    src["skipped_shots"] = [dict(s, reached=True) for s in eng.skipped_shots] + \
        [dict(s, reached=False) for s in eng.replay_source["skipped_shots"] if s["replay_uid"] not in seen]
    src["chaff_skipped"] = eng.chaff_skipped
    src["world_time_s"] = eng.time
    return src


def fork_snapshot(eng):
    snap = analysis_snapshot(eng)
    for row, p in zip(snap["planes"], eng.planes):
        rid = eng.rids[p.ident]
        info = eng.track.info.get(rid) or {}
        tracked = p.flight.tracked
        row.update(player=info.get("player"), ai=info.get("ai", False), replay_id=rid, tracked=tracked,
                   controllable=tracked and getattr(p.controller, "pilot", None) is not None,
                   release_at=eng.release_at.get(p.ident) if tracked else None,
                   attitude_source="estimated" if tracked else "model")
        if p.ident in eng.ghosts and not tracked:
            row["ghost"] = eng.ghosts[p.ident]
        if tracked:
            row["phase"] = "track"
    snap["map_half_m"] = eng.map_half_m
    src = eng.replay_source
    roster = {str(p.ident): dict(player=(eng.track.info.get(eng.rids[p.ident]) or {}).get("player"),
                                 aircraft=p.aircraft, team=p.team) for p in eng.planes}
    snap["fork"] = dict(file=src.get("file"), t_fork=src.get("t_fork"), t0=eng.offset, offset=eng.offset,
                        me=eng.fork_me, released=src.get("released"), skipped=len(eng.skipped_shots),
                        notes=src.get("notes"), roster=roster)
    return snap

