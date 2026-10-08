#!/usr/bin/env python3
"""War Thunder replay -> engagement replay JSONL (the format scripts/rl_dashboard.py plays).

Input: an export of the missile_sim replay inspector (branch codex/replay-missile-inspector, local branch
wt-overlay-exports): ``match_export.json`` from tools/replay_inspector/inspect_match.py (preferred) or
``raw_extended.json`` from inspect_extended.py; a file, an export directory, or folders searched recursively (default
outputs/wt_replays and outputs/wt_replay_exports; only those two file names and .wrpl are looked at, so *.vromfs.bin and
other files next to them are ignored). A .wrpl is not read here; the inspector backend (WrplReplayParser with this
branch's patches, plus the game's aces / game / char .vromfs.bin of the replay's version) exports it first:

    PYTHONPATH=<backend>/python/main python3 tools/replay_inspector/inspect_match.py \
        --game <dir with the .vromfs.bin> --replay '<dir>/#2026.10.08 11.49.38.wrpl' --output outputs/wt_replay_exports/<name>

Output: outputs/engagements/wt_real/<replay name>.jsonl, one file per input:
  header   planes [{id, team, aircraft, name, archetype (player), skill ("AI" for unowned aircraft), missile, missiles,
           chaff, ..., wt{eid, player, team_source, samples, gaps, ...}}], the standard plane_columns / missile_columns,
           map_half_m, frame_dt_s=0.25, time_limit_s, and ``source``: provenance, frame, and ``fields`` saying for
           each item whether it came from the replay or was inferred;
  frame    every 0.25 s of replay time, rows interpolated linearly between the decoded samples (about 4 Hz);
  event    launch (target and target_basis.source), missile_end (outcome, outcome_basis, evidence, closest approach,
           end altitude / speed / flight-path angle), kill and death (kill feed), damage (severe / critical damage
           messages), chaff and flare (countermeasure projectiles grouped per release), death "left_replay";
  end      reason "replay_end" with the usual result summary.

Frame: ENU metres, x = Dagor X, y = Dagor Z, z = Dagor Y (altitude), recentred on the middle of the aircraft tracks
(offset in source.origin_dagor_xz). Dagor taken as left-handed with Y up; were it not, the map would be mirrored
east-west, with no range, altitude, angle or speed changed.

Where each item comes from (match export first, then the fallback used for raw_extended.json):
  team        MPlayer.team of the owning player, else the unit's army (BaseExtReflectable.unitArmyNo) -- replay;
              fallback: 2-means on spawn positions, checked against shooter / target pairs; --team NAME=0|1 overrides
  kills       KillMessage (killer, victim, weapon name, weapon and death type) -- replay. The kill feed's weapon name
              is often not the missile that hit, so the missile is credited by geometry (closest approach <= 150 m of
              the victim) and kill_weapon_match records whether the names agree; fallback: Unit.killed_at_ms + a hit
  damage      SevereDamage / CriticalDamage messages (replay) and damage-model (FM_DVM 0xf09a) messages (replay, payload
              undecoded) on the target near the missile's end
  chaff/flare countermeasure projectiles (replay: owner, time); flare vs chaff from the projectile's spin (u12_3) or
              lifetime (candidate: spinning ones live 1.7-4.5 s, the others 15-20 s)
  target      seeker block of the missile sync (candidate layout from the 2026-09-27 sample; the aircraft within 300 m
              of its tracked position, majority vote) > kill feed > the shooter's (6, 0) target designation at launch
              (candidate) > geometry (heading error vs line of sight plus closest approach; "ambiguous" when the
              runner-up is within 10 points)
  outcome     kill feed or damage messages with the missile within 150 m of the target, else closest approach:
              hit, hit_nokill, miss, target_dead (target killed before the missile got there), unknown
  velocity    decoded aircraft sync velocity (upstream scale, matches position differences to ~4 m/s median),
              else finite differences of the positions; missiles: sync b7 velocity (candidate scale) at their samples
  missile end the entity's destroy time when it comes within 3 s of the last sample (far missiles are synced only
              every 0.8-1.6 s), with the missile carried on straight from its last sample to that time
  not in the data: radar / RWR display state (sensor lists and designations exist only as candidate fields in the
              export), seeker / datalink on-off, loadout (the missiles column counts down launches seen), terrain.

Run:  .venv/bin/python scripts/wt_replay_import.py [inputs ...] [--out-dir DIR] [--team NAME=0] [--all-units]
Pure standard library (Python 3.10+).
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = [ROOT / "outputs" / "wt_replays", ROOT / "outputs" / "wt_replay_exports"]
DEFAULT_OUT = ROOT / "outputs" / "engagements" / "wt_real"
IMPORTER_VERSION = 2

UNKNOWN_MS = 0xFFFFFFFF          # inspector sentinel for "no such event"
FRAME_DT = 0.25
PLANE_COLUMNS = ["id", "x", "y", "z", "vx", "vy", "vz", "heading_deg", "missiles", "chaff", "phase"]
MISSILE_COLUMNS = ["uid", "owner", "target", "x", "y", "z", "vx", "vy", "vz", "heading_deg", "age_s", "seeker",
                   "datalink"]

# outcome thresholds
REACH_M = 40.0          # closest approach at or below this, with the missile ending there: it reached the target
NEAR_M = 150.0          # ... at or below this plus a kill at the missile's end: hit (sample-spacing allowance)
END_AT_CPA_S = 0.6      # the missile ended at its closest approach when it ended no later than this after it
KILL_BEFORE_S = 1.0     # the kill-feed time may precede the missile's last sample by this much ...
KILL_AFTER_S = 5.0      # ... or follow it by this much and still count for the missile
KILL_FEED_AFTER_S = 30.0   # a kill-feed entry naming the shooter, the target and this missile type counts this late
DESTROY_EXTRA_S = 3.0   # at most this long from the last missile sample to the entity's destroy time; client replays
                        # sync far missiles only every 0.8-1.6 s, so the last sample can be ~1 km short of the target
EXTRAP_S = 0.5          # aircraft tracks are extrapolated at most this far past their ends (geometry only)
DIST_PTS_PER_KM = 10.0  # target score: median azimuth error (deg) + this per km of closest approach (capped) ...
DIST_CAP_KM = 5.0
CPA_BONUS = ((150.0, 40.0), (500.0, 20.0))   # ... - 40 when it came within 150 m, - 20 within 500 m ...
SAME_TEAM_PTS = 60.0    # ... + this for a candidate on the shooter's team
AMBIGUOUS_MARGIN = 10.0 # runner-up within this many points (and no close pass): the target is ambiguous
CM_RE = re.compile(r"flare|chaff|countermeasure|dipole", re.I)   # weapon names treated as countermeasures
SEEK_MATCH_M = 300.0    # seeker block position within this of an aircraft = a vote for that aircraft as the target
DESIG_MATCH_M = 500.0   # shooter designation within this of an aircraft at launch = designated target
DESIGNATION_TRACK = (6, 0)   # target designation type (t#_1, t#_2) that sits on enemy aircraft (median 92 m)
DVM_HIT_ID = 0xF09A     # FM_DVM (type 16) message id seen on aircraft with a missile within 100 m
WT_TEAM = {1: 0, 2: 1}  # replay team / army number -> dashboard team
GAP_S = 2.0             # samples further apart than this are not used together for a velocity
ATTITUDE_OK_DEG = 15.0  # median nose-vs-velocity angle above this -> the decoded attitude is not trusted


# -- small vector helpers --------------------------------------------------------------------------------------

def sub(a, b):
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])


def add(a, b):
    return (a[0]+b[0], a[1]+b[1], a[2]+b[2])


def mul(a, s):
    return (a[0]*s, a[1]*s, a[2]*s)


def dot(a, b):
    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]


def norm(a):
    return math.sqrt(dot(a, a))


def angle_deg(a, b):
    na, nb = norm(a), norm(b)
    if na < 1e-9 or nb < 1e-9:
        return None
    return math.degrees(math.acos(max(-1.0, min(1.0, dot(a, b)/(na*nb)))))


def azimuth_gap_deg(v, los):
    """Angle between the horizontal parts of two vectors (insensitive to loft); None when either is ~vertical."""
    hv, hl = math.hypot(v[0], v[1]), math.hypot(los[0], los[1])
    if hv < 30.0 or hl < 300.0:
        return None
    c = (v[0]*los[0]+v[1]*los[1])/(hv*hl)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def compass_deg(v):
    return math.degrees(math.atan2(v[0], v[1])) % 360.0


def ms(value):
    """Inspector millisecond field -> seconds, or None for the sentinel / missing."""
    if value is None or not isinstance(value, (int, float)) or value == UNKNOWN_MS or value < 0:
        return None
    return value/1000.0


# -- Dagor -> ENU ------------------------------------------------------------------------------------------------

def dagor_to_enu(p):
    return (float(p[0]), float(p[2]), float(p[1]))


def nose_enu(euler):
    """Body x axis from the decoded Euler angles [bank (X), heading (Y), attitude (Z)], radians, with the axis order
    of analyze_extended.axes() (R_y(heading) R_z(attitude) R_x(bank)), mapped to ENU."""
    if not euler or len(euler) < 3 or any(e is None or not math.isfinite(e) for e in euler[:3]):
        return None
    _, h, a = (float(e) for e in euler[:3])
    fx, fy, fz = math.cos(h)*math.cos(a), math.sin(a), -math.sin(h)*math.cos(a)
    return (fx, fz, fy)


class Track:
    """One object's decoded samples: times (s, strictly increasing), ENU positions, finite-difference velocities and
    (optionally) nose unit vectors."""

    def __init__(self, samples, origin=(0.0, 0.0)):
        by_time = {}
        for s in samples:
            t, xyz = s.get("t_ms"), s.get("xyz")
            if t is None or not xyz or any(c is None or not math.isfinite(c) for c in xyz):
                continue
            by_time[int(t)] = s          # a repeated packet time keeps the last state sent with it
        self.t, self.p, self.nose = [], [], []
        for t_ms in sorted(by_time):
            s = by_time[t_ms]
            e, n, u = dagor_to_enu(s["xyz"])
            self.t.append(t_ms/1000.0)
            self.p.append((e-origin[0], n-origin[1], u))
            self.nose.append(nose_enu(s.get("euler_raw")))
        self.v = [self._velocity(i) for i in range(len(self.t))]

    def _velocity(self, i):
        t, p, n = self.t, self.p, len(self.t)
        if n < 2:
            return (0.0, 0.0, 0.0)
        j = i-1 if i > 0 and t[i]-t[i-1] <= GAP_S else i
        k = i+1 if i < n-1 and t[i+1]-t[i] <= GAP_S else i
        if j == k:                                   # isolated sample: use the nearer neighbour
            if i == 0:
                j, k = 0, 1
            elif i == n-1:
                j, k = n-2, n-1
            else:
                j, k = (i-1, i) if t[i]-t[i-1] <= t[i+1]-t[i] else (i, i+1)
        return mul(sub(p[k], p[j]), 1.0/(t[k]-t[j]))

    def set_velocity(self, samples, origin_unused=None, max_dt=0.3):
        """Replace the finite-difference velocity at each sample by the nearest decoded velocity (ENU tuples with time
        in s) within max_dt. Returns the number of samples replaced."""
        if not samples:
            return 0
        ts = [s[0] for s in samples]
        n = 0
        for i, t in enumerate(self.t):
            j = bisect.bisect_left(ts, t)
            best = None
            for k in (j-1, j):
                if 0 <= k < len(ts) and abs(ts[k]-t) <= max_dt and (best is None or abs(ts[k]-t) < abs(ts[best]-t)):
                    best = k
            if best is not None:
                self.v[i] = samples[best][1]
                n += 1
        return n

    def __len__(self):
        return len(self.t)

    @property
    def first(self):
        return self.t[0]

    @property
    def last(self):
        return self.t[-1]

    def at(self, t, extrap=0.0):
        """(position, velocity, nose or None) at time t; None outside [first-extrap, last+extrap]."""
        ts = self.t
        if not ts or t < ts[0]-extrap-1e-9 or t > ts[-1]+extrap+1e-9:
            return None
        if t <= ts[0]:
            return add(self.p[0], mul(self.v[0], t-ts[0])), self.v[0], self.nose[0]
        if t >= ts[-1]:
            return add(self.p[-1], mul(self.v[-1], t-ts[-1])), self.v[-1], self.nose[-1]
        i = bisect.bisect_right(ts, t)
        a, b = i-1, i
        f = (t-ts[a])/(ts[b]-ts[a])
        p = add(self.p[a], mul(sub(self.p[b], self.p[a]), f))
        v = add(self.v[a], mul(sub(self.v[b], self.v[a]), f))
        na, nb = self.nose[a], self.nose[b]
        nose = None
        if na is not None and nb is not None:
            m = add(na, mul(sub(nb, na), f))
            nm = norm(m)
            nose = mul(m, 1.0/nm) if nm > 1e-9 else na
        return p, v, nose

    def gaps(self, t0=None, t1=None):
        ts = [t for t in self.t if (t0 is None or t >= t0) and (t1 is None or t <= t1)]
        d = [b-a for a, b in zip(ts, ts[1:])]
        return (statistics.median(d) if d else None), (max(d) if d else None)


# -- input -------------------------------------------------------------------------------------------------------

def find_inputs(paths):
    found, wrpl = [], []
    for raw in paths:
        p = Path(raw)
        if p.is_file() and p.name.endswith(".json"):
            found.append(p)
        elif p.is_file() and p.suffix.lower() == ".wrpl":
            wrpl.append(p)
        elif p.is_dir():                                   # only export files and .wrpl; *.vromfs.bin etc. ignored
            for d in sorted({q.parent for q in p.rglob("*.json") if q.name in ("match_export.json",
                                                                                "raw_extended.json")}):
                found.append(d/"match_export.json" if (d/"match_export.json").exists() else d/"raw_extended.json")
            wrpl.extend(sorted(q for q in p.rglob("*") if q.suffix.lower() == ".wrpl"))
        else:
            print(f"skip {p}: not found", file=sys.stderr)
    return found, wrpl


def replay_name(raw, path):
    src = raw.get("source")
    base = Path(src.replace("\\", "/")).stem if isinstance(src, str) and src else path.parent.name
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("_.") or "wt_replay"
    return name


# -- teams -------------------------------------------------------------------------------------------------------

def two_means(points):
    """Labels 0/1 for 2-D points (farthest pair seeds, a few Lloyd steps) and the centroid separation."""
    n = len(points)
    if n < 2:
        return [0]*n, 0.0
    best = (-1.0, 0, 1)
    for i in range(n):
        for j in range(i+1, n):
            d = math.dist(points[i], points[j])
            if d > best[0]:
                best = (d, i, j)
    c = [points[best[1]], points[best[2]]]
    labels = [0]*n
    for _ in range(30):
        new = [0 if math.dist(p, c[0]) <= math.dist(p, c[1]) else 1 for p in points]
        groups = [[p for p, k in zip(points, new) if k == g] for g in (0, 1)]
        if not groups[0] or not groups[1]:
            break
        c = [(sum(p[0] for p in g)/len(g), sum(p[1] for p in g)/len(g)) for g in groups]
        if new == labels:
            break
        labels = new
    return labels, math.dist(c[0], c[1])


def infer_teams(planes, pairs, overrides):
    """planes: list of dicts with id, player, aircraft, track. pairs: (shooter, target, weight) from a first target
    pass. Returns (team per id, info)."""
    keys = {}
    for p in planes:                                      # one team per player across respawns
        keys.setdefault(p["player"] or f"#unit{p['id']}", []).append(p)
    groups = list(keys.values())
    spawn = [g[0]["track"].p[0][:2] for g in groups]      # first sample of the player's earliest unit
    labels, separation = two_means(spawn)
    team = {}
    for g, k in zip(groups, labels):
        for p in g:
            team[p["id"]] = k
    basis = "spawn_2means"
    agree = sum(w for a, b, w in pairs if team[a] != team[b])
    clash = sum(w for a, b, w in pairs if team[a] == team[b])
    if pairs and (separation < 5000.0 or clash > agree):  # spawn clustering is not convincing: colour the pairs
        colour = {}
        adj = {}
        for a, b, w in pairs:
            adj.setdefault(a, []).append(b)
            adj.setdefault(b, []).append(a)
        for start in sorted(adj):
            if start in colour:
                continue
            colour[start] = team[start]
            stack = [start]
            while stack:
                u = stack.pop()
                for w in adj[u]:
                    if w not in colour:
                        colour[w] = 1-colour[u]
                        stack.append(w)
        for g in groups:                                  # player majority, spawn label for the rest
            votes = [colour[p["id"]] for p in g if p["id"] in colour]
            if votes:
                k = 1 if sum(votes)*2 > len(votes) else 0
                for p in g:
                    team[p["id"]] = k
        basis = "missile_pairs"
        agree = sum(w for a, b, w in pairs if team[a] != team[b])
        clash = sum(w for a, b, w in pairs if team[a] == team[b])
    fixed = {}
    for spec in overrides:
        key, _, val = spec.partition("=")
        if val not in ("0", "1"):
            raise SystemExit(f"--team {spec}: expected NAME=0 or NAME=1")
        hit = [p for p in planes if key in (p["player"], p["aircraft"], str(p["id"]))]
        if not hit:
            print(f"warning: --team {spec} matches no aircraft", file=sys.stderr)
        for p in hit:
            for q in keys[p["player"] or f"#unit{p['id']}"]:
                fixed[q["id"]] = int(val)
    if fixed:
        flips = sum(1 for i, k in fixed.items() if team[i] != k)
        if flips*2 > len(fixed):                          # align the inferred labels with the overrides
            team = {i: 1-k for i, k in team.items()}
        team.update(fixed)
        basis += "+overrides"
    else:                                                 # deterministic labels: the first unit is team 0
        first = min(planes, key=lambda p: p["id"])["id"] if planes else None
        if first is not None and team[first] == 1:
            team = {i: 1-k for i, k in team.items()}
    info = dict(basis=basis, spawn_separation_m=round(separation), pair_weight_cross_team=agree,
                pair_weight_same_team=clash, overrides=sorted(overrides))
    return team, info


# -- missiles: target, closest approach, outcome -----------------------------------------------------------------

def infer_target(m, shooter, planes, team=None):
    """The aircraft missile ``m`` flew toward. Returns (target id or None, info dict)."""
    t0, t1 = m.first, m.last
    flight = t1-t0
    skip = min(1.0, 0.3*flight)
    idx, last_t = [], -1e9
    for i, t in enumerate(m.t):                           # ~4 Hz over the flight, every sample in the last 3 s
        if t < t0+skip:
            continue
        if t-last_t >= 0.25 or t >= t1-3.0:
            idx.append(i)
            last_t = t
    if not idx:
        idx = list(range(len(m.t)))
    cands = []
    for c in planes:
        if c["id"] == shooter:
            continue
        tr = c["track"]
        if tr.last+EXTRAP_S < t0 or tr.first > t1:
            continue
        killed = c["killed_s"]
        if killed is not None and killed < t0:
            continue
        angs, dists = [], []
        for i in idx:
            st = tr.at(m.t[i], EXTRAP_S)
            if st is None:
                continue
            los = sub(st[0], m.p[i])
            dists.append(norm(los))
            a = azimuth_gap_deg(m.v[i], los)
            if a is None:
                a = angle_deg(m.v[i], los)
            if a is not None:
                angs.append((m.t[i], a))
        if len(dists) < 3 or not angs:
            continue
        k = max(1, int(round(len(angs)*0.8)))
        med = statistics.median(a for _, a in angs[:k])                     # first 80 % of the flight
        early = [a for t, a in angs if t <= t0+skip+0.5*flight] or [angs[0][1]]
        med = 0.5*(med+statistics.median(early))                            # ... and its first half
        dmin = min(dists)
        score = med+DIST_PTS_PER_KM*min(dmin/1000.0, DIST_CAP_KM)
        score -= CPA_BONUS[0][1] if dmin <= CPA_BONUS[0][0] else CPA_BONUS[1][1] if dmin <= CPA_BONUS[1][0] else 0.0
        same = team is not None and shooter is not None and team.get(c["id"]) == team.get(shooter)
        if same:
            score += SAME_TEAM_PTS
        cands.append((score, c["id"], med, dmin, same))
    if not cands:
        return None, dict(basis="none", candidates=0)
    cands.sort()
    score, cid, med, dmin, same = cands[0]
    basis = "cpa" if dmin <= NEAR_M else "guidance" if med <= 25.0 else "weak"
    info = dict(basis=basis, score=round(score, 1), az_err_deg=round(med, 1), min_range_m=round(dmin),
                candidates=len(cands), same_team=same)
    if len(cands) > 1:
        info["runner_up"] = cands[1][1]
        info["margin"] = round(cands[1][0]-score, 1)
        if cands[1][0]-score < AMBIGUOUS_MARGIN and basis != "cpa":
            info["ambiguous"] = True
    return cid, info


def closest_approach(m, tgt, t_destroy, t_cut=None):
    """(cpa_m, t_cpa) between missile track m and target track tgt, segment-wise on relative positions, plus the
    straight continuation from the missile's last sample to its destroy time; times after t_cut are ignored.
    (None, None) without overlap."""
    rel = []
    for t, p in zip(m.t, m.p):
        if t_cut is not None and t > t_cut:
            break
        st = tgt.at(t, EXTRAP_S)
        if st is not None:
            rel.append((t, sub(st[0], p)))
    if t_destroy is not None and rel and rel[-1][0] == m.last and t_destroy > m.last \
            and (t_cut is None or t_destroy <= t_cut):
        te = min(t_destroy, m.last+DESTROY_EXTRA_S)
        st = tgt.at(te, EXTRAP_S)
        if st is not None:
            p_end = add(m.p[-1], mul(m.v[-1], te-m.last))
            rel.append((te, sub(st[0], p_end)))
    if not rel:
        return None, None
    best, best_t = norm(rel[0][1]), rel[0][0]
    for (ta, ra), (tb, rb) in zip(rel, rel[1:]):
        if tb-ta > GAP_S:
            continue
        d = sub(rb, ra)
        dd = dot(d, d)
        s = 0.0 if dd < 1e-12 else max(0.0, min(1.0, -dot(ra, d)/dd))
        dist = norm(add(ra, mul(d, s)))
        if dist < best:
            best, best_t = dist, ta+s*(tb-ta)
        if norm(rb) < best:
            best, best_t = norm(rb), tb
    return best, best_t


def judge(m, t_end, tgt, evidence=None):
    """Outcome of one missile against its target dict (or None). ``evidence`` holds replay messages about this
    shooter / target pair around the missile's end: kill_t (kill feed: shooter killed the target with this weapon
    type), damage_t (severe / critical damage message from the shooter), dvm_t (damage-model message on the target)."""
    if tgt is None:
        return dict(outcome="unknown", basis="no_target")
    ev = evidence or {}
    tr, killed = tgt["track"], tgt["killed_s"]
    cut = None if killed is None else killed+1.0        # ignore a wreck falling after the kill-feed entry
    cpa, t_cpa = closest_approach(m, tr, t_end, cut)
    out = dict(cpa_m=None if cpa is None else round(cpa, 1), t_cpa=None if t_cpa is None else round(t_cpa, 3),
               target_killed_at=killed, evidence={k: v for k, v in ev.items() if v is not None} or None)
    near = cpa is not None and cpa <= NEAR_M and t_end-t_cpa <= END_AT_CPA_S+0.5
    if ev.get("kill_t") is not None and (cpa is None or near):
        out.update(outcome="hit", basis="kill_feed")
        return out
    if near and (ev.get("damage_t") is not None or ev.get("dvm_t") is not None):
        out.update(outcome="hit_nokill", basis="damage_msg" if ev.get("damage_t") is not None else "dvm_msg")
        return out
    if cpa is None:
        out.update(outcome="unknown", basis="no_target_track")
        return out
    ended_there = t_end-t_cpa <= END_AT_CPA_S
    kill_ok = killed is not None and t_end-KILL_BEFORE_S <= killed <= t_end+KILL_AFTER_S
    covered = tr.last+EXTRAP_S >= min(t_end, t_cpa+1.0)
    horizon = min(t_end, tr.last+EXTRAP_S, cut if cut is not None else t_end)
    closing_at_cut = t_cpa >= horizon-END_AT_CPA_S      # still closing when the target (track) went away
    if ended_there and cpa <= REACH_M:
        out.update(outcome="hit" if kill_ok else "hit_nokill", basis="cpa+kill" if kill_ok else "cpa")
    elif ended_there and cpa <= NEAR_M and kill_ok:
        out.update(outcome="hit", basis="near+kill")
    elif killed is not None and killed <= t_end+0.5 and closing_at_cut:
        out.update(outcome="target_dead", basis="target_killed_before_reaching")
    elif not covered:
        out.update(outcome="unknown", basis="target_track_ended")
    else:
        out.update(outcome="miss", basis="flyby" if cpa <= REACH_M else "cpa")
    return out


# -- fields of the match export (inspect_match.py) ---------------------------------------------------------------

def parse_kv(text):
    out = {}
    for kv in (text or "").split(";"):
        if kv:
            k, _, v = kv.partition("=")
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


def seeker_target(sync_rows, planes, origin, shooter):
    """Missile seeker block (inspect_match.py rockets[].sync: [t_ms, b7 vx, vy, vz, prefix, bits, px, py, pz, ...]):
    the aircraft nearest to the candidate tracked-target position, voted over the rows where it lies within
    SEEK_MATCH_M. Returns (plane id or None, info)."""
    votes, dists, rows = {}, {}, 0
    for row in sync_rows or ():
        if len(row) < 9 or row[4] is None or any(c is None for c in row[6:9]):
            continue
        rows += 1
        t = row[0]/1000.0
        e, n, u = dagor_to_enu(row[6:9])
        pos = (e-origin[0], n-origin[1], u)
        best = None
        for p in planes:
            if p["id"] == shooter:
                continue
            st = p["track"].at(t, EXTRAP_S)
            if st is not None:
                d = norm(sub(st[0], pos))
                if best is None or d < best[0]:
                    best = (d, p["id"])
        if best and best[0] <= SEEK_MATCH_M:
            votes[best[1]] = votes.get(best[1], 0)+1
            dists.setdefault(best[1], []).append(best[0])
    if not votes:
        return None, dict(rows=rows, votes=0)
    pid = max(votes, key=votes.get)
    share = votes[pid]/sum(votes.values())
    info = dict(rows=rows, votes=votes[pid], share=round(share, 2), median_m=round(statistics.median(dists[pid]), 1))
    if votes[pid] < 3 or share < 0.6:
        return None, info
    return pid, info


def fm_rows_by_eid(raw):
    """inspect_match.py fm_probe -> {eid: [(t_s, merged key/value dict)]} with the discrete state carried forward."""
    out = {}
    for e, block in (raw.get("fm_probe") or {}).items():
        rows, last = [], {}
        for t, disc, cont in block.get("rows") or ():
            if disc:
                last = parse_kv(disc)
            rows.append((t/1000.0, dict(last, **parse_kv(cont))))
        out[int(e)] = rows
    return out


def designations(row, origin):
    """Target designations of one aircraft sync row with a world position: [(type1, type2, ENU position)]."""
    out = []
    for k in range(int(row.get("nt", 0) or 0)):
        p = f"t{k}_"
        if row.get(p+"c") == 1 and p+"6x" in row:
            e, n, u = dagor_to_enu((row[p+"6x"], row[p+"6y"], row[p+"6z"]))
            out.append((int(row.get(p+"1", -1)), int(row.get(p+"2", -1)), (e-origin[0], n-origin[1], u)))
    return out


def designated_target(fm_rows, t0, planes, origin, shooter):
    """The aircraft under the shooter's type (6, 0) designation in the last sync row before launch (that designation
    sits on the target the missile seeker later tracks at 272 of 282 launches in the 2026-10-08 replays)."""
    if not fm_rows:
        return None, None
    ts = [r[0] for r in fm_rows]
    i = bisect.bisect_right(ts, t0)-1
    if i < 0 or t0-ts[i] > 1.5:
        return None, None
    best = None
    for k1, k2, pos in designations(fm_rows[i][1], origin):
        if (k1, k2) != DESIGNATION_TRACK:
            continue
        for p in planes:
            if p["id"] == shooter:
                continue
            st = p["track"].at(t0, EXTRAP_S)
            if st is not None:
                d = norm(sub(st[0], pos))
                if d <= DESIG_MATCH_M and (best is None or d < best[0]):
                    best = (d, p["id"])
    return (best[1], round(best[0], 1)) if best else (None, None)


def classify_cm(cm):
    """Flare or chaff for one countermeasure projectile: spin in its creation record (u12_3) means flare; without
    spin data, a lifetime under 8 s. In the 2026-10-08 replays spinning ones lived 1.7-4.5 s and the others 15-20 s."""
    name = cm.get("launcher") or ""
    if "chaff_only" in name or "only_chaff" in name:
        return "chaff", "launcher"
    spin = cm.get("spin")
    if spin and all(c is not None for c in spin):
        return ("flare" if any(abs(c) > 1e-6 for c in spin) else "chaff"), "spin"
    life = None
    if cm.get("destroyed_at_ms") not in (None, UNKNOWN_MS) and cm.get("created_at_ms") not in (None, UNKNOWN_MS):
        life = (cm["destroyed_at_ms"]-cm["created_at_ms"])/1000.0
    if life is not None:
        return ("flare" if life < 8.0 else "chaff"), "lifetime"
    return "unknown", "none"


# -- conversion --------------------------------------------------------------------------------------------------

def convert(path, out_dir, team_overrides, all_units, name_override=None, taken=None):
    raw = json.loads(path.read_text(encoding="utf-8"))
    units, rockets = raw.get("units") or [], raw.get("rockets") or []
    replay_len = ms(raw.get("replay_length_ms"))

    # origin: middle of the aircraft tracks (Dagor X/Z)
    xs, zs = [], []
    for u in units:
        for s in u.get("track") or ():
            if s.get("xyz") and all(c is not None for c in s["xyz"]):
                xs.append(s["xyz"][0])
                zs.append(s["xyz"][2])
    origin = ((min(xs)+max(xs))/2, (min(zs)+max(zs))/2) if xs else (0.0, 0.0)

    # aircraft
    cand = []
    skipped = dict(ground=0, empty=0, slow=0)
    for u in units:
        name = u.get("unit_name") or ""
        ground = u.get("unit_type") != 2 if "unit_type" in u else "/" in name   # 2 = AircraftType
        if not all_units and ground:                     # tanks, fortifications (tankModels/... names)
            skipped["ground"] += 1
            continue
        tr = Track(u.get("track") or (), origin)
        if len(tr) < 2:
            skipped["empty"] += 1
            continue
        vmax = max(norm(v) for v in tr.v)
        if not all_units and vmax < 25.0:
            skipped["slow"] += 1
            continue
        cand.append((tr.first, u.get("eid") or 0, u, tr))
    cand.sort(key=lambda c: (c[0], c[1]))
    player_team, player_name = {}, {}                    # match export: MPlayer.team / name per owned unit
    for pl in raw.get("players") or ():
        for e in pl.get("owned_unit_eids") or ():
            player_team[e], player_name[e] = pl.get("team"), pl.get("name")
    fm_probe = raw.get("fm_probe") or {}
    planes, n_vel = [], 0
    for pid, (_, _, u, tr) in enumerate(cand):
        killed = ms(u.get("killed_at_ms"))
        eid = u.get("eid")
        real_team, team_src = None, None
        if WT_TEAM.get(player_team.get(eid)) is not None:
            real_team, team_src = WT_TEAM[player_team[eid]], "player"
        elif WT_TEAM.get(u.get("army")) is not None:
            real_team, team_src = WT_TEAM[u["army"]], "army"
        vel = (fm_probe.get(str(eid)) or {}).get("velocity") or ()
        n_vel += tr.set_velocity([(r[0]/1000.0, (r[1], r[3], r[2])) for r in vel])   # Dagor (x, y up, z) -> ENU
        planes.append(dict(id=pid, eid=eid, uid=u.get("uid"), aircraft=u.get("unit_name") or "?",
                           player=player_name.get(eid) or u.get("player_internal_name") or "",
                           owner_pid=u.get("owner_pid"), track=tr, killed_s=killed,
                           destroyed_s=ms(u.get("destroyed_at_ms")), created_s=ms(u.get("created_at_ms")),
                           real_team=real_team, team_src=team_src))
    by_eid = {p["eid"]: p for p in planes if p["eid"] is not None}
    by_id = {p["id"]: p for p in planes}
    fm_rows = fm_rows_by_eid(raw)
    plane_of = lambda e: by_eid[e]["id"] if e in by_eid else None  # noqa: E731

    # replay messages: kill feed, damage messages, damage-model (DVM) messages per aircraft
    kill_feed, damage_msgs = [], []
    for msg in raw.get("battle_messages") or ():
        t_msg = msg["time_ms"]/1000.0
        if msg["kind"] == "KillMessage":
            kill_feed.append(dict(t=t_msg, victim=plane_of(msg.get("offended_eid")),
                                  killer=plane_of(msg.get("offender_eid")), weapon=msg.get("used_weapon") or None,
                                  weapon_type=msg.get("weapon_type"), death_type=msg.get("death_type"),
                                  killer_vehicle=msg.get("offender_vehicle")))
        elif msg["kind"] in ("SevereDamageMessage", "CriticalDamageMessage"):
            damage_msgs.append(dict(t=t_msg, victim=plane_of(msg.get("offended_eid")),
                                    by=plane_of(msg.get("offender_eid")),
                                    severity="severe" if msg["kind"].startswith("Severe") else "critical",
                                    fire=bool(msg.get("is_fire"))))
    dvm = {}
    for row in (raw.get("unit_mpi") or {}).get("rows") or ():
        if row[1] == 16 and row[4] == DVM_HIT_ID and row[3] in by_eid:
            dvm.setdefault(by_eid[row[3]]["id"], []).append(row[0]/1000.0)
    has_feed = "battle_messages" in raw

    # attitude check: decoded nose vs track velocity
    nose_err = []
    for p in planes:
        tr = p["track"]
        for v, n in zip(tr.v, tr.nose):
            if n is not None and norm(v) > 60.0:
                a = angle_deg(v, n)
                if a is not None:
                    nose_err.append(a)
    nose_med = statistics.median(nose_err) if nose_err else None
    attitude_ok = nose_med is not None and nose_med <= ATTITUDE_OK_DEG

    # missiles; countermeasure projectiles (match export "countermeasures", or rockets with such names) -> events
    mlist, cms = [], []
    for r in rockets:
        owner = by_eid.get(r.get("owner_eid"))
        if CM_RE.search(r.get("weapon_name") or ""):
            t_cm = ms(r.get("created_at_ms"))
            if t_cm is None and r.get("track"):
                t_cm = ms(r["track"][0].get("t_ms"))
            if owner is not None and t_cm is not None:
                cms.append((t_cm, owner["id"], "chaff", r.get("weapon_name"), "name"))
            continue
        tr = Track(r.get("track") or (), origin)
        if len(tr) < 2:
            continue
        b7 = [(row[0]/1000.0, (row[1], row[3], row[2])) for row in r.get("sync") or ()
              if len(row) > 3 and None not in row[1:4]]
        tr.set_velocity(b7, max_dt=0.02)                   # missile sync velocity (b7, candidate scale) at its samples
        mlist.append(dict(eid=r.get("eid"), weapon=r.get("weapon_name") or "?", track=tr,
                          shooter=owner["id"] if owner else None, created_s=ms(r.get("created_at_ms")),
                          destroyed_s=ms(r.get("destroyed_at_ms")), sync=r.get("sync"),
                          template=r.get("template")))
    for cm in raw.get("countermeasures") or ():
        owner = by_eid.get(cm.get("owner_eid"))
        t_cm = ms(cm.get("created_at_ms"))
        if owner is not None and t_cm is not None:
            kind, basis = classify_cm(cm)
            cms.append((t_cm, owner["id"], kind, cm.get("launcher"), basis))
    mlist.sort(key=lambda m: (m["track"].first, m["eid"] or 0))
    for uid, m in enumerate(mlist):
        m["uid"] = uid
        d = m["destroyed_s"]
        m["t_end"] = d if d is not None and m["track"].last <= d <= m["track"].last+DESTROY_EXTRA_S else m["track"].last

    # teams: replay fields when present (MPlayer.team, else unit army), inference for the rest
    known = {p["id"]: p["real_team"] for p in planes if p["real_team"] is not None}
    if planes and len(known) == len(planes) and not team_overrides:
        team = dict(known)
        team_info = dict(basis="replay", player=sum(p["team_src"] == "player" for p in planes),
                         army=sum(p["team_src"] == "army" for p in planes), inferred=0)
    else:
        pairs = []
        for m in mlist:
            tid, info = infer_target(m["track"], m["shooter"], planes)
            if tid is not None and m["shooter"] is not None and info["basis"] in ("cpa", "guidance") \
                    and not info.get("ambiguous"):
                pairs.append((m["shooter"], tid, 2 if info["basis"] == "cpa" else 1))
        team, team_info = infer_teams(planes, pairs, team_overrides)
        if known:
            flips = sum(1 for i, k in known.items() if team[i] != k)
            if flips*2 > len(known):                       # align inferred labels with the replay's numbering
                team = {i: 1-k for i, k in team.items()}
            if not team_overrides:
                team.update(known)
            team_info.update(basis=team_info["basis"]+"+replay", replay_known=len(known),
                             inferred=len(planes)-len(known))

    def weapon_match(a, b):
        strip = lambda w: re.sub(r"_default$", "", w or "")  # noqa: E731
        return bool(a) and bool(b) and strip(a) == strip(b)

    # targets: seeker block > kill feed > shooter designation at launch > geometry; then the outcome
    target_src = {}
    for m in mlist:
        geo_id, geo_info = infer_target(m["track"], m["shooter"], planes, team)
        seek_id, seek_info = seeker_target(m.get("sync"), planes, origin, m["shooter"])
        des_id, des_d = designated_target(fm_rows.get(by_id[m["shooter"]]["eid"]) if m["shooter"] is not None
                                          else None, m["track"].first, planes, origin, m["shooter"])
        # the kill feed's weapon name is often not the missile that hit (it names e.g. R-73 for an R-77 kill), so a
        # kill counts for this missile when the shooter killed an aircraft the missile passed within NEAR_M of
        feed_id = None
        for k in sorted(kill_feed, key=lambda k: abs(k["t"]-m["t_end"])):
            if k["killer"] == m["shooter"] and m["shooter"] is not None and k["victim"] is not None \
                    and m["t_end"]-KILL_BEFORE_S <= k["t"] <= m["t_end"]+KILL_FEED_AFTER_S:
                cpa_k, t_k = closest_approach(m["track"], by_id[k["victim"]]["track"], m["t_end"], k["t"]+1.0)
                if cpa_k is not None and cpa_k <= NEAR_M and m["t_end"]-t_k <= END_AT_CPA_S+0.5:
                    feed_id = k["victim"]
                    break
        for src, tid in (("seeker", seek_id), ("kill_feed", feed_id), ("designation", des_id), ("geometry", geo_id)):
            if tid is not None:
                break
        else:
            src, tid = "none", None
        info = dict(geo_info) if src == "geometry" else dict(basis=src)
        info.update(source=src, seeker=dict(seek_info, target=seek_id), designation=des_id, designation_m=des_d,
                    kill_feed=feed_id, geometry=geo_id)
        target_src[src] = target_src.get(src, 0)+1
        m["target"], m["target_info"] = tid, info
        ev = {}
        if tid is not None and m["shooter"] is not None:
            for k in kill_feed:
                if k["killer"] == m["shooter"] and k["victim"] == tid \
                        and m["t_end"]-KILL_BEFORE_S <= k["t"] <= m["t_end"]+KILL_FEED_AFTER_S:
                    ev["kill_t"] = round(k["t"], 3)
                    ev["kill_weapon"] = k["weapon"]
                    ev["kill_weapon_match"] = weapon_match(k["weapon"], m["weapon"])
            for dmsg in damage_msgs:
                if dmsg["by"] == m["shooter"] and dmsg["victim"] == tid \
                        and m["t_end"]-KILL_BEFORE_S <= dmsg["t"] <= m["t_end"]+KILL_AFTER_S:
                    ev["damage_t"] = round(dmsg["t"], 3)
        for t_d in dvm.get(tid, ()):
            if m["t_end"]-0.5 <= t_d <= m["t_end"]+1.5:
                ev["dvm_t"] = round(t_d, 3)
        m["judge"] = judge(m["track"], m["t_end"], by_id.get(tid), ev)

    # kills and deaths: the kill feed when the export has it, else the kill time plus a missile hit
    kills, deaths, events = [], [], []
    if has_feed:
        for k in kill_feed:
            if k["victim"] is None:
                continue
            cands = [m for m in mlist if m["shooter"] == k["killer"] and k["killer"] is not None
                     and m["target"] == k["victim"] and m["judge"]["outcome"] in ("hit", "hit_nokill")
                     and m["t_end"]-KILL_BEFORE_S <= k["t"] <= m["t_end"]+KILL_FEED_AFTER_S]
            hit = min(cands, key=lambda m: (not weapon_match(k["weapon"], m["weapon"]), abs(k["t"]-m["t_end"]))) \
                if cands else None
            dt = k["death_type"]
            cause = "crash" if dt == -1 else "left" if dt == -2 else \
                {3: "missile", 1: "bullet", 2: "bomb", 4: "torpedo"}.get(k["weapon_type"], "killed")
            rec = dict(victim=k["victim"], cause=cause, killer=k["killer"], time_s=round(k["t"], 3),
                       uid=hit["uid"] if hit else None, weapon=k["weapon"], killer_vehicle=k["killer_vehicle"],
                       missile=hit["weapon"] if hit else None, source="kill_feed",
                       friendly_fire=bool(k["killer"] is not None and team.get(k["killer"]) == team.get(k["victim"])))
            if k["victim"] in {d["victim"] for d in deaths}:
                continue
            kills.append(rec)
            deaths.append(rec)
    else:
        for p in planes:
            if p["killed_s"] is None:
                continue
            hits = [m for m in mlist if m["target"] == p["id"] and m["judge"]["outcome"] == "hit"]
            hit = min(hits, key=lambda m: abs(m["t_end"]-p["killed_s"])) if hits else None
            killer = hit["shooter"] if hit else None
            rec = dict(victim=p["id"], cause="missile" if hit else "killed", killer=killer,
                       time_s=round(p["killed_s"], 3), uid=hit["uid"] if hit else None, source="inferred",
                       friendly_fire=bool(killer is not None and team.get(killer) == team.get(p["id"])))
            kills.append(rec)
            deaths.append(rec)
    deaths_by = {d["victim"]: d for d in deaths}
    for p in planes:                                       # the dashboard hides an aircraft from its death on
        if p["id"] in deaths_by and p["killed_s"] is None:
            p["killed_s"] = deaths_by[p["id"]]["time_s"]

    # alive interval of every aircraft in the output
    for p in planes:
        tr = p["track"]
        p["t_out"] = min(tr.last, p["killed_s"]) if p["killed_s"] is not None else tr.last
    t_first = min([p["track"].first for p in planes]+[m["track"].first for m in mlist], default=0.0)
    t_last = max([p["t_out"] for p in planes]+[m["t_end"] for m in mlist], default=0.0)

    # launches per shooter
    launch_times = {}
    for m in mlist:
        if m["shooter"] is not None:
            launch_times.setdefault(m["shooter"], []).append(m["track"].first)
    for v in launch_times.values():
        v.sort()
    weapon_of = {}
    for m in mlist:
        if m["shooter"] is not None:
            weapon_of.setdefault(m["shooter"], []).append(m["weapon"])

    def heading(p, st):
        pos, v, nose = st
        if attitude_ok and nose is not None and math.hypot(nose[0], nose[1]) > 1e-3:
            return compass_deg(nose)
        return compass_deg(v) if math.hypot(v[0], v[1]) > 1.0 else 0.0

    # events
    for m in mlist:
        tr = m["track"]
        t0 = tr.first
        sh = by_id.get(m["shooter"])
        tg = by_id.get(m["target"])
        s_st = sh["track"].at(t0, EXTRAP_S) if sh else None
        t_st = tg["track"].at(t0, EXTRAP_S) if tg else None
        rng = round(norm(sub(t_st[0], tr.p[0]))) if t_st else None
        ob = ob_v = nose_list = None
        if s_st and t_st:
            los = sub(t_st[0], s_st[0])
            ob_v = angle_deg(s_st[1], los)
            if attitude_ok and s_st[2] is not None:
                ob = angle_deg(s_st[2], los)
                nose_list = [round(c, 4) for c in s_st[2]]
        left = None
        if m["shooter"] is not None:
            lt = launch_times[m["shooter"]]
            left = len(lt)-bisect.bisect_right(lt, t0)
        r1 = lambda x: None if x is None else round(x, 1)  # noqa: E731
        events.append(dict(type="event", t=round(t0, 3), kind="launch", uid=m["uid"], shooter=m["shooter"],
                           target=m["target"], missile=m["weapon"], mode=None, range_m=rng, left=left,
                           altitude_m=r1(s_st[0][2]) if s_st else r1(tr.p[0][2]),
                           target_altitude_m=r1(t_st[0][2]) if t_st else None,
                           speed_mps=r1(norm(s_st[1])) if s_st else None,
                           target_speed_mps=r1(norm(t_st[1])) if t_st else None,
                           off_boresight_deg=r1(ob), off_boresight_vel_deg=r1(ob_v), shooter_nose_enu=nose_list,
                           target_basis=m["target_info"], wt_eid=m["eid"], template=m.get("template"),
                           created_at=None if m["created_s"] is None else round(m["created_s"], 3)))
        j = m["judge"]
        res = {"hit": "fuse", "hit_nokill": "fuse", "miss": "miss", "target_dead": "target_dead"}.get(j["outcome"],
                                                                                                     "unknown")
        end_rng = None
        if tg:
            st = tg["track"].at(tr.last, EXTRAP_S)
            end_rng = round(norm(sub(st[0], tr.p[-1])), 1) if st else None
        gap = None
        if tg:
            gap = tg["track"].gaps(m["t_end"]-10.0, m["t_end"])[1]
        v_end = tr.v[-1]
        events.append(dict(type="event", t=round(m["t_end"], 3), kind="missile_end", uid=m["uid"],
                           shooter=m["shooter"], target=m["target"], result=res, miss_m=j.get("cpa_m"),
                           flight_s=round(m["t_end"]-t0, 2), outcome=j["outcome"], outcome_basis=j["basis"],
                           evidence=j.get("evidence"), t_cpa=j.get("t_cpa"), end_range_m=end_rng,
                           target_killed_at=j.get("target_killed_at"),
                           target_max_gap_s=None if gap is None else round(gap, 2),
                           missile_median_dt_s=None if tr.gaps()[0] is None else round(tr.gaps()[0], 3),
                           end_alt_m=round(tr.p[-1][2], 1), end_speed_mps=round(norm(v_end), 1),
                           end_fpa_deg=round(math.degrees(math.atan2(v_end[2], math.hypot(v_end[0], v_end[1]))), 1)
                           if norm(v_end) > 1 else None))
    groups = []                                            # one event per release (both dispensers, same type)
    for t_cm, pid, kind, wname, basis in sorted(cms):
        g = groups[-1] if groups else None
        if g and g["plane"] == pid and g["kind"] == kind and t_cm-g["t"] <= 0.12:
            g["n"] += 1
        else:
            groups.append(dict(t=t_cm, plane=pid, kind=kind, n=1, item=wname, basis=basis))
    groups.sort(key=lambda g: g["t"])
    for g in groups:
        events.append(dict(type="event", t=round(g["t"], 3), kind=g["kind"] if g["kind"] != "unknown" else "chaff",
                           plane=g["plane"], n=g["n"], left="?", item=g["item"], cm_type=g["kind"],
                           cm_basis=g["basis"]))
    for dmsg in damage_msgs:
        if dmsg["victim"] is not None:
            events.append(dict(type="event", t=round(dmsg["t"], 3), kind="damage", plane=dmsg["victim"],
                               by=dmsg["by"], severity=dmsg["severity"], fire=dmsg["fire"]))
    for d in kills:
        events.append(dict(type="event", t=d["time_s"], kind="kill", **d))
    for p in planes:
        d = deaths_by.get(p["id"])
        if d is not None:
            st = p["track"].at(d["time_s"], EXTRAP_S) or p["track"].at(p["track"].last, 0.0)
            events.append(dict(type="event", t=d["time_s"], kind="death", plane=p["id"], cause=d["cause"],
                               killer=d["killer"], altitude_m=round(st[0][2], 1), speed_mps=round(norm(st[1]), 1)))
        elif p["track"].last < t_last-5.0:                # gone from the replay without a kill-feed entry
            st = p["track"].at(p["track"].last, 0.0)
            events.append(dict(type="event", t=round(p["track"].last, 3), kind="death", plane=p["id"],
                               cause="left_replay", killer=None, altitude_m=round(st[0][2], 1),
                               speed_mps=round(norm(st[1]), 1)))
            p["left_replay"] = True

    # frames
    frames = []
    t = math.floor(t_first/FRAME_DT)*FRAME_DT
    n_frames = 0
    while t <= t_last+1e-9:
        rows = []
        for p in planes:
            tr = p["track"]
            if t < tr.first-1e-9 or t > p["t_out"]+1e-9:
                continue
            st = tr.at(t)
            if st is None:
                continue
            pos, v, _ = st
            lt = launch_times.get(p["id"], [])
            left = len(lt)-bisect.bisect_right(lt, t)
            rows.append([p["id"], round(pos[0], 1), round(pos[1], 1), round(pos[2], 1), round(v[0], 1),
                         round(v[1], 1), round(v[2], 1), round(heading(p, st), 1) % 360.0, left, "?", ""])
        mrows = []
        for m in mlist:
            tr = m["track"]
            if t < tr.first-1e-9 or t > tr.last+1e-9:
                continue
            pos, v, _ = tr.at(t)
            mrows.append([m["uid"], m["shooter"], m["target"], round(pos[0], 1), round(pos[1], 1), round(pos[2], 1),
                          round(v[0], 1), round(v[1], 1), round(v[2], 1), round(compass_deg(v), 1) % 360.0,
                          round(t-tr.first, 2), 0, 0])
        frames.append(dict(type="frame", t=round(t, 3), planes=rows, missiles=mrows))
        n_frames += 1
        t += FRAME_DT

    # header
    extent = 0.0
    for p in planes:
        for pos in p["track"].p:
            extent = max(extent, abs(pos[0]), abs(pos[1]))
    for m in mlist:
        for pos in m["track"].p:
            extent = max(extent, abs(pos[0]), abs(pos[1]))
    map_half = max(10000.0, math.ceil(extent*1.05/1000.0)*1000.0)
    hplanes = []
    for p in planes:
        weapons = weapon_of.get(p["id"], [])
        main = max(set(weapons), key=weapons.count) if weapons else None
        med, mx = p["track"].gaps()
        hplanes.append(dict(id=p["id"], team=team[p["id"]], aircraft=p["aircraft"],
                            name=f"{p['player'] or '?'}:{p['aircraft']}", archetype=p["player"] or None,
                            skill="AI" if p["team_src"] == "army" else None,
                            missile=main, missiles=len(weapons), chaff="?", rcs_ratio=None, radar=None, rwr=None,
                            mass_kg=None, script=None,
                            wt=dict(eid=p["eid"], uid=p["uid"], player=p["player"], owner_pid=p["owner_pid"],
                                    samples=len(p["track"]), median_dt_s=None if med is None else round(med, 3),
                                    max_gap_s=None if mx is None else round(mx, 2),
                                    first_s=round(p["track"].first, 3), last_s=round(p["track"].last, 3),
                                    killed_at_s=p["killed_s"], left_replay=bool(p.get("left_replay")),
                                    team_source=p["team_src"] or "inferred")))
    try:
        source_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        source_sha = None
    outcomes, outcome_basis, cm_counts = {}, {}, {}
    for m in mlist:
        outcomes[m["judge"]["outcome"]] = outcomes.get(m["judge"]["outcome"], 0)+1
        outcome_basis[m["judge"]["basis"]] = outcome_basis.get(m["judge"]["basis"], 0)+1
    for g in groups:
        cm_counts[g["kind"]] = cm_counts.get(g["kind"], 0)+g["n"]
    header = dict(type="header", version=1, seed=None, map_half_m=map_half, tick_s=None, frame_dt_s=FRAME_DT,
                  time_limit_s=round(replay_len if replay_len is not None else t_last, 3), planes=hplanes,
                  plane_columns=PLANE_COLUMNS, missile_columns=MISSILE_COLUMNS,
                  source=dict(kind="wt_replay", importer_version=IMPORTER_VERSION, export_file=str(path),
                              export_sha256=source_sha, replay=raw.get("source"),
                              replay_sha256=raw.get("source_sha256"), backend_revision=raw.get("backend_revision"),
                              replay_length_s=replay_len, origin_dagor_xz=[round(origin[0], 2), round(origin[1], 2)],
                              frame="ENU m: x=Dagor X, y=Dagor Z, z=Dagor Y (up), minus origin_dagor_xz; "
                                    "t = replay packet time, s",
                              teams=team_info, skipped_units=skipped,
                              attitude=dict(ok=attitude_ok, median_nose_vs_velocity_deg=None if nose_med is None
                                            else round(nose_med, 2), samples=len(nose_err),
                                            heading_from="nose" if attitude_ok else "velocity"),
                              thresholds=dict(reach_m=REACH_M, near_m=NEAR_M, end_at_cpa_s=END_AT_CPA_S,
                                              kill_before_s=KILL_BEFORE_S, kill_after_s=KILL_AFTER_S),
                              missile_outcomes=outcomes, outcome_basis=outcome_basis, target_source=target_src,
                              countermeasures=cm_counts,
                              fields=dict(
                                  team=team_info["basis"],
                                  kills="replay kill feed (KillMessage: killer, victim, weapon, death/weapon type)"
                                  if has_feed else "inferred: kill time + missile hit",
                                  damage=f"replay messages: {len(damage_msgs)} severe/critical, "
                                         f"{sum(len(v) for v in dvm.values())} DVM 0x{DVM_HIT_ID:x}"
                                  if has_feed else "none",
                                  countermeasures="replay projectiles (time, owner); flare/chaff from spin/lifetime "
                                                  "(candidate)" if raw.get("countermeasures") is not None else
                                                  "none in this export",
                                  target="seeker block (candidate layout) > kill feed > shooter designation (6,0) "
                                         "(candidate) > geometry; see launch target_basis.source",
                                  aircraft_velocity=f"decoded sync velocity at {n_vel} samples (candidate scale), "
                                                    "else finite differences",
                                  outcome="kill feed / damage messages / DVM messages + closest approach; "
                                          "see missile_end outcome_basis"),
                              unknown=["radar / RWR display state (sensor and designation fields exported only as "
                                       "candidates)", "seeker/datalink on/off", "loadout (missiles column = launches "
                                       "seen)", "terrain (ground impacts not identified)"]))

    # end record
    alive_end = [0, 0]
    for p in planes:
        if p["killed_s"] is None and not p.get("left_replay"):
            alive_end[team[p["id"]]] += 1
    summary_planes = []
    for p in planes:
        d = deaths_by.get(p["id"])
        summary_planes.append(dict(id=p["id"], alive=d is None and not p.get("left_replay"),
                                   kills=sum(1 for k in kills if k["killer"] == p["id"]), assists=0,
                                   launches=len(weapon_of.get(p["id"], [])), chaff_used=None, missiles_left=None,
                                   death=None if d is None else dict(cause=d["cause"], time_s=d["time_s"],
                                                                     killer=d["killer"]),
                                   fm_faults=0))
    end = dict(type="end", reason="replay_end", t=round(t_last, 3),
               result=dict(teams_alive=alive_end, kills=kills, deaths=deaths, launches=len(mlist),
                           planes=summary_planes))

    # write: header, frames and events in time order (an event goes before the frame at or after it), end
    name = name_override or replay_name(raw, path)
    if taken is not None:                                  # two exports of one replay in one run: keep both
        if name in taken:
            name = f"{name}_{re.sub(r'[^A-Za-z0-9._-]+', '_', path.parent.name)}"
        taken.add(name)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir/f"{name}.jsonl"
    events.sort(key=lambda e: e["t"])
    with out.open("w", encoding="utf-8") as f:
        f.write(json.dumps(header, separators=(",", ":"), allow_nan=False)+"\n")
        ei = 0
        for fr in frames:
            while ei < len(events) and events[ei]["t"] <= fr["t"]:
                f.write(json.dumps(events[ei], separators=(",", ":"), allow_nan=False)+"\n")
                ei += 1
            f.write(json.dumps(fr, separators=(",", ":"), allow_nan=False)+"\n")
        for e in events[ei:]:
            f.write(json.dumps(e, separators=(",", ":"), allow_nan=False)+"\n")
        f.write(json.dumps(dict(type="event", t=end["t"], kind="end", reason="replay_end"),
                           separators=(",", ":"))+"\n")
        f.write(json.dumps(end, separators=(",", ":"), allow_nan=False)+"\n")
    return out, dict(aircraft=len(planes), missiles=len(mlist), frames=n_frames, outcomes=outcomes, teams=team_info,
                     targets=target_src, outcome_basis=outcome_basis, countermeasures=cm_counts, kills=len(kills),
                     attitude_ok=attitude_ok, nose_vs_velocity_deg=None if nose_med is None else round(nose_med, 1),
                     t=[round(t_first, 2), round(t_last, 2)], skipped=skipped)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="*", default=[str(p) for p in DEFAULT_IN if p.exists()],
                    help="raw_extended.json files, export directories or folders to search "
                         "(default outputs/wt_replays and outputs/wt_replay_exports)")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT), help="output folder (default outputs/engagements/wt_real)")
    ap.add_argument("--name", help="output file name (single input only)")
    ap.add_argument("--team", action="append", default=[], metavar="NAME=0|1",
                    help="fix a team: player name, aircraft name or plane id (repeatable)")
    ap.add_argument("--all-units", action="store_true", help="keep ground units and units that never move")
    args = ap.parse_args(argv)
    found, wrpl = find_inputs(args.inputs)
    if wrpl and not found:
        print(f"{len(wrpl)} .wrpl file(s) found ({', '.join(p.name for p in wrpl[:5])}"
              f"{', ...' if len(wrpl) > 5 else ''}): these need the replay inspector backend first "
              "(tools/replay_inspector/inspect_match.py, with the game's aces/game/char .vromfs.bin); point this script "
              "at its output directories (match_export.json).",
              file=sys.stderr)
    if not found:
        print("no match_export.json / raw_extended.json found", file=sys.stderr)
        return 1
    if args.name and len(found) > 1:
        ap.error("--name needs a single input")
    taken = set()
    for path in found:
        out, info = convert(path, Path(args.out_dir), args.team, args.all_units, args.name, taken)
        print(f"{path} -> {out}")
        print("   " + json.dumps(info))
    return 0


if __name__ == "__main__":
    sys.exit(main())
