#!/usr/bin/env python3
"""Per-missile table and hit-rate summaries from replay JSONL files, to calibrate the notch / look-up model.

Reads replays in our engagement format: the War Thunder imports written by scripts/wt_replay_import.py
(outputs/engagements/wt_real, the default) and also the simulator's own replays, so real and simulated shots can be
put side by side (summaries are grouped by source kind unless --group-by says otherwise).

Per missile:
  shooter / target aircraft and their altitudes at launch, launch range, altitude difference, aspect;
  off-boresight at launch: from the decoded nose (WT import with trusted attitude, ``ob_src`` = nose), else from the
    shooter's velocity vector (``ob_src`` = vel);
  flight time and closest approach (CPA);
  outcome: hit (reached the target and it was killed), hit_nokill (reached it, no kill-feed entry), miss,
    target_dead (target killed by something else first), unknown. Simulator results map fuse -> hit,
    lifetime/ground -> miss, target_dead -> target_dead;
  terminal phase = the last --terminal-s seconds (default 10) up to the closest approach (for a miss the missile may
    fly on for a long time after it), and the seeker phase = range <= --seeker-range-m (default 16 km):
      look_deg   elevation of the missile->target line above the missile's horizontal (+ = look-up, - = look-down);
      vr_mps     target velocity along that line, v_target . LOS (the ground-relative Doppler of the target as the
                 seeker sees it; |vr| <= 50 m/s is the notch band of our simulator);
      notch_frac share of terminal samples in the band; full_frac share also >= 2 deg below the horizon (the
                 simulator's clutter-notch condition); beamed = notch_frac >= --beam-frac (default 0.5);
  countermeasures: chaff (and, for WT imports, flare) projectiles released by the target in the terminal window,
    the first one's time relative to the closest approach, and chaff releases before the window;
  seeker_family: radar / ir / other from the weapon name (simulator missiles count as radar); summaries use radar
    missiles only unless --seeker says otherwise;
  outcome provenance: target_basis (seeker / kill_feed / designation / geometry ...) and outcome_basis from the import;
  ground impact: simulator result "ground"; for WT misses only a candidate (no terrain heights): ended faster than
    250 m/s, diving more than 10 deg, below 1500 m.

Summaries (hit rate = (hit + hit_nokill) / (hit + hit_nokill + miss), Wilson 95 % interval): overall; by terminal look
class (up >= +2 deg, level, down <= -2 deg); by beamed; by the simulator's full notch condition; look x beamed; by
target chaff in the terminal window, look x chaff, beamed x chaff, look x beamed x chaff; by launch range; by
target-minus-shooter altitude at launch; by target altitude at CPA, look x target altitude; and the share of misses
ending as (candidate) ground impacts by look. WT missiles whose target could not be told apart (basis "ambiguous" or
"none") are listed but left out of the summaries unless --all-targets.

Run:  .venv/bin/python scripts/wt_replay_missiles.py [replays or folders ...] [--csv out.csv] [--json out.json]
Pure standard library.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = ROOT / "outputs" / "engagements" / "wt_real"
DEFAULT_OUT = ROOT / "outputs" / "wt_replay_analysis"
SIM_RESULT = {"fuse": "hit", "lifetime": "miss", "ground": "miss", "target_dead": "target_dead"}
GOOD_OUTCOMES = ("hit", "hit_nokill", "miss")
# seeker family from the weapon name (WT blk names and our missile ids)
RADAR_RE = re.compile(r"aim_?120|aim7|aim_?54|r_?77|rvv|r_27e?r|r_27r|r_24r|r_23r|pl_?12|pl_?15|sd_?10|mica_em|aam4|"
                      r"derby|darter|fakour|super_530d|530d|skyflash|aspide|meteor|r_?37|aim_?260", re.I)
IR_RE = re.compile(r"aim_?9|r_?73|r_27e?t|r_60|r_13|pl_?8|pl_?5|pl_?9|pl_?10|magic|mica_ir|iris|asraam|python|pyton|"
                   r"aam3|aam5|r_?74|sidewinder|kd_88_missile_ir", re.I)
NON_AAM_RE = re.compile(r"kh_|agm|kd_88|paveway|gbu|165mm|_vt_", re.I)


def seeker_family(name):
    if not name:
        return "unknown"
    if NON_AAM_RE.search(name) and not re.search(r"missile_ir", name):
        return "other"
    if RADAR_RE.search(name):
        return "radar"
    if IR_RE.search(name):
        return "ir"
    return "unknown"


def sub(a, b):
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])


def dot(a, b):
    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]


def norm(a):
    return math.sqrt(dot(a, a))


def angle_deg(a, b):
    na, nb = norm(a), norm(b)
    if na < 1e-9 or nb < 1e-9:
        return None
    return math.degrees(math.acos(max(-1.0, min(1.0, dot(a, b)/(na*nb)))))


def look_deg(los):
    return math.degrees(math.atan2(los[2], math.hypot(los[0], los[1])))


class Series:
    """Rows of one object from the frames: times and (pos, vel) tuples, linearly interpolated."""

    def __init__(self):
        self.t, self.p, self.v, self.extra = [], [], [], []

    def add(self, t, p, v, extra=None):
        self.t.append(t)
        self.p.append(p)
        self.v.append(v)
        self.extra.append(extra)

    def at(self, t, slack=1e-6):
        """(pos, vel) at t; within ``slack`` of either end the end row is used, further out None."""
        ts = self.t
        if not ts or t < ts[0]-slack or t > ts[-1]+slack:
            return None
        t = min(max(t, ts[0]), ts[-1])
        i = bisect.bisect_left(ts, t)
        if ts[i] == t:
            return self.p[i], self.v[i]
        a, b = i-1, i
        f = (t-ts[a])/(ts[b]-ts[a])
        lerp = lambda x, y: tuple(xa+(ya-xa)*f for xa, ya in zip(x, y))  # noqa: E731
        return lerp(self.p[a], self.p[b]), lerp(self.v[a], self.v[b])


def load(path):
    header, events = None, []
    planes, missiles = {}, {}
    pcol = mcol = None
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                o = json.loads(line)
            except ValueError:
                continue
            typ = o.get("type")
            if typ == "header":
                header = o
                pcol = {n: i for i, n in enumerate(o["plane_columns"])}
                mcol = {n: i for i, n in enumerate(o["missile_columns"])}
            elif typ == "frame" and pcol is not None:
                t = o["t"]
                for r in o.get("planes") or ():
                    planes.setdefault(r[pcol["id"]], Series()).add(
                        t, (r[pcol["x"]], r[pcol["y"]], r[pcol["z"]]), (r[pcol["vx"]], r[pcol["vy"]], r[pcol["vz"]]))
                for r in o.get("missiles") or ():
                    missiles.setdefault(r[mcol["uid"]], Series()).add(
                        t, (r[mcol["x"]], r[mcol["y"]], r[mcol["z"]]), (r[mcol["vx"]], r[mcol["vy"]], r[mcol["vz"]]),
                        r[mcol["target"]])
            elif typ == "event":
                events.append(o)
    if header is None:
        raise ValueError("no header line")
    return header, planes, missiles, events


def analyse_file(path, args):
    header, planes, missiles, events = load(path)
    src = header.get("source") or {}
    kind = src.get("kind") or "sim"
    meta = {p["id"]: p for p in header.get("planes") or ()}
    launch = {e["uid"]: e for e in events if e.get("kind") == "launch"}
    mend = {e["uid"]: e for e in events if e.get("kind") == "missile_end"}
    chaff, flare = {}, {}
    for e in events:
        if e.get("kind") == "chaff":
            chaff.setdefault(e.get("plane"), []).append((e["t"], e.get("n", 1)))
        elif e.get("kind") == "flare":
            flare.setdefault(e.get("plane"), []).append((e["t"], e.get("n", 1)))
    has_chaff = bool(chaff) or bool(flare) or (src.get("countermeasures") is not None)
    kill_src = (src.get("fields") or {}).get("kills", "simulator" if kind != "wt_replay" else "inferred")
    weapon_re = re.compile(args.weapon) if args.weapon else None
    rows = []
    for uid in sorted(set(launch) | set(missiles)):
        L, E, ms = launch.get(uid), mend.get(uid), missiles.get(uid)
        if L is None or ms is None or len(ms.t) < 2:
            continue
        weapon = L.get("missile") or "?"
        if weapon_re and not weapon_re.search(weapon):
            continue
        shooter = L.get("shooter")
        target = E.get("target") if E and E.get("target") is not None else L.get("target")
        if target is None and ms.extra[-1] is not None:
            target = ms.extra[-1]
        t0 = L["t"]
        t_end = E["t"] if E else ms.t[-1]
        row = dict(file=path.stem, source=kind, uid=uid, missile=weapon,
                   seeker_family="radar" if kind != "wt_replay" else seeker_family(weapon), shooter=shooter,
                   shooter_ac=(meta.get(shooter) or {}).get("aircraft"), target=target,
                   target_ac=(meta.get(target) or {}).get("aircraft"), t_launch=round(t0, 2),
                   flight_s=round(E["flight_s"], 2) if E and E.get("flight_s") is not None else round(t_end-t0, 2))
        # outcome
        if E is None:
            outcome, basis = "unknown", "no_missile_end"
        elif "outcome" in E:
            outcome, basis = E["outcome"], E.get("outcome_basis")
        else:
            outcome, basis = SIM_RESULT.get(E.get("result"), "unknown"), f"sim:{E.get('result')}"
        row.update(outcome=outcome, outcome_basis=basis)
        tb = L.get("target_basis") or {}
        tq = "sim" if kind != "wt_replay" else ("ambiguous" if tb.get("ambiguous") else tb.get("basis", "none"))
        row["target_basis"] = tq
        row["kill_source"] = kill_src if outcome == "hit" else None
        if E is not None and kind == "wt_replay":
            row.update(end_alt_m=E.get("end_alt_m"), end_speed_mps=E.get("end_speed_mps"),
                       end_fpa_deg=E.get("end_fpa_deg"))
        elif E is not None and E.get("result") == "ground":
            row["ground_impact"] = "sim"
        sh, tg = planes.get(shooter), planes.get(target)
        # launch geometry
        s_st = sh.at(t0, 0.13) if sh else None
        t_st = tg.at(t0, 0.13) if tg else None
        if s_st is None and sh:
            s_st = sh.at(min(max(t0, sh.t[0]), sh.t[-1]))
        if t_st is None and tg and tg.t[0] <= t0+0.5:
            t_st = tg.at(min(max(t0, tg.t[0]), tg.t[-1]))
        alt_s = L.get("altitude_m", s_st[0][2] if s_st else None)
        alt_t = L.get("target_altitude_m") if L.get("target_altitude_m") is not None else (t_st[0][2] if t_st else None)
        rng = L.get("range_m")
        if rng is None and s_st and t_st:
            rng = norm(sub(t_st[0], s_st[0]))
        ob, ob_src, aspect = L.get("off_boresight_deg"), "nose", None
        if s_st and t_st:
            los = sub(t_st[0], s_st[0])
            if ob is None:
                ob, ob_src = angle_deg(s_st[1], los), "vel"
            aspect = angle_deg(t_st[1], sub(s_st[0], t_st[0]))     # 0 = target nose-on to the shooter
        elif ob is None:
            ob_src = None
        row.update(shooter_alt_m=r0(alt_s), target_alt_m=r0(alt_t),
                   dalt_m=r0(alt_t-alt_s) if alt_t is not None and alt_s is not None else None,
                   launch_range_km=None if rng is None else round(rng/1000.0, 2), off_boresight_deg=r1(ob),
                   ob_src=ob_src if ob is not None else None, aspect_deg=r1(aspect))
        # missile-target geometry over the flight (up to the target's last frame)
        geo = []
        if tg:
            for t, p, v in zip(ms.t, ms.p, ms.v):
                st = tg.at(t)
                if st is None:
                    continue
                los = sub(st[0], p)
                r = norm(los)
                if r < 1e-6:
                    continue
                u = tuple(c/r for c in los)
                geo.append(dict(t=t, r=r, look=look_deg(los), vr=dot(st[1], u),
                                vc=-dot(sub(st[1], v), u), zm=p[2], zt=st[0][2]))
        if not geo:
            row.update(note="no target geometry")
            rows.append(row)
            continue
        i_min = min(range(len(geo)), key=lambda i: geo[i]["r"])
        t_ref = geo[i_min]["t"]
        if E and E.get("t_cpa") is not None and geo[0]["t"] <= E["t_cpa"] <= geo[-1]["t"]+0.25:
            t_ref = E["t_cpa"]
        cpa = E.get("miss_m") if E and E.get("miss_m") is not None and kind == "wt_replay" else geo[i_min]["r"]
        term = [g for g in geo if t_ref-args.terminal_s-1e-6 <= g["t"] <= t_ref+1e-6]
        seek = [g for g in geo if g["r"] <= args.seeker_range_m and g["t"] <= t_ref+1e-6]
        band = lambda g: abs(g["vr"]) <= args.notch_mps  # noqa: E731
        down = lambda g: g["look"] <= -args.notch_look_deg  # noqa: E731
        last3 = [g for g in term if g["t"] >= t_ref-3.0] or term[-1:]
        n = len(term)
        notch_frac = sum(band(g) for g in term)/n if n else None
        full_frac = sum(band(g) and down(g) for g in term)/n if n else None
        look_mean = sum(g["look"] for g in term)/n if n else None
        row.update(cpa_m=r1(cpa), t_cpa=round(t_ref, 2), term_samples=n,
                   look_seeker_on_deg=r1(seek[0]["look"]) if seek else None,
                   seeker_on_range_km=round(seek[0]["r"]/1000.0, 2) if seek else None,
                   look_term_mean_deg=r1(look_mean), look_term_min_deg=r1(min(g["look"] for g in term)) if n else None,
                   look_term_max_deg=r1(max(g["look"] for g in term)) if n else None,
                   look_at_cpa_deg=r1(term[-1]["look"]) if n else None,
                   vr_abs_mean_mps=r1(sum(abs(g["vr"]) for g in term)/n) if n else None,
                   vr_abs_min_mps=r1(min(abs(g["vr"]) for g in term)) if n else None,
                   vr_abs_last3_mps=r1(sum(abs(g["vr"]) for g in last3)/len(last3)) if last3 else None,
                   closing_mean_mps=r1(sum(g["vc"] for g in term)/n) if n else None,
                   notch_frac=r2(notch_frac), full_notch_frac=r2(full_frac),
                   missile_alt_cpa_m=r0(term[-1]["zm"]) if n else None,
                   target_alt_cpa_m=r0(term[-1]["zt"]) if n else None)
        lm = look_mean
        row["look_class"] = None if lm is None else "up" if lm >= args.notch_look_deg else \
            "down" if lm <= -args.notch_look_deg else "level"
        row["beamed"] = None if notch_frac is None else notch_frac >= args.beam_frac
        row["sim_notch"] = None if full_frac is None else full_frac >= args.beam_frac
        if has_chaff and target is not None:
            win = lambda lst: [(t, k) for t, k in lst if t_ref-args.terminal_s <= t <= t_ref+0.5]  # noqa: E731
            ch, fl = win(chaff.get(target, ())), win(flare.get(target, ()))
            row["chaff_term"] = sum(k for _, k in ch)
            row["flare_term"] = sum(k for _, k in fl) if kind == "wt_replay" else None
            row["chaff_first_s"] = round(min(t for t, _ in ch)-t_ref, 2) if ch else None   # relative to the CPA
            row["chaff_releases_term"] = len(ch)
            before = [t for t, _ in chaff.get(target, ()) if t0 <= t < t_ref-args.terminal_s]
            row["chaff_before_term"] = len(before)
        else:
            row["chaff_term"] = "n/a"
        if kind == "wt_replay" and outcome == "miss" and row.get("end_speed_mps") is not None:
            # candidate ground impact: no terrain heights in the export, so: ended fast (> 250 m/s), diving
            # (< -10 deg) and low (< 1500 m) - a missile that times out has slowed down first
            row["ground_impact"] = "candidate" if (row["end_speed_mps"] > 250 and (row.get("end_fpa_deg") or 0) < -10
                                                   and (row.get("end_alt_m") or 1e9) < 1500) else "no"
        if E and E.get("target_max_gap_s") is not None:
            row["target_max_gap_s"] = E["target_max_gap_s"]
        rows.append(row)
    return rows, dict(file=str(path), kind=kind, attitude=(src.get("attitude") or {}).get("ok"),
                      teams=(src.get("teams") or {}).get("basis"), missiles=len(rows))


def r0(x):
    return None if x is None else round(x)


def r1(x):
    return None if x is None else round(x, 1)


def r2(x):
    return None if x is None else round(x, 2)


def wilson(k, n, z=1.96):
    if n == 0:
        return None, None
    p = k/n
    d = 1+z*z/n
    c = (p+z*z/(2*n))/d
    h = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return max(0.0, c-h), min(1.0, c+h)


def usable(row, args):
    if row.get("outcome") not in GOOD_OUTCOMES or row.get("term_samples", 0) < 4:
        return False
    if args.seeker != "all" and row.get("seeker_family") != args.seeker:
        return False
    if row.get("flight_s", 0) < args.min_flight_s:
        return False
    if row["source"] == "wt_replay" and not args.all_targets and row.get("target_basis") in ("none", "ambiguous"):
        return False
    return True


def bin_range(km):
    if km is None:
        return None
    for lo, hi in ((0, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 50)):
        if km < hi:
            return f"{lo:>2}-{hi} km"
    return "50+ km"


def bin_dalt(m):
    if m is None:
        return None
    if m >= 2000:
        return "tgt >=2 km above"
    if m >= 500:
        return "tgt 0.5-2 km above"
    if m > -500:
        return "within 0.5 km"
    if m > -2000:
        return "tgt 0.5-2 km below"
    return "tgt >=2 km below"


def bin_alt(m):
    if m is None:
        return None
    for hi, name in ((1000, "<1 km"), (3000, "1-3 km"), (6000, "3-6 km"), (9000, "6-9 km")):
        if m < hi:
            return name
    return ">=9 km"


SUMMARIES = [
    ("overall", lambda r: "all"),
    ("terminal look (mean)", lambda r: r.get("look_class")),
    ("beamed (|vr|<=band for >= beam-frac of terminal)", lambda r: None if r.get("beamed") is None
     else "beamed" if r["beamed"] else "not beamed"),
    ("sim notch (beamed AND look-down)", lambda r: None if r.get("sim_notch") is None
     else "in sim notch" if r["sim_notch"] else "outside"),
    ("look x beamed", lambda r: None if r.get("look_class") is None or r.get("beamed") is None
     else f"{r['look_class']:>5} / {'beamed' if r['beamed'] else 'not beamed'}"),
    ("target chaff in terminal window", lambda r: None if not isinstance(r.get("chaff_term"), int)
     else "chaff" if r["chaff_term"] > 0 else "no chaff"),
    ("look x chaff", lambda r: None if r.get("look_class") is None or not isinstance(r.get("chaff_term"), int)
     else f"{r['look_class']:>5} / {'chaff' if r['chaff_term'] > 0 else 'no chaff'}"),
    ("beamed x chaff", lambda r: None if r.get("beamed") is None or not isinstance(r.get("chaff_term"), int)
     else f"{'beamed' if r['beamed'] else 'not beamed'} / {'chaff' if r['chaff_term'] > 0 else 'no chaff'}"),
    ("look x beamed x chaff", lambda r: None if r.get("look_class") is None or r.get("beamed") is None
     or not isinstance(r.get("chaff_term"), int) else
     f"{r['look_class']:>5} / {'beamed' if r['beamed'] else 'not beamed'} / {'chaff' if r['chaff_term'] > 0 else 'no chaff'}"),
    ("launch range", lambda r: bin_range(r.get("launch_range_km"))),
    ("target - shooter altitude at launch", lambda r: bin_dalt(r.get("dalt_m"))),
    ("target altitude at CPA", lambda r: bin_alt(r.get("target_alt_cpa_m"))),
    ("look x target altitude at CPA", lambda r: None if r.get("look_class") is None
     else f"{r['look_class']:>5} / {bin_alt(r.get('target_alt_cpa_m'))}"),
    ("miss end (ground impact candidate) by look", lambda r: None if r.get("outcome") != "miss"
     or r.get("ground_impact") is None or r.get("look_class") is None
     else f"{r['look_class']:>5} / {'ground?' if r['ground_impact'] in ('candidate', 'sim') else 'air'}"),
]


def summarise(rows, args):
    groups = {}
    for r in rows:
        key = "all" if args.group_by == "none" else r["source"] if args.group_by == "source" else r["file"]
        groups.setdefault(key, []).append(r)
    out = {}
    for g, rs in groups.items():
        use = [r for r in rs if usable(r, args)]
        excl = {}
        for r in rs:
            if not usable(r, args):
                why = r.get("outcome") if r.get("outcome") not in GOOD_OUTCOMES else \
                    f"seeker {r.get('seeker_family')}" if args.seeker != "all" and r.get("seeker_family") != args.seeker \
                    else \
                    f"target {r.get('target_basis')}" if r["source"] == "wt_replay" and r.get("target_basis") in (
                        "none", "ambiguous") else "short/no geometry"
                excl[why] = excl.get(why, 0)+1
        tables = {}
        for name, key in SUMMARIES:
            cells = {}
            for r in use:
                k = key(r)
                if k is None:
                    continue
                c = cells.setdefault(k, dict(n=0, hit=0, hit_nokill=0, miss=0, chaff=0))
                c["n"] += 1
                c[r["outcome"]] += 1
                if isinstance(r.get("chaff_term"), int) and r["chaff_term"] > 0:
                    c["chaff"] += 1
            for c in cells.values():
                k = c["hit"]+c["hit_nokill"]
                lo, hi = wilson(k, c["n"])
                c.update(rate=round(k/c["n"], 3), ci95=[round(lo, 3), round(hi, 3)])
            tables[name] = dict(sorted(cells.items()))
        out[g] = dict(missiles=len(rs), used=len(use), excluded=excl, tables=tables)
    return out


COLS = [("file", 18), ("uid", 4), ("missile", 16), ("shooter_ac", 14), ("target_ac", 14), ("shooter_alt_m", 6),
        ("target_alt_m", 6), ("launch_range_km", 6), ("off_boresight_deg", 5), ("flight_s", 6), ("outcome", 11),
        ("cpa_m", 7), ("look_term_mean_deg", 6), ("look_at_cpa_deg", 6), ("vr_abs_mean_mps", 6),
        ("vr_abs_last3_mps", 6), ("notch_frac", 5), ("full_notch_frac", 5), ("chaff_term", 5), ("flare_term", 5),
        ("target_basis", 11)]
HEAD = ["file", "uid", "missile", "shooter", "target", "alt_s", "alt_t", "R_km", "OB", "tof", "outcome", "cpa_m",
        "look", "lookE", "|vr|", "|vr|3", "nf", "full", "chaf", "flar", "tgt"]


def fmt(v, w):
    s = "-" if v is None else f"{v}"
    return s[:w].rjust(w) if isinstance(v, (int, float)) else s[:w].ljust(w)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="*", default=[str(DEFAULT_IN)], help="replay .jsonl files or folders (recursive)")
    ap.add_argument("--terminal-s", type=float, default=10.0, help="terminal window before the closest approach")
    ap.add_argument("--seeker-range-m", type=float, default=16000.0, help="range for the seeker-on look angle")
    ap.add_argument("--notch-mps", type=float, default=50.0, help="|v_target . LOS| band of the notch")
    ap.add_argument("--notch-look-deg", type=float, default=2.0, help="look-down needed for the clutter notch")
    ap.add_argument("--beam-frac", type=float, default=0.5, help="share of the terminal window in the band = beamed")
    ap.add_argument("--min-flight-s", type=float, default=2.0)
    ap.add_argument("--weapon", help="regex on the missile name")
    ap.add_argument("--all-targets", action="store_true", help="also count WT missiles with an ambiguous target")
    ap.add_argument("--seeker", choices=("radar", "ir", "all"), default="radar",
                    help="seeker family in the summaries (from the weapon name; simulator missiles count as radar)")
    ap.add_argument("--group-by", choices=("source", "file", "none"), default="source")
    ap.add_argument("--csv", help=f"per-missile CSV (default {DEFAULT_OUT.relative_to(ROOT)}/missiles.csv)")
    ap.add_argument("--json", help=f"summary JSON (default {DEFAULT_OUT.relative_to(ROOT)}/summary.json)")
    ap.add_argument("--no-write", action="store_true", help="print only")
    ap.add_argument("--quiet", action="store_true", help="summaries only, no per-missile table")
    args = ap.parse_args(argv)
    files = []
    for raw in args.inputs:
        p = Path(raw)
        if p.is_dir():
            files.extend(sorted(p.rglob("*.jsonl")))
        elif p.is_file():
            files.append(p)
        else:
            print(f"skip {p}: not found", file=sys.stderr)
    rows, infos = [], []
    for f in files:
        try:
            rs, info = analyse_file(f, args)
        except (ValueError, KeyError) as e:
            print(f"skip {f}: {e}", file=sys.stderr)
            continue
        rows.extend(rs)
        infos.append(info)
    if not rows:
        print("no missiles found", file=sys.stderr)
        return 1
    if not args.quiet:
        print("  ".join(fmt(h, w) for h, (_, w) in zip(HEAD, COLS)))
        for r in rows:
            print("  ".join(fmt(r.get(k), w) for k, w in COLS))
        print("look = mean terminal look angle (deg, + up), lookE = at CPA, |vr| = mean |v_target.LOS| (m/s), "
              "|vr|3 = last 3 s, nf / full = share of terminal in the band / band + look-down, OB = off-boresight")
    summary = summarise(rows, args)
    for g, s in summary.items():
        print(f"\n== {g}: {s['missiles']} missiles, {s['used']} in the summaries; left out: {s['excluded'] or '-'}")
        for name, cells in s["tables"].items():
            print(f"  {name}")
            for k, c in cells.items():
                print(f"    {k:<28} n={c['n']:<4} hit={c['hit']:<3} hit_nokill={c['hit_nokill']:<3} miss={c['miss']:<4}"
                      f" rate={c['rate']:.2f}  95% [{c['ci95'][0]:.2f}, {c['ci95'][1]:.2f}]"
                      + (f"  chaff>0: {c['chaff']}" if any(isinstance(r.get('chaff_term'), int) for r in rows) else ""))
    settings = dict(terminal_s=args.terminal_s, seeker_range_m=args.seeker_range_m, notch_mps=args.notch_mps,
                    notch_look_deg=args.notch_look_deg, beam_frac=args.beam_frac, min_flight_s=args.min_flight_s,
                    weapon=args.weapon, all_targets=args.all_targets, seeker=args.seeker)
    if not args.no_write:
        csv_path = Path(args.csv) if args.csv else DEFAULT_OUT/"missiles.csv"
        json_path = Path(args.json) if args.json else DEFAULT_OUT/"summary.json"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        json_path.write_text(json.dumps(dict(settings=settings, files=infos, summary=summary), indent=1),
                             encoding="utf-8")
        print(f"\nwrote {csv_path} and {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
