#!/usr/bin/env python3
"""Plot an engagement replay (scripts/run_engagement.py): top-down tracks, altitude against time, and a phase strip.

    outputs/.mlenv/bin/python scripts/plot_engagement.py outputs/engagements/duel.jsonl [--out duel.png]

Top view (east right, north up, map square drawn): aircraft tracks in team colours (team 0 blue, team 1 orange; a
filled circle marks the spawn, the last point the end of life), missiles as thin lines in the shooter's colour, markers
for launches (triangle), kills (cross on the victim), other deaths (square), chaff releases (small dots), RWR missile
warnings (diamond on the warned aircraft) and phase changes (tick; labelled for matches of up to four aircraft).
Below: altitude against time, with the same event ticks, and for up to four aircraft a phase strip per aircraft; for
larger matches the second strip shows how many aircraft are alive per team and how many missiles are in flight.
Needs matplotlib (outputs/.mlenv).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

TEAM = ("#2a78d6", "#eb6834")                      # reference palette slots 1 and 2 (validated adjacent pair)
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
PHASES = {"": "#c9c8c2", "climb": "#2a78d6", "suppress": "#e87ba4", "advance": "#1baf7a", "evade": "#e34948",
          "recommit": "#eda100", "round2": "#4a3aa7", "crawl": "#008300", "popup": "#eb6834", "rush": "#52514e",
          "home": "#9a9993"}


def load(path):
    header, frames, events, end = None, [], [], None
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        kind = row["type"]
        if kind == "header":
            header = row
        elif kind == "frame":
            frames.append(row)
        elif kind == "event":
            events.append(row)
        elif kind == "end":
            end = row
    return header, frames, events, end


def tracks(frames):
    planes, missiles = {}, {}
    for f in frames:
        for p in f["planes"]:
            planes.setdefault(p[0], []).append((f["t"], p[1], p[2], p[3], p[10]))
        for m in f["missiles"]:
            missiles.setdefault(m[0], dict(owner=m[1], target=m[2], pts=[]))["pts"].append((f["t"], m[3], m[4], m[5]))
    return planes, missiles


def position(planes, ident, t):
    pts = planes.get(ident)
    if not pts:
        return None
    best = min(pts, key=lambda p: abs(p[0]-t))
    return best


def style_axis(ax):
    ax.set_facecolor(SURFACE)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color=GRID, linewidth=.6)
    ax.set_axisbelow(True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("replay", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--title")
    args = parser.parse_args(argv)
    header, frames, events, end = load(args.replay)
    planes, missiles = tracks(frames)
    info = {p["id"]: p for p in header["planes"]}
    team_of = {i: p["team"] for i, p in info.items()}
    small = len(info) <= 4
    half = header["map_half_m"]/1000.
    duration = frames[-1]["t"]

    fig = plt.figure(figsize=(17, 8.6), facecolor=SURFACE)
    grid = fig.add_gridspec(2, 2, width_ratios=(1.0, 1.25), height_ratios=(3, 2 if small else 1.6), wspace=.2, hspace=.22,
                            left=.05, right=.985, top=.94, bottom=.2)
    top = fig.add_subplot(grid[:, 0])
    alt, strip = fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, 1])
    for ax in (top, alt, strip):
        style_axis(ax)

    # -- top view ------------------------------------------------------------------------------------------------
    top.plot([-half, half, half, -half, -half], [-half, -half, half, half, -half], color=MUTED, linewidth=1.2)
    for uid, m in missiles.items():
        pts = m["pts"]
        top.plot([p[1]/1000. for p in pts], [p[2]/1000. for p in pts], color=TEAM[team_of[m["owner"]]], linewidth=.7,
                 alpha=.55, zorder=2)
    for ident, pts in planes.items():
        c = TEAM[team_of[ident]]
        top.plot([p[1]/1000. for p in pts], [p[2]/1000. for p in pts], color=c, linewidth=1.6 if small else 1.1, zorder=3)
        top.plot(pts[0][1]/1000., pts[0][2]/1000., "o", color=c, markersize=6 if small else 4, markeredgecolor=SURFACE,
                 markeredgewidth=1, zorder=4)
        if small:
            top.annotate(f"{ident}: {info[ident]['aircraft']}\n{info[ident]['archetype']}/{info[ident]['skill']}",
                         (pts[0][1]/1000., pts[0][2]/1000.), textcoords="offset points", xytext=(8, -4 if team_of[ident] else 8),
                         fontsize=8, color=INK)
    size = 40 if small else 16
    for e in events:
        k, t = e["kind"], e["t"]
        if k == "launch":
            p = position(planes, e["shooter"], t)
            if p:
                top.plot(p[1]/1000., p[2]/1000., "^", color=TEAM[team_of[e["shooter"]]], markersize=8 if small else 5,
                         markeredgecolor=INK, markeredgewidth=.6, zorder=5)
        elif k == "kill":
            p = position(planes, e["victim"], t)
            if p:
                top.plot(p[1]/1000., p[2]/1000., "X", color=INK, markersize=11 if small else 7, markeredgecolor=SURFACE,
                         markeredgewidth=.8, zorder=6)
        elif k == "death" and e["cause"] != "missile":
            p = position(planes, e["plane"], t)
            if p:
                top.plot(p[1]/1000., p[2]/1000., "s", color=INK, markersize=7 if small else 5, zorder=6)
        elif k == "chaff":
            p = position(planes, e["plane"], t)
            if p:
                top.plot(p[1]/1000., p[2]/1000., ".", color=MUTED, markersize=3, zorder=3)
        elif k == "rwr" and e["warning"] == "missile":
            p = position(planes, e["plane"], t)
            if p:
                top.plot(p[1]/1000., p[2]/1000., "D", markerfacecolor="none", markeredgecolor=INK, markersize=7 if small else 4,
                         zorder=6)
        elif k == "phase" and small:
            p = position(planes, e["plane"], t)
            if p:
                top.annotate(e["to"], (p[1]/1000., p[2]/1000.), textcoords="offset points", xytext=(4, 4), fontsize=6.5,
                             color=MUTED)
    top.set_aspect("equal")
    top.set_xlim(-half*1.04, half*1.04)
    top.set_ylim(-half*1.04, half*1.04)
    top.set_xlabel("east (km)", color=MUTED, fontsize=9)
    top.set_ylabel("north (km)", color=MUTED, fontsize=9)
    legend = [Line2D([], [], color=TEAM[0], linewidth=2, label="team 0 (spawns south, flies north)"),
              Line2D([], [], color=TEAM[1], linewidth=2, label="team 1 (spawns north, flies south)"),
              Line2D([], [], color=MUTED, linewidth=.8, label="missile"),
              Line2D([], [], color=INK, marker="^", linestyle="", markersize=7, label="launch"),
              Line2D([], [], color=INK, marker="D", markerfacecolor="none", linestyle="", markersize=6, label="RWR missile warning"),
              Line2D([], [], color=INK, marker="X", linestyle="", markersize=8, label="kill"),
              Line2D([], [], color=INK, marker="s", linestyle="", markersize=6, label="crash / left the map"),
              Line2D([], [], color=MUTED, marker=".", linestyle="", markersize=5, label="chaff")]
    top.legend(handles=legend, loc="upper center", bbox_to_anchor=(.5, -.08), fontsize=8, frameon=False, ncol=4)
    kills = sum(1 for e in events if e["kind"] == "kill")
    launches = sum(1 for e in events if e["kind"] == "launch")
    top.set_title(args.title or f"{args.replay.stem}: seed {header['seed']}, {duration:.0f} s, {launches} launches, {kills} kills"
                  + (f", ended: {end['reason']}" if end else ""), color=INK, fontsize=11, loc="left")

    # -- altitude ------------------------------------------------------------------------------------------------
    for ident, pts in planes.items():
        alt.plot([p[0] for p in pts], [p[3]/1000. for p in pts], color=TEAM[team_of[ident]], linewidth=1.4 if small else .8,
                 alpha=1. if small else .8)
    for e in events:
        if e["kind"] == "launch":
            alt.axvline(e["t"], color=MUTED, linewidth=.4, alpha=.5)
        elif e["kind"] == "kill":
            alt.axvline(e["t"], color=INK, linewidth=.9, alpha=.8)
    alt.set_ylabel("altitude (km)", color=MUTED, fontsize=9)
    alt.set_xlim(0, duration)
    alt.set_xticklabels([])
    alt.text(.995, .93, "grey lines: launches; black: kills", transform=alt.transAxes, ha="right", va="top", fontsize=7.5, color=MUTED)

    # -- phase strip / counts ------------------------------------------------------------------------------------
    if small:
        order = sorted(planes)
        for row, ident in enumerate(order):
            pts = planes[ident]
            start = pts[0][0]
            for a, b in zip(pts, pts[1:]+[(pts[-1][0]+.25, 0, 0, 0, "")]):
                strip.barh(row, b[0]-a[0], left=a[0], color=PHASES.get(a[4], "#c9c8c2"), height=.7, linewidth=0)
        strip.set_yticks(range(len(order)))
        strip.set_yticklabels([f"{i}: {info[i]['aircraft'][:16]}\n{info[i]['archetype']}" for i in order], fontsize=7.5, color=MUTED)
        used = sorted({p[4] for pts in planes.values() for p in pts}, key=lambda s: list(PHASES).index(s) if s in PHASES else 99)
        strip.legend(handles=[Line2D([], [], color=PHASES.get(s, "#c9c8c2"), linewidth=6, label=s or "-") for s in used],
                     loc="upper center", bbox_to_anchor=(.5, -.3), ncol=len(used), fontsize=8, frameon=False)
        strip.set_ylim(len(order)-.4, -.6)
    else:
        times = [f["t"] for f in frames]
        for team in (0, 1):
            strip.plot(times, [sum(1 for p in f["planes"] if team_of[p[0]] == team) for f in frames], color=TEAM[team], linewidth=1.6,
                       label=f"team {team} alive")
        strip.plot(times, [len(f["missiles"]) for f in frames], color=INK, linewidth=1., label="missiles in flight")
        strip.legend(loc="upper right", fontsize=8, frameon=False, ncol=3)
        strip.set_ylabel("count", color=MUTED, fontsize=9)
    strip.set_xlim(0, duration)
    strip.set_xlabel("time (s)", color=MUTED, fontsize=9)

    out = args.out or args.replay.with_suffix(".png")
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
