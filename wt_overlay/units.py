"""Airframe equipment from the game files: radar, RWR, air-to-air missiles, countermeasures.

scripts/import_datamine_units.py downloads each catalog aircraft's unit file and the
sensor and weapon-container files it uses (at the FM catalog's datamine revision),
parses them here and writes data/units/{equipment,radars,rwrs}.json; ``load`` reads
those back. Numbers are the game files' (A). What the fields mean is inferred from
their names and the radars' state machines (C/D):

  * Scan patterns: ``width`` is the azimuth half-width about the scan centre, bars
    of ``barHeight`` stack in elevation, one frame takes ``period`` seconds
    (user-confirmed for the F-16C's TWS, 2026-10-05).
  * TWS: a track is dropped ``timeout_s`` after its last detection and extrapolated
    until then (user: about 8 s on the F-16C, with lingering tracks); a new track
    needs ``track_time_min_s`` (user: about 2 s, unsure).
  * Electronically scanned radars alternate the selected TWS scan with a fast
    ``fast_pattern`` over their whole field of regard that only updates existing
    tracks (``fast_timeout_s``); mechanically scanned ones update a track only when
    the selected scan sweeps it.
  * The beam width is the transceiver antenna's ``angleHalfSens``; the track gate is
    ``posGateRange`` (metres, growing with the time since the last detection up to
    ``posGateMaxTime``) for confirmed tracks and ``posGateRangeInitial`` for new ones;
    ``rollStabLimit`` / ``pitchStabLimit`` are the attitudes up to which the scan is
    stabilised against the horizon (wt_overlay/sensors.py uses all of these).
  * NCTR: a radar whose target type table ``targetTypeId`` has an entry for rocket
    propulsion (``targetPropulsion`` type "rocket", the "hud/rocket" icon) names a
    missile track as a missile (``identifies_missiles``).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re

DATA_DIR = Path(__file__).resolve().parents[1]/"data"/"units"


def as_list(x):
    return x if isinstance(x, list) else ([] if x is None else [x])


def stem(blk: str) -> str:
    """'gameData/Weapons/rocketGuns/us_aim_120b_default.blk' -> 'us_aim_120b_default'."""
    return blk.replace("\\", "/").split("/")[-1].rsplit(".", 1)[0].lower()


def missile_id(blk: str) -> str:
    """Weapon file stem without the loadout-variant suffixes: us_aim_120b_default_switzerland -> us_aim_120b."""
    s = re.sub(r"_default.*$", "", stem(blk))
    return re.sub(r"_(bol_pod|switzerland)$", "", s)


@dataclass(frozen=True)
class ScanPattern:
    name: str
    azimuth_limits_deg: tuple
    elevation_limits_deg: tuple
    half_width_deg: float | None
    bars: int
    bar_height_deg: float | None
    period_s: float | None
    center_elevation_deg: float = 0.          # file centerElevation: offset of the scan centre from the commanded one
    roll_stab_limit_deg: float | None = None  # scan stays level up to this bank / pitch (None: not given)
    pitch_stab_limit_deg: float | None = None


@dataclass(frozen=True)
class Waveform:
    transceiver: str
    signal: str
    range_m: float | None           # detection range of a target of reference_rcs_m2
    reference_rcs_m2: float | None
    doppler_min_mps: float | None   # closing-speed window the signal passes (None: no Doppler filter)
    doppler_max_mps: float | None
    main_beam_notch_mps: float | None
    ground_clutter: bool | None
    distance_max_m: float | None
    band: tuple = ()                          # bands the transmitter emits (an RWR must list one)
    beam_azimuth_deg: float | None = None     # antenna angleHalfSens, half width of the beam
    beam_elevation_deg: float | None = None
    range_max_m: float | None = None          # transceiver rangeMax: nothing is seen beyond, whatever its RCS
    distance_min_m: float | None = None
    range_finder: bool = True                 # False: velocity-only signal (HPRF velocity search), no range
    air_target: bool = True                   # False: surface-search signal, aircraft are not targets

    @property
    def measures_doppler(self) -> bool:
        return self.doppler_min_mps is not None or self.doppler_max_mps is not None


@dataclass(frozen=True)
class Gate:
    """TWS association gate (posGate* of matchTargetsOfInterest / updateTargetOfInterest)."""
    range_m: tuple                 # radius at no age and at max_time_s since the last detection
    max_time_s: float | None
    initial_range_m: float | None  # radius for a track that is not yet confirmed
    initial_time_s: tuple


@dataclass(frozen=True)
class Tws:
    patterns: tuple
    waveforms: tuple
    timeout_s: float | None
    track_limit: int | None
    track_time_min_s: float | None
    fast_pattern: ScanPattern | None = None
    fast_timeout_s: float | None = None
    gate: Gate | None = None


@dataclass(frozen=True)
class Radar:
    id: str
    name: str
    search_patterns: tuple
    search_waveforms: tuple
    tws: Tws | None
    stt_coast_s: float | None
    identifies_missiles: bool = False   # NCTR: targetTypeId lists rocket-propelled targets

    @property
    def electronic(self) -> bool:
        return self.tws is not None and self.tws.fast_pattern is not None

    @property
    def field_of_regard_deg(self) -> float | None:
        limits = [abs(v) for p in (*self.search_patterns, *(self.tws.patterns if self.tws else ()))
                  for v in p.azimuth_limits_deg]
        return max(limits) if limits else None


@dataclass(frozen=True)
class Rwr:
    id: str
    name: str
    range_m: float | None
    sectors: tuple              # (azimuth, elevation, azimuth width, elevation width, angle finder)
    detects_tracking: bool
    detects_launch: bool
    tracks_targets: bool
    targets_max: int | None
    signal_hold_s: float | None
    target_hold_s: float | None
    new_target_hold_s: float | None
    bands: tuple
    range_finder_m: tuple | None


@dataclass(frozen=True)
class Equipment:
    aircraft: str
    radar: str | None
    rwr: str | None
    mlws: bool
    missiles: dict = field(default_factory=dict)   # air-to-air missile id -> most carried at once
    countermeasures: int = 0                         # default loadout
    countermeasures_max: int = 0                     # with the largest dispenser options and pods


def _pattern(name, p):
    return ScanPattern(name, tuple(p.get("azimuthLimits") or ()), tuple(p.get("elevationLimits") or ()),
                       p.get("width"), int(p.get("barsCount") or 1), p.get("barHeight"), p.get("period"),
                       p.get("centerElevation") or 0., p.get("rollStabLimit"), p.get("pitchStabLimit"))


def _waveform(d, transceiver, signal):
    t = (d.get("transivers") or {}).get(transceiver) or {}
    s = (d.get("signals") or {}).get(signal) or {}
    doppler = s.get("dopplerSpeed") or {}
    distance = s.get("distance") or {}
    antenna = t.get("antenna") or {}
    azimuth, elevation = antenna.get("azimuth") or antenna, antenna.get("elevation") or antenna
    return Waveform(transceiver, signal, t.get("range"), t.get("rcs"),
                    doppler.get("minValue") if doppler else None, doppler.get("maxValue") if doppler else None,
                    s.get("mainBeamNotchWidth"), s.get("groundClutter"), distance.get("maxValue"),
                    tuple(int(b) for b in as_list(t.get("band"))), azimuth.get("angleHalfSens"),
                    elevation.get("angleHalfSens"), t.get("rangeMax"), distance.get("minValue"),
                    bool(s.get("rangeFinder", True)), bool(s.get("aircraftAsTarget", True)))


def _templates(d):
    return ((d.get("fsms") or {}).get("main") or {}).get("actionsTemplates") or {}


def _set_patterns(d, set_name):
    sp, sets = d.get("scanPatterns") or {}, d.get("scanPatternSets") or {}
    names = [n for k, v in (sets.get(set_name) or {}).items() if k.startswith("scanPattern") for n in as_list(v)]
    return tuple(_pattern(n, sp[n]) for n in names if n in sp)


def _mode(d, kind):
    """(scan-pattern set, waveforms) the radar's '<kind>' air search mode uses ('Tws' or '')."""
    templates = _templates(d)
    common = [v for k, v in templates.items() if re.fullmatch(rf"set(Radar)?{kind}SearchModeCommon", k)]
    set_name = next(((t.get("setScanPatternSet") or {}).get("scanPatternSet") for t in common
                     if t.get("setScanPatternSet")), None)
    if set_name is None:
        set_name = next((k for k in (kind.lower(), f"radar{kind}", "search", "radarSearch", "radarSearchManual")
                         if k in (d.get("scanPatternSets") or {})), None)
    waveforms = []
    for name, t in templates.items():
        if not re.fullmatch(rf"set(Mprf|Hprf|HprfVelocity|Lprf|Pulse|)?{kind}SearchMode", name) or not isinstance(t, dict):
            continue
        tr, sig = (t.get("setTransiver") or {}).get("transiver"), (t.get("setSignal") or {}).get("signal")
        if tr and sig:
            waveforms.append(_waveform(d, tr, sig))
    return (_set_patterns(d, set_name) if set_name else ()), tuple(waveforms)


def _transitions(fsm):
    for t in (fsm.get("transitions") or {}).values():
        yield from as_list(t)


def _match(acts):
    """The track-matching action of a TWS transition: matchTargetsOfInterest (matched when a scan finishes)
    or updateTargetOfInterest (updated per detection, e.g. N001/N019 and the electronic radars)."""
    for key in ("matchTargetsOfInterest", "updateTargetOfInterest"):
        found = next((m for m in as_list(acts.get(key)) if isinstance(m, dict) and m), None)
        if found:
            return found
    return {}


def parse_radar(ident: str, d: dict) -> Radar:
    fsms = d.get("fsms") or {}
    search_patterns, search_waveforms = _mode(d, "")
    tws = None
    if "tws" in fsms:
        patterns, waveforms = _mode(d, "Tws")
        timeout = fast_timeout = fast = limit = time_min = gate = None
        for t in _transitions(fsms["tws"]):
            acts = t.get("actions") or {}
            fast_name = (acts.get("scan") or {}).get("scanPattern")
            clear = (acts.get("clearTargetsOfInterest") or {}).get("timeOut")
            if fast_name:
                fast, fast_timeout = _pattern(fast_name, (d.get("scanPatterns") or {}).get(fast_name) or {}), clear
            elif clear is not None:
                timeout = clear
            match = _match(acts)
            limit, time_min = match.get("limit", limit), match.get("timeMin", (match.get("timeLimits") or (time_min,))[0])
            if match.get("posGateRange"):
                gate = Gate(tuple(match["posGateRange"]), match.get("posGateMaxTime"), match.get("posGateRangeInitial"),
                            tuple(match.get("posGateTimeInitial") or ()))
        tws = Tws(patterns or search_patterns, waveforms or search_waveforms, timeout, limit, time_min, fast, fast_timeout, gate)
    coast = None
    track = fsms.get("track") or {}
    for t in [*(track.get("actionsTemplates") or {}).values(), *(t.get("actions") or {} for t in _transitions(track))]:
        coast = (t.get("clearTargetsOfInterest") or {}).get("timeOut", coast) if isinstance(t, dict) else coast
    rocket = any(isinstance(p, dict) and p.get("type") == "rocket" for e in as_list(d.get("targetTypeId"))
                 if isinstance(e, dict) for p in as_list(e.get("targetPropulsion")))
    return Radar(ident, d.get("name") or ident, search_patterns, search_waveforms, tws, coast, rocket)


def parse_rwr(ident: str, d: dict) -> Rwr:
    sectors = tuple((r.get("azimuth"), r.get("elevation"), r.get("azimuthWidth"), r.get("elevationWidth"),
                     bool(r.get("angleFinder"))) for r in as_list((d.get("receivers") or {}).get("receiver")))
    bands = tuple(sorted(int(k[4:]) for k, v in d.items() if re.fullmatch(r"band\d+", k) and any(as_list(v))))
    return Rwr(ident, d.get("name") or ident, d.get("range"), sectors, bool(d.get("detectTracking")),
               bool(d.get("detectLaunch")), bool(d.get("targetTracking")), d.get("trackedTargetsMax"),
               d.get("signalHoldTime"), d.get("targetHoldTime"), d.get("newTargetHoldTime"), bands,
               tuple(d["targetRange"]) if d.get("targetRangeFinder") and d.get("targetRange") else None)


def _countermeasures(preset):
    return sum(w.get("bullets") or 0 for w in as_list(preset.get("Weapon")) if "countermeasure" in w.get("blk", "").lower())


def parse_unit(aircraft: str, unit: dict, sensor_types: dict, containers: dict) -> Equipment:
    """sensor_types: sensor id -> type ('radar', 'rwr', ...); containers: container stem -> (inner blk, count)."""
    sensors = [stem(s["blk"]) for s in as_list((unit.get("sensors") or {}).get("sensor"))]
    kinds = [sensor_types.get(s) for s in sensors]
    radar = next((s for s, k in zip(sensors, kinds) if k == "radar"), None)
    rwr = next((s for s, k in zip(sensors, kinds) if k == "rwr"), None)
    missiles, cm_default, cm_max = {}, 0, 0
    common = sum(w.get("bullets") or 0 for w in as_list((unit.get("commonWeapons") or {}).get("Weapon"))
                 if "countermeasure" in w.get("blk", "").lower())
    for slot in as_list((unit.get("WeaponSlots") or {}).get("WeaponSlot")):
        presets = as_list(slot.get("WeaponPreset"))
        best = {}
        for p in presets:
            if "air_to_air" not in str(p.get("iconType", "")):
                continue
            carried = {}   # a preset can list the same missile more than once (e.g. a tandem pair under the fuselage)
            for w in as_list(p.get("Weapon")):
                if "blk" not in w:
                    continue
                blk, n = w["blk"], w.get("bullets") or 1
                if "countermeasure" in blk.lower():
                    continue
                if "/containers/" in blk.lower() and stem(blk) in containers:
                    blk, n = containers[stem(blk)][0], containers[stem(blk)][1]*(w.get("bullets") or 1)
                m = missile_id(blk)
                carried[m] = carried.get(m, 0)+n
            for m, n in carried.items():
                best[m] = max(best.get(m, 0), n)
        for m, n in best.items():
            missiles[m] = missiles.get(m, 0)+n
        counts = [_countermeasures(p) for p in presets]
        if slot.get("index") == 0 and counts:
            cm_default += counts[0]
        cm_max += max(counts, default=0)
    return Equipment(aircraft, radar, rwr, "mlws" in kinds, dict(sorted(missiles.items())),
                     common+cm_default, common+cm_max)


def _tupled(cls, data):
    """Rebuild a dataclass from its asdict() form (lists back to tuples, nested patterns and waveforms)."""
    if cls is ScanPattern:
        return ScanPattern(**{k: tuple(v) if isinstance(v, list) else v for k, v in data.items()})
    if cls is Waveform:
        return Waveform(**{**data, "band": tuple(data.get("band") or ())})
    if cls is Gate:
        return Gate(tuple(data["range_m"]), data["max_time_s"], data["initial_range_m"], tuple(data["initial_time_s"]))
    if cls is Tws:
        return Tws(tuple(_tupled(ScanPattern, p) for p in data["patterns"]),
                   tuple(_tupled(Waveform, w) for w in data["waveforms"]), data["timeout_s"], data["track_limit"],
                   data["track_time_min_s"], data["fast_pattern"] and _tupled(ScanPattern, data["fast_pattern"]),
                   data["fast_timeout_s"], data.get("gate") and _tupled(Gate, data["gate"]))
    if cls is Radar:
        return Radar(data["id"], data["name"], tuple(_tupled(ScanPattern, p) for p in data["search_patterns"]),
                     tuple(_tupled(Waveform, w) for w in data["search_waveforms"]),
                     data["tws"] and _tupled(Tws, data["tws"]), data["stt_coast_s"], data.get("identifies_missiles", False))
    if cls is Rwr:
        return Rwr(**{k: (tuple(tuple(x) if isinstance(x, list) else x for x in v) if isinstance(v, list) else v)
                      for k, v in data.items()})
    return cls(**data)


def dump(path: Path, items: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({k: asdict(v) for k, v in sorted(items.items())}, ensure_ascii=False, indent=1))


@dataclass(frozen=True)
class Units:
    equipment: dict
    radars: dict
    rwrs: dict

    def radar_of(self, aircraft: str) -> Radar | None:
        e = self.equipment.get(aircraft)
        return self.radars.get(e.radar) if e and e.radar else None

    def rwr_of(self, aircraft: str) -> Rwr | None:
        e = self.equipment.get(aircraft)
        return self.rwrs.get(e.rwr) if e and e.rwr else None


def load(data_dir: Path | None = None) -> Units:
    folder = Path(data_dir or DATA_DIR)
    read = lambda name, cls: {k: _tupled(cls, v) for k, v in json.loads((folder/name).read_text()).items()}  # noqa: E731
    return Units(read("equipment.json", Equipment), read("radars.json", Radar), read("rwrs.json", Rwr))
