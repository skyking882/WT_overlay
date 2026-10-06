"""Radar and RWR sensor model: what a pilot could know about the other aircraft, tick by tick.

A multi-aircraft engagement simulator owns one ``RadarSensor`` and one ``RwrSensor`` per
aircraft (parameters from wt_overlay/units.py, i.e. the game files) and calls them every
tick with the truth of the world. They return pictures built only from what the sensor
could have measured; the truth ids behind each entry sit in separate ``truth_ids`` tuples
for evaluation and must never reach a policy.

Frame and conventions. World = ENU metres, z up, flat ground at z = 0 (escape.to_enu
converts missile_sim's x, up, z frame). Attitude: heading in degrees clockwise from north,
pitch positive nose up, roll positive right wing down (as wt_overlay/attitude.py). Angles
relative to the nose are azimuth positive right, elevation positive up; world angles are
bearing (compass, 0 = north, clockwise) and elevation above the horizon. Closing speed is
positive when the range shrinks.

Typical use, once per tick for every aircraft (build ``truths``, the TargetTruth of every
aircraft, once and share it; each radar skips its ``owner``)::

    radar = RadarSensor(units.load().radar_of(aircraft), owner=ident)   # once
    radar.set_mode("tws", pattern_index=0, centre_az_deg=0., centre_el_deg=0., t=0.)
    picture = radar.update(t, dt, own, truths)               # RadarPicture: contacts, hits, stt_state, events
    lit = radar.illuminated(t, own, [(id, position), ...], dt)    # ids of the receivers its beam is on now
    emission = radar.emission(ident, t, own, receiver_position, dt)    # for one receiver; covers_receiver inside
    rwr_picture = rwr_sensor.update(t, receiver_own, emissions)        # RwrPicture of RwrContact

Radar semantics (D = inferred, never from a file unless stated; the grades come from the
units.py docstring and docs/rl_design.md section 8):

  * Scan (D). A pattern is a raster: ``bars`` rows of ``bar_height_deg`` stacked about the
    centre elevation (first row on top), one frame per ``period_s``, each row takes
    period/bars and the beam centre sweeps the azimuth ``centre +- half_width_deg`` across
    it, alternating direction row by row. The player sets the centre (azimuth from the
    nose, elevation) and it is clamped so the pattern stays inside the pattern's azimuth /
    elevation limits. The scan is level (stabilised against bank and pitch) while |roll|
    and |pitch| are within the file's rollStabLimit / pitchStabLimit, else fixed to the
    airframe; a target must also lie inside the limits in airframe axes (gimbal limits).
  * Beam (A for the numbers, D for the use). The transceiver's ``angleHalfSens`` (azimuth,
    elevation; 3 deg when absent) is the beam half width: a target is illuminated while the
    beam centre is within that of it in azimuth and elevation (independent rectangular
    coordinates). Detection happens on the first tick of a dwell in which the target is
    illuminated and detectable, at most once per frame, so rows that overlap in elevation
    do not detect twice.
  * Detectability (D). Range limit = ``range_m`` * (rcs / reference rcs)^(1/4), capped by the
    transceiver ``rangeMax`` and the signal's ``distance`` limits (min and max). Aircraft
    RCS is not in the game files that are cached, so it is a per-target value with the
    sensor default 5 m^2. Optional ``ramp_fraction`` (needs an rng): the detection
    probability falls linearly to 0 over that fraction below the limit, rolled once per
    target and frame; default 0 is a hard cut. The Doppler window [doppler_min, doppler_max]
    applies to the closing speed. Look-down = line of sight at least ``look_down_deg``
    (2, as missile_sim's look_down_angle) below the horizon; then a waveform with a main beam
    notch loses targets whose own radial speed |v_target . los| is under half the notch
    width (the width is read as full width centred on zero), and a ``ground_clutter``
    waveform loses targets lower than R*sin(beam half elevation) above the ground (they sit
    inside the beam's ground footprint). Waveforms without ``range_finder`` (HPRF velocity
    search) give angles and closing speed but no range or position; surface-search signals
    (``air_target`` false) detect no aircraft. Multipath, sidelobes,
    jamming and chaff are not modelled; there is no measurement noise.
  * Tracks (TWS, D except the file numbers). A detection joins the nearest existing track
    inside its gate (file ``posGateRange``, growing with the time since the last detection
    up to ``posGateMaxTime``, at least 100 m; ``posGateRangeInitial`` for unconfirmed tracks,
    which are dropped after ``posGateTimeInitial[1]``), else it starts an unconfirmed track.
    A track is confirmed (and shown, with an id) when it has two detections, is
    ``track_time_min_s`` (2) old and fewer than ``track_limit`` tracks exist. Position and
    velocity follow an alpha-beta filter (alpha 1: measurements are exact; beta 0.6),
    positions are extrapolated at constant velocity and a track is dropped ``timeout_s``
    after its last detection. ``extrapolated`` is set when a track has missed its expected
    revisit (1.5 frame periods, 2 fast-pattern periods for electronic radars, 1.5 ticks in
    STT); ``age_s`` is the plain time since the last detection. Electronically scanned
    radars also refresh existing tracks anywhere inside the fast pattern's field of
    regard (at most once per fast period), drop them ``fast_timeout_s`` after the last
    refresh, and take new tracks only from the selected scan. While running, their fast
    scan illuminates the whole field of regard for the RWR (``fast_scan_rwr``).
  * Search: detections only, as blips without ids (one per target, replaced when the target
    is detected again, dropped after 1.5 frame periods). STT: one designated target
    (from a TWS track id, or a truth id for scripted designation), measured every tick
    while inside the field of regard (all patterns) and detectable, otherwise coasted for
    the file's ``stt_coast_s`` (3 s when absent) and then lost, after which the radar returns to its previous scan
    mode with no tracks. TWS tracks are discarded when STT starts. STT uses the selected
    waveform (the file's separate track signals are not extracted).

RWR semantics (D): an emission is received when the emitter's beam covers the receiver this
tick (``Emission.covers_receiver``; ``RadarSensor.illuminated`` computes it), its band is
in ``Rwr.bands``, it is within ``Rwr.range_m`` (a plain gate) and it lies inside one of the
RWR's sectors (centre +- width / 2, in airframe axes). A sector with an angle finder
reports the true direction (plus a Gaussian error of ``angle_sigma_deg`` when an rng is
given), any other the sector centre. Contacts are held ``new_target_hold_s`` after a first
illumination and ``target_hold_s`` (else ``signal_hold_s``, else 3 s) after later ones,
capped at ``targets_max`` by priority (missile, lock, nearest). ``detects_tracking`` shows
an STT lock as ``tracking`` (a radar that lost its STT target and coasts counts as search);
air radars give no launch warning, a missile seeker (kind 'missile') gives a missile
warning. A range-finding RWR reports the range clamped to ``range_finder_m``. An emitter
heard again after more than 0.25 s of silence is a new illumination.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

from . import units as un

DEFAULT_RCS_M2 = 5.
DEFAULT_BEAM_DEG = 3.            # beam half width when the file gives no antenna angleHalfSens
DEFAULT_TRACK_TIME_MIN_S = 2.
DEFAULT_TIMEOUT_S = 8.
DEFAULT_STT_COAST_S = 3.
DEFAULT_RWR_HOLD_S = 3.
LOOK_DOWN_DEG = 2.               # depression below which the line of sight counts as look-down
NOTCH_HALF_FRACTION = .5         # notch half width = main_beam_notch_mps * this
GATE_FLOOR_M = 100.
NEW_ILLUMINATION_GAP_S = .25      # an RWR emitter silent this long and heard again is a new illumination
DEFAULT_GATE = un.Gate((0., 300.), 2., 1500., (2., 4.))
VELOCITY_GAIN = .6               # beta of the alpha-beta filter
MODES = ("off", "search", "tws", "stt")
RAD = 180./math.pi


def wrap_deg(angle):
    return (angle+180.) % 360.-180.


@dataclass(frozen=True)
class OwnState:
    """Own aircraft: ENU position and velocity (m, m/s), heading / pitch / roll in degrees."""
    position: tuple
    velocity: tuple
    heading_deg: float
    pitch_deg: float = 0.
    roll_deg: float = 0.
    axes: tuple = field(init=False, repr=False, compare=False)   # nose, right, up unit vectors in ENU
    level: tuple = field(init=False, repr=False, compare=False)  # (sin, cos) of the heading

    def __post_init__(self):
        h, p, r = (math.radians(x) for x in (self.heading_deg, self.pitch_deg, self.roll_deg))
        sh, ch, sp, cp, sr, cr = math.sin(h), math.cos(h), math.sin(p), math.cos(p), math.sin(r), math.cos(r)
        nose, right, up = (cp*sh, cp*ch, sp), (ch, -sh, 0.), (-sp*sh, -sp*ch, cp)
        object.__setattr__(self, "axes", (nose, tuple(cr*a-sr*b for a, b in zip(right, up)),
                                          tuple(cr*b+sr*a for a, b in zip(right, up))))
        object.__setattr__(self, "level", (sh, ch))


@dataclass(frozen=True)
class TargetTruth:
    """An aircraft as the simulator knows it. ``rcs_m2`` None: the sensor's default."""
    id: object
    position: tuple
    velocity: tuple
    rcs_m2: float | None = None


@dataclass(frozen=True)
class Emission:
    """One emitter as an RWR could see it this tick. ``emitter_id`` is truth (kept out of contacts), unique per
    emitter (a missile has its own);
    ``radar_id`` is the radar type the RWR symbol would show. ``covers_receiver``: the emitter's
    beam is on the receiver now (RadarSensor.illuminated; for a missile, its seeker is active on it)."""
    emitter_id: object
    position: tuple
    kind: str                       # 'search' | 'tws' | 'stt' | 'missile'
    band: int | None = 8
    covers_receiver: bool = True
    radar_id: str | None = None


@dataclass(frozen=True)
class RadarContact:
    """A radar track, blip or raw detection, as the pilot would read it. Angles are relative to the
    nose at the time of the picture; a blip or track keeps its world position, not its angles."""
    kind: str                       # picture: 'track' | 'blip' | 'stt'; hits: 'scan' | 'fast' | 'stt'
    track_id: int | None
    range_m: float | None           # None: the waveform measures no range
    azimuth_deg: float
    elevation_deg: float
    bearing_deg: float              # world, compass
    world_elevation_deg: float
    closing_speed_mps: float | None  # None: the waveform measures no Doppler
    position: tuple | None          # ENU estimate at the picture time
    velocity: tuple | None          # ENU estimate (tracks only)
    updated_s: float                # time of the last detection
    age_s: float
    extrapolated: bool
    mark_id: int | None = None        # observed map association, assigned by ObservationBuilder


@dataclass(frozen=True)
class RadarPicture:
    """Everything except ``truth_ids`` and ``hit_truth_ids`` (parallel to ``contacts`` / ``hits``,
    evaluation only) may be shown to a policy."""
    time_s: float
    mode: str
    contacts: tuple
    truth_ids: tuple
    hits: tuple                     # detections made in this update (diagnostics; tracks are what a pilot sees)
    hit_truth_ids: tuple
    stt_state: str | None           # 'acquiring' | 'tracking' | 'coasting' | None
    events: tuple                   # 'stt_lost'


@dataclass(frozen=True)
class RwrContact:
    contact_id: int
    kind: str                       # 'search' | 'tws' | 'stt' | 'missile'
    azimuth_deg: float              # relative to the nose when last illuminated (sector centre without angle finder)
    elevation_deg: float
    bearing_deg: float              # world compass of the same reading
    range_m: float | None
    tracking: bool                  # lock warning (STT seen by an RWR that detects tracking)
    missile_warning: bool
    band: int | None
    radar_id: str | None
    new: bool                       # illuminated only once so far
    illuminated: bool               # received in this very update
    age_s: float                    # since the last illumination


@dataclass(frozen=True)
class RwrPicture:
    time_s: float
    contacts: tuple
    truth_ids: tuple                # emitter ids, evaluation only


def relative_angles(own: OwnState, point):
    """(range m, azimuth deg, elevation deg) of ``point`` in the airframe axes of ``own``."""
    d = (point[0]-own.position[0], point[1]-own.position[1], point[2]-own.position[2])
    nose, right, up = own.axes
    x, y, z = (d[0]*a[0]+d[1]*a[1]+d[2]*a[2] for a in (nose, right, up))
    return math.sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]), math.atan2(y, x)*RAD, math.atan2(z, math.hypot(x, y))*RAD


def world_angles(d):
    """(bearing deg, elevation deg) of an ENU offset."""
    return math.atan2(d[0], d[1])*RAD % 360., math.atan2(d[2], math.hypot(d[0], d[1]))*RAD


def _limits(patterns, axis):
    values = [v for p in patterns for v in (p.azimuth_limits_deg if axis == 0 else p.elevation_limits_deg)]
    return (min(values), max(values)) if values else (-60., 60.)


def _clamp(x, lo, hi):
    return (lo+hi)/2. if lo > hi else min(max(x, lo), hi)


class _Hit:
    __slots__ = ("truth", "t", "pos", "los", "range", "closing", "source")

    def __init__(self, truth, t, pos, los, rng, closing, source):
        self.truth, self.t, self.pos, self.los, self.range, self.closing, self.source = truth, t, pos, los, rng, closing, source


class _Track:
    __slots__ = ("id", "pos", "vel", "t_first", "t_last", "hits", "truth", "closing", "stamp")

    def __init__(self, hit, tick):
        self.id = None
        self.pos, self.vel, self.t_first, self.t_last, self.hits = hit.pos, None, hit.t, hit.t, 1
        self.truth, self.closing, self.stamp = hit.truth, hit.closing, tick

    def at(self, t):
        if self.vel is None:
            return self.pos
        age = t-self.t_last
        return (self.pos[0]+self.vel[0]*age, self.pos[1]+self.vel[1]*age, self.pos[2]+self.vel[2]*age)

    def update(self, hit, tick):
        dt = hit.t-self.t_last
        if dt > 1e-9:
            pred = self.at(hit.t)
            if self.vel is None:
                self.vel = tuple((n-o)/dt for n, o in zip(hit.pos, self.pos))
            else:
                self.vel = tuple(v+VELOCITY_GAIN*(n-p)/dt for v, n, p in zip(self.vel, hit.pos, pred))
        self.pos, self.t_last, self.hits, self.truth, self.closing, self.stamp = hit.pos, hit.t, self.hits+1, hit.truth, hit.closing, tick


class RadarSensor:
    """One aircraft's radar. ``set_mode`` selects mode, pattern, scan centre and waveform; ``update``
    advances the scan over the tick (t-dt, t] and returns the picture; ``illuminated`` /
    ``beam_covers`` tell whether the beam is on a point (for RWRs). Pass the whole world's targets
    to every radar and ``owner`` (the aircraft's own id) to have it skipped.
    """

    def __init__(self, radar: un.Radar, rcs_m2=DEFAULT_RCS_M2, rng=None, *, owner=None, ramp_fraction=0.,
                 look_down_deg=LOOK_DOWN_DEG, fast_scan_rwr=True, stt_coast_s=None):
        tws = radar.tws
        if not any(w.range_m for w in (*radar.search_waveforms, *(tws.waveforms if tws else ()))):
            raise ValueError(f"{radar.id} has no radar waveform")
        patterns = [*radar.search_patterns, *(tws.patterns if tws else ()), *([tws.fast_pattern] if tws and tws.fast_pattern else [])]
        self.radar, self.rcs_m2, self.owner = radar, rcs_m2, owner
        self._rng, self._ramp, self._fast_rwr = rng, ramp_fraction, fast_scan_rwr
        self._sin_down = -math.sin(math.radians(look_down_deg))
        self.stt_coast_s = stt_coast_s if stt_coast_s is not None else \
            DEFAULT_STT_COAST_S if radar.stt_coast_s is None else radar.stt_coast_s
        self._for = (*_limits(patterns, 0), *_limits(patterns, 1))   # airframe limits over every pattern
        fast = tws.fast_pattern if tws else None
        self._fast_for = (*_limits([fast], 0), *_limits([fast], 1)) if fast else None
        self._gate = (tws.gate if tws and tws.gate else DEFAULT_GATE)
        self._time_min = (tws.track_time_min_s if tws and tws.track_time_min_s is not None else DEFAULT_TRACK_TIME_MIN_S)
        self._track_limit = tws.track_limit if tws else None
        self._timeout = (tws.fast_timeout_s if radar.electronic and tws.fast_timeout_s is not None else
                         tws.timeout_s if tws and tws.timeout_s is not None else DEFAULT_TIMEOUT_S)
        self.mode = "off"
        self.stt_state = None
        self._t = self._dt = None
        self._tick = 0
        self._select = None            # (mode, pattern index, waveform index) of the running scan
        self._resume = None            # scan configuration to return to after STT
        self._pattern = self._wf = self._scan = None
        self._pat_box = self._for            # airframe limits of the selected pattern
        self._t_ref = 0.
        self._scan_id = 0
        self._seg_key = self._segs = None
        self._beam = (DEFAULT_BEAM_DEG, DEFAULT_BEAM_DEG, math.sin(math.radians(DEFAULT_BEAM_DEG)))
        self._rlim = {}
        self._done = {}
        self._tracks, self._blips = [], {}
        self._next_id = 1
        self._last_fast = -math.inf
        self._allow = math.inf
        self._stt = self._stt_truth = None
        self._stt_since = 0.

    # -- mode ---------------------------------------------------------------------------------------------

    @property
    def emission_kind(self):
        """'search' | 'tws' | 'stt' | None: what an RWR would classify this radar as right now."""
        if self.mode == "stt":
            return "stt" if self.stt_state == "tracking" else "search"
        return None if self.mode == "off" else self.mode

    @property
    def band(self):
        return self._wf.band[0] if self._wf is not None and self._wf.band else 8

    def set_mode(self, mode, pattern_index=0, centre_az_deg=0., centre_el_deg=0., waveform_index=0, *,
                 stt_track=None, stt_truth=None, t=None):
        """Select 'off' | 'search' | 'tws' | 'stt'. The centre is relative to the nose in azimuth and to the
        horizon in elevation (the airframe while the scan is not stabilised). Changing mode or pattern
        restarts the frame at ``t`` (default: the last update). STT needs ``stt_track`` (a track id of this
        radar's picture) or ``stt_truth`` (a truth id, for scripted designation)."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        now = (self._t or 0.) if t is None else t
        if mode == "off":
            self._reset(None)
            return
        if mode == "stt":
            self._start_stt(stt_track, stt_truth, now)
            return
        tws = self.radar.tws
        patterns, waves = (self.radar.search_patterns, self.radar.search_waveforms) if mode == "search" else \
            ((tws.patterns, tws.waveforms) if tws else ((), ()))
        if not 0 <= pattern_index < len(patterns) or not 0 <= waveform_index < len(waves):
            raise ValueError(f"{self.radar.id} has no {mode} pattern {pattern_index} / waveform {waveform_index}")
        pattern, wf = patterns[pattern_index], waves[waveform_index]
        if not pattern.period_s or not wf.range_m:
            raise ValueError(f"{self.radar.id} {mode} pattern {pattern.name} / waveform {wf.signal} is unusable")
        select = (mode, pattern_index, waveform_index)
        if mode != self.mode:
            self._reset(mode)
        if select != self._select:
            self._t_ref, self._done = now, {}
        self._select, self._pattern, self._wf = select, pattern, wf
        self._pat_box = (*_limits([pattern], 0), *_limits([pattern], 1))
        self._resume = (mode, pattern_index, centre_az_deg, centre_el_deg, waveform_index)
        ha, he = wf.beam_azimuth_deg or DEFAULT_BEAM_DEG, wf.beam_elevation_deg or DEFAULT_BEAM_DEG
        self._beam, self._rlim = (ha, he, math.sin(math.radians(he))), {}
        w, bars, bh = pattern.half_width_deg or 0., pattern.bars, pattern.bar_height_deg or 0.
        az_lo, az_hi = pattern.azimuth_limits_deg or (-180., 180.)
        el_lo, el_hi = pattern.elevation_limits_deg or (-90., 90.)
        height = bars*bh
        c_el = _clamp(centre_el_deg+pattern.center_elevation_deg, el_lo+height/2., el_hi-height/2.)
        self._scan = (pattern.period_s, bars, pattern.period_s/bars, w, bh, _clamp(centre_az_deg, az_lo+w, az_hi-w),
                      c_el+(height-bh)/2.)
        self._scan_id += 1
        self._allow = (2.*tws.fast_pattern.period_s if mode == "tws" and tws.fast_pattern and tws.fast_pattern.period_s
                       else 1.5*pattern.period_s)

    def _reset(self, mode):
        self.mode, self.stt_state = mode or "off", None
        self._tracks, self._blips, self._stt, self._stt_truth = [], {}, None, None
        if mode is None:
            self._select = self._resume = None

    def _start_stt(self, stt_track, stt_truth, now):
        seed = None
        if stt_track is not None:
            seed = next((k for k in self._tracks if k.id == stt_track), None)
            if seed is None:
                raise ValueError(f"no track {stt_track} to lock")
            stt_truth = seed.truth
        elif stt_truth is None:
            raise ValueError("stt needs stt_track or stt_truth")
        if self._wf is None:
            waves = (self.radar.tws.waveforms if self.radar.tws else ()) or self.radar.search_waveforms
            self._wf = next(w for w in waves if w.range_m)
            self._beam = (self._wf.beam_azimuth_deg or DEFAULT_BEAM_DEG, self._wf.beam_elevation_deg or DEFAULT_BEAM_DEG,
                          math.sin(math.radians(self._wf.beam_elevation_deg or DEFAULT_BEAM_DEG)))
            self._rlim = {}
        self._tracks, self._blips = [], {}
        self.mode, self.stt_state, self._stt, self._stt_truth, self._stt_since = "stt", "acquiring", seed, stt_truth, now
        self._allow = math.inf

    def _end_stt(self, t):
        resume = self._resume
        self._stt = self._stt_truth = None
        if resume is None:
            self._reset(None)
        else:
            self.mode, self.stt_state = "off", None
            self.set_mode(*resume, t=t)

    def tracked_ids(self):
        """Truth ids of the targets this radar holds a track on right now (a set; evaluation and the engagement
        manager only, never shown to a policy): confirmed TWS tracks, extrapolated ones included until they time
        out, or the STT target while it is tracked or coasting."""
        if self.mode == "tws":
            return {k.truth for k in self._tracks if k.id is not None}
        if self.mode == "stt" and self._stt is not None:
            return {self._stt.truth}
        return set()

    # -- geometry -----------------------------------------------------------------------------------------

    def _limit(self, wf, rcs):
        """Detection range for a target of ``rcs`` (cached per rcs for the current waveform)."""
        r = self._rlim.get(rcs)
        if r is None:
            r = (wf.range_m or 0.)*(rcs/(wf.reference_rcs_m2 or 1.))**.25
            for cap in (wf.range_max_m, wf.distance_max_m):
                if cap:
                    r = min(r, cap)
            self._rlim[rcs] = r
        return r

    def _level(self, own):
        """Is the scan stabilised against the horizon at this attitude (else it is fixed to the airframe)?"""
        p = self._pattern
        if p is None:
            return True
        return (abs(own.roll_deg) <= (180. if p.roll_stab_limit_deg is None else p.roll_stab_limit_deg)
                and abs(own.pitch_deg) <= (90. if p.pitch_stab_limit_deg is None else p.pitch_stab_limit_deg))

    @staticmethod
    def _scan_angles(own, d, level):
        if level:
            sh, ch = own.level
            x, y, z = d[0]*sh+d[1]*ch, d[0]*ch-d[1]*sh, d[2]
        else:
            x, y, z = (d[0]*a[0]+d[1]*a[1]+d[2]*a[2] for a in own.axes)
        return math.atan2(y, x)*RAD, math.atan2(z, math.hypot(x, y))*RAD

    @staticmethod
    def _in_box(own, d, box):
        """Is the offset ``d`` inside (azimuth low, high, elevation low, high) in airframe axes?"""
        x, y, z = (d[0]*a[0]+d[1]*a[1]+d[2]*a[2] for a in own.axes)
        az, el = math.atan2(y, x)*RAD, math.atan2(z, math.hypot(x, y))*RAD
        return box[0] <= az <= box[1] and box[2] <= el <= box[3]

    def _segments(self, t0, t1):
        """Beam-centre sweeps (elevation, azimuth low, azimuth high) in the scan frame over (t0, t1]."""
        key = (t0, t1, self._scan_id)
        if self._seg_key == key:
            return self._segs
        period, bars, bar_s, w, bh, c_az, e_top = self._scan
        if t1-t0 >= period:
            t0 = t1-period
        a, b = t0-self._t_ref, t1-self._t_ref
        k, segs = math.floor(a/bar_s), []
        while True:
            base = k*bar_s
            f0, f1 = (max(a, base)-base)/bar_s, (min(b, base+bar_s)-base)/bar_s
            i = k % bars
            x0, x1 = (-w+2.*w*f0, -w+2.*w*f1) if i % 2 == 0 else (w-2.*w*f0, w-2.*w*f1)
            segs.append((e_top-i*bh, c_az+min(x0, x1), c_az+max(x0, x1)))
            if base+bar_s >= b:
                break
            k += 1
        self._seg_key, self._segs = key, segs
        return segs

    def beam_covers(self, t, own: OwnState, point, dt=0.):
        """Is the beam on ``point`` at ``t`` (or at any time in (t-dt, t])? For TWS / search this is true for a
        short dwell each frame; in STT while the point is within the beam of the designated target."""
        return bool(self.illuminated(t, own, ((True, point),), dt))

    def illuminated(self, t, own: OwnState, receivers, dt=0.):
        """Ids of the (id, position) ``receivers`` the beam covers at ``t`` (any time in (t-dt, t]). The scan,
        the airframe limits and the beam are evaluated once for all of them. An electronic radar in TWS also
        covers its whole field of regard with the fast scan (``fast_scan_rwr``)."""
        mode, ox, oy, oz = self.mode, *own.position
        ha, he, _ = self._beam
        fast_box = self._fast_for if (self._fast_rwr and mode == "tws") else None
        if mode == "stt":
            if self._stt is None:
                return []
            c = self._stt.at(t)
            level = self._level(own)
            az, el = self._scan_angles(own, (c[0]-ox, c[1]-oy, c[2]-oz), level)
            segs = [(el, az, az)]
        elif mode in ("search", "tws"):
            level, segs = self._level(own), self._segments(t-dt, t)
        else:
            return []
        box = self._for if mode == "stt" else self._pat_box
        out = []
        for rid, p in receivers:
            d = (p[0]-ox, p[1]-oy, p[2]-oz)
            if fast_box is not None and self._in_box(own, d, fast_box):
                out.append(rid)
                continue
            az, el = self._scan_angles(own, d, level)
            for e, lo, hi in segs:
                if abs(el-e) <= he and lo-ha <= az <= hi+ha:
                    if self._in_box(own, d, box):
                        out.append(rid)
                    break
        return out

    def emission(self, emitter_id, t, own: OwnState, receiver_position, dt=0.):
        """The Emission an RWR at ``receiver_position`` would get from this radar (None when it is off)."""
        kind = self.emission_kind
        if kind is None:
            return None
        return Emission(emitter_id, own.position, kind, self.band, self.beam_covers(t, own, receiver_position, dt),
                        self.radar.id)

    # -- detection ----------------------------------------------------------------------------------------

    def _try(self, own, tg, d, rng, wf, box, source, t):
        """A detection of ``tg`` (offset ``d``, range ``rng``) if it passes range, airframe limits, Doppler,
        notch and clutter; None otherwise. No beam test."""
        if not wf.air_target or rng > self._limit(wf, tg.rcs_m2 or self.rcs_m2) or rng < 1. or \
                (wf.distance_min_m and rng < wf.distance_min_m):
            return None
        if not self._in_box(own, d, box):
            return None
        inv = 1./rng
        lx, ly, lz = d[0]*inv, d[1]*inv, d[2]*inv
        tv, ov = tg.velocity, own.velocity
        closing = -((tv[0]-ov[0])*lx+(tv[1]-ov[1])*ly+(tv[2]-ov[2])*lz)
        if (wf.doppler_min_mps is not None and closing < wf.doppler_min_mps) or \
                (wf.doppler_max_mps is not None and closing > wf.doppler_max_mps):
            return None
        if lz <= self._sin_down:
            if wf.main_beam_notch_mps and abs(tv[0]*lx+tv[1]*ly+tv[2]*lz) < NOTCH_HALF_FRACTION*wf.main_beam_notch_mps:
                return None
            if wf.ground_clutter and tg.position[2] < rng*self._beam[2]:
                return None
        return _Hit(tg.id, t, tg.position if wf.range_finder else None, (lx, ly, lz), rng if wf.range_finder else None,
                    closing if wf.measures_doppler else None, source)

    def _scan_hits(self, t0, t, own, targets):
        """Targets the selected scan illuminated and could detect during (t0, t], at most one per frame each."""
        wf = self._wf
        if not wf.air_target:
            return []
        segs = self._segments(t0, t)
        level = self._level(own)
        frame = math.floor((t-self._t_ref)/self._scan[0])
        ha, he, _ = self._beam
        ox, oy, oz = own.position
        sh, ch = own.level
        axes = own.axes
        done, owner, hits = self._done, self.owner, []
        for tg in targets:
            if tg.id == owner or done.get(tg.id) == frame:
                continue
            p = tg.position
            dx, dy, dz = p[0]-ox, p[1]-oy, p[2]-oz
            r2 = dx*dx+dy*dy+dz*dz
            lim = self._limit(wf, tg.rcs_m2 or self.rcs_m2)
            if r2 > lim*lim:
                continue
            if level:
                x, y, z = dx*sh+dy*ch, dx*ch-dy*sh, dz
            else:
                x, y, z = (dx*a[0]+dy*a[1]+dz*a[2] for a in axes)
            az, el = math.atan2(y, x)*RAD, math.atan2(z, math.hypot(x, y))*RAD
            for e, lo, hi in segs:
                if abs(el-e) <= he and lo-ha <= az <= hi+ha:
                    break
            else:
                continue
            rng = math.sqrt(r2)
            hit = self._try(own, tg, (dx, dy, dz), rng, wf, self._pat_box, "scan", t)
            if hit is None:
                continue
            done[tg.id] = frame
            if self._ramp and self._rng is not None and rng > (1.-self._ramp)*lim and \
                    self._rng.random() > (lim-rng)/(self._ramp*lim):
                continue
            hits.append(hit)
        return hits

    # -- tracking -----------------------------------------------------------------------------------------

    def _gate_m(self, trk, t):
        g = self._gate
        if trk.id is None:
            return g.initial_range_m or 1500.
        r0, r1 = g.range_m
        return max(GATE_FLOOR_M, r0+(r1-r0)*min(1., (t-trk.t_last)/(g.max_time_s or 2.)))

    def _ingest(self, hit, t):
        """Join ``hit`` to the nearest track (distance / gate) that has not been updated this tick, else start one."""
        best, best_score = None, math.inf
        for trk in self._tracks:
            if trk.stamp == self._tick:
                continue
            p, q = trk.at(t), hit.pos
            d = math.sqrt((p[0]-q[0])**2+(p[1]-q[1])**2+(p[2]-q[2])**2)
            g = self._gate_m(trk, t)
            if d <= g and d/g < best_score:
                best, best_score = trk, d/g
        if best is None:
            self._tracks.append(_Track(hit, self._tick))
        else:
            best.update(hit, self._tick)

    def _fast_refresh(self, t, own, targets):
        """Electronic radars: re-detect existing tracks anywhere in the fast pattern's field of regard."""
        fp = self.radar.tws.fast_pattern
        if t-self._last_fast < (fp.period_s or 0.)*.999:
            return []
        self._last_fast = t
        ox, oy, oz = own.position
        hits = []
        for trk in self._tracks:
            if trk.stamp == self._tick:
                continue
            ep, g2 = trk.at(t), self._gate_m(trk, t)**2
            near = [((tg.position[0]-ep[0])**2+(tg.position[1]-ep[1])**2+(tg.position[2]-ep[2])**2, i)
                    for i, tg in enumerate(targets) if tg.id != self.owner]
            for d2, i in sorted(x for x in near if x[0] <= g2):
                tg = targets[i]
                d = (tg.position[0]-ox, tg.position[1]-oy, tg.position[2]-oz)
                hit = self._try(own, tg, d, math.sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]), self._wf, self._fast_for, "fast", t)
                if hit is not None and hit.pos is not None:
                    trk.update(hit, self._tick)
                    hits.append(hit)
                    break
        return hits

    def _confirm_and_expire(self, t):
        """Confirm tracks that were detected this tick and have aged enough; drop the ones past their timeout."""
        g = self._gate
        window = max(g.initial_time_s[1] if len(g.initial_time_s) > 1 else 4., 1.5*self._scan[0])
        confirmed = sum(k.id is not None for k in self._tracks)
        for trk in self._tracks:
            if trk.id is None and trk.stamp == self._tick and trk.hits >= 2 and t-trk.t_first >= self._time_min-1e-9 and \
                    (self._track_limit is None or confirmed < self._track_limit):
                trk.id, self._next_id, confirmed = self._next_id, self._next_id+1, confirmed+1
        self._tracks = [k for k in self._tracks if t-k.t_last <= (self._timeout if k.id is not None else window)]

    def _update_stt(self, t, own, targets, hits, events):
        tg = next((x for x in targets if x.id == self._stt_truth), None)
        hit = None
        if tg is not None:
            d = (tg.position[0]-own.position[0], tg.position[1]-own.position[1], tg.position[2]-own.position[2])
            hit = self._try(own, tg, d, math.sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]), self._wf, self._for, "stt", t)
        if hit is not None and hit.pos is not None:
            if self._stt is None:
                self._stt = _Track(hit, self._tick)
            else:
                self._stt.update(hit, self._tick)
            self.stt_state = "tracking"
            hits.append(hit)
            return
        last = self._stt.t_last if self._stt is not None else self._stt_since
        if t-last > self.stt_coast_s:
            events.append("stt_lost")
            self._end_stt(t)
        else:
            self.stt_state = "coasting" if self._stt is not None else "acquiring"

    # -- tick ---------------------------------------------------------------------------------------------

    def update(self, t, dt, own: OwnState, targets, report=True):
        """Advance the radar over (t-dt, t]. ``targets``: TargetTruth list (the owner is skipped).
        Returns a RadarPicture, or None when ``report`` is false (state still advances; ``picture`` builds it)."""
        self._tick += 1
        self._t, self._dt = t, dt
        hits, events = [], []
        if self.mode == "stt":
            self._update_stt(t, own, targets, hits, events)
        elif self.mode in ("search", "tws"):
            scan_hits = self._scan_hits(t-dt, t, own, targets)
            hits.extend(scan_hits)
            if self.mode == "search":
                for h in scan_hits:
                    self._blips[h.truth] = h
                hold = 1.5*self._scan[0]
                self._blips = {k: h for k, h in self._blips.items() if t-h.t <= hold}
            else:
                for h in scan_hits:
                    if h.pos is not None:
                        self._ingest(h, t)
                if self.radar.electronic:
                    hits.extend(self._fast_refresh(t, own, targets))
                self._confirm_and_expire(t)
        return self.picture(t, own, hits, events) if report else None

    def picture(self, t, own: OwnState, hits=(), events=()):
        """The picture at ``t`` from the current state (tracks extrapolated to ``t``)."""
        contacts, truth = [], []
        if self.mode == "tws":
            for trk in sorted((k for k in self._tracks if k.id is not None), key=lambda k: k.id):
                contacts.append(self._contact("track", trk, t, own))
                truth.append(trk.truth)
        elif self.mode == "search":
            for h in sorted(self._blips.values(), key=lambda h: -h.t):
                contacts.append(self._hit_contact("blip", h, t, own))
                truth.append(h.truth)
        elif self.mode == "stt" and self._stt is not None:
            contacts.append(self._contact("stt", self._stt, t, own))
            truth.append(self._stt.truth)
        return RadarPicture(t, self.mode, tuple(contacts), tuple(truth),
                            tuple(self._hit_contact(h.source, h, t, own) for h in hits), tuple(h.truth for h in hits),
                            self.stt_state, tuple(events))

    def _allow_s(self):
        """Longest time since the last detection before a track counts as extrapolated (a missed revisit)."""
        dt = self._dt or 0.
        return 1.5*dt if self.mode == "stt" else max(self._allow, 1.5*dt)

    def _contact(self, kind, trk, t, own):
        pos, vel, closing = trk.at(t), trk.vel, trk.closing
        d = (pos[0]-own.position[0], pos[1]-own.position[1], pos[2]-own.position[2])
        rng, az, el = relative_angles(own, pos)
        age = t-trk.t_last
        if not self._wf.measures_doppler:
            closing = None
        elif vel is not None and age > 1e-9 and rng > 1.:
            closing = -sum((v-o)*x/rng for v, o, x in zip(vel, own.velocity, d))
        bearing, world_el = world_angles(d)
        return RadarContact(kind, trk.id, rng, az, el, bearing, world_el, closing, pos, vel, trk.t_last, age,
                            age > self._allow_s()+1e-9)

    def _hit_contact(self, kind, hit, t, own):
        if hit.pos is not None:
            d = (hit.pos[0]-own.position[0], hit.pos[1]-own.position[1], hit.pos[2]-own.position[2])
            rng = relative_angles(own, hit.pos)[0]
        else:
            d, rng = hit.los, None
        _, az, el = relative_angles(own, (own.position[0]+d[0], own.position[1]+d[1], own.position[2]+d[2]))
        bearing, world_el = world_angles(d)
        return RadarContact(kind, None, rng, az, el, bearing, world_el, hit.closing, hit.pos, None, hit.t, t-hit.t, False)


class RwrSensor:
    """One aircraft's RWR. ``update`` takes the emissions of every emitter that may reach it this tick."""

    def __init__(self, rwr: un.Rwr, rng=None, angle_sigma_deg=2.):
        self.rwr, self._rng, self._sigma = rwr, rng, angle_sigma_deg
        hold = rwr.target_hold_s if rwr.target_hold_s is not None else rwr.signal_hold_s
        self._hold = DEFAULT_RWR_HOLD_S if hold is None else hold
        self._hold_new = self._hold if rwr.new_target_hold_s is None else rwr.new_target_hold_s
        self._contacts, self._tick, self._next_id = {}, 0, 1

    def _sector(self, az, el):
        best, best_score = None, math.inf
        for k, (saz, sel, w, h, _) in enumerate(self.rwr.sectors):
            daz, d_el = abs(wrap_deg(az-saz)), abs(el-sel)
            if daz <= w/2. and d_el <= h/2.:
                score = daz/w+d_el/h
                if score < best_score:
                    best, best_score = k, score
        return best

    def update(self, t, own: OwnState, emitters) -> RwrPicture:
        """Receive this tick's emissions, age the held contacts and return the contacts by priority."""
        self._tick += 1
        rwr = self.rwr
        ox, oy, oz = own.position
        for e in emitters:
            if not e.covers_receiver or (e.band is not None and e.band not in rwr.bands):
                continue
            d = (e.position[0]-ox, e.position[1]-oy, e.position[2]-oz)
            rng = math.sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2])
            if rng < 1. or (rwr.range_m and rng > rwr.range_m):
                continue
            _, az, el = relative_angles(own, e.position)
            k = self._sector(az, el)
            if k is None:
                continue
            sector_az, sector_el, _, _, finder = rwr.sectors[k]
            if finder:
                bearing = world_angles(d)[0]
                if self._rng is not None and self._sigma:
                    noise = self._rng.gauss(0., self._sigma)
                    az, el, bearing = az+noise, el+self._rng.gauss(0., self._sigma), bearing+noise
            else:
                az, el, bearing = sector_az, sector_el, own.heading_deg+sector_az
            c = self._contacts.get(e.emitter_id)
            if c is None:
                c = self._contacts[e.emitter_id] = dict(id=self._next_id, count=0, tick=-1)
                self._next_id += 1
            if c["tick"] != self._tick-1 or t-c["last"] > NEW_ILLUMINATION_GAP_S:   # a gap: a new illumination
                c["count"] += 1
            c.update(tick=self._tick, last=t, kind=e.kind, azimuth=az, elevation=el, bearing=bearing % 360., band=e.band,
                     radar_id=e.radar_id, rank=rng,
                     range=None if rwr.range_finder_m is None else min(max(rng, rwr.range_finder_m[0]), rwr.range_finder_m[1]))
        rows = []
        for key, c in list(self._contacts.items()):
            if t-c["last"] > (self._hold_new if c["count"] == 1 else self._hold)+1e-9:
                del self._contacts[key]
            else:
                rows.append((key, c))
        rows.sort(key=lambda kc: (kc[1]["kind"] != "missile", kc[1]["kind"] != "stt", kc[1]["rank"]))
        for key, _ in rows[rwr.targets_max or len(rows):]:
            del self._contacts[key]
        rows = rows[:rwr.targets_max or len(rows)]
        contacts = tuple(RwrContact(c["id"], c["kind"], c["azimuth"], c["elevation"], c["bearing"],
                                    c["range"], c["kind"] == "stt" and rwr.detects_tracking, c["kind"] == "missile",
                                    c["band"], c["radar_id"], c["count"] == 1, c["tick"] == self._tick, t-c["last"])
                         for _, c in rows)
        return RwrPicture(t, contacts, tuple(key for key, _ in rows))
