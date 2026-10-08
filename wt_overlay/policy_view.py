"""Opt-in policy observation views (MatchEnv config; absent = the observation as before, bit for bit).

Only what the policy is given changes. The engagement, the sensor models, the scripted pilots (they keep full
information on purpose), the critic's truth tokens, target legality (track ids, masks, launch_ok) and execution are
as without these options: rl_observation.select_view_entities keeps a full-information twin of every shown entity
for the executor and the masks, and MatchEnv computes the scripts' labels from the plain observation.

User facts (War Thunder, 2026-10-08) behind the options:

``radar_display: "bscope"`` (or a dict ``{"mode": "bscope", ...}`` with RADAR_DISPLAY keys; all values grade D):
  * The B-scope shows no numbers. Range (vertical position) and azimuth (horizontal position, from the nose) are what a
    perfect reader of the screen gets: quantised to ``range_res_frac`` of the display range scale (the radar's
    farthest reportable range from the radar data, display_range(), or ``display_range_m``) and ``az_res_deg``, with
    optional Gaussian reading noise (``range_noise_frac`` of the scale, ``az_noise_deg``) drawn once per radar refresh
    of the contact (a new ``updated_s``) and held until the next.
  * Every contact (search, TWS, STT) shows a velocity vector: the target's own velocity, truth at the contact's last
    refresh, horizontal only (the B-scope is a 2D picture; the vertical slot is not a speed, see below), held between
    refreshes. Near the notch, r = |v_target . LOS| < ``notch_band_mps``, it is unstable: each refresh draws a missing
    vector with probability dropout_p*k, else a direction error of sigma jitter_deg*k and a speed factor error of sigma
    jitter_speed_frac*k, k = 1 - r/notch_band_mps.
  * No closure rate (unknown) and no altitude: the elevation coverage of the scan at the refresh (RadarSensor.
    elevation_coverage: the bars and the beam about the antenna's elevation) with the displayed range gives an
    altitude band; its centre goes into the elevation and altitude fields, its half width (``band_half_width``) into
    the vz slot. STT is the same (the scan the radar returns to). Reach-table times use only these values.
  * Own missiles ('shot' entities) are on the B-scope at their range / azimuth (quantised the same way, no
    altitude) while inside the display (radar azimuth limits, display range); before the seeker is active the aim point
    (the missile's datalink / INS target estimate, quantised, no altitude) goes into the vx / vy slots. The active,
    datalink and target-mark flags are kept. The 3D marker of an own missile is not used.
  * The 3D target box (range, closure, relative altitude) stays exact.
  * The team map is not part of this option: the engagement puts every radar track on the team map with its exact
    position (20 s hold), so radar_display alone still gives the policy exact positions through 'map' entities. Use
    it with map_spotting_m (the game's map rule), which takes radar tracks off the policy's map.

``launch_zone_info: False``: no per-entity "if fired now" flight / seeker-on times (the game's dynamic launch zone does
not exist for the policy). The own vector's rmax (a static head-on figure of the own missile) stays.

``map_spotting_m``: enemy map marks come only from spotting, an enemy within this distance of the plane or of a
living airborne teammate (a simplification the user accepted); radar tracks no longer mark enemies for the policy.
``map_hold_s`` (default engagement.MAP_HOLD_S = 20 s): how long the policy's enemy marks are held.

``wreck_s``: a shot-down aircraft stays a wreck for the policy aircraft's radars and eyes (engagement.wreck_state);
its enemy mark stops being refreshed and expires, a dead teammate stays on the map as a wreck for wreck_s, and an own
missile whose target died keeps showing that target's mark and the support it had while the wreck is tracked.
"""
from __future__ import annotations

from dataclasses import replace
import math
import random

from .engagement import MAP_HOLD_S, MISSILE_TRUTH, MapMark
from .escape import to_enu
from .intent import wrap

RADAR_DISPLAY = dict(range_res_frac=.005, az_res_deg=.5, range_noise_frac=0., az_noise_deg=0., display_range_m=None,
                     notch_band_mps=60., jitter_deg=60., jitter_speed_frac=.4, dropout_p=.3, band_half_width=True)
FALLBACK_DISPLAY_RANGE_M = 120000.


def _number(name, v, positive=False, at_most=None):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 or (positive and v == 0) \
            or (at_most is not None and v > at_most):
        raise ValueError(f'{name} must be a number '+('> 0' if positive else '>= 0')+
                         ('' if at_most is None else f' and <= {at_most:g}'))
    return float(v)


def radar_display_settings(value):
    """config radar_display -> the RADAR_DISPLAY settings filled in, or None (absent / None). ValueError otherwise."""
    if value is None:
        return None
    if value == 'bscope':
        value = {'mode': 'bscope'}
    if not isinstance(value, dict) or value.get('mode') != 'bscope' or set(value)-set(RADAR_DISPLAY)-{'mode'}:
        raise ValueError('radar_display must be "bscope" or {"mode": "bscope", ...} with keys from '+', '.join(RADAR_DISPLAY))
    out = dict(RADAR_DISPLAY)
    for k, v in value.items():
        if k == 'mode':
            continue
        if k == 'band_half_width':
            if not isinstance(v, bool):
                raise ValueError('radar_display band_half_width must be True or False')
            out[k] = v
        elif k == 'display_range_m':
            out[k] = None if v is None else _number('radar_display display_range_m', v, positive=True)
        else:
            out[k] = _number('radar_display '+k, v, at_most=1. if k == 'dropout_p' else None)
    return out


def view_settings(config):
    """The policy view options of a MatchEnv config, validated; None when they are all absent / off (no view)."""
    rd = radar_display_settings(config.get('radar_display'))
    lz = config.get('launch_zone_info', True)
    if not isinstance(lz, bool):
        raise ValueError('launch_zone_info must be True or False')
    spot, hold, wreck = (config.get(k) for k in ('map_spotting_m', 'map_hold_s', 'wreck_s'))
    spot = None if spot is None else _number('map_spotting_m', spot, positive=True)
    hold = None if hold is None else _number('map_hold_s', hold)
    wreck = None if wreck is None else _number('wreck_s', wreck, positive=True)
    if rd is None and lz and spot is None and hold is None and wreck is None:
        return None
    # marks: the policy's map marks are built here (policy_marks) from the spotting table (map_spotting_m) or the
    # engagement's team marks (map_hold_s / wreck_s only)
    return dict(radar_display=rd, launch_zone_info=lz, map_spotting_m=spot, map_hold_s=MAP_HOLD_S if hold is None else hold,
                wreck_s=wreck, marks=spot is not None or hold is not None or wreck is not None)


_DISPLAY_RANGE = {}


def display_range(radar):
    """The B-scope's range scale (D): the farthest range any of the radar's range-finding air waveforms can report,
    min(transceiver rangeMax, signal distance max) where given, else its detection range."""
    found = _DISPLAY_RANGE.get(radar.id)
    if found is None:
        found = 0.
        for w in (*radar.search_waveforms, *(radar.tws.waveforms if radar.tws else ())):
            if w.range_finder and w.air_target:
                caps = [x for x in (w.range_max_m, w.distance_max_m) if x]
                found = max(found, min(caps) if caps else (w.range_m or 0.))
        found = _DISPLAY_RANGE[radar.id] = found or FALLBACK_DISPLAY_RANGE_M
    return found


def quantise(x, step):
    return x if step <= 0. else round(x/step)*step


def _state_at(eng, ident, t):
    """Truth (position, velocity) of aircraft ``ident`` at ``t`` (its wreck after death, wreck_s); the latest state when
    the flight history no longer holds ``t``."""
    q = eng.planes[ident]
    if not q.alive and eng.wreck_s is not None:
        state = eng.wreck_state(ident, t)
        if state is not None:
            return state
    try:
        return q.state_at(t)
    except ValueError:
        return q.flight.state.position, q.flight.state.velocity


def _truth(eng, tid, t):
    """(position, velocity) behind a radar contact of truth id ``tid`` at its refresh time ``t``; None if unknown."""
    if tid is None:
        return None
    if tid >= MISSILE_TRUTH:
        m = next((m for m in eng.missiles if m.uid == tid-MISSILE_TRUTH), None)
        return None if m is None else (m.pos_enu, m.vel_enu)
    return _state_at(eng, tid, t)


class PolicyView:
    """One policy aircraft's view state (snapshot-copied with the env): the B-scope display state per entity key, the
    reading-noise / notch-jitter generator (its own, seeded per episode and aircraft), the memories of
    rl_observation.select_view_entities and the wreck_s record of each own missile's target."""

    def __init__(self, settings, seed, ident):
        self.settings, self.ident = settings, ident
        self.rng = random.Random(f'{seed}:policy_view:{ident}')
        self.state = {}         # B-scope: key -> refresh time, held noise, altitude band, held vector
        self.memory = {}        # shown entities (select_view_entities)
        self.full_memory = {}   # their full-information twins
        self.shots = {}         # wreck_s: own missile uid -> (target ident, datalink) while that target lived

    # -- entry point ------------------------------------------------------------------------------------------

    def show(self, obs, observed, plane, eng, reach):
        """The policy's entities for raw_entities(obs) ``observed`` (same keys and order)."""
        s, rd = self.settings, self.settings['radar_display']
        out = list(observed)
        if rd is not None or not s['launch_zone_info']:
            truths = plane.picture.truth_ids if plane.picture is not None else ()
            for i, c in enumerate(obs.radar):
                e = out[i]   # raw_entities lists the radar contacts first, in picture order
                if e.kind != 'radar':
                    raise ValueError('radar entities out of order')
                if rd is not None:
                    e = self._contact(e, c, truths[i] if i < len(truths) else None, obs, plane, eng, reach)
                if not s['launch_zone_info']:
                    e = replace(e, flight_time=None, seeker_time=None)
                out[i] = e
            if rd is not None:
                current = {out[i].key for i in range(len(obs.radar))}
                for key in [k for k in self.state if k not in current]:
                    del self.state[key]
        shots = {m.uid: m for m in eng.missiles if m.shooter is plane}
        if s['wreck_s'] is not None:
            for uid in [u for u in self.shots if u not in shots]:
                del self.shots[uid]
        for i, e in enumerate(out):
            if e.kind == 'shot':
                out[i] = self._shot(e, shots.get(e.key[1]), obs, plane, eng)
        return out

    # -- B-scope radar contacts -------------------------------------------------------------------------------

    def _scale(self, plane):
        rd = self.settings['radar_display']
        if rd['display_range_m'] is not None:
            return rd['display_range_m']
        return display_range(plane.radar.radar) if plane.radar is not None else FALLBACK_DISPLAY_RANGE_M

    def _contact(self, e, c, tid, obs, plane, eng, reach):
        rd, own = self.settings['radar_display'], obs.own
        scale_m = self._scale(plane)
        st = self.state.get(e.key)
        if st is None or st['updated'] != c.updated_s:
            st = self.state[e.key] = self._refresh(c, tid, obs, plane, eng, scale_m)
        az = quantise(wrap(c.bearing_deg-own.heading_deg)+st['az_noise'], rd['az_res_deg'])
        bearing = (own.heading_deg+az) % 360.
        vel = None if st['vector'] is None else (st['vector'][0], st['vector'][1], None)
        if c.range_m is None:   # no range finder: an angle and the elevation coverage's centre only
            return replace(e, bearing=bearing, elevation=st['el_centre'], distance=None, closure=None, position=None,
                           velocity=vel, flight_time=None, seeker_time=None)
        r = max(0., quantise(c.range_m+st['range_noise'], rd['range_res_frac']*scale_m))
        lo, hi = st['band']
        zc, half = (lo+hi)/2., (hi-lo)/2.
        ox, oy, oz = own.position
        dz = zc-oz
        h = math.sqrt(max(0., r*r-dz*dz))
        el = math.degrees(math.atan2(dz, h)) if r > 0. else 0.
        b = math.radians(bearing)
        pos = (ox+h*math.sin(b), oy+h*math.cos(b), zc)
        flight = seeker = None
        if reach is not None and self.settings['launch_zone_info']:
            aspect = 0.
            if vel is not None:
                dx, dy = ox-pos[0], oy-pos[1]
                denom = math.hypot(dx, dy)*math.hypot(vel[0], vel[1])
                if denom > 0:
                    aspect = math.degrees(math.acos(max(-1., min(1., (dx*vel[0]+dy*vel[1])/denom))))
            timing = reach.times(own.altitude_m, own.speed_mps, zc-own.altitude_m, aspect, r)
            if timing is not None:
                flight, seeker = timing
        return replace(e, bearing=bearing, elevation=el, distance=r, closure=None, position=pos, velocity=vel,
                       flight_time=flight, seeker_time=seeker, band_half=half if rd['band_half_width'] else None)

    def _refresh(self, c, tid, obs, plane, eng, scale_m):
        """A new radar picture of the contact: reading noise, altitude band and velocity vector, held until the next."""
        rd, rng = self.settings['radar_display'], self.rng
        rn = rng.gauss(0., rd['range_noise_frac']*scale_m) if rd['range_noise_frac'] > 0. else 0.
        an = rng.gauss(0., rd['az_noise_deg']) if rd['az_noise_deg'] > 0. else 0.
        cov = plane.radar.elevation_coverage() if plane.radar is not None else None
        lo_el, hi_el = (-90., 90.) if cov is None else (max(-90., cov[0]), min(90., cov[1]))
        st = dict(updated=c.updated_s, range_noise=rn, az_noise=an, el_centre=(lo_el+hi_el)/2., band=None)
        if c.range_m is not None:
            r = max(0., quantise(c.range_m+rn, rd['range_res_frac']*scale_m))
            oz = obs.own.position[2]
            lo = max(0., oz+r*math.sin(math.radians(lo_el)))
            st['band'] = (lo, max(lo, oz+r*math.sin(math.radians(hi_el))))
        st['vector'] = self._vector(c, tid, plane, eng)
        return st

    def _vector(self, c, tid, plane, eng):
        """The displayed horizontal velocity (east, north) at a refresh, or None (missing)."""
        rd, rng = self.settings['radar_display'], self.rng
        truth = _truth(eng, tid, c.updated_s)
        if truth is None:   # a missile already gone: the radar's own estimate
            if c.velocity is None or c.position is None:
                return None
            truth = (c.position, c.velocity)
        pos, vel = truth
        own = _state_at(eng, plane.ident, c.updated_s)[0]
        los = tuple(a-b for a, b in zip(pos, own))
        n = math.sqrt(sum(x*x for x in los))
        radial = abs(sum(a*b for a, b in zip(vel, los)))/n if n > 0. else math.inf
        vx, vy = vel[0], vel[1]
        band = rd['notch_band_mps']
        if band > 0. and radial < band:
            k = 1.-radial/band
            if rng.random() < rd['dropout_p']*k:
                return None
            turn = math.radians(rng.gauss(0., rd['jitter_deg']*k))
            factor = max(0., 1.+rng.gauss(0., rd['jitter_speed_frac']*k))
            cs, sn = math.cos(turn), math.sin(turn)
            vx, vy = (vx*cs+vy*sn)*factor, (vy*cs-vx*sn)*factor
        return vx, vy

    # -- own missiles -----------------------------------------------------------------------------------------

    def _shot(self, e, m, obs, plane, eng):
        aim_wreck = None
        if self.settings['wreck_s'] is not None and m is not None:
            e, aim_wreck = self._wreck_shot(e, m, plane, eng)
        if self.settings['radar_display'] is None:
            return e
        reading = None if m is None else self._read(m.pos_enu, obs, plane)
        if reading is None:
            return replace(e, bearing=None, elevation=None, distance=None, aim=None)
        aim = None
        if not m.seeker_on:
            point = aim_wreck
            if point is None and m.runtime is not None:
                track = getattr(m.runtime, 'track', None)
                if track is not None and track.valid:
                    point = to_enu(track.position)
            spot = None if point is None else self._read(point, obs, plane)
            if spot is not None:
                b = math.radians(spot[0])
                aim = (spot[1]*math.sin(b), spot[1]*math.cos(b))
        return replace(e, bearing=reading[0], elevation=None, distance=reading[1], aim=aim)

    def _read(self, point, obs, plane):
        """(bearing, range) of a point as read off the B-scope (quantised, reading noise drawn now), None outside the
        display: no radar, parked, beyond the radar's azimuth limits or the range scale."""
        rd, own = self.settings['radar_display'], obs.own
        if plane.radar is None or obs.grounded:
            return None
        scale_m = self._scale(plane)
        d = tuple(a-b for a, b in zip(point, own.position))
        rng_m = math.sqrt(sum(x*x for x in d))
        az = wrap(math.degrees(math.atan2(d[0], d[1]))-own.heading_deg)
        if abs(az) > (plane.radar.radar.field_of_regard_deg or 60.) or rng_m > scale_m:
            return None
        if rd['range_noise_frac'] > 0.:
            rng_m += self.rng.gauss(0., rd['range_noise_frac']*scale_m)
        if rd['az_noise_deg'] > 0.:
            az += self.rng.gauss(0., rd['az_noise_deg'])
        az = quantise(az, rd['az_res_deg'])
        return (own.heading_deg+az) % 360., max(0., quantise(rng_m, rd['range_res_frac']*scale_m))

    def _wreck_shot(self, e, m, plane, eng):
        """wreck_s: while the target an own missile was guided at is a wreck, the missile keeps showing that target's
        mark and the support it had, as long as the own radar tracks the wreck (the sim's datalink drops and the
        missile may retarget at the death; the policy must not see that). Returns (entity, aim point or None)."""
        rec = self.shots.get(m.uid)
        if rec is None and m.wreck_shot:
            rec = self.shots[m.uid] = (m.target.ident, True)
        if rec is not None and not eng.planes[rec[0]].alive:
            state = eng.wreck_state(rec[0])
            if state is not None:
                support = rec[1] and rec[0] in plane.tracked
                return replace(e, mark_id=eng.mark_ids[rec[0]], support=support), (state[0] if support else None)
        if m.target.alive:
            self.shots[m.uid] = (m.target.ident, m.datalink)
        return e, None


# -- map marks (map_spotting_m, map_hold_s, wreck_s) -----------------------------------------------------------

def update_spotting(eng, tables, radius):
    """map_spotting_m: put every living airborne enemy within ``radius`` of a living airborne member of a team on that
    team's policy mark table ({ident: (x, y, z, time)}); a parked enemy leaves it (as the engagement's landing does)."""
    t = eng.time
    for team in (0, 1):
        spotters = [p.own.position for p in eng.live if p.team == team and not p.grounded]
        table = tables[team]
        for q in eng.live:
            if q.team == team:
                continue
            if q.grounded:
                table.pop(q.ident, None)
                continue
            pos = q.own.position
            if any(math.dist(pos, s) <= radius for s in spotters):
                table[q.ident] = (pos[0], pos[1], pos[2], t)


def policy_marks(eng, plane, settings, table):
    """The policy's map marks: living teammates now (and, wreck_s, dead ones as wrecks until wreck_s), then the enemies
    of ``table`` (the spotting table, or the team's engagement marks) held map_hold_s; a dead enemy's mark is kept until
    it expires under wreck_s (no longer refreshed), else it goes at the death as before. Enemy marks have no altitude."""
    t, wreck = eng.time, settings['wreck_s'] is not None
    marks = []
    for q in eng.live:
        if q.team == plane.team and q is not plane:
            v = q.own.velocity
            marks.append(MapMark(eng.mark_ids[q.ident], q.own.position[0], q.own.position[1], q.own.position[2], t,
                                 True, math.degrees(math.atan2(v[0], v[1])) % 360.))
    if wreck:
        for ident in sorted(eng.wrecks):
            q = eng.planes[ident]
            state = eng.wreck_state(ident) if q.team == plane.team and q is not plane else None
            if state is not None:
                (x, y, z), v = state
                marks.append(MapMark(eng.mark_ids[ident], x, y, z, t, True, math.degrees(math.atan2(v[0], v[1])) % 360.))
    for ident in sorted(table):
        x, y, z, seen = table[ident]
        q = eng.planes[ident]
        if t-seen <= settings['map_hold_s'] and not q.grounded and (q.alive or wreck):
            marks.append(MapMark(eng.mark_ids[ident], x, y, None, seen))
    return tuple(marks)


def without_wrecks(raw, plane, eng):
    """wreck_s: the observation a script reads for a policy aircraft (its labels): no wreck radar contacts, sightings,
    boxes or contrails (the scripts know who is dead)."""
    dead = {eng.mark_ids[q.ident]: q.ident for q in eng.planes if not q.alive}
    if not dead:
        return raw
    truths = plane.picture.truth_ids if plane.picture is not None else ()
    keep = [i for i, _ in enumerate(raw.radar)
            if not (i < len(truths) and truths[i] < MISSILE_TRUTH and not eng.planes[truths[i]].alive)]
    if len(keep) == len(raw.radar) and not any(s.ref in dead for s in (*raw.visual, *raw.boxes, *raw.contrails)):
        return raw
    return replace(raw, radar=tuple(raw.radar[i] for i in keep),
                   radar_missiles=tuple(raw.radar_missiles[i] for i in keep) if raw.radar_missiles else raw.radar_missiles,
                   visual=tuple(s for s in raw.visual if s.ref not in dead),
                   boxes=tuple(b for b in raw.boxes if b.ref not in dead),
                   contrails=tuple(c for c in raw.contrails if c.ref not in dead))
