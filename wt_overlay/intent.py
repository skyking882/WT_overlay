"""Shared, observation-only intent decoding and player execution model (training spec 3–5).

Pointers are stable observation keys internally; wire actions contain indices into the
current entity list. The actor's sampled action is never replaced by executed commands.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
import math
import random

from .engagement import Action, RadarCommand, STT_TRACK, equipment_data
from .flight import FlightCommand, KeyboardCommand

HEAD_NAMES = ('maneuver_ref','target','view_object','maneuver','vertical','speed','chaff',
              'radar_mode','antenna','weapon','view_mode','look_az','look_el','kb_roll','kb_pitch')
CAT_SIZES = dict(maneuver=11,vertical=5,speed=3,chaff=3,radar_mode=3,antenna=5,
                weapon=2,view_mode=3,look_az=8,look_el=3,kb_roll=3,kb_pitch=3)
POINTERS = ('maneuver_ref','target','view_object')
SAMPLE_ORDER = ('view_mode','maneuver_ref','target','view_object','maneuver','vertical','speed',
                'chaff','radar_mode','antenna','weapon','look_az','look_el','kb_roll','kb_pitch')
ANTENNA_DEG = (20.,10.,0.,-10.,-20.)
# User-specified launchAngleMax entries; aliases are carried weapon ids (not seeker FOV).
ALL_ASPECT = frozenset(('su_r_77_1','su_rvv_sd','cn_pl12a','us_aim_120d','fr_mica_em'))


def wrap(deg):
    return (deg+180.) % 360.-180.


def launch_limit(aircraft, missile):
    eq = equipment_data().equipment.get(aircraft)
    radar = equipment_data().radars.get(eq.radar) if eq else None
    return min(radar.field_of_regard_deg or 60., 180. if missile in ALL_ASPECT else 60.) if radar else 0.


@dataclass(frozen=True)
class Intent:
    maneuver_ref: object = None
    target: object = None
    view_object: object = None
    maneuver: int = 0
    vertical: int = 0
    speed: int = 0
    chaff: int = 0
    radar_mode: int = 0
    antenna: int = 2
    weapon: int = 0
    view_mode: int = 0
    look_az: int = 0
    look_el: int = 1
    kb_roll: int = 1
    kb_pitch: int = 1

    def indices(self, entities):
        keys = [e.key for e in entities]
        return {h: (keys.index(getattr(self,h)) if getattr(self,h) in keys else len(keys))
                if h in POINTERS else getattr(self,h) for h in HEAD_NAMES}

    @classmethod
    def from_indices(cls, action, entities):
        n = len(entities)
        values = {}
        if set(action) != set(HEAD_NAMES):
            raise ValueError('actions must contain exactly the 15 intent heads')
        for h in HEAD_NAMES:
            i = action[h]
            k = n+1 if h in POINTERS else CAT_SIZES[h]
            if type(i) is not int or not 0 <= i < k:
                raise ValueError('invalid action index for '+h)
            values[h] = (None if i == n else entities[i].key) if h in POINTERS else i
        return cls(**values)


def selected_mask(head, n, masks, chosen):
    row = masks[head]
    if head in ('maneuver_ref','view_object','look_az','look_el','kb_roll','kb_pitch'):
        return row[chosen['view_mode']]
    if head in ('maneuver','vertical'):
        return row[chosen['view_mode']][int(chosen['maneuver_ref'] == n)]
    if head in ('weapon','radar_mode'):
        return row[chosen['target']]
    return row


def legalize(want, entities, masks):
    """Project a script's discrete proposal onto its observation's legal heads."""
    n, chosen = len(entities), {}
    values = want.indices(entities)
    for h in SAMPLE_ORDER:
        row = selected_mask(h,n,masks,chosen)
        i = values[h]
        chosen[h] = i if row[i] else min((j for j,v in enumerate(row) if v),key=lambda j:abs(j-i))
    return {h:chosen[h] for h in HEAD_NAMES}


@dataclass(frozen=True)
class DelayPath:
    median_range: tuple
    sigma: float
    bounds: tuple

    def __post_init__(self):
        if len(self.median_range)!=2 or len(self.bounds)!=2 or self.sigma<0 \
                or min(self.median_range)<=0 or min(self.bounds)<0 \
                or self.median_range[0]>self.median_range[1] or self.bounds[0]>self.bounds[1]:
            raise ValueError('invalid delay distribution')


class IntentExecutor:
    """Hold starts at publication. Visible new alerts release it, never truth.

    A changed proposal is one event. Repeating it does not resample or restart
    its delay. One-shot weapon/chaff requests are events even when repeated.
    Pending events execute in publication order; older delayed events cannot
    undo a newer event that already executed. All state is snapshot-copyable.
    """
    def __init__(self, seed, *, path='follow', delay=None, hold_s=2., error_deg=5.,
                 reject_p=.05, authority=1., home_xy=(0.,0.)):
        if path not in ('follow','autonomous'):
            raise ValueError('unknown execution path')
        if hold_s<0 or error_deg<0 or not 0<=reject_p<=1 or not 0<authority<=1:
            raise ValueError('invalid execution parameters')
        self.rng = random.Random(seed)
        settings = {'follow':DelayPath((.5,1.2),.3,(.25,2.5)),
                    'autonomous':DelayPath((1.,2.5),.4,(.5,5.))}
        for name, d in (delay or {}).items():
            if name not in settings:
                raise ValueError('unknown delay path '+name)
            old=settings[name]
            settings[name]=DelayPath(tuple(d.get('median_range',old.median_range)),
                                     d.get('sigma',old.sigma),tuple(d.get('bounds',old.bounds)))
        self.path, self.distribution = path, settings[path]
        self.personal_median = self.rng.uniform(*self.distribution.median_range)
        self.hold_s,self.error_deg,self.reject_p,self.authority = hold_s,error_deg,reject_p,authority
        self.home_xy = home_xy
        self.published = self.executed = Intent()
        self.hold_until = self.published_at = 0.
        self.head_since = {h:0. for h in HEAD_NAMES}
        self.alerts = set()
        self.pending = []
        self.sequence = self.applied_sequence = 0
        self.last_delay = None
        self.last_noise = 0.
        self.initial = True
        self.aim_heading = self.aim_altitude = None
        self.free_since = None
        self.release_count = self.rejected = 0
        self.last_chaff = -1e9

    def notice(self, obs):
        alerts = {('r',c.contact_id,'m' if c.missile_warning else 'l') for c in obs.rwr
                  if c.illuminated and (c.missile_warning or c.tracking)}
        alerts.update(('maw',s.ref) for s in obs.maw)
        alerts.update(('marker',s.ref) for s in obs.missile_marks)
        if alerts-self.alerts:
            self.hold_until = obs.time_s
            self.release_count += 1
        self.alerts = alerts

    def held(self, now):
        return self.published.view_mode == 0 and now < self.hold_until-1e-9

    def publish(self, intent, obs):
        self.notice(obs)
        old, now = self.published, obs.time_s
        leaving = old.view_mode != 0 and intent.view_mode == 0
        if self.held(now) and intent.view_mode==0 and not leaving:
            if any(getattr(intent,h)!=getattr(old,h) for h in ('maneuver_ref','maneuver','vertical')):
                raise ValueError('held high-level intent cannot change without a visible new warning')
        changed = self.initial or intent != old or intent.weapon or intent.chaff==1
        if not changed:
            return
        high = any(getattr(intent,h)!=getattr(old,h) for h in ('maneuver_ref','maneuver','vertical'))
        if intent.view_mode==0 and (high or leaving or self.initial):
            self.hold_until = now+self.hold_s
        for h in HEAD_NAMES:
            if self.initial or getattr(intent,h)!=getattr(old,h):
                self.head_since[h] = now
        self.initial = False
        self.published, self.published_at = intent, now
        self.sequence += 1
        if self.rng.random() < self.reject_p:
            self.rejected += 1
            return
        d = self.distribution
        delay = max(d.bounds[0],min(d.bounds[1],self.rng.lognormvariate(math.log(self.personal_median),d.sigma)))
        self.last_delay = delay
        noise = self.rng.gauss(0.,self.error_deg)
        self.pending.append((now+delay,self.sequence,intent,noise,leaving))

    def advance(self, eng, plane, obs, entities):
        now = eng.time
        ready = sorted((ev for ev in self.pending if ev[0]<=now+1e-9),key=lambda ev:(ev[0],ev[1]))
        self.pending = [ev for ev in self.pending if ev[0]>now+1e-9]
        for _,seq,intent,noise,leaving in ready:
            if seq <= self.applied_sequence:
                continue
            self.applied_sequence = seq
            self.last_noise = noise
            self.executed = intent
            if intent.view_mode==0:
                self.free_since = None
                self._aim(obs,entities,intent,noise,leaving)
            elif self.free_since is None:
                self.free_since = now
            action = self._action(plane,obs,entities,intent,fire_once=True)
            eng.apply(plane,action)
        # Continuous keyboard flight, radar pointing and tracking camera follow
        # the latest available observation, not the entity's true position.
        intent = self.executed
        if intent.view_mode==0 and self.aim_heading is not None and (intent.maneuver_ref is not None or intent.maneuver in (7,10)):
            self._aim(obs,entities,intent,self.last_noise,False)
        eng.apply(plane,self._action(plane,obs,entities,intent,fire_once=False))
        camera = plane.camera
        if intent.view_mode==2:
            camera.point(plane.own.heading_deg+45.*intent.look_az,(-30.,0.,30.)[intent.look_el],2)
        elif intent.view_mode==1:
            entity = next((e for e in entities if e.key==intent.view_object),None)
            if entity is not None and entity.bearing is not None:
                camera.point(entity.bearing,entity.elevation or 0.,1)
        else:
            aim_el=plane.own.pitch_deg if self.aim_altitude is None else max(-30.,min(30.,
                     math.degrees(math.asin(max(-1.,min(1.,(self.aim_altitude-obs.own.altitude_m)/max(1.,6*obs.own.speed_mps)))))))
            camera.point(self.aim_heading if self.aim_heading is not None else plane.own.heading_deg,
                         aim_el,0)
        camera.advance(plane.own,1/48)
        if intent.chaff==2 and now-self.last_chaff>=.5 and plane.chaff:
            eng.drop_chaff(plane,1)
            self.last_chaff=now

    def _aim(self, obs, entities, i, noise, leaving):
        own=obs.own
        ref=next((e for e in entities if e.key==i.maneuver_ref),None)
        bearing=own.heading_deg if ref is None or ref.bearing is None else ref.bearing
        offsets={1:0.,2:-40.,3:40.,4:-90.,5:90.,6:180.,8:-40.,9:40.}
        if i.maneuver==7:
            x,y,_=own.position
            half=obs.map_half_m
            margin=own.speed_mps**2/(2*9.80665)+own.speed_mps*7.+3000.
            if min(half-abs(x),half-abs(y))<margin:
                if half-abs(x)<half-abs(y):
                    sy=1. if own.velocity[1]>=0. else -1.
                    if half-abs(y)<margin:
                        sy=-1. if y>0. else 1.
                    heading=math.degrees(math.atan2(-(.35 if x>0. else -.35),sy))
                else:
                    sx=1. if own.velocity[0]>=0. else -1.
                    if half-abs(x)<margin:
                        sx=-1. if x>0. else 1.
                    heading=math.degrees(math.atan2(sx,-(.35 if y>0. else -.35)))
            else:
                heading=math.degrees(math.atan2(-x,-y))
        elif i.maneuver==10:
            dx,dy=self.home_xy[0]-own.position[0],self.home_xy[1]-own.position[1]
            heading=own.heading_deg+3. if math.hypot(dx,dy)<12000. else math.degrees(math.atan2(dx,dy))
        elif i.maneuver in offsets:
            heading=bearing+offsets[i.maneuver]
        else:
            heading=own.heading_deg if leaving or self.aim_heading is None else self.aim_heading
        if i.maneuver or leaving or self.aim_heading is None:
            self.aim_heading=(heading+noise)%360.
        if i.vertical or leaving or self.aim_altitude is None:
            self.aim_altitude=(own.altitude_m,11000.,8000.,100.,max(100.,own.altitude_m-3000.))[i.vertical]

    def _action(self, plane, obs, entities, i, fire_once):
        if i.view_mode:
            flight=KeyboardCommand(i.kb_roll-1,i.kb_pitch-1,(1,0,-1)[i.speed],i.speed==2,self.authority)
        else:
            from .rl_observation import energy_state
            if getattr(self,'performance_at',None)!=obs.time_s:
                self.performance_ratio=energy_state(obs.own,plane.flight.model,plane.flight.load_limits)[2]
                self.performance_at=obs.time_s
            ratio=self.performance_ratio
            warning=bool(obs.maw or obs.missile_marks or any(c.missile_warning for c in obs.rwr))
            max_load=2. if i.weapon else 3. if not warning and ratio is not None and ratio<1. else 9.
            flight=FlightCommand(heading_deg=self.aim_heading,altitude_m=self.aim_altitude,max_load=max_load,
                                 throttle_percent=(110.,85.,0.)[i.speed],airbrake_allowed=i.speed==2,
                                 speed_mps=50. if i.speed==2 else None)
        target=next((e for e in entities if e.key==i.target and e.track_id is not None),None)
        track=None if target is None else target.track_id
        data=plane.radar.radar if plane.radar else None
        patterns=data.tws.patterns if data and data.tws else ()
        pattern=0
        if patterns and i.radar_mode==1:
            pattern=min(range(len(patterns)),key=lambda j:patterns[j].half_width_deg or 180.)
        mode='stt' if i.radar_mode==2 and track is not None else 'tws'
        radar_ref=target or next((e for e in entities if e.key==i.maneuver_ref and e.bearing is not None),None)
        az=0. if radar_ref is None or radar_ref.bearing is None else wrap(radar_ref.bearing-plane.own.heading_deg)
        radar=RadarCommand(mode,pattern,az,ANTENNA_DEG[i.antenna],track)
        return Action(flight,radar,track if fire_once and i.weapon else None,
                      1 if fire_once and i.chaff==1 else 0)


def from_flight_action(action, obs, entities, *, phase='', home_xy=(0.,0.), guard=False):
    """Quantise an archetype's geometric proposal to the public 15 heads.

    No continuous command bypasses the executor. The legacy geometric adapter
    remains available to existing callers of Pilot.decide().
    """
    f, own = action.flight, obs.own
    target=next((e for e in entities if e.track_id==action.fire),None) if action.fire is not None else None
    # A script never takes a missile track (radar_sees_missiles) as its reference or radar target.
    candidates=[e for e in entities if e.bearing is not None and e.kind in ('radar','box','map','visual','contrail','rwr')
                and not e.friend and e.missile is None]
    ref=target or (min(candidates,key=lambda e:abs(wrap(e.bearing-(own.heading_deg+action.radar.azimuth_deg)))) if candidates and action.radar else None)
    heading=own.heading_deg if f.heading_deg is None else f.heading_deg
    if f.direction is not None:
        heading=math.degrees(math.atan2(f.direction[0],f.direction[1]))%360.
    relative=wrap(heading-(ref.bearing if ref else own.heading_deg))
    offsets=(0.,0.,-40.,40.,-90.,90.,180.,0.,-40.,40.,0.)
    maneuver=min((1,2,3,4,5,6),key=lambda j:abs(wrap(relative-offsets[j])))
    if phase=='home':
        maneuver=10
    if guard:
        maneuver=7
    altitude=own.altitude_m if f.altitude_m is None else f.altitude_m
    vertical=0 if abs(altitude-own.altitude_m)<400. else min(range(1,4),key=lambda j:abs(altitude-(11000.,8000.,100.)[j-1]))
    if f.direction and f.direction[2]<-.05:
        vertical=4
    if ref is None and maneuver in (1,2,3,4,5,8,9):
        # The first flank is an absolute look-direction reference observed from
        # own state; absent a contact the public actions still allow a reversal.
        maneuver=0
    speed=2 if f.airbrake_allowed else 0 if f.speed_mps is None else 1
    radar_target=target
    if radar_target is None and action.radar is not None:
        tracks=[e for e in entities if e.kind=='radar' and e.track_id is not None and e.bearing is not None
                and e.missile is None]
        if tracks:
            wanted=own.heading_deg+action.radar.azimuth_deg
            radar_target=min(tracks,key=lambda e:abs(wrap(e.bearing-wanted)))
    antenna=min(range(5),key=lambda j:abs((action.radar.elevation_deg if action.radar else 0.)-ANTENNA_DEG[j]))
    return Intent(maneuver_ref=None if ref is None else ref.key,target=None if radar_target is None else radar_target.key,
                  maneuver=maneuver,vertical=vertical,speed=speed,chaff=1 if action.chaff else 0,
                  antenna=antenna,weapon=int(target is not None))
