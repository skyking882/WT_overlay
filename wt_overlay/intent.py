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
from .fm import atmosphere

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
# vertical_mode 'angle' (opt-in, IntentExecutor): the vertical head is a flight-path angle kept until another option
# executes. 0 level: capture the altitude when the option is (re)selected; 1 steep climb, 2 shallow climb, 3 shallow
# descent, 4 dive (degrees, D). The default 'altitude' keeps 0 hold target / 11 km / 8 km / deck / 3 km lower.
VERTICAL_MODES = ('altitude','angle')
VERTICAL_ANGLES_DEG = (None,25.,10.,-10.,-30.)
ANGLE_CLIMB_MIN_SPEED_MPS = 250.   # climb options stop climbing below this speed (the scripts' rule, archetypes)
# from_flight_action in 'angle' mode: a script's altitude target -> option (D). Within the deadband level; farther
# than STEEP_M climb steeply / dive, else the shallow angle. A dive toward a floor levels above it by the deadband,
# LEAD_S of sink rate and a pull-out at PULL_G net (3 g). (A narrower deadband near the deck made crawlers bounce
# between descent and level into the 100 m floor: each switch is a new event with its own delay, so the level-off
# lands delay x sink rate lower.)
SCRIPT_ALT_DEADBAND_M, SCRIPT_ALT_STEEP_M, SCRIPT_FLOOR_LEAD_S, SCRIPT_FLOOR_PULL_G = 300., 2500., 2., 2.
VIEW_MODELS = ('full','object_only')   # view_model: 'object_only' masks look-direction and the keyboard heads


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
    The hold on maneuver_ref / maneuver / vertical survives free look: entering
    or leaving it neither clears nor restarts a running hold, and the exit must
    keep the held values (aim_heads, the last mouse-aim publication). Leaving
    free look is never rejected: the pilot always takes back normal control (the
    rejection draw is made and ignored). A rejected proposal that is still being
    published is noticed again renotice_s after it was passed over (D) and
    offered afresh. RNG draws per offer, in order: rejection, then delay and
    error if it is taken up.

    Options (all opt-in, the defaults keep the old behaviour; MatchEnv passes them in ``execution``):
    vertical_mode 'angle' (VERTICAL_ANGLES_DEG), view_model 'object_only' (look-object keeps flying the last mouse-aim
    intent; look-direction and the keyboard are masked), entity_memory_s (rl_observation.select_entities keeps any enemy
    entity seen within that many seconds, extrapolated), maw_entities False (no MAW entities, own MAW flag or MAW hold
    release: every aircraft gives the actor the same inputs).
    """
    def __init__(self, seed, *, path='follow', delay=None, hold_s=2., error_deg=5.,
                 reject_p=.05, renotice_s=2., authority=1., home_xy=(0.,0.), ground_floor=True, deck_m=100.,
                 airfield=None, vertical_mode='altitude', view_model='full', entity_memory_s=None, maw_entities=True):
        if path not in ('follow','autonomous'):
            raise ValueError('unknown execution path')
        if hold_s<0 or error_deg<0 or not 0<=reject_p<=1 or not renotice_s>=0 or not 0<authority<=1:
            raise ValueError('invalid execution parameters')
        if vertical_mode not in VERTICAL_MODES or view_model not in VIEW_MODELS or not isinstance(maw_entities,bool) \
                or (entity_memory_s is not None and (isinstance(entity_memory_s,bool) or
                                                     not isinstance(entity_memory_s,(int,float)) or not entity_memory_s>0)):
            raise ValueError("vertical_mode must be 'altitude' or 'angle', view_model 'full' or 'object_only', "
                             'entity_memory_s None or a positive number of seconds, maw_entities True or False')
        self.vertical_mode, self.view_model = vertical_mode, view_model
        self.entity_memory_s = None if entity_memory_s is None else float(entity_memory_s)
        self.maw_entities = maw_entities
        self.rng = random.Random(seed)
        # ground_floor False: manoeuvre commands carry no ground floor (no pull-out help); deck_m: the lowest altitude
        # the deck / dive options aim for.
        self.ground_floor, self.deck_m = bool(ground_floor), float(deck_m)
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
        self.renotice_s = renotice_s
        self.unheeded_at = None   # when the latest publication was last rejected (None: it was taken up)
        self.home_xy = home_xy
        # airfield (opt-in, engagement.airfield_settings): go-home (maneuver 10) flies straight to home_xy, inside
        # approach_m down to approach_alt_m at approach_ias_kmh; plane.want_home tells the engagement to land.
        self.airfield, self.approach, self.takeoffs = airfield, False, 0
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
        # vertical_mode 'angle': the flight-path angle flown (None: capture aim_altitude) and the option it came from
        self.aim_climb = self.aim_vertical = None
        # maneuver_ref / maneuver / vertical of the last mouse-aim publication: what a hold keeps through free look
        self.aim_heads = (None,0,0)
        self.free_since = None
        self.release_count = self.rejected = 0
        self.last_chaff = -1e9

    def notice(self, obs):
        alerts = {('r',c.contact_id,'m' if c.missile_warning else 'l') for c in obs.rwr
                  if c.illuminated and (c.missile_warning or c.tracking)}
        if self.maw_entities:
            alerts.update(('maw',s.ref) for s in obs.maw)
        alerts.update(('marker',s.ref) for s in obs.missile_marks)
        if alerts-self.alerts:
            self.hold_until = obs.time_s
            self.release_count += 1
        self.alerts = alerts

    def held(self, now):
        # Whatever the view: free look neither clears nor bypasses the hold.
        return now < self.hold_until-1e-9

    def mouse(self, intent):
        """True when ``intent`` is flown by the mouse-aim command: the aim view, and look-object under view_model
        'object_only' (the camera follows the object, the aircraft keeps the last mouse-aim intent)."""
        return intent.view_mode==0 or (intent.view_mode==1 and self.view_model=='object_only')

    def publish(self, intent, obs):
        self.notice(obs)
        old, now = self.published, obs.time_s
        leaving = not self.mouse(old) and self.mouse(intent)   # back from keyboard flight
        held, aiming = self.held(now), self.mouse(intent)
        heads = (intent.maneuver_ref,intent.maneuver,intent.vertical)
        if held and aiming and heads!=self.aim_heads:
            raise ValueError('held high-level intent cannot change without a visible new warning')
        if aiming and intent.view_mode!=0 and heads!=self.aim_heads:
            raise ValueError('look-object (view_model object_only) keeps the last mouse-aim intent')
        changed = self.initial or intent != old or intent.weapon or intent.chaff==1
        if not changed:
            if self.unheeded_at is not None and now >= self.unheeded_at+self.renotice_s-1e-9:
                self._offer(intent,now,False)
            return
        high = aiming and heads!=self.aim_heads
        # A change of the high-level heads, the first publication and taking back mouse flight start a hold; a hold
        # that is running is kept as it is.
        if aiming and (high or self.initial or (leaving and not held)):
            self.hold_until = now+self.hold_s
        if aiming:
            self.aim_heads = heads
        for h in HEAD_NAMES:
            if self.initial or getattr(intent,h)!=getattr(old,h):
                self.head_since[h] = now
        self.initial = False
        self.published, self.published_at = intent, now
        self.sequence += 1
        self._offer(intent,now,leaving)

    def _offer(self, intent, now, leaving):
        # A re-notice keeps the publication's sequence number: it is the same event, taken up late. Leaving free look
        # is never rejected (the draw is made and ignored, so later draws stay in order): the pilot always takes back
        # normal control; the hold fix above already keeps the exit from changing the held heads.
        if self.rng.random() < self.reject_p and not leaving:
            self.rejected += 1
            self.unheeded_at = now
            return
        self.unheeded_at = None
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
                # Overtaken by a newer publication that already executed: its persistent heads are stale, but a
                # one-shot press (fire, single chaff) still happens, at the target it was pressed for. (Before this,
                # a fire request followed by a quicker 'no fire' publication was silently dropped: scripts got 23
                # launches from 33 requests.)
                if intent.weapon or intent.chaff==1:
                    shot = replace(self.executed, weapon=intent.weapon, chaff=1 if intent.chaff==1 else self.executed.chaff,
                                   target=intent.target if intent.weapon else self.executed.target)
                    eng.apply(plane,self._action(plane,obs,entities,shot,fire_once=True))
                continue
            self.applied_sequence = seq
            self.last_noise = noise
            # Back from the keyboard also when that publication was passed over and taken up on a later notice.
            leaving = leaving or (not self.mouse(self.executed) and self.mouse(intent))
            self.executed = intent
            if intent.view_mode==0:
                self.free_since = None
            elif self.free_since is None:
                self.free_since = now
            if self.mouse(intent):
                self._aim(obs,entities,intent,noise,leaving)
            action = self._action(plane,obs,entities,intent,fire_once=True)
            eng.apply(plane,action)
        if self.airfield is not None:
            plane.want_home = self.executed.maneuver==10
            if plane.takeoffs!=self.takeoffs:
                # Just took off: aim afresh from the runway heading (the old aim pointed at the airfield).
                # (vertical_mode 'angle': the vertical option is taken up afresh, so level captures the take-off altitude)
                self.takeoffs,self.aim_heading,self.aim_vertical = plane.takeoffs,None,None
                if self.aim_altitude is None:
                    self.aim_altitude = plane.own.position[2]
                if self.mouse(self.executed):
                    self._aim(obs,entities,self.executed,self.last_noise,False)
        # Continuous keyboard flight, radar pointing and tracking camera follow
        # the latest available observation, not the entity's true position.
        intent = self.executed
        if self.mouse(intent) and self.aim_heading is not None and (intent.maneuver_ref is not None or intent.maneuver in (7,10)):
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
            aim_el=max(-30.,min(30.,self.aim_climb)) if self.aim_climb is not None else \
                plane.own.pitch_deg if self.aim_altitude is None else max(-30.,min(30.,
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
        self.approach=False
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
            if self.airfield is not None:   # straight to the airfield; the approach inside approach_m (_action)
                heading=math.degrees(math.atan2(dx,dy))
                self.approach=math.hypot(dx,dy)<=self.airfield['approach_m']
            else:
                heading=own.heading_deg+3. if math.hypot(dx,dy)<12000. else math.degrees(math.atan2(dx,dy))
        elif i.maneuver in offsets:
            heading=bearing+offsets[i.maneuver]
        else:
            heading=own.heading_deg if leaving or self.aim_heading is None else self.aim_heading
        if i.maneuver or leaving or self.aim_heading is None:
            self.aim_heading=(heading+noise)%360.
        if self.vertical_mode=='angle':
            # Persistent: only another option (or taking back mouse flight) changes it; level captures the altitude now.
            if i.vertical!=self.aim_vertical or leaving or self.aim_altitude is None:
                self.aim_vertical,self.aim_climb,self.aim_altitude=i.vertical,VERTICAL_ANGLES_DEG[i.vertical],own.altitude_m
        elif i.vertical or leaving or self.aim_altitude is None:
            self.aim_altitude=(own.altitude_m,11000.,8000.,self.deck_m,max(self.deck_m,own.altitude_m-3000.))[i.vertical]

    def _action(self, plane, obs, entities, i, fire_once):
        if not self.mouse(i):
            flight=KeyboardCommand(i.kb_roll-1,i.kb_pitch-1,(1,0,-1)[i.speed],i.speed==2,self.authority)
        else:
            from .rl_observation import energy_state
            if getattr(self,'performance_at',None)!=obs.time_s:
                self.performance_ratio=energy_state(obs.own,plane.flight.model,plane.flight.load_limits)[2]
                self.performance_at=obs.time_s
            ratio=self.performance_ratio
            warning=bool(obs.maw or obs.missile_marks or any(c.missile_warning for c in obs.rwr))
            max_load=2. if i.weapon else 3. if not warning and ratio is not None and ratio<1. else 9.
            altitude,speed,brake=self.aim_altitude,50. if i.speed==2 else None,i.speed==2
            extra={} if self.ground_floor else {'floor_m':None}
            climb=None
            if self.approach:   # airfield approach: down to approach_alt_m, throttle back (airbrake) to approach_ias_kmh
                a=self.airfield
                altitude,brake=a['approach_alt_m'],True
                speed=a['approach_ias_kmh']/3.6/math.sqrt(atmosphere(max(0.,min(19999.,obs.own.altitude_m)))[0]/1.225)
            elif self.aim_climb is not None:   # vertical_mode 'angle': hold the flight-path angle, climbs not below 250 m/s
                altitude,climb=None,self.aim_climb
                if climb>0.:
                    extra=dict(extra,min_speed_mps=ANGLE_CLIMB_MIN_SPEED_MPS)
            flight=FlightCommand(heading_deg=self.aim_heading,altitude_m=altitude,climb_deg=climb,max_load=max_load,
                                 throttle_percent=(110.,85.,0.)[i.speed],airbrake_allowed=brake,
                                 speed_mps=speed,**extra)
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


def angle_option(f, own):
    """vertical_mode 'angle': the vertical option for a script's FlightCommand ``f``. An altitude target within the
    deadband levels (0); above it a climb (1 steep beyond SCRIPT_ALT_STEEP_M, else 2), below it a descent (4 dive beyond
    SCRIPT_ALT_STEEP_M, else 3). A flight-path command (direction / climb_deg) takes the nearest angle; descending toward
    its floor it levels in time: the deadband, SCRIPT_FLOOR_LEAD_S of sink rate and a SCRIPT_FLOOR_PULL_G pull-out above
    the floor."""
    alt=own.altitude_m
    if f.direction is not None or (f.altitude_m is None and f.climb_deg is not None):
        if f.direction is not None:
            n=math.sqrt(sum(x*x for x in f.direction))
            gamma=0. if n<1e-9 else math.degrees(math.asin(max(-1.,min(1.,f.direction[2]/n))))
        else:
            gamma=f.climb_deg
        sink=max(0.,-own.velocity[2])
        if gamma<0. and f.floor_m is not None and alt-f.floor_m<SCRIPT_ALT_DEADBAND_M+SCRIPT_FLOOR_LEAD_S*sink+\
                sink*sink/(2.*SCRIPT_FLOOR_PULL_G*9.80665):
            return 0
        return min(range(5),key=lambda j:abs(gamma-(VERTICAL_ANGLES_DEG[j] or 0.)))
    target=alt if f.altitude_m is None else f.altitude_m
    gap=target-alt
    if abs(gap)<SCRIPT_ALT_DEADBAND_M:
        return 0
    if gap>0.:
        return 1 if gap>SCRIPT_ALT_STEEP_M else 2
    return 4 if -gap>SCRIPT_ALT_STEEP_M else 3


def from_flight_action(action, obs, entities, *, phase='', home_xy=(0.,0.), guard=False, vertical_mode=None):
    """Quantise an archetype's geometric proposal to the public 15 heads.

    No continuous command bypasses the executor. The legacy geometric adapter
    remains available to existing callers of Pilot.decide(). ``vertical_mode``
    (None: the one ``entities`` carries, rl_observation.EntityList, from the
    script's own executor; 'altitude' without one) picks the vertical options:
    'altitude' quantises the target to 11 km / 8 km / 100 m, 'angle' translates
    it into level / climb / descent (angle_option).
    """
    f, own = action.flight, obs.own
    mode=vertical_mode or getattr(entities,'vertical_mode','altitude')
    target=next((e for e in entities if e.track_id==action.fire),None) if action.fire is not None else None
    # A script never takes a missile track (radar_sees_missiles) as its reference or radar target, nor an entity only
    # carried from memory (entity_memory_s).
    candidates=[e for e in entities if e.bearing is not None and e.kind in ('radar','box','map','visual','contrail','rwr')
                and not e.friend and e.missile is None and not getattr(e,'memory',False)]
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
    if mode=='angle':
        vertical=angle_option(f,own)
    else:
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
