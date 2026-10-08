"""Pure-Python feature encoding and precomputed conditional masks (spec 2,13.1).

Feature layouts are named below; each scalar is followed in the second half by
its validity flag. Selection, memory, and masks consume Observation only.
The separate truth encoder is called only after actor features are complete.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
import json
from functools import lru_cache

from .intent import HEAD_NAMES, CAT_SIZES, POINTERS, launch_limit, wrap
from .fm import atmosphere
from .engagement import equipment_data, STT_TRACK

OWN_FIELDS = ('time','team','x','y','altitude','vx','vy','vz','speed','ias','mach','heading_sin',
              'heading_cos','pitch_sin','pitch_cos','roll_sin','roll_cos','aoa','load','engine',
              'missiles','chaff','radar_mode','stt','maw','energy_height','ps','ias_corner',
              'load_max','load_min','thrust_weight','drag_weight','radar_range','radar_for',
              'rwr_range','rmax','rne','edge_x_pos','edge_x_neg','edge_y_pos','edge_y_neg',
              'view_mode','camera_az_sin','camera_az_cos','camera_el_sin','camera_el_cos',
              'free_age','hold_remaining')
ENTITY_FIELDS = ('kind','bearing_sin','bearing_cos','elevation_sin','elevation_cos','range','closure',
                 'airframe_load_limit','radar_for','altitude','vx','vy','vz','speed','energy','flight_time','seeker_time','age',
                 'extrapolated','tracking_me','friend','support','active','aircraft_type')
KINDS = ('radar','rwr','maw','flame','visual','box','map','friend','shot','missile_marker','contrail')
MISSILE_TYPE_ID = -1.   # aircraft_type of a radar track an NCTR radar names a missile (aircraft are in (0, 1])


def pair(values):
    return [0. if v is None else float(v) for v in values]+[0. if v is None else 1. for v in values]


def angle(deg):
    return (None,None) if deg is None else (math.sin(math.radians(deg)),math.cos(math.radians(deg)))


def scale(v, divisor):
    return None if v is None else v/divisor


def turn(x, y, heading_deg):
    """East/north components -> (right, forward) components in the horizontal frame of ``heading_deg`` (compass):
    a vector of bearing b gets bearing b-heading_deg. None or 0 leaves the vector as it is."""
    if not heading_deg or x is None or y is None:
        return x, y
    c, s = math.cos(math.radians(heading_deg)), math.sin(math.radians(heading_deg))
    return x*c-y*s, y*c+x*s


def relative(deg, heading_deg):
    return deg if deg is None or not heading_deg else (deg-heading_deg) % 360.


def edge_ray(x, y, bearing_deg, half):
    """Distance from (x, y) along compass ``bearing_deg`` to the edge of the map square; 0 outside the square."""
    if abs(x) > half or abs(y) > half:
        return 0.
    dx, dy = math.sin(math.radians(bearing_deg)), math.cos(math.radians(bearing_deg))
    t = math.inf
    for p, d in ((x, dx), (y, dy)):
        if d > 1e-9:
            t = min(t, (half-p)/d)
        elif d < -1e-9:
            t = min(t, (-half-p)/d)
    return t


# observation_frame "egocentric": these own slots carry frame-free quantities instead (same width, so a world-frame
# checkpoint can be fine-tuned). Everything else in the own vector is already independent of map axes.
EGO_OWN = dict(team='home_range', x='home_sin', y='home_cos', vx='enemy_home_sin', vy='enemy_home_cos',
               heading_sin='enemy_home_range', heading_cos='edge_min', edge_x_pos='edge_ahead',
               edge_x_neg='edge_right', edge_y_pos='edge_behind', edge_y_neg='edge_left')
OWN_FIELDS_EGO = tuple(EGO_OWN.get(f, f) for f in OWN_FIELDS)



def energy_state(own, model, load_limits):
    h,v=own.altitude_m,own.speed_mps
    energy=h+v*v/(2*9.80665)
    if not (0<=h<=20000. and v>=50.):
        return energy,None,None,None,None,None,None
    rho,sound=atmosphere(h)
    ias=v*math.sqrt(rho/1.225)
    if v/sound>2.35 or not -60.<=own.aoa_deg<=60.:
        return energy,None,None,ias,v/sound,None,None
    thrust,drag,_,alpha=model.forces_at_aoa(h,v,own.aoa_deg,own.engine_percent)
    lift_q=model._aero.lookup(v/sound,20.)[0]
    corner=math.sqrt(2*model.mass*9.80665*load_limits[1]/(rho*lift_q)) if lift_q>0 else None
    weight=model.mass*9.80665
    return energy,(thrust*math.cos(alpha)-drag)*v/weight,v/corner if corner else None,ias,v/sound,thrust/weight,drag/weight


@dataclass(frozen=True)
class Entity:
    key: tuple
    kind: str
    bearing: float | None = None
    elevation: float | None = None
    distance: float | None = None
    closure: float | None = None
    position: tuple | None = None
    velocity: tuple | None = None
    age: float = 0.
    extrapolated: bool = False
    tracking: bool = False
    friend: bool = False
    support: bool | None = None
    active: bool | None = None
    track_id: int | None = None
    mark_id: int | None = None
    aircraft: str | None = None
    warning: str | None = None
    flight_time: float | None = None
    seeker_time: float | None = None
    launch_ok: bool = False
    missile: int | None = None      # truth (radar_sees_missiles): uid of the missile behind a radar entity; never encoded
    memory: bool = False            # entity_memory_s: carried from an earlier observation, not pinned; never encoded
    # radar_display 'bscope' (policy_view.py): the altitude band's half width (encoded in the vz slot, whose vertical
    # speed the B-scope does not show) and an own missile's aim point (east, north offset from the own aircraft,
    # encoded in the vx / vy slots of a 'shot' entity, which has no velocity). None leaves those slots as they were.
    band_half: float | None = None
    aim: tuple | None = None


class EntityList(list):
    """select_entities' result under vertical_mode 'angle': the entity list also carries the vertical_mode of the
    executor that flies this aircraft, so a script's proposal (intent.from_flight_action) is translated for it."""
    vertical_mode = 'altitude'


# Turn-aware extrapolation of an entity no longer observed (D): its horizontal turn rate is estimated from the last
# two distinct measurements of its velocity (at most TURN_WINDOW_S apart, lateral acceleration capped at TURN_MAX_G);
# slower turns than TURN_MIN_DEG_S are carried straight, faster ones along the arc for at most TURN_CARRY_MAX_DEG of
# turn, then straight on. Entities without a velocity estimate keep their last position.
TURN_MIN_DEG_S, TURN_MAX_G, TURN_WINDOW_S, TURN_CARRY_MAX_DEG = .05, 9., 10., 180.


def raw_entities(obs, reach=None, maw=True):
    out=[]
    boxes={b.ref:b for b in obs.boxes}
    for c,uid in zip(obs.radar,obs.radar_missiles or (None,)*len(obs.radar)):
        key=('radar',c.track_id) if c.track_id is not None else ('blip',c.mark_id) if uid is None else ('blip','missile',uid)
        b=boxes.get(c.mark_id)
        flight,seeker=None,None
        if reach is not None and c.position is not None and c.range_m is not None:
            aspect=0.
            if c.velocity is not None:
                d=tuple(a-b for a,b in zip(obs.own.position,c.position))
                denom=math.sqrt(sum(x*x for x in d)*sum(x*x for x in c.velocity))
                if denom>0:
                    aspect=math.degrees(math.acos(max(-1.,min(1.,sum(a*b for a,b in zip(d,c.velocity))/denom))))
            timing=reach.times(obs.own.altitude_m,obs.own.speed_mps,c.position[2]-obs.own.altitude_m,aspect,c.range_m)
            if timing is not None:
                flight,seeker=timing
        valid=c.kind in ('track','stt') and (c.track_id is not None or c.kind=='stt')
        out.append(Entity(key,'radar',c.bearing_deg,c.world_elevation_deg,c.range_m,c.closing_speed_mps,
                          c.position,c.velocity,c.age_s,c.extrapolated,track_id=STT_TRACK if c.kind=='stt' else c.track_id,
                          mark_id=c.mark_id,aircraft=c.target_type if b is None else b.aircraft,flight_time=flight,
                          seeker_time=seeker,missile=uid,
                          launch_ok=valid and abs(c.azimuth_deg)<=launch_limit(obs.own.aircraft,obs.own.missile_id)))
    for c in obs.rwr:
        known=[eq.aircraft for eq in equipment_data().equipment.values() if eq.radar==c.radar_id] if c.radar_id else []
        aircraft=known[0] if len(known)==1 else None
        warning='missile' if c.missile_warning else 'lock' if c.tracking else 'track' if c.kind=='tws' else 'search'
        out.append(Entity(('rwr',c.contact_id),'rwr',c.bearing_deg,c.elevation_deg,c.range_m,
                          age=c.age_s,tracking=c.tracking,warning=warning,aircraft=aircraft))
    # maw False (executor maw_entities False): no MAW entities, so every aircraft gives the actor the same inputs.
    for group in (obs.maw if maw else (),obs.flames,obs.visual,obs.contrails,obs.missile_marks):
        for c in group:
            kind='visual' if c.kind=='aircraft' else c.kind
            out.append(Entity((kind,c.ref),kind,c.bearing_deg,c.elevation_deg,c.range_m,
                              age=obs.time_s-c.time_s,mark_id=c.ref if kind in ('visual','contrail') else None))
    for b in obs.boxes:
        out.append(Entity(('box',b.ref),'box',b.bearing_deg,b.elevation_deg,b.range_m,b.closing_speed_mps,
                          age=obs.time_s-b.time_s,mark_id=b.ref,aircraft=b.aircraft))
    own=obs.own
    for m in obs.marks:
        d=(m.x-own.position[0],m.y-own.position[1],None if m.z is None else m.z-own.position[2])
        bearing=math.degrees(math.atan2(d[0],d[1]))%360.
        distance=math.hypot(d[0],d[1])
        el=None if d[2] is None else math.degrees(math.atan2(d[2],max(1.,distance)))
        out.append(Entity(('map',m.mark_id),'friend' if m.friend else 'map',bearing,el,distance,
                          position=(m.x,m.y,m.z),age=obs.time_s-m.time_s,friend=m.friend,mark_id=m.mark_id))
    for s in obs.shots:
        out.append(Entity(('shot',s.uid),'shot',s.bearing_deg,s.elevation_deg,s.range_m,age=s.age_s,
                          friend=True,support=s.datalink,active=s.active,mark_id=s.target_mark))
    return out


def turn_track(entry, e, now):
    """(measurement time, heading deg, turn rate deg/s) of an observed entity with a velocity estimate, from the memory
    ``entry`` of its last observation: the heading change between the last two distinct measurements (the time of a
    measurement is now - age), capped at TURN_MAX_G lateral acceleration. None without a velocity."""
    v=e.velocity
    speed=0. if v is None or e.position is None else math.hypot(v[0],v[1])
    if speed<1.:
        return None
    t,h=now-e.age,math.degrees(math.atan2(v[0],v[1]))
    last=entry[2] if entry is not None and len(entry)>2 else None
    if last is None or t-last[0]>TURN_WINDOW_S:
        return (t,h,0.)
    if t<=last[0]+1e-6:
        return last    # no new measurement (a track coasting at constant velocity): keep the estimate
    limit=math.degrees(TURN_MAX_G*9.80665/speed)
    return (t,h,max(-limit,min(limit,wrap(h-last[1])/(t-last[0]))))


def carried(entry, now, own):
    """The entity of memory ``entry`` (its last observation, when, turn track) carried to ``now``: along its turn (at most
    TURN_CARRY_MAX_DEG of it, then straight on), or straight at its last velocity; bearing, elevation and distance from
    the own position. It is extrapolated, with no track and no launch. A velocity without a vertical component (the
    B-scope's horizontal vector, policy_view.py) carries the altitude level and keeps the vertical component unknown."""
    v=entry[0].velocity
    if v is not None and v[2] is None:
        e=carried((replace(entry[0],velocity=(v[0],v[1],0.)),*entry[1:]),now,own)
        return replace(e,velocity=(e.velocity[0],e.velocity[1],None))
    previous,when=entry[0],entry[1]
    turn=entry[2] if len(entry)>2 and entry[2] is not None else None
    age=now-when
    pos,vel=previous.position,previous.velocity
    if pos is not None and vel is not None:
        w=0. if turn is None else math.radians(turn[2])
        if abs(w)<math.radians(TURN_MIN_DEG_S) or age<=0.:
            pos=tuple(a+b*age for a,b in zip(pos,vel))
        else:
            arc=min(age,math.radians(TURN_CARRY_MAX_DEG)/abs(w))
            speed,h0=math.hypot(vel[0],vel[1]),math.atan2(vel[0],vel[1])
            h1=h0+w*arc
            x=pos[0]+speed/w*(math.cos(h0)-math.cos(h1))
            y=pos[1]+speed/w*(math.sin(h1)-math.sin(h0))
            vel=(speed*math.sin(h1),speed*math.cos(h1),vel[2])
            pos=(x+vel[0]*(age-arc),y+vel[1]*(age-arc),pos[2]+vel[2]*age)
    bearing,elevation,distance=previous.bearing,previous.elevation,previous.distance
    if pos is not None and all(v is not None for v in pos):
        d=tuple(b-a for a,b in zip(own.position,pos))
        distance=math.sqrt(sum(v*v for v in d))
        bearing=math.degrees(math.atan2(d[0],d[1]))%360.
        elevation=math.degrees(math.atan2(d[2],math.hypot(d[0],d[1])))
    return replace(previous,age=previous.age+age,position=pos,velocity=vel,bearing=bearing,elevation=elevation,
                   distance=distance,extrapolated=True,track_id=None,launch_ok=False)


def entity_priority(e, pinned, support_marks):
    """Sort key of the actor's entities (training spec 2): pinned references and the targets of supported own missiles,
    missile warnings, locks, radar contacts by range, RWR by threat, then the rest by range."""
    r=e.distance if e.distance is not None else 0.
    if e.key in pinned or (e.mark_id in support_marks and e.kind in ('radar','box','map')):
        return (0,0.,e.age,str(e.key))
    if e.kind in ('maw','missile_marker') or e.warning=='missile':
        return (1,0.,e.age,str(e.key))
    if e.tracking:
        return (2,0.,e.age,str(e.key))
    if e.kind=='radar':
        return (3,r,e.age,str(e.key))
    if e.kind=='rwr':
        return (4,{'missile':0,'lock':1,'track':2,'search':3}[e.warning],e.age,str(e.key))
    if e.kind in ('friend','map'):
        return (5,r,e.age,str(e.key))
    return (4 if e.kind=='flame' else 5,r,e.age,str(e.key))


def select_entities(obs, executor, memory, reach=None, limit=64):
    """The actor's entities (at most ``limit``, in the priority order of training spec 2) and the number dropped.
    ``memory`` maps an entity key to (last observation, time, turn track). Executor options: maw_entities False drops
    MAW entities; entity_memory_s keeps any enemy entity seen within that many seconds as an extrapolated memory entity
    (no track, no launch, never a target), only in the slots the observed entities leave free."""
    now=obs.time_s
    all_entities=raw_entities(obs,reach,getattr(executor,'maw_entities',True))
    keys={e.key for e in all_entities}
    pinned={executor.published.maneuver_ref,executor.published.view_object}
    aim=getattr(executor,'aim_heads',None)
    if aim is not None and aim[0] is not None and executor.held(now):
        pinned.add(aim[0])   # a held reference stays through free look (the exit has to keep it)
    # A held reference lost by the sensors is carried from its last observation.
    for key in pinned:
        if key is not None and key not in keys and key in memory:
            all_entities.append(carried(memory[key],now,obs.own))
    span=getattr(executor,'entity_memory_s',None)
    def remembered(key, entry):
        e=entry[0]
        return span is not None and key not in keys and key not in pinned and now-entry[1]<=span+1e-9 \
            and not e.friend and e.kind not in ('friend','shot')
    extra=[replace(carried(entry,now,obs.own),tracking=False,memory=True)
           for key,entry in memory.items() if remembered(key,entry)]
    for e in all_entities:
        if not (e.key in pinned and e.key not in keys):
            memory[e.key]=(e,now,turn_track(memory.get(e.key),e,now))
    # Retain only current / explicitly pinned estimates (and, with entity_memory_s, recent enemies); no world cache.
    for key in list(memory):
        if key not in keys and key not in pinned and not remembered(key,memory[key]):
            del memory[key]
    support_marks={s.target_mark for s in obs.shots if s.datalink and not s.active and s.target_mark is not None}
    def priority(e):
        return entity_priority(e,pinned,support_marks)
    all_entities.sort(key=priority)
    kept,dropped=all_entities[:limit],max(0,len(all_entities)-limit)
    if extra:   # memory entities never displace an observed one and are not counted as dropped
        kept=kept+sorted(extra,key=priority)[:max(0,limit-len(kept))]
    if getattr(executor,'vertical_mode','altitude')!='altitude':
        kept=EntityList(kept)
        kept.vertical_mode=executor.vertical_mode
    return kept,dropped


def select_view_entities(obs, executor, view, show, reach=None, limit=64):
    """select_entities for a policy aircraft with an opt-in policy view (policy_view.PolicyView): ``show(obs, entities)``
    turns raw_entities(obs) into what the policy may know (same keys, degraded values). The priority order, the limit, the
    references carried from memory and entity_memory_s all use only the shown entities and their own memory
    (view.memory); each one has a full-information twin (the same key from the same observation, carried from
    view.full_memory) for the executor and the action masks, so legality and execution are as without the view.
    Returns (shown, full, dropped), parallel lists."""
    now=obs.time_s
    full_obs=raw_entities(obs,reach,getattr(executor,'maw_entities',True))
    shown_obs=show(obs,full_obs)
    if [e.key for e in shown_obs]!=[e.key for e in full_obs]:
        raise ValueError('a policy view must keep the observed entities and their order')
    memories=(view.memory,view.full_memory)
    lists=[list(shown_obs),list(full_obs)]
    keys={e.key for e in full_obs}
    pinned={executor.published.maneuver_ref,executor.published.view_object}
    aim=getattr(executor,'aim_heads',None)
    if aim is not None and aim[0] is not None and executor.held(now):
        pinned.add(aim[0])
    for key in pinned:
        if key is not None and key not in keys and key in memories[0]:
            for lst,memory in zip(lists,memories):
                lst.append(carried(memory[key],now,obs.own))
    span=getattr(executor,'entity_memory_s',None)
    def remembered(key, entry):
        e=entry[0]
        return span is not None and key not in keys and key not in pinned and now-entry[1]<=span+1e-9 \
            and not e.friend and e.kind not in ('friend','shot')
    extra_keys=[key for key,entry in memories[0].items() if remembered(key,entry)]
    extras=[[replace(carried(memory[key],now,obs.own),tracking=False,memory=True) for key in extra_keys]
            for memory in memories]
    for lst,memory in zip(lists,memories):
        for e in lst:
            if not (e.key in pinned and e.key not in keys):
                memory[e.key]=(e,now,turn_track(memory.get(e.key),e,now))
        for key in list(memory):
            if key not in keys and key not in pinned and not remembered(key,memory[key]):
                del memory[key]
    shown=lists[0]
    support_marks={e.mark_id for e in shown if e.kind=='shot' and e.support and not e.active and e.mark_id is not None}
    order=sorted(range(len(shown)),key=lambda i:entity_priority(shown[i],pinned,support_marks))
    dropped=max(0,len(order)-limit)
    kept=[[lst[i] for i in order[:limit]] for lst in lists]
    if extra_keys:   # memory entities never displace an observed one and are not counted as dropped
        free=max(0,limit-len(kept[0]))
        eorder=sorted(range(len(extra_keys)),key=lambda i:entity_priority(extras[0][i],pinned,support_marks))[:free]
        kept=[k+[x[i] for i in eorder] for k,x in zip(kept,extras)]
    if getattr(executor,'vertical_mode','altitude')!='altitude':
        kept=[EntityList(k) for k in kept]
        for k in kept:
            k.vertical_mode=executor.vertical_mode
    return kept[0],kept[1],dropped


def single(k, selected):
    return [i==selected for i in range(k)]


def masks_for(obs, entities, executor, last_launch):
    n=len(entities)
    none=single(n+1,n)
    refs=[e.bearing is not None for e in entities]+[True]
    # A missile track (truth, so also one the radar did not name) is no target unless allow_missile_targets:
    # this masks the weapon and STT as well.
    target=[e.track_id is not None and e.kind=='radar' and (e.missile is None or obs.missile_targets)
            for e in entities]+[True]
    aim_refs=refs
    held=executor.held(obs.time_s)
    # The held values are those of the last mouse-aim publication (aim_heads), also through free look.
    p=executor.published
    ref,old_m,old_v=getattr(executor,'aim_heads',(p.maneuver_ref,p.maneuver,p.vertical))
    keys=[e.key for e in entities]
    old=dict(maneuver_ref=keys.index(ref) if ref in keys else n,maneuver=old_m,vertical=old_v)
    if held:
        aim_refs=single(n+1,old['maneuver_ref'])
    # view_model 'object_only': no look-direction view, no keyboard; look-object keeps the last mouse-aim intent.
    obj=getattr(executor,'view_model','full')=='object_only'
    free=(lambda k,i:single(k,i)) if obj else (lambda k,i:[True]*k)
    air=not obs.grounded   # airfield: no weapon or chaff on the ground
    m=dict(view_mode=[True,any(refs[:-1]),not obj],target=target,speed=[True]*3,
           chaff=[True,obs.own.chaff>0 and air,obs.own.chaff>0 and air],antenna=[True]*5,
           maneuver_ref=[aim_refs,single(n+1,old['maneuver_ref']) if obj else none[:],none[:]],
           view_object=[none[:],refs[:],none[:]],
           look_az=[single(8,0),single(8,0),free(8,0)],
           look_el=[single(3,1),single(3,1),free(3,1)],
           kb_roll=[single(3,1),free(3,1),free(3,1)],kb_pitch=[single(3,1),free(3,1),free(3,1)])
    for h in ('maneuver','vertical'):
        k=CAT_SIZES[h]
        # A hold keeps the held values through free look; taking back mouse flight after it is a fresh proposal.
        row=single(k,old[h]) if held else [True]*k
        look=single(k,old[h]) if obj else single(k,0)
        m[h]=[[row[:],row[:]], [look[:],look[:]],[single(k,0),single(k,0)]]
    fire=[]
    radar=[]
    eq=equipment_data().equipment.get(obs.own.aircraft)
    have_radar=eq is not None and eq.radar is not None
    for idx in range(n+1):
        e=None if idx==n else entities[idx]
        tracked=e is not None and target[idx]
        can_fire=tracked and e.launch_ok and obs.own.missiles>0 and obs.time_s-last_launch>=1.-1e-9 and air
        fire.append([True,bool(can_fire)])
        radar.append([True,have_radar,have_radar and tracked])
    m['weapon'],m['radar_mode']=fire,radar
    return m


@lru_cache(maxsize=None)
def aircraft_description(aircraft):
    from .fm import load_aircraft
    from .fm.catalog import find_aircraft
    from .flight import read_structure
    from .match import load_model
    if aircraft is None:
        pool=load_model()['aircraft_frequency']['weights']
        values=[aircraft_description(a) for a in pool]
        return tuple(sum(v[j] for v in values)/len(values) for j in (0,1))
    fm=load_aircraft(aircraft)
    structure=read_structure(json.loads(find_aircraft(aircraft).path.read_text()))
    load=structure.limits(fm.empty_mass_kg*1.3)[1]/12.
    eq=equipment_data().equipment.get(aircraft)
    radar=equipment_data().radars.get(eq.radar) if eq else None
    return load,0. if radar is None else (radar.field_of_regard_deg or 0.)/180.


def entity_vector(e, heading_deg=None):
    """``heading_deg`` (egocentric frame: the observer's heading): the bearing is given relative to the nose and the
    horizontal velocity as (right, forward) components. None keeps compass bearings and east/north velocities."""
    a,b=angle(relative(e.bearing,heading_deg));c,d=angle(e.elevation)
    pos=e.position or (None,None,None);vel=e.velocity or (None,None,None)
    if heading_deg is not None:
        vel=(*turn(vel[0],vel[1],heading_deg),vel[2])
    # A B-scope velocity has no vertical component (None): the speed is the length of the horizontal vector shown.
    speed=None if e.velocity is None else math.sqrt(sum(v*v for v in e.velocity if v is not None))
    energy=None if pos[2] is None or speed is None else pos[2]+speed*speed/(2*9.80665)
    slots=[scale(v,1000.) for v in vel]
    if e.aim is not None:   # bscope own missile before its seeker is active: the aim point, scaled like a range
        ax,ay=turn(e.aim[0],e.aim[1],heading_deg) if heading_deg is not None else e.aim
        slots=[ax/120000.,ay/120000.,None]
    if e.band_half is not None:   # bscope radar contact: half width of the altitude band, scaled like an altitude
        slots[2]=scale(e.band_half,20000.)
    # An observed type is represented by catalog order; absent type stays invalid.
    from .fm.catalog import aircraft_catalog
    types=[a.id for a in aircraft_catalog()]
    missile=e.aircraft=='missile'
    type_id=None if e.aircraft is None else MISSILE_TYPE_ID if missile else (types.index(e.aircraft)+1)/len(types)
    descriptions=aircraft_description(e.aircraft) if e.kind in ('radar','rwr','visual','box','map','contrail','friend') \
        and not missile else (None,None)
    values=[KINDS.index(e.kind)/10.,a,b,c,d,scale(e.distance,120000.),scale(e.closure,1000.),
            *descriptions,scale(pos[2],20000.),
            *slots,scale(speed,1000.),scale(energy,40000.),
            scale(e.flight_time,300.),scale(e.seeker_time,300.),e.age/60.,float(e.extrapolated),
            float(e.tracking),float(e.friend),e.support,e.active,type_id]
    return pair(values)


def own_vector(obs, plane, executor, reach=None, judge=None, ego=None):
    """``ego`` = (home_xy, enemy_home_xy) selects the egocentric frame (see EGO_OWN): no map coordinates, compass
    heading or team; instead range and nose-relative bearing of the own and the enemy home, distance to the nearest
    map edge (negative outside) and the distance to the edge straight ahead, right, behind and left."""
    o=obs.own
    perf=energy_state(o,plane.flight.model,plane.flight.load_limits)
    en,ps,corner,ias,mach,thrust,drag=perf
    ha,hb=angle(o.heading_deg);pa,pb=angle(o.pitch_deg);ra,rb=angle(o.roll_deg)
    cam=plane.camera
    ca,cb=angle(None if cam.bearing_deg is None else wrap(cam.bearing_deg-o.heading_deg))
    ce,cf=angle(cam.elevation_deg)
    radar=plane.radar.radar if plane.radar else None
    rwr=plane.rwr.rwr if plane.rwr else None
    waves=() if radar is None else radar.search_waveforms
    rmax=None if reach is None else reach.rmax(o.altitude_m,o.speed_mps,0.,0.)
    if rmax is None and judge is not None:
        rmax=judge.rmax(o.altitude_m,o.speed_mps,0.)
    x,y,_=o.position;half=obs.map_half_m
    values=[obs.time_s/900.,float(o.team),x/64000.,y/64000.,o.altitude_m/20000.,
            *[v/1000. for v in o.velocity],o.speed_mps/1000.,scale(ias,1000.),mach,
            ha,hb,pa,pb,ra,rb,o.aoa_deg/60.,o.load/12.,o.engine_percent/110.,
            o.missiles/16.,o.chaff/1000.,{'off':0.,'search':1/3,'tws':2/3,'stt':1.}[o.radar_mode],
            None if o.stt_state is None else {'acquiring':0.,'tracking':1.,'coasting':.5}[o.stt_state],
            float(o.has_maw and getattr(executor,'maw_entities',True)),en/40000.,scale(ps,300.),corner,
            plane.flight.load_limits[1]/12.,plane.flight.load_limits[0]/12.,thrust,drag,
            None if not waves else max(w.range_m or 0. for w in waves)/200000.,
            None if radar is None else scale(radar.field_of_regard_deg,180.),
            None if rwr is None else scale(rwr.range_m,200000.),scale(rmax,120000.),
            getattr(o,'fuel_fraction',None),   # the 'rne' slot (never computed): fuel share of the initial load with fuel on
            (half-x)/128000.,(half+x)/128000.,(half-y)/128000.,(half+y)/128000.,
            executor.executed.view_mode/2.,ca,cb,ce,cf,
            0. if executor.free_since is None else (obs.time_s-executor.free_since)/30.,
            max(0.,executor.hold_until-obs.time_s)/2.]
    if ego is not None:
        hd=o.heading_deg
        for (hx,hy),(r,sn,cs) in zip(ego,(('home_range','home_sin','home_cos'),
                                          ('enemy_home_range','enemy_home_sin','enemy_home_cos'))):
            dx,dy=hx-x,hy-y
            values[OWN_FIELDS_EGO.index(r)]=math.hypot(dx,dy)/128000.
            values[OWN_FIELDS_EGO.index(sn)],values[OWN_FIELDS_EGO.index(cs)]=angle(math.degrees(math.atan2(dx,dy))-hd)
        values[OWN_FIELDS_EGO.index('edge_min')]=min(half-abs(x),half-abs(y))/128000.
        for name,offset in (('edge_ahead',0.),('edge_right',90.),('edge_behind',180.),('edge_left',270.)):
            values[OWN_FIELDS_EGO.index(name)]=min(edge_ray(x,y,hd+offset,half),2.*half)/128000.
    return pair(values)


def prev_vector(executor, entities, now):
    indices=executor.published.indices(entities)
    values=[indices[h]/max(1,len(entities) if h in POINTERS else CAT_SIZES[h]-1) for h in HEAD_NAMES]
    ages=[(now-executor.head_since[h])/60. for h in HEAD_NAMES]
    return values+ages+[1.]*15+[(now-executor.published_at)/30.,max(0.,executor.hold_until-now)/2.,
                               executor.published.view_mode/2.]


def truth_vectors(eng, plane, ego=False):
    """Privileged critic tokens; never called by sorting or mask construction. ``ego``: positions relative to the
    observing aircraft and, like velocities and headings, in its heading frame (altitude stays absolute)."""
    if ego:
        ox,oy,_=plane.own.position if plane.grounded else plane.flight.state.position
        hd=plane.own.heading_deg if plane.grounded else plane.flight.attitude()[0]
        def place(v):
            return (*turn(v[0]-ox,v[1]-oy,hd),v[2])
        def move(v):
            return (*turn(v[0],v[1],hd),v[2])
    else:
        hd=None
        place=move=lambda v:v
    tokens=[]
    for p in eng.planes:
        if p.grounded:   # airfield: parked at zero altitude and speed (the frozen flight state is the touchdown)
            pos,vel,(h,pi,r),engine,load=p.own.position,(0.,0.,0.),(p.own.heading_deg,0.,0.),0.,1.
        else:
            s=p.flight.state
            pos,vel,(h,pi,r),engine,load=s.position,s.velocity,p.flight.attitude(),s.engine_percent,p.flight.load
        vals=[0.,float(p.team==plane.team),p.ident/32.,*[x/64000. for x in place(pos)],
              *[v/1000. for v in move(vel)],*angle(relative(h,hd)),pi/90.,r/180.,float(p.alive),
              p.missiles/16.,engine/110.,load/12.,None,None,None]
        tokens.append(pair(vals))
    # Preserve aircraft first, then closest missiles if the truth cap is reached.
    for m in sorted(eng.missiles,key=lambda m:(math.dist(m.pos_enu,plane.own.position),m.uid)):
        vals=[1.,float(m.shooter.team==plane.team),m.uid/1024.,*[x/64000. for x in place(m.pos_enu)],
              *[v/1000. for v in move(m.vel_enu)],None,None,None,None,float(not m.done),
              m.target.ident/32.,m.time_s/300.,float(m.time_s<=m.info.burn_s),float(m.seeker_on),
              float(m.datalink),m.shooter.ident/32.]
        tokens.append(pair(vals))
    return tokens[:128]
