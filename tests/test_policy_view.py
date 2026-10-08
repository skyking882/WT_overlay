"""Opt-in policy views (wt_overlay/policy_view.py): B-scope radar reading, no launch-zone times, map spotting, wrecks.

Each option changes only what the policy aircraft are given; absent, everything is bit-identical."""
from __future__ import annotations

import contextlib
from dataclasses import replace
import hashlib
import io
import json
import math
import random
from types import SimpleNamespace
import unittest

from rl import check_env, spec, wire
from wt_overlay.engagement import RadarCommand
from wt_overlay.intent import Intent, legalize, selected_mask, wrap
from wt_overlay.policy_view import PolicyView, display_range, view_settings
from wt_overlay.reach import ReachTable
from wt_overlay.rl_env import MatchEnv
from wt_overlay.rl_observation import ENTITY_FIELDS, OWN_FIELDS, entity_vector, own_vector

ZERO=dict(reject_p=0.,error_deg=0.,delay={p:dict(median_range=(.001,.001),sigma=0.,bounds=(0.,0.))
                                        for p in ('follow','autonomous')})
GE=dict(aircraft='f_15c_golden_eagle',archetype='middle',skill='top',altitude_m=8000.,mach=1.)
SM2=dict(aircraft='su_30sm2',archetype='middle',skill='top',altitude_m=8000.,mach=1.)
TEAMS=[[GE],[SM2]]
TEAMS2=[[GE,dict(GE,archetype='left')],[SM2,dict(SM2,archetype='left')]]
LOW=[[GE],[dict(SM2,altitude_m=5000.)]]                 # the target 3 km below: its altitude is not the band centre
N=len(ENTITY_FIELDS)
OFF=dict(radar_display=None,launch_zone_info=True,map_spotting_m=None,map_hold_s=None,wreck_s=None)


def make(seed=5, **kw):
    config=dict(teams=TEAMS,range_km=40.,controlled='all',execution=ZERO,time_limit_s=120.)
    config.update(kw)
    e=MatchEnv(config,seed)
    e.reset()
    return e


def slot(row, name):
    """(value, validity) of entity field ``name`` in an encoded row."""
    k=ENTITY_FIELDS.index(name)
    return row[k],row[N+k]


def still(e):
    """Legal actions that reference no entity (the same in every view): flying on, no target."""
    return {aid:legalize(Intent(),e._entities[aid],o.masks) for aid,o in e._observations.items()}


def contact_of(e, aid, truth):
    """(index in raw.radar, RadarContact) of the radar contact on ``truth`` of aircraft ``aid``, or (None, None)."""
    p=e.engagement.planes[aid]
    if p.picture is None:
        return None,None
    for i,t in enumerate(p.picture.truth_ids):
        if t==truth:
            return i,p.picture.contacts[i]
    return None,None


def fly_until_tracked(e, aid=0, truth=1, steps=60, actions=still):
    for _ in range(steps):
        _,c=contact_of(e,aid,truth)
        if c is not None and c.track_id is not None and c.kind=='track':
            return c
        e.step(actions(e))
    raise AssertionError('no TWS track')


def key_of(c):
    """raw_entities' key of an aircraft radar contact."""
    return ('radar',c.track_id) if c.track_id is not None else ('blip',c.mark_id)


def shown_of(e, aid, c):
    """(shown, full) entities of radar contact ``c``."""
    keys=[x.key for x in e._shown[aid]]
    i=keys.index(key_of(c))
    return e._shown[aid][i],e._entities[aid][i]


def pairs(e, aid=0, kind='radar'):
    return [(s,f) for s,f in zip(e._shown[aid],e._entities[aid]) if s.kind==kind]


def stable(result):
    obs,r,d,i=result
    return json.dumps(({a:wire.pack_obs(o,True) for a,o in obs.items()},r,d,i),sort_keys=True,default=repr)


def dense_table():
    """A reach table that answers everywhere a 1v1 can be (straight lines, D)."""
    axes=dict(altitude_m=[0.,20000.],speed_mps=[50.,800.],delta_altitude_m=[-20000.,20000.],aspect_deg=[0.,180.])
    cells=[]
    for a in axes['altitude_m']:
        for v in axes['speed_mps']:
            for d in axes['delta_altitude_m']:
                for asp in axes['aspect_deg']:
                    cells.append(dict(point=[a,v,d,asp],rmax_m=80000.,censored=False,samples=[
                        dict(range_m=0.,flight_s=0.,seeker_on_s=0.),dict(range_m=200000.,flight_s=200.+asp/9.,
                                                                          seeker_on_s=150.+d/1000.)]))
    return ReachTable(dict(version=1,axes=axes,cells=cells))


class DefaultTests(unittest.TestCase):
    def test_absent_and_off_values_are_bit_identical(self):
        for config,seed,mode in ((dict(),5,'scripted'),
                                 (dict(radar_sees_missiles=True,observation_frame='egocentric',
                                       execution=dict(ZERO,entity_memory_s=10.)),7,'random')):
            digests=[]
            for extra in ({},OFF):
                e=MatchEnv(dict(dict(teams=TEAMS,range_km=40.,controlled='all',execution=ZERO,time_limit_s=30.),
                                **config,**extra),seed)
                self.assertIsNone(e.view)
                rng=random.Random(3);h=hashlib.sha256()
                for _ in range(2):   # two episodes
                    obs=e.reset()
                    self.assertFalse(hasattr(e,'views') or hasattr(e,'_shown'))
                    h.update(json.dumps({a:wire.pack_obs(o,True) for a,o in obs.items()},default=repr).encode())
                    while not e.over and e._observations:
                        if mode=='scripted':
                            acts=e.scripted_actions()
                        else:
                            acts={}
                            for aid,o in e._observations.items():
                                chosen={}
                                for head in spec.SAMPLE_ORDER:
                                    row=selected_mask(head,len(o.entities),o.masks,chosen)
                                    chosen[head]=rng.choice([i for i,b in enumerate(row) if b])
                                acts[aid]=chosen
                        h.update(stable(e.step(acts)).encode())
                    h.update(json.dumps(e.engagement.log,default=repr).encode())
                digests.append(h.hexdigest())
            self.assertEqual(digests[0],digests[1])

    def test_bad_values_fail(self):
        for bad in (dict(radar_display='ascope'),dict(radar_display=dict(mode='bscope',az_res=1.)),
                    dict(radar_display=dict(mode='bscope',dropout_p=1.5)),
                    dict(radar_display=dict(mode='bscope',range_res_frac=-1.)),
                    dict(radar_display=dict(mode='bscope',band_half_width=1)),dict(radar_display=dict(az_res_deg=1.)),
                    dict(launch_zone_info=0),dict(map_spotting_m=0),dict(map_spotting_m='8km'),dict(map_hold_s=-1.),
                    dict(wreck_s=0),dict(wreck_s=True)):
            with self.assertRaises(ValueError,msg=str(bad)):
                MatchEnv(dict(team_size=1,**bad),1)
        s=view_settings(dict(radar_display=dict(mode='bscope',az_res_deg=1.,display_range_m=80000)))
        self.assertEqual((s['radar_display']['az_res_deg'],s['radar_display']['display_range_m'],
                          s['radar_display']['range_res_frac']),(1.,80000.,.005))

    def test_contract_checks_pass_with_every_option(self):
        config=dict(team_size=2,time_limit_s=40.,self_play_prob=.5,radar_sees_missiles=True,
                    observation_frame='egocentric',radar_display=dict(mode='bscope',range_noise_frac=.002,az_noise_deg=.3),
                    launch_zone_info=False,map_spotting_m=8000.,wreck_s=20.)
        out=io.StringIO()
        with contextlib.redirect_stdout(out):
            code=check_env.main(['--env','wt_overlay.rl_env:MatchEnv','--config',json.dumps(config),
                                 '--episodes','3','--max-steps','90'])
        report=json.loads(out.getvalue())
        self.assertEqual((code,report['errors']),(0,[]))


class BScopeTests(unittest.TestCase):
    """radar_display 'bscope': what a perfect reader of the B-scope gets."""

    def setUp(self):
        self.e=make(radar_display='bscope',teams=LOW)
        fly_until_tracked(self.e)

    def check_reading(self, e, aid=0):
        own=e._raw[aid].own
        radar=e.engagement.planes[aid].radar.radar
        res=.005*display_range(radar)
        checked=0
        for s,f in pairs(e,aid):
            az=wrap(s.bearing-own.heading_deg)
            self.assertAlmostEqual(az/.5,round(az/.5),places=6)
            self.assertLessEqual(abs(wrap(s.bearing-f.bearing)),.25+1e-6)
            self.assertAlmostEqual(s.distance/res,round(s.distance/res),places=6)
            self.assertLessEqual(abs(s.distance-f.distance),res/2.+1e-6)
            self.assertIsNone(s.closure);self.assertIsNotNone(f.closure)
            checked+=1
        return checked

    def test_range_and_bearing_are_quantised_and_closure_is_unknown(self):
        e=self.e
        self.assertEqual(display_range(e.engagement.planes[0].radar.radar),150000.)
        self.assertGreater(self.check_reading(e),0)
        rows=e._observations[0].entities
        for i,x in enumerate(e._shown[0]):
            self.assertEqual(rows[i],entity_vector(x))
            if x.kind=='radar':
                self.assertEqual(slot(rows[i],'closure'),(0.,0.))
        for _ in range(8):   # the reading follows the contact every decision
            e.step(still(e))
            self.check_reading(e)

    def test_altitude_is_the_band_centre_in_both_frames(self):
        for frame in ('world','egocentric'):
            e=self.e if frame=='world' else make(radar_display='bscope',observation_frame=frame,teams=LOW)
            if frame=='egocentric':
                fly_until_tracked(e)
            p=e.engagement.planes[0]
            e.views[0].state.clear()   # a refresh at this decision: the band comes from this range and altitude
            e.observe()
            own=e._raw[0].own
            i,c=contact_of(e,0,1)
            s,_=shown_of(e,0,c)
            st=e.views[0].state[s.key]
            lo,hi=st['band']
            self.assertAlmostEqual(s.position[2],(lo+hi)/2.)
            self.assertAlmostEqual(s.band_half,(hi-lo)/2.)
            self.assertNotAlmostEqual(s.position[2],c.position[2],places=0)
            truth=e.engagement.planes[1].state_at(c.updated_s)[0][2]
            self.assertTrue(lo-200.<=truth<=hi+200.)
            low,high=p.radar.elevation_coverage()
            self.assertEqual((low,high),(-9.25,9.25))      # 6 bars of 2.5 deg and the beam, level
            self.assertAlmostEqual(lo,max(0.,own.position[2]+s.distance*math.sin(math.radians(low))))
            self.assertAlmostEqual(hi,own.position[2]+s.distance*math.sin(math.radians(high)))
            dz=s.position[2]-own.position[2]
            self.assertAlmostEqual(s.elevation,math.degrees(math.atan2(dz,math.sqrt(s.distance**2-dz*dz))))
            heading=own.heading_deg if frame=='egocentric' else None
            row=e._observations[0].entities[e._shown[0].index(s)]
            self.assertEqual(row,entity_vector(s,heading))
            self.assertAlmostEqual(slot(row,'altitude')[0],s.position[2]/20000.)
            self.assertAlmostEqual(slot(row,'elevation_sin')[0],math.sin(math.radians(s.elevation)))
            self.assertEqual(slot(row,'vz'),(s.band_half/20000.,1.))
            if frame=='egocentric':
                rel=(s.bearing-own.heading_deg)%360.
                self.assertAlmostEqual(slot(row,'bearing_sin')[0],math.sin(math.radians(rel)))
                self.assertAlmostEqual(rel/.5,round(rel/.5),places=6)

    def test_stt_is_degraded_too(self):
        e=self.e;eng=e.engagement;p=eng.planes[0]
        _,c=contact_of(e,0,1)
        eng._set_radar(p,RadarCommand('stt',0,0.,0.,c.track_id))
        e.observe()
        self.assertEqual(p.radar.mode,'stt')
        i,c=contact_of(e,0,1)
        self.assertEqual(c.kind,'stt')
        s,f=shown_of(e,0,c)
        self.assertEqual(s.key,f.key)
        self.assertIsNone(s.closure)
        self.assertNotAlmostEqual(s.position[2],f.position[2],places=0)
        self.assertEqual(s.velocity[2],None)
        self.check_reading(e)

    def test_velocity_in_every_mode_exact_away_from_the_notch(self):
        e=self.e;eng=e.engagement;p=eng.planes[0]
        def check():
            n=0
            for i,c in enumerate(e._raw[0].radar):
                truth=p.picture.truth_ids[i]
                pos,vel=eng.planes[truth].state_at(c.updated_s)
                own=p.state_at(c.updated_s)[0]
                los=[a-b for a,b in zip(pos,own)]
                radial=abs(sum(a*b for a,b in zip(vel,los)))/math.sqrt(sum(x*x for x in los))
                self.assertGreater(radial,60.)
                self.assertEqual(shown_of(e,0,c)[0].velocity,(vel[0],vel[1],None))
                n+=1
            return n
        self.assertEqual(check(),1)                        # TWS
        # search: the executors only command TWS / STT, so the engagement is stepped on its own here
        eng._set_radar(p,RadarCommand('search',0,0.,0.))
        for _ in range(20):
            for _ in range(24):
                eng.step()
            e.observe()
            if e._raw[0].radar:
                break
        self.assertEqual(e._raw[0].radar[0].kind,'blip')
        self.assertIsNone(shown_of(e,0,e._raw[0].radar[0])[1].velocity)   # a blip has no velocity estimate
        self.assertEqual(check(),1)

    def test_notch_jitter_dropout_and_determinism(self):
        settings=view_settings(dict(radar_display='bscope'))
        def stub(target_velocity):
            own=SimpleNamespace(ident=0,alive=True,state_at=lambda t:((0.,0.,8000.),(0.,250.,0.)))
            tgt=SimpleNamespace(ident=1,alive=True,state_at=lambda t:((0.,50000.,8000.),target_velocity))
            return own,SimpleNamespace(planes=[own,tgt],missiles=[],wreck_s=None)
        c=SimpleNamespace(updated_s=1.,velocity=None,position=None)
        def draws(velocity, seed=1, n=3000):
            view=PolicyView(settings,seed,0)
            own,eng=stub(velocity)
            return [view._vector(c,1,own,eng) for _ in range(n)]
        beam=draws((300.,0.,0.))                           # r = 0: full jitter
        missing=sum(v is None for v in beam)/len(beam)
        self.assertAlmostEqual(missing,.3,delta=.03)
        errors=[math.degrees(math.atan2(v[0]*0.-v[1]*1.,v[0]*1.+v[1]*0.)) for v in beam if v is not None]
        sd=math.sqrt(sum(x*x for x in errors)/len(errors))
        self.assertAlmostEqual(sd,60.,delta=6.)
        speeds=[math.hypot(*v)/300. for v in beam if v is not None]
        self.assertAlmostEqual(math.sqrt(sum((x-1.)**2 for x in speeds)/len(speeds)),.4,delta=.05)
        half=draws((300.,30.,0.))                          # r = 30: half the band
        self.assertAlmostEqual(sum(v is None for v in half)/len(half),.15,delta=.03)
        self.assertEqual(set(draws((300.,80.,0.),n=50)),{(300.,80.)})   # outside the band: exact
        self.assertEqual(draws((300.,0.,0.),n=200),beam[:200])           # deterministic per seed
        self.assertNotEqual(draws((300.,0.,0.),seed=2,n=200),beam[:200])

    def test_vector_is_held_between_refreshes_and_snapshot_restores_the_draws(self):
        # a mechanically scanned radar (Captor-M, 1.7 s frames): several decisions see the same refresh; electronic
        # radars refresh their tracks every fast-scan period, so every decision is a refresh for them
        cfg=dict(radar_display=dict(mode='bscope',notch_band_mps=2000.,dropout_p=0.,range_noise_frac=.002,
                                    az_noise_deg=.3),teams=[[dict(GE,aircraft='ef_2000_block_10')],LOW[1]])
        e=make(**cfg)
        fly_until_tracked(e)
        last,held,changed=None,0,0
        for _ in range(16):
            e.step(still(e))
            _,c=contact_of(e,0,1)
            s,_=shown_of(e,0,c)
            truth=e.engagement.planes[1].state_at(c.updated_s)[1]
            self.assertNotAlmostEqual(s.velocity[0],truth[0],places=3)   # always inside this wide band: jittered
            if last is not None and last[0]==c.updated_s:
                self.assertEqual(s.velocity,last[1]);held+=1
            elif last is not None:
                self.assertNotEqual(s.velocity,last[1]);changed+=1
            last=(c.updated_s,s.velocity)
        self.assertGreater(held,0);self.assertGreater(changed,0)
        # the same seed gives the same draws; snapshot / restore brings them back
        a,b=make(**cfg),make(**cfg)
        for _ in range(12):
            self.assertEqual(stable(a.step(a.scripted_actions())),stable(b.step(b.scripted_actions())))
        state=a.snapshot()
        acts=[]
        first=[]
        for _ in range(10):
            acts.append(a.scripted_actions());first.append(stable(a.step(acts[-1])))
        a.restore(state)
        self.assertEqual([stable(a.step(x)) for x in acts],first)

    def test_own_missile_on_the_b_scope(self):
        for frame in ('world','egocentric'):
            e=make(radar_display='bscope',observation_frame=frame)
            fly_until_tracked(e)
            eng=e.engagement
            m=eng.fire(eng.planes[0],eng.planes[1])
            for _ in range(6):
                e.step(still(e))
            ((s,f),)=pairs(e,0,'shot')
            own=e._raw[0].own
            self.assertFalse(m.seeker_on)
            self.assertEqual((s.support,s.active,s.mark_id),(f.support,f.active,f.mark_id))
            self.assertEqual(s.mark_id,eng.mark_ids[1])
            self.assertIsNone(s.elevation);self.assertIsNone(s.position);self.assertIsNone(s.velocity)
            d=[a-b for a,b in zip(m.pos_enu,own.position)]
            res=.005*150000.
            self.assertLessEqual(abs(s.distance-math.sqrt(sum(x*x for x in d))),res/2.+1e-6)
            az=wrap(s.bearing-own.heading_deg)
            self.assertAlmostEqual(az/.5,round(az/.5),places=6)
            # the aim point: the missile's own target estimate (datalink: the target), quantised, no altitude
            tp=eng.planes[1].flight.state.position
            ax,ay=s.aim
            self.assertLessEqual(abs(math.hypot(ax,ay)-math.dist(tp,own.position)),res)
            row=e._observations[0].entities[e._shown[0].index(s)]
            heading=own.heading_deg if frame=='egocentric' else None
            self.assertEqual(row,entity_vector(s,heading))
            self.assertEqual(slot(row,'vz'),(0.,0.));self.assertEqual(slot(row,'speed'),(0.,0.))
            self.assertEqual(slot(row,'altitude'),(0.,0.));self.assertEqual(slot(row,'elevation_sin'),(0.,0.))
            if frame=='world':
                self.assertEqual(slot(row,'vx'),(ax/120000.,1.))
            else:
                h=math.radians(own.heading_deg)
                self.assertAlmostEqual(slot(row,'vy')[0],(ay*math.cos(h)+ax*math.sin(h))/120000.)
        # active seeker: no circle
        m.seeker_on=True
        e.observe()
        self.assertIsNone(pairs(e,0,'shot')[0][0].aim)

    def test_scripted_episode_labels_masks_and_boxes_are_unchanged(self):
        for kw in (dict(),dict(policy_ids=[0])):
            on=make(radar_display=dict(mode='bscope',range_noise_frac=.003,az_noise_deg=.5),**kw)
            off=make(**kw)
            boxes=0
            for _ in range(70):
                if on.over:
                    break
                la,lb=on.scripted_actions(),off.scripted_actions()
                self.assertEqual(set(la),set(lb))
                for aid in la:
                    ea,eb=on._entities[aid],off._entities[aid]
                    self.assertEqual(Intent.from_indices(la[aid],ea),Intent.from_indices(lb[aid],eb))
                    if aid in on._observations:
                        ma,mb=on._observations[aid].masks,off._observations[aid].masks
                        ka,kb=[x.key for x in ea],[x.key for x in eb]
                        self.assertEqual(sorted(ka,key=str),sorted(kb,key=str))
                        for key in ka:
                            i,j=ka.index(key),kb.index(key)
                            for head in ('target','weapon','radar_mode'):
                                self.assertEqual(ma[head][i],mb[head][j],head)
                            self.assertEqual(ma['view_object'][1][i],mb['view_object'][1][j])
                            self.assertEqual(ea[i].launch_ok,eb[j].launch_ok)
                            self.assertEqual(ea[i],eb[j])           # the executor's twins are the plain entities
                        sb=[x for x in on._shown[aid] if x.kind=='box']
                        self.assertEqual(sb,[x for x in eb if x.kind=='box'])
                        boxes+=len(sb)
                on.step(la);off.step(lb)
            self.assertGreater(boxes,0)
            self.assertEqual(on.engagement.log,off.engagement.log)


class LaunchZoneTests(unittest.TestCase):
    def test_flight_and_seeker_times_are_unknown(self):
        on,off=make(launch_zone_info=False),make()
        for e in (on,off):
            fly_until_tracked(e)
            e.reach[0]=dense_table()
            e.observe()
        (s,f),=pairs(on)
        (x,)=[x for x in off._entities[0] if x.kind=='radar']
        self.assertIsNotNone(x.flight_time);self.assertIsNotNone(x.seeker_time)
        self.assertEqual((s.flight_time,s.seeker_time),(None,None))
        self.assertEqual(replace(s,flight_time=x.flight_time,seeker_time=x.seeker_time),x)
        row=on._observations[0].entities[on._shown[0].index(s)]
        self.assertEqual((slot(row,'flight_time'),slot(row,'seeker_time')),((0.,0.),(0.,0.)))
        self.assertEqual(on._observations[0].masks,off._observations[0].masks)
        self.assertEqual(on._observations[0].own,off._observations[0].own)   # rmax stays

    def test_bscope_times_use_only_the_shown_values(self):
        e=make(radar_display='bscope')
        fly_until_tracked(e)
        table=e.reach[0]=dense_table()
        e.observe()
        (s,f),=pairs(e)
        own=e._raw[0].own
        dx,dy=own.position[0]-s.position[0],own.position[1]-s.position[1]
        aspect=math.degrees(math.acos((dx*s.velocity[0]+dy*s.velocity[1])/(math.hypot(dx,dy)*math.hypot(*s.velocity[:2]))))
        self.assertEqual((s.flight_time,s.seeker_time),
                         table.times(own.altitude_m,own.speed_mps,s.position[2]-own.altitude_m,aspect,s.distance))
        self.assertNotEqual(s.seeker_time,f.seeker_time)


class SpottingTests(unittest.TestCase):
    """map_spotting_m: the policy's enemy marks come only from spotting (near the plane or a living teammate)."""

    def enemy_marks(self, e, aid):
        return [m for m in e._raw[aid].marks if not m.friend]

    def test_radar_tracks_no_longer_mark_enemies(self):
        on,off=make(map_spotting_m=8000.),make()
        for _ in range(30):
            on.step(still(on));off.step(still(off))
        self.assertTrue(contact_of(on,0,1)[1] is not None)
        self.assertTrue(self.enemy_marks(off,0))                   # the engagement marks radar tracks
        self.assertFalse(self.enemy_marks(on,0))                   # 30+ km away: not spotted
        self.assertFalse([x for x in on._shown[0] if x.kind=='map'])
        self.assertEqual(on.engagement.marks,off.engagement.marks)  # the shared table is the scripts', unchanged

    def test_spotting_range_and_hold(self):
        radius=34000.
        e=make(map_spotting_m=radius,map_hold_s=3.,policy_ids=[0],teams=TEAMS2)
        eng=e.engagement
        seen=0
        for _ in range(60):
            e.step(still(e))
            if 0 not in e._observations:
                break
            t=eng.time
            spotters=[p.own.position for p in eng.live if p.team==0 and not p.grounded]
            marks={m.mark_id:m for m in self.enemy_marks(e,0)}
            for q in eng.live:
                if q.team==0:
                    continue
                near=any(math.dist(q.own.position,s)<=radius for s in spotters)
                m=marks.get(eng.mark_ids[q.ident])
                if near:
                    self.assertEqual((m.x,m.y,m.z,m.time_s),(q.own.position[0],q.own.position[1],None,t))
                    seen+=1
                elif m is not None:
                    self.assertLessEqual(t-m.time_s,3.+1e-9)
        self.assertGreater(seen,0)

    def test_scripts_marks_and_actions_are_unchanged(self):
        on=make(map_spotting_m=8000.,teams=TEAMS2,policy_ids=[0])
        off=make(teams=TEAMS2,policy_ids=[0])
        for _ in range(40):
            if on.over:
                break
            on.scripted_actions();off.scripted_actions()
            for aid in (1,2,3):
                if aid in on._raw:
                    self.assertEqual(on._raw[aid],off._raw[aid])
                    self.assertEqual(on._scripts_cache[aid],off._scripts_cache[aid])
            on.step(still(on));off.step(still(off))
        self.assertEqual(on.engagement.log,off.engagement.log)

    def test_snapshot_restores_the_spotting_table(self):
        e=make(map_spotting_m=40000.,teams=TEAMS2,policy_ids=[0])
        for _ in range(4):
            e.step(still(e))
        state=e.snapshot()
        self.assertTrue(e.policy_mark_tables[0])
        first=[stable(e.step(still(e))) for _ in range(4)]
        e.restore(state)
        self.assertEqual([stable(e.step(still(e))) for _ in range(4)],first)


class WreckTests(unittest.TestCase):
    """wreck_s: a shot-down aircraft stays a wreck for the policy aircraft's radars and eyes."""

    def kill(self, e, victim=2, killer=0):
        eng=e.engagement
        eng._kill(eng.planes[victim],eng.planes[killer],'missile',None)
        return eng.time

    # enemies without missiles, so the policy aircraft lives through the wreck time
    TEAMS=[TEAMS2[0],[dict(SM2,missiles=0),dict(SM2,archetype='left',missiles=0)]]

    def make(self, **kw):
        e=make(teams=self.TEAMS,policy_ids=[0],**kw)
        fly_until_tracked(e,0,2)
        return e

    def test_wreck_is_tracked_and_seen_for_wreck_s_then_gone(self):
        on,off=self.make(wreck_s=20.),self.make()
        eng=on.engagement
        _,c=contact_of(on,0,2)
        track,mark=c.track_id,eng.mark_ids[2]
        t0=self.kill(on);self.kill(off)
        refreshed=0
        while eng.time<t0+25.:
            on.step(still(on));off.step(still(off))
            _,c=contact_of(on,0,2)
            _,d=contact_of(off,0,2)
            boxed=[b for b in on._raw[0].boxes if b.ref==mark]
            if eng.time<=t0+20.:
                self.assertEqual(c.track_id,track)
                refreshed+=c.updated_s>t0
                self.assertEqual(len(boxed),1)
                i=[x.key for x in on._entities[0]].index(('radar',track))
                self.assertTrue(on._observations[0].masks['target'][i])        # a legal target
            else:
                self.assertFalse(boxed)
                self.assertTrue(c is None or c.updated_s<=t0+20.+1e-9)
            self.assertFalse([b for b in off._raw[0].boxes if b.ref==mark])
            self.assertTrue(d is None or d.updated_s<=t0)
            _,k=contact_of(on,1,2)   # the scripted teammate's radar: no wreck (a dead track coasts, as before)
            self.assertTrue(k is None or k.updated_s<=t0)
            self.assertFalse([b for b in on._raw[1].boxes if b.ref==mark])
        self.assertGreater(refreshed,40)
        self.assertIsNone(eng.wreck_state(2))

    def test_firing_at_a_wreck_is_wasted(self):
        e=self.make(wreck_s=20.)
        eng=e.engagement;p=eng.planes[0]
        _,c=contact_of(e,0,2)
        self.kill(e)
        e.step(still(e))
        left=p.missiles;p.last_launch=-1e9
        m=eng.launch(p,c.track_id)
        self.assertIsNotNone(m);self.assertTrue(m.wreck_shot);self.assertEqual(p.missiles,left-1)
        for _ in range(24):
            e.step(still(e))
        self.assertIs(m.target,eng.planes[2])                       # never retargets
        self.assertFalse(any(x['kind']=='retarget' and x['uid']==m.uid for x in eng.log))
        ((s,f),)=pairs(e,0,'shot')
        self.assertEqual(s.mark_id,eng.mark_ids[2])
        self.assertFalse(m.datalink)
        self.assertTrue(s.support)                                   # the policy sees it supported while tracked
        # without the option a dead contact is no launch target
        off=self.make();_,c=contact_of(off,0,2);self.kill(off);off.step(still(off))
        off.engagement.planes[0].last_launch=-1e9
        self.assertIsNone(off.engagement.launch(off.engagement.planes[0],c.track_id))

    def test_no_exact_death_signal(self):
        e=self.make(wreck_s=20.,radar_display='bscope')
        eng=e.engagement;p=eng.planes[0]
        m=eng.fire(p,eng.planes[2])                 # an own missile at the victim, guided by datalink
        e.step(still(e))
        mark=eng.mark_ids[2];friend_mark=eng.mark_ids[1]
        before={x.key for x in e._shown[0]}
        self.assertIn(('map',mark),before);self.assertIn(('map',friend_mark),before)
        t0=self.kill(e);self.kill(e,victim=1,killer=3)
        e.step(still(e))
        after={x.key:x for x in e._shown[0]}
        self.assertTrue(before<=set(after))         # nothing the policy had disappears at the deaths
        self.assertTrue(after[('map',friend_mark)].friend)
        shot=after[('shot',m.uid)]
        self.assertEqual((shot.mark_id,shot.support),(mark,True))
        self.assertTrue(m.retargeted or not m.datalink)              # what the sim did and the policy cannot see
        self.assertNotIn('alive',OWN_FIELDS)
        while eng.time<t0+20.5:
            e.step(still(e))
        keys={x.key for x in e._shown[0]}
        self.assertNotIn(('map',friend_mark),keys)                    # the dead teammate has gone after wreck_s
        self.assertNotIn(('map',mark),keys)                           # the enemy mark expired (not refreshed)

    def test_scripts_read_no_wrecks_and_snapshot_restores(self):
        e=self.make(wreck_s=20.)
        eng=e.engagement
        self.kill(e)
        for _ in range(3):
            e.step(still(e))
        self.assertTrue(any(t==2 for t in eng.planes[0].picture.truth_ids))
        mark=eng.mark_ids[2]
        script=e._script_raw[0]
        self.assertFalse([c for c in script.radar if c.mark_id==mark])
        self.assertFalse([b for b in script.boxes if b.ref==mark])
        target=e.scripted_actions()[0]['target']
        ents=e._entities[0]
        self.assertTrue(target==len(ents) or ents[target].mark_id!=mark)
        state=e.snapshot()
        first=[stable(e.step(still(e))) for _ in range(5)]
        e.restore(state)
        self.assertEqual([stable(e.step(still(e))) for _ in range(5)],first)


if __name__=='__main__':
    unittest.main()
