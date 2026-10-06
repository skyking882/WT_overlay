"""Environment contract, observation causality and exact replay of mutable state."""
from __future__ import annotations

import contextlib
import copy
from dataclasses import replace
import io
import json
import math
import pickle
import random
from types import SimpleNamespace
import unittest

from rl import check_env, spec, wire
from wt_overlay.engagement import Camera, Sighting, LauncherSupport
from wt_overlay.flight import KeyboardCommand, FlightCommand
from wt_overlay.intent import Intent, IntentExecutor, HEAD_NAMES, launch_limit, selected_mask
from wt_overlay import match
from wt_overlay.rl_env import MatchEnv, DT_STEP
from wt_overlay.rl_observation import (Entity, select_entities, masks_for, OWN_FIELDS, OWN_FIELDS_EGO, ENTITY_FIELDS,
                                        raw_entities, entity_vector)
from wt_overlay.sensors import RwrContact
from wt_overlay.reach import ReachTable

ZERO=dict(reject_p=0.,error_deg=0.,delay={p:dict(median_range=(.001,.001),sigma=0.,bounds=(0.,0.))
                                        for p in ('follow','autonomous')})
TEAMS=[[dict(aircraft='f_15c_golden_eagle',archetype='middle',skill='top',altitude_m=8000.,mach=1.)],
       [dict(aircraft='su_30sm2',archetype='middle',skill='top',altitude_m=8000.,mach=1.)]]


def env(seed=5, **kwargs):
    config=dict(teams=TEAMS,range_km=40.,controlled='all',execution=ZERO,time_limit_s=120.)
    config.update(kwargs)
    e=MatchEnv(config,seed)
    e.reset()
    return e


def random_actions(e,rng):
    result={}
    for aid,o in e._observations.items():
        chosen={}
        for h in spec.SAMPLE_ORDER:
            row=selected_mask(h,len(o.entities),o.masks,chosen)
            chosen[h]=rng.choice([i for i,b in enumerate(row) if b])
        result[aid]=chosen
    return result


def stable(result):
    obs,r,d,i=result
    return ({a:wire.pack_obs(o,True) for a,o in obs.items()},r,d,i)


class CameraTests(unittest.TestCase):
    def test_projection_and_joint_rate_limit(self):
        own=SimpleNamespace(heading_deg=0.,pitch_deg=0.)
        camera=Camera(90.)
        self.assertTrue(camera.sees(own,45.,0.))
        self.assertFalse(camera.sees(own,46.,0.))
        self.assertFalse(camera.sees(own,0.,40.))
        camera.advance(own,0.)
        camera.point(180.,0.,2)
        camera.advance(own,.25)
        self.assertAlmostEqual(abs((camera.bearing_deg+180)%360-180),45.)
        camera.point(270.,60.,2)
        old=(camera.bearing_deg,camera.elevation_deg)
        camera.advance(own,.1)
        self.assertLessEqual(math.hypot((camera.bearing_deg-old[0]+180)%360-180,camera.elevation_deg-old[1]),18.+1e-9)

    def test_camera_changes_only_visual_channels(self):
        e=env(range_km=6.)
        p=e.engagement.planes[0]
        p.camera.point(0.,0.)
        p.camera.advance(p.own,1.)
        front=e.engagement.observe(p)
        p.camera.point(180.,0.,2)
        p.camera.advance(p.own,1.)
        back=e.engagement.observe(p)
        self.assertTrue(front.visual and front.boxes)
        self.assertFalse(back.visual or back.boxes)
        self.assertEqual(front.radar,back.radar)
        self.assertEqual(front.rwr,back.rwr)
        self.assertEqual(front.maw,back.maw)
        self.assertEqual(front.marks,back.marks)

    def test_contrail_is_visible_at_any_distance_above_9500(self):
        teams=copy.deepcopy(TEAMS)
        teams[1][0]['altitude_m']=11000.
        e=env(teams=teams,range_km=100.)
        p=e.engagement.planes[0]
        front=e.engagement.observe(p)
        self.assertTrue(front.contrails)
        self.assertFalse(front.visual)
        p.camera.point(180.,0.,2);p.camera.advance(p.own,1.)
        self.assertFalse(e.engagement.observe(p).contrails)


class ExecutorTests(unittest.TestCase):
    def test_hold_starts_at_publication_and_only_visible_new_alert_releases(self):
        e=env();raw=e._raw[0]
        x=IntentExecutor(1,**ZERO)
        x.publish(Intent(maneuver=2,vertical=1),raw)
        self.assertEqual(x.hold_until,2.)
        self.assertEqual(x.executed.maneuver,0)
        with self.assertRaises(ValueError):
            x.publish(Intent(maneuver=3,vertical=1),replace(raw,time_s=.5))
        silent=replace(raw,time_s=.6,truth=SimpleNamespace(incoming_missile=True))
        x.notice(silent);self.assertTrue(x.held(.6))
        warning=RwrContact(5,'missile',0.,0.,0.,None,False,True,8,None,True,True,0.)
        x.notice(replace(raw,time_s=.7,rwr=(warning,)))
        self.assertFalse(x.held(.7))
        x.publish(Intent(maneuver=3),replace(raw,time_s=.7,rwr=(warning,)))
        self.assertAlmostEqual(x.hold_until,2.7)
        x.notice(replace(raw,time_s=.8,rwr=(warning,)))
        self.assertTrue(x.held(.8))

    def test_delay_paths_have_one_personal_median_and_do_not_stack(self):
        e=env();raw=e._raw[0]
        for path,bounds,medians in [('follow',(.25,2.5),(.5,1.2)),('autonomous',(.5,5.),(1.,2.5))]:
            x=IntentExecutor(4,path=path,reject_p=0.)
            median=x.personal_median
            self.assertTrue(medians[0]<=median<=medians[1])
            times=[]
            for j in range(200):
                now=j*3.
                x.publish(Intent(look_az=j%8,view_mode=2),replace(raw,time_s=now))
                times.append(x.pending[-1][0]-now)
                self.assertEqual(x.personal_median,median)
            self.assertTrue(all(bounds[0]-1e-9<=t<=bounds[1]+1e-9 for t in times))
            old=len(x.pending)
            x.publish(x.published,replace(raw,time_s=601.))
            self.assertEqual(len(x.pending),old)

    def test_rejection_is_an_execution_effect_not_action_mutation(self):
        e=env();x=IntentExecutor(1,reject_p=1.)
        proposal=Intent(maneuver=6)
        x.publish(proposal,e._raw[0])
        self.assertEqual(x.published,proposal)
        self.assertEqual(x.executed,Intent())
        self.assertEqual(x.pending,[])
        self.assertEqual(x.rejected,1)

    def test_free_look_keyboard_and_fresh_exit(self):
        e=env();action=e.scripted_actions()
        for a in action.values():
            a.update(view_mode=2,maneuver_ref=0,maneuver=0,vertical=0,look_az=4,kb_roll=2,kb_pitch=2)
        # Pointer "none" is the last option, including when n=0.
        for aid,a in action.items():
            a['maneuver_ref']=a['view_object']=len(e._entities[aid])
        e.step(action)
        self.assertIsInstance(e.engagement.planes[0].flight.command,KeyboardCommand)
        self.assertEqual(e.engagement.planes[0].flight.command.pitch,1)
        obs=e._observations[0]
        self.assertEqual(sum(obs.masks['maneuver'][2][1]),1)
        action=e.scripted_actions()
        for a in action.values():
            a.update(view_mode=0,maneuver=6,vertical=2,look_az=0,look_el=1,kb_roll=1,kb_pitch=1)
        e.step(action)
        self.assertIsInstance(e.engagement.planes[0].flight.command,FlightCommand)
        self.assertAlmostEqual(e.executors[0].hold_until,DT_STEP+2.)


class ContractTests(unittest.TestCase):
    def test_script_climb_completes_at_discrete_altitude_and_edge_target_updates(self):
        e=env()
        pilot=e.pilots[0]
        pilot.managed_execution=True
        pilot.peak_done=True
        pilot.p=replace(pilot.p,level_alt_m=8900.,flank_dist_m=0.)
        raw=e._raw[0]
        pilot.decide(raw)
        self.assertNotEqual(pilot.phase,'climb')
        ex=e.executors[0]
        ex.executed=Intent(maneuver=7)
        ex.aim_heading=0.
        corner=replace(raw,own=replace(raw.own,position=(63000.,63000.,8000.),velocity=(300.,300.,0.)))
        ex._aim(corner,[],ex.executed,0.,False)
        self.assertLess(math.sin(math.radians(ex.aim_heading)),0.)
        self.assertLess(math.cos(math.radians(ex.aim_heading)),0.)

    def test_shapes_masks_and_script_labels_roundtrip(self):
        e=env()
        self.assertEqual(tuple(HEAD_NAMES),spec.HEAD_NAMES)
        self.assertEqual(len(OWN_FIELDS),48);self.assertEqual(len(ENTITY_FIELDS),24)
        for step in range(25):
            acts=e.scripted_actions()
            self.assertEqual(acts,e.scripted_actions())
            for aid,o in e._observations.items():
                wire.pack_obs(o,True)
                self.assertEqual(spec.check_mask_shapes(len(o.entities),o.masks),[])
                self.assertEqual(spec.empty_selectable_rows(len(o.entities),o.masks),[])
                ents=e._entities[aid]
                self.assertEqual(Intent.from_indices(acts[aid],ents).indices(ents),acts[aid])
                _,forced,illegal=spec.canonicalize_action(len(ents),o.masks,acts[aid])
                self.assertEqual(illegal,[])
                if e.executors[aid].held(e.engagement.time):
                    self.assertEqual(sum(o.masks['maneuver_ref'][0]),1)
                    self.assertEqual(sum(o.masks['maneuver'][0][0]),1)
            e.step(acts)

    def test_truth_cannot_change_actor_selection_or_masks(self):
        e=env();raw=e._raw[0];x=e.executors[0]
        a,d=select_entities(raw,x,{},None)
        secret=replace(raw,truth=SimpleNamespace(planes='new hidden planes'))
        b,d2=select_entities(secret,x,{},None)
        self.assertEqual(a,b);self.assertEqual(d,d2)
        self.assertEqual(masks_for(raw,a,x,-1e9),masks_for(secret,b,x,-1e9))
        # Critic-only changes must not move target, sorting, or actor feature rows.
        e.engagement.planes[1].flight.state=replace(e.engagement.planes[1].flight.state,
                                                   velocity=(999.,0.,0.))
        self.assertEqual(raw_entities(raw),raw_entities(secret))

    def test_priority_retains_references_and_warning_without_range(self):
        e=env();raw=e._raw[0];x=e.executors[0]
        cues=tuple(Sighting('aircraft',i,0.,0.,None,0.) for i in range(80))
        warning=RwrContact(500,'missile',0.,0.,0.,None,False,True,8,None,True,True,0.)
        raw=replace(raw,visual=cues,rwr=(warning,))
        x.published=Intent(maneuver_ref=('visual',79))
        selected,dropped=select_entities(raw,x,{})
        self.assertEqual(len(selected),64)
        self.assertEqual(dropped,17)
        self.assertEqual(selected[0].key,('visual',79))
        self.assertEqual(selected[1].key,('rwr',500))

    def test_launch_angle_is_shared_with_weapon_mask(self):
        self.assertEqual(launch_limit('su_30sm2','su_r_77_1'),120.)
        self.assertEqual(launch_limit('f_15c_golden_eagle','us_aim_120d'),70.)
        self.assertEqual(launch_limit('ef_2000a_aesa','us_aim_120c_5'),60.)

    def test_death_rewards_and_timeout_final_observation(self):
        e=env(time_limit_s=DT_STEP)
        obs,rewards,dones,info=e.step(e.scripted_actions())
        self.assertTrue(info['timeout']);self.assertTrue(all(dones.values()))
        self.assertEqual(set(obs),{0,1});self.assertEqual(info['dt'],DT_STEP)
        e=env()
        p=e.engagement.planes[0]
        p.flight.state=replace(p.flight.state,position=(0.,0.,-100.))
        obs,r,d,i=e.step(e.scripted_actions())
        self.assertEqual(r[0],-2.);self.assertTrue(d[0]);self.assertNotIn(0,obs)
        self.assertEqual(i['events']['death'],1)
        self.assertEqual(i['events']['crash'],1)

    def test_egocentric_observation_ignores_how_the_spawn_is_turned(self):
        # The same match with its spawn picture turned by 90 degrees (the square map maps onto itself): world-frame
        # vectors change, egocentric ones (own, entities, critic truth) stay the same for both sides.
        orig=match.spawn_layout
        def run(theta,frame):
            match.spawn_layout=lambda seed,layout,*a:(theta,0.,0.) if layout else None
            try:
                e=env(observation_frame=frame,spawn_layout=dict(rotate=True),time_limit_s=60.)
                for _ in range(3):
                    obs,_,_,_=e.step(e.scripted_actions())
                return obs
            finally:
                match.spawn_layout=orig
        def close(u,v):
            return len(u)==len(v) and all(abs(x-y)<1e-6 for x,y in zip(u,v))
        a,b=run(0.,'world'),run(90.,'world')
        self.assertFalse(close(a[0].own,b[0].own))
        a,b=run(0.,'egocentric'),run(90.,'egocentric')
        for aid in (0,1):
            self.assertTrue(close(a[aid].own,b[aid].own))
            self.assertTrue(all(close(x,y) for x,y in zip(a[aid].entities,b[aid].entities)))
            self.assertTrue(all(close(x,y) for x,y in zip(a[aid].truth,b[aid].truth)))

    def test_egocentric_own_vector_has_no_team_or_map_axes(self):
        same=[[dict(aircraft='su_30sm2',archetype='middle',skill='top',altitude_m=8000.,mach=1.)]]*2
        e=env(teams=same,observation_frame='egocentric')
        o0,o1=e._observations[0].own,e._observations[1].own
        n=len(OWN_FIELDS_EGO)
        # Mirrored spawns differ by spawn jitter only. Each side starts at its home (bearing undefined there) and
        # flies at the enemy home.
        for f in ('home_range','enemy_home_sin','enemy_home_cos','enemy_home_range',
                  'edge_ahead','edge_right','edge_behind','edge_left','edge_min'):
            k=OWN_FIELDS_EGO.index(f)
            self.assertAlmostEqual(o0[k],o1[k],delta=.02,msg=f)
        for o in (o0,o1):
            self.assertLess(o[OWN_FIELDS_EGO.index('home_range')],.02)
            self.assertGreater(o[OWN_FIELDS_EGO.index('enemy_home_cos')],.99)
        with self.assertRaises(ValueError):
            MatchEnv(dict(observation_frame='team'),0)

    def test_spawn_layout_turns_and_shifts_inside_the_map(self):
        for seed in range(8):
            m=match.random_match(seed,team_size=4,layout=dict(rotate=True,offset_km=20.))
            theta,dx,dy=m.notes['layout']
            for s in m.specs:
                self.assertLess(max(abs(s.position[0]),abs(s.position[1])),m.map_half_m-7000.)
                p=s.controller
                facing=math.degrees(math.atan2(p.enemy_xy[0]-p.home_xy[0],p.enemy_xy[1]-p.home_xy[1]))%360.
                self.assertAlmostEqual(math.cos(math.radians(facing-p.forward_deg)),1.,places=9)
        plain=match.random_match(3,team_size=2)
        self.assertEqual([s.position for s in plain.specs],[s.position for s in match.random_match(3,team_size=2,layout=None).specs])
        self.assertNotIn('layout',plain.notes)
        with self.assertRaises(ValueError):
            MatchEnv(dict(spawn_layout=dict(spin=True)),0)

    def test_script_perturbation_varies_pilots_and_is_off_by_default(self):
        from wt_overlay.archetypes import PERTURBATION, PilotParams
        plain=match.random_match(4,team_size=8)
        again=match.random_match(4,team_size=8,perturb=None)
        self.assertEqual([s.controller.p for s in plain.specs],[s.controller.p for s in again.specs])
        for s in plain.specs:
            p=s.controller.p
            self.assertEqual((p.commit_alt_m,p.commit_delay_s,p.early_recommit_s,p.evade_inner),(None,0.,None,False))
        shaken=match.random_match(4,team_size=8,perturb={})
        changed=[a.controller.p!=b.controller.p for a,b in zip(plain.specs,shaken.specs)]
        self.assertTrue(any(changed) and not all(changed))      # p=0.5 of the pilots
        for a,b in zip(plain.specs,shaken.specs):
            q=b.controller.p
            if q!=a.controller.p:
                lo,hi=PERTURBATION['level_alt_m']
                self.assertTrue(lo<=q.level_alt_m<=hi);self.assertEqual(q.commit_alt_m,q.level_alt_m)
                self.assertEqual((q.archetype,q.skill),(a.controller.p.archetype,a.controller.p.skill))
        # Everything else in the match (aircraft, spawns) is untouched.
        self.assertEqual([(s.aircraft,s.position) for s in plain.specs],[(s.aircraft,s.position) for s in shaken.specs])
        self.assertEqual([s.controller.p for s in match.random_match(4,team_size=8,perturb=dict(p=0.)).specs],
                         [s.controller.p for s in plain.specs])
        with self.assertRaises(ValueError):
            MatchEnv(dict(script_perturbation=dict(wobble=1.)),0)

    def test_downed_aircraft_still_scores_with_its_missiles(self):
        # 2v2, so the match goes on after aircraft 0 is down (its wingman 1 is alive); 2 and 3 are the enemies.
        e=env(teams=[TEAMS[0]*2,TEAMS[1]*2])
        p0,p1,p2,p3=e.engagement.planes
        p0.flight.state=replace(p0.flight.state,position=(0.,0.,-100.))
        obs,r,d,i=e.step(e.scripted_actions())
        self.assertEqual(r[0],-2.);self.assertTrue(d[0]);self.assertEqual(i['tallies'][0],[0,1])
        self.assertFalse(e.over)
        self.assertFalse(e.pending_credit())
        # A missile of the downed aircraft still flying is credit owed to it.
        e.engagement.missiles.append(SimpleNamespace(done=False,shooter=p0))
        self.assertTrue(e.pending_credit())
        e.engagement.missiles.pop()
        # ...and when it kills, the reward goes to info late_rewards (the aircraft no longer acts).
        eng=e.engagement
        step=eng.step
        def killing_step():
            eng.step=step
            step()
            eng._kill(p2,p0,'missile',None)
        eng.step=killing_step
        obs,r,d,i=e.step(e.scripted_actions())
        self.assertEqual(r[2],-2.);self.assertNotIn(0,r);self.assertEqual(i['late_rewards'],{0:1.})
        self.assertEqual(i['tallies'][0],[1,0]);self.assertEqual(i['tallies'][2],[0,1])

    def test_structural_speed_tears_the_wings_off_beyond_vne(self):
        for structural in (False,True):
            e=env(structural_speed=structural)
            p0=e.engagement.planes[0]
            p0.flight.state=replace(p0.flight.state,position=(0.,0.,600.),velocity=(0.,560.,0.))
            deaths=[]
            for _ in range(4):
                obs,r,d,i=e.step(e.scripted_actions())
                deaths+=[x for x in e.engagement.log if x['kind']=='death' and x['plane']==0]
                if 0 not in e._observations:
                    break
            if structural:
                self.assertEqual(deaths[0]['cause'],'overspeed');self.assertEqual(r[0],-2.)
                self.assertEqual(i['events']['overspeed'],1)
            else:
                self.assertEqual(deaths,[])

    def test_timeout_reward_makes_the_time_limit_terminal(self):
        e=env(time_limit_s=DT_STEP,timeout_reward=-.5)
        obs,rewards,dones,info=e.step(e.scripted_actions())
        self.assertEqual(rewards,{0:-.5,1:-.5});self.assertTrue(all(dones.values()))
        self.assertFalse(info['timeout']);self.assertTrue(info['time_limit']);self.assertEqual(obs,{})
        # A match decided on the last step is not a timeout: the crash ends it and nobody gets the timeout reward.
        e=env(time_limit_s=DT_STEP,timeout_reward=-.5)
        p=e.engagement.planes[0]
        p.flight.state=replace(p.flight.state,position=(0.,0.,-100.))
        obs,r,d,i=e.step(e.scripted_actions())
        self.assertEqual(r,{0:-2.,1:0.});self.assertFalse(i['time_limit'])
        with self.assertRaises(ValueError):
            MatchEnv(dict(timeout_reward=True),0)

    def test_determinism_and_snapshot_with_live_missile_and_pending_actions(self):
        e=env(execution=dict(reject_p=.05,error_deg=5.))
        for _ in range(12):
            e.step(e.scripted_actions())
        # Explicit launch exercises observer, controller, chaff and datalink state.
        m=e.engagement.fire(*e.engagement.planes)
        self.assertIsInstance(m.runtime.provider.launcher_support,LauncherSupport)
        e.engagement.drop_chaff(e.engagement.planes[1],2)
        e.observe()
        snapshot=e.snapshot()
        rng=random.Random(9)
        actions=random_actions(e,rng)
        first=stable(e.step(actions))
        e.restore(snapshot)
        self.assertIsNot(e.engagement,snapshot['engagement'])
        self.assertIs(e.engagement.missiles[0].runtime.provider.launcher_support.missile,
                      e.engagement.missiles[0])
        self.assertEqual(first,stable(e.step(actions)))
        # Restoration must not mutate the reusable saved state.
        e.restore(snapshot);self.assertEqual(first,stable(e.step(actions)))
        fresh=env(execution=dict(reject_p=.05,error_deg=5.))
        for _ in range(12):fresh.step(fresh.scripted_actions())
        fresh.engagement.fire(*fresh.engagement.planes)
        fresh.engagement.drop_chaff(fresh.engagement.planes[1],2)
        fresh.observe()
        self.assertEqual(first,stable(fresh.step(actions)))

    def test_dead_target_missile_can_reacquire_friend_and_no_friendly_kill_reward(self):
        e=env(range_km=30.)
        eng=e.engagement
        for _ in range(12):e.step(e.scripted_actions())
        shooter,target=eng.planes
        m=eng.fire(shooter,target)
        eng._kill(target,None,'crash',None)
        # Place shooter ahead of missile inside seeker FOV to exercise friend selection.
        state=shooter.flight.state
        shooter.flight.state=replace(state,position=(state.position[0],state.position[1]+1000.,state.position[2]))
        eng._retarget(m)
        self.assertIs(m.target,shooter)
        self.assertFalse(m.done)
        eng._kill(shooter,shooter,'missile',m)
        self.assertEqual(shooter.kills,0)
        self.assertTrue(any(e['kind']=='friendly_fire' for e in eng.log))

    def test_16v16_random_legal_policy_for_120_seconds(self):
        e=MatchEnv(dict(team_size=16,controlled='all',time_limit_s=120.),31)
        e.reset();rng=random.Random(17)
        for _ in range(288):
            # Random keyboard pushes can kill every controlled plane; terminal
            # episodes reset while the stress run still covers 120 simulated s.
            if e.over or not e._observations:
                e.reset()
            obs,r,d,info=e.step(random_actions(e,rng))
            for o in obs.values():wire.pack_obs(o,True)
            self.assertEqual(e.engagement.missile_errors,0)
        self.assertEqual(288*DT_STEP,120.)


class ReachTests(unittest.TestCase):
    def test_sparse_table_does_not_invent_missing_times(self):
        axes=dict(altitude_m=[8000.],speed_mps=[300.],delta_altitude_m=[0.],aspect_deg=[0.])
        table=ReachTable(dict(version=1,axes=axes,cells=[dict(point=[8000.,300.,0.,0.],rmax_m=60000.,
                    censored=False,samples=[dict(range_m=10000.,flight_s=10.,seeker_on_s=0.),
                                           dict(range_m=60000.,flight_s=70.,seeker_on_s=50.)])]))
        self.assertEqual(table.rmax(8000.,300.),60000.)
        self.assertEqual(table.times(8000.,300.,0.,0.,35000.),(40.,25.))
        self.assertIsNone(table.times(8000.,300.,0.,0.,90000.))
        self.assertIsNone(table.rmax(9000.,300.))


class SelfPlayTests(unittest.TestCase):
    """self_play_prob: each reset draws the episode kind from the env's own episode RNG."""
    CONFIG=dict(teams=TEAMS,range_km=40.,execution=ZERO,time_limit_s=20.)

    def make(self,seed=5,**kwargs):
        return MatchEnv(dict(self.CONFIG,**kwargs),seed)

    def kinds(self,seed,n,**kwargs):
        e=self.make(seed,**kwargs)
        out=[]
        for _ in range(n):
            obs=e.reset()
            out.append((e.episode_kind,tuple(sorted(obs)),e.scenario))
        return out

    def test_kinds_are_deterministic_per_seed_and_mixed(self):
        a=self.kinds(11,8,self_play_prob=.5)
        self.assertEqual(a,self.kinds(11,8,self_play_prob=.5))
        self.assertEqual({k for k,_,_ in a},{'self_play','vs_script'})
        self.assertNotEqual([k for k,_,_ in a],[k for k,_,_ in self.kinds(12,8,self_play_prob=.5)])
        # The kind is a function of the seed only: it does not depend on what the policy does in between.
        e=self.make(11,self_play_prob=.5)
        rng=random.Random(1)
        seen=[]
        for _ in range(4):
            e.reset();seen.append(e.episode_kind)
            e.step(random_actions(e,rng))
        self.assertEqual(seen,[k for k,_,_ in a[:4]])

    def test_self_play_controls_every_slot_and_other_episodes_the_configured_ones(self):
        for configured,expect in ((None,(0,)),([1],(1,)),('all',(0,1))):
            extra={} if configured is None else dict(controlled=configured)
            rows=self.kinds(11,8,self_play_prob=.5,**extra)
            for kind,agents,scenario in rows:
                if kind=='self_play':
                    self.assertEqual(agents,(0,1));self.assertTrue(scenario.endswith(':self_play'))
                else:
                    self.assertEqual(agents,expect);self.assertNotIn('self_play',scenario)
        # policy_ids is honoured the same way, and a 2v2 self-play episode controls all four aircraft.
        for kind,agents,_ in self.kinds(11,8,self_play_prob=.5,policy_ids=[1]):
            self.assertEqual(agents,(0,1) if kind=='self_play' else (1,))
        big=MatchEnv(dict(team_size=2,execution=ZERO,time_limit_s=20.,self_play_prob=1.),3)
        self.assertEqual(sorted(big.reset()),[0,1,2,3])

    def test_self_play_episode_runs_with_both_agents_on_the_follow_path(self):
        e=self.make(self_play_prob=1.)
        obs=e.reset()
        self.assertEqual((e.episode_kind,e.self_play,e.policy_ids),('self_play',True,(0,1)))
        self.assertTrue(all(ex.path=='follow' for ex in e.executors.values()))
        rng=random.Random(3)
        for _ in range(5):
            obs,r,d,info=e.step(random_actions(e,rng))
            self.assertEqual((info['self_play'],info['episode_kind']),(True,'self_play'))
            self.assertEqual(info['events']['self_play_decisions'],2)
            self.assertEqual(set(r),{0,1});self.assertEqual(set(obs),{0,1})
        snap=e.snapshot()
        e.restore(snap)
        self.assertEqual(e.episode_kind,'self_play')

    def test_vs_script_episode_reports_its_kind_and_zero_self_play_decisions(self):
        e=self.make(self_play_prob=0.)
        e.reset()
        obs,r,d,info=e.step(e.scripted_actions())
        self.assertEqual((info['self_play'],info['episode_kind']),(False,'vs_script'))
        self.assertEqual(info['events']['self_play_decisions'],0)
        self.assertEqual(set(obs),{0})
        self.assertEqual(e.executors[0].path,'follow');self.assertEqual(e.executors[1].path,'autonomous')

    def test_option_absent_changes_nothing(self):
        absent=self.make();zero=self.make(self_play_prob=0.)
        a,b=absent.reset(),zero.reset()
        self.assertEqual(absent.scenario,zero.scenario)
        self.assertEqual({k:wire.pack_obs(o) for k,o in a.items()},{k:wire.pack_obs(o) for k,o in b.items()})
        self.assertEqual(sorted(a),[0])
        self.assertIsNone(absent.episode_kind)
        actions=absent.scripted_actions()
        _,_,_,ia=absent.step(actions)
        _,_,_,ib=zero.step(actions)
        for key in ('self_play','episode_kind'):self.assertNotIn(key,ia)
        self.assertNotIn('self_play_decisions',ia['events'])
        self.assertEqual(ia['events'],{k:v for k,v in ib['events'].items() if k!='self_play_decisions'})
        # No extra RNG draw without the option: the episode seeds of both runs stay the same ones.
        self.assertEqual(absent.rng.random(),zero.rng.random())

    def test_invalid_probability_is_rejected(self):
        for bad in (-.1,1.5,'half',None,True):
            with self.assertRaises(ValueError,msg=repr(bad)):
                MatchEnv(dict(self.CONFIG,self_play_prob=bad),1)

    def test_contract_checks_pass_with_mixed_episodes(self):
        config=dict(team_size=1,time_limit_s=25.,self_play_prob=.5)
        out=io.StringIO()
        with contextlib.redirect_stdout(out):
            code=check_env.main(['--env','wt_overlay.rl_env:MatchEnv','--config',json.dumps(config),
                                 '--episodes','4','--max-steps','60'])
        report=json.loads(out.getvalue())
        self.assertEqual((code,report['errors']),(0,[]))
        stats=report['stats']
        self.assertGreater(stats['episodes_self_play'],0);self.assertGreater(stats['episodes_vs_script'],0)
        self.assertEqual(stats['controlled_agents_self_play'],2*stats['episodes_self_play'])
        self.assertEqual(stats['controlled_agents_vs_script'],stats['episodes_vs_script'])


if __name__=='__main__':
    unittest.main()
