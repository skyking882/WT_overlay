"""Real multi-agent training adapter for Engagement; no numpy or torch imports.

MatchEnv(config, seed), reset/step/scripted_actions/snapshot/restore follow
rl_training_spec 13/13.1 and can replace FakeMatchEnv in the existing trainer.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import random

from .engagement import Camera, default_library, MAP_HALF_M
from .intent import Intent, IntentExecutor, legalize, selected_mask, SAMPLE_ORDER
from . import match
from .rl_observation import (select_entities, masks_for, own_vector, entity_vector,
                             prev_vector, truth_vectors)
from .reach import ReachTable

DT_STEP = 20/48


@dataclass
class AgentObs:
    own: list
    entities: list
    prev_intent: list
    masks: dict
    truth: list
    aircraft: str
    dt: float


class ScriptController:
    """Adapter used by stand-alone matches as well as MatchEnv scripts."""
    def __init__(self, pilot, executor, reach=None):
        self.pilot,self.executor,self.reach=pilot,executor,reach
        self.memory={}
        self.entities=[]
        self.observation=None
        self.labels=None

    @property
    def phase(self):
        return self.pilot.phase

    def describe(self):
        return self.pilot.describe()

    def decide(self, obs):
        self.observation=obs
        self.executor.notice(obs)
        self.entities,_=select_entities(obs,self.executor,self.memory,self.reach)
        # Read only the pilot's last *observed* launch time, not a hidden target.
        masks=masks_for(obs,self.entities,self.executor,self.pilot.shot_t)
        proposal=self.pilot.propose(obs,self.entities)
        self.labels=legalize(proposal,self.entities,masks)
        self.executor.publish(Intent.from_indices(self.labels,self.entities),obs)
        return None

    def advance(self, eng, plane):
        if self.observation is not None:
            self.executor.advance(eng,plane,self.observation,self.entities)


def equip_scripts(eng, *, fov_range=(90.,120.), execution=None, reach_dir=None):
    """Upgrade generated stand-alone match scripts to the common intent route."""
    rng=random.Random(f'{eng.seed}:intent-cameras')
    for p in eng.planes:
        if p.controller is None or not hasattr(p.controller,'propose'):
            continue
        pilot=p.controller
        pilot.performance_model=p.flight.model
        p.camera=Camera(rng.uniform(*fov_range))
        p.camera.advance(p.own,0.)
        executor=IntentExecutor(f'{eng.seed}:executor:{p.ident}',path='autonomous',
                                home_xy=pilot.home_xy,**(execution or {}))
        reach=ReachTable.load(p.missile_id,reach_dir) if p.missile_id else None
        p.controller=ScriptController(pilot,executor,reach)
    return eng


class MatchEnv:
    def __init__(self, config=None, seed=0):
        self.config=copy.deepcopy(config or {})
        self.seed=int(seed)
        self.rng=random.Random(f'{seed}:episodes')
        self.episode_count=0
        self.engagement=None
        self.scenario=None
        self.self_play=False
        self.history=False
        self.frozen_ids=()
        self.episode_kind=None
        self._observations={}
        self._entities={}
        self._raw={}
        self._scripts_cache=None
        n=self.config.get('team_size',1)
        if not isinstance(n,int) or not 1<=n<=16:
            raise ValueError('team_size must be 1..16')
        for name in ('self_play_prob','history_prob'):
            p=self.config.get(name,0.)
            if isinstance(p,bool) or not isinstance(p,(int,float)) or not 0.<=p<=1.:
                raise ValueError(name+' must be a number in [0,1]')
        if self.config.get('self_play_prob',0.)+self.config.get('history_prob',0.)>1.+1e-9:
            raise ValueError('self_play_prob + history_prob must not exceed 1')
        t=self.config.get('timeout_reward')
        if t is not None and (isinstance(t,bool) or not isinstance(t,(int,float))):
            raise ValueError('timeout_reward must be a number')
        if not isinstance(self.config.get('policy_ground_floor',True),bool):
            raise ValueError('policy_ground_floor must be True or False')
        deck=self.config.get('policy_deck_m',100.)
        if isinstance(deck,bool) or not isinstance(deck,(int,float)) or not 0<deck<=8000:
            raise ValueError('policy_deck_m must be a number in (0, 8000]')
        frame=self.config.get('observation_frame','world')
        if frame not in ('world','egocentric'):
            raise ValueError("observation_frame must be 'world' or 'egocentric'")
        self.egocentric=frame=='egocentric'
        layout=self.config.get('spawn_layout')
        if layout is not None and (not isinstance(layout,dict) or set(layout)-{'rotate','offset_km','margin_km'}):
            raise ValueError("spawn_layout must be a dict with keys from 'rotate', 'offset_km', 'margin_km'")
        from .archetypes import PERTURBATION
        perturb=self.config.get('script_perturbation')
        if perturb is not None and (not isinstance(perturb,dict) or set(perturb)-set(PERTURBATION)):
            raise ValueError('script_perturbation must be a dict with keys from archetypes.PERTURBATION')
        # Opt-in (docs/radar_missile_detection_spec.md): radars see enemy missiles in flight; allow_missile_targets
        # would let target / weapon / STT pick a missile track (off: masked by truth, identified or not).
        for name in ('radar_sees_missiles','allow_missile_targets'):
            if not isinstance(self.config.get(name,False),bool):
                raise ValueError(name+' must be True or False')
        self.library=default_library(self.config.get('missile_sim'))

    def reset(self):
        self.episode_count+=1
        episode_seed=self.rng.randrange(2**63)
        c=self.config
        # With self_play_prob or history_prob >0 the episode kind comes from the env's own episode RNG (one draw per
        # reset, so kinds are a function of the env seed). Without them no draw is made and episodes are unchanged.
        # A history episode draws once more: the side flown by the frozen past policy (env.frozen_ids).
        p_self,p_hist=c.get('self_play_prob',0.),c.get('history_prob',0.)
        u=self.rng.random() if p_self>0. or p_hist>0. else 1.
        self.self_play=u<p_self
        self.history=not self.self_play and u<p_self+p_hist
        frozen_side=self.rng.randrange(2) if self.history else None
        model=copy.deepcopy(c.get('model') or match.load_model(c.get('model_path')))
        if 'aircraft_pool' in c:
            allowed=set(c['aircraft_pool'])
            model['aircraft_frequency']['weights']={k:v for k,v in model['aircraft_frequency']['weights'].items() if k in allowed}
            if not model['aircraft_frequency']['weights']:
                raise ValueError('aircraft_pool has no aircraft in the match model')
        kwargs=dict(model=model,library=self.library,map_half_m=c.get('map_half_m',MAP_HALF_M))
        if 'teams' in c:
            if len(c['teams'])!=2 or any(not 1<=len(t)<=16 for t in c['teams']):
                raise ValueError('teams must contain 1..16 aircraft on each side')
            generated=match.scenario(c['teams'],episode_seed,range_km=c.get('range_km'),layout=c.get('spawn_layout'),
                                     perturb=c.get('script_perturbation'),**kwargs)
        else:
            generated=match.random_match(episode_seed,team_size=c.get('team_size',1),layout=c.get('spawn_layout'),
                                         perturb=c.get('script_perturbation'),**kwargs)
        fov=c.get('fov_range',(90.,120.))
        if len(fov)!=2 or not 0<fov[0]<=fov[1]<180.:
            raise ValueError('fov_range must be ordered within (0,180)')
        rcs=c.get('rcs_m2',5.)
        for i,s in enumerate(generated.specs):
            s.camera=Camera(self.rng.uniform(*fov))
            s.rcs_m2=float(rcs.get(i,rcs.get(s.aircraft,5.)) if isinstance(rcs,dict) else rcs)
            if s.rcs_m2<=0:
                raise ValueError('aircraft RCS must be positive')
        ids=c.get('policy_ids',c.get('controlled',None))
        if ids is None:
            ids=list(range(c.get('n_agents',1)))
        elif ids=='all':
            ids=list(range(len(generated.specs)))
        if not ids or len(set(ids))!=len(ids) or any(type(i) is not int or not 0<=i<len(generated.specs) for i in ids):
            raise ValueError('policy_ids must be distinct valid aircraft slots')
        if self.self_play or self.history:
            ids=list(range(len(generated.specs)))   # validated above, so a bad config fails on every episode kind
        self.policy_ids=tuple(ids)
        self.frozen_ids=tuple(i for i,s in enumerate(generated.specs) if s.team==frozen_side) if self.history else ()
        self.pilots={i:s.controller for i,s in enumerate(generated.specs)}
        for s in generated.specs:
            s.controller=None
        # A step always completes exactly 20 ticks, including after a terminal tick.
        limit=c.get('time_limit_s',c.get('max_steps',2160)*DT_STEP)
        if limit<=0:
            raise ValueError('time_limit_s must be positive')
        self.engagement=generated.engagement(time_limit_s=limit,decision_ticks=20,intent_layer=False,
                                            multipath_gain=c.get('multipath_gain'),
                                            seeker_search=c.get('seeker_search'),
                                            structural_speed=bool(c.get('structural_speed',False)),
                                            missile_marker_range_m=c.get('missile_marker_range_m',10000.),
                                            radar_sees_missiles=c.get('radar_sees_missiles',False),
                                            allow_missile_targets=c.get('allow_missile_targets',False))
        eng=self.engagement
        self.executors={}
        self.reach={}
        self.memory={}
        self.last_launch={i:-1e9 for i in range(len(eng.planes))}
        for p in eng.planes:
            pilot=self.pilots[p.ident]
            pilot.performance_model=p.flight.model
            pilot.managed_execution=True
            for name,value in c.get('target_selection',{}).items():
                if name not in ('candidate_count','side_weight','threat_weight','over_shoulder_p'):
                    raise ValueError('unknown target_selection setting '+name)
                setattr(pilot,name,value)
            # User 2026-10-06: policy aircraft may get no pull-out help (policy_ground_floor False) and a lower deck
            # (policy_deck_m, e.g. 10 m); scripted pilots keep theirs.
            floor_kw={} if p.ident not in ids else dict(ground_floor=c.get('policy_ground_floor',True),
                                                         deck_m=c.get('policy_deck_m',100.))
            ex=IntentExecutor(f'{episode_seed}:execute:{p.ident}',path='follow' if p.ident in ids else 'autonomous',**floor_kw,
                              home_xy=pilot.home_xy,**c.get('execution',{}))
            self.executors[p.ident]=ex
            self.reach[p.ident]=ReachTable.load(p.missile_id,c.get('reach_dir')) if p.missile_id else None
            pilot.reach=self.reach[p.ident]
            self.memory[p.ident]={}
            p.camera.advance(p.own,0.)
        self.over=False
        self.scenario=f'match:{episode_seed}:{len(generated.specs)}'
        if 'self_play_prob' in c or 'history_prob' in c:
            self.episode_kind='self_play' if self.self_play else 'history' if self.history else 'vs_script'
            if self.self_play or self.history:
                self.scenario+=':'+self.episode_kind
        else:
            self.episode_kind=None
        return self.observe()

    def observe(self):
        eng=self.engagement
        self._observations={}
        self._entities={}
        self._raw={}
        self._scripts_cache=None
        self.dropped_entities=0
        for p in eng.live:
            raw=eng.observe(p)
            ex=self.executors[p.ident]
            ex.notice(raw)
            entities,dropped=select_entities(raw,ex,self.memory[p.ident],self.reach[p.ident])
            self._raw[p.ident],self._entities[p.ident]=raw,entities
            if p.ident not in self.policy_ids:
                continue
            self.dropped_entities+=dropped
            masks=masks_for(raw,entities,ex,self.last_launch[p.ident])
            # observation_frame 'egocentric': nothing tied to map axes, spawn side or team (rl_observation.EGO_OWN).
            pilot=self.pilots[p.ident]
            ego=(pilot.home_xy,pilot.enemy_xy) if self.egocentric else None
            heading=raw.own.heading_deg if self.egocentric else None
            actor=AgentObs(own_vector(raw,p,ex,self.reach[p.ident],pilot.judge,ego),
                           [entity_vector(e,heading) for e in entities],
                           prev_vector(ex,entities,eng.time),masks,[],p.aircraft,DT_STEP)
            actor.truth=truth_vectors(eng,p,self.egocentric)
            self._observations[p.ident]=actor
        return self._observations

    def scripted_actions(self):
        if self.engagement is None or self.over:
            return {}
        if self._scripts_cache is None:
            labels={}
            for aid,raw in self._raw.items():
                ex=self.executors[aid]
                entities=self._entities[aid]
                obs=self._observations.get(aid)
                masks=obs.masks if obs is not None else masks_for(raw,entities,ex,self.last_launch[aid])
                pilot=self.pilots[aid]
                proposal=pilot.propose(raw,entities)
                labels[aid]=legalize(proposal,entities,masks)
            self._scripts_cache=labels
        return {aid:dict(a) for aid,a in self._scripts_cache.items() if aid in self._observations}

    def step(self, actions):
        if self.engagement is None or self.over:
            raise RuntimeError('reset before stepping a new match')
        if set(actions)!=set(self._observations):
            raise ValueError('actions must match all living policy agents')
        # Decode and validate every action before mutating any simulation state.
        decoded={}
        for aid,action in actions.items():
            ents=self._entities[aid]
            decoded[aid]=Intent.from_indices(action,ents)
            for h in SAMPLE_ORDER:
                if not selected_mask(h,len(ents),self._observations[aid].masks,action)[action[h]]:
                    raise ValueError('illegal action '+h+' for agent '+str(aid))
        if any(p.ident not in self.policy_ids for p in self.engagement.live):
            self.scripted_actions()
        eng=self.engagement
        active=list(actions)
        start=len(eng.log)
        for p in eng.live:
            aid=p.ident
            intent=decoded[aid] if aid in decoded else Intent.from_indices(self._scripts_cache[aid],self._entities[aid])
            self.executors[aid].publish(intent,self._raw[aid])
            pilot=self.pilots[aid]
            if p.phase!=pilot.phase and aid not in self.policy_ids:
                eng.event('phase',plane=aid,frm=p.phase,to=pilot.phase)
                p.phase=pilot.phase
        for _ in range(20):
            for p in eng.live:
                self.executors[p.ident].advance(eng,p,self._raw[p.ident],self._entities[p.ident])
            eng.step()
        new=eng.log[start:]
        rewards={aid:0. for aid in active}
        events=dict(launch=0,kill=0,assist=0,death=0,dropped_entities=0,friendly_fire=0,retarget=0,
                    crash=0,out_of_bounds=0,overspeed=0,missile_error=0)
        if eng.radar_sees_missiles:
            events['radar_missile_tracks']=0   # missile tracks the radars started this step (all aircraft)
        # A policy aircraft that is already down still scores with the missiles it left in the air: those rewards
        # go to info['late_rewards'] (the trainer adds them to its final step). tallies: kills / deaths per policy
        # aircraft this step, so episode outcomes need not be read back from summed rewards.
        late,tallies={},{}
        for e in new:
            kind=e['kind']
            if kind in events:
                events[kind]+=1
            if kind=='kill' and e['killer'] in self.policy_ids:
                k=e['killer']
                if k in rewards:
                    rewards[k]+=1.
                else:
                    late[k]=late.get(k,0.)+1.
                tallies.setdefault(k,[0,0])[0]+=1
            elif kind=='assist' and e['plane'] in self.policy_ids:
                if e['plane'] in rewards:
                    rewards[e['plane']]+=.3
                else:
                    late[e['plane']]=late.get(e['plane'],0.)+.3
            elif kind=='death':
                if e['plane'] in rewards:
                    rewards[e['plane']]-=2.
                if e['plane'] in self.policy_ids:
                    tallies.setdefault(e['plane'],[0,0])[1]+=1
                if e['cause'] in ('crash','out_of_bounds','overspeed'):
                    events[e['cause']]+=1
            elif kind=='launch':
                self.last_launch[e['shooter']]=eng.planes[e['shooter']].last_launch
            elif kind=='radar_missile_track':
                events['radar_missile_tracks']+=1
        self.over=eng.reason is not None
        timeout=eng.reason=='time_limit'
        penalty=self.config.get('timeout_reward')
        if timeout and penalty is not None:
            # Opt-in: running out of time is a result of its own. Every policy aircraft still alive gets the
            # reward and the time limit becomes a terminal state (info timeout False: no value bootstrap).
            for aid in rewards:
                if eng.planes[aid].alive:
                    rewards[aid]+=float(penalty)
            timeout=False
        dones={aid:self.over or not eng.planes[aid].alive for aid in active}
        obs=self.observe()
        events['dropped_entities']=self.dropped_entities
        info=dict(timeout=timeout,events=events,reason=eng.reason,time_s=eng.time,dt=DT_STEP)
        if penalty is not None:
            info['time_limit']=eng.reason=='time_limit'
        if late:
            info['late_rewards']=late
        if tallies:
            info['tallies']=tallies
        if self.episode_kind is not None:
            # Decisions taken in self-play episodes this step (agent-summed like the other events).
            events['self_play_decisions']=len(active) if self.self_play else 0
            info.update(self_play=self.self_play,episode_kind=self.episode_kind)
            if 'history_prob' in self.config:
                # history episodes: decisions of the current policy (the frozen side's are not counted)
                events['history_decisions']=sum(1 for a in active if a not in self.frozen_ids) if self.history else 0
                info['frozen_ids']=list(self.frozen_ids)
        if self.over and not timeout:
            obs={}
            self._observations={}
        return obs,rewards,dones,info

    def pending_credit(self):
        """True while a policy aircraft that is already down can still be credited (info['late_rewards']), so a
        trainer should play the match on before resetting it: one of its missiles is still flying, or a teammate can
        still kill an enemy it shot at within the assist window (Engagement._kill: a teammate of the killer whose
        missile at the victim was in flight within the last 20 s gets an assist, alive or not). The frozen side of a
        history episode (frozen_ids) trains nothing, so nothing is owed to it."""
        eng=self.engagement
        if eng is None or self.over:
            return False
        owed={i for i in self.policy_ids if i not in self.frozen_ids and not eng.planes[i].alive}
        if not owed:
            return False
        if any(not m.done and m.shooter.ident in owed for m in eng.missiles):
            return True
        t=eng.time
        for v in eng.live:
            for shooter,_,t_end in v.missile_hist:
                team=eng.planes[shooter].team
                if shooter in owed and team!=v.team and (t_end is None or t_end>=t-20.):
                    # a kill needs a possible killer: a live teammate, or a teammate's missile still in flight
                    if any(p.team==team and p.ident!=shooter for p in eng.live) or \
                            any(not m.done and m.shooter.team==team and m.shooter.ident!=shooter for m in eng.missiles):
                        return True
        return False

    def snapshot(self):
        if self.engagement is None:
            raise RuntimeError('reset before snapshot')
        return self._copy_state(self.__dict__)

    def restore(self, state):
        st=self._copy_state(state)
        self.__dict__.clear()
        self.__dict__.update(st)

    @staticmethod
    def _copy_state(state):
        # Immutable FM force tables and source models are shared; mutable numeric
        # caches, RNGs, sensor/controller history and callback graph are copied.
        eng=state['engagement']
        memo={id(state['library']):state['library']}
        if eng is not None:
            for p in eng.planes:
                model=p.flight.model
                for immutable in (model._aero,model.model):
                    memo[id(immutable)]=immutable
        return copy.deepcopy(state,memo)
