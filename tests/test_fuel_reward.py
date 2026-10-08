"""Opt-in reward anti-spam (assist_rule, launch_reward, retarget_kill_reward), fuel and the airfield approach time
(docs/fuel_spec.md)."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import math
import pickle
import unittest

from rl import wire
from wt_overlay import match
from wt_overlay.engagement import AIRFIELD, FUEL, MISSILE_MASS_KG, ReplayWriter
from wt_overlay.flight import (FUEL_MASS_STEP_KG, Aircraft, Flameout, FlightCommand, FuelTank, aircraft_model,
                               fuel_data)
from wt_overlay.fm import atmosphere
from wt_overlay.rl_env import MatchEnv
from wt_overlay.turn import ManeuverModel

ZERO = dict(reject_p=0., error_deg=0., delay={p: dict(median_range=(.001, .001), sigma=0., bounds=(0., 0.))
                                              for p in ('follow', 'autonomous')})
GE = dict(aircraft='f_15c_golden_eagle', archetype='middle', skill='top', altitude_m=8000., mach=1.)
SM2 = dict(aircraft='su_30sm2', archetype='middle', skill='top', altitude_m=8000., mach=1.)
TEAMS1 = [[GE], [SM2]]
TEAMS2 = [[GE, GE], [SM2, SM2]]


def env(seed=5, teams=TEAMS2, **kwargs):
    config = dict(teams=teams, range_km=40., controlled='all', execution=ZERO, time_limit_s=120.)
    config.update(kwargs)
    e = MatchEnv(config, seed)
    e.reset()
    return e


def act(e, aid, maneuver=0):
    """A legal intent flying ``maneuver`` with no reference, target, weapon or chaff (no policy launch)."""
    n = len(e._entities[aid])
    return dict(maneuver_ref=n, target=n, view_object=n, maneuver=maneuver, vertical=0, speed=0, chaff=0,
                radar_mode=0, antenna=2, weapon=0, view_mode=0, look_az=0, look_el=1, kb_roll=1, kb_pitch=1)


def quiet(e, maneuvers=None):
    return {a: act(e, a, (maneuvers or {}).get(a, 0)) for a in e._observations}


def after_next_tick(eng, fn):
    """Run ``fn`` once right after the engine's next tick (inside the next MatchEnv.step)."""
    step = eng.step

    def once():
        eng.step = step
        t = step()
        fn()
        return t
    eng.step = once


def kinds(e, kind, **match_):
    return [x for x in e.engagement.log if x['kind'] == kind and all(x.get(k) == v for k, v in match_.items())]


def reward_env(**kwargs):
    """A 2v2 whose missiles end when their target dies (no retarget): a missile just off the rail would otherwise
    reacquire its own shooter right behind it (Engagement._retarget) and add a friendly kill to the step."""
    e = env(**kwargs)
    e.engagement.retarget_dead = False
    return e


def missile_kill(eng, victim, m):
    """Missile ``m`` fuses on ``victim``, as Engagement._settle settles a fuse."""
    m.done, m.event = True, 'fuse'
    m.hist[2] = eng.time
    eng._kill(victim, m.shooter, 'missile', m)


class DefaultsTests(unittest.TestCase):
    """Keys absent and keys present as None give the same match, bit for bit."""

    NONE = dict(assist_rule=None, launch_reward=None, retarget_kill_reward=None, fuel=None)

    def digest(self, **extra):
        e = MatchEnv(dict(teams=TEAMS2, range_km=40., policy_ids=[0], time_limit_s=60., **extra), 11)
        obs = e.reset()
        out = [{a: wire.pack_obs(o, True) for a, o in obs.items()}]
        while not e.over and len(out) < 72:
            lab = e.scripted_actions()
            obs, r, d, i = e.step({a: lab[a] for a in e._observations})
            out.append(({a: wire.pack_obs(o, True) for a, o in obs.items()}, r, d, i))
        raw = [e._raw[a].own for a in sorted(e._raw)]
        self.assertTrue(all(o.fuel_kg is None and o.mass_kg is None and not o.bingo for o in raw))
        return hashlib.sha256(pickle.dumps((out, e.engagement.log, [repr(o) for o in raw]))).hexdigest()

    def test_env_absent_equals_none(self):
        self.assertEqual(self.digest(), self.digest(**self.NONE))

    def test_replay_absent_equals_none(self):
        lines = []
        for kw in ({}, dict(fuel=None, assist_rule=None)):
            eng = match.scenario(TEAMS2, 11, range_km=40.).engagement(replay=ReplayWriter(None), **kw)
            eng.run(until_s=20.)
            lines.append(eng.replay.lines)
        self.assertEqual(lines[0], lines[1])
        self.assertNotIn('fuel', lines[0][0])
        self.assertIn('"plane_columns":["id","x","y","z","vx","vy","vz","heading_deg","missiles","chaff","phase"]',
                      lines[0][0])

    def test_bad_values_fail(self):
        for bad in (dict(assist_rule='last_shot'), dict(launch_reward=True), dict(launch_reward='x'),
                    dict(retarget_kill_reward=float('nan')), dict(fuel=[]), dict(fuel=dict(load=1.)),
                    dict(fuel=dict(fraction=[0., 1.])), dict(fuel=dict(fraction=[.8, .5])),
                    dict(fuel=dict(fraction=[.5, 1.2])), dict(fuel=dict(bingo=1.)), dict(fuel=dict(bingo=True)),
                    dict(fuel=dict(tanks='all'))):
            with self.assertRaises(ValueError, msg=str(bad)):
                MatchEnv(bad, 0)


class AssistRuleTests(unittest.TestCase):
    """assist_rule 'first_shot': only a teammate who shot at the victim before the killing missile, once per pair."""

    def scene(self, rule):
        e = reward_env(**({} if rule is None else dict(assist_rule=rule)))
        eng = e.engagement
        p0, p1, p2, p3 = eng.planes
        e.step(quiet(e))
        eng.fire(p1, p2)                          # p1 shoots p2 first ...
        e.step(quiet(e))
        kill2 = eng.fire(p0, p2)                  # ... p0's later missile kills p2
        kill3 = eng.fire(p0, p3)                  # p0 shoots p3 first ...
        e.step(quiet(e))
        eng.fire(p1, p3)                          # ... and p1 only after it
        after_next_tick(eng, lambda: (missile_kill(eng, p2, kill2), missile_kill(eng, p3, kill3)))
        _, r, _, info = e.step(quiet(e))
        return e, r, info

    def test_first_shot_credits_only_the_earlier_shooter_once(self):
        e, r, info = self.scene('first_shot')
        eng = e.engagement
        self.assertEqual([(x['plane'], x['victim']) for x in kinds(e, 'assist')], [(1, 2)])
        self.assertEqual(info['events']['assist'], 1)
        self.assertAlmostEqual(r[1], .3);self.assertEqual(r[0], 2.)
        self.assertEqual((eng.planes[1].assists, eng.assisted), (1, {(1, 2)}))
        # once per (shooter, victim) in the match: the same pair is never credited again
        p0, p1, p2, p3 = eng.planes
        p2.alive = True;eng.live.append(p2)
        p2.missile_hist.append([1, eng.time-5., None])
        eng._kill(p2, p0, 'missile', None)
        self.assertEqual(len(kinds(e, 'assist')), 1)

    def test_default_rule_credits_every_teammate_in_the_window(self):
        e, r, info = self.scene(None)
        self.assertEqual([(x['plane'], x['victim']) for x in kinds(e, 'assist')], [(1, 2), (1, 3)])
        self.assertAlmostEqual(r[1], .6);self.assertEqual(info['events']['assist'], 2)
        self.assertEqual(e.engagement.assisted, set())


class LaunchRewardTests(unittest.TestCase):
    def launches(self, **kw):
        e = reward_env(**kw)
        eng = e.engagement
        p0, p1, p2, p3 = eng.planes
        after_next_tick(eng, lambda: (eng.fire(p0, p2), eng.fire(p0, p3), eng.fire(p1, p2)))
        lab = e.scripted_actions()
        _, r, _, info = e.step({a: act(e, a) if a in (0, 1) else lab[a] for a in e._observations})
        n = {s: len(kinds(e, 'launch', shooter=s)) for s in range(4)}
        return r, info, n

    def test_each_launch_of_a_policy_aircraft_only(self):
        r, info, n = self.launches(controlled=None, policy_ids=[0], launch_reward=-.05)
        self.assertEqual(n[0], 2);self.assertGreaterEqual(n[1], 1)
        self.assertEqual(set(r), {0});self.assertAlmostEqual(r[0], -.1)
        self.assertNotIn('late_rewards', info)                  # the scripted p1 gets nothing
        r, info, n = self.launches(launch_reward=-.05)          # every slot a policy aircraft
        for s in range(4):
            self.assertAlmostEqual(r[s], -.05*n[s])
        r, info, n = self.launches(controlled=None, policy_ids=[0])
        self.assertEqual(r, {0: 0.})                             # absent: no launch reward


class RetargetKillTests(unittest.TestCase):
    """retarget_kill_reward: a kill by a missile that took a new target after its own died."""

    def retargeted(self, e, shooter_down=False):
        eng = e.engagement
        p0, p1, p2, p3 = eng.planes
        e.step(quiet(e))
        m = eng.fire(p0, p2)
        if shooter_down:
            eng._kill(p0, None, 'crash', None)
        eng._kill(p2, None, 'crash', None)
        # p3 straight ahead of the missile, inside its seeker cone
        u = [x/math.sqrt(sum(v*v for v in m.vel_enu)) for x in m.vel_enu]
        p3.flight.state = replace(p3.flight.state, position=tuple(a+2000.*b for a, b in zip(m.pos_enu, u)))
        eng._retarget(m)
        self.assertIs(m.target, p3);self.assertIn(m.uid, eng.retargeted_uids)
        e.observe()
        after_next_tick(eng, lambda: missile_kill(eng, p3, m))
        return e.step(quiet(e))

    def test_retarget_kill_credits_the_setting_also_late(self):
        _, r, _, info = self.retargeted(reward_env(retarget_kill_reward=.5))
        self.assertEqual(r[0], .5);self.assertEqual(info['tallies'][0], [1, 0])
        self.assertEqual(info['events']['kill'], 1)
        _, r, _, info = self.retargeted(reward_env(retarget_kill_reward=.5), shooter_down=True)
        self.assertNotIn(0, r);self.assertEqual(info['late_rewards'], {0: .5});self.assertEqual(info['tallies'][0], [1, 0])
        _, r, _, info = self.retargeted(reward_env())             # absent: a full kill
        self.assertEqual(r[0], 1.)
        _, r, _, info = self.retargeted(reward_env(), shooter_down=True)
        self.assertEqual(info['late_rewards'], {0: 1.})

    def test_a_direct_kill_is_still_worth_one(self):
        e = reward_env(retarget_kill_reward=.5)
        eng = e.engagement
        p0, p1, p2, p3 = eng.planes
        m = eng.fire(p0, p2)
        after_next_tick(eng, lambda: missile_kill(eng, p2, m))
        _, r, _, info = e.step(quiet(e))
        self.assertEqual(r[0], 1.);self.assertEqual(eng.retargeted_uids, set())


class FuelTests(unittest.TestCase):
    def test_load_mass_burn_and_raw_observation(self):
        e = env(teams=TEAMS1, fuel={})
        eng = e.engagement
        header = eng._header()
        self.assertEqual(header['plane_columns'][-1], 'fuel_kg')
        for p in eng.planes:
            tank, data = p.flight.fuel, fuel_data(p.aircraft)
            self.assertTrue(FUEL['fraction'][0]*data.max_kg <= tank.initial_kg <= data.max_kg)
            missile = eng.missile_mass(p.missile_id)
            self.assertNotEqual(missile, MISSILE_MASS_KG)             # the missile_sim launch mass
            self.assertAlmostEqual(p.flight.model.mass, data.empty_kg+tank.initial_kg+p.missiles*missile)
            self.assertEqual(header['fuel']['loads'][p.ident], [round(tank.initial_kg, 1), data.max_kg])
            self.assertEqual(header['planes'][p.ident]['mass_kg'], round(p.flight.model.mass))
        p0 = eng.planes[0]
        tank = p0.flight.fuel
        for _ in range(12):
            _, _, _, info = e.step(quiet(e))
        self.assertEqual(info['events']['flameout'], 0)
        self.assertLess(tank.kg, tank.initial_kg)
        self.assertLess(p0.flight.model.mass, tank.empty_kg+tank.initial_kg+tank.payload_kg)
        self.assertLess(abs(p0.flight.model.mass-tank.mass_kg), FUEL_MASS_STEP_KG)
        own = e._raw[0].own
        self.assertEqual((own.fuel_kg, own.fuel_fraction, own.mass_kg, own.bingo),
                         (tank.kg, tank.kg/tank.initial_kg, tank.mass_kg, False))
        self.assertEqual(len(e._observations[0].own), len(env(teams=TEAMS1)._observations[0].own))   # no new widths
        # a launch takes the missile's mass away
        before = tank.payload_kg
        eng.fire(p0, eng.planes[1])
        self.assertAlmostEqual(before-tank.payload_kg, eng.missile_mass(p0.missile_id))
        eng.replay = ReplayWriter(None)
        eng._frame()
        self.assertEqual(eval(eng.replay.lines[-1].replace('null', 'None'))['planes'][0][-1], round(tank.kg, 1))

    def test_internal_tanks_and_fraction(self):
        e = env(teams=TEAMS1, fuel=dict(tanks='internal', fraction=[1., 1.]))
        ge = e.engagement.planes[0]
        self.assertEqual(ge.flight.fuel.initial_kg, fuel_data('f_15c_golden_eagle').internal_kg)
        self.assertLess(fuel_data('f_15c_golden_eagle').internal_kg, fuel_data('f_15c_golden_eagle').max_kg)

    def flown(self, aircraft, throttle, fuel_kg=5000., seconds=4.):
        model = aircraft_model(aircraft, mass_kg=20000.)
        v = .9*atmosphere(8000.)[1]
        a = Aircraft(model, (0., 0., 8000.), (0., v, 0.))
        a.fuel = FuelTank(fuel_data(aircraft), fuel_kg, 2000.)
        a.fuel.sync(model, force=True)
        a.command = FlightCommand(heading_deg=0., altitude_m=8000., throttle_percent=throttle)
        for _ in range(round(seconds*48)):
            a.step()
        return a

    def test_afterburner_burns_faster_than_military_power(self):
        for aircraft in ('f_15c_golden_eagle', 'su_30sm2', 'f_16c_block_50'):
            mil, ab = (5000.-self.flown(aircraft, t).fuel.kg for t in (100., 110.))
            self.assertGreater(mil, 0., aircraft);self.assertGreater(ab, 1.5*mil, aircraft)
            idle = 5000.-self.flown(aircraft, 0.).fuel.kg
            self.assertLess(idle, mil, aircraft)
        a = self.flown('f_15c_golden_eagle', 110.)
        self.assertLess(a.model.mass, 2000.+fuel_data('f_15c_golden_eagle').empty_kg+5000.)

    def test_empty_tanks_give_no_thrust(self):
        a = self.flown('f_15c_golden_eagle', 110., fuel_kg=1., seconds=1.)
        self.assertTrue(a.fuel.out);self.assertEqual(a.fuel.kg, 0.);self.assertIs(type(a.model), Flameout)
        self.assertEqual(a.model.forces_at_aoa(8000., 250., 3., 110.)[0], 0.)
        lit = aircraft_model('f_15c_golden_eagle', mass_kg=20000.)
        self.assertGreater(lit.forces_at_aoa(8000., 250., 3., 110.)[0], 0.)
        self.assertEqual(a.model.forces_at_aoa(8000., 250., 3.)[1:], lit.forces_at_aoa(8000., 250., 3.)[1:])
        # it glides on (no thrust: the speed bleeds) and still turns
        speed = a.speed
        a.command = FlightCommand(heading_deg=90., altitude_m=8000., throttle_percent=110.)
        for _ in range(5*48):
            a.step()
        self.assertTrue(a.alive);self.assertLess(a.speed, speed-10.);self.assertGreater(a.attitude()[0], 20.)

    def test_flameout_event_once_and_the_plane_flies_on(self):
        e = env(teams=TEAMS1, fuel={})
        p0 = e.engagement.planes[0]
        p0.flight.fuel.kg = .01
        _, _, _, info = e.step(quiet(e))
        self.assertEqual(info['events']['flameout'], 1)
        self.assertEqual([x['plane'] for x in kinds(e, 'flameout')], [0])
        for _ in range(4):
            _, _, _, info = e.step(quiet(e))
            self.assertEqual(info['events']['flameout'], 0)
        self.assertTrue(p0.alive);self.assertIs(type(p0.flight.model), Flameout)
        self.assertEqual((e._raw[0].own.fuel_kg, e._raw[0].own.fuel_fraction), (0., 0.))

    def test_bingo_flag(self):
        e = env(teams=TEAMS1, fuel=dict(bingo=.5))
        tank = e.engagement.planes[0].flight.fuel
        tank.kg = .4*tank.initial_kg
        e.observe()
        self.assertTrue(e._raw[0].own.bingo)
        tank.kg = .6*tank.initial_kg
        e.observe()
        self.assertFalse(e._raw[0].own.bingo)

    def test_rearm_refuels_to_the_initial_load(self):
        e = env(teams=TEAMS1, range_km=60., fuel={}, airfield={}, time_limit_s=600.)
        eng = e.engagement
        ge = eng.planes[0]
        tank = ge.flight.fuel
        e.step(quiet(e, {0: 10}))
        eng.fire(ge, eng.planes[1])
        tank.kg = 1e-9
        tank.burn(ge.flight.model, ge.flight.state)          # dry: flamed out
        self.assertIs(type(ge.flight.model), Flameout)
        eng._land(ge)
        for _ in range(60):                                   # 25 s with go-home chosen: the turnaround passes
            e.step(quiet(e, {0: 10}))
        rearm = kinds(e, 'rearm', plane=0)
        self.assertEqual(len(rearm), 1);self.assertEqual(rearm[0]['fuel_kg'], round(tank.initial_kg, 1))
        self.assertEqual((tank.kg, tank.out), (tank.initial_kg, False))
        self.assertIs(type(ge.flight.model), ManeuverModel)
        full = tank.empty_kg+tank.initial_kg+ge.spec.missiles*eng.missile_mass(ge.missile_id)
        self.assertEqual((tank.payload_kg+tank.empty_kg+tank.kg, ge.flight.model.mass), (full, full))
        e.step(quiet(e))                                      # another maneuver: take off with the same tank
        self.assertFalse(ge.grounded);self.assertIs(ge.flight.fuel, tank)
        for _ in range(4):
            e.step(quiet(e))
        self.assertLess(tank.kg, tank.initial_kg)

    def test_snapshot_restores_the_tank(self):
        e = env(teams=TEAMS1, fuel={}, execution=dict(ZERO, reject_p=.05, error_deg=5.))
        for _ in range(3):
            e.step(quiet(e))
        e.engagement.planes[1].flight.fuel.kg = .05
        snap = e.snapshot()

        def play():
            out = [e.step(quiet(e)) for _ in range(6)]
            return pickle.dumps(([(wire_obs(o), r, d, i) for o, r, d, i in out],
                                 [(p.flight.fuel.kg, p.flight.model.mass, type(p.flight.model).__name__)
                                  for p in e.engagement.planes]))
        first = play()
        e.restore(snap)
        self.assertEqual(first, play())


def wire_obs(obs):
    return {a: wire.pack_obs(o, True) for a, o in obs.items()}


class ApproachTests(unittest.TestCase):
    """Default airfield: the policy path flying go-home at full throttle from 40 km out (8 km high, Mach 0.9; the
    measurement of docs/airfield_rearm_spec.md section 9) lands 76-86 s after it crosses the approach radius (the
    scripts, slower at the radius, 90-103 s)."""

    def test_about_90_s_from_the_approach_radius(self):
        e = MatchEnv(dict(teams=TEAMS1, range_km=60., controlled='all', time_limit_s=900., airfield={}), 1)
        e.reset()
        eng = e.engagement
        radius = AIRFIELD['approach_m']
        v = .9*atmosphere(8000.)[1]
        for p, side in zip(eng.planes, (1., -1.)):   # beside their airfields, far from each other, flying at it
            hx, hy = p.airfield_xy
            pos, vel = (hx+side*40000., hy, 8000.), (-side*v, 0., 0.)
            p.flight.state = replace(p.flight.state, position=pos, velocity=vel)
            p.flight._ring.clear();p.flight._ring.append((pos, vel));p.flight._first = p.flight.tick
        e.observe()
        inside = {}
        while not e.over and not all(p.grounded for p in eng.planes) and eng.time < 260.:
            e.step(quiet(e, {0: 10, 1: 10}))
            for p in eng.planes:
                x, y, _ = p.own.position
                if p.ident not in inside and math.hypot(x-p.airfield_xy[0], y-p.airfield_xy[1]) <= radius:
                    inside[p.ident] = eng.time
        for p in eng.planes:
            land = kinds(e, 'landing', plane=p.ident)
            self.assertTrue(land, p.aircraft)
            self.assertTrue(70. <= land[0]['t']-inside[p.ident] <= 100., (p.aircraft, land[0]['t']-inside[p.ident]))


if __name__ == '__main__':
    unittest.main()


class FuelOwnVectorTests(unittest.TestCase):
    def test_fuel_share_fills_the_rne_slot(self):
        from wt_overlay.rl_env import MatchEnv
        from wt_overlay.rl_observation import OWN_FIELDS_EGO, OWN_FIELDS
        for frame, fields in (("egocentric", OWN_FIELDS_EGO), ("world", OWN_FIELDS)):
            i, n = fields.index("rne"), len(fields)                            # values first, then validity flags
            off = MatchEnv(dict(model_path="data/match/top_tier_s1_far.json", team_size=1, time_limit_s=30.,
                                observation_frame=frame), 5)
            on = MatchEnv(dict(model_path="data/match/top_tier_s1_far.json", team_size=1, time_limit_s=30.,
                               observation_frame=frame, fuel={"tanks": "internal"}), 5)
            vo, vn = off.reset()[0].own, on.reset()[0].own
            self.assertEqual((vo[i], vo[n+i]), (0., 0.))                     # fuel off: slot stays invalid
            self.assertEqual(vn[n+i], 1.)
            self.assertAlmostEqual(vn[i], 1., places=3)                         # full initial load at spawn

