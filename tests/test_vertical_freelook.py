"""Opt-in vertical_mode / view_model / entity_memory_s / maw_entities and the free-look hold (rl_training_spec 3-4)."""
from __future__ import annotations

from dataclasses import replace
import math
from types import SimpleNamespace
import unittest

from rl import spec, wire
from wt_overlay.engagement import Action, Sighting
from wt_overlay.flight import FlightCommand, KeyboardCommand
from wt_overlay.intent import (Intent, IntentExecutor, angle_option, from_flight_action,
                               SCRIPT_ALT_DEADBAND_M, SCRIPT_ALT_STEEP_M)
from wt_overlay.rl_env import MatchEnv, DT_STEP
from wt_overlay.rl_observation import (Entity, EntityList, OWN_FIELDS, carried, masks_for, own_vector, select_entities,
                                        turn_track)
from wt_overlay.sensors import RadarContact

ZERO = dict(reject_p=0., error_deg=0., delay={p: dict(median_range=(.001, .001), sigma=0., bounds=(0., 0.))
                                             for p in ('follow', 'autonomous')})
TEAMS = [[dict(aircraft='f_15c_golden_eagle', archetype='middle', skill='top', altitude_m=8000., mach=1.)],
         [dict(aircraft='su_30sm2', archetype='middle', skill='top', altitude_m=8000., mach=1.)]]
DEFAULTS = dict(vertical_mode='altitude', view_model='full', entity_memory_s=None, maw_entities=True)


def env(seed=5, execution=None, **kwargs):
    config = dict(teams=TEAMS, range_km=40., controlled='all', execution=dict(ZERO, **(execution or {})),
                  time_limit_s=120.)
    config.update(kwargs)
    e = MatchEnv(config, seed)
    e.reset()
    return e


def neutral(e, **heads):
    """Every living agent: no reference / target / object, keep heading, the given heads."""
    acts = {}
    for aid in e._observations:
        n = len(e._entities[aid])
        a = dict(maneuver_ref=n, target=n, view_object=n, maneuver=0, vertical=0, speed=0, chaff=0, radar_mode=0,
                 antenna=2, weapon=0, view_mode=0, look_az=0, look_el=1, kb_roll=1, kb_pitch=1)
        a.update(heads)
        acts[aid] = a
    return acts


def stable(result):
    obs, r, d, i = result
    return ({a: wire.pack_obs(o, True) for a, o in obs.items()}, r, d, i)


def contact(track_id, position, velocity, t, age=0.):
    return RadarContact('track', track_id, 30000., 0., 0., 0., 0., -300., tuple(position), tuple(velocity), t-age, age,
                        False)


class DefaultsTests(unittest.TestCase):
    def test_options_present_but_default_change_nothing(self):
        runs = []
        for execution in (None, DEFAULTS):
            e = env(seed=7, execution=execution, controlled=[0])
            out = [stable((e._observations, {}, {}, {}))]
            for _ in range(30):
                out.append(stable(e.step(e.scripted_actions())))
            runs.append(out)
            self.assertEqual(type(e._entities[0]), list)   # no EntityList in the default vertical mode
        self.assertEqual(runs[0], runs[1])

    def test_bad_option_values_fail(self):
        for bad in (dict(vertical_mode='alt'), dict(view_model='object'), dict(entity_memory_s=0.),
                    dict(entity_memory_s=True), dict(maw_entities=1)):
            with self.assertRaises(ValueError):
                IntentExecutor(1, **bad)


class AngleModeTests(unittest.TestCase):
    def test_each_option_climbs_or_descends_at_its_angle_and_level_holds(self):
        expected = {1: 25., 2: 10., 3: -10., 4: -30.}
        for v in range(5):
            e = env(execution=dict(vertical_mode='angle'), range_km=100.)
            start = {p.ident: p.own.position[2] for p in e.engagement.planes}
            for _ in range(24):   # 10 s
                e.step(neutral(e, vertical=v))
            for p in e.engagement.planes:
                climb = p.own.position[2]-start[p.ident]
                gamma = math.degrees(math.asin(p.own.velocity[2]/math.sqrt(sum(x*x for x in p.own.velocity))))
                if v == 0:
                    self.assertLess(abs(climb), 30., p.aircraft)
                else:
                    self.assertGreater(climb*expected[v], 0., (v, p.aircraft))
                    self.assertAlmostEqual(gamma, expected[v], delta=2., msg=(v, p.aircraft))
                    self.assertIsInstance(p.flight.command, FlightCommand)
                    self.assertEqual(p.flight.command.climb_deg, expected[v])
        # Level captures the altitude when it is selected after a climb, and holds it.
        e = env(execution=dict(vertical_mode='angle'), range_km=100.)
        for _ in range(10):
            e.step(neutral(e, vertical=2))
        e.step(neutral(e, vertical=0))
        captured = {p.ident: e.executors[p.ident].aim_altitude for p in e.engagement.planes}
        self.assertTrue(all(c > 8100. for c in captured.values()))
        lows, highs = {i: math.inf for i in captured}, {i: -math.inf for i in captured}
        for k in range(36):
            e.step(neutral(e, vertical=0))   # publishing the same option keeps it (no recapture)
            for p in e.engagement.planes:
                self.assertEqual(e.executors[p.ident].aim_altitude, captured[p.ident])
                if k >= 12:
                    lows[p.ident] = min(lows[p.ident], p.own.position[2])
                    highs[p.ident] = max(highs[p.ident], p.own.position[2])
        for i, c in captured.items():
            self.assertLess(highs[i]-c, 150., e.engagement.planes[i].aircraft)
            self.assertLess(c-lows[i], 50., e.engagement.planes[i].aircraft)

    def test_climb_options_stop_climbing_below_250_mps(self):
        e = env(execution=dict(vertical_mode='angle'), range_km=100.)
        for _ in range(5):   # past the 2 s hold of the first publication
            e.step(neutral(e, vertical=1))
        command = e.engagement.planes[0].flight.command
        self.assertEqual((command.climb_deg, command.min_speed_mps, command.altitude_m), (25., 250., None))
        e.step(neutral(e, vertical=3))
        command = e.engagement.planes[0].flight.command
        self.assertEqual((command.climb_deg, command.min_speed_mps), (-10., None))

    def test_script_altitude_target_mapping(self):
        own = SimpleNamespace(altitude_m=5000., velocity=(0., 250., 0.))
        table = [(5000.+SCRIPT_ALT_DEADBAND_M-1., 0), (5000.-SCRIPT_ALT_DEADBAND_M+1., 0), (5400., 2),
                 (5000.+SCRIPT_ALT_STEEP_M+1., 1), (4600., 3), (5000.-SCRIPT_ALT_STEEP_M-1., 4), (None, 0)]
        for target, option in table:
            self.assertEqual(angle_option(FlightCommand(heading_deg=0., altitude_m=target), own), option, target)
        for gamma, option in ((-40., 4), (-15., 3), (-2., 0), (8., 2), (20., 1)):
            g = math.radians(gamma)
            f = FlightCommand(direction=(0., math.cos(g), math.sin(g)), floor_m=None)
            self.assertEqual(angle_option(f, own), option, gamma)
            self.assertEqual(angle_option(FlightCommand(climb_deg=gamma, floor_m=None), own), option, gamma)
        # A dive levels off above its floor: the deadband, 2 s of the sink rate and a 3 g pull-out (here 755 m).
        dive = FlightCommand(direction=(0., .7, -.7), floor_m=2500.)
        self.assertEqual(angle_option(dive, SimpleNamespace(altitude_m=3300., velocity=(0., 200., -100.))), 4)
        self.assertEqual(angle_option(dive, SimpleNamespace(altitude_m=3200., velocity=(0., 200., -100.))), 0)

    def test_from_flight_action_takes_the_mode_from_the_executor(self):
        old, new = env(), env(execution=dict(vertical_mode='angle'))
        self.assertIsInstance(new._entities[0], EntityList)
        self.assertEqual(new._entities[0].vertical_mode, 'angle')
        def at(obs, altitude):
            return replace(obs, own=replace(obs.own, altitude_m=altitude, position=(*obs.own.position[:2], altitude)))
        raw, high = at(old._raw[0], 150.), at(old._raw[0], 8000.)
        popup = Action(FlightCommand(heading_deg=0., altitude_m=3000.), None, None, 0)
        level = Action(FlightCommand(heading_deg=0., altitude_m=5000.), None, None, 0)
        # Old quantisation: a 3 km pop-up is the deck (100 m), a 5 km level from 8 km is the 8 km step.
        self.assertEqual(from_flight_action(popup, raw, old._entities[0]).vertical, 3)
        self.assertEqual(from_flight_action(level, high, old._entities[0]).vertical, 2)
        self.assertEqual(from_flight_action(popup, raw, new._entities[0]).vertical, 1)
        self.assertEqual(from_flight_action(level, high, new._entities[0]).vertical, 4)
        self.assertEqual(from_flight_action(level, high, old._entities[0], vertical_mode='angle').vertical, 4)

    def test_crawler_pops_up_to_its_popup_altitude(self):
        teams = [[dict(aircraft='su_30sm2', archetype='crawler', skill='top', altitude_m=150., mach=.9)],
                 [dict(aircraft='f_15c_golden_eagle', archetype='middle', skill='top', altitude_m=8000., mach=1.)]]
        peaks = {}
        for mode in ('altitude', 'angle'):
            e = env(teams=teams, range_km=60., controlled=[1], execution=dict(vertical_mode=mode), seed=3)
            pilot = e.pilots[0]
            pilot.p = replace(pilot.p, popup_range_m=80000., popup_alt_m=2000., second_round_m=1000., p_react=0.,
                              p_lock_react=0.)
            peak = 0.
            for _ in range(96):   # 40 s; the enemy flies straight and never shoots
                e.step(neutral(e))
                peak = max(peak, e.engagement.planes[0].own.position[2])
            self.assertEqual(pilot.phase, 'popup')
            peaks[mode] = peak
        self.assertLess(peaks['altitude'], 600.)                 # old: 2000 m quantised to the 100 m deck
        self.assertGreater(peaks['angle'], 2000.-SCRIPT_ALT_DEADBAND_M-100.)
        self.assertLess(peaks['angle'], 2000.+SCRIPT_ALT_DEADBAND_M)


class FreeLookHoldTests(unittest.TestCase):
    def test_hold_survives_free_look_and_the_exit_keeps_the_held_values(self):
        raw = env()._raw[0]
        x = IntentExecutor(1, **ZERO)
        x.publish(Intent(maneuver=2, vertical=1), raw)
        self.assertEqual(x.hold_until, 2.)
        x.publish(Intent(view_mode=2, kb_pitch=2), replace(raw, time_s=.4))     # entering free look: still held
        self.assertTrue(x.held(.8))
        self.assertEqual(x.hold_until, 2.)
        with self.assertRaises(ValueError):                                    # the exit cannot bring a new manoeuvre
            x.publish(Intent(maneuver=6), replace(raw, time_s=.8))
        x.publish(Intent(maneuver=2, vertical=1), replace(raw, time_s=.8))      # the held values: fine, hold unchanged
        self.assertEqual((x.hold_until, x.aim_heads), (2., (None, 2, 1)))
        self.assertTrue(x.pending[-1][4])                                      # it is the exit
        x.publish(Intent(view_mode=2), replace(raw, time_s=1.2))
        x.publish(Intent(maneuver=6), replace(raw, time_s=2.4))                # hold over: a fresh exit restarts it
        self.assertAlmostEqual(x.hold_until, 4.4)

    def test_leaving_free_look_is_never_rejected(self):
        raw = env()._raw[0]
        x = IntentExecutor(3, reject_p=0.)
        x.publish(Intent(view_mode=2, kb_pitch=0), raw)
        x.reject_p = 1.
        x.publish(Intent(maneuver=6), replace(raw, time_s=1.))                 # the exit always goes through
        self.assertEqual((x.rejected, x.unheeded_at, x.published.view_mode, len(x.pending)), (0, None, 0, 2))
        x.publish(Intent(maneuver=4), replace(raw, time_s=3.5))                # a later change can still be rejected
        self.assertEqual((x.rejected, x.unheeded_at), (1, 3.5))

    def test_overtaken_fire_request_still_fires(self):
        """A fire press with a longer delay than the following 'no fire' publication is not dropped."""
        from types import SimpleNamespace
        e = env()
        for _ in range(400):                                                   # fly until a launchable track exists
            if any(en.track_id is not None and en.launch_ok for en in e._entities[0]):
                break
            e.step(e.scripted_actions())
        else:
            self.skipTest('no launchable radar track within 400 steps')
        raw = e._raw[0]; plane = e.engagement.planes[0]; ents = e._entities[0]
        x = IntentExecutor(3, reject_p=0.)
        key = next(en.key for en in ents if en.track_id is not None and en.launch_ok)
        x.publish(Intent(target=key), raw)
        fired = []
        eng = SimpleNamespace(time=0., apply=lambda p, a: fired.append(a))
        x.advance(eng, plane, raw, ents)                                       # the first publication executes
        x.publish(Intent(target=key, weapon=1), replace(raw, time_s=0.5))      # fire: seq 2
        x.publish(Intent(target=key), replace(raw, time_s=1.))                 # no fire: seq 3
        x.pending = [(2.5, 2, Intent(target=key, weapon=1), 0., False), (1.5, 3, Intent(target=key), 0., False)]
        eng.time = 1.6; x.advance(eng, plane, raw, ents)                       # seq 3 executes first
        self.assertEqual(x.applied_sequence, 3)
        eng.time = 2.6; x.advance(eng, plane, raw, ents)                       # the overtaken fire still happens
        self.assertTrue(any(a.fire is not None for a in fired), [a.fire for a in fired])

    def test_flicker_through_free_look_does_not_unlock_the_manoeuvre_heads(self):
        e = env()
        e.step(neutral(e, maneuver=6))                                         # published at t=0: held until 2 s
        e.step(neutral(e, view_mode=2, look_az=4))
        for aid, o in e._observations.items():
            ex = e.executors[aid]
            self.assertTrue(ex.held(e.engagement.time))
            n = len(e._entities[aid])
            for nulled in (0, 1):
                self.assertEqual(o.masks['maneuver'][0][nulled], [i == 6 for i in range(11)])
                self.assertEqual(sum(o.masks['vertical'][0][nulled]), 1)
            self.assertEqual(o.masks['maneuver_ref'][0], [i == n for i in range(n+1)])
        with self.assertRaises(ValueError):
            e.step(neutral(e, maneuver=4))                                    # masked: the env refuses it
        before = e.executors[0].hold_until
        e.step(neutral(e, maneuver=6))                                         # exit with the held values
        self.assertEqual(e.executors[0].hold_until, before)
        self.assertIsInstance(e.engagement.planes[0].flight.command, FlightCommand)

    def test_held_reference_stays_an_entity_through_free_look(self):
        raw = env()._raw[0]
        x = IntentExecutor(1, **ZERO)
        memory = {}
        seen = replace(raw, radar=(contact(4, (0., 30000., 8000.), (0., -250., 0.), 0.),), radar_missiles=None)
        select_entities(seen, x, memory)
        x.publish(Intent(maneuver_ref=('radar', 4), maneuver=1), seen)
        x.publish(Intent(view_mode=2), replace(seen, time_s=.4))
        self.assertIsNone(x.published.maneuver_ref)
        lost = replace(raw, time_s=.8, radar=(), radar_missiles=None)
        ents, _ = select_entities(lost, x, memory)                              # the sensors lost it: carried
        keys = [e.key for e in ents]
        self.assertIn(('radar', 4), keys)
        masks = masks_for(lost, ents, x, -1e9)
        self.assertEqual(masks['maneuver_ref'][0], [i == keys.index(('radar', 4)) for i in range(len(ents)+1)])
        late = replace(raw, time_s=2.5, radar=(), radar_missiles=None)       # hold over: no longer kept
        self.assertNotIn(('radar', 4), [e.key for e in select_entities(late, x, memory)[0]])


class ObjectOnlyTests(unittest.TestCase):
    def test_masks_and_look_object_keeps_flying_the_mouse_aim_intent(self):
        e = env(execution=dict(view_model='object_only'))
        for _ in range(6):   # past the hold of the first publication
            e.step(neutral(e))
        e.step(neutral(e, maneuver=6))
        o = e._observations[0]
        n = len(e._entities[0])
        self.assertEqual(o.masks['view_mode'][2], False)
        for row in range(3):
            self.assertEqual(o.masks['kb_roll'][row], [False, True, False])
            self.assertEqual(o.masks['kb_pitch'][row], [False, True, False])
            self.assertEqual(o.masks['look_az'][row], [i == 0 for i in range(8)])
            self.assertEqual(o.masks['look_el'][row], [False, True, False])
        self.assertEqual(o.masks['maneuver'][1][1], [i == 6 for i in range(11)])
        self.assertEqual(o.masks['maneuver_ref'][1], [i == n for i in range(n+1)])
        self.assertEqual(spec.empty_selectable_rows(n, o.masks), [])
        self.assertTrue(o.masks['view_mode'][1])
        obj = next(i for i, x in enumerate(e._entities[0]) if x.bearing is not None)
        look = neutral(e, maneuver=6, view_mode=1)
        look[0]['view_object'] = obj
        heading = e.executors[0].aim_heading
        e.step(look)
        ex, plane = e.executors[0], e.engagement.planes[0]
        self.assertEqual(ex.executed.view_mode, 1)
        self.assertIsInstance(plane.flight.command, FlightCommand)
        self.assertEqual(ex.aim_heads, (None, 6, 0))
        self.assertEqual(plane.camera.mode, 1)
        with self.assertRaises(ValueError):    # look-object cannot change the manoeuvre
            ex.publish(replace(ex.published, maneuver=3), replace(e._raw[0], time_s=e.engagement.time+5.))
        self.assertIsNotNone(heading)

    def test_full_model_still_flies_the_keyboard_in_free_look(self):
        e = env()
        e.step(neutral(e, view_mode=2, kb_pitch=2))
        self.assertIsInstance(e.engagement.planes[0].flight.command, KeyboardCommand)


class EntityMemoryTests(unittest.TestCase):
    def circling(self, t, centre=(5000., 20000., 8000.), radius=3000., speed=250.):
        w = speed/radius
        a = w*t
        pos = (centre[0]+radius*math.sin(a), centre[1]+radius*math.cos(a)-radius, centre[2])
        vel = (speed*math.cos(a), -speed*math.sin(a), 0.)
        return pos, vel

    def test_turn_aware_extrapolation_beats_a_straight_line_on_a_circling_target(self):
        raw = env()._raw[0]
        x = IntentExecutor(1, **ZERO)
        x.published = Intent(maneuver_ref=('radar', 7))
        memory = {}
        for k in range(8):   # observed every 0.42 s for 3 s
            t = k*DT_STEP
            pos, vel = self.circling(t)
            obs = replace(raw, time_s=t, radar=(contact(7, pos, vel, t),), radar_missiles=None)
            select_entities(obs, x, memory)
        entry = memory[('radar', 7)]
        self.assertAlmostEqual(entry[2][2], math.degrees(250./3000.), delta=.2)   # clockwise (compass) turn
        last = 7*DT_STEP
        for lost in (4., 10.):
            t = last+lost
            ents, _ = select_entities(replace(raw, time_s=t, radar=(), radar_missiles=None), x, memory)
            e = next(e for e in ents if e.key == ('radar', 7))
            self.assertTrue(e.extrapolated)
            self.assertIsNone(e.track_id)
            truth = self.circling(t)[0]
            straight = tuple(p+v*lost for p, v in zip(entry[0].position, entry[0].velocity))
            self.assertLess(math.dist(e.position, truth), .1*math.dist(straight, truth), lost)
        # A straight flyer is carried exactly as before (no turn estimate).
        straight_entry = (replace(entry[0], velocity=(0., 250., 0.)), entry[1], (0., 0., 0.))
        moved = carried(straight_entry, entry[1]+5., raw.own)
        self.assertEqual(moved.position, tuple(p+v*5. for p, v in zip(entry[0].position, (0., 250., 0.))))
        self.assertIsNone(turn_track(None, Entity(('rwr', 1), 'rwr', 10.), 0.))

    def test_entity_memory_keeps_a_lost_enemy_for_n_seconds(self):
        raw = env()._raw[0]
        for span in (None, 5.):
            x = IntentExecutor(1, entity_memory_s=span, **ZERO)
            memory = {}
            pos, vel = (0., 30000., 8000.), (0., -250., 0.)
            for k in range(3):
                t = k*DT_STEP
                obs = replace(raw, time_s=t, radar=(contact(9, (pos[0], pos[1]-250.*t, pos[2]), vel, t),),
                              radar_missiles=None)
                select_entities(obs, x, memory)
            seen = 2*DT_STEP
            kept = []
            for dt in (1., 4.9, 5.2):
                obs = replace(raw, time_s=seen+dt, radar=(), radar_missiles=None)
                ents, dropped = select_entities(obs, x, memory)
                mem = [e for e in ents if e.key == ('radar', 9)]
                kept.append(bool(mem))
                if mem:
                    e = mem[0]
                    self.assertTrue(e.extrapolated and e.memory)
                    self.assertEqual((e.track_id, e.launch_ok, e.tracking), (None, False, False))
                    self.assertAlmostEqual(e.position[1], 30000.-250.*(seen+dt), delta=1e-6)
                    masks = masks_for(obs, ents, x, -1e9)
                    i = ents.index(e)
                    self.assertFalse(masks['target'][i])
                    self.assertEqual(masks['weapon'][i], [True, False])
                    self.assertTrue(masks['maneuver_ref'][0][i])
                self.assertEqual(dropped, 0)
            self.assertEqual(kept, [False, False, False] if span is None else [True, True, False])

    def test_memory_entities_never_displace_observed_ones(self):
        raw = env()._raw[0]
        x = IntentExecutor(1, entity_memory_s=30., **ZERO)
        memory = {}
        select_entities(replace(raw, time_s=0., radar=(contact(9, (0., 30000., 8000.), (0., -250., 0.), 0.),),
                                radar_missiles=None), x, memory)
        cues = tuple(Sighting('aircraft', i, float(i), 0., None, 1.) for i in range(80))
        ents, dropped = select_entities(replace(raw, time_s=1., radar=(), radar_missiles=None, visual=cues), x, memory)
        self.assertEqual((len(ents), dropped), (64, 16 + len(raw_rest(raw))))
        self.assertFalse(any(e.memory for e in ents))
        ents, dropped = select_entities(replace(raw, time_s=2., radar=(), radar_missiles=None, visual=cues[:10]), x,
                                        memory)
        self.assertTrue(any(e.memory and e.key == ('radar', 9) for e in ents))
        self.assertEqual(dropped, 0)


def raw_rest(raw):
    """Entities of ``raw`` other than its radar contacts and sightings (RWR, marks, ...)."""
    from wt_overlay.rl_observation import raw_entities
    return [e for e in raw_entities(replace(raw, radar=(), radar_missiles=None, visual=()))]


class MawTests(unittest.TestCase):
    def test_maw_entities_off_removes_maw_from_entities_flag_and_hold_release(self):
        e = env()
        raw, plane = e._raw[0], e.engagement.planes[0]
        maw = replace(raw, maw=(Sighting('maw', 3, 10., 0., None, raw.time_s),), own=replace(raw.own, has_maw=True))
        for flag in (True, False):
            x = IntentExecutor(1, maw_entities=flag, **ZERO)
            x.publish(Intent(maneuver=2), raw)
            ents, _ = select_entities(maw, x, {})
            self.assertEqual(any(en.kind == 'maw' for en in ents), flag)
            own = own_vector(maw, plane, x)
            self.assertEqual(own[OWN_FIELDS.index('maw')], 1. if flag else 0.)
            x.notice(replace(maw, time_s=.5))
            self.assertEqual(x.held(.5), not flag)


if __name__ == '__main__':
    unittest.main()
