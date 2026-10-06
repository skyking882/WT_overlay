"""Engagement manager, scripted pilots and match generator (docs/engagement_spec.md)."""
import dataclasses
import json
import math
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
MISSILE_SIM = ROOT.parent/"missle_sim"
HAVE_MISSILE_SIM = (MISSILE_SIM/"src"/"aim120_model"/"surface_runtime.py").exists()

from wt_overlay import archetypes, match  # noqa: E402
from wt_overlay.engagement import (MAP_HALF_M, OPEN_CAMERA, Action, Camera, Engagement, FlightCommand, MissileTarget, PlaneSpec,  # noqa: E402
                                   RadarCommand, ReplayWriter, default_library, LAUNCH_OPTIONS)
from wt_overlay.escape import from_enu  # noqa: E402

H = 1/48


class Scripted:
    """Flies a held heading and altitude, scans toward the enemy and fires once when a track is inside ``fire_range``;
    ``after`` = (seconds after the shot, heading) turns the aircraft away."""
    phase = "script"

    def __init__(self, heading, fire_range=None, after=None, altitude=8000., speed=300.):
        self.heading, self.fire_range, self.after, self.altitude, self.speed = heading, fire_range, after, altitude, speed
        self.shot_at = None

    def decide(self, obs):
        fire = None
        heading = self.heading
        if self.fire_range and self.shot_at is None and obs.radar and obs.own.missiles:
            c = obs.radar[0]
            if c.track_id is not None and c.range_m < self.fire_range:
                fire, self.shot_at = c.track_id, obs.time_s
        if self.shot_at is not None and self.after and obs.time_s-self.shot_at >= self.after[0]:
            heading = self.after[1]
        return Action(FlightCommand(heading_deg=heading, altitude_m=self.altitude, speed_mps=self.speed),
                      RadarCommand("tws", 0, 0., 0.), fire)


def spec(aircraft, team, position, heading, controller, missile="us_aim_120c_5", missiles=8, chaff=40, speed=300., **kw):
    h = math.radians(heading)
    return PlaneSpec(aircraft, team, position, (speed*math.sin(h), speed*math.cos(h), 0.), controller, missile, missiles, chaff,
                     kw.pop("rcs_ratio", 1.), kw.pop("flame_probability", 0.), 1.3, **kw)


def duel(shooter_controller, target_controller, range_m=40000., seed=1, target_aircraft="su_30sm2", **kw):
    specs = [spec("f_15c_golden_eagle", 0, (0., -range_m/2, 8000.), 0., shooter_controller),
             spec(target_aircraft, 1, (0., range_m/2, 8000.), 180., target_controller, "su_r_77_1", 10, 60)]
    return Engagement(specs, seed, replay=ReplayWriter(None), **kw)


def events(eng, kind):
    return [e for e in eng.log if e["kind"] == kind]


@unittest.skipUnless(HAVE_MISSILE_SIM, "missile_sim not present")
class FrameTests(unittest.TestCase):
    """The missile target proxy hands missile_sim the aircraft in its own frame and time."""

    @classmethod
    def setUpClass(cls):
        cls.lib = default_library()
        sys.path.insert(0, str(MISSILE_SIM/"src"))

    def legacy(self, profile_id, distance, azimuth, shooter_alt, target_alt, shooter_speed, target_speed, heading):
        from aim120_model.public_api import simulate
        scenario = dict(launch_speed_kmh=shooter_speed*3.6, launch_altitude_m=shooter_alt, launch_pitch_deg=0,
                        launch_heading_deg=heading, target_speed_kmh=target_speed*3.6, target_altitude_m=target_alt,
                        initial_distance_m=distance, target_azimuth_deg=azimuth, target_heading_deg=0,
                        target_course_reference="relative_to_los", target_vertical_heading_deg=0, target_constant_turn_g=0,
                        max_simulation_time_s=150, observation_mode="sensor_track", loft_enabled=True)
        r = simulate(self.lib.profile(profile_id), scenario, clutter_model="look_down_angle", clutter_min_depression_deg=2.,
                     cw_on_clear_beam=True, require_seeker_lock=True)
        return r["summary"]["termination_event"], r["summary"]["flight_time_s"]

    def proxy_run(self, profile_id, shooter_pos, shooter_vel, target_pos, target_vel, t_launch, trim=False):
        """Run a missile against a constant-velocity ENU target through MissileTarget; world clock offset by t_launch."""
        origin = tuple(p-v*t_launch for p, v in zip(target_pos, target_vel))

        class Target:
            ident, bundles = 1, []

            def state_at(self, world):
                return tuple(p+v*world for p, v in zip(origin, target_vel)), target_vel

        eng = SimpleNamespace(library=self.lib, chaff_specs={1: self.lib.ChaffSpec(rcs_ratio=1.)})
        m = SimpleNamespace(target=Target(), t_launch=t_launch, pos_enu=None, vel_enu=None)
        proxy = MissileTarget(eng, m)
        speed = math.sqrt(sum(v*v for v in shooter_vel))
        runtime = self.lib.create(
            self.lib.profile(profile_id), launch_position_m=from_enu(shooter_pos), launch_velocity_mps=from_enu(shooter_vel),
            launch_pitch_deg=math.degrees(math.asin(shooter_vel[2]/speed)),
            launch_heading_deg=math.degrees(math.atan2(-shooter_vel[1], shooter_vel[0])), target=proxy,
            launcher_support=lambda t, truth: "", **LAUNCH_OPTIONS)
        ticks = 0
        while not runtime.done:
            runtime.step()
            ticks += 1
            if trim and ticks % 96 == 0:
                del runtime._rows[:-1]       # what Engagement does to keep concurrent missiles small
        return runtime.event, runtime.time_s

    def test_dropping_old_samples_does_not_change_a_flight(self):
        args = ("us_aim_120c_5", (0., 0., 8000.), (300., 0., 0.), (35000., 0., 8000.), (-280., 0., 0.), 3.)
        self.assertEqual(self.proxy_run(*args), self.proxy_run(*args, trim=True))

    def test_proxy_matches_the_legacy_interface_in_any_orientation_and_clock_offset(self):
        for profile in ("us_aim_120c_5", "su_r_77_1", "cn_pl12"):
            # East-facing: ENU x is missile_sim x. North-facing: missile_sim heading -90 (z = -north).
            east = self.proxy_run(profile, (0., 0., 8000.), (300., 0., 0.), (30000., 0., 8000.), (-280., 0., 0.), 12.5)
            north = self.proxy_run(profile, (1000., 2000., 8000.), (0., 300., 0.), (1000., 32000., 8000.), (0., -280., 0.), 7.)
            old_east = self.legacy(profile, 30000., 0., 8000., 8000., 300., 280., 0.)
            old_north = self.legacy(profile, 30000., -90., 8000., 8000., 300., 280., -90.)
            for new, old in ((east, old_east), (north, old_north)):
                self.assertEqual(new[0], "fuse")
                self.assertEqual(old[0], "proximity_fuse")
                self.assertLess(abs(new[1]-old[1]), .1, profile)

    def test_a_level_flying_aircraft_gives_nearly_the_same_missile_as_a_straight_target(self):
        # The aircraft holds speed and altitude (autothrottle); its history through the proxy, from the launch tick on,
        # must end in the same fuse at nearly the same time as a constant-velocity target of the launch velocity.
        class Hold:
            phase = ""

            def decide(self, obs):
                return Action(FlightCommand(heading_deg=180., altitude_m=8000., speed_mps=280.), RadarCommand("tws"))

        eng = duel(Scripted(0., altitude=8000.), Hold(), range_m=40000.)
        shooter, target = eng.planes
        for _ in range(48*6):
            eng.step()
        self.assertIn(target.ident, shooter.tracked)
        v_t = target.flight.state.velocity
        p_t = target.flight.state.position
        p_s, v_s = shooter.flight.state.position, shooter.flight.state.velocity
        eng.fire(shooter, target)
        m = eng.missiles[0]
        while not m.done and eng.reason is None:
            eng.step()
        self.assertEqual(m.event, "fuse")
        old = self.legacy("us_aim_120c_5", math.dist(p_s, p_t), 0., p_s[2], p_t[2], math.sqrt(sum(x*x for x in v_s)),
                          math.sqrt(sum(x*x for x in v_t)), 0.)
        self.assertEqual(old[0], "proximity_fuse")
        self.assertLess(abs(m.time_s-old[1]), .5)  # Speed and altitude drift a little; the frames and clock must agree.

    def test_chaff_reflectors_are_released_below_the_aircraft_with_the_plain_rcs_ratio(self):
        eng = duel(Scripted(0.), Scripted(180.), seed=2)
        target = eng.planes[1]
        for _ in range(24):
            eng.step()
        eng.drop_chaff(target, 3)
        self.assertEqual(target.chaff, 57)
        m = SimpleNamespace(target=target, t_launch=0., pos_enu=None, vel_enu=None)
        proxy = MissileTarget(eng, m)
        self.assertEqual(proxy.rcs_m2, 1.)
        out = proxy.decoys_at(eng.time+.5)
        self.assertEqual(len(out), 3)
        self.assertTrue(all(r.kind == "chaff" and abs(r.rcs_m2-1.) < 1e-9 for r in out))
        after = proxy.decoys_at(eng.time+20.)   # past the 15 s lifetime
        self.assertEqual(after, [])


@unittest.skipUnless(HAVE_MISSILE_SIM, "missile_sim not present")
class HeadOnTests(unittest.TestCase):
    def test_target_gets_the_rwr_warning_when_the_seeker_comes_on_and_evades_outward_with_chaff(self):
        params = archetypes.sample_params(match.load_model(), "su_30sm2", "left", "top", random.Random(5))
        params = dataclasses.replace(params, defend_maneuver="beam", defend_dive=False, delay_median_s=.5,
                                     recommit_clear_s=30., suppress=False, level_alt_m=8000., peak_alt_m=None,
                                     chaff_style="continuous", flank_offset_deg=30.)
        pilot = archetypes.Pilot(params, random.Random(1), team_forward=(0., -1.), home_xy=(0., 20000.),
                                 enemy_xy=(0., -20000.), missile_id=None)
        eng = duel(Scripted(0., fire_range=39000.), pilot, range_m=40000.)
        eng.planes[1].has_maw = False
        eng.run(until_s=70.)
        shooter, target = eng.planes
        launches, seeker, rwr = events(eng, "launch"), events(eng, "seeker_on"), events(eng, "rwr")
        self.assertEqual(len(launches), 1)
        self.assertEqual(launches[0]["mode"], "tws")
        self.assertEqual(len(seeker), 1)
        warned = [e for e in rwr if e["plane"] == target.ident and e["warning"] == "missile"]
        self.assertTrue(warned)
        self.assertGreaterEqual(warned[0]["t"], seeker[0]["t"])
        self.assertLess(warned[0]["t"]-seeker[0]["t"], .1)   # TWS shot: silent until the seeker is on
        self.assertFalse([e for e in rwr if e["t"] < seeker[0]["t"] and e["warning"] == "missile"])
        phases = [e for e in eng.log if e["kind"] == "phase" and e["plane"] == target.ident and e["to"] == "evade"]
        self.assertTrue(phases)
        self.assertGreaterEqual(phases[0]["t"], seeker[0]["t"])
        self.assertLess(phases[0]["t"]-seeker[0]["t"], 6.)
        # The missile comes from the south; the left flyer of the south-facing team turns east (outer = left = east).
        frames = [json.loads(line) for line in eng.replay.lines if '"type":"frame"' in line]
        t_evade = phases[0]["t"]
        after = [f for f in frames if t_evade+6. <= f["t"] <= t_evade+8.]
        self.assertTrue(after)
        row = next(p for p in after[0]["planes"] if p[0] == target.ident)
        self.assertLess(abs(row[7]-90.), 30., row[7])
        self.assertLess(target.chaff, 60)
        self.assertEqual([e["n"] for e in events(eng, "chaff")], [1]*len(events(eng, "chaff")))
        self.assertTrue(events(eng, "chaff"))

    def test_kill_goes_to_the_shooter(self):
        eng = duel(Scripted(0., fire_range=39000.), None, range_m=40000.)
        eng.run(until_s=120.)
        shooter, target = eng.planes
        kills = events(eng, "kill")
        self.assertEqual(len(kills), 1)
        self.assertEqual((kills[0]["victim"], kills[0]["killer"]), (target.ident, shooter.ident))
        self.assertEqual(shooter.kills, 1)
        self.assertFalse(target.alive)
        end = [e for e in eng.log if e["kind"] == "missile_end"][0]
        self.assertEqual(end["result"], "fuse")
        self.assertEqual(eng.reason, "annihilation")
        self.assertEqual(eng.deaths[0]["cause"], "missile")

    def test_launch_rules(self):
        eng = duel(Scripted(0.), Scripted(180.))
        shooter, target = eng.planes
        for _ in range(48*5):
            eng.step()
        self.assertIn(target.ident, shooter.tracked)
        obs = eng.observe(shooter)
        track = obs.radar[0].track_id
        self.assertIsNone(eng.launch(target, 9999))              # not a track of the picture
        first = eng.launch(shooter, track)
        self.assertIsNotNone(first)
        self.assertEqual(shooter.missiles, 7)
        self.assertIsNone(eng.launch(shooter, track))            # not within 1 s of the last launch
        for _ in range(49):
            eng.step()
        eng.observe(shooter)
        self.assertIsNone(eng.launch(shooter, 9999))             # not a track of the picture
        shooter.missiles = 0
        self.assertIsNone(eng.launch(shooter, track))            # nothing left to fire
        shooter.missiles = 3
        self.assertIsNotNone(eng.launch(shooter, track))
        # A radar that is off holds no track: no launch.
        shooter.radar.set_mode("off")
        shooter.picture, shooter.tracked = None, set()
        shooter.last_launch = -10.
        self.assertIsNone(eng.launch(shooter, track))

    def test_stt_lock_warns_the_target_and_an_stt_shot_is_logged_as_such(self):
        from wt_overlay.engagement import STT_TRACK

        class Lock(Scripted):
            def decide(self, obs):
                radar, fire = RadarCommand("tws", 0, 0., 0.), None
                if obs.radar and obs.radar[0].kind == "track" and obs.own.radar_mode == "tws":
                    radar = RadarCommand("stt", stt_track=obs.radar[0].track_id)
                elif obs.own.radar_mode == "stt":
                    radar = RadarCommand("stt", stt_track=None)
                    if (not self.shot_at and obs.radar and obs.radar[0].kind == "stt" and obs.radar[0].range_m < 39000.
                            and obs.own.stt_state == "tracking"):
                        fire, self.shot_at = STT_TRACK, obs.time_s
                return Action(FlightCommand(heading_deg=0., altitude_m=8000., speed_mps=300.), radar, fire)

        eng = duel(Lock(0.), Scripted(180.), range_m=40000.)
        eng.run(until_s=20.)
        locks = [e for e in events(eng, "rwr") if e["warning"] == "lock" and e["plane"] == 1]
        self.assertTrue(locks)
        launches = events(eng, "launch")
        self.assertEqual(len(launches), 1)
        self.assertEqual(launches[0]["mode"], "stt")
        self.assertGreaterEqual(launches[0]["t"], locks[0]["t"]-1.)

    def test_observation_has_no_truth(self):
        eng = duel(Scripted(0.), Scripted(180.), range_m=30000.)
        for _ in range(48*8):
            eng.step()
        obs = eng.observe(eng.planes[0])
        self.assertIsNone(obs.truth)
        self.assertTrue(obs.radar)
        self.assertFalse(hasattr(obs.radar[0], "truth_id"))
        self.assertFalse(any(hasattr(c, "truth") or hasattr(c, "emitter_id") for c in obs.rwr))
        self.assertEqual(obs.own.team, 0)
        debug = duel(Scripted(0.), Scripted(180.), truth_debug=True)
        self.assertIs(debug.observe(debug.planes[0]).truth, debug)


@unittest.skipUnless(HAVE_MISSILE_SIM, "missile_sim not present")
class ObservationTests(unittest.TestCase):
    def test_maw_and_flames_only_during_the_motor_burn_and_only_for_the_target(self):
        specs = [spec("f_15c_golden_eagle", 0, (0., -5000., 8000.), 0., Scripted(0.)),
                 spec("j_16", 1, (0., 4000., 8000.), 180., Scripted(180.), "cn_pl12a", 4, 40, flame_probability=1.)]
        eng = Engagement(specs, 3, replay=ReplayWriter(None))
        shooter, target = eng.planes
        self.assertTrue(target.has_maw)
        self.assertFalse(shooter.has_maw)
        for _ in range(48*6):
            eng.step()
        self.assertIn(target.ident, shooter.tracked)
        m = eng.fire(shooter, target)
        self.assertTrue(m.flame_seen)
        eng.step()
        obs = eng.observe(target)
        self.assertEqual([s.ref for s in obs.maw], [m.uid])
        self.assertEqual([s.ref for s in obs.flames], [m.uid])
        bearing = obs.maw[0].bearing_deg
        self.assertTrue(170. < bearing < 190., bearing)             # the missile is south of the target
        self.assertEqual(eng.observe(shooter).maw, ())               # nobody warns the shooter
        self.assertEqual(eng.observe(shooter).flames, ())
        # Past the burn nothing shows, whatever the distance.
        m.time_s = m.info.burn_s+.1
        obs = eng.observe(target)
        self.assertEqual((obs.maw, obs.flames), ((), ()))

    def test_flame_is_drawn_per_missile_with_the_planes_probability(self):
        hits = 0
        for seed in range(60):
            eng = duel(Scripted(0.), Scripted(180.), range_m=30000., seed=seed)
            eng.planes[1].flame_p = .5
            for _ in range(48*5):
                eng.step()
            hits += eng.fire(eng.planes[0], eng.planes[1]).flame_seen
        self.assertTrue(15 <= hits <= 45, hits)

    def test_map_marks_carry_what_teammates_found_and_visual_sightings_are_close_only(self):
        specs = [spec("f_15c_golden_eagle", 0, (0., -20000., 8000.), 0., Scripted(0.)),
                 spec("f_16c_block_52_aesa", 0, (30000., -20000., 8000.), 0., Scripted(0.), "us_aim_120c_5", 6, 40),
                 spec("su_30sm2", 1, (0., 10000., 8000.), 180., Scripted(180.), "su_r_77_1", 5)]
        eng = Engagement(specs, 2)
        eng.planes[1].radar.set_mode("off")      # the second friend sees nothing itself
        eng.planes[1].controller = None
        for _ in range(48*8):
            eng.step()
        eng.observe(eng.planes[0])               # the radar picture feeds the team's marks
        obs = eng.observe(eng.planes[1])
        enemies = [m for m in obs.marks if not m.friend]
        friends = [m for m in obs.marks if m.friend]
        self.assertEqual(len(friends), 1)
        self.assertEqual(len(enemies), 1)
        enemy = eng.planes[2]
        self.assertAlmostEqual(enemies[0].y, enemy.flight.state.position[1], delta=1500.)
        self.assertLess(eng.time-enemies[0].time_s, 1.)
        self.assertEqual(obs.visual, ())          # 30 km away
        self.assertEqual(obs.radar, ())           # its own radar is off
        self.assertEqual(len({m.mark_id for m in friends+enemies}), 2)   # opaque ids, one per aircraft
        # Once nobody sees the enemy a mark goes stale after 20 s.
        eng.planes[0].radar.set_mode("off")
        eng.planes[0].controller = None
        for _ in range(48*25):
            eng.step()
        self.assertFalse([m for m in eng.observe(eng.planes[1]).marks if not m.friend])

    def test_the_camera_gates_only_what_the_eyes_give(self):
        class Blind(Camera):
            def sees(self, own, bearing_deg, elevation_deg):
                return False

        class Cone(Camera):
            """Looks along the nose, 40 degrees half-angle."""

            def sees(self, own, bearing_deg, elevation_deg):
                d_az = (bearing_deg-own.heading_deg+180.) % 360.-180.
                return abs(d_az) < 40. and abs(elevation_deg-own.pitch_deg) < 40.

        def world(camera_target, camera_shooter):
            specs = [spec("f_15c_golden_eagle", 0, (0., -5000., 8000.), 0., Scripted(0.), camera=camera_shooter),
                     spec("j_16", 1, (0., 4000., 8000.), 180., Scripted(180.), "cn_pl12a", 4, 40, flame_probability=1.,
                          camera=camera_target)]
            eng = Engagement(specs, 3)
            for _ in range(48*6):
                eng.step()
            m = eng.fire(eng.planes[0], eng.planes[1])
            eng.step()
            return eng, m

        eng, m = world(None, None)                      # the default: no gating
        self.assertIs(eng.planes[0].camera, OPEN_CAMERA)
        target, shooter = eng.observe(eng.planes[1]), eng.observe(eng.planes[0])
        self.assertTrue(target.visual and target.flames and target.maw and target.radar)
        self.assertEqual(len(shooter.shots), 1)
        self.assertAlmostEqual(shooter.shots[0].range_m, math.dist(m.pos_enu, eng.planes[0].own.position), delta=.5)
        self.assertIsNotNone(shooter.shots[0].bearing_deg)
        eng, m = world(Blind(), Blind())
        target, shooter = eng.observe(eng.planes[1]), eng.observe(eng.planes[0])
        self.assertEqual((target.visual, target.flames), ((), ()))          # eyes only: gated
        self.assertTrue(target.maw and target.radar and target.rwr is not None)   # radar, MAW, RWR: not gated
        self.assertTrue([k for k in target.marks if not k.friend] and shooter.radar)   # map marks: not gated
        self.assertEqual(len(shooter.shots), 1)                              # the pilot knows he fired,
        self.assertEqual((shooter.shots[0].bearing_deg, shooter.shots[0].range_m), (None, None))   # but sees no marker
        self.assertIn(eng.planes[1].ident, eng.marks[0])                      # the radar still marked the enemy on the map
        eng, m = world(Cone(), Cone())                  # the nose cameras: the shooter's missile is ahead, the plume behind
        target, shooter = eng.observe(eng.planes[1]), eng.observe(eng.planes[0])
        self.assertTrue(target.flames)                                        # the missile comes at the target's nose
        self.assertIsNotNone(shooter.shots[0].bearing_deg)

    def test_enemies_within_8_km_are_seen_by_bearing_only(self):
        specs = [spec("f_15c_golden_eagle", 0, (0., -3000., 8000.), 0., None),
                 spec("su_30sm2", 1, (3000., 3000., 8200.), 180., None, "su_r_77_1", 5)]
        eng = Engagement(specs, 2)
        obs = eng.observe(eng.planes[0])
        self.assertEqual(len(obs.visual), 1)
        sight = obs.visual[0]
        self.assertAlmostEqual(sight.bearing_deg, math.degrees(math.atan2(3000., 6000.)), delta=.01)
        self.assertGreater(sight.elevation_deg, 0.)
        self.assertIsNone(sight.range_m)


@unittest.skipUnless(HAVE_MISSILE_SIM, "missile_sim not present")
class SupportTests(unittest.TestCase):
    def test_datalink_breaks_when_the_shooter_turns_out_of_radar_range(self):
        eng = duel(Scripted(0., fire_range=39000., after=(1., 180.)), Scripted(180.), range_m=40000.)
        eng.run(until_s=60.)
        lost = events(eng, "datalink_lost")
        self.assertEqual(len(lost), 1)
        self.assertEqual(lost[0]["reason"], "track_lost")
        seeker = events(eng, "seeker_on")
        self.assertTrue(not seeker or lost[0]["t"] < seeker[0]["t"])
        self.assertTrue(events(eng, "track_lost"))

    def test_datalink_breaks_when_the_shooter_dies(self):
        eng = duel(Scripted(0., fire_range=39000.), Scripted(180.), range_m=40000.)
        while not eng.missiles and eng.reason is None:
            eng.step()
        for _ in range(48*4):
            eng.step()
        eng._kill(eng.planes[0], None, "crash", None)
        for _ in range(48*2):
            eng.step()
        lost = events(eng, "datalink_lost")
        self.assertEqual([e["reason"] for e in lost], ["shooter_dead"])
        self.assertTrue(eng.missiles)    # the missile flies on

    def test_datalink_holds_while_the_shooter_keeps_the_track(self):
        eng = duel(Scripted(0., fire_range=39000.), Scripted(180.), range_m=40000.)
        eng.run(until_s=40.)
        lost = events(eng, "datalink_lost")
        self.assertTrue(all(e["reason"] == "seeker_track" for e in lost))


@unittest.skipUnless(HAVE_MISSILE_SIM, "missile_sim not present")
class WorldTests(unittest.TestCase):
    def test_default_map_is_128_km_and_leaving_it_for_15_s_is_a_loss(self):
        self.assertEqual(MAP_HALF_M, 64000.)
        specs = [spec("f_15c_golden_eagle", 0, (0., 62000., 8000.), 0., Scripted(0.)),
                 spec("su_30sm2", 1, (-60000., -60000., 8000.), 180., Scripted(180.), "su_r_77_1", 5)]
        eng = Engagement(specs, 1)
        self.assertEqual(eng.map_half_m, 64000.)
        eng.run(until_s=25.)
        self.assertFalse(eng.planes[0].alive)
        self.assertEqual(eng.planes[0].death["cause"], "out_of_bounds")
        self.assertGreater(eng.planes[0].death["time_s"], 15.)
        wide = Engagement([dataclasses.replace(specs[0]), dataclasses.replace(specs[1])], 1, map_half_m=100000.)
        wide.run(until_s=25.)
        self.assertTrue(all(p.alive for p in wide.planes))

    def test_crash_is_a_death_and_the_clock_ends_the_match(self):
        class Dive:
            phase = ""

            def decide(self, obs):
                return Action(FlightCommand(direction=(0., 1., -1.), floor_m=-1000., max_load=1.5), RadarCommand("tws"))

        specs = [spec("f_15c_golden_eagle", 0, (0., -20000., 2500.), 0., Dive()),
                 spec("su_30sm2", 1, (0., 20000., 8000.), 180., Scripted(180.), "su_r_77_1", 5)]
        eng = Engagement(specs, 1, time_limit_s=90.)
        eng.run()
        self.assertEqual(eng.planes[0].death["cause"], "crash")
        self.assertEqual(eng.reason, "annihilation")
        timed = Engagement([spec("f_15c_golden_eagle", 0, (0., -50000., 8000.), 0., Scripted(0.)),
                            spec("su_30sm2", 1, (0., 50000., 8000.), 180., Scripted(180.), "su_r_77_1", 5)], 1, time_limit_s=3.)
        timed.run()
        self.assertEqual((timed.reason, round(timed.time, 6)), ("time_limit", 3.))

    def test_stalemate_ends_a_match_that_flew_apart(self):
        specs = [spec("f_15c_golden_eagle", 0, (0., -50000., 8000.), 180., Scripted(180.)),
                 spec("su_30sm2", 1, (0., 50000., 8000.), 0., Scripted(0.), "su_r_77_1", 5)]
        eng = Engagement(specs, 1, map_half_m=300000.)
        eng.run()
        self.assertEqual(eng.reason, "stalemate")
        self.assertGreaterEqual(eng.time, 60.)
        self.assertLess(eng.time, 400.)

    def test_deterministic_replay_for_a_seed(self):
        def run(seed):
            m = match.scenario([[dict(aircraft="su_30sm2", archetype="middle", skill="normal"),
                                 dict(aircraft="f_16c_block_52_aesa", archetype="left", skill="top")],
                                [dict(aircraft="ef_2000_aesa", archetype="left", skill="normal"),
                                 dict(aircraft="j_16", archetype="crawler", skill="normal")]], seed, range_km=60.)
            replay = ReplayWriter(None)
            eng = m.engagement(replay=replay)
            eng.run(until_s=100.)
            return replay.lines

        a, c, b = run(4), run(5), run(4)   # another seed in between: no state may leak from run to run
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertGreater(len(a), 400)

    def test_replay_format(self):
        eng = duel(Scripted(0., fire_range=39000.), Scripted(180.))
        eng.run(until_s=12.)
        rows = [json.loads(line) for line in eng.replay.lines]
        self.assertEqual(rows[0]["type"], "header")
        self.assertEqual(rows[0]["map_half_m"], 64000.)
        frames = [r for r in rows if r["type"] == "frame"]
        self.assertAlmostEqual(frames[1]["t"]-frames[0]["t"], .25)
        self.assertEqual(len(frames[0]["planes"][0]), len(rows[0]["plane_columns"]))
        self.assertEqual(frames[1]["planes"][0][10], "script")
        flying = [m for f in frames for m in f["missiles"]]
        self.assertTrue(flying)
        self.assertEqual(len(flying[0]), len(rows[0]["missile_columns"]))
        self.assertTrue(0. <= flying[0][9] < 360.)
        self.assertEqual({r["type"] for r in rows}, {"header", "frame", "event"})


@unittest.skipUnless(HAVE_MISSILE_SIM, "missile_sim not present")
class SmokeTests(unittest.TestCase):
    def test_16v16_runs_120_s_with_finite_numbers(self):
        m = match.random_match(7)
        replay = ReplayWriter(None)
        eng = m.engagement(replay=replay)
        result = eng.run(until_s=120.)
        self.assertEqual(round(result.time_s, 6), 120.)
        self.assertEqual(result.missile_errors, 0)
        self.assertEqual(len(eng.planes), 32)

        def finite(x):
            if isinstance(x, float):
                self.assertTrue(math.isfinite(x))
            elif isinstance(x, (list, tuple)):
                for y in x:
                    finite(y)
            elif isinstance(x, dict):
                for y in x.values():
                    finite(y)
        for line in replay.lines:
            finite(json.loads(line))
        for p in eng.planes:
            finite(p.flight.state.position)
            finite(p.flight.state.velocity)
        self.assertGreater(len(replay.lines), 400)
        self.assertEqual(sum(p["fm_faults"] for p in result.planes), 0)


class ArchetypeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = match.load_model()

    def test_behaviour_constants_match_pk_behaviour(self):
        sys.path.insert(0, str(ROOT/"scripts"))
        from pk_behaviour import Behaviour
        b = Behaviour()
        self.assertEqual((archetypes.P_REACT["normal"], archetypes.P_CORRECT["normal"], archetypes.P_LOCK_REACT["normal"]),
                         (b.p_react, b.p_correct, b.p_lock_react))
        self.assertEqual((archetypes.DELAY_MEDIAN_S, archetypes.DELAY_SIGMA), (b.delay_median_s, b.delay_sigma))
        self.assertEqual(archetypes.REPERTOIRE, b.repertoire)

    def test_sampled_parameters_follow_the_json(self):
        rng = random.Random(3)
        prm = self.model["parameters"]
        draws = [archetypes.sample_params(self.model, "su_30sm2", "middle", "normal", rng) for _ in range(400)]
        self.assertTrue(all(7500 <= d.level_alt_m <= 9000 for d in draws))
        peaks = sum(1 for d in draws if d.peak_alt_m == 11000.)
        self.assertGreater(peaks, 40)
        self.assertLess(peaks, 120)                                       # about 20 %
        self.assertTrue(all(d.suppress for d in draws))                   # middle: always
        self.assertTrue(all(.8 <= d.suppress_fraction <= 1. for d in draws))
        self.assertTrue(all(3. <= d.recommit_clear_s <= 10. for d in draws))
        self.assertTrue(all(20000. <= d.second_round_m <= 30000. for d in draws))
        self.assertTrue(any(d.early_left for d in draws))                 # J-16 / Su-30SM2 middle players
        left = [archetypes.sample_params(self.model, "f_15c_golden_eagle", "left", "normal", rng) for _ in range(600)]
        share = sum(d.suppress for d in left)/len(left)
        self.assertAlmostEqual(share, prm["suppression_shot_probability"]["left"], delta=.07)
        self.assertTrue(all(30. <= d.flank_offset_deg <= 60. for d in left))
        self.assertTrue(all(not d.early_left for d in left))
        crawler = archetypes.sample_params(self.model, "j_10c", "crawler", "normal", rng)
        self.assertTrue(30. <= crawler.crawler_alt_m <= 200.)
        self.assertEqual(archetypes.sample_params(self.model, "j_10c", "rusher", "normal", rng).p_react, archetypes.RUSH_P_REACT)
        self.assertEqual(archetypes.sample_params(self.model, "j_10c", "left", "top", rng).p_correct, 1.)
        with self.assertRaises(ValueError):
            archetypes.sample_params(self.model, "j_10c", "ace", "normal", rng)

    def pilot(self, archetype="left", team=0, **override):
        params = archetypes.sample_params(self.model, "f_15c_golden_eagle", archetype, "normal", random.Random(11))
        params = dataclasses.replace(params, **override)
        forward = (0., 1.) if team == 0 else (0., -1.)
        return archetypes.Pilot(params, random.Random(2), team_forward=forward, home_xy=(0., -50000. if team == 0 else 50000.),
                                enemy_xy=(0., 50000. if team == 0 else -50000.), missile_id=None)

    def observation(self, **own):
        from wt_overlay.engagement import Observation, OwnObs
        fields = dict(time_s=0., team=0, aircraft="f_15c_golden_eagle", position=(0., -50000., 3000.),
                      velocity=(0., 300., 0.), heading_deg=0., pitch_deg=0., roll_deg=0., speed_mps=300., altitude_m=3000.,
                      aoa_deg=1., load=1., engine_percent=100., missile_id=None, missiles=8, chaff=50, radar_mode="tws",
                      stt_state=None, has_maw=False)
        fields.update(own)
        return Observation(fields["time_s"], OwnObs(**fields), (), (), (), (), (), (), (), 64000.)

    def test_flank_heading_is_left_for_left_flyers_and_right_for_right_flyers(self):
        for team, forward_deg in ((0, 0.), (1, 180.)):
            left = self.pilot("left", team, flank_offset_deg=40.).decide(self.observation(team=team, heading_deg=forward_deg))
            right = self.pilot("right", team, flank_offset_deg=40.).decide(self.observation(team=team, heading_deg=forward_deg))
            self.assertAlmostEqual((left.flight.heading_deg-forward_deg) % 360., 320., places=6)   # counter-clockwise 40
            self.assertAlmostEqual((right.flight.heading_deg-forward_deg) % 360., 40., places=6)

    def test_reaction_delay_is_log_normal_with_median_1_5_s(self):
        from wt_overlay.sensors import RwrContact
        delays, reacted = [], 0
        for seed in range(600):
            pilot = self.pilot("left")
            pilot.rng = random.Random(seed)
            contact = RwrContact(1, "missile", 0., 0., 0., 12000., False, True, 8, None, True, True, 0.)
            obs = dataclasses.replace(self.observation(), rwr=(contact,))
            pilot.decide(obs)
            if pilot.pending is not None:
                reacted += 1
                delays.append(pilot.pending.at)
        delays.sort()
        self.assertAlmostEqual(reacted/600, .85, delta=.05)
        self.assertAlmostEqual(delays[len(delays)//2], 1.5, delta=.15)
        self.assertLess(delays[int(.05*len(delays))], 1.)
        self.assertGreater(delays[int(.95*len(delays))], 2.2)

    def test_top_pilots_always_react_and_are_always_right(self):
        from wt_overlay.sensors import RwrContact
        for seed in range(50):
            pilot = self.pilot("left", p_react=1., p_correct=1., p_lock_react=1.)
            pilot.rng = random.Random(seed)
            contact = RwrContact(1, "missile", 0., 0., 0., 12000., False, True, 8, None, True, True, 0.)
            pilot.decide(dataclasses.replace(self.observation(), rwr=(contact,)))
            self.assertIsNotNone(pilot.pending)
            self.assertTrue(pilot.pending.correct)

    def test_wrong_picks_come_from_the_repertoire(self):
        pilot = self.pilot("left")
        plans = {pilot._plan(False) for _ in range(300)}
        self.assertEqual(plans, set(archetypes.REPERTOIRE))
        self.assertEqual(pilot._plan(True)[0], 90. if pilot.p.defend_maneuver == "beam" else 0.)

    def test_range_judge_bins_and_caches(self):
        judge = archetypes.range_judge("us_aim_120c_5")
        hot = judge.rmax(8000., 300., 0.)
        cold = judge.rmax(8000., 300., 180.)
        self.assertGreater(hot, cold)
        self.assertGreater(hot, 15000.)
        beam = judge.rmax(8000., 300., 90.)
        self.assertTrue(cold < beam < hot)
        n = len(judge._cache)
        judge.rmax(8000., 300., 0.)     # a repeated reading costs no new model evaluation
        self.assertEqual(len(judge._cache), n)
        self.assertAlmostEqual(judge.rmax(8000., 300., 90.), judge.rmax(8010., 300.5, 90.), delta=100.)   # interpolated: continuous
        self.assertGreater(judge.rmax(11000., 400., 180.), judge.rmax(3000., 250., 180.))   # high and fast reaches farther
        self.assertLessEqual(hot, 45000.)   # the hit-probability network stops at 45 km: the hot line is capped

    def test_reach_lines_equal_the_envelope_lines(self):
        from wt_overlay import offense
        advisor = offense.OffenseAdvisor("cn_pl12")
        hot, cold = advisor.reach_lines(8000., 300.)
        row = advisor.envelope(8000., 300., azimuths_deg=(0.,), alt_diffs_m=(0.,)).lines[(0., 0.)]
        self.assertEqual((hot, cold), (row["rmax_hot"], row["rmax_cold"]))
        with self.assertRaises(ValueError):
            advisor.reach_lines(float("nan"), 300.)

    def test_a_lost_enemy_is_looked_for_where_he_was_heading_and_then_at_the_spawn(self):
        pilot = self.pilot("left")
        seen = SimpleNamespace(position=(0., 20000., 8000.), velocity=(0., -300., 0.))
        obs = dataclasses.replace(self.observation(time_s=10.), radar=(seen,))
        self.assertEqual(pilot._anchor(obs)[3], "radar")
        later = self.observation(time_s=20.)                      # no track, no mark
        x, y, z, source = pilot._anchor(later)
        self.assertEqual(source, "memory")
        self.assertAlmostEqual(y, 20000.-300.*10., delta=1e-6)    # carried on at his last velocity
        self.assertIsNone(z)
        self.assertEqual(pilot._anchor(self.observation(time_s=10.+archetypes.LAST_KNOWN_S+1.))[3], "spawn")
        mark = SimpleNamespace(friend=False, time_s=95., x=500., y=-3000., z=None)
        obs = dataclasses.replace(self.observation(time_s=100.), marks=(mark,))
        self.assertEqual(pilot._anchor(obs)[3], "mark")

    def test_a_pilot_without_missiles_flies_to_the_spawn_point_and_loiters(self):
        pilot = self.pilot("left", team=0)                       # team 0 home is (0, -50000)
        far = self.observation(position=(20000., 10000., 8000.), velocity=(0., -300., 0.), heading_deg=180., missiles=0)
        action = pilot.decide(far)
        self.assertEqual(pilot.phase, "home")
        self.assertAlmostEqual(action.flight.heading_deg, math.degrees(math.atan2(-20000., -60000.)) % 360., delta=.01)
        near = self.observation(position=(1000., -49000., 8000.), velocity=(0., -300., 0.), heading_deg=180., missiles=0, time_s=1.)
        turn = pilot.decide(near).flight.heading_deg
        self.assertAlmostEqual((turn-180.) % 360., archetypes.HOME_ORBIT_TURN_DEG, delta=.01)   # a gentle turn, not a straight run to the edge

    def test_boundary_guard_turns_inward(self):
        pilot = self.pilot("left")
        cmd = FlightCommand(heading_deg=90.)
        obs = self.observation(position=(60000., 0., 8000.), velocity=(400., 0., 0.), heading_deg=90.)
        guarded = pilot._guard(obs, cmd, "round2")
        self.assertTrue(pilot.guard)
        self.assertAlmostEqual(guarded.heading_deg, 270., places=6)
        calm = self.observation(position=(0., 0., 8000.), velocity=(400., 0., 0.), heading_deg=90.)
        self.assertIs(pilot._guard(calm, cmd, "round2"), cmd)
        self.assertFalse(pilot.guard)


class MatchTests(unittest.TestCase):
    @unittest.skipUnless(HAVE_MISSILE_SIM, "missile_sim not present")
    def test_random_match_follows_the_model(self):
        model = match.load_model()
        sep = []
        mix = {}
        counts = {}
        for seed in range(30):
            m = match.random_match(seed)
            self.assertEqual(len(m.specs), 32)
            self.assertEqual(sum(1 for s in m.specs if s.team == 0), 16)
            sep.append(m.separation_m)
            ys = {t: [s.position[1] for s in m.specs if s.team == t] for t in (0, 1)}
            self.assertLess(max(ys[0]), 0.)
            self.assertGreater(min(ys[1]), 0.)
            self.assertAlmostEqual(min(ys[1])-max(ys[0]), m.separation_m, delta=2*match.ALONG_JITTER_M+1.)
            for s in m.specs:
                self.assertLessEqual(abs(s.position[0]), 15000.+1.)
                self.assertTrue(2000. <= s.position[2] <= 3000.)
                speed = math.hypot(s.velocity[0], s.velocity[1])
                from wt_overlay.fm import atmosphere
                self.assertTrue(.8 <= speed/atmosphere(s.position[2])[1] <= 1.2)
                self.assertEqual(s.velocity[1] > 0, s.team == 0)    # heading at the enemy
                eq = match.equipment_data().equipment[s.aircraft]
                self.assertIn(s.missile, eq.missiles)
                self.assertEqual(s.missiles, eq.missiles[s.missile])
                self.assertLessEqual(s.chaff, eq.countermeasures)
                self.assertGreaterEqual(s.chaff, round(.5*eq.countermeasures)-1)
                self.assertIn(s.rcs_ratio, (.5, 1., 2.))
                counts[s.aircraft] = counts.get(s.aircraft, 0)+1
                mix[s.skill] = mix.get(s.skill, 0)+1
        self.assertTrue(90000. <= min(sep) and max(sep) <= 110000.)
        self.assertAlmostEqual(mix["top"]/sum(mix.values()), .2, delta=.04)
        weights = {a: w for a, w in model["aircraft_frequency"]["weights"].items()}
        total = sum(weights.values())
        n = sum(counts.values())
        for aircraft in ("su_30sm2", "j_16", "su_30sm", "f_15c_golden_eagle"):
            self.assertAlmostEqual(counts.get(aircraft, 0)/n, weights[aircraft]/total, delta=.03)
        self.assertFalse(set(counts)-set(weights))
        self.assertFalse({"rafale_c_f3", "rafale_m_f3r", "rafale_eg_greece"} & set(counts))

    @unittest.skipUnless(HAVE_MISSILE_SIM, "missile_sim not present")
    def test_archetype_mix_follows_group_priors_around_a_dirichlet(self):
        model = match.load_model()
        picks = {"left": 0, "middle": 0, "right": 0, "crawler": 0, "rusher": 0}
        total = 0
        for seed in range(60):
            m = match.random_match(seed)
            for s in m.specs:
                if s.aircraft in ("f_15c_golden_eagle", "j_15t"):
                    picks[s.archetype] += 1
                    total += 1
        self.assertGreater(total, 100)
        prior = model["groups"][2]["prior"]
        self.assertGreater(picks["left"]/total, .7)
        self.assertAlmostEqual(picks["left"]/total, prior["left"], delta=.1)
        rng = random.Random(1)
        draws = [match.dirichlet(prior, 20., rng) for _ in range(500)]
        self.assertAlmostEqual(sum(d["left"] for d in draws)/500, prior["left"], delta=.02)
        self.assertGreater(max(d["left"] for d in draws)-min(d["left"] for d in draws), .05)   # it does vary by match

    @unittest.skipUnless(HAVE_MISSILE_SIM, "missile_sim not present")
    def test_small_scenarios(self):
        m = match.scenario([["su_30sm2"], ["f_15c_golden_eagle"]], 3, range_km=80.)
        self.assertEqual(m.separation_m, 80000.)
        self.assertEqual(len(m.specs), 2)
        self.assertEqual(m.specs[0].missile, "su_r_77_1")
        two = match.scenario([[dict(aircraft="j_16", archetype="left"), dict(aircraft="j_10c", archetype="crawler")],
                              [dict(aircraft="ef_2000_aesa", archetype="middle", skill="top", missiles=3),
                               "f_16c_block_52_aesa"]], 1)
        self.assertEqual([s.archetype for s in two.specs][:3], ["left", "crawler", "middle"])
        self.assertEqual(two.specs[2].missiles, 3)
        self.assertEqual(two.specs[2].skill, "top")
        self.assertTrue(90000. <= two.separation_m <= 110000.)
        with self.assertRaises(ValueError):
            match.scenario([[dict(aircraft="j_16", archetype="ace")], ["j_10c"]])


class SensorAccessorTests(unittest.TestCase):
    def test_tracked_ids_reports_tws_tracks_and_stt(self):
        from wt_overlay import units
        from wt_overlay.sensors import OwnState, RadarSensor, TargetTruth
        radar = RadarSensor(units.load().radar_of("f_16c_block_52_aesa"), owner=0)
        radar.set_mode("tws", 0, 0., 0., t=0.)
        self.assertEqual(radar.tracked_ids(), set())
        own = OwnState((0., 0., 8000.), (0., 300., 0.), 0.)
        target = TargetTruth(7, (0., 30000., 8000.), (0., -300., 0.))
        t = 0.
        for _ in range(48*8):
            t += H
            radar.update(t, H, own, [target], report=False)
        self.assertEqual(radar.tracked_ids(), {7})
        radar.set_mode("stt", stt_truth=7, t=t)
        for _ in range(10):
            t += H
            radar.update(t, H, own, [target], report=False)
        self.assertEqual(radar.tracked_ids(), {7})
        radar.set_mode("off")
        self.assertEqual(radar.tracked_ids(), set())


if __name__ == "__main__":
    unittest.main()
