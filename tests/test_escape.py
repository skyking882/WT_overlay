"""Offline FM evader: protocol, beam/drag geometry and optional missile_sim coupling."""
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

from wt_overlay.escape import EvasionPilot, FMEvader, Perception, fm_evader_factory, from_enu, to_enu
from wt_overlay.fm import load_aircraft
from wt_overlay.turn import ManeuverModel, norm

MISSILE_SIM = Path(__file__).resolve().parents[2]/"missle_sim"


class HeadOn:
    """missile_sim frame (x, up, z): 1000 km/h toward the origin at 8 km."""
    velocity = (-1000/3.6, 0., 0.)

    def state_at(self, t):
        return SimpleNamespace(position=(20000+self.velocity[0]*t, 8000., 0.), velocity=self.velocity)


def fly(evader, seconds, missile=(0., 8000., 0.)):
    states = []
    for k in range(1, int(seconds*48)+1):
        evader.observe_missile((k-1)/48, missile, (0., 0., 0.))
        states.append(evader.state_at(k/48))
    return states


def radial(state, missile=(0., 8000., 0.)):
    p, v = to_enu(state.position), to_enu(state.velocity)
    m = to_enu(missile)
    los = (p[0]-m[0], p[1]-m[1])
    return (los[0]*v[0]+los[1]*v[1])/math.hypot(*los)


class EvaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = ManeuverModel(load_aircraft("f_16c_block_50"), 12000.)

    def test_frames_round_trip_and_keep_up_axis(self):
        v = (1., 2., 3.)
        self.assertEqual(from_enu(to_enu(v)), v)
        self.assertEqual(to_enu((0., 5., 0.))[2], 5.)

    def test_reproduces_scenario_target_until_start(self):
        base = HeadOn()
        evader = FMEvader(base, EvasionPilot("beam", 4.), self.model)
        for t in (0., 1.5, 4.):
            self.assertEqual(evader.state_at(t), base.state_at(t))

    def test_beam_reaches_zero_radial_within_aoa_ceiling(self):
        evader = FMEvader(HeadOn(), EvasionPilot("beam", 0.), self.model)
        states = fly(evader, 12)
        self.assertLess(abs(radial(states[-1])), 5.)
        self.assertTrue(all(s.aoa_deg <= 20.+1e-9 for s in evader._states))
        self.assertLess(evader.min_speed_mps, 1000/3.6)  # The turn costs energy.
        self.assertIsNone(evader.fault)

    def test_drag_reverses_in_a_level_turn(self):
        evader = FMEvader(HeadOn(), EvasionPilot("drag", 0.), self.model)
        states = fly(evader, 15)
        speed = norm(states[-1].velocity)
        self.assertGreater(radial(states[-1]), .95*speed)
        self.assertLess(abs(states[-1].position[1]-8000.), 600.)

    def test_dive_trades_altitude_for_speed(self):
        level, dive = (FMEvader(HeadOn(), EvasionPilot("drag", 0., dive_deg=d), self.model) for d in (0., 20.))
        a, b = fly(level, 20)[-1], fly(dive, 20)[-1]
        self.assertLess(b.position[1], a.position[1])
        self.assertGreater(norm(b.velocity), norm(a.velocity))

    def test_perception_switches_between_missile_and_launcher(self):
        evader = FMEvader(HeadOn(), EvasionPilot("beam", 0.), self.model, perception=Perception(2., 5000.))
        target = to_enu(HeadOn().state_at(0.).position)
        launcher_velocity = (300., 0., 0.)
        # Missile flies an offset path, so its direction differs from the launcher's.
        evader.observe_missile(0., (0., 8000., 0.), launcher_velocity)
        evader.observe_missile(1., (900., 8000., -3000.), (900., 0., 0.))
        self.assertEqual(evader._reference(target), to_enu((900., 8000., -3000.)))  # Motor burning.
        evader.observe_missile(5., (6000., 8000., -3000.), (900., 0., 0.))
        self.assertEqual(evader._reference(target), to_enu((1500., 8000., 0.)))  # Launcher held course.
        evader.observe_missile(9., (16500., 8000., -500.), (900., 0., 0.))
        self.assertEqual(evader._reference(target), to_enu((16500., 8000., -500.)))  # Seeker active.
        evader.observe_missile(12., (19000., 8000., 9000.), (900., 0., 0.))
        self.assertEqual(evader._reference(target), to_enu((19000., 8000., 9000.)))  # Latched.

    def test_launcher_crank_path(self):
        v = (300., 0., 0.)
        straight = Perception(2., 16000.)
        self.assertEqual(straight.launcher_offset(v, 10.), (3000., 0., 0.))
        rate = .1
        crank = Perception(2., 16000., crank_rad=math.pi/2, crank_rate_rad_s=rate, crank_start_s=1.)
        radius, turn = 300./rate, (math.pi/2)/rate
        before = crank.launcher_offset(v, 1.)
        self.assertEqual(before, (300., 0., 0.))
        done = crank.launcher_offset(v, 1.+turn)
        self.assertAlmostEqual(done[0], 300.+radius, places=6)
        self.assertAlmostEqual(done[1], radius, places=6)
        later = crank.launcher_offset(v, 1.+turn+2.)
        self.assertAlmostEqual(later[0], done[0], places=6)
        self.assertAlmostEqual(later[1], done[1]+600., places=6)
        right = Perception(2., 16000., crank_rad=-math.pi/2, crank_rate_rad_s=rate)
        self.assertAlmostEqual(right.launcher_offset(v, turn)[1], -radius, places=6)

    def test_truth_perception_always_uses_the_missile(self):
        evader = FMEvader(HeadOn(), EvasionPilot("beam", 0.), self.model)
        evader.observe_missile(0., (0., 8000., 0.), (300., 0., 0.))
        evader.observe_missile(5., (6000., 8000., -3000.), (900., 0., 0.))
        self.assertEqual(evader._reference((20000., 0., 8000.)), to_enu((6000., 8000., -3000.)))
        self.assertEqual(evader.describe()["perception"], "truth")

    def test_pilot_validation(self):
        for bad in (dict(kind="loop", start_s=0.), dict(kind="beam", start_s=-1.),
                    dict(kind="beam", start_s=0., max_load=0.), dict(kind="drag", start_s=0., dive_deg=70.)):
            with self.assertRaises(ValueError):
                EvasionPilot(**bad)


@unittest.skipUnless((MISSILE_SIM/"src"/"aim120_model"/"evasive_target.py").exists(), "missile_sim not present")
class MissileSimCouplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(MISSILE_SIM/"src"))
        from aim120_model.profile_catalog import load_profile_catalog
        from aim120_model.public_api import simulate
        from aim120_model.target import TargetState
        from missile_gui.library import scan_library
        _, catalog = load_profile_catalog(MISSILE_SIM)
        profiles, _ = scan_library(catalog["profiles_dir"], MISSILE_SIM)
        cls.profile = {p["missile_id"]: p for p in profiles}["us_aim_120c_5"]
        cls.simulate, cls.state = staticmethod(simulate), TargetState
        cls.model = ManeuverModel(load_aircraft("f_16c_block_50"), 12000.)
        cls.scenario = dict(launch_speed_kmh=1100, launch_altitude_m=8000, launch_pitch_deg=0, launch_heading_deg=0,
                            target_speed_kmh=1000, target_altitude_m=8000, initial_distance_m=20000,
                            target_azimuth_deg=0, target_heading_deg=0, target_course_reference="relative_to_los",
                            target_vertical_heading_deg=0, target_constant_turn_g=0, max_simulation_time_s=40,
                            observation_mode="sensor_track", loft_enabled=True)

    def test_late_start_matches_default_runtime(self):
        default = self.simulate(self.profile, self.scenario)
        late = self.simulate(self.profile, self.scenario, target_factory=fm_evader_factory(
            EvasionPilot("beam", 1e6), self.model, self.state))
        self.assertEqual(late["summary"], default["summary"])
        self.assertEqual(late["model"]["target_model"]["model"], "wt_overlay_fm_pilot")

    def test_evasion_changes_the_engagement(self):
        default = self.simulate(self.profile, self.scenario)
        evaded = self.simulate(self.profile, self.scenario, target_factory=fm_evader_factory(
            EvasionPilot("beam", 5.), self.model, self.state))
        key = lambda r: (r["summary"]["termination_event"], r["summary"]["flight_time_s"])  # noqa: E731
        self.assertNotEqual(key(evaded), key(default))
        self.assertLess(evaded["model"]["target_model"]["min_speed_mps"], 1000/3.6)


if __name__ == "__main__":
    unittest.main()
