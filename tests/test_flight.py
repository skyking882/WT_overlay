"""Command-driven aircraft: follows commands within load and roll-rate limits, history queries, FMEvader unchanged."""
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

from wt_overlay.contracts import G
from wt_overlay.escape import EvasionPilot, FMEvader
from wt_overlay.flight import (DEFAULT_LOAD_LIMITS, SUBSTEP_S, Aircraft, FlightCommand, FlightParams, KeyboardCommand,
                               aircraft_model, model_with_mass, read_structure)
from wt_overlay.fm import load_aircraft
from wt_overlay.turn import ManeuverModel, add, cross, dot, norm, rotate, scale, transport, unit

H = SUBSTEP_S


def wrap(deg):
    return (deg+180.) % 360.-180.


def fly(aircraft, seconds, command=None):
    if command is not None:
        aircraft.command = command
    for _ in range(round(seconds/H)):
        aircraft.step()
    return aircraft


class HeadOn:
    velocity = (-1000/3.6, 0., 0.)

    def state_at(self, t):
        return SimpleNamespace(position=(20000+self.velocity[0]*t, 8000., 0.), velocity=self.velocity)


class FlightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = aircraft_model("f_15c_golden_eagle", mass_factor=1.3)

    def new(self, altitude=3000., speed=330., heading=0., **kwargs):
        h = math.radians(heading)
        return Aircraft(self.model, (0., 0., altitude), (speed*math.sin(h), speed*math.cos(h), 0.), keep_history=True,
                        **kwargs)

    def test_turns_to_a_compass_heading_within_load_and_roll_rate(self):
        for target in (90., -60., 175.):
            a = self.new(heading=0.)
            rolls, loads = [], []
            a.command = FlightCommand(heading_deg=target % 360., max_load=7.)
            for _ in range(round(30/H)):
                a.step()
                rolls.append(a.attitude()[2])
                loads.append(a.load)
            self.assertLess(abs(wrap(a.attitude()[0]-target)), 1., target)
            self.assertLessEqual(max(loads), 7.*1.05, target)
            limit = FlightParams().roll_rate_deg_s*H
            self.assertLessEqual(max(abs(wrap(b-c)) for b, c in zip(rolls[1:], rolls)), limit+.3, target)
            self.assertGreater(max(loads), 6.)  # It actually pulled.
            self.assertIsNone(None if a.faults == 0 else a.faults)

    def test_reversal_is_a_horizontal_turn(self):
        a = fly(self.new(heading=0.), 24, FlightCommand(heading_deg=180.))
        self.assertLess(abs(wrap(a.attitude()[0]-180.)), 1.)
        self.assertLess(abs(a.altitude-3000.), 600.)  # No loop over or under.

    def test_captures_altitude_and_speed(self):
        a = fly(self.new(altitude=3000., speed=300.), 100,
                FlightCommand(heading_deg=0., altitude_m=7000., speed_mps=330., min_speed_mps=250.))
        self.assertLess(abs(a.altitude-7000.), 10.)
        self.assertLess(abs(a.speed-330.), 2.)
        self.assertGreaterEqual(min(norm(s.velocity) for s in a.states), 250.)
        down = fly(self.new(altitude=7000., speed=330.), 80, FlightCommand(heading_deg=90., altitude_m=3000.))
        self.assertLess(abs(down.altitude-3000.), 10.)
        self.assertLess(abs(wrap(down.attitude()[0]-90.)), 1.)

    def test_slows_to_a_lower_speed_with_the_airbrake_allowed(self):
        free = fly(self.new(speed=330.), 40, FlightCommand(altitude_m=3000., speed_mps=240.))
        braked = fly(self.new(speed=330.), 40, FlightCommand(altitude_m=3000., speed_mps=240., airbrake_allowed=True))
        self.assertLess(abs(braked.speed-240.), 2.)
        self.assertGreater(braked.state.engine_percent, -1.)
        self.assertLessEqual(max(s.airbrake for s in free.states), 0.)
        self.assertGreater(max(s.airbrake for s in braked.states), 0.5)

    def test_throttle_command_holds_a_setting(self):
        a = fly(self.new(), 20, FlightCommand(heading_deg=0., altitude_m=3000., throttle_percent=60.))
        self.assertAlmostEqual(a.state.engine_percent, 60., delta=.5)

    def test_direction_vector_and_flight_path_angle(self):
        a = fly(self.new(), 12, FlightCommand(direction=(1., 1., 0.)))
        self.assertLess(abs(wrap(a.attitude()[0]-45.)), 1.)
        b = fly(self.new(), 8, FlightCommand(heading_deg=0., climb_deg=15.))
        v = b.state.velocity
        self.assertAlmostEqual(math.degrees(math.asin(v[2]/norm(v))), 15., delta=1.5)

    def test_ground_is_death_and_the_floor_protects(self):
        dive = FlightCommand(direction=(0., 1., -1.), floor_m=-1000., max_load=2.)
        a = fly(self.new(altitude=800., speed=250.), 60, dive)
        self.assertTrue(a.crashed and not a.alive)
        frozen = a.state
        a.step()
        self.assertIs(a.state, frozen)
        safe = fly(self.new(altitude=3000., speed=250.), 40, FlightCommand(direction=(0., 1., -1.), floor_m=300.))
        self.assertTrue(safe.alive)
        self.assertGreater(safe.min_altitude_m, 100.)

    def test_pull_out_outranks_a_turn_demanded_in_a_steep_dive(self):
        # Diving 50 degrees at 7 km while told to turn 90 degrees and keep diving: the floor makes it pull up wings level first.
        g = math.radians(-50)
        a = Aircraft(self.model, (0., 0., 7000.), (0., 300*math.cos(g), 300*math.sin(g)))
        a.command = FlightCommand(direction=(1., 0., -1.), floor_m=2500.)
        for _ in range(48*40):
            a.step()
        self.assertTrue(a.alive)
        self.assertGreater(a.min_altitude_m, 1500.)

    def test_engine_table_limits_the_speed_and_a_climb_stops_below_the_ceiling(self):
        a = fly(self.new(altitude=6000., speed=320.), 200, FlightCommand(heading_deg=0., altitude_m=6000.))
        self.assertEqual(a.faults, 0)
        self.assertLessEqual(a.speed, a.v_engine_mps+1.)
        self.assertLessEqual(a.speed, 2.*340.+1.)        # Mach 2 at this altitude, whatever the engine table allows
        up = fly(self.new(altitude=14000., speed=320.), 120, FlightCommand(heading_deg=0., altitude_m=30000., max_climb_deg=60.,
                                                                         min_speed_mps=200.))
        self.assertEqual(up.faults, 0)
        self.assertLess(up.altitude, up.ceiling_m)

    def test_state_at_is_exact_on_ticks_and_linear_between(self):
        a = fly(self.new(history=200), 2, FlightCommand(heading_deg=60.))
        for k in (0, 5, 96):
            p, v = a.state_at(k*H)
            self.assertEqual((p, v), (a.states[k].position, a.states[k].velocity))
        p, v = a.state_at(10.5*H)
        for x, y, z in zip(p, a.states[10].position, a.states[11].position):
            self.assertAlmostEqual(x, (y+z)/2, places=9)
        self.assertEqual(a.state_at(96*H+1e-12)[0], a.states[96].position)  # float noise on a tick time
        with self.assertRaises(ValueError):
            a.state_at(97*H)
        short = self.new(history=10)
        fly(short, 1)
        short.state_at(short.time)
        with self.assertRaises(ValueError):
            short.state_at(0.)

    def test_offset_start_time(self):
        a = Aircraft(self.model, (0., 0., 3000.), (0., 300., 0.), t0=12.5)
        fly(a, 1)
        self.assertAlmostEqual(a.time, 13.5)
        self.assertEqual(a.state_at(13.5)[0], a.state.position)

    def test_below_the_fm_table_speed_falls_ballistically_and_recovers(self):
        a = Aircraft(self.model, (0., 0., 6000.), (0., 40., 0.))
        fly(a, 6, FlightCommand(heading_deg=0.))
        self.assertGreater(a.faults, 0)
        self.assertTrue(a.alive)
        self.assertTrue(math.isfinite(a.altitude))
        self.assertGreater(a.speed, 50.)  # Falling regained flying speed and the tables apply again.

    def test_attitude_of_a_banked_turn(self):
        a = fly(self.new(heading=0.), 2, FlightCommand(heading_deg=90.))
        heading, pitch, roll = a.attitude()
        self.assertGreater(roll, 60.)  # Right turn: right wing down.
        a = fly(self.new(heading=0.), 2, FlightCommand(heading_deg=270.))
        self.assertLess(a.attitude()[2], -60.)

    def test_command_validation(self):
        with self.assertRaises(ValueError):
            FlightCommand(heading_deg=float("nan"))
        with self.assertRaises(ValueError):
            FlightCommand(max_load=0.)
        with self.assertRaises(ValueError):
            FlightCommand(direction=(1., 2.))
        with self.assertRaises(ValueError):
            FlightParams(roll_rate_deg_s=0.)

    def test_model_with_mass_shares_tables_but_not_the_aoa_cache(self):
        heavy = model_with_mass(self.model, self.model.mass*1.2)
        self.assertIs(heavy._aero, self.model._aero)
        self.assertAlmostEqual(heavy.mass, self.model.mass*1.2)
        self.assertIsNot(heavy._aoa_cache, self.model._aoa_cache)
        light = fly(Aircraft(self.model, (0., 0., 3000.), (0., 330., 0.)), 5, FlightCommand(heading_deg=90.))
        loaded = fly(Aircraft(heavy, (0., 0., 3000.), (0., 330., 0.)), 5, FlightCommand(heading_deg=90.))
        self.assertGreater(loaded.state.aoa_deg, light.state.aoa_deg)  # The same load needs more AoA when heavy.
        with self.assertRaises(ValueError):
            model_with_mass(self.model, 0.)

    def test_deterministic(self):
        runs = [fly(self.new(), 10, FlightCommand(heading_deg=130., altitude_m=3500.)).state for _ in range(2)]
        self.assertEqual(runs[0], runs[1])


def legacy_fly(ev, s, h):
    """FMEvader._fly as it was before the physics moved to flight.advance (verbatim copy)."""
    from wt_overlay.escape import _State
    p = ev.pilot
    speed = norm(s.velocity)
    direction = unit(s.velocity)
    gravity_perp = add((0., 0., G), scale(direction, -G*direction[2]))
    demand = gravity_perp
    desired = ev._desired(s.position, s.velocity)
    if desired is not None:
        along = dot(desired, direction)
        perp = add(desired, scale(direction, -along))
        error = math.atan2(norm(perp), along)
        if norm(perp) < 1e-6:
            perp = s.normal if along < 0 else (0., 0., 0.)
        perp = ev._tilted(perp, direction, error, speed, s.position[2])
        if norm(perp) > 1e-9:
            demand = add(demand, scale(unit(perp), speed*error/p.turn_time_constant_s))
    load = min(norm(demand)/G, p.max_load)
    want = unit(demand) if norm(demand) > 1e-9 else s.normal
    phi = math.atan2(dot(direction, cross(s.normal, want)), dot(s.normal, want))
    limit = math.radians(p.roll_rate_deg_s)*h
    normal = rotate(s.normal, direction, max(-limit, min(limit, phi)))
    load *= max(0., dot(normal, want))
    alpha_target = ev._alpha_for(s.position[2], speed, load, s.aoa_deg)
    alpha_mid = alpha_target+(s.aoa_deg-alpha_target)*math.exp(-h/(2*p.load_response_s))
    alpha_end = alpha_target+(s.aoa_deg-alpha_target)*math.exp(-h/p.load_response_s)
    target_kmh = None if ev._hot else p.segment_full(ev._times[-1]-p.start_s)[3]
    throttle, brake = p.throttle_percent, 0.
    if target_kmh is not None:
        excess = speed-target_kmh/3.6
        throttle, brake = ((0., 1.) if excess > 5. else (p.throttle_percent, 0.) if excess < -5.
                           else (s.engine_percent, 0.))
    engine = throttle+(s.engine_percent-throttle)*math.exp(-h/p.engine_response_s)
    step = (ev.model.airbrake_speed or 1.)*h
    airbrake = min(brake, s.airbrake+step) if brake > s.airbrake else max(brake, s.airbrake-step)
    thrust, drag, lift, alpha = ev.model.forces_at_aoa(s.position[2], speed, alpha_mid,
                                                      (s.engine_percent+engine)/2, (s.airbrake+airbrake)/2)
    mass = ev.model.mass
    acceleration = add(add(scale(direction, (thrust*math.cos(alpha)-drag)/mass),
                           scale(normal, (lift+thrust*math.sin(alpha))/mass)), (0., 0., -G))
    velocity = add(s.velocity, scale(acceleration, h))
    normal = transport(normal, direction, unit(velocity))
    position = add(s.position, scale(add(s.velocity, velocity), h/2))
    return _State(position, velocity, normal, alpha_end, engine, airbrake)


class KeyboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = aircraft_model("f_15c_golden_eagle", mass_factor=1.3)

    def new(self, altitude=6000., speed=300., **kwargs):
        return Aircraft(self.model, (0., 0., altitude), (0., speed, 0.), keep_history=True, **kwargs)

    def run_keys(self, command, seconds, **kwargs):
        a = self.new(**kwargs)
        a.command = command
        rows = []
        for _ in range(round(seconds/H)):
            a.step()
            rows.append((a.load, a.attitude()[2], a._roll_rate, a.state.engine_percent, a.state.airbrake))
        return a, rows

    def test_structure_comes_from_the_fm_file(self):
        structure = self.model.structure
        self.assertEqual((structure.crit_neg_n, structure.crit_pos_n), (-400000., 1190000.))
        self.assertEqual((structure.mult_neg, structure.mult_pos), (.9, .9))   # lower value of each overloadMult pair
        n_neg, n_pos = structure.limits(self.model.mass)
        self.assertAlmostEqual(n_pos, .9*1190000./(self.model.mass*9.80665), places=9)
        self.assertAlmostEqual(n_neg, -.9*400000./(self.model.mass*9.80665), places=9)
        self.assertGreater(structure.limits(self.model.mass/1.3)[1], n_pos)       # lighter: more allowed
        self.assertEqual(Aircraft(self.model, (0., 0., 6000.), (0., 300., 0.)).load_limits, (n_neg, n_pos))

    def test_every_catalog_aircraft_in_the_match_model_has_sane_limits(self):
        import json
        weights = json.loads((Path(__file__).resolve().parents[1]/"data"/"match"/"top_tier.json").read_text())
        for aircraft in weights["aircraft_frequency"]["weights"]:
            model = aircraft_model(aircraft, mass_factor=1.3)
            n_neg, n_pos = model.structure.limits(model.mass)
            self.assertTrue(3. < n_pos < 12., (aircraft, n_pos))
            self.assertTrue(-6. < n_neg < -.5, (aircraft, n_neg))
            self.assertIsNotNone(model.structure.crit_pos_n, aircraft)

    def test_read_structure_cases(self):
        self.assertEqual(read_structure({}).limits(10000.), DEFAULT_LOAD_LIMITS)
        raw = {"Aerodynamics": {"WingPlaneSweep0": {"Strength": {"CritOverload": [-100000., 500000.]}}},
               "Instructor": {"limitOverload": True, "overloadMult": [.8, .9, .7, .95], "limitLoadfactor": True,
                              "loadFactorLimit": [-4., 6.]}}
        s = read_structure(raw)
        self.assertEqual((s.mult_pos, s.mult_neg, s.cap_neg, s.cap_pos), (.8, .7, -4., 6.))
        n_neg, n_pos = s.limits(5000.)
        self.assertEqual(n_pos, 6.)                                   # 0.8 x 500000 / (5000 g) = 8.2, capped at 6
        self.assertAlmostEqual(n_neg, -.7*100000./(5000*9.80665), places=9)
        off = read_structure({"Instructor": {"limitOverload": False, "overloadMult": [.5, .5, .5, .5]},
                              "Aerodynamics": {"WingPlane": {"Strength": {"CritOverload": [-1., 490332.5]}}}})
        self.assertAlmostEqual(off.limits(10000.)[1], 5., places=6)     # no multiplier when the Instructor does not limit
        plain = Aircraft(ManeuverModel(load_aircraft("f_16c_block_50"), 12000.), (0., 0., 6000.), (0., 300., 0.))
        self.assertEqual(plain.load_limits, DEFAULT_LOAD_LIMITS)         # a model built without the FM file's structure

    def test_full_pull_reaches_the_airframe_limit_and_not_beyond(self):
        a, rows = self.run_keys(KeyboardCommand(pitch=1), 5)
        n_neg, n_pos = a.load_limits
        loads = [r[0] for r in rows]
        self.assertGreater(max(loads), .95*n_pos)
        self.assertLess(max(loads), 1.03*n_pos)
        self.assertGreater(loads[round(1.5/H)], .7*n_pos)             # bang-bang: it is there within a second or two
        a, rows = self.run_keys(KeyboardCommand(pitch=-1), 4)
        self.assertLess(min(r[0] for r in rows), .95*n_neg)
        self.assertGreater(min(r[0] for r in rows), 1.05*n_neg)
        pulled = self.new()
        pulled.command = KeyboardCommand(pitch=1)
        for _ in range(round(2/H)):
            pulled.step()
        pulled.command = KeyboardCommand(pitch=0)
        for _ in range(round(3/H)):
            pulled.step()
        self.assertLess(pulled.load, 2.)                              # released: back toward 1 g

    def test_roll_rate_and_authority(self):
        a, rows = self.run_keys(KeyboardCommand(roll=1), 2)
        self.assertAlmostEqual(math.degrees(max(r[2] for r in rows)), 120., delta=.5)
        self.assertTrue(70. < rows[round(1/H)-1][1] < 95.)             # right wing down, about 80 degrees in a second
        left, rows_left = self.run_keys(KeyboardCommand(roll=-1), 1)
        self.assertLess(rows_left[-1][1], -60.)
        half, rows_half = self.run_keys(KeyboardCommand(roll=1, authority=.5), 2)
        self.assertAlmostEqual(math.degrees(max(r[2] for r in rows_half)), 60., delta=.5)
        _, pull_half = self.run_keys(KeyboardCommand(pitch=1, authority=.5), 4)
        n_pos = a.load_limits[1]
        self.assertAlmostEqual(max(r[0] for r in pull_half), 1.+.5*(n_pos-1.), delta=.35)
        hold, rows_hold = self.run_keys(KeyboardCommand(), 3)
        self.assertLess(abs(rows_hold[-1][1]), .5)                    # no input, no roll
        self.assertLess(max(abs(r[2]) for r in rows_hold), 1e-9)

    def test_throttle_and_airbrake_keys(self):
        a, rows = self.run_keys(KeyboardCommand(throttle=-1), 4)
        engine = [r[3] for r in rows]
        self.assertLess(engine[-1], 15.)
        self.assertTrue(all(x >= y-1e-9 for x, y in zip(engine, engine[1:])))
        low = self.new(engine_percent=40.)
        low.command = KeyboardCommand(throttle=1)
        for _ in range(round(4/H)):
            low.step()
        self.assertGreater(low.state.engine_percent, 90.)
        self.assertLessEqual(low.state.engine_percent, self.model.max_throttle+1e-9)
        _, hold = self.run_keys(KeyboardCommand(), 3)
        self.assertAlmostEqual(hold[-1][3], 110., delta=1.)
        _, brake = self.run_keys(KeyboardCommand(airbrake=True), 3)
        self.assertGreater(brake[-1][4], .99)
        _, off = self.run_keys(KeyboardCommand(airbrake=False), 3)
        self.assertEqual(off[-1][4], 0.)

    def test_switching_between_the_two_command_types(self):
        a = self.new()
        a.command = FlightCommand(heading_deg=0., altitude_m=6000., throttle_percent=70.)
        for _ in range(round(4/H)):
            a.step()
        engine = a.state.engine_percent
        a.command = KeyboardCommand(roll=1)
        a.step()
        self.assertAlmostEqual(a.state.engine_percent, engine, delta=1.5)    # keyboard mode starts where the engine is
        for _ in range(round(1/H)):
            a.step()
        a.command = FlightCommand(heading_deg=0., altitude_m=6000.)
        for _ in range(round(20/H)):
            a.step()
        self.assertEqual(a.faults, 0)
        self.assertLess(abs(wrap(a.attitude()[0])), 2.)
        self.assertLess(abs(a.altitude-6000.), 100.)

    def test_the_structure_cap_can_limit_a_flight_command(self):
        for limited in (False, True):
            a = self.new(speed=330., limit_structure=limited)
            a.command = FlightCommand(heading_deg=120., max_load=9.)
            top = 0.
            for _ in range(round(8/H)):
                a.step()
                top = max(top, a.load)
            if limited:
                self.assertLess(top, 1.03*a.load_limits[1])
            else:
                self.assertGreater(top, 1.2*a.load_limits[1])

    def test_keyboard_command_validation(self):
        for bad in (dict(roll=2), dict(pitch=-2), dict(throttle=1.5), dict(authority=0.), dict(authority=1.5),
                    dict(authority=float("nan"))):
            with self.assertRaises(ValueError):
                KeyboardCommand(**bad)
        self.assertEqual(KeyboardCommand().authority, 1.)

    def test_keyboard_flight_is_deterministic_and_stays_in_the_tables(self):
        runs = []
        for _ in range(2):
            a = self.new()
            for k in range(round(30/H)):
                a.step(KeyboardCommand(roll=(1, 0, -1)[k//48 % 3], pitch=(1, 1, 0)[k//96 % 3], throttle=0, airbrake=k % 400 < 80))
            runs.append(a.state)
            self.assertEqual(a.faults, 0)
        self.assertEqual(runs[0], runs[1])


class FMEvaderUnchangedTests(unittest.TestCase):
    """The shared physics must leave FMEvader bit-identical: compare against the pre-refactor body."""

    def run_case(self, aircraft, mass, pilot, seconds):
        from wt_overlay.fm import load_aircraft
        from wt_overlay.turn import ManeuverModel
        model = ManeuverModel(load_aircraft(aircraft), mass)
        new = FMEvader(HeadOn(), pilot, model)
        old = FMEvader(HeadOn(), pilot, model)
        old._fly = lambda s, h: legacy_fly(old, s, h)
        for ev in (new, old):
            for k in range(1, round(seconds*48)+1):
                ev.observe_missile((k-1)/48, (0., 8000., 0.), (0., 0., 0.))
                ev.state_at(k/48)
        self.assertEqual(new._states, old._states)
        self.assertGreater(len(new._states), seconds*40)

    def test_beam_drag_and_a_slow_plan(self):
        self.run_case("f_16c_block_50", 12000., EvasionPilot("beam", 0.), 6)
        self.run_case("f_16c_block_50", 12000., EvasionPilot("drag", 0., dive_deg=20., speed_kmh=700.), 8)
        self.run_case("su_30sm2", 25000., EvasionPilot("beam", 1., plane_deg=45., dive_deg=20.,
                                                       then=((3., 0., 0., 0., 800.),)), 6)


if __name__ == "__main__":
    unittest.main()
