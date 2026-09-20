"""Geometry, forces, bounded keyboard search and stale-result acceptance checks."""
from concurrent.futures import Future
from dataclasses import asdict, replace
import math
from threading import Event
import unittest
from unittest.mock import patch

from wt_overlay.app import OverlayController
from wt_overlay.contracts import FlightState, G, KeyboardTurnSettings, PerformanceCondition
from wt_overlay.fm import load_aircraft
from wt_overlay.turn import (Action, Cancelled, ManeuverModel, Motion, TurnPlan, TurnSession, TurnStep,
                             VelocityGoal, angle, dot, flight_frame, norm, rotate, search_turn,
                             transport, unit, validate_settings)


def sample(t=0, **changes):
    return replace(FlightState(t, True, altitude_m=5000, tas_mps=250, ias_mps=200,
        vertical_speed_mps=0, pitch_deg=4, roll_deg=0, heading_deg=0,
        aoa_deg=4, aos_deg=0, mass_kg=23000, aircraft_id="su_27sm", throttle_percent=110), **changes)


def telemetry_for_motion(model, motion, t):
    alpha = model.forces(motion.altitude, motion.speed, motion.load)[3]
    v, n = unit(motion.velocity), motion.normal
    forward = tuple(a*math.cos(alpha)+b*math.sin(alpha) for a, b in zip(v, n))
    up = tuple(-a*math.sin(alpha)+b*math.cos(alpha) for a, b in zip(v, n))
    heading, pitch = math.atan2(forward[0], forward[1]), math.asin(forward[2])
    right = (math.cos(heading), -math.sin(heading), 0)
    level_up = (-math.sin(pitch)*math.sin(heading), -math.sin(pitch)*math.cos(heading), math.cos(pitch))
    roll = math.atan2(dot(up, right), dot(up, level_up))
    return sample(t, altitude_m=motion.altitude, tas_mps=motion.speed, vertical_speed_mps=motion.velocity[2],
        heading_deg=math.degrees(heading), pitch_deg=math.degrees(pitch), roll_deg=math.degrees(roll),
        aoa_deg=math.degrees(alpha), throttle_percent=motion.throttle_percent)


class GeometryTests(unittest.TestCase):
    def test_velocity_corrects_aoa_sideslip_and_uses_vertical_speed(self):
        velocity, normal = flight_frame(sample())
        self.assertAlmostEqual(angle(velocity, (0, 1, 0)), 0)
        self.assertAlmostEqual(dot(unit(velocity), normal), 0)
        self.assertAlmostEqual(norm(velocity), 250)
        velocity, normal = flight_frame(sample(roll_deg=90, pitch_deg=0, aoa_deg=10))
        self.assertLess(velocity[0], 0)  # Velocity lies left of the nose at right bank.
        self.assertGreater(normal[0], .9)
        self.assertAlmostEqual(angle(velocity, (0, 1, 0)), 10)
        velocity, _ = flight_frame(sample(vertical_speed_mps=100))
        self.assertAlmostEqual(velocity[2], 100)

    def test_heading_wrap_roll_wrap_and_inverted_frame(self):
        left, _ = flight_frame(sample(heading_deg=359))
        right, _ = flight_frame(sample(heading_deg=1))
        self.assertAlmostEqual(angle(left, right), 2)
        _, a = flight_frame(sample(roll_deg=179, aoa_deg=0, pitch_deg=0))
        _, b = flight_frame(sample(roll_deg=-179, aoa_deg=0, pitch_deg=0))
        self.assertAlmostEqual(angle(a, b), 2)
        self.assertLess(a[2], -.99)
        self.assertAlmostEqual(norm(transport((0, 0, 1), (0, 1, 0), (1, 0, 0))), 1)

    def test_beam_goal_is_not_ninety_degrees_from_start(self):
        threat = VelocityGoal((1, 0, 0), kind="beam")
        self.assertEqual(threat.remaining((0, 1, 0)), 0)
        self.assertGreater(threat.remaining((1, 0, 0)), 80)
        turn = VelocityGoal((0, 1, 0), 90)
        self.assertEqual(turn.remaining((1, 0, 0)), 0)

    def test_missing_pose_and_near_vertical_are_not_fabricated(self):
        for change in ({"heading_deg": None}, {"aoa_deg": None}, {"roll_deg": math.nan},
                       {"vertical_speed_mps": 250}, {"aos_deg": 30}, {"valid": False}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                flight_frame(sample(**change))

    def test_missing_readings_are_named_and_low_speed_has_a_separate_reason(self):
        with self.assertRaisesRegex(ValueError, "缺少转向读数：航向 compass、滚转 aviahorizon_roll"):
            flight_frame(sample(heading_deg=None, roll_deg=None))
        with self.assertRaisesRegex(ValueError, "缺少转向读数：侧滑 AoS"):
            flight_frame(sample(aos_deg=None))
        with self.assertRaisesRegex(ValueError, "低于转向计算下限"):
            flight_frame(sample(tas_mps=40))


class ManeuverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fm = load_aircraft("su_27sm")

    def setUp(self):
        self.model = ManeuverModel(self.fm, 23000)
        self.settings = KeyboardTurnSettings()
        self.initial = Motion(5000, (0, 250, 0), (0, 0, 1), 0, 1)

    def test_signed_lift_and_original_static_force_agreement(self):
        for load in (-2, 0, 1, 5, 9):
            thrust, drag, lift, alpha = self.model.forces(5000, 250, load)
            self.assertTrue(all(math.isfinite(v) for v in (thrust, drag, lift, alpha)))
            point = self.fm.evaluate(PerformanceCondition(5000, 250, 23000, aoa_deg=math.degrees(alpha)))
            self.assertTrue(point.valid)
            self.assertAlmostEqual(point.drag_n, drag)
            self.assertAlmostEqual(point.lift_n, lift)
            self.assertAlmostEqual(point.thrust_n, thrust)
        self.assertLess(self.model.forces(5000, 250, -2)[2], 0)

    def test_roll_and_load_build_up_and_decay_instead_of_jumping(self):
        x = self.model.step(self.initial, Action(1, 1), .1, self.settings)
        self.assertGreater(x.roll_rate, 0)
        self.assertLess(x.roll_rate, math.radians(self.settings.roll_rate_deg_s))
        self.assertGreater(x.load, 1)
        self.assertLess(x.load, 9)
        y = self.model.step(x, Action(0, 0), .1, self.settings)
        self.assertLess(y.roll_rate, x.roll_rate)
        self.assertLess(y.load, x.load)
        self.assertAlmostEqual(dot(unit(y.velocity), y.normal), 0, places=10)
        self.assertAlmostEqual(norm(y.normal), 1)

    def test_throttle_ramp_spool_and_force_endpoints(self):
        military = self.fm.evaluate(PerformanceCondition(5000, 250, 23000, afterburner=False)).thrust_n
        maximum = self.fm.evaluate(PerformanceCondition(5000, 250, 23000)).thrust_n
        self.assertEqual(self.model.forces(5000, 250, 1, 0)[0], 0)
        self.assertAlmostEqual(self.model.forces(5000, 250, 1, 50)[0], military/2)
        self.assertAlmostEqual(self.model.forces(5000, 250, 1, 100)[0], military)
        self.assertAlmostEqual(self.model.forces(5000, 250, 1, 110)[0], maximum)
        reduced = self.model.step(self.initial, Action(0, 0, -1), .1, self.settings)
        self.assertEqual(reduced.throttle_percent, 105)
        self.assertGreater(reduced.engine_throttle_percent, reduced.throttle_percent)
        self.assertLess(reduced.engine_throttle_percent, 110)
        coast = self.model.step(reduced, Action(0, 0, 0), .1, self.settings)
        self.assertEqual(coast.throttle_percent, reduced.throttle_percent)
        self.assertLess(coast.engine_throttle_percent, reduced.engine_throttle_percent)
        dry = ManeuverModel(self.fm, 23000, afterburner=False)
        initial = replace(self.initial, throttle_percent=99, engine_throttle_percent=99)
        self.assertEqual(dry.step(initial, Action(0, 0, 1), .1, self.settings).throttle_percent, 100)
        idle = replace(self.initial, throttle_percent=1, engine_throttle_percent=20)
        self.assertEqual(self.model.step(idle, Action(0, 0, -1), .1, self.settings).throttle_percent, 0)

    def test_search_can_choose_reduced_throttle_and_respects_speed_floor(self):
        # A load-limited, fast case: slowing down can increase angular progress.
        fast = replace(self.initial, velocity=(0, 400, 0), normal=(1, 0, 0), load=9)
        s = replace(self.settings, angle_deg=90, horizon_s=12)
        result = search_turn(self.model, fast, VelocityGoal((0, 1, 0), 90), s, 4500, budget_s=3)
        self.assertTrue(result.reached)
        self.assertIsNotNone(result.action)
        self.assertEqual(result.action.throttle, -1)
        hold, cut = fast, fast
        for _ in range(20):
            hold = self.model.step(hold, Action(0, 1, 0), .15, s)
            cut = self.model.step(cut, Action(0, 1, -1), .15, s)
        self.assertLess(cut.speed, hold.speed)
        self.assertGreater(angle(cut.velocity, fast.velocity), angle(hold.velocity, fast.velocity))
        self.assertLess(cut.energy, hold.energy)
        constrained = replace(s, minimum_tas_mps=405)
        with self.assertRaises(ValueError):
            search_turn(self.model, fast, VelocityGoal((0, 1, 0), 30), constrained, 4500)

    def test_short_step_energy_balance_and_initial_bank_change_trajectory(self):
        dt = 1e-4
        t, d, _, alpha = self.model.forces(5000, 250, 1)
        x = self.model.step(self.initial, None, dt, self.settings)
        expected = 250*(t*math.cos(alpha)-d)/(23000*G)
        self.assertAlmostEqual((x.energy-self.initial.energy)/dt, expected, delta=.02)
        right = replace(self.initial, normal=(1, 0, 0), load=5)
        turned = self.model.step(right, None, .1, self.settings)
        self.assertGreater(turned.velocity[0], 0)
        upside_down = self.model.step(replace(self.initial, normal=(0, 0, -1)), None, .1, self.settings)
        self.assertLess(upside_down.velocity[2], x.velocity[2])

    def test_all_requested_angles_have_bounded_candidates(self):
        for target in (30, 45, 90, 120):
            with self.subTest(target=target):
                s = replace(self.settings, angle_deg=target)
                result = search_turn(self.model, self.initial, VelocityGoal((0, 1, 0), target), s, 4500, budget_s=3)
                self.assertTrue(result.reached)
                self.assertIsNotNone(result.action)
                self.assertGreater(result.duration_s, s.reaction_s)
                self.assertLessEqual(result.duration_s, s.horizon_s)
                self.assertGreater(len(result.steps), 0)
                self.assertLessEqual(len(result.steps), 4)
                self.assertEqual(result.action, result.steps[0].action)
                self.assertTrue(all(x.duration_s >= s.hold_s-1e-8 for x in result.steps[:-1]))

    def test_cancellation_invalid_limits_and_incomplete_search_have_no_eta(self):
        stop = Event(); stop.set()
        with self.assertRaises(Cancelled):
            search_turn(self.model, self.initial, VelocityGoal((0, 1, 0)), self.settings, 4500, stop)
        with self.assertRaises(ValueError):
            search_turn(self.model, self.initial, VelocityGoal((0, 1, 0)), self.settings, 5100)
        result = search_turn(self.model, self.initial, VelocityGoal((0, 1, 0)), self.settings, 4500, budget_s=0)
        self.assertFalse(result.reached)
        self.assertIsNone(result.duration_s)
        self.assertIsNone(result.action)
        for change in ({"hold_s": .01}, {"max_load": math.nan}, {"angle_deg": 60}, {"min_load": 1}):
            with self.assertRaises(ValueError):
                validate_settings(replace(self.settings, **change))

    def test_receding_search_completes_from_different_initial_attitudes(self):
        # Closed-loop surrogate test, not an in-game maneuver validation.
        for normal in ((0, 0, 1), (1, 0, 0), (0, 0, -1)):
            with self.subTest(normal=normal):
                state = replace(self.initial, normal=normal)
                goal = VelocityGoal((0, 1, 0), 45)
                for _ in range(18):
                    plan = search_turn(self.model, state, goal, self.settings, 4500, budget_s=2)
                    self.assertIsNotNone(plan.action)
                    for _ in range(4):
                        state = self.model.step(state, plan.action, .15, self.settings)
                    self.assertGreaterEqual(state.altitude, 4500)
                    self.assertGreaterEqual(state.speed, self.settings.minimum_tas_mps)
                    if goal.remaining(state.velocity) == 0:
                        break
                self.assertEqual(goal.remaining(state.velocity), 0)


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.fm = load_aircraft("su_27sm")
        self.session = TurnSession()
        self.settings = KeyboardTurnSettings(angle_deg=30)
        self.addCleanup(self.session.close)

    def update(self, t, **changes):
        return self.session.update(sample(t, **changes), self.fm, 23000, True, 0, self.settings)

    def test_anchoring_angle_completion_latch_and_short_loss_recovery(self):
        self.assertEqual(self.update(0).phase, "读取姿态")
        self.update(.1)
        goal = self.session.goal
        self.update(.2, heading_deg=10)
        self.assertEqual(self.session.goal, goal)
        done = self.update(.3, heading_deg=31)
        self.assertEqual(done.phase, "到达")
        self.assertEqual(done.action, "松开机动键")
        self.assertEqual(self.update(.4, heading_deg=5).phase, "到达")
        self.update(.5, valid=False)
        self.assertFalse(self.session.require_restart)
        self.assertEqual(self.session.goal, goal)
        self.assertEqual(self.update(.6).phase, "读取姿态")
        self.assertEqual(self.update(.7).phase, "到达")
        self.session.reset()
        self.assertEqual(self.update(.8).phase, "读取姿态")

    def test_sampling_hiccup_and_invalid_pose_recover_without_moving_origin(self):
        self.update(0); self.update(.1)
        goal = self.session.goal
        self.assertEqual(self.update(.9, heading_deg=5).phase, "读取姿态")
        self.assertNotEqual(self.update(1., heading_deg=6).phase, "重新开始")
        bad = self.update(1.1, aoa_deg=None)
        self.assertFalse(bad.available)
        self.assertTrue(bad.reason)
        self.update(1.2)
        self.assertNotEqual(self.update(1.3).phase, "重新开始")
        self.assertIs(self.session.goal, goal)

    def test_limits_show_cause_and_resume_without_resetting_altitude_floor(self):
        self.update(0); self.update(.1)
        goal, floor = self.session.goal, self.session.floor
        result = self.update(.2, tas_mps=90)
        self.assertEqual(result.phase, "速度不足")
        self.assertTrue(result.reason)
        self.update(.3)
        self.assertNotEqual(self.update(.4).phase, "重新开始")
        self.assertIs(self.session.goal, goal)
        self.assertEqual(self.session.floor, floor)
        self.assertEqual(self.update(.5, altitude_m=4400).phase, "高度不足")

    def test_long_data_loss_requires_restart_and_keeps_explanation(self):
        self.update(0); self.update(.1)
        self.update(.2, valid=False)
        result = self.update(3.)
        self.assertEqual(result.phase, "重新开始")
        self.assertIn("中断", result.reason)
        self.assertEqual(self.update(3.1).reason, result.reason)
        self.session.reset()
        self.assertEqual(self.update(3.2).phase, "读取姿态")
        self.assertNotEqual(self.update(3.3).phase, "重新开始")

    def test_missing_throttle_pauses_and_actual_throttle_seeds_plan(self):
        result = self.update(0, throttle_percent=None)
        self.assertFalse(result.available)
        self.assertIn("油门", result.reason)
        self.update(.1, throttle_percent=70)
        self.update(.2, throttle_percent=70)
        self.assertEqual(self.session.engine_throttle, 70)

    def test_observed_load_is_not_body_ny_and_mass_does_not_move_origin(self):
        self.update(0, normal_load_g=99)
        self.update(.1, normal_load_g=99)
        goal = self.session.goal
        self.assertLess(self.session.model.observed_load(5000, 250, 4), 9)
        self.session.update(sample(.2, heading_deg=5), self.fm, 23500, True, 0, self.settings)
        self.assertEqual(self.session.goal, goal)

    def test_stale_plan_is_rejected_without_old_command(self):
        self.update(0); self.update(.1)
        self.session.cancel.set()
        velocity, normal = flight_frame(sample())
        future = Future()
        future.set_result(TurnPlan(Motion(5000, velocity, normal, 0, 2), Action(1, 1), 4, -100, True))
        self.session.future = future
        result = self.update(.2, heading_deg=20)
        self.assertFalse(result.available)
        self.assertEqual(result.action, "")

    def test_matching_action_is_published_without_unverified_eta(self):
        self.update(0); self.update(.1)
        self.session.cancel.set()
        velocity, normal = flight_frame(sample())
        future = Future()
        load = self.session.model.observed_load(5000, 250, 4)
        future.set_result(TurnPlan(Motion(5000, velocity, normal, 0, load), Action(1, 1), 4, -100, True))
        self.session.future = future
        result = self.update(.2)
        self.assertTrue(result.available)
        self.assertEqual(result.action, "右滚＋拉杆")
        self.assertIsNone(result.duration_s)  # Rebased sequence has not reached the goal.
        self.assertEqual(result.step_count, 1)
        self.assertEqual(result.next_action, "继续规划")

    def test_throttle_recommendation_and_stale_throttle_rejection(self):
        self.update(0); self.update(.1)
        self.session.cancel.set()
        velocity, normal = flight_frame(sample())
        load = self.session.model.observed_load(5000, 250, 4)
        future = Future()
        future.set_result(TurnPlan(Motion(5000, velocity, normal, 0, load), Action(0, 1, -1), 4, -100, True))
        self.session.future = future
        result = self.update(.2)
        self.assertTrue(result.available)
        self.assertEqual(result.throttle_command, -1)
        self.assertEqual(result.throttle_percent, 110)
        self.assertEqual(result.target_throttle_percent, 50)
        result = self.update(.3, throttle_percent=50)
        self.assertFalse(result.available)
        self.assertIsNone(result.target_throttle_percent)

    def test_normal_rolling_follows_committed_sequence_and_previews_next_action(self):
        self.settings = replace(self.settings, angle_deg=120)
        self.update(0); self.update(.1)
        self.session.cancel.set()
        v, n = flight_frame(sample())
        load = self.session.model.observed_load(5000, 250, 4)
        initial = Motion(5000, v, n, 0, load)
        future = Future()
        steps = (TurnStep(Action(1, 1), 1.2), TurnStep(Action(0, 1), 1.2))
        future.set_result(TurnPlan(initial, steps[0].action, None, None, False, steps=steps))
        self.session.future = future
        result = self.update(.2)
        self.assertTrue(result.available)
        self.assertIn("停止滚转", result.next_action)
        execution, origin = self.session.execution, self.session.goal
        for i in range(1, 18):
            elapsed = i*.1
            motion = execution.reference(elapsed)
            state = telemetry_for_motion(self.session.model, motion, .2+elapsed)
            result = self.session.update(state, self.fm, 23000, True, 0, self.settings)
            self.assertTrue(result.available, (elapsed, result))
            self.assertIs(self.session.execution, execution)
            self.assertIs(self.session.goal, origin)
            self.assertIsNone(self.session.future)
            self.assertEqual(result.roll_command, 1 if elapsed < 1.5-1e-8 else 0)
            if i == 11:
                self.assertGreater(angle(initial.normal, motion.normal), 30)
        self.assertEqual(result.step_index, 2)

    def test_model_limit_follows_live_progress_and_missing_pose_withdraws_cue(self):
        self.settings = replace(self.settings, angle_deg=90)
        self.update(0); self.update(.1)
        result = self.update(.2, heading_deg=20, aoa_deg=35)
        self.assertTrue(result.available)
        self.assertEqual(result.phase, "机动跟随")
        self.assertEqual(result.action, "保持机动")
        self.assertIsNone(result.duration_s)
        self.assertIsNone(result.step_index)
        self.assertAlmostEqual(result.turned_deg, 20)
        self.assertAlmostEqual(result.remaining_deg, 70)
        self.assertFalse(result.progress_stale)
        missing = self.update(.3, pitch_deg=None)
        self.assertEqual(missing.turned_deg, result.turned_deg)
        self.assertTrue(missing.progress_stale)
        self.assertEqual(missing.phase, "缺少姿态")
        self.assertFalse(missing.available)
        self.assertEqual(missing.action, "")

    def test_j16_high_aoa_follows_progress_previews_release_and_completes(self):
        # AoA, bank, pitch, Ny and Vy match the screenshot. Height/speed/heading
        # are explicit fixture choices because those readings are not in the crop.
        self.fm = load_aircraft("j_16")
        self.settings = replace(self.settings, angle_deg=90)
        def maneuver(t, heading):
            return self.update(t, aircraft_id="j_16", aoa_deg=21.6, aos_deg=.1,
                pitch_deg=-8.5, roll_deg=-90.3, normal_load_g=8.3,
                vertical_speed_mps=-30.9, heading_deg=heading)
        maneuver(0, 0)
        first = maneuver(.1, 0)
        self.assertEqual(first.phase, "机动跟随")
        self.assertIn("共同失速前", first.reason)
        self.assertIsNone(self.session.future)
        for i in range(1, 30):
            result = maneuver(.1+i*.2, i*3)
            self.assertTrue(result.available)
            self.assertEqual(result.phase, "机动跟随")
            self.assertIsNone(result.duration_s)
            self.assertIsNone(result.step_index)
            self.assertIsNone(result.target_throttle_percent)
            if i == 10:
                self.assertEqual(result.action, "保持机动")
        self.assertEqual(result.action, "准备松键")
        self.assertGreater(result.remaining_deg, 0)
        reached = maneuver(6.1, 99)
        self.assertEqual(reached.phase, "到达")
        self.assertEqual(reached.action, "松开机动键")
        self.assertEqual(reached.remaining_deg, 0)

    def test_following_detects_no_progress_and_respects_real_height_speed_limits(self):
        self.settings = replace(self.settings, angle_deg=90)
        for i in range(6):
            result = self.update(i*.1, aoa_deg=35)
        self.assertEqual(result.action, "检查转向")
        self.assertEqual(result.next_action, "转角未增加")
        self.assertFalse(self.update(.6, aoa_deg=35, tas_mps=90).available)
        self.assertEqual(self.update(.7, aoa_deg=35, tas_mps=90).phase, "速度不足")
        self.assertEqual(self.update(.8, aoa_deg=35, altitude_m=4400).phase, "高度不足")

    def test_fm_load_limit_uses_progress_cues_without_reinterpreting_body_ny(self):
        self.settings = replace(self.settings, angle_deg=90)
        self.update(0, tas_mps=400, aoa_deg=10, normal_load_g=8)
        result = self.update(.1, tas_mps=400, aoa_deg=10, normal_load_g=8)
        self.assertEqual(result.phase, "机动跟随")
        self.assertIn("载荷超出", result.reason)
        self.assertTrue(result.available)
        self.assertGreater(self.session.model.observed_load(5000, 400, 10), self.settings.max_load)
        self.assertIsNone(result.duration_s)
        self.assertIsNone(self.session.future)

    def test_following_continues_while_model_recovers_and_accepts_a_new_sequence(self):
        self.update(0, aoa_deg=35); self.update(.1, aoa_deg=35)
        self.assertTrue(self.session.following)
        # Pending planning must not cancel itself to keep a progress cue visible.
        future = Future()
        self.session.future = future
        self.session.last_submit = .15
        pending = self.update(.2, aoa_deg=4)
        self.assertEqual(pending.phase, "机动跟随")
        self.assertIs(self.session.future, future)
        self.assertFalse(future.cancelled())
        v, n = flight_frame(sample())
        load = self.session.model.observed_load(5000, 250, 4)
        candidate = TurnPlan(Motion(5000, v, n, 0, load), Action(1, 1), None, None, False)
        future.set_result(candidate)
        with patch("wt_overlay.turn.prepare_execution", side_effect=ValueError("动作序列超出所设限制")):
            rejected = self.update(.3)
        self.assertEqual(rejected.phase, "机动跟随")
        self.assertTrue(rejected.available)
        self.session.future = Future()
        self.session.future.set_result(candidate)
        self.session.last_submit = .35
        resumed = self.update(.4)
        self.assertTrue(resumed.available)
        self.assertEqual(resumed.phase, "转向")
        self.assertFalse(self.session.following)
        self.assertEqual(resumed.step_index, 1)


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.state = sample()
        self.controller = OverlayController(client=self)
        self.addCleanup(self.controller.stop)

    def poll(self, time_s=None):
        return replace(self.state, time_s=time_s)

    def test_mutually_exclusive_directors_and_disconnect_recovery(self):
        c = self.controller
        c.submit({"action": "climb_enabled", "enabled": True})
        c.tick(0)
        c.submit({"action": "turn_enabled", "enabled": True})
        self.assertFalse(c.tick(.1).climb_enabled)
        snapshot = c.tick(.2)
        self.assertTrue(snapshot.turn_enabled)
        self.assertIsNone(snapshot.climb)
        self.assertIsNotNone(c._turn_session.goal)
        model, goal = c.model, c._turn_session.goal
        self.state = replace(self.state, valid=False)
        self.assertFalse(c.tick(.3).turn.available)
        self.state = sample()
        self.assertEqual(c.tick(.4).turn.phase, "读取姿态")
        self.assertIs(c.model, model)
        self.assertIs(c._turn_session.goal, goal)
        c.submit({"action": "turn_restart"})
        self.assertEqual(c.tick(.5).turn.phase, "读取姿态")
        c.submit({"action": "climb_enabled", "enabled": True})
        self.assertFalse(c.tick(.6).turn_enabled)
        self.assertIsNone(c._turn_session.future)

    def test_turn_settings_validation_and_stale_snapshot_mask(self):
        c = self.controller
        with self.assertRaises(ValueError):
            c.submit({"action": "turn_target", "settings": {"angle_deg": 11}})
        c.submit({"action": "turn_target", "settings": asdict(KeyboardTurnSettings(angle_deg=45))})
        c.submit({"action": "turn_enabled", "enabled": True})
        c.tick(0); c.tick(.1)
        self.assertEqual(c._turn_session.goal.angle_deg, 45)
        with patch("wt_overlay.app.time.monotonic", return_value=c._published_at+3):
            snapshot = c.get_snapshot()
        self.assertFalse(snapshot.turn.available)
        self.assertEqual(snapshot.turn.action, "")


if __name__ == "__main__":
    unittest.main()
