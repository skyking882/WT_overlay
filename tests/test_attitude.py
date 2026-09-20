"""Calibrated attitude kinematics and the user's missing-horizon J-16 capture."""
from dataclasses import replace
import math
import unittest

from wt_overlay.attitude import AttitudeEstimator
from wt_overlay.app import OverlayController
from wt_overlay.telemetry import parse_telemetry


def j16(t=0, **changes):
    # Relevant fields copied from the user's actual two-endpoint capture.
    state = {"valid": True, "H, m": 5058, "TAS, km/h": 1506, "IAS, km/h": 1164,
             "M": 1.30, "AoA, deg": -.9, "AoS, deg": 0., "Ny": 1.04,
             "Vy, m/s": 45.4, "Wx, deg/s": 0, "Mfuel, kg": 7637,
             "throttle 1, %": 110, "thrust 1, kgs": 13881, "thrust 2, kgs": 13881}
    sample = parse_telemetry(state, {"valid": True, "army": "air", "type": "j_16",
        "compass": 262.117523, "throttle": 1.1, "stick_elevator": -.101141}, t)
    return replace(sample, **changes)


class AttitudeTests(unittest.TestCase):
    def setUp(self):
        self.estimator = AttitudeEstimator()

    def test_user_capture_requires_explicit_level_calibration(self):
        raw = j16()
        self.assertIsNone(raw.pitch_deg)
        self.assertIsNone(self.estimator.update(raw).roll_deg)
        estimated = self.estimator.update(raw, calibrate_sign=1)
        self.assertAlmostEqual(estimated.roll_deg, 0)
        self.assertAlmostEqual(estimated.pitch_deg, math.degrees(math.asin(45.4/(1506/3.6)))-.9)
        self.assertIsNone(raw.pitch_deg)
        self.assertEqual(estimated.raw_indicators, raw.raw_indicators)
        self.assertTrue(self.estimator.estimated)

    def test_roll_rate_integration_and_direction_selection(self):
        for sign in (-1, 1):
            estimator = AttitudeEstimator()
            estimator.update(j16(), calibrate_sign=sign)
            for i in range(1, 11):
                raw = j16(i*.1, raw_state={"Wx, deg/s": 90})
                estimate = estimator.update(raw)
            self.assertAlmostEqual(estimate.roll_deg, sign*85.5, places=6)
            self.assertTrue(estimator.estimated)

    def test_body_rate_yaw_coupling_and_compass_wrap_keep_wings_level(self):
        pitch, aoa, speed = 10., 2., 250.
        rate = -10*math.sin(math.radians(pitch))
        def sample(t):
            return j16(t, heading_deg=(359.7+10*t)%360, tas_mps=speed, aoa_deg=aoa,
                vertical_speed_mps=speed*math.sin(math.radians(pitch-aoa)), raw_state={"Wx, deg/s": rate})
        self.estimator.update(sample(0), calibrate_sign=1)
        for i in range(1, 21):
            result = self.estimator.update(sample(i*.1))
        self.assertAlmostEqual(result.roll_deg, 0, places=6)
        self.assertAlmostEqual(result.pitch_deg, pitch, places=6)

    def test_loss_aircraft_switch_timeout_and_raw_attitude_priority(self):
        self.estimator.update(j16(), calibrate_sign=1)
        self.assertIsNone(self.estimator.update(j16(.1, valid=False)).pitch_deg)
        self.assertIsNone(self.estimator.update(j16(.2)).pitch_deg)
        self.assertIn("重新校准", self.estimator.reason)
        self.estimator.update(j16(.3), calibrate_sign=1)
        self.assertIsNone(self.estimator.update(j16(.4, aircraft_id="j_11b")).pitch_deg)
        self.estimator.update(j16(.5), calibrate_sign=1)
        self.assertIsNone(self.estimator.update(j16(1.5)).pitch_deg)
        self.estimator.update(j16(2), calibrate_sign=1)
        self.estimator.calibrated_at = -60
        self.assertIsNone(self.estimator.update(j16(2.1)).pitch_deg)
        raw = j16(2.2, pitch_deg=9, roll_deg=75)
        self.assertIs(self.estimator.update(raw), raw)
        self.assertFalse(self.estimator.estimated)

    def test_bad_calibration_near_vertical_and_missing_rate_do_not_invent_pose(self):
        for changes in ({"raw_state": {}}, {"raw_state": {"Wx, deg/s": 40}},
                        {"vertical_speed_mps": 410}, {"aos_deg": 20}):
            with self.subTest(changes=changes):
                self.estimator.reset()
                self.assertIsNone(self.estimator.update(j16(**changes), calibrate_sign=1).pitch_deg)
                self.assertFalse(self.estimator.estimated)

    def test_controller_calibration_uses_estimates_only_for_turn_guidance(self):
        class Client:
            def poll(self, time_s=None):
                return j16(time_s)
        c = OverlayController(client=Client(), mass_kg=23000)
        self.addCleanup(c.stop)
        c.submit({"action": "turn_enabled", "enabled": True})
        before = c.tick(0)
        self.assertEqual(before.turn.phase, "需要校准")
        c.submit({"action": "pose_calibrate", "roll_sign": 1})
        c.tick(.1)
        after = c.tick(.2)
        self.assertIsNotNone(after.turn.estimated_pitch_deg)
        self.assertIsNone(after.state.pitch_deg)
        self.assertIsNone(after.state.roll_deg)
        self.assertEqual(after.turn.phase, "计算")
        self.assertEqual(after.advice, before.advice)
        self.assertIsNotNone(c._turn_session.goal)


if __name__ == "__main__":
    unittest.main()
