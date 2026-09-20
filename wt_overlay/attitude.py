"""Calibrated kinematic attitude estimate for aircraft without horizon outputs.

Requires an explicit upright, wings-level seed. Wx is a body-axis rate, not
Euler roll rate: phi_dot = p + psi_dot*sin(theta). No force/load inversion.
The Wx sign must be checked in game; both sign choices are exposed at calibration.
"""
from dataclasses import replace
import math

from .climb import valid_number


def wrap(degrees):
    return (degrees+180.) % 360.-180.


def pitch_from_velocity(state, roll_deg):
    alpha, beta, roll = map(math.radians, (state.aoa_deg, state.aos_deg, roll_deg))
    a = math.cos(alpha)*math.cos(beta)
    b = math.sin(alpha)*math.cos(beta)*math.cos(roll)+math.sin(beta)*math.sin(roll)
    radius = math.hypot(a, b)
    vertical = state.vertical_speed_mps/state.tas_mps
    if radius < .2 or abs(vertical/radius) > 1:
        raise ValueError("姿态估计几何不可解，请平翼重新校准")
    phase = math.atan2(b, a)
    principal = math.asin(vertical/radius)
    candidates = [wrap(math.degrees(phase+x)) for x in (principal, math.pi-principal)]
    candidates = [x for x in candidates if abs(x) < 75.]
    if len(candidates) != 1:
        raise ValueError("接近垂直或姿态分支不确定，请平翼重新校准")
    return candidates[0]


class AttitudeEstimator:
    """Raw attitude takes priority; estimates expire instead of silently drifting."""
    def __init__(self):
        self.reset()

    def reset(self, reason="缺少姿态，请机翼水平后点击平翼校准"):
        self.previous = None
        self.calibrated_at = None
        self.identity = None
        self.roll = self.pitch = None
        self.sign = 1
        self.reason = reason
        self.estimated = False

    def update(self, state, *, calibrate_sign=None):
        self.estimated = False
        if valid_number(state.pitch_deg) and valid_number(state.roll_deg) and state.valid:
            self.reset("使用 8111 姿态读数，无需平翼校准")
            return state
        identity = (state.source, state.aircraft_id)
        if self.identity is not None and identity != self.identity and state.valid and state.aircraft_id:
            self.reset("机型已改变，请平翼重新校准")
        rate = state.raw_state.get("Wx, deg/s")
        required = (state.time_s, state.heading_deg, state.tas_mps, state.vertical_speed_mps,
                    state.aoa_deg, state.aos_deg, rate)
        if not state.valid or not all(valid_number(x) for x in required):
            if self.previous is not None:
                self.reset("姿态估计所需数据中断，请平翼重新校准")
            elif calibrate_sign is not None:
                self.reason = "校准需要航向、TAS、Vy、AoA、AoS 和 Wx 读数"
            return state
        if state.tas_mps < 50 or abs(state.aos_deg) > 10 or abs(rate) > 400:
            self.reset("状态超出姿态估计范围，请平翼重新校准")
            return state
        try:
            if calibrate_sign is not None:
                if calibrate_sign not in (-1, 1):
                    raise ValueError("滚转率方向须为 +1 或 -1")
                if abs(rate) > 3 or abs(state.aos_deg) > 3:
                    raise ValueError("校准时请停止滚转并减小侧滑")
                self.roll = 0.
                self.pitch = pitch_from_velocity(state, self.roll)
                self.sign = calibrate_sign
                self.calibrated_at = state.time_s
                self.identity = identity
            elif self.previous is None:
                return state
            else:
                t0, heading0, rate0 = self.previous
                dt = state.time_s-t0
                if not .01 <= dt <= .6:
                    raise ValueError("滚转率采样不连续，请平翼重新校准")
                if state.time_s-self.calibrated_at > 60:
                    raise ValueError("姿态校准已超过 60 秒，请平翼重新校准")
                yaw_change = wrap(state.heading_deg-heading0)
                if abs(yaw_change)/dt > 180:
                    raise ValueError("航向读数跳变，请平翼重新校准")
                body_change = self.sign*(rate0+rate)*dt/2
                predicted_roll = wrap(self.roll+body_change+yaw_change*math.sin(math.radians(self.pitch)))
                predicted_pitch = pitch_from_velocity(state, predicted_roll)
                self.roll = wrap(self.roll+body_change+yaw_change*math.sin(math.radians((self.pitch+predicted_pitch)/2)))
                self.pitch = pitch_from_velocity(state, self.roll)
            self.previous = (state.time_s, state.heading_deg, rate)
            self.estimated = True
            self.reason = (f"姿态使用平翼校准估计：俯仰 {self.pitch:+.1f}°，滚转 {self.roll:+.1f}°；"
                           f"校准后 {state.time_s-self.calibrated_at:.1f} s，Wx 方向 {self.sign:+d}")
            return replace(state, pitch_deg=self.pitch, roll_deg=self.roll)
        except (ValueError, OverflowError) as exc:
            self.reset(str(exc))
            return state
