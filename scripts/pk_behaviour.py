#!/usr/bin/env python3
"""Opponent behaviour for hit probabilities (no numpy: shared by the surrogate and PyPy sims).

The escape surrogate answers "does the target escape if it starts plan P at time
t?". The hit probability averages that over how real opponents behave. Every
number here is a guess to be revisited (user estimates of 2026-10-04, D-grade):

  * Detection. ARH missiles give no launch warning (TWS or STT). The target
    learns of the shot at the earliest of: the active seeker's RWR warning
    (t_active, always noticed); the motor-burn marker (noticed with P_SEE_BURN,
    taken as at launch); for STT shots, the lock warning prompting a precautionary
    defence (P_LOCK_REACT, at launch).
  * Human delay after detection: log-normal, median DELAY_MEDIAN_S.
  * P_REACT of opponents defend at all (the rest fly on and are hit whenever an
    unevaded shot hits). Of those, P_CORRECT pick the best of the REPERTOIRE for the
    situation; the rest pick one of it at random.
  * Defenders drop chaff; its strength is unknown: CHAFF_RATIOS equally likely.

Range and geometry enter only through the surrogate, so e.g. a wrong defence at
long range may still escape.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Behaviour:
    p_react: float = .85
    p_correct: float = .60
    p_see_burn: float = .5
    p_lock_react: float = .5
    delay_median_s: float = 1.5
    delay_sigma: float = .4          # log-normal shape: 5-95 % about 0.8-2.9 s
    delay_points: int = 5
    chaff_ratios: tuple = (.5, 1., 2.)
    # (target_deg, plane_deg, dive_deg, speed_kmh; >= 1250 = full throttle): level beam, diving beams,
    # split-S into the beam, drag, diving drag, and throttle-back + airbrake to 600 km/h then a level beam.
    repertoire: tuple = ((90., 0., 0., 1500.), (90., 0., 20., 1500.), (90., 0., 40., 1500.), (90., 90., 20., 1500.),
                         (0., 0., 0., 1500.), (0., 0., 20., 1500.), (90., 0., 0., 600.))

    def delays(self):
        """Equal-probability quantile points of the human delay."""
        return [self.delay_median_s*math.exp(self.delay_sigma*_norm_ppf((k+.5)/self.delay_points))
                for k in range(self.delay_points)]

    def detections(self, mode, t_active):
        """[(probability, detection time)] for 'tws' or 'stt' shots."""
        early = 1-(1-self.p_see_burn)*((1-self.p_lock_react) if mode == "stt" else 1.)
        late = t_active if t_active is not None else math.inf
        return [(early, 0.), (1-early, late)]


def _norm_ppf(q):
    # Acklam's rational approximation; ample for a handful of quadrature points.
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549671324928485e+00,
         4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00)
    if q < .02425:
        t = math.sqrt(-2*math.log(q))
        return (((((c[0]*t+c[1])*t+c[2])*t+c[3])*t+c[4])*t+c[5])/((((d[0]*t+d[1])*t+d[2])*t+d[3])*t+1)
    if q > 1-.02425:
        return -_norm_ppf(1-q)
    t = q-.5
    r = t*t
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*t/(((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
