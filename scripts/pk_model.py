#!/usr/bin/env python3
"""Hit probability of a shot against a population of opponents (offensive HUD).

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

import math
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pk_behaviour import Behaviour  # noqa: E402,F401


CHAFF_COLUMNS = (11, 12)  # train_surrogate.state_features: has-chaff flag, log2 ratio


@torch.no_grad()
def hit_probability(surrogate, state_vec, mode="tws", behaviour: Behaviour = Behaviour()):
    """P(hit) of a shot now, and its parts: P(reach), and P(hit | reach) per behaviour branch.

    state_vec: train_surrogate.state_features + evader_features; its chaff entries are
    replaced by the behaviour's chaff ratios for defenders. The launch net must output
    [hit logit, log time of flight, t_active (s, /10)]."""
    f = np.array(state_vec, dtype=np.float32)
    out = surrogate.launch(torch.tensor(f[None]))[0].numpy()
    p_reach = 1/(1+math.exp(-out[0]))
    tof = float(math.exp(out[1]))
    t_active = max(0., float(out[2])*10) if len(out) > 2 else None
    plans = np.array(behaviour.repertoire, dtype=np.float32)
    starts, weights = [], []
    for p_det, t_det in behaviour.detections(mode, t_active):
        for delay in behaviour.delays():
            starts.append(t_det+delay)
            weights.append(p_det/behaviour.delay_points)
    starts, weights = np.array(starts, dtype=np.float32), np.array(weights)
    best = np.zeros(len(starts))
    mean = np.zeros(len(starts))
    for ratio in behaviour.chaff_ratios:
        g = f.copy()
        g[CHAFF_COLUMNS[0]], g[CHAFF_COLUMNS[1]] = 1., math.log2(ratio)
        n, m = len(starts), len(plans)
        x = np.concatenate([np.repeat(g[None], n*m, 0), np.full((n*m, 1), tof, np.float32),
                            np.repeat(np.minimum(starts, tof), m)[:, None], np.tile(plans, (n, 1))], axis=1)
        p = 1/(1+np.exp(-surrogate.escape(torch.tensor(x))[:, 0].numpy().reshape(n, m)))
        p[starts >= tof] = 0.  # The missile has arrived before the defence begins.
        best += p.max(axis=1)/len(behaviour.chaff_ratios)
        mean += p.mean(axis=1)/len(behaviour.chaff_ratios)
    escape_correct = float(weights@best)
    escape_wrong = float(weights@mean)
    b = behaviour
    p_hit_given_reach = (1-b.p_react)+b.p_react*(b.p_correct*(1-escape_correct)+(1-b.p_correct)*(1-escape_wrong))
    return dict(p_hit=p_reach*p_hit_given_reach, p_reach=p_reach, p_hit_given_reach=p_hit_given_reach,
                escape_correct=escape_correct, escape_wrong=escape_wrong, t_active=t_active, tof=tof)


@torch.no_grad()
def hit_probability_batch(surrogate, states, modes=("tws", "stt"), behaviours=None, chunk=1024):
    """P(hit) for many engagements at once: {(behaviour name, mode): array}, plus 'p_reach'.

    states: [N, n_state] of train_surrogate.state_features + evader_features. Same model as
    hit_probability (detection x delay quadrature, chaff ratios, repertoire best / mean)."""
    behaviours = behaviours or {"normal": Behaviour()}
    states = np.asarray(states, dtype=np.float32)
    out = {key: np.zeros(len(states)) for key in ((b, m) for b in behaviours for m in modes)}
    out["p_reach"] = np.zeros(len(states))
    plans = None
    for lo in range(0, len(states), chunk):
        f = states[lo:lo+chunk]
        n = len(f)
        launch = surrogate.launch(torch.tensor(f)).numpy()
        p_reach = 1/(1+np.exp(-launch[:, 0]))
        tof = np.exp(launch[:, 1])
        t_active = np.maximum(0., launch[:, 2]*10)
        out["p_reach"][lo:lo+n] = p_reach
        for bname, b in behaviours.items():
            plans = np.array(b.repertoire, dtype=np.float32)
            delays = np.array(b.delays(), dtype=np.float32)
            m = len(plans)
            # Escape outcome tables for detection at launch and at t_active: [n, delays] best / mean.
            tables = {}
            for which, t_det in (("early", np.zeros(n, np.float32)), ("late", t_active.astype(np.float32))):
                starts = t_det[:, None]+delays[None, :]                      # [n, d]
                best = np.zeros(starts.shape)
                mean = np.zeros(starts.shape)
                for ratio in b.chaff_ratios:
                    g = f.copy()
                    g[:, CHAFF_COLUMNS[0]], g[:, CHAFF_COLUMNS[1]] = 1., math.log2(ratio)
                    d = starts.shape[1]
                    x = np.concatenate([np.repeat(g, d*m, 0), np.repeat(tof, d*m)[:, None].astype(np.float32),
                                        np.repeat(np.minimum(starts, tof[:, None]).reshape(-1), m)[:, None],
                                        np.tile(plans, (n*d, 1))], axis=1).astype(np.float32)
                    p = 1/(1+np.exp(-surrogate.escape(torch.tensor(x))[:, 0].numpy().reshape(n, d, m)))
                    p[starts >= tof[:, None]] = 0.
                    best += p.max(axis=2)/len(b.chaff_ratios)
                    mean += p.mean(axis=2)/len(b.chaff_ratios)
                tables[which] = (best.mean(axis=1), mean.mean(axis=1))
            for mode in modes:
                early = 1-(1-b.p_see_burn)*((1-b.p_lock_react) if mode == "stt" else 1.)
                esc_c = early*tables["early"][0]+(1-early)*tables["late"][0]
                esc_w = early*tables["early"][1]+(1-early)*tables["late"][1]
                given = (1-b.p_react)+b.p_react*(b.p_correct*(1-esc_c)+(1-b.p_correct)*(1-esc_w))
                out[(bname, mode)][lo:lo+n] = p_reach*given
    return out
