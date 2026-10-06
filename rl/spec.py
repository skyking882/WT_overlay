"""Interface-contract constants shared by the trainer and the environment side.

Pure Python only (no numpy, no torch) so that PyPy workers can import it.
Everything here follows docs/rl_training_spec.md sections 3, 4 and 13 (13.1: mask format).
"""
from __future__ import annotations

import math

DT_STEP = 20.0 / 48.0          # seconds per env.step (20 ticks of 1/48 s)
GAMMA_BASE = 0.995             # per second
LAMBDA_BASE = 0.95             # per second

# Episode kinds an env may report (MatchEnv with self_play_prob / history_prob: env.episode_kind,
# info["episode_kind"]). An env that reports none counts as vs-script, the behaviour before self-play existed.
KIND_SCRIPT = "vs_script"      # the configured controlled slots (default slot 0) fly the policy, the rest scripts
KIND_SELF = "self_play"        # every aircraft slot is policy-controlled
KIND_HIST = "history"          # every slot is policy-controlled; one side (env.frozen_ids) flies a frozen past policy

OWN_DIM = 96
ENT_DIM = 48
INTENT_DIM = 48
TRUTH_DIM = 40
MAX_ENT = 64
MAX_TRUTH = 128
PTR_CAP = MAX_ENT + 1          # pointer-head width when padded (n + 1 <= 65)

# Heads in the order of the contract table (section 13). This is also the column
# order of every action vector / action tuple on the wire.
# antenna: 5 options +20 / +10 / 0 / -10 / -20 deg (spec 13.2; 3 options before that revision).
HEAD_NAMES = (
    "maneuver_ref", "target", "view_object",
    "maneuver", "vertical", "speed", "chaff", "radar_mode", "antenna", "weapon",
    "view_mode", "look_az", "look_el", "kb_roll", "kb_pitch",
)
N_HEADS = len(HEAD_NAMES)
HEAD_INDEX = {h: i for i, h in enumerate(HEAD_NAMES)}
POINTER_HEADS = ("maneuver_ref", "target", "view_object")
CAT_HEADS = tuple(h for h in HEAD_NAMES if h not in POINTER_HEADS)
CAT_SIZES = {
    "maneuver": 11, "vertical": 5, "speed": 3, "chaff": 3, "radar_mode": 3,
    "antenna": 5, "weapon": 2, "view_mode": 3, "look_az": 8, "look_el": 3,
    "kb_roll": 3, "kb_pitch": 3,
}

# Layout of per-option tables over all categorical heads (reference-policy log-probs).
CAT_OFFSETS = {}
_c = 0
for _h in CAT_HEADS:
    CAT_OFFSETS[_h] = _c
    _c += CAT_SIZES[_h]
CAT_TOTAL = _c                 # 52

# Option meanings (indices) that the trainer needs to know about.
VIEW_AIM, VIEW_OBJECT, VIEW_DIRECTION = 0, 1, 2
WEAPON_NO, WEAPON_FIRE = 0, 1
MANEUVER_KEEP = 0
NULL = "null"                  # marker: the pointer head's "no object" option (index n)

# Sampling order (section 3 / 13.1): view mode first, then maneuver reference, target and view
# object, then everything else.
SAMPLE_ORDER = (
    "view_mode", "maneuver_ref", "target", "view_object",
    "maneuver", "vertical", "speed", "chaff", "radar_mode", "antenna", "weapon",
    "look_az", "look_el", "kb_roll", "kb_pitch",
)

# ---------------------------------------------------------------------------
# Mask format (section 13.1). The environment precomputes conditional legality, because
# sampling runs in the learner process without access to the env:
#
#   kind      masks[head] shape          row selected by                      heads
#   flat      K                           -                                    view_mode speed chaff antenna target
#   view      3 x K                       chosen view_mode                     maneuver_ref view_object look_az
#                                                                              look_el kb_roll kb_pitch
#   view_ref  3 x 2 x K                   chosen view_mode, then 1 if the      maneuver vertical
#                                         chosen maneuver_ref is "none"
#   target    (n+1) x K                   chosen target index                  weapon radar_mode
#
# K = n + 1 for pointer heads (last option = "none"), else CAT_SIZES[head]. A held head has
# exactly one True; every selected 1-D mask must contain at least one True.
# ---------------------------------------------------------------------------
MASK_KIND = {
    "view_mode": "flat", "speed": "flat", "chaff": "flat", "antenna": "flat", "target": "flat",
    "maneuver_ref": "view", "view_object": "view", "look_az": "view", "look_el": "view",
    "kb_roll": "view", "kb_pitch": "view",
    "maneuver": "view_ref", "vertical": "view_ref",
    "weapon": "target", "radar_mode": "target",
}


def head_size(head: str, n: int) -> int:
    """Number of options of a head for an observation with n entities."""
    return n + 1 if head in POINTER_HEADS else CAT_SIZES[head]


def head_cap(head: str) -> int:
    return PTR_CAP if head in POINTER_HEADS else CAT_SIZES[head]


def mask_shape(head: str, n: int):
    k = head_size(head, n)
    kind = MASK_KIND[head]
    if kind == "flat":
        return (k,)
    if kind == "view":
        return (3, k)
    if kind == "view_ref":
        return (3, 2, k)
    return (n + 1, k)


def mask_cap_shape(head: str):
    k = head_cap(head)
    kind = MASK_KIND[head]
    if kind == "flat":
        return (k,)
    if kind == "view":
        return (3, k)
    if kind == "view_ref":
        return (3, 2, k)
    return (PTR_CAP, k)


def _prod(t):
    r = 1
    for x in t:
        r *= x
    return r


MASK_OFFSET = {}
MASK_SIZE = {}
_o = 0
for _h in HEAD_NAMES:
    MASK_OFFSET[_h] = _o
    MASK_SIZE[_h] = _prod(mask_cap_shape(_h))
    _o += MASK_SIZE[_h]
MASK_BYTES = _o                # width of the padded, flattened per-observation mask block

# Fallback option if a selected mask row is empty (an env bug; the trainer counts these).
DEFAULT_OPTION = {h: 0 for h in HEAD_NAMES}
DEFAULT_OPTION.update({"maneuver_ref": NULL, "target": NULL, "view_object": NULL,
                       "look_el": 1, "kb_roll": 1, "kb_pitch": 1})


def default_option(head: str, n: int) -> int:
    d = DEFAULT_OPTION[head]
    return n if d == NULL else d


def select_row(head, n, masks, chosen):
    """The 1-D mask of `head` given the already chosen earlier heads (reference semantics)."""
    m = masks[head]
    kind = MASK_KIND[head]
    if kind == "flat":
        return m
    if kind == "view":
        return m[chosen["view_mode"]]
    if kind == "view_ref":
        return m[chosen["view_mode"]][1 if chosen["maneuver_ref"] == n else 0]
    return m[chosen["target"]]


def effective_mask(head, n, masks, chosen):
    """Reference (pure Python) legal-option mask of one head: the selected row, with the
    empty-row fallback (default option) the trainer applies."""
    row = [bool(x) for x in select_row(head, n, masks, chosen)]
    k = head_size(head, n)
    if len(row) != k:
        raise ValueError("mask %s has a selected row of length %d, expected %d" % (head, len(row), k))
    if not any(row):
        row = [False] * k
        row[default_option(head, n)] = True
    return row


def check_mask_shapes(n, masks):
    """List of structural problems of an obs.masks dict (empty list = fine)."""
    probs = []
    for h in HEAD_NAMES:
        if h not in masks:
            probs.append("missing head %s" % h)
    for k in masks:
        if k not in HEAD_INDEX:
            probs.append("unknown key %r (the old 'head|cond' tables are replaced by the shapes of section 13.1)" % k)
    if probs:
        return probs
    for h in HEAD_NAMES:
        shape = mask_shape(h, n)
        m = masks[h]
        try:
            _check_nested(m, shape)
        except ValueError as e:
            probs.append("masks[%s]: %s (expected shape %s)" % (h, e, "x".join(map(str, shape))))
    return probs


def _check_nested(m, shape):
    if len(m) != shape[0]:
        raise ValueError("length %d where %d expected" % (len(m), shape[0]))
    if len(shape) > 1:
        for r in m:
            _check_nested(r, shape[1:])


def empty_selectable_rows(n, masks):
    """Rows that can actually be selected by some legal sequence of earlier choices but contain no
    True (violates the env guarantee of 13.1). Returns a list of strings."""
    bad = []

    def legal(head, chosen):
        return [i for i, ok in enumerate(select_row(head, n, masks, chosen)) if ok]

    if not any(masks["view_mode"]):
        bad.append("view_mode")
    for vm in [i for i, ok in enumerate(masks["view_mode"]) if ok]:
        ch = {"view_mode": vm}
        mrs = legal("maneuver_ref", ch)
        if not mrs:
            bad.append("maneuver_ref[view_mode=%d]" % vm)
        for h in ("view_object", "look_az", "look_el", "kb_roll", "kb_pitch"):
            if not legal(h, ch):
                bad.append("%s[view_mode=%d]" % (h, vm))
        for mr in mrs:
            ch2 = dict(ch, maneuver_ref=mr)
            for h in ("maneuver", "vertical"):
                if not legal(h, ch2):
                    bad.append("%s[view_mode=%d, ref_is_null=%d]" % (h, vm, int(mr == n)))
    for h in ("speed", "chaff", "antenna", "target"):
        if not any(masks[h]):
            bad.append(h)
    for t in [i for i, ok in enumerate(masks["target"]) if ok]:
        for h in ("weapon", "radar_mode"):
            if not any(masks[h][t]):
                bad.append("%s[target=%d]" % (h, t))
    return bad


def illegal_heads(n, masks, actions):
    """Heads whose chosen option is not legal under the conditional rules.

    An out-of-range or masked option is reported; later heads are then evaluated as if the first
    legal option had been chosen for it, so the check never raises on bad actions.
    """
    chosen = {}
    bad = []
    for head in SAMPLE_ORDER:
        m = effective_mask(head, n, masks, chosen)
        a = actions[head]
        if isinstance(a, bool) or not isinstance(a, int) or not (0 <= a < len(m)) or not m[a]:
            bad.append(head)
            a = m.index(True)
        chosen[head] = a
    return bad


def canonicalize_action(n, masks, actions):
    """Make an intent consistent with the masks (used for BC labels).

    Heads that have exactly one legal option are forced to it (e.g. the mouse-flight heads while
    the view is free); a choice that is illegal although several options exist is replaced by the
    nearest legal one. Returns (canonical {head: int}, forced heads, illegal heads).
    """
    chosen, forced, illegal = {}, [], []
    for head in SAMPLE_ORDER:
        m = effective_mask(head, n, masks, chosen)
        a = actions[head]
        if not (0 <= a < len(m) and m[a]):
            legal = [i for i, ok in enumerate(m) if ok]
            (forced if len(legal) == 1 else illegal).append(head)
            a = min(legal, key=lambda i: abs(i - a))
        chosen[head] = a
    return {h: chosen[h] for h in HEAD_NAMES}, forced, illegal


def all_true_masks(n):
    """obs.masks with every option legal (right shapes); handy for benchmarks and tests."""
    def build(shape):
        return [True] * shape[0] if len(shape) == 1 else [build(shape[1:]) for _ in range(shape[0])]
    return {h: build(mask_shape(h, n)) for h in HEAD_NAMES}


def gamma_lambda(dt: float, gamma_base: float = GAMMA_BASE, lambda_base: float = LAMBDA_BASE):
    return gamma_base ** dt, lambda_base ** dt


def log_k(k: int) -> float:
    return math.log(k)
