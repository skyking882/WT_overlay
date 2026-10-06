"""Wire format between environment workers (pure Python / PyPy) and the learner.

An AgentObs (object with attributes, or a dict) is packed on the worker side into a
plain tuple of ints, floats, strs, bytes and a dict of bytes, so that messages are
pickle-able by both CPython and PyPy (protocol 4) and cheap to decode into tensors
on the learner side without per-float Python work.

obs wire tuple:
    (own: bytes[96 f32], n: int, ent: bytes[n*48 f32], prev: bytes[48 f32],
     m: int, truth: bytes[m*40 f32],
     masks: bytes[spec.MASK_BYTES] (0/1; every head's table of section 13.1 flattened row-major at
            spec.MASK_OFFSET[head] with pointer widths / target rows zero padded to 65),
     aircraft: str, dt: float)
action wire: tuple of 15 ints in spec.HEAD_NAMES order.
"""
from __future__ import annotations

import itertools
from array import array

from rl import spec

PICKLE_PROTOCOL = 4        # supported by CPython 3.8+ and PyPy 3.x; plain types only


class ContractError(ValueError):
    pass


def _get(obs, name):
    if isinstance(obs, dict):
        return obs[name]
    return getattr(obs, name)


def _f32(values, what, expect=None):
    try:
        a = array("f", values)
    except (TypeError, ValueError) as e:
        raise ContractError("%s: not a flat list of numbers (%s)" % (what, e))
    if expect is not None and len(a) != expect:
        raise ContractError("%s: expected %d values, got %d" % (what, expect, len(a)))
    return a


def pack_masks(masks, n):
    """obs.masks (section 13.1 shapes) -> padded flat bytes of length spec.MASK_BYTES."""
    probs = spec.check_mask_shapes(n, masks)
    if probs:
        raise ContractError("masks: " + "; ".join(probs))
    buf = bytearray(spec.MASK_BYTES)
    for h in spec.HEAD_NAMES:
        off = spec.MASK_OFFSET[h]
        kcap = spec.head_cap(h)
        m = masks[h]
        kind = spec.MASK_KIND[h]
        if kind == "flat":
            rows = [(0, m)]
        elif kind == "view":
            rows = [(r, m[r]) for r in range(3)]
        elif kind == "view_ref":
            rows = [(r * 2 + q, m[r][q]) for r in range(3) for q in range(2)]
        else:
            rows = [(r, m[r]) for r in range(n + 1)]
        for ri, row in rows:
            base = off + ri * kcap
            for j, v in enumerate(row):
                if v:
                    buf[base + j] = 1
    return bytes(buf)


def pack_obs(obs, check_finite=False):
    own = _f32(_get(obs, "own"), "own", spec.OWN_DIM)
    ents = _get(obs, "entities")
    n = len(ents)
    if n > spec.MAX_ENT:
        raise ContractError("entities: %d > %d" % (n, spec.MAX_ENT))
    for e in ents:
        if len(e) != spec.ENT_DIM:
            raise ContractError("entities: row of width %d, expected %d" % (len(e), spec.ENT_DIM))
    ent = _f32(itertools.chain.from_iterable(ents), "entities")
    prev = _f32(_get(obs, "prev_intent"), "prev_intent", spec.INTENT_DIM)
    truth_rows = _get(obs, "truth")
    m = len(truth_rows)
    if m > spec.MAX_TRUTH:
        raise ContractError("truth: %d > %d" % (m, spec.MAX_TRUTH))
    for e in truth_rows:
        if len(e) != spec.TRUTH_DIM:
            raise ContractError("truth: row of width %d, expected %d" % (len(e), spec.TRUTH_DIM))
    truth = _f32(itertools.chain.from_iterable(truth_rows), "truth")
    if check_finite:
        for name, arr in (("own", own), ("entities", ent), ("prev_intent", prev), ("truth", truth)):
            for x in arr:
                if x != x or x in (float("inf"), float("-inf")):
                    raise ContractError("%s contains a non-finite value" % name)
    masks = _get(obs, "masks")
    mask_bytes = pack_masks(masks, n)
    dt = float(_get(obs, "dt"))
    return (own.tobytes(), n, ent.tobytes(), prev.tobytes(), m, truth.tobytes(),
            mask_bytes, str(_get(obs, "aircraft")), dt)


def pack_action(act):
    """{head: int} -> tuple of 15 ints."""
    return tuple(int(act[h]) for h in spec.HEAD_NAMES)


def unpack_action(t):
    return {h: int(v) for h, v in zip(spec.HEAD_NAMES, t)}
