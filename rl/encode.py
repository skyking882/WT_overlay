"""Batching of observations into padded tensors and masks.

* decode_wire(list of obs wire tuples)  -> Decoded   (ragged rows, one entry per obs)
* Decoded.to_batch(...)                  -> Batch     ([B,1,...] padded, for rollouts)
* StepStore                              -> flat storage of many steps, gather(idx[B,W]) -> Batch
  (used for the PPO round buffer and the BC dataset; entity and truth rows are stored
  ragged so padding costs nothing in memory)

Padding never reaches a loss: every Batch carries `valid` (a real decision) and the entity /
truth counts; padded entity rows are zero and masked everywhere in the model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from rl import spec

_F32 = torch.float32


def _frombytes(b: bytes, dtype, width: int):
    if len(b) == 0:
        return torch.zeros(0, width, dtype=dtype) if width else torch.zeros(0, dtype=dtype)
    t = torch.frombuffer(bytearray(b), dtype=dtype)
    return t.view(-1, width) if width else t


@dataclass
class Batch:
    own: torch.Tensor        # [B,T,96]
    ent: torch.Tensor        # [B,T,N,48]
    ent_n: torch.Tensor      # [B,T] long
    prev: torch.Tensor       # [B,T,48]
    truth: torch.Tensor      # [B,T,M,40]
    truth_n: torch.Tensor    # [B,T] long
    masks: Dict[str, torch.Tensor]   # head -> [B,T,*spec.mask_shape] bool (section 13.1 tables, pointer
                                     # widths and target rows sliced to N+1)
    valid: torch.Tensor      # [B,T] bool
    first: torch.Tensor      # [B,T] bool
    dt: torch.Tensor         # [B,T]
    act: Optional[torch.Tensor] = None   # [B,T,15] long

    @property
    def shape(self):
        return self.ent_n.shape

    def to(self, device):
        return Batch(
            self.own.to(device), self.ent.to(device), self.ent_n.to(device), self.prev.to(device),
            self.truth.to(device), self.truth_n.to(device),
            {k: v.to(device) for k, v in self.masks.items()}, self.valid.to(device), self.first.to(device), self.dt.to(device),
            None if self.act is None else self.act.to(device))

    def time_slice(self, a, b):
        s = slice(a, b)
        return Batch(self.own[:, s], self.ent[:, s], self.ent_n[:, s], self.prev[:, s], self.truth[:, s],
                     self.truth_n[:, s], {k: v[:, s] for k, v in self.masks.items()}, self.valid[:, s], self.first[:, s],
                     self.dt[:, s], None if self.act is None else self.act[:, s])


def pad_rows(rows, starts, counts, cap):
    """Ragged rows -> dense [..., cap, D] (zeros beyond counts). starts/counts share a shape."""
    d = rows.shape[-1]
    if rows.shape[0] == 0:
        return torch.zeros(*counts.shape, cap, d, dtype=rows.dtype)
    ar = torch.arange(cap)
    idx = starts.unsqueeze(-1) + ar
    ok = ar < counts.unsqueeze(-1)
    out = rows[idx.clamp(max=rows.shape[0] - 1)]
    return out * ok.unsqueeze(-1).to(out.dtype)


def masks_from_flat(flat, n_cap):
    """[..., MASK_BYTES] bool -> {head: [..., *shape]} with pointer widths / target rows cut to n_cap = N+1."""
    lead = flat.shape[:-1]
    out = {}
    for h in spec.HEAD_NAMES:
        off, size = spec.MASK_OFFSET[h], spec.MASK_SIZE[h]
        v = flat[..., off:off + size].reshape(*lead, *spec.mask_cap_shape(h))
        if h in spec.POINTER_HEADS:
            v = v[..., :n_cap]
        if spec.MASK_KIND[h] == "target":
            v = v[..., :n_cap, :]
        out[h] = v
    return out


class Decoded:
    """Ragged decode of B wire observations (one per row of own/prev/...)."""

    def __init__(self, obs_list: List[tuple]):
        B = len(obs_list)
        self.B = B
        self.own = _frombytes(b"".join(o[0] for o in obs_list), _F32, spec.OWN_DIM)
        self.ent_n = torch.tensor([o[1] for o in obs_list], dtype=torch.long)
        self.ent_rows = _frombytes(b"".join(o[2] for o in obs_list), _F32, spec.ENT_DIM)
        self.prev = _frombytes(b"".join(o[3] for o in obs_list), _F32, spec.INTENT_DIM)
        self.truth_n = torch.tensor([o[4] for o in obs_list], dtype=torch.long)
        self.truth_rows = _frombytes(b"".join(o[5] for o in obs_list), _F32, spec.TRUTH_DIM)
        self.mask = _frombytes(b"".join(o[6] for o in obs_list), torch.uint8, spec.MASK_BYTES).bool()
        self.aircraft = [o[7] for o in obs_list]
        dt = torch.tensor([o[8] for o in obs_list], dtype=_F32)
        self.dt = torch.where(dt > 1e-6, dt, torch.full_like(dt, spec.DT_STEP))   # dt <= 0 (e.g. a reset obs) -> 20/48

    def to_batch(self, first=None, valid=None) -> Batch:
        """[B,1,...] batch; entities padded to the largest count in this decode."""
        B = self.B
        N = max(1, int(self.ent_n.max())) if B else 1
        Mx = max(1, int(self.truth_n.max())) if B else 1
        e_start = torch.cumsum(self.ent_n, 0) - self.ent_n
        t_start = torch.cumsum(self.truth_n, 0) - self.truth_n
        ent = pad_rows(self.ent_rows, e_start, self.ent_n, N)
        truth = pad_rows(self.truth_rows, t_start, self.truth_n, Mx)
        if first is None:
            first = torch.zeros(B, dtype=torch.bool)
        if valid is None:
            valid = torch.ones(B, dtype=torch.bool)
        masks = {k: v.unsqueeze(1) for k, v in masks_from_flat(self.mask, N + 1).items()}
        return Batch(
            own=self.own.unsqueeze(1), ent=ent.unsqueeze(1), ent_n=self.ent_n.unsqueeze(1),
            prev=self.prev.unsqueeze(1), truth=truth.unsqueeze(1), truth_n=self.truth_n.unsqueeze(1),
            masks=masks, valid=valid.unsqueeze(1), first=first.unsqueeze(1), dt=self.dt.unsqueeze(1))


class StepStore:
    """Flat storage of n_steps decision steps with ragged entity / truth rows.

    Slots are filled with put(); unfilled slots are padding (valid False, zero data).
    """

    def __init__(self, n_steps: int):
        N = n_steps
        self.N = N
        self.own = torch.zeros(N, spec.OWN_DIM)
        self.prev = torch.zeros(N, spec.INTENT_DIM)
        self.mask = torch.zeros(N, spec.MASK_BYTES, dtype=torch.bool)
        self.ent_start = torch.zeros(N, dtype=torch.long)
        self.ent_n = torch.zeros(N, dtype=torch.long)
        self.tr_start = torch.zeros(N, dtype=torch.long)
        self.tr_n = torch.zeros(N, dtype=torch.long)
        self.act = torch.zeros(N, spec.N_HEADS, dtype=torch.long)
        self.valid = torch.zeros(N, dtype=torch.bool)
        self.first = torch.zeros(N, dtype=torch.bool)
        self.dt = torch.full((N,), spec.DT_STEP)
        self._ent_parts: List[torch.Tensor] = []
        self._tr_parts: List[torch.Tensor] = []
        self.n_ent_rows = 0
        self.n_tr_rows = 0
        self.ent_rows = torch.zeros(0, spec.ENT_DIM)
        self.truth_rows = torch.zeros(0, spec.TRUTH_DIM)

    # ------------------------------------------------------------------ writing
    def put(self, idx: torch.Tensor, dec: Decoded, first=None, act=None):
        """Write the B decoded observations into slots idx [B] (rows appended in order)."""
        if dec.B == 0:
            return
        self.own[idx] = dec.own
        self.prev[idx] = dec.prev
        self.mask[idx] = dec.mask
        self.ent_start[idx] = self.n_ent_rows + torch.cumsum(dec.ent_n, 0) - dec.ent_n
        self.ent_n[idx] = dec.ent_n
        self.tr_start[idx] = self.n_tr_rows + torch.cumsum(dec.truth_n, 0) - dec.truth_n
        self.tr_n[idx] = dec.truth_n
        self._ent_parts.append(dec.ent_rows)
        self._tr_parts.append(dec.truth_rows)
        self.n_ent_rows += dec.ent_rows.shape[0]
        self.n_tr_rows += dec.truth_rows.shape[0]
        self.dt[idx] = dec.dt
        self.valid[idx] = True
        if first is not None:
            self.first[idx] = first
        if act is not None:
            self.act[idx] = act

    def finalize(self):
        if self._ent_parts:
            self.ent_rows = torch.cat([self.ent_rows] + self._ent_parts, 0)
            self._ent_parts = []
        if self._tr_parts:
            self.truth_rows = torch.cat([self.truth_rows] + self._tr_parts, 0)
            self._tr_parts = []

    def copy_from(self, other: "StepStore", src: slice, dst_start: int):
        """Copy the contiguous step range `src` of `other` into this store at dst_start.

        Rows of the copied steps are appended (so call before any put() in this store, in the
        same order as the steps are laid out, to keep rows contiguous per step range).
        """
        other.finalize()
        a, b = src.start, src.stop
        n = b - a
        dst = slice(dst_start, dst_start + n)
        for name in ("own", "prev", "mask", "ent_n", "tr_n", "act", "valid", "first", "dt"):
            getattr(self, name)[dst] = getattr(other, name)[a:b]
        for st, nn_, rows_name, part_name, cnt_name in (
                ("ent_start", "ent_n", "ent_rows", "_ent_parts", "n_ent_rows"),
                ("tr_start", "tr_n", "truth_rows", "_tr_parts", "n_tr_rows")):
            starts = getattr(other, st)[a:b]
            counts = getattr(other, nn_)[a:b]
            mask = counts > 0
            if bool(mask.any()):
                lo = int(starts[mask].min())
                hi = int((starts + counts)[mask].max())
                getattr(self, part_name).append(getattr(other, rows_name)[lo:hi])
                new = starts - lo + getattr(self, cnt_name)
                getattr(self, st)[dst] = torch.where(mask, new, torch.zeros_like(new))
                setattr(self, cnt_name, getattr(self, cnt_name) + (hi - lo))
            else:
                getattr(self, st)[dst] = 0

    # ------------------------------------------------------------------ reading
    def gather(self, idx: torch.Tensor) -> Batch:
        """idx [B,W] long; entries < 0 are padding. Returns a Batch of shape [B,W]."""
        self.finalize()
        ok = idx >= 0
        ii = idx.clamp(min=0)
        okf = ok.unsqueeze(-1)
        ent_n = torch.where(ok, self.ent_n[ii], torch.zeros_like(ii))
        tr_n = torch.where(ok, self.tr_n[ii], torch.zeros_like(ii))
        N = max(1, int(ent_n.max()))
        Mx = max(1, int(tr_n.max()))
        ent = pad_rows(self.ent_rows, self.ent_start[ii], ent_n, N)
        truth = pad_rows(self.truth_rows, self.tr_start[ii], tr_n, Mx)
        masks = masks_from_flat(self.mask[ii] & okf, N + 1)
        return Batch(
            own=self.own[ii] * okf, ent=ent, ent_n=ent_n, prev=self.prev[ii] * okf, truth=truth, truth_n=tr_n,
            masks=masks,
            valid=self.valid[ii] & ok, first=self.first[ii] & ok, dt=self.dt[ii],
            act=self.act[ii] * okf)
