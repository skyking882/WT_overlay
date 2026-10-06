"""Actor and critic networks of docs/rl_training_spec.md section 6.

Parameter counts (checked by rl/tests/test_params.py): actor 643,125 with the 5-option antenna head of
spec 13.2 (642,355 with the original 3 options), critic 650,627.

Conventions
-----------
* Pointer-head options live in "env space": option i < n is entity i, option n is the
  head's learned "none" key, positions > n are padding and always masked. So the model
  action index equals the env action index and nothing needs converting.
* The feature of a *selected* object that conditions later heads is the key vector of
  that option under the pointer head that selected it (the head's null key for "none").
  That is what makes the conditioning widths of the spec's parameter table (64 each) work.
* A masked option has probability exactly 0. A head with exactly one legal option has
  log-prob 0 and zero entropy. A head with K > 1 contributes entropy / log K.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl import spec
from rl.encode import Batch

NEG = -1.0e9
D = 64
H = 256


class PreLNBlock(nn.Module):
    """Pre-LN Transformer encoder layer: d=64, 4 heads, FFN=128, dropout 0, GELU.

    Structurally identical to nn.TransformerEncoderLayer(d_model=64, nhead=4,
    dim_feedforward=128, dropout=0.0, activation="gelu", norm_first=True, batch_first=True);
    written out so that training and inference use the same code path (no fast path)
    and fully masked rows cannot produce NaN. Equivalence is unit-tested.
    """

    def __init__(self, d=D, heads=4, ffn=128):
        super().__init__()
        self.h = heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ff1 = nn.Linear(d, ffn)
        self.ff2 = nn.Linear(ffn, d)

    def forward(self, x, key_mask):
        # x [M,N,d], key_mask [M,N] bool (True = may be attended to; every row has one True)
        M, N, d = x.shape
        a = self.ln1(x)
        qkv = self.qkv(a).view(M, N, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], attn_mask=key_mask[:, None, None, :])
        x = x + self.proj(o.transpose(1, 2).reshape(M, N, d))
        return x + self.ff2(F.gelu(self.ff1(self.ln2(x))))


class EntityEncoder(nn.Module):
    """MLP in->64->64 (GELU), 2 pre-LN Transformer layers, attention pooling."""

    def __init__(self, in_dim, d=D):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, d), nn.GELU(), nn.Linear(d, d), nn.GELU())
        self.blocks = nn.ModuleList([PreLNBlock(d) for _ in range(2)])
        self.pool = nn.Linear(d, 1)

    def forward(self, x, mask):
        """x [M,N,in], mask [M,N] bool of valid entities -> (per-entity [M,N,d], pooled [M,d])."""
        h = self.mlp(x)
        attn = mask.clone()
        attn[:, 0] |= ~mask.any(-1)          # rows without entities attend to slot 0 (output discarded)
        for blk in self.blocks:
            h = blk(h, attn)
        s = self.pool(h).squeeze(-1).masked_fill(~mask, NEG)
        w = torch.softmax(s, -1) * mask       # all-masked row -> weights 0 -> pooled 0
        return h, (w.unsqueeze(-1) * h).sum(1)


def gru_unroll(cell: nn.GRUCell, x, h0, first, valid):
    """GRU over [B,T,*] with per-step reset and padding handling.

    first[b,t]: reset the carried state to 0 before step t (episode start).
    valid[b,t]: if False the step is padding: the state is carried through unchanged.
    Uses the parameters of an nn.GRUCell (identical count to a 1-layer nn.GRU).
    Returns (outputs [B,T,H] = state after each step, final state [B,H]).
    """
    B, T, _ = x.shape
    gi = F.linear(x, cell.weight_ih, cell.bias_ih)
    h = h0
    outs = []
    for t in range(T):
        h = h * (~first[:, t]).unsqueeze(-1).to(h.dtype)
        gh = F.linear(h, cell.weight_hh, cell.bias_hh)
        ir, iz, inn = gi[:, t].chunk(3, -1)
        hr, hz, hn = gh.chunk(3, -1)
        r = torch.sigmoid(ir + hr)
        z = torch.sigmoid(iz + hz)
        n = torch.tanh(inn + r * hn)
        h_new = (1.0 - z) * n + z * h
        h = torch.where(valid[:, t].unsqueeze(-1), h_new, h)
        outs.append(h)
    return torch.stack(outs, 1), h


class PointerHead(nn.Module):
    def __init__(self, q_in, d=D):
        super().__init__()
        self.q = nn.Linear(q_in, d)
        self.k = nn.Linear(d, d)
        self.null = nn.Parameter(torch.randn(d) * 0.1)
        self.scale = d ** -0.5

    def keys(self, e, n):
        """e [M,N,d] -> keys [M,N+1,d] with the null key at position n (env-space layout)."""
        M, N, d = e.shape
        k = torch.cat([self.k(e), e.new_zeros(M, 1, d)], 1)
        is_null = torch.arange(N + 1, device=e.device)[None, :] == n[:, None]
        return torch.where(is_null.unsqueeze(-1), self.null.expand(M, N + 1, d), k)

    def logits(self, q_in, keys):
        return torch.einsum("md,mkd->mk", self.q(q_in), keys) * self.scale


@dataclass
class HeadOut:
    actions: torch.Tensor          # [M,15] long, columns in spec.HEAD_NAMES order
    logp: torch.Tensor             # [M,15] log-prob of the taken option (0 for single-option heads)
    ent: torch.Tensor              # [M,15] entropy / log K, 0 where K <= 1
    k: torch.Tensor                # [M,15] number of legal options
    legal: torch.Tensor            # [M,15] bool: taken option is legal under the effective mask
    fallback: torch.Tensor         # [M,15] bool: the selected mask row was empty (env bug), default used
    logp_all: Optional[Dict[str, torch.Tensor]] = None   # head -> [M,K] masked log-softmax
    eff: Optional[Dict[str, torch.Tensor]] = None        # head -> [M,K] effective mask


def _onehot_pos(pos, idx):
    return pos[None, :] == idx[:, None]


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.ent = EntityEncoder(spec.ENT_DIM)
        self.own = nn.Sequential(nn.Linear(spec.OWN_DIM, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU())
        self.fuse = nn.Linear(128 + D + spec.INTENT_DIM, H)
        self.fuse_ln = nn.LayerNorm(H)
        self.gru = nn.GRUCell(H, H)
        self.ptr = nn.ModuleDict({
            "maneuver_ref": PointerHead(H),
            "target": PointerHead(H + D),            # + selected maneuver-ref feature
            "view_object": PointerHead(H),
        })
        cond_in = {}
        for h in ("maneuver", "vertical", "speed", "chaff"):
            cond_in[h] = H + D                       # GRU + maneuver ref
        for h in ("radar_mode", "antenna", "weapon"):
            cond_in[h] = H + 2 * D                   # GRU + maneuver ref + target
        for h in ("view_mode", "look_az", "look_el"):
            cond_in[h] = H
        for h in ("kb_roll", "kb_pitch"):
            cond_in[h] = H + 2 * D                   # GRU + maneuver ref + view object
        self.cond_in = cond_in
        self.cat = nn.ModuleDict({h: nn.Linear(cond_in[h], spec.CAT_SIZES[h]) for h in spec.CAT_HEADS})
        for lin in self.cat.values():
            nn.init.orthogonal_(lin.weight, gain=0.01)
            nn.init.zeros_(lin.bias)

    # ---------------------------------------------------------------- encoding
    def encode(self, b: Batch, entities: bool = True):
        """-> x [B,T,256] GRU input; e [B,T,N,64] per-entity Transformer output (or None)."""
        B, T, N = b.ent.shape[:3]
        mask = torch.arange(N, device=b.ent.device)[None, :] < b.ent_n.reshape(B * T, 1)
        e, pooled = self.ent(b.ent.reshape(B * T, N, spec.ENT_DIM), mask)
        o = self.own(b.own.reshape(B * T, spec.OWN_DIM))
        x = F.gelu(self.fuse_ln(self.fuse(torch.cat([o, pooled, b.prev.reshape(B * T, -1)], -1))))
        return x.view(B, T, H), (e.view(B, T, N, D) if entities else None)

    def unroll(self, x, h0, first, valid):
        return gru_unroll(self.gru, x, h0, first, valid)

    # ---------------------------------------------------------------- heads
    @staticmethod
    def _select(head, masks, acts, n, pos):
        """The 1-D mask of `head` for the already sampled earlier heads (section 13.1 tables).

        Returns (row [M,K] bool, fallback [M] bool). An empty selected row is an env bug; it is
        replaced by the head's default option and reported through `fallback`.
        """
        m = masks[head]
        kind = spec.MASK_KIND[head]
        ar = torch.arange(m.shape[0], device=m.device)
        if kind == "flat":
            row = m
        elif kind == "view":
            row = m[ar, acts["view_mode"]]
        elif kind == "view_ref":
            row = m[ar, acts["view_mode"], (acts["maneuver_ref"] == n).long()]
        else:                                    # target-indexed
            row = m[ar, acts["target"].clamp(max=m.shape[1] - 1)]
        empty = ~row.any(-1)
        d = spec.DEFAULT_OPTION[head]
        def_idx = n if d == spec.NULL else torch.full_like(n, d)
        return torch.where(empty.unsqueeze(-1), _onehot_pos(pos, def_idx), row), empty

    def heads(self, h, e, n, masks, actions=None, gen=None, greedy=False, keep_dists=False) -> HeadOut:
        """Evaluate (actions given) or sample (actions None) all 15 heads in conditioning order.

        h [M,256]; e [M,N,64]; n [M] entity counts; masks {head: [M,*shape]} bool (13.1 tables,
        pointer widths and target rows N+1); actions [M,15] long.
        """
        M, N = e.shape[0], e.shape[1]
        dev = h.device
        pos_ptr = torch.arange(N + 1, device=dev)
        keys = {}
        acts, feats = {}, {}
        logp = {}; ent = {}; kk = {}; legal = {}; fb = {}
        dists = {} if keep_dists else None
        effs = {} if keep_dists else None
        ar = torch.arange(M, device=dev)

        def finish(head, logits, pos):
            eff, fb[head] = self._select(head, masks, acts, n, pos)
            lg = logits.masked_fill(~eff, NEG)
            lp_all = torch.log_softmax(lg, -1)
            if actions is not None:
                a = actions[:, spec.HEAD_INDEX[head]]
            elif greedy:
                a = lg.argmax(-1)
            else:
                u = torch.rand(lg.shape, generator=gen, device=dev).clamp_(1e-10, 1.0 - 1e-7)
                a = (lg - torch.log(-torch.log(u))).argmax(-1)
            acts[head] = a
            logp[head] = lp_all.gather(1, a.unsqueeze(1)).squeeze(1)
            legal[head] = eff.gather(1, a.unsqueeze(1)).squeeze(1)
            K_l = eff.sum(-1)
            kk[head] = K_l
            p = lp_all.exp()
            hent = -(p * lp_all).sum(-1)
            ent[head] = torch.where(K_l > 1, hent / torch.log(K_l.clamp(min=2).to(hent.dtype)), torch.zeros_like(hent))
            if keep_dists:
                dists[head] = lp_all
                effs[head] = eff

        def pointer(head, q_in):
            ph = self.ptr[head]
            ks = ph.keys(e, n)
            keys[head] = ks
            finish(head, ph.logits(q_in, ks), pos_ptr)
            feats[head] = ks[ar, acts[head]]

        def categorical(head, parts):
            inp = torch.cat([h] + [feats[p] for p in parts], -1) if parts else h
            finish(head, self.cat[head](inp), torch.arange(spec.CAT_SIZES[head], device=dev))

        categorical("view_mode", ())
        pointer("maneuver_ref", h)
        pointer("target", torch.cat([h, feats["maneuver_ref"]], -1))
        pointer("view_object", h)
        for hd in ("maneuver", "vertical", "speed", "chaff"):
            categorical(hd, ("maneuver_ref",))
        for hd in ("radar_mode", "antenna", "weapon"):
            categorical(hd, ("maneuver_ref", "target"))
        for hd in ("look_az", "look_el"):
            categorical(hd, ())
        for hd in ("kb_roll", "kb_pitch"):
            categorical(hd, ("maneuver_ref", "view_object"))
        order = spec.HEAD_NAMES
        return HeadOut(
            actions=torch.stack([acts[x] for x in order], -1),
            logp=torch.stack([logp[x] for x in order], -1),
            ent=torch.stack([ent[x] for x in order], -1),
            k=torch.stack([kk[x] for x in order], -1),
            legal=torch.stack([legal[x] for x in order], -1),
            fallback=torch.stack([fb[x] for x in order], -1),
            logp_all=dists, eff=effs,
        )

    # ---------------------------------------------------------------- convenience
    def heads_from_batch(self, b: Batch, hs, e, actions=None, gen=None, greedy=False, keep_dists=False):
        """hs [B,T,256], e [B,T,N,64]; flattens [B,T] -> M and returns HeadOut with leading [B*T]."""
        B, T, N = e.shape[:3]
        M = B * T
        masks = {k: v.reshape(M, *v.shape[2:]) for k, v in b.masks.items()}
        return self.heads(
            hs.reshape(M, H), e.reshape(M, N, D), b.ent_n.reshape(M), masks,
            None if actions is None else actions.reshape(M, spec.N_HEADS),
            gen=gen, greedy=greedy, keep_dists=keep_dists)

    @torch.no_grad()
    def act(self, b: Batch, h, gen=None, greedy=False):
        """One decision for B streams (T=1). h [B,256] raw carried state. -> (HeadOut, new h)."""
        x, e = self.encode(b)
        hs, h_new = self.unroll(x, h, b.first, b.valid)
        out = self.heads_from_batch(b, hs, e, None, gen=gen, greedy=greedy)
        return out, h_new


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.ent = EntityEncoder(spec.ENT_DIM)
        self.truth = EntityEncoder(spec.TRUTH_DIM)
        self.own = nn.Sequential(nn.Linear(spec.OWN_DIM, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU())
        self.fuse = nn.Linear(128 + D + D + spec.INTENT_DIM, H)
        self.fuse_ln = nn.LayerNorm(H)
        self.gru = nn.GRUCell(H, H)
        self.value = nn.Linear(H, 1)
        nn.init.orthogonal_(self.value.weight, gain=1.0)
        nn.init.zeros_(self.value.bias)

    def encode(self, b: Batch):
        B, T, N = b.ent.shape[:3]
        Mx = b.truth.shape[2]
        dev = b.ent.device
        m1 = torch.arange(N, device=dev)[None, :] < b.ent_n.reshape(B * T, 1)
        m2 = torch.arange(Mx, device=dev)[None, :] < b.truth_n.reshape(B * T, 1)
        _, p1 = self.ent(b.ent.reshape(B * T, N, spec.ENT_DIM), m1)
        _, p2 = self.truth(b.truth.reshape(B * T, Mx, spec.TRUTH_DIM), m2)
        o = self.own(b.own.reshape(B * T, spec.OWN_DIM))
        x = F.gelu(self.fuse_ln(self.fuse(torch.cat([o, p1, p2, b.prev.reshape(B * T, -1)], -1))))
        return x.view(B, T, H)

    def unroll(self, x, h0, first, valid):
        return gru_unroll(self.gru, x, h0, first, valid)

    def values(self, hs):
        return self.value(hs).squeeze(-1)

    def forward(self, b: Batch, h0):
        """Full sequence -> (values [B,T], states [B,T,256])."""
        hs, _ = self.unroll(self.encode(b), h0, b.first, b.valid)
        return self.values(hs), hs


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
