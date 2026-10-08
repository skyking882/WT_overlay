"""Round buffer, GAE and the post-rollout passes (critic values, BC-reference distributions).

Layout of one PPO round
-----------------------
S streams x (P + T) ticks, flat step index  i = (t + P) * S + s  for tick t in [-P, T).
P = burn_in: the last P ticks of the previous round are copied in front, so that the burn-in
of the first segment sees real history. Loss ticks are t in [0, T). Segment k of a stream
covers loss ticks [kL, kL+L) with the burn-in window [kL-B, kL) in front of it.

Recurrent state handling
------------------------
* actor: the raw state *entering* tick kL-B (k = 1..K) is stored during the rollout (R2D2 style
  stored state); the state for k = 0 is carried over from the previous round.
* critic: stored the same way, computed by the post-pass (the critic is not run per tick).
* BC reference: frozen, so its states are exact; it runs once over the whole round after
  sampling, teacher-forced on the sampled actions, and its distributions are stored per step.
"""
from __future__ import annotations

from typing import Dict, List

import torch

from rl import spec
from rl.encode import Decoded, StepStore
from rl.model import NEG, Actor, Critic

H = 256


class RoundBuffer:
    def __init__(self, S, T, L, B):
        assert T % L == 0 and B <= L
        self.S, self.T, self.L, self.B = S, T, L, B
        self.P = B
        self.K = T // L
        self.n_steps = (self.P + T) * S
        self.store = StepStore(self.n_steps)
        N = self.n_steps
        self.logp = torch.zeros(N)
        self.logp_heads = torch.zeros(N, spec.N_HEADS)   # per-head part of logp (diagnostics only)
        self.reward = torch.zeros(N)
        self.done = torch.zeros(N, dtype=torch.bool)      # terminal (agent died / match ended)
        self.trunc = torch.zeros(N, dtype=torch.bool)     # truncated: bootstrap from the value net
        self.boot_final = torch.zeros(N, dtype=torch.bool)  # bootstrap obs available (else own value)
        self.aircraft = torch.zeros(N, dtype=torch.long)
        # teacher labels (env teacher, docs/kickstart_spec.md): option per head of the decision, -1 = no label;
        # teacher_name: index into teacher_names (the sampler's list), -1 = none. Frozen-side steps are never stored.
        self.teacher = torch.full((N, spec.N_HEADS), -1, dtype=torch.int16)
        self.teacher_name = torch.full((N,), -1, dtype=torch.int8)
        self.teacher_names: List[str] = []
        self.boot_obs: List[tuple] = []                   # (flat idx, wire obs) of final timeout obs
        self.end_obs: Dict[int, tuple] = {}               # stream -> wire obs pending after the last tick
        self.end_first: Dict[int, bool] = {}
        self.h_actor: Dict[int, torch.Tensor] = {}        # k -> [S,256] state entering tick kL-B
        self.h_critic: Dict[int, torch.Tensor] = {}
        # outputs of the post-pass / GAE (loss region only, laid out [T*S], index t*S+s)
        self.value = torch.zeros(T * S)
        self.boot_value = torch.zeros(T * S)
        self.end_value = torch.zeros(S)
        self.end_valid = torch.zeros(S, dtype=torch.bool)
        self.adv = torch.zeros(T * S)
        self.ret = torch.zeros(T * S)
        self.ref_cat = None       # [T*S, 50]
        self.ref_ptr = None       # [T*S, 3, 65]
        self.h_critic_end = None  # [S,256] critic state after the last tick
        self.h_ref_end = None

    def extend(self, n):
        """Append n ticks (a multiple of L) of padding to the loss region, before the post-pass: room for the settle
        ticks of Sampler._settle. Flat indices of the stored steps (and the stored entity rows) do not change."""
        assert n % self.L == 0 and not self.h_critic
        add = n * self.S
        st = self.store
        for obj, names in ((st, ("own", "prev", "mask", "ent_start", "ent_n", "tr_start", "tr_n", "act", "valid",
                                 "first")),
                           (self, ("logp", "logp_heads", "reward", "done", "trunc", "boot_final", "aircraft"))):
            for name in names:
                x = getattr(obj, name)
                setattr(obj, name, torch.cat([x, x.new_zeros((add,) + tuple(x.shape[1:]))]))
        for name in ("teacher", "teacher_name"):        # padding carries no label (-1)
            x = getattr(self, name)
            setattr(self, name, torch.cat([x, x.new_full((add,) + tuple(x.shape[1:]), -1)]))
        st.dt = torch.cat([st.dt, torch.full((add,), spec.DT_STEP)])
        st.N += add
        self.T += n
        self.K = self.T // self.L
        self.n_steps += add
        TS = self.T * self.S
        self.value, self.boot_value, self.adv, self.ret = (torch.zeros(TS) for _ in range(4))

    # step index helpers
    def flat(self, t, s):
        return (t + self.P) * self.S + s

    def loss_view(self, x):
        """flat per-step array -> [T,S] view of the loss region."""
        return x[self.P * self.S:].view(self.T, self.S)

    def window_index(self, s: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """[n] stream, [n] segment -> [n, B+L] flat step indices of burn-in + segment."""
        t = k.unsqueeze(1) * self.L - self.B + torch.arange(self.B + self.L).unsqueeze(0)
        return (t + self.P) * self.S + s.unsqueeze(1)

    def loss_index(self, s, k):
        """[n],[n] -> [n,L] loss-region flat index t*S+s (for adv/ret/ref tables)."""
        t = k.unsqueeze(1) * self.L + torch.arange(self.L).unsqueeze(0)
        return t * self.S + s.unsqueeze(1)

    def n_valid(self):
        return int(self.loss_view(self.store.valid).sum())


# ---------------------------------------------------------------------------
# GAE
# ---------------------------------------------------------------------------

def compute_gae(r, v, done, trunc, valid, dt, boot_v, end_v, end_valid, gamma_base=spec.GAMMA_BASE,
                lambda_base=spec.LAMBDA_BASE):
    """Generalised advantage estimation with per-step discount from the real step length.

    All inputs are [T,S] (end_* are [S]). gamma_t = gamma_base**dt_t, lambda_t = lambda_base**dt_t.
    done: terminal step (value of the next state is 0). trunc: timeout / lost agent, the
    next-state value is boot_v (bootstrapped, not terminal). The chain also stops at the end
    of the round, where end_v bootstraps streams whose episode continues (end_valid).
    Returns (adv, ret) both [T,S]; invalid steps get 0.
    """
    T, S = r.shape
    gam = gamma_base ** dt
    lam = lambda_base ** dt
    adv = torch.zeros_like(r)
    last = torch.zeros(S, dtype=r.dtype)
    for t in range(T - 1, -1, -1):
        if t == T - 1:
            nxt_v, nxt_ok = end_v, end_valid
        else:
            nxt_v, nxt_ok = v[t + 1], valid[t + 1]
        cont = valid[t] & ~done[t] & ~trunc[t] & nxt_ok
        nv = torch.where(cont, nxt_v, torch.where(trunc[t], boot_v[t], torch.zeros_like(nxt_v)))
        delta = r[t] + gam[t] * nv - v[t]
        last = delta + gam[t] * lam[t] * cont.to(r.dtype) * last
        last = torch.where(valid[t], last, torch.zeros_like(last))
        adv[t] = last
    return adv, (adv + v) * valid.to(r.dtype)


# ---------------------------------------------------------------------------
# post-rollout passes
# ---------------------------------------------------------------------------

def _chunks(n, size):
    for a in range(0, n, size):
        yield list(range(a, min(n, a + size)))


@torch.no_grad()
def critic_postpass(buf: RoundBuffer, critic: Critic, h_carry: torch.Tensor, h_prefix: torch.Tensor,
                    device, chunk=8):
    """Run the critic over the loss ticks of every stream.

    h_carry: critic state entering tick 0 (carried from the previous round's last tick).
    h_prefix: critic state entering tick -B (the previous round's stored state at tick T-B).

    Fills buf.value, buf.h_critic[k] (k = 1..K), buf.boot_value (timeout final obs and the
    fallback to the step's own value), buf.end_value and returns the critic state after the
    last tick (the carry for the next round).
    """
    S, T, L, B, P, K = buf.S, buf.T, buf.L, buf.B, buf.P, buf.K
    critic.eval()
    buf.h_critic[0] = h_prefix.clone()
    hs_all = torch.zeros(S, T, H)
    vals = torch.zeros(S, T)
    t_idx = torch.arange(T)
    for ss in _chunks(S, chunk):
        s_t = torch.tensor(ss)
        idx = (t_idx.unsqueeze(0) + P) * S + s_t.unsqueeze(1)           # [c,T]
        b = buf.store.gather(idx).to(device)
        v, hs = critic(b, h_carry[s_t].to(device))
        vals[s_t] = v.cpu()
        hs_all[s_t] = hs.cpu()
    buf.value = vals.t().reshape(-1).clone()                                # [T*S] index t*S+s
    for k in range(1, K + 1):
        t0 = k * L - B                                                      # state entering tick t0
        buf.h_critic[k] = hs_all[:, t0 - 1].clone() if t0 >= 1 else h_carry.clone()
    h_end = hs_all[:, T - 1].clone()
    # bootstrap values: timeouts (final obs) and the end of the round
    boot = buf.boot_value.view(T, S)
    if buf.boot_obs:
        dec = Decoded([w for _, w in buf.boot_obs])
        ts = torch.tensor([(i // S) - P for i, _ in buf.boot_obs])
        ss = torch.tensor([i % S for i, _ in buf.boot_obs])
        b = dec.to_batch().to(device)
        v, _ = critic(b, hs_all[ss, ts].to(device))
        boot[ts, ss] = v.squeeze(1).cpu()
    if buf.end_obs:
        ss_l = sorted(buf.end_obs)
        dec = Decoded([buf.end_obs[s] for s in ss_l])
        ss = torch.tensor(ss_l)
        first = torch.tensor([buf.end_first[s] for s in ss_l])
        b = dec.to_batch(first=first).to(device)
        v, _ = critic(b, h_end[ss].to(device))
        buf.end_value[ss] = v.squeeze(1).cpu()
        buf.end_valid[ss] = ~first
    # fallback bootstrap (lost agents / timeout without a final obs): the step's own value
    need_fb = buf.loss_view(buf.trunc) & ~buf.loss_view(buf.boot_final)
    vv = buf.value.view(T, S)
    boot.copy_(torch.where(need_fb, vv, boot))
    return h_end


@torch.no_grad()
def ref_postpass(buf: RoundBuffer, ref: Actor, h_carry: torch.Tensor, device, chunk=8):
    """Teacher-forced pass of the frozen BC reference over the loss ticks.

    Stores the reference's masked log-softmax tables: ref_cat [T*S,50], ref_ptr [T*S,3,65]
    (pointer tables padded with NEG), flat index t*S+s. Returns the reference state after the
    last tick.
    """
    S, T, P = buf.S, buf.T, buf.P
    ref.eval()
    cat = torch.full((T, S, spec.CAT_TOTAL), NEG)
    ptr = torch.full((T, S, 3, spec.PTR_CAP), NEG)
    h_end = h_carry.clone()
    t_idx = torch.arange(T)
    for ss in _chunks(S, chunk):
        s_t = torch.tensor(ss)
        idx = (t_idx.unsqueeze(0) + P) * S + s_t.unsqueeze(1)
        b = buf.store.gather(idx).to(device)
        x, e = ref.encode(b)
        hs, hl = ref.unroll(x, h_carry[s_t].to(device), b.first, b.valid)
        out = ref.heads_from_batch(b, hs, e, actions=b.act, keep_dists=True)
        c = len(ss)
        for hname in spec.CAT_HEADS:
            off, sz = spec.CAT_OFFSETS[hname], spec.CAT_SIZES[hname]
            cat[:, s_t, off:off + sz] = out.logp_all[hname].view(c, T, sz).permute(1, 0, 2).cpu()
        for j, hname in enumerate(spec.POINTER_HEADS):
            lp = out.logp_all[hname].view(c, T, -1).cpu()
            ptr[:, s_t, j, :lp.shape[-1]] = lp.permute(1, 0, 2)
        h_end[s_t] = hl.cpu()
    buf.ref_cat = cat.reshape(T * S, spec.CAT_TOTAL)
    buf.ref_ptr = ptr.reshape(T * S, 3, spec.PTR_CAP)
    buf.h_ref_end = h_end
    return h_end


def finish_round(buf: RoundBuffer, critic: Critic, h_critic_carry, h_critic_prefix, device,
                 gamma_base, lambda_base):
    """critic post-pass + GAE; fills buf.adv / buf.ret (raw, not normalised)."""
    h_end = critic_postpass(buf, critic, h_critic_carry, h_critic_prefix, device)
    T, S = buf.T, buf.S
    lv = buf.loss_view
    adv, ret = compute_gae(
        lv(buf.reward), buf.value.view(T, S), lv(buf.done), lv(buf.trunc), lv(buf.store.valid),
        lv(buf.store.dt), buf.boot_value.view(T, S), buf.end_value, buf.end_valid, gamma_base, lambda_base)
    buf.adv = adv.reshape(-1).clone()
    buf.ret = ret.reshape(-1).clone()
    return h_end
