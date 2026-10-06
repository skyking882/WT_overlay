"""Recurrent PPO (docs/rl_training_spec.md sections 7 and 9).

* Minibatches are 16 *segments* of 80 steps, each with up to 16 burn-in steps in front
  (no gradient, no loss) that re-advance the stored hidden state with the current parameters.
* Joint-action ratio exp(sum over heads of logp_new - logp_old), clipped once (eps = 0.15).
* Separate Adam optimisers for actor (1e-4) and critic (3e-4), grad clip 0.5 each.
* target-KL 0.02 on the joint action (k3 estimator): once a minibatch exceeds it, the remaining
  actor updates of the round are skipped (critic keeps training).
* actor loss = PPO - ent_coef * normalised entropy + beta * KL(pi_BC || pi)
  (KL averaged over heads with >1 legal option) [+ sum over ppo.head_kl heads of coef_h * KL_h, the head's k3 KL to
  the behaviour policy per valid step; coef_h adapts per round toward its target]; critic loss = MSE / (return std)^2.
* advantages normalised over the whole valid batch of the round; padding never enters any
  loss or statistic.
"""
from __future__ import annotations

import time
from typing import Dict, Optional

import torch

from rl import spec
from rl.buffer import RoundBuffer
from rl.model import Actor, Critic, HeadOut


class EntropySchedule:
    """0.01 for the first `hold` decisions, linear to `end_coef` at `end`; decay pauses on collapse."""

    def __init__(self, c0, c1, hold, end, pause_below):
        self.c0, self.c1, self.hold, self.end, self.pause_below = c0, c1, hold, end, pause_below
        self.paused_decisions = 0.0
        self.paused = False

    def coef(self, decisions):
        d = decisions - self.paused_decisions
        if d <= self.hold:
            return self.c0
        if d >= self.end:
            return self.c1
        return self.c0 + (self.c1 - self.c0) * (d - self.hold) / (self.end - self.hold)

    def observe(self, mean_norm_entropy, decisions_in_round):
        """Call after a round: a collapsed entropy freezes the schedule for this round's decisions."""
        self.paused = mean_norm_entropy < self.pause_below
        if self.paused:
            self.paused_decisions += decisions_in_round

    def state_dict(self):
        return {"paused_decisions": self.paused_decisions, "paused": self.paused}

    def load_state_dict(self, d):
        self.paused_decisions = d["paused_decisions"]
        self.paused = d["paused"]


class BetaSchedule:
    """BC-reference KL weight: constant for `hold` decisions then linear to 0 at `end`."""

    def __init__(self, b0, hold, end):
        self.b0, self.hold, self.end = b0, hold, end

    def beta(self, decisions):
        if decisions <= self.hold:
            return self.b0
        if decisions >= self.end:
            return 0.0
        return self.b0 * (self.end - decisions) / (self.end - self.hold)


def clipped_surrogate(ratio, adv, clip):
    """PPO objective term min(r A, clip(r, 1-eps, 1+eps) A), clipped once on the JOINT-action ratio."""
    return torch.min(ratio * adv, ratio.clamp(1.0 - clip, 1.0 + clip) * adv)


def actor_window(actor: Actor, batch, h0, burn, actions, keep_dists=False, grad=True):
    """Forward the actor over [burn-in | segment] windows. Returns HeadOut over the segment part.

    batch has T = burn + L steps. The burn-in part runs without gradient from the stored state
    h0 [B,256]; the segment part runs with gradient from the burned-in state.
    Output tensors are flattened [B*L].
    """
    if burn > 0:
        with torch.no_grad():
            xb, _ = actor.encode(batch.time_slice(0, burn), entities=False)
            _, hb = actor.unroll(xb, h0, batch.first[:, :burn], batch.valid[:, :burn])
    else:
        hb = h0
    seg = batch.time_slice(burn, batch.shape[1])
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        x, e = actor.encode(seg)
        hs, _ = actor.unroll(x, hb, seg.first, seg.valid)
        out = actor.heads_from_batch(seg, hs, e, actions=actions, keep_dists=keep_dists)
    return out, seg


def critic_window(critic: Critic, batch, h0, burn):
    if burn > 0:
        with torch.no_grad():
            xb = critic.encode(batch.time_slice(0, burn))
            _, hb = critic.unroll(xb, h0, batch.first[:, :burn], batch.valid[:, :burn])
    else:
        hb = h0
    seg = batch.time_slice(burn, batch.shape[1])
    x = critic.encode(seg)
    hs, _ = critic.unroll(x, hb, seg.first, seg.valid)
    return critic.values(hs)


def ref_kl(out: HeadOut, ref_cat, ref_ptr):
    """Per-step KL(pi_ref || pi), averaged over heads with more than one legal option. [M]."""
    kls = []
    for hname in spec.HEAD_NAMES:
        lp = out.logp_all[hname]
        eff = out.eff[hname]
        if hname in spec.POINTER_HEADS:
            lr = ref_ptr[:, spec.POINTER_HEADS.index(hname), :lp.shape[-1]]
        else:
            off, sz = spec.CAT_OFFSETS[hname], spec.CAT_SIZES[hname]
            lr = ref_cat[:, off:off + sz]
        pr = lr.exp()
        kl = torch.where(eff, pr * (lr - lp), torch.zeros_like(lp)).sum(-1)
        kls.append(kl)
    kl = torch.stack(kls, -1)                       # [M,15]
    active = (out.k > 1).to(kl.dtype)
    return (kl * active).sum(-1) / active.sum(-1).clamp(min=1.0), kl, active


class PPOTrainer:
    def __init__(self, cfg, actor: Actor, critic: Critic, ref: Optional[Actor], device):
        self.cfg = cfg
        self.p = cfg.ppo
        self.actor, self.critic, self.ref, self.device = actor, critic, ref, device
        p = self.p
        self.opt_a = torch.optim.Adam(actor.parameters(), lr=p.lr_actor, betas=(0.9, 0.999), eps=p.adam_eps, weight_decay=0.0)
        self.opt_c = torch.optim.Adam(critic.parameters(), lr=p.lr_critic, betas=(0.9, 0.999), eps=p.adam_eps, weight_decay=0.0)
        self.ent_sched = EntropySchedule(p.ent_coef, p.ent_coef_end, p.ent_hold, p.ent_end, p.ent_pause_below)
        unknown = set(p.ent_head_scale) - set(spec.HEAD_NAMES)
        if unknown:
            raise ValueError("ppo.ent_head_scale has unknown heads: %s" % sorted(unknown))
        self.ent_scale = torch.tensor([float(p.ent_head_scale.get(h, 1.0)) for h in spec.HEAD_NAMES],
                                      device=self.device)
        self.beta_sched = BetaSchedule(p.kl_beta, p.kl_beta_hold, p.kl_beta_end)
        # ppo.head_kl: head -> (index, target, coef_min, coef_max); the adapted coefficients live in the trainer state
        unknown = set(p.head_kl) - set(spec.HEAD_NAMES)
        if unknown:
            raise ValueError("ppo.head_kl has unknown heads: %s" % sorted(unknown))
        self.head_kl = {h: (spec.HEAD_INDEX[h], float(v["target"]), float(v.get("coef_min", 0.01)),
                            float(v.get("coef_max", 100.0))) for h, v in p.head_kl.items()}
        self.head_kl_coef = {h: float(v.get("coef", 1.0)) for h, v in p.head_kl.items()}
        self.decisions = 0
        self.round = 0
        self.gen = torch.Generator()
        self.gen.manual_seed(cfg.run.seed + 17)

    # ------------------------------------------------------------------ state
    def state_dict(self):
        return {"opt_a": self.opt_a.state_dict(), "opt_c": self.opt_c.state_dict(),
                "ent": self.ent_sched.state_dict(), "decisions": self.decisions, "round": self.round,
                "gen": self.gen.get_state(), "head_kl_coef": dict(self.head_kl_coef)}

    def load_state_dict(self, d):
        self.opt_a.load_state_dict(d["opt_a"])
        self.opt_c.load_state_dict(d["opt_c"])
        self.ent_sched.load_state_dict(d["ent"])
        self.decisions = d["decisions"]
        self.round = d["round"]
        self.gen.set_state(d["gen"])
        for h, c in (d.get("head_kl_coef") or {}).items():    # heads still configured keep their adapted coef
            if h in self.head_kl_coef:
                self.head_kl_coef[h] = float(c)

    # ------------------------------------------------------------------ update
    def update(self, buf: RoundBuffer) -> Dict:
        p, dev = self.p, self.device
        S, L, B = buf.S, buf.L, buf.B
        t_start = time.time()
        valid = buf.loss_view(buf.store.valid).reshape(-1)             # [T*S] index t*S+s
        n_valid = int(valid.sum())
        if n_valid == 0:
            raise RuntimeError("round without a single valid decision (every stream idle?)")
        adv_raw, ret = buf.adv, buf.ret
        mean, std = adv_raw[valid].mean(), adv_raw[valid].std(unbiased=False)
        adv = torch.where(valid, (adv_raw - mean) / (std + 1e-8), torch.zeros_like(adv_raw))
        ret_std = float(ret[valid].std(unbiased=False)) if n_valid > 1 else 1.0
        vscale = max(ret_std, p.value_std_floor)                       # fixed for the whole round
        v_old = buf.value
        ev_den = float(ret[valid].var(unbiased=False))
        explained_var = 1.0 - float((ret[valid] - v_old[valid]).var(unbiased=False)) / ev_den if ev_den > 1e-12 else float("nan")

        ent_coef = self.ent_sched.coef(self.decisions)
        beta = self.beta_sched.beta(self.decisions) if self.ref is not None else 0.0
        warm = self.round < p.critic_warmup_rounds
        # segments with at least one valid loss step
        segs = []
        vm = buf.loss_view(buf.store.valid)                            # [T,S]
        for s in range(S):
            for k in range(buf.K):
                if bool(vm[k * L:(k + 1) * L, s].any()):
                    segs.append((s, k))
        n_seg = len(segs)
        seg_s = torch.tensor([a for a, _ in segs])
        seg_k = torch.tensor([b for _, b in segs])

        acc = {"pg": 0.0, "v": 0.0, "kl_old": 0.0, "clip": 0.0, "ent": 0.0, "kl_ref": 0.0,
               "gn_a": 0.0, "gn_c": 0.0, "n_a": 0, "n_c": 0, "illegal": 0, "fallback": 0, "steps": 0.0,
               "kl_max": 0.0, "kl_last": 0.0, "kl_skips": 0,
               "ratio_max": 0.0}
        head_ent = torch.zeros(spec.N_HEADS)
        head_cnt = torch.zeros(spec.N_HEADS)
        head_kl = torch.zeros(spec.N_HEADS)
        head_klc = torch.zeros(spec.N_HEADS)
        head_k3 = torch.zeros(spec.N_HEADS)             # per-head k3 KL to the behaviour policy, sum over actor steps
        hk = [(h,) + self.head_kl[h] + (self.head_kl_coef[h],) for h in self.head_kl]   # coefs fixed for the round
        hk_acc = {h: 0.0 for h in self.head_kl}
        actor_stopped = False
        stopped_at = -1
        stop_diag = skip_diag = None
        mb_count = 0
        actor_steps = 0
        self.actor.train()
        self.critic.train()
        mb = p.minibatch_segments
        for epoch in range(p.epochs):
            perm = torch.randperm(n_seg, generator=self.gen)
            for a in range(0, n_seg, mb):
                sel = perm[a:a + mb]
                s_b, k_b = seg_s[sel], seg_k[sel]
                widx = buf.window_index(s_b, k_b)                      # [b, B+L] flat step ids
                lidx = buf.loss_index(s_b, k_b)                        # [b, L] t*S+s ids
                batch = buf.store.gather(widx).to(dev)
                h0a = torch.stack([buf.h_actor[int(k)][int(s)] for s, k in zip(s_b, k_b)]).to(dev)
                h0c = torch.stack([buf.h_critic[int(k)][int(s)] for s, k in zip(s_b, k_b)]).to(dev)
                old_logp = buf.logp[widx[:, B:]].reshape(-1).to(dev)
                a_b = adv[lidx].reshape(-1).to(dev)
                r_b = ret[lidx].reshape(-1).to(dev)
                lm = batch.time_slice(B, B + L).valid.reshape(-1)
                n_lm = lm.sum().clamp(min=1).to(torch.float32)
                lmf = lm.to(torch.float32)
                mb_count += 1
                # ---------------- actor
                if (not warm) and not actor_stopped:
                    out, seg = actor_window(self.actor, batch, h0a, B, batch.act[:, B:], keep_dists=True)
                    logp = out.logp.sum(-1)
                    logr = (logp - old_logp).clamp(-20.0, 20.0)
                    ratio = logr.exp()
                    with torch.no_grad():
                        kl_old = (((ratio - 1.0) - logr) * lmf).sum() / n_lm
                        clipf = (((ratio - 1.0).abs() > p.clip).to(torch.float32) * lmf).sum() / n_lm
                        old_h = buf.logp_heads[widx[:, B:]].reshape(-1, spec.N_HEADS).to(dev)
                        lr_h = (out.logp.reshape(-1, spec.N_HEADS) - old_h).clamp(-20.0, 20.0)
                        k3 = ((lr_h.exp() - 1.0) - lr_h) * lmf.unsqueeze(-1)
                    acc["kl_last"] = float(kl_old)
                    acc["kl_max"] = max(acc["kl_max"], float(kl_old))
                    skip_mb = False
                    if p.kl_mode == "skip":
                        if float(kl_old) > p.target_kl_skip:
                            skip_mb = True
                            acc["kl_skips"] += 1
                        elif (acc["kl_old"] + float(kl_old)) / (acc["n_a"] + 1) > p.target_kl:
                            actor_stopped = True
                            stopped_at = mb_count
                    elif float(kl_old) > p.target_kl:
                        actor_stopped = True
                        stopped_at = mb_count
                    if actor_stopped or (skip_mb and skip_diag is None):
                        # Diagnostics: which heads moved in the minibatch that stopped the actor (or the first one
                        # skipped): per-head k3 KL to the behaviour policy, and how many valid steps changed a head's
                        # log-prob by more than 1.
                        with torch.no_grad():
                            big = ((lr_h.abs() > 1.0).to(torch.float32) * lmf.unsqueeze(-1)).sum(0)
                            diag = {"minibatch": mb_count, "kl": float(kl_old), "steps": int(n_lm),
                                    "kl_heads": {h: round(float(v), 5) for h, v in
                                                 zip(spec.HEAD_NAMES, (k3.sum(0) / n_lm).cpu()) if v > 1e-5},
                                    "steps_logp_moved_gt1": {h: int(v) for h, v in
                                                             zip(spec.HEAD_NAMES, big.cpu()) if v > 0}}
                        if actor_stopped:
                            stop_diag = diag
                        else:
                            skip_diag = diag
                    if not actor_stopped and not skip_mb:
                        pg = -clipped_surrogate(ratio, a_b, p.clip)
                        pg_loss = (pg * lmf).sum() / n_lm
                        active = (out.k > 1).to(torch.float32)
                        ent_step = (out.ent * active).sum(-1) / active.sum(-1).clamp(min=1.0)
                        ent_mean = (ent_step * lmf).sum() / n_lm
                        # Bonus uses per-head scales (ppo.ent_head_scale); the logged entropy stays unweighted.
                        bonus_step = (out.ent * active * self.ent_scale).sum(-1) / active.sum(-1).clamp(min=1.0)
                        ent_bonus = (bonus_step * lmf).sum() / n_lm
                        if self.ref is not None:
                            rc = buf.ref_cat[lidx.reshape(-1)].to(dev)
                            rp = buf.ref_ptr[lidx.reshape(-1)][:, :, :out.logp_all["target"].shape[-1]].to(dev)
                            klr_step, kl_heads, _ = ref_kl(out, rc, rp)
                            klr = (klr_step * lmf).sum() / n_lm
                        else:
                            klr = torch.zeros((), device=dev)
                            kl_heads = None
                        loss = pg_loss - ent_coef * ent_bonus + beta * klr
                        for h, i, _, _, _, c in hk:             # ppo.head_kl: k3 KL of one head, with gradient
                            lr_i = (out.logp.reshape(-1, spec.N_HEADS)[:, i] - old_h[:, i]).clamp(-20.0, 20.0)
                            kl_i = (((lr_i.exp() - 1.0) - lr_i) * lmf).sum() / n_lm
                            loss = loss + c * kl_i
                            hk_acc[h] += float(kl_i.detach())
                        self.opt_a.zero_grad(set_to_none=True)
                        loss.backward()
                        gn = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), p.max_grad_norm)
                        self.opt_a.step()
                        actor_steps += 1
                        with torch.no_grad():
                            acc["pg"] += float(pg_loss); acc["ent"] += float(ent_mean)
                            acc["kl_old"] += float(kl_old); acc["clip"] += float(clipf)
                            acc["kl_ref"] += float(klr); acc["gn_a"] += float(gn); acc["n_a"] += 1
                            acc["ratio_max"] = max(acc["ratio_max"], float((ratio * lmf).max()))
                            acc["illegal"] += int(((~out.legal) & lm.unsqueeze(-1)).sum())
                            acc["fallback"] += int((out.fallback & lm.unsqueeze(-1)).sum())
                            acc["steps"] += float(lmf.sum())
                            head_k3 += (k3.sum(0) / n_lm).cpu()
                            m2 = lmf.unsqueeze(-1) * active
                            head_ent += (out.ent * m2).sum(0).cpu()
                            head_cnt += m2.sum(0).cpu()
                            if kl_heads is not None:
                                head_kl += (kl_heads * m2).sum(0).cpu()
                                head_klc += m2.sum(0).cpu()
                # ---------------- critic
                v_new = critic_window(self.critic, batch, h0c, B).reshape(-1)
                v_loss = (((v_new - r_b) ** 2) * lmf).sum() / n_lm / (vscale ** 2)
                self.opt_c.zero_grad(set_to_none=True)
                v_loss.backward()
                gc = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), p.max_grad_norm)
                self.opt_c.step()
                acc["v"] += v_loss.item(); acc["gn_c"] += float(gc); acc["n_c"] += 1
        na, nc = max(acc["n_a"], 1), max(acc["n_c"], 1)
        names = spec.HEAD_NAMES
        head_kl_m = {}
        for h, _, target, lo, hi, c in hk:
            # adapt toward the target for the next round: x1.5 above 1.5 * target, /1.5 below target / 1.5
            kl_h = hk_acc[h] / acc["n_a"] if acc["n_a"] else None
            nxt = c if kl_h is None else min(max(c * 1.5 if kl_h > 1.5 * target else c / 1.5 if kl_h < target / 1.5
                                                 else c, lo), hi)
            self.head_kl_coef[h] = nxt
            head_kl_m[h] = {"kl": kl_h, "coef": c, "coef_next": nxt, "target": target}
        cnt = head_cnt.clamp(min=1)
        mean_ent = acc["ent"] / na if acc["n_a"] else float("nan")
        m = {
            "ent_coef": ent_coef, "kl_beta": beta,
            "pg_loss": acc["pg"] / na, "value_loss": acc["v"] / nc,
            # value_loss is normalised by value_scale^2; this is its RMS error in reward units
            "value_rmse": (acc["v"] / nc) ** 0.5 * vscale,
            "entropy": mean_ent,
            "kl_target": acc["kl_old"] / na,         # joint-action KL to the behaviour policy (k3), mean over actor steps
            "kl_target_max": acc["kl_max"],          # largest per-minibatch KL seen, incl. the one that stopped the actor
            "kl_target_last": acc["kl_last"],
            "kl_stop": stop_diag,                    # per-head view of the minibatch that stopped the actor, if any
            "kl_skipped_minibatches": acc["kl_skips"],   # kl_mode "skip": minibatches left out for a KL spike
            "kl_skip_first": skip_diag,
            "kl_ref": acc["kl_ref"] / na if self.ref is not None else float("nan"),   # to the frozen BC policy
            "clip_frac": acc["clip"] / na,
            "ratio_max": acc["ratio_max"],
            "grad_norm_actor": acc["gn_a"] / na, "grad_norm_critic": acc["gn_c"] / nc,
            "explained_variance": explained_var,
            "adv_mean_raw": float(mean), "adv_std_raw": float(std), "return_std": ret_std, "value_scale": vscale,
            "actor_steps": actor_steps, "minibatches": mb_count, "n_segments": n_seg,
            "actor_stopped_at_minibatch": stopped_at, "illegal_actions_taken": acc["illegal"], "mask_fallbacks": acc["fallback"],
            "entropy_head": {names[i]: float(head_ent[i] / cnt[i]) for i in range(spec.N_HEADS) if head_cnt[i] > 0},
            "kl_ref_head": {names[i]: float(head_kl[i] / head_klc[i].clamp(min=1)) for i in range(spec.N_HEADS) if head_klc[i] > 0},
            # per-head k3 KL to the behaviour policy per valid step, mean over the applied actor minibatches (like
            # kl_target; the joint k3 is not the sum of these)
            "kl_target_head": {names[i]: float(head_k3[i] / na) for i in range(spec.N_HEADS) if head_k3[i] > 0},
            "head_active_frac": {names[i]: float(head_cnt[i] / max(acc["steps"], 1.0)) for i in range(spec.N_HEADS)},
            "t_update": time.time() - t_start,
            "decisions_in_round": n_valid,
        }
        if hk:
            m["head_kl"] = head_kl_m             # {head: {kl (round mean), coef (used), coef_next, target}}
        self.decisions += n_valid
        self.round += 1
        if acc["n_a"] and mean_ent == mean_ent:
            self.ent_sched.observe(mean_ent, n_valid)
        m["entropy_paused"] = self.ent_sched.paused
        return m
