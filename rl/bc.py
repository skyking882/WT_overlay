"""Behaviour cloning from scripted pilots (docs/rl_training_spec.md section 8).

Data: episodes of env.scripted_actions() (intents before executor noise), collected by the
workers, canonicalised against the masks (section 13.1), and stored in shards (truth tokens are
dropped: BC trains the actor only). Split 80/20 grouped by episode. Windows of 80 steps with 16
burn-in steps (zero initial state at the window start, no loss on burn-in), batch 32 windows,
Adam 3e-4, grad clip 0.5, at most 10 epochs, stop after 3 epochs without validation improvement.

Loss: per-head masked cross-entropy (pointer conditioning uses the demonstrated objects),
class weights fire x5 and maneuver switch (maneuver != keep) x3, normalised by the weight sum
inside each head, summed over heads. Heads with a single legal option carry no loss.
Reports: per-head validation loss and accuracy, fire precision / recall.
"""
from __future__ import annotations

import json
import os
import random
import time
from types import SimpleNamespace
from typing import Dict, List

import torch

from rl import spec
from rl.encode import Decoded, StepStore
from rl.model import Actor
from rl.ppo import actor_window

H = 256
IDX = spec.HEAD_INDEX


# ---------------------------------------------------------------------------
# shards
# ---------------------------------------------------------------------------

def build_shard(episodes) -> dict:
    wires, acts, trajs = [], [], []
    off = 0
    events: Dict[str, int] = {}
    for ep in episodes:
        for k, v in ep["events"].items():
            events[k] = events.get(k, 0) + v
        for aid, ag in ep["agents"].items():
            steps = ag["steps"]
            for o, a in steps:
                wires.append(o[:4] + (0, b"") + o[6:])        # drop the truth tokens
                acts.append(a)
            trajs.append({"episode": ep["episode_id"], "scenario": ep["scenario"], "agent": str(aid),
                          "aircraft": ag["aircraft"], "offset": off, "length": len(steps), "ret": ag["ret"],
                          "timeout": ep["timeout"]})
            off += len(steps)
    dec = Decoded(wires)
    return {"own": dec.own, "prev": dec.prev, "mask": dec.mask,
            "ent_n": dec.ent_n, "ent_rows": dec.ent_rows, "dt": dec.dt, "act": torch.tensor(acts, dtype=torch.long),
            "trajs": trajs, "events": events, "n_episodes": len(episodes)}


def save_shard(path, shard):
    tmp = path + ".tmp"
    torch.save(shard, tmp)
    os.replace(tmp, path)


class BCDataset:
    """All shards in one StepStore, trajectories split by episode."""

    def __init__(self, shards: List[dict], val_frac=0.2, seed=0):
        total = sum(int(s["own"].shape[0]) for s in shards)
        self.store = StepStore(total)
        self.trajs: List[dict] = []
        self.events: Dict[str, int] = {}
        self.n_episodes = 0
        off = 0
        for sh in shards:
            n = int(sh["own"].shape[0])
            if n == 0:
                continue
            ns = SimpleNamespace(B=n, own=sh["own"], prev=sh["prev"], mask=sh["mask"],
                                 ent_n=sh["ent_n"], ent_rows=sh["ent_rows"],
                                 truth_n=torch.zeros(n, dtype=torch.long), truth_rows=torch.zeros(0, spec.TRUTH_DIM),
                                 dt=sh["dt"])
            self.store.put(torch.arange(off, off + n), ns, act=sh["act"])
            for t in sh["trajs"]:
                t = dict(t)
                t["offset"] += off
                self.trajs.append(t)
                self.store.first[t["offset"]] = True
            for k, v in sh["events"].items():
                self.events[k] = self.events.get(k, 0) + v
            self.n_episodes += sh["n_episodes"]
            off += n
        self.store.finalize()
        eps = sorted({t["episode"] for t in self.trajs})
        rng = random.Random(seed)
        rng.shuffle(eps)
        n_val = max(1, int(round(len(eps) * val_frac))) if len(eps) > 1 else 0
        val_eps = set(eps[:n_val])
        self.split = {"train": [i for i, t in enumerate(self.trajs) if t["episode"] not in val_eps],
                      "val": [i for i, t in enumerate(self.trajs) if t["episode"] in val_eps]}

    def segments(self, which, seg_len):
        out = []
        for i in self.split[which]:
            for start in range(0, self.trajs[i]["length"], seg_len):
                out.append((i, start))
        return out

    def window_index(self, items, burn, seg_len):
        rows = []
        ar = torch.arange(burn + seg_len)
        for i, start in items:
            t = self.trajs[i]
            local = start - burn + ar
            idx = t["offset"] + local
            idx = torch.where((local >= 0) & (local < t["length"]), idx, torch.full_like(idx, -1))
            rows.append(idx)
        return torch.stack(rows)

    def stats(self):
        st = self.store
        n = st.N
        act = st.act
        valid = st.valid
        first = st.first
        vm = act[:, IDX["view_mode"]]
        fire = (act[:, IDX["weapon"]] == 1) & valid
        man = act[:, IDX["maneuver"]]
        # switches: the maneuver / vertical / reference changes against the previous step of the same trajectory
        prev_same = torch.zeros(n, dtype=torch.bool)
        prev_same[1:] = ~first[1:]
        def changes(h):
            a = act[:, IDX[h]]
            d = torch.zeros(n, dtype=torch.bool)
            d[1:] = a[1:] != a[:-1]
            return int((d & prev_same & valid).sum())
        enter_free = torch.zeros(n, dtype=torch.bool)
        enter_free[1:] = (vm[1:] != 0) & (vm[:-1] == 0)
        n_dec = int(valid.sum())
        tr = self.trajs
        return {
            "decisions": n_dec, "trajectories": len(tr), "episodes": self.n_episodes,
            "scenarios": len({t["scenario"] if t["scenario"] is not None else t["episode"] for t in tr}),
            "train_trajectories": len(self.split["train"]), "val_trajectories": len(self.split["val"]),
            "fire_steps": int(fire.sum()),
            "maneuver_switch_events": changes("maneuver"), "vertical_switch_events": changes("vertical"),
            "maneuver_ref_switch_events": changes("maneuver_ref"),
            "chaff_drop_steps": int(((act[:, IDX["chaff"]] > 0) & valid).sum()),
            "free_look_entries": int((enter_free & valid).sum()),
            "env_events": dict(self.events),
            "mean_trajectory_return": sum(t["ret"] for t in tr) / max(1, len(tr)),
            "mean_trajectory_length": n_dec / max(1, len(tr)),
        }


# ---------------------------------------------------------------------------
# loss / metrics
# ---------------------------------------------------------------------------

def head_weights(act, w_fire, w_switch):
    """[M,15] class weights from the demonstrated actions."""
    w = torch.ones(act.shape, dtype=torch.float32)
    w[:, IDX["weapon"]] = torch.where(act[:, IDX["weapon"]] == 1, torch.full_like(w[:, 0], w_fire), w[:, 0])
    w[:, IDX["maneuver"]] = torch.where(act[:, IDX["maneuver"]] != spec.MANEUVER_KEEP,
                                        torch.full_like(w[:, 0], w_switch), w[:, 0])
    return w


def bc_forward(actor: Actor, batch, burn, cfg_bc, device, keep_dists=False, grad=True):
    B = batch.shape[0]
    h0 = torch.zeros(B, H, device=device)
    out, seg = actor_window(actor, batch, h0, burn, batch.act[:, burn:], keep_dists=keep_dists, grad=grad)
    lm = seg.valid.reshape(-1)
    active = lm.unsqueeze(-1) & (out.k > 1) & out.legal                # [M,15]
    w = head_weights(seg.act.reshape(-1, spec.N_HEADS), cfg_bc.w_fire, cfg_bc.w_switch).to(device)
    return out, seg, lm, active, w


def bc_loss(out, active, w):
    """Sum over heads of the weight-normalised masked cross-entropy. Returns (loss, per-head tensors)."""
    nll = -out.logp
    wa = w * active.to(w.dtype)
    num = (nll * wa).sum(0)
    den = wa.sum(0)
    per_head = num / den.clamp(min=1e-8)
    return per_head.sum(), num, den


def train_bc(cfg, actor: Actor, ds: BCDataset, device, log=print, state_path=None, resume=True):
    """Returns the report dict; actor holds the best validation weights on return."""
    bc = cfg.bc
    opt = torch.optim.Adam(actor.parameters(), lr=bc.lr)
    rng = random.Random(bc.seed)
    train_seg = ds.segments("train", bc.seg_len)
    val_seg = ds.segments("val", bc.seg_len)
    best = {"val": float("inf"), "epoch": -1, "state": None}
    history = []
    bad = 0
    start_epoch = 0
    if state_path and resume and os.path.exists(state_path):
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        actor.load_state_dict(st["actor"])
        opt.load_state_dict(st["opt"])
        best, history, bad, start_epoch = st["best"], st["history"], st["bad"], st["epoch"]
        rng.setstate(st["rng"])
        log("bc: resumed after epoch %d (best val %.4f)" % (start_epoch, best["val"]))
    t0 = time.time()
    for epoch in range(start_epoch, bc.max_epochs):
        if bad >= bc.patience:
            break
        actor.train()
        rng.shuffle(train_seg)
        tot, nb = 0.0, 0
        for a in range(0, len(train_seg), bc.batch):
            if bc.max_batches_per_epoch and nb >= bc.max_batches_per_epoch:
                break
            widx = ds.window_index(train_seg[a:a + bc.batch], bc.burn_in, bc.seg_len)
            batch = ds.store.gather(widx).to(device)
            out, seg, lm, active, w = bc_forward(actor, batch, bc.burn_in, bc, device)
            loss, _, _ = bc_loss(out, active, w)
            if not bool(active.any()):
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), bc.grad_clip)
            opt.step()
            tot += loss.item()
            nb += 1
        val = evaluate_bc(cfg, actor, ds, val_seg, device) if val_seg else None
        rec = {"epoch": epoch + 1, "train_loss": tot / max(nb, 1), "batches": nb, "seconds": time.time() - t0}
        if val is not None:
            rec["val"] = val
            vl = val["loss"]
            if vl < best["val"] - 1e-6:
                best = {"val": vl, "epoch": epoch + 1, "state": {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()}}
                bad = 0
            else:
                bad += 1
        else:                                  # no validation data: keep the last weights
            best = {"val": rec["train_loss"], "epoch": epoch + 1, "state": {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()}}
        history.append(rec)
        log("bc epoch %d  train %.4f  val %s  bad %d" % (epoch + 1, rec["train_loss"], "%.4f" % val["loss"] if val else "-", bad))
        if state_path:
            tmp = state_path + ".tmp"
            torch.save({"actor": actor.state_dict(), "opt": opt.state_dict(), "best": best, "history": history,
                        "bad": bad, "epoch": epoch + 1, "rng": rng.getstate()}, tmp)
            os.replace(tmp, state_path)
    if best["state"] is not None:
        actor.load_state_dict(best["state"])
    final = evaluate_bc(cfg, actor, ds, val_seg, device) if val_seg else None
    return {"best_epoch": best["epoch"], "best_val_loss": best["val"], "history": history, "final_val": final,
            "train_segments": len(train_seg), "val_segments": len(val_seg)}


@torch.no_grad()
def evaluate_bc(cfg, actor: Actor, ds: BCDataset, segs, device, batch_size=32):
    """Per-head validation loss, accuracy and fire precision / recall (greedy among legal options)."""
    bc = cfg.bc
    actor.eval()
    num = torch.zeros(spec.N_HEADS); den = torch.zeros(spec.N_HEADS)
    nll_u = torch.zeros(spec.N_HEADS); cnt = torch.zeros(spec.N_HEADS); correct = torch.zeros(spec.N_HEADS)
    tp = fp = fn = tn = 0
    illegal = torch.zeros(spec.N_HEADS)
    for a in range(0, len(segs), batch_size):
        widx = ds.window_index(segs[a:a + batch_size], bc.burn_in, bc.seg_len)
        batch = ds.store.gather(widx).to(device)
        out, seg, lm, active, w = bc_forward(actor, batch, bc.burn_in, bc, device, keep_dists=True, grad=False)
        _, nm, dn = bc_loss(out, active, w)
        num += nm.cpu(); den += dn.cpu()
        act_f = active.to(torch.float32)
        nll_u += (-out.logp * act_f).sum(0).cpu(); cnt += act_f.sum(0).cpu()
        illegal += ((~out.legal) & lm.unsqueeze(-1)).sum(0).cpu().to(torch.float32)
        for h, name in enumerate(spec.HEAD_NAMES):
            pred = out.logp_all[name].argmax(-1)
            correct[h] += float(((pred == out.actions[:, h]) & active[:, h]).sum())
            if name == "weapon":
                m = active[:, h]
                pf, tf = (pred == 1) & m, (out.actions[:, h] == 1) & m
                tp += int((pf & tf).sum()); fp += int((pf & ~tf).sum()); fn += int((~pf & tf).sum()); tn += int((~pf & ~tf & m).sum())
    per_head = (num / den.clamp(min=1e-8))
    names = spec.HEAD_NAMES
    return {
        "loss": float(per_head.sum()),
        "head_loss_weighted": {names[i]: float(per_head[i]) for i in range(spec.N_HEADS) if den[i] > 0},
        "head_loss": {names[i]: float(nll_u[i] / cnt[i]) for i in range(spec.N_HEADS) if cnt[i] > 0},
        "head_acc": {names[i]: float(correct[i] / cnt[i]) for i in range(spec.N_HEADS) if cnt[i] > 0},
        "head_samples": {names[i]: int(cnt[i]) for i in range(spec.N_HEADS)},
        "fire": {"precision": tp / (tp + fp) if tp + fp else float("nan"), "recall": tp / (tp + fn) if tp + fn else float("nan"),
                 "tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "illegal_labels": {names[i]: int(illegal[i]) for i in range(spec.N_HEADS) if illegal[i] > 0},
    }


# ---------------------------------------------------------------------------
# data collection to shards (resumable)
# ---------------------------------------------------------------------------

class StopCollect(Exception):
    pass


def collect_to_shards(pool, cfg, out_dir, seed, log=print, stop=None):
    """Collect scripted episodes until cfg.bc.collect_decisions decisions exist on disk."""
    os.makedirs(out_dir, exist_ok=True)
    meta_path = os.path.join(out_dir, "meta.json")
    meta = {"decisions": 0, "shards": 0, "illegal": {}, "forced": {}}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    need = cfg.bc.collect_decisions - meta["decisions"]
    if need <= 0:
        log("bc data: %d decisions already collected" % meta["decisions"])
        return meta
    pending: List[dict] = []
    pend_dec = [0]

    def flush():
        if not pending:
            return
        sh = build_shard(pending)
        n = int(sh["own"].shape[0])
        path = os.path.join(out_dir, "shard_%05d.pt" % meta["shards"])
        save_shard(path, sh)
        meta["shards"] += 1
        meta["decisions"] += n
        with open(meta_path + ".tmp", "w") as f:
            json.dump(meta, f)
        os.replace(meta_path + ".tmp", meta_path)
        log("bc data: shard %d, %d decisions (total %d)" % (meta["shards"], n, meta["decisions"]))
        pending.clear()
        pend_dec[0] = 0

    def on_episodes(eps):
        for ep in eps:
            pending.append(ep)
            pend_dec[0] += sum(len(a["steps"]) for a in ep["agents"].values())
        if pend_dec[0] >= cfg.bc.shard_decisions:
            flush()
        if stop is not None and stop():
            flush()
            raise StopCollect()

    # collect in slices so that a job-time-limit kill loses at most one shard
    try:
        _, illegal, forced = pool.collect_bc(need, cfg.bc.episodes_per_request, seed, on_episodes=on_episodes)
    except StopCollect:
        log("bc data: stopped on request after %d decisions" % meta["decisions"])
        return meta
    flush()
    for k, v in illegal.items():
        meta["illegal"][k] = meta["illegal"].get(k, 0) + v
    for k, v in forced.items():
        meta["forced"][k] = meta["forced"].get(k, 0) + v
    with open(meta_path + ".tmp", "w") as f:
        json.dump(meta, f)
    os.replace(meta_path + ".tmp", meta_path)
    return meta


def load_shards(out_dir) -> List[dict]:
    names = sorted(n for n in os.listdir(out_dir) if n.startswith("shard_") and n.endswith(".pt"))
    return [torch.load(os.path.join(out_dir, n), map_location="cpu", weights_only=False) for n in names]
