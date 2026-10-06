"""Benchmarks: actor inference latency and PPO update-step time.

    python -m rl.bench [--threads 1 4] [--entities 20 64] [--batch 1 64] [--iters 200]
                       [--device cpu|cuda] [--update-iters 5] [--skip-update]

Inference = one decision for B streams: encode (entity MLP + Transformer + pooling + own MLP
+ fusion) + GRU step + the 15 sequential heads incl. masked sampling, on tensors already
prepared (`forward`); `prep+forward` adds decoding the wire bytes into padded tensors.
Update = one PPO minibatch: 16 segments x (16 burn-in + 80) steps, actor forward / backward /
clip / Adam step plus critic forward / backward / step.
"""
from __future__ import annotations

import argparse
import json
import random
import time

import torch

from rl import spec, wire
from rl.encode import Batch, Decoded, masks_from_flat
from rl.model import Actor, Critic, count_params


def synthetic_wire(n, rng, m=48):
    ents = [[rng.uniform(-1, 1) for _ in range(spec.ENT_DIM)] for _ in range(n)]
    masks = spec.all_true_masks(n)
    obs = {"own": [rng.uniform(-1, 1) for _ in range(spec.OWN_DIM)], "entities": ents,
           "prev_intent": [0.0] * spec.INTENT_DIM, "masks": masks,
           "truth": [[rng.uniform(-1, 1) for _ in range(spec.TRUTH_DIM)] for _ in range(m)],
           "aircraft": "x", "dt": spec.DT_STEP}
    return wire.pack_obs(obs)


def stats(ts):
    ts = sorted(ts)
    return {"mean_ms": 1000 * sum(ts) / len(ts), "p95_ms": 1000 * ts[int(0.95 * (len(ts) - 1))],
            "min_ms": 1000 * ts[0]}


def bench_inference(actor, n, bsz, iters, device, rng):
    wires = [synthetic_wire(n, rng) for _ in range(bsz)]
    gen = torch.Generator(device=device)
    gen.manual_seed(0)
    h = torch.zeros(bsz, 256, device=device)
    prep, fwd, both = [], [], []
    for i in range(iters + 20):
        t0 = time.perf_counter()
        dec = Decoded(wires)
        b = dec.to_batch(first=torch.zeros(bsz, dtype=torch.bool)).to(device)
        t1 = time.perf_counter()
        out, h_new = actor.act(b, h, gen)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        if i >= 20:
            prep.append(t1 - t0)
            fwd.append(t2 - t1)
            both.append(t2 - t0)
    return {"forward": stats(fwd), "prep": stats(prep), "prep+forward": stats(both)}


def bench_update(actor, critic, n, device, iters, rng, segs=16, burn=16, seg_len=80):
    B, T = segs, burn + seg_len
    g = torch.Generator()
    g.manual_seed(1)
    ent = torch.randn(B, T, n, spec.ENT_DIM, generator=g)
    ent_n = torch.full((B, T), n, dtype=torch.long)
    tr = torch.randn(B, T, 48, spec.TRUTH_DIM, generator=g)
    masks = masks_from_flat(torch.ones(B, T, spec.MASK_BYTES, dtype=torch.bool), n + 1)
    batch = Batch(torch.randn(B, T, spec.OWN_DIM, generator=g), ent, ent_n, torch.zeros(B, T, spec.INTENT_DIM), tr,
                  torch.full((B, T), 48, dtype=torch.long), masks,
                  torch.ones(B, T, dtype=torch.bool), torch.zeros(B, T, dtype=torch.bool), torch.full((B, T), spec.DT_STEP))
    batch = batch.to(device)
    opt_a = torch.optim.Adam(actor.parameters(), lr=1e-4, eps=1e-5)
    opt_c = torch.optim.Adam(critic.parameters(), lr=3e-4, eps=1e-5)
    from rl.ppo import actor_window, critic_window
    h0 = torch.zeros(B, 256, device=device)
    with torch.no_grad():
        x, e = actor.encode(batch)
        hs, _ = actor.unroll(x, h0, batch.first, batch.valid)
        act = actor.heads_from_batch(batch, hs, e, None, gen=torch.Generator(device=device).manual_seed(0)).actions.view(B, T, -1)
    times, t_a, t_c = [], [], []
    for i in range(iters + 2):
        t0 = time.perf_counter()
        out, seg = actor_window(actor, batch, h0, burn, act[:, burn:], keep_dists=True)
        loss = -out.logp.sum(-1).mean() - 0.01 * out.ent.mean()
        opt_a.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
        opt_a.step()
        t1 = time.perf_counter()
        v = critic_window(critic, batch, h0, burn)
        vl = (v ** 2).mean()
        opt_c.zero_grad()
        vl.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
        opt_c.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        if i >= 2:
            t_a.append(t1 - t0)
            t_c.append(t2 - t1)
            times.append(t2 - t0)
    return {"minibatch": stats(times), "actor_part": stats(t_a), "critic_part": stats(t_c)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--entities", type=int, nargs="+", default=[20, 64])
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 64])
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--update-iters", type=int, default=5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--skip-update", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    device = torch.device(a.device)
    torch.manual_seed(0)
    actor, critic = Actor().to(device), Critic().to(device)
    actor.eval()
    rng = random.Random(0)
    res = {"params": {"actor": count_params(actor), "critic": count_params(critic)}, "device": a.device,
           "torch": torch.__version__, "inference": [], "update": []}
    for th in a.threads:
        torch.set_num_threads(th)
        for n in a.entities:
            for bsz in a.batch:
                r = bench_inference(actor, n, bsz, a.iters, device, rng)
                res["inference"].append({"threads": th, "entities": n, "batch": bsz, **r})
                print("inference threads=%d entities=%d batch=%d  forward mean %.2f ms p95 %.2f ms | prep %.2f ms | prep+forward mean %.2f ms p95 %.2f ms" % (
                    th, n, bsz, r["forward"]["mean_ms"], r["forward"]["p95_ms"], r["prep"]["mean_ms"],
                    r["prep+forward"]["mean_ms"], r["prep+forward"]["p95_ms"]), flush=True)
    if not a.skip_update:
        actor.train()
        for th in a.threads:
            torch.set_num_threads(th)
            for n in a.entities:
                r = bench_update(actor, critic, n, device, a.update_iters, rng)
                res["update"].append({"threads": th, "entities": n, **r})
                print("update minibatch (16 seg x 16+80 steps) threads=%d entities=%d: %.0f ms (actor %.0f + critic %.0f)" % (
                    th, n, r["minibatch"]["mean_ms"], r["actor_part"]["mean_ms"], r["critic_part"]["mean_ms"]), flush=True)
    if a.json:
        print(json.dumps(res))
    return res


if __name__ == "__main__":
    main()
