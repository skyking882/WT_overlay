"""Training CLI.

    python -m rl.train run         --config smoke|workstation|gpu_cluster|file.json [--run-dir D] [--set a.b=v ...]
    python -m rl.train collect-bc  ...   scripted episodes -> <run_dir>/bc_data shards
    python -m rl.train bc          ...   behaviour cloning -> <run_dir>/bc_actor.pt, bc_report.json
    python -m rl.train ppo         ...   recurrent PPO from the BC actor (resumes from <run_dir>/ppo)
    python -m rl.train show-config ...

`run` executes the three stages in order and skips what is already finished, so after a
job-time-limit kill the same command continues. Exit code 0 = finished, 75 = stopped early
(wall limit / SIGTERM / SIGUSR1) after checkpointing: run it again to continue.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from rl import config as C
from rl import runtime as R
from rl import spec

T0 = time.time()


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------

def _pool(cfg, run_dir):
    from rl.pool import WorkerPool
    w = cfg.workers
    pool = WorkerPool(w, cfg.env, run_dir if cfg.run.log_workers else None).start()
    info = pool.describe()
    kinds = {}
    for i in info:
        k = "%s %s" % (i.get("impl"), i.get("version", ""))
        kinds[k] = kinds.get(k, 0) + 1
    R.log("workers: %s" % kinds)
    return pool


def stage_collect_bc(cfg, run_dir, stop):
    from rl import bc
    R.init_torch(cfg)
    out_dir = os.path.join(run_dir, "bc_data")
    pool = _pool(cfg, run_dir)
    try:
        pool.init_envs(0, cfg.env.seed)
        return bc.collect_to_shards(pool, cfg, out_dir, cfg.env.seed + 5, log=R.log, stop=stop.is_set)
    finally:
        pool.close()


def stage_bc(cfg, run_dir, stop):
    import torch
    from rl import bc
    from rl.model import Actor
    device = R.init_torch(cfg)
    R.set_threads(cfg.resources.threads_update)
    out_path = os.path.join(run_dir, "bc_actor.pt")
    if os.path.exists(out_path):
        R.log("bc: %s exists, skipping" % out_path)
        return
    shards = bc.load_shards(os.path.join(run_dir, "bc_data"))
    if not shards:
        raise SystemExit("no BC data in %s/bc_data; run collect-bc first" % run_dir)
    ds = bc.BCDataset(shards, cfg.bc.val_frac, cfg.bc.seed)
    del shards
    stats = ds.stats()
    R.log("bc dataset: %s" % json.dumps({k: v for k, v in stats.items() if k != "env_events"}))
    torch.manual_seed(cfg.run.seed)
    actor = Actor().to(device)
    report = bc.train_bc(cfg, actor, ds, device, log=R.log, state_path=os.path.join(run_dir, "bc_state.pt"))
    report["dataset"] = stats
    # let the cloned policy fly by itself and compare with the script (reward / events per decision)
    pool = _pool(cfg, run_dir)
    try:
        report["flying"] = fly_eval(cfg, actor, device, pool, ds, stats)
    finally:
        pool.close()
    R.atomic_torch_save({"actor": actor.state_dict(), "report": report}, out_path)
    with open(os.path.join(run_dir, "bc_report.json"), "w") as f:
        json.dump(report, f, indent=1, default=R._json_default)
    R.log("bc done: best epoch %s val loss %.4f; flying %s" % (report["best_epoch"], report["best_val_loss"],
                                                              json.dumps(report["flying"])))


def fly_eval(cfg, actor, device, pool, ds, stats):
    from rl.rollout import Sampler
    sampler = Sampler(cfg, pool, device, seed=cfg.run.seed + 99)
    sampler.start(cfg.env.seed + 424242)
    rew = dec = 0
    ev = {}
    eps = []
    R.set_threads(cfg.resources.threads_infer)
    for _ in range(cfg.bc.eval_rounds):
        buf, st = sampler.collect(actor)
        v = buf.loss_view(buf.store.valid)
        rew += float(buf.loss_view(buf.reward)[v].sum())
        dec += int(v.sum())
        for k, x in st["events"].items():
            ev[k] = ev.get(k, 0) + x
        eps += st["episodes"]
    n_dec = max(stats["decisions"], 1)
    return {
        "policy": {"decisions": dec, "reward_per_decision": rew / max(dec, 1),
                   "events_per_100_decisions": {k: 100.0 * v / max(dec, 1) for k, v in ev.items()}},
        "script": {"decisions": stats["decisions"],
                   "reward_per_decision": stats["mean_trajectory_return"] * stats["trajectories"] / n_dec,
                   "events_per_100_decisions": {k: 100.0 * v / n_dec for k, v in stats["env_events"].items()}},
    }


def stage_ppo(cfg, run_dir, stop):
    import copy
    import torch
    from rl.model import Actor, Critic
    from rl.ppo import PPOTrainer
    from rl.rollout import RoundAborted, Sampler
    device = R.init_torch(cfg)
    torch.manual_seed(cfg.run.seed)
    ckpt_dir = os.path.join(run_dir, "ppo")
    os.makedirs(ckpt_dir, exist_ok=True)
    metrics = R.MetricsLog(os.path.join(run_dir, "metrics.jsonl"))
    actor, critic = Actor().to(device), Critic().to(device)
    ref = None
    starts = 0
    path = R.latest_checkpoint(ckpt_dir)
    if path:
        ck = torch.load(path, map_location=device, weights_only=False)
        actor.load_state_dict(ck["actor"])
        critic.load_state_dict(ck["critic"])
        if ck.get("ref") is not None:
            ref = Actor().to(device)
            ref.load_state_dict(ck["ref"])
        starts = ck.get("starts", 0) + 1
        R.log("ppo: resumed from %s (round %d, %d decisions)" % (path, ck["trainer"]["round"], ck["trainer"]["decisions"]))
    else:
        bc_path = os.path.join(run_dir, "bc_actor.pt")
        if os.path.exists(bc_path):
            actor.load_state_dict(torch.load(bc_path, map_location=device, weights_only=False)["actor"])
            ref = copy.deepcopy(actor)
            R.log("ppo: starting from the BC actor")
        else:
            R.log("ppo: WARNING no bc_actor.pt, starting from a random actor without reference policy")
    if ref is not None:
        ref.eval()
        for p in ref.parameters():
            p.requires_grad_(False)
    trainer = PPOTrainer(cfg, actor, critic, ref, device)
    if path:
        trainer.load_state_dict(ck["trainer"])
    metrics.truncate_after(trainer.round)
    pool = _pool(cfg, run_dir)
    sampler = Sampler(cfg, pool, device, seed=cfg.run.seed + 1000 * starts)
    round_times = []
    code = R.EXIT_DONE

    def save(tag=""):
        payload = {"actor": actor.state_dict(), "critic": critic.state_dict(),
                   "ref": None if ref is None else ref.state_dict(), "trainer": trainer.state_dict(),
                   "cfg": cfg.to_dict(), "starts": starts, "aircraft": sampler.aircraft_names}
        p = R.checkpoint_path(ckpt_dir, trainer.round)
        R.atomic_torch_save(payload, p)
        R.prune_checkpoints(ckpt_dir, cfg.run.keep_ckpts)
        R.log("checkpoint %s%s" % (os.path.basename(p), tag))

    try:
        sampler.start(cfg.env.seed + 1_000_003 * starts)
        while trainer.round < cfg.run.rounds:
            if stop.is_set():
                R.log("stop requested (%s)" % stop.reason)
                code = R.EXIT_RESUME_ME
                break
            if cfg.run.max_wall_s > 0:
                left = cfg.run.max_wall_s - (time.time() - T0)
                if left < cfg.run.wall_margin_s + 1.2 * max(round_times or [0.0]):
                    R.log("wall budget nearly used (%.0f s left): checkpoint and stop" % left)
                    code = R.EXIT_RESUME_ME
                    break
            t_round = time.time()
            R.set_threads(cfg.resources.threads_infer)
            t0 = time.time()
            try:
                buf, st = sampler.collect(actor, abort=stop.is_set)
            except RoundAborted:
                R.log("stop requested (%s) during sampling: dropping the partial round" % stop.reason)
                code = R.EXIT_RESUME_ME
                break
            t_collect = time.time() - t0
            R.set_threads(cfg.resources.threads_update)
            t0 = time.time()
            sampler.finish(buf, critic, ref, cfg.ppo.gamma_base, cfg.ppo.lambda_base)
            t_post = time.time() - t0
            m = trainer.update(buf)
            rec = round_record(cfg, trainer, buf, st, m, sampler)
            rec["time"] = {"sample": t_collect, "inference": st["t_infer"], "env_wait": st["t_env"],
                           "worker_max_sum": st["t_worker"], "postpass": t_post, "update": m.get("t_update", 0.0),
                           "round": time.time() - t_round}
            metrics.write(rec)
            round_times.append(time.time() - t_round)
            R.log(summary_line(rec))
            if trainer.round % cfg.run.ckpt_every == 0 or trainer.round >= cfg.run.rounds:
                save()
        if code != R.EXIT_DONE:
            save(" (early stop)")
    finally:
        pool.close()
    return code


def round_record(cfg, trainer, buf, st, m, sampler):
    import torch
    S, T = buf.S, buf.T
    v = buf.loss_view(buf.store.valid)
    n_valid = int(v.sum())
    rew = buf.loss_view(buf.reward)
    air = buf.loss_view(buf.aircraft)
    per_air = {}
    for i, name in enumerate(sampler.aircraft_names):
        mk = v & (air == i)
        n = int(mk.sum())
        if n:
            per_air[name] = {"decisions": n, "reward_per_decision": float(rew[mk].sum()) / n}
    eps = st["episodes"]
    ep_by_air = {}
    kinds = {}
    ret_by_kind = {}
    for e in eps:
        a, ret, ln, kind = e[:4]
        d = ep_by_air.setdefault(a, [0, 0.0])
        d[0] += 1
        d[1] += ret
        kinds[kind] = kinds.get(kind, 0) + 1
        r = ret_by_kind.setdefault(e[4] if len(e) > 4 else spec.KIND_SCRIPT, [0, 0.0])
        r[0] += 1
        r[1] += ret
    for name, (n, tot) in ep_by_air.items():
        per_air.setdefault(name, {})["episodes"] = n
        per_air[name]["episode_return_mean"] = tot / n
    outcomes = st.get("outcomes", [])
    by_kind = outcomes_by_kind(outcomes, st.get("decisions_by_kind"), eps)
    # The long-standing keys describe the policy against scripts: self-play results are ~50/50 by construction and
    # would hide them, so they leave these (they stay in outcomes_by_kind). Without any vs-script episode: all.
    base = [o for o in outcomes if _kind_of(o) != spec.KIND_SELF] or outcomes
    rec = {
        "round": trainer.round, "decisions_total": trainer.decisions, "decisions_in_round": n_valid,
        "valid_fraction": n_valid / float(S * T),
        "reward_per_decision": float(rew[v].sum()) / max(n_valid, 1),
        "episodes_finished": len(eps), "episode_kinds": kinds,
        "episode_ends_by_kind": episode_ends_by_kind(eps),
        # Mean over every finished agent-episode of the round, vs-script and self-play mixed.
        "episode_return_mean": (sum(e[1] for e in eps) / len(eps)) if eps else None,
        # Self-play returns are about -0.5 by construction (+1 / -2 between two copies), so mixed returns mislead.
        "episode_return_by_kind": {k: tot / n for k, (n, tot) in ret_by_kind.items()},
        "late_rewards": {"credited": st.get("late_credited", 0), "dropped": st.get("late_dropped", 0)},
        "events": st["events"], "lost_agents": st["lost"], "rollout_mask_fallbacks": st["mask_fallbacks"],
        "outcomes": outcome_counts(base),
        "outcomes_by_kind": by_kind,
        "per_aircraft": per_air,
    }
    extra = sampler_extras(st)
    if extra:
        rec["sampler_stats"] = extra
    rec.update(outcome_rates(rec["outcomes"]))
    for o in base:
        pa = per_air.setdefault(o[0], {})
        pa[o[1]] = pa.get(o[1], 0) + 1
    rec.update(m)
    return rec


# Sampler stats keys round_record turns into keys of its own; anything else a sampler reports goes to sampler_stats.
ST_KNOWN = frozenset(("t_infer", "t_env", "t_worker", "episodes", "events", "lost", "mask_fallbacks",
                      "decisions_by_kind", "outcomes", "late_credited", "late_dropped", "n_valid"))


def _is_num(x):
    return isinstance(x, (int, float))


def sampler_extras(st):
    """Sampler statistics without a record key of their own (e.g. added by a newer rollout): numbers and flat dicts
    of numbers pass through unchanged, so they are kept in metrics.jsonl and the dashboard can list them."""
    return {k: v for k, v in st.items() if k not in ST_KNOWN and
            (_is_num(v) or (isinstance(v, dict) and all(_is_num(x) for x in v.values())))}


def _kind_of(outcome):
    """Episode kind of an outcome tuple (air, outcome, kills, deaths[, kind]); older 4-tuples are vs-script."""
    return outcome[4] if len(outcome) > 4 else spec.KIND_SCRIPT


def episode_ends_by_kind(episodes):
    """{episode kind: {end kind: agent-episodes}} from the sampler's (aircraft, return, length, end, kind) tuples.
    End kinds: terminal, timeout (also a time limit made terminal by timeout_reward), lost."""
    out = {}
    for e in episodes:
        d = out.setdefault(_kind_of(e), {})
        d[e[3]] = d.get(e[3], 0) + 1
    return out


def _paired(outcomes, episodes):
    """The sampler appends an agent-episode's outcome and its episode tuple together, so the i-th of each describe
    the same agent-episode. None when the lists do not line up (then no outcome is split by how the episode ended)."""
    if episodes is None or len(episodes) != len(outcomes):
        return None
    pairs = list(zip(outcomes, episodes))
    if any(o[0] != e[0] or _kind_of(o) != _kind_of(e) for o, e in pairs):
        return None
    return pairs


def outcome_counts(outcomes):
    """Per-episode results of policy agents: win (killed, survived), loss (died, no kill), trade, none."""
    c = {"win": 0, "loss": 0, "trade": 0, "none": 0, "kills": 0, "deaths": 0}
    for o in outcomes:
        c[o[1]] += 1
        c["kills"] += o[2]
        c["deaths"] += o[3]
    return c


def outcome_rates(oc):
    """win_rate (wins per finished agent-episode) and exchange (kills per death); None when undefined."""
    n_out = sum(oc[k] for k in ("win", "loss", "trade", "none"))
    return {"win_rate": oc["win"] / n_out if n_out else None,
            "exchange": oc["kills"] / oc["deaths"] if oc["deaths"] else None}


def outcomes_by_kind(outcomes, decisions=None, episodes=None):
    """{episode kind: outcome counts + episodes, win_rate, exchange, decisions} for every kind that finished an
    episode or acted this round. Counts are per agent-episode, so a self-play episode contributes two outcomes
    (its win and its loss), and its win_rate is ~0.5 by construction.

    With the sampler's ``episodes`` (aircraft, return, length, end, kind) also: trade_rate (trades per agent-episode),
    timeouts / timeout_rate (agent-episodes that ran into the time limit, whatever their outcome), and the "none"
    outcomes (survived without a kill) split into none_timeout (time ran out) and none_other (e.g. the opponent
    crashed); the split is None when outcomes and episodes do not line up."""
    groups = {}
    for o in outcomes:
        groups.setdefault(_kind_of(o), []).append(o)
    for kind in decisions or {}:
        groups.setdefault(kind, [])
    pairs = _paired(outcomes, episodes)
    out = {}
    for kind, rows in groups.items():
        c = outcome_counts(rows)
        c["episodes"] = len(rows)
        c.update(outcome_rates(c))
        c["decisions"] = int((decisions or {}).get(kind, 0))
        if episodes is not None:
            n = len(rows)
            ends = [e[3] for e in episodes if _kind_of(e) == kind]
            c["trade_rate"] = c["trade"] / n if n else None
            c["timeouts"] = sum(1 for x in ends if x == "timeout")
            c["timeout_rate"] = c["timeouts"] / len(ends) if ends else None
            if pairs is None:
                c["none_timeout"] = c["none_other"] = None
            else:
                nones = [e[3] for o, e in pairs if _kind_of(o) == kind and o[1] == "none"]
                c["none_timeout"] = sum(1 for x in nones if x == "timeout")
                c["none_other"] = len(nones) - c["none_timeout"]
        out[kind] = c
    return out


def summary_line(r):
    t = r["time"]
    line = ("round %d  dec %d  valid %.2f  rew/dec %.4f  ent %.3f  klT %.4f  klBC %.4f  clip %.3f  vloss %.3f  "
            "vrms %.3f  EV %.2f  | sample %.1fs (inf %.1f env %.1f) post %.1fs upd %.1fs" % (
                r["round"], r["decisions_total"], r["valid_fraction"], r["reward_per_decision"],
                r.get("entropy", float("nan")), r.get("kl_target", float("nan")), r.get("kl_ref", float("nan")),
                r.get("clip_frac", float("nan")), r.get("value_loss", float("nan")),
                r.get("value_rmse", float("nan")), r.get("explained_variance", float("nan")), t["sample"],
                t["inference"], t["env_wait"], t["postpass"], t["update"]))
    kinds = r.get("outcomes_by_kind") or {}
    if spec.KIND_SELF in kinds:         # with self-play, the vs-script result must not be lost in the totals
        def f(x):
            return "-" if x is None else "%.2f" % x
        vs, sp = kinds.get(spec.KIND_SCRIPT) or {}, kinds[spec.KIND_SELF]
        ret = r.get("episode_return_by_kind") or {}
        line += "  | vs-script: eps %d win %s ex %s ret %s dec %d  self-play: eps %d ret %s dec %d" % (
            vs.get("episodes", 0), f(vs.get("win_rate")), f(vs.get("exchange")), f(ret.get(spec.KIND_SCRIPT)),
            vs.get("decisions", 0), sp["episodes"], f(ret.get(spec.KIND_SELF)), sp["decisions"])
    return line


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_config(args) -> C.Config:
    cfg = None
    if args.run_dir:
        saved = os.path.join(C.expand(args.run_dir), "config.json")
        if os.path.exists(saved) and not args.config_given:
            cfg = C.Config.load(saved)
    if cfg is None:
        if args.config.endswith(".json") or os.path.exists(args.config):
            cfg = C.Config.load(args.config)
        else:
            cfg = C.preset(args.config)
    if args.run_dir:
        cfg.run.run_dir = args.run_dir
    if args.rounds is not None:
        cfg.run.rounds = args.rounds
    C.apply_overrides(cfg, args.set or [])
    cfg.validate()
    return cfg


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "collect-bc", "bc", "ppo", "show-config"])
    ap.add_argument("--config", default=None, help="preset name (smoke|workstation|gpu_cluster) or JSON file; "
                                                    "default: <run-dir>/config.json if present, else smoke")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--rounds", type=int, default=None, help="total PPO rounds (run.rounds)")
    ap.add_argument("--set", action="append", help="override, e.g. --set ppo.clip=0.1 (repeatable)")
    args = ap.parse_args(argv)
    args.config_given = args.config is not None
    args.config = args.config or "smoke"
    cfg = load_config(args)
    if args.cmd == "show-config":
        print(json.dumps(cfg.to_dict(), indent=1, sort_keys=True))
        return 0
    run_dir = C.expand(cfg.run.run_dir)
    os.makedirs(run_dir, exist_ok=True)
    cores = R.early_setup(cfg)        # core cap + thread defaults BEFORE torch is imported
    cfg.save(os.path.join(run_dir, "config.json"))
    R.log("run dir %s | config %s | cores %s" % (run_dir, cfg.name, "uncapped" if cores is None else "%d allowed (%s..%s)" % (len(cores), cores[0], cores[-1])))
    stop = R.StopFlag()
    code = 0
    if args.cmd in ("run", "collect-bc"):
        stage_collect_bc(cfg, run_dir, stop)
    if args.cmd in ("run", "bc") and not stop.is_set():
        stage_bc(cfg, run_dir, stop)
    if args.cmd in ("run", "ppo") and not stop.is_set():
        code = stage_ppo(cfg, run_dir, stop)
    if stop.is_set() and code == 0:
        code = R.EXIT_RESUME_ME
    return code


if __name__ == "__main__":
    sys.exit(main())
