"""Opponent pool of history episodes (env.config history_prob > 0): frozen copies of past actors.

* snapshots: a copy of the current actor every league.snapshot_every rounds, and one when the pool has none yet (the
  run's starting actor); the newest league.keep are kept. Files <run_dir>/league/snap_r<round>.pt hold {"actor": state
  dict, "round"}, so they also work as --opponent-checkpoint of rl.eval_replay. The member list goes into the PPO
  checkpoint (state_dict); files of dropped snapshots are removed after the next checkpoint (prune_files).
* references: fixed checkpoints (league.references: any file with an "actor" state dict, e.g. the best checkpoint of
  an earlier run), always in the pool; a copy is kept in <run_dir>/league/ for a resume after the original is gone.
Members are inference-only Actors (eval, no grad) on the learner's device, loaded once when they join the pool, never
per episode. A history episode draws its opponent uniformly (pick) and keeps that object until it ends, even if the
member leaves the pool meanwhile.
"""
from __future__ import annotations

import copy
import os
import random

import torch

from rl.model import Actor
from rl.runtime import atomic_torch_save


class League:
    def __init__(self, lcfg, device, run_dir=None, seed=0, log=print):
        self.c, self.device, self.log = lcfg, device, log
        self.dir = os.path.join(run_dir, "league") if run_dir else None
        self.members = []          # {"name", "kind": snapshot | reference, "round", "file", "actor"}
        self.rng = random.Random(seed)
        self.last_snapshot = None

    @staticmethod
    def _frozen(actor):
        actor.eval()
        for p in actor.parameters():
            p.requires_grad_(False)
            p.grad = None
        return actor

    def _load(self, path):
        a = Actor()
        a.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["actor"])
        return self._frozen(a.to(self.device))

    def add(self, name, actor, kind="snapshot", round_no=None, file=None):
        self.members.append(dict(name=name, kind=kind, round=round_no, file=file, actor=self._frozen(actor)))

    def snapshots(self):
        return [m for m in self.members if m["kind"] == "snapshot"]

    def names(self):
        return [m["name"] for m in self.members]

    def pick(self):
        """(name, actor) of a uniformly drawn member: the frozen opponent of one history episode."""
        if not self.members:
            raise RuntimeError("the league has no opponent (league.snapshot_every = 0 and no league.references?)")
        m = self.members[self.rng.randrange(len(self.members))]
        return m["name"], m["actor"]

    # ------------------------------------------------------------------ pool changes
    def snapshot(self, actor, round_no):
        """Copy the current actor into the pool (file in the run dir); the oldest snapshots beyond keep leave it."""
        name = "r%06d" % round_no
        a = self._frozen(copy.deepcopy(actor))
        file = None
        if self.dir:
            os.makedirs(self.dir, exist_ok=True)
            file = "snap_%s.pt" % name
            atomic_torch_save({"actor": a.state_dict(), "round": round_no}, os.path.join(self.dir, file))
        self.members = [m for m in self.members if m["name"] != name]
        self.add(name, a, "snapshot", round_no, file)
        drop = {id(m) for m in self.snapshots()[:-max(1, self.c.keep)]}
        self.members = [m for m in self.members if id(m) not in drop]
        self.last_snapshot = round_no
        return name

    def after_round(self, actor, round_no):
        """After the update of round `round_no`: a snapshot every snapshot_every rounds. Returns its name or None."""
        n = self.c.snapshot_every
        if n > 0 and round_no % n == 0 and round_no != self.last_snapshot:
            return self.snapshot(actor, round_no)
        return None

    def load_references(self, paths):
        for i, p in enumerate(paths or []):
            src = os.path.expanduser(p)
            cache = os.path.join(self.dir, "ref%d_%s" % (i, os.path.basename(src))) if self.dir else None
            if os.path.exists(src):
                a = self._load(src)
                if cache:
                    os.makedirs(self.dir, exist_ok=True)
                    atomic_torch_save({"actor": a.state_dict(), "source": src}, cache)
            elif cache and os.path.exists(cache):
                self.log("league: reference %s is gone, using the copy %s" % (src, cache))
                a = self._load(cache)
            else:
                raise FileNotFoundError("league.references: %s does not exist" % p)
            self.add("ref%d:%s" % (i, os.path.basename(src)), a, "reference", None,
                     os.path.basename(cache) if cache else None)

    # ------------------------------------------------------------------ persistence
    def state_dict(self):
        return {"snapshots": [{k: m[k] for k in ("name", "round", "file")} for m in self.snapshots()],
                "last_snapshot": self.last_snapshot}

    def load_state_dict(self, d):
        """Snapshots listed in a checkpoint (references come from the config: load_references)."""
        for s in (d or {}).get("snapshots", []):
            path = os.path.join(self.dir, s["file"]) if self.dir and s.get("file") else None
            if path is None or not os.path.exists(path):
                self.log("league: snapshot %s (%s) is missing, left out of the pool" % (s["name"], path))
                continue
            self.add(s["name"], self._load(path), "snapshot", s["round"], s["file"])
        self.last_snapshot = (d or {}).get("last_snapshot")

    def prune_files(self):
        """Remove snapshot files that left the pool. Call right after a checkpoint (which no longer lists them)."""
        if not self.dir or not os.path.isdir(self.dir):
            return
        keep = {m["file"] for m in self.snapshots()}
        for f in os.listdir(self.dir):
            if f.startswith("snap_") and f.endswith(".pt") and f not in keep:
                try:
                    os.remove(os.path.join(self.dir, f))
                except OSError:
                    pass
