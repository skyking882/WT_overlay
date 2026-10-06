"""FakeMatchEnv: a stand-in for wt_overlay/rl_env.py that obeys the section-13 contract.

Pure Python (random, math, copy only), so it runs under CPython and PyPy. It is not a
combat simulator. It is a small abstract game whose structure is rich enough to
exercise everything on the training side:

* variable entity counts (0..64, optionally above 64 to exercise truncation),
* the section-13.1 masks: flat lists, 3 x K tables by view_mode, 3 x 2 x K tables by view_mode and
  "maneuver_ref is none", (n+1) x K tables by the chosen target,
* hold-rule single-option masks for maneuver_ref / maneuver / vertical (2.0 s hold,
  emergency release on a new visible warning),
* a camera: flame / visual / target-box entities are only visible inside the view cone,
* random deaths (incoming missiles), kills, assists, timeouts and natural match ends,
* truth tokens (aircraft + missiles, 40 wide) for the critic,
* `scripted_actions()`, `snapshot()` / `restore()`, dt = 20/48 s.

The hidden structure that makes learning possible: an incoming missile is often announced
by a visible MAW contact; evading (defensive maneuver, dive, chaff) while it is present
sharply cuts the hit probability, and firing at a close radar track gives a kill chance.
"""
from __future__ import annotations

import copy
import math
import random

from rl import spec

ENT_TYPES = ("radar", "rwr", "maw", "flame", "visual", "box", "map", "mate", "own_missile", "inferred")
T_RADAR, T_RWR, T_MAW, T_FLAME, T_VISUAL, T_BOX, T_MAP, T_MATE, T_OWNMSL, T_INFER = range(10)
CAMERA_TYPES = (T_FLAME, T_VISUAL, T_BOX)
DEFENSIVE_MANEUVERS = (4, 5, 6, 10)      # left/right nine-line, turn-away, go home
NEEDS_REF_MANEUVERS = (1, 2, 3)          # aim / offset left / offset right need a reference
AIRCRAFT = ("f16", "j10c", "typhoon", "su30", "f15c", "mig35")

DEFAULTS = dict(
    n_agents=4,              # max number of policy-controlled agents per episode
    min_agents=1,
    max_steps=120,           # timeout length
    max_entities=64,         # wire cap; the env drops the rest and counts them
    min_entities=0,
    target_entities_max=40,  # upper end of the random entity-count walk
    overflow=False,          # if true the walk can exceed max_entities (tests truncation)
    hold_steps=5,            # 2.0 s at 0.4167 s
    p_incoming=0.04,         # per step per agent
    p_background_death=0.0005,
    p_match_end=0.0008,
    p_ignore=0.05,           # executor: intent not executed
    strict=True,             # raise on illegal actions
    camera_fov_deg=(90.0, 120.0),
    aircraft=AIRCRAFT[:4],
)


class IllegalAction(Exception):
    pass


class AgentObs:
    """Section-13 AgentObs; plain lists and floats."""
    __slots__ = ("own", "entities", "prev_intent", "masks", "truth", "aircraft", "dt")

    def __init__(self, own, entities, prev_intent, masks, truth, aircraft, dt):
        self.own = own
        self.entities = entities
        self.prev_intent = prev_intent
        self.masks = masks
        self.truth = truth
        self.aircraft = aircraft
        self.dt = dt


def _antenna_bin(elev):
    """Nearest of +20 / +10 / 0 / -10 / -20 degrees (option 0..4) for an elevation in rad."""
    deg = elev * 57.29578
    return min(range(5), key=lambda i: abs(deg - (20 - 10 * i)))


def _sin(x):
    return math.sin(x)


def _cos(x):
    return math.cos(x)


class FakeMatchEnv:
    def __init__(self, config=None, seed=0):
        cfg = dict(DEFAULTS)
        cfg.update(config or {})
        self.cfg = cfg
        self.seed = seed
        self.rng = random.Random(seed)
        self.script_rng = random.Random(seed * 7919 + 13)
        self.episode_count = 0
        self.scenario = None
        self.S = None
        self._cache = {}

    # ------------------------------------------------------------------ helpers
    def _uid(self):
        self.S["next_uid"] += 1
        return self.S["next_uid"]

    def _new_contact(self, typ=None):
        rng = self.rng
        if typ is None:
            typ = rng.choices(
                (T_RADAR, T_RWR, T_FLAME, T_VISUAL, T_BOX, T_MAP, T_MATE),
                weights=(8, 3, 1, 1, 1, 2, 2))[0]
        c = {
            "uid": self._uid(), "type": typ,
            "rng": rng.uniform(5.0, 120.0), "brg": rng.uniform(-math.pi, math.pi),
            "elev": rng.uniform(-0.3, 0.3), "spd": rng.uniform(150.0, 500.0),
            "age": 0.0, "extrap": 0, "lock": 0, "lock_left": 0,
            "perf": [rng.random(), rng.random(), rng.random()],
            "support": 0, "noise": [rng.uniform(-1, 1) * 0.3 for _ in range(24)],
        }
        return c

    def _target_count(self):
        S = self.S
        cap = self.cfg["target_entities_max"]
        if self.rng.random() < 0.15:
            S["target_n"] = max(self.cfg["min_entities"], min(cap, S["target_n"] + self.rng.choice((-6, -3, 3, 6))))
        return S["target_n"]

    # ------------------------------------------------------------------ reset
    def reset(self):
        cfg = self.cfg
        rng = self.rng
        self.episode_count += 1
        self.scenario = "ep%d" % self.episode_count
        k = rng.randint(max(1, cfg["min_agents"]), max(1, cfg["n_agents"]))
        fov = math.radians(rng.uniform(*cfg["camera_fov_deg"]))
        S = {
            "t": 0, "next_uid": 0, "agents": {}, "contacts": {}, "threats": [], "shots": [],
            "enemies": [], "fov": fov, "target_n": rng.randint(cfg["min_entities"], max(cfg["min_entities"], cfg["target_entities_max"] // 2)),
            "over": False,
        }
        self.S = S
        for i in range(k):
            aid = "p%d" % i
            S["agents"][aid] = {
                "alive": True, "aircraft": rng.choice(cfg["aircraft"]),
                "missiles": rng.randint(2, 6), "chaff": rng.randint(10, 40),
                "view_mode": 0, "view_obj": None, "look_az": 0, "look_el": 1, "free_steps": 0,
                "ref": None, "man": 0, "vert": 0, "hold": 0,
                "last": {h: (spec.default_option(h, 0) if h not in spec.POINTER_HEADS else 0) for h in spec.HEAD_NAMES},
                "since": {h: 0 for h in spec.HEAD_NAMES},
                "chaff_recent": 0, "stt": 0, "ant": 1, "warn_new": False, "maw_seen": set(),
                "ret": 0.0, "perf": [rng.random() for _ in range(3)],
            }
            S["contacts"][aid] = []
            n0 = rng.randint(cfg["min_entities"], max(cfg["min_entities"], S["target_n"]))
            for _ in range(n0):
                S["contacts"][aid].append(self._new_contact())
        for _ in range(6):
            S["enemies"].append({"pos": [rng.uniform(-1, 1) for _ in range(3)], "vel": [rng.uniform(-1, 1) for _ in range(3)], "alive": True})
        return self.observe()

    # ------------------------------------------------------------------ observation
    def _view_dir(self, a):
        if a["view_mode"] == 1 and a["view_obj"] is not None:
            for c in self._contacts_of(a):
                if c["uid"] == a["view_obj"]:
                    return c["brg"]
            return 0.0
        if a["view_mode"] == 2:
            return a["look_az"] * math.pi / 4.0
        return 0.0

    def _contacts_of(self, a):
        for aid, ag in self.S["agents"].items():
            if ag is a:
                return self.S["contacts"][aid]
        return []

    def _visible_contacts(self, aid):
        S = self.S
        a = S["agents"][aid]
        vd = self._view_dir(a)
        half = S["fov"] / 2.0
        out = []
        for c in S["contacts"][aid]:
            if c["type"] in CAMERA_TYPES:
                d = abs((c["brg"] - vd + math.pi) % (2 * math.pi) - math.pi)
                if d > half:
                    continue
            out.append(c)
        return out

    def _priority(self, c, a):
        if c["uid"] == a["ref"] or c["uid"] == a["view_obj"] or (c["type"] == T_OWNMSL and c["support"]):
            return 0
        if c["type"] in (T_MAW, T_INFER):
            return 1
        if c["lock"]:
            return 2
        if c["type"] == T_RADAR:
            return 3
        if c["type"] == T_RWR:
            return 4
        return 5

    def _entity_features(self, c):
        f = [0.0] * spec.ENT_DIM
        f[c["type"]] = 1.0
        f[10] = c["rng"] / 150.0
        f[11] = _sin(c["brg"]); f[12] = _cos(c["brg"])
        f[13] = _sin(c["elev"]); f[14] = _cos(c["elev"])
        f[15] = c["spd"] / 600.0
        f[16] = min(c["age"], 20.0) / 20.0
        f[17] = float(c["extrap"])
        f[18] = float(c["lock"])
        f[19], f[20], f[21] = c["perf"]
        f[22] = float(c["support"])
        f[23] = 1.0
        f[24:48] = c["noise"]
        return f

    def _build_masks(self, aid, ents):
        """Section 13.1 masks: unconditional lists, 3 x K tables by view_mode, 3 x 2 x K by
        view_mode and "maneuver_ref is none", (n+1) x K by the chosen target. The env precomputes the
        view-mode gating itself: in a row where a head is inactive only its inactive option is True."""
        a = self.S["agents"][aid]
        n = len(ents)
        K = n + 1
        idx = {c["uid"]: i for i, c in enumerate(ents)}
        types = [c["type"] for c in ents]

        def one(k, i):
            return [j == i for j in range(k)]

        # hold rule: while held (aiming, within 2.0 s of the last change) maneuver_ref / maneuver /
        # vertical have exactly one True, the current value
        held = a["hold"] > 0 and a["view_mode"] == 0
        ref_idx = None
        if held:
            ref_idx = n if a["ref"] is None else idx.get(a["ref"], None)
            if ref_idx is None:
                a["hold"] = 0                   # referenced entity vanished: release the hold
                held = False
        has_chaff = a["chaff"] > 0
        m = {}
        m["view_mode"] = [True] * 3
        m["speed"] = [True] * 3
        m["chaff"] = [True, has_chaff, has_chaff]
        m["antenna"] = [True] * 5
        m["target"] = [True] * K
        m["maneuver_ref"] = [one(K, ref_idx) if held else [True] * K, one(K, n), one(K, n)]
        m["view_object"] = [one(K, n), ([True] * n + [False]) if n > 0 else [True], one(K, n)]
        m["look_az"] = [one(8, 0), one(8, 0), [True] * 8]
        m["look_el"] = [one(3, 1), one(3, 1), [True] * 3]
        m["kb_roll"] = [one(3, 1), [True] * 3, [True] * 3]
        m["kb_pitch"] = [one(3, 1), [True] * 3, [True] * 3]

        def man_row(ref_null):
            if held:
                return one(11, a["man"])
            return [not (mv in NEEDS_REF_MANEUVERS and ref_null) for mv in range(11)]

        keep = one(11, 0)
        m["maneuver"] = [[man_row(False), man_row(True)], [keep, keep], [keep, keep]]
        vrow = one(5, a["vert"]) if held else [True] * 5
        vkeep = one(5, 0)
        m["vertical"] = [[vrow, vrow], [vkeep, vkeep], [vkeep, vkeep]]
        fire_ok = [a["missiles"] > 0 and types[j] == T_RADAR for j in range(n)] + [False]
        m["weapon"] = [[True, bool(fire_ok[j])] for j in range(K)]
        m["radar_mode"] = [[True, True, bool(j < n and types[j] == T_RADAR)] for j in range(K)]
        return m

    def _truth_tokens(self):
        S = self.S
        toks = []
        for i, (aid, a) in enumerate(sorted(S["agents"].items())):
            t = [0.0] * spec.TRUTH_DIM
            t[0] = 1.0                       # aircraft flag
            t[2] = 1.0 if a["alive"] else 0.0
            t[3] = 1.0                       # controlled
            t[4] = a["missiles"] / 6.0
            t[5] = a["chaff"] / 40.0
            t[6 + (i % 8)] = 1.0
            toks.append(t)
        for e in S["enemies"]:
            t = [0.0] * spec.TRUTH_DIM
            t[0] = 1.0; t[2] = 1.0 if e["alive"] else 0.0
            for j in range(3):
                t[14 + j] = e["pos"][j]; t[17 + j] = e["vel"][j]
            toks.append(t)
        ids = sorted(S["agents"])
        for th in S["threats"]:
            t = [0.0] * spec.TRUTH_DIM
            t[1] = 1.0                       # missile flag
            t[2] = 1.0
            t[20 + min(ids.index(th["aid"]), 7)] = 1.0     # target id
            t[28] = th["tti"] / 15.0
            t[29] = 1.0 if th["visible_at"] > 0 else 0.0   # will be announced at all
            t[30] = 1.0 if th["tti"] <= th["visible_at"] else 0.0
            toks.append(t)
        for sh in S["shots"]:
            t = [0.0] * spec.TRUTH_DIM
            t[1] = 1.0; t[2] = 1.0; t[3] = 1.0
            t[28] = sh["left"] / 16.0
            t[31] = sh["p_hit"]
            t[32] = 1.0
            toks.append(t)
        return toks[:spec.MAX_TRUTH]

    def _prev_intent(self, a):
        v = [0.0] * spec.INTENT_DIM
        for i, h in enumerate(spec.HEAD_NAMES):
            if h in spec.POINTER_HEADS:
                v[i] = float(a["last"][h]) / 64.0
            else:
                v[i] = float(a["last"][h]) / max(1, spec.CAT_SIZES[h] - 1)
            v[15 + i] = min(a["since"][h], 40) / 40.0
        return v

    def _own(self, aid, a, n, ents):
        S = self.S
        v = [0.0] * spec.OWN_DIM
        v[0] = 0.7 + 0.1 * _sin(S["t"] * 0.3 + a["perf"][0] * 6)
        v[1] = 0.5 + 0.2 * _sin(S["t"] * 0.05 + a["perf"][1] * 6)
        v[2] = a["missiles"] / 6.0
        v[3] = a["chaff"] / 40.0
        v[4] = S["t"] / float(self.cfg["max_steps"])
        v[5] = a["hold"] / float(self.cfg["hold_steps"])
        v[6 + a["view_mode"]] = 1.0
        v[9] = min(a["free_steps"], 40) / 40.0
        v[10] = _sin(a["look_az"] * math.pi / 4.0); v[11] = _cos(a["look_az"] * math.pi / 4.0)
        v[12] = float(a["look_el"] - 1)
        v[13] = n / 64.0
        v[14] = 1.0 if any(c["type"] == T_MAW for c in ents) else 0.0
        v[15] = 1.0 if any(c["lock"] for c in ents) else 0.0
        v[16] = a["chaff_recent"] / 3.0
        for i, p in enumerate(a["perf"]):
            v[17 + i] = p
        v[20 + self.cfg["aircraft"].index(a["aircraft"]) % 6] = 1.0
        v[26] = len(S["agents"]) / 8.0
        v[27] = 1.0                           # validity flag for the block above
        return v

    def observe(self):
        """Observation for every alive controlled agent (cached masks for strict checking)."""
        S = self.S
        out = {}
        self._cache = {}
        dropped_total = 0
        for aid in sorted(S["agents"]):
            a = S["agents"][aid]
            if not a["alive"]:
                continue
            vis = self._visible_contacts(aid)
            vis.sort(key=lambda c: (self._priority(c, a), c["rng"]))
            cap = self.cfg["max_entities"]
            dropped = max(0, len(vis) - cap)
            ents = vis[:cap]
            S.setdefault("dropped", {})[aid] = dropped
            masks = self._build_masks(aid, ents)
            self._cache[aid] = {"n": len(ents), "masks": masks, "ents": ents}
            out[aid] = AgentObs(
                own=self._own(aid, a, len(ents), ents),
                entities=[self._entity_features(c) for c in ents],
                prev_intent=self._prev_intent(a),
                masks=masks,
                truth=self._truth_tokens(),
                aircraft=a["aircraft"],
                dt=spec.DT_STEP,
            )
        return out

    # ------------------------------------------------------------------ stepping
    def _check_actions(self, actions):
        S = self.S
        for aid, a in S["agents"].items():
            if a["alive"] and aid not in actions:
                raise KeyError("missing action for agent %s" % aid)
        for aid, act in actions.items():
            if aid not in self._cache:
                raise KeyError("action for dead or unknown agent %s" % aid)
            for h in spec.HEAD_NAMES:
                if h not in act:
                    raise KeyError("agent %s: head %s missing" % (aid, h))
            if self.cfg["strict"]:
                c = self._cache[aid]
                bad = spec.illegal_heads(c["n"], c["masks"], act)
                if bad:
                    raise IllegalAction("agent %s: illegal options for heads %s (actions=%r)" % (aid, bad, act))

    def step(self, actions):
        cfg = self.cfg
        rng = self.rng
        S = self.S
        if S is None or S["over"]:
            raise RuntimeError("step() after the episode ended; call reset()")
        self._check_actions(actions)
        S["t"] += 1
        rewards = {}
        dones = {}
        ev = {"launch": 0, "kill": 0, "assist": 0, "death": 0, "dropped_entities": 0}
        for aid in sorted(actions):
            a = S["agents"][aid]
            act = actions[aid]
            ents = self._cache[aid]["ents"]
            n = len(ents)
            rewards[aid] = 0.0
            ignored = rng.random() < cfg["p_ignore"]
            prev_vm = a["view_mode"]
            vm = act["view_mode"]
            a["view_mode"] = vm
            a["free_steps"] = a["free_steps"] + 1 if vm != 0 else 0
            if vm == 1:
                vo = act["view_object"]
                a["view_obj"] = ents[vo]["uid"] if vo < n else None
            else:
                a["view_obj"] = None
            if vm == 2:
                a["look_az"] = act["look_az"]; a["look_el"] = act["look_el"]
            # flight intent (mouse) only while aiming
            if vm == 0:
                if not ignored:
                    ref = act["maneuver_ref"]
                    ref_uid = ents[ref]["uid"] if ref < n else None
                    changed = (prev_vm != 0 or ref_uid != a["ref"] or act["maneuver"] != a["man"]
                               or act["vertical"] != a["vert"])
                    a["ref"], a["man"], a["vert"] = ref_uid, act["maneuver"], act["vertical"]
                    a["hold"] = cfg["hold_steps"] if changed else max(0, a["hold"] - 1)
                else:
                    a["hold"] = max(0, a["hold"] - 1)
            else:
                a["hold"] = 0
            # book-keeping for prev_intent
            for h in spec.HEAD_NAMES:
                if act[h] == a["last"][h]:
                    a["since"][h] += 1
                else:
                    a["since"][h] = 0
                a["last"][h] = act[h]
            # chaff, radar, speed
            a["chaff_recent"] = max(0, a["chaff_recent"] - 1)
            if act["chaff"] > 0 and a["chaff"] > 0:
                a["chaff"] -= 1
                a["chaff_recent"] = 3
            a["stt"] = 1 if act["radar_mode"] == 2 else 0
            a["ant"] = act["antenna"]
            a["slow"] = 1 if act["speed"] == 2 else 0
            a["kb_active"] = 1 if (vm != 0 and act["kb_roll"] != 1 and act["kb_pitch"] != 1) else 0
            # weapon
            if act["weapon"] == 1 and act["target"] < n and a["missiles"] > 0 and ents[act["target"]]["type"] == T_RADAR:
                tc = ents[act["target"]]
                a["missiles"] -= 1
                ev["launch"] += 1
                p = max(0.05, 0.85 - 0.0075 * tc["rng"]) + (0.1 if a["stt"] else 0.0)
                p += 0.08 if a["ant"] == _antenna_bin(tc["elev"]) else 0.0
                S["shots"].append({"aid": aid, "uid": tc["uid"], "left": rng.randint(8, 16), "p_hit": min(0.95, p)})
                tc["support"] = 1
        # --- contacts evolve
        for aid in sorted(actions):
            a = S["agents"][aid]
            cs = S["contacts"][aid]
            a["warn_new"] = False
            keep = []
            for c in cs:
                if c["type"] in (T_MAW, T_OWNMSL):
                    keep.append(c)
                    continue
                c["rng"] = max(2.0, c["rng"] + rng.gauss(-0.4, 1.0))
                c["brg"] = (c["brg"] + rng.gauss(0, 0.03) + math.pi) % (2 * math.pi) - math.pi
                c["elev"] = max(-0.5, min(0.5, c["elev"] + rng.gauss(0, 0.01)))
                if c["type"] == T_RADAR and rng.random() < 0.8:
                    c["age"] = 0.0
                else:
                    c["age"] += spec.DT_STEP
                c["extrap"] = 1 if c["age"] > 2.0 else 0
                if c["lock_left"] > 0:
                    c["lock_left"] -= 1
                    if c["lock_left"] == 0:
                        c["lock"] = 0
                elif c["type"] == T_RADAR and rng.random() < 0.01:
                    c["lock"] = 1; c["lock_left"] = rng.randint(3, 8); a["warn_new"] = True
                if rng.random() < 0.03 or c["age"] > 15.0:
                    continue
                keep.append(c)
            # keep contacts referenced by an in-flight shot alive
            S["contacts"][aid] = keep
            tn = self._target_count()
            cap_total = cfg["max_entities"] + (30 if cfg["overflow"] else 0)
            tn = min(tn, cap_total)
            while len(S["contacts"][aid]) < tn and rng.random() < 0.7:
                S["contacts"][aid].append(self._new_contact())
            while len(S["contacts"][aid]) > tn and rng.random() < 0.5:
                victim = rng.randrange(len(S["contacts"][aid]))
                if S["contacts"][aid][victim]["type"] not in (T_MAW, T_OWNMSL):
                    S["contacts"][aid].pop(victim)
        # --- incoming threats
        for aid in sorted(actions):
            a = S["agents"][aid]
            if not any(th["aid"] == aid for th in S["threats"]) and rng.random() < cfg["p_incoming"]:
                S["threats"].append({"aid": aid, "tti": rng.randint(7, 14), "visible_at": rng.choice((9, 9, 8, 7, 0)), "uid": None})
        for th in list(S["threats"]):
            a = S["agents"][th["aid"]]
            th["tti"] -= 1
            cs = S["contacts"][th["aid"]]
            if th["visible_at"] > 0 and th["tti"] <= th["visible_at"] and th["uid"] is None and th["tti"] > 0:
                c = self._new_contact(T_MAW)
                c["rng"] = 5.0 + th["tti"]; c["age"] = 0.0
                cs.append(c)
                th["uid"] = c["uid"]
                a["warn_new"] = True
            if th["tti"] <= 0:
                evasive = 0.0
                if a["view_mode"] == 0 and a["man"] in DEFENSIVE_MANEUVERS:
                    evasive += 0.35
                if a["view_mode"] == 0 and a["vert"] in (3, 4):
                    evasive += 0.2
                if a["view_mode"] != 0 and a.get("kb_active"):
                    evasive += 0.35
                if a["chaff_recent"] > 0:
                    evasive += 0.3
                if a.get("slow"):
                    evasive -= 0.1
                p_hit = min(0.95, max(0.05, 0.9 - evasive))
                S["threats"].remove(th)
                if th["uid"] is not None:
                    S["contacts"][th["aid"]] = [c for c in S["contacts"][th["aid"]] if c["uid"] != th["uid"]]
                if rng.random() < p_hit:
                    self._kill(th["aid"], rewards, dones, ev)
        # --- shots resolve
        for sh in list(S["shots"]):
            sh["left"] -= 1
            if sh["left"] <= 0:
                S["shots"].remove(sh)
                a = S["agents"][sh["aid"]]
                if a["alive"] and rng.random() < sh["p_hit"]:
                    rewards[sh["aid"]] = rewards.get(sh["aid"], 0.0) + 1.0
                    ev["kill"] += 1
                    S["contacts"][sh["aid"]] = [c for c in S["contacts"][sh["aid"]] if c["uid"] != sh["uid"]]
            elif rng.random() < 0.01:
                a = S["agents"][sh["aid"]]
                if a["alive"]:
                    rewards[sh["aid"]] = rewards.get(sh["aid"], 0.0) + 0.3
                    ev["assist"] += 1
        # --- background death
        for aid in sorted(actions):
            if S["agents"][aid]["alive"] and rng.random() < cfg["p_background_death"]:
                self._kill(aid, rewards, dones, ev)
        # --- enemies drift (truth only)
        for e in S["enemies"]:
            for j in range(3):
                e["pos"][j] = max(-1.5, min(1.5, e["pos"][j] + 0.05 * e["vel"][j]))
        # --- hold release on a new visible warning (emergency release)
        for aid in sorted(actions):
            a = S["agents"][aid]
            if a["alive"] and a["warn_new"]:
                a["hold"] = 0
        for aid in actions:
            a = S["agents"][aid]
            a["ret"] += rewards.get(aid, 0.0)
        for aid in S["agents"]:
            if aid in actions and S["agents"][aid]["alive"]:
                dones.setdefault(aid, False)
        # --- episode end
        alive = [aid for aid, a in S["agents"].items() if a["alive"]]
        info = {"timeout": False, "events": ev}
        final_obs_for = []
        if not alive:
            S["over"] = True
        elif S["t"] >= cfg["max_steps"]:
            S["over"] = True
            info["timeout"] = True
            final_obs_for = alive
            for aid in alive:
                dones[aid] = True
        elif rng.random() < cfg["p_match_end"]:
            S["over"] = True
            for aid in alive:
                dones[aid] = True
        obs = {}
        if alive and (not S["over"] or info["timeout"]):
            obs = self.observe()
        else:
            self._cache = {}
        ev["dropped_entities"] = sum(S.get("dropped", {}).get(aid, 0) for aid in obs)
        for aid in actions:
            rewards.setdefault(aid, 0.0)
            dones.setdefault(aid, False)
        return obs, rewards, dones, info

    def _kill(self, aid, rewards, dones, ev):
        S = self.S
        a = S["agents"][aid]
        if not a["alive"]:
            return
        a["alive"] = False
        rewards[aid] = rewards.get(aid, 0.0) - 2.0
        dones[aid] = True
        ev["death"] += 1
        S["threats"] = [th for th in S["threats"] if th["aid"] != aid]

    # ------------------------------------------------------------------ scripted intents
    def scripted_actions(self):
        """Intents of a simple scripted pilot, before executor noise. Uses only visible state."""
        S = self.S
        srng = self.script_rng
        out = {}
        for aid in sorted(self._cache):
            a = S["agents"][aid]
            c = self._cache[aid]
            ents, n = c["ents"], c["n"]
            threat = any(e["type"] == T_MAW for e in ents)
            tracks = [i for i, e in enumerate(ents) if e["type"] == T_RADAR]
            tracks.sort(key=lambda i: ents[i]["rng"])
            tgt = tracks[0] if tracks else n
            want = {h: spec.default_option(h, n) for h in spec.HEAD_NAMES}
            # view: mostly aim; sometimes look at the threat / around for a few steps
            if a["view_mode"] != 0 and a["free_steps"] < 3 and not threat:
                want["view_mode"] = a["view_mode"]
            elif srng.random() < 0.08 and not threat:
                want["view_mode"] = srng.choice((1, 2))
            else:
                want["view_mode"] = 0
            want["target"] = tgt
            maws = [i for i, e in enumerate(ents) if e["type"] == T_MAW]
            want["view_object"] = maws[0] if maws else (tgt if tgt < n else 0)
            want["look_az"] = srng.randrange(8)
            want["look_el"] = 1
            want["maneuver_ref"] = tgt
            if threat:
                want["maneuver"] = 4 if srng.random() < 0.5 else 5
                want["vertical"] = 4
                want["speed"] = 0
                want["chaff"] = 1 if a["chaff"] > 0 else 0
            else:
                want["maneuver"] = 1 if tgt < n else 0
                want["vertical"] = 0
                want["speed"] = 1
                want["chaff"] = 0
            close = tgt < n and ents[tgt]["rng"] < 55.0
            want["radar_mode"] = 2 if close else 0
            if tgt < n:
                want["antenna"] = _antenna_bin(ents[tgt]["elev"])
            want["weapon"] = 1 if (close and a["missiles"] > 0 and srng.random() < 0.5) else 0
            want["kb_roll"] = 0 if threat else 1
            want["kb_pitch"] = 2 if threat else 1
            out[aid] = self._legalize(c, want)
        return out

    def _legalize(self, c, want):
        n, masks = c["n"], c["masks"]
        chosen = {}
        for h in spec.SAMPLE_ORDER:
            m = spec.effective_mask(h, n, masks, chosen)
            w = want[h]
            if not (0 <= w < len(m)) or not m[w]:
                legal = [i for i, ok in enumerate(m) if ok]
                w = min(legal, key=lambda i: abs(i - w))
            chosen[h] = w
        return {h: chosen[h] for h in spec.HEAD_NAMES}

    # ------------------------------------------------------------------ snapshot / restore
    def snapshot(self):
        return copy.deepcopy({
            "S": self.S, "rng": self.rng.getstate(), "script_rng": self.script_rng.getstate(),
            "episode_count": self.episode_count, "scenario": self.scenario,
        })

    def restore(self, state):
        st = copy.deepcopy(state)
        self.S = st["S"]
        self.rng.setstate(st["rng"])
        self.script_rng.setstate(st["script_rng"])
        self.episode_count = st["episode_count"]
        self.scenario = st["scenario"]
        if self.S is not None and not self.S["over"]:
            self.observe()        # rebuild the mask cache (pure function of the state)
        else:
            self._cache = {}
