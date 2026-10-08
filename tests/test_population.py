"""Population v2 of the scripted opponents (docs/population_spec.md).

The fast tests check the switch (the v1 scripts are untouched without it), the draws and the match files. The behaviour
tests fly twelve seeded 240 s scripted matches (six 1v1s, six 2v2s; in parallel processes unless
POPULATION_TEST_WORKERS=1) once and check what the pilots did:

  (a) turn-in times spread (coefficient of variation >= 0.3 within an archetype),
  (b) one pilot flies at least three different evasions,
  (c) the scripts' altitudes cover at least four of the bands 0-0.5, 0.5-2, 2-5, 5-8, 8-10, 10+ km,
  (d) a spammer fires two missiles within 30 s,
  (e) a wingman stays 3-8 km from his lead for most of the lead's climb,
  (f) top pilots react faster than normal ones on the executor's delay path,
  (g) the replay header carries every sampled parameter.

The pusher tests fly six 45 s 1v1s in the controlled study's co-heading geometry (push_game,
docs/analysis_offboresight_push.md): at 10 km a Su-30SM2 pusher fires at once far off the nose, ripples a few seconds
apart and pursues; a J-16 pusher turns in only as far as its 70 degree limit needs; at 22 km the Su-30SM2 turns to
40 degrees before firing.

The matches use the executor's vertical_mode 'angle' when intent.py has it (any altitude target is flown), else the
default altitude options.
"""
from __future__ import annotations

import concurrent.futures
from dataclasses import fields, replace
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import unittest

from wt_overlay import archetypes, intent, match
from wt_overlay.engagement import ReplayWriter

ROOT = Path(__file__).resolve().parents[1]
UNTIL_S = 240.
BANDS_M = (0., 500., 2000., 5000., 8000., 10000., math.inf)
EXECUTION = dict(vertical_mode="angle") if "angle" in getattr(intent, "VERTICAL_MODES", ()) else {}


def m(aircraft, archetype=None, skill=None):
    member = dict(aircraft=aircraft)
    if archetype:
        member["archetype"] = archetype
    if skill:
        member["skill"] = skill
    return member


LEFT = m("f_15c_golden_eagle", "left", "normal")    # the same left flyer in every 1v1: only his own draws differ
MATCHES = (
    (1, [[LEFT], [m("ef_2000_aesa", "spammer", "top")]], {}),
    (2, [[LEFT], [m("f_15c_golden_eagle", "speedster", "normal")]], {}),
    (3, [[LEFT], [m("f_16c_block_52_aesa", "deck_notcher", "top")]], {}),
    (4, [[LEFT], [m("su_30sm2", "middle", "top")]], {}),
    (5, [[LEFT], [m("saab_jas39e", "crawler", "normal")]], {}),
    (6, [[LEFT], [m("fa_18e_block_2", "rusher", "normal")]], {}),
    (7, [[m("f_15c_golden_eagle", "left", "top"), m("f_15c_golden_eagle", "pair", "normal")],
         [m("ef_2000_aesa", "spammer", "normal"), m("su_30sm2", "middle", "top")]], dict(spread_km=5.)),
    (8, [[m("j_16", "bait", "normal"), m("su_30sm2", "middle", "normal")],
         [m("f_16c_block_52_aesa", "left", "top"), m("j_10c", "right", "normal")]], {}),
    (9, [[m("su_30sm2"), m("ef_2000_aesa")], [m("f_15c_golden_eagle"), m("j_10c")]], {}),
    (10, [[m("su_30mkm", "pair", "top"), m("su_30sm", "left", "normal")],
          [m("saab_jas39e", "deck_notcher", "normal"), m("ef_2000_fgr4", "spammer", "top")]], dict(spread_km=5.)),
    (11, [[m("f_15c_golden_eagle", "left", "normal"), m("j_15t", "middle", "normal")],
          [m("mig_35", "left", "top"), m("su_30sm2", "spammer", "normal")]], {}),
    (12, [[m("ef_2000a_aesa", "middle", "normal"), m("f_15c_golden_eagle", "left", "normal")],
          [m("j_16", "speedster", "top"), m("j_10c", "crawler", "normal")]], {}),
)


def fly(case, population="v2", until_s=UNTIL_S, execution=None):
    """One scripted match of MATCHES; what the tests read from it (plain data, so it can come from a worker)."""
    seed, teams, kwargs = case
    built = match.scenario(teams, seed, range_km=100., population=population, **kwargs)
    writer = ReplayWriter(None, keep=True)
    eng = built.engagement(replay=writer, execution=EXECUTION if execution is None else execution)
    delays = {}
    for p in eng.planes:
        ex, record = p.controller.executor, delays.setdefault(p.ident, [])

        def offer(intent_, now, leaving, ex=ex, original=ex._offer, record=record):
            n = len(ex.pending)
            original(intent_, now, leaving)
            if len(ex.pending) > n:          # taken up (not rejected): the delay just drawn
                record.append(ex.last_delay)
        ex._offer = offer
    eng.run(until_s=until_s)
    header = json.loads(writer.lines[0])
    frames = []
    for line in writer.lines:
        row = json.loads(line)
        if row.get("type") == "frame":
            frames.append((row["t"], [(r[0], r[1], r[2], r[3]) for r in row["planes"]]))
    planes = []
    for p in eng.planes:
        pilot = p.controller.pilot
        planes.append(dict(id=p.ident, team=p.team, archetype=p.spec.archetype, skill=p.spec.skill,
                           events=getattr(pilot, "events", []), lead=getattr(pilot, "lead_ident", None),
                           personal_median=p.controller.executor.personal_median,
                           delay_median=pilot.p.delay_median_s, delays=delays[p.ident],
                           params={f.name: archetypes._plain(getattr(pilot.p, f.name)) for f in fields(pilot.p)}))
    log = [e for e in eng.log if e["kind"] in ("phase", "launch", "kill", "death")]
    return dict(seed=seed, header=header, planes=planes, log=log, frames=frames,
                digest=hashlib.sha256("\n".join(writer.lines).encode()).hexdigest())


def fly_all(population="v2", cases=MATCHES):
    workers = int(os.environ.get("POPULATION_TEST_WORKERS", min(6, os.cpu_count() or 1)))
    if workers <= 1:
        return [fly(c, population) for c in cases]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fly, cases, [population]*len(cases)))


# The pusher in the controlled study's geometry: (seed, pusher aircraft, his missile, lateral offset m). The
# Su-30SM2's R-77-1 may be fired 120 degrees off the nose, the J-16's PL-12A 70 (intent.launch_limit).
PUSH_CASES = ((2, "su_30sm2", "su_r_77_1", 10000.), (6, "su_30sm2", "su_r_77_1", 10000.),
              (3, "j_16", "cn_pl12a", 10000.), (6, "j_16", "cn_pl12a", 10000.),
              (2, "su_30sm2", "su_r_77_1", 22000.), (6, "su_30sm2", "su_r_77_1", 22000.))
PUSH_UNTIL_S = 45.


def push_game(case, until_s=PUSH_UNTIL_S):
    """A top pusher (team 0) and an unarmed target (team 1: a normal v2 middle flyer without missiles, so he heads home
    and defends) fly east side by side at 9 km and Mach 0.9, the case's offset apart, the target on the pusher's left
    (the study, docs/analysis_offboresight_push.md: co-heading at 8-22 km lateral offset). Each team has the other on
    the map from the start, kept fresh as a teammate's radar would. Returns the pusher's launches (time, off-boresight
    at the launch from truth), his attack plans, and every second the target's bearing off his nose, the range and
    his phase."""
    from wt_overlay.fm import atmosphere
    seed, aircraft, missile, offset = case
    target = "su_30sm2" if aircraft == "j_16" else "j_16"
    built = match.scenario([[dict(aircraft=aircraft, archetype="pusher", skill="top", missile=missile)],
                            [dict(aircraft=target, archetype="middle", skill="normal", missiles=0)]], seed,
                           range_km=50., population="v2")
    speed = .9*atmosphere(9000.)[1]
    for s in built.specs:
        s.position, s.velocity = (-30000., (.5 if s.team else -.5)*offset, 9000.), (speed, 0., 0.)
    eng = built.engagement(execution=EXECUTION)
    pusher, other = eng.planes

    def bearing_off(a, b):
        pa, pb, v = a.flight.state.position, b.flight.state.position, a.flight.state.velocity
        return archetypes.wrap(archetypes.bearing_of(pb[0]-pa[0], pb[1]-pa[1])-archetypes.bearing_of(v[0], v[1]))

    launches, fire = [], eng.fire

    def record(plane, aim):
        if plane is pusher:
            launches.append((round(eng.time, 2), round(abs(bearing_off(plane, aim)), 1)))
        return fire(plane, aim)
    eng.fire = record
    track, next_s, dead_t = [], 0., None
    while eng.reason is None and eng.time < until_s:
        for p in eng.planes:                 # the map marks a teammate's radar would keep fresh
            q = other if p is pusher else pusher
            if q.alive:
                eng.marks[p.team][q.ident] = (*q.flight.state.position, eng.time)
        eng.step()
        if not other.alive and dead_t is None:
            dead_t = eng.time
        if eng.time >= next_s and pusher.alive and other.alive:
            next_s += 1.
            track.append((round(eng.time, 2), round(bearing_off(pusher, other), 1), pusher.phase,
                          round(math.dist(pusher.flight.state.position, other.flight.state.position))))
    pilot = pusher.controller.pilot
    plans = [{k: v for k, v in e.items() if k != "uids"} for e in pilot.events if e["kind"] == "attack"]
    return dict(case=case, launches=launches, plans=plans, track=track, dead_t=dead_t, gap=pilot.p.salvo_gap_s,
                limit=intent.launch_limit(aircraft, missile), top=pilot.push_top, events=pilot.events)


def push_all(cases=PUSH_CASES):
    workers = int(os.environ.get("POPULATION_TEST_WORKERS", min(6, os.cpu_count() or 1)))
    if workers <= 1:
        return [push_game(c) for c in cases]
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(workers, len(cases))) as pool:
        return list(pool.map(push_game, cases))


def commit_times(results):
    """{archetype: [turn-in times]} of the climbing v2 pilots (never-commit pilots left out)."""
    out = {}
    for r in results:
        for p in r["planes"]:
            for e in p["events"]:
                if e["kind"] == "commit" and e["trigger"] != "never":
                    out.setdefault(p["archetype"], []).append(e["t"])
    return out


def cv(values):
    return statistics.pstdev(values)/statistics.mean(values)


class SwitchTests(unittest.TestCase):
    """Without the option the v1 population flies exactly as before; v2 keeps the match the seed would build."""

    def test_v1_is_unchanged_without_the_key(self):
        teams = [["su_30sm2", "f_15c_golden_eagle"], ["j_10c", "ef_2000_aesa"]]
        digests = []
        for kwargs in ({}, dict(population="v1"), dict(perturb={"population": "v1"}), dict(perturb={"population": None})):
            built = match.scenario(teams, 7, range_km=100., **kwargs)
            for s in built.specs:
                self.assertIs(type(s.controller), archetypes.Pilot)
                self.assertFalse(set(s.controller.describe()) & archetypes.V2_FIELDS)
                self.assertFalse(hasattr(s, "script_params"))
            writer = ReplayWriter(None, keep=True)
            built.engagement(replay=writer).run(until_s=60.)
            digests.append(hashlib.sha256("\n".join(writer.lines).encode()).hexdigest())
        self.assertEqual(len(set(digests)), 1)
        # The perturbation is still the v1 one ({} = its defaults), with or without a population key.
        plain = match.random_match(4, team_size=8, perturb={})
        keyed = match.random_match(4, team_size=8, perturb={"population": "v1", "p": .5})
        self.assertEqual([s.controller.p for s in plain.specs], [s.controller.p for s in keyed.specs])

    def test_v2_keeps_aircraft_spawns_and_loadouts(self):
        for build in (lambda **k: match.random_match(5, team_size=4, **k),
                      lambda **k: match.scenario([["su_30sm2", "j_10c"], ["f_15c_golden_eagle", "ef_2000_aesa"]], 3, **k)):
            old, new = build(), build(population="v2")
            key = lambda s: (s.aircraft, s.team, s.position, s.velocity, s.missile, s.missiles, s.chaff, s.rcs_ratio,
                             s.mass_factor, s.skill)
            self.assertEqual([key(s) for s in old.specs], [key(s) for s in new.specs])
            self.assertTrue(all(type(s.controller) is archetypes.PilotV2 for s in new.specs))
            self.assertTrue(all(s.controller.p.population == "v2" for s in new.specs))
            self.assertEqual(new.notes["population"], "v2")

    def test_the_switch_is_validated(self):
        with self.assertRaises(ValueError):
            match.scenario([["su_30sm2"], ["j_10c"]], 1, population="v3")
        with self.assertRaises(ValueError):     # v2 replaces the v1 perturbation
            match.scenario([["su_30sm2"], ["j_10c"]], 1, perturb={"population": "v2", "p": .3})
        with self.assertRaises(ValueError):
            match.scenario([["su_30sm2"], ["j_10c"]], 1, population="v1", perturb={"population": "v2"})
        with self.assertRaises(ValueError):     # new archetypes only in v2
            match.scenario([[dict(aircraft="su_30sm2", archetype="spammer")], ["j_10c"]], 1)
        two = match.scenario([[dict(aircraft="su_30sm2", archetype="spammer")], ["j_10c"]], 1,
                             perturb={"population": "v2"})
        self.assertEqual(two.specs[0].archetype, "spammer")

    def test_match_env_takes_the_population_through_script_perturbation(self):
        from wt_overlay.rl_env import MatchEnv
        teams = [[dict(aircraft="f_15c_golden_eagle", archetype="left", skill="top")],
                 [dict(aircraft="su_30sm2", archetype="spammer", skill="normal")]]
        e = MatchEnv(dict(teams=teams, range_km=80., time_limit_s=30., script_perturbation={"population": "v2"}), 3)
        e.reset()
        header = e.engagement._header()
        script = header["planes"][1]["script"]
        self.assertEqual((script["population"], script["archetype"]), ("v2", "spammer"))
        pilot, ex = e.pilots[1], e.executors[1]
        self.assertEqual(ex.path, "autonomous")
        self.assertAlmostEqual(ex.personal_median, pilot.p.delay_median_s, places=12)
        policy = e.executors[0]                         # the policy slot keeps its own path and median draw
        self.assertEqual(policy.path, "follow")
        self.assertTrue(.5 <= policy.personal_median <= 1.2)
        for _ in range(8):
            e.step(e.scripted_actions())
        plain = MatchEnv(dict(teams=[teams[0], [dict(teams[1][0], archetype="middle")]], range_km=80.,
                              time_limit_s=30.), 3)
        plain.reset()
        self.assertIsNone(plain.engagement._header()["planes"][1]["script"])   # v1 under MatchEnv: as before
        with self.assertRaises(ValueError):
            MatchEnv(dict(teams=teams, script_perturbation={"population": "v2", "p": .5}), 3).reset()


class PilotRuleTests(unittest.TestCase):
    """Rules shared by both populations: the climb under vertical_mode 'angle' and bingo fuel."""

    @classmethod
    def setUpClass(cls):
        cls.model = match.load_model()

    def pilot(self, cls=archetypes.Pilot, archetype="middle", **override):
        params = archetypes.sample_params(self.model, "f_15c_golden_eagle", archetype, "normal", random.Random(11))
        if cls is archetypes.PilotV2:
            params = archetypes.population_params(params, archetype, self.model, "f_15c_golden_eagle", random.Random(1))
        params = replace(params, peak_alt_m=None, **override)
        return cls(params, random.Random(2), team_forward=(0., 1.), home_xy=(0., -50000.), enemy_xy=(0., 50000.))

    def obs(self, t, altitude, **own):
        from wt_overlay.engagement import Observation, OwnObs
        values = dict(time_s=t, team=0, aircraft="f_15c_golden_eagle", position=(0., -40000., altitude),
                      velocity=(0., 300., 0.), heading_deg=0., pitch_deg=0., roll_deg=0., speed_mps=300.,
                      altitude_m=altitude, aoa_deg=1., load=1., engine_percent=100., missile_id=None, missiles=6,
                      chaff=30, radar_mode="tws", stt_state=None, has_maw=False)
        values.update(own)
        return Observation(t, OwnObs(**values), (), (), (), (), (), (), (), 64000.)

    def test_angle_mode_climb_ends_at_the_own_level_altitude(self):
        for mode, left in (("angle", True), ("altitude", False)):
            pilot = self.pilot(level_alt_m=8750.)
            pilot.managed_execution, pilot.vertical_mode = True, mode
            pilot.decide(self.obs(0., 8550.))           # 200 m short of 8750: the angle executor's level-off
            self.assertEqual(pilot.phase != "climb", left, mode)
        # The stall fallback: a climb that stopped (here 1 km short) also counts as done under 'angle'.
        pilot = self.pilot(level_alt_m=9500.)
        pilot.managed_execution, pilot.vertical_mode = True, "angle"
        for t in range(0, 34, 2):
            pilot.decide(self.obs(float(t), 8500.))
        self.assertNotEqual(pilot.phase, "climb")

    def test_bingo_fuel_sends_a_pilot_home_and_keeps_him_there(self):
        from wt_overlay.engagement import OwnObs
        if "bingo" not in {f.name for f in fields(OwnObs)}:
            self.skipTest("no fuel model in this tree")
        for cls, archetype in ((archetypes.Pilot, "middle"), (archetypes.PilotV2, "middle")):
            pilot = self.pilot(cls, archetype)
            pilot.decide(self.obs(0., 8000., bingo=True))
            self.assertEqual(pilot.phase, "home", cls.__name__)
            pilot.decide(self.obs(1., 8000., bingo=True))
            self.assertEqual(pilot.phase, "home")
            pilot.decide(self.obs(2., 8000.))             # refuelled (airfield): back out
            self.assertNotEqual(pilot.phase, "home")


class DrawTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = match.load_model()

    def draws(self, archetype, skill="normal", n=300, aircraft="f_15c_golden_eagle"):
        rng = random.Random(f"{archetype}:{skill}")
        out = []
        for i in range(n):
            base = archetypes.sample_params(self.model, aircraft, archetypes.BASE_ARCHETYPE.get(archetype, archetype),
                                            skill, rng)
            out.append(archetypes.population_params(base, archetype, self.model, aircraft, random.Random(i)))
        return out

    def test_v2_fields_are_the_new_ones(self):
        names = [f.name for f in fields(archetypes.PilotParams)]
        self.assertEqual(set(names[names.index("population"):]), set(archetypes.V2_FIELDS))

    def test_draws_follow_the_documented_ranges(self):
        s = archetypes.population_settings(self.model)
        cm, dc = s["commit"], s["doctrine"]
        left = self.draws("left")
        self.assertTrue(all(20. <= d.flank_offset_deg <= 60. for d in left))
        self.assertTrue(all(d.commit_triggers[-1] == "timer" for d in left))
        self.assertTrue(all(30000. <= d.commit_contact_m <= 70000. for d in left))
        timers = sorted(d.commit_timer_s for d in left)
        self.assertTrue(60. < timers[len(timers)//2] < 180.)                         # log-normal around 60-180 s
        self.assertTrue(all(cm["timer_s"]["clip"][0] <= t <= cm["timer_s"]["clip"][1] for t in timers))
        self.assertGreater(len({d.commit_triggers for d in left}), 4)               # pilots differ in what moves them
        self.assertTrue(any(d.commit_never for d in left))
        self.assertTrue(all(.5 <= d.launch_fraction <= 1.1 for d in left))
        self.assertTrue(all(d.shots_per_engagement in (1, 2, 3) for d in left))
        self.assertTrue(all(2. <= d.salvo_gap_s <= 8. for d in left))
        self.assertTrue(all(0. <= d.p_stt <= .4 for d in left))
        self.assertTrue(all(abs(sum(w for _, w in d.evade_weights)-1.) < 1e-9 for d in left))
        self.assertTrue(all(1.5 <= d.delay_median_s <= 2.5 for d in left))
        self.assertTrue(all(.8 <= d.delay_median_s <= 1.5 for d in self.draws("left", "top", 50)))
        wide = sum(not 20000. <= d.second_round_m <= 30000. for d in left)/len(left)
        self.assertTrue(.05 < wide < dc["p_second_round_wide"])                      # the wide draw, partly inside 20-30
        spam = self.draws("spammer", n=100)
        self.assertTrue(all(d.shots_per_engagement in (2, 3) and .9 <= d.launch_fraction <= 1.1 for d in spam))
        self.assertTrue(all(dict(d.support_weights) == {"drag": 1.} for d in spam))
        fast = self.draws("speedster", n=50)
        self.assertTrue(all(9500. <= d.level_alt_m <= 11500. and d.p_react == d.p_lock_react == 1. for d in fast))
        self.assertTrue(all(not d.commit_triggers for d in fast))
        deck = self.draws("deck_notcher", n=50, aircraft="j_10c")
        self.assertTrue(all(30. <= d.crawler_alt_m <= 200. and 0. <= d.deck_offset_deg <= 40. for d in deck))
        self.assertTrue(all("dive20" not in dict(d.evade_weights) for d in deck))
        pair = self.draws("pair", n=50)
        self.assertTrue(all(3000. <= d.pair_offset_m <= 8000. and d.commit_triggers[0] == "lead" for d in pair))
        rush = self.draws("rusher", n=50)
        self.assertTrue(all(d.p_react == archetypes.RUSH_P_REACT and d.delay_median_s >= 1.5*1.3 for d in rush))
        push = self.draws("pusher", n=200, aircraft="su_30sm2")
        self.assertTrue(all(d.shots_per_engagement in (2, 3) and 3. <= d.salvo_gap_s <= 6. for d in push))
        self.assertTrue(.25 < sum(d.shots_per_engagement == 3 for d in push)/len(push) < .55)     # 2 / 3: 0.6 / 0.4
        self.assertTrue(all(.8 <= d.launch_fraction <= 1. and dict(d.support_weights) == {"straight": 1.}
                            for d in push))                                                 # pursues, never cranks
        self.assertTrue(all(8. <= d.push_wait_s <= 16. and d.p_stt == 0. and d.commit_never is None for d in push))
        self.assertTrue(all(d.commit_triggers[0] == "envelope" and d.commit_triggers[-1] == "timer" for d in push))
        self.assertTrue(all(20. <= abs(d.flank_offset_deg) <= 60. for d in push))
        self.assertTrue(.3 < sum(d.flank_offset_deg < 0. for d in push)/len(push) < .7)     # both flanks
        self.assertTrue(all(d.push_wait_s == 0. for d in left+spam+fast+deck+pair+rush))

    def test_match_files_carry_the_same_valid_population_section(self):
        sections = []
        for name in ("top_tier.json", "top_tier_s1_far.json"):
            model = json.loads((ROOT/"data"/"match"/name).read_text())
            sections.append((model["population"], [g.get("prior_v2_extra") for g in model["groups"]]))
            s = archetypes.population_settings(model)          # unknown keys would raise
            self.assertEqual(set(s), set(archetypes.POPULATION_V2))
            for g in model["groups"]:
                extra = g["prior_v2_extra"]
                self.assertEqual(set(extra), set(archetypes.NEW_ARCHETYPES))
                self.assertTrue(all(.03 <= v <= .08 for v in extra.values()))
                self.assertTrue(.03 <= extra["pusher"] <= .06)                  # about the spammer's weight
                self.assertTrue(.2 <= sum(extra.values()) <= .3)
                self.assertEqual(set(g["prior"]), set(archetypes.ARCHETYPES))   # the C-grade prior is untouched
            self.assertEqual(max(model["groups"], key=lambda g: g["prior_v2_extra"]["pusher"])["aircraft"],
                             ["j_16", "su_30sm2"])        # the most where the R-77-1 shoots 120 degrees off the nose
        self.assertEqual(sections[0], sections[1])
        # The shipped files mirror the code defaults (a file's values override them, so change both together).
        self.assertEqual(sections[0][0]["v2"], archetypes.POPULATION_V2)
        with self.assertRaises(ValueError):
            archetypes.population_settings({"population": {"v2": {"commit": {"wobble": 1}}}})

    def test_new_archetypes_appear_and_team_ones_need_a_teammate(self):
        seen, solo = {}, set()
        for seed in range(40):
            for s in match.random_match(seed, team_size=4, population="v2").specs:
                seen[s.archetype] = seen.get(s.archetype, 0)+1
            for s in match.random_match(seed, team_size=1, population="v2").specs:
                solo.add(s.archetype)
        n = sum(seen.values())
        self.assertTrue(set(archetypes.NEW_ARCHETYPES) <= set(seen))
        self.assertTrue(.17 < sum(seen[a] for a in archetypes.NEW_ARCHETYPES)/n < .37)    # the groups' 0.25-0.28
        self.assertTrue(.02 < seen["pusher"]/n < .09)                                      # the groups' 0.04-0.06
        self.assertFalse(solo & set(archetypes.TEAM_ARCHETYPES))
        self.assertIn("pusher", solo)                                                      # 1v1s have pushers too
        built = match.scenario([[dict(aircraft="su_30sm2", archetype="pair"), dict(aircraft="j_10c", archetype="left"),
                                 dict(aircraft="j_16", archetype="crawler")], ["f_15c_golden_eagle"]], 2,
                               population="v2")
        wing = built.specs[0].controller
        self.assertEqual(wing.lead_ident, 1)                    # nearest climbing teammate, not the crawler
        self.assertIs(wing.radio, built.specs[1].controller.radio)
        self.assertIsNot(wing.radio, built.specs[3].controller.radio)
        self.assertEqual(built.specs[0].script_params["pair_lead"], 1)


class PusherTests(unittest.TestCase):
    """The pusher in the controlled study's co-heading geometry (push_game, docs/analysis_offboresight_push.md; six
    45 s 1v1s, about 20 s in parallel): at 10 km the Su-30SM2 fires at once with the target abeam, follows up a few
    seconds later while turning in and pursues; the J-16 (PL-12A, 70 degrees) turns toward the target only to the 40
    degree offset and never fires beyond its limit; at 22 km the Su-30SM2 turns to 40 degrees before firing."""

    @classmethod
    def setUpClass(cls):
        cls.results = {r["case"]: r for r in push_all()}

    def cases(self, aircraft, offset):
        return [self.results[c] for c in PUSH_CASES if c[1] == aircraft and c[3] == offset]

    def offsets(self, r, t0, t1):
        """|bearing off the nose| of the target from t0 to t1 while the pusher pushes."""
        return [abs(az) for t, az, phase, _ in r["track"] if t0 <= t <= t1 and phase == "push"]

    def check_common(self, r):
        """The envelope commits him at once; every proposal inside the limit; the ripple's presses at least the drawn
        gap apart and its launches a few seconds apart; a re-push only once the target turned back hot or left the
        beam; pursuit after the shot (no crank): the target within 25 degrees of the nose 8-16 s after the first
        launch."""
        commit = next(e for e in r["events"] if e["kind"] == "commit")
        self.assertEqual((commit["t"], commit["trigger"]), (0., "envelope"), r["case"])
        self.assertTrue(r["launches"], r["plans"])
        first = r["plans"][0]
        self.assertFalse(first["repush"])
        self.assertIn(first["shots"], (2, 3))
        self.assertGreaterEqual(first["launched"], 2, first)
        self.assertTrue(all(b-a >= r["gap"]-1e-6 for a, b in zip(first["shot_t"], first["shot_t"][1:])), first)
        ripple = [t for t, _ in r["launches"]][:first["launched"]]
        self.assertTrue(all(b-a <= r["gap"]+4. for a, b in zip(ripple, ripple[1:])), (r["gap"], ripple))
        for plan in r["plans"]:                           # the airframe azimuth the launch limit is checked against
            self.assertTrue(all(a <= r["limit"] for a in plan["az"] if a is not None), plan)
            self.assertNotIn("crank_t", plan)
        for plan in r["plans"][1:]:
            self.assertTrue(plan["repush"] and abs(plan["aspect"]-90.) > 20., plan)
        t0 = r["launches"][0][0]
        pursuit = self.offsets(r, t0+8., t0+16.)
        self.assertGreaterEqual(len(pursuit), 3, (r["launches"], r["dead_t"]))
        self.assertLessEqual(statistics.median(pursuit), 25., pursuit)

    def test_su_30sm2_at_10_km_fires_at_once_far_off_the_nose_follows_up_and_pursues(self):
        for r in self.cases("su_30sm2", 10000.):
            self.assertEqual((r["limit"], r["top"]), (120., 90.))
            self.check_common(r)
            t0, oba0 = r["launches"][0]
            self.assertLessEqual(t0, 12., r["launches"])                 # within seconds of the shot becoming legal
            self.assertGreater(oba0, 60., r["launches"])                 # with the target near abeam
            first = r["plans"][0]
            self.assertGreater(first["oba"][0], 60., first)
            self.assertLessEqual(first["range_m"][0], 16000.)
            self.assertLessEqual(r["launches"][1][0]-t0, 10., r["launches"])   # the follow-up comes quickly

    def test_j_16_turns_in_only_as_needed_and_never_fires_beyond_its_limit(self):
        for r in self.cases("j_16", 10000.):
            self.assertEqual((r["limit"], r["top"]), (70., 40.))
            self.check_common(r)
            t0, oba0 = r["launches"][0]
            self.assertTrue(all(oba <= r["limit"] for _, oba in r["launches"]), r["launches"])
            self.assertTrue(all(a <= r["limit"] for p in r["plans"] for a in p["oba"] if a is not None), r["plans"])
            before = self.offsets(r, 0., t0)
            self.assertLess(min(before), 75., before)      # it turned toward the target abeam (90 degrees at the start)
            self.assertGreater(oba0, 25., r["launches"])  # ... but only to the 40 degree offset, not nose on

    def test_su_30sm2_at_22_km_turns_to_40_degrees_before_firing(self):
        for r in self.cases("su_30sm2", 22000.):
            self.check_common(r)
            first = r["plans"][0]
            self.assertGreater(first["range_m"][0], 16000.)
            for plan in r["plans"]:                       # beyond 16 km nothing far off the nose (horizontally)
                self.assertTrue(all(a <= 45. for a, d in zip(plan["oba"], plan["range_m"]) if d > 16000.), plan)
            t0, oba0 = r["launches"][0]
            self.assertLess(oba0, 55., r["launches"])
            self.assertLess(min(self.offsets(r, 0., t0)), 60.)


class BehaviourTests(unittest.TestCase):
    """Twelve 240 s scripted matches, flown once (about two minutes in six processes)."""

    @classmethod
    def setUpClass(cls):
        cls.results = fly_all("v2")

    def test_a_turn_in_times_spread(self):
        times = commit_times(self.results)
        left = times["left"]
        self.assertGreaterEqual(len(left), 8)
        self.assertGreaterEqual(cv(left), .3, left)
        groups = [v for v in times.values() if len(v) >= 4]
        pooled = math.sqrt(sum(statistics.pvariance(v)*len(v) for v in groups)/sum(len(v) for v in groups))
        self.assertGreaterEqual(pooled/statistics.mean([t for v in groups for t in v]), .3, times)
        # Not the flank geometry: a v1 flanker turns in exactly when his lateral offset reaches the drawn flank
        # distance (10-25 km); a v2 one wherever his trigger finds him. Several kinds of trigger fire.
        lateral, kinds = [], set()
        for r in self.results:
            for p in r["planes"]:
                turn = [e for e in p["events"] if e["kind"] == "commit" and e["trigger"] != "never"]
                kinds.update(e["trigger"] for e in turn)
                if turn and p["archetype"] in ("left", "right"):
                    track = [(t, row) for t, rows in r["frames"] for row in rows if row[0] == p["id"]]
                    row = next(row for t, row in track if t >= turn[0]["t"])
                    lateral.append(abs(row[1]-track[0][1][1]))
        self.assertGreaterEqual(sum(not 10000. <= x <= 25000. for x in lateral)/len(lateral), .3, lateral)
        self.assertGreaterEqual(len(kinds), 3, kinds)

    def test_b_one_pilot_flies_three_evasions(self):
        # In the matches a pilot changes style between its evasions; the draw itself (one pilot's propensities over
        # 40 events) yields at least 3 styles. The match count depends on how many evasions the seeds produce.
        best = max(len({e.get("flown", e["style"]) for e in p["events"] if e["kind"] == "evade"} - {"ignore"})
                   for r in self.results for p in r["planes"])
        self.assertGreaterEqual(best, 2)
        from wt_overlay import match as M
        gen = M.random_match(7, team_size=2, population="v2", model=M.load_model("data/match/top_tier_s1_far.json"))
        pilot = next(s.controller for s in gen.specs if "bait" not in dict(s.controller.p.evade_weights))
        styles = {pilot._draw_event(None, float(i), True, "test")["style"] for i in range(40)}
        self.assertGreaterEqual(len(styles - {"ignore"}), 3, styles)

    def test_c_altitudes_cover_four_bands(self):
        counts = [0]*(len(BANDS_M)-1)
        for r in self.results:
            for _, rows in r["frames"]:
                for row in rows:
                    counts[next(i for i in range(len(counts)) if BANDS_M[i] <= row[3] < BANDS_M[i+1])] += 1
        shares = [c/sum(counts) for c in counts]
        self.assertGreaterEqual(sum(s >= .005 for s in shares), 4, shares)

    def test_d_a_spammer_fires_twice_within_30_s(self):
        spells = []
        for r in self.results:
            for p in r["planes"]:
                if p["archetype"] == "spammer":
                    t = [e["t"] for e in r["log"] if e["kind"] == "launch" and e["shooter"] == p["id"]]
                    spells.append(any(b-a <= 30. for a, b in zip(t, t[1:])))
        self.assertTrue(spells and any(spells))

    def test_e_a_wingman_keeps_station_through_the_climb(self):
        checked = 0
        for r in self.results:
            for p in r["planes"]:
                if p["archetype"] != "pair" or p["lead"] is None:
                    continue
                turned = [e["t"] for e in r["log"] if e["kind"] == "phase" and e["plane"] == p["lead"] and e["frm"] == "climb"]
                end = turned[0] if turned else math.inf
                d = []
                for t, rows in r["frames"]:
                    if t > end:
                        break
                    a = next(x for x in rows if x[0] == p["id"])
                    b = next(x for x in rows if x[0] == p["lead"])
                    d.append(math.hypot(a[1]-b[1], a[2]-b[2]))
                self.assertGreater(len(d), 100)
                self.assertGreaterEqual(sum(2900. <= x <= 8100. for x in d)/len(d), .6, d[::20])
                checked += 1
        self.assertEqual(checked, 2)

    def test_f_skill_sets_the_reaction_delay(self):
        by = {"normal": [], "top": []}
        for r in self.results:
            for p in r["planes"]:
                self.assertAlmostEqual(p["personal_median"], p["delay_median"], places=12)
                by[p["skill"]].extend(p["delays"])
        med = {k: statistics.median(v) for k, v in by.items()}
        self.assertLess(med["top"]+.3, med["normal"], med)
        self.assertTrue(.8 <= med["top"] <= 1.6 and 1.4 <= med["normal"] <= 2.6, med)

    def test_g_the_header_carries_every_draw(self):
        names = {f.name for f in fields(archetypes.PilotParams)}
        for r in self.results:
            for p, h in zip(r["planes"], r["header"]["planes"]):
                script = h["script"]
                self.assertTrue(names <= set(script), names-set(script))
                for k in names:
                    self.assertEqual(script[k], p["params"][k], k)
                if p["archetype"] == "pair":
                    self.assertEqual(script["pair_lead"], p["lead"])

    def test_events_record_every_evasion_draw(self):
        for r in self.results:
            for p in r["planes"]:
                for e in p["events"]:
                    if e["kind"] == "evade" and e["style"] != "ignore":
                        self.assertIn(e["chaff"], archetypes.CHAFF_RHYTHMS)
                        self.assertIn(e["recommit"], archetypes.RECOMMIT_MODES)
                        self.assertTrue(e["style"] in archetypes.STYLE_PLANS)
                    elif e["kind"] == "attack":
                        self.assertIn(e["support"], archetypes.SUPPORT_MODES)


if __name__ == "__main__":
    unittest.main()
