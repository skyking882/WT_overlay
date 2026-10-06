"""Seeker re-acquisition bench for missile_sim (acceptance tests of docs/seeker_search_spec.md).

Two experiments with a kinematic constant-speed target (1000 km/h, 7 g turn limit) and the WT_overlay engagement's
launch options (look-down clutter 2 deg, cw on clear beam, fuse only after a seeker lock):

  weave  launcher 8 km / 1300 km/h, target hot or oblique at 8 km or 3 km, 10-25 km; the target starts an evasion
         12 s or 6 s before the unevaded impact: drag, steady beam, horizontal weaves, vertical weaves.
  blind  the target flies straight and level; for BLIND_S seconds the seeker cannot see it (RCS ~0: needs an
         RCS-aware seeker, i.e. --sim-options '{"seeker_search": {}}'); then it keeps its line or turns / climbs /
         dives: separates "lock broken, kept flying straight" from "lock broken, changed line" (user report: the
         first gets fused on INS, GNSS missiles most of all; the second escapes)
  notch  look-down shot (target 2-3.5 km); the target beams for 9 s (the seeker loses it in the clutter notch), then
         resumes its old heading ('back'), keeps beaming ('hold'), turns 45 deg, climbs 20 deg or dives 15 deg;
         counts lock losses, re-locks and hits.

    .venv/bin/python scripts/bench_seeker_reacquire.py notch            (run with the missile_sim venv)
    .venv/bin/python scripts/bench_seeker_reacquire.py all --sim-options '{"seeker_search": {...}}'

--sim-options is merged into the simulate() keyword options, so the same bench compares the current seeker with the
opt-in search model. Baseline results (2026-10-06, current seeker) are in docs/seeker_search_spec.md.
MISSILE_SIM overrides the missile_sim checkout (default: the sibling ../missle_sim).
"""
import argparse, bisect, collections, json, math, os, sys
from pathlib import Path
MS = Path(os.environ.get("MISSILE_SIM", Path(__file__).resolve().parents[2] / "missle_sim"))
sys.path.insert(0, str(MS / "src"))
from aim120_model.profile_catalog import load_profile_catalog
from aim120_model.public_api import simulate, create_surface_missile, validate_scenario, _case_from_scenario
from aim120_model.surface_runtime import SurfaceRuntime, simulate_surface
from aim120_model.target import TargetState
from missile_gui.library import scan_library, public_profile

_, catalog = load_profile_catalog(MS)
profiles, _ = scan_library(catalog["profiles_dir"], MS)
PROFILES = {p["missile_id"]: p for p in profiles}
OPTS = dict(clutter_model="look_down_angle", clutter_min_depression_deg=2., cw_on_clear_beam=True, require_seeker_lock=True)
SUB = 1/192
LOCK_STATS = collections.defaultdict(list)


def record_locks(missile, experiment, mode, result):
    """Compare sustained lock losses, retaining the profile prolongation window."""
    summary = result['summary']
    if 'lock_loss_count' in summary:
        stats = (summary['first_lock_time_s'], summary['lock_loss_count'], summary['reacquire_delays_s'])
    else:
        profile = PROFILES[missile]
        radar = ((profile['guidance'].get('sensor_model') or {}).get('radar_seeker') or {})
        hold = radar.get('prolongation_time_max_s')
        hold = 1. if hold is None else hold
        first = last = loss = None
        count, delays = 0, []
        for row in result['samples']:
            t = row['time_s']
            if row['track_mode'] == 'radar_track':
                if first is None: first = t
                if loss is not None: delays.append(t-loss); loss = None
                last = t
            elif last is not None and loss is None and t-last > hold:
                count += 1; loss = t
        stats = (first, count, delays)
    LOCK_STATS[(missile, experiment, mode)].append(stats)


class Weaver:
    """After start_s: heading = (away from the missile) + offset(t); offset per mode, turn limited to max_g."""
    def __init__(self, base, mode, start_s, max_g=7., period_s=6., amp_deg=60.):
        self.base, self.mode, self.start = base, mode, start_s
        self.max_g, self.period, self.amp = max_g, period_s, amp_deg
        self.missile = None
        self.times, self.states = [start_s], [base.state_at(start_s)]
        self.side = None

    def observe_missile(self, t, position, velocity):
        self.missile = tuple(position)

    def describe(self):
        return dict(mode=self.mode, start_s=self.start)

    def offset(self, t, p, v):
        k = int((t - self.start) // (self.period / 2)) % 2      # flips every half period
        if self.mode == "drag":
            return 0.
        if self.mode == "beam":
            return 90. * self.side
        if self.mode == "weave_drag":        # +-amp around the drag heading
            return self.amp * (1 if k == 0 else -1)
        if self.mode == "weave_beam":        # +-30 deg around one beam side
            return 90. * self.side + 30. * (1 if k == 0 else -1)
        if self.mode == "beam_flip":         # beam, switching sides every half period (through the tail)
            return 90. * self.side * (1 if k == 0 else -1)
        if self.mode in ("vweave_drag", "vweave_hot"):
            return 0.
        if self.mode == "vweave_beam":
            return 90. * self.side
        raise ValueError(self.mode)

    def gamma(self, t):
        """Desired flight-path angle (deg): vertical weave modes alternate +-25 deg every half period."""
        if not self.mode.startswith("vweave"):
            return None
        k = int((t - self.start) // (self.period / 2)) % 2
        return 25. if k == 0 else -25.

    def state_at(self, t):
        if t <= self.start:
            return self.base.state_at(t)
        while self.times[-1] < t - 1e-12:
            self.step()
        i = bisect.bisect_left(self.times, t - 1e-12)
        if abs(self.times[i] - t) <= 1e-12:
            return self.states[i]
        t0, t1 = self.times[i-1], self.times[i]; s0, s1 = self.states[i-1], self.states[i]
        w = (t - t0) / (t1 - t0)
        return TargetState(tuple(a + w*(b-a) for a, b in zip(s0.position, s1.position)),
                           tuple(a + w*(b-a) for a, b in zip(s0.velocity, s1.velocity)))

    def step(self):
        s = self.states[-1]; p, v = s.position, s.velocity
        t = self.times[-1]
        new_v = v
        speed = math.hypot(v[0], v[2])
        if self.missile is not None and speed > 1e-9:
            los = (p[0] - self.missile[0], p[2] - self.missile[2])
            away = math.atan2(los[1], los[0])
            if self.side is None:
                cross = los[0]*v[2] - los[1]*v[0]
                self.side = 1. if cross >= 0 else -1.
            desired = away + math.radians(self.offset(t, p, v)) if self.mode != "vweave_hot" else math.atan2(v[2], v[0])
            err = (desired - math.atan2(v[2], v[0]) + math.pi) % (2*math.pi) - math.pi
            lim = self.max_g * 9.80665 / speed * SUB
            turn = max(-lim, min(lim, err))
            c, sn = math.cos(turn), math.sin(turn)
            new_v = (c*v[0] - sn*v[2], v[1], sn*v[0] + c*v[2])
            g = self.gamma(t)
            if g is not None:                # pitch the velocity toward the desired path angle, same g budget
                full = math.sqrt(sum(x*x for x in new_v)); hor = math.hypot(new_v[0], new_v[2])
                cur = math.atan2(new_v[1], hor)
                e2 = math.radians(g) - cur
                l2 = self.max_g * 9.80665 / full * SUB
                cur += max(-l2, min(l2, e2))
                if p[1] < 300. and cur < 0: cur = 0.          # do not fly into the ground
                h2 = full*math.cos(cur); f = h2/max(hor, 1e-9)
                new_v = (new_v[0]*f, full*math.sin(cur), new_v[2]*f)
        mid = tuple((a+b)/2 for a, b in zip(v, new_v))
        self.times.append(t + SUB)
        self.states.append(TargetState(tuple(a + b*SUB for a, b in zip(p, mid)), new_v))


def scenario(range_m, tgt_alt, aspect_course):
    return dict(launch_speed_kmh=1300., launch_altitude_m=8000., launch_pitch_deg=0, launch_heading_deg=0,
                target_speed_kmh=1000., target_altitude_m=tgt_alt, initial_distance_m=range_m, target_azimuth_deg=0.,
                target_heading_deg=aspect_course, target_course_reference="relative_to_los",
                target_vertical_heading_deg=0, target_constant_turn_g=0., max_simulation_time_s=150.,
                observation_mode="sensor_track", loft_enabled=True)


def run_weave(missile, sc, mode, start_s):
    fac = None if mode == "none" else (lambda base: Weaver(base, mode, start_s))
    r = simulate(PROFILES[missile], sc, target_factory=fac, early_miss_s=2., **OPTS)
    record_locks(missile, 'weave', mode, r)
    s = r["summary"]
    return s["termination_event"] in ("proximity_fuse", "hit"), s.get("last_radar_reject_reason"), s["flight_time_s"]




NOTCH_S = 9.


class NotchEvader:
    def __init__(self, base, mode, start_s, max_g=7.):
        self.base, self.mode, self.start, self.max_g = base, mode, start_s, max_g
        self.missile = None; self.side = None
        s0 = base.state_at(start_s)
        self.h0 = math.atan2(s0.velocity[2], s0.velocity[0])          # original heading (x/z plane)
        self.times, self.states = [start_s], [s0]

    def observe_missile(self, t, position, velocity):
        self.missile = tuple(position)

    def describe(self):
        return dict(mode=self.mode)

    def state_at(self, t):
        if t <= self.start:
            return self.base.state_at(t)
        while self.times[-1] < t - 1e-12:
            self.step()
        i = bisect.bisect_left(self.times, t - 1e-12)
        if abs(self.times[i] - t) <= 1e-12:
            return self.states[i]
        t0, t1 = self.times[i-1], self.times[i]; s0, s1 = self.states[i-1], self.states[i]
        w = (t - t0) / (t1 - t0)
        return TargetState(tuple(a + w*(b-a) for a, b in zip(s0.position, s1.position)),
                           tuple(a + w*(b-a) for a, b in zip(s0.velocity, s1.velocity)))

    def targets(self, t, p, v):
        """(desired heading rad, desired flight-path angle deg)."""
        los = (p[0] - self.missile[0], p[2] - self.missile[2])
        away = math.atan2(los[1], los[0])
        if self.side is None:
            self.side = 1. if los[0]*v[2] - los[1]*v[0] >= 0 else -1.
        if t < self.start + NOTCH_S or self.mode == "hold":
            return away + self.side*math.pi/2, 0.
        if self.mode == "back":
            return self.h0, 0.
        if self.mode == "turn45":
            return self.h0 - self.side*math.radians(45.), 0.
        if self.mode == "climb":
            return self.h0, 20.
        if self.mode == "dive":
            return self.h0, -15.
        raise ValueError(self.mode)

    def step(self):
        s = self.states[-1]; p, v = s.position, s.velocity; t = self.times[-1]
        new_v = v
        if self.missile is not None:
            hd, gd = self.targets(t, p, v)
            speed = math.hypot(v[0], v[2])
            err = (hd - math.atan2(v[2], v[0]) + math.pi) % (2*math.pi) - math.pi
            lim = self.max_g*9.80665/max(speed, 1.)*SUB
            turn = max(-lim, min(lim, err)); c, sn = math.cos(turn), math.sin(turn)
            new_v = (c*v[0] - sn*v[2], v[1], sn*v[0] + c*v[2])
            full = math.sqrt(sum(x*x for x in new_v)); hor = math.hypot(new_v[0], new_v[2])
            cur = math.atan2(new_v[1], hor)
            cur += max(-self.max_g*9.80665/full*SUB, min(self.max_g*9.80665/full*SUB, math.radians(gd) - cur))
            if p[1] < 300. and cur < 0: cur = 0.
            f = full*math.cos(cur)/max(hor, 1e-9)
            new_v = (new_v[0]*f, full*math.sin(cur), new_v[2]*f)
        mid = tuple((a+b)/2 for a, b in zip(v, new_v))
        self.times.append(t + SUB); self.states.append(TargetState(tuple(a + b*SUB for a, b in zip(p, mid)), new_v))


def run_notch(missile, sc, mode, start):
    fac = None if mode is None else (lambda base: NotchEvader(base, mode, start))
    r = simulate(PROFILES[missile], sc, target_factory=fac, early_miss_s=2., **OPTS)
    record_locks(missile, 'notch', mode or 'none', r)
    modes = [x["track_mode"] for x in r["samples"]]
    first = next((i for i, m in enumerate(modes) if m == "radar_track"), None)
    lost = relock = False
    if first is not None:
        for m in modes[first:]:
            if m != "radar_track": lost = True
            elif lost: relock = True; break
    hit = r["summary"]["termination_event"] in ("proximity_fuse", "hit")
    return hit, lost, relock, r["summary"]["flight_time_s"]


def coverage(missiles):
    """Bounded three-entry coverage with actual acquisition for every active radar."""
    defaults = json.loads((MS/'config/effective_surface_defaults.json').read_text())
    counts = collections.Counter()
    for missile in missiles:
        p = PROFILES[missile]
        sc = scenario(2000., 8000., 0.)
        sc.update(max_simulation_time_s=.95, loft_enabled=False)
        options = dict(OPTS)
        # Existing radar-only options are inapplicable to the kinematic provider.
        if not p['guidance'].get('sensor_model'):
            for key in ('clutter_model', 'clutter_min_depression_deg', 'cw_on_clear_beam', 'require_seeker_lock'):
                options.pop(key, None)
        norm = validate_scenario(sc)
        reference = SurfaceRuntime(p, norm, _case_from_scenario(norm), defaults)
        family = 'active' if getattr(getattr(reference.provider, 'radar', None), 'active', False) else 'not_applicable'
        public = simulate(p, sc, **options)
        direct = simulate_surface(p, sc, **options)
        assert public == direct, missile
        stepping = create_surface_missile(p, launch_position_m=reference.creation['position_xyz_m'],
            launch_velocity_mps=reference.creation['velocity_xyz_mps'], launch_pitch_deg=0., launch_heading_deg=0.,
            target=reference.target, loft=False, end_time_s=.95, **options)
        while not stepping.done: stepping.step()
        assert public['samples'] == stepping.result()['samples'], missile
        if options.get('seeker_search') is not None and family == 'active':
            assert public['summary']['first_lock_time_s'] is not None, missile
        counts[family] += 1
    print('coverage: all three entries; active radar acquisition:', dict(counts))


def print_lock_stats():
    for (missile, experiment, mode), stats in sorted(LOCK_STATS.items()):
        delays = [delay for _, _, values in stats for delay in values]
        mean = sum(delays)/len(delays) if delays else None
        print('locks: %-15s %-5s %-11s runs=%d first_locks=%d losses=%d reacquires=%d mean_delay_s=%s' %
              (missile, experiment, mode, len(stats), sum(first is not None for first, _, _ in stats),
               sum(losses for _, losses, _ in stats), len(delays), 'n/a' if mean is None else '%.3f' % mean))


class BlindEvader(NotchEvader):
    """Straight and level until start_s; from then the seeker cannot see it for BLIND_S seconds (RCS ~0, only with an
    RCS-aware seeker such as seeker_search) and it flies on per mode: 'straight' keeps its line (the INS estimate stays
    right), 'turn45' / 'beam' turn 45 / 90 deg, 'climb' / 'dive' pitch +20 / -15 deg."""
    def __init__(self, base, mode, start_s, max_g=7., rcs_m2=5.):
        super().__init__(base, mode, start_s, max_g)
        self.rcs_normal, self.last_t = rcs_m2, 0.

    @property
    def rcs_m2(self):
        return 1e-6 if self.start <= self.last_t < self.start + BLIND_S else self.rcs_normal

    def state_at(self, t):
        self.last_t = max(self.last_t, t)
        return super().state_at(t)

    def targets(self, t, p, v):
        if self.side is None:
            los = (p[0] - self.missile[0], p[2] - self.missile[2])
            self.side = 1. if los[0]*v[2] - los[1]*v[0] >= 0 else -1.
        return {"straight": (self.h0, 0.), "turn45": (self.h0 - self.side*math.radians(45.), 0.),
                "beam": (self.h0 - self.side*math.pi/2, 0.), "climb": (self.h0, 20.), "dive": (self.h0, -15.)}[self.mode]


BLIND_S = 4.
BLIND_MODES = ("straight", "turn45", "beam", "climb", "dive")


def run_blind(missile, sc, mode, start):
    fac = None if mode is None else (lambda base: BlindEvader(base, mode, start))
    r = simulate(PROFILES[missile], sc, target_factory=fac, early_miss_s=2., **OPTS)
    modes = [x["track_mode"] for x in r["samples"]]
    times = [x["time_s"] for x in r["samples"]]
    first = next((i for i, m in enumerate(modes) if m == "radar_track"), None)
    lost_at = relock_at = None
    if first is not None:
        for i in range(first, len(modes)):
            if modes[i] != "radar_track" and lost_at is None:
                lost_at = times[i]
            elif modes[i] == "radar_track" and lost_at is not None:
                relock_at = times[i]; break
    hit = r["summary"]["termination_event"] in ("proximity_fuse", "hit")
    lock_t = times[first] if first is not None else None
    return hit, lost_at, relock_at, lock_t, r["summary"]["flight_time_s"]


def blind_experiment(missiles):
    agg = collections.defaultdict(lambda: [0, 0, 0, 0, 0.])
    for missile in missiles:
        for rng, alt in ((12000., 6000.), (16000., 6000.), (20000., 6000.), (14000., 3000.), (18000., 3000.)):
            sc = scenario(rng, alt, 180.)
            hit0, _, _, lock_t, tof = run_blind(missile, sc, None, 0.)
            if not hit0 or lock_t is None:
                continue
            for lead in (10., 6.):
                start = max(lock_t + 1., tof - lead)
                for mode in BLIND_MODES:
                    hit, lost_at, relock_at, _, _ = run_blind(missile, sc, mode, start)
                    a = agg[(missile, mode)]; a[0] += 1; a[1] += lost_at is not None; a[2] += relock_at is not None
                    a[3] += hit; a[4] += (relock_at - start) if relock_at is not None else 0.
    print("blind (%.0f s seeker blackout while flying straight, then):" % BLIND_S)
    print("       %-15s %-9s %4s %10s %10s %6s %16s" % ("missile", "mode", "runs", "lock lost", "re-locked", "hit", "relock after (s)"))
    for (missile, mode), (n, lost, relock, hit, delay) in sorted(agg.items()):
        print("       %-15s %-9s %4d %10d %10d %6d %16s" % (missile, mode, n, lost, relock, hit,
                                                          "%.1f" % (delay/relock) if relock else "-"))


WEAVE_MODES = ("none", "drag", "beam", "weave_drag", "weave_beam", "beam_flip", "vweave_hot", "vweave_drag",
               "vweave_beam")
NOTCH_MODES = ("hold", "back", "turn45", "climb", "dive")


def weave_experiment(missiles):
    res = collections.defaultdict(list); reasons = collections.defaultdict(collections.Counter)
    for missile in missiles:
        for tgt_alt in (8000., 3000.):
            for course in (180., 135.):
                for rng in (10000., 15000., 20000., 25000.):
                    sc = scenario(rng, tgt_alt, course)
                    hit0, _, tof = run_weave(missile, sc, "none", 0.)
                    for lead in (12., 6.):
                        start = max(0., tof - lead)
                        for mode in WEAVE_MODES:
                            hit, why, _ = (hit0, None, tof) if mode == "none" else run_weave(missile, sc, mode, start)
                            res[(lead, mode)].append(hit)
                            if not hit and why:
                                reasons[(lead, mode)][why] += 1
    for lead in (12., 6.):
        print("weave: evasion starts %.0f s before the unevaded impact" % lead)
        for mode in WEAVE_MODES:
            h = res[(lead, mode)]
            print("  %-11s hit %2d/%-2d (%3.0f%%)   last seeker reject when missed: %s" % (
                mode, sum(h), len(h), 100*sum(h)/len(h), dict(reasons[(lead, mode)])))


def notch_experiment(missiles):
    agg = collections.defaultdict(lambda: [0, 0, 0, 0])
    for missile in missiles:
        for rng, alt in ((10000., 2000.), (13000., 2000.), (16000., 2000.), (19000., 2000.), (13000., 3500.),
                         (16000., 3500.)):
            sc = scenario(rng, alt, 180.)
            hit0, _, _, tof = run_notch(missile, sc, None, 0.)
            if not hit0:
                continue
            for lead in (22., 16.):
                for mode in NOTCH_MODES:
                    hit, lost, relock, _ = run_notch(missile, sc, mode, max(0., tof - lead))
                    a = agg[(missile, mode)]; a[0] += 1; a[1] += lost; a[2] += relock; a[3] += hit
    print("notch: %-15s %-7s %4s %10s %10s %6s" % ("missile", "after", "runs", "lock lost", "re-locked", "hit"))
    for (missile, mode), (n, lost, relock, hit) in sorted(agg.items()):
        print("       %-15s %-7s %4d %10d %10d %6d" % (missile, mode, n, lost, relock, hit))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("experiment", choices=("weave", "notch", "blind", "all", "coverage"))
    ap.add_argument("--sim-options", default="{}", help="JSON merged into the simulate() options")
    ap.add_argument("--missiles", default="us_aim_120c_5,us_aim_120d,su_r_77_1", help="comma-separated profile IDs or all runnable profiles")
    a = ap.parse_args()
    OPTS.update(json.loads(a.sim_options))
    missiles = ([p['missile_id'] for p in profiles if public_profile(p)['runnable']]
                if a.missiles == 'all' else a.missiles.split(','))
    if a.experiment == 'coverage':
        coverage(missiles)
    if a.experiment in ("weave", "all"):
        weave_experiment(missiles)
    if a.experiment in ("notch", "all"):
        notch_experiment(missiles)
    if a.experiment in ("blind", "all"):
        blind_experiment(missiles)
    print_lock_stats()
