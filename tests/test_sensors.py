"""Radar and RWR sensor model on synthetic geometry, plus the shipped F-16C radar."""
from dataclasses import replace
import math
import random
import unittest

from wt_overlay import units
from wt_overlay.sensors import (Emission, OwnState, RadarSensor, RwrSensor, TargetTruth, relative_angles, world_angles)

DT = 1/48


def pattern(width=25., bars=2, height=3.2, period=1.5, az=(-60., 60.), el=(-60., 60.), **kw):
    return units.ScanPattern("p", az, el, width, bars, height, period, **kw)


def waveform(**kw):
    base = dict(transceiver="mprf", signal="mprfSearch", range_m=62000., reference_rcs_m2=3., doppler_min_mps=-2500.,
                doppler_max_mps=2500., main_beam_notch_mps=60., ground_clutter=False, distance_max_m=74000., band=(8,),
                beam_azimuth_deg=3.3, beam_elevation_deg=4.5, range_max_m=150000., distance_min_m=500., range_finder=True)
    return units.Waveform(**{**base, **kw})


def radar(electronic=False, wf=None, timeout=8., fast_timeout=4., limit=10, tws_pattern=None, stab=(70., 70.)):
    wf = wf or waveform()
    fast = pattern(width=60., bars=24, height=5., period=.0416, roll_stab_limit_deg=stab[0], pitch_stab_limit_deg=stab[1]) \
        if electronic else None
    mid = tws_pattern or pattern(roll_stab_limit_deg=stab[0], pitch_stab_limit_deg=stab[1])
    tws = units.Tws((mid,), (wf,), timeout, limit, 2., fast, fast_timeout if electronic else None,
                    units.Gate((0., 300.), 2., 1500., (2., 4.)))
    return units.Radar("test", "Test", (pattern(width=60., period=3.6, roll_stab_limit_deg=stab[0],
                                                pitch_stab_limit_deg=stab[1]),), (wf,), tws, 3.)


def own(position=(0., 0., 3000.), velocity=(0., 0., 0.), heading=0., pitch=0., roll=0.):
    return OwnState(position, velocity, heading, pitch, roll)


def bandit(ident="b", position=(0., 30000., 3000.), velocity=(0., 0., 0.), rcs=None):
    return TargetTruth(ident, position, velocity, rcs)


def moving(ident, p0, v, t, rcs=None):
    return TargetTruth(ident, tuple(p+x*t for p, x in zip(p0, v)), v, rcs)


def at_azimuth(az_deg, range_m=30000., z=3000.):
    return (range_m*math.sin(math.radians(az_deg)), range_m*math.cos(math.radians(az_deg)), z)


def run(sensor, seconds, own_at, targets_at, t0=0.):
    """Step ``sensor`` from t0 for ``seconds`` at 48 Hz; own_at(t) -> OwnState, targets_at(t) -> list."""
    pictures, k0 = [], round(t0/DT)
    for k in range(k0+1, k0+round(seconds/DT)+1):
        t = k*DT
        pictures.append(sensor.update(t, DT, own_at(t), targets_at(t)))
    return pictures


def scan_hits(pictures):
    return [h for p in pictures for h in p.hits if h.kind == "scan"]


def tracking(mode="tws", electronic=False, **kw):
    sensor = RadarSensor(radar(electronic, **{k: v for k, v in kw.items() if k in ("wf", "timeout", "limit", "tws_pattern",
                                                                                  "fast_timeout", "stab")}),
                         owner="me")
    sensor.set_mode(mode, 0, kw.get("az", 0.), kw.get("el", 0.), t=0.)
    return sensor


class GeometryTests(unittest.TestCase):
    def test_axes_and_angles(self):
        east = own(heading=90.)
        rng, az, el = relative_angles(own((0., 0., 0.)), (1000., 1000., 1000.))
        self.assertAlmostEqual(az, 45.)
        self.assertAlmostEqual(el, math.degrees(math.atan(1000/math.hypot(1000, 1000))))
        self.assertAlmostEqual(rng, 1000*math.sqrt(3))
        self.assertAlmostEqual(relative_angles(east, (0., 1000., 3000.))[1], -90.)   # north is to the left of an eastbound jet
        self.assertAlmostEqual(relative_angles(own(pitch=30.), (0., 1000., 3000.+1000*math.tan(math.radians(30))))[2], 0.)
        bank = own((0., 0., 0.), roll=90.)   # right wing down: the right-hand body axis points at the ground
        self.assertAlmostEqual(relative_angles(bank, (0., 0., -1000.))[1], 90.)
        self.assertAlmostEqual(relative_angles(bank, (0., 0., -1000.))[2], 0.)
        self.assertAlmostEqual(relative_angles(bank, (1000., 0., 0.))[2], 90.)   # east is over the canopy
        self.assertEqual(world_angles((0., 100., 0.)), (0., 0.))
        self.assertAlmostEqual(world_angles((100., 0., 100.))[0], 90.)
        self.assertAlmostEqual(world_angles((100., 0., 100.))[1], 45.)


class ScanTests(unittest.TestCase):
    def test_target_in_the_volume_is_detected_about_once_per_frame(self):
        sensor = tracking()
        pics = run(sensor, 15., lambda t: own(), lambda t: [bandit(position=at_azimuth(10.))])
        hits = scan_hits(pics)
        self.assertTrue(9 <= len(hits) <= 11, len(hits))
        gaps = [b.updated_s-a.updated_s for a, b in zip(hits, hits[1:])]
        self.assertTrue(all(1.0 < g < 2.0 for g in gaps), gaps)
        self.assertAlmostEqual(hits[0].range_m, 30000., delta=1.)
        self.assertAlmostEqual(hits[0].azimuth_deg, 10., delta=3.3)

    def test_targets_outside_the_volume_are_never_detected(self):
        for position in (at_azimuth(40.), at_azimuth(-40.), (0., 30000., 3000.+30000*math.tan(math.radians(20.))),
                         (0., 30000., 3000.-30000*math.tan(math.radians(20.)))):
            pics = run(tracking(), 15., lambda t: own(), lambda t: [bandit(position=position)])
            self.assertEqual(scan_hits(pics), [], position)

    def test_beam_width_extends_the_volume_by_the_beam_half_width(self):
        # Pattern half width 25 deg + 3.3 deg beam: 27 deg is illuminated, 30 deg is not.
        self.assertTrue(scan_hits(run(tracking(), 4., lambda t: own(), lambda t: [bandit(position=at_azimuth(27.))])))
        self.assertFalse(scan_hits(run(tracking(), 4., lambda t: own(), lambda t: [bandit(position=at_azimuth(30.))])))

    def test_scan_centre_is_clamped_to_the_limits(self):
        sensor = tracking(az=90.)   # limits +-60, half width 25: the centre is pulled in to 35
        self.assertTrue(scan_hits(run(sensor, 4., lambda t: own(), lambda t: [bandit(position=at_azimuth(55.))])))
        self.assertFalse(scan_hits(run(tracking(az=90.), 4., lambda t: own(), lambda t: [bandit(position=at_azimuth(5.))])))

    def test_scan_centre_elevation_follows_the_player_and_the_horizon(self):
        high = (0., 30000., 3000.+30000*math.tan(math.radians(20.)))
        self.assertFalse(scan_hits(run(tracking(), 4., lambda t: own(), lambda t: [bandit(position=high)])))
        self.assertTrue(scan_hits(run(tracking(el=20.), 4., lambda t: own(), lambda t: [bandit(position=high)])))
        # Nose up 20 deg, scan centred on the horizon: a level target is still found (the scan is stabilised).
        self.assertTrue(scan_hits(run(tracking(), 4., lambda t: own(pitch=20.), lambda t: [bandit()])))

    def test_scan_is_stabilised_in_bank_up_to_the_limit(self):
        position = at_azimuth(20.)
        for roll, seen in ((45., True), (80., False)):
            pics = run(tracking(), 4., lambda t: own(roll=roll), lambda t: [bandit(position=position)])
            self.assertEqual(bool(scan_hits(pics)), seen, roll)

    def test_airframe_limits_gate_the_field_of_regard(self):
        # +-10 deg elevation gimbal limit. Banked 60 deg (the scan stays level) a target 20 deg off the nose on the
        # horizon is 17 deg below the airframe's elevation limit, so the radar cannot look at it.
        limited = radar(tws_pattern=pattern(el=(-10., 10.)))
        for roll, seen in ((0., True), (60., False)):
            sensor = RadarSensor(limited, owner="me")
            sensor.set_mode("tws", 0, 0., 0., t=0.)
            pics = run(sensor, 4., lambda t: own(roll=roll), lambda t: [bandit(position=at_azimuth(20.))])
            self.assertEqual(bool(scan_hits(pics)), seen, roll)

    def test_beam_covers_is_a_short_dwell_each_frame(self):
        sensor = tracking()
        sensor.update(DT, DT, own(), [])
        p = at_azimuth(10.)
        covered = [sensor.beam_covers(k*DT, own(), p, DT) for k in range(1, 73)]
        self.assertTrue(4 <= sum(covered) <= 20, sum(covered))
        self.assertFalse(any(sensor.beam_covers(k*DT, own(), at_azimuth(40.)) for k in range(1, 73)))
        self.assertEqual(sensor.illuminated(1., own(), ((1, p), (2, at_azimuth(40.))), 1.5), [1])

    def test_electronic_fast_scan_illuminates_the_whole_field_of_regard(self):
        esa = tracking(electronic=True)
        mech = tracking()
        for sensor, expected in ((esa, True), (mech, False)):
            sensor.update(DT, DT, own(), [])
            seen = any(sensor.beam_covers(k*DT, own(), at_azimuth(45.)) for k in range(1, 73))
            self.assertEqual(seen, expected)
        self.assertFalse(any(esa.beam_covers(k*DT, own(), at_azimuth(75.)) for k in range(1, 73)))


class DetectabilityTests(unittest.TestCase):
    def seen(self, distance, rcs, sensor=None, **kw):
        sensor = sensor or tracking(**kw)
        return bool(scan_hits(run(sensor, 3., lambda t: own(), lambda t: [bandit(position=(0., distance, 3000.), rcs=rcs)])))

    def test_range_scales_with_the_fourth_root_of_rcs(self):
        self.assertTrue(self.seen(60000., 3.))
        self.assertFalse(self.seen(64000., 3.))
        self.assertTrue(self.seen(68000., 5.))      # 62 km * (5/3)^(1/4) = 70.4 km
        self.assertFalse(self.seen(72000., 5.))
        self.assertTrue(self.seen(40000., 1.))      # 62 km * (1/3)^(1/4) = 47.2 km
        self.assertFalse(self.seen(50000., 1.))
        self.assertFalse(self.seen(75000., 400.))   # capped by the signal's distance limit (74 km)

    def test_default_rcs_applies_when_the_target_has_none(self):
        self.assertTrue(self.seen(68000., None))
        self.assertFalse(self.seen(72000., None))

    def test_minimum_range(self):
        self.assertFalse(self.seen(300., 5.))

    def test_surface_search_waveforms_see_no_aircraft(self):
        self.assertFalse(self.seen(20000., 5., wf=waveform(air_target=False)))

    def test_probability_ramp_thins_detections_near_the_limit(self):
        rng = random.Random(3)
        sensor = RadarSensor(radar(), rng=rng, owner="me", ramp_fraction=.5)
        sensor.set_mode("tws", 0, 0., 0., t=0.)
        far = (0., 69000., 3000.)   # 98 % of the 70.4 km limit: nearly always missed
        near = (0., 40000., 3000.)
        hits_far = hits_near = 0
        for _ in range(3):
            hits_far += len(scan_hits(run(sensor, 30., lambda t: own(), lambda t: [bandit("far", far, rcs=5.)])))
            hits_near += len(scan_hits(run(sensor, 30., lambda t: own(), lambda t: [bandit("near", near, rcs=5.)])))
            sensor.set_mode("search", t=0.)
            sensor.set_mode("tws", 0, 0., 0., t=0.)
        self.assertGreater(hits_near, 50)
        self.assertLess(hits_far, hits_near/2)

    def test_notch_hides_a_beaming_target_in_look_down_only(self):
        beam = (250., 0., 0.)   # 4 deg below / above the horizon at 30 km is 2.1 km
        for z, seen in ((8000.-2100., False), (8000., True), (8000.+2100., True)):
            pics = run(tracking(), 4., lambda t: own((0., 0., 8000.)), lambda t: [bandit(position=(0., 30000., z), velocity=beam)])
            self.assertEqual(bool(scan_hits(pics)), seen, z)

    def test_notch_does_not_hide_a_closing_target_or_a_fast_enough_radial_speed(self):
        below = (0., 30000., 8000.-2100.)
        for velocity in ((0., -250., 0.), (0., -100., 0.), (0., 100., 0.)):
            pics = run(tracking(), 4., lambda t: own((0., 0., 8000.)), lambda t: [bandit(position=below, velocity=velocity)])
            self.assertTrue(scan_hits(pics), velocity)
        pics = run(tracking(), 4., lambda t: own((0., 0., 8000.)),
                   lambda t: [bandit(position=below, velocity=(0., 20., 0.))])   # radial 20 m/s < 30 m/s half width
        self.assertFalse(scan_hits(pics))

    def test_hprf_doppler_window_rejects_slow_closing_and_receding_targets(self):
        hprf = waveform(transceiver="hprf", signal="hprfSearch", doppler_min_mps=40., doppler_max_mps=1500.,
                        main_beam_notch_mps=None, distance_min_m=5000.)
        for closing, seen in ((10., False), (-100., False), (200., True), (1700., False)):
            sensor = tracking(wf=hprf)
            # Own flies north at 250 m/s toward a target at 250-closing m/s northbound, 30 km ahead.
            pics = run(sensor, 4., lambda t: own(velocity=(0., 250., 0.)),
                       lambda t: [bandit(position=(0., 30000., 3000.), velocity=(0., 250.-closing, 0.))])
            hits = scan_hits(pics)
            self.assertEqual(bool(hits), seen, closing)
            if seen:
                self.assertAlmostEqual(hits[0].closing_speed_mps, closing, delta=1e-6)

    def test_ground_clutter_waveform_loses_low_targets_in_look_down(self):
        lprf = waveform(transceiver="lprf", signal="lprfSearch", doppler_min_mps=None, doppler_max_mps=None,
                        main_beam_notch_mps=None, ground_clutter=True, range_m=85000., reference_rcs_m2=5.)
        low, high = (0., 30000., 1500.), (0., 30000., 3000.)   # beam footprint at 30 km: 30000*sin(4.5 deg) = 2.35 km
        for position, seen in ((low, False), (high, True)):
            sensor = tracking(wf=lprf, el=-6.)
            pics = run(sensor, 4., lambda t: own((0., 0., 5000.)), lambda t: [bandit(position=position, rcs=5.)])
            hits = scan_hits(pics)
            self.assertEqual(bool(hits), seen, position)
            if hits:
                self.assertIsNone(hits[0].closing_speed_mps)   # no Doppler on this waveform
        # Level with the target there is no look-down, hence no clutter.
        pics = run(tracking(wf=lprf), 4., lambda t: own((0., 0., 1500.)), lambda t: [bandit(position=low, rcs=5.)])
        self.assertTrue(scan_hits(pics))

    def test_velocity_search_gives_closing_speed_without_range(self):
        velocity_only = waveform(transceiver="hprfVelocity", signal="hprfVelocitySearch", doppler_min_mps=40.,
                                 doppler_max_mps=1500., main_beam_notch_mps=None, distance_min_m=None, range_finder=False)
        sensor = RadarSensor(radar(wf=velocity_only), owner="me")
        sensor.set_mode("search", 0, 0., 0., t=0.)
        pics = run(sensor, 8., lambda t: own(velocity=(0., 250., 0.)),
                   lambda t: [bandit(position=at_azimuth(10.), velocity=(0., -250., 0.))])
        blip = pics[-1].contacts[0]
        self.assertIsNone(blip.range_m)
        self.assertIsNone(blip.position)
        self.assertAlmostEqual(blip.closing_speed_mps, 500.*math.cos(math.radians(10.)), delta=1e-6)
        self.assertAlmostEqual(blip.bearing_deg, 10., delta=1e-6)

    def test_look_down_threshold_is_configurable(self):
        beam = (250., 0., 0.)
        position = (0., 30000., 3000.-700.)   # 1.3 deg below the horizon: below the 2 deg default, so not look-down
        self.assertTrue(scan_hits(run(tracking(), 4., lambda t: own(), lambda t: [bandit(position=position, velocity=beam)])))
        strict = RadarSensor(radar(), owner="me", look_down_deg=0.)
        strict.set_mode("tws", 0, 0., 0., t=0.)
        self.assertFalse(scan_hits(run(strict, 4., lambda t: own(), lambda t: [bandit(position=position, velocity=beam)])))

    def test_owner_is_skipped(self):
        sensor = tracking()
        pics = run(sensor, 3., lambda t: own(), lambda t: [bandit("me", position=at_azimuth(0., 10000.))])
        self.assertFalse(scan_hits(pics))


class TwsTrackTests(unittest.TestCase):
    def sim(self, sensor, seconds, p0=None, v=(0., -250., 0.), t0=0., **kw):
        p0 = p0 or at_azimuth(10.)
        return run(sensor, seconds, kw.get("own_at", lambda t: own()), lambda t: [moving("b", p0, v, t)], t0=t0)

    def test_new_track_appears_only_after_the_minimum_time_with_two_detections(self):
        sensor = tracking()
        pics = self.sim(sensor, 10.)
        hits = scan_hits(pics)
        first = next(p for p in pics if p.contacts)
        self.assertGreaterEqual(first.time_s-hits[0].updated_s, 2.-1e-9)
        self.assertLess(first.time_s-hits[0].updated_s, 4.)
        self.assertGreaterEqual(sum(1 for h in hits if h.updated_s <= first.time_s), 2)
        self.assertEqual(first.contacts[0].track_id, 1)
        self.assertEqual(first.truth_ids, ("b",))
        # Frames of 1.2 s (narrow TWS): the third detection, 2.4 s after the first, confirms.
        narrow = tracking(tws_pattern=pattern(width=10., bars=4, period=1.2))
        pics = self.sim(narrow, 10., p0=at_azimuth(2.))
        hits = scan_hits(pics)
        first = next(p for p in pics if p.contacts)
        self.assertAlmostEqual(first.time_s-hits[0].updated_s, 2.4, delta=.2)

    def test_track_estimates_position_velocity_and_closing_speed(self):
        sensor = tracking()
        pics = self.sim(sensor, 12.)
        c = pics[-1].contacts[0]
        truth = moving("b", at_azimuth(10.), (0., -250., 0.), 12.)
        for a, b in zip(c.position, truth.position):
            self.assertAlmostEqual(a, b, delta=1.)
        for a, b in zip(c.velocity, (0., -250., 0.)):
            self.assertAlmostEqual(a, b, delta=2.)
        los = [x/math.dist(c.position, (0., 0., 3000.)) for x in (c.position[0], c.position[1], c.position[2]-3000.)]
        self.assertAlmostEqual(c.closing_speed_mps, 250.*los[1]*-1.*-1., delta=2.)
        self.assertAlmostEqual(c.range_m, math.dist(c.position, (0., 0., 3000.)), delta=1e-6)
        self.assertFalse(c.extrapolated)
        self.assertLess(c.age_s, 2.)

    def test_picture_separates_truth_from_what_a_policy_sees(self):
        pic = self.sim(tracking(), 6.)[-1]
        self.assertEqual(pic.truth_ids, ("b",))
        for obj in (*pic.contacts, *pic.hits):
            self.assertFalse(any("truth" in name or name == "id" for name in vars(obj)), vars(obj))
            self.assertNotIn("b", [getattr(obj, name) for name in vars(obj)])

    def test_track_limit_caps_confirmed_tracks(self):
        sensor = tracking(limit=2)
        pics = run(sensor, 10., lambda t: own(), lambda t: [bandit(str(i), at_azimuth(-12.+8.*i, 30000.+3000.*i), (0., 0., 0.))
                                                          for i in range(4)])
        self.assertEqual(len(pics[-1].contacts), 2)

    def leave_volume(self, sensor, own_at=lambda t: own()):
        """Track a target, then slew the scan away at t = 8 s. Returns the pictures before and after the slew."""
        truth = lambda t: [moving("b", at_azimuth(40.), (0., -100., 0.), t)]  # noqa: E731
        before = run(sensor, 8., own_at, truth)
        self.assertTrue(before[-1].contacts)
        sensor.set_mode("tws", 0, -35., 0.)
        return before, run(sensor, 14., own_at, truth, t0=8.)

    def test_mechanical_radar_extrapolates_then_drops_a_target_that_left_the_volume(self):
        sensor = tracking(az=35.)
        before, after = self.leave_volume(sensor)
        self.assertFalse(scan_hits(after))
        last = scan_hits(before)[-1].updated_s
        self.assertTrue(after[0].contacts)
        coasting = [p for p in after if p.contacts and p.contacts[0].extrapolated]
        self.assertTrue(coasting)
        self.assertTrue(all(p.contacts[0].age_s > 1.5*1.5 for p in coasting))
        self.assertTrue(all(not p.contacts[0].extrapolated for p in after if p.contacts and p.contacts[0].age_s < 1.5*1.5))
        gone = next(p for p in after if not p.contacts)
        self.assertAlmostEqual(gone.time_s-last, 8., delta=2*DT+1e-6)   # timeout_s after the last detection

    def test_extrapolation_continues_at_constant_velocity(self):
        sensor = tracking(az=35.)
        _, after = self.leave_volume(sensor)
        a, b = (next(p for p in after if p.contacts and p.contacts[0].extrapolated and p.time_s > t) for t in (11., 13.))
        v = a.contacts[0].velocity
        moved = tuple(y-x for x, y in zip(a.contacts[0].position, b.contacts[0].position))
        for m, vi in zip(moved, v):
            self.assertAlmostEqual(m, vi*(b.time_s-a.time_s), delta=.5)

    def test_electronic_radar_keeps_refreshing_a_track_outside_the_selected_volume(self):
        sensor = tracking(electronic=True, az=35.)
        _, after = self.leave_volume(sensor)
        self.assertTrue(all(p.contacts for p in after))
        self.assertTrue(all(not p.contacts[0].extrapolated and p.contacts[0].age_s <= 2*.0416+DT for p in after))
        self.assertTrue(any(h.kind == "fast" for p in after for h in p.hits))
        self.assertFalse(scan_hits(after))

    def test_electronic_radar_does_not_start_tracks_from_the_fast_scan(self):
        sensor = tracking(electronic=True)   # volume +-25 deg; the target sits at 40 deg, inside the field of regard
        pics = self.sim(sensor, 10., p0=at_azimuth(40.))
        self.assertFalse(any(p.contacts or p.hits for p in pics))

    def test_electronic_radar_drops_a_track_fast_timeout_after_leaving_the_field_of_regard(self):
        sensor = tracking(electronic=True, az=35.)
        # Turn away 110 deg at 8 s: the target at 40 deg is 70 deg off the nose, outside +-60.
        turned = lambda t: own(heading=110. if t > 8. else 0.)  # noqa: E731
        run(sensor, 8., turned, lambda t: [moving("b", at_azimuth(40.), (0., -100., 0.), t)])
        last = sensor.picture(8., turned(8.)).contacts[0].updated_s
        after = run(sensor, 8., turned, lambda t: [moving("b", at_azimuth(40.), (0., -100., 0.), t)], t0=8.)
        refreshed = max(p.contacts[0].updated_s for p in after if p.contacts)
        self.assertLess(refreshed-last, 1.)
        gone = next(p for p in after if not p.contacts)
        self.assertAlmostEqual(gone.time_s-refreshed, 4., delta=2*DT+1e-6)   # fast_timeout_s

    def test_a_target_that_jumps_out_of_the_gate_starts_a_new_track(self):
        sensor = tracking()
        positions = lambda t: at_azimuth(10.) if t < 8. else at_azimuth(10., 30000.+6000.)  # noqa: E731
        pics = run(sensor, 14., lambda t: own(), lambda t: [bandit(position=positions(t))])
        ids = {c.track_id for p in pics for c in p.contacts}
        self.assertEqual(ids, {1, 2})

    def test_unconfirmed_candidates_expire(self):
        sensor = tracking()
        pics = run(sensor, 1.6, lambda t: own(), lambda t: [bandit(position=at_azimuth(10.))])
        self.assertTrue(scan_hits(pics) and not any(p.contacts for p in pics))
        pics = run(sensor, 8., lambda t: own(), lambda t: [], t0=1.6)
        self.assertFalse(sensor._tracks)


class SearchSttTests(unittest.TestCase):
    def test_search_gives_blips_without_ids_or_tracks(self):
        sensor = RadarSensor(radar(), owner="me")
        sensor.set_mode("search", 0, 0., 0., t=0.)
        pics = run(sensor, 12., lambda t: own(), lambda t: [bandit("a", at_azimuth(10.)), bandit("b", at_azimuth(-20.))])
        last = pics[-1]
        self.assertEqual(last.mode, "search")
        self.assertEqual({c.kind for c in last.contacts}, {"blip"})
        self.assertTrue(all(c.track_id is None and c.velocity is None and not c.extrapolated for c in last.contacts))
        self.assertEqual(sorted(last.truth_ids), ["a", "b"])
        self.assertEqual(len(last.contacts), 2)   # a blip is replaced by the next detection of the same target
        self.assertTrue(all(0. <= c.age_s <= 1.5*3.6 for c in last.contacts))
        self.assertTrue(any(p.contacts and p.contacts[0].age_s > 1. for p in pics))

    def test_blips_fade_after_one_and_a_half_frames(self):
        sensor = RadarSensor(radar(), owner="me")
        sensor.set_mode("search", 0, 0., 0., t=0.)
        run(sensor, 6., lambda t: own(), lambda t: [bandit("a", at_azimuth(10.))])
        pics = run(sensor, 8., lambda t: own(), lambda t: [], t0=6.)
        self.assertTrue(pics[0].contacts)
        gone = next(p for p in pics if not p.contacts)
        self.assertLess(gone.time_s-pics[0].contacts[0].updated_s, 1.5*3.6+.2)

    def locked(self, **kw):
        sensor = tracking(**kw)
        run(sensor, 8., lambda t: own(), lambda t: [moving("b", at_azimuth(10.), (0., -100., 0.), t)])
        self.assertEqual(len(sensor.picture(8., own()).contacts), 1)
        sensor.set_mode("stt", stt_track=1)
        return sensor

    def test_stt_tracks_every_tick_while_detectable(self):
        sensor = self.locked()
        pics = run(sensor, 5., lambda t: own(), lambda t: [moving("b", at_azimuth(10.), (0., -100., 0.), t)], t0=8.)
        self.assertTrue(all(p.mode == "stt" and p.stt_state == "tracking" and len(p.contacts) == 1 for p in pics))
        c = pics[-1].contacts[0]
        self.assertEqual((c.kind, c.track_id, c.age_s, c.extrapolated), ("stt", 1, 0., False))
        self.assertEqual(sensor.emission_kind, "stt")
        self.assertTrue(all(len(p.hits) == 1 and p.hits[0].kind == "stt" for p in pics))
        for a, b in zip(c.position, moving("b", at_azimuth(10.), (0., -100., 0.), 13.).position):
            self.assertAlmostEqual(a, b, delta=.5)

    def test_stt_coasts_then_loses_the_target_and_returns_to_the_scan_mode(self):
        sensor = self.locked()
        truth = lambda t: [moving("b", at_azimuth(10.), (0., -100., 0.), t)]  # noqa: E731
        turned = lambda t: own(heading=100. if t > 9. else 0.)                 # noqa: E731
        run(sensor, 1., lambda t: own(), truth, t0=8.)
        pics = run(sensor, 5., turned, truth, t0=9.)
        coasting = [p for p in pics if p.stt_state == "coasting"]
        self.assertTrue(coasting)
        self.assertTrue(all(p.contacts and p.contacts[0].extrapolated == (p.contacts[0].age_s > 1.5*DT) for p in coasting))
        self.assertTrue(coasting[-1].contacts[0].extrapolated)
        lost = next(p for p in pics if "stt_lost" in p.events)
        self.assertAlmostEqual(lost.time_s-9., 3., delta=2*DT+.05)
        self.assertEqual(sensor.mode, "tws")
        self.assertIsNone(sensor.stt_state)
        self.assertEqual(sensor.update(lost.time_s+DT, DT, turned(lost.time_s), truth(lost.time_s+DT)).contacts, ())

    def test_stt_coast_time_comes_from_the_radar_file_unless_overridden(self):
        long_coast = replace(radar(), stt_coast_s=6.)
        self.assertEqual(RadarSensor(long_coast).stt_coast_s, 6.)
        self.assertEqual(RadarSensor(long_coast, stt_coast_s=1.).stt_coast_s, 1.)
        self.assertEqual(RadarSensor(replace(radar(), stt_coast_s=None)).stt_coast_s, 3.)

    def test_stt_beam_follows_the_target(self):
        sensor = self.locked()
        sensor.update(8.+DT, DT, own(), [moving("b", at_azimuth(10.), (0., -100., 0.), 8.+DT)])
        self.assertTrue(sensor.beam_covers(8.1, own(), moving("b", at_azimuth(10.), (0., -100., 0.), 8.1).position))
        self.assertFalse(sensor.beam_covers(8.1, own(), at_azimuth(20.)))

    def test_stt_by_truth_id_acquires_a_target_in_the_beam(self):
        sensor = RadarSensor(radar(), owner="me")
        sensor.set_mode("search", 0, 0., 0., t=0.)
        sensor.set_mode("stt", stt_truth="b", t=0.)
        pics = run(sensor, 1., lambda t: own(), lambda t: [bandit("b", at_azimuth(10.))])
        self.assertEqual(pics[0].stt_state, "tracking")
        with self.assertRaises(ValueError):
            sensor.set_mode("stt")

    def test_mode_errors(self):
        sensor = RadarSensor(radar(), owner="me")
        with self.assertRaises(ValueError):
            sensor.set_mode("scan")
        with self.assertRaises(ValueError):
            sensor.set_mode("tws", 3)
        with self.assertRaises(ValueError):
            sensor.set_mode("stt", stt_track=7)
        no_tws = RadarSensor(replace(radar(), tws=None), owner="me")
        with self.assertRaises(ValueError):
            no_tws.set_mode("tws")
        no_wave = replace(radar(), search_waveforms=(), tws=None)
        with self.assertRaises(ValueError):
            RadarSensor(no_wave)
        sensor.set_mode("tws", 0, 0., 0., t=0.)
        sensor.set_mode("off")
        self.assertIsNone(sensor.emission_kind)
        self.assertEqual(sensor.update(1., 1., own(), [bandit()]).contacts, ())


def rwr(**kw):
    base = dict(id="r", name="R", range_m=70000., sectors=((0., 0., 180., 90., True), (-180., 0., 180., 90., True)),
                detects_tracking=True, detects_launch=False, tracks_targets=True, targets_max=12, signal_hold_s=5.,
                target_hold_s=5., new_target_hold_s=3., bands=(4, 5, 6, 7, 8, 9), range_finder_m=(5000., 50000.))
    return units.Rwr(**{**base, **kw})


def emitter(ident="e", az=0., el=0., rng=20000., kind="tws", band=8, covers=True, radar_id="radar"):
    a, e = math.radians(az), math.radians(el)
    return Emission(ident, (rng*math.cos(e)*math.sin(a), rng*math.cos(e)*math.cos(a), rng*math.sin(e)), kind, band, covers, radar_id)


MIRAGE = ((0., 0., 120., 90., False), (-110., 0., 100., 90., False), (110., 0., 100., 90., False), (180., 0., 120., 90., False))


class RwrTests(unittest.TestCase):
    def one(self, sensor, e, t=0., o=None):
        return sensor.update(t, o or own((0., 0., 0.)), [e]).contacts

    def test_angle_finder_reports_the_true_bearing_and_others_the_sector_centre(self):
        finder = RwrSensor(rwr())
        c, = self.one(finder, emitter(az=30., el=10.))
        self.assertAlmostEqual(c.azimuth_deg, 30., places=6)
        self.assertAlmostEqual(c.elevation_deg, 10., places=6)
        self.assertAlmostEqual(c.bearing_deg, 30., places=6)
        for az, centre in ((30., 0.), (100., 110.), (-100., -110.), (150., 180.), (-170., 180.)):
            quad = RwrSensor(rwr(sectors=MIRAGE))
            c, = self.one(quad, emitter(az=az))
            self.assertAlmostEqual(wrap(c.azimuth_deg-centre), 0., places=6, msg=az)
            self.assertEqual(c.elevation_deg, 0.)

    def test_reading_follows_the_own_heading(self):
        sensor = RwrSensor(rwr())
        c, = self.one(sensor, emitter(az=0.), o=own((0., 0., 0.), heading=90.))
        self.assertAlmostEqual(c.azimuth_deg, -90., places=6)
        self.assertAlmostEqual(c.bearing_deg, 0., places=6)

    def test_held_contact_keeps_its_last_reading_like_a_scope_symbol(self):
        sensor = RwrSensor(rwr())
        sensor.update(0., own((0., 0., 0.)), [emitter(az=30.)])
        c, = sensor.update(1., own((0., 0., 0.), heading=90.), []).contacts
        self.assertAlmostEqual(c.azimuth_deg, 30., places=6)
        self.assertAlmostEqual(c.bearing_deg, 30., places=6)
        self.assertFalse(c.illuminated)

    def test_elevation_sector_is_a_blind_zone(self):
        sensor = RwrSensor(rwr())
        self.assertEqual(self.one(sensor, emitter(el=60.)), ())     # elevation width 90: +-45 deg
        self.assertEqual(len(self.one(RwrSensor(rwr()), emitter(el=30.))), 1)
        self.assertEqual(len(self.one(RwrSensor(rwr(sectors=((0., 0., 180., 180., True),))), emitter(el=60.))), 1)

    def test_sector_gaps_hear_nothing(self):
        sensor = RwrSensor(rwr(sectors=((0., 0., 100., 90., True),)))
        self.assertEqual(self.one(sensor, emitter(az=90.)), ())
        self.assertEqual(len(self.one(RwrSensor(rwr(sectors=((0., 0., 100., 90., True),))), emitter(az=40.))), 1)

    def test_range_band_and_beam_filters(self):
        sensor = RwrSensor(rwr())
        self.assertEqual(self.one(sensor, emitter(rng=80000.)), ())
        self.assertEqual(self.one(RwrSensor(rwr()), emitter(band=3)), ())
        self.assertEqual(len(self.one(RwrSensor(rwr()), emitter(band=9))), 1)
        self.assertEqual(self.one(RwrSensor(rwr()), emitter(covers=False)), ())
        self.assertEqual(len(self.one(RwrSensor(rwr(bands=(8,))), emitter(band=8))), 1)
        self.assertEqual(self.one(RwrSensor(rwr(bands=(8,))), emitter(band=9)), ())

    def test_new_contacts_are_held_briefly_and_confirmed_ones_longer(self):
        o = own((0., 0., 0.))
        sensor = RwrSensor(rwr())
        sensor.update(0., o, [emitter()])
        self.assertTrue(sensor.update(2.9, o, []).contacts[0].new)
        self.assertEqual(sensor.update(3.1, o, []).contacts, ())   # new_target_hold_s = 3
        sensor = RwrSensor(rwr())
        sensor.update(0., o, [emitter()])
        c, = sensor.update(1., o, [emitter()]).contacts
        self.assertFalse(c.new)
        self.assertTrue(c.illuminated)
        c, = sensor.update(5.9, o, []).contacts
        self.assertFalse(c.illuminated)
        self.assertAlmostEqual(c.age_s, 4.9)
        self.assertEqual(sensor.update(6.1, o, []).contacts, ())   # target_hold_s = 5 after the last illumination
        # Contiguous ticks are one illumination, not many.
        sensor = RwrSensor(rwr())
        for k in range(10):
            pic = sensor.update(k*DT, o, [emitter()])
        self.assertTrue(pic.contacts[0].new)
        self.assertEqual(sensor.update(10*DT+3.1, o, []).contacts, ())

    def test_signal_hold_is_the_fallback_hold(self):
        o = own((0., 0., 0.))
        sensor = RwrSensor(rwr(target_hold_s=None, new_target_hold_s=None, signal_hold_s=2.))
        sensor.update(0., o, [emitter()])
        self.assertTrue(sensor.update(1.9, o, []).contacts)
        self.assertFalse(sensor.update(2.1, o, []).contacts)

    def test_contact_ids_are_stable_while_held_and_new_after_expiry(self):
        o = own((0., 0., 0.))
        sensor = RwrSensor(rwr())
        a = sensor.update(0., o, [emitter("e")]).contacts[0].contact_id
        self.assertEqual(sensor.update(1., o, [emitter("e")]).contacts[0].contact_id, a)
        sensor.update(10., o, [])
        self.assertNotEqual(sensor.update(11., o, [emitter("e")]).contacts[0].contact_id, a)

    def test_stt_gives_a_lock_warning_only_to_rwrs_that_detect_tracking(self):
        c, = self.one(RwrSensor(rwr()), emitter(kind="stt"))
        self.assertTrue(c.tracking)
        self.assertFalse(c.missile_warning)
        c, = self.one(RwrSensor(rwr(detects_tracking=False)), emitter(kind="stt"))
        self.assertFalse(c.tracking)
        for kind in ("search", "tws"):
            self.assertFalse(self.one(RwrSensor(rwr()), emitter(kind=kind))[0].tracking)

    def test_missile_seeker_is_a_missile_warning_and_air_radars_are_not(self):
        c, = self.one(RwrSensor(rwr()), emitter(kind="missile"))
        self.assertTrue(c.missile_warning)
        self.assertFalse(c.tracking)
        for kind in ("search", "tws", "stt"):
            self.assertFalse(self.one(RwrSensor(rwr()), emitter(kind=kind))[0].missile_warning)

    def test_range_finder_reports_a_clamped_range_and_others_none(self):
        c, = self.one(RwrSensor(rwr()), emitter(rng=20000.))
        self.assertAlmostEqual(c.range_m, 20000.)
        self.assertEqual(self.one(RwrSensor(rwr()), emitter(rng=2000.))[0].range_m, 5000.)
        self.assertEqual(self.one(RwrSensor(rwr()), emitter(rng=60000.))[0].range_m, 50000.)
        self.assertIsNone(self.one(RwrSensor(rwr(range_finder_m=None)), emitter())[0].range_m)

    def test_contact_cap_keeps_missiles_then_locks_then_the_nearest(self):
        o = own((0., 0., 0.))
        sensor = RwrSensor(rwr(targets_max=2))
        pic = sensor.update(0., o, [emitter("far", az=10., rng=40000.), emitter("near", az=20., rng=10000.),
                                    emitter("lock", az=30., rng=60000., kind="stt")])
        self.assertEqual(pic.truth_ids, ("lock", "near"))
        pic = sensor.update(.1, o, [emitter("m", az=40., rng=30000., kind="missile")])
        self.assertEqual(pic.truth_ids[0], "m")
        self.assertEqual(len(pic.contacts), 2)

    def test_angle_error_applies_only_with_an_rng(self):
        noisy = RwrSensor(rwr(), rng=random.Random(1), angle_sigma_deg=3.)
        values = {round(noisy.update(float(k), own((0., 0., 0.)), [emitter("e%d" % k, az=30.)]).contacts[-1].azimuth_deg, 3)
                  for k in range(5)}
        self.assertGreater(len(values), 1)
        self.assertTrue(all(abs(v-30.) < 15. for v in values))
        self.assertAlmostEqual(self.one(RwrSensor(rwr()), emitter(az=30.))[0].azimuth_deg, 30., places=6)

    def test_pictures_keep_the_emitter_ids_apart(self):
        pic = RwrSensor(rwr()).update(0., own((0., 0., 0.)), [emitter("secret")])
        self.assertEqual(pic.truth_ids, ("secret",))
        self.assertFalse(any("secret" in str(v) for v in vars(pic.contacts[0]).values()))


class RadarRwrChainTests(unittest.TestCase):
    def test_scan_illumination_is_periodic_and_the_rwr_holds_it_between_sweeps(self):
        radar_sensor = tracking()
        victim = RwrSensor(rwr())
        spot = at_azimuth(10., 30000., 3000.)
        flashes, present, steps = 0, 0, 0
        for k in range(1, 15*48+1):
            t = k*DT
            radar_sensor.update(t, DT, own(), [], report=False)
            e = radar_sensor.emission("tws-radar", t, own(), spot, DT)
            pic = victim.update(t, own(spot, (0., 0., 0.), 180.), [e])
            if pic.contacts:
                present += 1
                flashes += pic.contacts[0].illuminated
            steps += 1
        self.assertEqual(e.kind, "tws")
        self.assertEqual(e.band, 8)
        self.assertEqual(e.radar_id, "test")
        self.assertTrue(0 < flashes < steps/3)
        self.assertGreater(present, steps*.9)   # held through the 1.5 s between illuminations

    def test_stt_lock_reaches_the_target_rwr_as_a_lock_warning(self):
        sensor = tracking()
        spot = (0., 20000., 3000.)
        run(sensor, 8., lambda t: own(), lambda t: [bandit(position=spot)])
        sensor.set_mode("stt", stt_track=1)
        victim = RwrSensor(rwr())
        sensor.update(8.+DT, DT, own(), [bandit(position=spot)])
        c, = victim.update(8.+DT, own(spot, (0., 0., 0.), 180.), [sensor.emission("x", 8.+DT, own(), spot)]).contacts
        self.assertTrue(c.tracking)
        self.assertEqual(c.kind, "stt")


def wrap(a):
    return (a+180.) % 360.-180.


def radar_file():
    """Game-file layout with antenna widths, bands, a missing track limit (updateTargetOfInterest) and timeLimits."""
    pat = lambda w, bars, period: dict(type="pyramide", azimuthLimits=[-60., 60.], elevationLimits=[-60., 60.], width=w,  # noqa: E731
                                       barHeight=3.2, barsCount=bars, period=period, rollStabLimit=70., pitchStabLimit=60.,
                                       centerElevation=-2.)
    transceiver = {"range": 62000., "rcs": 3., "rangeMax": 150000., "band": [8, 9],
                   "antenna": {"azimuth": {"angleHalfSens": 3.3}, "elevation": {"angleHalfSens": 4.5}}}
    return {"type": "radar", "name": "Test",
            "transivers": {"mprf": transceiver, "hprf": {**transceiver, "band": 8, "antenna": {"angleHalfSens": 2.}}},
            "signals": {"mprfSearch": {"dopplerSpeed": {"minValue": -2500., "maxValue": 2500.}, "mainBeamNotchWidth": 60.,
                                       "distance": {"minValue": 500., "maxValue": 74000.}},
                        "hprfVelocitySearch": {"rangeFinder": False, "dopplerSpeedFinder": True,
                                               "dopplerSpeed": {"minValue": 40., "maxValue": 1500.}}},
            "scanPatterns": {"twsMedium": pat(25., 2, 1.5)},
            "scanPatternSets": {"tws": {"scanPattern1": "twsMedium"}, "search": {"scanPattern1": "twsMedium"}},
            "fsms": {"main": {"actionsTemplates": {
                         "setSearchModeCommon": {"setScanPatternSet": {"scanPatternSet": "search"}},
                         "setTwsSearchModeCommon": {"setScanPatternSet": {"scanPatternSet": "tws"}},
                         "setTwsSearchMode": {"setTransiver": {"transiver": "mprf"}, "setSignal": {"signal": "mprfSearch"}},
                         "setHprfVelocitySearchMode": {"setTransiver": {"transiver": "hprf"},
                                                       "setSignal": {"signal": "hprfVelocitySearch"}}}},
                     "tws": {"transitions": {
                         "scan": {"event": "update", "actions": {"clearTargetsOfInterest": {"timeOut": 12.}}},
                         "add": {"event": "targetDetected", "actions": {"updateTargetOfInterest": {
                             "limit": 10, "timeLimits": [2., 2.], "posGateRange": [0., 1000.], "posGateMaxTime": 2.,
                             "posGateRangeInitial": 1500., "posGateTimeInitial": [2., 4.]}}}}}}}


class ExtractionTests(unittest.TestCase):
    def test_beam_band_gate_and_stabilisation_come_from_the_file(self):
        r = units.parse_radar("test", radar_file())
        wf = r.tws.waveforms[0]
        self.assertEqual((wf.beam_azimuth_deg, wf.beam_elevation_deg, wf.band, wf.range_max_m, wf.distance_min_m),
                         (3.3, 4.5, (8, 9), 150000., 500.))
        self.assertTrue(wf.range_finder)
        self.assertTrue(wf.measures_doppler)
        velocity = r.search_waveforms[0]   # the HPRF velocity search: one antenna width for both axes, no range
        self.assertEqual((velocity.beam_azimuth_deg, velocity.beam_elevation_deg, velocity.band, velocity.range_finder),
                         (2., 2., (8,), False))
        self.assertEqual((velocity.doppler_min_mps, velocity.doppler_max_mps), (40., 1500.))
        p = r.tws.patterns[0]
        self.assertEqual((p.center_elevation_deg, p.roll_stab_limit_deg, p.pitch_stab_limit_deg), (-2., 70., 60.))
        self.assertEqual(r.tws.gate, units.Gate((0., 1000.), 2., 1500., (2., 4.)))
        self.assertEqual((r.tws.track_limit, r.tws.track_time_min_s, r.tws.timeout_s), (10, 2., 12.))

    def test_round_trip_through_json_keeps_the_new_fields(self):
        import tempfile
        from pathlib import Path
        r = units.parse_radar("test", radar_file())
        with tempfile.TemporaryDirectory() as folder:
            units.dump(Path(folder)/"radars.json", {"test": r})
            loaded = units._tupled(units.Radar, __import__("json").loads((Path(folder)/"radars.json").read_text())["test"])
        self.assertEqual(loaded, r)

    @unittest.skipUnless((units.DATA_DIR/"radars.json").exists(), "no imported unit data")
    def test_shipped_electronic_radars_now_carry_a_track_limit_and_gate(self):
        u = units.load()
        typhoon = u.radar_of("ef_2000_typhoon_aesa")
        self.assertEqual((typhoon.tws.track_limit, typhoon.tws.track_time_min_s), (40, 2.))
        self.assertEqual(typhoon.tws.gate.range_m, (0., 1000.))
        apg68 = u.radar_of("f_16c_block_50")
        self.assertEqual(apg68.tws.gate, units.Gate((0., 300.), 2., 1500., (2., 4.)))
        wf = apg68.tws.waveforms[0]
        self.assertEqual((wf.beam_azimuth_deg, wf.beam_elevation_deg, wf.band, wf.range_max_m), (3.3, 4.5, (8,), 150000.))
        self.assertEqual(apg68.tws.patterns[0].roll_stab_limit_deg, 70.)
        self.assertTrue(wf.air_target)
        self.assertFalse(u.radar_of("mirage_2000d_r1").search_waveforms[0].air_target)   # Antilope: surface search only


@unittest.skipUnless((units.DATA_DIR/"equipment.json").exists(), "no imported unit data")
class ShippedRadarTests(unittest.TestCase):
    def test_f16c_tws_sees_a_head_on_5m2_target_at_30_km_within_one_frame(self):
        r = units.load().radar_of("f_16c_block_50")
        sensor = RadarSensor(r, rcs_m2=5., owner="me")
        sensor.set_mode("tws", 0, 0., 0., t=0.)
        period = r.tws.patterns[0].period_s
        pics = run(sensor, period, lambda t: own((0., 250.*t, 5000.), (0., 250., 0.)),
                   lambda t: [moving("b", (0., 30000., 5000.), (0., -250., 0.), t)])
        hits = scan_hits(pics)
        self.assertEqual(len(hits), 1)
        self.assertAlmostEqual(hits[0].range_m, 30000.-500.*hits[0].updated_s, delta=1.)
        self.assertAlmostEqual(hits[0].closing_speed_mps, 500., delta=1e-6)
        # And not at 80 km: the limit for 5 m^2 is 62 km * (5/3)^(1/4) = 70 km.
        sensor = RadarSensor(r, rcs_m2=5., owner="me")
        sensor.set_mode("tws", 0, 0., 0., t=0.)
        self.assertFalse(scan_hits(run(sensor, 2*period, lambda t: own(), lambda t: [bandit(position=(0., 80000., 3000.))])))

    def test_f16c_tracks_the_target_and_the_picture_is_usable(self):
        r = units.load().radar_of("f_16c_block_50")
        sensor = RadarSensor(r, rcs_m2=5., owner="me")
        sensor.set_mode("tws", 1, 0., 0., t=0.)   # twsNarrow: 10 deg, 4 bars, 1.2 s
        pics = run(sensor, 8., lambda t: own((0., 250.*t, 5000.), (0., 250., 0.)),
                   lambda t: [moving("b", (0., 40000., 5000.), (0., -250., 0.), t)])
        c = pics[-1].contacts[0]
        self.assertEqual(c.track_id, 1)
        self.assertAlmostEqual(c.closing_speed_mps, 500., delta=3.)

    def test_every_shipped_radar_and_rwr_runs_in_each_available_mode(self):
        u = units.load()
        targets = [bandit(str(i), at_azimuth(-30.+15.*i, 25000.+5000.*i, 3000.+500.*i), (0., -200., 0.)) for i in range(5)]
        ran = 0
        for ident, r in u.radars.items():
            try:
                sensor = RadarSensor(r, owner="me")
            except ValueError:
                continue
            modes = [m for m, ok in (("search", r.search_patterns and r.search_waveforms), ("tws", r.tws and r.tws.patterns)) if ok]
            for mode in modes:
                sensor.set_mode(mode, 0, 0., 0., t=0.)
                for k in range(1, 3*48):
                    pic = sensor.update(k*DT, DT, own((0., 0., 3000.), (0., 200., 0.), 0., 5., 20.), targets)
                self.assertEqual(pic.mode, mode)
                e = sensor.emission("x", 3., own(), at_azimuth(0.))
                self.assertIn(e.kind, ("search", "tws"))
                ran += 1
        self.assertGreater(ran, 100)
        for ident, w in u.rwrs.items():
            sensor = RwrSensor(w)
            sensor.update(0., own((0., 0., 0.)), [emitter(az=a) for a in (0., 90., 180., -90.)])
        self.assertTrue(u.rwrs)


class MissileTargetTests(unittest.TestCase):
    """Engagement(radar_sees_missiles=True) hands the radars enemy missiles of MISSILE_RCS_M2 under the normal rules."""

    def test_head_on_missile_is_first_seen_near_70_km_by_the_reference_radar(self):
        from wt_overlay.engagement import MISSILE_RCS_M2, MISSILE_TRUTH
        r = units.load().radars["us_an_apg_63_v_3"]
        wf = r.tws.waveforms[0]
        self.assertEqual((wf.signal, wf.range_m, wf.reference_rcs_m2), ("mprfSearch", 70000., 1.))
        sensor = RadarSensor(r, owner="me")
        sensor.set_mode("tws", 0, 0., 0., t=0.)
        self.assertAlmostEqual(sensor._limit(wf, MISSILE_RCS_M2), 70000.)
        ident, closing = MISSILE_TRUTH+3, 300.+1000.
        own_at = lambda t: own((0., 300.*t, 8000.), (0., 300., 0.))  # noqa: E731
        missile = lambda t: [moving(ident, (0., 90000., 8000.), (0., -1000., 0.), t, MISSILE_RCS_M2)]  # noqa: E731
        hits = [h for p in run(sensor, 20., own_at, missile) for h in p.hits]
        self.assertTrue(hits)
        first = hits[0].range_m
        self.assertLessEqual(first, 70000.)
        self.assertGreater(first, 70000.-closing*r.tws.patterns[0].period_s)
        self.assertIn(ident, sensor.tracked_ids())

    def test_missiles_take_tws_track_slots(self):
        from wt_overlay.engagement import MISSILE_RCS_M2, MISSILE_TRUTH
        # The missile and the first aircraft are swept (and confirmed) before the second aircraft.
        targets = lambda t: [bandit(MISSILE_TRUTH, at_azimuth(-5., 25000.), (0., 0., 0.), MISSILE_RCS_M2),  # noqa: E731
                             bandit("a", at_azimuth(0.)), bandit("b", at_azimuth(20.))]
        full, capped = tracking(limit=3), tracking(limit=2)
        run(full, 8., lambda t: own(), targets)
        run(capped, 8., lambda t: own(), targets)
        self.assertEqual(full.tracked_ids(), {MISSILE_TRUTH, "a", "b"})
        self.assertEqual(capped.tracked_ids(), {MISSILE_TRUTH, "a"})


if __name__ == "__main__":
    unittest.main()
