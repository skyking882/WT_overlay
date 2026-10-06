import tempfile
import unittest
from pathlib import Path

from wt_overlay import units


def radar_file(electronic=False):
    """A minimal radar in the game's layout: TWS scan set and waveform picked by action templates."""
    scan = {"actions": {"scan": {}, "extrapolateTargetsOfInterest": {}, "clearTargetsOfInterest": {"timeOut": 6. if electronic else 8.}}}
    transitions = {"scan": scan,
                   "matchTargets": {"event": "scanFinished",
                                    "actions": {"matchTargetsOfInterest": {"limit": 10, "timeMin": 2.}}}}
    if electronic:
        transitions["scanTrack"] = {"stateFrom": "search", "stateTo": "track", "event": "update",
                                    "actions": {"scan": {"scanPattern": "fastTws"},
                                                "clearTargetsOfInterest": {"timeOut": 4.}}}
    pattern = lambda w, bars, period: dict(type="pyramide", azimuthLimits=[-60., 60.], elevationLimits=[-60., 60.],  # noqa: E731
                                           width=w, barHeight=3.2, barsCount=bars, period=period)
    return {"type": "radar", "name": "Test radar",
            "transivers": {"mprf": {"range": 62000., "rcs": 3.}},
            "signals": {"mprfSearch": {"dopplerSpeed": {"minValue": -2500., "maxValue": 2500.}, "mainBeamNotchWidth": 60.,
                                       "groundClutter": False, "distance": {"maxValue": 74000.}}},
            "scanPatterns": {"searchWide": pattern(60., 2, 3.6), "twsMedium": pattern(25., 2, 1.5),
                             "twsNarrow": pattern(10., 4, 1.2), "fastTws": pattern(100., 80, .0416)},
            "scanPatternSets": {"search": {"scanPattern1": "searchWide"},
                                "tws": {"scanPattern1": "twsMedium", "scanPattern2": "twsNarrow"}},
            "fsms": {"main": {"actionsTemplates": {
                         "setSearchModeCommon": {"setScanPatternSet": {"scanPatternSet": "search"}},
                         "setTwsSearchModeCommon": {"setScanPatternSet": {"scanPatternSet": "tws"}},
                         "setMprfSearchMode": {"setTransiver": {"transiver": "mprf"}, "setSignal": {"signal": "mprfSearch"}},
                         "setTwsSearchMode": {"setTransiver": {"transiver": "mprf"}, "setSignal": {"signal": "mprfSearch"}}}},
                     "tws": {"stateInit": "search", "transitions": transitions},
                     "track": {"actionsTemplates": {"extrapolate": {"clearTargetsOfInterest": {"timeOut": 3.}}}}}}


class RadarTests(unittest.TestCase):
    def test_mechanical_tws_follows_the_action_templates(self):
        r = units.parse_radar("test", radar_file())
        self.assertFalse(r.electronic)
        self.assertEqual([(p.name, p.half_width_deg, p.bars, p.period_s) for p in r.tws.patterns],
                         [("twsMedium", 25., 2, 1.5), ("twsNarrow", 10., 4, 1.2)])
        self.assertEqual((r.tws.timeout_s, r.tws.track_limit, r.tws.track_time_min_s), (8., 10, 2.))
        wave = r.tws.waveforms[0]
        self.assertEqual((wave.transceiver, wave.range_m, wave.reference_rcs_m2, wave.main_beam_notch_mps), ("mprf", 62000., 3., 60.))
        self.assertEqual([p.name for p in r.search_patterns], ["searchWide"])
        self.assertEqual((r.stt_coast_s, r.field_of_regard_deg), (3., 60.))

    def test_nctr_names_missiles_when_the_type_table_has_rocket_propulsion(self):
        jets = [{"name": "hud/single jet", "targetPropulsion": {"type": "jet", "num": 1}},
                {"name": "hud/multi jet", "targetPropulsion": [{"type": "jet", "num": 2}, {"type": "jet", "num": 3}]},
                {"name": "hud/small", "sizeRange": [0., 5.]}]
        rocket = {"name": "hud/rocket", "targetPropulsion": {"type": "rocket"}}
        self.assertFalse(units.parse_radar("test", radar_file()).identifies_missiles)
        self.assertFalse(units.parse_radar("test", dict(radar_file(), targetTypeId=jets)).identifies_missiles)
        r = units.parse_radar("test", dict(radar_file(), targetTypeId=[*jets, rocket]))
        self.assertTrue(r.identifies_missiles)
        listed = dict(rocket, targetPropulsion=[{"type": "jet", "num": 1}, {"type": "rocket"}])
        self.assertTrue(units.parse_radar("test", dict(radar_file(), targetTypeId=[listed])).identifies_missiles)
        with tempfile.TemporaryDirectory() as folder:
            units.dump(Path(folder)/"radars.json", {"test": r})
            for name in ("equipment.json", "rwrs.json"):
                (Path(folder)/name).write_text("{}")
            self.assertEqual(units.load(Path(folder)).radars["test"], r)

    def test_electronic_tws_has_a_fast_track_update(self):
        r = units.parse_radar("test", radar_file(electronic=True))
        self.assertTrue(r.electronic)
        self.assertEqual((r.tws.fast_pattern.name, r.tws.fast_pattern.period_s), ("fastTws", .0416))
        self.assertEqual((r.tws.timeout_s, r.tws.fast_timeout_s), (6., 4.))
        self.assertEqual([p.name for p in r.tws.patterns], ["twsMedium", "twsNarrow"])


class UnitTests(unittest.TestCase):
    def unit(self):
        aam = "missile_type_f_air_to_air_midrange"
        return {"sensors": {"sensor": [{"blk": "gameData/sensors/test_radar.blk"}, {"blk": "gameData/sensors/test_rwr.blk"}]},
                "WeaponSlots": {"WeaponSlot": [
                    {"index": 0, "WeaponPreset": [
                        {"name": "gun_common", "Weapon": [{"blk": "gameData/Weapons/cannon.blk", "bullets": 500},
                                                          {"blk": "gameData/Weapons/countermeasure_split_launcher_jet.blk", "bullets": 30},
                                                          {"blk": "gameData/Weapons/countermeasure_split_launcher_jet.blk", "bullets": 30}]},
                        {"name": "gun_ltc", "Weapon": [{"blk": "gameData/Weapons/countermeasure_split_launcher_jet.blk", "bullets": 120}]}]},
                    {"index": 1, "WeaponPreset": [
                        {"name": "a", "iconType": aam, "Weapon": {"blk": "gameData/Weapons/rocketGuns/us_aim_120b_default.blk", "bullets": 1}},
                        {"name": "b", "iconType": aam, "Weapon": {"blk": "gameData/Weapons/containers/rack_x2.blk"}},
                        {"name": "pod", "iconType": aam, "Weapon": {"blk": "gameData/Weapons/countermeasure_pod_bol.blk", "bullets": 160}}]},
                    {"index": 2, "WeaponPreset": {"name": "bomb", "iconType": "bombs", "Weapon": {"blk": "gameData/Weapons/mk82.blk"}}}]}}

    def test_missiles_count_racks_and_skip_countermeasure_pods(self):
        e = units.parse_unit("jet", self.unit(), {"test_radar": "radar", "test_rwr": "rwr"},
                             {"rack_x2": ("gameData/Weapons/rocketGuns/us_aim_120c_5.blk", 2)})
        self.assertEqual((e.radar, e.rwr, e.mlws), ("test_radar", "test_rwr", False))
        self.assertEqual(e.missiles, {"us_aim_120b": 1, "us_aim_120c_5": 2})
        self.assertEqual((e.countermeasures, e.countermeasures_max), (60, 280))

    def test_a_preset_listing_one_missile_twice_carries_two(self):
        # Su-30SM2 style: a fuselage slot whose preset lists the missile twice (a tandem pair) next to a single.
        aam = "missile_type_f_air_to_air_midrange"
        unit = {"WeaponSlots": {"WeaponSlot": [
            {"index": 1, "WeaponPreset": {"name": "one", "iconType": aam,
                                          "Weapon": {"blk": "gameData/Weapons/rocketGuns/su_r_77_1.blk", "bullets": 1}}},
            {"index": 2, "WeaponPreset": [
                {"name": "single", "iconType": aam, "Weapon": {"blk": "gameData/Weapons/rocketGuns/su_r_77_1.blk", "bullets": 1}},
                {"name": "pair", "iconType": aam + "_group",
                 "Weapon": [{"blk": "gameData/Weapons/rocketGuns/su_r_77_1.blk", "bullets": 1},
                            {"blk": "gameData/Weapons/rocketGuns/su_r_77_1.blk", "bullets": 1}]}]}]}}
        self.assertEqual(units.parse_unit("jet", unit, {}, {}).missiles, {"su_r_77_1": 3})

    def test_round_trip_through_json(self):
        radar = units.parse_radar("test_radar", radar_file(electronic=True))
        rwr = units.parse_rwr("test_rwr", {"type": "rwr", "name": "RWR", "range": 70000., "band8": True, "band9": [True, False],
                                           "detectTracking": True, "targetHoldTime": 5., "targetRangeFinder": True,
                                           "targetRange": [5000., 50000.],
                                           "receivers": {"receiver": [{"azimuth": 0., "elevation": 0., "azimuthWidth": 180.,
                                                                       "elevationWidth": 90., "angleFinder": True}]}})
        self.assertEqual(rwr.bands, (8, 9))
        unit = units.parse_unit("jet", self.unit(), {"test_radar": "radar", "test_rwr": "rwr"}, {})
        with tempfile.TemporaryDirectory() as folder:
            units.dump(Path(folder)/"equipment.json", {"jet": unit})
            units.dump(Path(folder)/"radars.json", {"test_radar": radar})
            units.dump(Path(folder)/"rwrs.json", {"test_rwr": rwr})
            loaded = units.load(Path(folder))
        self.assertEqual(loaded.radar_of("jet"), radar)
        self.assertEqual(loaded.rwr_of("jet"), rwr)
        self.assertEqual(loaded.equipment["jet"], unit)


class ShippedUnitsTests(unittest.TestCase):
    @unittest.skipUnless((units.DATA_DIR/"equipment.json").exists(), "no imported unit data")
    def test_f16c_and_typhoon_radars(self):
        u = units.load()
        apg68 = u.radar_of("f_16c_block_50")
        self.assertFalse(apg68.electronic)
        self.assertEqual((apg68.tws.timeout_s, apg68.tws.track_time_min_s), (8., 2.))
        self.assertTrue(u.radar_of("ef_2000_typhoon_aesa").electronic)
        self.assertEqual(u.equipment["f_16c_block_50"].missiles["us_aim_120a"], 6)
        # Full loads (user 2026-10-06: Golden Eagle 12 AIM-120D, Su-30SM2 12 R-77-1).
        self.assertEqual(u.equipment["f_15c_golden_eagle"].missiles["us_aim_120d"], 12)
        self.assertEqual(u.equipment["su_30sm2"].missiles["su_r_77_1"], 12)

    @unittest.skipUnless((units.DATA_DIR/"radars.json").exists(), "no imported unit data")
    def test_every_top_tier_radar_identifies_missiles(self):
        import json
        u = units.load()
        model = json.loads((units.DATA_DIR.parent/"match"/"top_tier.json").read_text())
        aircraft = list(model["aircraft_frequency"]["weights"])
        self.assertEqual(len(aircraft), 19)
        self.assertTrue(all(u.radar_of(a).identifies_missiles for a in aircraft))
        # Not only AESA: the mechanically scanned CAPTOR-M and N011M and the N035E too; older radars do not.
        self.assertTrue(all(u.radars[r].identifies_missiles for r in ("uk_captor_m", "su_n_011m", "su_n_035e")))
        self.assertFalse(u.radars["us_an_apg_68_v_9"].identifies_missiles)


if __name__ == "__main__":
    unittest.main()
