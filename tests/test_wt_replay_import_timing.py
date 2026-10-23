"""Native missile observations must survive resampling without changing attribution tracks."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("wt_import_timing", ROOT / "scripts/wt_replay_import.py")
imp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(imp)


class NativeMissileTimingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        # 32 observations, with a 14-tick hole after the first 31. Another missile samples inside that hole.
        self.ticks = list(range(480, 511)) + [524]
        self.states = []
        rockets = []
        for eid, ticks in [(301, self.ticks), (302, list(range(511, 525)))]:
            track = []
            for i, tick in enumerate(ticks):
                packet = 10030 if eid == 301 and i < 2 else round(tick/48*1000)+30
                xyz = [float(tick-480)*10, 6000., 5000.]
                track.append(dict(t_ms=packet, xyz=xyz, euler_raw=[0., 0., 0.]))
                self.states.append(dict(kind="rocket", eid=eid, packet_ms=packet, b5=tick, xyz=xyz,
                                        euler_raw=[0., 0., 0.], b7_candidate=[float(tick), 2., 3.],
                                        sf_raw={"1": tick-480+1}, sf_decoded_s={"1": (tick-480)/48}))
            rockets.append(dict(eid=eid, owner_eid=101, weapon_name="us_aim_120c_5", track=track,
                                created_at_ms=10000, destroyed_at_ms=11000,
                                creation_raw_fields=dict(creation_time=10., starting_pos=[0., 6000., 5000.])))
        self.raw = dict(source="timing.wrpl", source_sha256="a"*64, replay_length_ms=20000, rockets=rockets,
                        units=[dict(eid=101, uid=1, unit_type=2, unit_name="f_16c", army=1,
                                    track=[dict(t_ms=t, xyz=[0., 6000., t*.25]) for t in range(0, 20001, 250)]),
                               dict(eid=202, uid=2, unit_type=2, unit_name="su_30sm2", army=2,
                                    track=[dict(t_ms=t, xyz=[1000., 6000., 10000.-t*.25])
                                           for t in range(0, 20001, 250)])])
        self.export = self.root / "match_export.json"
        self.export.write_text(json.dumps(self.raw))
        self.sidecar = self.root / "states.jsonl"
        self.write_states()

    def write_states(self, states=None, sha=None):
        header = dict(type="header", source_sha256=sha or self.raw["source_sha256"])
        self.sidecar.write_text("\n".join(json.dumps(x) for x in [header]+(self.states if states is None else states)))

    def convert(self, native=True):
        out, _ = imp.convert(self.export, self.root / ("native" if native else "packet"), [], False,
                             weapon_states=self.sidecar if native else None)
        rows = [json.loads(l) for l in out.read_text().splitlines()]
        return out, rows

    @staticmethod
    def launches(rows):
        return {r["wt_eid"]: r for r in rows if r.get("kind") == "launch"}

    def test_same_packet_distinct_ticks_and_b7_pairing(self):
        _, rows = self.convert()
        uid = self.launches(rows)[301]["uid"]
        observations = [(f, s) for f in rows if f.get("type") == "frame"
                        for s in f["native_missile_samples"] if s["uid"] == uid]
        self.assertEqual(len(observations), 32)
        self.assertEqual([s["b5"] for f, s in observations], self.ticks)
        for f, s in observations:
            self.assertAlmostEqual(f["t"], s["b5"]/48, places=9)
            missile = next(r for r in f["missiles"] if r[0] == uid)
            self.assertEqual(missile[6:9], [float(s["b5"]), 3., 2.])
            self.assertAlmostEqual(missile[10], f["t"]-10, places=9)
        self.assertEqual([s["packet_ms"] for f, s in observations[:2]], [10030, 10030])
        self.assertAlmostEqual(observations[1][0]["t"]-observations[0][0]["t"], 1/48, places=9)
        self.assertNotEqual(observations[1][0]["missiles"][0][10], .02)

    def test_gap_and_other_missiles_native_frame_is_not_a_native_observation(self):
        _, rows = self.convert()
        uid = self.launches(rows)[301]["uid"]
        frames = [f for f in rows if f.get("type") == "frame" and 510/48 < f["t"] < 524/48-1e-9]
        self.assertTrue(frames)
        for frame in frames:
            self.assertTrue(any(r[0] == uid for r in frame["missiles"]))
            self.assertNotIn(uid, frame["native_missile_uids"])
            self.assertTrue(frame["native_missile_uids"])  # the other missile really is observed
        samples = [f["t"] for f in rows if f.get("type") == "frame" and uid in f["native_missile_uids"]]
        self.assertAlmostEqual(samples[-1]-samples[-2], 14/48, places=9)

    def test_packet_fallback_background_and_attribution_are_preserved(self):
        _, packet = self.convert(False)
        _, native = self.convert()
        self.assertEqual(self.launches(packet), self.launches(native))
        self.assertEqual([r for r in packet if r.get("type") == "event"],
                         [r for r in native if r.get("type") == "event"])
        header = packet[0]
        self.assertIsNone(header["frame_dt_s"])
        self.assertEqual(header["base_frame_dt_s"], .25)
        sample = next(s for f in packet if f.get("type") == "frame"
                      for s in f["native_missile_samples"])
        self.assertEqual(sample["clock"], "packet_ms/1000")
        self.assertNotIn("b5", sample)
        self.assertEqual(header["source"]["sampling"]["missile_samples"][0]["unresolved_packet_time_collisions"], 1)
        self.assertTrue(any(f.get("t") == 10.03 for f in packet if f.get("type") == "frame"))
        # Reader accepts variable frame times and extra provenance fields.
        from wt_overlay.sandbox_replay import ReplayTrack
        path, _ = self.convert()
        replay = ReplayTrack(path)
        self.assertIsNotNone(replay.missile_at(self.launches(native)[301]["uid"], 10.5))

    def test_wrong_source_or_track_missing_clock_and_partial_sidecars_fail(self):
        self.write_states(sha="b"*64)
        with self.assertRaisesRegex(ValueError, "source_sha256"):
            self.convert()
        wrong = [dict(s) for s in self.states]
        wrong[0]["xyz"] = [999., 6000., 5000.]
        self.write_states(wrong)
        with self.assertRaisesRegex(ValueError, "matching export state"):
            self.convert()
        wrong = [dict(s) for s in self.states]
        del wrong[0]["b5"]
        self.write_states(wrong)
        with self.assertRaisesRegex(ValueError, "decoded b5"):
            self.convert()
        self.write_states(self.states[1:])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.convert()


if __name__ == "__main__":
    unittest.main()
