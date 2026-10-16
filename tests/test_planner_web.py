"""Local opening-shot page (wt_overlay.planner_web): options, input checks, the HTTP API with a stubbed planner,
one request at a time, and (WT_SLOW=1, missile_sim present) one real computation."""

from dataclasses import replace
import json
import os
import threading
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request

from wt_overlay.engagement import DEFAULT_MISSILE_SIM
from wt_overlay.opening import OpeningPlan
from wt_overlay.planner_web import server

HAVE_MISSILE_SIM = (DEFAULT_MISSILE_SIM/"src"/"aim120_model").exists()
PLAN = OpeningPlan(True, climb_deg=15., pull_s=45., pitch_deg=24., launch_s=46.5, window_s=(35., 51.5),
                   launch_range_m=65500., hit_s=95.1, flight_s=48.6, launch_altitude_m=6987., launch_tas_mps=426.,
                   target_altitude_m=9998., profile=tuple((float(t), 2500.+100.*t, 290.+3.*t, 15.) for t in range(47)))
TRACE = dict(own=[(0., 0., 2500.), (46.5, 16000., 7000.)], missile=[(46.5, 16000., 7000.), (95.1, 59000., 9998.)],
             enemy=[(0., 100000., 2500.), (95.1, 59000., 9998.)], launch=(46.5, 16000., 7000.),
             end=(95.1, 59000., 9998.), hit=True, event="fuse", launch_speed_mps=426., launch_path_deg=24.)


def form(**changes):
    body = dict(aircraft="ef_2000_typhoon_aesa", missile="us_aim_120c_5", altitude_m=2500, mach=.85, fuel_percent=50,
                missiles=6, enemy=dict(aircraft="ef_2000_aesa", distance_km=100, fuel_percent=100, climb_deg=15,
                                       level_m=10000))
    body.update(changes)
    return body


@unittest.skipUnless(HAVE_MISSILE_SIM, "missile_sim not present")
class RequestTests(unittest.TestCase):
    def test_form_to_planner_input(self):
        request, inputs = server.build_request(form())
        self.assertEqual((request.aircraft_id, request.missile, request.t_s, request.path_deg),
                         ("ef_2000_typhoon_aesa", "us_aim_120c_5", 0., 0.))
        self.assertAlmostEqual(request.tas_mps, .85*330.6, delta=1.)    # Mach 0.85 at 2500 m, ISA (a = 330.6 m/s)
        self.assertEqual((request.spawn_altitude_m, request.spawn_tas_mps), (request.altitude_m, request.tas_mps))
        self.assertEqual(request.settings.distance_m, 100000.)
        self.assertAlmostEqual(inputs["fuel_kg"], 2325., delta=1.)       # half of 4650 kg internal
        self.assertGreater(inputs["mass_kg"], 11220.+2325.+6*150.)

    def test_bad_input_is_named(self):
        for change, word in ((dict(aircraft="nope"), "机型"), (dict(missile="cn_pl15"), "导弹"),
                             (dict(altitude_m=50), "起始高度"), (dict(mach="fast"), "马赫"),
                             (dict(fuel_percent=0), "内油"), (dict(enemy=dict(aircraft="nope")), "敌机"),
                             (dict(enemy=dict(aircraft="ef_2000_aesa", distance_km=5, fuel_percent=100, climb_deg=15,
                                              level_m=10000)), "间距")):
            with self.subTest(change=change), self.assertRaisesRegex(server.InputError, word):
                server.build_request(form(**change))

    def test_options_list_radar_missiles_without_pl15(self):
        options = server.Planner(1).options()
        aircraft = {a["id"]: a for a in options["aircraft"]}
        self.assertEqual([m["id"] for m in aircraft["ef_2000_typhoon_aesa"]["missiles"]][0], "us_aim_120c_5")
        ids = {m["id"] for m in options["radar_missiles"]}
        self.assertIn("us_aim_54c", ids)
        self.assertNotIn("cn_pl15", ids)
        self.assertNotIn("us_aim9m_sidewinder", ids)
        self.assertEqual(options["enemy"]["aircraft"], "ef_2000_aesa")

    def test_steps_read_like_the_cockpit(self):
        _, inputs = server.build_request(form())
        rows = server.steps(PLAN, inputs, TRACE)
        self.assertEqual([r["title"] for r in rows], ["全加力，爬升 15°", "拉起到 24°", "发射", "命中"])
        self.assertEqual([r["t"] for r in rows], [0., 45., 46.5, 95.1])
        self.assertIn("65.5 km", rows[2]["detail"])
        self.assertIn("59.0 km", rows[3]["detail"])
        self.assertEqual(len(server.steps(replace(PLAN, pitch_deg=None), inputs, TRACE)), 3)


@unittest.skipUnless(HAVE_MISSILE_SIM, "missile_sim not present")
class HttpTests(unittest.TestCase):
    def setUp(self):
        self.planner = server.Planner(1)
        self.http = server.PlannerServer(("127.0.0.1", 0), self.planner)
        self.thread = threading.Thread(target=self.http.serve_forever, kwargs=dict(poll_interval=.05), daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.http.server_address[1]}"

    def tearDown(self):
        self.http.shutdown()
        self.http.server_close()
        self.planner.close()

    def get(self, path):
        with urllib.request.urlopen(self.url+path, timeout=10) as response:
            return response.status, response.read()

    def post(self, body):
        data = json.dumps(body).encode()
        request = urllib.request.Request(self.url+"/api/plan", data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    def test_page_assets_and_options(self):
        status, body = self.get("/")
        self.assertEqual(status, 200)
        self.assertIn("开局抽奖".encode(), body)
        for asset in ("app.js", "app.css"):
            self.assertEqual(self.get(f"/static/{asset}")[0], 200)
        with self.assertRaises(urllib.error.HTTPError):
            self.get("/static/../server.py")
        status, body = self.get("/api/options")
        self.assertTrue(json.loads(body)["ok"])

    def test_plan_round_trip_with_a_stubbed_search(self):
        def fake_run(function, jobs):
            return [TRACE for _ in jobs]
        with patch.object(server, "plan_opening", return_value=PLAN), patch.object(self.planner, "run", fake_run):
            status, data = self.post(form())
        self.assertEqual(status, 200)
        self.assertTrue(data["ok"])
        self.assertEqual(data["plan"]["hit_s"], 95.1)
        self.assertEqual(data["steps"][-1]["title"], "命中")
        self.assertEqual(data["trace"]["end"][2], 9998.)
        self.assertEqual(data["inputs"]["missile_name"], "AIM-120C-5")

    def test_bad_input_and_busy(self):
        status, data = self.post(form(altitude_m=-1))
        self.assertEqual((status, data["error"]["code"]), (422, "invalid_input"))
        self.planner.lock.acquire()
        try:
            status, data = self.post(form())
            self.assertEqual((status, data["error"]["code"]), (409, "busy"))
        finally:
            self.planner.lock.release()

    def test_a_plan_that_cannot_hit_has_no_trace(self):
        with patch.object(server, "plan_opening", return_value=OpeningPlan(False, "打不到")):
            status, data = self.post(form())
        self.assertEqual(status, 200)
        self.assertFalse(data["plan"]["available"])
        self.assertNotIn("trace", data)


@unittest.skipUnless(HAVE_MISSILE_SIM and os.environ.get("WT_SLOW"), "set WT_SLOW=1 (missile_sim, real search)")
class RealPlanTests(unittest.TestCase):
    def test_typhoon_120c5(self):
        planner = server.Planner(4)
        try:
            result = planner.plan(form())
        finally:
            planner.close()
        self.assertTrue(result["plan"]["available"])
        self.assertLess(result["plan"]["hit_s"], 110.)
        self.assertTrue(result["trace"]["hit"])
        self.assertAlmostEqual(result["trace"]["end"][0], result["plan"]["hit_s"], delta=.5)


if __name__ == "__main__":
    unittest.main()
