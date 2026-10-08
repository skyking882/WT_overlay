"""Focused real-model sandbox contracts, including its independent live API."""
import copy
import json
import math
from pathlib import Path
import tempfile
import threading
import time
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from wt_overlay.engagement import ReplayWriter
from wt_overlay.flight import FlightCommand, KeyboardCommand
from wt_overlay.sandbox import (SandboxServer, SandboxSession, analysis_snapshot, build_sandbox,
                                preset, set_order, validate_sandbox)
from wt_overlay.sim import build_simulation, validate_scenario

HAVE_MODELS = (Path(__file__).resolve().parents[2]/"missle_sim"/"src"/"aim120_model").exists()


class ScenarioTests(unittest.TestCase):
    def test_multiple_units_and_old_singleton_contract(self):
        scene = preset(True)
        original = copy.deepcopy(scene)
        self.assertEqual([len(t) for t in validate_sandbox(scene)["teams"]], [2, 2])
        self.assertEqual(scene, original)
        with self.assertRaises(ValueError):
            validate_scenario(scene)
        for n in (0, 17):
            scene["teams"][0] = [copy.deepcopy(original["teams"][0][0]) for _ in range(n)]
            with self.assertRaises(ValueError):
                validate_sandbox(scene)
        scene = preset()
        scene["teams"][0][0]["position_m"][0] = float("nan")
        with self.assertRaises(ValueError):
            validate_sandbox(scene)


@unittest.skipUnless(HAVE_MODELS, "sibling missile models not installed")
class SandboxTests(unittest.TestCase):
    def test_multi_deployment_exact_coordinates_and_observation_boundary(self):
        scene = preset(True)
        eng = build_sandbox(scene)
        self.assertEqual(len(eng.planes), 4)
        self.assertEqual([p.flight.state.position for p in eng.planes],
                         [tuple(p["position_m"]) for t in scene["teams"] for p in t])
        for _ in range(48):
            eng.step()
        for p in eng.planes:
            self.assertIsNone(p.controller.auto.observation.truth)
            self.assertFalse(p.controller.auto.pilot.debug)
            self.assertTrue(90 <= p.camera.fov_deg <= 120)
        self.assertFalse(eng.truth_debug)

    def test_navigation_and_keyboard_commands_use_existing_physics(self):
        eng = build_sandbox(preset())
        set_order(eng, 0, dict(mode="manual", heading_deg=90, altitude_m=3500, speed_mps=280))
        before = eng.planes[0].flight.state
        for _ in range(240):
            eng.step()
        p = eng.planes[0]
        self.assertIsInstance(p.flight.command, FlightCommand)
        self.assertGreater(p.own.heading_deg, 10)
        self.assertGreater(p.flight.state.position[2], before.position[2])
        self.assertEqual(eng.planes[1].controller.mode, "auto")
        self.assertIsNone(eng.planes[1].controller.auto.observation.truth)
        set_order(eng, 0, dict(mode="pilot"))
        p.controller.keys = dict(roll=1, pitch=1, throttle=0, airbrake=False)
        p.controller.keys_until = time.monotonic()+10
        previous = p.flight.state
        for _ in range(48):
            eng.step()
        self.assertIsInstance(p.flight.command, KeyboardCommand)
        self.assertNotEqual(p.flight.state.velocity, previous.velocity)
        self.assertGreater(abs(p.own.roll_deg), 30)
        p.controller.keys = dict(roll=0, pitch=0, throttle=0, airbrake=False, level=True)
        for _ in range(4*48):
            eng.step()
        self.assertLess(abs(p.own.roll_deg), 10)
        p.controller.keys_until = 0
        eng.step()
        self.assertEqual(p.flight.command.roll, 0)
        self.assertEqual(p.flight.command.pitch, 0)
        set_order(eng, 0, dict(mode="auto"))
        eng.step()
        self.assertEqual(p.controller.mode, "auto")
        self.assertIsNone(p.controller.auto.observation.truth)

    def test_autonomous_wrapper_preserves_standalone_world(self):
        scene = preset()
        scene["time_limit_s"] = 15
        headless = build_simulation(scene)
        sandbox = build_sandbox(scene)
        headless.run()
        sandbox.run()
        self.assertEqual([p.flight.state for p in headless.planes], [p.flight.state for p in sandbox.planes])
        self.assertEqual(headless.log, sandbox.log)
        tick = sandbox.tick
        sandbox.step()
        self.assertEqual(sandbox.tick, tick)

    def test_session_pause_step_terminal_reset_and_cancelled_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = SandboxSession(output_root=tmp)
            try:
                self.send(session, "step", seconds=1)
                self.wait(session, lambda s:s["status"] == "paused")
                state = session.state()
                self.assertEqual(state["time_s"], 1)
                time.sleep(.05)
                self.assertEqual(session.state()["tick"], 48)
                self.send(session, "order", plane=0, order=dict(mode="pilot"))
                self.send(session, "keys", plane=0, keys=dict(roll=1,pitch=1,throttle=0,airbrake=False), error=True)
                self.send(session, "chaff", plane=0)
                self.send(session, "fire", plane=0, track=999, error=True)
                self.send(session, "reset")
                partial = session.state()["output"]
                self.assertFalse(partial["terminal"])
                self.assertEqual(partial["status"], "cancelled")
                rows = [json.loads(s) for s in (Path(partial["path"])/"replay.jsonl").read_text().splitlines()]
                self.assertFalse(any(r["type"] == "end" for r in rows))
                self.assertTrue(any(r.get("kind") == "sandbox_cancel" for r in rows))
                scene = preset()
                scene["time_limit_s"] = 1
                self.send(session, "configure", scenario=scene)
                self.send(session, "speed", speed=64)
                self.send(session, "start")
                self.wait(session, lambda s:s["status"] == "completed")
                state = session.state()
                self.assertEqual(state["time_s"], 1)
                self.assertTrue(state["output"]["terminal"])
                self.send(session, "start", error=True)
                self.send(session, "step", error=True)
                self.assertEqual(session.state()["tick"], 48)
                self.send(session, "reset")
                self.assertEqual(session.state()["status"], "setup")
            finally:
                session.close()

    def test_http_reads_remain_responsive_during_live_physics_and_commands_are_local(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = SandboxSession(output_root=tmp)
            server = SandboxServer(("127.0.0.1",0),session)
            thread = threading.Thread(target=server.serve_forever,daemon=True)
            thread.start()
            root = f"http://127.0.0.1:{server.server_port}"
            try:
                with urlopen(root+"/api/catalog") as response:
                    self.assertTrue(json.load(response)["aircraft"])
                self.send(session,"speed",speed=64)
                self.send(session,"start")
                started = time.monotonic()
                with urlopen(root+"/api/state",timeout=2) as response:
                    self.assertEqual(json.load(response)["status"],"running")
                self.assertLess(time.monotonic()-started, 2)
                for headers in ({"Content-Type":"application/json"},
                                {"Content-Type":"application/json","X-Sandbox-Token":server.token,"Origin":"http://elsewhere.invalid"}):
                    request = Request(root+"/api/command",data=b'{"action":"pause"}',headers=headers)
                    with self.assertRaises(HTTPError) as error:
                        urlopen(request)
                    self.assertEqual(error.exception.code,403)
                    error.exception.close()
                request = Request(root+"/api/command",data=b'{"action":"pause"}',
                                  headers={"Content-Type":"application/json","X-Sandbox-Token":server.token})
                with urlopen(request) as response:
                    self.assertEqual(response.status,202)
                    jid = json.load(response)["id"]
                self.wait(session,lambda s:s["status"] == "paused")
                self.assertTrue(session.job(jid)["done"])
            finally:
                server.shutdown()
                server.server_close()
                session.close()

    def send(self, session, action, error=False, **fields):
        jid = session.submit(dict(action=action, **fields))
        until = time.monotonic()+10
        while time.monotonic()<until:
            receipt = session.job(jid)
            if receipt["done"]:
                self.assertEqual("error" in receipt,error,receipt)
                return receipt
            time.sleep(.01)
        self.fail("command did not complete")

    def wait(self,session,condition):
        until = time.monotonic()+10
        while time.monotonic()<until:
            if condition(session.state()): return
            time.sleep(.01)
        self.fail(f"world did not reach state: {session.state()['status']}")


if __name__ == "__main__":
    unittest.main()
