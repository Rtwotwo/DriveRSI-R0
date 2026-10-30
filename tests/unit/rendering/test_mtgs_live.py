"""Loopback controls must preserve the world clock and reject foreign writes."""

import copy
import http.client
import json
import tempfile
import threading
import unittest
from pathlib import Path

from driversi.rendering.gaussian.mtgs_live import LiveController, make_live_server


class FakeSession:
    def __init__(self, output=None, dt=.2):
        self.output = output
        self.dt = dt
        self.current = {"tick": 0, "controlled_ids": ["leader", "follower"], "done": False}
        self.generation = 0
        self.calls = []
        self.fail_step = False
        self.fail_reset = False

    def state(self):
        return {**self.current, "world_tick": self.current["tick"], "dt": self.dt,
                "generation": self.generation, "status": "running", "actors": []}

    def advance(self, overrides):
        if self.fail_step:
            raise RuntimeError("Renderer did not produce a frame")
        self.calls.append(copy.deepcopy(overrides))
        self.current["tick"] += 1

    def reset(self):
        if self.fail_reset:
            raise RuntimeError("Reset renderer failed")
        self.generation += 1
        self.current["tick"] = 0
        self.current["done"] = False


class ControllerTests(unittest.TestCase):
    def test_brake_duration_uses_physics_time_and_snapshot_cannot_modify_it(self):
        session = FakeSession(dt=.2)
        controller = LiveController(session)
        snapshot = controller.control({"operation": "brake", "actor_id": "leader"})
        self.assertEqual(snapshot["scheduled_brake"]["until_tick"], 5)
        snapshot["scheduled_brake"]["actor_id"] = "follower"
        for _ in range(6):
            controller.control({"operation": "step"})
        self.assertEqual(session.calls, [{"leader": 1}] * 5 + [{}])
        self.assertIsNone(controller.state()["scheduled_brake"])

    def test_tick_pump_uses_declared_period_and_never_catches_up_by_skipping_steps(self):
        from unittest.mock import patch

        session = FakeSession(dt=.2)
        controller = LiveController(session)
        controller.playing = True
        waits = []

        class OneIteration:
            def is_set(self):
                return bool(waits)

            def wait(self, seconds):
                waits.append(seconds)

        controller.stop = OneIteration()
        with patch("driversi.rendering.gaussian.mtgs_live.time.perf_counter", side_effect=[1., 1.05]):
            controller.run()
        self.assertAlmostEqual(waits[0], .15)
        self.assertEqual(session.current["tick"], 1)

    def test_failed_step_and_reset_cannot_keep_autoplay_running(self):
        session = FakeSession()
        controller = LiveController(session)
        controller.control({"operation": "play"})
        session.fail_step = True
        with self.assertRaisesRegex(RuntimeError, "Renderer"):
            controller.control({"operation": "step"})
        self.assertFalse(controller.state()["playing"])
        self.assertIn("Renderer", controller.state()["error"])
        with self.assertRaisesRegex(ValueError, "Reset"):
            controller.control({"operation": "play"})
        session.fail_reset = True
        with self.assertRaisesRegex(RuntimeError, "Reset renderer"):
            controller.control({"operation": "reset"})
        self.assertIn("Reset renderer", controller.state()["error"])
        session.fail_reset = False
        controller.control({"operation": "reset"})
        self.assertIsNone(controller.state()["error"])

    def test_control_from_previous_reset_generation_cannot_mutate_new_episode(self):
        controller = LiveController(FakeSession())
        controller.control({"operation": "reset", "expected_generation": 0})
        with self.assertRaisesRegex(ValueError, "Episode changed"):
            controller.control({"operation": "step", "expected_generation": 0})
        self.assertEqual(controller.state()["world_tick"], 0)

    def test_invalid_physics_clock_is_rejected(self):
        for dt in [None, 0, -1, float("inf"), float("nan"), True]:
            with self.subTest(dt=dt), self.assertRaisesRegex(ValueError, "physics dt"):
                LiveController(FakeSession(dt=dt))

    def test_reset_failure_uses_world_generation_not_stale_image_generation(self):
        class StaleFrameSession(FakeSession):
            def state(self):
                return {**super().state(), "generation": 0, "world_generation": 2}

        controller = LiveController(StaleFrameSession())
        with self.assertRaisesRegex(ValueError, "Episode changed"):
            controller.control({"operation": "reset", "expected_generation": 0})
        controller.control({"operation": "reset", "expected_generation": 2})


class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "session"
        self.root.mkdir()
        (self.root / "frame.png").write_bytes(b"png-fixture")
        (Path(self.temp.name) / "private.png").write_bytes(b"private")
        (self.root / "private.json").write_text("private", encoding="utf-8")
        self.session = FakeSession(self.root)
        self.server = make_live_server(self.session)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.close_server)
        status, raw, _ = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.token = json.loads(raw.decode().split("const CONFIG=", 1)[1].split(",$=", 1)[0])["token"]

    def close_server(self):
        self.server.shutdown()
        self.thread.join(5)
        self.server.server_close()

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=3)
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            return response.status, response.read(), dict(response.getheaders())
        finally:
            connection.close()

    def control(self, payload, **headers):
        return self.request("POST", "/control", json.dumps(payload),
                            {"Content-Type": "application/json", "X-DriveRSI-Token": self.token,
                             **headers})

    def test_state_requires_token_and_nonascii_tokens_fail_closed(self):
        for headers in [{}, {"X-DriveRSI-Token": "wrong"}, {"X-DriveRSI-Token": "é"}]:
            self.assertEqual(self.request("GET", "/state", headers=headers)[0], 403)
        self.assertEqual(self.request("GET", "/state",
                                     headers={"X-DriveRSI-Token": self.token})[0], 200)

    def test_cross_origin_or_rebinding_requests_never_mutate_world(self):
        for headers in [{"Origin": "https://foreign.invalid"}, {"Origin": "null"},
                        {"Host": "foreign.invalid"}, {"Sec-Fetch-Site": "cross-site"}]:
            with self.subTest(headers=headers):
                self.assertEqual(self.control({"operation": "step"}, **headers)[0], 403)
        self.assertEqual(self.session.current["tick"], 0)
        self.assertEqual(self.control({"operation": "step"})[0], 200)
        self.assertEqual(self.session.current["tick"], 1)

    def test_malformed_bodies_are_errors_not_world_actions(self):
        cases = [(b"{", {}), (b"[]", {}), (b"x" * 4097, {}),
                 (b'{"operation":"step"}', {"Transfer-Encoding": "chunked"}),
                 (b'{"operation":"step"}', {"Content-Type": "text/plain"})]
        for body, extra in cases:
            status, _, _ = self.request("POST", "/control", body,
                                       {"Content-Type": "application/json",
                                        "X-DriveRSI-Token": self.token, **extra})
            self.assertEqual(status, 400)
        self.assertEqual(self.session.current["tick"], 0)

    def test_artifacts_remain_inside_root_and_do_not_expose_other_files(self):
        for path, expected in [("/artifacts/frame.png", 200),
                               ("/artifacts/%2e%2e/private.png", 404),
                               ("/artifacts/private.json", 404),
                               ("/artifacts/none.png", 404),
                               ("/artifacts/%00.png", 400),
                               ("/artifacts/%5c..%5cprivate.png", 400)]:
            with self.subTest(path=path):
                status, _, headers = self.request("GET", path)
                self.assertEqual(status, expected)
                self.assertEqual(headers["Cross-Origin-Resource-Policy"], "same-origin")
        self.assertEqual(self.request("GET", "/artifacts/frame.png",
                                     headers={"Sec-Fetch-Site": "cross-site"})[0], 403)

    def test_step_failure_is_visible_without_fabricating_a_later_tick(self):
        self.session.fail_step = True
        status, raw, _ = self.control({"operation": "step"})
        self.assertEqual(status, 500)
        self.assertIn("Renderer", json.loads(raw)["error"])
        _, raw, _ = self.request("GET", "/state", headers={"X-DriveRSI-Token": self.token})
        state = json.loads(raw)
        self.assertEqual(state["world_tick"], 0)
        self.assertFalse(state["playing"])
        self.assertIn("Renderer", state["error"])


if __name__ == "__main__":
    unittest.main()
