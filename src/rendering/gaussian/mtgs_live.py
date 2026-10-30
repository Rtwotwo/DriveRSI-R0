"""Loopback-only, single-session controls for the world-driven renderer."""

from __future__ import annotations

import copy
import json
import math
import mimetypes
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


class LiveController:
    def __init__(self, session):
        self.session = session
        self.lock = threading.RLock()
        self.playing = False
        self.error = None
        self.brake = None
        self.stop = threading.Event()
        initial = session.state()
        self.dt = initial.get("dt", initial.get("dt_seconds"))
        if type(self.dt) not in (int, float) or not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("Live playback requires a positive finite physics dt")

    def state(self):
        with self.lock:
            return copy.deepcopy({**self.session.state(), "playing": self.playing,
                                  "error": self.error, "scheduled_brake": self.brake})

    def _step(self):
        overrides = {}
        if self.brake:
            if self.session.current["tick"] < self.brake["until_tick"]:
                overrides[self.brake["actor_id"]] = 1
            else:
                self.brake = None
        try:
            self.session.advance(overrides)
        except Exception as error:
            self.playing = False
            self.error = str(error)
            raise
        if self.session.current["done"]:
            self.playing = False

    def control(self, value: dict):
        with self.lock:
            if not isinstance(value, dict):
                raise ValueError("Control must be an object")
            operation = value.get("operation")
            if operation not in {"play", "pause", "step", "reset", "brake"}:
                raise ValueError("Unknown control")
            allowed = {"operation", "expected_generation"}
            if operation == "brake":
                allowed.add("actor_id")
            if set(value)-allowed:
                raise ValueError("Unexpected control fields")
            snapshot = self.session.state()
            generation = snapshot.get("world_generation", snapshot.get("generation"))
            if ("expected_generation" in value
                    and value["expected_generation"] != generation):
                raise ValueError("Episode changed; refresh the state before controlling it")
            if operation == "pause":
                self.playing = False
            elif operation == "reset":
                self.playing = False
                try:
                    self.session.reset()
                except Exception as error:
                    self.error = str(error)
                    raise
                self.error, self.brake = None, None
            elif operation == "brake":
                actor = value.get("actor_id")
                if actor not in self.session.current["controlled_ids"]:
                    raise ValueError("Brake requires a controlled actor ID")
                if self.session.current["done"] or self.error:
                    raise ValueError("Reset the ended/failed episode first")
                duration_ticks = max(1, math.ceil(1.0 / self.dt - 1e-9))
                self.brake = {"actor_id": actor,
                              "until_tick": self.session.current["tick"]+duration_ticks}
            else:
                if self.session.current["done"] or self.error:
                    raise ValueError("Reset the ended/failed episode first")
                if operation == "step":
                    self.playing = False
                    self._step()
                else:
                    self.playing = True
            return self.state()

    def run(self):
        while not self.stop.is_set():
            started = time.perf_counter()
            with self.lock:
                if self.playing:
                    try:
                        self._step()
                    except Exception:
                        pass  # Error has been published; no silent replay fallback.
            self.stop.wait(max(0.001, self.dt-(time.perf_counter()-started)))


def make_live_server(session, *, port: int = 0):
    from driversi.rendering.gaussian.mtgs_live_viewer import live_html

    controller = LiveController(session)
    token = secrets.token_urlsafe(32)
    root = Path(session.output).resolve(strict=True)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def reply(self, status, data, content_type="application/json; charset=utf-8"):
            if isinstance(data, (dict, list)):
                data = json.dumps(data, ensure_ascii=False, allow_nan=False).encode()
            elif isinstance(data, str):
                data = data.encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
            self.send_header("X-Frame-Options", "DENY")
            self.end_headers()
            self.wfile.write(data)

        def valid_host(self):
            return len(self.headers.get_all("Host", [])) == 1 and self.headers.get("Host") in {
                f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}

        def same_origin(self):
            origin = self.headers.get("Origin")
            return (self.valid_host()
                    and len(self.headers.get_all("Origin", [])) <= 1
                    and self.headers.get("Sec-Fetch-Site") != "cross-site"
                    and (origin is None or origin == "http://"+self.headers["Host"]))

        def valid_token(self):
            if len(self.headers.get_all("X-DriveRSI-Token", [])) != 1:
                return False
            try:
                supplied = self.headers["X-DriveRSI-Token"].encode("ascii")
            except UnicodeEncodeError:
                return False
            return secrets.compare_digest(supplied, token.encode("ascii"))

        def do_GET(self):
            if not self.same_origin():
                self.reply(403, {"error": "Same-origin loopback request required"})
                return
            try:
                path = unquote(urlparse(self.path).path, errors="strict")
                if "\x00" in path or "\\" in path:
                    raise ValueError("Invalid artifact path")
            except ValueError:
                self.reply(400, {"error": "Invalid request path"})
                return
            if path == "/":
                self.reply(200, live_html(token), "text/html; charset=utf-8")
            elif path == "/state":
                if not self.valid_token():
                    self.reply(403, {"error": "Session token required"})
                    return
                self.reply(200, controller.state())
            elif path.startswith("/artifacts/"):
                try:
                    target = (root/path.removeprefix("/artifacts/")).resolve()
                except (OSError, ValueError):
                    self.reply(404, {"error": "Unknown image artifact"})
                    return
                if (not target.is_relative_to(root) or not target.is_file()
                        or target.suffix.lower() != ".png"):
                    self.reply(404, {"error": "Unknown image artifact"})
                    return
                try:
                    data = target.read_bytes()
                except OSError:
                    self.reply(404, {"error": "Image artifact is unavailable"})
                    return
                self.reply(200, data, mimetypes.guess_type(target)[0] or "image/png")
            else:
                self.reply(404, {"error": "Unknown endpoint"})

        def do_POST(self):
            if not self.same_origin() or not self.valid_token():
                self.reply(403, {"error": "Same-origin session token required"})
                return
            if self.path != "/control":
                self.reply(404, {"error": "Unknown endpoint"})
                return
            try:
                if (self.headers.get("Transfer-Encoding") is not None
                        or len(self.headers.get_all("Content-Length", [])) != 1):
                    raise ValueError("One fixed JSON body length is required")
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 4096 or self.headers.get_content_type() != "application/json":
                    raise ValueError("Small JSON control body required")
                self.connection.settimeout(5)
                body = self.rfile.read(size)
                if len(body) != size:
                    raise ValueError("Incomplete JSON body")
                value = json.loads(body)
                self.reply(200, controller.control(value))
            except (ValueError, TypeError, TimeoutError) as error:
                self.reply(400, {"error": str(error)})
            except Exception as error:
                self.reply(500, {"error": str(error)})

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    server.controller = controller
    return server


def serve_session(session, *, port: int):
    server = make_live_server(session, port=port)
    controller = server.controller
    pump = threading.Thread(target=controller.run, daemon=True)
    pump.start()
    print(json.dumps({"kind": "mtgs_live_session", "url": f"http://127.0.0.1:{server.server_port}",
                      "output": str(session.output), "single_session": True}), flush=True)
    try:
        server.serve_forever(poll_interval=.2)
    finally:
        controller.stop.set()
        pump.join(timeout=30)
        server.server_close()
