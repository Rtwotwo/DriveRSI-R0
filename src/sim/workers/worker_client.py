"""One persistent local physics process; requests always concern a whole world."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
from pathlib import Path


class WorldWorker:
    def __init__(self, *, python: str | Path, scene: str | Path, road_graph: str | Path,
                 native_path: str | Path, route_tasks: str | Path | None = None,
                 desired_speed: float = 8., timeout: float = 120.,
                 population_config: str | Path | None = None):
        self.timeout = timeout
        self._serial = 0
        self._responses: queue.Queue = queue.Queue()
        environment = os.environ.copy()
        environment["DRIVERSI_NATIVE_PATH"] = str(Path(native_path).resolve(strict=True))
        # The selected interpreter must have this project installed (editable or wheel).
        executable = Path(python).absolute()
        if not executable.is_file():
            raise ValueError(f"Physics Python does not exist: {executable}")
        # Resolving a venv's interpreter symlink would discard its environment.
        command = [str(executable), "-m", "driversi.sim.workers.worker",
                   "--scene", str(Path(scene).resolve(strict=True)),
                   "--road-graph", str(Path(road_graph).resolve(strict=True)),
                   "--desired-speed", str(desired_speed)]
        if route_tasks is not None:
            command += ["--route-tasks", str(Path(route_tasks).resolve(strict=True))]
        if population_config is not None:
            command += ["--population-config", str(Path(population_config).resolve(strict=True))]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        text=True, encoding="utf-8", bufsize=1, env=environment)
        threading.Thread(target=self._read_responses, daemon=True).start()
        try:
            self.current = self._receive()
            if self.current.get("status") != "ready":
                raise RuntimeError(f"Physics worker did not initialize: {self.current}")
        except BaseException:
            self.close()
            raise

    def _read_responses(self):
        try:
            for line in self.process.stdout:
                self._responses.put(json.loads(line))
        except Exception as error:
            self._responses.put(error)
        finally:
            self._responses.put(EOFError("Physics worker closed its response stream"))

    def _receive(self) -> dict:
        try:
            response = self._responses.get(timeout=self.timeout)
        except queue.Empty as error:
            raise TimeoutError("Physics worker response timed out") from error
        if isinstance(response, BaseException):
            raise RuntimeError("Physics worker communication failed") from response
        if not isinstance(response, dict) or response.get("schema_version") != 1:
            raise RuntimeError("Invalid physics response schema")
        return response

    def request(self, operation: str, **values) -> dict:
        self._serial += 1
        request = {"operation": operation, "request_id": self._serial, **values}
        self.process.stdin.write(json.dumps(request, allow_nan=False) + "\n")
        self.process.stdin.flush()
        response = self._receive()
        if response.get("request_id") != self._serial or response.get("operation") != operation:
            raise RuntimeError("Out-of-order physics response")
        if response.get("status") == "error":
            raise RuntimeError(f"Physics request failed: {response.get('error')}")
        self.current = response
        return response

    def close(self):
        process = getattr(self, "process", None)
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        if process.stdin:
            process.stdin.close()
        if process.stdout:
            process.stdout.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
