"""Read-only recorded world for camera re-rendering, never a training simulator."""

import copy
import hashlib
import json
from pathlib import Path


class RecordedWorldWorker:
    def __init__(self, *, manifest, scene, road_graph, **unused):
        self.path = Path(manifest)
        record = json.loads(self.path.read_text())
        for name, path in (("scene", scene), ("road_graph", road_graph)):
            if (
                hashlib.sha256(Path(path).read_bytes()).hexdigest()
                != record["inputs"][name]["sha256"]
            ):
                raise ValueError("Recorded world input mismatch: " + name)
        initial = record["initial_world_by_generation"]
        if len(initial) != 1 or len({x["generation"] for x in record["steps"]}) != 1:
            raise ValueError("Re-render exactly one recorded generation")
        self.current = copy.deepcopy(next(iter(initial.values())))
        if {x["generation"] for x in record["steps"]} != {int(next(iter(initial)))}:
            raise ValueError("Recorded generation mismatch")
        self.steps = copy.deepcopy(record["steps"])
        ticks = [self.current["tick"]] + [x["tick"] for x in self.steps]
        if any(b != a + 1 for a, b in zip(ticks, ticks[1:])):
            raise ValueError("Incomplete recorded action sequence")
        self.index = 0
        self.closed = False

    def request(self, operation, *, expected_tick, overrides=None, **kwargs):
        if self.closed or operation != "step" or overrides or kwargs:
            raise ValueError(
                "Recorded world cannot simulate, intervene, reset or accept new actions"
            )
        if expected_tick != self.current["tick"] or self.index >= len(self.steps):
            raise ValueError("Stale or exhausted recorded world")
        self.current.update(copy.deepcopy(self.steps[self.index]))
        self.index += 1
        return copy.deepcopy(self.current)

    def close(self):
        self.closed = True
