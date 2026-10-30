"""Explicit lane and branch intentions, separate from observed state and log futures."""

import json
from dataclasses import dataclass, field
from pathlib import Path

from driversi.contracts.scene import Scene


@dataclass(frozen=True)
class RouteTasks:
    scenario_id: str
    source_log: str
    initial_nodes: dict[str, str] = field(default_factory=dict)
    purpose: str = "explicit engineering intentions; not inferred human intent"
    successors: dict[str, dict[str, str]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "RouteTasks":
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.pop("schema_version", None) != 1:
            raise ValueError("Unsupported route task schema")
        return cls(**value)

    def validate_scene(self, scene: Scene) -> None:
        if self.scenario_id != scene.metadata["scenario_id"] or self.source_log != scene.metadata["source_log"]:
            raise ValueError("Route tasks belong to a different source scene")
        controlled = {actor for actor, active in zip(scene.actor_ids, scene.controlled) if active}
        if not isinstance(self.initial_nodes, dict) or not set(self.initial_nodes) <= controlled:
            raise ValueError("Route task IDs must be controlled actors in this scene")

        def valid_node(node):
            return isinstance(node, str) and node.startswith(("lane:", "connector:")) and bool(node.split(":", 1)[1])

        if any(not valid_node(node) for node in self.initial_nodes.values()):
            raise ValueError("Route tasks require lane/connector node IDs")
        if not isinstance(self.successors, dict) or not set(self.successors) <= controlled:
            raise ValueError("Successor task IDs must be controlled actors in this scene")
        for choices in self.successors.values():
            if not isinstance(choices, dict) or any(not valid_node(k) or not valid_node(v) for k, v in choices.items()):
                raise ValueError("Successor tasks require lane/connector node ID mappings")

    def to_dict(self) -> dict:
        return {"schema_version": 1, "scenario_id": self.scenario_id, "source_log": self.source_log,
                "initial_nodes": dict(self.initial_nodes), "purpose": self.purpose,
                "successors": {actor: dict(choices) for actor, choices in self.successors.items()}}
