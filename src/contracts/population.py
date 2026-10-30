"""Immutable, versioned policy assignments for one synchronous episode.

Only declared engineering controllers are accepted. External policy state stays
with the caller; this contract does not make external rollouts training-ready.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

_OBSERVATIONS = {
    "route-idm-v2": "metric-v1",
    "route-conflict-v1": "metric-v1",
    "external-joint-v1": "external-unverified",
    "external-native-state-v1": "native-integrator-state-v1",
}
_ROLES = {"participant", "candidate", "reference_opponent"}


def _text(value, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value


@dataclass(frozen=True)
class PopulationPolicy:
    policy_id: str
    version: str
    controller: str
    parameters: tuple[tuple[str, float], ...] = ()

    def __post_init__(self):
        _text(self.policy_id, "policy_id")
        _text(self.version, "version")
        _text(self.controller, "controller")
        if self.controller not in _OBSERVATIONS:
            raise ValueError("Unsupported population controller")
        parameters = tuple(sorted(tuple(item) for item in self.parameters))
        if any(len(item) != 2 for item in parameters) or len({key for key, _ in parameters}) != len(
            parameters
        ):
            raise ValueError("Policy parameters must have unique names")
        for key, value in parameters:
            _text(key, "parameter name")
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("Policy parameters must be finite numbers, not booleans")
        if self.controller in {"external-joint-v1", "external-native-state-v1"} and parameters:
            raise ValueError("External policies do not accept controller parameters")
        object.__setattr__(self, "parameters", parameters)

    @property
    def observation_version(self) -> str:
        return _OBSERVATIONS[self.controller]

    def to_dict(self) -> dict:
        return {
            "policy_id": self.policy_id,
            "version": self.version,
            "controller": self.controller,
            "parameters": dict(self.parameters),
            "observation_version": self.observation_version,
            "trainable": False,
        }


@dataclass(frozen=True)
class ActorPolicyBinding:
    actor_id: str
    policy_id: str
    role: str = "participant"

    def __post_init__(self):
        _text(self.actor_id, "actor_id")
        _text(self.policy_id, "policy_id")
        _text(self.role, "role")
        if self.role not in _ROLES:
            raise ValueError("Unknown population role")

    def to_dict(self) -> dict:
        return {"actor_id": self.actor_id, "policy_id": self.policy_id, "role": self.role}


@dataclass(frozen=True)
class EpisodePopulation:
    policies: tuple[PopulationPolicy, ...]
    assignments: tuple[ActorPolicyBinding, ...]

    def __post_init__(self):
        policies = tuple(sorted(self.policies, key=lambda item: item.policy_id))
        assignments = tuple(sorted(self.assignments, key=lambda item: item.actor_id))
        policy_ids = {item.policy_id for item in policies}
        actor_ids = {item.actor_id for item in assignments}
        if (
            not policies
            or not assignments
            or len(policy_ids) != len(policies)
            or len(actor_ids) != len(assignments)
        ):
            raise ValueError("Population requires unique policies and actor assignments")
        if any(item.policy_id not in policy_ids for item in assignments):
            raise ValueError("Assignment references an unknown policy")
        object.__setattr__(self, "policies", policies)
        object.__setattr__(self, "assignments", assignments)

    @property
    def behavior_policies(self) -> dict:
        policies = {item.policy_id: item for item in self.policies}
        return {
            binding.actor_id: {
                "policy_id": binding.policy_id,
                "version": policies[binding.policy_id].version,
                "controller": policies[binding.policy_id].controller,
                "role": binding.role,
                "observation_version": policies[binding.policy_id].observation_version,
                "trainable": False,
            }
            for binding in self.assignments
        }

    @property
    def external_actor_ids(self) -> tuple[str, ...]:
        return tuple(
            actor
            for actor, policy in self.behavior_policies.items()
            if policy["controller"] == "external-joint-v1"
        )

    def validate_actors(self, actor_ids: tuple[str, ...]) -> None:
        if len(set(actor_ids)) != len(actor_ids) or set(actor_ids) != {
            item.actor_id for item in self.assignments
        }:
            raise ValueError("Population must cover exactly every controlled actor")

    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "policies": [item.to_dict() for item in self.policies],
            "assignments": [item.to_dict() for item in self.assignments],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "EpisodePopulation":
        if (
            not isinstance(payload, dict)
            or set(payload) != {"schema_version", "policies", "assignments"}
            or type(payload["schema_version"]) is not int
            or payload["schema_version"] != 1
            or not isinstance(payload["policies"], list)
            or not isinstance(payload["assignments"], list)
        ):
            raise ValueError("Invalid population schema")
        policies = []
        for item in payload["policies"]:
            required = {"policy_id", "version", "controller", "parameters"}
            if (
                not isinstance(item, dict)
                or not required <= set(item)
                or set(item) - required - {"observation_version", "trainable"}
                or not isinstance(item["parameters"], dict)
            ):
                raise ValueError("Invalid population policy fields")
            policy = PopulationPolicy(
                item["policy_id"],
                item["version"],
                item["controller"],
                tuple(item["parameters"].items()),
            )
            if (
                item.get("observation_version", policy.observation_version)
                != policy.observation_version
                or item.get("trainable", False) is not False
            ):
                raise ValueError("Policy observation or training declaration is unsupported")
            policies.append(policy)
        assignments = []
        for item in payload["assignments"]:
            if not isinstance(item, dict) or set(item) != {"actor_id", "policy_id", "role"}:
                raise ValueError("Invalid population assignment fields")
            assignments.append(ActorPolicyBinding(**item))
        return cls(tuple(policies), tuple(assignments))

    @classmethod
    def load(cls, path: Path) -> "EpisodePopulation":
        def pairs(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise ValueError(f"Duplicate population JSON field: {key}")
                result[key] = value
            return result

        return cls.from_dict(json.loads(Path(path).read_text(), object_pairs_hook=pairs))
