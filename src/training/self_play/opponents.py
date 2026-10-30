"""Explicit immutable identities for stateless frozen structured opponents.

Manifests select trusted local inference code. They never infer imports from model
files, download weights, or claim driving quality from interface compatibility.
"""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from driversi.agents.interfaces.config import load_inference, normalize_agent, read_agent_config
from driversi.configuration import read_mapping
from driversi.contracts.learning import ACTION_SIZE, OBSERVATION_SIZE
from driversi.data.assets.mtgs_assets import file_sha256
from driversi.data.scenes.scene_store import source_inventory
from driversi.training.sampling.environment import validate_development_paths


class FrozenOpponentPool:
    """Uniform pool ordering is part of checkpoint and behavior-index identity."""
    def __init__(self, manifest):
        if isinstance(manifest, FrozenOpponentPool):
            raise TypeError("Reuse an existing FrozenOpponentPool directly")
        if isinstance(manifest, (str, Path)):
            path = Path(manifest).resolve()
            raw, base = read_mapping(path), path.parent
        elif isinstance(manifest, dict):
            raw, base = copy.deepcopy(manifest), Path.cwd()
        else:
            raise TypeError("Opponent manifest must be a path or mapping")
        version = raw.get("schema_version")
        fields = {"schema_version", "policies"} | ({"sampling"} if version == 2 else set())
        if (set(raw) != fields or type(version) is not int
                or version not in (1, 2) or not isinstance(raw["policies"], list)
                or not raw["policies"]):
            raise ValueError("Invalid opponent pool schema")
        rows, policies, ids, source_hashes = [], [], set(), {}
        for row in raw["policies"]:
            if not isinstance(row, dict) or set(row) != {"id", "checkpoint", "agent_config", "training_paths"}:
                raise ValueError("Opponent entries require id, checkpoint, agent_config and training_paths")
            name = row["id"]
            if not isinstance(name, str) or not name.strip() or name in ids:
                raise ValueError("Opponent IDs must be nonempty and unique")
            ids.add(name)
            def resolve(value):
                if not isinstance(value, (str, Path)) or not str(value):
                    raise ValueError("Opponent file paths must be nonempty")
                return (base / value).resolve()
            checkpoint = resolve(row["checkpoint"])
            if checkpoint.suffix != ".npz":
                raise ValueError("Frozen opponent checkpoints must be NPZ inference exports")
            config = (normalize_agent(row["agent_config"]) if isinstance(row["agent_config"], dict)
                      else read_agent_config(resolve(row["agent_config"])))
            if not isinstance(row["training_paths"], list) or not row["training_paths"]:
                raise ValueError("Declared nonempty opponent training_paths required")
            paths = [str(resolve(value)) for value in row["training_paths"]]
            groups = sorted(validate_development_paths(paths))
            inventory = source_inventory(paths, .02)
            for path in paths:
                if path not in source_hashes:
                    source_hashes[path] = file_sha256(Path(path))
            before = file_sha256(checkpoint)
            policy = load_inference(checkpoint, config)
            if file_sha256(checkpoint) != before:
                raise ValueError("Opponent checkpoint changed while loading")
            # Existing NPZ backends expose independent NumPy weights. Explicit
            # read-only arrays prevent accidental training/alias mutation.
            for value in getattr(policy, "weights", {}).values():
                if isinstance(value, np.ndarray):
                    value.flags.writeable = False
            rows.append({"id": name, "checkpoint": str(checkpoint), "agent_config": config,
                         "training_paths": paths, "training_source_groups": groups,
                         "training_inventory": inventory,
                         "training_sha256": {path: source_hashes[path] for path in paths},
                         "policy_version": policy.metadata["policy_version"],
                         "checkpoint_sha256": before})
            policies.append(policy)
        self._policies = tuple(policies)
        self._rows = tuple(rows)
        self.training_paths = tuple(dict.fromkeys(path for row in rows for path in row["training_paths"]))
        payload = {"schema_version": 1, "sampling": "uniform_episode_fixed", "policies": rows}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                           allow_nan=False).encode()).hexdigest()
        self._identity = {**payload, "sha256": digest}
        self._probabilities = None
        if version == 2:
            from driversi.training.self_play.opponent_priority import priority_from_report
            sampling = raw["sampling"]
            required = {"method", "matchup_report", "uniform_mix", "power", "min_actor_episodes"}
            if not isinstance(sampling, dict) or set(sampling) != required or sampling["method"] != "failure_priority_v1":
                raise ValueError("Invalid frozen priority sampling configuration")
            report_path = resolve(sampling["matchup_report"])
            sampling_identity = priority_from_report(read_mapping(report_path), self._identity,
                **{k: sampling[k] for k in ("uniform_mix", "power", "min_actor_episodes")})
            self._probabilities = np.array([r["probability"] for r in sampling_identity["evidence"]])
            payload = {"schema_version": 2, "sampling": sampling_identity, "policies": rows}
            digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                               allow_nan=False).encode()).hexdigest()
            self._identity = {**payload, "sha256": digest}
        for index in range(len(rows)):
            self.predict(index, np.zeros((2, OBSERVATION_SIZE), dtype=np.float32))

    @property
    def policies(self):
        return copy.deepcopy(self._rows)

    @property
    def identity(self):
        return copy.deepcopy(self._identity)

    def __len__(self):
        return len(self._policies)

    def sample_indices(self, rng, count):
        """One-based behavior indices; legacy uniform draw sequence is unchanged."""
        if type(count) is not int or count < 0:
            raise ValueError("Nonnegative opponent count required")
        if self._probabilities is None:
            return rng.integers(1, len(self) + 1, count)
        return rng.choice(len(self), size=count, p=self._probabilities) + 1

    def predict(self, index, observations):
        if type(index) is not int or not 0 <= index < len(self):
            raise ValueError("Invalid opponent index")
        x = np.array(observations, dtype=np.float32, copy=True)
        if x.ndim != 2 or x.shape[1] != OBSERVATION_SIZE or not np.isfinite(x).all():
            raise ValueError("Invalid opponent observations")
        x.flags.writeable = False
        logits, values = self._policies[index].predict(x)
        logits, values = np.asarray(logits), np.asarray(values)
        if (logits.shape != (len(x), ACTION_SIZE) or values.shape != (len(x),)
                or logits.dtype.kind not in "fiu" or values.dtype.kind not in "fiu"
                or not np.isfinite(logits).all() or not np.isfinite(values).all()):
            raise ValueError("Opponent predict must return finite [batch,12] logits and [batch] values")
        return logits.copy(), values.copy()

    def behavior_policies(self, learner_version):
        if not isinstance(learner_version, str) or not learner_version.strip():
            raise ValueError("Nonempty learner policy version required")
        return [{"index": 0, "role": "learner", "policy_version": learner_version}, *[
            {"index": index + 1, "role": "frozen_opponent", "id": row["id"],
             "policy_version": row["policy_version"], "checkpoint": row["checkpoint"],
             "checkpoint_sha256": row["checkpoint_sha256"], "agent_config": copy.deepcopy(row["agent_config"])}
            for index, row in enumerate(self._rows)]]
