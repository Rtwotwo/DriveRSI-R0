"""Action-conditioned structured world sequences from actual closed-loop rollouts.

History, planned joint commands, and future labels have distinct accessors.
Replay actors remain exogenous; these records are not a complete simulator state.
"""
import argparse
import copy
import json
import re
from collections import OrderedDict, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from driversi.contracts.driving_task import tasks_from_metadata
from driversi.contracts.scene import MAX_WORLD_ACTORS, Scene
from driversi.contracts.visual import frozen_array
from driversi.data.scenes.scene_store import LazyScenes, rollout_sources

STATE_VERSION = "puffer-state13-motion3-v1"
STATE_COLUMNS = ("x", "y", "z", "yaw", "vx", "vy", "vz", "goal_x", "goal_y",
                 "stopped", "collision", "offroad", "reached_goal", "a_long", "a_lat", "steering")


MIXED_FIELDS = ("learner_mask", "behavior_policy_index", "on_policy_eligible")


def validate_behavior_table(table, learner_version):
    """Validate declared behavior identities without importing models or reading weights."""
    if not isinstance(table, list) or len(table) < 2:
        raise ValueError("Mixed behavior requires a learner and frozen opponent table")
    names = set()
    for index, row in enumerate(table):
        expected = ({"index", "role", "policy_version"} if index == 0 else
                    {"index", "role", "id", "policy_version", "checkpoint", "checkpoint_sha256", "agent_config"})
        if (not isinstance(row, dict) or set(row) != expected or type(row.get("index")) is not int
                or row["index"] != index or not isinstance(row.get("policy_version"), str)
                or not row["policy_version"].strip()):
            raise ValueError("Invalid mixed behavior table entry")
        if index == 0:
            if row["role"] != "learner" or row["policy_version"] != learner_version:
                raise ValueError("Mixed learner behavior version mismatch")
        elif (row["role"] != "frozen_opponent" or not isinstance(row["id"], str)
              or not row["id"].strip() or row["id"] in names
              or not isinstance(row["checkpoint"], str) or not Path(row["checkpoint"]).is_absolute()
              or not isinstance(row["checkpoint_sha256"], str)
              or re.fullmatch(r"[0-9a-f]{64}", row["checkpoint_sha256"]) is None
              or not isinstance(row["agent_config"], dict)):
            raise ValueError("Invalid frozen opponent identity")
        if index:
            names.add(row["id"])
    return copy.deepcopy(table)


def rollout_behavior(metadata, data):
    """Return checked mixed-role masks; legacy single-policy batches stay unchanged."""
    declared = ("behavior_policies" in metadata or "sampling_mode" in metadata
                or any(key in data for key in MIXED_FIELDS))
    if not declared:
        return None
    if (metadata.get("sampling_mode") != "frozen_history_uniform"
            or any(key not in data for key in MIXED_FIELDS)):
        raise ValueError("Incomplete mixed behavior record")
    table = validate_behavior_table(metadata.get("behavior_policies"), metadata.get("policy_version"))
    valid = data["valid"]
    learner, index, eligible = (np.asarray(data[key]) for key in MIXED_FIELDS)
    if (valid.dtype != bool or valid.ndim != 2
            or any(value.shape != valid.shape for value in (learner, index, eligible))
            or learner.dtype != bool or eligible.dtype != bool or index.dtype.kind not in "iu"
            or (index < -1).any() or (index >= len(table)).any()
            or (valid & (index < 0)).any() or not np.array_equal(learner, index == 0)
            or (eligible & ~valid).any()):
        raise ValueError("Invalid mixed behavior indices or learning masks")
    return table, index, valid & learner & eligible


@dataclass(frozen=True)
class JointSequence:
    metadata: dict
    states: np.ndarray  # [T+1, all_actor, 16]; unavailable fields zero, never future padding
    actions: np.ndarray  # [T, all_actor]; controlled commands, -1 for exogenous replay
    action_applied: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    events: np.ndarray
    controlled: np.ndarray
    state_field_valid: np.ndarray
    scene_context: dict
    behavior_policy_index: np.ndarray | None = None
    learning_eligible: np.ndarray | None = None
    presence: np.ndarray | None = None
    route_support_end: np.ndarray | None = None

    def __post_init__(self):
        m = self.metadata
        c, p = m.get("context_steps"), m.get("prediction_steps")
        ids = tuple(m.get("actor_ids", ()))
        if (m.get("schema_version") not in (1, 2) or m.get("state_version") != STATE_VERSION
                or any(type(v) is not int or v < 1 for v in (c, p))
                or not ids or len(set(ids)) != len(ids) or m.get("dt_seconds") != .1):
            raise ValueError("Invalid world-model sequence contract")
        length, count = c + p, len(ids)
        presence = np.ones((length+1,count),bool) if self.presence is None else np.asarray(self.presence)
        if presence.shape != (length+1,count) or presence.dtype != bool or not presence[:,self.controlled].all():
            raise ValueError("Invalid sequence presence")
        object.__setattr__(self,"presence",frozen_array(presence))
        shapes = {"states": (length + 1, count, 16), "actions": (length, count),
                  "action_applied": (length, count), "rewards": (length, count),
                  "terminated": (length, count), "truncated": (length, count),
                  "events": (length, count, 3), "controlled": (count,), "state_field_valid": (count, 16)}
        for name, shape in shapes.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"Invalid sequence {name}")
        for name in ("action_applied", "terminated", "truncated", "events", "controlled", "state_field_valid"):
            if np.asarray(getattr(self, name)).dtype != bool:
                raise ValueError("Sequence masks must be boolean")
        if (self.actions.dtype.kind not in "iu" or not self.controlled.any()
                or (self.actions[:, self.controlled] < 0).any()
                or (self.actions[:, self.controlled] >= 12).any()
                or (self.actions[:, ~self.controlled] != -1).any()
                or self.action_applied[:, ~self.controlled].any()
                or ((self.terminated | self.truncated) & ~self.action_applied).any()
                or (self.terminated & self.truncated).any()
                or (self.states[:, ~self.state_field_valid] != 0).any()):
            raise ValueError("Sequence action/mask semantics differ")
        if (type(m.get("start_tick")) is not int or m["start_tick"] < 0
                or m.get("end_tick") != m["start_tick"] + length
                or type(m.get("max_steps")) is not int or m["end_tick"] > m["max_steps"]
                or len(m.get("policy_versions", [])) != length):
            raise ValueError("Invalid sequence clock or behavior versions")
        if m["schema_version"] == 2:
            index, eligible = self.behavior_policy_index, self.learning_eligible
            tables = m.get("behavior_policies")
            if (not isinstance(index, np.ndarray) or index.shape != (length, count)
                    or index.dtype.kind not in "iu" or not isinstance(eligible, np.ndarray)
                    or eligible.shape != index.shape or eligible.dtype != bool
                    or not isinstance(tables, list) or len(tables) != length
                    or (index[:, ~self.controlled] != -1).any()
                    or (index[:, self.controlled] < 0).any()
                    or not np.all(index == index[0])
                    or (eligible & (~self.action_applied | (index != 0))).any()):
                raise ValueError("Invalid sequence mixed behavior metadata")
            versions = []
            for t, table in enumerate(tables):
                table = validate_behavior_table(table, m["policy_versions"][t])
                if table[1:] != tables[0][1:]:
                    raise ValueError("Frozen sequence opponent identities changed")
                if (index[t] >= len(table)).any():
                    raise ValueError("Sequence behavior index outside policy table")
                versions.append([None if i < 0 else table[int(i)]["policy_version"] for i in index[t]])
            if m.get("actor_policy_versions") != versions:
                raise ValueError("Sequence actor behavior versions differ from policy table")
            object.__setattr__(self, "behavior_policy_index", frozen_array(index))
            object.__setattr__(self, "learning_eligible", frozen_array(eligible))
        elif (self.behavior_policy_index is not None or self.learning_eligible is not None
              or "behavior_policies" in m or "actor_policy_versions" in m):
            raise ValueError("Mixed behavior requires sequence schema version 2")
        if self.route_support_end is not None:
            support = np.asarray(self.route_support_end)
            if support.shape != (length, count) or support.dtype != bool or (support & ~self.truncated).any():
                raise ValueError("Invalid sequence route support mask")
            object.__setattr__(self, "route_support_end", frozen_array(support))
        task_fields = {"driving_tasks", "task_goals_xy"} if "driving_tasks" in self.scene_context["metadata"] else set()
        if task_fields:
            tasks_from_metadata(self.scene_context["metadata"], [a for a,c in zip(ids,self.controlled) if c])
            if self.route_support_end is None:
                raise ValueError("Task sequence missing route support mask")
        allowed = {"metadata", "actor_ids", "kinds", "dimensions", "road_points", "road_offsets", "road_types"}
        if (set(self.scene_context) != allowed or tuple(self.scene_context["actor_ids"]) != ids
                or set(self.scene_context["metadata"]) !=
                ({"scenario_id", "source_log", "split", "map_name", "origin_xyz"} | task_fields)):
            raise ValueError("Scene context must contain only declared static fields")
        for name in ("states", "actions", "action_applied", "rewards", "terminated", "truncated",
                     "events", "controlled", "state_field_valid"):
            object.__setattr__(self, name, frozen_array(getattr(self, name)))

    def model_inputs(self) -> dict:
        c = self.metadata["context_steps"]
        return {"state_history": self.states[:c + 1], "presence_history": self.presence[:c+1], "action_history": self.actions[:c],
                "history_action_applied": self.action_applied[:c],
                "future_actions": self.actions[c:], "controlled": self.controlled,
                "state_field_valid": self.state_field_valid, "scene": self.scene_context,
                "elapsed_seconds": (self.metadata["start_tick"] + c) * .1,
                "remaining_seconds": (self.metadata["max_steps"] - self.metadata["start_tick"] - c) * .1}

    def training_targets(self) -> dict:
        c = self.metadata["context_steps"]
        result = {"future_states": self.states[c + 1:], "future_presence": self.presence[c+1:], "action_applied": self.action_applied[c:],
                  "rewards": self.rewards[c:], "terminated": self.terminated[c:],
                  "truncated": self.truncated[c:], "events": self.events[c:]}
        if self.route_support_end is not None:
            result["route_support_end"] = self.route_support_end[c:]
        if self.learning_eligible is not None:
            result["policy_learning_eligible"] = self.learning_eligible[c:]
        return result

    def save(self, path: Path) -> None:
        if path.exists():
            raise FileExistsError(path)
        arrays = {k: v for k, v in vars(self).items() if isinstance(v, np.ndarray)}
        np.savez_compressed(path, metadata=json.dumps(self.metadata), **arrays)


def static_context(scene: Scene) -> dict:
    keys = ("scenario_id", "source_log", "split", "map_name", "origin_xyz")
    if "driving_tasks" in scene.metadata:
        keys += ("driving_tasks", "task_goals_xy")
    return {"metadata": copy.deepcopy({k: scene.metadata[k] for k in keys}),
            "actor_ids": scene.actor_ids,
            **{k: frozen_array(getattr(scene, k)) for k in
               ("kinds", "dimensions", "road_points", "road_offsets", "road_types")}}


def iter_joint_sequences(run: Path, *, context_steps=4, prediction_steps=8, stride=4,
                         statistics=None):
    if any(type(v) is not int or v < 1 for v in (context_steps, prediction_steps, stride)):
        raise ValueError("Context, prediction and stride must be positive integers")
    run = Path(run)
    files = sorted((run / "rollouts").glob("batch-*.npz"))
    if not files:
        raise ValueError("No actual rollout batches (rank-sharded DDP sequence export is not supported)")
    stats = {} if statistics is None else statistics
    stats.update(joint_transitions=0, windows=0, reset_boundaries=0, short_segments=0)
    length = context_steps + prediction_steps
    history, previous, seen, source_paths = {}, {}, {}, None
    scenes, contexts, inventories = None, OrderedDict(), {}
    previous_batch = None
    world_count = None
    mixed_mode, frozen_table = None, None
    fields = ("actions", "valid", "terminated", "truncated", "scene_index", "episode", "tick",
              "world_before", "world_after", "world_motion_before", "world_motion_after",
              "world_actor_count", "events", "rewards")
    for file in files:
        batch_index = int(file.stem.split("-")[-1])
        if previous_batch is not None and batch_index != previous_batch + 1:
            raise ValueError("Missing or out-of-order rollout batch")
        previous_batch = batch_index
        with np.load(file, allow_pickle=False) as archive:
            meta = json.loads(str(archive["metadata"]))
            if (meta.get("joint_state_version") != STATE_VERSION or meta.get("dt_seconds") != .1
                    or any(k not in archive for k in fields)):
                raise ValueError("World-model input requires recorded motion and the declared 10Hz state version")
            data = {k: archive[k].copy() for k in fields}
            data.update({k: archive[k].copy() for k in (*MIXED_FIELDS,"world_presence_before","world_presence_after","route_support_end") if k in archive})
        behavior = rollout_behavior(meta, data)
        if mixed_mode is not None and mixed_mode != (behavior is not None):
            raise ValueError("Behavior recording mode changed within a run")
        mixed_mode = behavior is not None
        if behavior is not None:
            table, behavior_index, learning_eligible = behavior
            if frozen_table is not None and frozen_table != table[1:]:
                raise ValueError("Frozen opponent identities changed within a run")
            frozen_table = table[1:]
        meta = rollout_sources(meta, run, inventories)
        paths = meta["scene_paths"]
        if source_paths is not None and paths != source_paths:
            raise ValueError("Source scene order changed within a run")
        source_paths = paths
        if scenes is None:
            scenes = LazyScenes(paths, 32)
        before = data["world_before"]
        if before.ndim != 4 or not 1 <= before.shape[2] <= MAX_WORLD_ACTORS or before.shape[3] != 13:
            raise ValueError("Expected full joint-world state arrays")
        steps, worlds = before.shape[:2]
        capacity = before.shape[2]
        for key in ("world_presence_before","world_presence_after"):
            if key in data and (data[key].shape != (steps,worlds,capacity) or data[key].dtype != bool):
                raise ValueError("Invalid world presence mask")
        if world_count is not None and worlds != world_count:
            raise ValueError("World lane count changed")
        world_count = worlds
        for key in fields:
            value = data[key]
            shape = ((steps, worlds, capacity, 13) if key in ("world_before", "world_after") else
                     (steps, worlds, capacity, 3) if key.startswith("world_motion") else
                     (steps, worlds) if key == "world_actor_count" else
                     (steps, worlds * 32, 3) if key == "events" else (steps, worlds * 32))
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"Invalid recorded {key}")
        for key in ("valid", "terminated", "truncated", "events"):
            if data[key].dtype != bool:
                raise ValueError("Recorded masks must be boolean")
        for key in ("actions", "scene_index", "episode", "tick", "world_actor_count"):
            if data[key].dtype.kind not in "iu":
                raise ValueError("Recorded indices/actions must be integers")
        if not isinstance(meta.get("policy_version"), str) or not meta["policy_version"]:
            raise ValueError("Missing frozen behavior version")
        for t in range(steps):
            for w in range(worlds):
                start = w * 32
                scene_index = int(data["scene_index"][t, start])
                if not 0 <= scene_index < len(paths):
                    raise ValueError("Scene index outside source inventory")
                scene = scenes[scene_index]
                if scene_index not in contexts:
                    contexts[scene_index] = static_context(scene)
                    while len(contexts) > 32:
                        contexts.popitem(last=False)
                contexts.move_to_end(scene_index)
                ids = tuple(a for a, flag in zip(scene.actor_ids, scene.controlled) if flag)
                if (list(scene.actor_ids) != meta["actor_ids"][scene_index]
                        or list(ids) != meta["controlled_ids"][scene_index]):
                    raise ValueError("Recorded/source actor order differs")
                n, a = len(ids), len(scene.actor_ids)
                sl = slice(start, start + n)
                if int(data["world_actor_count"][t, w]) != a:
                    raise ValueError("Recorded world population differs from source")
                episode, tick = int(data["episode"][t, start]), int(data["tick"][t, start])
                for key, expected in (("scene_index", scene_index), ("episode", episode), ("tick", tick)):
                    if expected < 0 or not (data[key][t, sl] == expected).all():
                        raise ValueError("Actor clocks/scene identities disagree")
                present = scene.presence_at(tick)
                next_present = scene.presence_at(tick+1)
                for key, expected in (("world_presence_before",present),("world_presence_after",next_present)):
                    if key not in data and scene.presence is not None:
                        raise ValueError("Lifecycle recording missing presence mask")
                    if key in data and (not np.array_equal(data[key][t,w,:a],expected) or data[key][t,w,a:].any()):
                        raise ValueError("World presence differs from source lifecycle")
                valid = data["valid"][t, sl]
                term, trunc = data["terminated"][t, sl], data["truncated"][t, sl]
                if ((term | trunc) & ~valid).any() or (term & trunc).any():
                    raise ValueError("Invalid episode boundary masks")
                commands = data["actions"][t, sl]
                if (commands < 0).any() or (commands >= 12).any():
                    raise ValueError("Invalid submitted joint action")
                state = np.concatenate((before[t, w, :a], data["world_motion_before"][t, w, :a]), axis=1)
                successor = np.concatenate((data["world_after"][t, w, :a],
                                            data["world_motion_after"][t, w, :a]), axis=1)
                ownership = None
                if behavior is not None:
                    ownership = behavior_index[t, sl].copy()
                    if (ownership < 0).any() or (behavior_index[t, start+n:start+32] != -1).any():
                        raise ValueError("Behavior ownership differs from controlled population")
                old = previous.get(w)
                if old is not None:
                    if episode == old["episode"]:
                        if (scene_index != old["scene"] or tick != old["tick"] + 1 or old["ended"]
                                or not np.array_equal(state, old["after"])
                                or behavior is not None and not np.array_equal(ownership, old["ownership"])):
                            raise ValueError("Joint sequence discontinuity within episode")
                    else:
                        if episode != old["episode"] + 1 or tick != 0 or not old["ended"]:
                            raise ValueError("Invalid reset boundary")
                        stats["reset_boundaries"] += 1
                        stats["short_segments"] += int(seen[w] < length)
                        history[w].clear()
                        seen[w] = 0
                previous[w] = {"episode": episode, "scene": scene_index, "tick": tick,
                               "after": successor, "ended": bool((~valid | term | trunc).all()),
                               "ownership": ownership}
                seen[w] = seen.get(w, 0) + 1
                history.setdefault(w, deque(maxlen=length))
                actions = np.full(a, -1, dtype=np.int64)
                actions[scene.controlled] = commands
                row = {"state": state, "next": successor, "actions": actions,
                       "policy_version": meta["policy_version"], "tick": tick,
                       "presence": present, "next_presence": next_present}
                for key, value in (("action_applied", valid), ("terminated", term), ("truncated", trunc),
                                   ("rewards", data["rewards"][t, sl]), ("events", data["events"][t, sl])):
                    full = np.zeros((a,) + value.shape[1:], dtype=value.dtype)
                    full[scene.controlled] = value
                    row[key] = full
                if behavior is not None:
                    full_index = np.full(a, -1, dtype=np.int64)
                    full_index[scene.controlled] = ownership
                    full_eligible = np.zeros(a, dtype=bool)
                    full_eligible[scene.controlled] = learning_eligible[t, sl]
                    row.update(behavior_policy_index=full_index, learning_eligible=full_eligible,
                               behavior_policies=copy.deepcopy(table))
                if "driving_tasks" in scene.metadata:
                    if "route_support_end" not in data:
                        raise ValueError("Task rollout is missing route support end masks")
                    support = np.asarray(data["route_support_end"])
                    if support.shape != data["valid"].shape or support.dtype != bool or (support[t,sl] & ~trunc).any():
                        raise ValueError("Invalid route support end masks")
                    full = np.zeros(a, bool)
                    full[scene.controlled] = support[t,sl]
                    row["route_support_end"] = full
                history[w].append(row)
                stats["joint_transitions"] += 1
                if len(history[w]) != length or (seen[w] - length) % stride:
                    continue
                rows = list(history[w])
                states = np.stack([rows[0]["state"], *(item["next"] for item in rows)])
                field_valid = np.ones((a, 16), bool)
                field_valid[~scene.controlled, 7:9] = False
                field_valid[~scene.controlled, 9:16] = False  # Passive contacts are separate events, not terminal flags.
                states[:, ~field_valid] = 0
                metadata = {"schema_version": 1, "kind": "joint-world-model-sequence", "state_version": STATE_VERSION,
                            "state_columns": STATE_COLUMNS, "dt_seconds": .1, "world_lane": w,
                            "episode": episode, "scene_index": scene_index, "actor_ids": scene.actor_ids,
                            "start_tick": rows[0]["tick"], "end_tick": tick + 1,
                            "context_steps": context_steps, "prediction_steps": prediction_steps,
                            "max_steps": len(scene.times) - 1,
                            "policy_versions": [item["policy_version"] for item in rows],
                            "eligibility": {"structured_dynamics_engineering": True, "expert_supervision": False,
                                            "photoreal_training": False},
                            "limits": ["Not a complete Markov simulator state", "Replay is exogenous",
                                       "Fixed actor slots with source-supported replay lifetimes", "Future masks are labels, not inputs"]}
                mixed_arrays = {"presence": np.stack([rows[0]["presence"], *(item["next_presence"] for item in rows)])}
                if "driving_tasks" in scene.metadata:
                    mixed_arrays["route_support_end"] = np.stack([item["route_support_end"] for item in rows])
                if behavior is not None:
                    mixed_arrays.update({key: np.stack([item[key] for item in rows]) for key in
                                    ("behavior_policy_index", "learning_eligible")})
                    metadata.update(schema_version=2,
                        behavior_policies=[item["behavior_policies"] for item in rows],
                        actor_policy_versions=[[None if i < 0 else item["behavior_policies"][int(i)]["policy_version"]
                                                for i in item["behavior_policy_index"]] for item in rows],
                        policy_versions_semantics="learner_version_per_step_not_all_actor_behavior",
                        learning_eligibility_semantics="policy_update_mask_not_dynamics_data_filter")
                stats["windows"] += 1
                yield JointSequence(metadata, states,
                                    **{k: np.stack([item[k] for item in rows]) for k in
                                       ("actions", "action_applied", "rewards", "terminated", "truncated", "events")},
                                    controlled=scene.controlled, state_field_valid=field_valid,
                                    scene_context=contexts[scene_index], **mixed_arrays)
    stats["short_segments"] += sum(count < length for count in seen.values())


def export_sequences(run: Path, output: Path, *, context_steps=4, prediction_steps=8, stride=4) -> dict:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    (output / "sequences").mkdir()
    (output / "scenes").mkdir()
    manifest = {"schema_version": 1, "kind": "structured-world-model-dataset", "status": "writing",
                "source_run": str(run.resolve()), "context_steps": context_steps,
                "prediction_steps": prediction_steps, "stride": stride, "statistics": {}, "sequences": []}
    inventory = output / "manifest.json"
    inventory.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    contexts = set()
    try:
        for sequence in iter_joint_sequences(run, context_steps=context_steps, prediction_steps=prediction_steps,
                                             stride=stride, statistics=manifest["statistics"]):
            m = sequence.metadata
            name = f"world-{m['world_lane']:03d}-episode-{m['episode']:06d}-tick-{m['start_tick']:04d}"
            path = output / "sequences" / (name + ".npz")
            sequence.save(path)
            sid = m["scene_index"]
            context_path = output / "scenes" / f"scene-{sid:05d}.npz"
            if sid not in contexts:
                c = sequence.scene_context
                np.savez_compressed(context_path, metadata=json.dumps(c["metadata"]),
                                    actor_ids=np.asarray(c["actor_ids"]),
                                    **{k: v for k, v in c.items() if isinstance(v, np.ndarray)})
                contexts.add(sid)
            manifest["sequences"].append({"path": str(path.relative_to(output)),
                                          "scene_context": str(context_path.relative_to(output))})
        manifest["status"] = "complete"
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        raise
    finally:
        inventory.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_sequence(root: Path, index: int) -> JointSequence:
    root = root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if (manifest.get("schema_version") != 1 or manifest.get("kind") != "structured-world-model-dataset"
            or manifest.get("status") != "complete"):
        raise ValueError("Expected a complete supported dataset")
    item = manifest["sequences"][index]
    paths = [(root / item[k]).resolve() for k in ("path", "scene_context")]
    if any(not p.is_relative_to(root) for p in paths):
        raise ValueError("Dataset reference outside its root")
    with np.load(paths[1], allow_pickle=False) as archive:
        context = {k: frozen_array(archive[k]) for k in archive.files if k not in ("metadata", "actor_ids")}
        context.update(metadata=json.loads(str(archive["metadata"])), actor_ids=tuple(archive["actor_ids"].tolist()))
    with np.load(paths[0], allow_pickle=False) as archive:
        return JointSequence(metadata=json.loads(str(archive["metadata"])), scene_context=context,
                             **{k: archive[k].copy() for k in archive.files if k != "metadata"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context-steps", type=int, default=4)
    parser.add_argument("--prediction-steps", type=int, default=8)
    parser.add_argument("--stride", type=int, default=4)
    args = parser.parse_args()
    result = export_sequences(args.run, args.output, context_steps=args.context_steps,
                              prediction_steps=args.prediction_steps, stride=args.stride)
    print(json.dumps(result["statistics"]))


if __name__ == "__main__":
    main()
