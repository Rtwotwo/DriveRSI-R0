"""Causal, model-neutral structured history for privileged continuous policies.

Inspired by DriveRL's timestamp/identity/mask boundary, not its checkpoint tensor
schema. All positions (including ego) and velocities refer to object CENTERS.
A downstream native-model adapter must explicitly convert its own conventions.
Source time follows nuPlan lidar associations, not simultaneous sensor capture.
Pose measurement offsets remain explicit. No route, goal, future label, source future object or model-capacity truncation
is exposed here. This is structured ground truth, not camera perception.
"""

import copy
from collections import deque

import numpy as np

from driversi.data.participants.nuplan_motion import REAR_AXLE_TO_CENTER, ego_center_motion

VERSION = "causal-world-history-v1"


class StructuredHistory:
    def __init__(self, scene, world, capacity_steps):
        self.scene = scene
        self.dt_us = round(float(scene.times[1] - scene.times[0]) * 1e6)
        self.span_us = (capacity_steps - 1) * self.dt_us
        source = scene.metadata.get("nuplan_source", {})
        self.anchor_us = int(source.get("anchor_timestamp_us", 0))
        history = source.get("history", [])
        self.initial_pose_offset_us = (
            (history[-1]["ego_pose_timestamp_us"] - self.anchor_us) if history else None
        )
        self.seed = self._source_history(source)
        self.records = deque(copy.deepcopy(self.seed))
        self.append_world(world)

    def _source_history(self, source):
        history = source.get("history", [])
        if not history:
            return []
        times = [v["timestamp_us"] for v in history]
        if any(type(t) is not int for t in times) or any(a >= b for a, b in zip(times, times[1:])):
            raise ValueError("Source history timestamps must increase")
        if times[-1] != self.anchor_us or any(t > self.anchor_us for t in times):
            raise ValueError("Source history must end at anchor without future leakage")
        if any(abs(v["ego_pose_timestamp_us"] - v["timestamp_us"]) > 50000 for v in history):
            raise ValueError("Source pose/lidar association exceeds 50ms alignment tolerance")
        origin = np.asarray(self.scene.metadata["origin_xyz"][:2])
        ego_index = self.scene.actor_ids.index("ego")
        records = []
        for i, value in enumerate(history[:-1]):
            stamp = value["timestamp_us"]
            if stamp < self.anchor_us - self.span_us - 150000:
                continue
            pos, yaw, velocity, valid = ego_center_motion(value, history[i - 1] if i else None)
            actors = {
                "ego": {
                    "state": np.r_[pos - origin, yaw, velocity, self.scene.dimensions[ego_index]]
                    .astype(float)
                    .tolist(),
                    "velocity_valid": valid,
                    "kind": "vehicle",
                }
            }
            for obj in value["objects"]:
                token = obj["track_token"]
                if not isinstance(token, str) or not token or token in actors:
                    raise ValueError("Duplicate or invalid history track identity")
                pose = np.asarray(obj["pose_global"])
                velocity = obj["velocity_global"]
                actors[token] = {
                    "state": [
                        *(pose[:2] - origin),
                        pose[3],
                        *(velocity[:2] if velocity is not None else [0.0, 0.0]),
                        *obj["dimensions_lwh"],
                    ],
                    "velocity_valid": velocity is not None,
                    "kind": obj["category"],
                }
            records.append(
                {
                    "timestamp_us": stamp,
                    "origin": "source_history",
                    "ego_pose_offset_us": value["ego_pose_timestamp_us"] - stamp,
                    "actors": actors,
                }
            )
        return records

    def world_record(self, world):
        state = world.state()
        present = world.presence()
        kinds = self.scene.metadata.get("source_actor_kinds", {})
        # Include only physically present actors. Future and pending identities
        # remain in the simulator registry, never in this observation registry.
        actors = {}
        for i in np.flatnonzero(present):
            actor = self.scene.actor_ids[i]
            actors[actor] = {
                "state": np.r_[state[i, :2], state[i, 3], state[i, 4:6], self.scene.dimensions[i]]
                .astype(float)
                .tolist(),
                "velocity_valid": True,
                "kind": kinds.get(
                    actor, "vehicle" if actor == "ego" else str(int(self.scene.kinds[i]))
                ),
            }
        return {
            "timestamp_us": self.anchor_us + world.observe().step * self.dt_us,
            "origin": "physical_world",
            "ego_pose_offset_us": 0,
            "actors": actors,
        }

    def append_world(self, world):
        record = self.world_record(world)
        if self.records and record["timestamp_us"] <= self.records[-1]["timestamp_us"]:
            raise ValueError("World history clock must advance")
        self.records.append(record)
        cutoff = record["timestamp_us"] - self.span_us - 150000
        while self.records and self.records[0]["timestamp_us"] < cutoff:
            self.records.popleft()

    def snapshot(self):
        return {"version": VERSION, "records": copy.deepcopy(list(self.records))}

    def restore(self, payload, world):
        if (
            not isinstance(payload, dict)
            or set(payload) != {"version", "records"}
            or payload["version"] != VERSION
        ):
            raise ValueError("Invalid structured history checkpoint")
        records = copy.deepcopy(payload["records"])
        terminal = self.world_record(world)
        cutoff = terminal["timestamp_us"] - self.span_us - 150000
        expected_source = [r for r in self.seed if r["timestamp_us"] >= cutoff]
        if (
            not isinstance(records, list)
            or not records
            or len(records) > len(self.seed) + self.span_us // self.dt_us + 4
        ):
            raise ValueError("Invalid structured history length")
        if [r for r in records if r.get("origin") == "source_history"] != expected_source:
            raise ValueError("Structured source history differs from initialization")
        expected_times = list(
            range(
                max(
                    self.anchor_us,
                    ((cutoff - self.anchor_us + self.dt_us - 1) // self.dt_us) * self.dt_us
                    + self.anchor_us,
                ),
                terminal["timestamp_us"] + 1,
                self.dt_us,
            )
        )
        world_records = [r for r in records if r.get("origin") == "physical_world"]
        if [r.get("timestamp_us") for r in world_records] != expected_times:
            raise ValueError("Structured world history must be contiguous")
        previous = None
        for record in records:
            if set(record) != {"timestamp_us", "origin", "ego_pose_offset_us", "actors"} or record[
                "origin"
            ] not in {
                "source_history",
                "physical_world",
            }:
                raise ValueError("Invalid structured history record")
            stamp = record["timestamp_us"]
            if type(stamp) is not int or (previous is not None and stamp <= previous):
                raise ValueError("Invalid structured history timestamp")
            previous = stamp
            offset = record["ego_pose_offset_us"]
            if (
                type(offset) is not int
                or abs(offset) > 50000
                or (record["origin"] == "physical_world" and offset != 0)
            ):
                raise ValueError("Invalid structured timestamp alignment")
            if not isinstance(record["actors"], dict):
                raise ValueError("Invalid structured history actors")
            for actor, value in record["actors"].items():
                if (
                    not isinstance(actor, str)
                    or not actor
                    or set(value) != {"state", "velocity_valid", "kind"}
                ):
                    raise ValueError("Invalid structured history actor")
                state = np.asarray(value["state"], dtype=float)
                if (
                    state.shape != (8,)
                    or not np.isfinite(state).all()
                    or (state[5:] <= 0).any()
                    or type(value["velocity_valid"]) is not bool
                    or not isinstance(value["kind"], str)
                ):
                    raise ValueError("Invalid structured history state")
                if record["origin"] == "physical_world" and actor not in self.scene.actor_ids:
                    raise ValueError("Unknown physical actor in history")
        if records[-1] != terminal:
            raise ValueError("Structured history terminal state differs from world")
        self.records = deque(records)

    def sample(self, *, actor="ego", frames=4, interval=0.5, max_age=0.15):
        if (
            type(frames) is not int
            or not 1 <= frames <= 256
            or isinstance(interval, bool)
            or not np.isfinite(interval)
            or interval <= 0
        ):
            raise ValueError("Invalid history sampling request")
        if (
            isinstance(max_age, bool)
            or not np.isfinite(max_age)
            or not 0 <= max_age <= min(0.15, interval)
        ):
            raise ValueError("History tolerance must be finite and <= 0.15s and interval")
        step = round(interval * 1e6)
        if step <= 0 or (frames - 1) * step > self.span_us:
            raise ValueError("History request exceeds retained time support")
        raw = list(self.records)
        anchor = raw[-1]
        if actor not in anchor["actors"]:
            raise ValueError("Anchor actor is not physically present")
        requested = anchor["timestamp_us"] - np.arange(frames - 1, -1, -1, dtype=np.int64) * step
        times = np.asarray([r["timestamp_us"] for r in raw], dtype=np.int64)
        indices = np.searchsorted(times, requested, side="right") - 1
        valid = (indices >= 0) & (requested - times[np.maximum(indices, 0)] <= round(max_age * 1e6))
        selected = [raw[i] if ok else None for i, ok in zip(indices, valid)]
        ids = sorted({a for r in selected if r for a in r["actors"]})
        ids.remove(actor)
        ids.insert(0, actor)
        states = np.zeros((len(ids), frames, 8), dtype=np.float64)
        mask = np.zeros((len(ids), frames), bool)
        velocity_mask = mask.copy()
        derivatives = np.zeros((len(ids), frames, 3), dtype=np.float64)
        derivative_mask = mask.copy()
        kinds = []
        origin = np.asarray(anchor["actors"][actor]["state"][:3], dtype=float)
        c, s = np.cos(origin[2]), np.sin(origin[2])
        rotation = np.array([[c, s], [-s, c]])
        for j, identity in enumerate(ids):
            kinds.append(
                next(
                    r["actors"][identity]["kind"]
                    for r in reversed(selected)
                    if r and identity in r["actors"]
                )
            )
            for t, record in enumerate(selected):
                if record is None or identity not in record["actors"]:
                    continue
                item = record["actors"][identity]
                value = np.array(item["state"], dtype=float)
                value[:2] = rotation @ (value[:2] - origin[:2])
                value[2] = (value[2] - origin[2] + np.pi) % (2 * np.pi) - np.pi
                value[3:5] = rotation @ value[3:5]
                if not item["velocity_valid"]:
                    value[3:5] = 0
                states[j, t] = value
                mask[j, t] = True
                velocity_mask[j, t] = item["velocity_valid"]
                if t and velocity_mask[j, t - 1] and velocity_mask[j, t]:
                    continuous = all(
                        identity in r["actors"] and r["origin"] == record["origin"]
                        for r in raw[indices[t - 1] : indices[t] + 1]
                    )
                    dt = (record["timestamp_us"] - selected[t - 1]["timestamp_us"]) / 1e6
                    if continuous and 0 < dt <= interval + max_age:
                        derivatives[j, t, :2] = (states[j, t, 3:5] - states[j, t - 1, 3:5]) / dt
                        derivatives[j, t, 2] = (
                            (states[j, t, 2] - states[j, t - 1, 2] + np.pi) % (2 * np.pi) - np.pi
                        ) / dt
                        derivative_mask[j, t] = True
        return {
            "version": VERSION,
            "actor_ids": tuple(ids),
            "kinds": tuple(kinds),
            "states": states,
            "present": mask,
            "velocity_valid": velocity_mask,
            "derivatives": derivatives,
            "derivative_valid": derivative_mask,
            "frame_valid": valid,
            "requested_timestamp_us": requested,
            "sampled_timestamp_us": np.array(
                [r["timestamp_us"] if r else -1 for r in selected], dtype=np.int64
            ),
            "origins": tuple(r["origin"] if r else "unavailable" for r in selected),
            "ego_pose_measurement_offset_us": np.array(
                [r["ego_pose_offset_us"] if r else 0 for r in selected], dtype=np.int64
            ),
            "initial_pose_measurement_offset_us": self.initial_pose_offset_us,
            "timestamp_basis": "source_lidar_association_then_simulation_clock; not exact sensor synchrony",
            "anchor_timestamp_us": anchor["timestamp_us"],
            "anchor_pose_world": origin,
            "reference": "anchor_actor_center_axes; all object centers; states=x,y,yaw,vx,vy,length,width,height",
            "ego_rear_axle_to_center_m": REAR_AXLE_TO_CENTER if actor == "ego" else None,
            "scope": "privileged structured history; no native model tensor or sensor equivalence claimed",
        }
