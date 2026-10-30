"""JSON transport for a same-scene, same-build native world snapshot."""

import base64
import hashlib
import json

from driversi.contracts.transition import Snapshot


def encode_world(snapshot: Snapshot, *, tick: int, configuration: dict, actor_ids) -> dict:
    return {"version": "world-checkpoint-v1", "scene_key": snapshot.scene_key,
            "build_id": snapshot.build_id, "tick": tick, "actor_ids": list(actor_ids),
            "configuration": json.loads(json.dumps(configuration, allow_nan=False)), "alive": list(snapshot.alive),
            "episode_done": snapshot.episode_done,
            **({"runtime": snapshot.runtime} if snapshot.runtime else {}),
            "native_state": base64.b64encode(snapshot.native_state).decode("ascii"),
            "native_sha256": hashlib.sha256(snapshot.native_state).hexdigest()}


def decode_world(value: dict, *, initial: Snapshot, configuration: dict, actor_ids) -> Snapshot:
    fields = {"version", "scene_key", "build_id", "tick", "actor_ids", "configuration",
              "alive", "episode_done", "native_state", "native_sha256"}
    if (not isinstance(value, dict) or set(value) not in (fields, fields | {"runtime"})
            or value["version"] != "world-checkpoint-v1"
            or value["scene_key"] != initial.scene_key or value["build_id"] != initial.build_id
            or value["configuration"] != json.loads(json.dumps(configuration, allow_nan=False)) or value["actor_ids"] != list(actor_ids)
            or type(value["tick"]) is not int or value["tick"] < 0
            or not isinstance(value["alive"], list) or len(value["alive"]) != len(initial.alive)
            or any(type(v) is not bool for v in value["alive"])
            or type(value["episode_done"]) is not bool
            or not isinstance(value["native_state"], str) or len(value["native_state"]) > 128*1024*1024):
        raise ValueError("Incompatible or invalid world checkpoint")
    try:
        raw = base64.b64decode(value["native_state"], validate=True)
    except (ValueError, UnicodeError) as error:
        raise ValueError("Invalid native checkpoint encoding") from error
    if hashlib.sha256(raw).hexdigest() != value["native_sha256"]:
        raise ValueError("Corrupt native checkpoint")
    return Snapshot(value["scene_key"], value["build_id"], raw, tuple(value["alive"]), value["episode_done"], value.get("runtime", {}))
