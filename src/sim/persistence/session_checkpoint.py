"""Context-bound MTGS checkpoints and prefix-preserving independent branches.

Images are dependencies in the server recording, not embedded in the NPZ.
Forks copy those immutable files and own their world/history/memory/recording.
The shared MTGS runtime and frozen policy weights must be used sequentially.
"""

import copy
import hashlib
import json
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SessionCheckpoint:
    payload: dict

    def save(self, path: Path):
        arrays = {}

        def pack(value):
            if isinstance(value, np.ndarray):
                if value.dtype.hasobject:
                    raise ValueError("Object arrays cannot be checkpointed")
                name = f"array_{len(arrays)}"
                arrays[name] = value
                return {"__checkpoint_array__": name}
            if isinstance(value, dict):
                return {k: pack(v) for k, v in value.items()}
            if isinstance(value, (tuple, list)):
                return [pack(v) for v in value]
            return value

        metadata = json.dumps(pack(self.payload), allow_nan=False)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            np.savez_compressed(stream, metadata=metadata, **arrays)

    @classmethod
    def load(cls, path: Path):
        with np.load(path, allow_pickle=False) as archive:
            def unpack(value):
                if isinstance(value, dict):
                    if set(value) == {"__checkpoint_array__"}:
                        return archive[value["__checkpoint_array__"]].copy()
                    return {k: unpack(v) for k, v in value.items()}
                if isinstance(value, list):
                    return [unpack(v) for v in value]
                return value
            payload = unpack(json.loads(str(archive["metadata"])))
        if payload.get("version") != "mtgs-session-checkpoint-v1":
            raise ValueError("Unsupported session checkpoint version")
        return cls(payload)


def context(session):
    m = session.manifest
    return {"inputs": {k: v["sha256"] for k, v in m["inputs"].items()
                       if k != "effective_population"},
            "assets": {k: v["sha256"] for k, v in m["assets"].items()},
            "backend": m["backend"], "renderer_scale": session.runtime.scale,
            "sensor_stride": session.sensor_stride,
            "camera_ids": m["camera_ids"], "height_mode": m["height_mode"],
            "population": m["behavior_policies"], "sampling": m["sampling"]}


def snapshot_session(session):
    if session.manifest["status"] not in {"running", "completed"}:
        raise RuntimeError("Cannot checkpoint an initializing or failed session")
    world = session.worker.request("snapshot")["snapshot"]
    if world["tick"] != session.current["tick"]:
        raise ValueError("Worker and session checkpoint clocks differ")
    files = {}
    for frame in session.manifest["frames"]:
        for camera in frame["cameras"]:
            for key in ("image", "valid", "pinhole", "source", "surface", "alpha", "depth"):
                name = camera.get(key)
                if name and name not in files:
                    path = (session.output/name).resolve()
                    if not path.is_relative_to(session.output) or not path.is_file():
                        raise ValueError("Checkpoint image dependency is missing or outside recording")
                    files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return SessionCheckpoint({"version": "mtgs-session-checkpoint-v1", "context": copy.deepcopy(context(session)),
        "source_root": str(session.output), "files": files, "world": world,
        "current": copy.deepcopy(session.current), "generation": session.generation,
        "manifest": copy.deepcopy(session.manifest), "latest": copy.deepcopy(session.latest),
        "sensor_buffers": copy.deepcopy(session._sensor_buffers),
        "source_images": {str(k): v.relative_to(session.output).as_posix() for k, v in session._source_images.items()},
        "sampler": session.visual_sampler.snapshot() if session.visual_sampler is not None else None})


def fork_session(session, output: Path, checkpoint: SessionCheckpoint | None = None):
    from driversi.sim.workers.worker_client import WorldWorker

    checkpoint = checkpoint or snapshot_session(session)
    payload = checkpoint.payload
    if payload.get("version") != "mtgs-session-checkpoint-v1" or payload["context"] != context(session):
        raise ValueError("Session checkpoint context differs from renderer, assets, policy or scene")
    if (payload["current"]["tick"] != payload["world"]["tick"]
            or payload["latest"]["generation"] != payload["generation"]
            or payload["latest"]["tick"] > payload["current"]["tick"]):
        raise ValueError("Invalid session checkpoint clocks")
    output, source = Path(output).resolve(), Path(payload["source_root"]).resolve()
    if output.exists():
        raise FileExistsError("Fork output must be new")
    # Validate every source dependency before creating a branch or a worker.
    for name, digest in payload["files"].items():
        path = (source/name).resolve()
        if (not path.is_relative_to(source) or not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != digest):
            raise ValueError("Checkpoint image dependency changed or disappeared")
    output.mkdir(parents=True)
    child = type(session).__new__(type(session))
    child.__dict__ = session.__dict__.copy()
    child.worker = None
    child.output, child.session_id = output, uuid.uuid4().hex
    child.current, child.latest = copy.deepcopy(payload["current"]), copy.deepcopy(payload["latest"])
    child.manifest = copy.deepcopy(payload["manifest"])
    child.generation = payload["generation"]
    child._sensor_buffers = copy.deepcopy(payload["sensor_buffers"])
    for buffers in child._sensor_buffers.values():
        for pixels in buffers.values():
            pixels.setflags(write=False)
    child._source_images = {int(k): output/v for k, v in payload["source_images"].items()}
    child._samples = dict(session._samples)
    child.map = copy.deepcopy(session.map)
    child.visual_sampler = None
    try:
        for name in payload["files"]:
            target = output/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source/name, target)
        child.worker = WorldWorker(**session._worker_settings)
        restored = child.worker.request("restore", expected_tick=0, snapshot=payload["world"])
        for key in ("tick", "state", "motion", "alive", "done", "all_actor_ids", "controlled_ids", "behavior_policies"):
            if restored[key] != child.current[key]:
                raise ValueError(f"Restored world differs at {key}")
        parent = {"session_id": child.manifest["session_id"], "generation": child.generation,
                  "tick": child.current["tick"], "inherited_steps": len(child.manifest["steps"]),
                  "inherited_decisions": len(child.manifest.get("decisions", [])),
                  "inherited_frames": len(child.manifest["frames"]),
                  "prefix_is_new_sample": False, "runtime_execution": "sequential_shared_renderer"}
        child.manifest.setdefault("branch_lineage", []).append(parent)
        child.manifest["session_id"] = child.session_id
        for record in child.manifest.get("decisions", []):
            record["identity"]["session_id"] = child.session_id
        if payload["sampler"] is not None:
            sampler = copy.copy(session.visual_sampler)
            sampler.session = child
            sampler.restore(payload["sampler"])
            if sampler.held_record is not None:
                sampler.held_record["identity"]["session_id"] = child.session_id
            child.visual_sampler = sampler
        for name, rows in (("decisions", child.manifest.get("decisions", [])),
                           ("transitions", child.manifest["steps"])):
            with (output/(name+".jsonl")).open("x", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row, allow_nan=False, ensure_ascii=False)+"\n")
        child._publish()
        return child
    except BaseException:
        child.close()
        raise
