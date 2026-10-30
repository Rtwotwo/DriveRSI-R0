"""Validate managed visual inputs and export an inspection sampling sequence."""

import hashlib
import json
from pathlib import Path

import numpy as np

from driversi.agents.visual_control.image_moment import ImageMomentPolicy
from driversi.contracts.sampling import ActorVisualInput, CameraInput


def export_visual_samples(manifest_path: Path, output: Path) -> dict:
    from PIL import Image

    manifest_path, output = Path(manifest_path).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Visual sample output must be new")
    source_bytes = manifest_path.read_bytes()
    doc = json.loads(source_bytes)
    sampling = doc.get("sampling", {})
    if (doc.get("kind") != "mtgs_world_session" or doc.get("status") != "completed"
            or doc.get("trainable") is not False or sampling.get("decision_mode") != "visual"
            or doc.get("protocol", {}).get("rgb_is_policy_input") is not True):
        raise ValueError("A completed managed visual sampling session is required")
    policy = ImageMomentPolicy()
    if sampling.get("policy") != policy.describe():
        raise ValueError("This exporter requires the declared diagnostic policy implementation")
    actor_ids = sampling["visual_actor_ids"]
    if not actor_ids or len(set(actor_ids)) != len(actor_ids):
        raise ValueError("Invalid visual actor identity")
    root = manifest_path.parent
    decisions, steps, frames = doc["decisions"], doc["steps"], doc["frames"]
    if not decisions or not steps or not frames:
        raise ValueError("Missing sampling records")
    try:
        with (root/"decisions.jsonl").open() as journal:
            persisted = [json.loads(next(journal)) for _ in decisions]
    except StopIteration as error:
        raise ValueError("Decision journal is shorter than frozen manifest") from error
    if persisted != decisions:
        raise ValueError("Decision journal differs from the manifest")
    known = {}
    for frame in frames:
        for view in frame["cameras"]:
            key = (frame["generation"], frame["tick"], view["owner_actor_id"], view["camera_id"])
            if key in known:
                raise ValueError("Duplicate camera frame identity")
            known[key] = view
    pixels, hashes = {}, {}

    def image(record, mode):
        path = (root/record["path"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Input artifact is missing or outside session")
        if path not in pixels:
            raw = path.read_bytes()
            hashes[path] = hashlib.sha256(raw).hexdigest()
            with Image.open(path) as im:
                pixels[path] = np.asarray(im.convert(mode)).copy()
        if hashes[path] != record["sha256"]:
            raise ValueError("Input artifact hash mismatch")
        return pixels[path]

    by_id = {}
    previous = {}
    for decision in decisions:
        identity = decision["identity"]
        generation, tick = identity["generation"], identity["tick"]
        if (decision["decision_id"] != len(by_id) or decision["observation_tick"] != tick
                or identity["session_id"] != doc["session_id"]
                or decision["preprocessing"] != "identity_uint8_rgb_v1"
                or [x["actor_id"] for x in decision["actors"]] != actor_ids
                or decision["policy"] != policy.describe()):
            raise ValueError("Decision identity/preprocessing/policy is inconsistent")
        expected_versions = {a: {"policy_id": p["policy_id"], "version": p["version"]}
                             for a, p in doc["behavior_policies"].items()}
        if identity["policy_versions"] != expected_versions:
            raise ValueError("Policy binding/version changed")
        prior = previous.get(generation)
        expected_memory = prior["memory_after"] if prior else policy.initial_memories(len(actor_ids)).tolist()
        if decision["memory_before"] != expected_memory or (prior and decision["rng_before"] != prior["rng_after"]):
            raise ValueError("Actor memory/RNG continuity is broken")
        if prior and tick <= prior["observation_tick"]:
            raise ValueError("Decision ticks must advance within each generation")
        observations = []
        for actor in decision["actors"]:
            if actor["history_ticks"][-1] != tick:
                raise ValueError("Actor input does not end at the decision tick")
            history = []
            for history_tick, views in zip(actor["history_ticks"], actor["history"], strict=True):
                if history_tick > tick:
                    raise ValueError("Future images cannot be actor input")
                cameras = []
                for view in views:
                    source = known.get((generation, history_tick, actor["actor_id"], view["camera_id"]))
                    if source is None or view["rgb"]["path"] != source["image"] or view["pixel_valid"]["path"] != source["valid"]:
                        raise ValueError("Image owner/tick/source path mismatch")
                    for key, source_key in (("intrinsic", "K"), ("distortion", "distortion"),
                                            ("camera_to_body", "camera_to_body"), ("camera_model", "camera_model"),
                                            ("timestamp_us", "camera_timestamp_us")):
                        if view[key] != source[source_key]:
                            raise ValueError("Actual input calibration differs from capture")
                    rgb = image(view["rgb"], "RGB")
                    valid = image(view["pixel_valid"], "L") > 0
                    if (view["rgb_pixel_sha256"] != source["image_pixel_sha256"]
                            or view["valid_pixel_sha256"] != source["valid_pixel_sha256"]
                            or hashlib.sha256(rgb.tobytes()).hexdigest() != view["rgb_pixel_sha256"]
                            or hashlib.sha256(valid.tobytes()).hexdigest() != view["valid_pixel_sha256"]
                            or rgb.shape != (view["height"], view["width"], 3)):
                        raise ValueError("Recorded files do not reproduce the actual input pixels")
                    cameras.append(CameraInput(view["camera_id"], rgb, valid, np.asarray(view["intrinsic"]),
                        np.asarray(view["distortion"]), np.asarray(view["camera_to_body"]), view["camera_model"],
                        view["timestamp_us"], view["rgb"]["path"], view["pixel_valid"]["path"]))
                history.append(tuple(cameras))
            observations.append(ActorVisualInput(actor["actor_id"], tuple(history), tuple(actor["history_ticks"]),
                np.asarray(actor["telemetry"]), np.asarray(actor["navigation"]), actor["alive"]))
        rngs = tuple(np.random.default_rng() for _ in actor_ids)
        for rng, state in zip(rngs, decision["rng_before"], strict=True):
            rng.bit_generator.state = state
        result = policy.infer(tuple(observations), np.asarray(decision["memory_before"]), rngs)
        if ({a: int(v) for a, v in zip(actor_ids, result.actions)} != decision["actions"]
                or result.memories.tolist() != decision["memory_after"]
                or [rng.bit_generator.state for rng in rngs] != decision["rng_after"]
                or list(result.diagnostics) != decision["diagnostics"]):
            raise ValueError("Recorded policy result cannot be reproduced from actual inputs/state")
        by_id[decision["decision_id"]] = decision
        previous[generation] = decision
    groups = {}
    used = set()
    for step in steps:
        g = step["generation"]
        group = groups.setdefault(g, [])
        if step["tick"] != len(group)+1:
            raise ValueError("Physical steps must be contiguous within generation")
        reference = step["policy_decision"]
        decision = by_id[reference["decision_id"]]
        obs = decision["observation_tick"]
        available = [d for d in decisions if d["identity"]["generation"] == g and d["observation_tick"] <= step["tick"]-1]
        if not available or max(available, key=lambda d: d["observation_tick"])["decision_id"] != decision["decision_id"]:
            raise ValueError("A physical step reused a superseded decision")
        if (decision["identity"]["generation"] != g or obs > step["tick"]-1
                or reference["observation_tick"] != obs
                or reference["newly_decided"] != (obs == step["tick"]-1)):
            raise ValueError("A physical step references a wrong-generation or future decision")
        if any(step["policy_actions"][a] != decision["actions"][a] for a in actor_ids):
            raise ValueError("Executed policy batch differs from the recorded visual decision")
        used.add(reference["decision_id"])
        group.append(step)
    if used != set(by_id):
        raise ValueError("Uncommitted or unreferenced decisions cannot be exported")
    initial = doc.get("initial_world_by_generation", {})
    if set(initial) != {str(g) for g in groups}:
        raise ValueError("Complete initial worlds are required for every generation")
    for generation, rows in groups.items():
        world = initial[str(generation)]
        if world.get("tick") != 0 or any(row["all_actor_ids"] != world.get("all_actor_ids") for row in rows):
            raise ValueError("Initial world tick/actor ordering differs from physical sequence")
    if any(not rows[-1]["done"] for rows in groups.values()):
        raise ValueError("Every exported generation must finish")
    intervals = []
    for decision in decisions:
        g, start = decision["identity"]["generation"], decision["observation_tick"]
        rows = [s for s in groups[g] if s["policy_decision"]["decision_id"] == decision["decision_id"]]
        end = rows[-1]["tick"]
        successor = next((f for f in frames if f["generation"] == g and f["tick"] == end), None)
        if successor is None:
            raise ValueError("Decision interval lacks a real successor camera capture")
        targets = []
        for camera in successor["cameras"]:
            artifacts = {}
            for field, mode, pixel_key in (("image", "RGB", "image_pixel_sha256"),
                                            ("valid", "L", "valid_pixel_sha256")):
                path = root/camera[field]
                record = {"path": camera[field], "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                array = image(record, mode)
                if field == "valid":
                    array = array > 0
                if hashlib.sha256(array.tobytes()).hexdigest() != camera[pixel_key]:
                    raise ValueError("Successor camera pixels differ from capture")
                artifacts[field] = record
            targets.append({"camera_id": camera["camera_id"], "owner_actor_id": camera["owner_actor_id"],
                            **artifacts})
        intervals.append({"generation": g, "start_tick": start, "end_tick": end,
            "decision_id": decision["decision_id"], "joint_actions": [{"tick": s["tick"],
                "policy_actions": s["policy_actions"], "executed_actions": s["executed_actions"]} for s in rows],
            "successor_cameras": targets})
    if manifest_path.read_bytes() != source_bytes:
        raise ValueError("Source session changed during validation")
    result = {"schema_version": 1, "kind": "managed_visual_sampling_episode", "status": "completed",
        "trainable": False, "source": {"manifest": str(manifest_path), "artifact_root": str(root),
            "sha256": hashlib.sha256(source_bytes).hexdigest()}, "sampling": sampling,
        "eligibility": {"sampling_inspection": True, "visual_rl": False, "teacher_imitation": False,
                        "production_world_model": False},
        "limits": doc["limits"], "unavailable_fields": ["reward", "log_probability", "teacher_labels", "world_snapshot"],
        "permissions": {"decisions": "actual_actor_inputs_and_own_memory",
                        "physics": "privileged_world_not_actor_input", "initial_world": "privileged_world_not_actor_input"},
        "summary": {"generations": len(groups), "decisions": len(decisions), "physics_steps": len(steps),
                    "visual_actors": len(actor_ids), "actor_decisions": len(decisions)*len(actor_ids),
                    "input_image_files": len(pixels), "seconds": len(steps)*doc.get("dt_seconds", .1)}, "files": []}
    result["branch_lineage"] = doc.get("branch_lineage", [])
    output.mkdir(parents=True)
    initial_path = output/"initial-world.json"
    initial_path.write_text(json.dumps(initial, allow_nan=False, ensure_ascii=False))
    result["initial_world"] = {"path": initial_path.name,
                               "sha256": hashlib.sha256(initial_path.read_bytes()).hexdigest()}
    for name, rows in (("decisions", decisions), ("physics", steps), ("decision-intervals", intervals)):
        path = output/(name+".jsonl")
        with path.open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False, ensure_ascii=False)+"\n")
        result["files"].append({"path": path.name, "records": len(rows), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (output/"manifest.json").write_text(json.dumps(result, allow_nan=False, ensure_ascii=False, indent=2))
    return result
