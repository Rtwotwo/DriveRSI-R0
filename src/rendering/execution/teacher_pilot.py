"""Render frozen learned privileged policies through the existing external-action boundary."""
import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

from driversi.training.legacy.frozen import NpzPolicy
from driversi.contracts.population import ActorPolicyBinding, EpisodePopulation, PopulationPolicy
from driversi.contracts.scene import Scene
from driversi.data.assets.mtgs_episode import export_mtgs_episode
from driversi.rendering.gaussian.mtgs_artifacts import validate_image_write_workers
from driversi.rendering.gaussian.mtgs_live_viewer import write_session_viewer
from driversi.rendering.gaussian.mtgs_runtime import MTGSRuntime
from driversi.rendering.gaussian.mtgs_session import MTGSWorldSession
from driversi.sim.persistence.session_checkpoint import SessionCheckpoint
from driversi.training.sampling.environment import OBSERVATION_VERSION


def decision(session, policy, override=None):
    identity = session.decision_identity()
    observation = session.worker.request("privileged_observe")
    if observation["tick"] != identity["tick"] or observation["observation_version"] != OBSERVATION_VERSION:
        raise ValueError("Privileged observation clock or schema mismatch")
    x = np.asarray(observation["privileged_features"], dtype=np.float32)
    logits, values = policy.predict(x)
    actions = {actor: int(action) for actor, action in
               zip(observation["controlled_ids"], logits.argmax(axis=1), strict=True)}
    record = {"identity": identity, "input_mode": "privileged_structured", "rgb_is_policy_input": False,
              "policy_version": policy.metadata["policy_version"], "observations": x.tolist(),
              "actor_ids": observation["controlled_ids"], "logits": logits.tolist(),
              "values": values.tolist(), "actions": actions, "selection": "argmax",
              "behavior_log_probability": 0., "approved_teacher": False}
    # Input journal precedes world mutation; network softmax is not the probability of argmax behavior.
    with (session.output / "teacher-decisions.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")
        stream.flush()
    session.advance(override, external_actions=actions, expected_tick=identity["tick"],
                    decision_identity=identity, policy_decision={
                        "decision_id": f"{identity['session_id']}:{identity['tick']}",
                        "input_mode": "privileged_structured", "selection": "argmax",
                        "observation_tick": identity["tick"], "policy_version": policy.metadata["policy_version"],
                        "inputs": "teacher-decisions.jsonl", "rgb_is_policy_input": False})


def run(config, model_path, output, fork_tick=11):
    image_write_workers = validate_image_write_workers(config.get("image_write_workers", 1))
    from PIL import Image

    if output.exists():
        raise FileExistsError(output)
    policy = NpzPolicy(model_path)
    scene = Scene.load(Path(config["scene"]))
    controlled = tuple(a for a, flag in zip(scene.actor_ids, scene.controlled) if flag)
    population = EpisodePopulation(
        (PopulationPolicy("learned-teacher-candidate", policy.metadata["policy_version"], "external-joint-v1"),),
        tuple(ActorPolicyBinding(a, "learned-teacher-candidate", "candidate") for a in controlled))
    output.mkdir(parents=True)
    population_path = output / "population.json"
    population_path.write_text(json.dumps(population.to_dict(), indent=2), encoding="utf-8")
    runtime = MTGSRuntime(**{k: Path(config[k]) for k in ("config_path", "checkpoint_path", "road_block_config")},
                          scale=config.get("scale", .5), trusted_assets=True)
    settings = {k: Path(config[k]) for k in ("scene", "road_graph", "binding", "world_python", "native_path")}
    settings.update(cameras=tuple(config["cameras"]), sensor_stride=config["sensor_stride"],
                    desired_speed=config["desired_speed"], virtual_camera_actors=tuple(config["virtual_camera_actors"]),
                    height_mode=config["height_mode"], population_config=population_path,
                    image_write_workers=image_write_workers)
    sessions = []
    start = time.perf_counter()
    try:
        parent = MTGSWorldSession(runtime=runtime, output=output / "candidate", **settings)
        sessions.append(parent)
        shutil.copyfile(model_path, parent.output / "teacher.npz")
        parent.manifest["teacher_artifact"] = "teacher.npz"
        parent.manifest["limits"] = [x for x in parent.manifest["limits"] if x != "external_policy_inputs_unrecorded"]
        parent.manifest["limits"].append("learned_teacher_candidate_not_approved")
        parent.manifest["protocol"] = {"kind": "learned_privileged_population", "rgb_is_policy_input": False}
        parent.manifest["sampling"].update(decision_mode="privileged_structured", policy=policy.metadata,
                                           inputs_recorded=True, approved_teacher=False)
        while parent.current["tick"] < fork_tick and not parent.current["done"]:
            decision(parent, policy)
        if parent.current["done"]:
            raise ValueError("Candidate ended before the configured branch point")
        checkpoint = parent.snapshot()
        checkpoint.save(output / "checkpoint.npz")
        disk = SessionCheckpoint.load(output / "checkpoint.npz")
        for name in ("replay", "brake"):
            child = parent.fork(output / name, disk)
            sessions.append(child)
            shutil.copyfile(model_path, child.output / "teacher.npz")
            # Teacher is stateless; input records before the fork are inherited context.
            prefix = []
            for line in (parent.output / "teacher-decisions.jsonl").read_text().splitlines():
                row = json.loads(line)
                row["identity"]["session_id"] = child.session_id
                prefix.append(json.dumps(row))
            (child.output / "teacher-decisions.jsonl").write_text("\n".join(prefix) + "\n", encoding="utf-8")
        replay, brake = sessions[1:]
        for session in sessions:
            while not session.current["done"]:
                tick = session.current["tick"]
                override = {config["brake_actor"]: 1} if session is brake and tick < fork_tick + 10 else None
                decision(session, policy, override)
            session._publish()
            write_session_viewer(session.output / "manifest.json")
            export_mtgs_episode(session.output / "manifest.json", output / (session.output.name + "-sequence"))
        fields = ("state", "motion", "policy_actions", "executed_actions", "valid", "terminated", "truncated", "events")
        exact = len(parent.manifest["steps"]) == len(replay.manifest["steps"]) and all(
            a[k] == b[k] for a, b in zip(parent.manifest["steps"], replay.manifest["steps"]) for k in fields)
        exact_rgb, compared_rgb = 0, 0
        for a, b in zip(parent.manifest["frames"], replay.manifest["frames"], strict=True):
            if a["tick"] <= fork_tick:
                continue
            for av, bv in zip(a["cameras"], b["cameras"], strict=True):
                with Image.open(parent.output / av["image"]) as image:
                    left = np.asarray(image)
                with Image.open(replay.output / bv["image"]) as image:
                    right = np.asarray(image)
                compared_rgb += 1
                exact_rgb += int(np.array_equal(left, right))
        action_changes = []
        for a, b in zip(parent.manifest["steps"], brake.manifest["steps"]):
            changed = [actor for actor in controlled if actor != config["brake_actor"]
                       and a["policy_actions"][actor] != b["policy_actions"][actor]]
            if changed:
                action_changes.append({"tick": b["tick"], "actors": changed})
        result = {"kind": "learned-privileged-render-pilot", "policy": policy.metadata,
                  "controlled_actors": len(controlled), "rgb_is_policy_input": False,
                  "fork_tick": fork_tick, "physics_replay_exact": exact,
                  "new_suffix_images_exact": exact_rgb, "new_suffix_images_compared": compared_rgb,
                  "intervention_other_actor_action_changes": action_changes,
                  "override_action_difference_steps": sum(s["executed_actions"] != s["policy_actions"] for s in brake.manifest["steps"]),
                  "session_seconds_excluding_model_load": time.perf_counter() - start,
                  "branches": {s.output.name: {"steps": len(s.manifest["steps"]),
                      "events": {k: sum(sum(row["events"][k]) for row in s.manifest["steps"]) for k in ("collision", "offroad", "goal")}}
                               for s in sessions}, "trainable": False, "approved_teacher": False}
        (output / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        if not exact:
            raise AssertionError("Learned policy same-state replay differs")
        return result
    finally:
        for session in sessions:
            session.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(json.loads(args.config.read_text()), args.model, args.output)))


if __name__ == "__main__":
    main()
