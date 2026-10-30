"""Exercise compiled real DB scenes through the existing joint-action interface.

This checks data plumbing, action response and exact restoration, not policy quality.
No model, optimiser or sensor renderer is loaded.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from driversi.contracts.learning import JointDecision
from driversi.contracts.scene import Scene
from driversi.training.sampling.environment import TeacherEnvironment, validate_initial_world


def run(manifest_path, output, limit=24):
    manifest_path, output = Path(manifest_path), Path(output)
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("source_format") != "nuplan_db" or manifest.get("state") != "complete":
        raise ValueError("Expected a complete compiled nuPlan catalogue manifest")
    if not 1 <= limit <= 256:
        raise ValueError("Validation limit must be 1..256")
    records = []
    for row in [v for v in manifest["records"] if v["status"] == "compiled"][:limit]:
        scene = Scene.load(manifest_path.parent / row["scene"])
        record = {
            "scene": row["scene"],
            "token": row["token"],
            "source_log": row["source_log"],
            "city": row["map_name"],
            "actors": len(scene.actor_ids),
            "controlled": int(scene.controlled.sum()),
            "history_frames": len(scene.metadata["nuplan_source"]["history"]),
            "trainable": False,
        }
        env = None
        try:
            env = TeacherEnvironment(scene)
            try:
                validate_initial_world(env)
                record["initial_admission"] = "passed"
            except ValueError as error:
                record["initial_admission"] = "rejected"
                record["initial_reason"] = str(error)
            initial = env.world.state()[env.indices]
            record["controlled_initial_state_valid"] = bool(
                env.world.observe().alive.all()
                and not initial[:, 10:12].any()
                and (np.linalg.norm(initial[:, :2] - initial[:, 7:9], axis=1) > 2).all()
            )
            record["ego_is_policy_controlled"] = "ego" in env.world.actor_ids
            first = env.snapshot()
            observations = env.observation()

            def advance(action):
                obs = env.observation()
                return env.advance(
                    JointDecision(
                        obs.key,
                        obs.actor_ids,
                        np.full(len(obs.actor_ids), action, dtype=np.int64),
                        "nuplan-interface-diagnostic-v1",
                    )
                )

            advance(0)
            a = env.world.state().copy()
            env.restore(first)
            advance(11)
            b = env.world.state().copy()
            record["different_joint_actions_change_state"] = bool(not np.array_equal(a, b))
            env.restore(first)
            prefix = 0
            while not env.world.episode_done and prefix < 10:
                advance(6)
                prefix += 1
            snapshot = env.snapshot()
            suffix = []
            while not env.world.episode_done:
                step = advance(6)
                suffix.append(
                    (
                        env.world.state().copy(),
                        step.rewards.copy(),
                        step.end_reasons.copy(),
                        step.transition.valid.copy(),
                        step.observation.features.copy(),
                        env.world.presence().copy(),
                    )
                )
            record["steps"] = prefix + len(suffix)
            record["suffix_steps"] = len(suffix)
            record["valid_agent_steps_in_suffix"] = sum(int(v[3].sum()) for v in suffix)
            record["last_collision_flags"] = int(env.world.state()[:, 10].sum())
            record["last_offroad_flags"] = int(env.world.state()[:, 11].sum())
            env.restore(snapshot)
            for expected in suffix:
                step = advance(6)
                actual = (
                    env.world.state(),
                    step.rewards,
                    step.end_reasons,
                    step.transition.valid,
                    step.observation.features,
                    env.world.presence(),
                )
                for left, right in zip(expected, actual, strict=True):
                    np.testing.assert_array_equal(left, right)
            record["restore"] = "exact" if suffix else "not_exercised_early_termination"
            record["observation_shape"] = list(observations.features.shape)
            record["status"] = "executed"
        except (ValueError, RuntimeError, AssertionError) as error:
            record.update(status="failed", reason=f"{type(error).__name__}: {error}")
        finally:
            if env is not None:
                env.close()
        records.append(record)
    report = {
        "schema_version": 1,
        "kind": "nuplan-interface-acceptance",
        "manifest": str(manifest_path.resolve()),
        "validation": "joint-action data interface; fixed diagnostic actions, not policy training or driving certification",
        "training_started": False,
        "records": records,
        "summary": {
            "scenes": len(records),
            "executed": sum(v["status"] == "executed" for v in records),
            "multi_agent_initially_valid": sum(
                v.get("initial_admission") == "passed" for v in records
            ),
            "controlled_initial_states_valid": sum(
                v.get("controlled_initial_state_valid", False) for v in records
            ),
            "exact_restores": sum(v.get("restore") == "exact" for v in records),
            "action_response": sum(
                v.get("different_joint_actions_change_state", False) for v in records
            ),
            "cities": sorted({v["city"] for v in records}),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=24)
    args = parser.parse_args()
    print(json.dumps(run(args.manifest, args.output, args.limit)["summary"]))
