"""Read-only alignment audit of archived native DriveRL execution, never training data admission."""

import argparse
import json
from pathlib import Path

import numpy as np

from driversi.agents.backends.native import digest
from driversi.contracts.scene import Scene
from driversi.runtime.budget import write_json
from driversi.training.contracts.samples import NativeWorldSequence


def inspect(source):
    frames = json.loads((source / "native-history.json").read_text())
    decisions = json.loads((source / "decisions.json").read_text())
    metadata = Scene.load(source / "scene.npz").metadata
    n = len(frames)
    if n < 17 or len(decisions) != n or [d["tick"] for d in decisions] != list(range(n)):
        raise ValueError("Requires an uninterrupted recorded pre-action history")
    actions = tuple(d["executed_action"] for d in decisions[:-1])
    interventions = tuple(
        "harness_repair" if d.get("repair_active") else None for d in decisions[:-1]
    )
    times = np.asarray([int(f["ego"][-1]) * 1000 for f in frames], dtype=np.int64)
    sequence = NativeWorldSequence(
        metadata["source_log"],
        source.name,
        "driverl_recorded_native_executed_action_v1",
        times,
        tuple(frames),
        actions,
        interventions,
        16,
    )
    policy = sequence.policy_inputs()
    world = sequence.world_inputs()
    targets = sequence.targets()
    assert len(policy["observations"]) == 16 and len(targets["observations"]) == n - 16
    assert "future_executed_actions" not in policy
    assert len(world["future_executed_actions"]) == len(targets["observations"])
    assert np.array_equal(world["future_executed_actions"], np.asarray(actions)[15:])
    policy["observations"][0]["ego"][0] = float("nan")
    assert np.isfinite(sequence.observations[0]["ego"][0])
    mismatch = sum(
        not np.array_equal(d["native_action"], d["executed_action"]) for d in decisions[:-1]
    )
    if any(
        not np.array_equal(d["native_action"], d["executed_action"]) and not flag
        for d, flag in zip(decisions[:-1], interventions)
    ):
        raise ValueError("Unlabeled executed-action change")
    return {
        "source": str(source),
        "source_log": metadata["source_log"],
        "map_name": metadata["map_name"],
        "input_hashes": {
            n: digest(source / n) for n in ("native-history.json", "decisions.json", "scene.npz")
        },
        "recorded_states": len(frames),
        "complete_transitions": n - 1,
        "excluded_last_action": "no_native_successor_frame_recorded",
        "duration_seconds": float((times[-1] - times[0]) * 1e-9),
        "history_frames": 16,
        "history_span_seconds": float((times[15] - times[0]) * 1e-9),
        "intervention_transitions": sum(x is not None for x in interventions),
        "native_executed_action_mismatches": mismatch,
        "scope": "10Hz structured sequence alignment, not a 2Hz camera model dataset",
        "policy_future_hidden": True,
        "world_conditions_on_actual_executed_actions": True,
        "on_policy_admissible": False,
        "reason": "no_native_sampling_likelihoods_and_contains_interventions",
        "trainable": False,
        "quality_admission": "not_evaluated",
        "new_rollout_steps": 0,
        "new_optimizer_updates": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "schema_version": 1,
        "scope": "archived_real_data_contract_validation_only",
        "results": [inspect(s.resolve()) for s in args.source],
        "trainable": False,
    }
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
