"""Check recorded on-policy batches against their frozen behavior models."""
import argparse
import json
from pathlib import Path

import numpy as np

from driversi.agents.interfaces.config import load_inference
from driversi.data.assets.mtgs_assets import file_sha256
from driversi.data.datasets.world_model import rollout_behavior
from driversi.training.legacy.advantages import generalized_advantage


def audit_run(root: Path) -> dict:
    root = Path(root)
    count = learner_count = opponent_count = mixed_batches = 0
    max_logprob_error = max_value_error = max_next_value_error = 0.
    files = sorted((root / "rollouts").glob("batch-*.npz"))
    if not files:
        raise ValueError("No rollout batches (rank-sharded DDP audit is not supported)")
    frozen_cache, frozen_identity = {}, None
    mixed_mode = None
    for path in files:
        with np.load(path, allow_pickle=False) as batch:
            metadata = json.loads(str(batch["metadata"]))
            policy = load_inference(path.with_name(path.name.replace("batch-", "policy-")), metadata.get("agent"))
            if metadata["policy_version"] != policy.metadata["policy_version"]:
                raise ValueError("Batch behavior version mismatch")
            valid = batch["valid"]
            if valid.dtype != np.bool_ or valid.ndim != 2:
                raise ValueError("Invalid sample mask")
            n = int(valid.sum())
            if n != metadata["valid_transitions"] or n == 0:
                raise ValueError("Invalid effective sample count")
            behavior = rollout_behavior(metadata, batch)
            if mixed_mode is not None and mixed_mode != (behavior is not None):
                raise ValueError("Behavior recording mode changed within a run")
            mixed_mode = behavior is not None
            policies, index, learning = [policy], np.zeros(valid.shape, dtype=np.int64), valid
            if behavior is not None:
                table, index, learning = behavior
                if frozen_identity is not None and frozen_identity != table[1:]:
                    raise ValueError("Frozen opponent identities changed within a run")
                frozen_identity = table[1:]
                for row in table[1:]:
                    checkpoint = Path(row["checkpoint"])
                    if file_sha256(checkpoint) != row["checkpoint_sha256"]:
                        raise ValueError("Frozen opponent checkpoint hash mismatch")
                    key = json.dumps(row, sort_keys=True, allow_nan=False)
                    if key not in frozen_cache:
                        frozen_cache[key] = load_inference(checkpoint, row["agent_config"])
                    opponent = frozen_cache[key]
                    if (opponent.metadata["policy_version"] != row["policy_version"]
                            or file_sha256(checkpoint) != row["checkpoint_sha256"]):
                        raise ValueError("Frozen opponent behavior identity mismatch")
                    policies.append(opponent)
                if metadata.get("learner_transitions") != int(learning.sum()):
                    raise ValueError("Mixed learner transition count mismatch")
                mixed_batches += 1
                opponent_count += int((valid & (index > 0)).sum())
            for name in ("actions", "log_probs", "values", "next_values"):
                if batch[name].shape != valid.shape:
                    raise ValueError("Invalid recorded behavior arrays")
                if not np.isfinite(batch[name]).all():
                    raise ValueError("Nonfinite behavior probability/value")
            if (batch["observations"].shape[:2] != valid.shape
                    or batch["next_observations"].shape != batch["observations"].shape
                    or not np.isfinite(batch["observations"]).all()
                    or not np.isfinite(batch["next_observations"]).all()):
                raise ValueError("Invalid behavior observations or successor")
            for policy_index, behavior_policy in enumerate(policies):
                mask = valid & (index == policy_index)
                size = int(mask.sum())
                if not size:
                    continue
                actions = batch["actions"][mask]
                if (not np.issubdtype(actions.dtype, np.integer)
                        or (actions < 0).any() or (actions >= 12).any()):
                    raise ValueError("Invalid behavior action")
                logits, values = behavior_policy.predict(batch["observations"][mask])
                if (logits.shape != (size, 12) or values.shape != (size,)
                        or not np.isfinite(logits).all() or not np.isfinite(values).all()):
                    raise ValueError("Frozen policy returned invalid predictions")
                stabilized = logits - logits.max(axis=1, keepdims=True)
                log_probs = stabilized - np.log(np.exp(stabilized).sum(axis=1, keepdims=True))
                error = float(np.abs(log_probs[np.arange(size), actions] - batch["log_probs"][mask]).max())
                value_error = float(np.abs(values - batch["values"][mask]).max())
                if error > 2e-5 or value_error > 2e-5:
                    raise ValueError("Stored behavior probability/value cannot be reproduced")
                if behavior is not None:
                    _, successor_values = behavior_policy.predict(batch["next_observations"][mask])
                    if successor_values.shape != (size,) or not np.isfinite(successor_values).all():
                        raise ValueError("Invalid frozen successor values")
                    next_error = float(np.abs(successor_values - batch["next_values"][mask]).max())
                    if next_error > 2e-5:
                        raise ValueError("Stored successor value differs from assigned behavior policy")
                    max_next_value_error = max(max_next_value_error, next_error)
                max_logprob_error = max(max_logprob_error, error)
                max_value_error = max(max_value_error, value_error)
            advantage, returns = generalized_advantage(batch["rewards"], batch["values"],
                batch["next_values"], learning, batch["terminated"], batch["truncated"])
            if not np.array_equal(advantage, batch["advantages"]) or not np.array_equal(returns, batch["returns"]):
                raise ValueError("Stored advantage/return differs from mask-aware GAE")
            if (batch["terminated"] & ~valid).any() or (batch["truncated"] & ~valid).any():
                raise ValueError("Invalid terminal sample")
            count += n
            learner_count += int(learning.sum())
    report = json.loads((root / "report.json").read_text())
    if report["resume_from"] is None and count != report["valid_transitions"]:
        raise ValueError("Run count does not match actual valid batch entries")
    result = {"batches": len(files), "valid_transitions_in_this_run": count,
              "frozen_policy_logprob_reproduced": True, "gae_reproduced": True,
              "max_logprob_error": max_logprob_error, "max_value_error": max_value_error,
              "teacher_quality_certified": False}
    if mixed_batches:
        result.update(mixed_behavior_batches=mixed_batches, frozen_opponent_identities_verified=True,
                      learner_transitions_in_this_run=learner_count,
                      opponent_transitions_in_this_run=opponent_count,
                      max_next_value_error=max_next_value_error)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit_run(args.run)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
