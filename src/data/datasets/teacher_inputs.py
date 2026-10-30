"""Replay known stateless privileged model inputs before permitting explicit RGB non-use."""
import json
from pathlib import Path

import numpy as np

from driversi.training.legacy.frozen import NpzPolicy


def validate_teacher_inputs(document, root: Path):
    if document.get("teacher_artifact") != "teacher.npz":
        raise ValueError("Learned teacher requires its frozen teacher.npz artifact")
    policy = NpzPolicy(root / "teacher.npz")
    rows = [json.loads(line) for line in (root / "teacher-decisions.jsonl").read_text().splitlines()]
    steps = document.get("steps", [])
    if len(rows) != len(steps) or not rows:
        raise ValueError("Teacher input journal must cover every physical decision")
    controlled = document["controlled_ids"]
    version = policy.metadata["policy_version"]
    if any(p["version"] != version for p in document["behavior_policies"].values()):
        raise ValueError("Teacher weights and population version differ")
    for row, step in zip(rows, steps, strict=True):
        identity = row.get("identity", {})
        if (identity.get("tick") != step["tick"] - 1
                or identity.get("generation") != step["generation"]
                or identity.get("session_id") != document["session_id"]
                or row.get("actor_ids") != controlled
                or row.get("policy_version") != version
                or row.get("input_mode") != "privileged_structured"
                or row.get("rgb_is_policy_input") is not False
                or row.get("selection") != "argmax"
                or row.get("behavior_log_probability") != 0.):
            raise ValueError("Teacher decision identity/input mode mismatch")
        logits, values = policy.predict(row["observations"])
        actions = dict(zip(controlled, logits.argmax(axis=1).tolist(), strict=True))
        if (row.get("actions") != actions or step["policy_actions"] != actions
                or not np.allclose(row["logits"], logits, rtol=1e-6, atol=1e-6)
                or not np.allclose(row["values"], values, rtol=1e-6, atol=1e-6)):
            raise ValueError("Teacher decision cannot be reproduced from recorded inputs")
    return {"policy": policy.metadata, "decisions": len(rows), "stateless": True,
            "selection": "argmax", "behavior_log_probability": 0.,
            "input_journal": "teacher-decisions.jsonl", "weights": "teacher.npz",
            "recomputed_outputs": True, "current_world_input_truth_independently_certified": False}
