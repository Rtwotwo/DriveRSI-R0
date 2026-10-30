"""Continuous reached-state contract shared by perturbation and learned policies.

A pending recovery task is not a teacher label, an RGB sample or an RL batch.
"""

import json
from pathlib import Path

import numpy as np

from driversi.agents.backends.native import digest


def verify_reached_state(scene, rollout, world, anchor_tick, *, actor="ego"):
    """Check the restored full world against an executed, contiguous prefix."""
    if type(anchor_tick) is not int or anchor_tick < 15:
        raise ValueError("Four actual history frames require at least 15 ticks")
    with np.load(rollout, allow_pickle=False) as archive:
        meta = json.loads(str(archive["metadata"]))
        if meta["dt_seconds"] != 0.1 or list(archive["actor_ids"]) != list(scene.actor_ids):
            raise ValueError("Reached-state clock or actor identity mismatch")
        states, motion, presence = [archive[k] for k in ("states", "motion", "presence")]
        if len(states) != len(archive["actions"]) + 1 or anchor_tick >= len(states):
            raise ValueError("Reached anchor absent from actual execution")
        if len(motion) != len(states) or len(presence) != len(states):
            raise ValueError("Incomplete world history")
        if (
            not np.isfinite(states[: anchor_tick + 1]).all()
            or not np.isfinite(motion[: anchor_tick + 1]).all()
        ):
            raise ValueError("Nonfinite executed history")
        ids = tuple(archive["controlled_ids"])
        if actor not in ids:
            raise ValueError("Recovery owner must have an executed policy slot")
        observation = world.observe()
        if observation.step != anchor_tick or not observation.alive[ids.index(actor)]:
            raise ValueError("Anchor must be live and at the executed tick")
        for name, recorded, actual in (
            ("state", states[anchor_tick], world.state()),
            ("motion", motion[anchor_tick], world.motion()),
            ("presence", presence[anchor_tick], world.presence()),
        ):
            if not np.array_equal(recorded, actual):
                raise ValueError("Reached anchor " + name + " mismatch")
        index = scene.actor_ids.index(actor)
        if not presence[anchor_tick - 15 : anchor_tick + 1, index].all():
            raise ValueError("Recovery owner missing from arrival history")
        ticks = list(range(anchor_tick - 15, anchor_tick + 1, 5))
        return {
            "schema_version": 1,
            "kind": "reached_state_recovery_task",
            "actor": actor,
            "source_log": scene.metadata["source_log"],
            "anchor_tick": anchor_tick,
            "history_ticks": ticks,
            "history_span_seconds": 1.5,
            "history_state_indices": "all_world_actor_slots_including_presence",
            "rollout": str(Path(rollout).resolve()),
            "rollout_sha256": digest(rollout),
            "available_future_ticks": len(states) - 1 - anchor_tick,
            "required_teacher_future_ticks": 40,
            "required_teacher_future_seconds": 4.0,
            "arrival_verified": True,
            "teacher_qualified": False,
            "counterfactual_score": None,
            "original_log_score_applicable": False,
            "history_images": "require_rendering_from_this_execution",
            "training_uses": {
                "supervised_planning": "pending_same_anchor_teacher_and_visual_admission",
                "world_model": "pending_action_sensor_alignment_and_quality_admission",
                "on_policy_rl": "not_admitted_no_policy_sampling_likelihoods",
            },
            "trainable": False,
        }
