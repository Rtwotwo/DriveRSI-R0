"""Bounded learned-policy collection for reached-state recovery, not an RL batch."""

import argparse
from pathlib import Path

import numpy as np
import torch

from driversi.agents.backends.native import digest, project_root
from driversi.configuration import read_mapping
from driversi.contracts.road import RoadGraph
from driversi.contracts.scene import Scene
from driversi.rendering.bev.replay import export_replay
from driversi.rsi.correction.reached_state import verify_reached_state
from driversi.runtime.budget import write_json
from driversi.sim.persistence.recording import RolloutRecord, write_rollout
from driversi_agents.driverl.closed_loop import DriveRLSession
from driversi_agents.driverl.learning.learner import deterministic_training


def session(plan, checkpoint):
    cfg = plan["model_config"]
    return DriveRLSession(
        Scene.load(Path(plan["scene"])),
        repository=project_root() / cfg["repository"] / "DriveRL",
        checkpoint=checkpoint,
        maps_root=cfg["paths"]["maps"],
        device="cuda:0",
        parity=False,
    )


def collect(plan):
    deterministic_training()
    torch.set_num_threads(4)
    output = Path(plan["output"])
    reports = {}
    for branch, checkpoint in (
        ("parent", plan["parent_checkpoint"]),
        ("candidate", plan["checkpoint"]),
    ):
        torch.manual_seed(42)
        out = output / branch
        out.mkdir()
        s = session(plan, checkpoint)
        record = RolloutRecord.start(s.world)
        contacts = []
        try:
            s.scene.save(out / "scene.npz")
            for tick in range(plan["steps"]):
                if s.world.episode_done or not s.world.observe().alive[0]:
                    break
                tr = s.step()
                record.append(
                    s.world, s.last_decision["executed_actions"], tr, s.last_decision["diagnostics"]
                )
                # Capture the reached state immediately, including a final supported
                # state with no future observations. No extra action is required.
                if s.world.observe().step == plan["anchor_tick"] and s.world.observe().alive[0]:
                    torch.save(s.snapshot(), out / "runtime.pt")
                    s.world.snapshot().save(out / "anchor.npz")
                if s.world.contacts():
                    contacts.append({"tick": tick + 1, "contacts": s.world.contacts()})
            protocol = {
                "kind": "learned_policy_collection",
                "trainable": False,
                "title": "DriveRL " + branch + " reached-state collection",
                "note": "Diagnostic collection: candidate not promoted; rule traffic, no teacher or RGB admission.",
                "producer_checkpoint_sha256": digest(checkpoint),
                "anchor_tick": plan["anchor_tick"],
            }
            write_rollout(
                s.scene,
                out / "rollout.npz",
                record=record,
                snapshot=s.initial,
                controlled_ids=s.world.actor_ids,
                policy="DriveRL-" + branch,
                controller_configuration=s.configuration,
                protocol=protocol,
            )
            export_replay(
                s.scene,
                out / "rollout.npz",
                out / "viewer.html",
                inspection=protocol,
                road_graph=RoadGraph.load(Path(plan["road_graph"])),
                compress_payload=True,
            )
            reports[branch] = {
                "steps": len(record.actions),
                "contacts": contacts,
                "checkpoint": checkpoint,
                "checkpoint_sha256": digest(checkpoint),
                "all_world_objects": len(s.scene.actor_ids),
                "traffic": "existing_reactive_rule_controller",
                "learned_self_play": False,
                "complete": len(record.actions) == plan["steps"],
                "recovery_available": (out / "runtime.pt").is_file(),
            }
            write_json(out / "execution.json", reports[branch])
            if not reports[branch]["recovery_available"]:
                continue
            s.restore(torch.load(out / "runtime.pt", map_location="cpu", weights_only=False))
            reached = verify_reached_state(
                s.scene, out / "rollout.npz", s.world, plan["anchor_tick"]
            )
            reached.update(
                producer="learned_policy",
                policy_checkpoint=checkpoint,
                policy_checkpoint_sha256=digest(checkpoint),
                student_artifact=plan["student_identity"] if branch == "candidate" else None,
                world_snapshot=str(out / "anchor.npz"),
                runtime_snapshot=str(out / "runtime.pt"),
                policy_memory="native_planner_controller_route_history",
                model_promoted=False,
                selection="fixed_pilot_anchor_not_automatic_weakness_mining",
                learned_self_play=False,
            )
            write_json(out / "recovery-task.json", reached)
        finally:
            s.close()
    with (
        np.load(output / "parent/rollout.npz", allow_pickle=False) as a,
        np.load(output / "candidate/rollout.npz", allow_pickle=False) as b,
    ):
        comparison = {
            "same_initial_state": bool(np.array_equal(a["states"][0], b["states"][0])),
            "different_action_steps": int(
                np.any(
                    a["actions"][: min(len(a["actions"]), len(b["actions"]))]
                    != b["actions"][: min(len(a["actions"]), len(b["actions"]))],
                    axis=(1, 2),
                ).sum()
            ),
            "anchor_max_state_difference": (
                float(
                    np.abs(
                        a["states"][plan["anchor_tick"]] - b["states"][plan["anchor_tick"]]
                    ).max()
                )
                if min(len(a["states"]), len(b["states"])) > plan["anchor_tick"]
                else None
            ),
            "parent_role": plan["parent_role"],
            "is_performance_evaluation": False,
        }
    if not comparison["same_initial_state"]:
        raise ValueError("Parent/candidate did not start from the same world")
    report = {
        "branches": reports,
        "comparison": comparison,
        "trainable": False,
        "promoted": False,
        "charged_agent_steps": sum(x["steps"] * x["all_world_objects"] for x in reports.values()),
    }
    write_json(output / "collection.json", report)


def verify(plan):
    """New process restores all runtime memory and recomputes the saved suffix."""
    deterministic_training()
    torch.set_num_threads(4)
    out = Path(plan["output"]) / "candidate"
    s = session(plan, plan["checkpoint"])
    try:
        s.restore(torch.load(out / "runtime.pt", map_location="cpu", weights_only=False))
        reached = verify_reached_state(s.scene, out / "rollout.npz", s.world, plan["anchor_tick"])
        with np.load(out / "rollout.npz", allow_pickle=False) as a:
            actual_steps = len(a["actions"])
            for tick in range(plan["anchor_tick"], actual_steps):
                tr = s.step()
                for x, y in (
                    (s.world.state(), a["states"][tick + 1]),
                    (s.world.motion(), a["motion"][tick + 1]),
                    (s.world.presence(), a["presence"][tick + 1]),
                    ([s.last_decision["executed_actions"]["ego"]], a["actions"][tick]),
                    (tr.rewards, a["rewards"][tick]),
                    (tr.valid, a["valid"][tick]),
                    (tr.terminated, a["terminated"][tick]),
                    (tr.truncated, a["truncated"][tick]),
                ):
                    np.testing.assert_array_equal(x, y)
        write_json(
            Path(plan["output"]) / "restore.json",
            {
                "exact": True,
                "steps": actual_steps - plan["anchor_tick"],
                "history_ticks": reached["history_ticks"],
                "scope": "candidate_world_actions_rewards_masks_native_memory",
            },
        )
    finally:
        s.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("plan")
    p.add_argument("--verify", action="store_true")
    a = p.parse_args()
    plan = read_mapping(a.plan)
    for path, sha in plan["files"].items():
        if digest(path) != sha:
            raise ValueError("Collection input changed: " + path)
    (verify if a.verify else collect)(plan)


if __name__ == "__main__":
    main()
