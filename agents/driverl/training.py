"""DriveRL paper-derived online learner adapter for the common lifecycle."""

import json
from pathlib import Path

from driversi.agents.backends.native import asset_files, digest, path_at, project_root
from driversi.configuration import read_mapping
from driversi.training.orchestration.state import fingerprint
from driversi.training.sampling.environment import source_log_group


def run(plan):
    import numpy as np
    import torch

    from driversi.contracts.scene import Scene
    from driversi_agents.driverl.closed_loop import DriveRLSession
    from driversi_agents.driverl.learning.learner import deterministic_training
    from driversi_agents.driverl.learning.runner import execute

    root, out = project_root(), Path(plan["output"])
    req, cfg, data = plan["resolved"]["request"], plan["model_config"], plan["data"]
    previous = read_mapping(plan["previous"]) if plan["previous"] else None
    deterministic_training()
    torch.set_num_threads(4)
    if plan["phase"] != "evaluate":
        report = execute(
            plan["resolved"]["request_path"],
            data["train"]["scene"],
            data["train"]["registry"],
            out / "online",
            burn_in=20,
            seed=req["pilot"]["seed"],
            minibatch_size=req["pilot"]["batch_size"],
            resume=previous["recovery_checkpoint"] if previous else None,
            curriculum=data["train"].get("curriculum"),
            curriculum_arm=data["train"].get("curriculum_arm"),
        )
        if report["status"] != "completed":
            raise RuntimeError("DriveRL online training rejected: " + str(report.get("reason")))
        state = torch.load(
            report["training_recovery_checkpoint"], map_location="cpu", weights_only=False
        )
        snap = {
            "model": state["actor"],
            "optimizer": state["learner"]["optimizer"],
            "scheduler": state["learner"]["scheduler"],
            "rng": state["rng"],
            "environment": state["environment"],
            "observation": state["collector"]["observation"],
        }
        report["consumed"]["decision_calls"] = report["consumed"]["environment_steps"]
        report["consumed"]["sim_agent_steps"] = (
            report["burn_in_physics_steps"] + report["physics_steps"]
        )
        if data["train"].get("curriculum"):
            report["consumed"]["sim_agent_steps"] *= report["world_slots"]
        report["consumed"]["sim_step_scope"] = (
            "all_world_slots_physics_steps_including_burn_in" if data["train"].get("curriculum")
            else "ego_physics_steps_including_burn_in")
        files = {p: digest(p) for p in asset_files(out / "online/checkpoints")}
        return {
            "status": "completed",
            "reason": None,
            "checkpoint": report["checkpoint"]["path"],
            "checkpoint_files": files,
            "recovery_checkpoint": report["training_recovery_checkpoint"],
            "updates": report["consumed"]["successful_optimizer_steps"],
            "total_updates": state["learner"]["updates"],
            "state_fingerprint": fingerprint(snap),
            "resume_from": previous["recovery_checkpoint"] if previous else None,
            "metrics": report["metrics"],
            "consumed": report["consumed"],
            "curriculum": report.get("curriculum"),
            "implementation_origin": "paper_derived_not_official_training_release",
            "checkpoint_state": [
                "model",
                "optimizer",
                "scheduler",
                "rng",
                "world",
                "policy_memory",
                "reward_history",
            ],
        }
    scene = Scene.load(data["development"]["scene"])
    if source_log_group(scene.metadata["source_log"]) not in plan["source_groups"]["development"]:
        raise ValueError("Development scene identity mismatch")
    metrics = {}
    evaluation_slots = len(scene.actor_ids)
    parent = path_at(root, cfg["weights"]["checkpoint"])
    steps = req["pilot"]["evaluation_steps"]
    for name, checkpoint in (("parent", parent), ("candidate", previous["checkpoint"])):
        torch.manual_seed(req["pilot"]["seed"])
        session = DriveRLSession(
            scene,
            repository=root / cfg["repository"] / "DriveRL",
            checkpoint=checkpoint,
            maps_root=cfg["paths"]["maps"],
            device="cuda:0",
            parity=False,
        )
        rows, contacts, travelled = [], 0, 0.0
        try:
            old = session.world.state()[0, :2].copy()
            for _ in range(steps):
                if session.world.episode_done:
                    break
                session.step()
                now = session.world.state()[0, :2].copy()
                travelled += float(np.linalg.norm(now - old))
                old = now
                collisions = session.world.contacts()
                contacts += bool(collisions)
                rows.append(
                    {
                        "ego": session.world.state()[0].tolist(),
                        "contacts": collisions,
                        "decision": session.last_decision,
                    }
                )
            if len(rows) != steps:
                raise ValueError("Development closed loop ended early; no shortened comparison")
            metrics[name] = {
                "travelled_m": travelled,
                "contact_free_fraction": 1 - contacts / steps,
                "recorded_steps": len(rows),
            }
            (out / (name + "-rollout.json")).write_text(
                json.dumps(rows, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
            )
        finally:
            session.close()
    return {
        "status": "completed",
        "reason": None,
        "metrics": metrics,
        "directions": {"travelled_m": 1, "contact_free_fraction": 1},
        "primary_metric": "travelled_m",
        "safety_higher": ["contact_free_fraction"],
        "metric_scope": "fixed_short_closed_loop_engineering_not_official_nuplan_score",
        "evaluated_checkpoint": previous["checkpoint"],
        "consumed": {"environment_steps": 2 * steps, "scoring_agent_steps": 0,
                     **({"sim_agent_steps": 2 * steps * evaluation_slots,
                         "sim_step_scope": "all_world_slots_parent_and_candidate"}
                        if data["train"].get("curriculum") else {})},
    }
