"""Bounded DriveRL paper learner through DriveRSI's unified online training engine."""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from driversi.agents.backends.native import digest, path_at, project_root
from driversi.configuration import read_mapping
from driversi.contracts.scene import Scene
from driversi.training.contracts.job import resolve_request
from driversi.training.online_rl.online import RolloutSession
from driversi.training.orchestration.engine import run_online
from driversi.training.sampling.environment import source_log_group
from driversi_agents.driverl.closed_loop import DriveRLSession
from driversi_agents.driverl.learning import checkpoint
from driversi_agents.driverl.learning.environment import NativeEnvironment
from driversi_agents.driverl.learning.learner import (
    LearningConfig,
    NativeActor,
    PaperLearner,
    advance_training_generation,
    deterministic_training,
)


def execute(
    request,
    scene_path,
    source_registry,
    output,
    *,
    burn_in=20,
    seed=42,
    minibatch_size=16,
    resume=None,
    curriculum=None,
    curriculum_arm=None,
):
    started = time.monotonic()
    root = project_root()
    r = resolve_request(request)
    cfg = read_mapping(r["model_config"])
    if r["model"] != "driverl" or r["request"]["mode"] != "online_rl":
        raise ValueError("DriveRL online job required")
    if cfg["settings"]["release_id"] != "driverl-teacher-u2400":
        raise ValueError("teacher initialization only")
    if not 0 <= burn_in <= 40:
        raise ValueError("bounded history burn-in required")
    if r["request"]["budget"]["max_environment_steps"] > 110:
        raise ValueError("single-world pilot maximum 110 decisions")
    source = read_mapping(source_registry)
    if (
        set(source)
        != {"schema_version", "source_log", "scene_sha256", "protected_logs", "split", "db_path"}
        or source["schema_version"] != 1
        or source["split"] != "train"
    ):
        raise ValueError("explicit source registry required")
    if digest(scene_path) != source["scene_sha256"]:
        raise ValueError("scene changed")
    if (
        r["request"]["budget"]["max_optimizer_steps"] > 16
        or r["request"]["budget"]["max_scoring_agent_steps"] != 0
    ):
        raise ValueError("pilot permits at most 16 optimizer updates and no scoring queries")
    official = (
        root
        / "src/vendor/driving/navsim-v1.1/navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml"
    )
    if not official.is_file():
        raise ValueError("official protected source list missing")
    protected = source["protected_logs"] + read_mapping(official)["log_names"]
    scene = Scene.load(scene_path)
    if source_log_group(source["source_log"]) in {source_log_group(x) for x in protected}:
        raise ValueError("protected training source")
    # Dataset partition plus source-group registry, not an arbitrary train flag.
    db = Path(source["db_path"]).resolve(strict=True)
    if any(part in ("val", "test") for part in db.parts) or db.stem != source["source_log"]:
        raise ValueError("heldout or mismatched DB")
    deterministic_training()
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model_path = path_at(root, cfg["weights"]["checkpoint"])
    s = DriveRLSession(
        scene,
        repository=path_at(root, cfg["repository"]) / "DriveRL",
        checkpoint=model_path,
        maps_root=cfg["paths"]["maps"],
        device="cuda:" + str(cfg["settings"]["gpu"]),
        parity=False,
    )
    try:
        if (
            s.scenario.log_name != source["source_log"]
            or Path(scene.metadata["nuplan_source"]["db_path"]).resolve() != db
        ):
            raise ValueError("source identity mismatch")
        curriculum_value = None
        executed_burn_in = 0
        if curriculum:
            from driversi.rsi.correction.curriculum import validate_curriculum
            curriculum_value, selected = validate_curriculum(curriculum, scene=scene_path,
                source_log=source["source_log"], checkpoint=model_path, arm=curriculum_arm)
            bundle = torch.load(selected["snapshot"], map_location="cpu", weights_only=False)
            s.restore(bundle["runtime"])
            if int(s.world.observe().step) != selected["tick"] or not s.world.observe().alive[0]:
                raise ValueError("Curriculum reset does not match its live native anchor")
        else:
            for _ in range(burn_in):
                if s.world.episode_done:
                    raise ValueError("burn-in ended before training")
                s.step()
                executed_burn_in += 1
        env = NativeEnvironment(
            s, source_log=source["source_log"], protected_logs=protected, seed=seed
        )
        actor = NativeActor(s.planner._agent, digest(model_path))
        collector = RolloutSession(env, actor, memory=None, protected_logs=protected)
        output = Path(output).resolve()
        weight_out = output / "checkpoints" if output.is_relative_to(root / "exp") else root / "exp" / output.relative_to(root / "outputs") / "checkpoints"
        learner = PaperLearner(
            actor.agent,
            actor.policy_version,
            weight_out,
            root,
            LearningConfig(minibatch_size=minibatch_size),
        )
        seal = {
            "request": r["request_sha256"],
            "model_config": r["model_config_sha256"],
            "scene": source["scene_sha256"],
            "registry": digest(source_registry),
            "model": digest(model_path),
            "seed": seed,
            "burn_in": burn_in,
            **({"curriculum": digest(curriculum), "curriculum_arm": curriculum_arm}
               if curriculum else {}),
            "source_revision": cfg["revision"],
            "closed_loop": digest(Path(__file__).parents[1] / "closed_loop.py"),
            "implementation": {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")},
        }
        if resume:
            checkpoint.restore(resume, collector, learner, seal=seal)
        physics_before = env.physics_steps
        report = run_online(
            request,
            output,
            session=collector,
            learner=learner,
            chunk_steps=min(16, r["request"]["budget"]["max_environment_steps"]),
        )
        report["implementation_origin"] = "paper_derived_not_official_training_release"
        report["reward_protocol"] = "paper-hard-goal-native-six-soft-v1"
        report["burn_in_physics_steps"] = executed_burn_in
        report["world_slots"] = len(s.scene.actor_ids)
        report["curriculum"] = ({"manifest": str(curriculum), "arm": curriculum_arm,
            "anchor_tick": selected["tick"], "teacher_actions_used_as_PPO": False,
            "fresh_current_policy_sampling": True} if curriculum_value else None)
        report["end_to_end_wall_seconds"] = time.monotonic() - started
        report["wall_budget_scope"] = (
            "collection_and_update_excluding_model_load_and_explicit_burn_in"
        )
        report["world_events"] = env.world_events
        report["physics_steps"] = env.physics_steps - physics_before
        report["metrics"] = learner.last_metrics
        report["native_value_channels"] = 9
        report["reward_memory"] = "persistent_ego_state_identity_mapped_neighbors_and_baselines"
        if report["status"] == "completed":
            collector = advance_training_generation(
                collector, learner, digest(report["checkpoint"]["path"])
            )
            report["training_generation_advanced"] = True
        if report["status"] == "completed" and not collector.unresolved_step:
            report["training_recovery_checkpoint"] = str(
                checkpoint.save(weight_out / "training-state.pt", collector, learner, seal=seal)
            )
            report["recovery_protocol"] = (
                "deterministic_collection_and_update_boundaries_single_worker"
            )
            report["exact_resume_supported"] = (
                False  # A job must compare a replay before claiming this.
            )
        else:
            report["training_recovery_checkpoint"] = None
            report["recovery_rejection"] = "not_a_confirmed_synchronized_update_boundary"
        (output / "paper-learning-report.json").write_text(json.dumps(report, indent=2))
        return report
    finally:
        s.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--request", required=True)
    p.add_argument("--scene", required=True)
    p.add_argument("--source-registry", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--burn-in", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--minibatch-size", type=int, default=16)
    p.add_argument("--resume")
    a = p.parse_args()
    print(
        json.dumps(
            execute(
                a.request,
                a.scene,
                a.source_registry,
                a.output,
                burn_in=a.burn_in,
                seed=a.seed,
                minibatch_size=a.minibatch_size,
                resume=a.resume,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
