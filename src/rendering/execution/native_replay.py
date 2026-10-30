"""Exact native-integrator replay followed by GS capture. No new policy inference."""

import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from driversi.agents.backends.native import digest
from driversi.contracts.population import ActorPolicyBinding, EpisodePopulation, PopulationPolicy
from driversi.contracts.scene import Scene
from driversi.rendering.bev.replay import _load_run, _scene_key
from driversi.rendering.execution.challenge_replay import verify_world_record
from driversi.rendering.gaussian.mtgs_artifacts import validate_image_write_workers


def recorded_generator_provenance(metadata):
    """Carry recorded generator roles; replay never infers a model identity."""
    protocol = metadata.get("protocol") or {}
    if not isinstance(protocol, dict):
        raise ValueError("Recorded controller protocol must be an object")
    return {
        "policy": metadata.get("policy", "unknown"),
        "controller_protocol": copy.deepcopy(protocol),
        "executed_teacher_backend": (metadata.get("controller_configuration") or {}).get(
            "executed_teacher_backend", "unknown"
        ),
        "source_of_truth": "sealed_rollout_metadata",
    }


def run(config_path, execution, output, branch="teacher"):
    from driversi.rendering.execution.challenge_replay import _runtime, _session
    from driversi.rendering.gaussian.mtgs_live_viewer import write_session_viewer
    from driversi.sim.workers.worker_client import WorldWorker

    config = json.loads(Path(config_path).read_text())
    image_write_workers = validate_image_write_workers(config.get("image_write_workers", 1))
    execution = Path(execution)
    output = Path(output)
    binding = json.loads(Path(config["binding"]).read_text())
    capture_end = binding.get("render_end_tick")
    if capture_end is not None and (
        type(capture_end) is not int or not 60 <= capture_end <= 70 or capture_end % 5
    ):
        raise ValueError(
            "Native history/future replay requires capture through tick60 and at most70"
        )
    output.mkdir(parents=True, exist_ok=False)
    report = dict(
        kind="native_integrator_gs_replay",
        status="running",
        trainable=False,
        branch=branch,
        verified_cpu_steps=0,
        verified_render_steps=0,
    )
    session = None

    def dump():
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))

    dump()
    try:
        scene_path = execution / "scene.npz"
        scene = Scene.load(scene_path)
        rollout = execution / (branch + ".npz")
        target_path = execution / (branch + "-targets.json")
        if digest(config["scene"]) != digest(scene_path) or digest(config["road_graph"]) != digest(
            execution / "road-graph.json"
        ):
            raise ValueError("Renderer inputs differ from native branch")
        metadata = _load_run(scene, rollout, _scene_key(scene))["metadata"]
        if metadata.get("action_space") != "driverl_native_jerk_steering_rate":
            raise ValueError("Native continuous control recording required")
        with np.load(rollout, allow_pickle=False) as data:
            arrays = {
                k: data[k].copy()
                for k in (
                    "states",
                    "motion",
                    "presence",
                    "actions",
                    "valid",
                    "terminated",
                    "truncated",
                    "collision",
                    "offroad",
                    "goal",
                    "route_support_end",
                )
            }
            controlled = tuple(str(x) for x in data["controlled_ids"])
        targets = json.loads(target_path.read_text())
        if len(targets) != len(arrays["actions"]) or len(targets) != len(scene.times) - 1:
            raise ValueError("Complete native episode and targets required")
        bundle = SimpleNamespace(
            arrays=arrays, metadata=metadata, scene=scene, controlled=controlled
        )
        population = EpisodePopulation(
            (PopulationPolicy("recorded-native", digest(rollout), "external-native-state-v1"),),
            tuple(ActorPolicyBinding(a, "recorded-native") for a in controlled),
        )
        pop = output / "population.json"
        pop.write_text(json.dumps(population.to_dict()))
        settings = dict(
            scene=scene_path,
            road_graph=execution / "road-graph.json",
            binding=Path(config["binding"]),
            world_python=Path(config["world_python"]),
            native_path=Path(config["native_path"]),
            population_config=pop,
            output=output / "session",
            cameras=tuple(config["cameras"]),
            sensor_stride=5,
            height_mode=config.get("height_mode", "reference-track"),
            visual_policy=False,
            registration_mode=config.get("registration_mode", "legacy-map-axes"),
            require_complete_assets=config.get("require_complete_assets", False),
            image_write_workers=image_write_workers,
        )
        report["source_files"] = {
            str(p): digest(p)
            for p in (scene_path, rollout, target_path, Path(config_path), Path(config["binding"]))
        }
        with WorldWorker(
            python=settings["world_python"],
            native_path=settings["native_path"],
            scene=scene_path,
            road_graph=settings["road_graph"],
            population_config=pop,
        ) as worker:
            verify_world_record(bundle, worker.current, 0)
            for tick, (target, action) in enumerate(zip(targets, arrays["actions"])):
                current = worker.request(
                    "native_step",
                    expected_tick=tick,
                    native_targets=target,
                    native_actions=dict(zip(controlled, action.tolist())),
                )
                verify_world_record(bundle, current, tick + 1)
                report["verified_cpu_steps"] = tick + 1
        dump()
        runtime = _runtime(config)
        from driversi.data.assets.mtgs_coverage import runtime_nodes, verified_asset_preflight

        nodes = runtime_nodes(runtime)
        (output / "asset-catalog.json").write_text(
            json.dumps(
                {
                    "checkpoint_sha256": digest(config["checkpoint_path"]),
                    "metadata_sha256": digest(runtime.metadata_path),
                    "nodes": nodes,
                },
                indent=2,
            )
        )
        preflight = verified_asset_preflight(binding, nodes, scene)
        (output / "asset-preflight.json").write_text(json.dumps(preflight, indent=2))
        report["asset_preflight"] = preflight
        dump()
        if settings["require_complete_assets"] and not preflight["complete"]:
            raise ValueError("Source reconstruction assets incomplete")
        session = _session(runtime, settings)
        verify_world_record(bundle, session.current, 0)
        session.manifest["native_replay"] = {
            "source_rollout": digest(rollout),
            "source_targets": digest(target_path),
            "original_controller_provenance": recorded_generator_provenance(metadata),
            "rendered_rgb_used_for_actions": False,
            "trainable": False,
        }
        capture_end = session.capture_end_tick
        for tick, (target, action) in enumerate(
            zip(targets[:capture_end], arrays["actions"][:capture_end])
        ):
            session.advance(
                external_actions=dict(zip(controlled, action.tolist())),
                native_targets=target,
                expected_tick=tick,
                decision_identity=session.decision_identity(),
                policy_decision={
                    "input_mode": "recorded_native_integrator",
                    "observation_tick": tick,
                    "rgb_is_policy_input": False,
                },
            )
            verify_world_record(bundle, session.current, tick + 1)
            report["verified_render_steps"] = tick + 1
        session.manifest["status"] = "completed"
        session.manifest["native_replay"]["capture_end_tick"] = capture_end
        session.manifest["native_replay"]["full_physics_steps"] = len(targets)
        session._publish()
        write_session_viewer(session.output / "manifest.json")
        report.update(
            status="completed",
            viewer="session/viewer.html",
            manifest="session/manifest.json",
            visual_admission=False,
        )
    except Exception as exc:
        import traceback

        (output / "failure-traceback.txt").write_text(traceback.format_exc())
        report.update(status="failed", reason=type(exc).__name__ + ": " + str(exc))
    finally:
        if session:
            session.close()
        dump()
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--execution", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--branch", choices=["teacher", "baseline"], default="teacher")
    a = p.parse_args()
    report = run(a.config, a.execution, a.output, a.branch)
    print(json.dumps(report))
    return int(report["status"] != "completed")


if __name__ == "__main__":
    raise SystemExit(main())
