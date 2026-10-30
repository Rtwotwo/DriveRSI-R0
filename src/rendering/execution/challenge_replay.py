"""Strict GS rendering of an already executed bounded challenge action sequence.

This bridge re-executes recorded joint actions; it does not perform new policy
inference. The existing session captures before returning from advance, so the
per-step equality check is explicitly after capture, never a pre-render claim.
"""

import argparse
import copy
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from driversi.contracts.context import scene_fingerprint
from driversi.contracts.population import ActorPolicyBinding, EpisodePopulation, PopulationPolicy
from driversi.contracts.road import RoadGraph
from driversi.contracts.scene import Scene
from driversi.rendering.bev.replay import _load_run, _scene_key
from driversi.rendering.gaussian.mtgs_artifacts import validate_image_write_workers

VERSION = "bounded-challenge-action-render-v1"
SOURCE_VERSION = "bounded-longitudinal-challenge-v1"


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path):
    def invalid(value):
        raise ValueError("Nonfinite JSON value: " + value)

    return json.loads(Path(path).read_text(), parse_constant=invalid)


def _dump(path, data):
    pending = path.with_suffix(".pending")
    pending.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def _artifact(root, name, digest):
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError("Challenge artifacts must use relative paths")
    path = (root / name).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file() or _sha(path) != digest:
        raise ValueError("Challenge artifact hash/path mismatch: " + name)
    return path


@dataclass(frozen=True)
class ReplayInput:
    config: dict
    report: dict
    candidate: dict
    scene: Scene
    graph: RoadGraph
    arrays: dict
    metadata: dict
    paths: dict
    hashes: dict
    controlled: tuple[str, ...]
    derived_binding: dict | None = None


def load_replay_input(
    config_path, report_path, candidate_id=None, *, allow_diagnostic_failure=False
):
    """Validate immutable source artifacts without loading Torch or a renderer."""
    config_path, report_path = Path(config_path).resolve(), Path(report_path).resolve()
    config, report = _json(config_path), _json(report_path)
    validate_image_write_workers(config.get("image_write_workers", 1))
    if type(allow_diagnostic_failure) is not bool:
        raise ValueError("Diagnostic failure mode must be explicit boolean")
    if isinstance(report, dict) and report.get("kind") in {
        "bounded-lateral-pair",
        "two-stage-reactive-augmentation-v1",
    }:
        return _load_lateral_input(
            config_path,
            report_path,
            config,
            report,
            candidate_id,
            allow_diagnostic_failure=allow_diagnostic_failure,
        )
    if (
        not isinstance(report, dict)
        or report.get("schema_version") != 1
        or report.get("kind") != SOURCE_VERSION
        or report.get("trainable") is not False
        or report.get("status") not in ("completed", "partial")
    ):
        raise ValueError("Require a completed/partial bounded challenge report")
    chosen = candidate_id if candidate_id is not None else report.get("selected_candidate")
    rows = [row for row in report.get("attempts", []) if row.get("id") == chosen]
    if len(rows) != 1 or rows[0].get("status") != "completed":
        raise ValueError("Candidate must identify exactly one completed challenge")
    row = rows[0]
    if row.get("metrics", {}).get("constraint_violations") != []:
        raise ValueError("Candidate has unresolved execution constraints")
    root = report_path.parent
    inputs = report.get("input_artifacts", {})
    paths = {"report": report_path, "config": config_path}
    for key, name in (("scene", "scene.npz"), ("road_graph", "road_graph.json")):
        item = inputs.get(name, {})
        paths[key] = _artifact(root, item.get("path"), item.get("sha256"))
    if "route_tasks.json" in inputs:
        tasks = inputs["route_tasks.json"]
        route_tasks = _json(_artifact(root, tasks.get("path"), tasks.get("sha256")))
        if route_tasks.get("initial_nodes") or route_tasks.get("successors"):
            raise ValueError("MTGS bridge cannot silently discard explicit route overrides")
    paths["rollout"] = _artifact(root, row.get("rollout"), row.get("rollout_sha256"))
    if not isinstance(config, dict):
        raise ValueError("Renderer config must be an object")
    for key in (
        "scene",
        "road_graph",
        "binding",
        "world_python",
        "native_path",
        "config_path",
        "checkpoint_path",
        "road_block_config",
    ):
        value = config.get(key)
        if not isinstance(value, str) or not Path(value).is_absolute() or not Path(value).is_file():
            raise ValueError("Renderer config needs an existing absolute path: " + key)
    for key in ("scene", "road_graph"):
        if _sha(config[key]) != _sha(paths[key]):
            raise ValueError("Renderer config differs from packed challenge " + key)
    paths["binding"] = Path(config["binding"]).resolve()
    binding = _json(paths["binding"])
    if binding.get("inputs", {}).get("scene", {}).get("sha256") != _sha(paths["scene"]):
        raise ValueError("Explicit reconstruction binding does not match challenge scene hash")
    scene, graph = Scene.load(paths["scene"]), RoadGraph.load(paths["road_graph"])
    graph.validate_scene(scene)
    if (
        binding.get("log_name") != scene.metadata["source_log"]
        or binding.get("map_name") != scene.metadata["map_name"]
        or binding.get("source_start_timestamp_us")
        != scene.metadata.get("source_start_timestamp_us")
    ):
        raise ValueError("Reconstruction binding source identity differs")
    key = _scene_key(scene)
    checked = _load_run(scene, paths["rollout"], key)
    metadata = checked["metadata"]
    protocol = metadata.get("protocol", {})
    if (
        report.get("scenario_id") != scene.metadata["scenario_id"]
        or report.get("scene_key") != key
        or metadata.get("backend_build") != report.get("backend_build")
        or protocol.get("kind") != SOURCE_VERSION
        or protocol.get("branch") != row["id"]
        or protocol.get("trainable") is not False
        or protocol.get("details", {}).get("stop_reason") != "episode_boundary"
    ):
        raise ValueError("Rollout provenance/build/branch differs from challenge report")
    with np.load(paths["rollout"], allow_pickle=False) as bundle:
        arrays = {
            name: bundle[name].copy()
            for name in (
                "states",
                "motion",
                "actions",
                "valid",
                "terminated",
                "truncated",
                "collision",
                "offroad",
                "goal",
            )
        }
        controlled = tuple(str(a) for a in bundle["controlled_ids"])
        for name in ("presence", "route_support_end"):
            if name in bundle.files:
                arrays[name] = bundle[name].copy()
    steps = len(arrays["actions"])
    if not steps or not (
        steps == len(scene.times) - 1 or arrays["states"][-1, scene.controlled, 9].all()
    ):
        raise ValueError("Candidate recording is not a full episode boundary")
    for value in arrays.values():
        value.setflags(write=False)
    hashes = {name: _sha(path) for name, path in paths.items()}
    return ReplayInput(
        config, report, row, scene, graph, arrays, metadata, paths, hashes, controlled
    )


def _load_lateral_input(
    config_path, report_path, config, report, candidate_id, *, allow_diagnostic_failure
):
    """Task-aware strict bridge. Failure render requests are explicitly diagnostic."""
    if (
        report.get("schema_version") != 1
        or report.get("export_status") != "complete"
        or report.get("trainable") is not False
    ):
        raise ValueError("Require a complete unqualified lateral source package")
    chosen = candidate_id or "challenge"
    if chosen not in {"baseline", "challenge"}:
        raise ValueError("Lateral candidate must be baseline or challenge")
    branch = report.get("branches", {}).get(chosen, {})
    root = report_path.parent
    paths = {"config": config_path, "report": report_path}
    for key, name in (("scene", "scene.npz"), ("road_graph", "road_graph.json")):
        item = report.get("input_artifacts", {}).get(name, {})
        paths[key] = _artifact(root, item.get("path"), item.get("sha256"))
    paths["rollout"] = _artifact(root, branch.get("rollout"), branch.get("sha256"))
    if not isinstance(config, dict):
        raise ValueError("Renderer config must be an object")
    for key in (
        "scene",
        "road_graph",
        "binding",
        "world_python",
        "native_path",
        "config_path",
        "checkpoint_path",
        "road_block_config",
    ):
        if (
            not isinstance(config.get(key), str)
            or not Path(config[key]).is_absolute()
            or not Path(config[key]).is_file()
        ):
            raise ValueError("Renderer config needs an existing absolute path: " + key)
    if _sha(config["road_graph"]) != _sha(paths["road_graph"]):
        raise ValueError("Renderer road graph differs from lateral package")
    scene, original = Scene.load(paths["scene"]), Scene.load(Path(config["scene"]))
    graph = RoadGraph.load(paths["road_graph"])
    graph.validate_scene(scene)
    paths["binding"] = Path(config["binding"]).resolve()
    binding = _json(paths["binding"])
    if binding.get("inputs", {}).get("scene", {}).get("sha256") != _sha(config["scene"]):
        raise ValueError("Original reconstruction binding scene hash differs")
    if any(
        binding.get(key) != scene.metadata.get(meta)
        for key, meta in (
            ("log_name", "source_log"),
            ("map_name", "map_name"),
            ("source_start_timestamp_us", "source_start_timestamp_us"),
        )
    ):
        raise ValueError("Reconstruction binding source identity differs")
    comparable = copy.deepcopy(scene)
    for key in ("driving_tasks", "task_goals_xy"):
        if key in original.metadata:
            comparable.metadata[key] = copy.deepcopy(original.metadata[key])
        else:
            comparable.metadata.pop(key, None)
    if scene_fingerprint(comparable) != scene_fingerprint(original):
        raise ValueError("Only declared driving task changes may reuse reconstruction binding")
    derived = copy.deepcopy(binding)
    derived["inputs"]["scene"] = {"path": str(paths["scene"]), "sha256": _sha(paths["scene"])}
    derived["task_only_derivation"] = {
        "source_binding_sha256": _sha(paths["binding"]),
        "source_scene_sha256": _sha(config["scene"]),
        "changed_fields": ["driving_tasks", "task_goals_xy"],
        "all_other_scene_content_equal": True,
    }
    key = _scene_key(scene)
    metadata = _load_run(scene, paths["rollout"], key)["metadata"]
    protocol = metadata.get("protocol", {})
    augmentation = report.get("kind") == "two-stage-reactive-augmentation-v1"
    protocol_keys = (
        ("actor", "phases", "explore_speed", "continuation_configuration", "source_history_sha256")
        if augmentation
        else ("challenger", "responder", "intervention_tick", "candidate_route", "task_semantics")
    )
    if augmentation:
        item = report.get("input_artifacts", {}).get("source-history.json", {})
        history_path = _artifact(root, item.get("path"), item.get("sha256"))
        if _sha(history_path) != report.get("source_history_sha256"):
            raise ValueError("Augmentation source history differs")
        paths["source_history"] = history_path
    if (
        metadata.get("backend_build") != report.get("backend_build")
        or report.get("scene_key") != key
        or report.get("scenario_id") != scene.metadata["scenario_id"]
        or protocol.get("branch") != chosen
        or protocol.get("kind") != report.get("kind")
        or protocol.get("trainable") is not False
        or any(protocol.get(k) != report.get(k) for k in protocol_keys)
    ):
        raise ValueError("Lateral rollout provenance/build/branch differs")
    with np.load(paths["rollout"], allow_pickle=False) as data:
        arrays = {
            name: data[name].copy()
            for name in (
                "states",
                "motion",
                "actions",
                "presence",
                "valid",
                "terminated",
                "truncated",
                "collision",
                "offroad",
                "goal",
            )
        }
        if "driving_tasks" in scene.metadata:
            arrays["route_support_end"] = data["route_support_end"].copy()
        controlled = tuple(str(a) for a in data["controlled_ids"])
    ticks = len(arrays["actions"])
    if not ticks or not (
        ticks == len(scene.times) - 1 or arrays["states"][-1, scene.controlled, 9].all()
    ):
        raise ValueError("Lateral render requires a full recorded episode boundary")
    qualified = (
        not report[chosen]["bounds_failures"]
        and not arrays["collision"].any()
        and not arrays["offroad"].any()
        and (augmentation or chosen == "baseline" or report.get("maneuver_validation_pass") is True)
    )
    if not qualified and not allow_diagnostic_failure:
        raise ValueError(
            "Lateral maneuver failed qualification; explicitly allow diagnostic failure render"
        )
    row = {
        "id": chosen,
        "status": "completed",
        "diagnostic_failure": not qualified,
        "maneuver_validation_pass": report.get("maneuver_validation_pass") is True,
    }
    for value in arrays.values():
        value.setflags(write=False)
    hashes = {name: _sha(path) for name, path in paths.items()}
    return ReplayInput(
        config, report, row, scene, graph, arrays, metadata, paths, hashes, controlled, derived
    )


def verify_world_record(bundle, current, tick):
    """Exact comparison; does not accept tolerance, actor reorder or lost masks."""
    arrays = bundle.arrays
    if (
        current.get("build_id") != bundle.metadata["backend_build"]
        or current.get("all_actor_ids") != list(bundle.scene.actor_ids)
        or current.get("controlled_ids") != list(bundle.controlled)
        or current.get("tick") != tick
        or current.get("dt_seconds") != 0.1
    ):
        raise ValueError(f"World identity/build/order differs at tick {tick}")
    for name, expected in (("state", arrays["states"][tick]), ("motion", arrays["motion"][tick])):
        actual = np.asarray(current.get(name))
        if actual.shape != expected.shape or not np.array_equal(actual, expected):
            raise ValueError(f"{name} differs at tick {tick}")
    if "presence" in arrays and current.get("presence") != arrays["presence"][tick].tolist():
        raise ValueError(f"Presence differs at tick {tick}")
    alive = arrays["states"][tick, bundle.scene.controlled, 9] == 0
    done = tick >= len(bundle.scene.times) - 1 or not alive.any()
    if current.get("alive") != alive.tolist() or current.get("done") is not bool(done):
        raise ValueError(f"Alive/done differs at tick {tick}")
    if tick:
        index = tick - 1
        expected_actions = dict(
            zip(bundle.controlled, arrays["actions"][index].tolist(), strict=True)
        )
        if any(
            current.get(name) != expected_actions for name in ("policy_actions", "executed_actions")
        ):
            raise ValueError(f"Recorded joint action differs at tick {tick}")
        for name in ("valid", "terminated", "truncated"):
            if current.get(name) != arrays[name][index].tolist():
                raise ValueError(f"{name} differs at tick {tick}")
        expected_events = {
            name: arrays[name][index].tolist() for name in ("collision", "offroad", "goal")
        }
        if "route_support_end" in arrays:
            expected_events["route_support_end"] = arrays["route_support_end"][index].tolist()
        if current.get("events") != expected_events:
            raise ValueError(f"Events differ at tick {tick}")


def replay_into_session(session, bundle, verification):
    """Images are output only. Each existing capture precedes this equality check."""
    verify_world_record(bundle, session.current, 0)
    verification["initial_state_exact"] = True
    for tick, row in enumerate(bundle.arrays["actions"]):
        identity = session.decision_identity()
        if identity.get("tick") != tick:
            raise ValueError("Session decision clock differs from replay clock")
        actions = dict(zip(bundle.controlled, row.tolist(), strict=True))
        session.advance(
            external_actions=actions,
            expected_tick=tick,
            decision_identity=identity,
            policy_decision={
                "decision_id": f"recorded:{bundle.hashes['rollout']}:{tick}",
                "input_mode": "recorded_joint_action",
                "selection": "recorded",
                "observation_tick": tick,
                "rgb_is_policy_input": None,
                "rendered_rgb_used_for_replay_action": False,
            },
        )
        verify_world_record(bundle, session.current, tick + 1)
        verification["verified_steps"] = tick + 1


def _preflight(bundle, settings):
    from driversi.sim.workers.worker_client import WorldWorker

    with WorldWorker(
        python=settings["world_python"],
        native_path=settings["native_path"],
        scene=settings["scene"],
        road_graph=settings["road_graph"],
        population_config=settings["population_config"],
    ) as worker:
        verify_world_record(bundle, worker.current, 0)
        for tick, actions in enumerate(bundle.arrays["actions"]):
            current = worker.request(
                "step",
                expected_tick=tick,
                external_actions=dict(zip(bundle.controlled, actions.tolist(), strict=True)),
            )
            verify_world_record(bundle, current, tick + 1)
    return len(bundle.arrays["actions"])


def _runtime(config):
    from driversi.rendering.gaussian.mtgs_runtime import MTGSRuntime

    return MTGSRuntime(
        **{
            key: Path(config[key])
            for key in ("config_path", "checkpoint_path", "road_block_config")
        },
        scale=config.get("scale", 0.5),
        source_travel_id=_json(Path(config["binding"]))["travel_id"],
        trusted_assets=True,
    )


def _session(runtime, settings):
    from driversi.rendering.gaussian.mtgs_session import MTGSWorldSession

    return MTGSWorldSession(runtime=runtime, **settings)


def run(config_path, report_path, output, *, candidate_id=None, allow_diagnostic_failure=False):
    """Load once, re-execute once, and retain explicit failure evidence on mismatch."""
    from driversi.data.assets.mtgs_episode import export_mtgs_episode
    from driversi.rendering.gaussian.mtgs_live_viewer import write_session_viewer

    started = time.perf_counter()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    verification = {
        "kind": VERSION,
        "schema_version": 1,
        "status": "running",
        "phase": "validate_source",
        "stage_seconds": {},
        "verified_steps": 0,
        "trainable": False,
        "approved_teacher": False,
        "official_score": None,
        "rendered_rgb_used_for_replay_actions": False,
        "original_policy_inputs": "unverified",
        "verification_timing": "CPU full-action replay before model load; initial and step captures precede equality checks in MTGS session",
        "scope": "reproduction of recorded reactive interaction, not new inference or GS quality certification",
    }
    _dump(output / "verification.json", verification)
    session = None
    stage_started = started

    def phase(name):
        nonlocal stage_started
        now = time.perf_counter()
        verification["stage_seconds"][verification["phase"]] = now - stage_started
        verification["phase"] = name
        verification["wall_seconds"] = now - started
        stage_started = now
        _dump(output / "verification.json", verification)

    try:
        bundle = load_replay_input(
            config_path,
            report_path,
            candidate_id,
            allow_diagnostic_failure=allow_diagnostic_failure,
        )
        verification.update(
            candidate_id=bundle.candidate["id"],
            diagnostic_failure=bundle.candidate.get("diagnostic_failure", False),
            requested_steps=len(bundle.arrays["actions"]),
            native_build=bundle.metadata["backend_build"],
            inputs={
                name: {"path": str(bundle.paths[name]), "sha256": digest}
                for name, digest in bundle.hashes.items()
            },
        )
        provenance = output / "provenance"
        provenance.mkdir()
        names = {
            "report": "report.json",
            "source_history": "source-history.json",
            "config": "render-config.json",
            "scene": "scene.npz",
            "road_graph": "road_graph.json",
            "rollout": "rollout.npz",
            "binding": "binding.json",
        }
        for name, path in bundle.paths.items():
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != bundle.hashes[name]:
                raise ValueError("Input changed while copying provenance: " + name)
            (provenance / names[name]).write_bytes(raw)
        if bundle.derived_binding is not None:
            (provenance / "binding.json").rename(provenance / "source-binding.json")
            _dump(provenance / "binding.json", bundle.derived_binding)
        population = EpisodePopulation(
            (
                PopulationPolicy(
                    "recorded-actions", "sha256:" + bundle.hashes["rollout"], "external-joint-v1"
                ),
            ),
            tuple(
                ActorPolicyBinding(actor, "recorded-actions", "reference_opponent")
                for actor in bundle.controlled
            ),
        )
        population_path = provenance / "population.json"
        _dump(population_path, population.to_dict())
        config = bundle.config
        settings = dict(
            scene=provenance / "scene.npz",
            road_graph=provenance / "road_graph.json",
            binding=provenance / "binding.json",
            world_python=Path(config["world_python"]),
            native_path=Path(config["native_path"]),
            population_config=population_path,
            output=output / "session",
            cameras=tuple(config["cameras"]),
            sensor_stride=config.get("sensor_stride", 2),
            desired_speed=config.get("desired_speed", 8.0),
            virtual_camera_actors=tuple(config.get("virtual_camera_actors", [])),
            height_mode=config.get("height_mode", "source-time"),
            visual_policy=False,
            image_write_workers=config.get("image_write_workers", 1),
        )
        phase("cpu_full_replay")
        verification["cpu_preflight_verified_steps"] = _preflight(bundle, settings)
        verification["cpu_initial_preflight_exact"] = True
        phase("load_renderer")
        runtime = _runtime(config)
        phase("initialize_render_session")
        session = _session(runtime, settings)
        session.manifest["challenge_replay"] = {
            "version": VERSION,
            "source_rollout_sha256": bundle.hashes["rollout"],
            "source_report_sha256": bundle.hashes["report"],
            "candidate_id": bundle.candidate["id"],
            "diagnostic_failure": bundle.candidate.get("diagnostic_failure", False),
            "maneuver_validation_pass": bundle.candidate.get("maneuver_validation_pass"),
            "source_kind": bundle.report["kind"],
            "rendered_rgb_used_for_replay_actions": False,
            "original_policy_inputs": "unverified",
        }
        # Keep external-policy RGB use unknown in the strict inspection exporter;
        # the separate bridge declaration proves only this replay's RGB non-use.
        phase("capture_recorded_actions")
        replay_into_session(session, bundle, verification)
        if not session.current["done"]:
            raise ValueError("Replay did not finish its recorded episode")
        phase("export")
        session._publish()
        write_session_viewer(session.output / "manifest.json")
        export_mtgs_episode(session.output / "manifest.json", output / "sequence")
        verification.update(
            status="completed",
            manifest="session/manifest.json",
            viewer="session/viewer.html",
            sequence="sequence/manifest.json",
        )
    except BaseException as error:
        verification.update(
            status="failed", failure={"type": type(error).__name__, "reason": str(error)}
        )
        if session is not None:
            try:
                session._failed(error)
            except Exception as recording_error:
                verification["failure_recording_error"] = str(recording_error)
        _dump(output / "verification.json", verification)
        if not isinstance(error, Exception):
            raise
    finally:
        if session is not None:
            try:
                session.close()
            except Exception as close_error:
                verification["close_error"] = str(close_error)
                if verification["status"] != "failed":
                    verification.update(
                        status="failed",
                        failure={"type": type(close_error).__name__, "reason": str(close_error)},
                    )
                _dump(output / "verification.json", verification)
    phase("completed" if verification["status"] == "completed" else "failed")
    _dump(output / "verification.json", verification)
    return verification


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--candidate")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-diagnostic-failure", action="store_true")
    args = parser.parse_args(argv)
    result = run(
        args.config,
        args.report,
        args.output,
        candidate_id=args.candidate,
        allow_diagnostic_failure=args.allow_diagnostic_failure,
    )
    print(json.dumps(result, allow_nan=False))
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
