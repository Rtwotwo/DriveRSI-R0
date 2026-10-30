"""Build a fixed-source interaction evidence collection, including rejected cases."""

import hashlib
import json
import math
import pickle
import time
import tomllib
from pathlib import Path

import numpy as np

from driversi.agents.traffic_control.routing import inspect_routes
from driversi.contracts.scene import Scene
from driversi.data.scenes.context import compile_context
from driversi.data.road_network.lane_graph import compile_lane_graph
from driversi.data.road_network.maps import attach_nuplan_map
from driversi.data.sources.navsim import (
    audit_horizons,
    compile_frames,
    load_log,
    plan_horizon,
    validate_horizon_request,
)
from driversi.evaluation.benchmarks.following import evaluate_following
from driversi.evaluation.benchmarks.interaction import probe_interaction
from driversi.evaluation.benchmarks.real_scenes import summarize_rollout, verify_replay
from driversi.rendering.inspection.gallery import export_gallery
from driversi.rendering.bev.replay import export_replay
from driversi.sim.execution.rollout import rollout

PROBE_STATUSES = ("verified", "evidence_insufficient", "not_applicable", "failed", "not_run")


def _write_json(path: Path, value: dict) -> None:
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                       encoding="utf-8")
    pending.replace(path)


def interaction_duration(path: Path) -> dict:
    """Count active control intervals, not rendered frames or inferred interactions."""
    with np.load(path, allow_pickle=False) as record:
        valid = record["valid"]
        dt = json.loads(str(record["metadata"]))["dt_seconds"]
        return {
            "actual_seconds": len(valid) * dt,
            "any_controlled_seconds": float(np.count_nonzero(valid.any(axis=1)) * dt),
            "two_controlled_seconds": float(np.count_nonzero(valid.sum(axis=1) >= 2) * dt),
            "per_actor_active_seconds": {
                str(actor): float(count * dt)
                for actor, count in zip(record["controlled_ids"], valid.sum(axis=0), strict=True)
            },
            "interpretation": "active control exposure, not proof of causal interaction",
        }


def _probe_with_views(scene, graph, context, folder: Path) -> dict:
    """Keep probe failure separate from the already verified base rollout."""
    output = folder / "probe"
    relative = Path("real") / folder.name / "probe"
    probe = {"status": "failed", "reason": "Probe did not return a result"}
    try:
        probe = probe_interaction(scene, graph, output)
        if probe.get("status") not in PROBE_STATUSES[:-1]:
            raise ValueError("Unexpected completed interaction probe status")
        probe["report_path"] = (relative / "report.json").as_posix()
        probe["views"] = []
        recordings = probe.get("recordings") or {}
        assets, lengths, seconds = {}, {}, {}
        for branch, name in recordings.items():
            if branch not in ("baseline", "lead_brake", "action_replay_keep", "action_replay_brake"):
                raise ValueError("Unknown probe recording branch")
            path = output / name
            if Path(name).is_absolute() or not path.resolve().is_relative_to(output.resolve()):
                raise ValueError("Probe recording must remain in its output directory")
            assets[branch] = path
            with np.load(path, allow_pickle=False) as record:
                lengths[branch] = len(record["valid"])
                seconds[branch] = lengths[branch] * json.loads(str(record["metadata"]))["dt_seconds"]
        probe["recording_seconds"] = seconds
        probe["comparison_notes"] = []
        for branch, label, reference in (
            ("baseline", "跟驰基准", None),
            ("lead_brake", "前车制动", "baseline"),
            ("action_replay_keep", "后车动作开环 · 前车维持", None),
            ("action_replay_brake", "后车动作开环 · 前车制动", "action_replay_keep"),
        ):
            if branch not in assets:
                continue
            comparison = None
            if reference in assets:
                if lengths[branch] == lengths[reference]:
                    comparison = assets[reference]
                    label += " / 基准对照"
                else:
                    probe["comparison_notes"].append(
                        f"{branch} ({seconds[branch]:g} s) 与 {reference} ({seconds[reference]:g} s) "
                        "提前结束时刻不同，分别展示，未补帧或裁剪。")
            page = output / (branch + ".html")
            export_replay(scene, assets[branch], page, compare_path=comparison,
                          road_graph=graph, scene_context=context)
            probe["views"].append({"label": label, "branch": branch,
                                   "path": (relative / page.name).as_posix(),
                                   "comparison": reference if comparison else None})
    except Exception as error:
        # probe_interaction persists its own partial report before raising.
        # Recover that evidence without changing base-run status or retrying.
        if not probe.get("recordings") and (output / "report.json").is_file():
            try:
                probe = json.loads((output / "report.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
        previous = probe.get("status")
        probe.update(status="failed", reason=f"Probe or presentation error: {type(error).__name__}: {error}",
                     status_before_error=previous)
        if (output / "report.json").is_file():
            probe["report_path"] = (relative / "report.json").as_posix()
    return probe


def _real_case(case: dict, scene_dir: Path, dataset_root: Path, folder: Path,
               duration: float, max_actors: int, *, duration_mode: str | None = None,
               min_duration: float = .1) -> dict:
    item = {"id": Path(case["scene"]).stem, "source_scene": case["scene"],
            "requested_seconds": duration, "status": "compile_rejected",
            "interaction_probe": {"status": "not_run", "reason": "Probe has not run"},
            "qualification": {"source_time": "not_checked", "route_task": "not_checked",
                              "joint_control": "not_checked", "interaction": "not_run"}}
    folder.mkdir()
    stage = "compile"
    try:
        previous = Scene.load(scene_dir / case["scene"])
        if previous.metadata["scenario_id"] != case["scenario_id"]:
            raise ValueError("Frozen scene identity mismatch")
        log_name = previous.metadata["source_log"]
        if Path(log_name).name != log_name:
            raise ValueError("Source log name must be a filename stem")
        log_file = dataset_root / "navsim_logs/mini" / (log_name + ".pkl")
        item.update(map_name=previous.metadata["map_name"], source_log=log_name)
        source = load_log(log_file, trusted=True)
        start = previous.metadata["source_start_frame"]
        previous_duration = float(previous.times[-1] - previous.times[0])
        if duration_mode is not None:
            plan = plan_horizon(source, start=start, duration=duration,
                                mode=duration_mode, min_duration=min_duration)
            item["horizon_plan"] = plan
            item["planned_seconds"] = plan["planned_duration_seconds"]
            item["qualification"]["source_time"] = plan["status"]
            _write_json(folder / "horizon_plan.json", plan)
            if plan["status"] != "supported":
                raise ValueError("Horizon rejected: " + ", ".join(plan["rejection_reasons"]))
        else:
            item["planned_seconds"] = duration
        horizon = audit_horizons(source, start=start,
                                 durations=sorted({previous_duration, item["planned_seconds"]}))
        _write_json(folder / "source_horizons.json", horizon)
        duration_options = ({} if duration_mode is None else
                            {"duration_mode": duration_mode, "min_duration": min_duration})
        scene = compile_frames(source, start=start, duration=duration,
                               split=previous.metadata["split"], max_actors=max_actors,
                               **duration_options)
        identity = ("scenario_id", "source_log", "map_name", "source_start_timestamp_us")
        if any(scene.metadata[key] != previous.metadata[key] for key in identity):
            raise ValueError("Source no longer matches frozen scene identity")
        scene.metadata["source_file"] = str(log_file)
        scene.metadata["gallery_source_scene"] = case["scene"]
        population = {"previous_duration_seconds": previous_duration,
                      "previous_world": len(previous.actor_ids), "world": len(scene.actor_ids),
                      "controlled": int(scene.controlled.sum()),
                      "retained_previous_ids": sorted(set(previous.actor_ids) & set(scene.actor_ids)),
                      "lost_previous_ids": sorted(set(previous.actor_ids) - set(scene.actor_ids)),
                      "new_world_ids": sorted(set(scene.actor_ids) - set(previous.actor_ids))}
        item["population"] = population
        attach_nuplan_map(scene, dataset_root / "maps")
        scene.save(folder / "scene.npz")
        del source
        stage = "routes"
        graph = compile_lane_graph(scene, dataset_root / "maps")
        graph.save(folder / "lanes.json")
        routing = inspect_routes(scene, graph)
        item["qualification"]["route_task"] = "resolved" if routing["all_resolved"] else "unresolved"
        _write_json(folder / "routes.json", routing)
        item["route_warnings"] = [actor for actor, value in routing["actors"].items()
                                  if value.get("initial_route_feasibility", {}).get(
                                      "insufficient_stopping_distance")]
        if not routing["all_resolved"]:
            item.update(status="unresolved_task", error="Unresolved routes: " + ", ".join(
                actor for actor, value in routing["actors"].items() if not value["resolved"]))
        else:
            stage = "rollout"
            rollout(scene, folder / "rollout.npz", policy="follow", road_graph=graph)
            item["summary"] = summarize_rollout(scene, folder / "rollout.npz")
            item["interaction"] = interaction_duration(folder / "rollout.npz")
            item["qualification"]["joint_control"] = (
                "available" if item["interaction"]["two_controlled_seconds"] > 0 else "unavailable")
            item["actual_seconds"] = item["interaction"]["actual_seconds"]
            stage = "verification"
            item["verification"] = verify_replay(scene, folder / "rollout.npz")
            failures = [name for name, value in item["verification"].items() if value is False]
            stage = "context_and_viewer"
            context = compile_context(scene, log_file, dataset_root / "maps", trusted=True)
            context.save(folder / "context.json")
            export_replay(scene, folder / "rollout.npz", folder / "viewer.html",
                          road_graph=graph, scene_context=context)
            item["viewer"] = (Path("real") / folder.name / "viewer.html").as_posix()
            item.update(status="contract_failure" if failures else "complete")
            if failures:
                item["error"] = "Failed recording contracts: " + ", ".join(failures)
                item["interaction_probe"]["reason"] = "Base recording contract verification failed"
            elif duration_mode is not None:
                item["interaction_probe"] = _probe_with_views(scene, graph, context, folder)
                item["qualification"]["interaction"] = item["interaction_probe"]["status"]
    except (ValueError, KeyError, TypeError, OSError, RuntimeError, EOFError,
            pickle.UnpicklingError) as error:
        item.update(status="compile_rejected" if stage == "compile" else "runtime_error",
                    error=f"{stage}: {type(error).__name__}: {error}")
    item["details"] = (Path("real") / folder.name / "case.json").as_posix()
    _write_json(folder / "case.json", item)
    return item


def _gallery_cases(config: dict, selection: dict, following_config: dict) -> list[dict]:
    """Resolve budgets before execution without changing any frozen source start."""
    version = config.get("schema_version")
    if type(version) is not int or version not in (1, 2):
        raise ValueError("Gallery configuration requires schema v1 or v2")
    cases = selection.get("cases", [])
    names = [case["scene"] for case in cases]
    if (selection.get("schema_version") != 1 or not cases or len(set(names)) != len(names)
            or any(Path(name).name != name for name in names)):
        raise ValueError("Require nonempty unique frozen scene filenames")

    def positive_seconds(value):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0):
            raise ValueError("Gallery durations must be finite positive seconds")
        return float(value)

    requested = positive_seconds(config.get("duration_seconds"))
    minimum = positive_seconds(config.get("min_duration_seconds", .1))
    mode = config.get("duration_mode", "exact")
    if version == 1:
        if requested != 9.0 or config.get("duration_mode", "exact") != "exact":
            raise ValueError("Gallery schema v1 requires an exact 9-second duration")
        if config.get("cases") or any("duration_seconds" in case for case in cases):
            raise ValueError("Per-case durations require gallery schema v2")
        if following_config["duration_seconds"] != requested:
            raise ValueError("Schema v1 real and paired scenarios require the same horizon")
    elif config.get("duration_mode", "exact") not in ("exact", "fit_source"):
        raise ValueError("Gallery duration_mode must be exact or fit_source")
    overrides = config.get("cases", [])
    if not isinstance(overrides, list):
        raise ValueError("Per-case budgets must be an array of case tables")
    indexed = {}
    for override in overrides:
        if (set(override) != {"scene", "duration_seconds"} or override["scene"] not in names
                or override["scene"] in indexed):
            raise ValueError("Per-case budgets require unique frozen scene names and durations")
        indexed[override["scene"]] = positive_seconds(override["duration_seconds"])
    resolved = []
    for case in cases:
        duration = positive_seconds(indexed.get(case["scene"], case.get("duration_seconds", requested)))
        validate_horizon_request(duration, mode, minimum)
        resolved.append({**case, "duration_seconds": duration})
    return resolved


def build_interaction_gallery(config_path: Path, scene_dir: Path, dataset_root: Path,
                               output: Path, *, trusted: bool = False) -> dict:
    """A completed collection retains failures; completion is not qualification."""
    if not trusted:
        raise ValueError("Gallery compilation requires --trusted-pickle")
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    selection_path = config_path.parent / config["selection_file"]
    following_path = config_path.parent / config["following_config_file"]
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    following_config = tomllib.loads(following_path.read_text(encoding="utf-8"))
    cases = _gallery_cases(config, selection, following_config)
    case_count = len(following_config["speeds_mps"]) * len(following_config["bumper_gaps_m"])
    selected = config["selected_following_cases"]
    if (not selected or len(set(selected)) != len(selected)
            or any(type(index) is not int or not 0 <= index < case_count for index in selected)):
        raise ValueError("Invalid preselected following case indices")
    if type(config["max_actors"]) is not int or not 1 <= config["max_actors"] <= 32:
        raise ValueError("Invalid physical population capacity")
    output.mkdir(parents=True, exist_ok=False)
    (output / "real").mkdir()
    (output / "config.toml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    _write_json(output / "selection.json", selection)
    report = {"schema_version": config["schema_version"], "kind": "interaction_gallery", "status": "running",
              "requested_seconds": config["duration_seconds"], "real_cases": [],
              "configuration": config, "selection_rule": selection.get("selection_rule"),
              "following": {"selected_cases": [], "duration_seconds": following_config["duration_seconds"],
                            "intervention_seconds": following_config.get("intervention_seconds")},
              "requested_cases": cases,
              "scope": "development evidence; no long-world, traffic-rule or training certification"}
    report["code_sha256"] = {path.relative_to(Path(__file__).parents[2]).as_posix():
                              hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in sorted(Path(__file__).parents[2].rglob("*.py"))}
    started = time.perf_counter()

    def save():
        _write_json(output / "report.json", report)
        export_gallery(report, output / "index.html")

    try:
        save()
        evaluate_following(following_path, output / "following")
        following = json.loads((output / "following/report.json").read_text(encoding="utf-8"))
        checks = [value for case in following["cases"] for value in case["checks"].values()]
        report["following"].update(
            report_path="following/report.json", case_count=len(following["cases"]),
            checks_total=len(checks), checks_passed=sum(checks), passed=following["passed"],
            independent_templates=following["independent_templates"],
            backend_build=following["backend_build"],
        )
        for index in selected:
            item = following["cases"][index]
            folder = output / "following" / f"case-{index:03d}"
            views = []
            for label, mode, alternative, kind in (
                ("前车减速 / 维持：反应式后车", "reactive", "brake", "lead_intervention"),
                ("远车减速 / 维持：无关干预", "reactive", "far", "far_negative"),
                ("前车减速 / 维持：回放后车", "replay", "brake", "replay_negative"),
            ):
                scene = Scene.load(folder / f"{mode}-scene.npz")
                page = folder / f"{kind}.html"
                export_replay(scene, folder / f"{mode}-{alternative}-rollout.npz", page,
                              compare_path=folder / f"{mode}-keep-rollout.npz")
                views.append({"label": label, "kind": kind, "path": page.relative_to(output).as_posix()})
            report["following"]["selected_cases"].append({
                "case_index": index, "speed_mps": item["speed_mps"], "gap_m": item["initial_bumper_gap_m"],
                "passed": item["passed"], "checks_total": len(item["checks"]),
                "checks_passed": sum(item["checks"].values()), "metrics": item["modes"], "views": views,
            })
        save()
        for case in cases:
            duration_options = ({} if config["schema_version"] == 1 else
                                {"duration_mode": config.get("duration_mode", "exact"),
                                 "min_duration": config.get("min_duration_seconds", .1)})
            report["real_cases"].append(_real_case(
                case, scene_dir, dataset_root, output / "real" / Path(case["scene"]).stem,
                case["duration_seconds"], config["max_actors"], **duration_options,
            ))
            save()
        report["elapsed_seconds"] = time.perf_counter() - started
        report["status"] = "complete"
        report["stats"] = {status: sum(case["status"] == status for case in report["real_cases"])
                           for status in ("complete", "compile_rejected", "unresolved_task",
                                          "runtime_error", "contract_failure")}
        report["interaction_probe_counts"] = {
            status: sum((case.get("interaction_probe") or {}).get("status", "not_run") == status
                        for case in report["real_cases"])
            for status in PROBE_STATUSES}
        save()
    except Exception as error:
        report.update(status="error", error=f"{type(error).__name__}: {error}",
                      elapsed_seconds=time.perf_counter() - started)
        # Preserve the original failure if the storage/rendering failure also
        # prevents finalizing the partial report.
        try:
            save()
        except Exception:
            pass
        raise
    return {"output": str(output), "status": report["status"], "stats": report["stats"],
            "following_passed": following["passed"], "paired_checks": len(checks),
            "interaction_probe_counts": report["interaction_probe_counts"],
            "probe_failures": report["interaction_probe_counts"]["failed"]}
