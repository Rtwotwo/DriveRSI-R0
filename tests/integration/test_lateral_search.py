import json

import numpy as np
import pytest

from driversi.contracts.road import RoadGraph
from driversi.evaluation.search.lateral_search import SearchBudget, run


def inputs(scene, root):
    scene.times = np.arange(91) * 0.1
    scene.poses = np.zeros((91, 3, 4))
    scene.poses[:, :, 0] = scene.times[:, None] * 6 + [0, 18, -20]
    scene.poses[:, :, 1] = [0, 3.5, -8]
    scene.velocities = np.zeros((91, 3, 3))
    scene.velocities[:, :, 0] = 6
    scene.road_points[scene.road_points[:, 0] == 60, 0] = 160
    scene.save(root / "source.npz")
    graph = RoadGraph({"schema_version": 1, "map_name": "straight", "map_version": None,
                       "origin_xyz": [0, 0, 0], "bounds_local": [-40, -20, 160, 20]}, {
        f"lane:{key}": {"kind": "lane", "points": [[-40, y], [140, y]],
                       "outgoing": [], "speed_limit_mps": None}
        for key, y in (("a", 0), ("b", 3.5))})
    graph.save(root / "graph.json")
    return root / "source.npz", root / "graph.json"


def candidate(**overrides):
    return {"challenger": "b", "responder": "a", "length_m": 30,
            "intervention_tick": 10, **overrides}


def test_search_keeps_rejections_and_selects_only_completed_maneuver(straight_scene, tmp_path):
    paths = inputs(straight_scene, tmp_path)
    out = tmp_path / "search"
    result = run(*paths, [candidate(length_m=3), candidate()], out)
    assert result["status"] == "completed"
    assert result["attempts"][0]["status"] == "rejected"
    assert result["attempts"][1]["maneuver_validation_pass"]
    assert result["selected"] == "candidate-001"
    assert result["charged_joint_steps"] == 200  # two previews + two 90-step branches
    assert not result["trainable"]
    report = json.loads((out / "candidate-001/report.json").read_text())
    assert report["export_status"] == "complete" and set(report["branches"]) == {"baseline", "challenge"}
    with np.load(out / "candidate-001/challenge.npz") as data:
        assert data["presence"].shape == (91, 3)
        assert "route_support_end" in data
        assert json.loads(str(data["metadata"]))["protocol"]["branch"] == "challenge"


def test_budget_stops_mid_preview_without_a_selected_artifact(straight_scene, tmp_path):
    paths = inputs(straight_scene, tmp_path)
    result = run(*paths, [candidate()], tmp_path / "search", budget=SearchBudget(1, 5, 60))
    assert result["status"] == "partial" and result["selected"] is None
    assert result["charged_joint_steps"] == 5
    assert result["attempts"][0]["status"] == "budget_exhausted"
    assert not (tmp_path / "search/candidate-000").exists()


def test_attempt_limit_and_heldout_rejection(straight_scene, tmp_path):
    paths = inputs(straight_scene, tmp_path)
    result = run(*paths, [candidate(length_m=1), candidate()], tmp_path / "search",
                 budget=SearchBudget(1, 1000, 60))
    assert result["status"] == "partial" and len(result["attempts"]) == 1
    straight_scene.metadata["split"] = "navtest"
    straight_scene.save(paths[0])
    with pytest.raises(ValueError, match="Official"):
        run(*paths, [candidate()], tmp_path / "heldout")
    assert not (tmp_path / "heldout").exists()


def render_config(paths, root):
    import sys

    from driversi.contracts.scene import Scene
    from driversi.data.assets.mtgs_assets import file_sha256
    source = Scene.load(paths[0])
    binding = {"inputs": {"scene": {"sha256": file_sha256(paths[0])}},
               "log_name": source.metadata["source_log"], "map_name": source.metadata["map_name"],
               "source_start_timestamp_us": source.metadata.get("source_start_timestamp_us")}
    (root / "binding.json").write_text(json.dumps(binding))
    (root / "asset").write_text("fixture only")
    config = {"scene": str(paths[0]), "road_graph": str(paths[1]), "binding": str(root / "binding.json"),
              "world_python": sys.executable, "cameras": ["CAM_F0"],
              **{key: str(root / "asset") for key in (
                  "native_path", "config_path", "checkpoint_path", "road_block_config")}}
    (root / "render.json").write_text(json.dumps(config))
    return root / "render.json"


def test_lateral_render_binding_and_exact_physics_masks(straight_scene, tmp_path):
    from driversi.rendering.execution.challenge_replay import load_replay_input, verify_world_record
    from driversi.sim.backends.pufferdrive import PufferDriveWorld
    paths = inputs(straight_scene, tmp_path)
    out = tmp_path / "search"
    run(*paths, [candidate()], out)
    config = render_config(paths, tmp_path)
    original_binding = (tmp_path / "binding.json").read_bytes()
    bundle = load_replay_input(config, out / "candidate-000/report.json")
    assert bundle.derived_binding["task_only_derivation"]["all_other_scene_content_equal"]
    assert bundle.derived_binding["inputs"]["scene"]["sha256"] != bundle.derived_binding["task_only_derivation"]["source_scene_sha256"]
    assert not bundle.candidate["diagnostic_failure"]
    actions, transition = None, None
    with PufferDriveWorld(bundle.scene) as world:
        for tick in range(91):
            row = {"build_id": bundle.metadata["backend_build"], "all_actor_ids": list(bundle.scene.actor_ids),
                   "controlled_ids": list(bundle.controlled), "tick": tick, "dt_seconds": 0.1,
                   "state": world.state().tolist(), "motion": world.motion().tolist(),
                   "presence": world.presence().tolist(), "alive": world.observe().alive.tolist(),
                   "done": world.episode_done}
            if tick:
                row.update(policy_actions=actions, executed_actions=actions,
                           valid=transition.valid.tolist(), terminated=transition.terminated.tolist(),
                           truncated=transition.truncated.tolist(),
                           events={k: v.tolist() for k, v in transition.events.items()})
            verify_world_record(bundle, row, tick)
            if tick == 0:
                row["presence"][0] = not row["presence"][0]
                with pytest.raises(ValueError, match="Presence"):
                    verify_world_record(bundle, row, tick)
            if tick < 90:
                actions = dict(zip(bundle.controlled, bundle.arrays["actions"][tick].tolist()))
                transition = world.step(actions)
    assert (tmp_path / "binding.json").read_bytes() == original_binding
    # A changed collision body is not a task-only change, even if the original binding is refreshed.
    straight_scene.dimensions[0, 0] += 0.1
    straight_scene.save(paths[0])
    render_config(paths, tmp_path)
    with pytest.raises(ValueError, match="Only declared driving task"):
        load_replay_input(config, out / "candidate-000/report.json")


def test_uncompleted_maneuver_requires_explicit_diagnostic_render(straight_scene, tmp_path):
    from driversi.rendering.execution.challenge_replay import load_replay_input
    paths = inputs(straight_scene, tmp_path)
    out = tmp_path / "search"
    result = run(*paths, [candidate(length_m=30, intervention_tick=80)], out)
    assert result["selected"] is None and result["attempts"][0]["status"] == "completed"
    config = render_config(paths, tmp_path)
    report = out / "candidate-000/report.json"
    with pytest.raises(ValueError, match="explicitly allow diagnostic"):
        load_replay_input(config, report)
    bundle = load_replay_input(config, report, allow_diagnostic_failure=True)
    assert bundle.candidate["diagnostic_failure"]
    assert bundle.report["trainable"] is False
    data = json.loads(report.read_text())
    data["challenger"] = "replay"
    report.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="provenance"):
        load_replay_input(config, report, allow_diagnostic_failure=True)


def test_search_index_keeps_rejected_and_budget_exhausted_attempts(straight_scene, tmp_path):
    paths = inputs(straight_scene, tmp_path)
    out = tmp_path / "indexed"
    run(*paths, [candidate(length_m=3), candidate()], out,
        budget=SearchBudget(max_joint_steps=15))
    page = (out / "index.html").read_text()
    assert "candidate-000" in page and "candidate-001" in page
    assert "rejected" in page and "budget_exhausted" in page
    assert "不是学习型对手" in page
