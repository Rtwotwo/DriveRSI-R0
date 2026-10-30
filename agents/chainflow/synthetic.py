"""Evidence-bound synthetic transport for the unmodified ChainFlow stage1 model.

A branch, not an operator-supplied trainable flag, owns every identity, state,
image and metric cache. Native math runs in the isolated NAVSIM environment.
"""

import hashlib
import json
import lzma
import pickle
from pathlib import Path

from driversi.agents.backends.native import digest

CAMERAS = ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0")


def validate_visual_admission(path):
    """Use the same core verifier across the native Python 3.9 boundary."""
    import subprocess
    import sys

    from driversi.agents.backends.native import project_root

    if sys.version_info >= (3,10):
        from driversi.rendering.diagnostics.worldengine_admission import validate_admission
        return validate_admission(path)
    runtime = project_root()/".venv/bin/python"
    code = "import json,sys;from driversi.rendering.diagnostics.worldengine_admission import validate_admission;print(json.dumps(validate_admission(sys.argv[1]),allow_nan=False))"
    result = subprocess.run([str(runtime),"-c",code,str(path)],capture_output=True,text=True,timeout=300,check=True)
    return json.loads(result.stdout)


def load_physical_branch(scene_path, rollout_path):
    """Read sealed numeric transport without importing the simulation runtime."""
    from types import SimpleNamespace

    import numpy as np

    with np.load(scene_path,allow_pickle=False) as payload:
        scene = SimpleNamespace(metadata=json.loads(str(payload["metadata"])),actor_ids=tuple(payload["actor_ids"].tolist()),kinds=payload["kinds"].copy(),dimensions=payload["dimensions"].copy(),poses=payload["poses"].copy(),times=payload["times"].copy())
    with np.load(rollout_path,allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata"]))
        ids = tuple(payload["actor_ids"].tolist())
        run = dict(states=payload["states"].copy(),presence=payload["presence"].copy(),metadata=metadata)
    states,presence = run["states"],run["presence"]
    if ids != scene.actor_ids or len(ids) != len(set(ids)) or "ego" not in ids or metadata.get("scene_metadata") != scene.metadata:
        raise ValueError("Physical scene and rollout identities differ")
    if states.ndim != 3 or states.shape[1:] != (len(ids),13) or presence.shape != states.shape[:2] or presence.dtype.kind != "b" or not np.isfinite(states).all() or metadata.get("dt_seconds") != .1:
        raise ValueError("Malformed native physical state transport")
    if scene.dimensions.shape != (len(ids),3) or not np.isfinite(scene.dimensions).all() or (scene.dimensions<=0).any():
        raise ValueError("Malformed physical object dimensions")
    return scene,run


def _read(path):
    return json.loads(Path(path).read_text())


def _sealed(files, paths=()):
    if not isinstance(files, dict) or not files:
        raise ValueError("Synthetic evidence requires sealed files")
    if any(str(path) not in files for path in paths):
        raise ValueError("Required synthetic evidence is not sealed")
    for name, sha in files.items():
        if not Path(name).is_absolute() or digest(name) != sha:
            raise ValueError("Synthetic evidence changed: " + name)


def synthetic_token(branch_files, anchor_tick):
    """Unique physical branch identity; never reuse a parent benchmark token."""
    named = [(Path(name).name,sha) for name,sha in branch_files.items()]
    if len({name for name,_ in named}) != len(named):
        raise ValueError("Ambiguous physical branch filenames")
    payload = json.dumps([sorted(named), anchor_tick], separators=(",", ":"))
    return "synthetic_" + hashlib.sha256(payload.encode()).hexdigest()[:32]



def validate_source_train_membership(parent, source_frames, membership=None, *, metadata=None, anchor_tick=None):
    """Separate official split membership from the physically reached anchor.

    A membership token never supplies navigation, time, geometry or visual
    identity. Both tokens must be present at their exact indices in one sealed
    official training log; an arbitrary nearest token is never inferred.
    """
    from driversi_agents.chainflow.data import official_train_logs, official_train_tokens

    if not isinstance(parent, dict) or set(parent) != {"source_log", "token", "log_path", "anchor_index"}:
        raise ValueError("Malformed actual synthetic parent")
    source_log = parent["source_log"]
    if source_log not in official_train_logs() or Path(parent["log_path"]).stem != source_log:
        raise ValueError("Synthetic source is not the exact official navtrain log")

    def exact_frame(record, role):
        index = record["anchor_index"]
        if type(index) is not int or not 3 <= index < len(source_frames):
            raise ValueError(role + " requires an actual raw frame index")
        frame = source_frames[index]
        if frame["log_name"] != source_log or frame["token"] != record["token"]:
            raise ValueError(role + " token/log not present at declared raw anchor")
        return frame

    exact_frame(parent, "Reached parent")
    if membership is None:
        if parent["token"] not in official_train_tokens():
            raise ValueError("Nonofficial reached anchor requires explicit source_train_membership")
        return None
    if not isinstance(membership, dict) or set(membership) != {"source_log", "token", "anchor_index", "reference_role"}:
        raise ValueError("Malformed source_train_membership")
    if membership["reference_role"] != "official_split_membership_not_physical_anchor":
        raise ValueError("Training membership cannot replace the physical anchor")
    if membership["source_log"] != source_log:
        raise ValueError("Membership and reached anchor must use the same exact source log")
    if membership["token"] not in official_train_tokens():
        raise ValueError("Membership token is not in the exact official navtrain allowlist")
    exact_frame(membership, "Official membership")
    if metadata is None or anchor_tick is None:
        raise ValueError("Derived membership requires sealed physical scene evidence")
    if metadata.get("source_train_membership") != membership or metadata.get("split") != "train" or metadata.get("source_log") != source_log:
        raise ValueError("Training membership differs from sealed physical scene")
    raw_path = metadata.get("navsim_source", {}).get("log_path")
    if not raw_path or Path(raw_path).resolve() != Path(parent["log_path"]).resolve():
        raise ValueError("Membership raw log differs from physical branch source")
    start = metadata.get("source_start_frame")
    if type(anchor_tick) is not int or anchor_tick < 0 or anchor_tick % 5 or type(start) is not int or start < 3 or start >= len(source_frames):
        raise ValueError("Derived physical anchor must align with a real 2Hz source frame")
    if parent["anchor_index"] != start + anchor_tick // 5 or source_frames[start]["token"] != metadata.get("scenario_id") or source_frames[start]["log_name"] != source_log:
        raise ValueError("Membership cannot replace the actual reached physical anchor")
    return dict(membership)


def validate_admission(row, outer_files=None):
    from driversi.rsi.correction.trajectory_admission import validate_trajectory_evidence

    ref = row.get("admission")
    if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}:
        raise ValueError("Synthetic ChainFlow requires canonical sealed visual admission")
    path = ref["path"]
    if not Path(path).is_absolute() or digest(path) != ref["sha256"]:
        raise ValueError("Synthetic admission receipt changed")
    proof = _read(path)
    expected_keys = {"schema_version", "kind", "parent", "branch", "visual_admission", "native", "files"}
    version = proof.get("schema_version")
    if version in (2, 3, 4):
        expected_keys.add("source_train_membership")
    if version in (3, 4):
        expected_keys.add("image_contract")
    if type(version) is not int or version not in (1, 2, 3, 4) or set(proof) != expected_keys or proof["kind"] != "chainflow_synthetic_admission":
        raise ValueError("Malformed synthetic admission receipt")
    if version in (3, 4):
        from driversi.data.datasets.synthetic_storage import HISTORY_ONLY_IMAGE_CONTRACT, ANCHOR_ONLY_IMAGE_CONTRACT
        if proof["image_contract"] != ({3: HISTORY_ONLY_IMAGE_CONTRACT, 4: ANCHOR_ONLY_IMAGE_CONTRACT}[version]):
            raise ValueError("History-only admission requires the explicit image contract")
    parent, branch, native, files = (proof[k] for k in ("parent", "branch", "native", "files"))
    if set(parent) != {"source_log", "token", "log_path", "anchor_index"} or parent["source_log"] != row["source_log"]:
        raise ValueError("Synthetic parent provenance differs")
    if version == 1 or (version in (3, 4) and proof["source_train_membership"] is None):
        from driversi_agents.chainflow.data import official_train_logs, official_train_tokens
        if parent["source_log"] not in official_train_logs() or parent["token"] not in official_train_tokens():
            raise ValueError("Nonofficial reached anchor requires explicit source_train_membership")
    if set(branch) != {"report", "scene", "rollout", "trajectory_window"}:
        raise ValueError("Incomplete sealed physical branch")
    if set(native) != {"log_path", "sensor_root", "token", "metric_cache", "training_metric_cache", "cache_receipt"}:
        raise ValueError("Incomplete native synthetic transport")
    if any(native[k] != row[k] for k in native if k != "cache_receipt"):
        raise ValueError("Native synthetic payload differs from admitted payload")
    if row["anchor_index"] != 3 or row["token"] == parent["token"] or not row["token"].startswith("synthetic_"):
        raise ValueError("Synthetic token must be distinct from official parent token")
    visual_ref = proof["visual_admission"]
    if not isinstance(visual_ref, dict) or set(visual_ref) != {"path", "sha256"}:
        raise ValueError("Independent visual evidence reference required")
    _sealed(files, [parent["log_path"], *branch.values(), native["log_path"], native["metric_cache"], native["training_metric_cache"], native["cache_receipt"], visual_ref["path"]])
    if files[visual_ref["path"]] != visual_ref["sha256"]:
        raise ValueError("Visual certificate reference changed")
    if outer_files is not None:
        if outer_files.get(path) != ref["sha256"] or any(outer_files.get(k) != v for k, v in files.items()):
            raise ValueError("Manifest must seal the complete synthetic evidence closure")
    report = _read(branch["report"])
    validate_trajectory_evidence(report["trajectory_evidence"])
    if report.get("source_log") != parent["source_log"] or report.get("teacher_qualified") is not True:
        raise ValueError("Unqualified or unrelated teacher branch")
    if report.get("trajectory_window_sha256") != files[branch["trajectory_window"]]:
        raise ValueError("Trajectory window differs from selected branch")
    for name in ("scene", "rollout"):
        if report.get("files", {}).get(branch[name]) != files[branch[name]]:
            raise ValueError("Selected physical branch changed")
    if version == 2 or (version in (3, 4) and proof["source_train_membership"] is not None):
        if not isinstance(proof["source_train_membership"], dict):
            raise ValueError("Derived admission requires explicit source membership")
        with open(parent["log_path"], "rb") as stream:
            source_frames = pickle.load(stream)
        scene, _ = load_physical_branch(branch["scene"], branch["rollout"])
        validate_source_train_membership(parent, source_frames, proof["source_train_membership"], metadata=scene.metadata, anchor_tick=report["anchor_tick"])
    expected_token = synthetic_token({branch[k]: files[branch[k]] for k in ("scene", "rollout")}, report["anchor_tick"])
    if row["token"] != expected_token:
        raise ValueError("Synthetic token does not identify this physical branch")
    visual = validate_visual_admission(visual_ref["path"])
    if visual.get("admitted") is not True or visual.get("source_log") != parent["source_log"] or visual.get("parent_token") != parent["token"]:
        raise ValueError("Independent visual admission rejected or belongs to another source")
    if visual.get("branch_report_sha256") != files[branch["report"]] or visual.get("trajectory_window") != branch["trajectory_window"]:
        raise ValueError("Visual admission belongs to another branch/window")
    if any(files.get(k) != v for k, v in visual.get("files", {}).items()):
        raise ValueError("Visual evidence not sealed by dataset")
    receipt = _read(native["cache_receipt"])
    if receipt.get("kind") != "chainflow_counterfactual_cache_v1" or receipt.get("token") != row["token"]:
        raise ValueError("Native cache provenance differs")
    if receipt.get("branch_files") != {branch[k]: files[branch[k]] for k in ("scene", "rollout")}:
        raise ValueError("Native scoring context belongs to another branch")
    if "trajectory_admission_profile" in receipt:
        expected_profile = report["trajectory_evidence"].get("admission_profile", "strict_world95_v1")
        if receipt["trajectory_admission_profile"] != expected_profile:
            raise ValueError("Native scoring policy differs from sealed physical branch")
        validate_counterfactual_score(receipt["score"], expected_profile)
    if receipt.get("cache_files") != {native[k]: files[native[k]] for k in ("metric_cache", "training_metric_cache")}:
        raise ValueError("Native scoring cache changed")
    return proof


def validate_window(row, window, proof):
    """Rebuild ego/annotation expectations from executed physics, never from GT."""
    import numpy as np

    from driversi.data.datasets.navsim_synthetic import (
        _rotation,
        causal_signal_records,
        frame_geometry,
        rear_axle_states,
    )

    parent, branch, native = (proof[k] for k in ("parent", "branch", "native"))
    with open(parent["log_path"], "rb") as stream:
        source = pickle.load(stream)
    index = parent["anchor_index"]
    if type(index) is not int or index < 0 or index >= len(source) or source[index]["token"] != parent["token"] or source[index]["log_name"] != parent["source_log"]:
        raise ValueError("Synthetic parent token/log not present at declared raw anchor")
    scene, run = load_physical_branch(branch["scene"],branch["rollout"])
    if scene.metadata.get("split") != "train" or scene.metadata.get("source_log") != parent["source_log"]:
        raise ValueError("Executed branch is not the exact official training parent")
    report = _read(branch["report"])
    anchor = report["anchor_tick"]
    start = scene.metadata["source_start_frame"]
    if anchor % 5 or index != start + anchor // 5 or source[start]["token"] != scene.metadata["scenario_id"]:
        raise ValueError("Official parent must be the reached anchor, not another log token")
    validate_source_train_membership(parent, source, proof.get("source_train_membership"), metadata=scene.metadata, anchor_tick=anchor)
    if proof.get("source_train_membership") != scene.metadata.get("source_train_membership"):
        raise ValueError("Native source membership differs from the sealed physical branch")
    ticks = list(range(anchor - 15, anchor + 51, 5))
    if len(window) != 14 or ticks[0] < 2 or ticks[-1] >= len(run["states"]):
        raise ValueError("Four new histories and ten real executed future points required")
    rear = rear_axle_states(run["states"], scene.actor_ids.index("ego"))
    origin = np.asarray(scene.metadata["origin_xyz"])
    timestamp = scene.metadata["navsim_source"]["anchor_timestamp_us"]
    log_id = Path(native["log_path"]).stem
    parent_frame = source[index]
    timeline = [dict(timestamp_us=timestamp+round(item["time"]*1e6), records=[[str(i),bool(red)] for i,red in item["records"]],available=item.get("available",False)) for item in scene.metadata["traffic_lights"]]
    root = Path(native["sensor_root"]).resolve(strict=True)
    if log_id == parent["source_log"]:
        raise ValueError("Synthetic native log must have a distinct identity")
    visual = _read(proof["visual_admission"]["path"])
    render_path = Path(visual["render_report"])
    render = _read(render_path)
    rendered = {f["tick"]:f for f in render["frames"]}
    from driversi.data.datasets.synthetic_storage import history_only_images, image_frame_indices
    history_only = history_only_images(proof.get("image_contract"))
    rendered_indices = image_frame_indices(proof.get("image_contract"))
    rendered_contract = render.get("image_contract")
    if (history_only and rendered_contract != proof["image_contract"]) or (not history_only and rendered_contract not in (None, "history_future_rgb_v1")):
        raise ValueError("Admitted render and native image contracts differ")
    required_ticks = [ticks[i] for i in rendered_indices] if history_only else ticks
    if len(rendered) != len(render["frames"]) or (history_only and set(rendered) != set(required_ticks)) or not set(required_ticks) <= set(rendered):
        raise ValueError("Rendered image times do not match the declared input window")
    for index, (frame, tick) in enumerate(zip(window, ticks)):
        pose = rear[tick]
        expected_token = row["token"] if index == 3 else row["token"] + "_" + str(tick)
        if frame["token"] != expected_token or frame["frame_idx"] != index:
            raise ValueError("Native synthetic frame identity differs")
        if frame["log_name"] != log_id or int(frame["timestamp"]) != timestamp + tick * 100000:
            raise ValueError("Synthetic frame identity or physical time differs")
        expected_signals, signals_known = causal_signal_records(timeline,int(frame["timestamp"]))
        if not signals_known or frame["traffic_lights"] != expected_signals or frame["roadblock_ids"] != [str(v) for v in parent_frame["roadblock_ids"]]:
            raise ValueError("Synthetic navigation or signals differ from the declared source task")
        np.testing.assert_array_equal(frame["driving_command"],parent_frame["driving_command"])
        np.testing.assert_allclose(frame["ego2global_translation"], pose[:3] + origin, atol=1e-7, rtol=0)
        q = np.asarray(frame["ego2global_rotation"])
        actual_yaw = 2 * np.arctan2(q[3], q[0])
        if abs(np.arctan2(np.sin(actual_yaw-pose[3]), np.cos(actual_yaw-pose[3]))) > 1e-7:
            raise ValueError("Synthetic heading is not executed heading")
        velocity = (rear[tick, :3]-rear[tick-1, :3]) / .1
        previous = (rear[tick-1, :3]-rear[tick-2, :3]) / .1
        rotation = _rotation(pose[3])
        dynamics = np.concatenate(((velocity@rotation)[:2], (((velocity-previous)/.1)@rotation)[:2]))
        np.testing.assert_allclose(frame["ego_dynamic_state"], dynamics, atol=1e-6, rtol=0)
        expected = frame_geometry(scene, {"state":run["states"][tick], "presence":run["presence"][tick]}, pose)
        for key in ("gt_boxes", "gt_velocity_3d"):
            np.testing.assert_array_equal(frame["anns"][key], expected[key])
        for key in ("gt_names", "track_tokens"):
            if list(frame["anns"][key]) != expected[key]:
                raise ValueError("Synthetic world population differs from executed branch")
        if history_only and index not in rendered_indices:
            if any(any(value is not None for value in info.values()) for info in frame["cams"].values()):
                raise ValueError("Unrendered frames cannot contain images or calibration")
            continue
        for camera in CAMERAS:
            image = (root / frame["cams"][camera]["data_path"]).resolve(strict=True)
            if not image.is_relative_to(root) or str(image) not in proof["files"]:
                raise ValueError("Synthetic history/future image not independently sealed")
            capture = rendered[tick]["cameras"][camera]
            original = (render_path.parent / capture["file"]).resolve(strict=True)
            if proof["files"].get(str(original)) != proof["files"][str(image)]:
                raise ValueError("Native image differs from the admitted rendered tick/camera")
            calibration = frame["cams"][camera]
            transform = np.asarray(capture["camera_to_rear_axle"])
            for key, expected in (("sensor2lidar_rotation",transform[:3,:3]),("sensor2lidar_translation",transform[:3,3]),("cam_intrinsic",capture["distorted_intrinsic"]),("distortion",capture["distortion"])):
                np.testing.assert_array_equal(calibration[key],expected)
    return scene, run, anchor


def validate_cache_context(row, cache, proof, scene, run, anchor, *, training):
    """Check the actual cache geometry at every 10Hz tick, not just a receipt flag."""
    import numpy as np
    from shapely.geometry import Polygon

    from driversi.data.datasets.navsim_synthetic import rear_axle_states

    receipt = _read(proof["native"]["cache_receipt"])
    rear = rear_axle_states(run["states"], scene.actor_ids.index("ego"))
    origin = np.asarray(scene.metadata["origin_xyz"])
    actual = [cache.ego_state.rear_axle.x, cache.ego_state.rear_axle.y, cache.ego_state.rear_axle.heading]
    expected = [rear[anchor,0]+origin[0], rear[anchor,1]+origin[1], rear[anchor,3]]
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-7)
    for i in range(51):
        tick = anchor + i
        present = [j for j,t in enumerate(scene.actor_ids) if t != "ego" and run["presence"][tick,j]]
        expected_tokens = {scene.actor_ids[j] for j in present}
        occupancy = cache.observation[i]
        actual_tokens = set(occupancy.tokens)
        light_prefix = cache.observation.red_light_token
        actual_tokens = {token for token in actual_tokens if light_prefix not in token}
        if actual_tokens != expected_tokens:
            raise ValueError("Counterfactual scoring cache drops or borrows physical objects")
        for j in present:
            state = run["states"][tick,j]
            length,width = scene.dimensions[j,:2]
            local = np.array([[length/2,width/2],[length/2,-width/2],[-length/2,-width/2],[-length/2,width/2]])
            c,s = np.cos(state[3]),np.sin(state[3])
            polygon = Polygon(local@np.array([[c,s],[-s,c]]) + state[:2] + origin[:2])
            if polygon.symmetric_difference(occupancy[scene.actor_ids[j]]).area > 1e-5:
                raise ValueError("Counterfactual cache object pose differs from physical branch")
    if training:
        progress = np.asarray(cache.pdm_progress)
        if progress.size != 1 or not np.isfinite(progress).all() or float(progress.reshape(-1)[0]) < 0 or not np.allclose(progress, receipt["native_reference_progress"], rtol=0, atol=1e-8):
            raise ValueError("Native proposal reference progress changed")


def validate_counterfactual_score(score, admission_profile="strict_world95_v1"):
    """Use the producer's sealed policy without changing native scoring math."""
    import math

    from driversi.rsi.correction.trajectory_admission import trajectory_policy

    policy = trajectory_policy(admission_profile)
    value = score.get("score")
    valid = type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1
    if not valid or not (value > 0.95 if policy["strict_gt"] else value >= 0.95):
        raise ValueError("Counterfactual PDM does not admit this future")
    if policy["extra_world_vetoes"] and score.get("driving_direction_compliance") != 1:
        raise ValueError("Counterfactual direction does not meet strict trajectory policy")


def write_counterfactual_caches(context, egos, observations, output, *, native_log_name, token, branch_files, admission_profile="strict_world95_v1"):
    """Persist unmodified NAVSIM cache classes from the reached branch context.

    Caller must supply the 51 actually executed states/observations and the
    before-branch PDM reference from prepare_reference, including trajectory.
    This produces evidence; only the separate visual gate can admit training.
    """
    import numpy as np
    from navsim.planning.metric_caching.metric_cache import MetricCache
    from navsim.planning.metric_caching.train_metric_chache import MetricCache as TrainingCache
    from navsim.planning.simulation.planner.pdm_planner.observation.pdm_observation import (
        PDMObservation,
    )
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

    from driversi.evaluation.metrics.counterfactual_pdm import score_execution

    _sealed(branch_files)
    if not token.startswith("synthetic_") or not native_log_name.startswith("synthetic_"):
        raise ValueError("Distinct synthetic native identity required")
    score = score_execution(context, egos, observations)
    validate_counterfactual_score(score, admission_profile)
    sampling = TrajectorySampling(num_poses=40, interval_length=.1)
    observation = PDMObservation(TrajectorySampling(num_poses=50, interval_length=.1), sampling,100,observation_sample_res=1)
    observation.update_detections_tracks(observations)
    # Use the student's released evaluation conversion, rather than copying
    # another runtime's prepared float array or its tiny interpolation residuals.
    from navsim.evaluate.pdm_score import get_trajectory_as_array
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator

    raw_reference = get_trajectory_as_array(context["trajectory"],sampling,context["initial_ego"].time_point)
    native_reference = PDMSimulator(sampling).simulate_proposals(raw_reference[None],context["initial_ego"])
    scorer = PDMScorer(sampling)
    scorer.score_proposals(native_reference, observation, context["centerline"],context["route_lane_ids"],context["drivable_area_map"])
    progress = float(scorer._progress_raw[0] * scorer._multi_metrics[:,0].prod())
    out = Path(output)
    evaluation_path = out / "evaluation" / native_log_name / "synthetic" / token / "metric_cache.pkl"
    training_path = out / "training" / native_log_name / "synthetic" / token / "metric_cache.pkl"
    if evaluation_path.exists() or training_path.exists():
        raise FileExistsError("Synthetic caches are immutable")
    evaluation = MetricCache(evaluation_path,context["trajectory"],context["initial_ego"],observation,context["centerline"],context["route_lane_ids"],context["drivable_area_map"])
    training = TrainingCache(training_path,context["initial_ego"],observation,context["centerline"],context["route_lane_ids"],context["drivable_area_map"],np.asarray([progress]))
    for path, cache in ((evaluation_path,evaluation),(training_path,training)):
        path.parent.mkdir(parents=True)
        path.write_bytes(lzma.compress(pickle.dumps(cache,protocol=4),preset=0))
    receipt = dict(kind="chainflow_counterfactual_cache_v1",token=token,native_log_name=native_log_name,branch_files=branch_files,cache_files={str(p):digest(p) for p in (evaluation_path,training_path)},anchor_timestamp_us=context["anchor_timestamp_us"],native_reference_progress=progress,reference_conversion="unchanged_chainflow_get_trajectory_as_array_then_native_simulator",actual_world_steps=50,score=score,trajectory_admission_profile=admission_profile)
    receipt_path = out / "receipt.json"
    receipt_path.write_text(json.dumps(receipt,indent=2,allow_nan=False))
    return dict(metric_cache=str(evaluation_path),training_metric_cache=str(training_path),cache_receipt=str(receipt_path))


def validate_rendered_calibration(report, capture):
    """Bind refined optics to the actual certified rendering backend."""
    protocols = {
        "worldengine_portable": "reconstruction_colmap_pose_intrinsics_distortion",
        "mtgs_native": "native_checkpoint_optimized_camera_and_colmap_optics",
    }
    backend = report.get("renderer_backend", "worldengine_portable")
    if backend not in protocols or capture.get("calibration_source") != protocols[backend]:
        raise ValueError("Refined reconstruction optics do not match the rendering backend")
    if capture.get("camera_to_rear_axle_reference") != "reconstruction_refined_nuplan_rear_axle":
        raise ValueError("Refined camera must identify its rear-axle reference")


def export_native_frames(scene_path, rollout_path, render_report, output, *, anchor_tick=20, future_points=10, image_contract=None):
    """Export a declared native horizon using the exact refined rendered optics.

    This function does not grant admission. The explicit history-only contract
    omits future pixels while retaining all executed future states and targets.
    Legacy callers retain their complete-image transport unchanged.
    """
    import shutil

    import numpy as np

    from driversi.data.datasets.navsim_synthetic import (
        _rotation,
        causal_signal_records,
        frame_geometry,
        rear_axle_states,
    )

    scene_path, rollout_path, render_report = [Path(p).resolve(strict=True) for p in (scene_path,rollout_path,render_report)]
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    scene, run = load_physical_branch(scene_path,rollout_path)
    if scene.metadata.get("split") != "train" or scene.metadata.get("pose_reference") != "vehicle_center" or scene.metadata.get("coordinate_frame") != "map_axes_translated_from_initial_ego":
        raise ValueError("Explicit training world and pose contracts required")
    raw_log = Path(scene.metadata["navsim_source"]["log_path"]).resolve(strict=True)
    with raw_log.open("rb") as handle:
        source_frames = pickle.load(handle)
    start = scene.metadata["source_start_frame"]
    if type(anchor_tick) is not int or anchor_tick % 5:
        raise ValueError("Native anchor must coincide with a 2Hz source frame")
    parent_index = start + anchor_tick // 5
    parent_frame = source_frames[parent_index]
    if source_frames[start]["token"] != scene.metadata["scenario_id"] or parent_frame["log_name"] != scene.metadata["source_log"]:
        raise ValueError("Executed scene does not match actual raw parent")
    parent = dict(source_log=scene.metadata["source_log"], token=parent_frame["token"], log_path=str(raw_log), anchor_index=parent_index)
    membership = validate_source_train_membership(parent, source_frames, scene.metadata.get("source_train_membership"), metadata=scene.metadata, anchor_tick=anchor_tick)
    roads = [str(v) for v in parent_frame["roadblock_ids"]]
    command = np.asarray(parent_frame["driving_command"])
    if not roads or command.shape != (4,) or not np.isin(command,[0,1]).all() or command.sum() != 1:
        raise ValueError("Actual current navigation task required")
    report = _read(render_report)
    branch_files = {str(p):digest(p) for p in (scene_path,rollout_path)}
    if any(report.get("inputs",{}).get(k) != v for k,v in branch_files.items()) or report.get("status") != "rendered_diagnostic":
        raise ValueError("Renderer did not consume this exact executed branch")
    _sealed(report["inputs"])
    from driversi.data.datasets.synthetic_storage import history_only_images, image_frame_indices
    history_only = history_only_images(image_contract)
    rendered_indices = image_frame_indices(image_contract)
    rendered_contract = report.get("image_contract")
    if (history_only and rendered_contract != image_contract) or (not history_only and rendered_contract not in (None, "history_future_rgb_v1")):
        raise ValueError("Requested and rendered image contracts differ")
    if type(future_points) is not int or future_points not in (8, 10):
        raise ValueError("Native export requires eight or ten actual future points")
    if history_only and future_points != 10:
        raise ValueError("History-only training preserves ten actual future states")
    ticks = list(range(anchor_tick-15,anchor_tick+future_points*5+1,5))
    if ticks[0] < 2 or ticks[-1] >= len(run["states"]) or report.get("history_ticks") != ticks[:4] or report.get("future_ticks") != ticks[4:]:
        raise ValueError("Complete physical and rendered native window required")
    captures = {frame["tick"]:frame for frame in report["frames"]}
    image_ticks = [ticks[i] for i in rendered_indices] if history_only else ticks
    if len(captures) != len(report["frames"]) or not set(image_ticks) <= set(captures) or (history_only and set(captures) != set(image_ticks)):
        raise ValueError("Missing/repeated render time")
    from driversi.data.datasets.synthetic_storage import sensor_relative_path, write_window_manifest

    token = synthetic_token(branch_files,anchor_tick)
    log_id = token + "_log"
    rear = rear_axle_states(run["states"],scene.actor_ids.index("ego"))
    origin = np.asarray(scene.metadata["origin_xyz"])
    start_us = int(scene.metadata["navsim_source"]["anchor_timestamp_us"])
    timeline = [dict(timestamp_us=start_us+round(row["time"]*1e6),records=[[str(i),bool(red)] for i,red in row["records"]],available=row.get("available",False)) for row in scene.metadata["traffic_lights"]]
    sensor_root = output / "sensors"
    camera_names = ("CAM_F0","CAM_L0","CAM_L1","CAM_L2","CAM_R0","CAM_R1","CAM_R2","CAM_B0")
    frames, files = [], {str(p):digest(p) for p in (scene_path,rollout_path,render_report,raw_log)}
    for index,tick in enumerate(ticks):
        pose = rear[tick]
        timestamp = start_us+tick*100000
        signals,known = causal_signal_records(timeline,timestamp)
        if not known:
            raise ValueError("Unknown source signal context blocks native synthetic training")
        rotation = _rotation(pose[3])
        velocity = (rear[tick,:3]-rear[tick-1,:3])/.1
        previous = (rear[tick-1,:3]-rear[tick-2,:3])/.1
        dynamics = np.concatenate(((velocity@rotation)[:2],(((velocity-previous)/.1)@rotation)[:2]))
        frame_token = token if index == 3 else token + "_" + str(tick)
        annotations = frame_geometry(scene,{"state":run["states"][tick],"presence":run["presence"][tick]},pose)
        annotations["instance_tokens"] = [hashlib.sha256((frame_token+track).encode()).hexdigest()[:32] for track in annotations["track_tokens"]]
        cameras = {name:dict(data_path=None,sensor2lidar_rotation=None,sensor2lidar_translation=None,cam_intrinsic=None,distortion=None) for name in camera_names}
        for name in CAMERAS if (not history_only or index in rendered_indices) else ():
            capture = captures[tick]["cameras"][name]
            validate_rendered_calibration(report, capture)
            transform = np.asarray(capture.get("camera_to_rear_axle"),dtype=float)
            intrinsic = np.asarray(capture["distorted_intrinsic"],dtype=float)
            distortion = np.asarray(capture["distortion"],dtype=float)
            if transform.shape != (4,4) or intrinsic.shape != (3,3) or distortion.shape != (5,) or not all(np.isfinite(x).all() for x in (transform,intrinsic,distortion)):
                raise ValueError("Finite explicit refined camera-to-rear calibration required")
            relative = sensor_relative_path(log_id, name, frame_token, index < 4)
            destination = sensor_root/relative
            source = render_report.parent/capture["file"]
            if not source.resolve(strict=True).is_relative_to(render_report.parent):
                raise ValueError("Render image outside its sealed package")
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,destination)
            if digest(source) != digest(destination):
                raise ValueError("Rendered pixel copy changed")
            files[str(source.resolve())] = digest(source)
            files[str(destination)] = digest(destination)
            cameras[name] = dict(data_path=relative.as_posix(),sensor2lidar_rotation=transform[:3,:3],sensor2lidar_translation=transform[:3,3],cam_intrinsic=intrinsic,distortion=distortion)
        frames.append(dict(token=frame_token,frame_idx=index,timestamp=timestamp,log_name=log_id,log_token=log_id,scene_token=log_id,scene_name=log_id,map_location=scene.metadata["map_name"],roadblock_ids=roads,ego2global_translation=pose[:3]+origin,ego2global_rotation=np.asarray([np.cos(pose[3]/2),0,0,np.sin(pose[3]/2)]),ego_dynamic_state=dynamics,driving_command=command.copy(),anns=annotations,cams=cameras,traffic_lights=signals,lidar_path="",lidar2ego_translation=np.zeros(3),lidar2ego_rotation=[1.,0.,0.,0.]))
    log_path = output/"logs"/(log_id+".pkl")
    log_path.parent.mkdir(parents=True)
    log_path.write_bytes(pickle.dumps(frames,protocol=4))
    files[str(log_path)] = digest(log_path)
    result = dict(token=token,native_log_name=log_id,log_path=str(log_path),sensor_root=str(sensor_root),anchor_index=3,image_files=[str(sensor_root/frames[3]["cams"][name]["data_path"]) for name in CAMERAS],parent=dict(source_log=scene.metadata["source_log"],token=parent_frame["token"],log_path=str(raw_log),anchor_index=parent_index),files=files,trainable=False)
    result.update(write_window_manifest(output, frames, ticks, log_path, files, image_contract=image_contract))
    if membership is not None:
        result["source_train_membership"] = membership
    (output/"export.json").write_text(json.dumps(result,indent=2))
    return result


def seal_admission(exported, caches, branch, visual_admission, output):
    """Create and immediately revalidate the immutable student evidence closure."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    visual = _read(visual_admission)
    visual_ref = dict(path=str(Path(visual_admission).resolve()),sha256=digest(visual_admission))
    files = dict(exported["files"])
    for path in [*branch.values(), *caches.values(), visual_ref["path"]]:
        files[str(Path(path).resolve())] = digest(path)
    for section in (visual.get("files",{}), _read(branch["report"]).get("upstream_source_files",{})):
        for path,sha in section.items():
            if path in files and files[path] != sha:
                raise ValueError("Inconsistent evidence closure")
            files[path] = sha
    native = {key:exported[key] for key in ("log_path","sensor_root","token")}
    native.update(caches)
    proof = dict(schema_version=1,kind="chainflow_synthetic_admission",parent=exported["parent"],branch=branch,visual_admission=visual_ref,native=native,files=files)
    if "source_train_membership" in exported:
        proof["schema_version"] = 2
        proof["source_train_membership"] = exported["source_train_membership"]
    if "image_contract" in exported:
        from driversi.data.datasets.synthetic_storage import HISTORY_ONLY_IMAGE_CONTRACT, ANCHOR_ONLY_IMAGE_CONTRACT
        if exported["image_contract"] not in (HISTORY_ONLY_IMAGE_CONTRACT, ANCHOR_ONLY_IMAGE_CONTRACT):
            raise ValueError("Unknown native history-only contract")
        proof.update(schema_version=4 if exported["image_contract"] == ANCHOR_ONLY_IMAGE_CONTRACT else 3, image_contract=exported["image_contract"],
                     source_train_membership=exported.get("source_train_membership"))
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(proof,indent=2,allow_nan=False))
    row = dict(sample_id=exported["token"],source_log=exported["parent"]["source_log"],kind="synthetic_navtrain",log_path=exported["log_path"],sensor_root=exported["sensor_root"],anchor_index=3,token=exported["token"],metric_cache=caches["metric_cache"],training_metric_cache=caches["training_metric_cache"],image_files=exported["image_files"],admission=dict(path=str(output),sha256=digest(output)))
    files_with_receipt = {**files,str(output):digest(output)}
    validate_admission(row,files_with_receipt)
    return row, files_with_receipt


def validate_cache_pair(evaluation, training, proof):
    """Recompute the native reference baseline; do not trust a scalar receipt."""
    import numpy as np
    from navsim.evaluate.pdm_score import get_trajectory_as_array
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

    if evaluation.route_lane_ids != training.route_lane_ids:
        raise ValueError("Evaluation and native proposal route context differs")
    for attribute in ("centerline", "drivable_area_map"):
        # Both caches are produced from exactly one native context. Pickle bytes
        # are not a cross-version format guarantee; within the pinned runtime
        # unequal objects are not silently accepted as equivalent.
        if pickle.dumps(getattr(evaluation,attribute),protocol=4) != pickle.dumps(getattr(training,attribute),protocol=4):
            raise ValueError("Evaluation and proposal geometry differs: " + attribute)
    sampling = TrajectorySampling(num_poses=40,interval_length=.1)
    reference = get_trajectory_as_array(evaluation.trajectory,sampling,evaluation.ego_state.time_point)
    simulated = PDMSimulator(sampling).simulate_proposals(reference[None],evaluation.ego_state)
    scorer = PDMScorer(sampling)
    scorer.score_proposals(simulated,evaluation.observation,evaluation.centerline,evaluation.route_lane_ids,evaluation.drivable_area_map)
    expected = float(scorer._progress_raw[0]*scorer._multi_metrics[:,0].prod())
    if not np.allclose(training.pdm_progress,expected,rtol=0,atol=1e-7):
        raise ValueError("Counterfactual native reference progress was not reconstructed correctly")
    receipt = _read(proof["native"]["cache_receipt"])
    if not np.isclose(receipt["native_reference_progress"],expected,rtol=0,atol=1e-7):
        raise ValueError("Counterfactual reference receipt differs from native computation")
