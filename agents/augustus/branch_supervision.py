"""Same-branch prediction supervision for the unchanged AUGUstus SFT model.

This is a data annotation protocol, not a new model/loss or a reproduction of
AutoDrive-P3's unpublished annotation heuristic. Geometry drafts need review.
No training cache is published before independent visual and label admission.
"""

import hashlib
import io
import itertools
import json
from pathlib import Path

import numpy as np

CAMERAS = ("CAM_L0", "CAM_F0", "CAM_R0")
POLICY = {
    "version": "executed_geometry_prediction_v1",
    "horizon_seconds": 4.0,
    "near_metres": 10.0,
    "middle_metres": 30.0,
    "front_half_width_metres": 2.0,
    "current_object_radius_metres": 50.0,
    "multi_camera_occurrence": "one_largest_clipped_box_per_actor",
    "frame": "future_ego_rear_axle_x_forward_y_left",
    "bbox_frame": "current_stitched_image_2048x512",
    "occlusion": "independent_review_required",
}


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))


def seal(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pose(frame):
    q = np.asarray(frame["ego2global_rotation"], float)
    p = np.asarray(frame["ego2global_translation"], float)
    if q.shape != (4,) or p.shape != (3,) or not np.isfinite([*q, *p]).all():
        raise ValueError("Explicit finite ego pose required")
    return p, 2 * np.arctan2(q[3], q[0])


def trajectories(frames):
    if len(frames) < 12:
        raise ValueError("Four actual history and eight actual future frames required")
    ts = np.asarray([f["timestamp"] for f in frames[:12]], dtype=np.int64)
    if not np.all(np.abs(np.diff(ts) - 500000) < 50000):
        raise ValueError("Continuous 2 Hz history/future required")
    if len({f["log_name"] for f in frames[:12]}) != 1:
        raise ValueError("Mixed physical branches")
    origin, yaw = pose(frames[3])
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[c, -s], [s, c]])
    result = []
    for frame in frames[:12]:
        p, h = pose(frame)
        result.append([*((p[:2] - origin[:2]) @ rotation), (h - yaw + np.pi) % (2 * np.pi) - np.pi])
    return np.asarray(result[:4]), np.asarray(result[4:])


def future_position(xy):
    xy = np.asarray(xy, float)
    if xy.shape != (2,) or not np.isfinite(xy).all():
        raise ValueError("Finite measured future position required")
    x, y = xy
    if x < 0:
        return "behind"
    distance = np.linalg.norm(xy)
    depth = (
        "near"
        if distance < POLICY["near_metres"]
        else "middle"
        if distance < POLICY["middle_metres"]
        else "far"
    )
    side = (
        "left"
        if y > POLICY["front_half_width_metres"]
        else "right"
        if y < -POLICY["front_half_width_metres"]
        else "front"
    )
    return depth + ", " + side


def stitched_bbox(box, camera, name, size):
    """OpenCV optical projection, native crop then native 2048x512 resize.

    Only fully in-front 3D boxes are drafted; partial near-plane boxes are
    reported as unprojectable, never silently interpreted as absent objects.
    """
    import cv2

    b = np.asarray(box, float)
    if b.shape != (7,) or not np.isfinite(b).all() or np.any(b[3:6] <= 0):
        raise ValueError("Finite oriented physical box required")
    c, s = np.cos(b[6]), np.sin(b[6])
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    corners = np.asarray(list(itertools.product((-1, 1), repeat=3))) * b[3:6] / 2
    corners = corners @ rotation.T + b[:3]
    rot, trans = (
        np.asarray(camera["sensor2lidar_rotation"]),
        np.asarray(camera["sensor2lidar_translation"]),
    )
    points = (corners - trans) @ rot
    if (points[:, 2] <= 0.1).all():
        return None, "behind_camera"
    if (points[:, 2] <= 0.1).any():
        from driversi_agents.augustus.projection import outside_crop_frustum

        if outside_crop_frustum(corners, camera, name, size):
            return None, "outside_crop"
        return None, "near_plane_intersection"
    pixels, _ = cv2.projectPoints(
        points,
        np.zeros(3),
        np.zeros(3),
        np.asarray(camera["cam_intrinsic"], float),
        np.asarray(camera["distortion"], float),
    )
    pixels = pixels.reshape(-1, 2) * np.array([1920 / size[0], 1080 / size[1]])
    # Upstream crops: side views [28:-28,416:-416], front view [:-56].
    left, top, width = (0, 0, 1920) if name == "CAM_F0" else (416, 28, 1088)
    lo, hi = pixels.min(0) - [left, top], pixels.max(0) - [left, top]
    lo, hi = np.maximum(lo, 0), np.minimum(hi, [width, 1024])
    if (hi - lo < 2).any():
        return None, "outside_crop"
    offset = {"CAM_L0": 0, "CAM_F0": 1088, "CAM_R0": 3008}[name]
    scale = np.array([0.5, 0.5])
    lo, hi = (lo + [offset, 0]) * scale, (hi + [offset, 0]) * scale
    return [*np.floor(lo).astype(int).tolist(), *np.ceil(hi).astype(int).tolist()], None


def prediction_draft(frames, image_sizes):
    """Language targets use actual object identity at the same branch's t+4s."""
    trajectories(frames)
    current, future = frames[3], frames[11]
    now, later = current["anns"], future["anns"]
    tokens = list(now["track_tokens"])
    future_tokens = list(later["track_tokens"])
    if len(set(tokens)) != len(tokens) or len(set(future_tokens)) != len(future_tokens):
        raise ValueError("Ambiguous object identities")
    future_index = {token: i for i, token in enumerate(future_tokens)}
    rows, unresolved = [], []
    for i, token in enumerate(tokens):
        box = np.asarray(now["gt_boxes"][i], float)
        if np.linalg.norm(box[:2]) > POLICY["current_object_radius_metres"]:
            continue
        choices = []
        for name in CAMERAS:
            bbox, reason = stitched_bbox(box, current["cams"][name], name, image_sizes[name])
            if bbox:
                choices.append((bbox, name))
            elif reason == "near_plane_intersection":
                unresolved.append(dict(actor_id=token, camera=name, reason=reason))
        if not choices:
            continue
        if token not in future_index:
            unresolved.append(dict(actor_id=token, reason="future_identity_unavailable"))
            continue
        # The same object can straddle two cameras. Annotate one largest view,
        # never duplicate it as two independent traffic participants.
        bbox, name = max(choices, key=lambda x: (x[0][2] - x[0][0]) * (x[0][3] - x[0][1]))
        xy = np.asarray(later["gt_boxes"][future_index[token]][:2], float)
        rows.append(
            dict(
                actor_id=token,
                camera=name,
                bbox_2d=bbox,
                future_position=future_position(xy),
                measured_future_xy=xy.tolist(),
                category=str(now["gt_names"][i]),
            )
        )
    return dict(
        schema_version=1,
        policy=POLICY,
        objects=rows,
        unresolved=unresolved,
        trainable=False,
        review_required=True,
    )


def format_prediction(rows):
    values = [dict(bbox_2d=x["bbox_2d"], future_position=x["future_position"]) for x in rows]
    if not values:
        values = [dict(bbox_2d=[], future_position=[])]
    return "```json\n" + json.dumps(values, ensure_ascii=False) + "\n```"


def native_row(frames, sensor_root, output_video, draft):
    """Preserve native stitch/crop/JPEG/H264 and prompt-field conventions."""
    import cv2
    from moviepy.editor import ImageSequenceClip
    from PIL import Image

    from driversi_agents.augustus.data import validate_row

    history, future = trajectories(frames)
    images, evidence = [], []
    root = Path(sensor_root).resolve(strict=True)
    for frame in frames[:4]:
        views = []
        for name in CAMERAS:
            path = (root / frame["cams"][name]["data_path"]).resolve(strict=True)
            if not path.is_relative_to(root):
                raise ValueError("Image escaped branch sensor root")
            image = cv2.imread(str(path))
            if image is None or image.shape[:2] not in ((540, 960), (1080, 1920)):
                raise ValueError("Unsupported rendered camera geometry")
            # Existing MTGS output is half resolution; this is declared data
            # upsampling, not a change to native model preprocessing.
            if image.shape[:2] == (540, 960):
                image = cv2.resize(image, (1920, 1080))
            views.append(image[:-56] if name == "CAM_F0" else image[28:-28, 416:-416])
            evidence.append(
                dict(path=str(path), sha256=seal(path), camera=name, token=frame["token"])
            )
        stitched = cv2.resize(np.concatenate(views, axis=1), (2048, 512))
        ok, jpeg = cv2.imencode(".jpg", stitched)
        if not ok:
            raise ValueError("Native intermediate JPEG failed")
        images.append(np.asarray(Image.open(io.BytesIO(jpeg.tobytes())).convert("RGB")))
    output_video = Path(output_video)
    output_video.parent.mkdir(parents=True, exist_ok=True)
    clip = ImageSequenceClip(images, fps=2.0)
    try:
        clip.write_videofile(
            str(output_video), codec="libx264", audio=False, logger=None, threads=2
        )
    finally:
        clip.close()
    anchor = frames[3]
    command = {
        (1, 0, 0, 0): "TURN LEFT",
        (0, 1, 0, 0): "GO FORWARD",
        (0, 0, 1, 0): "TURN RIGHT",
    }.get(tuple(anchor["driving_command"]))
    if command is None:
        raise ValueError("Unknown navigation cannot be inferred from future labels")
    v = np.asarray(anchor["ego_dynamic_state"], float)
    if v.shape != (4,) or not np.isfinite(v).all():
        raise ValueError("Malformed current telemetry")
    row = dict(
        id=anchor["token"],
        video=output_video.read_bytes(),
        scene_token=anchor["log_name"],
        frame_id=anchor["token"],
        driving_command=command,
        ego_dynamic_state=f"vx: {v[0]:.2f} m/s, vy: {v[1]:.2f} m/s, ax: {v[2]:.2f} m/s², ay: {v[3]:.2f} m/s²",
        historical_x_y_angle=json.dumps(history.tolist()),
        future_x_y_angle=json.dumps(future.tolist()),
        historical_vx_vy=json.dumps(
            [np.asarray(f["ego_dynamic_state"][:2], float).tolist() for f in frames[:4]]
        ),
        solution_prediction=format_prediction(draft["objects"]),
        solution_planning="",
        solution_perception="",
        reasoning_trace="",
    )
    validate_row(row)
    return row, evidence


def geometry_label_contract(preview, output):
    """Declare measured labels without inventing a visual reviewer or unknown futures."""
    draft_path = Path(preview) / "language-draft.json"
    draft = json.loads(draft_path.read_text())
    value = dict(
        kind="executed_geometry_label_contract",
        version=1,
        draft_sha256=seal(draft_path),
        branch_report_sha256=draft["branch_report_sha256"],
        actor_ids=[x["actor_id"] for x in draft["objects"]],
        omitted_unresolved=draft["unresolved"],
        visibility_reviewed=False,
        occlusion_policy_override="experimental_geometry_only_no_independent_visibility_claim",
        annotation_protocol="executed_geometry_unreviewed_v1",
        native_loss_unchanged=True,
    )
    path = Path(output) / "label-review.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    from driversi.rsi.experiments.continuation import frozen_json

    frozen_json(path, value)
    return value


def require_label_contract(draft_path, review_path, visual):
    if visual.get("visual_quality_profile") != "experimental_numeric25_v1":
        return require_label_review(draft_path, review_path)
    draft, contract = [json.loads(Path(p).read_text()) for p in (draft_path, review_path)]
    if (
        contract.get("kind") != "executed_geometry_label_contract"
        or contract.get("annotation_protocol") != "executed_geometry_unreviewed_v1"
        or contract.get("draft_sha256") != seal(draft_path)
        or contract.get("branch_report_sha256") != draft["branch_report_sha256"]
        or contract.get("actor_ids") != [x["actor_id"] for x in draft["objects"]]
        or contract.get("omitted_unresolved") != draft["unresolved"]
        or contract.get("visibility_reviewed") is not False
    ):
        raise ValueError("Sealed same-branch geometry label contract required")
    return draft


def require_label_review(draft_path, review_path):
    draft = json.loads(Path(draft_path).read_text())
    review = json.loads(Path(review_path).read_text())
    if (
        review.get("draft_sha256") != seal(draft_path)
        or review.get("approved") is not True
        or not review.get("reviewer")
        or draft["unresolved"]
        or review.get("checked_actor_ids") != sorted(x["actor_id"] for x in draft["objects"])
        or any(
            review.get(k) is not True
            for k in ("boxes_aligned", "visibility_checked", "future_identity_checked")
        )
    ):
        raise ValueError("Same-branch language supervision has not passed independent review")
    return draft


def verify_export_binding(exported, required_files, expected_parent):
    """An existing export must include every selected input, not their overlap."""
    from driversi.rsi.plugins.synthesis import check_files

    if any(exported.get("files", {}).get(k) != v for k, v in required_files.items()):
        raise ValueError("Existing export lacks exact selected branch inputs")
    if exported.get("parent") != expected_parent or exported.get("anchor_index") != 3:
        raise ValueError("Existing export parent or anchor differs from selected branch")
    check_files(exported["files"])


def selected_export(branch_path, render_path, output):
    """Export actual geometry, without granting visual or training admission."""
    import pickle

    from driversi.contracts.scene import Scene
    from driversi_agents.chainflow.synthetic import export_native_frames

    branch = json.loads(branch_path.read_text())
    execution = Path(branch["execution"]).resolve(strict=True)
    scene_path, rollout_path = execution / "scene.npz", execution / "teacher.npz"
    scene = Scene.load(scene_path)
    raw_path = Path(scene.metadata["navsim_source"]["log_path"]).resolve(strict=True)
    with raw_path.open("rb") as stream:
        raw = pickle.load(stream)
    anchor = scene.metadata["source_start_frame"] + 4
    expected_parent = dict(
        source_log=scene.metadata["source_log"],
        token=raw[anchor]["token"],
        log_path=str(raw_path),
        anchor_index=anchor,
    )
    if branch["source_log"] != expected_parent["source_log"]:
        raise ValueError("Selected branch source differs from physical world")
    required = {
        str(p.resolve()): seal(p) for p in (scene_path, rollout_path, render_path, raw_path)
    }
    export_path = output / "native/export.json"
    if export_path.exists():
        exported = json.loads(export_path.read_text())
    else:
        from driversi.data.datasets import synthetic_storage
        from driversi.data.datasets.synthetic_publish import publish_package
        from driversi_agents.chainflow import synthetic

        package_identity = dict(
            source_files=required,
            future_points=branch["future_frames"],
            exporter_sha256=seal(synthetic.__file__),
            storage_sha256=seal(synthetic_storage.__file__),
            layout="logs_sensors_history_future_v2",
        )
        exported = publish_package(
            package_identity,
            lambda staging: export_native_frames(
                scene_path,
                rollout_path,
                render_path,
                staging,
                future_points=branch["future_frames"],
            ),
            output / "native",
        )
    verify_export_binding(exported, required, expected_parent)
    return exported


def diagnostic_preview(branch_report, render_report, output):
    """Build inspectable joint labels even for rejected pixels; never an HF cache.

    Geometry previews help diagnose projection/annotation errors independently
    of appearance repair. This entry point cannot publish trainable records.
    """
    import pickle

    from PIL import Image, ImageDraw

    from driversi.agents.backends.native import project_root

    branch_path, render_path = [
        Path(p).resolve(strict=True) for p in (branch_report, render_report)
    ]
    output = Path(output).resolve()
    if not output.is_relative_to(project_root() / "exp"):
        raise ValueError("Synthetic artifacts belong under exp")
    output.mkdir(parents=True, exist_ok=True)
    exported = selected_export(branch_path, render_path, output)
    with Path(exported["log_path"]).open("rb") as stream:
        frames = pickle.load(stream)
    sizes = {
        name: Image.open(Path(exported["sensor_root"]) / frames[3]["cams"][name]["data_path"]).size
        for name in CAMERAS
    }
    draft = prediction_draft(frames, sizes)
    draft.update(
        branch_report_sha256=seal(branch_path), native_log_sha256=seal(exported["log_path"])
    )
    draft_path = output / "language-draft.json"
    if draft_path.exists() and json.loads(draft_path.read_text()) != draft:
        raise ValueError("Same-branch draft changed")
    draft_path.write_text(json.dumps(draft, indent=2, allow_nan=False))
    row, images = native_row(frames, exported["sensor_root"], output / "history.mp4", draft)
    import av

    with av.open(io.BytesIO(row["video"])) as container:
        decoded = [f.to_image() for f in container.decode(video=0)]
    overlay = decoded[-1].copy()
    draw = ImageDraw.Draw(overlay)
    for obj in draft["objects"]:
        draw.rectangle(obj["bbox_2d"], outline="red", width=3)
        draw.text(
            tuple(obj["bbox_2d"][:2]), obj["actor_id"] + " " + obj["future_position"], fill="yellow"
        )
    overlay.save(output / "anchor-label-overlay.png")
    (output / "row-preview.json").write_text(
        json.dumps({k: v for k, v in row.items() if k != "video"}, indent=2, ensure_ascii=False)
    )
    report = dict(
        kind="augustus_joint_supervision_preview",
        draft=str(draft_path),
        trainable=False,
        branch_report_sha256=seal(branch_path),
        render_report_sha256=seal(render_path),
        actual_history_frames=4,
        actual_future_points=8,
        native_decode_passed=True,
        objects=len(draft["objects"]),
        unresolved=draft["unresolved"],
        images=images,
        cache_published=False,
        admission="not_granted",
    )
    (output / "preview-report.json").write_text(json.dumps(report, indent=2))
    return report


def publish(branch_report, render_report, visual_admission, label_review, output):
    """Render-derived annotation preview, then admission-gated native HF cache.

    Run once without label_review to create a draft and overlays; a subsequent
    invocation with review publishes a separate cache directory. No dummy labels.
    """
    import pickle

    from PIL import Image

    from driversi.agents.backends.native import project_root
    from driversi.rsi.correction.trajectory_admission import validate_trajectory_evidence
    from driversi.rsi.plugins.synthesis import check_files
    from driversi_agents.chainflow.synthetic import validate_visual_admission

    branch_path = Path(branch_report).resolve(strict=True)
    branch = json.loads(branch_path.read_text())
    render_path = Path(render_report).resolve(strict=True)
    validate_trajectory_evidence(branch["trajectory_evidence"])
    visual = validate_visual_admission(visual_admission)
    if (
        branch.get("branch") != "teacher"
        or branch.get("teacher_qualified") is not True
        or not visual["admitted"]
    ):
        raise ValueError("Qualified teacher and visual branch required")
    if (
        visual.get("branch_report_sha256") != seal(branch_path)
        or Path(visual["render_report"]).resolve() != render_path
    ):
        raise ValueError("Visual admission belongs to a different branch or renderer")
    output = Path(output).resolve()
    if not output.is_relative_to(project_root() / "exp"):
        raise ValueError("Synthetic training artifacts belong under exp")
    files = {
        **branch["files"],
        **branch["upstream_source_files"],
        **visual["files"],
        str(branch_path): seal(branch_path),
        str(render_path): seal(render_path),
        str(Path(visual_admission).resolve()): seal(visual_admission),
        str(Path(__file__).resolve()): seal(__file__),
        str(Path(__file__).with_name("projection.py").resolve()): seal(
            Path(__file__).with_name("projection.py")
        ),
    }
    check_files(files)
    output.mkdir(parents=True, exist_ok=True)
    exported = selected_export(branch_path, render_path, output)
    with Path(exported["log_path"]).open("rb") as stream:
        frames = pickle.load(stream)
    sizes = {
        name: Image.open(Path(exported["sensor_root"]) / frames[3]["cams"][name]["data_path"]).size
        for name in CAMERAS
    }
    draft = prediction_draft(frames, sizes)
    draft.update(
        branch_report_sha256=seal(branch_path), native_log_sha256=seal(exported["log_path"])
    )
    draft_path = output / "language-draft.json"
    if draft_path.exists() and json.loads(draft_path.read_text()) != draft:
        raise ValueError("Same-branch draft changed")
    draft_path.write_text(json.dumps(draft, indent=2, allow_nan=False))
    row, images = native_row(frames, exported["sensor_root"], output / "history.mp4", draft)
    (output / "row-preview.json").write_text(
        json.dumps({k: v for k, v in row.items() if k != "video"}, indent=2, ensure_ascii=False)
    )
    if label_review is None:
        return dict(
            kind="augustus_joint_supervision_preview", draft=str(draft_path), trainable=False
        )
    require_label_contract(draft_path, label_review, visual)
    from datasets import Dataset

    cache = output / "cache/batch_0"
    if cache.exists():
        raise ValueError("Published cache already exists; do not replay training data side effects")
    Dataset.from_list([row]).save_to_disk(str(cache))
    files.update(exported["files"])
    for path in [
        draft_path,
        Path(label_review).resolve(),
        output / "history.mp4",
        output / "row-preview.json",
        *cache.iterdir(),
    ]:
        if path.is_file():
            files[str(path.resolve())] = seal(path)
    check_files(files)
    report = dict(
        kind="augustus_admitted_synthetic_sample",
        sample_id=row["frame_id"],
        cache=str(cache.parent),
        source_log=branch["source_log"],
        branch_report=str(branch_path),
        visual_admission=str(Path(visual_admission).resolve()),
        language_draft=str(draft_path),
        label_review=str(Path(label_review).resolve()),
        files=files,
        images=images,
        history_frames=4,
        future_points=8,
        future_seconds=4.0,
        native_loss_unchanged=True,
        on_policy=False,
        trainable=True,
        annotation_protocol=POLICY,
        label_evidence_mode="executed_geometry_unreviewed_v1"
        if visual.get("visual_quality_profile") == "experimental_numeric25_v1"
        else "independent_review",
        image_profile="existing_mtgs_960x540_upsampled_before_native_crop",
        record=dict(
            frame_id=row["frame_id"],
            log_name=branch["source_log"],
            timestamp_us=int(frames[3]["timestamp"]),
            ego_pose=[*pose(frames[3])[0][:2], pose(frames[3])[1]],
        ),
    )
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


def validate_sample(path):
    """Revalidate pixels, trajectory, language and cache closure at consumption."""
    from driversi.rsi.correction.trajectory_admission import validate_trajectory_evidence
    from driversi.rsi.plugins.synthesis import check_files
    from driversi_agents.chainflow.synthetic import validate_visual_admission

    value = json.loads(Path(path).read_text())
    if (
        value.get("kind") != "augustus_admitted_synthetic_sample"
        or value.get("trainable") is not True
        or value.get("annotation_protocol") != POLICY
        or value.get("on_policy") is not False
    ):
        raise ValueError("Reviewed, admitted native synthetic sample required")
    check_files(value["files"])
    for name in ("branch_report", "visual_admission", "language_draft", "label_review"):
        p = str(Path(value[name]).resolve(strict=True))
        if value["files"].get(p) != seal(p):
            raise ValueError("Synthetic sample has an unsealed admission reference")
    branch = json.loads(Path(value["branch_report"]).read_text())
    validate_trajectory_evidence(branch["trajectory_evidence"])
    visual = validate_visual_admission(value["visual_admission"])
    if (
        not visual["admitted"]
        or visual["branch_report_sha256"] != seal(value["branch_report"])
        or branch["source_log"] != value["source_log"]
        or branch["branch"] != "teacher"
    ):
        raise ValueError("Sample visual evidence or source belongs to another branch")
    expected_mode = (
        "executed_geometry_unreviewed_v1"
        if visual.get("visual_quality_profile") == "experimental_numeric25_v1"
        else "independent_review"
    )
    if value.get("label_evidence_mode", "independent_review") != expected_mode:
        raise ValueError("Declared label evidence mode differs from actual admission")
    draft = require_label_contract(value["language_draft"], value["label_review"], visual)
    if draft["branch_report_sha256"] != seal(value["branch_report"]):
        raise ValueError("Language supervision belongs to another branch")
    cache_files = {str(p.resolve()) for p in Path(value["cache"]).glob("batch_*/*") if p.is_file()}
    if not cache_files or not cache_files <= value["files"].keys():
        raise ValueError("Native synthetic cache is not sealed")
    if (
        value["record"]["frame_id"] != value["sample_id"]
        or value["record"]["log_name"] != value["source_log"]
    ):
        raise ValueError("Synthetic record source is inconsistent")
    return value


def normalize_joint_manifest(value, path, *, allow_smoke=False):
    """Translate an admitted mixture into the existing native loader contract."""
    import copy

    from driversi_agents.augustus.data import manifest

    if (
        set(value)
        != {
            "schema_version",
            "kind",
            "base_manifest",
            "base_cache_files",
            "synthetic_samples",
            "train_cache",
            "train_cache_files",
            "dataset_version",
        }
        or value["schema_version"] != 2
        or value["kind"] != "augustus_joint_cache"
        or not value["synthetic_samples"]
    ):
        raise ValueError("Malformed admitted mixture")
    ref = value["base_manifest"]
    base_path = (path.parent / ref["path"]).resolve(strict=True)
    if set(ref) != {"path", "sha256"} or seal(base_path) != ref["sha256"]:
        raise ValueError("Base dataset identity changed")
    if json.loads(base_path.read_text())["kind"] == "augustus_joint_cache":
        raise ValueError("Use a fixed original base, not recursively nested mixtures")
    result = copy.deepcopy(manifest(base_path, allow_smoke=allow_smoke))
    from driversi.rsi.plugins.synthesis import check_files
    from driversi_agents.augustus.data import cache_files

    if set(cache_files(result)) != set(value["base_cache_files"]):
        raise ValueError("Original cache closure changed")
    check_files(value["base_cache_files"])
    source_blocks = [copy.deepcopy(result["splits"]["train"])]
    for sample in value["synthetic_samples"]:
        sample_path = (path.parent / sample["path"]).resolve(strict=True)
        if set(sample) != {"path", "sha256"} or seal(sample_path) != sample["sha256"]:
            raise ValueError("Synthetic admission identity changed")
        admitted = validate_sample(sample_path)
        result["splits"]["train"]["records"].append(admitted["record"])
        source_blocks.append(dict(cache=admitted["cache"], records=[admitted["record"]]))
    result["splits"]["train"]["cache"] = str(
        (path.parent / value["train_cache"]).resolve(strict=True)
    )
    from driversi.rsi.plugins.synthesis import check_files

    actual_cache_files = {
        str(p.resolve())
        for p in Path(result["splits"]["train"]["cache"]).glob("batch_*/*")
        if p.is_file()
    }
    if not actual_cache_files or actual_cache_files != set(value["train_cache_files"]):
        raise ValueError("Mixed cache file closure changed")
    check_files(value["train_cache_files"])
    result["splits"]["train"]["joint_source_blocks"] = source_blocks
    result["dataset_version"] = value["dataset_version"]
    return result


def build_training_pair(base_manifest, samples, output, *, allow_smoke=False):
    """Same original pool; only the intervention adds admitted joint supervision.

    No optimizer or scoring is invoked. The unified lifecycle owns both budgets.
    Invalid previews are rejected before any dataset is published.
    """
    from datasets import concatenate_datasets

    from driversi.agents.backends.native import project_root
    from driversi.training.sampling.environment import source_log_group
    from driversi_agents.augustus.data import cache_files, manifest, read_cache, validate_row

    output = Path(output).resolve()
    if output.exists() or not output.is_relative_to(project_root() / "exp"):
        raise ValueError("New paired dataset output required under exp")
    original = manifest(base_manifest, allow_smoke=allow_smoke)
    admitted = [validate_sample(p) for p in samples]
    if not admitted:
        raise ValueError("At least one admitted synthetic sample required")
    forbidden = {source_log_group(x) for x in original["protected_logs"]}
    forbidden |= {
        source_log_group(x["log_name"])
        for s in ("validation", "development")
        for x in original["splits"][s]["records"]
    }
    if any(source_log_group(x["source_log"]) in forbidden for x in admitted):
        raise ValueError("Synthetic source overlaps protected/evaluation logs")
    datasets = [read_cache(original["splits"]["train"])]
    for sample in admitted:
        data = read_cache(dict(cache=sample["cache"], records=[sample["record"]]))
        row = data[0]
        validate_row(row)
        if row["solution_prediction"] != format_prediction(
            json.loads(Path(sample["language_draft"]).read_text())["objects"]
        ):
            raise ValueError("Cache language differs from reviewed geometry")
        datasets.append(data)
    merged = concatenate_datasets(datasets)
    if len(set(merged["frame_id"])) != len(merged):
        raise ValueError("Repeated original/synthetic sample identities")
    output.mkdir(parents=True)
    merged.save_to_disk(str(output / "intervention/train/batch_0"))
    (output / "control.json").write_text(json.dumps(original, indent=2))
    joint = dict(
        schema_version=2,
        kind="augustus_joint_cache",
        base_manifest=dict(path=str(Path(base_manifest).resolve()), sha256=seal(base_manifest)),
        base_cache_files={str(Path(p).resolve()): seal(p) for p in cache_files(original)},
        synthetic_samples=[dict(path=str(Path(p).resolve()), sha256=seal(p)) for p in samples],
        train_cache=str(output / "intervention/train"),
        train_cache_files={
            str(p.resolve()): seal(p)
            for p in (output / "intervention/train").glob("batch_*/*")
            if p.is_file()
        },
        dataset_version=original["dataset_version"] + ":joint:" + seal(samples[0])[:12],
    )
    (output / "intervention.json").write_text(json.dumps(joint, indent=2))
    manifest(output / "control.json", allow_smoke=allow_smoke)
    manifest(output / "intervention.json", allow_smoke=allow_smoke)
    return dict(
        control=str(output / "control.json"),
        intervention=str(output / "intervention.json"),
        original_rows=len(datasets[0]),
        synthetic_rows=len(admitted),
        native_loss_unchanged=True,
        training_started=False,
        student_gain=None,
    )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    for key in ("branch-report", "render-report", "visual-admission", "output"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--label-review")
    a = p.parse_args()
    print(
        json.dumps(
            publish(a.branch_report, a.render_report, a.visual_admission, a.label_review, a.output)
        )
    )
