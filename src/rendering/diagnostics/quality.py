"""Conservative MTGS metadata quality checks, not a photometric certificate.

Lens sampling support, geometry proxies and participant inventories answer
separate questions. Unknown reconstruction quality must never become approval.
"""

import math
from collections import Counter

import numpy as np

QUALITY_VERSION = "mtgs-frame-metadata-quality-v2"


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _pose_status(value):
    if value is None:
        return "unknown"
    try:
        pose = np.asarray(value)
        if (pose.dtype.kind not in "iuf" or pose.shape not in ((3, 4), (4, 4)) or not np.isfinite(pose).all()
                or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(pose[:3, :3]), 1., atol=1e-5)
                or pose.shape == (4, 4) and not np.allclose(pose[3], [0, 0, 0, 1])):
            return "failed"
    except (TypeError, ValueError):
        return "failed"
    return "passed"



def _calibration_checks(camera):
    """Match the source-lens runtime; do not infer missing calibration fields."""
    model = camera.get("camera_model")
    checks = {"camera_model": ("unknown" if model is None else "passed"
                               if model == "opencv_distorted_source_lens" else "failed")}
    for field, name in (("K", "camera_intrinsics"), ("distortion", "camera_distortion"),
                        ("camera_to_body", "camera_body_transform")):
        value = camera.get(field)
        if value is None:
            checks[name] = "unknown"
            continue
        try:
            array = np.asarray(value)
            valid = array.dtype.kind in "iuf" and np.isfinite(array).all()
            if field == "K":
                valid = (valid and array.shape == (3, 3) and array[0, 0] > 0 and array[1, 1] > 0
                         and np.allclose(array[2], [0., 0., 1.]))
            elif field == "distortion":
                valid = valid and array.ndim == 1 and len(array) in (4, 5, 8, 12, 14)
            else:
                valid = valid and array.shape == (4, 4) and _pose_status(array) == "passed"
        except (TypeError, ValueError):
            valid = False
        checks[name] = "passed" if valid else "failed"
    dimensions = [camera.get(k) for k in ("width", "height")]
    checks["camera_dimensions"] = ("unknown" if any(v is None for v in dimensions) else "passed"
                                   if all(type(v) is int and v > 0 for v in dimensions) else "failed")
    return checks


def assess_frame_quality(frame):
    """Recompute evidence from raw metadata; producer quality booleans are ignored.

    Diagnostic eligibility means metadata is coherent and some lens pixels exist.
    It does not certify appearance, physical registration or a novel viewpoint.
    No current MTGS binding contains the independent evidence needed to approve
    high-quality supervised data, so this version deliberately cannot do that.
    """
    if not isinstance(frame, dict):
        raise ValueError("Render quality needs a frame object")
    cameras = frame.get("cameras", [])
    if not isinstance(cameras, list) or any(not isinstance(c, dict) for c in cameras):
        raise ValueError("Render quality cameras must be objects")
    tick, generation, stamp = (frame.get(k) for k in ("tick", "generation", "simulation_timestamp_us"))
    clock_ok = (type(tick) is int and tick >= 0 and type(generation) is int and generation >= 0
                and type(stamp) is int and type(frame.get("render_tick")) is int
                and frame["render_tick"] == tick)
    ids = [c.get("camera_id") for c in cameras]
    ids_ok = bool(ids) and all(isinstance(c, str) and c for c in ids) and len(set(ids)) == len(ids)
    coverage = frame.get("coverage", {})
    if not isinstance(coverage, dict):
        coverage = {}
    inventories = coverage.get("source_inventories", [])
    if not isinstance(inventories, list):
        inventories = []
    result = []
    for camera in cameras:
        reasons = []
        cid = camera.get("camera_id")
        owner = camera.get("owner_actor_id")
        offset, source = camera.get("source_seed_error_us"), camera.get("source_timestamp_us")
        timing = (clock_ok and ids_ok and isinstance(owner, str) and bool(owner)
                  and type(source) is int and type(offset) is int
                  and source-stamp == offset and abs(offset) <= 60000)
        fraction = camera.get("source_lens_valid_fraction")
        lens = ("unknown" if fraction is None else "passed" if _number(fraction)
                and 0 < fraction <= 1 else "failed")
        pose = _pose_status(camera.get("camera_to_world"))
        calibration = _calibration_checks(camera)
        height = camera.get("height_binding", {})
        if not isinstance(height, dict):
            height = {}
        height_status = ("passed" if height.get("supported") is True
                         else "failed" if height.get("supported") is False else "unknown")
        support = camera.get("support", {})
        if not isinstance(support, dict):
            support = {}
        distance, angle = (support.get(k) for k in
                           ("distance_from_seed_camera_m", "rotation_from_seed_camera_degrees"))
        pose_distance_recorded = (_number(distance) and distance >= 0
                                  and _number(angle) and 0 <= angle <= 180)
        matches = [i for i in inventories if isinstance(i, dict) and i.get("camera_id") == cid]
        inventory = matches[0] if len(matches) == 1 else {}
        inventory_keys = ("source_without_reconstruction_node", "source_without_physics_actor",
                          "reconstruction_only_current_nodes")
        inventory_known = bool(inventory) and all(isinstance(inventory.get(k), list) for k in inventory_keys)
        participant_status = "unknown"
        if inventory_known:
            participant_status = "failed" if any(inventory[k] for k in inventory_keys) else "passed"
        if camera.get("virtual") is True and coverage.get("ego_appearance") == "camera_car_not_reconstructed":
            participant_status = "failed"
            reasons.append("ego_appearance_missing_for_virtual_view")
        checks = {"frame_identity_and_seed_time": "passed" if timing else "failed",
                  "lens_sampling_support": lens, "finite_rigid_camera_pose": pose,
                  "height_proxy": height_status, "participant_inventory": participant_status,
                  "seed_camera_displacement_recorded": "passed" if pose_distance_recorded else "unknown",
                  "reconstruction_quality": "unknown", "render_physics_registration": "unknown",
                  "novel_view_domain": "unknown", **calibration}
        for name, status in checks.items():
            if status != "passed":
                reasons.append(name+"_"+status)
        if camera.get("scene_height_supported") is not True:
            reasons.append("scene_height_proxy_not_supported")
        diagnostic = (timing and lens == "passed" and pose == "passed"
                      and all(status == "passed" for status in calibration.values()))
        result.append({"camera_id": cid, "owner_actor_id": owner,
                       "checks": checks, "reasons": reasons,
                       "diagnostic_usable": diagnostic, "high_quality_supervision_eligible": False,
                       "lens_valid_fraction": fraction if _number(fraction) else None,
                       "seed_camera_distance_m": distance if pose_distance_recorded else None,
                       "seed_camera_rotation_degrees": angle if pose_distance_recorded else None})
    return {"version": QUALITY_VERSION, "scope": "metadata_only_not_pixel_quality_or_visibility",
            "generation": generation, "tick": tick,
            "diagnostic_usable": bool(result) and all(r["diagnostic_usable"] for r in result),
            "high_quality_supervision_eligible": False,
            "not_measured": ["photometric_reconstruction", "temporal_flicker", "novel_view_accuracy",
                             "physical_surface_registration", "wall_clock_realtime"],
            "cameras": result}


def validate_frame_quality(frame):
    """Legacy absent reports stay unknown; contradictory claimed reports fail."""
    assessed = assess_frame_quality(frame)
    recorded = frame.get("render_quality")
    if recorded is not None and recorded != assessed:
        raise ValueError("Recorded render quality contradicts frame metadata")
    for camera, expected in zip(frame.get("cameras", []), assessed["cameras"], strict=True):
        if "quality" in camera and camera["quality"] != expected:
            raise ValueError("Recorded camera quality contradicts frame metadata")
    return assessed


def summarize_quality(reports):
    reports = list(reports)
    cameras = [camera for report in reports for camera in report["cameras"]]
    return {"version": QUALITY_VERSION, "frames": len(reports), "camera_frames": len(cameras),
            "diagnostic_usable_camera_frames": sum(c["diagnostic_usable"] for c in cameras),
            "high_quality_supervision_camera_frames": sum(c["high_quality_supervision_eligible"] for c in cameras),
            "reason_counts": dict(sorted(Counter(r for c in cameras for r in c["reasons"]).items())),
            "scope": "metadata_only_not_pixel_quality_or_visibility"}


def audit_manifest(path):
    """Inspect a saved session without loading models or asserting pixel quality."""
    import hashlib
    import json
    from pathlib import Path

    path = Path(path).resolve()
    raw = path.read_bytes()
    doc = json.loads(raw)
    if not isinstance(doc, dict) or doc.get("kind") != "mtgs_world_session":
        raise ValueError("Render audit requires an MTGS world session manifest")
    reports = [validate_frame_quality(frame) for frame in doc.get("frames", [])]
    captures = [f["capture_seconds"] for f in doc.get("frames", [])
                if _number(f.get("capture_seconds")) and f["capture_seconds"] >= 0]
    return {"manifest": str(path), "manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "status": doc.get("status"), "quality": summarize_quality(reports),
            "latency": {"scope": "recorded_capture_only_excludes_policy_physics_and_publish",
                        "measured_captures": len(captures),
                        "p50_seconds": float(np.percentile(captures, 50)) if captures else None,
                        "p95_seconds": float(np.percentile(captures, 95)) if captures else None},
            "qualification": "not_measured_no_new_render_or_pixel_inspection"}


def audit_session_inputs(config_path):
    """A checkpoint being present does not mean the paired physics scene exists."""
    import hashlib
    import json
    from pathlib import Path

    config_path = Path(config_path).resolve()
    raw = config_path.read_bytes()
    config = json.loads(raw)
    required = ("config_path", "checkpoint_path", "road_block_config", "scene", "road_graph",
                "binding", "world_python", "native_path")
    if config.get("population_config") is not None:
        required += ("population_config",)
    files = []
    for key in required:
        value = config.get(key)
        path = Path(value) if isinstance(value, str) and value else None
        present = path is not None and path.is_absolute() and path.is_file()
        files.append({"field": key, "path": str(path) if path else None,
                      "present": present, "bytes": path.stat().st_size if present else None})
    return {"config": str(config_path), "config_sha256": hashlib.sha256(raw).hexdigest(),
            "input_files_present": all(f["present"] for f in files), "files": files,
            "scope": "file_presence_only_not_asset_compatibility_or_render_quality"}


def main():
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifests = [audit_manifest(path) for path in args.manifest]
    report = {"schema_version": 1, "kind": "render_readiness_and_metadata_audit",
              "inputs": audit_session_inputs(args.session_config), "manifests": manifests,
              "manifest_count": len(manifests), "new_renders": 0,
              "pixel_quality_measured": False, "wall_clock_realtime_verified": False,
              "industrial_qualification": "not_established"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "inputs_ready": report["inputs"]["input_files_present"],
                      "manifest_count": len(manifests)}))


if __name__ == "__main__":
    main()
