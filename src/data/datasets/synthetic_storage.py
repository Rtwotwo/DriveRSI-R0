"""Partitioned immutable sensor payloads with a native NAVSIM log index.

The physical log remains one chronological sequence for native loaders.
Partitions are storage roles, never an admission or a change in supervision.
"""

import json
from pathlib import Path

HISTORY_ONLY_IMAGE_CONTRACT = "history_rgb_future_states_v1"
HISTORY_ONLY_LAYOUT = "logs_sensors_history_rgb_future_states_v3"
ANCHOR_ONLY_IMAGE_CONTRACT = "anchor_rgb_history_future_states_v1"
ANCHOR_ONLY_LAYOUT = "logs_sensors_anchor_rgb_history_future_states_v4"


def image_frame_indices(contract):
    """Rendered RGB indices; omitted/legacy contract preserves every frame."""
    if contract is None:
        return None
    if contract == HISTORY_ONLY_IMAGE_CONTRACT:
        return [0, 1, 2, 3]
    if contract == ANCHOR_ONLY_IMAGE_CONTRACT:
        return [3]
    raise ValueError("Unknown synthetic image contract")


def history_only_images(contract):
    return image_frame_indices(contract) is not None


def image_storage_layout(contract):
    indices = image_frame_indices(contract)
    if indices == [3]:
        return ANCHOR_ONLY_LAYOUT
    return HISTORY_ONLY_LAYOUT if indices is not None else "logs_sensors_history_future_v2"


def sensor_relative_path(log_id, camera, frame_token, history):
    for value in (log_id, camera, frame_token):
        if (
            not isinstance(value, str)
            or not value
            or value in {".", ".."}
            or any(c in value for c in ("/", "\\", "\x00"))
        ):
            raise ValueError("Single safe sensor path component required")
    if type(history) is not bool:
        raise ValueError("Explicit history/future role required")
    return Path("history" if history else "future") / log_id / camera / (frame_token + ".png")


def write_window_manifest(output, frames, ticks, log_path, files, *, image_contract=None):
    """Describe and seal actual paths; all pixels remain separate from metadata."""
    from driversi.agents.backends.native import digest

    history_only = history_only_images(image_contract)
    rendered_indices = image_frame_indices(image_contract)
    output, log_path = Path(output).resolve(), Path(log_path).resolve(strict=True)
    sensor_root = output / "sensors"
    if len(frames) not in (12, 14) or len(frames) != len(ticks):
        raise ValueError("Four history plus eight/ten actual future frames required")
    if not log_path.is_relative_to(output / "logs") or files.get(str(log_path)) != digest(log_path):
        raise ValueError("Sealed native log beneath logs/ required")
    timestamp = [int(f["timestamp"]) for f in frames]
    if any(b <= a for a, b in zip(timestamp, timestamp[1:])):
        raise ValueError("Storage cannot conceal non-monotonic timestamps")
    anchor = timestamp[3]
    sections = {"history": [], "future": []}
    seen = set()
    camera_set = None
    for index, (frame, tick) in enumerate(zip(frames, ticks)):
        role = "history" if index < 4 else "future"
        cameras = {}
        for camera, info in frame["cams"].items():
            if info.get("data_path") is None:
                continue
            relative = Path(info["data_path"])
            expected = sensor_relative_path(frame["log_name"], camera, frame["token"], index < 4)
            if relative != expected:
                raise ValueError("Sensor image assigned to the wrong history/future partition")
            path = (sensor_root / relative).resolve(strict=True)
            if not path.is_relative_to(sensor_root) or path in seen:
                raise ValueError("Escaped or duplicate synthetic sensor path")
            seen.add(path)
            sha = digest(path)
            if files.get(str(path)) != sha:
                raise ValueError("Unsealed or changed partitioned pixels")
            cameras[camera] = dict(path=path.relative_to(output).as_posix(), sha256=sha)
        if history_only and index not in rendered_indices:
            if cameras or any(any(value is not None for value in info.values()) for info in frame["cams"].values()):
                raise ValueError("Unrendered frames must contain numerical state and no camera payload")
        else:
            if len(cameras) != 4 or (camera_set is not None and camera_set != set(cameras)):
                raise ValueError("Four consistent rendered cameras per stored frame required")
            camera_set = set(cameras)
        sections[role].append(
            dict(
                frame_index=index,
                tick=int(tick),
                timestamp_us=timestamp[index],
                relative_to_anchor_s=(timestamp[index] - anchor) / 1e6,
                cameras=cameras,
            )
        )
    value = dict(
        schema_version=4 if rendered_indices == [3] else 3 if history_only else 2,
        kind="partitioned_navsim_synthetic_window",
        storage_layout=image_storage_layout(image_contract),
        branch_id=frames[3]["log_name"],
        native_log=dict(path=log_path.relative_to(output).as_posix(), sha256=digest(log_path)),
        anchor_index=3,
        policy_input_frame_indices=rendered_indices if history_only else [0, 1, 2, 3],
        future_images_are_policy_inputs=False,
        archived_future_points=len(frames) - 4,
        camera_names=sorted(camera_set),
        history=sections["history"],
        future=sections["future"],
        image_counts={
            role: sum(len(f["cameras"]) for f in rows) for role, rows in sections.items()
        },
        calibration_and_objects="native_log_per_frame",
        trainable=False,
    )
    if history_only:
        value.update(image_contract=image_contract, allowed_anchor_indices=[3],
                     rendered_frame_indices=rendered_indices, physical_history_frame_indices=[0, 1, 2, 3],
                     future_images_included=False)
    target = output / "manifests" / (frames[3]["log_name"] + ".window.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if target.exists() and target.read_text() != payload:
        raise ValueError("Synthetic window manifest is immutable")
    if not target.exists():
        target.write_text(payload)
    files[str(target)] = digest(target)
    result = dict(
        storage_layout=value["storage_layout"],
        window_manifest=str(target),
        image_counts=value["image_counts"],
        archived_future_points=len(frames) - 4,
    )
    if history_only:
        result.update(image_contract=image_contract, allowed_anchor_indices=[3],
                      future_images_included=False)
    return result
