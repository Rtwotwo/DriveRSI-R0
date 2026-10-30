"""Protect native camera ordering and bounded provider composition."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "multiview_gs_window", ROOT / "src/rendering/gaussian/window_cli.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fixtures(tmp_path, extra=""):
    config = tmp_path / "model.yaml"
    config.write_text(
        "config:\n  cam_f0: [3]\n  cam_b0: [3]\n  cam_l0: [3]\n  cam_r0: [3]\n  cam_l1: []\n  num_poses: 8\n  long_trajectory_additional_poses: 2\n"
        + extra
    )
    builder = tmp_path / "feature.py"
    builder.write_text(
        "def f(cameras):\n    cameras = [cameras.cam_f0, cameras.cam_b0, cameras.cam_l0, cameras.cam_l1, cameras.cam_r0]\n"
    )
    return config, builder


def test_native_camera_stack_order_is_not_export_order(tmp_path):
    value = MODULE.camera_contract(*fixtures(tmp_path))
    assert value["native_stack_order"] == ["CAM_F0", "CAM_B0", "CAM_L0", "CAM_R0"]
    assert value["history_ticks"] == [5, 10, 15, 20]
    assert value["future_ticks"] == list(range(25, 71, 5))


def test_changed_input_time_is_not_silently_accepted(tmp_path):
    config, builder = fixtures(tmp_path)
    config.write_text(config.read_text().replace("cam_f0: [3]", "cam_f0: [0, 1, 2, 3]"))
    with pytest.raises(ValueError, match="current-camera"):
        MODULE.camera_contract(config, builder)


def test_unknown_camera_order_fails_closed(tmp_path):
    config, builder = fixtures(tmp_path)
    builder.write_text("def f(cameras):\n    cameras = list(cameras)\n")
    with pytest.raises(ValueError, match="Unambiguous"):
        MODULE.camera_contract(config, builder)


def test_providers_retain_independent_admission_and_bounds():
    task = MODULE.task_spec("pilot", "teacher-task", "teacher", "/asset", "/env/python", 6)
    assert [row["provider"] for row in task["steps"]] == [
        "completed_navsim_teacher_reference",
        "native_branch_selection",
        "mtgs_selected_branch_render",
        "worldengine_source_reconstruction",
    ]
    assert task["steps"][1]["config"]["future_frames"] == 10
    assert task["steps"][2]["config"]["max_wall_seconds"] == 600
    assert all("train" not in row["provider"] for row in task["steps"])
    with pytest.raises(ValueError, match="GPU"):
        MODULE.task_spec("pilot", "teacher-task", "teacher", "/asset", "/env/python", -1)


INSPECT_SPEC = importlib.util.spec_from_file_location(
    "multiview_gs_inspect", ROOT / "src/rendering/diagnostics/window_review.py"
)
INSPECT = importlib.util.module_from_spec(INSPECT_SPEC)
INSPECT_SPEC.loader.exec_module(INSPECT)


def render_fixture():
    import copy

    frames = []
    rows = []
    for tick in range(0, 71, 5):
        token = f"capture{tick}"
        cameras = {
            cam: {
                "source_token": token,
                "object_bindings": {
                    "car": {"target_tick": tick, "target_physical_pose": [1, 2, 3, 4]}
                },
            }
            for cam in INSPECT.CAMERAS
        }
        frames.append({"tick": tick, "source_token": token, "cameras": copy.deepcopy(cameras)})
        rows.extend({"tick": tick, "camera": cam, "source_token": token} for cam in INSPECT.CAMERAS)
    return {
        "status": "rendered_diagnostic",
        "frames": frames,
        "history_ticks": [5, 10, 15, 20],
        "future_ticks": list(range(25, 71, 5)),
    }, {"frames": rows}


def test_inspection_rejects_cross_camera_state_mismatch():
    report, source = render_fixture()
    assert len(INSPECT.check_window(report, source)) == 60
    report["frames"][1]["cameras"]["CAM_L0"]["object_bindings"]["car"]["target_physical_pose"][
        0
    ] += 1
    with pytest.raises(ValueError, match="Different physical object"):
        INSPECT.check_window(report, source)


def test_inspection_rejects_source_swaps_and_missing_future():
    report, source = render_fixture()
    source["frames"][-1]["source_token"] = "othercapture"
    with pytest.raises(ValueError, match="synchronization"):
        INSPECT.check_window(report, source)
    report["frames"].pop()
    with pytest.raises(ValueError, match="Complete actual"):
        INSPECT.check_window(report, source)


def test_virtualenv_python_path_does_not_resolve_away_environment(tmp_path):
    binary = tmp_path / "nativepython"
    binary.write_text("binary")
    link = tmp_path / "venv" / "bin" / "python"
    link.parent.mkdir(parents=True)
    link.symlink_to(binary)
    assert MODULE.interpreter_path(link) == link.absolute()
    assert MODULE.interpreter_path(link) != binary.resolve()


def test_portable_static_background_inspection_still_requires_complete_timeline():
    report, source = render_fixture()
    report["renderer_backend"] = "worldengine_portable"
    for frame in report["frames"]:
        frame["unresolved_physical_boxes"] = {"parked": [[0, 0, 0]] * 8}
        for camera in frame["cameras"].values():
            camera.pop("object_bindings")
    assert len(INSPECT.check_window(report, source)) == 60
    source["frames"].pop()
    with pytest.raises(ValueError, match="Complete unique"):
        INSPECT.check_window(report, source)


def test_augustus_native_eight_future_images_keep_exact_coverage():
    report, source = render_fixture()
    report['future_ticks'] = list(range(25, 61, 5))
    report['frames'] = [f for f in report['frames'] if f['tick'] <= 60]
    source['frames'] = [f for f in source['frames'] if f['tick'] <= 60]
    assert len(INSPECT.check_window(report, source)) == 52
    source['frames'].pop()
    with pytest.raises(ValueError, match='Complete unique'):
        INSPECT.check_window(report, source)


def test_arbitrary_shortened_future_is_not_accepted():
    report, source = render_fixture()
    report['future_ticks'] = list(range(25, 56, 5))
    with pytest.raises(ValueError, match='eight-point or ten-point'):
        INSPECT.check_window(report, source)


def test_augustus_player_cannot_seek_nonexistent_future_images():
    page = INSPECT.viewer_html(['actor'], future_points=8)
    assert 'max="12"' in page and 'length:13' in page
    assert '===14' not in page and '25–70' not in page
    assert 'B0 仅用于' in page
    assert 'length:15' in INSPECT.viewer_html(['actor'])
