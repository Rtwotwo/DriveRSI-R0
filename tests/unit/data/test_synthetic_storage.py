import json
import pickle

import pytest
from PIL import Image

from driversi.agents.backends.native import digest
from driversi.data.datasets.synthetic_storage import sensor_relative_path, write_window_manifest


def sample(tmp_path):
    frames, files = [], {}
    for i in range(14):
        frame = dict(
            timestamp=1_000_000 + i * 500_000, token=f"frame{i}", log_name="branch", cams={}
        )
        for camera in ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0"):
            rel = sensor_relative_path("branch", camera, frame["token"], i < 4)
            p = tmp_path / "sensors" / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 8), (i, 20, 30)).save(p)
            files[str(p)] = digest(p)
            frame["cams"][camera] = dict(data_path=rel.as_posix())
        frames.append(frame)
    log = tmp_path / "logs/branch.pkl"
    log.parent.mkdir()
    log.write_bytes(pickle.dumps(frames))
    files[str(log)] = digest(log)
    return frames, files, log


def test_separate_pixels_native_log_and_history_only_encoder(tmp_path):
    from driversi.data.catalog.derived_features import history

    frames, files, log = sample(tmp_path)
    value = write_window_manifest(tmp_path, frames, list(range(5, 71, 5)), log, files)
    assert value["image_counts"] == dict(history=16, future=40)
    assert sorted(p.suffix for p in (tmp_path / "logs").iterdir()) == [".pkl"]
    assert all(
        isinstance(pickle.loads(p.read_bytes()), list) for p in (tmp_path / "logs").iterdir()
    )
    assert "/manifests/" in value["window_manifest"]
    manifest = json.loads(open(value["window_manifest"]).read())
    assert manifest["policy_input_frame_indices"] == [0, 1, 2, 3]
    assert not manifest["future_images_are_policy_inputs"]
    assert manifest["future"][0]["relative_to_anchor_s"] == 0.5
    assert manifest["future"][-1]["relative_to_anchor_s"] == 5
    assert write_window_manifest(tmp_path, frames, list(range(5, 71, 5)), log, files) == value
    export = tmp_path / "export.json"
    export.write_text(
        json.dumps(dict(log_path=str(log), sensor_root=str(tmp_path / "sensors"), files=files))
    )
    _, evidence, images = history(export, ["CAM_F0", "CAM_L0", "CAM_R0"], 8)
    assert len(images) == 12
    assert all("/history/" in e["path"] for e in evidence)
    assert [i.getpixel((0, 0))[0] for i in images] == [0] * 3 + [1] * 3 + [2] * 3 + [3] * 3


def test_wrong_role_and_pixel_mutation_are_rejected(tmp_path):
    frames, files, log = sample(tmp_path)
    original = frames[4]["cams"]["CAM_F0"]["data_path"]
    frames[4]["cams"]["CAM_F0"]["data_path"] = original.replace("future/", "history/", 1)
    with pytest.raises(ValueError, match="partition"):
        write_window_manifest(tmp_path, frames, list(range(14)), log, files)
    frames[4]["cams"]["CAM_F0"]["data_path"] = original
    (tmp_path / "sensors" / original).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        write_window_manifest(tmp_path, frames, list(range(14)), log, files)


@pytest.mark.parametrize("component", ["../escape", "x/y", "x\\y", ".."])
def test_path_components_cannot_escape(component):
    with pytest.raises(ValueError):
        sensor_relative_path(component, "CAM_F0", "frame", True)


@pytest.mark.parametrize(
    "native_loader",
    [
        "agents/augustus/native/AUGUstus_SFT/third_party/navsim/navsim/common/dataloader.py",
        "agents/chainflow/native/navsim/common/dataloader.py",
    ],
)
def test_native_unfiltered_log_discovery(tmp_path, native_loader):
    import ast
    from types import SimpleNamespace

    from driversi.agents.backends.native import project_root

    path = project_root() / native_loader
    if not path.exists():
        pytest.skip("Native implementation path not installed")
    source = ast.parse(path.read_text())
    node = next(
        n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "filter_scenes"
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            node,
        ],
        type_ignores=[],
    )
    ns = dict(pickle=pickle, tqdm=lambda x, **kw: x)
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), ns)
    frames, files, log = sample(tmp_path)
    write_window_manifest(tmp_path, frames, list(range(5, 71, 5)), log, files)
    config = SimpleNamespace(
        log_names=None,
        tokens=None,
        num_frames=14,
        frame_interval=14,
        has_route=False,
        num_history_frames=4,
        max_scenes=None,
    )
    result = ns["filter_scenes"](tmp_path / "logs", config)
    assert list(result) == ["frame3"] and len(result["frame3"]) == 14


@pytest.mark.parametrize("anchor_only",[False,True])
def test_explicit_history_contract_preserves_ten_numeric_future_frames(tmp_path, anchor_only):
    contract="anchor_rgb_history_future_states_v1" if anchor_only else "history_rgb_future_states_v1"
    frames, files, log=sample(tmp_path)
    for i,frame in enumerate(frames):
        if i == 3 or (not anchor_only and i < 4):continue
        for info in frame["cams"].values():
            image=tmp_path/"sensors"/info["data_path"]
            files.pop(str(image));image.unlink()
            info["data_path"]=None
    log.write_bytes(pickle.dumps(frames));files[str(log)]=digest(log)
    with pytest.raises(ValueError,match="Four consistent"):
        write_window_manifest(tmp_path,frames,list(range(14)),log,files)
    value=write_window_manifest(tmp_path,frames,list(range(14)),log,files,image_contract=contract)
    assert value["image_counts"] == dict(history=4 if anchor_only else 16,future=0)
    window=json.loads(open(value["window_manifest"]).read())
    assert window["schema_version"] == (4 if anchor_only else 3) and len(window["future"]) == 10
    assert window["rendered_frame_indices"] == ([3] if anchor_only else [0,1,2,3])
    assert all(f["cameras"] == {} for f in window["future"])
    assert window["allowed_anchor_indices"] == [3]
    assert window["future"][-1]["timestamp_us"] == frames[-1]["timestamp"]


def test_history_contract_rejects_future_rgb_and_unknown_contract(tmp_path):
    frames,files,log=sample(tmp_path)
    with pytest.raises(ValueError,match="no camera payload"):
        write_window_manifest(tmp_path,frames,list(range(14)),log,files,image_contract="history_rgb_future_states_v1")
    with pytest.raises(ValueError,match="Unknown"):
        write_window_manifest(tmp_path,frames,list(range(14)),log,files,image_contract="auto")
