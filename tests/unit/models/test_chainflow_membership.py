"""Official split membership never substitutes reached-anchor state or images."""

import copy
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from driversi.agents.backends.native import digest
from driversi_agents.chainflow import data, synthetic


@pytest.fixture
def source(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "official_train_logs", lambda: {"train-log", "other-train-log"})
    monkeypatch.setattr(data, "official_train_tokens", lambda: {"official-907", "foreign-official"})
    frames = [
        dict(
            token=f"raw-{i}",
            log_name="train-log",
            roadblock_ids=[f"road-{i}"],
            driving_command=[1, 0, 0, 0],
        )
        for i in range(940)
    ]
    frames[907]["token"] = "official-907"
    path = tmp_path / "train-log.pkl"
    path.write_bytes(pickle.dumps(frames))
    parent = dict(source_log="train-log", token="raw-925", log_path=str(path), anchor_index=925)
    member = dict(
        source_log="train-log",
        token="official-907",
        anchor_index=907,
        reference_role="official_split_membership_not_physical_anchor",
    )
    metadata = dict(
        source_train_membership=member,
        source_log="train-log",
        split="train",
        source_start_frame=921,
        scenario_id="raw-921",
        navsim_source={"log_path": str(path)},
    )
    return frames, parent, member, metadata


def test_canonical_anchor_needs_no_derived_membership(source):
    frames, parent, member, metadata = source
    parent.update(token="official-907", anchor_index=907)
    assert synthetic.validate_source_train_membership(parent, frames) is None


def test_derived_anchor_requires_explicit_same_log_raw_evidence(source):
    frames, parent, member, metadata = source
    with pytest.raises(ValueError, match="requires explicit"):
        synthetic.validate_source_train_membership(parent, frames)
    assert (
        synthetic.validate_source_train_membership(
            parent, frames, member, metadata=metadata, anchor_tick=20
        )
        == member
    )
    assert parent["anchor_index"] == 925 and parent["token"] == "raw-925"


@pytest.mark.parametrize(
    "mutate",
    [
        "different_log",
        "fake_official",
        "wrong_index",
        "missing_scene",
        "replace_parent",
        "wrong_role",
        "bool_index",
        "test_log",
    ],
)
def test_membership_cannot_launder_foreign_or_unbound_sources(source, mutate):
    frames, parent, member, metadata = copy.deepcopy(source)
    if mutate == "different_log":
        member["source_log"] = "other-train-log"
    elif mutate == "fake_official":
        member["token"] = "foreign-official"
    elif mutate == "wrong_index":
        member["anchor_index"] = 908
    elif mutate == "missing_scene":
        metadata.pop("source_train_membership")
    elif mutate == "replace_parent":
        parent.update(token="official-907", anchor_index=907)
    elif mutate == "wrong_role":
        member["reference_role"] = "actual_anchor"
    elif mutate == "bool_index":
        member["anchor_index"] = True
    elif mutate == "test_log":
        parent["source_log"] = "navtest-log"
    with pytest.raises(ValueError):
        synthetic.validate_source_train_membership(
            parent, frames, member, metadata=metadata, anchor_tick=20
        )


@pytest.mark.parametrize("history_only,explicit_contract,anchor_only", [(False,False,False),(False,True,False),(True,True,False),(True,True,True)])
def test_derived_export_keeps_925_navigation_and_executed_geometry(source, tmp_path, history_only, explicit_contract, anchor_only):
    frames, parent, member, metadata = source
    metadata.update(
        pose_reference="vehicle_center",
        coordinate_frame="map_axes_translated_from_initial_ego",
        origin_xyz=[1000.0, 2000.0, 0.0],
        map_name="fixture-map",
        traffic_lights=[dict(time=0.0, records=[], available=True)],
    )
    metadata["navsim_source"]["anchor_timestamp_us"] = 921 * 500000
    states = np.zeros((71, 1, 13))
    states[:, 0, 0] = 10.0 + np.arange(71) * 0.1
    scene = tmp_path / "scene.npz"
    run = tmp_path / "teacher.npz"
    np.savez(
        scene,
        metadata=json.dumps(metadata),
        actor_ids=np.array(["ego"]),
        kinds=np.array([1]),
        dimensions=np.array([[5.0, 2.0, 2.0]]),
        poses=states[:, :, :4],
        times=np.arange(71) * 0.1,
    )
    np.savez(
        run,
        metadata=json.dumps(dict(scene_metadata=metadata, dt_seconds=0.1)),
        actor_ids=np.array(["ego"]),
        states=states,
        presence=np.ones((71, 1), bool),
    )
    renders = []
    for tick in ([20] if anchor_only else range(5, 21, 5) if history_only else range(5, 71, 5)):
        cameras = {}
        for name in synthetic.CAMERAS:
            image = tmp_path / f"{tick}-{name}.png"
            image.write_bytes(f"unit-fixture-{tick}-{name}".encode())
            cameras[name] = dict(
                file=image.name,
                calibration_source="native_checkpoint_optimized_camera_and_colmap_optics",
                camera_to_rear_axle_reference="reconstruction_refined_nuplan_rear_axle",
                camera_to_rear_axle=np.eye(4).tolist(),
                distorted_intrinsic=np.eye(3).tolist(),
                distortion=[0.0] * 5,
            )
        renders.append(dict(tick=tick, cameras=cameras))
    report = tmp_path / "render.json"
    report.write_text(
        json.dumps(
            dict(
                renderer_backend="mtgs_native",
                status="rendered_diagnostic",
                inputs={str(p): digest(p) for p in (scene, run)},
                history_ticks=[5, 10, 15, 20],
                future_ticks=list(range(25, 71, 5)),
                frames=renders,
                **({"image_contract": "anchor_rgb_history_future_states_v1" if anchor_only else "history_rgb_future_states_v1" if history_only else "history_future_rgb_v1"} if explicit_contract else {}),
            )
        )
    )
    contract = "anchor_rgb_history_future_states_v1" if anchor_only else "history_rgb_future_states_v1" if history_only else None
    exported = synthetic.export_native_frames(scene, run, report, tmp_path / "export", image_contract=contract)
    actual = pickle.loads(Path(exported["log_path"]).read_bytes())
    assert exported["parent"] == parent
    assert exported["source_train_membership"] == member
    assert len(actual) == 14 and actual[3]["timestamp"] == 925 * 500000
    assert all(f["roadblock_ids"] == ["road-925"] for f in actual)
    np.testing.assert_allclose(
        actual[3]["ego2global_translation"], [1010.539, 2000.0, 0.0], atol=1e-10
    )
    assert all(f["roadblock_ids"] != ["road-907"] for f in actual)
    assert not exported["trainable"]
    assert exported["image_counts"] == dict(history=4 if anchor_only else 16, future=0 if history_only else 40)
    branch_report = tmp_path / "branch.json"
    branch_report.write_text(json.dumps(dict(anchor_tick=20)))
    visual = tmp_path / "visual.json"
    visual.write_text(json.dumps(dict(render_report=str(report))))
    proof = dict(parent=parent, branch=dict(scene=str(scene), rollout=str(run), report=str(branch_report)),
        native=dict(log_path=exported["log_path"], sensor_root=exported["sensor_root"]),
        visual_admission=dict(path=str(visual)), files=exported["files"], source_train_membership=member)
    if history_only:
        proof["image_contract"] = contract
        assert exported["allowed_anchor_indices"] == [3]
        assert not (Path(exported["sensor_root"]) / "future").exists()
        with pytest.raises(ValueError, match="ten actual future"):
            synthetic.export_native_frames(scene, run, report, tmp_path / "short_future", image_contract=contract, future_points=8)
        assert all(all(all(v is None for v in c.values()) for c in f["cams"].values()) for f in actual[4:])
    synthetic.validate_window(dict(token=exported["token"]), actual, proof)
    changed = copy.deepcopy(actual)
    changed[-1]["ego2global_translation"][0] += 1
    with pytest.raises(AssertionError):
        synthetic.validate_window(dict(token=exported["token"]), changed, proof)
    if anchor_only:
        assert all(all(all(v is None for v in c.values()) for c in f["cams"].values()) for f in actual[:3])
        changed = copy.deepcopy(actual)
        changed[0]["cams"]["CAM_F0"] = copy.deepcopy(actual[3]["cams"]["CAM_F0"])
        with pytest.raises(ValueError, match="Unrendered"):
            synthetic.validate_window(dict(token=exported["token"]), changed, proof)
    if history_only:
        changed = copy.deepcopy(actual)
        changed[4]["cams"]["CAM_F0"] = copy.deepcopy(actual[3]["cams"]["CAM_F0"])
        with pytest.raises(ValueError, match="Unrendered"):
            synthetic.validate_window(dict(token=exported["token"]), changed, proof)
        invalid = json.loads(report.read_text())
        invalid["frames"].append(dict(invalid["frames"][-1], tick=25))
        report.write_text(json.dumps(invalid))
        with pytest.raises(ValueError, match="render time"):
            synthetic.export_native_frames(scene, run, report, tmp_path / "bad_export", image_contract=contract)


def test_membership_parse_reused_by_bytes_and_refreshes_with_changed_content(tmp_path, monkeypatch):
    import os

    import yaml

    from driversi_agents.chainflow import data

    source = (
        tmp_path
        / "agents/chainflow/native/navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml"
    )
    source.parent.mkdir(parents=True)
    source.write_text("log_names: [log_a]\ntokens: [token_a]\n")
    monkeypatch.setattr(data, "project_root", lambda: tmp_path)
    data._parse_official_train_split.cache_clear()
    original = yaml.safe_load
    calls = []

    def read(content):
        calls.append(1)
        return original(content)

    monkeypatch.setattr(yaml, "safe_load", read)
    logs = data.official_train_logs()
    logs.add("not_official")
    assert data.official_train_logs() == {"log_a"}
    assert data.official_train_tokens() == {"token_a"}
    assert len(calls) == 1
    stat = source.stat()
    source.write_text("log_names: [log_b]\ntokens: [token_b]\n")
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert data.official_train_logs() == {"log_b"}
    assert data.official_train_tokens() == {"token_b"}
    assert len(calls) == 2
    data._parse_official_train_split.cache_clear()


@pytest.mark.parametrize("version,contract",[(3,"history_rgb_future_states_v1"),(4,"anchor_rgb_history_future_states_v1")])
def test_history_only_receipt_cannot_reanchor_in_unrendered_future(tmp_path, monkeypatch, version, contract):
    monkeypatch.setattr(data, "official_train_logs", lambda: {"train-log"})
    monkeypatch.setattr(data, "official_train_tokens", lambda: {"official"})
    native = {k: str(tmp_path / k) for k in ("log_path", "sensor_root", "metric_cache", "training_metric_cache", "cache_receipt")}
    native["token"] = "synthetic_branch"
    proof = dict(schema_version=version, kind="chainflow_synthetic_admission",
        image_contract=contract, source_train_membership=None,
        parent=dict(source_log="train-log",token="official",log_path=str(tmp_path/"train-log.pkl"),anchor_index=3),
        native=native, branch={k:str(tmp_path/k) for k in ("report","scene","rollout","trajectory_window")},
        visual_admission={}, files={})
    p=tmp_path/"admission.json";p.write_text(json.dumps(proof))
    row={k:v for k,v in native.items() if k != "cache_receipt"}
    row.update(anchor_index=4,source_log="train-log",admission=dict(path=str(p),sha256=digest(p)))
    with pytest.raises(ValueError,match="Synthetic token"):
        synthetic.validate_admission(row)
