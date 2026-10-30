import copy
import json

import numpy as np
import pytest

from driversi_agents.augustus.branch_supervision import (
    format_prediction,
    future_position,
    normalize_joint_manifest,
    prediction_draft,
    require_label_review,
    seal,
    trajectories,
    validate_sample,
    verify_export_binding,
)


def frames():
    camera = dict(
        sensor2lidar_rotation=np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0.0]]),
        sensor2lidar_translation=[0, 0, 1.5],
        cam_intrinsic=np.array([[400, 0, 480], [0, 400, 270], [0, 0, 1.0]]),
        distortion=np.zeros(5),
    )
    return [
        dict(
            timestamp=1000000 + i * 500000,
            log_name="synthetic-a",
            ego2global_translation=[i, 0, 0],
            ego2global_rotation=[1, 0, 0, 0],
            cams={c: copy.deepcopy(camera) for c in ("CAM_L0", "CAM_F0", "CAM_R0")},
            anns=dict(
                track_tokens=["car-a"], gt_boxes=[[20 - i, 0, 1, 4, 2, 2, 0]], gt_names=["vehicle"]
            ),
        )
        for i in range(12)
    ]


def test_real_horizon_and_current_origin():
    h, f = trajectories(frames())
    assert h.shape == (4, 3) and f.shape == (8, 3)
    np.testing.assert_equal(h[-1], 0)
    np.testing.assert_equal(f[:, 0], np.arange(1, 9))
    bad = frames()
    bad[7]["timestamp"] += 200000
    with pytest.raises(ValueError, match="Continuous"):
        trajectories(bad)


@pytest.mark.parametrize(
    "xy,label",
    [
        ([3, 0], "near, front"),
        ([15, 3], "middle, left"),
        ([40, -5], "far, right"),
        ([-1, 0], "behind"),
    ],
)
def test_explicit_label_policy(xy, label):
    assert future_position(xy) == label


def test_projection_and_identity_tracking():
    f = frames()
    sizes = {c: (960, 540) for c in f[3]["cams"]}
    draft = prediction_draft(f, sizes)
    assert len(draft["objects"]) == 1  # never count three cameras as three actors
    assert draft["objects"][0]["actor_id"] == "car-a"
    assert draft["objects"][0]["future_position"] == "near, front"
    f[-1]["anns"]["track_tokens"] = ["different-car"]
    draft = prediction_draft(f, sizes)
    assert not draft["objects"]
    assert draft["unresolved"][0]["reason"] == "future_identity_unavailable"


def test_no_future_disappearance_fallback():
    f = frames()
    f[-1]["anns"]["track_tokens"] = []
    f[-1]["anns"]["gt_boxes"] = []
    d = prediction_draft(f, {c: (960, 540) for c in f[3]["cams"]})
    assert d["unresolved"] and d["trainable"] is False


def test_review_is_bound_and_complete(tmp_path):
    p = tmp_path / "draft.json"
    p.write_text(json.dumps(dict(objects=[dict(actor_id="a")], unresolved=[])))
    q = tmp_path / "review.json"
    review = dict(
        draft_sha256=seal(p),
        approved=True,
        reviewer="independent",
        checked_actor_ids=["a"],
        boxes_aligned=True,
        visibility_checked=True,
        future_identity_checked=True,
    )
    q.write_text(json.dumps(review))
    require_label_review(p, q)
    p.write_text(json.dumps(dict(objects=[], unresolved=[])))
    with pytest.raises(ValueError, match="independent review"):
        require_label_review(p, q)


def test_native_empty_answer_is_explicit():
    assert json.loads(format_prediction([]).split("\n")[1]) == [
        dict(bbox_2d=[], future_position=[])
    ]


def test_export_resume_requires_exact_branch_not_overlapping_inputs(tmp_path):
    a = tmp_path / "branch-a"
    b = tmp_path / "branch-b"
    a.write_text("world a")
    b.write_text("world b")
    parent = dict(source_log="training-log", token="anchor", anchor_index=24, log_path="raw")
    exported = dict(files={str(a): seal(a)}, parent=parent, anchor_index=3)
    verify_export_binding(exported, {str(a): seal(a)}, parent)
    # No intersection used to pass and overwrite provenance with branch B.
    with pytest.raises(ValueError, match="exact selected branch"):
        verify_export_binding(exported, {str(b): seal(b)}, parent)
    with pytest.raises(ValueError, match="parent or anchor"):
        verify_export_binding(exported, {str(a): seal(a)}, {**parent, "token": "other"})
    a.write_text("changed physical world")
    with pytest.raises(ValueError):
        verify_export_binding(exported, {str(a): exported["files"][str(a)]}, parent)


def test_diagnostic_preview_cannot_become_a_training_sample(tmp_path):
    p = tmp_path / "preview.json"
    p.write_text(
        json.dumps(
            dict(
                kind="augustus_joint_supervision_preview",
                trainable=False,
                native_decode_passed=True,
            )
        )
    )
    with pytest.raises(ValueError, match="admitted native synthetic"):
        validate_sample(p)


def test_mixture_pins_base_and_rejects_changed_identity(tmp_path):
    base = tmp_path / "base.json"
    base.write_text("{}")
    value = dict(
        schema_version=2,
        kind="augustus_joint_cache",
        dataset_version="pilot",
        base_manifest=dict(path=str(base), sha256=seal(base)),
        base_cache_files={},
        synthetic_samples=[dict(path="sample", sha256="not-used")],
        train_cache="cache",
        train_cache_files={},
    )
    base.write_text('{"changed":true}')
    with pytest.raises(ValueError, match="Base dataset identity"):
        normalize_joint_manifest(value, tmp_path / "mixture.json")


def test_original_relabel_cannot_erase_joint_lineage(tmp_path):
    from driversi_agents.augustus.data import relabel

    p = tmp_path / "joint.json"
    p.write_text(json.dumps(dict(kind="augustus_joint_cache")))
    with pytest.raises(ValueError, match="cannot strip"):
        relabel(p, tmp_path / "unused-proposals", tmp_path / "must-not-exist", "original")
    assert not (tmp_path / "must-not-exist").exists()


def test_mixed_payload_rechecked_at_native_cache_consumption(tmp_path, monkeypatch):
    import sys
    import types

    from driversi_agents.augustus.data import read_cache

    class Rows(list):
        def __getitem__(self, key):
            return [r[key] for r in self] if isinstance(key, str) else super().__getitem__(key)

    original = dict(
        frame_id="original", video=b"real video", solution_prediction="original language"
    )
    synthetic = dict(
        frame_id="synthetic", video=b"rendered video", solution_prediction="reviewed language"
    )
    blocks = {}
    values = {}
    for name, rows in (
        ("base", [original]),
        ("synth", [synthetic]),
        ("mix", [original, {**synthetic, "solution_prediction": "changed target"}]),
    ):
        root = tmp_path / name
        (root / "batch_0").mkdir(parents=True)
        values[str(root / "batch_0")] = Rows(copy.deepcopy(rows))
        blocks[name] = dict(cache=str(root), records=[dict(frame_id=x["frame_id"]) for x in rows])
    # This exercises native composition validation without loading heavy HF dependencies.
    stub = types.ModuleType("datasets")
    stub.load_from_disk = lambda p: values[p]
    stub.concatenate_datasets = lambda ds: Rows(x for d in ds for x in d)
    monkeypatch.setitem(sys.modules, "datasets", stub)
    blocks["mix"]["joint_source_blocks"] = [blocks["base"], blocks["synth"]]
    with pytest.raises(ValueError, match="rows differ"):
        read_cache(blocks["mix"])
    values[str(tmp_path / "mix/batch_0")] = Rows([original, synthetic])
    assert len(read_cache(blocks["mix"])) == 2


def test_near_plane_outside_field_is_not_false_unresolved():
    from driversi_agents.augustus.branch_supervision import stitched_bbox
    cam=frames()[0]['cams']['CAM_F0']
    bbox,reason=stitched_bbox([0,30,1,4,2,2,0],cam,'CAM_F0',(960,540))
    assert bbox is None and reason=='outside_crop'


def test_near_plane_visible_box_remains_unresolved():
    from driversi_agents.augustus.branch_supervision import stitched_bbox
    cam=frames()[0]['cams']['CAM_F0']
    bbox,reason=stitched_bbox([0.5,0,1,4,2,2,0],cam,'CAM_F0',(960,540))
    assert bbox is None and reason=='near_plane_intersection'
