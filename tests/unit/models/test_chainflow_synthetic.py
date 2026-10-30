"""Synthetic evidence cannot be substituted by flags, parent tokens or stale files."""

import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from driversi.agents.backends.native import digest
from driversi_agents.chainflow import data, synthetic


def test_native_synthetic_scoring_namespaces_are_distinct():
    rows = [
        dict(token="parent", kind="original_navtrain", training_metric_cache="/original"),
        dict(token="synthetic_new", kind="synthetic_navtrain", training_metric_cache="/branch"),
    ]
    original, generated = data.scoring_cache_paths(rows)
    assert original == {"parent": Path("/original")}
    assert generated == {"synthetic_new": Path("/branch")}
    with pytest.raises(ValueError, match="Duplicate scoring"):
        data.scoring_cache_paths(rows + [dict(rows[1])])
    rows[1]["token"] = "another-official-token"
    with pytest.raises(ValueError, match="distinct native token"):
        data.scoring_cache_paths(rows)


@pytest.mark.parametrize("admission", [None, {"trainable": True}, {"admitted": True}, {}])
def test_boolean_flags_never_authorize_synthetic_training(admission):
    with pytest.raises(ValueError, match="canonical sealed visual admission"):
        synthetic.validate_admission({"admission": admission})


def test_branch_identity_depends_on_physics_and_anchor():
    files = {"/scene": "a", "/rollout": "b"}
    one = synthetic.synthetic_token(files, 20)
    assert one.startswith("synthetic_")
    assert one == synthetic.synthetic_token(dict(reversed(list(files.items()))), 20)
    assert one == synthetic.synthetic_token({"/copy/scene": "a", "/copy/rollout": "b"}, 20)
    assert one != synthetic.synthetic_token(files, 25)
    assert one != synthetic.synthetic_token({**files, "/rollout": "c"}, 20)


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "official_train_logs", lambda: {"train-log"})
    monkeypatch.setattr(data, "official_train_tokens", lambda: {"parent"})
    files = {}

    def write(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value))
        files[str(path)] = digest(path)
        return str(path)

    source = write("source.pkl", {})
    scene = write("scene.npz", {})
    rollout = write("teacher.npz", {})
    window = write("window.json", {})
    token = synthetic.synthetic_token({scene: files[scene], rollout: files[rollout]}, 20)
    score = dict(
        score=0.97,
        driving_direction_compliance=1.0,
        actual_world_seconds=5.0,
        protocol="navsim-v1-native-math-executed-counterfactual-v1",
    )
    branch_report = write(
        "branch.json",
        dict(
            source_log="train-log",
            anchor_tick=20,
            teacher_qualified=True,
            files={scene: files[scene], rollout: files[rollout]},
            trajectory_window_sha256=files[window],
            trajectory_evidence=dict(
                branch="teacher",
                completed_steps=50,
                producer_identity="producer",
                history_changed=True,
                restore_exact=True,
                arrival_contact_records=0,
                future_contact_records=0,
                events=dict(collision=0, offroad=0, route_support_end=0),
                score=score,
            ),
        ),
    )
    native_log = write(token + "_log.pkl", {})
    metric = write("metric.pkl", {})
    training = write("training.pkl", {})
    cache_receipt = write(
        "cache-receipt.json",
        dict(
            kind="chainflow_counterfactual_cache_v1",
            token=token,
            branch_files={scene: files[scene], rollout: files[rollout]},
            cache_files={metric: files[metric], training: files[training]},
        ),
    )
    visual = write("visual.json", {})
    verdict = dict(
        admitted=True,
        source_log="train-log",
        parent_token="parent",
        branch_report_sha256=files[branch_report],
        trajectory_window=window,
        files={visual: files[visual]},
    )
    module = ModuleType("driversi.rendering.diagnostics.worldengine_admission")
    module.validate_admission = lambda path: verdict
    monkeypatch.setitem(sys.modules, module.__name__, module)
    proof = dict(
        schema_version=1,
        kind="chainflow_synthetic_admission",
        parent=dict(source_log="train-log", token="parent", log_path=source, anchor_index=3),
        branch=dict(report=branch_report, scene=scene, rollout=rollout, trajectory_window=window),
        visual_admission=dict(path=visual, sha256=files[visual]),
        native=dict(
            log_path=native_log,
            sensor_root=str(tmp_path),
            token=token,
            metric_cache=metric,
            training_metric_cache=training,
            cache_receipt=cache_receipt,
        ),
        files=dict(files),
    )
    path = write("admission.json", proof)
    row = dict(
        source_log="train-log",
        token=token,
        log_path=native_log,
        sensor_root=str(tmp_path),
        anchor_index=3,
        metric_cache=metric,
        training_metric_cache=training,
        admission=dict(path=path, sha256=files[path]),
    )
    return row, proof, files, verdict


def test_exact_evidence_closure_is_required(evidence):
    row, proof, files, verdict = evidence
    assert synthetic.validate_admission(row, files) == proof
    truncated = dict(files)
    del truncated[proof["branch"]["rollout"]]
    with pytest.raises(ValueError, match="complete synthetic evidence closure"):
        synthetic.validate_admission(row, truncated)


@pytest.mark.parametrize(
    "field,value",
    [
        ("admitted", False),
        ("admitted", 1),
        ("parent_token", "another-parent"),
        ("source_log", "test-log"),
        ("branch_report_sha256", "other-branch"),
        ("trajectory_window", "other-window"),
    ],
)
def test_visual_receipt_must_admit_same_branch(evidence, field, value):
    row, proof, files, verdict = evidence
    verdict[field] = value
    with pytest.raises(ValueError, match="visual|Visual"):
        synthetic.validate_admission(row, files)


def test_mutated_physics_cannot_reuse_sealed_admission(evidence):
    row, proof, files, verdict = evidence
    Path(proof["branch"]["rollout"]).write_text("changed")
    with pytest.raises(ValueError, match="evidence changed"):
        synthetic.validate_admission(row, files)


def test_parent_token_cannot_be_used_as_synthetic_cache_key(evidence):
    row, proof, files, verdict = evidence
    row["token"] = "parent"
    with pytest.raises(ValueError, match="payload differs|distinct"):
        synthetic.validate_admission(row, files)


@pytest.fixture
def physical_cache(tmp_path):
    from types import SimpleNamespace

    import numpy as np
    from shapely.geometry import box

    states = np.zeros((71, 2, 13), dtype=float)
    states[:, 0, 0] = 1.461
    states[:, 1, 0] = 10
    presence = np.ones((71, 2), dtype=bool)
    scene = SimpleNamespace(
        actor_ids=("ego", "vehicle"),
        dimensions=np.array([[4.0, 2.0, 2.0], [4.0, 2.0, 2.0]]),
        metadata={"origin_xyz": [0.0, 0.0, 0.0]},
    )

    class Occupancy:
        tokens = ["vehicle"]
        polygon = box(8.0, -1.0, 12.0, 1.0)

        def __getitem__(self, key):
            return self.polygon

    class Observation:
        red_light_token = "red_light"
        frames = [Occupancy() for _ in range(51)]

        def __getitem__(self, index):
            return self.frames[index]

    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"native_reference_progress": 4.0}))
    cache = SimpleNamespace(
        ego_state=SimpleNamespace(rear_axle=SimpleNamespace(x=0.0, y=0.0, heading=0.0)),
        observation=Observation(),
        pdm_progress=np.array([4.0]),
    )
    return (
        cache,
        {"native": {"cache_receipt": str(receipt)}},
        scene,
        {"states": states, "presence": presence},
        20,
    )


def test_real_cache_context_checks_all_51_physical_ticks(physical_cache):
    cache, proof, scene, run, anchor = physical_cache
    synthetic.validate_cache_context({}, cache, proof, scene, run, anchor, training=True)
    cache.observation.frames[49].tokens = []
    with pytest.raises(ValueError, match="drops or borrows"):
        synthetic.validate_cache_context({}, cache, proof, scene, run, anchor, training=False)


def test_wrong_counterfactual_object_geometry_is_rejected(physical_cache):
    from shapely.geometry import box

    cache, proof, scene, run, anchor = physical_cache
    cache.observation.frames[30].polygon = box(9.0, -1.0, 13.0, 1.0)
    with pytest.raises(ValueError, match="object pose differs"):
        synthetic.validate_cache_context({}, cache, proof, scene, run, anchor, training=False)


def test_wrong_counterfactual_anchor_and_reference_are_rejected(physical_cache):
    import numpy as np

    cache, proof, scene, run, anchor = physical_cache
    cache.ego_state.rear_axle.x = 2.0
    with pytest.raises(AssertionError):
        synthetic.validate_cache_context({}, cache, proof, scene, run, anchor, training=False)
    cache.ego_state.rear_axle.x = 0.0
    cache.pdm_progress = np.array([0.0])
    with pytest.raises(ValueError, match="reference progress changed"):
        synthetic.validate_cache_context({}, cache, proof, scene, run, anchor, training=True)


def test_rendered_calibration_is_backend_specific():
    import pytest

    from driversi_agents.chainflow.synthetic import validate_rendered_calibration

    reference = {"camera_to_rear_axle_reference": "reconstruction_refined_nuplan_rear_axle"}
    native = {
        **reference,
        "calibration_source": "native_checkpoint_optimized_camera_and_colmap_optics",
    }
    portable = {
        **reference,
        "calibration_source": "reconstruction_colmap_pose_intrinsics_distortion",
    }
    validate_rendered_calibration({"renderer_backend": "mtgs_native"}, native)
    validate_rendered_calibration({"renderer_backend": "worldengine_portable"}, portable)
    for backend, capture in [
        ("mtgs_native", portable),
        ("worldengine_portable", native),
        ("arbitrary", native),
    ]:
        with pytest.raises(ValueError, match="backend"):
            validate_rendered_calibration({"renderer_backend": backend}, capture)
    with pytest.raises(ValueError, match="rear-axle"):
        validate_rendered_calibration(
            {"renderer_backend": "mtgs_native"},
            {**native, "camera_to_rear_axle_reference": "vehicle_center"},
        )


def test_schema2_admission_revalidates_membership_and_rejects_downgrade(
    evidence, monkeypatch, tmp_path
):
    import pickle
    from types import SimpleNamespace

    row, proof, files, verdict = evidence
    frames = [dict(token=f"raw-{i}", log_name="train-log") for i in range(940)]
    frames[907]["token"] = "parent"
    raw = tmp_path / "train-log.pkl"
    raw.write_bytes(pickle.dumps(frames))
    member = dict(
        source_log="train-log",
        token="parent",
        anchor_index=907,
        reference_role="official_split_membership_not_physical_anchor",
    )
    metadata = dict(
        source_train_membership=member,
        split="train",
        source_log="train-log",
        source_start_frame=921,
        scenario_id="raw-921",
        navsim_source=dict(log_path=str(raw)),
    )
    monkeypatch.setattr(
        synthetic, "load_physical_branch", lambda *args: (SimpleNamespace(metadata=metadata), {})
    )
    proof.update(
        schema_version=2,
        source_train_membership=member,
        parent=dict(source_log="train-log", token="raw-925", anchor_index=925, log_path=str(raw)),
    )
    proof["files"][str(raw)] = digest(raw)
    verdict["parent_token"] = "raw-925"
    path = Path(row["admission"]["path"])

    def save():
        path.write_text(json.dumps(proof))
        row["admission"]["sha256"] = digest(path)
        return {**proof["files"], str(path): digest(path)}

    assert synthetic.validate_admission(row, save()) == proof
    metadata["source_start_frame"] = 907
    with pytest.raises(ValueError, match="reached physical anchor"):
        synthetic.validate_admission(row, save())
    metadata["source_start_frame"] = 921
    frames[907]["token"] = "not-official-parent"
    raw.write_bytes(pickle.dumps(frames))
    proof["files"][str(raw)] = digest(raw)
    with pytest.raises(ValueError, match="not present"):
        synthetic.validate_admission(row, save())
    proof["schema_version"] = 1
    proof.pop("source_train_membership")
    with pytest.raises(ValueError, match="requires explicit source_train_membership"):
        synthetic.validate_admission(row, save())


@pytest.mark.parametrize("value", [0.95, 0.94, float("nan"), float("inf"), True, 1.01])
def test_numeric_cache_policy_rejects_invalid_or_boundary_score(value):
    with pytest.raises(ValueError, match="Counterfactual PDM"):
        synthetic.validate_counterfactual_score(
            {"score": value, "score_pass": True, "driving_direction_compliance": 1},
            "pdms_gt95_v1",
        )


def test_numeric_cache_policy_preserves_direction_as_diagnostic():
    score = {"score": 1.0, "score_pass": True, "driving_direction_compliance": 0.5}
    synthetic.validate_counterfactual_score(score, "pdms_gt95_v1")
    with pytest.raises(ValueError, match="direction"):
        synthetic.validate_counterfactual_score(score)
    assert score["driving_direction_compliance"] == 0.5
    synthetic.validate_counterfactual_score({"score": 0.95, "driving_direction_compliance": 1})
    with pytest.raises(ValueError, match="Unknown trajectory"):
        synthetic.validate_counterfactual_score(score, "invented")
