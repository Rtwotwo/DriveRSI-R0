import json
from pathlib import Path

import pytest

from driversi.agents.backends.native import digest
from driversi.data.datasets import synthetic_publish as mod


@pytest.fixture(autouse=True)
def classification_isolation(tmp_path, monkeypatch):
    from driversi.data.catalog import synthetic_classification
    from driversi.rsi.services import synthetic_classification as service

    monkeypatch.setattr(
        synthetic_classification, "library_root", lambda: tmp_path / "classification"
    )
    monkeypatch.setattr(service, "complete", lambda *args: dict(status="test_completed"))


def producer(staging):
    log = staging / "logs" / "branch.pkl"
    log.parent.mkdir(parents=True)
    log.write_bytes(b"opaque test log")
    sensor = staging / "sensors/history/branch/CAM_F0/frame.png"
    sensor.parent.mkdir(parents=True)
    sensor.write_bytes(b"opaque test pixel")
    value = dict(
        token="synthetic_fixture",
        storage_layout="logs_sensors_history_future_v2",
        trainable=False,
        log_path=str(log),
        sensor_root=str(staging / "sensors"),
        image_files=[str(sensor)],
        files={str(p): digest(p) for p in [log, sensor]},
    )
    (staging / "export.json").write_text(json.dumps(value))
    return value


def test_training_only_export_does_not_enqueue_classification(tmp_path, monkeypatch):
    from driversi.data.datasets import simscale_collection
    from driversi.data.catalog import synthetic_classification
    monkeypatch.setattr(simscale_collection,'project_root',lambda:tmp_path)
    root=tmp_path/'dataset';campaign=tmp_path/'exp/current'
    simscale_collection.initialize(root,campaign)
    monkeypatch.setattr(mod,'dataset_root',lambda:root/'synthetic_navsim')
    (root/'postprocessing.json').write_text(json.dumps(dict(schema_version=1,mode='training_only',
        campaign=str(campaign), features='deferred',scene_caption='deferred',classification='deferred',
        training_admission='unchanged')))
    monkeypatch.setattr(synthetic_classification,'enqueue',lambda *_:pytest.fail('classification called'))
    value=mod.publish_package(dict(branch='test'),producer,campaign/'native')
    assert value['trainable'] is False
    assert not (root/'synthetic_navsim/admitted').exists()


def test_atomic_canonical_package_resumes_without_copying(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    monkeypatch.setattr(mod, "dataset_root", lambda: root)
    alias = tmp_path / "exp/native"
    identity = dict(branch="sealed", version=2)
    export = mod.publish_package(identity, producer, alias)
    assert alias.is_symlink() and alias.resolve() == Path(export["storage_root"])
    assert all("/samples/" in p for p in export["files"])
    assert all("/.staging/" not in p for p in export["files"])
    assert not export["trainable"] and not (root / "admitted").exists()
    assert len(list((tmp_path / "classification/jobs").glob("*/job.json"))) == 1
    assert mod.publish_package(identity, lambda x: pytest.fail("repeated export"), alias) == export
    assert json.loads((alias / "export.json").read_text()) == export
    Path(export["log_path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        mod.publish_package(identity, producer, alias)


def test_failed_export_keeps_staging_but_exposes_no_native_alias(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    monkeypatch.setattr(mod, "dataset_root", lambda: root)
    alias = tmp_path / "exp/native"

    def fail(staging):
        staging.mkdir(parents=True)
        (staging / "partial").write_text("failed copy")
        raise RuntimeError("simulated interrupted export")

    with pytest.raises(RuntimeError):
        mod.publish_package(dict(branch="same"), fail, alias)
    assert not alias.exists() and not (root / "samples").exists()
    assert len(list((root / ".staging").iterdir())) == 1
    export = mod.publish_package(dict(branch="same"), producer, alias)
    assert alias.exists() and not export["trainable"]
    assert len(list((root / ".staging").iterdir())) == 1


def test_admission_index_rejects_flags_without_native_validation(tmp_path, monkeypatch):
    from driversi.data.datasets import native_admission as branch_supervision

    sample = tmp_path / "sample.json"
    sample.write_text("{}")
    feature = tmp_path / "feature.json"
    feature.write_text('{"trainable":true}')

    def reject(path):
        raise ValueError("native admission missing")

    monkeypatch.setattr(branch_supervision, "validate_sample", reject)
    with pytest.raises(ValueError, match="native admission"):
        mod.index_admitted_package(sample, feature)


@pytest.mark.parametrize("admitted", [False, True])
@pytest.mark.parametrize("canonical", [False, True])
def test_feature_worker_registers_only_admitted_canonical_package(
    tmp_path, monkeypatch, admitted, canonical
):
    import sys

    from driversi.data.catalog import derived_features
    from driversi.rsi.experiments import reactive_postprocess

    export = tmp_path / "export.json"
    export.write_text(json.dumps(dict(package_id="canonical") if canonical else {}))
    job = tmp_path / "job.json"
    job.write_text(
        json.dumps(
            dict(
                action="features",
                arguments=dict(
                    export_path=str(export),
                    database=str(tmp_path / "generated.sqlite"),
                    source_catalog=str(tmp_path / "source.sqlite"),
                    profile_id="test-profile",
                    sample_report=str(tmp_path / "sample.json"),
                    output=str(tmp_path / "features"),
                ),
            )
        )
    )
    monkeypatch.setattr(sys, "argv", ["worker", "--job", str(job)])
    monkeypatch.setattr(derived_features, "encode", lambda **kw: dict(trainable=admitted))
    calls = []

    def index(*args):
        calls.append(args)
        return dict(path="index", sha256="sealed")

    monkeypatch.setattr(mod, "index_admitted_package", index)
    reactive_postprocess.main()
    result = json.loads((tmp_path / "result.json").read_text())
    assert bool(calls) == (admitted and canonical)
    assert ("dataset_reference" in result) == (admitted and canonical)
    assert ("scene_classification" in result) == (admitted and canonical)


def test_admitted_receipts_are_idempotent_and_versioned_per_experiment(tmp_path, monkeypatch):
    from driversi.data.datasets import native_admission as branch_supervision

    root = tmp_path / "dataset"
    monkeypatch.setattr(mod, "dataset_root", lambda: root)
    export = mod.publish_package(dict(branch="same"), producer, tmp_path / "exp-a/native")
    export_path = Path(export["storage_root"]) / "export.json"
    monkeypatch.setattr(
        branch_supervision, "validate_sample", lambda p: json.loads(Path(p).read_text())
    )
    refs = []
    for name in ("exp-a", "exp-b"):
        folder = tmp_path / name
        folder.mkdir(exist_ok=True)
        sample = folder / "sample.json"
        sample.write_text(json.dumps(dict(sample_id="sample", files=export["files"])))
        feature = folder / "feature.json"
        feature.write_text(
            json.dumps(
                dict(
                    trainable=True,
                    parent_vectors_copied=False,
                    admission=dict(path=str(sample), sha256=digest(sample)),
                    export=str(export_path),
                    export_sha256=digest(export_path),
                    images=[dict(path=p, sha256=h) for p, h in export["files"].items()],
                )
            )
        )
        one = mod.index_admitted_package(sample, feature)
        assert mod.index_admitted_package(sample, feature) == one
        refs.append(one)
    assert refs[0]["package_id"] == refs[1]["package_id"]
    assert refs[0]["path"] != refs[1]["path"]
    assert all(digest(x["path"]) == x["sha256"] for x in refs)
    assert len(list((root / "admitted" / export["package_id"]).glob("*.json"))) == 2
    assert len(list((root / "samples").iterdir())) == 1


@pytest.mark.parametrize("contract",[None,"history_rgb_future_states_v1","anchor_rgb_history_future_states_v1"])
def test_history_layout_requires_explicit_contract(tmp_path,monkeypatch,contract):
    from driversi.data.datasets.synthetic_storage import HISTORY_ONLY_LAYOUT, ANCHOR_ONLY_LAYOUT
    layout=ANCHOR_ONLY_LAYOUT if contract == "anchor_rgb_history_future_states_v1" else HISTORY_ONLY_LAYOUT
    monkeypatch.setattr(mod,"dataset_root",lambda:tmp_path/"dataset")
    def history_producer(staging):
        value=producer(staging)
        value["storage_layout"]=layout
        if contract:value["image_contract"]=contract
        (staging/"export.json").write_text(json.dumps(value))
        return value
    if contract is None:
        with pytest.raises(ValueError,match="image contract"):
            mod.publish_package(dict(contract=contract),history_producer,tmp_path/"exp/native")
    else:
        result=mod.publish_package(dict(contract=contract),history_producer,tmp_path/"exp/native")
        assert result["storage_layout"]==layout
        assert result["image_contract"]==contract
