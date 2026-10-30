import copy
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from driversi.agents.backends.native import digest
from driversi.data.datasets import simscale_collection as mod


def training_fixture(tmp_path, monkeypatch):
    from driversi.data.datasets import synthetic_publish
    from driversi.rsi.correction import epdms_admission
    monkeypatch.setattr(epdms_admission, 'validate_sample', lambda _: dict(score=0.99))
    root, campaign, ref, frames, log = fixture_package(tmp_path, monkeypatch)
    monkeypatch.setattr(synthetic_publish, 'dataset_root', lambda: root/'synthetic_navsim')
    old = json.loads(ref.read_text())
    export_path = Path(old['export'])
    export = json.loads(export_path.read_text())
    export['package_id'] = 'package'
    export_path.write_text(json.dumps(export))
    sample_path = Path(old['sample_report'])
    row = {k:export[k] for k in ['token','anchor_index','log_path','sensor_root','image_files']}
    row['sample_id'] = export['token']
    sample_path.write_text(json.dumps(dict(sample_id=export['token'], row=row, files=export['files'])))
    (root/'postprocessing.json').write_text(json.dumps(dict(schema_version=1, mode='training_only',
        campaign=str(campaign), features='deferred', scene_caption='deferred',
        classification='deferred', training_admission='unchanged')))
    return root, sample_path, export_path


def test_training_only_publishes_without_retrieval_and_preserves_native(tmp_path, monkeypatch):
    from driversi.data.datasets.training_publish import index_training_package
    root, sample, export = training_fixture(tmp_path, monkeypatch)
    before = {p:p.read_bytes() for p in [sample,export]}
    result = index_training_package(sample, export)
    ref = json.loads(Path(result['path']).read_text())
    assert ref['trainable'] and not ref['indexable'] and 'feature_report' not in ref
    assert result == index_training_package(sample, export)
    assert len(mod.validate_members(root)) == 1
    assert all(p.read_bytes() == raw for p,raw in before.items())
    assert result['simscale_collection']['image_counts'] == dict(history=16,future=40)


@pytest.mark.parametrize('fault',['foreign_log','missing_future_seal','wrong_token','corrupt_pixels','native_reject'])
def test_training_only_never_skips_native_or_branch_binding(tmp_path, monkeypatch, fault):
    from driversi.data.datasets import native_admission
    from driversi.data.datasets.training_publish import index_training_package
    root, sample, export = training_fixture(tmp_path, monkeypatch)
    value = json.loads(sample.read_text())
    if fault == 'foreign_log':
        p=sample.parent/'foreign.pkl';p.write_bytes(b'other');value['row']['log_path']=str(p)
    elif fault == 'missing_future_seal':
        value['files'].pop(next(p for p in value['files'] if '/future/' in p))
    elif fault == 'wrong_token':value['row']['token']='synthetic_other'
    elif fault == 'corrupt_pixels':Path(value['row']['image_files'][0]).write_bytes(b'bad')
    else:
        def reject(_):raise ValueError('native rejected')
        monkeypatch.setattr(native_admission,'validate_sample',reject)
    sample.write_text(json.dumps(value))
    with pytest.raises(ValueError):index_training_package(sample,export)
    assert json.loads((root/'manifests/synthetic_driverl_v1/index.json').read_text())['count']==0


def fixture_package(tmp_path, monkeypatch, future=10):
    from driversi.data.datasets import native_admission
    monkeypatch.setattr(mod, "validate_epdms", lambda _: dict(score=0.99))
    monkeypatch.setattr(mod, "project_root", lambda: tmp_path)
    root, campaign = tmp_path / "dataset", tmp_path / "exp/run"
    mod.initialize(root, campaign)
    package = root / "synthetic_navsim/samples/package"
    package.mkdir(parents=True)
    frames, files, images = [], {}, []
    for i in range(4 + future):
        role = "history" if i < 4 else "future"
        cams = {}
        for camera in ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0"):
            rel = Path(role) / "branch" / camera / f"{i}.png"
            p = package / "sensors" / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(f"fixture-pixel-{i}-{camera}".encode())
            files[str(p)] = digest(p)
            if i == 3:
                images.append(str(p))  # Native ChainFlow exports anchor images here.
            cams[camera] = dict(data_path=rel.as_posix(), cam_intrinsic=np.eye(3))
        frames.append(dict(token=f"synthetic_{i}", log_name="branch", timestamp=i * 500000,
                           cams=cams, ego2global_translation=np.array([i, 0., 1.]),
                           anns=dict(gt_boxes=np.ones((1, 7))), driving_command=[0, 1, 0, 0]))
    log = package / "logs/branch.pkl"
    log.parent.mkdir()
    log.write_bytes(pickle.dumps(frames))
    files[str(log)] = digest(log)
    export = dict(token="synthetic_3", anchor_index=3, log_path=str(log),
                  sensor_root=str(package / "sensors"), image_files=images, files=files)
    ep = package / "export.json"
    ep.write_text(json.dumps(export))
    sample = campaign / "sample.json"
    sample.write_text(json.dumps(dict(sample_id="synthetic_3")))
    feature = campaign / "feature.json"
    feature.write_text("{}")
    ref = root / "synthetic_navsim/admitted/package/receipt.json"
    ref.parent.mkdir(parents=True)
    ref.write_text(json.dumps(dict(package_id="package", sample_id="synthetic_3",
        export=str(ep), export_sha256=digest(ep), sample_report=str(sample),
        sample_report_sha256=digest(sample), feature_report=str(feature),
        feature_report_sha256=digest(feature))))
    monkeypatch.setattr(native_admission, "validate_sample", lambda p: json.loads(Path(p).read_text()))
    return root, campaign, ref, frames, log


def test_clear_init_is_idempotent_and_never_erases(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "project_root", lambda: tmp_path)
    root, campaign = tmp_path / "dataset", tmp_path / "exp/run"
    old = root / "synthetic_navsim/samples/retained.json"
    old.parent.mkdir(parents=True)
    old.write_text("keep")
    with pytest.raises(ValueError, match="cleared"):
        mod.initialize(root, campaign)
    assert old.read_text() == "keep"
    old.unlink()
    first = mod.initialize(root, campaign)
    assert mod.initialize(root, campaign) == first
    with pytest.raises(ValueError, match="another campaign"):
        mod.initialize(root, tmp_path / "exp/other")


def test_simscale_loader_paths_and_native_metadata_parity(tmp_path, monkeypatch):
    root, campaign, ref, original, native_log = fixture_package(tmp_path, monkeypatch)
    source_bytes = native_log.read_bytes()
    result = mod.publish(ref, root)
    assert result["image_counts"] == dict(history=16, future=40)
    assert mod.publish(ref, root) == result
    assert native_log.read_bytes() == source_bytes
    frames = pickle.loads(Path(result["log_path"]).read_bytes())
    for i, (before, after) in enumerate(zip(original, frames)):
        expected = copy.deepcopy(before)
        for camera, info in expected["cams"].items():
            info["data_path"] = "/".join(info["data_path"].split("/")[1:])
            pixel = Path(result["sensor_root"]) / info["data_path"]
            assert pixel.is_symlink() and pixel.is_file()
            part = "sensor_blobs_hist" if i < 4 else "sensor_blobs_fut"
            assert (root / part / "synthetic_driverl_v1" / info["data_path"]).resolve() == pixel.resolve()
        assert pickle.dumps(expected) == pickle.dumps(after)
    index = json.loads((root / "manifests/synthetic_driverl_v1/index.json").read_text())
    assert index["count"] == 1 and index["samples"] == ["package"]
    assert len(mod.validate_members(root)) == 1
    pixel = Path(result['sensor_root']) / frames[0]['cams']['CAM_F0']['data_path']
    pixel.unlink()
    pixel.symlink_to(tmp_path / 'unrelated.png')
    with pytest.raises((ValueError, FileNotFoundError)):
        mod.validate_members(root)


@pytest.mark.parametrize("fault", ["short", "corrupt", "rejected", "foreign"])
def test_no_collection_index_for_invalid_samples(tmp_path, monkeypatch, fault):
    from driversi.data.datasets import native_admission
    root, campaign, ref, frames, log = fixture_package(tmp_path, monkeypatch, 8 if fault == "short" else 10)
    if fault == "corrupt":
        log.write_bytes(b"changed")
    elif fault == "rejected":
        def reject(p):
            raise ValueError("native admission rejected")
        monkeypatch.setattr(native_admission, "validate_sample", reject)
    elif fault == "foreign":
        data = json.loads(ref.read_text())
        other = tmp_path / "exp/other/sample.json"
        other.parent.mkdir(parents=True)
        other.write_text('{"sample_id":"synthetic_3"}')
        data.update(sample_report=str(other), sample_report_sha256=digest(other))
        ref.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        mod.publish(ref, root)
    assert json.loads((root / "manifests/synthetic_driverl_v1/index.json").read_text())["count"] == 0


def test_active_campaign_blocks_unrelated_native_export(tmp_path, monkeypatch):
    from driversi.data.datasets import synthetic_publish
    monkeypatch.setattr(mod, "project_root", lambda: tmp_path)
    monkeypatch.setattr(synthetic_publish, "dataset_root", lambda: tmp_path / "dataset/synthetic_navsim")
    mod.initialize(tmp_path / "dataset", tmp_path / "exp/active")
    with pytest.raises(ValueError, match="active campaign"):
        synthetic_publish.publish_package({}, lambda _: pytest.fail("must not execute"),
                                          tmp_path / "exp/other/native")


def test_active_campaign_resume_preserves_canonical_alias(tmp_path, monkeypatch):
    from driversi.data.catalog import synthetic_classification
    from driversi.data.datasets import synthetic_publish
    monkeypatch.setattr(mod, 'project_root', lambda: tmp_path)
    monkeypatch.setattr(synthetic_publish, 'dataset_root', lambda: tmp_path / 'dataset/synthetic_navsim')
    monkeypatch.setattr(synthetic_classification, 'enqueue', lambda *a: {})
    campaign = tmp_path / 'exp/active'
    mod.initialize(tmp_path / 'dataset', campaign)
    def producer(staging):
        staging.mkdir(parents=True)
        pixel = staging / 'fixture'
        pixel.write_bytes(b'fixture')
        return dict(trainable=False, storage_layout='logs_sensors_history_future_v2',
                    files={str(pixel): digest(pixel)})
    alias = campaign / 'branch/native'
    first = synthetic_publish.publish_package(dict(branch='fixture'), producer, alias)
    assert alias.is_symlink()
    assert synthetic_publish.publish_package(dict(branch='fixture'),
        lambda _: pytest.fail('Repeated export'), alias) == first


def test_pdms_transition_preserves_prior_collection(tmp_path, monkeypatch):
    root, campaign, ref, original, native_log = fixture_package(tmp_path, monkeypatch)
    mod.publish(ref, root)
    old_contract = (root / "collection.json").read_bytes()
    old_files = {p: p.read_bytes() for p in (root / "manifests/synthetic_driverl_v1").glob("*.json")}
    new = tmp_path / "exp/pdms"
    value = mod.activate_pdms(root, new)
    assert value["trajectory_quality"] is None
    assert value["trajectory_admission"]["operator"] == ">"
    assert mod.activate_pdms(root, new) == value
    assert all(p.read_bytes() == data for p, data in old_files.items())
    assert json.loads((root / "manifests/synthetic_driverl_v1/collection.json").read_text()) == json.loads(old_contract)
    assert mod.validate_members(root) == []
    with pytest.raises(ValueError, match="active campaign"):
        mod.require_campaign(root, campaign / "sample.json")
    assert mod.require_campaign(root, new / "sample.json") == value


@pytest.mark.parametrize("score,accepted", [(0.95,False),(0.949,False),(0.951,True)])
def test_pdms_boundary_rechecks_executed_evidence(tmp_path, score, accepted):
    evidence = dict(branch="teacher", completed_steps=50, failure=None,
                    producer_identity="native-controller", history_changed=True, restore_exact=True,
                    admission_profile="pdms_gt95_v1", score=dict(score=score, actual_world_seconds=5.0,
                    protocol="navsim-v1-native-math-executed-counterfactual-v1"))
    report = tmp_path / "branch.json"
    report.write_text(json.dumps(dict(trajectory_evidence=evidence)))
    proof = tmp_path / "admission.json"
    proof.write_text(json.dumps(dict(branch=dict(report=str(report)), files={str(report):digest(report)})))
    sample = dict(row=dict(admission=dict(path=str(proof),sha256=digest(proof))))
    if accepted:
        assert mod.validate_pdms(sample)["score"] == score
        report.write_text("{}")
        with pytest.raises(ValueError, match="evidence changed"):
            mod.validate_pdms(sample)
    else:
        with pytest.raises(ValueError):mod.validate_pdms(sample)


def test_canonical_index_rejects_pdms_before_commit(tmp_path, monkeypatch):
    from driversi.data.datasets import synthetic_publish
    root, campaign, ref, _, _ = fixture_package(tmp_path, monkeypatch)
    mod.activate_pdms(root, campaign)
    monkeypatch.setattr(synthetic_publish, "dataset_root", lambda: root / "synthetic_navsim")
    reference = json.loads(ref.read_text())
    sample = Path(reference["sample_report"])
    sample.write_text(json.dumps(dict(sample_id="synthetic_3",files={})))
    export_path = Path(reference["export"])
    export = json.loads(export_path.read_text());export["package_id"] = "package"
    export_path.write_text(json.dumps(export))
    feature = Path(reference["feature_report"])
    feature.write_text(json.dumps(dict(trainable=True,parent_vectors_copied=False,
        admission=dict(path=str(sample),sha256=digest(sample)),export=str(export_path),
        export_sha256=digest(export_path),images=[])))
    before = {p:p.read_bytes() for p in (root / "synthetic_navsim/admitted").rglob("*.json")}
    def reject(_):raise ValueError("strict PDMS rejected")
    monkeypatch.setattr(mod,"validate_pdms",reject)
    with pytest.raises(ValueError,match="strict PDMS rejected"):
        synthetic_publish.index_admitted_package(sample,feature)
    after = {p:p.read_bytes() for p in (root / "synthetic_navsim/admitted").rglob("*.json")}
    assert before == after
    assert json.loads((root / "manifests/synthetic_driverl_pdms_v2/index.json").read_text())["count"] == 0


def test_history_only_collection_preserves_legacy_and_future_numerics(tmp_path,monkeypatch):
    root,campaign,ref,frames,log=fixture_package(tmp_path,monkeypatch)
    oldpublished=mod.publish(ref,root)
    mod.activate_pdms(root,campaign)
    previous=(root/"collection.json").read_bytes()
    preserved=Path(oldpublished["log_path"]).read_bytes()
    nextcampaign=tmp_path/"exp/history-only"
    value=mod.activate_history_only(root,nextcampaign)
    assert mod.activate_history_only(root,nextcampaign)==value
    assert value["trajectory_admission"]["operator"] == ">"
    assert value["image_contract"] == "history_rgb_future_states_v1"
    assert Path(oldpublished["log_path"]).read_bytes()==preserved
    assert json.loads((root/"manifests/synthetic_driverl_pdms_v2/collection.json").read_text())==json.loads(previous)
    assert not (root/"sensor_blobs_fut"/value["split"]).exists()
    assert mod.validate_members(root)==[]
    assert mod.require_campaign(root,nextcampaign/"sample.json")==value
    with pytest.raises(ValueError,match="active campaign"):
        mod.require_campaign(root,campaign/"sample.json")


@pytest.mark.parametrize("anchor_only",[False,True])
@pytest.mark.parametrize("with_future_rgb",[False,True])
def test_history_only_collection_publishes_only_history_pixels(tmp_path,monkeypatch,with_future_rgb,anchor_only):
    root,campaign,ref,frames,log=fixture_package(tmp_path,monkeypatch)
    mod.activate_pdms(root,campaign)
    config=(mod.activate_anchor_only if anchor_only else mod.activate_history_only)(root,campaign)
    monkeypatch.setattr(mod,"validate_pdms",lambda _:dict(score=.99))
    reference=json.loads(ref.read_text())
    ep=Path(reference["export"]);export=json.loads(ep.read_text())
    export["image_contract"]=config["image_contract"]
    if not with_future_rgb:
        for i,f in enumerate(frames):
            if i == 3 or (not anchor_only and i < 4):continue
            for info in f["cams"].values():
                info.update({k:None for k in info})
        log.write_bytes(pickle.dumps(frames))
        export["files"][str(log)]=digest(log)
    ep.write_text(json.dumps(export));reference["export_sha256"]=digest(ep)
    ref.write_text(json.dumps(reference))
    if with_future_rgb:
        with pytest.raises(ValueError,match="Unrendered numerical"):
            mod.publish(ref,root)
        assert mod.validate_members(root)==[]
        return
    result=mod.publish(ref,root)
    assert result["image_counts"] == dict(history=4 if anchor_only else 16,future=0)
    assert result["allowed_anchor_indices"] == [3]
    actual=pickle.loads(Path(result["log_path"]).read_bytes())
    assert len(actual)==14 and all(pickle.dumps(a)==pickle.dumps(b) for a,b in zip(actual[4:],frames[4:]))
    assert not (root/"sensor_blobs_fut"/config["split"]).exists()
    assert len(mod.validate_members(root))==1


def test_anchor_collection_transition_preserves_history_contract(tmp_path,monkeypatch):
    monkeypatch.setattr(mod,"project_root",lambda:tmp_path)
    root=tmp_path/"dataset";campaign=tmp_path/"exp/history"
    old=mod.activate_history_only(root,campaign)
    new=mod.activate_anchor_only(root,tmp_path/"exp/anchor")
    assert new["schema_version"]==4 and new["split"]=="synthetic_driverl_anchor_v4"
    assert new["image_contract"]=="anchor_rgb_history_future_states_v1"
    assert json.loads((root/"manifests"/old["split"]/"collection.json").read_text())==old
    assert mod.activate_anchor_only(root,tmp_path/"exp/anchor")==new
    assert mod.validate_members(root)==[]
