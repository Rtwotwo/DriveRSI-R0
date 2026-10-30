import json
from pathlib import Path

import numpy as np
import pytest

from driversi.agents.backends.native import digest
from driversi.data.catalog import synthetic_classification as mod


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    root = tmp_path / "library"
    monkeypatch.setattr(mod, "library_root", lambda: root)
    log = tmp_path / "log.pkl"
    log.write_bytes(b"test")
    image = tmp_path / "anchor.png"
    image.write_bytes(b"pixels")
    export = tmp_path / "export.json"
    export.write_text(
        json.dumps(
            dict(
                token="synthetic_fixture",
                log_path=str(log),
                parent=dict(source_log="source"),
                files={str(p): digest(p) for p in (log, image)},
            )
        )
    )
    queued = mod.enqueue(export)
    job = json.loads(Path(queued["job"]).read_text())
    request = dict(evidence=[dict(source_file=str(image), source_sha256=digest(image))])
    return root, export, job, request


def test_queue_idempotent_and_original_rejected(corpus):
    root, p, job, _ = corpus
    assert mod.enqueue(p)["id"] == job["id"]
    assert len(list((root / "jobs").iterdir())) == 1
    value = json.loads(p.read_text())
    value["token"] = "original_token"
    p.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Generated token"):
        mod.enqueue(p)


def test_changed_source_is_rejected(corpus):
    _, p, job, _ = corpus
    export = mod.read_export(job)
    Path(export["log_path"]).write_bytes(b"changed")
    with pytest.raises(ValueError):
        mod.read_export(job)
    p.write_text("{}")
    with pytest.raises(ValueError, match="export changed"):
        mod.read_export(job)


def test_quarantine_without_native_admission(corpus):
    _, p, job, request = corpus
    assert mod.admission(job, json.loads(p.read_text()), request) == dict(
        valid=False, status="not_supplied"
    )


def test_native_admission_bound_to_exact_pixels_and_log(corpus, monkeypatch):
    from driversi.data.datasets import native_admission

    root, p, job, request = corpus
    export = json.loads(p.read_text())
    import hashlib

    from driversi.data.datasets import synthetic_publish

    dataset = p.parent / "dataset"
    monkeypatch.setattr(synthetic_publish, "dataset_root", lambda: dataset)
    package_identity = dict(branch="fixture")
    package = hashlib.sha256(
        json.dumps(package_identity, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    export.update(package_id=package, package_identity=package_identity)
    p = dataset / "samples" / package / "export.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps(export))
    job = json.loads(Path(mod.enqueue(p)["job"]).read_text())
    sample = p.parent / "sample.json"
    sample.write_text("{}")
    mod.enqueue(p, sample)
    valid = dict(sample_id=export["token"], source_log="source", files=export["files"])
    monkeypatch.setattr(native_admission, "validate_sample", lambda _: valid)
    assert mod.admission(job, export, request)["valid"]
    valid["files"] = dict(valid["files"])
    valid["files"][export["log_path"]] = "wrong"
    assert not mod.admission(job, export, request)["valid"]
    sample.write_text('{"changed":true}')
    assert not mod.admission(job, export, request)["valid"]


def test_topology_unknown_does_not_inherit_parent(corpus, tmp_path):
    _, _, job, _ = corpus
    member = dict(
        features={"anchor_speed_kmh": 12},
        map_query=dict(token=job["token"], map="missing", xy=[100, 200]),
    )
    m = mod.topology(member, tmp_path, tmp_path)
    assert m["map_labels"] == [] and m["lane_association"] == "unknown"
    assert m["features"] == member["features"]


def test_publish_replaces_same_scene_without_merging_originals(corpus):
    root, _, job, _ = corpus
    original = root.parent / "navtrain.jsonl"
    original.write_text("untouched")
    value = dict(
        id=job["id"],
        split="synthetic_navtrain",
        cluster=dict(leaf=3),
        native_admission=dict(valid=False),
    )
    mod.publish_record(value)
    mod.publish_record(value)
    assert len((root / "members.jsonl").read_text().splitlines()) == 1
    assert json.loads((root / "manifest.json").read_text())["qualified_at_last_check"] == 0
    assert original.read_text() == "untouched"


def test_frozen_hierarchy_assigns_new_scene_without_refit(tmp_path):
    import joblib

    from driversi.data.catalog.scene_hierarchy import fit_hierarchy, fit_representation, transform

    config = dict(
        max_text_features=200,
        text_dimensions=3,
        weights=dict(text=0.85, facts=0.15),
        seed=42,
        target_leaf_size=8,
        min_child_size=2,
        max_depth=5,
        split_fit_samples=30,
        image_vectors_used=False,
        cuts=dict(coarse=1, medium=2, fine=3),
    )
    captions = [
        dict(
            attributes=[dict(description=f"道路车辆 行人横道 第{i % 6}类", certainty="visible")],
            retrieval_tags=[],
        )
        for i in range(30)
    ]
    members = [dict(features=dict(anchor_speed_kmh=i, nav_command_slot=i % 3)) for i in range(30)]
    corpus = dict(members=members, captions=captions)
    model = fit_representation(corpus, config)
    x, _ = transform(corpus, model)
    nodes, _, centers, splits = fit_hierarchy(x, config)
    joblib.dump(model, tmp_path / "representation.joblib")
    (tmp_path / "hierarchy.json").write_text(
        json.dumps(dict(fit_split="navtrain", config=config, nodes=nodes))
    )
    np.savez(
        tmp_path / "hierarchy.npz",
        centers=centers,
        split_node_ids=list(splits),
        split_centers=list(splits.values()),
    )
    (tmp_path / "clusters.json").write_text(
        json.dumps(dict(nodes=[dict(n, label="fixture", radius_p99=0.5) for n in nodes]))
    )
    before = {p.name: digest(p) for p in tmp_path.iterdir()}
    one = mod.assign(captions[4], members[4], tmp_path)
    two = mod.assign(captions[4], members[4], tmp_path)
    assert one == two and one["refitted"] is False and one["image_vectors_used"] is False
    assert before == {p.name: digest(p) for p in tmp_path.iterdir()}


def test_request_resume_does_not_repeat_paid_response(corpus, monkeypatch, tmp_path):
    from driversi.data.catalog import bailian_caption as client

    root, export, job, _ = corpus
    request = dict(
        token=job["token"],
        scope="current_anchor_only",
        input_identity="fixture",
        prompt="Describe anchor",
        facts=dict(anchor_speed_kmh=12, navigation_command="GO FORWARD"),
        evidence=[dict(id="CAM_F0_anchor", file="unused", sha256="unused")],
    )
    member = dict(features={}, map_query={})
    monkeypatch.setattr(
        mod, "prepare_anchor", lambda *args: (json.loads(export.read_text()), request, member)
    )
    monkeypatch.setattr(mod, "topology", lambda member, *args: member)
    # A real receipt is retained; only scene transformation is replaced for this transport test.
    monkeypatch.setattr(mod, "assign", lambda *args: dict(leaf=1, out_of_train_radius=False))
    monkeypatch.setattr(mod, "publish_record", lambda value: None)
    monkeypatch.setattr(client, "payload", lambda *args: dict(model="test", content="same pixels"))
    monkeypatch.setattr(client, "credential", lambda *_: "unit-test-placeholder")
    cfg = dict(
        base_url="https://example.invalid/compatible-mode/v1",
        models=["test"],
        concurrency=1,
        requests_per_minute=60,
        max_requests=1,
        max_attempts=1,
        timeout_seconds=30,
        max_tokens=1800,
        image_max_side=768,
        jpeg_quality=80,
        enable_thinking=False,
    )
    private_cfg = dict(cfg)
    monkeypatch.setattr(client, "private_config", lambda *_: dict(private_cfg))
    text = "道路前方可见车辆及行人，路侧建筑和道路边缘可以辨认。" * 5
    desc = dict(
        scene_description_zh=text,
        attributes=[
            dict(
                dimension="road_layout",
                description="道路边缘",
                evidence_ids=["CAM_F0_anchor"],
                certainty="visible",
            )
            for _ in range(4)
        ],
        retrieval_tags=["道路", "车辆", "行人"],
        uncertainties=[],
    )
    calls = []

    def transport(*args):
        calls.append(1)
        return dict(
            choices=[dict(finish_reason="stop", message=dict(content=json.dumps(desc)))],
            usage=dict(prompt_tokens=100, completion_tokens=100),
        )

    cfg.update(
        provider_config_name="test.json",
        credential_name="test.env",
        model="test",
        max_output_tokens=1800,
        cost_reserved_input_tokens=16384,
        busy_input_cny_per_million=2,
        busy_output_cny_per_million=8,
        cost_cap_cny=30,
        reserved_input_tokens=2500,
        tokens_per_minute=1000000,
        maps_root=str(tmp_path),
        hierarchy="unused",
    )
    directory = root / "jobs" / job["id"]
    (directory / "caption-job.json").write_text(json.dumps(request))
    (directory / "topology.json").write_text("{}")
    first = mod.process(directory / "job.json", cfg, transport=transport)
    second = mod.process(directory / "job.json", cfg, transport=transport)
    assert first["status"] == "classified" and second["requests_reused"] == 1 and len(calls) == 1
    assert first["partition"] == "quarantine"
    with pytest.raises(ValueError, match="policy changed"):
        mod.process(directory / "job.json", dict(cfg, model="another"), transport=transport)
    assert len(calls) == 1


def test_harness_provider_catalog_registered():
    from driversi.rsi.services.catalog import describe

    assert any(r["name"] == "synthetic_scene_classification" for r in describe("data"))


def test_unknown_inflight_is_never_resent(tmp_path, monkeypatch):
    from driversi.data.catalog import bailian_caption as client
    from driversi.data.catalog.store import identity

    cfg = dict(
        base_url="https://example.invalid/compatible-mode/v1",
        models=["test"],
        concurrency=1,
        requests_per_minute=60,
        max_requests=1,
        max_attempts=1,
        timeout_seconds=30,
        max_tokens=1800,
        image_max_side=768,
        jpeg_quality=80,
        enable_thinking=False,
    )
    body = dict(model="test", content="unchanged")
    monkeypatch.setattr(client, "payload", lambda *args: body)
    signature = identity(dict(base_url=cfg["base_url"], payload=body))
    job = dict(token="synthetic_fixture", scope="current_anchor_only", input_identity="one")
    attempts = tmp_path / "attempts"
    attempts.mkdir()
    (attempts / (signature + "-0.json")).write_text(
        json.dumps(dict(status="inflight", request_identity=signature))
    )
    report = client.run(
        [job],
        cfg,
        tmp_path,
        key="unit-test-placeholder",
        transport=lambda *args: pytest.fail("Unknown request must not repeat"),
    )
    assert not report["complete"]
    assert report["rows"][0]["status"] == "uncertain_previous_attempt_requires_review"


def test_noncanonical_benchmark_never_qualified(corpus, monkeypatch):
    from driversi.data.datasets import native_admission

    _, p, job, request = corpus
    export = json.loads(p.read_text())
    sample = p.parent / "benchmark-sample.json"
    sample.write_text("{}")
    mod.enqueue(p, sample)
    monkeypatch.setattr(
        native_admission,
        "validate_sample",
        lambda _: dict(
            sample_id=export["token"], source_log="source", files=export["files"], trainable=True
        ),
    )
    assert mod.admission(job, export, request)["valid"] is False


def test_rate_backoff_and_auth_state_survive_next_scene(tmp_path, monkeypatch):
    from driversi.data.catalog import bailian_caption as client

    now = [100.0]
    monkeypatch.setattr(mod.time, "time", lambda: now[0])

    def sleep(delay):
        now[0] += delay

    monkeypatch.setattr(mod.time, "sleep", sleep)
    cfg = dict(
        requests_per_minute=60,
        reserved_input_tokens=1000,
        max_output_tokens=1000,
        tokens_per_minute=1000000,
    )

    def rejected(*args):
        raise client.APIError(429, True)

    with pytest.raises(client.APIError):
        mod.shared_transport(tmp_path, cfg, rejected)({}, "unused", {})
    times = []
    mod.shared_transport(tmp_path, cfg, lambda *args: times.append(now[0]))({}, "unused", {})
    assert times == [160.0]

    def auth(*args):
        raise client.APIError(403)

    with pytest.raises(client.APIError):
        mod.shared_transport(tmp_path, cfg, auth)({}, "unused", {})
    with pytest.raises(client.NotSent):
        mod.shared_transport(tmp_path, cfg, lambda *args: pytest.fail("No auth retry"))(
            {}, "unused", {}
        )
