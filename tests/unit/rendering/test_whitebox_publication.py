import json

import pytest

import driversi.visualization.publication.whitebox as publisher
from driversi.rendering.whitebox.exporter import dump, sha256


@pytest.fixture
def bundle(tmp_path):
    folder = tmp_path / "input"
    folder.mkdir()
    frames = [
        dict(
            time_seconds=i,
            timestamp_us=1000000 + i * 1000000,
            objects=[
                dict(
                    actor_id="ego",
                    kind="ego",
                    pose_xyz_yaw=[i, 0, 0, 0],
                    dimensions_lwh=[4, 2, 1.5],
                )
            ],
        )
        for i in range(2)
    ]
    base = dict(
        schema_version=1,
        trainable=False,
        origin_xyz=[0, 0, 0],
        source_log="log",
        source_split="train",
        scenario_id="anchor",
        coordinate_frame="local_metric_map_axes",
        frames=frames,
    )
    source = dict(
        base,
        role="source_log",
        provenance=dict(
            family="navsim",
            anchor_timestamp_us=1000000,
            pose_scope="test",
            path="/private/source.pkl",
        ),
    )
    branch = dict(
        base,
        role="executed_branch",
        provenance=dict(
            anchor_timestamp_us=1000000,
            branch_id="hash",
            scene="/private/scene.npz",
            rollout="/private/run.npz",
            controller=dict(secret_setting="must not publish"),
        ),
    )
    map_data = dict(
        features=[], source=dict(path="/private/map.gpkg", version="1", geometry_model="planar")
    )
    for filename, value in zip(publisher.FILES, (source, branch, map_data), strict=True):
        dump(folder / filename, value)
    dump(
        folder / "manifest.json",
        dict(schema_version=1, counts={}, sha256={f: sha256(folder / f) for f in publisher.FILES}),
    )
    (folder / "private.txt").write_text("must not copy")
    return folder


def test_publication_is_allowlisted_and_redacted(bundle, tmp_path):
    target = tmp_path / "site"
    publisher.publish(bundle, target, "test-case", "test")
    case = target / "test-case"
    assert {p.name for p in case.iterdir()} == set(publisher.FILES) | {
        "manifest.json",
        "viewer.html",
        "viewer.js",
        "geometry.js",
    }
    for name in publisher.FILES:
        assert "/private/" not in (case / name).read_text()
        assert "secret_setting" not in (case / name).read_text()
    manifest = json.loads((case / "manifest.json").read_text())
    for name in publisher.FILES:
        assert sha256(case / name) == manifest["sha256"][name]


def test_modified_bundle_cannot_publish(bundle, tmp_path):
    with (bundle / "source-structure.json").open("a") as f:
        f.write(" ")
    with pytest.raises(ValueError, match="hash"):
        publisher.publish(bundle, tmp_path / "site", "case", "test")
    assert not (tmp_path / "site/case").exists()


def test_existing_release_is_immutable(bundle, tmp_path):
    target = tmp_path / "site"
    publisher.publish(bundle, target, "case", "test")
    before = (target / "case/manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        publisher.publish(bundle, target, "case", "replace")
    assert (target / "case/manifest.json").read_bytes() == before


@pytest.mark.parametrize("name", ["../escape", "/absolute", "has space", "x?y"])
def test_unsafe_names_rejected(bundle, tmp_path, name):
    with pytest.raises(ValueError):
        publisher.publish(bundle, tmp_path / "site", name, "test")
