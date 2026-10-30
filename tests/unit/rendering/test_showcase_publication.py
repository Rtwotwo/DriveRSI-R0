"""Publication checks exercise isolated temporary galleries, never formal assets."""
import copy
import fcntl
import json

import pytest

import driversi.visualization.publication.gallery as publisher


def dump(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def add_case(root, cid):
    card = {
        "id": cid, "source_log": "log-" + cid, "scenario_id": "scenario-" + cid,
        "trainable": False, "learned_self_play": False, "frame_count": 2,
        "data": "data/" + cid + ".json", "custom_case": "preserved",
    }
    media = "media/" + cid + ".png"
    (root / "viz" / media).write_bytes(b"existing asset")
    data = {
        "id": cid, "metadata": copy.deepcopy(card),
        "frames": [
            {"tick": tick, "time_seconds": tick / 10, "actors": [
                {"id": "ego", "x": tick, "y": 0, "yaw": 0, "length": 4, "width": 2}],
             "cameras": [{"id": "front", "image": media, "source": media}]}
            for tick in (0, 1)
        ],
    }
    dump(root / "viz" / card["data"], data)
    return card


@pytest.fixture
def gallery(tmp_path):
    for directory in ("viz/data", "viz/media", "src/rendering/inspection"):
        (tmp_path / directory).mkdir(parents=True)
    (tmp_path / "src/rendering/inspection/showcase.html").write_text("new template")
    cards = [add_case(tmp_path, "first"), add_case(tmp_path, "multiview"),
             add_case(tmp_path, "third")]
    dump(tmp_path / "viz/data/cases.json", {
        "schema_version": 1, "cases": cards, "custom": {"retain": True},
        "presentation_scope": "specific retained scope",
    })
    (tmp_path / "viz/index.html").write_text("old template")
    return tmp_path


def read_catalog(root):
    return json.loads((root / "viz/data/cases.json").read_text())


def test_optional_update_preserves_order_extensions_and_media(gallery):
    catalog = read_catalog(gallery)
    update = {**catalog["cases"][1], "title": "updated"}
    update.pop("custom_case")
    dump(gallery / "viz/data/multiview-card.json", update)
    originals = {p: p.read_bytes() for base in ("viz/media", "viz/data")
                 for p in (gallery / base).glob("*") if p.name != "cases.json"}
    receipt = publisher.publish(gallery)
    result = read_catalog(gallery)
    assert [c["id"] for c in result["cases"]] == ["first", "multiview", "third"]
    assert result["cases"][1]["custom_case"] == "preserved"
    assert result["cases"][1]["title"] == "updated"
    assert result["presentation_scope"] == catalog["presentation_scope"]
    assert result["custom"] == {"retain": True}
    assert all(p.read_bytes() == raw for p, raw in originals.items())
    assert receipt["case_count"] == 3
    assert (gallery / "viz/index.html").read_text() == "new template"


def test_missing_optional_multiview_is_valid(gallery):
    before = read_catalog(gallery)
    assert publisher.publish(gallery)["case_count"] == 3
    assert read_catalog(gallery) == before


def test_new_optional_case_appends_without_reordering(gallery):
    extra = add_case(gallery, "new-case")
    dump(gallery / "viz/data/multiview-card.json", extra)
    publisher.publish(gallery)
    assert [c["id"] for c in read_catalog(gallery)["cases"]] == [
        "first", "multiview", "third", "new-case"]


@pytest.mark.parametrize("field", ["source_log", "scenario_id"])
def test_optional_origin_conflict_fails_before_write(gallery, field):
    before = (gallery / "viz/data/cases.json").read_bytes()
    update = read_catalog(gallery)["cases"][1]
    update[field] = "different"
    dump(gallery / "viz/data/multiview-card.json", update)
    with pytest.raises(ValueError, match="origin conflict"):
        publisher.publish(gallery)
    assert (gallery / "viz/data/cases.json").read_bytes() == before
    assert (gallery / "viz/index.html").read_text() == "old template"


def test_failed_replace_keeps_previous_file_and_removes_temporary(gallery, monkeypatch):
    before = (gallery / "viz/data/cases.json").read_bytes()

    def fail_replace(*args):
        raise OSError("injected replace failure")

    monkeypatch.setattr(publisher.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected"):
        publisher.publish(gallery)
    assert (gallery / "viz/data/cases.json").read_bytes() == before
    assert not list((gallery / "viz/data").glob(".*.tmp"))


def test_failure_after_catalog_never_leaves_partial_file(gallery, monkeypatch):
    real_replace = publisher.os.replace

    def fail_index(source, target):
        if target.name == "index.html":
            raise OSError("injected index failure")
        real_replace(source, target)

    monkeypatch.setattr(publisher.os, "replace", fail_index)
    with pytest.raises(OSError, match="index failure"):
        publisher.publish(gallery)
    assert read_catalog(gallery)["schema_version"] == 1
    assert (gallery / "viz/index.html").read_text() == "old template"
    assert not list((gallery / "viz").glob(".*.tmp"))


@pytest.mark.parametrize("path", ["../outside.json", "/tmp/file.json", "data/../first.json",
                                  "data//first.json", "data/.hidden.json",
                                  "data/back\\slash.json", "data/name.json?x=1",
                                  "data/%2e%2e.json"])
def test_invalid_case_paths_fail(gallery, path):
    catalog = read_catalog(gallery)
    catalog["cases"][0]["data"] = path
    dump(gallery / "viz/data/cases.json", catalog)
    with pytest.raises(ValueError):
        publisher.publish(gallery)
    assert (gallery / "viz/index.html").read_text() == "old template"


@pytest.mark.parametrize("name", ["data/first.json", "media/first.png", "index.html",
                                  "data/publication-checks.json", ".export.lock"])
def test_symlink_file_rejected_without_touching_target(gallery, name):
    target = gallery / "outside"
    target.write_text("do not touch")
    link = gallery / "viz" / name
    link.unlink(missing_ok=True)
    link.symlink_to(target)
    with pytest.raises(ValueError, match="Symlink"):
        publisher.publish(gallery)
    assert target.read_text() == "do not touch"


def test_symlink_directory_rejected(gallery):
    media = gallery / "viz/media"
    media.rename(gallery / "elsewhere")
    media.symlink_to(gallery / "elsewhere", target_is_directory=True)
    with pytest.raises(ValueError, match="Symlink"):
        publisher.publish(gallery)


def test_shared_export_lock_refuses_parallel_publication(gallery):
    with (gallery / "viz/.export.lock").open("w") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            publisher.publish(gallery)
    assert (gallery / "viz/index.html").read_text() == "old template"


@pytest.mark.parametrize("mutation", ["schema", "schema_bool", "cases_type", "duplicate_id",
                                      "shared_data", "qualification", "metadata_origin",
                                      "timestamp", "duplicate_actor", "virtual_source"])
def test_malformed_schema_or_evidence_rejected(gallery, mutation):
    catalog = read_catalog(gallery)
    data_file = gallery / "viz/data/first.json"
    data = json.loads(data_file.read_text())
    if mutation == "schema":
        catalog["schema_version"] = 2
    elif mutation == "schema_bool":
        catalog["schema_version"] = True
    elif mutation == "cases_type":
        catalog["cases"] = {}
    elif mutation == "duplicate_id":
        catalog["cases"].append(catalog["cases"][0])
    elif mutation == "shared_data":
        catalog["cases"][1]["data"] = catalog["cases"][0]["data"]
    elif mutation == "qualification":
        catalog["cases"][0]["trainable"] = True
    elif mutation == "metadata_origin":
        data["metadata"]["source_log"] = "wrong"
    elif mutation == "timestamp":
        data["frames"][1]["time_seconds"] = 0
    elif mutation == "duplicate_actor":
        data["frames"][0]["actors"] *= 2
    elif mutation == "virtual_source":
        data["frames"][0]["cameras"][0]["virtual"] = True
    dump(gallery / "viz/data/cases.json", catalog)
    dump(data_file, data)
    with pytest.raises(ValueError):
        publisher.publish(gallery)
    assert (gallery / "viz/index.html").read_text() == "old template"


@pytest.mark.parametrize("raw", ['{"schema_version":1,"schema_version":1,"cases":[]}',
                                  '{"schema_version":1,"cases":[],"value":NaN}'])
def test_invalid_json_rejected(gallery, raw):
    (gallery / "viz/data/cases.json").write_text(raw)
    with pytest.raises(ValueError):
        publisher.publish(gallery)
