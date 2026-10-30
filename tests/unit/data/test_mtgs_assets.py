"""Asset trust and extraction must not be confused with visual simulation readiness."""

import hashlib
import io
import pickle
import tarfile

import pytest

from driversi.data.assets.mtgs_assets import (
    BLOCK_ID,
    REVISION,
    ExtractionLimits,
    default_manifest,
    inspect_assets,
    inspect_block_metadata,
    safe_extract_tar,
    verify_asset,
)


def _archive(tmp_path, members):
    path = tmp_path / "asset.tar"
    with tarfile.open(path, "w") as bundle:
        for name, data in members:
            member = tarfile.TarInfo(name)
            if isinstance(data, bytes):
                member.size = len(data)
                bundle.addfile(member, io.BytesIO(data))
            else:
                member.type = tarfile.SYMTYPE
                member.linkname = data
                bundle.addfile(member)
    raw = path.read_bytes()
    return path, {"expected_size": len(raw), "expected_sha256": hashlib.sha256(raw).hexdigest()}


def test_manifest_is_fixed_revision_and_status_is_not_simulator_readiness(tmp_path):
    manifest = default_manifest()
    assert len(manifest["assets"]) == 4
    assert manifest["revision"] == REVISION
    assert manifest["block_id"] == BLOCK_ID
    assert not manifest["simulator_ready"] and not manifest["trainable"]
    for asset in manifest["assets"]:
        assert f"/resolve/{REVISION}/" in asset["download_url"]
        assert asset["download_url"].startswith("https://hf-mirror.com/")
        assert len(asset["sha256"]) == 64 and asset["size_bytes"] > 0
    report = inspect_assets(tmp_path)
    assert not report["all_downloaded"] and not report["all_verified"]
    assert all(not item["status"]["downloaded"] for item in report["assets"])
    with pytest.raises(ValueError, match="transport"):
        default_manifest(transport="http://untrusted.example")


def test_same_size_corruption_does_not_pass_verification(tmp_path):
    path = tmp_path / "download"
    expected = hashlib.sha256(b"official").hexdigest()
    path.write_bytes(b"of")
    partial = verify_asset(path, expected_size=8, expected_sha256=expected)
    assert partial["present"] and not partial["downloaded"] and not partial["verified"]
    path.write_bytes(b"changed!")
    corrupt = verify_asset(path, expected_size=8, expected_sha256=expected)
    assert corrupt["downloaded"] and not corrupt["verified"]
    assert corrupt["error"] == "sha256_mismatch"


def test_selected_checkpoint_extraction_preserves_archive_and_is_idempotent(tmp_path):
    archive, integrity = _archive(tmp_path, [
        (f"ckpts/{BLOCK_ID}/model.ckpt", b"checkpoint"),
        (f"ckpts/{BLOCK_ID}/config.yml", b"config"),
        ("ckpts/other-block/model.ckpt", b"other"),
    ])
    destination = tmp_path / "extracted"
    report = safe_extract_tar(archive, destination, required_component=BLOCK_ID, **integrity)
    assert report["verified"] and report["extracted"] and report["extracted_files"] == 2
    assert archive.exists() and not (destination / "ckpts/other-block").exists()
    checkpoint = destination / f"ckpts/{BLOCK_ID}/model.ckpt"
    assert checkpoint.read_bytes() == b"checkpoint"
    before = checkpoint.stat().st_mtime_ns
    safe_extract_tar(archive, destination, required_component=BLOCK_ID, **integrity)
    assert checkpoint.stat().st_mtime_ns == before
    with pytest.raises(ValueError, match="different selection"):
        safe_extract_tar(archive, destination, **integrity)


@pytest.mark.parametrize("bad_name", ["../outside", "/absolute", "C:/outside", "a\\b",
                                     "a/../outside", "a/CON", "a/b:stream"])
def test_unsafe_members_rejected_before_any_extraction(tmp_path, bad_name):
    archive, integrity = _archive(tmp_path, [("okay/file", b"safe"), (bad_name, b"unsafe")])
    destination = tmp_path / "out"
    with pytest.raises(ValueError, match="archive member"):
        safe_extract_tar(archive, destination, **integrity)
    assert not destination.exists()
    assert not list(tmp_path.glob(".out.extract-*"))


def test_link_in_unselected_tree_is_still_rejected(tmp_path):
    archive, integrity = _archive(tmp_path, [("selected/a", b"safe"), ("other/link", "../out")])
    with pytest.raises(ValueError, match="member type"):
        safe_extract_tar(archive, tmp_path / "out", include_prefixes=["selected"], **integrity)


@pytest.mark.parametrize("limits", [ExtractionLimits(max_members=1),
                                    ExtractionLimits(max_total_bytes=5),
                                    ExtractionLimits(max_file_bytes=2)])
def test_expansion_limits_apply_before_writing(tmp_path, limits):
    archive, integrity = _archive(tmp_path, [("a", b"abcd"), ("b", b"efgh")])
    with pytest.raises(ValueError, match="limit"):
        safe_extract_tar(archive, tmp_path / "out", limits=limits, **integrity)
    assert not (tmp_path / "out").exists()


def test_never_replace_existing_unmanaged_directory(tmp_path):
    archive, integrity = _archive(tmp_path, [("a", b"x")])
    destination = tmp_path / "out"
    destination.mkdir()
    (destination / "keep").write_text("user data")
    with pytest.raises(FileExistsError):
        safe_extract_tar(archive, destination, **integrity)
    assert (destination / "keep").read_text() == "user data"


def test_tampered_extraction_not_silently_reused(tmp_path):
    archive, integrity = _archive(tmp_path, [("a", b"abc")])
    destination = tmp_path / "out"
    safe_extract_tar(archive, destination, **integrity)
    (destination / "a").unlink()
    with pytest.raises(FileExistsError):
        safe_extract_tar(archive, destination, **integrity)


def test_trusted_metadata_reports_source_inventory_without_inventing_rendered_entities(tmp_path):
    path = tmp_path / "video_scene_dict.pkl"
    frames = [{"timestamp": 1_000_000, "gt_names": ["vehicle", "pedestrian"],
               "track_tokens": ["car-1", "ped-1"], "cams": {"CAM_F0": {}}, "skipped": False},
              {"timestamp": 1_500_000, "gt_names": ["vehicle", "traffic_cone"],
               "track_tokens": ["car-1", "cone-1"], "cams": {"CAM_F0": {}},
               "skipped": "out_of_region"}]
    path.write_bytes(pickle.dumps({"video-0": {"frame_infos": frames, "map_location": "Boston"}}))
    with pytest.raises(ValueError, match="trusted=True"):
        inspect_block_metadata(path)
    report = inspect_block_metadata(path, trusted=True)
    assert report["videos"][0]["duration_seconds"] == .5
    assert report["videos"][0]["usable_frames"] == 1
    assert report["source_object_observations"]["vehicle"] == 2
    assert report["source_unique_tracks_by_class"] == {
        "pedestrian": 1, "traffic_cone": 1, "vehicle": 1}
    assert report["renderer_object_inventory"] is None
    assert report["world_object_inventory"] is None
    assert not report["simulator_ready"] and not report["trainable"]
    frames[-1]["timestamp"] = 900_000
    path.write_bytes(pickle.dumps({"video-0": {"frame_infos": frames}}))
    with pytest.raises(ValueError, match="strictly"):
        inspect_block_metadata(path, trusted=True)
