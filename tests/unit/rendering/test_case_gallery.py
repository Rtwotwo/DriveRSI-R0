import hashlib
import json

import pytest

from driversi.rendering.inspection.case_gallery import write_candidate_viewer


@pytest.fixture
def catalog(tmp_path):
    root = tmp_path / "outputs"
    media = root / "demo"
    media.mkdir(parents=True)
    (media / "image.png").write_bytes(b"image")
    (media / "viewer.html").write_text("viewer")
    cases = root / "cases"
    (cases / "clips").mkdir(parents=True)
    clip = {"sensors": [{"cameras": [{"artifacts": {"image": {"path": "image.png"}}}]}]}
    raw = json.dumps(clip).encode()
    (cases / "clips/one.json").write_bytes(raw)
    doc = {"kind": "mtgs_long_tail_candidates", "status": "completed", "trainable": False,
           "source": {"artifact_root": str(media), "session_manifest": str(media / "paired-manifest.json")},
           "candidates": [{"candidate_id": "</script><script>bad()</script>",
                           "clip": {"path": "clips/one.json", "sha256": hashlib.sha256(raw).hexdigest()}}],
           "summary": {}, "eligibility": {}, "configuration": {}}
    path = cases / "manifest.json"
    path.write_text(json.dumps(doc))
    return path, root, doc


def test_viewer_references_server_images_and_escapes_metadata(catalog):
    manifest, root, _ = catalog
    output = write_candidate_viewer(manifest, root)
    html = output.read_text()
    assert "../demo/image.png" in html
    assert "</script><script>bad()" not in html
    assert not list(manifest.parent.rglob("*.png"))
    with pytest.raises(FileExistsError):
        write_candidate_viewer(manifest, root)


def test_viewer_rejects_media_outside_explicit_web_root(catalog):
    manifest, _, _ = catalog
    with pytest.raises(ValueError, match="web_root"):
        write_candidate_viewer(manifest, manifest.parent)
    assert not (manifest.parent / "viewer.html").exists()


def test_viewer_rejects_changed_clip_before_generating_page(catalog):
    manifest, root, _ = catalog
    (manifest.parent / "clips/one.json").write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        write_candidate_viewer(manifest, root)
    assert not (manifest.parent / "viewer.html").exists()
