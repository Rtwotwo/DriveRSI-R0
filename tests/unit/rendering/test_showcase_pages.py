"""Isolated local gallery export tests; no upload, real gallery writes, or real media copies."""
import errno
import hashlib
import json

import pytest

import driversi.visualization.publication.pages as pages


def dump(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def gallery(tmp_path):
    root = tmp_path / "project"
    viz = root / "viz"
    (viz / "data").mkdir(parents=True)
    (viz / "media").mkdir()
    card = {"id": "example", "source_log": "source-log-id", "scenario_id": "source-token",
            "frame_count": 2, "trainable": False, "learned_self_play": False,
            "data": "data/example.json", "thumbnail": "media/image.png",
            "summary": "failed candidate", "warnings": ["quality unverified"],
            "private_checkpoint": "/home/private/model.pt",
            "score": {"before": 98, "after": 92, "official_benchmark": False},
            "playback": {"mode": "precomputed", "performance": {"physics_seconds": 1}},
            "camera_coverage": {"configured_cameras": 1, "camera_owners": 1,
                                "controlled_actors": 2}}
    dump(viz / "data/cases.json", {"schema_version": 1, "cases": [card],
                                  "presentation_scope": "diagnostic", "private": "secret"})
    data = {"id": "example", "metadata": card, "roads": [[[0, 0], [1, 1]]],
            "routes": [{"actor_id": "ego", "points": [[0, 0], [1, 1]]}],
            "trajectories": [{"actor_id": "ego", "role": "executed", "points": [[0, 0]]}],
            "frames": [{"tick": tick, "time_seconds": tick / 10,
                        "actors": [{"id": "ego", "x": tick, "y": 0, "yaw": 0,
                                    "length": 4, "width": 2}],
                        "cameras": [{"id": "front", "image": "media/image.png",
                                     "source": "media/source.jpg", "virtual": False,
                                     "warnings": ["quality unverified"],
                                     "private": "/home/private/asset"}]} for tick in (0, 1)],
            "evidence": {"execution_directory": "/data/private/run", "files": {"/data/log": "x"},
                         "physics_dt_seconds": .1, "report_fields": {"arrival_tick": 1,
                                                                     "private_path": "/tmp/secret"},
                         "branches": {"teacher": {"failure": "collision", "trajectory_admitted": False,
                                                   "events": {"collision": 1},
                                                   "score": {"score": .92, "trainable": False}}},
                         "image_policy": "source /home/private/data at 8.130.174.55:32341",
                         "trainable": False}}
    dump(viz / "data/example.json", data)
    (viz / "media/image.png").write_bytes(b"exact PNG bytes\x00\xff")
    (viz / "media/source.jpg").write_bytes(b"exact JPEG bytes\x00\xfe")
    (viz / "media/unrelated-secret.bin").write_bytes(b"never export")
    (viz / "data/private-report.json").write_text('{"secret":true}')
    (viz / "index.html").write_text(
        '<html><header><nav class="headnav"></nav></header><a href="preview.svg">preview</a>'
        '<script>fetch("data/cases.json")</script>'
        '<p class="foot">all data 32341 /data/xinyao</p></html>'
    )
    (viz / "preview.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg"><text>SERVER 32341</text>'
        '<metadata>/data/private/test</metadata><a href="http://127.0.0.1:38516">link</a>'
        '<image href="data:image/png;base64,YWJj"/><rect width="5"/></svg>'
    )
    return root


def test_complete_public_package_preserves_media_and_qualifications(gallery):
    out = gallery.parent / "published"
    original = {p: p.read_bytes() for p in (gallery / "viz").rglob("*") if p.is_file()}
    report = pages.export(gallery, out)
    assert report["uploaded"] is False
    assert report["rendered_image_count"] == 2
    assert report["media_file_count"] == 2
    assert report["trainable"] is False
    for name in ("image.png", "source.jpg"):
        src = gallery / "viz/media" / name
        dst = out / "site/media" / name
        assert src.read_bytes() == dst.read_bytes()
        assert src.stat().st_ino != dst.stat().st_ino
    assert not (out / "site/media/unrelated-secret.bin").exists()
    assert not (out / "site/data/private-report.json").exists()
    assert not (out / "site/.nojekyll").exists()
    for path, before in original.items():
        assert path.read_bytes() == before
    data = json.loads((out / "site/data/example.json").read_text())
    assert data["metadata"]["warnings"] == ["quality unverified"]
    assert data["metadata"]["score"]["after"] == 92
    assert data["evidence"]["branches"]["teacher"]["failure"] == "collision"
    assert data["metadata"]["source_log"] == "source-log-id"
    assert data["evidence"]["report_fields"]["arrival_tick"] == 1
    text = "\n".join(p.read_text() for p in (out / "site").rglob("*")
                     if p.suffix in (".json", ".html", ".svg"))
    for private in ("/home/private", "/data/private", "8.130.174.55", "32341", "private_checkpoint"):
        assert private not in text
    assert 'href="data:image/png;base64,YWJj"' in (out / "site/preview.svg").read_text()
    assert '<rect width="5"' in (out / "site/preview.svg").read_text()
    assert 'fetch("data/cases.json")' in (out / "site/index.html").read_text()
    assert 'href="preview.svg"' in (out / "site/index.html").read_text()


def test_manifest_covers_every_payload_file(gallery):
    out = gallery.parent / "published"
    report = pages.export(gallery, out)
    manifest = json.loads((out / "package-manifest.json").read_text())
    assert report == manifest
    actual = {p.relative_to(out).as_posix() for p in out.rglob("*")
              if p.is_file() and p.name != "package-manifest.json"}
    assert actual == {entry["path"] for entry in manifest["files"]}
    assert manifest["file_count"] == len(actual)
    for entry in manifest["files"]:
        raw = (out / entry["path"]).read_bytes()
        assert entry["bytes"] == len(raw)
        assert entry["sha256"] == hashlib.sha256(raw).hexdigest()
    assert manifest["site_bytes"] == sum(p.stat().st_size for p in (out / "site").rglob("*")
                                          if p.is_file())
    assert not (out / ".github").exists()


def test_existing_output_is_never_overwritten(gallery):
    out = gallery.parent / "exists"
    out.mkdir()
    with pytest.raises(FileExistsError):
        pages.export(gallery, out)
    assert list(out.iterdir()) == []


def test_final_install_refuses_a_concurrent_directory(gallery, monkeypatch):
    out = gallery.parent / "published"
    install = pages.rename_new_directory

    def concurrent(source, target):
        target.mkdir()
        (target / "sentinel").write_text("other publisher")
        install(source, target)

    monkeypatch.setattr(pages, "rename_new_directory", concurrent)
    with pytest.raises(FileExistsError):
        pages.export(gallery, out)
    assert (out / "sentinel").read_text() == "other publisher"
    assert not list(out.parent.glob(".published-*"))


@pytest.mark.parametrize("mutation", ["symlink", "missing", "traversal", "duplicate", "nonfinite"])
def test_invalid_sources_rejected_and_staging_removed(gallery, mutation):
    viz = gallery / "viz"
    data = json.loads((viz / "data/example.json").read_text())
    if mutation == "symlink":
        (viz / "media/image.png").unlink()
        (viz / "media/image.png").symlink_to(viz / "media/source.jpg")
    elif mutation == "missing":
        (viz / "media/image.png").unlink()
    elif mutation == "traversal":
        data["frames"][0]["cameras"][0]["image"] = "media/../../secret"
    elif mutation == "duplicate":
        catalog = json.loads((viz / "data/cases.json").read_text())
        catalog["cases"] *= 2
        dump(viz / "data/cases.json", catalog)
    elif mutation == "nonfinite":
        data["roads"][0][0][0] = float("nan")
    dump(viz / "data/example.json", data)
    out = gallery.parent / "published"
    with pytest.raises(ValueError):
        pages.export(gallery, out)
    assert not out.exists()
    assert not list(out.parent.glob(".published-*"))


@pytest.mark.parametrize("limit", ["MAX_FILE_BYTES", "MAX_SITE_BYTES"])
def test_size_limits_are_strict_and_leave_no_partial_package(gallery, monkeypatch, limit):
    monkeypatch.setattr(pages, limit, 10)
    out = gallery.parent / "published"
    with pytest.raises(ValueError, match="limit"):
        pages.export(gallery, out)
    assert not out.exists()
    assert not list(out.parent.glob(".published-*"))


def test_output_symlink_parent_rejected(gallery):
    link = gallery.parent / "link"
    link.symlink_to(gallery.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        pages.export(gallery, link / "published")


def test_svg_local_links_and_active_elements_are_not_published():
    raw = b'<svg><script>alert(1)</script><foreignObject>html</foreignObject><text onclick="x()">ok</text><image href="file:///tmp/secret"/></svg>'
    result = pages.public_svg(raw).decode()
    assert "script" not in result and "foreignObject" not in result
    assert "onclick" not in result and "file:" not in result
    assert "ok" in result


def test_public_html_retains_local_link_safety_detection():
    raw = b'<script>let localPreview=["localhost","127.0.0.1"];</script><p class="foot">32341</p>'
    result = pages.public_html(raw).decode()
    assert '"localhost","127.0.0.1"' in result
    assert "32341" not in result


def test_public_text_redacts_paths_without_dropping_failure_meaning():
    result = pages.redact_text("failure at /secret or C:\\private\\model.pt and http://8.130.174.55:32341")
    assert "failure" in result
    assert "/secret" not in result and "C:" not in result and "8.130" not in result


def test_source_log_timestamp_is_not_mistaken_for_ip_address():
    source = "2021.09.15.16.17.26_veh-28_01085_01182"
    assert pages.redact_text(source) == source
    assert pages.redact_text("2021.10.21.15.14.23_veh-28_01085_01182") == "2021.10.21.15.14.23_veh-28_01085_01182"
    assert "8.130.174.55" not in pages.redact_text("at 8.130.174.55:32341 now")


@pytest.mark.parametrize("error_code", [errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP])
def test_nas_fallback_installs_new_target(gallery, monkeypatch, error_code):
    def unsupported(source, target):
        raise OSError(error_code, "unsupported")
    monkeypatch.setattr(pages, "_rename_noreplace", unsupported)
    out = gallery.parent / "published"
    assert pages.export(gallery, out)["case_count"] == 1
    assert (out / "site/index.html").is_file()


@pytest.mark.parametrize("error_code", [errno.EINVAL, errno.ENOSYS])
@pytest.mark.parametrize("nonempty", [False, True])
def test_nas_fallback_never_replaces_existing_target(tmp_path, monkeypatch, error_code, nonempty):
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    (source / "payload").write_text("new")
    target.mkdir()
    if nonempty:
        (target / "sentinel").write_text("old")
    identity = target.stat().st_ino
    def unsupported(source, target):
        raise OSError(error_code, "unsupported")
    monkeypatch.setattr(pages, "_rename_noreplace", unsupported)
    with pytest.raises(FileExistsError):
        pages.rename_new_directory(source, target)
    assert target.stat().st_ino == identity
    assert (source / "payload").read_text() == "new"
    assert not (target / "payload").exists()
    if nonempty:
        assert (target / "sentinel").read_text() == "old"


@pytest.mark.parametrize("foreign_content", [False, True])
def test_nas_failed_install_cleans_only_own_empty_claim(tmp_path, monkeypatch, foreign_content):
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    def unsupported(source, target):
        raise OSError(errno.EINVAL, "unsupported")
    def failed_rename(source, target):
        if foreign_content:
            (target / "foreign").write_text("preserve")
        raise OSError(errno.EIO, "injected error")
    monkeypatch.setattr(pages, "_rename_noreplace", unsupported)
    monkeypatch.setattr(pages.os, "rename", failed_rename)
    with pytest.raises(OSError, match="injected"):
        pages.rename_new_directory(source, target)
    assert source.is_dir()
    if foreign_content:
        assert (target / "foreign").read_text() == "preserve"
    else:
        assert not target.exists()



@pytest.fixture
def architecture(gallery):
    figures = gallery / "assets/figures"
    figures.mkdir(parents=True)
    (figures / "platform-progress.html").write_text(
        '<!doctype html><html><head><style>.stage{fill:url(#marker)}</style></head>'
        '<body><a href="../index.html">Gallery</a>'
        '<a href="../index.html?case=example">Case</a><a href="#stage=selfplay">Stage</a>'
        '<a href="platform-progress.svg" download>Diagram</a>'
        '<a href="platform-progress.json" download>Specification</a>'
        '<svg xmlns="http://www.w3.org/2000/svg"><use href="#marker"/></svg>'
        '<script>fetch("platform-progress.json").then(r=>r.json());'
        'document.querySelector("body").classList.add("ready");</script></body></html>',
        encoding="utf-8",
    )
    (figures / "platform-progress.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg"><defs><marker id="marker"/></defs>'
        '<path d="M0 0 L1 1" marker-end="url(#marker)"/><text>Platform</text></svg>',
        encoding="utf-8",
    )
    dump(figures / "platform-progress.json", {
        "schema_version": 1,
        "stages": [{"id": "selfplay", "status": "planned", "evidence": "src/rsi/"}],
    })
    (figures / "unrelated-secret.json").write_text('{"secret":true}')
    return figures


def test_architecture_bundle_is_optional_and_has_no_dangling_navigation(gallery):
    index = gallery / "viz/index.html"
    index.write_text(index.read_text().replace(
        '<nav class="headnav">', '<nav class="headnav"><a href="architecture/index.html">'
        '平台架构与进度</a>'
    ))
    out = gallery.parent / "published"
    pages.export(gallery, out)
    assert not (out / "site/architecture").exists()
    assert "architecture/index.html" not in (out / "site/index.html").read_text()


@pytest.mark.parametrize("existing_link", [False, True])
def test_architecture_bundle_is_bounded_linked_and_hashed(gallery, architecture, existing_link):
    index = gallery / "viz/index.html"
    if existing_link:
        index.write_text(index.read_text().replace(
            '<nav class="headnav">', '<nav class="headnav">'
            '<a href="architecture/index.html">平台架构与进度</a>'
        ))
    original = {p.name: p.read_bytes() for p in architecture.iterdir()}
    out = gallery.parent / "published"
    manifest = pages.export(gallery, out)
    html = (out / "site/index.html").read_text()
    assert html.count('href="architecture/index.html"') == 1
    assert "平台架构与进度" in html
    expected = {"site/architecture/" + name for name in pages.ARCHITECTURE_FILES.values()}
    entries = {entry["path"]: entry for entry in manifest["files"]}
    assert expected.issubset(entries)
    assert {p.name for p in (out / "site/architecture").iterdir()} == set(
        pages.ARCHITECTURE_FILES.values()
    )
    for source, name in pages.ARCHITECTURE_FILES.items():
        raw = (out / "site/architecture" / name).read_bytes()
        assert raw == original[source]
        assert entries["site/architecture/" + name]["sha256"] == hashlib.sha256(raw).hexdigest()
        assert entries["site/architecture/" + name]["bytes"] == len(raw)
    assert all(p.read_bytes() == original[p.name] for p in architecture.iterdir())
    actual = {p.relative_to(out).as_posix() for p in out.rglob("*")
              if p.is_file() and p.name != "package-manifest.json"}
    assert set(entries) == actual
    assert manifest["site_bytes"] == sum(p.stat().st_size for p in (out / "site").rglob("*")
                                          if p.is_file())


@pytest.mark.parametrize("mutation", [
    "partial", "symlink", "parent_symlink", "json_nonfinite", "json_duplicate",
    "json_private", "json_escaped_remote", "json_unknown_case", "html_remote", "html_relative_escape",
    "html_unknown_case", "html_script_src", "html_remote_fetch", "html_dynamic_fetch",
    "html_protocol_relative", "html_meta_refresh", "html_css_resource",
    "svg_relative_escape", "svg_script", "svg_entity", "svg_stylesheet", "svg_base", "size",
])
def test_invalid_architecture_bundle_leaves_no_partial_export(
        gallery, architecture, mutation, monkeypatch):
    html = architecture / "platform-progress.html"
    svg = architecture / "platform-progress.svg"
    spec = architecture / "platform-progress.json"
    if mutation == "partial":
        svg.unlink()
    elif mutation == "symlink":
        svg.unlink()
        svg.symlink_to(html)
    elif mutation == "parent_symlink":
        relocated = gallery / "figures-real"
        architecture.rename(relocated)
        architecture.symlink_to(relocated, target_is_directory=True)
    elif mutation == "json_nonfinite":
        spec.write_text('{"score": NaN}')
    elif mutation == "json_duplicate":
        spec.write_text('{"stage": 1, "stage": 2}')
    elif mutation == "json_unknown_case":
        spec.write_text('{"stages": [{"case_ids": ["missing"]}]}')
    elif mutation == "json_private":
        spec.write_text('{"source": "/home/private/model.pt"}')
    elif mutation == "json_escaped_remote":
        spec.write_text(r'{"source": "https\u003a//example.com/a"}')
    elif mutation == "html_remote":
        html.write_text('<img src="https://example.com/a.png">')
    elif mutation == "html_protocol_relative":
        html.write_text('<img src="//example.com/a.png">')
    elif mutation == "html_relative_escape":
        html.write_text('<img src="../../private.png">')
    elif mutation == "html_unknown_case":
        html.write_text('<a href="../index.html?case=unknown">Case</a>')
    elif mutation == "html_script_src":
        html.write_text('<script src="platform-progress.json"></script>')
    elif mutation == "html_remote_fetch":
        html.write_text('<script>fetch("../../private.json")</script>')
    elif mutation == "html_dynamic_fetch":
        html.write_text('<script>fetch(resource)</script>')
    elif mutation == "html_meta_refresh":
        html.write_text('<meta http-equiv="refresh" content="0;url=../private">')
    elif mutation == "html_css_resource":
        html.write_text('<style>body{background:url(../../private.png)}</style>')
    elif mutation == "svg_relative_escape":
        svg.write_text('<svg><image href="../../private.png"/></svg>')
    elif mutation == "svg_stylesheet":
        svg.write_text('<?xml-stylesheet href="../../private.css"?><svg/>')
    elif mutation == "svg_base":
        svg.write_text('<svg xml:base="../../private/"><use href="#image"/></svg>')
    elif mutation == "svg_script":
        svg.write_text('<svg><script>alert(1)</script></svg>')
    elif mutation == "svg_entity":
        svg.write_text('<!DOCTYPE svg [<!ENTITY local "hidden">]><svg>&local;</svg>')
    elif mutation == "size":
        monkeypatch.setattr(pages, "MAX_ARCHITECTURE_BYTES", 10)
    out = gallery.parent / "published"
    with pytest.raises(ValueError):
        pages.export(gallery, out)
    assert not out.exists()
    assert not list(out.parent.glob(".published-*"))


@pytest.mark.parametrize("limit", ["MAX_FILE_BYTES", "MAX_SITE_BYTES"])
def test_architecture_uses_existing_package_size_limits(gallery, architecture, monkeypatch, limit):
    assets = pages.architecture_assets(gallery, {"example"})
    stage = gallery.parent / "stage"
    stage.mkdir()
    package = pages.Package(stage)
    monkeypatch.setattr(pages, limit, 10)
    with pytest.raises(ValueError, match="limit"):
        package.add("site/architecture/platform-progress.json",
                    assets["site/architecture/platform-progress.json"])
