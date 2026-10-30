import json

import pytest

from driversi.agents.backends.native import digest
from driversi.rendering.gaussian.resident_session import ResidentRenderSession
from driversi.rendering.gaussian.source_session import SourceSession
from driversi.rendering.gaussian.worldengine_source import run_cached


def test_resident_terrain_change_fails_before_render(tmp_path):
    terrain = tmp_path / "road_height_map.npy"
    terrain.write_bytes(b"initial terrain")
    session = ResidentRenderSession.__new__(ResidentRenderSession)
    session.frozen = {str(terrain): digest(terrain)}
    session.check_inputs()
    terrain.write_bytes(b"changed terrain")
    with pytest.raises(ValueError, match="terrain"):
        session.check_inputs()


def test_cached_source_wrapper_binds_each_current_render(tmp_path):
    checkpoint, metadata = tmp_path / "asset.ckpt", tmp_path / "video_scene_dict.pkl"
    checkpoint.write_bytes(b"checkpoint")
    metadata.write_bytes(b"metadata")
    assets = {str(p): digest(p) for p in (checkpoint, metadata)}
    cached = tmp_path / "canonical.json"
    cached.write_text(json.dumps(dict(files={}, frames=[])))
    session = SourceSession.__new__(SourceSession)
    session.inputs = assets

    def cached_run(_report, destination):
        destination.mkdir(parents=True)
        return cached, True

    session.run = cached_run
    results = []
    for i in (1, 2):
        report = tmp_path / f"render-{i}.json"
        report.write_text(json.dumps(dict(inputs=assets, output_image_size=[960, 540], branch=i)))
        result = run_cached(report, tmp_path / f"out-{i}", session)
        assert result["render_report_sha256"] == digest(report)
        assert result["inputs"][str(report)] == digest(report)
        results.append(result)
    assert results[0]["render_report_sha256"] != results[1]["render_report_sha256"]
    metadata.write_bytes(b"a different asset")
    changed = tmp_path / "changed.json"
    changed.write_text(
        json.dumps(
            dict(
                inputs={str(checkpoint): digest(checkpoint), str(metadata): digest(metadata)},
                output_image_size=[960, 540],
            )
        )
    )
    with pytest.raises(ValueError, match="another asset"):
        run_cached(changed, tmp_path / "bad", session)
