"""Production entry points reject invalid concurrency before touching renderer assets."""

import json

import pytest

from driversi.rendering.execution import challenge_replay, native_replay, teacher_pilot
from driversi.rendering.gaussian.mtgs_session import run_matched_mtgs


@pytest.mark.parametrize("workers", [0, 5, True, 1.5, "4"])
@pytest.mark.parametrize("entry", ["native", "challenge", "teacher", "matched"])
def test_invalid_writer_config_fails_before_renderer_or_world(tmp_path, workers, entry):
    config = {"image_write_workers": workers}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    report_path = tmp_path / "source-report.json"
    report_path.write_text("{}")
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="image_write_workers"):
        if entry == "native":
            native_replay.run(config_path, tmp_path / "missing-execution", output)
        elif entry == "challenge":
            challenge_replay.load_replay_input(config_path, report_path)
        elif entry == "teacher":
            teacher_pilot.run(config, tmp_path / "missing-policy.npz", output)
        else:
            required = {
                key: tmp_path / key
                for key in (
                    "config_path", "checkpoint_path", "road_block_config", "scene",
                    "road_graph", "binding", "world_python", "native_path", "output",
                )
            }
            run_matched_mtgs(**required, image_write_workers=workers)
    assert not output.exists()
