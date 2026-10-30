from pathlib import Path

import pytest

from driversi.cli import main


def test_population_config_reaches_render_runner_from_cli(monkeypatch):
    captured = {}
    monkeypatch.setattr("driversi.rendering.gaussian.mtgs_session.run_matched_mtgs", lambda **kw: captured.update(kw))
    args = ["run-mtgs-session"]
    for name in ("config", "checkpoint", "road-block-config", "scene", "road-graph", "binding",
                 "world-python", "native-path", "output-dir"):
        args += ["--"+name, "/fixture/"+name]
    args += ["--population-config", "/fixture/population.json", "--height-mode", "reference-track"]
    assert main(args) == 0
    assert captured["population_config"] == Path("/fixture/population.json")
    assert captured["height_mode"] == "reference-track"


@pytest.mark.parametrize("flag,expected", [([], False), (["--visual-policy"], True),
                                           (["--no-visual-policy"], False)])
def test_visual_toggle_is_explicit_and_defaults_to_structured(monkeypatch, flag, expected):
    captured = {}
    monkeypatch.setattr("driversi.rendering.gaussian.mtgs_session.run_matched_mtgs", lambda **kw: captured.update(kw))
    args = ["run-mtgs-session"]
    for name in ("config", "checkpoint", "road-block-config", "scene", "road-graph", "binding",
                 "world-python", "native-path", "output-dir"):
        args += ["--"+name, "/fixture/"+name]
    assert main(args+flag) == 0
    assert captured["visual_policy"] is expected
