import json

import pytest

from driversi.workbench.opponent_curriculum import export_dashboard


def test_dashboard_escapes_source_labels_keeps_scope_and_requires_in_gallery_replays(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    viewer = tmp_path / "replays/viewer.html"
    viewer.parent.mkdir()
    viewer.write_text("<html>test</html>")
    label = "</script><script>alert(1)</script>"
    report = {"adaptive_opponents": {"state": {"history": []}, "refreshes_this_run": []},
              "updates": 0, "valid_transitions": 0, "seconds_including_eval_recording": 1}
    (run / "report.json").write_text(json.dumps(report))
    (run / "config.json").write_text(json.dumps({"suite": {
        "train": [{"path": "train"}], "curriculum": [{"path": "curriculum"}],
        "evaluation": [{"path": "evaluation"}], "limits": [label]}}))
    matches = {"matchups": [{"opponent_id": label, "episodes": [{"map_name": "city", "scenario_id": label,
        "source_log": "log", "seconds": 1, "roles": [], "learner": {}, "opponent": {},
        "replay": {"viewer": str(viewer)}}]}]}
    path = tmp_path / "matchups.json"
    path.write_text(json.dumps(matches))
    output = tmp_path / "index.html"
    result = export_dashboard(run, path, output)
    text = output.read_text()
    assert result["replays"] == 1 and result["cities"] == ["city"]
    assert label not in text and "\\u003c/script>" in text
    assert "replays/viewer.html" in text and "photoreal" not in result
    with pytest.raises(FileExistsError):
        export_dashboard(run, path, output)
    with pytest.raises(ValueError):
        export_dashboard(run, path, tmp_path / "other/index.html")
