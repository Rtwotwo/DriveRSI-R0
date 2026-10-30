import json

import pytest

from driversi.agents.backends.native import digest
from driversi.rsi.plugins import mtgs_window
from driversi.rsi.workflow.providers import builtin_providers


def test_full_mtgs_producer_has_no_admission_authority():
    provider = builtin_providers().get("mtgs_selected_branch_render")
    assert provider.inputs == (("branch", "selected_native_branch"),)
    assert provider.output == "worldengine_inspection_window"


@pytest.mark.parametrize("changed", ["weight", "selected_rollout"])
def test_tampered_assets_or_physics_rejected_before_worker(tmp_path, monkeypatch, changed):
    weight, rollout, report = [
        tmp_path / name for name in ("weight", "selected_rollout", "branch.json")
    ]
    weight.write_bytes(b"weight-v1")
    rollout.write_bytes(b"physics-v1")
    config = {"asset_files": {str(weight): digest(weight)}, "code_files": {}}
    report.write_text(
        json.dumps({"files": {str(rollout): digest(rollout)}, "upstream_source_files": {}})
    )
    (tmp_path / changed).write_bytes(b"changed")
    monkeypatch.setattr(
        mtgs_window, "run_process", lambda *a, **k: pytest.fail("Must reject before GPU work")
    )
    with pytest.raises(ValueError):
        mtgs_window.execute(
            config, {"branch": {"artifacts": {"report": str(report)}}}, tmp_path / "out", None
        )
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "qualified,anchor,future", [(False, 20, 10), (True, 19, 10), (True, 20, 9), (True, 20, True)]
)
def test_incomplete_or_unqualified_branch_never_renders(
    tmp_path, monkeypatch, qualified, anchor, future
):
    report = tmp_path / "branch.json"
    report.write_text(
        json.dumps(
            {
                "files": {},
                "upstream_source_files": {},
                "source_log": "train-log",
                "teacher_qualified": qualified,
                "anchor_tick": anchor,
                "future_frames": future,
            }
        )
    )
    monkeypatch.setattr(mtgs_window, "protect", lambda *a: None)
    monkeypatch.setattr(
        mtgs_window, "run_process", lambda *a, **k: pytest.fail("Must reject before GPU work")
    )
    with pytest.raises(ValueError, match="Qualified"):
        mtgs_window.execute(
            {"asset_files": {}, "code_files": {}},
            {"branch": {"artifacts": {"report": str(report)}}},
            tmp_path / "out",
            None,
        )
    assert not (tmp_path / "out").exists()
