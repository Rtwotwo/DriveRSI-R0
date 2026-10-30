"""Post-render review reuses real ledger identities, never file-shaped substitutes."""

import json
from pathlib import Path

import pytest

from driversi.rsi.plugins import chainflow_synthetic as bridge
from driversi.rsi.workflow.artifacts import seal
from driversi.rsi.workflow.store import HarnessStore


@pytest.fixture
def stored(tmp_path, monkeypatch):
    store = HarnessStore(tmp_path / "ledger.sqlite")
    stages, declarations, receipts = [], [], {}
    for port, kind in bridge._REVIEWED_KINDS.items():
        folder = tmp_path / "original" / port
        folder.mkdir(parents=True)
        path = folder / "report.json"
        report = {"source_log": "train-log", "files": {}, "upstream_source_files": {}}
        extra = {}
        if port == "branch":
            window = folder / "window.json"
            window.write_text("{}")
            extra["trajectory_window"] = str(window)
        elif port == "rendered":
            report = {
                "branch_identity": receipts["branch"]["summary"]["artifact_reference"]["identity"],
                "payload_files": {},
            }
            render = folder / "render.json"
            render.write_text(
                json.dumps({"inputs": {}, "frames": [{"tick": 20, "source_token": "parent"}]})
            )
            extra["render_report"] = str(render)
        else:
            report = {
                "render_identity": receipts["rendered"]["summary"]["artifact_reference"][
                    "identity"
                ],
                "files": {},
                "inputs": {},
                "producer_code": {},
            }
        path.write_text(json.dumps(report))
        declaration = dict(id=port, output_kind=kind, provider="test_" + port, provider_version="1")
        receipt = dict(
            status="completed",
            reason=None,
            sim_agent_steps=0,
            artifacts={"report": str(path), **extra},
            summary={},
            admission={"trainable": False},
        )
        receipt["summary"]["artifact_reference"] = seal(receipt, folder, "original", declaration)
        declarations.append(declaration)
        receipts[port] = receipt
        stages.append(dict(id=port, status="completed", result=receipt))
    store.register(
        dict(id="original", providers=[], source_groups=["train-log"], steps=declarations),
        tmp_path / "original",
    )
    state = store.claim("original")
    state.update(status="completed", stages=stages)
    store.save(state, "task/end", {"status": "completed"})
    names = [
        "agents/chainflow/synthetic_export.py",
        "agents/chainflow/synthetic.py",
        "agents/chainflow/data.py",
        "src/rendering/diagnostics/worldengine_admission.py",
        "src/evaluation/metrics/counterfactual_pdm.py",
        "src/rsi/correction/trajectory_admission.py",
        "src/data/datasets/navsim_synthetic.py",
        "src/rsi/plugins/chainflow_synthetic.py",
        "src/rsi/plugins/worldengine.py",
        "src/rsi/workflow/artifacts.py",
        "src/rsi/workflow/store.py",
        "python",
        "review.json",
    ]
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    (tmp_path / "maps").mkdir()
    monkeypatch.setattr(bridge, "project_root", lambda: tmp_path)
    config = {port: dict(source_task="original", source_stage=port) for port in receipts}
    config.update(
        python="python", maps_root="maps", max_wall_seconds=60, visual_review="review.json"
    )
    return store, config, receipts


def test_real_stored_references_preserve_original_identity(tmp_path, stored, monkeypatch):
    store, raw, originals = stored
    config, cap = bridge.prepare_reviewed(raw, tmp_path, store.registry.path)
    assert cap == 160
    assert config["source_groups"] == ["train-log"]
    seen = []

    def reject(path):
        certificate = json.loads(Path(path).read_text())
        seen.append(certificate)
        return dict(
            certificate,
            admitted=False,
            rejection_reasons=["heldout_source_reconstruction_below_27db"],
        )

    monkeypatch.setattr(
        "driversi.rendering.diagnostics.worldengine_admission.validate_admission", reject
    )
    monkeypatch.setattr(
        bridge,
        "execute",
        lambda *a: pytest.fail("Rejected visual evidence must never start native export"),
    )
    value = bridge.execute_reviewed(config, {}, tmp_path / "exp/reviewed", store.registry.path)
    assert value["status"] == "rejected"
    assert value["sim_agent_steps"] == 0
    assert value["summary"]["trainable"] is False
    assert (
        seen[0]["branch_identity"]
        == originals["branch"]["summary"]["artifact_reference"]["identity"]
    )
    assert seen[0]["render_report"] == originals["rendered"]["artifacts"]["render_report"]
    assert seen[0]["source_comparison"] == originals["source"]["artifacts"]["report"]
    assert Path(value["artifacts"]["visual_report"]).is_file()
    assert not (tmp_path / "exp/reviewed/export").exists()


@pytest.mark.parametrize(
    "fault", ["tampered", "identity", "stale", "protected", "review", "incomplete"]
)
def test_reuse_rejects_changed_ledger_or_evidence(tmp_path, stored, fault):
    store, raw, originals = stored
    config, _ = bridge.prepare_reviewed(raw, tmp_path, store.registry.path)
    if fault == "review":
        (tmp_path / "review.json").write_text('{"changed":true}')
    elif fault == "tampered":
        Path(originals["rendered"]["artifacts"]["render_report"]).write_text("{}")
    elif fault == "protected":
        with store.registry._connect() as db:
            db.execute("CREATE TABLE repair_gate_groups (log_group TEXT)")
            db.execute("INSERT INTO repair_gate_groups VALUES ('train-log')")
    else:
        state = store.get("original")
        stage = state["stages"][0]
        if fault == "identity":
            stage["result"]["summary"]["artifact_reference"]["identity"] = "forged"
        elif fault == "incomplete":
            stage["status"] = "failed"
        else:
            path = Path(stage["result"]["artifacts"]["report"])
            path.write_text('{"source_log":"train-log","new":true}')
            stage["result"]["summary"]["artifact_reference"] = seal(
                stage["result"], path.parent, "original", state["task"]["steps"][0]
            )
        with store.registry._connect() as db:
            db.execute(
                "UPDATE harness_tasks SET payload=? WHERE id='original'", (json.dumps(state),)
            )
    with pytest.raises(ValueError):
        bridge.execute_reviewed(config, {}, tmp_path / "exp/reviewed", store.registry.path)
    assert not (tmp_path / "exp/reviewed").exists()


def test_wrong_type_and_arbitrary_path_are_not_references(tmp_path, stored):
    store, raw, _ = stored
    raw["branch"] = dict(source_task="original", source_stage="rendered")
    with pytest.raises(ValueError, match="selected_native_branch"):
        bridge.prepare_reviewed(raw, tmp_path, store.registry.path)
    raw["branch"] = {"report": "/arbitrary/report.json"}
    with pytest.raises(ValueError):
        bridge.prepare_reviewed(raw, tmp_path, store.registry.path)
