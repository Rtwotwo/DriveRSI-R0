# ruff: noqa: E402
"""End-to-end campaigns use synthetic worlds and a deliberately rejecting test gate."""

import copy
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("driversi.sim._native")
from driversi.rsi.experiments.campaign import recover_campaign, register_campaign, run_campaign
from driversi.rsi.experiments.campaign_spec import resolve_campaign
from driversi.rsi.experiments.campaign_store import CampaignStore
from driversi.rsi.execution.jobs import process_identity

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def campaign_case(straight_scene, tmp_path):
    rows = []
    for name in ("mining", "development", "gate0", "gate1"):
        scene = copy.deepcopy(straight_scene)
        scene.metadata["source_log"] = name + "-log"
        path = tmp_path / (name + ".npz")
        scene.save(path)
        rows.append({"path": str(path)})
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "official_test": False,
                "train": rows[:1],
                "development": rows[1:2],
                "promotion_slots": [[rows[2]], [rows[3]]],
            }
        )
    )
    cap = {
        "gpu_seconds": 0,
        "cpu_seconds": 45,
        "llm_tokens": 0,
        "sim_agent_steps": 512,
        "rendered_frames": 0,
        "trials": 1,
    }
    raw = {
        "schema_version": 1,
        "id": "test-campaign",
        "suite": str(suite),
        "agent_config": str(ROOT / "tests/fixtures/legacy/agents/linear.yaml"),
        "initial_model": "test-seed",
        "initial_weights": None,
        "initial_training_groups": [],
        "rounds": 2,
        "seed": 1,
        "device": "cpu",
        "gpu_id": 0,
        "wall_seconds": 240,
        "budget": {**cap, "cpu_seconds": 400, "sim_agent_steps": 10000, "trials": 6},
        "stage_budget": {k: dict(cap) for k in ("mining", "training", "gate")},
        "gate": {
            "id": "TEST_ONLY_NEVER_PROMOTE",
            "metric": "return_sum_per_actor",
            "direction": "maximize",
            "min_gain": 99999,
            "min_scenes": 1,
            "min_groups": 1,
            "min_actors": 2,
            "max_collision": 0.1,
            "max_offroad": 0.1,
            "collision_tolerance": 0,
            "offroad_tolerance": 0,
        },
        "improver": {
            "name": "hardcase_curriculum_v1",
            "max_scenes": 1,
            "replay_fraction": 0.25,
            "recipe": "short",
        },
        "recipes": {
            "short": {
                "target_transitions": 16,
                "max_train_agent_steps": 64,
                "learning_rate": 0.001,
                "epochs": 1,
                "minibatch_size": 8,
            }
        },
        "training": {
            "worlds": 1,
            "rollout_steps": 4,
            "max_rollout_mib": 64,
            "recording": "metrics",
        },
    }
    return raw, tmp_path


def register(case):
    raw, root = case
    p = root / "campaign.json"
    p.write_text(json.dumps(raw))
    return register_campaign(root / "ledger.sqlite", p, root / "runs")


def test_two_round_rejection_retains_parent_and_private_gate(campaign_case):
    raw, root = campaign_case
    register(campaign_case)
    result = run_campaign(root / "ledger.sqlite", raw["id"])
    assert result["status"] == "completed", result["failure_reason"]
    assert result["incumbent"]["version"] == "test-seed"
    assert len(result["history"]) == 2 and result["cost"]["trials"] == 6
    assert result["cost"]["cpu_seconds"] > 0 and result["cost"]["sim_agent_steps"] > 0
    store = CampaignStore(root / "ledger.sqlite")
    for i, round_ in enumerate(result["history"]):
        assert round_["decision"]["outcome"] == "rejected"
        assert round_["parent"] == result["incumbent"]
        assert round_["proposal"]["selected_paths"] == result["spec"]["train_paths"]
        mining = store.jobs.get(round_["mining_job"])
        train = store.jobs.get(round_["training_job"])
        gate = store.jobs.get(round_["gate_job"])
        assert mining["progress"]["updates"] == gate["progress"]["updates"] == 0
        assert train["progress"]["updates"] > 0
        assert gate["evaluation"]["reference_model"] == "test-seed"
        assert [x["scene"] for x in gate["evaluation"]["episodes"]] == result["spec"][
            "promotion_slots"
        ][i]
        assert gate["evaluation"]["source_separation_checked"]
    public = json.dumps(store.registry.public_summary())
    assert "gate0.npz" not in public and "gate1.npz" not in public
    assert "gate0-log" not in public and "gate1-log" not in public
    assert "TEST_ONLY_NEVER_PROMOTE" in public
    with pytest.raises(ValueError):
        run_campaign(root / "ledger.sqlite", raw["id"])


@pytest.mark.parametrize(
    "mutation", ["overlap", "reuse_slot", "weights_lineage", "official", "budget"]
)
def test_campaign_rejects_invalid_fixed_protocol(campaign_case, mutation):
    raw, root = campaign_case
    suite = json.loads(Path(raw["suite"]).read_text())
    if mutation == "overlap":
        suite["development"] = suite["train"]
    elif mutation == "reuse_slot":
        suite["promotion_slots"][1] = suite["promotion_slots"][0]
    elif mutation == "weights_lineage":
        raw["initial_training_groups"] = ["gate0-log"]
    elif mutation == "official":
        suite["official_test"] = True
    else:
        raw["budget"]["trials"] = 2
    Path(raw["suite"]).write_text(json.dumps(suite))
    with pytest.raises(ValueError):
        register(campaign_case)


def test_cancelled_registration_keeps_slot_reserved_and_protocol_frozen(campaign_case):
    raw, root = campaign_case
    register(campaign_case)
    store = CampaignStore(root / "ledger.sqlite")
    state = store.acquire(raw["id"])
    state["spec"]["gate"]["min_gain"] = 0.001
    with pytest.raises(ValueError):
        store.update(state)
    with pytest.raises(ValueError):
        recover_campaign(root / "ledger.sqlite", raw["id"])
    store.cancel(raw["id"])
    other = {**raw, "id": "other"}
    with pytest.raises(ValueError):
        store.register(resolve_campaign(other, root), root / "other-runs")


@pytest.mark.parametrize("interruption", ["cancel", "kill"])
def test_campaign_interrupt_settlement(campaign_case, interruption):
    raw, root = campaign_case
    register(campaign_case)
    store = CampaignStore(root / "ledger.sqlite")
    child = None
    with (root / "controller.log").open("w") as log:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-B",
                "-m",
                "driversi.rsi",
                "--registry",
                str(root / "ledger.sqlite"),
                "campaign-run",
                raw["id"],
            ],
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                state = store.get(raw["id"])
                if state["job_ids"]:
                    try:
                        child = store.jobs.get(state["job_ids"][-1]).get("child")
                    except ValueError:
                        pass
                if child:
                    break
                if proc.poll() is not None:
                    pytest.fail((root / "controller.log").read_text())
                time.sleep(0.03)
            assert child
            if interruption == "cancel":
                store.cancel(raw["id"])
            else:
                os.kill(proc.pid, signal.SIGKILL)
            proc.wait(timeout=15)
            deadline = time.monotonic() + 5
            while process_identity(child["pid"]) == child["start"] and time.monotonic() < deadline:
                time.sleep(0.03)
            assert process_identity(child["pid"]) != child["start"]
            result = (
                recover_campaign(root / "ledger.sqlite", raw["id"])
                if interruption == "kill"
                else store.get(raw["id"])
            )
            assert result["status"] == ("failed" if interruption == "kill" else "cancelled")
            assert result["incumbent"]["version"] == "test-seed"
            assert result["history"] == [] and result["cost"]["trials"] == 1
            assert result["cost"]["cpu_seconds"] > 0
            assert all(
                store.jobs.get(j)["status"] not in ("queued", "running") for j in result["job_ids"]
            )
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=5)
            if child and process_identity(child["pid"]) == child["start"]:
                os.killpg(child["pid"], signal.SIGKILL)


def test_global_budget_reserves_gate_before_next_round(campaign_case):
    raw, root = campaign_case
    raw["budget"]["trials"] = 3
    register(campaign_case)
    result = run_campaign(root / "ledger.sqlite", raw["id"])
    assert result["status"] == "budget_exhausted", result["failure_reason"]
    assert len(result["history"]) == 1 and result["cost"]["trials"] == 3
    assert len(result["job_ids"]) == 3
    assert result["incumbent"]["version"] == "test-seed"
