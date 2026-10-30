import copy
from concurrent.futures import ThreadPoolExecutor

import pytest

from driversi.rsi.execution.jobs import JobStore


@pytest.fixture
def store_and_job(tmp_path):
    store = JobStore(tmp_path / "ledger.sqlite")
    arm = {"version": "initial", "initial_model": "seed0", "information_set": "data0", "tools": "tools0"}
    plan = {"schema_version": 1, "id": "p", "target": "model", "hypothesis": "TEST ONLY",
            "arms": {"baseline": arm, "candidate": {**arm, "version": "updated"}},
            "protocol": {"id": "e", "evaluator_version": "driversi.teacher.v1",
                         "role": "development", "task_set": "tasks", "metric": "goal_per_actor",
                         "direction": "maximize"},
            "data": {"development_groups": ["a"], "evaluation_groups": ["b"]},
            "budget_per_arm": {"gpu_seconds": 50, "cpu_seconds": 50, "llm_tokens": 0,
                               "sim_agent_steps": 500, "rendered_frames": 0, "trials": 3},
            "scene_ids": []}
    store.registry.register(plan)
    job = {"id": "j", "plan_id": "p", "arm": "candidate", "device": "cpu",
           "max_train_agent_steps": 100, "evaluation_max_agent_steps": 50, "agent": {},
           "seed": 0, "initial_model": "seed0", "initial_weights": None,
           "candidate_version": "updated", "sources": [], "eval_paths": [],
           "evaluator_version": "driversi.teacher.v1", "task_set": "tasks"}
    return store, job, tmp_path / "runs"


def failure(state):
    state = copy.deepcopy(state)
    state["cost"].update(trials=1, cpu_seconds=2, sim_agent_steps=10)
    return state


def test_reservation_blocks_concurrent_arm_and_manual_record(store_and_job):
    store, job, root = store_and_job
    state = store.claim(job, root)
    with pytest.raises(ValueError, match="active reservation"):
        store.claim({**job, "id": "second"}, root)
    attempt = {"id": "manual", "plan_id": "p", "arm": "candidate", "status": "failed",
               "cost": {**state["cost"], "trials": 1}, "metrics": {}, "artifacts": {},
               "candidate_model": None, "failure_reason": "test"}
    with pytest.raises(ValueError, match="reserved"):
        store.registry.record(attempt)
    store.settle(failure(state), "failed", "test failure")
    next_state = store.claim({**job, "id": "second"}, root)
    assert next_state["job"]["reserved"]["cpu_seconds"] == 48
    assert next_state["job"]["reserved"]["sim_agent_steps"] == 490
    assert next_state["job"]["reserved"]["trials"] == 2


def test_simulation_reserves_evaluation_too(store_and_job):
    store, job, root = store_and_job
    job["max_train_agent_steps"] = 480
    with pytest.raises(ValueError, match="evaluation"):
        store.claim(job, root)


def test_cancellation_wins_before_atomic_settlement(store_and_job):
    store, job, root = store_and_job
    state = store.claim(job, root)
    store.cancel(job["id"])
    state = failure(state)
    state["evaluation"] = {"goal_per_actor": 0.3}
    final = store.settle(state, "completed")
    assert final["status"] == "cancelled"
    row = store.registry.public_summary()["experiments"][0]["attempts"][0]
    assert row["status"] == "cancelled" and row["candidate_model"] is None
    with pytest.raises(ValueError, match="settled"):
        store.settle(state, "completed")


def test_same_protocol_and_initialization_are_bound(store_and_job):
    store, job, root = store_and_job
    state = store.claim(job, root)
    store.settle(failure(state), "failed", "test")
    with pytest.raises(ValueError, match="protocol"):
        store.claim({**job, "id": "changed", "task_set": "different"}, root)
    with pytest.raises(ValueError, match="initialization"):
        store.claim({**job, "id": "new-seed", "seed": 5}, root)


def test_concurrent_claim_is_atomic(store_and_job):
    store, job, root = store_and_job
    def claim(i):
        try:
            return store.claim({**job, "id": f"job{i}"}, root)["id"]
        except ValueError:
            return None
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(claim, range(2)))
    assert sum(x is not None for x in results) == 1


def test_missing_evaluation_does_not_complete(store_and_job):
    store, job, root = store_and_job
    state = failure(store.claim(job, root))
    with pytest.raises(ValueError, match="metric"):
        store.settle(state, "completed")
    assert store.get(job["id"])["status"] == "queued"


def test_changed_weight_file_cannot_reuse_initial_model_identity(store_and_job):
    store, job, root = store_and_job
    job.update(initial_weights="/model.pt", sources=[{"path": "/model.pt", "size": 10, "mtime_ns": 1}])
    state = store.claim(job, root)
    store.settle(failure(state), "failed", "test")
    job["sources"][0]["mtime_ns"] = 2
    with pytest.raises(ValueError, match="initialization"):
        store.claim({**job, "id": "changed"}, root)


def test_previously_overspent_resource_blocks_another_trial(store_and_job):
    store, job, root = store_and_job
    state = failure(store.claim(job, root))
    state["cost"]["rendered_frames"] = 1
    store.settle(state, "failed", "test overspend")
    with pytest.raises(ValueError, match="exhausted"):
        store.claim({**job, "id": "again"}, root)


def test_unknown_settlement_status_rejected(store_and_job):
    store, job, root = store_and_job
    state = store.claim(job, root)
    with pytest.raises(ValueError, match="terminal"):
        store.settle(state, "typo")
    assert store.get(job["id"])["status"] == "queued"
