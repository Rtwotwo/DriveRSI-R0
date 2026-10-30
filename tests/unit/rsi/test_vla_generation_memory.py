import pytest

from driversi.rsi.decisions.vla_memory import SceneSkillMemory, generation_schedule, make_case


def test_three_modes_keep_teacher_and_admission_fixed():
    match = dict(
        case_id="observed-contact",
        suggested_base_skill="contact_trace_audit",
        shared_conditions=["nearby_objects_dense"],
    )
    none = generation_schedule("no_memory", [])
    fixed = generation_schedule("fixed", [])
    memory = generation_schedule("memory", [match])
    assert not none["preflight"]
    assert fixed["preflight"] == memory["preflight"] == ["initial_contact_audit"]
    assert none["mandatory_postconditions"] == memory["mandatory_postconditions"]
    assert none["teacher_backend"] == fixed["teacher_backend"] == memory["teacher_backend"]
    with pytest.raises(ValueError):
        generation_schedule("no_memory", [match])


def test_generic_straight_driving_similarity_cannot_trigger_contact_skill():
    match = dict(
        case_id="irrelevant-contact",
        suggested_base_skill="contact_trace_audit",
        shared_conditions=["ego_moving", "history_near_straight"],
    )
    schedule = generation_schedule("memory", [match])
    assert not schedule["preflight"]
    assert not schedule["applicable_case_ids"]
    assert schedule["excluded_case_ids"] == ["irrelevant-contact"]


def test_fresh_memory_records_cost_without_learning_credit(tmp_path):
    args = (
        dict(parent="same"),
        dict(conditions=["nearby_objects_sparse"]),
        dict(source_split="navtrain", source_group="train-log"),
        dict(failed_physical_checks=[], learning_blockers=["visual"]),
        "executed-identity",
    )
    with pytest.raises(ValueError):
        make_case(*args, contacts_present=False, fresh_execution=True)
    case = make_case(
        *args,
        contacts_present=False,
        fresh_execution=True,
        cost=dict(generation_wall_seconds=2.0, sim_agent_steps=500),
    )
    memory = SceneSkillMemory(tmp_path / "memory.sqlite", "evolve")
    assert memory.record(case)["recorded"]
    assert not memory.record(case)["recorded"]
    memory.close()
    memory = SceneSkillMemory(tmp_path / "memory.sqlite", "frozen")
    assert len(memory.cases()) == 1
    assert memory.cases()[0]["student_gain_credited"] is False
    assert "fresh_privileged_generation" in memory.cases()[0]["evidence_scope"]
    assert not memory.record(case)["recorded"]
    memory.close()
