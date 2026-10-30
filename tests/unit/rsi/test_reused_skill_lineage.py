"""Receipt reuse must preserve executed skill identity, never rewrite physical data."""

import json
from types import SimpleNamespace

import pytest

from driversi.agents.backends.native import digest
from driversi.data.catalog.store import identity
from driversi.rsi.decisions.capability_skills import validate_sample_lineage


def test_reused_plan_requires_matching_context_controls_and_scene(tmp_path, monkeypatch):
    from driversi.contracts.scene import Scene

    def save(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value))
        return path

    def seal(value):
        return dict(value, plan_id=identity(value))

    assignment = dict(
        id="a-v000",
        skill="driverl_full_window",
        skill_version=1,
        source_log="train",
        token="t",
        sampling_seed=42,
    )
    context = dict(model="chainflow_stage1", parent_checkpoint="parent")
    old = seal(dict(kind="capability_skill_plan", assignments=[assignment], context=context))
    old_path = save("old-plan.json", old)
    scene = tmp_path / "scene.npz"
    scene.write_bytes(b"unchanged executed scene fixture")
    branch = save("branch.json", dict(files={str(scene): digest(scene)}))
    generation = save(
        "generation.json",
        dict(
            status="rendered",
            teacher=dict(
                plan_path=str(old_path),
                plan_sha256=digest(old_path),
                assignment_id=assignment["id"],
            ),
            branch_report=str(branch),
            files={str(branch): digest(branch)},
        ),
    )
    plan = seal(
        dict(
            kind="capability_skill_plan",
            assignments=[assignment],
            context=context,
            reused_executions={
                assignment["id"]: dict(path=str(generation), sha256=digest(generation))
            },
        )
    )
    proof = save(
        "admission.json",
        dict(
            parent=dict(source_log="train", token="t"),
            branch=dict(scene=str(scene)),
            files={str(scene): digest(scene)},
        ),
    )
    row = dict(kind="synthetic_navtrain", admission=dict(path=str(proof), sha256=digest(proof)))
    files = {str(proof): digest(proof), str(scene): digest(scene)}
    actual = dict(
        plan_id=old["plan_id"], assignment_id="a-v000", skill="driverl_full_window", skill_version=1
    )
    monkeypatch.setattr(
        Scene, "load", lambda p: SimpleNamespace(metadata={"capability_skill": actual})
    )
    assert validate_sample_lineage(plan, [row], files) == ["a-v000"]
    changed = {k: v for k, v in plan.items() if k != "plan_id"}
    changed["context"] = dict(context, parent_checkpoint="different-parent")
    with pytest.raises(ValueError, match="parent context"):
        validate_sample_lineage(seal(changed), [row], files)
    # New proposal IDs can collide with a different old ID. Resolve the sealed
    # old plan/assignment pair, never the same-looking ID in the new selection.
    renamed = {k: v for k, v in plan.items() if k != "plan_id"}
    renamed["assignments"] = [dict(assignment, id="new-v007"), dict(assignment, sampling_seed=99)]
    renamed["reused_executions"] = {"new-v007": plan["reused_executions"]["a-v000"]}
    assert validate_sample_lineage(seal(renamed), [row], files) == ["new-v007"]
    changed_controls = dict(
        renamed, assignments=[dict(renamed["assignments"][0], sampling_seed=100)]
    )
    with pytest.raises(ValueError, match="controls"):
        validate_sample_lineage(seal(changed_controls), [row], files)
    ambiguous = dict(
        renamed,
        assignments=[dict(assignment, id="first"), dict(assignment, id="second")],
        reused_executions={x: plan["reused_executions"]["a-v000"] for x in ["first", "second"]},
    )
    with pytest.raises(ValueError, match="unambiguous"):
        validate_sample_lineage(seal(ambiguous), [row], files)
    generation.write_text(generation.read_text() + " ")
    with pytest.raises(ValueError, match="receipt changed"):
        validate_sample_lineage(plan, [row], files)
