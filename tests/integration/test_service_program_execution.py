"""Real subprocess checks: frozen code guards stop downstream work."""
import pytest
pytest.importorskip("driversi.sim._native")
from driversi.rsi.workflow.engine import run
from tests.integration.test_driveharness_execution import task_case


def test_changed_code_after_real_stage_blocks_downstream(straight_scene,tmp_path):
    registry=task_case(straight_scene,tmp_path,schema=2)
    calls=[]
    def guard():
        calls.append(1)
        if len(calls)==2:raise ValueError("frozen_program_changed")
    result=run(registry,"workflow",evidence_guard=guard)
    assert result["status"]=="failed"
    assert result["failure_reason"]=="frozen_program_changed"
    assert result["sim_agent_steps"]==140
    assert result["stages"][0]["status"]=="completed"
    assert result["stages"][1]["status"]=="blocked"


def test_guard_checks_before_and_after_each_real_stage(straight_scene,tmp_path):
    registry=task_case(straight_scene,tmp_path,schema=2)
    calls=[]
    result=run(registry,"workflow",evidence_guard=lambda:calls.append(1))
    assert result["status"]=="completed" and len(calls)==4
