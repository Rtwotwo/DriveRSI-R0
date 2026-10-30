"""Preparation at an exhausted learning budget must release queue ownership."""
import json
from driversi.rsi.experiments import continual_curriculum as cc
from driversi.rsi.experiments.continual_execution import next_operation


def test_exhausted_active_problem_does_not_pin_queue(tmp_path):
    state=dict(model='chainflow_stage1',incumbent=dict(revision=0,pdms=94),
        active='p0',policy=dict(max_candidates=3,targets=[10,20],max_rounds=2,
            mastery_pdms=0,attempts_per_source=32,replay_limit=0),
        cases=[dict(id='p0',status='needs_revision',round=2,target=50,candidates=[],failure_classes=[]),
               dict(id='p1',status='queued',round=0,target=50,candidates=[],failure_classes=[])])
    (tmp_path/'private').mkdir()
    (tmp_path/'declaration.json').write_text(json.dumps(dict(inputs={})))
    cc.save(tmp_path,state)
    cc.prepare(tmp_path)
    saved=json.loads((tmp_path/'private/state.json').read_text())
    assert saved['active'] is None and saved['cases'][0]['status']=='budget_exhausted'
    assert next_operation(saved)==('prepare',None)
