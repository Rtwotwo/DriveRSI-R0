import pytest
from driversi_agents.chainflow.sampling import GlobalEpochCursor,distributed_settings


def rows(n=11,s=2):
    return [{'kind':'original_navtrain'} for _ in range(n)]+[{'kind':'synthetic_navtrain'} for _ in range(s)]


def test_epoch_covers_every_original_and_marks_padding():
    c=GlobalEpochCursor(rows(),42,2,2,4);seen=[];padded=[]
    while c.epoch==0:
        ids,info=c.next_batch()
        assert len(ids)==4
        seen.extend(i for i,p in zip(ids,info['padding']) if not p)
        padded.extend(i for i,p in zip(ids,info['padding']) if p)
    assert sorted(i for i in seen if i<11)==list(range(11))
    assert len(seen)==14 and len(padded)==2 and c.offset==0


def test_cursor_resume_across_epoch_is_exact():
    a=GlobalEpochCursor(rows(),7,1,2);a.next_batch();a.next_batch()
    b=GlobalEpochCursor(rows(),7,1,2);b.load_state_dict(a.state_dict())
    for _ in range(30):assert a.next_batch()==b.next_batch()
    assert a.state_dict()==b.state_dict()


def test_resume_refuses_rank_change_or_offset_corruption():
    a=GlobalEpochCursor(rows(),7,1,2);a.next_batch();state=a.state_dict()
    with pytest.raises(ValueError):GlobalEpochCursor(rows(),7,1,1).load_state_dict(state)
    with pytest.raises(ValueError):a.load_state_dict({**state,'offset':1})


def test_full_population_traversal_without_expensive_io():
    c=GlobalEpochCursor(rows(103288,3),42,4,8);seen=set()
    while c.epoch==0:
        ids,info=c.next_batch()
        seen.update(i for i,p in zip(ids,info['padding']) if i<103288 and not p)
    assert len(seen)==103288


@pytest.mark.parametrize('devices',[[6,6],[],[-1],['6']])
def test_invalid_devices(devices):
    with pytest.raises(ValueError):distributed_settings({'distributed_training':dict(devices=devices,checkpoint_interval=2,synthetic_every=8)})


def test_interval_checkpoint_requires_complete_local_publication(tmp_path):
    from driversi.training.orchestration.lifecycle import verify_interval_checkpoint
    from driversi.agents.backends.native import digest
    folder = tmp_path / "checkpoints" / "step-1"
    folder.mkdir(parents=True)
    model, recovery = folder / "model.ckpt", folder / "recovery.pt"
    model.write_bytes(b"model"); recovery.write_bytes(b"optimizer and ranks")
    plan = {"output": str(tmp_path), "context": "context", "resolved": {"request": {"budget": {"max_optimizer_steps": 2}}}}
    receipt = {"context": "context", "recovery_schema_version": 3, "updates": 1,
               "checkpoint": str(model), "recovery_checkpoint": str(recovery),
               "checkpoint_files": {str(x): digest(x) for x in (model, recovery)}}
    verify_interval_checkpoint(receipt, plan)
    recovery.write_bytes(b"truncated")
    with pytest.raises(ValueError, match="publication mismatch"):
        verify_interval_checkpoint(receipt, plan)


def test_recovery_finishes_remaining_schedule_without_overrun():
    from driversi_agents.chainflow.sampling import remaining_updates
    assert remaining_updates(51644, 100, 51644) == 51544
    assert remaining_updates(64, 63, 2) == 1
    assert remaining_updates(64, 2, 2) == 2
    with pytest.raises(ValueError, match='already completed'):
        remaining_updates(64, 64, 2)


def test_epoch_request_accepts_full_epoch_budget():
    from driversi.training.contracts.job import validate_request
    request=dict(schema_version=4,id='epoch',model_config='model.yaml',mode='supervised',initialization='model_weights',data='data.json',pilot=dict(seed=42,batch_size=8,evaluation_steps=12146),budget=dict(max_wall_seconds=86400,max_optimizer_steps=51644,max_environment_steps=0,max_scoring_agent_steps=300000000))
    assert validate_request(request) is request


def test_failed_resume_cost_keeps_all_ancestors(tmp_path):
    import json
    from driversi.training.orchestration.lifecycle import accumulated_training_cost
    previous=None
    for i,status in enumerate(('completed','failed')):
        folder=tmp_path/str(i);folder.mkdir()
        receipt=folder/'training-result.json'
        receipt.write_text(json.dumps(dict(status=status,context='fixed',recoverable_checkpoint=status=='failed',consumed=dict(wall_seconds=10,scoring_agent_steps=100))))
        (folder/'training-plan.json').write_text(json.dumps(dict(phase='train' if i==0 else 'resume',previous=previous)))
        previous=str(receipt)
    cost=accumulated_training_cost(dict(previous=previous),dict(consumed=dict(wall_seconds=5,scoring_agent_steps=20)))
    assert cost['wall_seconds']==25 and cost['scoring_agent_steps']==220
