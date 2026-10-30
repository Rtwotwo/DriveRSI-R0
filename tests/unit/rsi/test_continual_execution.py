"""Finite orchestration tests; mocked phases are not learning experiments."""
import json
from pathlib import Path

import pytest

from driversi.rsi.experiments import continual_curriculum as cc
from driversi.rsi.experiments import continual_execution as auto


def initial(tmp_path, status='queued'):
    cases=[dict(id=f'problem-{i}',query=f'private-{i}',status=status if i==0 else 'queued',
                round=0,target=10,failure_classes=[],candidates=[dict(token=f't{i}-{j}',
                source_log=f'log{i}-{j}') for j in range(3)]) for i in range(2)]
    state=dict(model='chainflow_stage1',incumbent=dict(revision=0,pdms=94),cases=cases,
        active=None if status=='queued' else cases[0]['id'],
        policy=dict(max_candidates=3,targets=[10,20],max_rounds=2,mastery_pdms=.95,
                    attempts_per_source=32,replay_limit=32))
    (tmp_path/'private').mkdir();cc.save(tmp_path,state)
    return state


def read(root):return json.loads((root/'private/state.json').read_text())


def test_automatic_path_uses_native_phases_and_stops_at_problem_budget(tmp_path,monkeypatch):
    initial(tmp_path);calls=[]
    def phase(name,next_status):
        def perform(root):
            state=read(root)
            case=next((c for c in state['cases'] if c['id']==state['active']),state['cases'][0])
            state['active']=case['id'];case['status']=next_status
            if name=='settle':state['active']=None
            calls.append(name);cc.save(root,state)
        return perform
    for name,status in [('prepare','ready'),('dispatch','data_ready'),('train_current','awaiting_release'),('settle','solved')]:
        monkeypatch.setattr(cc,name,phase(name,status))
    monkeypatch.setattr(auto,'required_gpus',lambda *a:[6])
    monkeypatch.setattr(auto,'gpu_snapshot',lambda:{6:0})
    result=auto.run(tmp_path,max_problems=1)
    assert calls==['prepare','dispatch','train_current','settle']
    assert result['automatic_execution']['status']=='problem_budget_completed'
    assert result['automatic_execution']['transitions']==4
    assert read(tmp_path)['cases'][1]['status']=='queued'


@pytest.mark.parametrize('status,reason',[('blocked_assets','blocked_assets'),
    ('unknown_execution','needs_reconciliation'),('dispatching','needs_reconciliation'),
    ('training','needs_reconciliation'),('generation_stopped','generation_shortage')])
def test_unknown_or_blocked_work_is_never_reexecuted(tmp_path,monkeypatch,status,reason):
    initial(tmp_path,status)
    for name in ('prepare','dispatch','train_current','settle'):
        monkeypatch.setattr(cc,name,lambda *a:pytest.fail('Blocked work repeated'))
    assert auto.run(tmp_path)['automatic_execution']['status']==reason
    assert auto.run(tmp_path)['automatic_execution']['transitions']==0


@pytest.mark.parametrize('memory',[500,715,96000])
def test_busy_gpu_blocks_before_native_dispatch(tmp_path,monkeypatch,memory):
    initial(tmp_path,'ready')
    monkeypatch.setattr(auto,'required_gpus',lambda *a:[6,7])
    monkeypatch.setattr(auto,'gpu_snapshot',lambda:{6:0,7:memory})
    monkeypatch.setattr(cc,'dispatch',lambda *a:pytest.fail('Busy GPU was used'))
    result=auto.run(tmp_path)
    assert result['automatic_execution']['status']=='blocked_resources'
    assert read(tmp_path)['cases'][0]['status']=='ready'


def test_failed_gpu_inventory_is_fail_closed(tmp_path,monkeypatch):
    initial(tmp_path,'data_ready')
    monkeypatch.setattr(auto,'required_gpus',lambda *a:[6])
    def failed():raise RuntimeError('inventory unavailable')
    monkeypatch.setattr(auto,'gpu_snapshot',failed)
    assert auto.run(tmp_path)['automatic_execution']['status']=='resource_check_failed'


def test_transition_cap_stops_even_a_non_progressing_phase(tmp_path,monkeypatch):
    initial(tmp_path)
    monkeypatch.setattr(cc,'prepare',lambda *a:None)
    assert auto.run(tmp_path,max_transitions=2)['automatic_execution']['transitions']==2
    assert read(tmp_path)['automatic_execution']['status']=='transition_budget_completed'


def test_stop_marker_and_candidate_contract(tmp_path):
    initial(tmp_path)
    (tmp_path/'STOP_AUTOMATIC').touch()
    assert auto.run(tmp_path)['automatic_execution']['status']=='stopped_by_request'
    state=read(tmp_path);state['cases'][0]['candidates'].append(state['cases'][0]['candidates'][0])
    cc.save(tmp_path,state)
    with pytest.raises(ValueError,match='candidate limits'):auto.run(tmp_path)


def test_runtime_failure_is_saved_without_retry(tmp_path,monkeypatch):
    initial(tmp_path,'ready');calls=[]
    monkeypatch.setattr(auto,'required_gpus',lambda *a:[6])
    monkeypatch.setattr(auto,'gpu_snapshot',lambda:{6:0})
    def fail(root):
        calls.append(1);state=read(root);state['cases'][0]['status']='unknown_execution'
        cc.save(root,state);raise RuntimeError('Native result unknown')
    monkeypatch.setattr(cc,'dispatch',fail)
    with pytest.raises(RuntimeError):auto.run(tmp_path)
    assert read(tmp_path)['automatic_execution']['status']=='execution_failed'
    assert auto.run(tmp_path)['automatic_execution']['status']=='needs_reconciliation'
    assert calls==[1]


@pytest.mark.parametrize('terminal,reason',[('blocked_assets','blocked_assets'),
                                          ('unknown_execution','needs_reconciliation')])
def test_final_transition_preserves_block_instead_of_budget_success(tmp_path,monkeypatch,terminal,reason):
    initial(tmp_path)
    def last(root):
        state=read(root);state['active']=state['cases'][0]['id'];state['cases'][0]['status']=terminal
        cc.save(root,state)
    monkeypatch.setattr(cc,'prepare',last)
    result=auto.run(tmp_path,max_transitions=1)
    assert result['automatic_execution']['status']==reason
    assert result['automatic_execution']['transitions']==1


@pytest.mark.parametrize('promotion',[True,False])
def test_shortage_target_retry_full_release_then_next_problem(tmp_path,monkeypatch,promotion):
    """Mock native phases: verifies routing, never reports an ML result."""
    from driversi.rsi.experiments import generation_recovery
    state=initial(tmp_path)
    state.update(retained=[],replay=[],rounds=[])
    state['policy'].update(strict_target=True,mastery_pdms=0,replay_limit=0)
    cc.save(tmp_path,state)
    (tmp_path/'generation-recovery-policy.json').write_text('{}')
    calls=[];dispatches={};trains={};parents=[]
    def prepare(root):
        state=read(root)
        c=next((c for c in state['cases'] if c['id']==state['active']),None)
        if c is None:c=next(c for c in state['cases'] if c['status']=='queued')
        state['active']=c['id'];c['status']='ready';calls.append((c['id'],'prepare'))
        parents.append((c['id'],state['incumbent']['revision']));cc.save(root,state)
    def dispatch(root):
        state=read(root);c=next(c for c in state['cases'] if c['id']==state['active'])
        n=dispatches[c['id']]=dispatches.get(c['id'],0)+1
        c['status']='generation_stopped' if c['id']=='problem-0' and n==1 else 'data_ready'
        calls.append((c['id'],'dispatch'));cc.save(root,state)
    def revise(root):
        state=read(root);c=next(c for c in state['cases'] if c['id']==state['active'])
        assert c['status']=='generation_stopped'
        c['status']='needs_revision';calls.append((c['id'],'revise'));cc.save(root,state)
    def train(root):
        state=read(root);c=next(c for c in state['cases'] if c['id']==state['active'])
        assert c['status']=='data_ready'
        n=trains[c['id']]=trains.get(c['id'],0)+1
        if c['id']=='problem-0' and n==1:
            cc.apply_verdict(state,c,dict(promoted=False,target_success=False),None,[])
        else:c['status']='awaiting_release'
        calls.append((c['id'],'train'));cc.save(root,state)
    def settle(root):
        state=read(root);c=next(c for c in state['cases'] if c['id']==state['active'])
        cc.apply_verdict(state,c,dict(promoted=promotion,target_success=True),
                         dict(revision=state['incumbent']['revision']+1,pdms=95),[])
        calls.append((c['id'],'settle'));cc.save(root,state)
    for name,fn in [('prepare',prepare),('dispatch',dispatch),('train_current',train),('settle',settle)]:
        monkeypatch.setattr(cc,name,fn)
    monkeypatch.setattr(generation_recovery,'revise',revise)
    monkeypatch.setattr(auto,'required_gpus',lambda *a:[6])
    monkeypatch.setattr(auto,'gpu_snapshot',lambda:{6:0})
    result=auto.run(tmp_path,max_problems=2,max_transitions=32)
    assert result['automatic_execution']['status']=='finished'
    assert ('problem-0','revise') in calls
    assert calls.index(('problem-0','settle')) < calls.index(('problem-1','prepare'))
    assert ('problem-1',1 if promotion else 0) in parents
    assert trains=={'problem-0':2,'problem-1':1}


def test_frozen_queue_can_exceed_initial_three_problem_pilot(tmp_path,monkeypatch):
    state=initial(tmp_path)
    import copy
    for i in range(2,5):
        c=copy.deepcopy(state['cases'][0]);c.update(id=f'problem-{i}',query=f'private-{i}')
        state['cases'].append(c)
    cc.save(tmp_path,state);seen=[]
    def complete(root):
        s=read(root);c=next(c for c in s['cases'] if c['status']=='queued')
        c['status']='solved';seen.append(c['id']);cc.save(root,s)
    monkeypatch.setattr(cc,'prepare',complete)
    result=auto.run(tmp_path,max_problems=5,max_transitions=30)
    assert result['automatic_execution']['status']=='finished' and len(seen)==5
    with pytest.raises(ValueError,match='frozen curriculum'):
        auto.run(tmp_path,max_problems=6,max_transitions=30)


def test_detached_unknown_blocks_but_known_shortage_advances(tmp_path):
    state=initial(tmp_path);state['cases'][0]['status']='needs_execution_reconciliation'
    assert auto.next_operation(state)==(None,'needs_reconciliation')
    state['cases'][0]['status']='generation_budget_exhausted'
    assert auto.next_operation(state)==('prepare',None)
