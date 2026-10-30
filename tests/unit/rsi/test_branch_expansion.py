import json
from pathlib import Path
import numpy as np
import pytest
from driversi.rsi.experiments.branch_expansion import candidate_grid, trajectory, novelty, _execute

def test_candidate_count_not_success_count_and_native_reference_is_not_self_play():
    c = candidate_grid()
    assert len(c) == 26
    assert len({(x['arrival_backend'],tuple(x['arrival_action'])) for x in c}) == 26
    assert all(abs(v) <= .25 for x in c for v in x['arrival_action'])
    assert c[0]['arrival_backend'] == 'driverl_native'

def test_diversity_requires_new_history_and_consumed_future():
    h=np.zeros((4,2));f=np.zeros((8,2));prior=[('old',(h,f))]
    thresholds=dict(history_rms_m=.15,future_rms_m=.5)
    assert not novelty((h,f+1),prior,thresholds)[0]
    assert not novelty((h+1,f),prior,thresholds)[0]
    assert not novelty((h+.01,f+.01),prior,thresholds)[0]
    assert novelty((h+.2,f+.6),prior,thresholds)[0]

def test_native_eight_point_window_ignores_later_divergence(tmp_path):
    a=np.zeros((71,1,13));b=a.copy();b[65:,0,0]=100
    for name,arr in [('a',a),('b',b)]:np.savez(tmp_path/(name+'.npz'),actor_ids=['ego'],states=arr)
    x,y=trajectory(tmp_path/'a.npz',8),trajectory(tmp_path/'b.npz',8)
    np.testing.assert_array_equal(x[1],y[1])
    assert not np.array_equal(trajectory(tmp_path/'a.npz',10)[1],trajectory(tmp_path/'b.npz',10)[1])

def test_truncated_or_nonfinite_physical_rollout_rejected(tmp_path):
    p=tmp_path/'bad.npz';np.savez(p,actor_ids=['ego'],states=np.zeros((61,1,13)))
    with pytest.raises(ValueError,match='Full finite'):trajectory(p)
    a=np.zeros((71,1,13));a[1,0,0]=np.nan;np.savez(p,actor_ids=['ego'],states=a)
    with pytest.raises(ValueError,match='Full finite'):trajectory(p)

def test_unresolved_task_is_not_relaunched(tmp_path):
    with pytest.raises(ValueError,match='recovery'):
        _execute(tmp_path,dict(id='x',status='running'))

def test_rendered_branch_does_not_earn_native_sample_qualification(tmp_path,monkeypatch):
    from driversi.rsi.experiments.branch_expansion import _admit_reports
    import driversi_agents.augustus.branch_supervision as native
    report=dict(target_qualified_samples=20,rows=[dict(rendered=True,distinct=True,
                render_task='x',trainable=False,visual_admitted=False)])
    _admit_reports(tmp_path,report,[])
    assert report['trainable_samples']==0 and report['qualified_sample_shortfall']==20
    p=tmp_path/'sample.json';p.write_text('{}')
    import driversi.rsi.experiments.branch_expansion as expansion
    monkeypatch.setattr(expansion,'_validate_batch_admission',lambda *a:None)
    monkeypatch.setattr(native,'validate_sample',lambda path:dict(sample_id='s',branch_report=str(tmp_path/'tasks/x/select/report.json')))
    _admit_reports(tmp_path,report,[p]);assert report['trainable_samples']==1
    with pytest.raises(ValueError,match='duplicate'):_admit_reports(tmp_path,report,[p,p])


def test_preflight_rejects_stale_completed_artifacts_before_execution(monkeypatch):
    import driversi.rsi.experiments.branch_expansion as module
    def reject(*a):raise ValueError('stale sealed artifact')
    monkeypatch.setattr(module,'verify',reject)
    completed=dict(id='old',status='completed',stages=[dict(id='x',result={})],
                   task={'steps':[dict(id='x')]})
    with pytest.raises(ValueError,match='stale'):
        module._preflight([dict(status='registered'),completed])
    with pytest.raises(ValueError,match='Reconcile'):
        module._preflight([dict(status='running'),completed])


def test_external_rerender_cannot_earn_batch_admission(tmp_path,monkeypatch):
    import driversi.rsi.experiments.branch_expansion as module
    state=dict(status='completed',stages=[dict(id=k,result={'artifacts':{'report':str(tmp_path/(k+'.json'))}})
        for k in ['select','render','source']])
    class Store:
        def __init__(self,*a):pass
        def get(self,*a):return state
    monkeypatch.setattr(module,'HarnessStore',Store)
    monkeypatch.setattr(module,'_verify_completed',lambda *a:None)
    state['stages'][0]['result']['artifacts']['trajectory_window']=str(tmp_path/'window.json')
    state['stages'][1]['result']['artifacts']['render_report']=str(tmp_path/'actual.json')
    wrapper={'render_report':{'nested':'actual producer schema'},'trajectory_window':{'history':[]}}
    (tmp_path/'render.json').write_text(json.dumps(wrapper))
    visual=dict(branch_report=str(tmp_path/'select.json'),render_report=str(tmp_path/'actual.json'),
                trajectory_window=str(tmp_path/'window.json'),source_comparison=str(tmp_path/'source.json'))
    q=tmp_path/'visual.json';q.write_text(json.dumps(visual))
    module._validate_batch_admission(tmp_path,{'render_task':'r'},{'visual_admission':str(q)})
    visual['render_report']=str(tmp_path/'external-rerender.json');q.write_text(json.dumps(visual))
    with pytest.raises(ValueError,match='another render'):
        module._validate_batch_admission(tmp_path,{'render_task':'r'},{'visual_admission':str(q)})


def test_supervisor_stops_when_coordinator_exits(tmp_path):
    import os
    import signal
    import subprocess
    import sys
    import time
    if sys.platform != 'linux':
        pytest.skip('Linux parent-death ownership')
    child_code = """
import os,sys,time
from pathlib import Path
import driversi.rsi.workflow.engine as engine
from driversi.rsi.experiments.branch_expansion import _supervise
root=Path(sys.argv[1])
def execute(*args):
    (root/'ready').write_text(str(os.getpid()))
    try:
        while True:time.sleep(.05)
    except KeyboardInterrupt:
        (root/'cleaned').write_text('cancelled')
        return {'status':'cancelled'}
engine.run=execute
raise SystemExit(_supervise(os.getppid(),str(root/'fake.sqlite'),'fake'))
"""
    parent_code = "import subprocess,sys,time;subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2]]);time.sleep(30)"
    parent = subprocess.Popen([sys.executable, '-c', parent_code, child_code, str(tmp_path)],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    child_pid = None
    try:
        deadline = time.monotonic()+10
        while not (tmp_path/'ready').exists() and time.monotonic()<deadline:
            time.sleep(.05)
        assert (tmp_path/'ready').exists(), 'supervisor did not start'
        child_pid = int((tmp_path/'ready').read_text())
        parent.terminate();parent.wait(timeout=3)
        deadline = time.monotonic()+5
        while not (tmp_path/'cleaned').exists() and time.monotonic()<deadline:
            time.sleep(.05)
        assert (tmp_path/'cleaned').read_text() == 'cancelled'
    finally:
        if parent.poll() is None:
            parent.terminate();parent.wait(timeout=3)
        if child_pid:
            try:os.kill(child_pid, signal.SIGTERM)
            except ProcessLookupError:pass
