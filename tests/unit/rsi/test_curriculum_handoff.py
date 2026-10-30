"""Controller boundary tests with no physical, GPU or paid execution."""
from contextlib import contextmanager
import json
from pathlib import Path
import pytest
from driversi.rsi.experiments import curriculum_handoff as handoff
from driversi.rsi.experiments import continual_curriculum as cc, continual_execution as auto


def test_owned_boundary_wait_counts_against_deadline(tmp_path,monkeypatch):
    root=tmp_path/'exp/campaign';root.mkdir(parents=True)
    out=root/'automatic/handoff'
    monkeypatch.setattr(handoff,'project_root',lambda:tmp_path)
    tick=[1000.]
    monkeypatch.setattr(handoff.time,'time',lambda:tick[0])
    monkeypatch.setattr(handoff.time,'sleep',lambda n:tick.__setitem__(0,tick[0]+n))
    identity=dict(pid=99,birth='x');calls=[identity,identity,None]
    monkeypatch.setattr(handoff,'process_identity',lambda *a:calls.pop(0))
    @contextmanager
    def locked(r):yield r
    monkeypatch.setattr(cc,'locked',locked)
    monkeypatch.setattr(handoff,'settled',lambda r:dict(known_results=3))
    def run(r,**kw):
        assert kw==dict(max_problems=5,max_transitions=100,max_wall_seconds=90)
        assert not (root/'STOP_AUTOMATIC').exists()
        return dict(automatic_execution=dict(status='finished'))
    monkeypatch.setattr(auto,'run',run)
    handoff.run(root,99,out,max_problems=5,max_transitions=100,deadline=1100)
    assert json.loads((out/'status.json').read_text())['automatic_status']=='finished'
    assert (out/'boundary-stop.txt').exists()


def test_unrelated_stop_is_preserved(tmp_path,monkeypatch):
    root=tmp_path/'exp/campaign';root.mkdir(parents=True)
    (root/'STOP_AUTOMATIC').write_text('user_stop')
    monkeypatch.setattr(handoff,'project_root',lambda:tmp_path)
    with pytest.raises(ValueError,match='Existing stop'):
        handoff.run(root,99,root/'automatic/h',max_problems=2,max_transitions=18,deadline=handoff.time.time()+100)
    assert (root/'STOP_AUTOMATIC').read_text()=='user_stop'


def test_unknown_work_never_relaunches(tmp_path):
    (tmp_path/'private').mkdir()
    (tmp_path/'private/state.json').write_text(json.dumps(dict(active=None,cases=[dict(id='x',status='training')])))
    with pytest.raises(ValueError,match='unresolved'):handoff.settled(tmp_path)
