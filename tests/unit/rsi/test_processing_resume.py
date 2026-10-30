import json
from pathlib import Path
import pytest
from driversi.agents.backends.native import digest
from driversi.rsi.experiments.processing_resume import settled_generation


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return path


def fixture_wave(tmp_path):
    plan=put(tmp_path/'trajectory-family-plan.json',dict(assignments=[dict(id='a',token='train',source_log='log')]))
    folder=tmp_path/'candidates/a/generation'
    native=put(folder/'native.json',dict(status='completed',source_files={}))
    teacher=dict(status='execution_or_admission_rejected',harness_status='completed',
                 plan_sha256=digest(plan),assignment_id='a',token='train',source_log='log',teacher_report=str(native))
    put(folder/'report.json',dict(status='teacher_rejected',teacher=teacher))
    put(folder.parent/'physical-prefetch.json',dict(status='completed',started=1,finished=2,pid=123))
    put(tmp_path/'physical-prefetch-status.json',dict(submitted=1,completed=1,errors={'a':'KeyboardInterrupt'},
          jobs=dict(a=dict(done=True,output=str(folder)))))
    return tmp_path


def test_completed_prefetch_reuses_exact_result_without_changing_cost_receipt(tmp_path,monkeypatch):
    from driversi.rsi.experiments.reactive_prefetch import physical_job
    from driversi.rsi.experiments import reactive_generation
    wave=fixture_wave(tmp_path);folder=wave/'candidates/a/generation';report=folder.parent/'physical-prefetch.json'
    setup=dict(asset={'token':'train'},plan_sha256=digest(wave/'trajectory-family-plan.json'),
               config=dict(input=dict(database=str(tmp_path/'catalog'))))
    put(folder/'generation.json',setup)
    monkeypatch.setattr(reactive_generation,'generate',lambda **_:pytest.fail('must not repeat physical work'))
    raw=report.read_bytes()
    job=dict(output=str(folder),render=False,asset=setup['asset'],plan_path=str(wave/'trajectory-family-plan.json'),
             assignment_id='a',database=str(tmp_path/'catalog'))
    assert physical_job(job)['status']=='teacher_rejected'
    assert physical_job(job)['status']=='teacher_rejected'
    assert report.read_bytes()==raw


def test_interrupt_after_terminal_receipt_can_be_reconciled(tmp_path):
    wave=fixture_wave(tmp_path);result=settled_generation(wave)
    assert result['known_results']==1 and not result['completed_work_reexecuted']
    assert result['interrupted_waits_reconciled']==['a']


@pytest.mark.parametrize('fault',['inflight','platform_error','unknown_native','wrong_source','unsealed_outcome'])
def test_unknown_or_foreign_work_never_resumes(tmp_path,fault):
    wave=fixture_wave(tmp_path)
    if fault in ('inflight','platform_error'):
        p=wave/'physical-prefetch-status.json';v=json.loads(p.read_text())
        if fault=='inflight':v['completed']=0
        else:v['errors']['a']='TimeoutError'
        put(p,v)
    elif fault=='unknown_native':put(wave/'candidates/a/generation/native.json',dict(status='running'))
    elif fault=='wrong_source':
        p=wave/'candidates/a/generation/report.json';v=json.loads(p.read_text());v['teacher']['token']='other';put(p,v)
    else:put(wave/'candidates/a/outcome.json',dict(status='teacher_rejected'))
    with pytest.raises(ValueError):settled_generation(wave)
