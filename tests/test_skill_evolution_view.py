import json
import sqlite3
import errno
import pytest
from pathlib import Path

from driversi.visualization.skill_evolution import project
from driversi.visualization.current import current_campaign
from driversi.visualization.skill_evolution import read


def put(root,name,value):
    p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))


def test_current_projection_excludes_private_case_material(tmp_path):
    root=tmp_path/'exp/current';run='runs/test/'
    plan=dict(task_id='test',episodes=1,skills={'version':'base'},initial_memory_revision=2,
              validation_drives_per_episode=3,total_scoring_agent_steps=12*80,max_doctor_requests=2,
              members=[dict(token='PRIVATE_TOKEN',role='development',source_log='PRIVATE_LOG')],
              credential_file='/PRIVATE_SECRET_PATH')
    put(root,'declaration/plan.json',plan)
    put(root,run+'report.json',dict(status='completed',elapsed_seconds=20,stages=[{'id':'decision-0','status':'completed','reason':'PRIVATE_DETAIL'}]))
    put(root,run+'observations/report.json',dict(caption_summary={'development_scenes':1,'attributes':7}))
    put(root,run+'proposal-0/report.json',dict(skill_id='following',experience_count=2,candidate={'skills':{'following':'Check the visible corridor.'}},rationale='PRIVATE_CAPTION'))
    put(root,run+'inference-0/report.json',dict(rows=12,parameters_frozen=True,optimizer_updates=0))
    comparisons={'incumbent':{'new_zero':[['PRIVATE_TOKEN',42],['PRIVATE_TOKEN',43]],'safety_regressions':[{'token':'PRIVATE_TOKEN'}]}}
    put(root,run+'decision-0/report.json',dict(active_version='base',memory_revision=3,
        decision={'accepted':False,'reasons':['development:new_zero_vs_incumbent'],
                  'cohorts':{'development':{'means':{'native':0.5,'incumbent':0.6,'candidate':0.4},'comparisons':comparisons}}}))
    out=current_campaign(dict(display_mode='current_only',current_skill_campaign='exp/current'),tmp_path)
    assert out['completed_predictions']==12 and out['weight_updates']==0 and out['memory_revision']==3
    assert out['episodes'][0]['failures']['development']['new_zero_scenes']==1
    assert out['episodes'][0]['means']['development']['candidate']==40
    assert 'PRIVATE_' not in json.dumps(out)


def test_no_receipts_are_not_reported_as_completed(tmp_path):
    out=project(tmp_path)
    assert out['status']=='preparing' and out['episodes']==[]


def test_old_live_synthesis_snapshot_cannot_replace_frozen_skill_view():
    root=Path(__file__).resolve().parents[1]
    source=(root/'src/visualization/static/app.js').read_text()
    assert "value.current?.kind!=='frozen_skill_evolution'" in source


def test_running_task_comes_from_readonly_journal(tmp_path):
    put(tmp_path,'declaration/plan.json',dict(task_id='live',episodes=0,skills={'version':'base'},
        total_scoring_agent_steps=0,max_doctor_requests=2,members=[]))
    with sqlite3.connect(tmp_path/'harness.sqlite') as db:
        db.execute('CREATE TABLE harness_tasks (id TEXT PRIMARY KEY,payload TEXT)')
        db.execute('INSERT INTO harness_tasks VALUES (?,?)',('live',json.dumps(dict(status='running',started_at=100))))
    out=project(tmp_path)
    assert out['status']=='running' and out['elapsed_seconds']>0


def test_completed_task_waits_for_final_decision_receipts(tmp_path):
    put(tmp_path,'declaration/plan.json',dict(task_id='settling',episodes=1,skills={'version':'base'},
        total_scoring_agent_steps=0,max_doctor_requests=2,members=[]))
    put(tmp_path,'runs/settling/report.json',dict(status='completed',elapsed_seconds=10))
    assert project(tmp_path)['status']=='finalizing_receipts'
    put(tmp_path,'runs/settling/decision-0/report.json',dict(active_version='base',memory_revision=1,
        decision={'accepted':False,'reasons':['rejected'],'cohorts':{}}))
    result=project(tmp_path)
    assert result['status']=='completed' and result['memory_revision']==1


def test_paired_projection_exposes_only_aggregate_verdicts(tmp_path):
    put(tmp_path,'comparison-config.json',{'python':'PRIVATE_RUNTIME'})
    put(tmp_path,'comparison-state.json',{'status':'completed','elapsed_seconds':42})
    put(tmp_path,'analysis/report.json',dict(shared_first_reply=True,paired_first_inference_equal=True,
        new_api_requests=3,conditions={'retain':{'private_caption':'PRIVATE_TEXT'}}))
    result=project(tmp_path)
    assert result['new_api_requests']==3 and len(result['conditions'])==2
    assert result['paired_first_inference_equal'] and 'PRIVATE_' not in json.dumps(result)


def test_stale_nfs_handle_is_reopened_with_finite_retries(tmp_path,monkeypatch):
    attempts=[]
    def transient(path):
        attempts.append(path)
        if len(attempts)<3:raise OSError(getattr(errno,'ESTALE',116),'stale handle')
        return '{"status":"running"}'
    monkeypatch.setattr(Path,'read_text',transient)
    monkeypatch.setattr('driversi.visualization.skill_evolution.time.sleep',lambda _:None)
    assert read(tmp_path/'progress.json')=={'status':'running'} and len(attempts)==3
    attempts.clear()
    def permanent(path):
        attempts.append(path)
        raise OSError(getattr(errno,'ESTALE',116),'still stale')
    monkeypatch.setattr(Path,'read_text',permanent)
    with pytest.raises(OSError):read(tmp_path/'progress.json')
    assert len(attempts)==3


def test_projection_does_not_hide_corrupt_receipts(tmp_path):
    path=tmp_path/'progress.json';path.write_text('{')
    with pytest.raises(json.JSONDecodeError):read(path)
