import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from driversi.agents.backends.native import digest
from driversi.rsi.experiments import asset_recovery as ar
from driversi.rsi.experiments.continual_execution import next_operation


def source(tmp_path, value=27.1):
    image=tmp_path/'image';image.write_bytes(b'original measured image')
    rows=[dict(tick=i*5,camera=c,measured_psnr_db=value,
               rendered_source_image=str(image),real_source_image=str(image))
          for i in range(15) for c in ('CAM_F0','CAM_L0','CAM_R0','CAM_B0')]
    r=dict(source_mode='canonical_source_pose_without_motion_v1',
        asset=dict(token='train',source_log='log'),operator='>',threshold=27.,
        mean_psnr_db=value,frames=rows,inputs={str(image):digest(image)},files={str(image):digest(image)},passed_preflight=True)
    p=tmp_path/'source.json';p.write_text(json.dumps(r));return p,r,image


def test_full_source_strict_threshold_and_tamper(tmp_path):
    p,r,image=source(tmp_path,27.)
    assert ar.verify_source(p,token='train',source_log='log') is None
    r['mean_psnr_db']=27.1
    p.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='mean'):ar.verify_source(p,token='train',source_log='log')
    p,r,image=source(tmp_path)
    assert ar.verify_source(p,token='train',source_log='log')['trainable'] is False
    image.write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):ar.verify_source(p,token='train',source_log='log')


def test_missing_camera_or_wrong_source_never_registers(tmp_path):
    p,r,_=source(tmp_path)
    with pytest.raises(ValueError,match='identity'):ar.verify_source(p,token='other',source_log='log')
    r['frames'].pop();p.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='four-camera'):ar.verify_source(p,token='train',source_log='log')


def test_completed_recipe_not_executed_twice(tmp_path,monkeypatch):
    child=Mock(pid=123);child.wait.return_value=0
    popen=Mock(return_value=child);monkeypatch.setattr(ar.subprocess,'Popen',popen)
    monkeypatch.setattr(ar,'project_root',lambda:tmp_path)
    a=ar.execute('python','registered_module',{'a':1},tmp_path/'step',timeout=10,env={})
    b=ar.execute('python','registered_module',{'a':1},tmp_path/'step',timeout=10,env={})
    assert a==b and popen.call_count==1


def test_unknown_running_recipe_never_repeated(tmp_path,monkeypatch):
    folder=tmp_path/'step';folder.mkdir();(folder/'running.json').write_text('{}')
    popen=Mock();monkeypatch.setattr(ar.subprocess,'Popen',popen)
    with pytest.raises(RuntimeError,match='unknown'):
        ar.execute('python','registered_module',{},folder,timeout=10,env={})
    assert not popen.called


def test_known_failed_strategy_reuses_receipt_and_can_escalate(tmp_path,monkeypatch):
    child=Mock(pid=123);child.wait.return_value=2
    popen=Mock(return_value=child);monkeypatch.setattr(ar.subprocess,'Popen',popen)
    monkeypatch.setattr(ar,'project_root',lambda:tmp_path)
    a=ar.execute('python','registered_module',{'a':1},tmp_path/'step',timeout=10,env={})
    b=ar.execute('python','registered_module',{'a':1},tmp_path/'step',timeout=10,env={})
    assert a==b and a['status']=='failed' and popen.call_count==1


def test_completed_pending_recovery_is_requeued_before_scheduling(tmp_path,monkeypatch):
    from driversi.rsi.experiments import continual_curriculum as cc
    state=dict(active=None,cases=[dict(id='a',status='asset_recovery_pending',candidates=[dict(token='x')])])
    monkeypatch.setattr(ar,'policy',lambda root:{'builder_gpus':[1]})
    monkeypatch.setattr(ar,'ready_sources',lambda root:{'x':{}})
    monkeypatch.setattr(ar,'read_mapping',lambda path:state)
    saved=Mock();monkeypatch.setattr(cc,'save',saved)
    ar.reconcile_pending(tmp_path)
    assert state['cases'][0]['status']=='queued'
    assert state['cases'][0]['asset_revision']==1 and saved.called
    assert next_operation(state)==('prepare',None)


def test_asset_block_dispatches_recovery_and_pending_is_not_finished():
    case=dict(id='a',status='blocked_assets')
    assert next_operation(dict(active='a',cases=[case]))==('recover_assets',None)
    case['status']='asset_recovery_pending'
    assert next_operation(dict(active=None,cases=[case]))==(None,'asset_recovery_in_progress')
    case['status']='asset_recovery_unknown'
    assert next_operation(dict(active=None,cases=[case]))==(None,'needs_reconciliation')
    case['status']='assets_unresolved'
    assert next_operation(dict(active=None,cases=[case]))==(None,'unresolved_assets')


def test_verified_asset_short_circuits_gpu_calls(tmp_path,monkeypatch):
    p,r,_=source(tmp_path)
    monkeypatch.setattr(ar,'project_root',lambda:tmp_path)
    worker=Mock();monkeypatch.setattr(ar,'execute',worker)
    out=ar.run(dict(token='train',source_log='log',output=str(tmp_path/'exp/task'),
        source_report=str(p),gpu=0))
    assert out['status']=='source_ready' and not out['trainable'] and not worker.called


def test_repair_budget_integer_minimum_and_resume_binding(tmp_path):
    p,_,_=source(tmp_path)
    stage=tmp_path/'geometry';job=dict(sensor_root='raw')
    value=ar.repair_payload(stage,str(p),job,'geometry',3599.5)
    assert value['max_wall_seconds']==3599 and type(value['max_wall_seconds']) is int
    assert ar.repair_payload(stage,str(p),job,'geometry',119.5) is None
    (stage/'execution').mkdir(parents=True)
    (stage/'execution/job.json').write_text(json.dumps(value))
    assert ar.repair_payload(stage,str(p),job,'geometry',2500.9)==value


def test_only_known_duplicate_shortage_selects_alternative():
    from driversi.rsi.experiments.generation_recovery import decision
    summary=dict(rows=[],anchors=[dict(search_stop_reason='consecutive_non_novel_native_rollouts')])
    config=dict(trajectory_family=dict(method='native_policy_sample_family_v1'))
    assert decision(summary,config,0)=='bounded_history_proposals'
    assert decision(summary,config,1)=='generation_budget_exhausted'
    summary['rows']=[dict(status='execution_failed')]
    assert decision(summary,config,0)=='needs_execution_reconciliation'
    assert decision(summary,config,1)=='needs_execution_reconciliation'
    summary['rows']=[dict(status='native_language_review_incomplete')]
    assert decision(summary,config,0)=='needs_execution_reconciliation'


@pytest.mark.parametrize('status,reason',[
    ('needs_execution_reconciliation','needs_reconciliation'),
    ('generation_unresolved','unresolved_generation'),
    ('generation_budget_exhausted','generation_budget_exhausted')])
def test_unresolved_generation_is_never_finished(status,reason):
    assert next_operation(dict(active=None,cases=[dict(id='a',status=status)]))==(None,reason)


def test_generation_alternative_debits_prior_work_without_increasing_budget():
    from driversi.rsi.experiments.generation_recovery import remaining_budget
    original=dict(attempts_per_anchor=32,render_cap_per_anchor=20,target_per_anchor=10,
                  global_attempt_cap=32,global_render_cap=20,wave_size=1,states=[{}])
    summary=dict(attempted=8,rendered=2,anchors=[dict(anchor='a',attempted=8)],
                 rows=[dict(candidate='a-v000',anchor_assignment_id='a',rendered=True),
                       dict(candidate='a-v001',anchor_assignment_id='a',rendered=True)])
    revised=remaining_budget(original,summary,24)
    assert revised['attempts_per_anchor']==24 and revised['global_attempt_cap']==24
    assert revised['render_cap_per_anchor']==18 and revised['global_render_cap']==18
    assert revised['target_per_anchor']==10 and original['global_attempt_cap']==32
    summary['attempted']=summary['anchors'][0]['attempted']=25
    with pytest.raises(ValueError,match='cannot cover quota'):remaining_budget(original,summary,24)


def test_multisource_remaining_render_budget_uses_real_outcome_identities():
    from driversi.rsi.experiments.generation_recovery import remaining_budget
    original=dict(attempts_per_anchor=32,render_cap_per_anchor=20,target_per_anchor=10,
                  global_attempt_cap=64,global_render_cap=40,wave_size=2,states=[{},{}])
    summary=dict(attempted=20,rendered=14,anchors=[dict(anchor='a',attempted=6),dict(anchor='b',attempted=14)],
        rows=[dict(candidate=f'{aid}-v{i:03d}',anchor_assignment_id=aid,rendered=True)
              for aid,n in [('a',2),('b',12)] for i in range(n)])
    with pytest.raises(ValueError,match='cannot cover quota'):remaining_budget(original,summary,24)


def test_per_source_render_limit_does_not_borrow_other_source_allowance():
    from driversi.rsi.experiments.reactive_cohort import render_budget_exhausted
    config=dict(max_rendered_branches=24,trajectory_family=dict(render_cap_per_anchor=12))
    rows=[dict(anchor_assignment_id='a',rendered=True) for _ in range(12)]
    assert render_budget_exhausted(config,rows,'a',12)
    assert not render_budget_exhausted(config,rows,'b',12)
    assert render_budget_exhausted(config,rows,'b',24)


def test_only_known_missing_diagnostic_directory_allows_one_resume(tmp_path):
    from driversi.rsi.services.reconstruction_recovery import eligible, directory
    native=tmp_path/'native';native.mkdir()
    recipe=tmp_path/'native-job.json';recipe.write_text(json.dumps(dict(output=str(native))))
    job=dict(output=str(tmp_path/'recovery'),reconstruction_job=str(recipe))
    (native/'report.json').write_text(json.dumps(dict(status='stage_failed',stage='registration')))
    log=native/'registration.log';log.write_text('FileNotFoundError: /owned/map_vis/registration.pdf')
    assert eligible(job)
    (native/'running.json').write_text('{}')
    assert not eligible(job)
    (native/'running.json').unlink();log.write_text('OutOfMemoryError')
    assert not eligible(job)
    log.write_text('FileNotFoundError: /owned/map_vis/registration.pdf')
    revised=directory(job);revised.mkdir(parents=True);(revised/'launch.json').write_text('{}')
    assert not eligible(job)
    assert ar.current_recovery_folder(job['output'])==revised
