"""Control-flow fixtures only; no GPU, reconstruction or model-benefit evidence."""
import copy
import json
from pathlib import Path

import pytest

from driversi.rsi.experiments import dynamic_history_campaign as campaign


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value));return path


def fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(campaign,'project_root',lambda:tmp_path)
    model=write(tmp_path/'model.json',dict(model='chainflow_stage1',python='/native/python'))
    base=write(tmp_path/'base.json',{})
    sources=[]
    for i in range(2):
        branches=[]
        for j in range(2):
            name=f'source{i}-branch{j}'
            branches.append(dict(id=name,generation_report=str(write(tmp_path/(name+'-g.json'),{})),
                                 diversity_report=str(write(tmp_path/(name+'-d.json'),{}))))
        sources.append(dict(id=f'source{i}',recovery_job=str(write(tmp_path/f'job{i}.json',{})),branches=branches))
    plan=dict(schema_version=1,id='dynamic-history-unit',model='chainflow_stage1',sources=sources,
        gates=dict(pdms_gt=.95,source_psnr_gt=25.0,image_contract=campaign.IMAGE_CONTRACT,
                   visual_profile=campaign.VISUAL_PROFILE),
        limits=dict(max_branches=4,minimum_samples=4,minimum_sources=2),
        runtime=dict(asset_python='/asset/python',render_python='/render/python',asset_gpus=[0,1],
            render_gpus=[2,3],asset_max_iterations=30001,asset_timeout_seconds=60),
        training=dict(base_manifest=str(base),model_config=str(model),synthetic_exposures=4,
            batch_size=2,max_updates_per_arm=96,target_token='private-target'))
    return write(tmp_path/'plan.json',plan),plan


def runners():
    calls=[]
    def assets(source,runtime,out):
        calls.append(('asset',source['id'],runtime['gpu']))
        return dict(status='completed',bundle_root=str(out/'bundle'))
    def branches(branch,asset,runtime,out):
        calls.append(('branch',branch['id'],runtime['gpu']))
        sample=write(out/'sample.json',dict(sample_id=branch['id']))
        value=dict(status='training_sample_ready',sample_report=str(sample))
        write(out/'report.json',value);return value
    def pair(plan,samples,out):
        calls.append(('pair',len(samples)))
        assert samples==sorted(samples)
        return dict(status='completed',promoted=False,sim_agent_steps=99)
    return calls,dict(asset_runner=assets,branch_runner=branches,pair_runner=pair,
        sample_validator=lambda p:json.loads(Path(p).read_text())['sample_id'],
        dataset_activator=lambda *a:None)


def test_automatic_assets_admission_native_training_and_idempotence(tmp_path,monkeypatch):
    path,_=fixture(tmp_path,monkeypatch);calls,kwargs=runners()
    out=tmp_path/'exp/run';result=campaign.run(path,out,**kwargs)
    assert result['status']=='completed' and result['admitted_samples']==4
    assert result['physical_rollouts_reused']==4 and result['new_teacher_rollouts']==0
    assert result['future_rgb_generated']==0 and result['training_started']
    assert len([c for c in calls if c[0]=='asset'])==2
    assert {c[2] for c in calls if c[0]=='asset'}<={0,1}
    assert {c[2] for c in calls if c[0]=='branch'}<={2,3}
    assert calls[-1]==('pair',4)
    before=list(calls)
    assert campaign.run(path,out,**kwargs)==result and calls==before


def test_known_unsupported_source_continues_other_source_but_cannot_train(tmp_path,monkeypatch):
    path,_=fixture(tmp_path,monkeypatch);calls,kwargs=runners();original=kwargs['asset_runner']
    kwargs['asset_runner']=lambda source,*a:(dict(status='unsupported',reason='visible_actor_no_asset')
        if source['id']=='source0' else original(source,*a))
    result=campaign.run(path,tmp_path/'exp/run',**kwargs)
    assert result['status']=='insufficient_admitted_data' and result['admitted_samples']==2
    assert not result['training_started'] and all(c[0]!='pair' for c in calls)


def test_unknown_operation_is_not_reissued_on_resume(tmp_path,monkeypatch):
    path,_=fixture(tmp_path,monkeypatch);calls,kwargs=runners();original=kwargs['branch_runner'];errors=[]
    def branch(b,*a):
        if b['id']=='source0-branch0':
            errors.append(b['id']);raise RuntimeError('lost child reply')
        return original(b,*a)
    kwargs['branch_runner']=branch;out=tmp_path/'exp/run'
    assert campaign.run(path,out,**kwargs)['status']=='blocked_unknown'
    assert campaign.run(path,out,**kwargs)['status']=='blocked_unknown'
    assert errors==['source0-branch0'] and all(c[0]!='pair' for c in calls)


def test_saved_final_callback_reply_is_reconciled_without_rerender(tmp_path,monkeypatch):
    path,_=fixture(tmp_path,monkeypatch);calls,kwargs=runners();original=kwargs['branch_runner']
    def interrupted(b,*a):
        value=original(b,*a)
        if b['id']=='source0-branch0':raise RuntimeError('interrupted after commit')
        return value
    kwargs['branch_runner']=interrupted;out=tmp_path/'exp/run'
    assert campaign.run(path,out,**kwargs)['status']=='blocked_unknown'
    assert campaign.run(path,out,**kwargs)['status']=='completed'
    assert len([c for c in calls if c[0]=='branch'])==4
    assert len([c for c in calls if c[0]=='pair'])==1


def test_input_change_never_reuses_old_campaign(tmp_path,monkeypatch):
    path,plan=fixture(tmp_path,monkeypatch);_,kwargs=runners();out=tmp_path/'exp/run'
    campaign.run(path,out,**kwargs)
    write(Path(plan['sources'][0]['branches'][0]['generation_report']),dict(changed=True))
    with pytest.raises(ValueError,match='changed'):campaign.run(path,out,**kwargs)


@pytest.mark.parametrize('kind',['pdms','psnr','future','duplicate','budget','one_source'])
def test_unapproved_policy_or_unbounded_batch_is_rejected(tmp_path,monkeypatch,kind):
    _,p=fixture(tmp_path,monkeypatch)
    if kind=='pdms':p['gates']['pdms_gt']=.9
    elif kind=='psnr':p['gates']['source_psnr_gt']=24
    elif kind=='future':p['gates']['image_contract']='legacy_full_rgb'
    elif kind=='duplicate':p['sources'][1]['branches'][0]['id']=p['sources'][0]['branches'][0]['id']
    elif kind=='budget':p['limits']['max_branches']=100
    else:p['sources']=p['sources'][:1]
    with pytest.raises(ValueError):campaign.validate_plan(p)


def test_actual_population_scales_equal_budget_without_padding():
    assert campaign.update_budget(12,4,4,96)==96
    assert campaign.update_budget(4,4,4,96)==32
    assert campaign.update_budget(5,4,4,96)==40
    with pytest.raises(ValueError):campaign.update_budget(13,4,4,96)


def test_legacy_package_cannot_fill_new_dynamic_history_quota(tmp_path):
    path=write(tmp_path/'sample.json',dict(kind='chainflow_admitted_synthetic_sample',
               image_contract='legacy_full_rgb',future_images_included=True,
               image_counts=dict(history=16,future=40)))
    with pytest.raises(ValueError,match='history-only'):campaign.validate_sample(path)


def test_service_is_classified_once():
    from driversi.rsi.services.catalog import describe
    rows=[r for r in describe() if r['name']=='dynamic_history_replay_campaign']
    assert len(rows)==1 and rows[0]['group']=='training'


def test_known_settled_storage_failure_retries_once_in_new_folder(tmp_path):
    calls=[]
    def run(out):
        calls.append(out)
        return dict(status='retryable',reason='known_transient_storage',result_known=True,no_inflight=True)
    result=campaign._stage(tmp_path/'stage',run)
    assert result['status']=='failed' and result['automatic_retries']==1
    assert len(calls)==2 and calls[0]!=calls[1]
    assert campaign._stage(tmp_path/'stage',run)==result and len(calls)==2


def test_unknown_cost_cannot_request_retry(tmp_path):
    calls=[]
    def run(out):
        calls.append(out);return dict(status='unknown',reason='known_transient_storage',result_known=False,no_inflight=False)
    assert campaign._stage(tmp_path/'stage',run)['status']=='unknown'
    assert len(calls)==1


def test_timeout_reaps_owned_child_process(tmp_path,monkeypatch):
    import sys
    monkeypatch.setattr(campaign,'project_root',lambda:tmp_path)
    result=campaign._owned_process([sys.executable,'-c','import time;time.sleep(30)'],tmp_path,.05)
    assert result['timed_out'] and result['no_inflight']
    assert not campaign._group_alive(result['process_group'])


def test_anchor_protocol_is_distinct_from_history_protocol(tmp_path,monkeypatch):
    _,plan=fixture(tmp_path,monkeypatch)
    plan['gates'].update(image_contract='anchor_rgb_history_future_states_v1',visual_profile='dynamic_anchor25_v1')
    assert campaign.validate_plan(plan)['gates']['image_contract']=='anchor_rgb_history_future_states_v1'
    plan['gates']['visual_profile']='dynamic_history25_v1'
    with pytest.raises(ValueError):campaign.validate_plan(plan)


def test_anchor_admission_rejects_old_16_image_package_before_native_import(tmp_path):
    path=write(tmp_path/'sample.json',dict(kind='chainflow_admitted_synthetic_sample',
        image_contract=campaign.IMAGE_CONTRACT,future_images_included=False,image_counts=dict(history=16,future=0)))
    with pytest.raises(ValueError,match='history-only'):
        campaign.validate_sample(path,image_contract='anchor_rgb_history_future_states_v1')


def test_prebuilt_adoption_waits_for_same_group_and_consumes_completed_receipt(tmp_path,monkeypatch):
    import os
    job=write(tmp_path/'job.json',dict(output=str(tmp_path/'native')))
    blob=tmp_path/'bundle/blob';blob.parent.mkdir();blob.write_bytes(b'unit fixture')
    source=dict(adopt_existing_pid=123,adopt_existing_start='456',adopt_host=os.uname().nodename)
    states=[[dict(pid=123,group=123,start='456')],[]]
    monkeypatch.setattr(campaign,'_job_processes',lambda job:states.pop(0))
    def finish(_):
        write(tmp_path/'native/report.json',dict(status='completed',bundle_root=str(blob.parent),
              outputs={str(blob):campaign.digest(blob)}))
    monkeypatch.setattr(campaign.time,'sleep',finish)
    out=tmp_path/'adoption';out.mkdir()
    result=campaign._adopt_asset(source,job,out,60)
    assert result['adopted_existing_asset'] and result['new_asset_launch'] is False
    assert (out/'adoption.json').exists()


@pytest.mark.parametrize('bad',[dict(pid=123,group=999,start='456'),dict(pid=123,group=123,start='wrong')])
def test_prebuilt_adoption_rejects_foreign_or_reused_pid(tmp_path,monkeypatch,bad):
    import os
    source=dict(adopt_existing_pid=123,adopt_existing_start='456',adopt_host=os.uname().nodename)
    monkeypatch.setattr(campaign,'_job_processes',lambda job:[bad])
    with pytest.raises(RuntimeError):campaign._adopt_asset(source,tmp_path/'job',tmp_path,60)


def test_entrypoint_creates_new_output_root_before_task_registration(tmp_path,monkeypatch):
    import sys
    path,_=fixture(tmp_path,monkeypatch);output=tmp_path/'new/output'
    monkeypatch.setattr(sys,'argv',['campaign','--plan',str(path),'--output',str(output)])
    def register(*a,**kw):
        assert output.is_dir();return dict(status='prepared',directory=str(output))
    monkeypatch.setattr(campaign,'task_state',register)
    campaign.main()


def test_definite_current_asset_exit_is_failure_not_unknown(tmp_path):
    path=write(tmp_path/'status.json',dict(status='failed',failed_at=12,error_type='CalledProcessError'))
    receipt=dict(returncode=1,no_inflight=True,timed_out=False,started_at=10,finished_at=13)
    value=campaign._known_asset_failure(receipt,path)
    assert value['status']=='failed' and value['result_known'] and not value['automatic_resend']
    for changed in (dict(no_inflight=False),dict(timed_out=True),dict(started_at=13),dict(returncode=0)):
        assert campaign._known_asset_failure(dict(receipt,**changed),path) is None


def test_verified_runtime_is_explicit_and_unset_preserves_native_command():
    runtime=dict(asset_python='/native/python')
    command=campaign.asset_command('/job.json',runtime)
    assert command[3]=='driversi.rendering.gaussian.dynamic_reconstruction' and '--witness' not in command
    selected=campaign.asset_command('/job.json',dict(runtime,gsplat_runtime_witness='/witness.json'))
    assert selected[3]=='driversi.rendering.gaussian.verified_gsplat_runtime'
    assert selected[-2:]==['--witness','/witness.json']


def test_interrupted_adoption_recovers_old_completed_output_without_launch(tmp_path,monkeypatch):
    import os
    recovery=write(tmp_path/'recovery.json',{})
    job=write(tmp_path/'old/job.json',dict(kind='native_mtgs_dynamic_v1',output=str(tmp_path/'old'),
        recovery_job=str(recovery),gpu=0))
    source=dict(prebuilt_asset_job=str(job),recovery_job=str(recovery),adopt_existing_pid=123,
        adopt_existing_start='456',adopt_host=os.uname().nodename)
    monkeypatch.setattr(campaign,'_job_processes',lambda job:[])
    calls=[]
    def invoke(out):calls.append(out);return dict(status='unknown',reason='adoption_interrupted')
    stage=tmp_path/'stage'
    recover=lambda:campaign.reconcile_asset(source,dict(asset_gpus=[0]),stage/'execution')
    assert campaign._stage(stage,invoke,recover=recover)['status']=='unknown'
    assert campaign._stage(stage,invoke,recover=recover)['status']=='unknown'
    blob=tmp_path/'bundle/blob';blob.parent.mkdir();blob.write_bytes(b'fixture')
    write(tmp_path/'old/report.json',dict(status='completed',bundle_root=str(blob.parent),
        outputs={str(blob):campaign.digest(blob)}))
    result=campaign._stage(stage,invoke,recover=recover)
    assert result['status']=='completed' and result['reconciled'] and len(calls)==1
    assert campaign._stage(stage,invoke,recover=recover)==result
