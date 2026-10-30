import json
from pathlib import Path
import pytest

from driversi.agents.backends.native import digest
from driversi.rsi.decisions.service_memory import ServiceMemory, review_learning
from driversi.rsi.experiments.augustus_round import _task, _verify, verify_program, _finalize
from driversi.data.catalog.store import identity


def feedback(gain=.1, parent=.1):
    return dict(kind='augustus_paired_feedback',status='completed',effective_recipe_verified=True,
                positive_lr_added_sample_ids=['a'],exposed_added_sample_ids=['a'],
                gain_vs_control_pdms_points=gain,gain_vs_parent_pdms_points=parent,safety_regressions=[])


def test_replay_is_required_before_any_measured_gain():
    good={'different_pids':True,'same_compared_state':True}
    x=review_learning(feedback(),[good,dict(good,same_compared_state=False)])
    assert x['node']=='resume_verification' and x['observed_gain'] is None
    assert not x['skill_gain_credited'] and not x['promoted']
    assert review_learning(feedback(),[good,good])['status']=='pilot_signal'


@pytest.mark.parametrize('gain,parent',[(0,.1),(-.1,.1),(.1,0),(.1,-.1)])
def test_improvement_requires_control_and_parent(gain,parent):
    good={'different_pids':True,'same_compared_state':True}
    assert review_learning(feedback(gain,parent),[good,good])['status']=='no_benefit'


def test_safety_regression_never_awards_positive_signal():
    f=feedback();f['safety_regressions']=[{'metric':'NC'}]
    good={'different_pids':True,'same_compared_state':True}
    assert review_learning(f,[good,good])['reason']=='safety_regression'


def test_every_added_sample_needs_positive_lr_exposure():
    f=feedback();f['exposed_added_sample_ids'].append('b')
    with pytest.raises(ValueError,match='paired feedback'):review_learning(f,[])


def case(tmp_path):
    f=tmp_path/'evidence.json';f.write_text('{}')
    return dict(program_id='p',context={'parent_assets_id':'sha','evaluator_id':'eval'},
                node='visual_admission',status='blocked',source_logs=['log-a'],
                conditions=['nearby_objects_dense'],promoted=False,skill_gain_credited=False,
                evidence={str(f):digest(f)})


def test_persistent_evidence_and_idempotent_write(tmp_path):
    c=case(tmp_path);p=tmp_path/'memory.sqlite'
    s=ServiceMemory(p,'evolve');assert s.record(c)['recorded'];s.close()
    s=ServiceMemory(p,'evolve');assert not s.record(c)['recorded'];assert len(s.rows())==1
    with pytest.raises(ValueError,match='Conflicting'):s.record(dict(c,status='different'))
    s.close()


def test_frozen_does_not_create_and_does_not_write(tmp_path):
    p=tmp_path/'absent.sqlite';s=ServiceMemory(p,'frozen')
    assert s.rows()==[] and not s.record({})['recorded'];s.close();assert not p.exists()
    c=case(tmp_path);s=ServiceMemory(p,'evolve');s.record(c);s.close();before=p.read_bytes()
    s=ServiceMemory(p,'frozen');assert s.rows();assert not s.record(c)['recorded'];s.close()
    assert p.read_bytes()==before


def test_memory_requires_content_context_and_conditions(tmp_path):
    c=case(tmp_path);s=ServiceMemory(tmp_path/'m.sqlite','evolve');s.record(c)
    assert not s.advise({'parent_assets_id':'other','evaluator_id':'eval'},['log-a'])['matches']
    assert not s.advise(c['context'],['log-b'])['matches']
    found=s.advise(c['context'],['log-b'],conditions=['nearby_objects_dense'])
    assert found['matches'][0]['applicability']=='condition_overlap_unvalidated'
    assert found['preflight_actions']==['verify_asset_coverage_and_registration_before_rollout']
    assert not found['cause_established'];s.close()


def test_changed_evidence_and_positive_credit_refused(tmp_path):
    c=case(tmp_path);s=ServiceMemory(tmp_path/'m.sqlite','evolve')
    with pytest.raises(ValueError,match='cannot promote'):s.record(dict(c,promoted=True))
    Path(next(iter(c['evidence']))).write_text('{"changed":true}')
    with pytest.raises(ValueError,match='changed'):s.record(c)
    s.close()


def test_task_uses_native_complete_dependency_chain(tmp_path):
    req=tmp_path/'request.json';req.write_text(json.dumps({'budget':{'max_wall_seconds':50,'max_scoring_agent_steps':100}}))
    t=_task('unit',req,tmp_path/'eval.json');steps={s['id']:s for s in t['steps']}
    assert steps['evaluate']['inputs']=={'student':'resume'}
    assert steps['feedback']['inputs']=={'student':'resume','replay':'replay','evaluation':'evaluate'}
    assert steps['train']['provider']=='native_model_unified_training'
    assert all(set(s['config']) <= {'request','evaluator'} for s in t['steps'])
    assert t['budget']['max_sim_agent_steps']==400


def test_code_and_input_closure_are_frozen(tmp_path,monkeypatch):
    from driversi.rsi.experiments import augustus_round as m
    source=tmp_path/'code.py';source.write_text('version=1')
    inputs=tmp_path/'input';inputs.write_text('x')
    p=dict(files={str(inputs):digest(inputs)},code_files={str(source):digest(source)})
    p['program_id']=identity(p);(tmp_path/'program.json').write_text(json.dumps(p))
    monkeypatch.setattr(m,'_code_files',lambda:[source])
    assert verify_program(tmp_path)['program_id']==p['program_id']
    source.write_text('version=2')
    with pytest.raises(ValueError,match='changed'):verify_program(tmp_path)


def test_final_writeback_crash_can_resume_without_duplicate(tmp_path):
    out=tmp_path/'episode';out.mkdir();(out/'program.json').write_text('{}')
    plan=dict(output=str(out),program_id='p',memory=str(tmp_path/'m.sqlite'),memory_mode='evolve',
              context={'parent':'sha'},source_logs=['log-a'],conditions=[])
    report=dict(node='visual_admission',status='blocked',training_executed=False,observed_gain=None)
    result=_finalize(plan,report)
    # Simulate crash after memory write, before next-iteration is durable.
    (out/'next-iteration.json').unlink()
    again=_finalize(plan,json.loads((out/'review.json').read_text()))
    assert result==again
    s=ServiceMemory(plan['memory'],'frozen');assert len(s.rows())==1;s.close()


def test_model_capability_is_not_branch_admission():
    from driversi.rsi.decisions.model_services import student_contract,diagnose_augustus_branch
    c=student_contract('augustus')
    assert c['new_history_training'] and c['paired_training_service']
    assert not c['paired_skill_feedback']
    d=diagnose_augustus_branch({'kind':'selected_native_branch','branch':'teacher'})
    assert not d['trainable'] and d['learning_blockers']
    assert d['skills'][-1]['checks']['counterfactual_joint_cache_connected']
    assert not d['skills'][-1]['checks']['this_branch_joint_supervision_verified']


def test_memory_executes_preflight_not_prior_verdict(monkeypatch):
    from driversi.rsi.experiments import augustus_round as m,branch_expansion as b
    calls=[]
    monkeypatch.setattr(b,'verified_plan',lambda p:calls.append(p) or {'plan_id':'verified-current'})
    plan=dict(selection_mode='memory',advice={'preflight_actions':['verify_asset_coverage_and_registration_before_rollout']},expansion_root='root')
    r=m.memory_preflight(plan)
    assert calls==['root'] and r['performed'][0]['current_plan_id']=='verified-current'
    assert not r['performed'][0]['visually_admitted'] and not r['verdict_reused']
    plan['selection_mode']='no_memory';assert m.memory_preflight(plan)['performed']==[]


def test_positive_memory_asks_for_more_validation_not_recipe_repair(tmp_path):
    c=case(tmp_path);c.update(node='learning_review',status='pilot_signal',observed_gain=.2)
    s=ServiceMemory(tmp_path/'m.sqlite','evolve');s.record(c)
    assert s.advise(c['context'],['log-a'])['preflight_actions']==['expand_independent_validation']
    s.close()


def test_failed_harness_node_preserves_reason_and_cost(tmp_path):
    from driversi.rsi.experiments.augustus_round import task_failure
    (tmp_path/'report.json').write_text('{}')
    s=dict(directory=str(tmp_path),status='failed',elapsed_seconds=12.,sim_agent_steps=0,
           stages=[dict(id='train',status='completed'),dict(id='replay',status='failed',reason='checkpoint_invalid')])
    r=task_failure(s)
    assert r['node']=='resume_verification' and r['failed_stage']['reason']=='checkpoint_invalid'
    assert r['partial_cost']['harness_wall_seconds']==12. and r['execution_evidence']


@pytest.mark.parametrize('changed',['branch_report','render_report','visual_admission','label_review','bytes'])
def test_publication_resume_binds_every_original_evidence(tmp_path,changed):
    from driversi.rsi.experiments.augustus_round import verify_publication_job
    paths={k:str(tmp_path/(k+'.json')) for k in ['branch_report','render_report','visual_admission','label_review']}
    for p in paths.values():Path(p).write_text('{}')
    Path(paths['visual_admission']).write_text(json.dumps({'render_report':paths['render_report']}))
    sample={k:paths[k] for k in ['branch_report','visual_admission','label_review']}
    sample['files']={p:digest(p) for p in paths.values()}
    verify_publication_job(sample,paths)
    if changed=='bytes':Path(paths['label_review']).write_text('{"changed":true}')
    else:paths[changed]=str(tmp_path/'other.json')
    with pytest.raises(ValueError,match='differs'):verify_publication_job(sample,paths)


def test_branch_context_uses_weights_recipe_and_verified_history(tmp_path,monkeypatch):
    from driversi.rsi.experiments.augustus_round import branch_context
    from driversi.rsi.decisions import vla_memory
    monkeypatch.setattr(vla_memory,'history_signature',lambda x: {'conditions':x['conditions']})
    p=tmp_path/'select';p.mkdir();w=tmp_path/'teacher.pt';w.write_bytes(b'weights-a')
    window=p/'trajectory-window.json';window.write_text(json.dumps({'conditions':['turning']}))
    task=tmp_path/'task.json';task.write_text(json.dumps({'steps':[{'provider':'navsim_native_teacher_probe','config':{'teacher_backend':'driverl_native','arrival_action':[0,0]}}]}))
    b=dict(files={},upstream_source_files={str(w):digest(w)},future_frames=8,
           trajectory_window_sha256=digest(window),trajectory_evidence={'teacher':'DriveRL_native_argmax'})
    bp=p/'report.json';bp.write_text(json.dumps(b))
    teacher,facts,files=branch_context(bp)
    assert teacher['weights']==[digest(w)] and teacher['generation_recipe']['arrival_action']==[0,0]
    assert facts==['turning'] and str(window)in files
    w.write_bytes(b'weights-b')
    with pytest.raises(ValueError,match='changed'):branch_context(bp)
    b['upstream_source_files'][str(w)]=digest(w);bp.write_text(json.dumps(b))
    assert branch_context(bp)[0]!=teacher
    window.write_text(json.dumps({'conditions':['straight']}))
    with pytest.raises(ValueError,match='History'):branch_context(bp)


def test_generation_cost_is_same_branch_and_not_double_counted(tmp_path):
    from driversi.rsi.experiments.augustus_round import generation_costs
    branch=tmp_path/'branch.json';receipt=tmp_path/'task.json'
    receipt.write_text(json.dumps(dict(status='completed',id='a',elapsed_seconds=118.6,sim_agent_steps=625,
        stages=[{'result':{'artifacts':{'report':str(branch)}}}])))
    c=generation_costs([receipt],[branch]);assert c['wall_seconds']==118.6 and c['sim_agent_steps']==625
    assert not c['complete_total_cost']
    with pytest.raises(ValueError,match='Duplicate'):generation_costs([receipt,receipt],[branch])
    with pytest.raises(ValueError,match='producer'):generation_costs([receipt],[tmp_path/'foreign.json'])
