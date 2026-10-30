import copy
import json
from pathlib import Path

import pytest

from driversi.rsi.experiments import continuation as c
from driversi.rsi.experiments.interfaces import interfaces, validate_population
from driversi.rsi.plugins import continuation as gate
from driversi.rsi.workflow.store import HarnessStore
from driversi.runtime.budget import write_json


def test_population_requires_exact_official_training_membership():
    official={'tokens':['a','b'],'log_names':['log']}
    rows=[dict(token=t,source_log='log',source_group='g') for t in ['a','b']]
    validate_population(rows,official)
    for invalid in [rows[:1],rows+rows[:1],[{**rows[0],'token':'test'},rows[1]],
                    [{**rows[0],'source_log':'test-log'},rows[1]]]:
        with pytest.raises(ValueError):validate_population(invalid,official)


def test_inventory_exposes_model_specific_semantics_and_limits():
    value=interfaces();models=value['models']
    assert set(models)=={'chainflow_stage1','driverl','augustus'}
    assert models['chainflow_stage1']['future']==10 and models['augustus']['future']==8
    assert models['driverl']['mode']=='online_rl'
    assert all(not row['full_scale_ready'] for row in models.values())
    assert value['schema_version']==2
    assert value['training_caps']['epoch_pair']['implemented']
    assert not value['training_caps']['epoch_pair']['full_103k_epoch_validated']
    assert value['training_caps']['legacy_pair']['per_rank_batch_size']==1
    assert value['time_contract']['native_PDM_scored_seconds']==4
    assert value['memory_contract']['no_model_prompt_injection']
    assert any(p['name']=='qualified_native_branch_selection' for p in value['providers'])


def evidence():
    return dict(branch='teacher',completed_steps=50,producer_identity='producer',history_changed=True,
        restore_exact=True,arrival_contact_records=0,future_contact_records=0,
        events=dict(collision=0,offroad=0,route_support_end=0),
        score=dict(score=.99,driving_direction_compliance=1.,actual_world_seconds=5.,
            protocol='navsim-v1-native-math-executed-counterfactual-v1'))


@pytest.mark.parametrize('mutation',[{}, {'arrival_contact_records':1}, {'future_contact_records':1},
    {'completed_steps':49},{'restore_exact':False},{'history_changed':False},
    {'events':{'collision':0,'offroad':0,'route_support_end':1}},
    {'score':dict(score=.94,driving_direction_compliance=1.,actual_world_seconds=5.,
        protocol='navsim-v1-native-math-executed-counterfactual-v1')}])
def test_training_selection_blocks_bad_trajectory_before_render(tmp_path,monkeypatch,mutation):
    path=tmp_path/'branch.json';write_json(path,{'trajectory_evidence':{**evidence(),**mutation}})
    def selected(*args):
        return dict(status='completed',reason=None,artifacts={'report':str(path)},admission={'trainable':False})
    monkeypatch.setattr(gate,'run_select',selected)
    result=gate.execute({}, {}, tmp_path/'out',tmp_path/'registry')
    assert result['status']==('rejected' if mutation else 'completed')
    assert not result['admission']['trainable']


def config(tmp_path):
    ref=dict(source_task='source',source_stage='selected')
    return dict(schema_version=1,id='round-01',registry=str(tmp_path/'exp/ledger.sqlite'),
        output=str(tmp_path/'exp/round-01'),model='chainflow_stage1',
        input={'kind':'rendered','branch':ref,'rendered':ref,'source':ref},render=None,
        export=dict(python='/python',maps_root='/maps',max_wall_seconds=60,
            visual_review=str(tmp_path/'review.json'),visual_quality_profile='source27_v1'),training=None)


def test_frozen_declaration_prevents_budget_or_branch_replacement(tmp_path):
    path=tmp_path/'declaration.json';c.frozen_json(path,{'steps':8});c.frozen_json(path,{'steps':8})
    with pytest.raises(ValueError):c.frozen_json(path,{'steps':16})
    assert json.loads(path.read_text())=={'steps':8}


def test_waiting_review_does_not_launch_or_promote(tmp_path,monkeypatch):
    from driversi.rsi.experiments import readiness
    monkeypatch.setattr(readiness,'inspect_stored_render',lambda *a:dict(blockers=[]))
    monkeypatch.setattr(c,'project_root',lambda:tmp_path)
    monkeypatch.setattr(c,'_qualified_references',lambda cfg:cfg['input'])
    monkeypatch.setattr(c,'task_state',lambda *a,**k:pytest.fail('No review: must not launch'))
    path=tmp_path/'config.json';write_json(path,config(tmp_path))
    first=c.run(path,execute=True);second=c.run(path,execute=True)
    assert first['status']==second['status']=='waiting_visual_review'
    assert not second['training_executed'] and not second['promoted']


def test_non_chainflow_continuation_is_explicitly_unsupported(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'project_root',lambda:tmp_path)
    value=config(tmp_path);value['model']='augustus'
    with pytest.raises(ValueError,match='ChainFlow'):c.validate_config(value)


def test_teacher_render_graph_has_qualified_gate_and_dependencies(tmp_path):
    value=config(tmp_path);value['input']={'kind':'teacher','source_task':'t','source_stage':'s'}
    value['render']=dict(provider='mtgs_selected_branch_render',config={'max_wall_seconds':600},source_config={'max_wall_seconds':600})
    task=c.render_task(value);nodes={s['id']:s for s in task['steps']}
    assert nodes['branch']['provider']=='qualified_native_branch_selection'
    assert nodes['render']['inputs']=={'branch':'branch'}
    assert nodes['source']['inputs']=={'render':'render'}


@pytest.mark.parametrize('status',['completed','rejected','failed','running','outcome_unknown'])
def test_existing_task_is_never_relaunched(tmp_path,monkeypatch,status):
    registry=tmp_path/'harness.sqlite';root=tmp_path/'tasks';spec=tmp_path/'task.json'
    task={'id':'known','providers':[],'source_groups':[]}
    store=HarnessStore(registry);store.register(task,root/'known')
    with store.registry._connect() as db:db.execute('UPDATE harness_tasks SET status=? WHERE id=?',(status,'known'))
    monkeypatch.setattr(c.engine,'run',lambda *a:pytest.fail('Existing job must not rerun'))
    state=c.task_state(registry,task,spec,root,execute=True)
    assert state['status']==status


def test_task_identity_collision_not_adopted(tmp_path):
    registry=tmp_path/'harness.sqlite';task={'id':'known','providers':[],'source_groups':[]}
    HarnessStore(registry).register(task,tmp_path/'another/known')
    with pytest.raises(ValueError,match='another continuation'):
        c.task_state(registry,task,tmp_path/'spec.json',tmp_path/'tasks',execute=True)


def test_same_round_mutual_exclusion(tmp_path):
    with c._lock(tmp_path/'round'):
        with pytest.raises(ValueError,match='owns'):
            with c._lock(tmp_path/'round'):pass


def test_missing_training_execution_flag_cannot_train(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'project_root',lambda:tmp_path)
    path=tmp_path/'config.json';write_json(path,config(tmp_path))
    with pytest.raises(ValueError,match='execution mode'):c.run(path,execute_training=True)


@pytest.mark.parametrize('wrong_link',[None,'render','source'])
def test_real_reference_contract_is_path_typed_and_branch_specific(tmp_path,monkeypatch,wrong_link):
    from driversi.rsi.plugins import chainflow_synthetic as synthesis
    branch=tmp_path/'branch.json';render=tmp_path/'render.json';source=tmp_path/'source.json'
    write_json(branch,dict(trajectory_evidence=evidence(),future_frames=10))
    write_json(render,dict(branch_identity='wrong' if wrong_link=='render' else 'branch-id'))
    write_json(source,dict(render_identity='wrong' if wrong_link=='source' else 'render-id'))
    def stored(ref,kind,registry):
        assert isinstance(registry,Path)
        name={'selected_native_branch':'branch','worldengine_inspection_window':'render','worldengine_source_comparison':'source'}[kind]
        return {},{'artifacts':{'report':str(tmp_path/(name+'.json'))},
                   'summary':{'artifact_reference':{'identity':name+'-id'}}},{}
    monkeypatch.setattr(synthesis,'_stored_stage',stored)
    if wrong_link:
        with pytest.raises(ValueError,match='another'):c._qualified_references(config(tmp_path))
    else:assert set(c._qualified_references(config(tmp_path)))=={'branch','rendered','source'}


def test_assignment_dispatches_once_into_shared_registry(tmp_path,monkeypatch):
    from driversi.rsi.experiments import skill_execution
    from driversi.rsi.decisions import capability_skills
    from driversi.agents.backends.native import digest
    monkeypatch.setattr(c,'project_root',lambda:tmp_path)
    monkeypatch.setattr(capability_skills,'validate_plan',lambda x:x)
    plan=tmp_path/'plan.json';write_json(plan,dict(context={'model':'chainflow_stage1'},assignments=[{'id':'chosen'}]))
    value=config(tmp_path);value['input']=dict(kind='assignment',plan=str(plan),assignment='chosen',
        database='/catalog',memory='/memory',maps_root='/maps',python='/python',checkpoint='/weights',device='cpu')
    value['render']=dict(provider='mtgs_selected_branch_render',config={},source_config={})
    path=tmp_path/'config.json';write_json(path,value);calls=[]
    def teacher(plan_path,assignment,database,output,memory,**kwargs):
        calls.append(kwargs['registry']);output.mkdir()
        result=dict(status='execution_or_admission_rejected',plan_sha256=digest(plan_path),assignment_id=assignment)
        write_json(output/'report.json',result);return result
    monkeypatch.setattr(skill_execution,'run_assignment',teacher)
    assert c.run(path,execute=True)['status']=='teacher_not_admitted'
    assert c.run(path,execute=True)['status']=='teacher_not_admitted'
    assert calls==[Path(value['registry'])]
    assert len((Path(value['output'])/'attempts.jsonl').read_text().splitlines())==2


def test_unexpected_failure_has_terminal_receipt(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'project_root',lambda:tmp_path)
    monkeypatch.setattr(c,'_advance',lambda *a:(_ for _ in ()).throw(AttributeError('test defect')))
    value=config(tmp_path);path=tmp_path/'config.json';write_json(path,value)
    with pytest.raises(AttributeError):c.run(path)
    assert read_result(value)['status']=='execution_error'


def read_result(value):
    return json.loads((Path(value['output'])/'report.json').read_text())


@pytest.mark.parametrize('section,key',[('export','visual_review'),('export','python'),('export','maps_root')])
def test_relative_continuation_paths_fail_before_any_job(tmp_path,monkeypatch,section,key):
    monkeypatch.setattr(c,'project_root',lambda:tmp_path)
    value=config(tmp_path);value[section][key]='relative/file'
    with pytest.raises(ValueError,match='absolute'):c.validate_config(value)


@pytest.mark.parametrize('key',['config_path','checkpoint_path','road_block_config','bundle'])
def test_renderer_native_paths_must_also_be_absolute(tmp_path,monkeypatch,key):
    monkeypatch.setattr(c,'project_root',lambda:tmp_path)
    value=config(tmp_path);value['input']=dict(kind='teacher',source_task='teacher',source_stage='teacher')
    value['render']=dict(provider='mtgs_selected_branch_render',config={key:'relative/asset'},source_config={})
    with pytest.raises(ValueError,match='absolute'):c.validate_config(value)


def test_known_preparation_failure_is_not_reclassified_unknown(tmp_path,monkeypatch):
    from driversi.rsi.decisions import capability_skills
    from driversi.agents.backends.native import digest
    monkeypatch.setattr(c,'project_root',lambda:tmp_path)
    monkeypatch.setattr(capability_skills,'validate_plan',lambda x:x)
    plan=tmp_path/'plan.json';write_json(plan,dict(context={'model':'chainflow_stage1'},assignments=[{'id':'chosen'}]))
    value=config(tmp_path);value['input']=dict(kind='assignment',plan=str(plan),assignment='chosen',
        database='/catalog',memory='/memory',maps_root='/maps',python='/python',checkpoint='/weights',device='cpu')
    value['render']=dict(provider='mtgs_selected_branch_render',config={},source_config={})
    dest=Path(value['output'])/(value['id']+'-teacher');dest.mkdir(parents=True)
    write_json(dest/'report.json',dict(plan_sha256=digest(plan),assignment_id='chosen',
        status='preparation_or_execution_failed',failure_reason='FileNotFoundError: checkpoint'))
    path=tmp_path/'config.json';write_json(path,value)
    report=c.run(path,execute=True)
    assert report['status']=='teacher_not_admitted'
    assert report['reason']=='FileNotFoundError: checkpoint'


@pytest.mark.parametrize('status,exit_code',[('export_failed',2),('teacher_not_admitted',2),
    ('teacher_outcome_unknown',2),('blocked',2),('paired_completed',0),('waiting_visual_review',0)])
def test_cli_exposes_failure_exit_codes(monkeypatch,status,exit_code):
    import runpy, sys
    from driversi.agents.backends.native import project_root
    monkeypatch.setattr(c,'run',lambda *a,**kw:dict(status=status))
    monkeypatch.setattr(sys,'argv',['run_platform_spine.py','continue','--config','unused'])
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(project_root()/'scripts/rsi/run_platform_spine.py'),run_name='__main__')
    assert error.value.code==exit_code


@pytest.mark.parametrize('mismatch',[None,'lineage','branch'])
def test_completed_assignment_reconciliation_preserves_cost_unknown(tmp_path,monkeypatch,mismatch):
    from types import SimpleNamespace
    from driversi.contracts.scene import Scene
    from driversi.rsi.plugins import chainflow_synthetic as synthesis
    value=config(tmp_path);output=Path(value['output'])/'round-01-teacher'
    output.mkdir(parents=True)
    plan_path=tmp_path/'plan.json';write_json(plan_path,{'id':'plan'})
    value['input']=dict(plan=str(plan_path),assignment='chosen')
    plan=dict(plan_id='plan',assignments=[dict(id='chosen',skill='reactive_new_history',token='train')])
    store=HarnessStore(Path(value['registry']))
    store.register(dict(id=output.name,providers=[],source_groups=[]),output/'tasks'/output.name)
    with store.registry._connect() as db:
        db.execute('UPDATE harness_tasks SET status=? WHERE id=?',('completed',output.name))
    teacher=tmp_path/'teacher.json';branch=tmp_path/'branch.json'
    write_json(teacher,dict(branches={'teacher':{'trajectory_admitted':True}}))
    write_json(branch,dict(execution_identity='wrong' if mismatch=='branch' else 'teacher-id'))
    def stored(reference,kind,registry):
        return {},dict(artifacts={'report':str(teacher if reference['source_stage']=='teacher' else branch),
            'execution/scene.npz':'scene.npz'},summary={'artifact_reference':{'identity':'teacher-id'}}),{}
    monkeypatch.setattr(synthesis,'_stored_stage',stored)
    monkeypatch.setattr(Scene,'load',lambda p:SimpleNamespace(metadata={
        'capability_skill':dict(plan_id='wrong' if mismatch=='lineage' else 'plan',assignment_id='chosen',skill='reactive_new_history'),
        'source_train_membership':{'token':'train'}}))
    monkeypatch.setattr(c.engine,'run',lambda *a:pytest.fail('Recovery must never rerun'))
    if mismatch:
        with pytest.raises(ValueError):c._reconcile_assignment(value,output,plan)
    else:
        first=c._reconcile_assignment(value,output,plan)
        assert first==c._reconcile_assignment(value,output,plan)
        assert first['status']=='trajectory_admitted' and first['total_attempt_cost_unknown']
        assert first['teacher_report']==str(teacher)
        assert (output/'recovery-teacher-reference.json').exists()
        assert not first['execution_memory_reconciled']
        assert not (output/'report.json').exists()


def test_continuation_exposes_epoch_and_per_rank_batch(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'project_root',lambda:tmp_path)
    value=config(tmp_path)
    value['training']=dict(base_manifest='/manifest',model_config='/model',updates_per_arm=256,
        capability_plan=None,skill_memory=None,epoch_training=True,batch_size=2)
    assert c.validate_config(value)['training']['batch_size']==2
    value['training']['epoch_training']=False
    with pytest.raises(ValueError,match='Legacy'): c.validate_config(value)
    value['training']['epoch_training']='true'
    with pytest.raises(ValueError,match='boolean'): c.validate_config(value)


def test_missing_review_does_not_hide_known_render_failure(tmp_path,monkeypatch):
    from driversi.rsi.experiments import readiness
    monkeypatch.setattr(c,'project_root',lambda:tmp_path)
    monkeypatch.setattr(c,'_qualified_references',lambda cfg:cfg['input'])
    monkeypatch.setattr(readiness,'inspect_stored_render',lambda *a:dict(
        blockers=['finite_terrain_support_missing'],trainable=False))
    monkeypatch.setattr(c,'task_state',lambda *a,**k:pytest.fail('Known invalid render must not launch'))
    path=tmp_path/'config.json';write_json(path,config(tmp_path))
    result=c.run(path,execute=True)
    assert result['status']=='render_not_ready' and not result['training_executed']
    assert result['readiness']['blockers']==['finite_terrain_support_missing']


def test_native_pair_compiler_receives_epoch_options(tmp_path,monkeypatch):
    from driversi.rsi.experiments import readiness
    from driversi.rsi.plugins import chainflow_synthetic
    value=config(tmp_path);value['training']=dict(base_manifest='/manifest',model_config='/model',
        updates_per_arm=256,capability_plan=None,skill_memory=None,epoch_training=True,batch_size=2)
    Path(value['export']['visual_review']).write_text('{}')
    monkeypatch.setattr(c,'project_root',lambda:tmp_path)
    monkeypatch.setattr(c,'_qualified_references',lambda cfg:cfg['input'])
    monkeypatch.setattr(readiness,'inspect_stored_render',lambda *a:dict(blockers=[]))
    monkeypatch.setattr(chainflow_synthetic,'_stored_stage',lambda *a:({},dict(artifacts={'dataset':'sample.json'}),{}))
    monkeypatch.setattr(c,'task_state',lambda registry,task,*a,**kw:dict(
        status='prepared' if task['id']=='pair' else 'completed',id=task['id'],stages=[]))
    received={}
    def compile_pair(*args,**kwargs):
        received.update(kwargs);out=args[3];out.mkdir();path=out/'task.json';write_json(path,{'id':'pair'});return path
    monkeypatch.setattr(chainflow_synthetic,'compile_paired_task',compile_pair)
    path=tmp_path/'config.json';write_json(path,value)
    result=c.run(path,execute=True)
    assert received['epoch_training'] is True and received['batch_size']==2
    assert received['updates_per_arm']==256 and result['status']=='paired_prepared'
