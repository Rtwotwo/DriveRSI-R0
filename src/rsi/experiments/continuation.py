"""Restartable handoffs across the existing Harness and native paired compiler.

This does not launch a population-wide job or treat a prepared pair as training.
"""
import fcntl
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

from driversi.agents.backends.native import project_root
from driversi.configuration import read_mapping
from driversi.rsi.contracts import _identity, _keys
from driversi.rsi.workflow import engine
from driversi.rsi.workflow.store import HarnessStore
from driversi.runtime.budget import write_json


def validate_config(config):
    _keys(config, ('schema_version','id','registry','output','model','input',
                  'render','export','training'), 'pipeline continuation')
    if config['schema_version'] != 1 or config['model'] != 'chainflow_stage1':
        raise ValueError('Synthetic paired continuation currently supports ChainFlow stage1 only')
    _identity(config['id'], 'continuation.id')
    if len(config['id']) > 90:
        raise ValueError('Continuation id must leave room for phase suffixes')
    value = config['input']
    if value.get('kind') in ('teacher','assignment'):
        if value['kind']=='teacher':
            _keys(value, ('kind','source_task','source_stage'), 'teacher source')
        else:
            _keys(value, ('kind','plan','assignment','database','memory','maps_root','python','checkpoint','device'), 'skill assignment')
        _keys(config['render'], ('provider','config','source_config'), 'render recipe')
        if config['render']['provider'] not in ('mtgs_selected_branch_render','worldengine_selected_branch_render'):
            raise ValueError('Use a registered same-branch GS renderer')
    elif value.get('kind') == 'rendered':
        _keys(value, ('kind','branch','rendered','source'), 'rendered sources')
        for k in ('branch','rendered','source'):
            _keys(value[k], ('source_task','source_stage'), k)
        if config['render'] is not None:
            raise ValueError('Stored rendering cannot be silently replaced')
    else:
        raise ValueError('Explicit skill assignment, teacher or rendered input required')
    _keys(config['export'], ('python','maps_root','max_wall_seconds','visual_review','visual_quality_profile'), 'export recipe')
    if config['training'] is not None:
        _keys(config['training'], ('base_manifest','model_config','updates_per_arm','capability_plan','skill_memory',
            *(k for k in ('epoch_training','batch_size') if k in config['training'])), 'paired recipe')
        epoch = config['training'].get('epoch_training', False)
        batch = config['training'].get('batch_size', 1)
        if type(epoch) is not bool or type(batch) is not int or not 1 <= batch <= 64:
            raise ValueError('Explicit boolean epoch training and per-rank batch size 1..64 required')
        if not epoch and batch != 1:
            raise ValueError('Legacy paired recipe uses batch size one')
        if bool(config['training']['capability_plan']) != bool(config['training']['skill_memory']):
            raise ValueError('Skill plan and memory must be supplied together')
    for key in ('registry','output'):
        path = Path(config[key])
        if not path.is_absolute() or not path.resolve().is_relative_to(project_root().resolve()/'exp'):
            raise ValueError('Registry and output must be absolute paths under exp')
    paths = [(config['export'], ('python','maps_root','visual_review'))]
    if value['kind']=='assignment':
        paths.append((value, ('plan','database','memory','maps_root','python','checkpoint')))
    if config['training'] is not None:
        paths.append((config['training'], ('base_manifest','model_config','capability_plan','skill_memory')))
    if config['render'] is not None:
        for recipe in ('config','source_config'):
            paths.append((config['render'][recipe], ('config','python','cuda_home',
                'config_path','checkpoint_path','road_block_config','bundle')))
    for mapping, keys in paths:
        for key in keys:
            if mapping.get(key) is not None and not Path(mapping[key]).is_absolute():
                raise ValueError('Continuation path must be absolute: '+key)
    return config


@contextmanager
def _lock(output):
    output.mkdir(parents=True, exist_ok=True)
    with (output/'continuation.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('Another worker owns this continuation') from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def frozen_json(path, value):
    path = Path(path)
    if path.exists():
        if read_mapping(path) != value:
            raise ValueError('Continuation declaration changed; use a new output: '+str(path))
    else:
        write_json(path, value)


def task_state(registry, task, spec_path, task_root, *, execute, evidence_guard=None):
    """Only registered tasks start. Finished or uncertain tasks never re-run."""
    frozen_json(spec_path, task)
    store = HarnessStore(registry)
    with store.registry._connect() as db:
        exists = db.execute('SELECT 1 FROM harness_tasks WHERE id=?',(task['id'],)).fetchone()
    if exists:
        state = store.get(task['id'])
        if Path(state['directory']).resolve() != (Path(task_root)/task['id']).resolve():
            raise ValueError('Task id belongs to another continuation')
    elif not execute:
        # Exercise all preparation checks, but do not acquire or launch a job.
        engine.prepare(task, Path(spec_path).parent, registry)
        return {'status':'prepared','id':task['id'],'stages':[]}
    else:
        state = engine.register(registry, spec_path, task_root)
    if execute and state['status'] == 'registered':
        return engine.run(registry, task['id'], **({'evidence_guard':evidence_guard} if evidence_guard else {}))
    return state


def render_task(config):
    render = config['render']; rid=config['id']+'-render'
    render_wall = render['config']['max_wall_seconds']+30
    source_wall = render['source_config']['max_wall_seconds']+30
    return dict(schema_version=2,id=rid,label='Qualified physical branch to GS and source comparison',
        budget=dict(max_wall_seconds=render_wall+source_wall+90,max_sim_agent_steps=0),steps=[
            dict(id='teacher',provider='completed_navsim_teacher_reference',inputs={},
                config=(dict(source_task=config['id']+'-teacher',source_stage='teacher')
                        if config['input']['kind']=='assignment' else
                        {k:config['input'][k] for k in ('source_task','source_stage')}),max_wall_seconds=30),
            dict(id='branch',provider='qualified_native_branch_selection',inputs={'world':'teacher'},
                config={'branch':'teacher','future_frames':10},max_wall_seconds=30),
            dict(id='render',provider=render['provider'],inputs={'branch':'branch'},
                config=render['config'],max_wall_seconds=render_wall),
            dict(id='source',provider='worldengine_source_reconstruction',inputs={'render':'render'},
                config=render['source_config'],max_wall_seconds=source_wall)])


def _references(config):
    if config['input']['kind']=='rendered':
        return {k:config['input'][k] for k in ('branch','rendered','source')}
    return {k:dict(source_task=config['id']+'-render',source_stage=v)
            for k,v in [('branch','branch'),('rendered','render'),('source','source')]}


def phase_summary(state):
    return dict(status=state['status'],task_id=state['id'],reason=state.get('failure_reason'),
        stages=[dict(id=x['id'],status=x['status'],reason=x.get('reason')) for x in state.get('stages',[])])


def _reconcile_assignment(config, output, plan):
    """Read an explicitly settled task; never restart or rewrite its old receipt."""
    from driversi.agents.backends.native import digest
    from driversi.contracts.scene import Scene
    from driversi.rsi.plugins.chainflow_synthetic import _stored_stage
    registry=Path(config['registry']);task_id=output.name
    store=HarnessStore(registry)
    with store.registry._connect() as db:
        exists=db.execute('SELECT 1 FROM harness_tasks WHERE id=?',(task_id,)).fetchone()
    if not exists:
        return None
    state=store.get(task_id)
    if Path(state['directory']).resolve()!=(output/'tasks'/task_id).resolve():
        raise ValueError('Teacher recovery task belongs to another assignment directory')
    if state['status']!='completed':
        return None
    _,teacher_result,_=_stored_stage(dict(source_task=task_id,source_stage='teacher'),
                                    'navsim_teacher_execution',registry)
    _,branch_result,_=_stored_stage(dict(source_task=task_id,source_stage='admission'),
                                   'selected_native_branch',registry)
    chosen=next(a for a in plan['assignments'] if a['id']==config['input']['assignment'])
    scene=Scene.load(teacher_result['artifacts']['execution/scene.npz'])
    lineage=scene.metadata.get('capability_skill',{})
    if any(lineage.get(k)!=v for k,v in dict(plan_id=plan['plan_id'],
            assignment_id=chosen['id'],skill=chosen['skill']).items()):
        raise ValueError('Recovered teacher does not belong to the declared skill assignment')
    if scene.metadata.get('source_train_membership',{}).get('token')!=chosen['token']:
        raise ValueError('Recovered teacher source token mismatch')
    teacher=read_mapping(teacher_result['artifacts']['report'])
    branch=read_mapping(branch_result['artifacts']['report'])
    admitted=teacher.get('branches',{}).get('teacher',{}).get('trajectory_admitted') is True
    if branch.get('execution_identity')!=teacher_result['summary']['artifact_reference']['identity']:
        raise ValueError('Recovered branch belongs to another teacher')
    outcome=dict(status='trajectory_admitted' if admitted else 'execution_or_admission_rejected',
        plan_sha256=digest(config['input']['plan']),assignment_id=chosen['id'],
        harness_status='completed',reconciled_from_sealed_task=task_id,
        total_attempt_cost_unknown=True,execution_memory_reconciled=False,
        note='Original interruption evidence retained; no invented cost or duplicate execution credit')
    frozen_json(output/'recovery-receipt.json',outcome)
    # Preserve the old interruption/recovery receipt; add the sealed reference
    # needed by downstream trajectory consumers without rerunning the teacher.
    teacher_path = teacher_result['artifacts']['report']
    frozen_json(output/'recovery-teacher-reference.json',
                dict(teacher_report=teacher_path, sha256=digest(teacher_path)))
    return {**outcome, 'teacher_report': teacher_path}


def _qualified_references(config):
    from driversi.rsi.plugins.chainflow_synthetic import _stored_stage, _REVIEWED_KINDS
    from driversi.rsi.correction.trajectory_admission import validate_trajectory_evidence
    refs=_references(config)
    values={k:_stored_stage(refs[k],kind,Path(config['registry']))[1] for k,kind in _REVIEWED_KINDS.items()}
    branch=read_mapping(values['branch']['artifacts']['report'])
    validate_trajectory_evidence(branch['trajectory_evidence'])
    if branch.get('future_frames')!=10:
        raise ValueError('ChainFlow continuation needs ten actual future points')
    rendered=read_mapping(values['rendered']['artifacts']['report'])
    source=read_mapping(values['source']['artifacts']['report'])
    if rendered['branch_identity']!=values['branch']['summary']['artifact_reference']['identity']:
        raise ValueError('Render belongs to another physical branch')
    if source['render_identity']!=values['rendered']['summary']['artifact_reference']['identity']:
        raise ValueError('Source comparison belongs to another rendering')
    return refs


def run(path, *, execute=False, execute_training=False):
    config=validate_config(read_mapping(path));output=Path(config['output'])
    if execute_training and not execute:
        raise ValueError('Training execution also requires execution mode')
    with _lock(output):
        frozen_json(output/'continuation.json',config)
        started=time.monotonic()
        report=dict(kind='native_rsi_continuation',id=config['id'],status='inspecting',
            invocation_pid=os.getpid(),phases={},training_executed=False,promoted=False,stable_gain_demonstrated=False)
        try:
            _advance(config,output,report,execute,execute_training)
        except (ValueError,OSError,KeyError,RuntimeError) as error:
            report.update(status='blocked',reason=type(error).__name__+': '+str(error))
        except BaseException as error:
            report.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'execution_error',
                          reason=type(error).__name__+': '+str(error))
            raise
        finally:
            report['continuation_wall_seconds']=time.monotonic()-started
            report['continuation_cost_scope']='this_invocation_not_total_pipeline_cost'
            write_json(output/'report.json',report)
            with (output/'attempts.jsonl').open('a') as log:
                log.write(json.dumps(report,allow_nan=False)+'\n')
        return report


def _advance(config, output, report, execute, execute_training):
    registry=Path(config['registry']);task_root=output/'tasks'
    if config['input']['kind']=='assignment':
        from driversi.rsi.experiments.skill_execution import run_assignment
        from driversi.rsi.decisions.capability_skills import validate_plan
        from driversi.agents.backends.native import digest
        value=config['input'];plan=validate_plan(read_mapping(value['plan']))
        if plan['context']['model']!=config['model']:
            raise ValueError('Selected skill belongs to another student')
        if not any(a['id']==value['assignment'] for a in plan['assignments']):
            raise ValueError('Assignment absent from the selected skill plan')
        destination=output/(config['id']+'-teacher');receipt=destination/'report.json'
        outcome=None
        if receipt.exists():
            outcome=read_mapping(receipt)
            if outcome.get('plan_sha256')!=digest(value['plan']) or outcome.get('assignment_id')!=value['assignment']:
                raise ValueError('Teacher attempt belongs to another skill declaration')
        if destination.exists() and (outcome is None or outcome.get('status') in
                ('running','interrupted','preparation_or_execution_failed')):
            reconciled=_reconcile_assignment(config,destination,plan)
            if reconciled is None:
                if outcome is None or outcome.get('status') in ('running','interrupted'):
                    report.update(status='teacher_outcome_unknown',reason='partial_teacher_attempt_requires_explicit_recovery');return
                # A known preparation failure may precede task registration.
                # Preserve its actual reason; there is no Harness job to recover.
            else:
                outcome=reconciled
        if outcome is None and not execute:
            report.update(status='teacher_assignment_prepared',assignment=value['assignment']);return
        elif outcome is None:
            outcome=run_assignment(value['plan'],value['assignment'],value['database'],destination,value['memory'],
                maps_root=value['maps_root'],python=value['python'],checkpoint=value['checkpoint'],
                device=value['device'],registry=registry)
        report['phases']['teacher']={'status':outcome['status'],'report':str(receipt),
                                    'reason':outcome.get('failure_reason')}
        if outcome.get('reconciled_from_sealed_task'):
            report['phases']['teacher'].update(report=str(destination/'recovery-receipt.json'),
                total_attempt_cost_unknown=True,execution_memory_reconciled=False)
        if outcome['status']!='trajectory_admitted':
            report.update(status='teacher_not_admitted',reason=outcome.get('failure_reason'));return
    if config['input']['kind'] in ('teacher','assignment'):
        state=task_state(registry,render_task(config),output/'render-task.json',task_root,execute=execute)
        report['phases']['render']=phase_summary(state)
        if state['status']!='completed':
            report['status']='render_'+state['status'];return
    refs=_qualified_references(config)
    report['source_references']=refs
    from driversi.rsi.experiments.readiness import inspect_stored_render
    readiness=inspect_stored_render(refs, registry, config['export']['visual_quality_profile'])
    write_json(output/'readiness.json',readiness)
    report['readiness']=readiness
    review=Path(config['export']['visual_review'])
    if not review.is_file():
        report.update(status='render_not_ready' if readiness['blockers'] else 'waiting_visual_review',
                      reason=';'.join(readiness['blockers']) or None,review_path=str(review));return
    task=dict(schema_version=2,id=config['id']+'-export',label='Same-branch independent admission and native dataset export',
        budget=dict(max_wall_seconds=config['export']['max_wall_seconds']+60,max_sim_agent_steps=160),
        steps=[dict(id='dataset',provider='chainflow_reviewed_synthetic_dataset',inputs={},
            config={**refs,**config['export']},max_wall_seconds=config['export']['max_wall_seconds']+30)])
    state=task_state(registry,task,output/'export-task.json',task_root,execute=execute)
    report['phases']['export']=phase_summary(state)
    if state['status']!='completed':
        report['status']='export_'+state['status']
        report['reason']=next((x.get('reason') for x in state.get('stages',[]) if x.get('reason')),state.get('failure_reason'))
        return
    from driversi.rsi.plugins.chainflow_synthetic import _stored_stage, compile_paired_task
    _,result,_=_stored_stage({'source_task':task['id'],'source_stage':'dataset'},'chainflow_admitted_synthetic_sample',registry)
    sample=result['artifacts']['dataset'];report['admitted_sample_report']=sample
    if config['training'] is None:
        report['status']='admitted_dataset_available';return
    training=config['training'];paired=output/(config['id']+'-pair');spec=paired/'task.json'
    marker=output/'paired-compile-intent.json'
    intent={**training,'sample_reports':[sample]}
    frozen_json(marker,intent)
    if not spec.exists():
        if paired.exists():
            raise ValueError('Incomplete pair preparation retained; inspect before a new attempt')
        spec=compile_paired_task(training['base_manifest'],[sample],training['model_config'],paired,
            updates_per_arm=training['updates_per_arm'],capability_plan=training['capability_plan'],
            skill_memory=training['skill_memory'],
            epoch_training=training.get('epoch_training', False),
            batch_size=training.get('batch_size', 1))
    state=task_state(registry,read_mapping(spec),spec,task_root,execute=execute_training)
    report['phases']['paired']=phase_summary(state)
    report['training_executed']=None if state['status'] not in ('prepared','registered') else False
    report['status']='paired_'+state['status']
    if state['status']=='completed':
        report['training_executed']=True
        report['feedback']=str(Path(state['directory'])/'paired_feedback/report.json')
        report['skill_memory_updated']=any(x['id']=='skill_memory' and x['status']=='completed' for x in state['stages'])
        report['next_action']='inspect_paired_gain_and_promotion_requirements'
