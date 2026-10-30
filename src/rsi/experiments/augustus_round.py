"""Compile and advance an immutable model-service program using DriveHarness.

The episode can start at an existing bounded generation plan or admitted native
samples. Neither reuse nor a completed renderer constitutes a new admission.
No arbitrary code/LLM output is executed and no navtrain student is scored.
"""
import copy
import json
from pathlib import Path

import yaml

from driversi.agents.backends.native import digest, project_root, path_at
from driversi.configuration import read_mapping
from driversi.data.catalog.store import identity
from driversi.rsi.decisions.service_memory import NODES, REPAIRS, ServiceMemory, review_learning
from driversi.runtime.budget import write_json


def _exp(path):
    path = (project_root()/path).resolve()
    if not path.is_relative_to(project_root().resolve()/'exp'):
        raise ValueError('All service outputs and memory must live under exp')
    return path


def _files(paths):
    return {str(Path(p).resolve(strict=True)): digest(p) for p in paths}


def _code_files():
    root = project_root()
    # Include shared backends, validators, runtime and vendored native evaluators,
    # not just the orchestrator's own package. Paths and hashes are evidence only.
    suffixes={'.py','.yaml','.yml','.so','.pyd','.cpp','.h','.hpp'}
    return [p for sub in ('src','agents','cpp')
            for p in (root/sub).rglob('*') if p.is_file() and p.suffix in suffixes
            and '__pycache__' not in p.parts and '.git' not in p.parts]


def _verify(files):
    for p, sha in files.items():
        if digest(p) != sha:
            raise ValueError('Frozen program input/code changed: '+p)


def branch_context(path):
    """Use the same teacher and reached-history identity for both data entries."""
    from driversi.rsi.decisions.vla_memory import history_signature
    path=Path(path).resolve(strict=True)
    branch=read_mapping(path)
    sources={**branch['files'],**branch['upstream_source_files']}
    _verify(sources)
    window=path.parent/'trajectory-window.json'
    if digest(window) != branch['trajectory_window_sha256']:
        raise ValueError('History conditions differ from selected branch')
    # Source paths/logs are evidence, not teacher version: cross-log recall must
    # remain possible when the teacher, recipe and measured conditions agree.
    teacher=dict(teacher=branch['trajectory_evidence']['teacher'],
        weights=sorted(v for p,v in sources.items() if Path(p).suffix in ('.pt','.ckpt','.safetensors')),
        native_future_points=branch['future_frames'])
    task=path.parent.parent/'task.json'
    if task.exists():
        rows=read_mapping(task)['steps']
        configs=[x['config'] for x in rows if x['provider']=='navsim_native_teacher_probe']
        if configs:
            keys=('teacher_backend','arrival_backend','arrival_action','recovery_seconds','position_gain','search_candidates')
            teacher['generation_recipe']={k:configs[0][k] for k in keys if k in configs[0]}
            sources.update(_files([task]))
    sources.update(_files([path,window]))
    return teacher,history_signature(read_mapping(window))['conditions'],sources


def generation_costs(receipts, branch_reports):
    """Only charge completed generation tasks which produced selected evidence."""
    rows=[];wanted={str(Path(p).resolve()) for p in branch_reports}
    for p in receipts:
        state=read_mapping(p)
        outputs={str(Path(v).resolve()) for s in state['stages']
                 for v in (s.get('result') or {}).get('artifacts',{}).values()}
        if state['status']!='completed' or not wanted.intersection(outputs):
            raise ValueError('Generation receipt is not a completed selected-branch producer')
        import math
        wall=state['elapsed_seconds'];steps=state['sim_agent_steps']
        if not isinstance(wall,(float,int)) or not math.isfinite(wall) or wall<0 or type(steps)is not int or steps<0:
            raise ValueError('Finite measured generation costs required')
        rows.append(dict(task_id=state['id'],receipt=str(p),wall_seconds=wall,sim_agent_steps=steps))
    if len({r['task_id']for r in rows})!=len(rows):
        raise ValueError('Duplicate generation cost receipt')
    return dict(tasks=rows,wall_seconds=sum(r['wall_seconds']for r in rows),
                sim_agent_steps=sum(r['sim_agent_steps']for r in rows),
                scope='observed_Harness_tasks_including_render_if_in_same_task',
                unmetered=['human_review','asset_preparation','unreferenced_search'],complete_total_cost=False)


def prepare(config_path):
    """Fix recipe, model bytes, evaluations, memory advice and budget before work."""
    config_path = Path(config_path).resolve(strict=True)
    c = read_mapping(config_path)
    required = {'id','output','memory','memory_mode','selection_mode','base_manifest',
                'request_template','evaluator_template','sample_reports','expansion_root'}
    if set(c)-{'publication_jobs','generation_receipts'} != required or c['selection_mode'] not in ('no_memory','fixed','memory'):
        raise ValueError('Explicit model-service program config required')
    if not c['id'] or any(x not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for x in c['id']):
        raise ValueError('Simple program identity required')
    if c['memory_mode'] not in ('evolve','frozen','off'):
        raise ValueError('Explicit memory mode required')
    output, memory = _exp(c['output']), _exp(c['memory'])
    if output.exists():
        raise ValueError('Use a new immutable program directory, or run the prepared program')
    paths = {k: (project_root()/c[k]).resolve(strict=True) for k in
             ('base_manifest','request_template','evaluator_template')}
    req = read_mapping(paths['request_template'])
    model_path = path_at(paths['request_template'].parent, req['model_config'])
    data_path = path_at(paths['request_template'].parent, req['data'])
    model, data = read_mapping(model_path), read_mapping(data_path)
    if model['model'] != 'augustus' or req['mode'] != 'supervised' or req['initialization'] != 'model_weights':
        raise ValueError('Native parent-initialized AUGUstus SFT only')
    if req['schema_version'] != 3 or not 1 <= req['budget']['max_optimizer_steps'] <= 8:
        raise ValueError('Bounded native lifecycle pilot required (1-8 updates per segment)')
    if data.get('evaluation_protocol') != 'official_navtest_v1_locked_pilot':
        raise ValueError('Locked official navtest evaluation required; no training-source scores')
    from driversi_agents.augustus.data import manifest, cache_files
    from driversi_agents.augustus.adapter import complete_assets
    original = manifest(paths['base_manifest'], allow_smoke=True)
    sample_paths = [(project_root()/p).resolve(strict=True) for p in c['sample_reports']]
    if not isinstance(c['sample_reports'], list) or len(sample_paths) > 20 or len(set(sample_paths)) != len(sample_paths):
        raise ValueError('Distinct bounded native sample list required')
    expansion = _exp(c['expansion_root']) if c['expansion_root'] else None
    publication_jobs=c.get('publication_jobs',[])
    if not isinstance(publication_jobs,list) or len(publication_jobs)>20:
        raise ValueError('Bounded explicit joint-supervision publication required')
    if not sample_paths and expansion is None and not publication_jobs:
        raise ValueError('An executable expansion or admitted samples are required')
    source_logs, conditions = set(), set()
    generators=[]
    readiness = []
    source_files = []
    publications=[]
    for job in publication_jobs:
        if set(job) != {'branch_report','render_report','visual_admission','label_review'}:
            raise ValueError('Same-branch native publication references required')
        resolved={k:str((project_root()/v).resolve(strict=True)) for k,v in job.items()}
        publications.append(resolved);source_files.extend(map(Path,resolved.values()))
        source_logs.add(read_mapping(resolved['branch_report'])['source_log'])
        teacher,facts,files=branch_context(resolved['branch_report'])
        generators.append(teacher);conditions.update(facts);source_files.extend(map(Path,files))
    if expansion:
        from driversi.rsi.experiments.branch_expansion import verified_plan
        generation = verified_plan(expansion)
        source_logs.add(generation['source']['source_log'])
        checkpoint=generation['teacher_template']['steps'][0]['config']['checkpoint']
        generators.append(dict(teacher_checkpoint=digest(checkpoint),
                               backend=generation['teacher_template']['steps'][0]['config']['teacher_backend']))
        source_files += [expansion/'plan.json']
    # Preview is insufficient. A failed admission is an observable blocked node,
    # not a preparation crash or permission to refresh historical hashes.
    from driversi_agents.augustus.branch_supervision import validate_sample
    for p in sample_paths:
        try:
            s = validate_sample(p)
        except (ValueError,FileNotFoundError) as exc:
            readiness.append(dict(path=str(p),admitted=False,reason=type(exc).__name__+': '+str(exc)))
            source_files.append(p)
            conditions.add('joint_supervision_evidence_version_mismatch')
            continue
        readiness.append(dict(path=str(p),admitted=True))
        source_logs.add(s['source_log'])
        source_files.extend([p, *map(Path, s['files'])])
        teacher,facts,files=branch_context(s['branch_report'])
        generators.append(teacher);conditions.update(facts);source_files.extend(map(Path,files))
    receipts=[(project_root()/p).resolve(strict=True) for p in c.get('generation_receipts',[])]
    branches=[j['branch_report']for j in publications]+[read_mapping(p)['branch_report']for p in sample_paths if read_mapping(p).get('branch_report')]
    measured_generation=generation_costs(receipts,branches)
    source_files.extend(receipts)
    parent = path_at(project_root(), model['weights']['parent'])
    assets = _files(complete_assets(parent))
    code_files=_files(_code_files())
    context = dict(model='augustus', parent_assets_id=identity(assets),
                   program_code_id=identity({str(Path(p).relative_to(project_root())):v for p,v in code_files.items()}), generator_context_id=identity(generators),
                   native_recipe_id=identity(dict(model={k:v for k,v in model.items() if k != 'paths'},
                       request={k:v for k,v in req.items() if k not in ('id','model_config','data')})),
                   evaluator_id=identity(_files([paths['evaluator_template'], data['evaluation_subset']])),
                   service_version=2)
    store = ServiceMemory(memory, 'off' if c['selection_mode'] != 'memory' else 'frozen')
    try:
        advice = store.advise(context, source_logs, conditions=conditions)
    finally:
        store.close()
    inputs = [config_path, *paths.values(), model_path, data_path, Path(data['evaluation_subset']),
              *source_files, *map(Path, cache_files(original))]
    plan = dict(kind='augustus_service_program', schema_version=1, id=c['id'],
                output=str(output), memory=str(memory), memory_mode=c['memory_mode'],
                selection_mode=c['selection_mode'], context=context, source_logs=sorted(source_logs),
                conditions=sorted(conditions), advice=advice, nodes=list(NODES),
                request_template=str(paths['request_template']), model_template=str(model_path),
                data_template=str(data_path), evaluator_template=str(paths['evaluator_template']),
                base_manifest=str(paths['base_manifest']), sample_reports=list(map(str,sample_paths)),
                sample_readiness=readiness,
                publication_jobs=publications,
                generation_costs=measured_generation,
                expansion_root=str(expansion) if expansion else None,
                entry='bounded_generation' if expansion else 'reuse_independently_admitted_data',
                files={**_files(inputs), **assets}, code_files=code_files,
                automatic_promotion=False, navtrain_student_scoring=False,
                skill_attribution='no_generation_policy_credit_without_prospective_assignment')
    plan['program_id'] = identity(plan)
    output.mkdir(parents=True)
    write_json(output/'program.json', plan)
    return plan


def verify_program(output):
    plan = read_mapping(Path(output)/'program.json')
    if plan['program_id'] != identity({k:v for k,v in plan.items() if k != 'program_id'}):
        raise ValueError('Immutable program changed')
    _verify(plan['files']); _verify(plan['code_files'])
    if _files(_code_files()) != plan['code_files']:
        raise ValueError('Executable code closure changed')
    return plan


def _task(task_id, request, evaluator):
    req = read_mapping(request)
    cap = req['budget']['max_wall_seconds'] + 300
    steps = [dict(id='train', provider='native_model_unified_training', inputs={}, config={'request':str(request)}, max_wall_seconds=cap)]
    for name in ('resume','replay'):
        steps.append(dict(id=name, provider='native_model_unified_resume', inputs={'student':'train'},
                          config={'request':str(request)}, max_wall_seconds=cap))
    steps += [dict(id='evaluate',provider='native_model_unified_evaluation',inputs={'student':'resume'},
                   config={'request':str(request),'evaluator':str(evaluator)},max_wall_seconds=cap),
              dict(id='feedback',provider='native_model_unified_feedback',inputs={'student':'resume','replay':'replay','evaluation':'evaluate'},config={},max_wall_seconds=90)]
    return dict(schema_version=2,id=task_id,label='Immutable AUGUstus paired model-service program',
                budget={'max_wall_seconds':4*cap+120,'max_sim_agent_steps':4*req['budget']['max_scoring_agent_steps']},steps=steps)


def memory_preflight(plan):
    """Execute evidence checks recommended by memory; old cases are not verdicts."""
    actions = plan['advice']['preflight_actions'] if plan['selection_mode']=='memory' else []
    if plan['selection_mode']=='fixed':
        actions=['verify_asset_coverage_and_registration_before_rollout']
    performed=[]
    for action in actions:
        if action == 'verify_asset_coverage_and_registration_before_rollout' and plan['expansion_root']:
            from driversi.rsi.experiments.branch_expansion import verified_plan
            current=verified_plan(plan['expansion_root'])
            performed.append(dict(skill='gs_asset_closure_preflight',action=action,
                current_plan_id=current['plan_id'],result='complete_asset_files_verified',
                visually_admitted=False,reason='Asset presence alone is not image qualification'))
        elif action in ('revise_data_intervention_then_repeat_matched_trial','expand_independent_validation'):
            performed.append(dict(skill='paired_protocol_preflight',action=action,
                result='exact_parent_native_recipe_and_locked_test_revalidated',
                evidence_id=identity(plan['context']),student_benefit_established=False))
        else:
            performed.append(dict(skill='deferred_specialist_repair',action=action,result='requires_new_revision',
                                  automatically_executed=False))
    return dict(kind='service_memory_preflight',selection_mode=plan['selection_mode'],performed=performed,
                verdict_reused=False,mandatory_admission_unchanged=True)


def compile_pair(plan, samples):
    """Use exact original pool and original loss; only added cache differs."""
    from driversi_agents.augustus.branch_supervision import build_training_pair
    from driversi_agents.augustus.data import manifest
    out = Path(plan['output'])/'pair'
    if out.exists():
        sealed = read_mapping(out/'compiled.json')
        _verify(sealed['files'])
        if sealed['program_id'] != plan['program_id'] or sealed['samples'] != _files(samples):
            raise ValueError('Previously compiled pair differs from program')
        return sealed
    out.mkdir()
    pair = build_training_pair(plan['base_manifest'], samples, out/'data', allow_smoke=True)
    files, tasks = [], {}
    for arm, key in (('control','control'),('treatment','intervention')):
        d = out/arm; d.mkdir()
        mp = Path(pair[key])
        data = read_mapping(plan['data_template'])
        data['manifest'] = str(mp)
        data['train']['source_logs'] = sorted({r['log_name'] for r in manifest(mp,allow_smoke=True)['splits']['train']['records']})
        write_json(d/'data.json',data)
        model = read_mapping(plan['model_template'])
        model['paths']['dataset_manifest'] = str(mp)
        model['paths']['development_metric_cache'] = data['metric_cache']
        (d/'model.yaml').write_text(yaml.safe_dump(model))
        req = read_mapping(plan['request_template'])
        req.update(id=plan['id']+'-'+arm, model_config=str(d/'model.yaml'),data=str(d/'data.json'))
        write_json(d/'request.json', req)
        evaluator = read_mapping(plan['evaluator_template'])
        evaluator['dataset_manifest_sha256'] = digest(mp)
        write_json(d/'evaluator.json',evaluator)
        write_json(d/'task.json',_task(req['id'],d/'request.json',d/'evaluator.json'))
        tasks[arm] = str(d/'task.json')
        files += list(d.glob('*'))
    files += [p for p in (out/'data').rglob('*') if p.is_file()]
    sealed = dict(program_id=plan['program_id'],tasks=tasks,samples=_files(samples),files=_files(files))
    write_json(out/'compiled.json',sealed)
    return sealed


def _run_task(plan, task_path, execute):
    from driversi.rsi.workflow import engine
    from driversi.rsi.workflow.store import HarnessStore
    from driversi.rsi.workflow.artifacts import verify
    output = Path(plan['output']); registry = output/'harness.sqlite'
    task_id = read_mapping(task_path)['id']
    store = HarnessStore(registry)
    try:
        state = store.get(task_id)
    except ValueError as e:
        if str(e) != 'Unknown Harness task': raise
        engine.register(registry,task_path,output/'tasks')
        state = store.get(task_id)
    if state['status'] == 'registered' and execute:
        verify_program(output)
        state = engine.run(registry,task_id,evidence_guard=lambda: verify_program(output))
    if state['status'] == 'running':
        # Owned-child reconciliation, not blind replay of partially applied steps.
        state = engine.recover(registry,task_id)
    for row in state['stages']:
        if row['status'] == 'completed':
            declaration = next(x for x in state['task']['steps'] if x['id']==row['id'])
            verify(row['result'],task_id,declaration)
    return state


def _finalize(plan, report):
    out = Path(plan['output'])
    report.update(program_id=plan['program_id'], context=plan['context'],
                  source_logs=plan['source_logs'], conditions=plan['conditions'],
                  promoted=False, skill_gain_credited=False,
                  next_action=REPAIRS[report['node']] if report['status'] != 'pilot_signal' else 'expand_independent_validation')
    write_json(out/'review.json',report)
    evidence = {**_files([out/'program.json',out/'review.json']),
                **report.get('execution_evidence', {}), **report.get('generation_report', {})}
    observation = {**report,'evidence':evidence}
    memory = ServiceMemory(plan['memory'],plan['memory_mode'])
    try:
        written = memory.record(observation)
        advisory = memory.advise(plan['context'],plan['source_logs'],conditions=plan['conditions'])
    finally:
        memory.close()
    write_json(out/'memory-receipt.json',written)
    write_json(out/'next-iteration.json',dict(advisory=advisory,student_parent_unchanged=True,
               next_action=report['next_action'], requires_new_program_revision=True,
               executed_new_training=False, evidence=evidence))
    return report


def task_failure(state):
    """Retain the actual failed node and spent cost, including partial updates."""
    failed=next((s for s in state['stages'] if s['status']!='completed'),None)
    node=('independent_evaluation' if failed and failed['id']=='evaluate' else
          'resume_verification' if failed and failed['id'] in ('resume','replay','feedback') else 'paired_training')
    receipt=Path(state['directory'])/'report.json'
    return dict(status='blocked',node=node,reason='native_harness_task_not_completed',
                training_executed=None,harness_status=state['status'],observed_gain=None,
                failed_stage={k:failed.get(k) for k in ('id','status','reason')} if failed else None,
                stages=[{k:s.get(k) for k in ('id','status','reason','sim_agent_steps')} for s in state['stages']],
                execution_evidence=_files([receipt]) if receipt.exists() else {},
                partial_cost={'harness_wall_seconds':state.get('elapsed_seconds'),
                              'sim_agent_steps':state.get('sim_agent_steps')},complete_total_cost=False)


def verify_publication_job(sample, job):
    """A same-branch cache with different rendering/labels is not this program."""
    for key in ('branch_report','visual_admission','label_review'):
        if Path(sample[key]).resolve() != Path(job[key]).resolve():
            raise ValueError('Published '+key+' differs from immutable job')
    visual=read_mapping(sample['visual_admission'])
    if Path(visual['render_report']).resolve() != Path(job['render_report']).resolve():
        raise ValueError('Published render differs from immutable job')
    for p in job.values():
        if sample['files'].get(str(Path(p).resolve())) != digest(p):
            raise ValueError('Published job evidence hash differs')


def run(output, *, execute=False):
    """Advance once. Resume reuses completed tasks; rejects unknown/failed work."""
    import fcntl
    out = _exp(output)
    with (out/'program.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan = verify_program(out)
        if (out/'review.json').exists():
            review = read_mapping(out/'review.json')
            _verify(review.get('execution_evidence', {})); _verify(review.get('generation_report', {}))
            if (out/'next-iteration.json').exists():
                receipt = read_mapping(out/'next-iteration.json')
                _verify(receipt['evidence'])
                return review
            # The process may have died after writing review but before memory.
            # Deterministic replay completes writeback without training again.
            return _finalize(plan,review)
        samples = list(plan['sample_reports'])
        preflight_path=out/'preflight.json'
        if not preflight_path.exists():
            write_json(preflight_path,memory_preflight(plan))
        if any(not r['admitted'] for r in plan['sample_readiness']):
            return _finalize(plan,dict(status='blocked',node='joint_supervision',
                training_executed=False,reason='declared_sample_admission_no_longer_valid',
                sample_readiness=plan['sample_readiness'],observed_gain=None,
                causal_attribution='evidence_version_mismatch_not_student_failure'))
        if plan['publication_jobs']:
            from driversi_agents.augustus.branch_supervision import publish,validate_sample
            for i,job in enumerate(plan['publication_jobs']):
                dest=out/'published'/str(i)
                try:
                    if dest.exists():
                        s=validate_sample(dest/'report.json')
                    else:
                        publish(**job,output=dest)
                        s=validate_sample(dest/'report.json')
                    verify_publication_job(s,job)
                    from driversi.rsi.services.synthetic_classification import complete
                    classification = complete(dest/'native/export.json', dest/'report.json')
                    write_json(dest/'scene-classification.json', classification)
                    samples.append(str(dest/'report.json'))
                except (ValueError,FileNotFoundError) as exc:
                    return _finalize(plan,dict(status='blocked',node='joint_supervision',
                        training_executed=False,reason=type(exc).__name__+': '+str(exc),observed_gain=None))
        if plan['expansion_root']:
            from driversi.rsi.experiments import branch_expansion
            try:
                if execute:
                    generation = branch_expansion.run(plan['expansion_root'])
                else:
                    p = Path(plan['expansion_root'])/'report.json'
                    generation = read_mapping(p) if p.exists() else None
                if samples and generation:
                    generation = branch_expansion.attach_admissions(plan['expansion_root'],samples)
            except (ValueError,FileNotFoundError,RuntimeError) as exc:
                return _finalize(plan,dict(status='blocked',node='physical_generation',
                    training_executed=False,reason=type(exc).__name__+': '+str(exc),
                    causal_attribution='generation_or_sealed_evidence_failure_not_student_failure',observed_gain=None))
            if generation is None:
                return dict(status='prepared',node='physical_generation',training_executed=False)
            samples = [r['admission_report'] for r in generation['rows'] if r.get('trainable')]
            if not samples:
                node = 'visual_admission' if generation['rendered'] else 'trajectory_admission'
                return _finalize(plan,dict(status='blocked',node=node,training_executed=False,
                    reason='no_independently_admitted_native_samples',
                    generation_counts={k:generation[k] for k in ('attempts','teacher_qualified','rendered','trainable_samples')},
                    generation_report=_files([Path(plan['expansion_root'])/'report.json']),
                    causal_attribution='unresolved',observed_gain=None))
        try:
            pair = compile_pair(plan,samples)
        except (ValueError,FileNotFoundError) as exc:
            return _finalize(plan,dict(status='blocked',node='joint_supervision',
                training_executed=False,reason=type(exc).__name__+': '+str(exc),observed_gain=None))
        if not execute:
            return dict(status='prepared',node='paired_training',pair=pair['tasks'],training_executed=False)
        states=[]
        for arm in ('control','treatment'):
            _verify(pair['files'])
            try:
                state=_run_task(plan,pair['tasks'][arm],True);states.append(state)
            except (ValueError,FileNotFoundError,RuntimeError) as exc:
                return _finalize(plan,dict(status='blocked',node='paired_training',
                    failed_arm=arm,training_executed=None,reason=type(exc).__name__+': '+str(exc),observed_gain=None))
            if state['status'] != 'completed':
                return _finalize(plan,dict(task_failure(state),failed_arm=arm))
        verify_program(out)
        from driversi.rsi.experiments.augustus_pair import compare
        try:
            feedback=compare(*(Path(s['directory']) for s in states),generation_costs=plan.get('generation_costs'))
        except (ValueError,FileNotFoundError) as exc:
            return _finalize(plan,dict(status='blocked',node='independent_evaluation',
                training_executed=True,reason=type(exc).__name__+': '+str(exc),observed_gain=None))
        write_json(out/'paired-feedback.json',feedback)
        checks=[read_mapping(Path(s['directory'])/'feedback/feedback.json')['cross_process_resume'] for s in states]
        review=review_learning(feedback,checks)
        execution_evidence = {**feedback['evidence_files'], **_files([out/'paired-feedback.json'])}
        execution_evidence.update(_files([Path(s['directory'])/'feedback/feedback.json' for s in states]))
        # Entire task wall includes replay; native lineage intentionally does not.
        # Keep both views as components, never add them to each other.
        task_costs={arm:dict(wall_seconds=s.get('elapsed_seconds'),sim_agent_steps=s.get('sim_agent_steps'))
                    for arm,s in zip(('control','treatment'),states)}
        task_receipts=[Path(s['directory'])/'report.json' for s in states]
        execution_evidence.update(_files(task_receipts))
        review.update(training_executed=True,observed_gain=review['observed_gain'],
                      data_entry=plan['entry'],admitted_samples=len(samples),
                      evaluation_scope=feedback['evaluation_scope'],evaluation_samples=feedback['evaluation_samples'],
                      resume_checks=checks,paired_feedback=_files([out/'paired-feedback.json']),
                      partial_cost=dict(native_learning=feedback['training_evaluation_costs'],
                          harness_tasks_including_replay=task_costs,generation=plan.get('generation_costs'),
                          components_overlap_do_not_sum=True),complete_total_cost=False,
                      generation_policy_causal_credit=False,execution_evidence=execution_evidence)
        return _finalize(plan,review)


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('prepare').add_argument('config')
    q=sub.add_parser('run');q.add_argument('output');q.add_argument('--execute',action='store_true')
    a=p.parse_args()
    report=prepare(a.config) if a.command=='prepare' else run(a.output,execute=a.execute)
    print(json.dumps({k:report.get(k) for k in ('program_id','status','node','training_executed','observed_gain','promoted')},allow_nan=False))


if __name__=='__main__':main()
