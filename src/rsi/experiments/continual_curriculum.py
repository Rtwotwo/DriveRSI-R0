"""One failure task at a time, using native generation and paired training.

Private state owns test identities. The display projection exports aggregates
and opaque problem labels only. This is a foreground command, not a monitor.
"""

import argparse
import copy
import fcntl
import json
import os
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

from driversi.agents.backends.native import digest, project_root
from driversi.configuration import read_mapping
from driversi.rsi.decisions.augmentation_budget import build_budget, require_localization_free
from driversi.rsi.decisions.continual_learning import SAFETY, scores
from driversi.rsi.experiments.continuation import frozen_json
from driversi.runtime.budget import write_json


@contextmanager
def locked(root):
    root = Path(root).resolve()
    if not root.is_relative_to(project_root() / 'exp'):
        raise ValueError('Curriculum must live under exp')
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'curriculum.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield root


def state_path(root):
    return Path(root) / 'private/state.json'


def project(state):
    """Explicit projection: never copy a test token, caption, path or prediction."""
    current = next((c for c in state['cases'] if c['id'] == state['active']), None)
    collection = None
    if state.get('dataset_collection'):
        path = Path(state['dataset_collection'])
        contract = read_mapping(path)
        index = read_mapping(path.parent / 'manifests' / contract['split'] / 'index.json')
        collection = dict(root='dataset/', split=contract['split'], samples=index['count'],
                          history_frames=contract['history_frames'], future_frames=contract['future_frames'],
                          storage=contract['storage'], native_target_unchanged=True,
                          trajectory_quality=contract.get('trajectory_quality'))
    return dict(schema_version=1, kind='current_continual_curriculum',
                dataset=collection,
                model=state['model'], revision=state['incumbent']['revision'],
                pdms=state['incumbent']['pdms'], cases=len(state['cases']),
                states=dict(Counter(c['status'] for c in state['cases'])),
                active=None if current is None else dict(
                    id=current['id'], status=current['status'], round=current['round'],
                    candidates=len(current['candidates']), target=current['target'],
                    qualified=current.get('qualified', 0),
                    asset_states=current.get('asset_states', {}),
                    failure_classes=current['failure_classes']),
                generation_started=state.get('generation_started', False),
                training_started=state.get('training_started', False),
                localization_used_for_selection=False, location_role='audit_only',
                candidate_selection=f"visual_top{state['policy']['max_candidates']}_no_geographic_filter",
                training_population=state['policy'].get('training_population', 'mixed'),
                visual_quality_profile=state['policy'].get('visual_quality_profile'),
                source_psnr_strictly_above_db=state['policy'].get('strict_source_psnr_gt'),
                target_success_rule='PDMS>0' if state['policy'].get('strict_target') else 'threshold',
                adaptive_evaluation=True, promoted_rounds=state['incumbent']['revision'],
                automatic_execution=None if not state.get('automatic_execution') else {
                    k: state['automatic_execution'][k] for k in (
                        'status', 'started_at', 'updated_at', 'transitions',
                        'max_problems', 'max_transitions')},
                policy={k: state['policy'][k] for k in (
                    'max_candidates', 'targets', 'max_rounds', 'mastery_pdms',
                    'attempts_per_source', 'replay_limit')})


def save(root, state):
    write_json(state_path(root), state)
    write_json(Path(root) / 'current.json', project(state))
    from driversi.visualization.curriculum_operations import refresh
    refresh(root)


def initialize(root, queue_root, wave_config, policy_path):
    root, queue_root = Path(root), Path(queue_root).resolve()
    if state_path(root).exists():
        raise ValueError('Curriculum already exists; use advance to resume')
    policy = read_mapping(policy_path)
    new_only = policy.get('training_population') == 'synthetic_only'
    cohort_mode=policy.get('cohort_mode')
    if cohort_mode not in (None,'global_50_min1_v1'):
        raise ValueError('Unknown cohort policy')
    if cohort_mode and not new_only:raise ValueError('Global cohort requires synthetic-only training')
    if new_only:
        from driversi.rendering.diagnostics.numeric_admission import strict_problem_profile
        if (policy['max_candidates'] != 20 or policy['max_rounds'] != 3
                or policy['targets'] != [10, 20, 20] or policy['mastery_pdms'] != 0
                or policy['strict_target'] is not True or policy['replay_limit'] != 0
                or policy['sft_epochs'] != 4 or policy['attempts_per_source'] != 32
                or policy['source_caps'] != [5,5,6] or policy['similarity_margins'] != [.03,.03,.06]
                or not strict_problem_profile(policy.get('visual_quality_profile'),policy.get('strict_source_psnr_gt'))):
            raise ValueError('Use the declared adaptive-source new-only preset')
    elif (policy['max_candidates'] != 3 or policy['max_rounds'] != 2
            or policy['targets'] != [10, 20] or policy['mastery_pdms'] != 0.95
            or policy['attempts_per_source'] != 32 or policy['replay_limit'] != 32):
        raise ValueError('Use the declared bounded curriculum preset')
    c = read_mapping(wave_config)
    if c.get('dataset_collection'):
        from driversi.data.datasets.simscale_collection import require_campaign
        path = Path(c['dataset_collection']).resolve(strict=True)
        if path != project_root() / 'dataset/collection.json' or require_campaign(path.parent, root) is None:
            raise ValueError('Curriculum must own the active generated dataset')
    if c.get('student_model') != 'chainflow_stage1':
        raise ValueError('Failure curriculum is bound to ChainFlow, not AUGUstus')
    queue = read_mapping(queue_root / 'training-production-queue.json')
    require_localization_free(read_mapping(queue_root / 'summary.json'),
                              read_mapping(queue_root / 'inputs.json'), queue)
    baseline = Path(read_mapping(c['baseline_config'])['output']) / 'baseline/report.json'
    report = read_mapping(baseline)
    expected = [r['token'] for r in report['rows']]
    values = scores(report, expected)
    checkpoint = str(Path(c['checkpoint']).resolve())
    sha = digest(checkpoint)
    if len(expected) != 12146 or report['files'].get(checkpoint) != sha:
        raise ValueError('Frozen full benchmark must belong to the initial checkpoint')
    mapped = read_mapping(queue_root / 'private/visual-query-mapping.json')['rows'][f"visual_top{policy['max_candidates']}"]
    if set(mapped) != {t for t, m in values.items() if m['PDMS'] == 0}:
        raise ValueError('Query set differs from this model baseline')
    known = {r['token']: r for r in queue['rows']}
    if set(known) & set(expected):
        raise ValueError('Training candidates overlap evaluation identities')
    cases = []
    for i, token in enumerate(sorted(mapped)):
        candidates = [dict(known[r['token']], max_similarity=r['similarity'], best_rank=r['original_rank'])
                      for r in mapped[token][:policy['max_candidates']]]
        if new_only and (not candidates or len({c['source_group'] for c in candidates}) != len(candidates)):
            raise ValueError('Problem sources must be nonempty independent training groups')
        from driversi.rsi.decisions.problem_learning import select_problem_sources
        first_count = len(select_problem_sources(candidates,0,policy)) if new_only else 1
        cases.append(dict(id=f'problem-{i+1:03d}', query=token, candidates=candidates,
                          status='queued', round=0, target=50 if cohort_mode else 10*first_count,
                          failure_classes=[k for k in SAFETY if values[token][k] < 1]))
    if new_only:
        # Readiness prioritizes which problem to attempt; it never replaces or
        # reranks that problem's visually retrieved training sources.
        budget = read_mapping(queue_root/'global-budget.json')
        registered = {r['token'] for r in budget['states'] if r['state']=='asset_registered_unadmitted'}
        cases.sort(key=lambda case:(-sum(r['token'] in registered for r in
            select_problem_sources(case['candidates'],0,policy)),case['id']))
    private = root / 'private'
    private.mkdir(parents=True, exist_ok=True)
    os.chmod(private, 0o700)
    frozen_json(root / 'declaration.json', dict(
        model=c['student_model'], queue_root=str(queue_root), wave_config=str(Path(wave_config).resolve()),
        policy=policy, baseline=str(baseline), localization_used_for_selection=False,
        adaptive_evaluation=True, inputs={str(p): digest(p) for p in (
            Path(wave_config), Path(policy_path), baseline, queue_root / 'inputs.json',
            queue_root / 'training-production-queue.json', queue_root / 'global-budget.json',
            queue_root / 'private/visual-query-mapping.json')}))
    state = dict(model=c['student_model'], policy=policy, cases=cases, active=None,
                 incumbent=dict(checkpoint=checkpoint, sha256=sha, evaluation=str(baseline),
                                revision=0, pdms=100 * report['metrics']['PDMS']),
                 retained=[], replay=[], rounds=[], generation_started=False, training_started=False)
    if c.get('dataset_collection'):
        state['dataset_collection'] = c['dataset_collection']
    save(root, state)
    return project(state)


def prepare(root):
    state = read_mapping(state_path(root))
    declaration = read_mapping(root / 'declaration.json')
    for p, sha in declaration['inputs'].items():
        if digest(p) != sha:
            raise ValueError('Declared curriculum input changed; version the campaign')
    from .asset_recovery import ready_sources, policy as recovery_policy
    recovery = recovery_policy(root)
    recovered = ready_sources(root) if recovery else {}
    if recovery:
        binding=('asset-recovery-policy-binding.json' if recovery['strict_source_psnr_gt']==27.
                 else 'asset-recovery-policy25-spatial-binding.json')
        frozen_json(root/binding,dict(policy=recovery))
        for case in state['cases']:
            from .source_expansion import readiness
            if case['status']=='asset_recovery_pending' and readiness(case,recovered):
                case['status']='queued'
                case['asset_revision']=case.get('asset_revision',0)+1
        # Readiness chooses the next problem; it cannot alter its visual pool.
        state['cases'].sort(key=lambda case:(
            -sum(r['token'] in recovered for r in case['candidates']),case['id']))
    active = next((c for c in state['cases'] if c['id'] == state['active']), None)
    if active is None:
        active = next((c for c in state['cases'] if c['status'] == 'queued'), None)
        if active is None:
            return project(state)
        state['active'] = active['id']
    if active['status'] in ('awaiting_release', 'dispatching', 'unknown_execution', 'training', 'data_ready'):
        raise ValueError('Settle current result before preparing another round')
    if active['round'] >= state['policy']['max_rounds']:
        active['status'] = 'budget_exhausted'
        state['active'] = None
        save(root, state)
        return project(state)
    source = Path(declaration['queue_root'])
    old_budget = read_mapping(source / 'global-budget.json')
    states = {r['token']: r['state'] for r in old_budget['states']}
    if recovery:
        # A legacy registry entry cannot substitute for current strict27 proof.
        states={token:('needs_asset_repair' if value=='asset_registered_unadmitted' else value)
                for token,value in states.items()}
    states.update({token:'asset_registered_unadmitted' for token in recovered})
    active['asset_states'] = dict(Counter(states[r['token']] for r in active['candidates']))
    ready = [r for r in active['candidates'] if states[r['token']] == 'asset_registered_unadmitted']
    new_only = state['policy'].get('training_population') == 'synthetic_only'
    from driversi.rsi.decisions.problem_learning import select_problem_sources
    from .generation_policy import started_attempts
    spent=started_attempts(root,active['id'])
    cumulative_cap=state['policy']['attempts_per_source']*state['policy']['max_rounds']
    global_mode=state['policy'].get('cohort_mode')=='global_50_min1_v1'
    target=50 if global_mode else state['policy']['targets'][active['round']]
    eligible=[r for r in active['candidates'] if cumulative_cap-spent.get(r['token'],0)>=(1 if global_mode else target)]
    if not eligible:
        active.update(status='generation_budget_exhausted',charged_attempts=spent)
        state['active']=None;save(root,state);return project(state)
    selected = (select_problem_sources(eligible,active['round'],state['policy'])
                if new_only else ([r for r in ready if r in eligible][:1] or eligible[:1]))
    if active.get('source_search'):
        from .source_expansion import select
        selected=select(active,eligible,1 if global_mode else target)
        if not selected:
            active['status']='generation_budget_exhausted';state['active']=None
            save(root,state);return project(state)
        active['source_selected_tokens']=[r['token'] for r in selected]
    if recovery:
        available=[r for r in selected if r['token'] in recovered]
        if available and not active.get('source_search'):
            active['pending_source_tokens']=[r['token'] for r in selected if r['token'] not in recovered]
            selected=available
        elif active.get('source_search'):
            active['pending_source_tokens']=[r['token'] for r in selected if r['token'] not in recovered]
    active['source_selected_tokens']=[r['token'] for r in selected]
    target = 50 if global_mode else state['policy']['targets'][active['round']]
    total_target = 50 if global_mode else target * len(selected)
    active['target'] = total_target
    folder = root / 'rounds' / f"{active['id']}-r{active['round']+1}"
    if active.get('asset_revision'):
        folder=folder.with_name(folder.name+f"-assets{active['asset_revision']}")
    folder.mkdir(parents=True, exist_ok=True)
    mining = folder / 'selection-input'
    mining.mkdir(exist_ok=True)
    frozen_json(mining / 'summary.json', read_mapping(source / 'summary.json'))
    narrowed = dict(rows=selected, contains_test_tokens=False, localization_used_for_selection=False)
    frozen_json(mining / 'training-production-queue.json', narrowed)
    registered = [t for t, s in states.items() if s == 'asset_registered_unadmitted']
    repair = [t for t, s in states.items() if s == 'needs_asset_repair']
    attempts=min(state['policy']['attempts_per_source'],*[cumulative_cap-spent.get(r['token'],0) for r in selected])
    budget = build_budget(selected, registered, repair, target=target,
                          attempts=attempts, wave_size=len(selected),
                          **({'global_target':50} if global_mode else {}))
    from driversi.data.catalog.store import identity
    budget['queue_identity'] = identity(narrowed)
    frozen_json(folder / 'global-budget.json', budget)
    config = read_mapping(declaration['wave_config'])
    placement_path=root/'execution-placement.json'
    if placement_path.exists():
        placement=read_mapping(placement_path)
        if set(placement)!={'production_host','evaluation_host','reason'} or not placement['reason']:
            raise ValueError('Explicit execution placement required')
        frozen_json(folder/'execution-placement.json',dict(placement=placement,sha256=digest(placement_path)))
        config.update({k:placement[k] for k in ('production_host','evaluation_host')})
    if recovery and all(r['token'] in recovered for r in selected):
        from driversi.data.catalog.store import identity
        files={}
        rows=[]
        for candidate in selected:
            receipt=recovered[candidate['token']]
            proof=read_mapping(receipt['source_report'])
            files.update(proof['inputs'])
            files[receipt['source_report']]=receipt['source_sha256']
            rows.append(dict(receipt['asset'],source_quality_evidence=receipt['source_report'],
                             source_quality_mean_db=receipt['mean_psnr_db']))
        coverage=dict(kind='training_gs_source_coverage',population='navtrain',schema_version=1,
            files=files,rows=rows,input_id=identity(dict(files=files,rows=rows)),diagnostics=[],
            training_members=103288,trainable=False,
            claim=f"strict{int(recovery['strict_source_psnr_gt'])}_source_preflight_not_branch_admission")
        frozen_json(folder/'recovered-coverage.json',coverage)
        config['coverage']=str(folder/'recovered-coverage.json')
    provenance = read_mapping(source / 'inputs.json')
    if state['incumbent']['revision'] > 0:
        # New derivation, not re-signing the original retrieval/evaluation receipt.
        baseline = read_mapping(state['incumbent']['evaluation'])
        baseline.update(completed=12146, expected=12146)
        current_base = folder / 'incumbent-context'
        frozen_json(current_base / 'baseline/report.json', baseline)
        base_config = read_mapping(config['baseline_config'])
        base_config.update(output=str(current_base), checkpoint=state['incumbent']['checkpoint'])
        frozen_json(current_base / 'config.json', base_config)
        config['baseline_config'] = str(current_base / 'config.json')
        provenance = dict(provenance, config_sha256=digest(config['baseline_config']),
                          baseline_sha256=digest(current_base / 'baseline/report.json'),
                          frozen_retrieval_origin=str(source / 'inputs.json'),
                          retrieval_recomputed=False,
                          selection_purpose='existing_candidate_pool_for_current_failure')
    frozen_json(mining / 'inputs.json', provenance)
    config.update(output=str(folder / 'waves/wave-0001'), mining_root=str(mining),
                  max_candidates=len(selected), max_samples=total_target, max_rendered_branches=total_target*2,
                  minimum_training_samples=total_target, enable_training_after_wave=False,
                  checkpoint=state['incumbent']['checkpoint'],
                  global_budget=str(folder / 'global-budget.json'),
                  global_budget_sha256=digest(folder / 'global-budget.json'))
    config['trajectory_family']['target_per_anchor'] = target
    if global_mode:
        config['trajectory_family']['cohort_mode']='global_50_min1_v1'
        config['trajectory_family']['anchor_budget_mode']='explore_until_global_limit'
    config['trajectory_family']['attempts_per_anchor'] = attempts
    config['trajectory_family']['seed_start'] += active['round'] * state['policy']['attempts_per_source']
    config['max_pair_optimizer_steps'] = 256 if new_only else 128
    model = read_mapping(config['training_model_config'])
    model['weights']['checkpoint'] = state['incumbent']['checkpoint']
    slots = total_target + min(len(state['replay']), state['policy']['replay_limit'])
    global_batch = 2 * len(model['settings']['distributed_training']['devices'])
    if new_only:
        config.update(training_population='synthetic_only', sft_epochs=state['policy']['sft_epochs'],
                      required_source_tokens=[r['token'] for r in selected], target_query_token=active['query'],
                      visual_quality_profile=state['policy']['visual_quality_profile'],
                      strict_source_psnr_gt=state['policy']['strict_source_psnr_gt'])
        model['settings']['distributed_training']['synthetic_every'] = 1
        model['settings']['training_recipe'].update(scheduler_dataset_size=total_target,
            scheduler_num_epochs=state['policy']['sft_epochs'])
    else:
        model['settings']['training_recipe']['scheduler_dataset_size'] = max(
            model['settings']['training_recipe']['scheduler_dataset_size'], 2 * slots * global_batch)
    frozen_json(folder / 'student-model.json', model)
    config['training_model_config'] = str(folder / 'student-model.json')
    from .generation_policy import apply as apply_generation_policy
    apply_generation_policy(root, config, folder)
    if global_mode:
        # Two workers preserve the existing durable physical-prefetch receipts.
        config['physical_prefetch_workers']=2
    from driversi.data.datasets.postprocessing import mode
    if mode(config) == 'training_only':
        config['postprocessing_mode'] = 'training_only'
    if (root/'operations-policy.json').exists() and config.get('visual_quality_profile')=='experimental_numeric25_adaptive_v1':
        from .problem_cohort import collect
        pool=collect(root,active,config,exclude_round=folder)
        active['cumulative_unique']=pool['qualified'];active['cohort_per_source']=pool['unique_per_source']
        active['qualified']=pool['qualified']
        frozen_json(folder/'prior-cohort.json',pool)
        config['prior_cohort']=dict(path=str(folder/'prior-cohort.json'),sha256=digest(folder/'prior-cohort.json'))
    frozen_json(folder / 'generation-config.json', config)
    frozen_json(folder / 'round.json', dict(problem=active['id'], parent=state['incumbent'],
                selected_training_tokens=[r['token'] for r in selected],
                replay=[] if new_only else state['replay'][-state['policy']['replay_limit']:],
                target=total_target, target_per_source=target,
                generation_config=str(folder / 'generation-config.json'),
                source_selection='available_within_frozen_visual_pool' if recovery else 'visual_pool',
                pending_source_tokens=active.get('pending_source_tokens',[])))
    assets_ready = all(states[r['token']]=='asset_registered_unadmitted' for r in selected) if new_only else bool(ready)
    active['selected_sources'] = len(selected)
    active.update(status='ready' if assets_ready else 'blocked_assets', current_round=str(folder))
    save(root, state)
    return project(state)


def dispatch(root):
    from driversi.rsi.experiments.reactive_autoloop import run
    state = read_mapping(state_path(root))
    case = next(c for c in state['cases'] if c['id'] == state['active'])
    folder = Path(case['current_round'])
    status_path = folder / 'waves/wave-0001/production-status.json'
    if case['status'] in ('dispatching', 'unknown_execution', 'awaiting_release'):
        raise ValueError('Unsettled current work cannot be dispatched again')
    if status_path.exists():
        return project(state)  # Prior native receipt governs recovery; no blind replay.
    case['status'] = 'dispatching'
    save(root, state)
    try:
        run(folder / 'generation-config.json', 'production')
    except BaseException:
        case['status'] = 'unknown_execution'
        save(root, state)
        raise
    result = read_mapping(status_path)
    case['qualified'] = result.get('qualified_samples', 0)
    case['status'] = ('data_ready' if result['status'] == 'generation_wave_completed'
                      else 'blocked_assets' if result['status'] == 'production_budget_blocked'
                      else 'generation_stopped')
    from .asset_recovery import policy as recovery_policy
    if case['status']=='generation_stopped' and recovery_policy(root):
        failures=[]
        for evidence in (folder/'waves/wave-0001').rglob('readiness.json'):
            row=read_mapping(evidence)
            if 'source_psnr_not_strictly_above_required_value' in row.get('blockers',[]):
                failures.append(str(evidence))
        if failures:
            case.update(status='blocked_assets',source_quality_failures=failures)
    state['generation_started'] = result.get('generation_started', result['status'] != 'production_budget_blocked')
    save(root, state)
    if (Path(root)/'operations-policy.json').exists():
        from .problem_cohort import update
        update(root)
        state=read_mapping(state_path(root))
    return project(state)


def train_current(root):
    from driversi.rsi.experiments.chainflow_campaign import train_pair
    state = read_mapping(state_path(root))
    case = next(c for c in state['cases'] if c['id'] == state['active'])
    if case['status'] != 'data_ready':
        raise ValueError('Native admitted generation must finish before training')
    folder = Path(case['current_round'])
    declaration = read_mapping(folder / 'round.json')
    parent = state['incumbent']
    if declaration['parent'] != parent or digest(parent['checkpoint']) != parent['sha256']:
        raise ValueError('Current incumbent changed before paired training')
    wave = folder / 'waves/wave-0001'
    summary = read_mapping(wave / 'trajectory-family-summary.json')
    training_config=folder/'generation-config.json'
    if (Path(root)/'operations-policy.json').exists():
        from .problem_cohort import training_summary
        from driversi.rsi.decisions.training_recipe import prepare_recipe
        base=read_mapping(training_config)
        summary=training_summary(root,case,base)
        training_config=prepare_recipe(root,case,base,summary)
    samples = summary['sample_reports']
    if state.get('dataset_collection'):
        from driversi.data.datasets.simscale_collection import validate_members
        members = validate_members(Path(state['dataset_collection']).parent)
        available = {str(Path(read_mapping(m['admission_reference'])['sample_report']).resolve())
                     for m in members}
        requested = {str(Path(p).resolve()) for p in samples + declaration['replay']}
        if not requested <= available:
            raise ValueError('Current training samples must be in the admitted dataset collection')
    frozen_json(folder / 'training-input.json', dict(samples=samples, replay=declaration['replay'],
                                                   parent=parent))
    case['status'] = 'training'
    state['training_started'] = True
    save(root, state)
    try:
        ok = train_pair(training_config, samples, wave / 'chainflow-pair',
                        replay_reports=declaration['replay'])
    except BaseException:
        case['status'] = 'unknown_execution'
        save(root, state)
        raise
    trial = read_mapping(wave/'chainflow-pair/training-status.json')
    if not ok and trial['status'] == 'target_still_zero':
        local_path = Path(trial['target_feedback']); local = read_mapping(local_path)
        if local['target_token'] != case['query'] or local['target_success'] is not False:
            raise ValueError('Local target feedback differs from active problem')
        decision = dict(status='target_still_zero',promoted=False,target_success=False,
            target_before=0,target_after=local['pdms']/100,full_benchmark_run=False,
            reason='increase_distinct_data_or_related_sources',target_evidence=str(local_path))
        frozen_json(folder/'local-learning-decision.json',decision)
        from driversi.rsi.decisions.service_memory import ServiceMemory
        memory = ServiceMemory(root/'service-memory.sqlite','evolve')
        try:
            memory.record(dict(program_id=str(folder.relative_to(root)),
                context=dict(model=state['model'],parent=parent['sha256'],protocol='adaptive_single_failure_target'),
                node='learning_review',status='no_benefit',evidence={str(local_path):digest(local_path)},
                source_logs=sorted({r['source_log'] for r in case['candidates']
                    if r['token'] in declaration['selected_training_tokens']}),
                conditions=case['failure_classes'],observed_gain=0,training_executed=True,
                promoted=False,skill_gain_credited=False))
        finally: memory.close()
        apply_verdict(state,case,decision,None,samples)
    else:
        case['status'] = 'awaiting_release' if ok else 'unknown_execution'
    save(root, state)
    return project(state)


def evaluation_from_native(value, expected, side='candidate'):
    if (value.get('model') != 'chainflow_stage1' or value.get('status') != 'completed'
            or value.get('phase') != 'evaluate'
            or value['evaluation_context']['value']['protocol'] != 'chainflow_native_navsim_pdm_v1'):
        raise ValueError('Completed native ChainFlow PDM evaluation required')
    report = dict(model='chainflow_stage1', complete=True, score_unit='0_to_1',
                  native_score_unmodified=True, rows=[dict(token=r['sample_id'],
                  source_log=r['source_log'], metrics=dict(
                      {k: v for k, v in r[side].items() if k != 'score'},
                      PDMS=r[side]['score']))
                  for r in value['paired_samples']])
    scores(report, expected)
    return report


def apply_verdict(state, case, decision, candidate, samples):
    """Only accepted full-evaluation evidence moves the incumbent pointer."""
    case['round'] += 1
    if decision['promoted']:
        state['incumbent'] = candidate
        if case['query'] not in state['retained']:
            state['retained'].append(case['query'])
        state['replay'] = (list(dict.fromkeys(state['replay'] + samples))[-state['policy']['replay_limit']:]
                           if state['policy']['replay_limit'] else [])
        case['status'] = 'solved'
        state['active'] = None
    else:
        case['status'] = ('target_improved_release_rejected' if decision.get('target_success')
                          and state['policy'].get('strict_target') else
                          'needs_revision' if case['round'] < state['policy']['max_rounds'] else
                          'unresolved_needs_strategy_revision' if state['policy'].get('strict_target') else 'budget_exhausted')
        if case['status'] in ('budget_exhausted', 'target_improved_release_rejected','unresolved_needs_strategy_revision'):
            state['active'] = None
    state['rounds'].append(dict(problem=case['id'], round=case['round'], **decision))


def settle(root):
    from driversi.rsi.decisions.continual_learning import decide
    from driversi.rsi.decisions.service_memory import ServiceMemory
    state = read_mapping(state_path(root))
    case = next(c for c in state['cases'] if c['id'] == state['active'])
    if case['status'] != 'awaiting_release':
        raise ValueError('A completed current paired trial is required')
    folder = Path(case['current_round'])
    pair = folder / 'waves/wave-0001/chainflow-pair'
    status = read_mapping(pair / 'training-status.json')
    task = Path(status['directory']).resolve()
    if status['status'] != 'completed' or not task.is_relative_to(pair.resolve()):
        raise ValueError('Completed owned native task required')
    old = state['incumbent']
    binding = read_mapping(folder / 'training-input.json')
    if binding['parent'] != old or digest(old['checkpoint']) != old['sha256']:
        raise ValueError('Parent changed before release')
    feedback_path = task / 'paired_feedback/report.json'
    feedback = read_mapping(feedback_path)
    recovery_path = task / 'treatment_feedback/feedback.json'
    recovery = read_mapping(recovery_path)['cross_process_resume']
    if (feedback.get('kind') != 'chainflow_paired_feedback'
            or feedback.get('status') != 'completed' or feedback.get('effective_recipe_verified') is not True
            or not feedback.get('added_sample_ids') or feedback.get('unconsumed_added_sample_ids')
            or set(feedback['added_sample_ids']) != set(feedback['exposed_added_sample_ids'])
            or not all(recovery.get(k) is True for k in ('different_pids', 'same_compared_state'))):
        raise ValueError('Verified exposure, paired recipe and process recovery required')
    parent = read_mapping(old['evaluation'])
    expected = [r['token'] for r in parent['rows']]
    if len(expected) != 12146:
        raise ValueError('Release requires all12146 official scenes')
    reports, checkpoints = {}, {}
    for arm in ('control', 'treatment'):
        plan = read_mapping(task / (arm + '_train/training-plan.json'))
        model = plan['model_config']
        if isinstance(model, str):
            model = read_mapping(model)
        actual = (project_root() / model['weights']['checkpoint']).resolve()
        if str(actual) != old['checkpoint'] or digest(actual) != old['sha256']:
            raise ValueError('Training arm was not initialized from current incumbent')
        student = read_mapping(task / (arm + '_resume/training-result.json'))
        value = read_mapping(task / (arm + '_evaluation/training-result.json'))
        checkpoint = Path(student['checkpoint']).resolve(strict=True)
        if (not checkpoint.is_relative_to(task) or value['evaluated_checkpoint'] != str(checkpoint)
                or student['checkpoint_files'].get(str(checkpoint)) != digest(checkpoint)
                or student.get('native_loss_unchanged') is not True):
            raise ValueError('Candidate evaluation/weights/native-loss evidence differs')
        reports[arm] = evaluation_from_native(value, expected)
        rerun_parent = evaluation_from_native(value, expected, 'parent')
        if scores(rerun_parent, expected) != scores(parent, expected):
            raise ValueError('Incumbent evaluation parity changed; diagnose protocol drift')
        checkpoints[arm] = checkpoint
    decision = decide(parent, reports['treatment'], reports['control'], expected=expected,
                      target=case['query'], retained=state['retained'],
                      native_paired_benefit=feedback.get('observed_pilot_benefit'),
                       mastery=state['policy']['mastery_pdms'],
                       strict_target=state['policy'].get('strict_target', False))
    if state['policy'].get('strict_target'):
        local = read_mapping(task/'treatment_pre_evaluation/target-feedback.json')
        if (local['target_token'] != case['query']
                or local['checkpoint'] != str(checkpoints['treatment'])
                or local['checkpoint_sha256'] != digest(checkpoints['treatment'])
                or local['target_success'] is not True):
            raise ValueError('Successful local target receipt must precede global release')
        decision.update(full_target_success=decision['target_success'],
                        target_success=local['target_success'],local_target_pdms=local['pdms'])
    checkpoint = checkpoints['treatment']
    reports['treatment']['files'] = {str(checkpoint): digest(checkpoint)}
    reports['treatment']['metrics'] = {'PDMS': decision['candidate_pdms'] / 100}
    reports['treatment'].update(completed=12146, expected=12146)
    evaluation = folder / 'candidate-evaluation.json'
    frozen_json(evaluation, reports['treatment'])
    frozen_json(folder / 'release-decision.json', decision)
    candidate = dict(checkpoint=str(checkpoint), sha256=digest(checkpoint),
                     evaluation=str(evaluation), revision=old['revision']+1,
                     pdms=decision['candidate_pdms'])
    memory = ServiceMemory(root / 'service-memory.sqlite', 'evolve')
    try:
        memory.record(dict(program_id=str(folder.relative_to(root)),
            context=dict(model=state['model'], parent=old['sha256'], protocol='full_navtest_adaptive'),
            node='learning_review', status='pilot_signal' if decision['promoted'] else 'no_benefit',
            evidence={str(p): digest(p) for p in (feedback_path, recovery_path, folder / 'release-decision.json')},
            source_logs=sorted({r['source_log'] for r in case['candidates']}),
            conditions=case['failure_classes'], observed_gain=decision['candidate_pdms']-decision['control_pdms'],
            training_cost=feedback.get('unique_training_cost'), training_executed=True,
            promoted=False, skill_gain_credited=False))
    finally:
        memory.close()
    apply_verdict(state, case, decision, candidate, binding['samples'])
    # Refresh queued tasks from the accepted model: avoid producing for solved transfers.
    if decision['promoted']:
        latest = scores(reports['treatment'], expected)
        for item in state['cases']:
            from driversi.rsi.decisions.problem_learning import target_success
            if item['status'] == 'queued' and target_success(latest[item['query']]['PDMS'],
                    state['policy']['mastery_pdms'], strict=state['policy'].get('strict_target', False)):
                item['status'] = 'resolved_by_transfer'
                state['retained'].append(item['query'])
    save(root, state)
    return project(state)


def defer(root):
    state = read_mapping(state_path(root))
    case = next(c for c in state['cases'] if c['id'] == state['active'])
    if case['status'] not in ('blocked_assets', 'generation_stopped', 'needs_revision', 'budget_exhausted'):
        raise ValueError('Cannot defer running or unsettled work')
    case['status'] = 'deferred'
    state['active'] = None
    save(root, state)
    return project(state)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('initialize', 'advance', 'dispatch', 'train', 'settle', 'defer', 'status', 'auto'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--queue-root', type=Path)
    p.add_argument('--wave-config', type=Path)
    p.add_argument('--policy', type=Path)
    p.add_argument('--max-problems', type=int, default=3)
    p.add_argument('--max-transitions', type=int, default=18)
    p.add_argument('--max-wall-seconds', type=int, default=21600)
    a = p.parse_args()
    with locked(a.output) as root:
        if a.command == 'auto':
            from .continual_execution import run
            value = run(root, max_problems=a.max_problems, max_transitions=a.max_transitions,
                        max_wall_seconds=a.max_wall_seconds)
        elif a.command == 'initialize':
            value = initialize(root, a.queue_root, a.wave_config, a.policy)
        elif a.command == 'advance':
            value = prepare(root)
        elif a.command == 'dispatch':
            value = dispatch(root)
        elif a.command in ('train', 'settle', 'defer'):
            value = {'train': train_current, 'settle': settle, 'defer': defer}[a.command](root)
        else:
            value = project(read_mapping(state_path(root)))
    print(json.dumps(value, ensure_ascii=False))
    if a.command == 'auto' and value['automatic_execution']['status'] not in (
            'finished', 'problem_budget_completed', 'transition_budget_completed', 'stopped_by_request'):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
