"""Reconcile terminal carried-cohort accounting errors without GPU replay."""
from pathlib import Path
from collections import Counter

from driversi.agents.backends.native import digest
from driversi.configuration import read_mapping
from driversi.runtime.budget import write_json
from .continuation import frozen_json

ERROR = 'ValueError: Family sample total differs from actual unique outcomes'


def eligible(root, case):
    if not case or case.get('status') != 'unknown_execution' or not case.get('current_round'):
        return False
    wave=Path(case['current_round'])/'waves/wave-0001'
    failure=wave/'production-failure.txt'
    return failure.exists() and failure.read_text().strip().splitlines()[-1] == ERROR


def rebuild(config, plan, summary, outcomes):
    from .problem_cohort import seed
    from .reactive_cohort import family_training_gate
    prior=seed(config,plan)
    rows=prior+outcomes
    assignments={a['id']:a for a in plan['assignments']}
    for row in outcomes:
        a=assignments[row['candidate']]
        if row['anchor_assignment_id']!=a['anchor_assignment_id']:
            raise ValueError('Outcome source differs from plan')
    qualified=[r for r in rows if r.get('trainable')]
    if len({r['dataset_reference']['package_id'] for r in qualified})!=len(qualified):
        raise ValueError('Repeated package in carried cohort')
    if len(qualified)!=summary['qualified'] or len(outcomes)!=summary['attempted']:
        raise ValueError('Terminal accounting witnesses disagree')
    fixed=dict(summary,rows=rows,reused_prior_cohort=len(prior),
        counts=dict(Counter(r['status'] for r in outcomes)),
        sample_reports=[r['sample_report'] for r in qualified])
    anchors=list(dict.fromkeys(a['anchor_assignment_id'] for a in plan['assignments']))
    gate=family_training_gate(fixed,anchors,config['trajectory_family']['target_per_anchor'])
    return fixed,gate


def recover(root):
    from . import continual_curriculum as cc
    from .processing_resume import settled_generation
    from .problem_cohort import update
    root=Path(root);state=read_mapping(cc.state_path(root))
    case=next(c for c in state['cases'] if c['id']==state['active'])
    if not eligible(root,case):raise ValueError('Not the registered cohort accounting failure')
    folder=Path(case['current_round']);wave=folder/'waves/wave-0001'
    if (folder/'training-input.json').exists():raise ValueError('Training has a separate recovery protocol')
    settlement=settled_generation(wave,allow_prior_completed=True)
    if read_mapping(wave/'physical-prefetch-status.json').get('errors'):
        raise ValueError('Physical work must be terminal without unresolved errors')
    if any(not (p/'completion.json').exists() for p in (wave/'resident-batches').glob('*') if (p/'execution').exists()):
        raise ValueError('Unsettled postprocessing')
    cfg=read_mapping(folder/'generation-config.json');plan=read_mapping(wave/'trajectory-family-plan.json')
    summary=read_mapping(wave/'trajectory-family-summary.json')
    if summary.get('status') not in ('completed','completed_with_shortages'):
        raise ValueError('Only terminal family summaries are recoverable')
    outcomes=[read_mapping(p) for p in sorted((wave/'candidates').glob('*/outcome.json'))]
    fixed,gate=rebuild(cfg,plan,summary,outcomes)
    evidence=folder/'cohort-accounting-recovery';evidence.mkdir(exist_ok=True)
    # Originals survive a retry after any derived-state write below.
    if not (evidence/'summary-before.json').exists():
        frozen_json(evidence/'summary-before.json',summary)
        frozen_json(evidence/'production-before.json',read_mapping(wave/'production-status.json'))
        frozen_json(evidence/'state-before.json',state)
        frozen_json(evidence/'settlement.json',settlement)
    write_json(wave/'trajectory-family-summary.json',fixed)
    write_json(wave/'production-outcomes.json',fixed)
    frozen_json(evidence/'result.json',dict(rule='carried_cohort_accounting_v1',
        executor='DriveHarness',qualified=fixed['qualified'],fresh_outcomes=len(outcomes),
        reused_prior_cohort=fixed['reused_prior_cohort'],gate=gate,
        known_physical_jobs=settlement['known_results'],physical_reexecution=False,
        render_reexecution=False,failure_sha256=digest(wave/'production-failure.txt')))
    production=read_mapping(evidence/'production-before.json')
    production.update(status='generation_wave_completed' if gate['passed'] else 'generation_family_shortage',
        qualified_samples=fixed['qualified'],accounting_recovery=str(evidence/'result.json'))
    production.pop('error_type',None)
    write_json(wave/'production-status.json',production)
    case.update(status='data_ready' if gate['passed'] else 'generation_stopped',
        qualified=fixed['qualified'],accounting_recovery=str(evidence/'result.json'))
    cc.save(root,state);update(root)
    return cc.project(read_mapping(cc.state_path(root)))
