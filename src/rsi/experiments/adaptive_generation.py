"""Outcome-driven bounded revisions using registered generation skills only."""
from collections import Counter
from pathlib import Path

from driversi.agents.backends.native import digest
from driversi.configuration import read_mapping
from driversi.rsi.experiments.continuation import frozen_json


def diagnose(summary, revision):
    rows=[x for x in summary['rows'] if not x.get('reused_prior_cohort')];counts=Counter(x['status'] for x in rows)
    if any(x in counts for x in ('execution_failed','visual_review_incomplete',
                                'native_language_review_incomplete')):
        action='needs_execution_reconciliation'
    elif any(x in counts for x in ('visual_rejected','visual_preflight_rejected')):
        action='visual_repair_required'
    elif revision>=2:
        action='generation_budget_exhausted'
    elif counts['teacher_rejected']>=max(1,len(rows)*.5):
        action='reduce_history_perturbation'
    elif counts['trajectory_duplicate_before_render']>=2:
        action='adapt_spatial_or_speed_diversity'
    else:
        action='expand_bounded_untried_proposals'
    return dict(action=action,counts=dict(counts),attempted=len(rows),
        qualified=summary['qualified'],rendered=summary['rendered'],
        thresholds_mutable=['diversity','proposal_amplitude'],pdms_strict_gt=.95,
        model_loss_unchanged=True)


def context(root, case):
    """Only already started proposals are charged/excluded; no future queries."""
    used={}
    for plan in (Path(root)/'rounds').glob(case['id']+'-*/waves/wave-0001/trajectory-family-plan.json'):
        for assignment in read_mapping(plan)['assignments']:
            if (plan.parent/'candidates'/assignment['id']/'generation').exists():
                row=assignment.get('history_perturbation')
                if row:used.setdefault(assignment['token'],[]).append(row)
    current=case.get('generation_adaptations',{}).get(str(case['round']),{})
    feedback={}
    for plan in (Path(root)/'rounds').glob(case['id']+'-*/waves/wave-0001/trajectory-family-plan.json'):
        assignments={a['id']:a for a in read_mapping(plan)['assignments']}
        for p in (plan.parent/'candidates').glob('*/outcome.json'):
            token=assignments[p.parent.name]['token'];status=read_mapping(p)['status']
            counts=feedback.setdefault(token,{})
            counts[status]=counts.get(status,0)+1
    return dict(revision=min(2,current.get('revision',0)),action=current.get('action','initial'),
        source_actions=current.get('source_actions',{}),used_proposals=used,source_feedback=feedback)


def source_actions(summary, assignments, revision):
    """One source's offroad failures must not shrink another source's grid."""
    tokens={a['id']:a['token'] for a in assignments};groups={}
    for row in summary['rows']:
        if row.get('reused_prior_cohort'):continue
        token=tokens[row['candidate']]
        groups.setdefault(token,[]).append(row)
    return {token:diagnose(dict(rows=rows,
        qualified=sum(r.get('trainable') is True for r in rows),
        rendered=sum(r.get('rendered') is True for r in rows)),revision)['action']
        for token,rows in groups.items()}


def revise(root):
    from . import continual_curriculum as cc
    from .generation_policy import started_attempts
    root=Path(root);state=read_mapping(cc.state_path(root))
    case=next(x for x in state['cases'] if x['id']==state['active'])
    if case['status']!='generation_stopped':raise ValueError('Settle native generation before adaptation')
    old=Path(case['current_round']);wave=old/'waves/wave-0001'
    summary_path=wave/'trajectory-family-summary.json';summary=read_mapping(summary_path)
    if summary.get('status') not in ('completed','completed_with_shortages'):
        raise ValueError('Only terminal family evidence supports revision')
    from .processing_resume import settled_generation
    settlement=settled_generation(wave,allow_prior_completed=True)
    if read_mapping(wave/'physical-prefetch-status.json').get('errors'):
        raise ValueError('Resolve every prefetch error before automatic policy revision')
    frozen_json(old/'adaptive-settlement.json',settlement)
    revisions=case.get('generation_revisions',{}).get(str(case['round']),0)
    spent=started_attempts(root,case['id'])
    cap=state['policy']['attempts_per_source']*state['policy']['max_rounds']
    limit=2
    if (root/'operations-policy.json').exists():
        limit=read_mapping(root/'operations-policy.json').get('max_generation_revisions',2)
        if type(limit) is not int or not 2<=limit<=6:raise ValueError('Bounded revision policy required')
    minimum=1 if state['policy'].get('cohort_mode')=='global_50_min1_v1' else 10
    bounded_revision=1 if revisions<limit and any(cap-n>=minimum for n in spent.values()) else 2
    decision=diagnose(summary,bounded_revision)
    decision['source_actions']=source_actions(summary,read_mapping(wave/'trajectory-family-plan.json')['assignments'],bounded_revision)
    clocks=[read_mapping(p) for p in (wave/'candidates').glob('*/physical-prefetch.json')]
    physical_seconds=sum(max(0.,float(v['finished'])-float(v['started'])) for v in clocks if v.get('finished') is not None)
    decision.update(physical_worker_seconds=physical_seconds,started_physical_tasks=len(clocks),charged_attempts=spent,cumulative_cap_per_source=cap,
        trigger_evidence={str(summary_path):digest(summary_path)},
        executor='DriveHarness',external_code_edit_required=False,
        previous_round=str(old),revision=revisions+1)
    from .source_expansion import policy as expansion_policy, propose, commit
    expansion=None
    if expansion_policy(root) and decision['action'] not in ('needs_execution_reconciliation','visual_repair_required'):
        expansion=propose(case,state['policy'],spent,
            read_mapping(old/'round.json')['selected_training_tokens'],summary,expansion_policy(root))
    if expansion:
        decision['source_expansion']=expansion
        # Source breadth is a scheduler action, not a new physical generation skill.
        if decision['action']=='generation_budget_exhausted':decision['action']='expand_bounded_untried_proposals'
    frozen_json(old/'adaptive-next-action.json',decision)
    if decision['action'] in ('needs_execution_reconciliation','visual_repair_required','generation_budget_exhausted'):
        case.update(status=decision['action'],generation_recovery_evidence=str(old/'adaptive-next-action.json'))
        state['active']=None;cc.save(root,state);return cc.project(state)
    case.setdefault('generation_revisions',{})[str(case['round'])]=revisions+1
    case.setdefault('generation_adaptations',{})[str(case['round'])]=dict(
        revision=revisions+1,action=decision['action'],source_actions=decision['source_actions'])
    case.update(status='needs_revision',asset_revision=case.get('asset_revision',0)+1,
        generation_recovery_evidence=str(old/'adaptive-next-action.json'))
    if expansion:
        case['source_search']=expansion
        case.setdefault('source_expansions',[]).append(expansion)
    cc.save(root,state)
    value=cc.prepare(root)
    after=read_mapping(cc.state_path(root));changed=next(x for x in after['cases'] if x['id']==case['id'])
    frozen_json(old/'adaptive-action-result.json',dict(action=decision['action'],
        next_round=changed.get('current_round'),status=changed['status'],
        wall_seconds_known=summary.get('elapsed_seconds'),previous_qualified=summary['qualified'],
        prior_samples_relabelled=False,executor='DriveHarness'))
    return value
