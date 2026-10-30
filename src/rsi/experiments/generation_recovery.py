"""Escalate a known duplicate-limited family to one declared alternative skill."""
from pathlib import Path

from driversi.agents.backends.native import digest
from driversi.configuration import read_mapping
from driversi.rsi.experiments.continuation import frozen_json


def remaining_budget(budget, summary, proposal_count):
    """Alternative generation spends the remainder of the original allocation."""
    import copy
    value=copy.deepcopy(budget)
    remaining=[budget['attempts_per_anchor']-a['attempted'] for a in summary['anchors']]
    attempts=min([proposal_count,*remaining])
    spent_attempts=summary['attempted'];spent_renders=summary['rendered']
    render_caps=[]
    for anchor in summary['anchors']:
        rendered=sum(bool(row.get('rendered')) for row in summary['rows']
                     if row['anchor_assignment_id']==anchor['anchor'])
        render_caps.append(budget['render_cap_per_anchor']-rendered)
    render_cap=min(render_caps)
    if attempts<budget['target_per_anchor'] or render_cap<budget['target_per_anchor']:
        raise ValueError('Remaining per-source generation budget cannot cover quota')
    value.update(attempts_per_anchor=attempts,render_cap_per_anchor=render_cap,
        global_attempt_cap=budget['global_attempt_cap']-spent_attempts,
        global_render_cap=budget['global_render_cap']-spent_renders)
    if value['wave_size']*attempts>value['global_attempt_cap'] or value['wave_size']*render_cap>value['global_render_cap']:
        # Conservatively reserve an equal allowance for every selected source.
        value['render_cap_per_anchor']=value['global_render_cap']//value['wave_size']
        value['attempts_per_anchor']=min(attempts,value['global_attempt_cap']//value['wave_size'])
    if min(value['attempts_per_anchor'],value['render_cap_per_anchor'])<budget['target_per_anchor']:
        raise ValueError('Remaining global generation budget cannot cover quota')
    for row in value['states']:row['attempt_cap']=value['attempts_per_anchor']
    return value


def decision(summary, config, revisions):
    if any(x.get('status') in ('execution_failed','visual_review_incomplete',
                              'native_language_review_incomplete') for x in summary['rows']):
        return 'needs_execution_reconciliation'
    if revisions>=1:return 'generation_budget_exhausted'
    if config['trajectory_family']['method']!='native_policy_sample_family_v1':
        return 'generation_budget_exhausted'
    if any(x.get('search_stop_reason')=='consecutive_non_novel_native_rollouts'
           for x in summary['anchors']):
        return 'bounded_history_proposals'
    return 'generation_unresolved'


def revise(root):
    from . import continual_curriculum as cc
    root=Path(root)
    recovery=read_mapping(root/'generation-recovery-policy.json')
    if recovery.get('mode')=='adaptive_source_v1':
        from .adaptive_generation import revise as adaptive_revise
        return adaptive_revise(root)
    state=read_mapping(cc.state_path(root))
    case=next(x for x in state['cases'] if x['id']==state['active'])
    if case['status']!='generation_stopped':raise ValueError('Only settled generation may be revised')
    old=Path(case['current_round']);config=read_mapping(old/'generation-config.json')
    summary_path=old/'waves/wave-0001/trajectory-family-summary.json'
    policy_path=root/'generation-recovery-policy.json';policy=read_mapping(policy_path)
    if policy.get('max_revisions_per_learning_round')!=1:raise ValueError('One finite alternative required')
    summary=read_mapping(summary_path)
    revision=case.get('generation_revisions',{}).get(str(case['round']),0)
    action=decision(summary,config,revision)
    if action!='bounded_history_proposals':
        case.update(status=action,generation_recovery_evidence=str(summary_path))
        state['active']=None;cc.save(root,state);return cc.project(state)
    template=read_mapping(policy['fallback_config'])
    if template['generation_skill']!='perturbed_history_tts':raise ValueError('Unregistered fallback skill')
    folder=old.with_name(old.name+'-history-proposals-v2');folder.mkdir(exist_ok=True)
    family=template['trajectory_family']
    parent_budget=Path(config['global_budget'])
    try:
        budget=remaining_budget(read_mapping(parent_budget),summary,len(family['proposal_grid']))
    except ValueError as exc:
        case.update(status='generation_budget_exhausted',generation_recovery_reason=str(exc),
                    generation_recovery_evidence=str(summary_path))
        state['active']=None;cc.save(root,state);return cc.project(state)
    frozen_json(folder/'global-budget.json',budget)
    config['global_budget']=str(folder/'global-budget.json')
    config['global_budget_sha256']=digest(folder/'global-budget.json')
    config['max_rendered_branches']=min(config['max_rendered_branches'],
        budget['wave_size']*budget['render_cap_per_anchor'])
    config['generation_skill']='perturbed_history_tts'
    config['trajectory_family'].update(method=family['method'],
        proposal_grid=family['proposal_grid'][:budget['attempts_per_anchor']],
        attempts_per_anchor=budget['attempts_per_anchor'],seed_start=family['seed_start'],
        render_cap_per_anchor=budget['render_cap_per_anchor'])
    config['output']=str(folder/'waves/wave-0001')
    from driversi.rsi.decisions.augmentation_budget import runtime_gate
    if not runtime_gate(config)['allowed']:raise ValueError('Alternative generation reservation rejected')
    # Native model/loss, selected sources, all admission gates and per-source
    # quota are inherited unchanged. Earlier samples remain in their own family.
    student=read_mapping(config['training_model_config'])
    frozen_json(folder/'student-model.json',student)
    config['training_model_config']=str(folder/'student-model.json')
    frozen_json(folder/'generation-config.json',config)
    declaration=read_mapping(old/'round.json')
    declaration.update(generation_config=str(folder/'generation-config.json'))
    frozen_json(folder/'round.json',declaration)
    frozen_json(folder/'generation-recovery.json',dict(
        previous_round=str(old),reason='consecutive_non_novel_native_rollouts',
        generator='bounded_history_then_native_driverl_correction',pure_driverl=False,
        evidence={str(p):digest(p) for p in (summary_path,policy_path,Path(policy['fallback_config']),parent_budget)},
        original_budget_debited=True,prior_attempts=summary['attempted'],prior_renders=summary['rendered'],
        quota_and_admission_unchanged=True,previous_samples_auto_imported=False))
    case.setdefault('generation_revisions',{})[str(case['round'])]=revision+1
    case.update(status='ready',current_round=str(folder))
    cc.save(root,state);return cc.project(state)
