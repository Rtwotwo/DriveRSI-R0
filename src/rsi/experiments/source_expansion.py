"""Bounded source breadth escalation from settled generation evidence."""
from pathlib import Path
from collections import Counter

from driversi.configuration import read_mapping
from driversi.agents.backends.native import digest
from .continuation import frozen_json

DEFAULT_POLICY=dict(schema_version=1,levels=[3,5,8,10],minimum_attempts=16,
    maximum_fresh_yield=.10,minimum_duplicate_fraction=.60,minimum_revisions=1,
    source_order='frozen_visual_rank',location_used=False)


def policy(root):
    p=Path(root)/'source-expansion-policy.json'
    if not p.exists():return None
    v=read_mapping(p)
    if v!=DEFAULT_POLICY:raise ValueError('Unregistered source expansion contract')
    return v


def propose(case, state_policy, spent, current_tokens, summary, p, *, asset_failure=False):
    rows=[r for r in summary.get('rows',[]) if not r.get('reused_prior_cohort')]
    counts=Counter(r['status'] for r in rows)
    unknown=set(counts)-{'training_sample_ready','teacher_rejected','trajectory_duplicate_before_render','deferred_for_source_coverage'}
    revisions=case.get('generation_revisions',{}).get(str(case['round']),0)
    rate=sum(r.get('trainable') is True for r in rows)/max(1,len(rows))
    duplicate=counts['trajectory_duplicate_before_render']/max(1,len(rows))
    cap=state_policy['attempts_per_source']*state_policy['max_rounds']
    target=1 if state_policy.get('cohort_mode')=='global_50_min1_v1' else state_policy['targets'][case['round']]
    exhausted=bool(current_tokens) and any(cap-spent.get(t,0)<target for t in current_tokens)
    low_yield = rate<=p['maximum_fresh_yield'] or duplicate>=p['minimum_duplicate_fraction']
    enough_evidence = revisions>=p['minimum_revisions'] and len(rows)>=p['minimum_attempts']
    if not asset_failure and (unknown or not (exhausted or (enough_evidence and low_yield))):
        return None
    previous=case.get('source_search',{}).get('limit',case.get('selected_sources',3))
    next_k=next((k for k in p['levels'] if k>previous),previous if exhausted else None)
    if next_k is None:return None
    excluded=set(case.get('unavailable_source_tokens',[]))
    ranked=sorted(case['candidates'],key=lambda r:(-r['max_similarity'],r['token']))
    eligible=[r for r in ranked if r['token'] not in excluded and cap-spent.get(r['token'],0)>=target]
    if len({r['source_log'] for r in eligible})!=len(eligible):raise ValueError('Distinct source logs required')
    by_token={r['token']:r for r in eligible}
    chosen=[t for t in current_tokens if t in by_token]
    chosen+= [r['token'] for r in eligible if r['token'] not in chosen]
    chosen=chosen[:next_k]
    if not chosen or not set(chosen)-set(current_tokens):return None
    ranks={r['token']:i+1 for i,r in enumerate(ranked)}
    return dict(action='expand_visual_sources',previous_limit=previous,limit=next_k,
        tokens=chosen,original_visual_ranks=[ranks[t] for t in chosen],
        reason='known_asset_failure' if asset_failure else
               'settled_source_attempt_budget_exhausted' if exhausted else 'settled_low_yield_or_duplicate_saturation',
        fresh_attempts=len(rows),fresh_qualified=sum(r.get('trainable') is True for r in rows),
        fresh_yield=rate,duplicate_fraction=duplicate,charged_attempts=spent,
        cumulative_cap_per_source=cap,excluded_unavailable=sorted(excluded),
        quality_gates_unchanged=True,pdms_strict_gt=.95,psnr_strict_gt=25.,
        location_used=False,executor='DriveHarness',per_source_target=target)


def select(case, eligible, target=10):
    by_token={r['token']:r for r in eligible}
    # Native production remains capped at128 samples and SFT at256 updates.
    return [by_token[t] for t in case['source_search']['tokens'] if t in by_token][:128//target]


def commit(case, decision):
    case['source_search']=decision
    if decision['action']=='expand_visual_sources':case.setdefault('source_expansions',[]).append(decision)
    case.update(status='needs_revision',asset_revision=case.get('asset_revision',0)+1)


def after_assets(root):
    from . import continual_curriculum as cc
    from .generation_policy import started_attempts
    root=Path(root);state=read_mapping(cc.state_path(root))
    case=next(c for c in state['cases'] if c['id']==state['active'])
    if case['status']!='assets_unresolved':raise ValueError('Only settled asset failures permit expansion')
    terminal={'source_preparation_failed','source_preparation_incomplete','recovery_exhausted','recovery_failed','source_ready'}
    rows=case.get('asset_recovery',[])
    if not rows or any(r['status'] not in terminal for r in rows):raise ValueError('Asset outcome not settled')
    failed=[r['token'] for r in rows if r['status']!='source_ready']
    case['unavailable_source_tokens']=sorted(set(case.get('unavailable_source_tokens',[])+failed))
    old=Path(case['current_round']);round_info=read_mapping(old/'round.json')
    decision=propose(case,state['policy'],started_attempts(root,case['id']),
        round_info['selected_training_tokens'],{},policy(root),asset_failure=True)
    if not decision:
        from .asset_recovery import ready_sources
        ready=ready_sources(root)
        cap=state['policy']['attempts_per_source']*state['policy']['max_rounds']
        target=1 if state['policy'].get('cohort_mode')=='global_50_min1_v1' else state['policy']['targets'][case['round']]
        spent=started_attempts(root,case['id'])
        usable=[t for t in round_info['selected_training_tokens'] if t in ready and cap-spent.get(t,0)>=target]
        if usable:
            ranks={r['token']:i+1 for i,r in enumerate(sorted(case['candidates'],key=lambda r:(-r['max_similarity'],r['token'])))}
            decision=dict(case['source_search'],action='retain_verified_sources',tokens=usable,
                original_visual_ranks=[ranks[t] for t in usable],
                reason='bounded_pool_exhausted_use_verified_subset',excluded_unavailable=case['unavailable_source_tokens'],
                charged_attempts=spent)
    frozen_json(old/'asset-source-expansion.json',dict(decision=decision,failed_sources=failed))
    if decision:
        commit(case,decision);cc.save(root,state);return cc.prepare(root)
    case['status']='generation_unresolved';state['active']=None;cc.save(root,state)
    return cc.project(state)


def readiness(case, ready):
    """An old ready asset must not repeatedly wake an expanded blocked cohort."""
    search=case.get('source_search')
    if not search:return any(r['token'] in ready for r in case['candidates'])
    tokens=case.get('source_selected_tokens',search['tokens'])
    return bool(tokens) and all(t in ready for t in tokens)
