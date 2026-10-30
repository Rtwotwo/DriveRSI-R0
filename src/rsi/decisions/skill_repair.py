"""Targeted skill changes, factorial attribution and executable regression memory."""
import copy
import statistics

from . import grounded_skills as grounded
from . import skill_doctor
from .skill_evolution import evaluate, identity
from .frozen_skills import SAFETY, choose_skill

PROTOCOL='targeted_ablation_v1'
ARMS=('native','incumbent','text_only','gate_only','combined')
VARIANTS=ARMS[2:]
NODE='following_speed_check'
PROMPT=grounded.PROMPT.replace('eligibility. Diagnosis','eligibility, modification_contract. Diagnosis')+'''
The fixed problem family is GO STRAIGHT with a failed measured TTC metric, routed to
following_speed_check. This is not proof that the cause is following too closely.
Produce BOTH a text revision and an eligibility revision. They are evaluated separately
and together; do not change other skills. The prior incumbent remains the comparator.
Add exactly one field modification_contract to the JSON. It contains target_metric="TTC",
target_cases (one to four supplied family case IDs), and gate_changes (one to four objects
with exactly case_id and expected_enabled boolean). Each claimed gate change must differ
from the incumbent's actual activation on that supplied current state. A narrower upper
speed bound cannot repair a failed activation whose speed remains inside both ranges.
Use observed failures and protected regressions, not arbitrary numeric thresholds. A text
repair must improve TTC on the declared target family without reintroducing remembered
safety regressions. Structural validity is not driving success. Conditions cannot name
scene tokens. Do not claim a referenced metric proves a causal diagnosis.
Keep the complete reply concise: about one hundred words of skill text and short
rationale, hypothesis and uncertainty, leaving room for the structured contract.
'''


def enabled(state,preset,node=NODE):
    skill=choose_skill(state,preset)
    if skill['id']!=node:return None
    return grounded.apply_gate(skill,preset)['enabled']


def variants(incumbent,candidate):
    out={}
    for name in VARIANTS:
        p=copy.deepcopy(incumbent)
        if name in ('text_only','combined'):p['skills'][NODE]=candidate['skills'][NODE]
        if name in ('gate_only','combined'):
            p.setdefault('eligibility',{})[NODE]=copy.deepcopy(candidate['eligibility'][NODE])
        p['version']='repair-'+name+'-'+identity({'skills':p['skills'],'eligibility':p.get('eligibility',{})})[:12]
        out[name]=p
    return out


def validate_contract(value,incumbent,candidate,cases,target_cases):
    if not isinstance(value,dict) or set(value)!={'target_metric','target_cases','gate_changes'}:
        raise ValueError('Modification contract schema mismatch')
    target=value['target_cases']
    if value['target_metric']!='TTC' or not isinstance(target,list) or not 1<=len(target)<=4:
        raise ValueError('Declare bounded TTC target cases')
    if len(set(target))!=len(target) or any(x not in target_cases for x in target):
        raise ValueError('Targets must be measured family development cases')
    changes=value['gate_changes']
    if not isinstance(changes,list) or not 1<=len(changes)<=4:raise ValueError('Declare gate effects')
    checks=[];seen=set()
    for x in changes:
        if not isinstance(x,dict) or set(x)!={'case_id','expected_enabled'} or type(x['expected_enabled']) is not bool:
            raise ValueError('Gate effect schema mismatch')
        case=x['case_id']
        if case not in cases or case in seen:raise ValueError('Unknown/duplicate gate effect case')
        seen.add(case);state=cases[case]['observed_state']
        before,after=enabled(state,incumbent),enabled(state,candidate)
        checks.append(dict(case_id=case,before=before,after=after,expected=x['expected_enabled'],
                           passed=before is not None and before!=after and after==x['expected_enabled']))
    return dict(text_changed=candidate['skills'][NODE]!=incumbent['skills'][NODE],
                gate_claims_passed=all(c['passed'] for c in checks),gate_checks=checks,
                target_cases=target,target_metric='TTC',causal_validity_verified=False)


def propose(development,state,output,config,credentials,target_cases,regressions,prior_events=()):
    group=development[NODE]
    cases={c['case_id']:c for c in group['difficult_observations']}
    # Target/regression coverage is assembled by the experiment, never inferred from captions.
    available=[x for x in target_cases if x in cases]
    if not available:raise ValueError('No supplied target evidence')
    history=[]
    for e in [*prior_events,*state['events']][-3:]:
        full=e.get('proposed_preset',e['candidate'])
        old=e.get('development',{}).get(NODE,{})
        history.append(dict(proposed_text=full['skills'][NODE],
            proposed_eligibility=full.get('eligibility',{}).get(NODE),
            variant_definitions=e.get('variant_definitions'),selected_variant=e.get('selected_variant'),
            diagnostic_variant=e.get('diagnostic_variant'),diagnosis=e.get('diagnosis'),
            development_mean_metrics=old.get('mean_metrics'),
            regression_observations=[{k:c[k] for k in ('case_id','metrics','comparisons','observed_state')}
                for c in old.get('difficult_observations',[]) if c.get('safety_regression')],
            modification_check=e.get('modification_check'),variant_feedback=e.get('variant_feedback'),
            accepted=e['decision']['accepted'],reasons=e['decision']['reasons']))
    facts=dict(selected_skill_id=NODE,allowed_commands=list(grounded.COMMANDS),
        incumbent_text=state['active']['skills'][NODE],incumbent_eligibility=state['active'].get('eligibility',{}).get(NODE),
        development=group,target_family_cases=available,regression_obligations=regressions,
        persisted_experience=history,validation_examples_available=False,caption_assisted_diagnosis=True)
    job=dict(prompt=PROMPT,evidence=[],response_max_tokens=1600,
        diagnosis_evidence_ids=[e['id'] for e in group['evidence']],facts=facts)
    saved=skill_doctor.cached_request(job,output,config,credentials)
    raw=None;check={};error=None;candidate=state['active']
    try:
        import json
        if saved['finish_reason']!='stop':raise ValueError('Incomplete repair response')
        raw=json.loads(saved['content']);contract=raw['modification_contract']
        candidate=grounded.validate({k:v for k,v in raw.items() if k!='modification_contract'},NODE,state['active'],group['evidence'])
        check=validate_contract(contract,state['active'],candidate,cases,available)
    except (ValueError,KeyError,TypeError) as exc:
        candidate=state['active'];error=type(exc).__name__+': '+str(exc)
    # Invalid proposals still have a complete finite comparison; no fabricated improvement.
    if error:
        presets={k:copy.deepcopy(state['active']) for k in VARIANTS}
    else:presets=variants(state['active'],candidate)
    return dict(skill_id=NODE,incumbent=state['active'],candidate=candidate,variants=presets,
        valid=error is None,proposal=raw,proposal_error=error,modification_check=check,
        experience_count=len(history),memory_revision=state['revision'],memory_hash=identity(state),
        request_job_hash=identity(job),usage=saved['usage'],doctor_seconds=saved['seconds'])


def mapped(rows,arm,tokens=None):
    return [{**r,'arm':'candidate' if r['arm']==arm else r['arm']} for r in rows
            if r['arm'] in ('native','incumbent',arm) and (tokens is None or r['token'] in tokens)]


def regression_checks(rows,arm,obligations):
    index={(r['token'],r['seed'],r['arm']):r for r in rows};checks=[]
    for obligation in obligations:
        t,s=obligation['token'],obligation['seed']
        a=index[t,s,'incumbent']['metrics'];b=index[t,s,arm]['metrics']
        bad=[m for m in SAFETY if b[m]<a[m]-1e-9]
        if a['PDMS']>0 and b['PDMS']==0:bad.append('new_zero')
        checks.append(dict(token=t,seed=s,passed=not bad,failed=bad,origin=obligation['origin']))
    return dict(passed=all(x['passed'] for x in checks),checks=checks)


def decide(rows,members,seeds,proposal,obligations):
    """Choose using development ONLY, then apply the unchanged cross-log gates."""
    expected={(m['token'],s,a) for m in members for s in seeds for a in ARMS}
    keys=[(r['token'],r['seed'],r['arm']) for r in rows]
    if len(keys)!=len(expected) or set(keys)!=expected:raise ValueError('Incomplete factorial rows')
    dev={m['token'] for m in members if m['role']=='development'}
    targets={m['token'] for m in members if m['role']=='development' and m.get('repair_family')}
    if not targets:raise ValueError('No fixed development TTC family')
    index={(r['token'],r['seed'],r['arm']):r for r in rows};results={}
    contract_targets=set(proposal.get('modification_check',{}).get('target_cases',[]))
    named={m['token'] for m in members if m.get('case_id') in contract_targets and m['role']=='development'}
    for arm in VARIANTS:
        evaluation=evaluate(mapped(rows,arm),members,seeds)
        regressions=regression_checks(rows,arm,obligations)
        gain=statistics.mean(index[t,s,arm]['metrics']['TTC']-index[t,s,'incumbent']['metrics']['TTC'] for t in targets for s in seeds)
        named_gain=statistics.mean(index[t,s,arm]['metrics']['TTC']-index[t,s,'incumbent']['metrics']['TTC'] for t in named for s in seeds) if named else 0
        action_changed=any(index[t,s,arm]['raw_trajectory']!=index[t,s,'incumbent']['raw_trajectory'] for t in named for s in seeds)
        reasons=[x for x in evaluation['reasons'] if x.startswith('development:')]
        if not proposal['valid']:reasons.append('invalid_modification_contract')
        if arm in ('text_only','combined') and not proposal.get('modification_check',{}).get('text_changed'):reasons.append('text_unchanged')
        if arm in ('gate_only','combined') and not proposal.get('modification_check',{}).get('gate_claims_passed'):reasons.append('gate_claim_not_realized')
        if gain<=1e-9 or named_gain<=1e-9 or not action_changed:reasons.append('declared_TTC_repair_not_observed')
        if not regressions['passed']:reasons.append('remembered_regression_failed')
        results[arm]=dict(development_eligible=not reasons,development_reasons=reasons,
            target_ttc_delta=gain,declared_target_ttc_delta=named_gain,raw_action_changed=action_changed,
            regression=regressions,full_decision=evaluation,
            development_pdms_delta=evaluation['cohorts']['development']['comparisons']['incumbent']['mean_delta'])
    ranked=sorted(VARIANTS,key=lambda a:(-results[a]['target_ttc_delta'],-results[a]['development_pdms_delta'],VARIANTS.index(a)))
    selected=next((a for a in ranked if results[a]['development_eligible']),None)
    diagnostic=selected or ranked[0]
    final=copy.deepcopy(results[diagnostic]['full_decision'])
    if selected is None:
        final['accepted']=False;final['reasons'].append('no_development_eligible_variant')
    # Check family transfer and broad retention independently; no validation-based reselection.
    subsets={}
    for group in ('family','control'):
        ms=[m for m in members if m['role']=='development' or m.get('validation_group')==group]
        if not any(m['role']=='validation' for m in ms):raise ValueError('Missing validation subgroup')
        ds=evaluate(mapped(rows,diagnostic,{m['token'] for m in ms}),ms,seeds)
        subsets[group]=ds
        subset_reasons=[x for x in ds['reasons'] if not (group=='control' and x=='validation:no_positive_transfer_over_incumbent')]
        if subset_reasons:
            final['accepted']=False;final['reasons']+=['subgroup_'+group+':'+x for x in subset_reasons]
    vt={m['token'] for m in members if m.get('validation_group')=='family'}
    transfer=statistics.mean(index[t,s,diagnostic]['metrics']['TTC']-index[t,s,'incumbent']['metrics']['TTC'] for t in vt for s in seeds)
    if transfer<=1e-9:final['accepted']=False;final['reasons'].append('no_cross_log_TTC_transfer')
    final['reasons']=list(dict.fromkeys(final['reasons']))
    return dict(selected_variant=selected,diagnostic_variant=diagnostic,variants=results,decision=final,
        validation_subgroups=subsets,validation_target_ttc_delta=transfer,selection_uses_validation=False)
