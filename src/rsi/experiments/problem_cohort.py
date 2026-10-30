"""Cross-revision, source-specific admitted cohorts with mutual novelty evidence."""
from collections import Counter
from pathlib import Path

from driversi.agents.backends.native import digest, project_root
from driversi.configuration import read_mapping
from driversi.runtime.budget import write_json
from .continuation import frozen_json


def failed_training_history(root, case):
    """Only completed, explicit target-zero trials establish prior consumption."""
    consumed=set();evidence={}
    for folder in sorted((Path(root)/'rounds').glob(case['id']+'-*')):
        decision=folder/'local-learning-decision.json'
        if not decision.exists():continue
        result=read_mapping(decision)
        if result.get('status')!='target_still_zero' or result.get('target_success') is not False:
            continue
        path=folder/'training-input.json'
        consumed.update(read_mapping(path)['samples'])
        evidence.update({str(path):digest(path),str(decision):digest(decision)})
    return consumed,evidence


def training_selection(tokens, rows, consumed=()):
    from driversi.rsi.decisions.global_cohort import select
    consumed=set(consumed)
    ordered=[r for r in rows if r['sample_report'] not in consumed]
    ordered += [r for r in rows if r['sample_report'] in consumed]
    gate=select(tokens,ordered,total=50)
    fresh=sum(r['sample_report'] not in consumed for r in gate['selected_rows'])
    minimum=10 if consumed else 0
    gate.update(minimum_fresh_samples=minimum,fresh_selected=fresh,
        fresh_shortage=max(0,minimum-fresh),passed=gate['passed'] and fresh>=minimum)
    return gate


def collect(root, case, config, *, exclude_round=None):
    from driversi.rsi.decisions.adaptive_diversity import compare, PROFILE
    tokens=config.get('required_source_tokens',[])
    kept={t:[] for t in tokens};rows=[];rejected=[];seen={};evidence={};raw=Counter()
    collection=read_mapping(project_root()/'dataset/collection.json')
    index=read_mapping(project_root()/'dataset/manifests'/collection['split']/'index.json')
    members=set(index['samples'])
    for folder in sorted((Path(root)/'rounds').glob(case['id']+'-*')):
        if exclude_round and folder.resolve()==Path(exclude_round).resolve():continue
        cfg=folder/'generation-config.json';wave=folder/'waves/wave-0001'
        plan=wave/'trajectory-family-plan.json'
        if not cfg.exists() or not plan.exists():continue
        if read_mapping(cfg).get('visual_quality_profile')!=config.get('visual_quality_profile'):continue
        assignments={a['id']:a for a in read_mapping(plan)['assignments']}
        for path in sorted((wave/'candidates').glob('*/outcome.json')):
            a=assignments[path.parent.name];token=a['token'];out=read_mapping(path)
            if token not in kept or not out.get('trainable'):continue
            package=out.get('dataset_reference',{}).get('package_id')
            if package in seen:
                if config['trajectory_family'].get('cohort_mode')=='global_50_min1_v1' and seen[package]!=token:
                    raise ValueError('Package has ambiguous source identity')
                continue
            if package not in members:raise ValueError('Admitted branch absent from canonical collection')
            if read_mapping(path.with_name('outcome-seal.json'))['sha256']!=digest(path):
                raise ValueError('Prior cohort outcome changed')
            for p,sha in {**out.get('files',{}),**out.get('trajectory_files',{})}.items():
                if digest(p)!=sha:raise ValueError('Cohort evidence changed')
            seen[package]=token;raw[token]+=1;desc=out['descriptor']
            if config.get('visual_quality_profile')!=PROFILE:continue
            decision=compare(desc,kept[token],desc['adaptive_diversity_policy'])
            evidence[str(path)]=digest(path)
            if not decision['distinct']:
                rejected.append(dict(package_id=package,source_token=token,reason='cross_revision_duplicate'));continue
            kept[token].append(desc)
            rows.append(dict(out,source_token=token,original_outcome=str(path),
                original_outcome_sha256=digest(path),cross_revision_diversity=decision,
                origin_plan=str(plan),origin_plan_sha256=digest(plan),origin_assignment=a))
    target=config['trajectory_family']['target_per_anchor']
    counts={t:len(v) for t,v in kept.items()}
    global_mode=config['trajectory_family'].get('cohort_mode')=='global_50_min1_v1'
    consumed=set()
    if global_mode:
        consumed,training_evidence=failed_training_history(root,case)
        evidence.update(training_evidence)
        gate=training_selection(tokens,rows,consumed)
    return dict(schema_version=1,problem=case['id'],target_per_source=1 if global_mode else target,
        raw_admitted_per_source=dict(raw),unique_per_source=counts,
        ready=gate['passed'] if global_mode else bool(tokens) and all(n>=target for n in counts.values()),
        rows=rows,rejected=rejected,evidence=evidence,
        qualified=len(rows),sample_reports=[r['sample_report'] for r in rows],
        previously_consumed_reports=sorted(consumed),minimum_fresh_samples=10 if consumed else 0,
        fresh_samples=sum(r['sample_report'] not in consumed for r in rows),
        visual_quality_profile=config.get('visual_quality_profile'),native_gates_unchanged=True)


def update(root):
    from . import continual_curriculum as cc
    root=Path(root);state=read_mapping(cc.state_path(root))
    case=next((c for c in state['cases'] if c['id']==state['active']),None)
    if not case or not case.get('current_round'):return None
    folder=Path(case['current_round']);cfg=read_mapping(folder/'generation-config.json')
    if cfg.get('visual_quality_profile')!='experimental_numeric25_adaptive_v1':return None
    value=collect(root,case,cfg)
    write_json(folder/'cumulative-cohort.json',value)
    case['cumulative_unique']=value['qualified'];case['cohort_per_source']=value['unique_per_source']
    if case['status'] in ('generation_stopped','data_ready'):
        case.update(status='data_ready' if value['ready'] else 'generation_stopped',
                    qualified=value['qualified'],cohort=str(folder/'cumulative-cohort.json'))
    cc.save(root,state)
    return value


def seed(config, plan):
    reference=config.get('prior_cohort')
    if not reference:return []
    if digest(reference['path'])!=reference['sha256']:raise ValueError('Prior cohort changed')
    pool=read_mapping(reference['path'])
    anchors={a['token']:a['anchor_assignment_id'] for a in plan['assignments']}
    result=[];counts=Counter();target=config['trajectory_family']['target_per_anchor']
    rows=pool['rows']
    if config['trajectory_family'].get('cohort_mode')=='global_50_min1_v1':
        available=[r for r in rows if r['source_token'] in anchors]
        covered={r['source_token'] for r in available};limit=50-len(set(anchors)-covered)
        consumed=set(pool.get('previously_consumed_reports',[]))
        if consumed:
            # A next learning trial cannot consist entirely of the previous50.
            # Never-consumed carryover counts as fresh; keep room for the rest.
            fresh=[r for r in available if r['sample_report'] not in consumed]
            available=fresh+[r for r in available if r['sample_report'] in consumed]
            limit=min(limit,40+min(10,len(fresh)))
        first=[];seen=set()
        for row in available:
            if row['source_token'] not in seen:first.append(row);seen.add(row['source_token'])
        rows=(first+[r for r in available if r not in first])[:limit]
    for row in rows:
        if row['source_token'] not in anchors:continue
        if counts[row['source_token']]>=target:continue
        if digest(row['original_outcome'])!=row['original_outcome_sha256']:raise ValueError('Cohort source changed')
        result.append(dict(row,anchor_assignment_id=anchors[row['source_token']],
            reused_prior_cohort=True,rendered=True))
        counts[row['source_token']]+=1
    return result


def training_summary(root, case, config):
    pool=collect(root,case,config)
    if not pool['ready']:raise ValueError('Cumulative per-source quota incomplete')
    folder=Path(case['current_round']);wave=folder/'waves/wave-0001'
    plan=read_mapping(wave/'trajectory-family-plan.json')
    anchors={a['token']:a['anchor_assignment_id'] for a in plan['assignments']}
    # Use exactly the selected per-source quota, never an accidental excess pool.
    selected=[]
    if config['trajectory_family'].get('cohort_mode')=='global_50_min1_v1':
        selection=training_selection(config['required_source_tokens'],pool['rows'],pool['previously_consumed_reports'])
        if not selection['passed']:raise ValueError('Global quota, source coverage or fresh-data quota incomplete')
        selected=[dict(r,anchor_assignment_id=anchors[r['source_token']],rendered=True)
                  for r in selection['selected_rows']]
    else:
        for token in config['required_source_tokens']:
            selected.extend([dict(r,anchor_assignment_id=anchors[token],rendered=True)
                for r in pool['rows'] if r['source_token']==token][:pool['target_per_source']])
    summary=dict(rows=selected,qualified=len(selected),sample_reports=[r['sample_report'] for r in selected],
        evidence=pool['evidence'],target_per_anchor=pool['target_per_source'],status='completed',
        plan=str(wave/'trajectory-family-plan.json'),plan_sha256=digest(wave/'trajectory-family-plan.json'))
    if config['trajectory_family'].get('cohort_mode')=='global_50_min1_v1':
        summary.update(minimum_fresh_samples=selection['minimum_fresh_samples'],fresh_selected=selection['fresh_selected'])
    combined=bind_training_plan(plan,selected,anchors)
    frozen_json(folder/'training-cohort-plan.json',combined)
    frozen_json(folder/'training-cohort.json',summary)
    return summary


def bind_training_plan(plan,selected,anchors):
    """Use the existing explicit reuse protocol; keep every original plan intact."""
    from driversi.data.catalog.store import identity
    from driversi.rsi.decisions.capability_skills import validate_plan,validate_sample_lineage
    import copy
    combined=copy.deepcopy(plan);combined.pop('plan_id',None)
    combined['assignments']=[];combined['reused_executions']={}
    for i,row in enumerate(selected):
        origin=validate_plan(read_mapping(row['origin_plan']))
        if digest(row['origin_plan'])!=row['origin_plan_sha256'] or origin['context']!=plan['context']:
            raise ValueError('Cross-round skill parent context differs')
        assignment=dict(row['origin_assignment'],id=f'cohort-{i:03d}',anchor_assignment_id=anchors[row['source_token']])
        combined['assignments'].append(assignment)
        combined['reused_executions'][assignment['id']]=dict(path=row['generation'],sha256=digest(row['generation']))
    combined['plan_id']=identity(combined)
    validate_plan(combined)
    for row in selected:
        sample=read_mapping(row['sample_report'])
        validate_sample_lineage(combined,[sample['row']],sample['files'])
    return combined
