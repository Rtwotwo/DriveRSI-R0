"""Rebuildable provenance views; canonical SimScale data never move or relabel."""
import time
from collections import Counter
from pathlib import Path

from driversi.agents.backends.native import digest
from driversi.configuration import read_mapping
from driversi.runtime.budget import write_json


def build(dataset, campaign):
    dataset,campaign=Path(dataset),Path(campaign)
    state=read_mapping(campaign/'private/state.json')
    origins={};source_rows={};training=[];exposures=Counter();planned=set()
    active=state.get('active')
    for folder in sorted((campaign/'rounds').glob('*')):
        declaration=folder/'round.json';plan=folder/'waves/wave-0001/trajectory-family-plan.json'
        if not declaration.exists() or not plan.exists():continue
        dec=read_mapping(declaration);problem=dec['problem'];wave=plan.parent
        assignments={a['id']:a for a in read_mapping(plan)['assignments']}
        for p in (wave/'candidates').glob('*'):
            if p.name not in assignments:continue
            a=assignments[p.name];key=(problem,a['token'])
            row=source_rows.setdefault(key,dict(problem=problem,source_token=a['token'],
                source_log=a['source_log'],attempted=0,admitted=0,rejections={},physical_worker_seconds=0.))
            if (p/'generation').exists():row['attempted']+=1
            clock=p/'physical-prefetch.json'
            if clock.exists():
                v=read_mapping(clock)
                if v.get('finished') is not None:row['physical_worker_seconds']+=max(0.,v['finished']-v['started'])
            out=p/'outcome.json'
            if not out.exists():continue
            v=read_mapping(out)
            if v.get('trainable'):
                row['admitted']+=1
                package=v.get('dataset_reference',{}).get('package_id')
                if package:origins.setdefault(package,[]).append(dict(problem=problem,source_token=a['token'],
                    source_log=a['source_log'],round=folder.name,candidate=a['id'],outcome=str(out),
                    sample_report=v.get('sample_report'),outcome_sha256=digest(out)))
            else:row['rejections'][v['status']]=row['rejections'].get(v['status'],0)+1
        recipe=folder/'training-recipe.json';status=wave/'chainflow-pair/training-status.json'
        if (folder/'training-input.json').exists():
            planned.update(read_mapping(folder/'training-input.json')['samples'])
        if recipe.exists():
            r=read_mapping(recipe)
            traces=[]
            for report in (wave/'chainflow-pair/tasks').glob('*/*/training-result.json'):
                relative=str(report.relative_to(wave))
                if report.parent.name not in ('treatment_train','treatment_resume'):continue
                actual=read_mapping(report)
                if actual.get('sample_trace'):traces.append(actual)
            actual=max(traces,key=lambda x:len(x['sample_trace'])) if traces else None
            counts=Counter(s for x in actual['sample_trace'] for s in x['sample_ids']) if actual else Counter()
            exposures.update(counts)
            training.append(dict(problem=problem,round=folder.name,recipe=r,
                observed_unique_samples=len(counts),observed_exposures=sum(counts.values()),
                actual_recipe=actual.get('effective_training_recipe') if actual else None,
                status=read_mapping(status) if status.exists() else {'status':
                    'running_awaiting_native_receipt' if any(c.get('current_round')==str(folder) and c['status']=='training'
                        for c in state['cases']) else 'planned_not_started'}))
    members=[];split_counts={}
    for index in (dataset/'manifests').glob('*/index.json'):
        d=read_mapping(index);split_counts[index.parent.name]=d['count']
        if len(set(d['samples']))!=d['count']:raise ValueError('Collection coverage mismatch')
        for package in d['samples']:
            manifest=index.parent/(package+'.json');item=read_mapping(manifest)
            members.append(dict(package_id=package,split=index.parent.name,manifest=str(manifest),
                manifest_sha256=digest(manifest),origins=origins.get(package,[]),
                lineage_status='current_campaign' if package in origins else 'preserved_prior_synthetic',
                metric=item.get('pdms') or item.get('epdms'),
                observed_exposures=exposures[item['token']],
                training_usage='consumed' if exposures[item['token']] else
                    'planned_not_observed' if any(o['sample_report'] in planned for o in origins.get(package,[]))
                    else 'not_observed_in_current_campaign'))
    current=next((x for x in state['cases'] if x['id']==active),None)
    pool=None
    if current and current.get('current_round'):
        p=Path(current['current_round'])/'cumulative-cohort.json'
        if p.exists():pool=read_mapping(p)
    for row in source_rows.values():
        row['cumulative_unique']=((pool or {}).get('unique_per_source',{}).get(row['source_token'])
            if row['problem']==active else None)
        row['target']=((1 if state['policy'].get('cohort_mode')=='global_50_min1_v1' else
            state['policy']['targets'][min(current['round'],len(state['policy']['targets'])-1)])
            if current and row['problem']==active else None)
        row['remaining']=None if row['cumulative_unique'] is None or row['target'] is None else max(0,row['target']-row['cumulative_unique'])
    recoveries=[read_mapping(p) for p in sorted((campaign/'recovery-memory').glob('event-*.json'))]
    value=dict(schema_version=1,generated_at=time.time(),active_problem=active,
        split_counts=split_counts,total_preserved=len(members),members=members,
        sources=list(source_rows.values()),training=training,recoveries=recoveries,
        preserved_prior=sum(not m['origins'] for m in members),
        note='Derived lineage only; dataset membership is not training consumption or student gain')
    write_json(dataset/'manifests/curriculum-lineage.json',value)
    return value


def public(value):
    """Allowlisted aggregate projection; no test token, private paths or log bodies."""
    sources=[{k:r[k] for k in ('problem','source_token','source_log','attempted','admitted',
        'rejections','physical_worker_seconds','cumulative_unique','target','remaining')}
        for r in value['sources'] if r['problem']==value['active_problem']]
    training=[]
    keys=('samples','epochs','devices','gpu_count','global_batch','optimizer_updates_per_arm',
          'min_exposures','max_exposures','scaled_base_lr','learning_rate_policy','precision',
          'selection_reason','learned_optimal')
    for t in value['training']:
        if t['problem']!=value['active_problem']:continue
        training.append(dict(problem=t['problem'],round=t['round'],
            recipe={k:t['recipe'][k] for k in keys if k in t['recipe']},
            status=t['status'].get('status'),updates_per_arm=t['status'].get('updates_per_arm'),
            observed_unique_samples=t.get('observed_unique_samples'),observed_exposures=t.get('observed_exposures')))
    return dict(generated_at=value['generated_at'],active_problem=value['active_problem'],
        total_preserved=value['total_preserved'],preserved_prior=value['preserved_prior'],
        split_counts=value['split_counts'],sources=sources,training=training,
        recoveries=[{k:e[k] for k in ('rule','problem','status','started','finished','settled_physical_jobs') if k in e}
            for e in value['recoveries']],read_only=True)
