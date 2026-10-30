"""Harness-owned bounded batches using the existing resident postprocessor."""
import time
from pathlib import Path

from driversi.agents.backends.native import digest
from driversi.configuration import read_mapping
from driversi.data.catalog.store import identity
from driversi.rsi.experiments.continuation import frozen_json


def recover_completed(root, plan):
    """Collect the original batch after partial write-back; never regroup it."""
    from driversi.runtime.budget import write_json
    assignments={x['id']:x for x in plan['assignments']}
    for directory in (Path(root)/'candidates').glob('*'):
        if (directory/'outcome.json').exists() and (directory/'outcome-seal.json').exists():continue
        reservation=directory/'generation/render-reservation.json'
        if not reservation.exists():continue
        reference=read_mapping(reservation);bound=Path(reference['batch'])
        if digest(bound)!=reference['sha256']:raise ValueError('Reservation changed')
        folder=bound.parent;seal=folder/'completion.json'
        if not seal.exists():
            raise RuntimeError('Original resident batch unsettled; no regrouping or resubmission')
        completed=read_mapping(seal)
        from .storage_recovery import result_paths
        manifest,report=result_paths(folder)
        if digest(report)!=completed['report_sha256'] or digest(manifest)!=completed['manifest_sha256']:
            raise ValueError('Batch result or manifest changed')
        spec=read_mapping(manifest);reserved=read_mapping(bound)
        if (Path(spec['reservation']).resolve()!=bound.resolve()
                or spec['reservation_sha256']!=digest(bound)
                or digest(spec['config'])!=reserved['config_sha256']):
            raise ValueError('Batch input binding changed')
        jobs=[j for j in reserved['jobs'] if j['generation']==str(directory/'generation/report.json')]
        if len(jobs)!=1:raise ValueError('Exactly one reserved input required')
        job=jobs[0]
        if (Path(job['diversity']).resolve()!=(directory/'diversity.json').resolve()
                or digest(job['generation'])!=job['sha256']
                or digest(job['diversity'])!=job['diversity_sha256']):
            raise ValueError('Prepared generation or diversity changed')
        rows=read_mapping(report)['rows']
        matched=[r for r in rows if r['prepared_generation']==str(directory/'generation/report.json')]
        if len(matched)!=1:raise ValueError('Exactly one original result required')
        outcome=matched[0]['outcome']
        for path,sha in outcome.get('files',{}).items():
            if digest(path)!=sha:raise ValueError('Admitted result changed')
        div=read_mapping(directory/'diversity.json');desc=div['description'];a=assignments[directory.name]
        for path,sha in desc['files'].items():
            if digest(path)!=sha:raise ValueError('Trajectory evidence changed')
        result=dict(outcome,candidate=a['id'],anchor_assignment_id=a['anchor_assignment_id'],
            sampling_seed=a['sampling_seed'],behavior_types=desc['behavior_types'],
            trajectory_files=desc['files'],distinct=div['decision']['distinct'],descriptor=desc,
            diversity=str(directory/'diversity.json'))
        if plan.get('trajectory_family',{}).get('cohort_mode')=='global_50_min1_v1':
            result['rendered']=True  # original reservation and completed batch already verified
        if (directory/'outcome.json').exists():
            if read_mapping(directory/'outcome.json')!=result:raise ValueError('Partial outcome differs from sealed batch')
        else:write_json(directory/'outcome.json',result)
        frozen_json(directory/'outcome-seal.json',dict(sha256=digest(directory/'outcome.json')))


def dispatch(config, root, pending, deadline):
    from .resident_batch import run
    c = read_mapping(config); policy = c['resident_postprocessing']
    jobs = []
    for assignment, directory, outcome in pending:
        generation = directory/'generation/report.json'
        diversity = directory/'diversity.json'
        jobs.append(dict(generation=str(generation), sha256=digest(generation),
            diversity=str(diversity), diversity_sha256=digest(diversity)))
    key = identity(dict(config=digest(config), jobs=jobs))[:20]
    folder = Path(root)/'resident-batches'/key
    folder.mkdir(parents=True,exist_ok=True)
    reservation = folder/'reservation.json'
    frozen_json(reservation, dict(config_sha256=digest(config), jobs=jobs,
        candidates=[a['id'] for a, _, _ in pending],
        admission=c['visual_quality_profile'], no_training_dispatch=True))
    for _, directory, _ in pending:
        frozen_json(directory/'generation/render-reservation.json',
                    dict(batch=str(reservation), sha256=digest(reservation)))
    report = folder/'execution/report.json'
    seal = folder/'completion.json'
    if seal.exists():
        from .storage_recovery import result_paths
        result_manifest,report=result_paths(folder)
        saved = read_mapping(seal)
        if digest(report) != saved['report_sha256'] or digest(result_manifest)!=saved['manifest_sha256']:
            raise ValueError('Resident result changed')
        result = read_mapping(report)
        for row in result['rows']:
            for path, sha in row['outcome'].get('files', {}).items():
                if digest(path) != sha:raise ValueError('Admitted output changed')
        return result['rows']
    if (folder/'execution').exists():
        raise RuntimeError('Unsettled resident batch; reconcile without repeating native work')
    seconds = min(7200, int(deadline-time.time()))
    if seconds < 60:raise TimeoutError('No remaining batch budget')
    manifest = folder/'manifest.json'
    from driversi.data.datasets.postprocessing import mode
    features = {} if mode(c) == 'training_only' else dict(
        feature_gpu=policy['feature_gpu'], feature_worker_config=c['generated_catalog']+'.feature-worker.json')
    frozen_json(manifest, dict(schema_version=1, purpose='production', mode='resident',
        output=str(folder/'execution'), config=str(config), jobs=jobs,
        reservation=str(reservation), reservation_sha256=digest(reservation),
        gpus=policy['gpus'][:min(len(pending),len(policy['gpus']))],
        **features,
        max_seconds=seconds, staging=policy.get('staging')))
    result = run(manifest)
    if result['status'] != 'completed':raise RuntimeError('Resident batch did not settle')
    frozen_json(seal, dict(report_sha256=digest(report), manifest_sha256=digest(manifest)))
    return result['rows']
