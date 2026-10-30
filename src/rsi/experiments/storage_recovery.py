"""Receipt-bound recovery of terminal PNG cleanup failures, never unknown work."""
import os
import time
from pathlib import Path

from driversi.agents.backends.native import digest
from driversi.configuration import read_mapping
from driversi.runtime.budget import write_json
from .continuation import frozen_json

RULE = 'png_cleanup_terminal_v1'


def result_paths(folder):
    folder = Path(folder)
    original = folder / 'manifest.json'
    seal = folder / 'completion.json'
    if seal.exists():
        value = read_mapping(seal)
        if 'result_manifest' not in value:
            return original,folder/'execution/report.json'
        manifest = Path(value.get('result_manifest', original))
        if not manifest.resolve().is_relative_to(folder.resolve()):
            raise ValueError('Recovery manifest escapes reservation')
        if value.get('original_manifest_sha256', digest(original)) != digest(original):
            raise ValueError('Original reservation manifest changed')
        report=Path(read_mapping(manifest)['output']) / 'report.json'
        if not report.resolve().is_relative_to(folder.resolve()):raise ValueError('Recovery output escapes reservation')
        return manifest, report
    return original, folder / 'execution/report.json'


def diagnose(manifest):
    """Deliberately narrow: one failed render, before export/admission began."""
    manifest = Path(manifest); spec = read_mapping(manifest); out = Path(spec['output'])
    if len(spec['jobs']) != 1 or len(spec['gpus']) != 1:
        raise ValueError('Storage recovery requires one unambiguous unfinished branch')
    if read_mapping(out/'status.json').get('status') != 'failed':
        raise ValueError('Not a terminal failed batch')
    settled = read_mapping(out/'final-service-settlement.json')
    if settled.get('errors') or any(not r['settled'] for r in settled['rows']):
        raise ValueError('Owned services have not settled')
    request = read_mapping(out/'request.json')
    if request['manifest_sha256'] != digest(manifest):
        raise ValueError('Executed manifest changed')
    reservation = read_mapping(spec['reservation'])
    if (digest(spec['reservation']) != spec['reservation_sha256']
            or reservation['jobs'] != spec['jobs']
            or reservation['config_sha256'] != digest(spec['config'])):
        raise ValueError('Reserved inputs changed')
    for job in spec['jobs']:
        if digest(job['generation']) != job['sha256'] or digest(job['diversity']) != job['diversity_sha256']:
            raise ValueError('Prepared physical evidence changed')
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        if int(p.parent.name) == os.getpid(): continue
        try: parts = p.read_bytes().split(b'\0')
        except (OSError, PermissionError): continue
        if any(str(out).encode() in x for x in parts) and any(
                b'driversi.' in x or b'parent_exec.py' in x for x in parts):
            raise ValueError('Old owned worker still alive')
    lane = out/'lane-0'; branch = lane/'branch-000'
    if (lane/'report.json').exists() or (branch/'admission').exists() or (branch/'generation/report.json').exists():
        raise ValueError('Published or ambiguous postprocessing must not repeat')
    failure = lane/'failure.txt'; text = failure.read_text()
    if not all(s in text for s in ('image_store.py','TemporaryDirectory', 'Directory not empty', 'driversi-png-')):
        raise ValueError('Unregistered storage fault')
    start = float(read_mapping(out/'launch.json')['started'])
    seconds = max(0., (out/'status.json').stat().st_mtime-start)
    return dict(rule=RULE, failure_sha256=digest(failure), elapsed_seconds=seconds,
                known_terminal=True, physical_reexecution=False, admitted_reexecution=False)


def recover_batch(folder):
    from .resident_batch import run
    folder = Path(folder); original = folder/'manifest.json'
    if (folder/'completion.json').exists(): return
    spec = read_mapping(original)
    attempts = sorted(folder.glob('storage-recovery-*/attempt.json'))
    previous = original if not attempts else attempts[-1].parent/'manifest.json'
    if seal_completed(folder, previous): return
    if len(attempts) >= 2: raise ValueError('Storage recovery retry budget exhausted')
    # A launched attempt without terminal evidence never becomes a new attempt.
    proof = diagnose(previous)
    spent = diagnose(original)['elapsed_seconds'] if attempts else proof['elapsed_seconds']
    for p in attempts:
        spent += diagnose(p.parent/'manifest.json')['elapsed_seconds']
    remaining = min(1800, int(spec['max_seconds']-spent))
    if remaining < 60: raise ValueError('Original postprocessing budget exhausted')
    directory = folder/f'storage-recovery-{len(attempts)+1:02d}'
    directory.mkdir()
    new = dict(spec, output=str(directory/'execution'), max_seconds=remaining)
    frozen_json(directory/'manifest.json', new)
    frozen_json(directory/'attempt.json', dict(proof, executor='DriveHarness', started=time.time(),
        input_manifest_sha256=digest(original), previous_manifest=str(previous),
        charged_prior_seconds=spent, max_seconds=remaining))
    try:
        result = run(directory/'manifest.json')
    except BaseException as exc:
        write_json(directory/'result.json',dict(status='failed',error_type=type(exc).__name__,finished=time.time()))
        if isinstance(exc, Exception):
            # Reconcile the new terminal receipt before using the second slot.
            # Unknown/unsettled failures and exhausted budgets fail in diagnose.
            return recover_batch(folder)
        raise
    if result['status'] != 'completed': raise ValueError('Recovery did not settle')
    if not seal_completed(folder,directory/'manifest.json'):raise ValueError('Recovery completion missing')
    write_json(directory/'result.json',dict(status='completed',finished=time.time(),
        production_increment=result['production_increment'],wall_seconds=result['wall_seconds']))


def seal_completed(folder, manifest):
    """Recover the commit boundary without rerunning completed GPU work."""
    folder,manifest=Path(folder),Path(manifest);spec=read_mapping(manifest);out=Path(spec['output'])
    if not (out/'report.json').exists():return False
    result=read_mapping(out/'report.json');settled=read_mapping(out/'final-service-settlement.json')
    original=read_mapping(folder/'manifest.json');reserved=read_mapping(spec['reservation'])
    if (result.get('status')!='completed' or read_mapping(out/'status.json').get('status')!='completed'
            or settled.get('errors') or any(not x['settled'] for x in settled['rows'])
            or read_mapping(out/'request.json')['manifest_sha256']!=digest(manifest)
            or spec['jobs']!=original['jobs'] or spec['reservation']!=original['reservation']
            or spec['config']!=original['config'] or digest(spec['config'])!=reserved['config_sha256']
            or digest(spec['reservation'])!=spec['reservation_sha256']
            or {x['prepared_generation'] for x in result['rows']}!={j['generation'] for j in spec['jobs']}):
        raise ValueError('Incomplete or changed completed batch')
    for job in spec['jobs']:
        if digest(job['generation'])!=job['sha256'] or digest(job['diversity'])!=job['diversity_sha256']:
            raise ValueError('Completed batch input changed')
    for row in result['rows']:
        for p,sha in row['outcome'].get('files',{}).items():
            if digest(p)!=sha:raise ValueError('Completed output changed')
    frozen_json(folder/'completion.json',dict(report_sha256=digest(out/'report.json'),
        manifest_sha256=digest(manifest),result_manifest=str(manifest),
        original_manifest_sha256=digest(folder/'manifest.json'),recovery_rule=RULE))
    return True


def recover_campaign(root):
    from . import continual_curriculum as cc
    from .processing_resume import settled_generation
    from .reactive_autoloop import run
    root=Path(root);state=read_mapping(cc.state_path(root))
    case=next(c for c in state['cases'] if c['id']==state['active'])
    if case['status']!='unknown_execution': raise ValueError('Only terminal unknown production is reconcilable')
    folder=Path(case['current_round']);wave=folder/'waves/wave-0001'
    if (folder/'training-input.json').exists(): raise ValueError('Training requires its native recovery protocol')
    settlement=settled_generation(wave,allow_prior_completed=True)
    pending=[p for p in (wave/'resident-batches').glob('*')
             if (p/'execution').exists() and not (p/'completion.json').exists()]
    for p in pending:
        prior=sorted(p.glob('storage-recovery-*/manifest.json'))
        latest=prior[-1] if prior else p/'manifest.json'
        if not (Path(read_mapping(latest)['output'])/'report.json').exists():diagnose(latest)
    memory=root/'recovery-memory';memory.mkdir(exist_ok=True)
    if not pending:
        events=[read_mapping(p) for p in sorted(memory.glob('event-*.json'))]
        matching=[v for v in events if v.get('wave')==str(wave) and v['status'] in ('recovering','recovered')]
        if not matching:raise ValueError('No registered recovery continuation')
        for p in matching[-1]['batches']:
            manifest,report=result_paths(p);seal=read_mapping(Path(p)/'completion.json')
            if digest(report)!=seal['report_sha256'] or digest(manifest)!=seal['manifest_sha256']:
                raise ValueError('Recovery continuation changed')
    if len(list(memory.glob('event-*.json')))>=8: raise ValueError('Campaign recovery budget exhausted')
    event=memory/f'event-{len(list(memory.glob("event-*.json")))+1:03d}.json'
    value=dict(rule=RULE,problem=case['id'],status='recovering',started=time.time(),
        wave=str(wave),
        executor='DriveHarness',settled_physical_jobs=settlement['known_results'],
        batches=[str(p) for p in pending],physical_reexecution=False)
    write_json(event,value)
    try:
        for p in pending:recover_batch(p)
        value.update(status='recovered',finished=time.time());write_json(event,value)
        case['status']='dispatching';cc.save(root,state)
        run(folder/'generation-config.json','production')
    except BaseException as exc:
        case['status']='unknown_execution';cc.save(root,state)
        # Preserve successful storage repair separately from a later production error.
        if value['status']!='recovered':
            value.update(status='failed',error_type=type(exc).__name__,finished=time.time());write_json(event,value)
        raise
    result=read_mapping(wave/'production-status.json')
    case['qualified']=result.get('qualified_samples',0)
    case['status']='data_ready' if result['status']=='generation_wave_completed' else 'generation_stopped'
    state['generation_started']=True;cc.save(root,state)
    if (root/'operations-policy.json').exists():
        from .problem_cohort import update
        update(root);state=read_mapping(cc.state_path(root))
    return cc.project(state)
