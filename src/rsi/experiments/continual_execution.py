"""Finite automatic execution of the existing continual curriculum commands.

This is a foreground job, not a recurring monitor. Native phase budgets remain
authoritative. Missing resources/evidence stop the job rather than retrying it.
"""
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from driversi.configuration import read_mapping
from driversi.runtime.budget import write_json


def gpu_snapshot():
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used',
                             '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, timeout=15)
    if result.returncode:
        raise RuntimeError('GPU inventory unavailable')
    rows = {}
    for line in result.stdout.splitlines():
        index, memory = line.split(',')
        rows[int(index)] = int(memory)
    if not rows:
        raise RuntimeError('Empty GPU inventory')
    return rows


def required_gpus(case, operation):
    config = read_mapping(Path(case['current_round']) / 'generation-config.json')
    if operation == 'dispatch':
        resident=config.get('resident_postprocessing')
        from driversi.data.datasets.postprocessing import mode
        feature = [] if mode(config) == 'training_only' or not resident else [resident['feature_gpu']]
        return sorted(set([config['production_gpu'],*resident['gpus'],*feature])) if resident else [config['production_gpu']]
    model = read_mapping(config['training_model_config'])
    return model['settings']['distributed_training']['devices']



def owned_feature_gpu(case):
    """Permit reuse only when every compute PID on that GPU is the exact service."""
    if not case.get('current_round'):return None
    config_path=Path(case['current_round'])/'generation-config.json'
    if not config_path.exists():return None
    config=read_mapping(config_path)
    from driversi.data.datasets.postprocessing import mode
    if mode(config) == 'training_only':return None
    resident=config.get('resident_postprocessing')
    if not resident:return None
    descriptor=Path(config['generated_catalog']+'.feature-worker.json')
    if not descriptor.exists():return None
    feature=read_mapping(descriptor);directory=Path(feature['directory'])
    if not (directory/'active.json').exists() or not (directory/'launch.json').exists():return None
    active=read_mapping(directory/'active.json');launch=read_mapping(directory/'launch.json')
    from driversi.data.catalog import feature_worker
    process_birth=feature_worker.process_birth
    if process_birth(active['pid'])!=active['birth'] or launch['pid']!=active['pid']:return None
    cmd=Path('/proc')/str(active['pid'])/'cmdline'
    from driversi.agents.backends.native import digest
    if not cmd.exists() or launch.get('config_sha256')!=digest(descriptor):return None
    actual=[x.decode() for x in cmd.read_bytes().split(b'\0') if x]
    expected=launch.get('command',[])
    if (actual!=expected or len(actual)!=4 or actual[1]!=str(Path(feature_worker.__file__).resolve())
            or actual[2]!='--config' or actual[3]!=str(descriptor.resolve())):return None
    ids=subprocess.run(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],capture_output=True,text=True,timeout=15)
    apps=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],capture_output=True,text=True,timeout=15)
    if ids.returncode or apps.returncode:return None
    mapping={int(line.split(',')[0]):line.split(',')[1].strip() for line in ids.stdout.splitlines()}
    uuid=mapping[feature['gpu']]
    pids={int(line.split(',')[1]) for line in apps.stdout.splitlines() if line.split(',')[0].strip()==uuid}
    return feature['gpu'] if pids=={active['pid']} else None

def next_operation(state):
    case = next((c for c in state['cases'] if c['id'] == state['active']), None)
    if case is None:
        if any(c['status'] in ('asset_recovery_unknown', 'needs_execution_reconciliation',
                              'unknown_execution', 'dispatching', 'training') for c in state['cases']):
            return None,'needs_reconciliation'
        if any(c['status']=='queued' for c in state['cases']):return 'prepare',None
        if any(c['status']=='asset_recovery_pending' for c in state['cases']):
            return None,'asset_recovery_in_progress'
        if any(c['status']=='asset_recovery_unknown' for c in state['cases']):
            return None,'needs_reconciliation'
        if any(c['status']=='needs_execution_reconciliation' for c in state['cases']):
            return None,'needs_reconciliation'
        if any(c['status']=='generation_unresolved' for c in state['cases']):
            return None,'unresolved_generation'
        if any(c['status']=='generation_budget_exhausted' for c in state['cases']):
            return None,'generation_budget_exhausted'
        if any(c['status']=='assets_unresolved' for c in state['cases']):
            return None,'unresolved_assets'
        return None,'finished'
    status = case['status']
    if status=='asset_recovery_pending':return None,'asset_recovery_in_progress'
    operations = {'queued':'prepare', 'ready':'dispatch', 'data_ready':'train',
                  'awaiting_release':'settle', 'needs_revision':'prepare',
                  'blocked_assets':'recover_assets'}
    if status in operations:
        return operations[status], None
    reasons = {'generation_stopped':'generation_shortage',
               'dispatching':'needs_reconciliation', 'training':'needs_reconciliation',
               'unknown_execution':'needs_reconciliation'}
    return None, reasons.get(status, 'state_requires_review')


def registered_recovery(root, case):
    from .cohort_recovery import eligible
    if eligible(root,case):return 'recover_accounting'
    if not case or case.get('status')!='unknown_execution' or not case.get('current_round'):
        return None
    path=Path(root)/'storage-recovery-policy.json'
    if not path.exists() or read_mapping(path).get('rule')!='png_cleanup_terminal_v1':return None
    from .storage_recovery import diagnose
    wave=Path(case['current_round'])/'waves/wave-0001'
    pending=[p for p in (wave/'resident-batches').glob('*')
        if (p/'execution').exists() and not (p/'completion.json').exists()]
    try:
        for folder in pending:
            previous=sorted(folder.glob('storage-recovery-*/manifest.json'))
            manifest=previous[-1] if previous else folder/'manifest.json'
            if not (Path(read_mapping(manifest)['output'])/'report.json').exists():diagnose(manifest)
        if pending:return 'recover_storage'
        failure=wave/'production-failure.txt'
        text=failure.read_text() if failure.exists() else ''
        if not all(s in text for s in ('image_store.py','TemporaryDirectory','Directory not empty','driversi-png-')):
            return None
        for event in (Path(root)/'recovery-memory').glob('event-*.json'):
            v=read_mapping(event)
            if v.get('wave')==str(wave) and v.get('status') in ('recovering','recovered'):
                return 'recover_storage'
    except (ValueError,FileNotFoundError,KeyError):
        return None
    return None


def run(root, *, max_problems=3, max_transitions=18, max_wall_seconds=21600):
    """Called under the curriculum's single global lock, including phase execution."""
    from . import continual_curriculum as cc
    if type(max_problems) is not int or max_problems < 1:
        raise ValueError('Positive explicit problem budget required')
    if type(max_transitions) is not int or not 1 <= max_transitions <= 64 * max_problems:
        raise ValueError('Automatic transition budget must be 1..64 per allowed problem')
    if type(max_wall_seconds) is not int or not 1<=max_wall_seconds<=86400:
        raise ValueError('Automatic wall budget must be 1..86400 seconds')
    root = Path(root)
    state = read_mapping(cc.state_path(root))
    if max_problems > max(3, len(state['cases'])):
        raise ValueError('Problem budget exceeds the frozen curriculum')
    counts = [len(c['candidates']) for c in state['cases']]
    if any(not 1 <= n <= state['policy']['max_candidates'] for n in counts) or any(
            len({r['source_log'] for r in c['candidates']}) != len(c['candidates'])
            for c in state['cases']):
        raise ValueError('Frozen candidate limits and distinct source logs required')
    # Numbered receipts belong to this campaign, never to another dashboard/run.
    folder = root / 'automatic'
    folder.mkdir(exist_ok=True)
    number = len(list(folder.glob('execution-*.json'))) + 1
    receipt = folder / f'execution-{number:03d}.json'
    now = lambda: datetime.now(timezone.utc).isoformat()
    record = dict(schema_version=1, status='starting', started_at=now(),
                  max_problems=max_problems, max_transitions=max_transitions,
                  max_wall_seconds=max_wall_seconds,
                  transitions=0, problems=[], commands=[],
                  candidate_min=min(counts), candidate_max=max(counts),
                  recurring_monitor=False, native_gates_unchanged=True)

    def save(status, **fields):
        record.update(status=status, updated_at=now(), **fields)
        write_json(receipt, record)
        state = read_mapping(cc.state_path(root))
        state['automatic_execution'] = {k: record[k] for k in (
            'status','started_at','updated_at','transitions','max_problems','max_transitions')}
        cc.save(root, state)
        return cc.project(state)

    save('starting')
    from .asset_recovery import recover_current, reconcile_pending, policy as recovery_policy
    operations = dict(prepare=cc.prepare, dispatch=cc.dispatch, train=cc.train_current,
                      settle=cc.settle,recover_assets=recover_current)
    from .generation_recovery import revise
    operations['revise_generation']=revise
    from .storage_recovery import recover_campaign
    operations['recover_storage']=recover_campaign
    from .cohort_recovery import recover as recover_accounting
    from .source_expansion import after_assets, policy as expansion_policy
    operations['recover_accounting']=recover_accounting
    operations['expand_sources']=after_assets
    started=time.monotonic()
    while record['transitions']<max_transitions:
        if time.monotonic()-started>=max_wall_seconds:
            return save('automatic_wall_budget_completed')
        if (root / 'STOP_AUTOMATIC').exists():
            return save('stopped_by_request')
        reconcile_pending(root)
        state = read_mapping(cc.state_path(root))
        operation, stop = next_operation(state)
        active=next((x for x in state['cases'] if x['id']==state['active']),None)
        if stop=='needs_reconciliation':
            recovery=registered_recovery(root,active)
            if recovery:operation,stop=recovery,None
        if active and active['status']=='assets_unresolved' and expansion_policy(root):
            operation,stop='expand_sources',None
        if stop=='generation_shortage' and (root/'generation-recovery-policy.json').exists():
            operation,stop='revise_generation',None
        if operation == 'recover_assets' and recovery_policy(root) is None:
            return save('blocked_assets',recovery_reason='backend_not_registered')
        if stop=='asset_recovery_in_progress':
            save('waiting_asset_recovery');time.sleep(30);continue
        if stop:
            return save(stop)
        case = next((c for c in state['cases'] if c['id'] == state['active']), None)
        if case is None:
            queued=[c for c in state['cases'] if c['status']=='queued']
            if recovery_policy(root):
                from .asset_recovery import ready_sources
                ready=ready_sources(root)
                queued.sort(key=lambda c:(-sum(x['token'] in ready for x in c['candidates']),c['id']))
            case=queued[0]
        if case['id'] not in record['problems']:
            if len(record['problems']) >= max_problems:
                return save('problem_budget_completed')
            record['problems'].append(case['id'])
        if operation in ('dispatch','train','recover_storage'):
            try:
                wanted = required_gpus(case, 'dispatch' if operation=='recover_storage' else operation)
                observed = gpu_snapshot()
            except Exception as exc:
                return save('resource_check_failed', error_type=type(exc).__name__)
            record['resource_check'] = dict(required=wanted, memory_used_mib=observed)
            reusable=owned_feature_gpu(case) if operation=='dispatch' else None
            if any(type(i) is not int or i not in observed or (observed[i] >= 500 and i!=reusable) for i in wanted):
                if recovery_policy(root):
                    save('waiting_resources');time.sleep(30);continue
                return save('blocked_resources')
        save('running', current_command=operation)
        try:
            from driversi.visualization.curriculum_operations import publishing
            with publishing(root):
                operations[operation](root)
        except Exception as exc:
            if operation=='dispatch' and (root/'storage-recovery-policy.json').exists():
                record['commands'].append(dict(operation=operation,problem=case['id'],
                    status='failed_pending_classification',error_type=type(exc).__name__,finished_at=now()))
                record['transitions']+=1
                save('reconciling_failure')
                continue
            save('execution_failed', error_type=type(exc).__name__)
            raise  # Preserve original failure; never repeat an unknown paid/GPU call.
        record['transitions'] += 1
        record['commands'].append(dict(operation=operation, problem=case['id'], finished_at=now()))
        save('running')
    # The final phase can itself discover a block or an unknown native outcome.
    # Report that terminal condition rather than masking it as budget success.
    operation, stop = next_operation(read_mapping(cc.state_path(root)))
    if operation=='recover_assets' and recovery_policy(root) is None:
        stop='blocked_assets'
    return save(stop or 'transition_budget_completed')
