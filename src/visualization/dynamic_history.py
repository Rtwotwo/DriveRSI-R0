"""Read-only current projection of the native dynamic/history campaign.

Publication lives only for the existing Harness operation. It starts no jobs,
retries no experiment, and performs no network/public-site deployment.
"""
import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .skill_evolution import read


def project(directory, plan_path=None):
    root = Path(directory)
    plan = read(root / 'plan.json') or (read(Path(plan_path)) if plan_path else {})
    summary = read(root / 'report.json') or read(root / 'progress.json')
    contract = summary.get('image_contract', plan.get('gates', {}).get('image_contract', 'history_rgb_future_states_v1'))
    anchor_only = contract == 'anchor_rgb_history_future_states_v1'
    started = read(root / 'started.json')
    status = summary.get('status', 'running' if started else 'preparing')
    source_rows = {row['id']: row for row in summary.get('sources', [])}
    sources = []
    admitted = int(summary.get('admitted_samples', 0))
    rejected = unknown = pending = 0
    for source in plan.get('sources', []):
        settled = source_rows.get(source['id'], {})
        folder = root / 'sources' / source['id'] / 'asset'
        status_paths = list(folder.glob('**/reconstruction/status.json'))
        if source.get('prebuilt_asset_job'):
            prior = read(Path(source['prebuilt_asset_job']))
            if prior.get('output'):
                prior_status = Path(prior['output']) / 'status.json'
                if prior_status.exists():
                    status_paths.append(prior_status)
        status_paths.sort(key=lambda path: path.stat().st_mtime)
        native = read(status_paths[-1]) if status_paths else {}
        point = read(status_paths[-1].parent / 'point-cloud-progress.json') if status_paths else {}
        result = read(folder / 'stage-result.json')
        branches = settled.get('branches', [])
        counts = {'admitted': 0, 'rejected': 0, 'unknown': 0, 'pending': len(source.get('branches', [])) - len(branches)}
        reasons = {}
        for branch in branches:
            if branch['status'] == 'training_sample_ready':
                counts['admitted'] += 1
            elif branch['status'] in ('unknown', 'running', 'blocked_unknown'):
                counts['unknown'] += 1
            else:
                counts['rejected'] += 1
            if branch.get('reason'):
                entries = branch['reason'] if isinstance(branch['reason'], list) else [branch['reason']]
                for reason in entries:
                    if not isinstance(reason, str):
                        reason = reason.get('code', 'structured_rejection') if isinstance(reason, dict) else 'structured_rejection'
                    reasons[reason] = reasons.get(reason, 0) + 1
        rejected += counts['rejected']; unknown += counts['unknown']; pending += counts['pending']
        sources.append({'id': source['id'], 'phase': native.get('status', settled.get('asset_status', 'pending')),
                        'asset_status': result.get('status', settled.get('asset_status', 'pending')),
                        'prepared_frames': point.get('completed_frames'),
                        'eligible_objects': native.get('eligible_objects'),
                        'bound_objects': len(result.get('object_tokens', [])) if result else None,
                        'unsupported_objects': len(result.get('unsupported_objects', [])) if result else None,
                        'counts': counts, 'rejections': reasons})
    training = summary.get('training', {})
    train_root = root / 'training/execution'
    recipe = read(train_root / 'training-input.json')
    stages = []
    database = train_root / 'harness.sqlite'
    if database.exists():
        with sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True) as db:
            row = db.execute('SELECT payload FROM harness_tasks ORDER BY rowid DESC LIMIT 1').fetchone()
        if row:
            task = json.loads(row[0])
            stages = [{'id': step['id'], 'status': step['status']} for step in task.get('stages', [])]
    updates = {'control': 0, 'treatment': 0}
    for filename in ('training-progress.json', 'training-result.json'):
        for path in (train_root / 'tasks').glob('**/' + filename):
            values = read(path)
            for arm in updates:
                if any(part.startswith(arm + '_') for part in path.relative_to(train_root).parts):
                    updates[arm] = max(updates[arm], int(values.get('total_updates', 0)))
    native_started = any(value > 0 for value in updates.values())
    return {'kind': 'dynamic_history_campaign', 'status': status,
            'model': plan.get('model', 'chainflow_stage1'), 'sources': sources,
            'planned_reused_rollouts': sum(len(row.get('branches', [])) for row in plan.get('sources', [])),
            'physical_rollouts_reused': int(summary.get('physical_rollouts_reused', 0)),
            'new_teacher_rollouts': int(summary.get('new_teacher_rollouts', 0)),
            'admitted_samples': admitted, 'rejected_samples': rejected, 'unknown_samples': unknown,
            'pending_samples': pending, 'history_images_per_sample': 4 if anchor_only else 16,
            'input_rgb_per_sample': 4 if anchor_only else 16, 'future_rgb_per_sample': 0,
            'rgb_scope': 'anchor_only' if anchor_only else 'historical_frames',
            'history_state_frames': 4, 'future_numeric_frames': 10,
            'image_contract': contract, 'native_loss_unchanged': True,
            'minimum_samples': plan.get('limits', {}).get('minimum_samples'),
            'minimum_sources': plan.get('limits', {}).get('minimum_sources'),
            'wall_seconds': time.time() - started['started_at'] if status == 'running' and started else summary.get('wall_seconds'),
            'promoted': summary.get('promoted', False),
            'reason': summary.get('reason'),
            'training': {'dispatch_started': bool(summary.get('training_started')),
                         'native_training_started': native_started,
                         'observed_updates_per_arm': updates,
                         'status': training.get('status', 'native_execution' if native_started else
                                                'preparing' if summary.get('training_started') else 'not_started'),
                         'control': 'real_replay', 'treatment': 'synthetic_1_real_7',
                         'optimizer_updates_per_arm': recipe.get('updates_per_arm', training.get('optimizer_updates_per_arm')),
                         'requested_synthetic_exposures': recipe.get('requested_synthetic_exposures'),
                         'global_batch_size': recipe.get('global_batch_size'), 'stages': stages}}


def refresh(directory, config_path=None):
    from driversi.agents.backends.native import project_root
    from .build import atomic
    root = project_root()
    config_path = config_path or root / 'src/config/visualization/platform.json'
    config = read(Path(config_path))
    route = config.get('current_dynamic_history_campaign')
    if not route:
        return False
    route = {'directory': route} if isinstance(route, str) else route
    selected = (root / route['directory']).resolve()
    if selected != Path(directory).resolve():
        return False
    if not selected.is_relative_to((root / 'exp').resolve()):
        raise ValueError('Current campaign must be under experiment storage')
    destination = Path(config['output']).resolve()
    if destination != Path(config['site_root']).resolve() / 'platform':
        raise ValueError('Use the existing unified platform destination')
    payload = read(destination / 'platform.json')
    if payload.get('schema_version') != 1 or payload.get('read_only') is not True:
        raise ValueError('Build the existing private platform before live projection')
    plan = root / route['plan'] if route.get('plan') else None
    payload['current'] = project(selected, plan)
    payload['generated_at'] = datetime.now(timezone.utc).isoformat()
    atomic(destination / 'platform.json', json.dumps(payload, ensure_ascii=False, allow_nan=False))
    return True


@contextmanager
def publishing(directory):
    """Lifetime-bound UI heartbeat; never an independent experiment monitor."""
    stop = threading.Event()
    def update():
        try:
            refresh(directory)
        except Exception as exc:
            # UI I/O must never abort native production or make fabricated status.
            from driversi.runtime.budget import write_json
            write_json(Path(directory) / 'visualization-error.json',
                       {'error_type': type(exc).__name__, 'time': time.time()})
    def loop():
        while not stop.wait(20):
            update()
    update()
    worker = threading.Thread(target=loop, daemon=True, name='dynamic-history-private-projection')
    worker.start()
    try:
        yield
    finally:
        stop.set(); worker.join(timeout=2); update()
