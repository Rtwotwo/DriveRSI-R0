"""Bounded per-source new-history expansion through existing Harness providers.

Teacher qualification, geometric diversity, rendering, visual admission and
native label admission are separate states. Nothing here scores a student.
"""
import copy
import hashlib
import json
import math
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from driversi.agents.backends.native import digest, project_root
from driversi.configuration import read_mapping
from driversi.rsi.workflow.artifacts import verify
from driversi.rsi.workflow.engine import register
from driversi.rsi.workflow.store import HarnessStore

_STOP = threading.Event()
_CHILDREN = {}
_CHILDREN_LOCK = threading.Lock()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)


def candidate_grid():
    # Same controller/physical limits as the existing bounded-arrival provider.
    return [dict(arrival_backend='driverl_native', arrival_action=[0., 0.])] + [
        dict(arrival_backend='bounded_control', arrival_action=[j, s])
        for j in (0., -.125, .125, -.25, .25)
        for s in (0., -.02, .02, -.04, .04)]


def prepare(teacher_template, render_template, output, *, target=20,
            max_renders=4, workers=2, history_rms_m=.15, future_rms_m=.5,
            existing_rollouts=()):
    from driversi.contracts.scene import Scene
    from driversi_agents.chainflow.benchmark import official_filter
    from driversi.data.assets.worldengine import bundle_files

    root, output = project_root(), Path(output).resolve()
    if not output.is_relative_to(root / 'exp'):
        raise ValueError('Expansion artifacts must stay in exp')
    if (type(target) is not int or not 1 <= target <= 100 or
            type(max_renders) is not int or not 0 <= max_renders <= target or
            type(workers) is not int or workers not in (1, 2)):
        raise ValueError('Bounded target, render quota and one/two workers required')
    for value in (history_rms_m, future_rms_m):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError('Positive finite diversity distances required')
    if (output / 'plan.json').exists():
        raise ValueError('Expansion plan is immutable; use run to resume')
    teacher, render = read_mapping(teacher_template), read_mapping(render_template)
    if ([s['provider'] for s in teacher['steps']] !=
            ['navsim_native_teacher_probe', 'native_branch_selection']):
        raise ValueError('Native teacher and branch selection template required')
    if [s['id'] for s in teacher['steps']] != ['teacher', 'select']:
        raise ValueError('Teacher template IDs must be teacher/select')
    c = teacher['steps'][0]['config']
    if c.get('teacher_backend') != 'driverl_native':
        raise ValueError('This expansion recipe freezes native DriveRL future policy')
    scene = Scene.load(Path(c['scene']))
    _, official = official_filter('navtrain')
    origin = scene.metadata['source_train_membership']
    if (scene.metadata['split'] != 'train' or
            origin['source_log'] != scene.metadata['source_log'] or
            origin['source_log'] not in set(official['log_names']) or
            origin['token'] not in set(official['tokens'])):
        raise ValueError('Official training-source membership required')
    if [s['provider'] for s in render['steps']] != [
            'completed_navsim_teacher_reference', 'native_branch_selection',
            'worldengine_selected_branch_render', 'worldengine_source_reconstruction']:
        raise ValueError('Existing sealed GS render/source comparison spine required')
    if [s['id'] for s in render['steps']] != ['reuse', 'select', 'render', 'source']:
        raise ValueError('Render template stage IDs differ from native spine')
    future_frames = teacher['steps'][1]['config']['future_frames']
    if type(future_frames) is not int or future_frames not in (8, 10):
        raise ValueError('Native 8/10-point selected future required')
    if render['steps'][1]['config']['future_frames'] != teacher['steps'][1]['config']['future_frames']:
        raise ValueError('Teacher and render windows differ')
    bundle = Path(render['steps'][2]['config']['bundle']).resolve()
    asset_files = bundle_files(bundle.parent, bundle.name)
    if not all(v['present'] for v in asset_files.values()):
        raise ValueError('Complete GS bundle required')
    inputs = [Path(teacher_template), Path(render_template), Path(c['scene']),
              Path(c['road_graph']), Path(c['checkpoint']),
              Path(scene.metadata['navsim_source']['log_path'])]
    split_file, _ = official_filter('navtrain')
    inputs.append(Path(split_file))
    existing_rollouts = [str(Path(p).resolve()) for p in existing_rollouts]
    for path in existing_rollouts:
        trajectory(path, future_frames)
        with np.load(path, allow_pickle=False) as z:
            m = json.loads(str(z['metadata']))['scene_metadata']
            for key in ('source_log', 'source_start_frame'):
                if m[key] != scene.metadata[key]:
                    raise ValueError('Dedup reference must share the exact source window')
            if m['navsim_source'] != scene.metadata['navsim_source']:
                raise ValueError('Dedup reference source coordinate/time binding differs')
        inputs.append(Path(path))
    for name in ('src/rsi/experiments/branch_expansion.py',
                 'src/evaluation/search/navsim_teacher.py',
                 'src/rsi/plugins/navsim_teacher.py', 'agents/driverl/closed_loop.py'):
        inputs.append(root / name)
    plan = dict(kind='bounded_branch_expansion', version=1, target_qualified_samples=target,
                candidates=candidate_grid(), max_renders=max_renders, workers=workers,
                existing_rollouts=existing_rollouts,
                future_frames=future_frames,
                source=origin, scene_source_token=scene.metadata['scenario_id'],
                diversity=dict(history_rms_m=history_rms_m, future_rms_m=future_rms_m,
                               rule='both_distances_at_least_threshold_against_each_accepted_branch'),
                teacher_template=teacher, render_template=render,
                files={str(p.resolve()): digest(p) for p in inputs},
                asset_files=asset_files, bundle=str(bundle),
                scored_student=False, learned_self_play=False,
                visual_admission_automatic=False, training_started=False)
    plan['plan_id'] = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    write(output / 'plan.json', plan)
    return plan


def verified_plan(output):
    from driversi.data.assets.worldengine import bundle_files
    plan = read_mapping(Path(output) / 'plan.json')
    body = {k: v for k, v in plan.items() if k != 'plan_id'}
    if hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() != plan['plan_id']:
        raise ValueError('Expansion plan changed')
    for path, sha in plan['files'].items():
        if digest(path) != sha:
            raise ValueError('Expansion implementation/source changed: ' + path)
    bundle = Path(plan['bundle'])
    if bundle_files(bundle.parent, bundle.name) != plan['asset_files']:
        raise ValueError('Expansion GS asset version changed')
    return plan


def trajectory(path, future_frames=8):
    if type(future_frames) is not int or future_frames not in (8, 10):
        raise ValueError('Native 8/10-point future required')
    with np.load(path, allow_pickle=False) as z:
        ids = list(z['actor_ids'])
        if ids.count('ego') != 1:
            raise ValueError('One physical ego required')
        xy = np.asarray(z['states'][:, ids.index('ego'), :2], dtype=np.float64)
        if len(xy) != 71 or not np.isfinite(xy).all():
            raise ValueError('Full finite 2s history plus 5s execution required')
        return xy[[5, 10, 15, 20]], xy[[20+5*i for i in range(1, future_frames+1)]]


def novelty(candidate, accepted, thresholds):
    distances = []
    for key, prior in accepted:
        h, f = (float(np.sqrt(np.mean(np.sum((a-b)**2, axis=1))))
                for a, b in zip(candidate, prior))
        distances.append(dict(attempt=key, history_rms_m=h, future_rms_m=f))
    passed = all(d['history_rms_m'] >= thresholds['history_rms_m'] and
                 d['future_rms_m'] >= thresholds['future_rms_m'] for d in distances)
    return passed, distances


def _existing_or_register(output, task):
    store = HarnessStore(output / 'harness.sqlite')
    try:
        state = store.get(task['id'])
    except ValueError as exc:
        if str(exc) != 'Unknown Harness task':
            raise
        path = output / 'specs' / (task['id'] + '.json')
        write(path, task)
        return register(output / 'harness.sqlite', path, output / 'tasks')
    actual = state['task']
    if actual['budget'] != task['budget'] or len(actual['steps']) != len(task['steps']):
        raise ValueError('Existing task does not belong to this expansion plan')
    for a, b in zip(actual['steps'], task['steps']):
        if (any(a[k] != b[k] for k in ('id', 'provider', 'inputs', 'max_wall_seconds'))
                or any(a['config'].get(k) != v for k, v in b['config'].items())):
            raise ValueError('Existing task configuration differs')
    return state


def _execute(output, state):
    if _STOP.is_set():
        return state
    if state['status'] == 'registered':
        with (output / 'specs' / (state['id'] + '.log')).open('w') as log:
            # Separate supervisors: subreaper/child ownership must not be shared
            # between Python threads running the engine in this process.
            import os
            process = subprocess.Popen([sys.executable, '-m', __name__, 'supervise',
                                        str(os.getpid()), str(output/'harness.sqlite'), state['id']],
                                       cwd=project_root(), stdout=log, stderr=subprocess.STDOUT)
            with _CHILDREN_LOCK:
                _CHILDREN[state['id']] = process
            try:
                process.wait()
            except BaseException:
                process.terminate()
                process.wait()
                raise
            finally:
                with _CHILDREN_LOCK:
                    _CHILDREN.pop(state['id'], None)
    elif state['status'] == 'running':
        raise ValueError('An attempt needs explicit Harness recovery; never relaunch it')
    return HarnessStore(output/'harness.sqlite').get(state['id'])


def _teacher_row(state, index):
    row = dict(index=index, task=state['id'], status=state['status'],
               reason=state.get('failure_reason'), wall_seconds=state.get('elapsed_seconds'),
               sim_agent_steps=state.get('sim_agent_steps'), teacher_qualified=False,
               distinct=False, rendered=False, visual_admitted=False, trainable=False)
    if state['status'] != 'completed':
        return row
    _verify_completed(state)
    selected = next(s for s in state['stages'] if s['id'] == 'select')
    path = selected['result']['artifacts']['report']
    report = read_mapping(path)
    row.update(selection=path, selection_sha256=digest(path),
               teacher_qualified=report['teacher_qualified'],
               pdms=report['trajectory_evidence']['score']['score'] * 100,
               rollout=str(Path(report['execution'])/'teacher.npz'))
    return row


def _verify_completed(state):
    if state['status'] == 'completed':
        for stage in state['stages']:
            declaration = next(s for s in state['task']['steps'] if s['id'] == stage['id'])
            verify(stage['result'], state['id'], declaration)


def _preflight(states):
    if any(s['status'] == 'running' for s in states):
        raise ValueError('Reconcile all unresolved Harness attempts before continuing')
    for state in states:
        _verify_completed(state)


def run(output):
    import fcntl
    output = Path(output).resolve()
    with (output/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = verified_plan(output)
        old = read_mapping(output/'report.json') if (output/'report.json').exists() else {}
        prior_admissions = [r['admission_report'] for r in old.get('rows', []) if r.get('trainable')]
        states = []
        for i, c in enumerate(plan['candidates']):
            t = copy.deepcopy(plan['teacher_template'])
            t['id'] = 'expand-' + plan['plan_id'][:10] + f'-{i:02d}'
            t['label'] = 'New history expansion; native teacher; no student scoring'
            t['steps'][0]['config'].update(c)
            states.append(_existing_or_register(output, t))
        old_renders = []
        for state in states:
            try:
                old_renders.append(HarnessStore(output/'harness.sqlite').get(state['id']+'-render'))
            except ValueError as exc:
                if str(exc) != 'Unknown Harness task':
                    raise
        _preflight(states + old_renders)
        rows = []
        _STOP.clear()
        pool = ThreadPoolExecutor(max_workers=plan['workers'])
        try:
            for i, state in enumerate(pool.map(lambda s: _execute(output, s), states)):
                rows.append(_teacher_row(state, i))
                write(output/'progress.json', dict(status='generating', completed=len(rows),
                                                   planned=len(states), rows=rows))
        except BaseException:
            _STOP.set()
            with _CHILDREN_LOCK:
                for process in _CHILDREN.values():
                    if process.poll() is None:
                        process.terminate()
            pool.shutdown(wait=True, cancel_futures=True)
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        verified_plan(output)
        accepted = [('existing:' + str(i), trajectory(p, plan['future_frames']))
                    for i, p in enumerate(plan['existing_rollouts'])]
        excluded_count = len(accepted)
        for row in rows:
            if row['teacher_qualified']:
                signature = trajectory(row['rollout'], plan['future_frames'])
                row['distinct'], row['diversity_comparisons'] = novelty(signature, accepted, plan['diversity'])
                if row['distinct']:
                    accepted.append((row['index'], signature))
                else:
                    row['reason'] = 'near_duplicate_history_or_future'
        for row in [r for r in rows if r['distinct']][:plan['max_renders']]:
            t = copy.deepcopy(plan['render_template'])
            t['id'] = row['task'] + '-render'
            t['steps'][0]['config']['source_task'] = row['task']
            state = _execute(output, _existing_or_register(output, t))
            row['render_task'] = state['id']
            row['render_status'] = state['status']
            row['render_wall_seconds'] = state.get('elapsed_seconds')
            if state['status'] == 'completed':
                _verify_completed(state)
                row['rendered'] = True
                row['visual_review_status'] = 'independent_review_and_native_labels_required'
            write(output/'progress.json', dict(status='rendering', rows=rows))
        verified_plan(output)
        # Rendering success is deliberately not a route to training admission.
        report = dict(kind='bounded_branch_expansion_result', plan_id=plan['plan_id'],
                      status='completed_with_quota_shortfall', source=plan['source'],
                      target_qualified_samples=plan['target_qualified_samples'],
                      attempts=len(rows), teacher_qualified=sum(r['teacher_qualified'] for r in rows),
                      distinct_qualified_trajectories=len(accepted)-excluded_count, rendered=sum(r['rendered'] for r in rows),
                      visually_admitted=0, trainable_samples=0,
                      qualified_sample_shortfall=plan['target_qualified_samples'],
                      rows=rows, native_model_loss_or_gates_changed=False,
                      student_scored=False, training_started=False,
                      limitation='Geometric variety is not independent scene coverage or proven student benefit')
        if prior_admissions:
            _admit_reports(output, report, prior_admissions)
        write(output/'report.json', report)
        return report


def attach_admissions(output, sample_reports):
    """Count only independently admitted native joint caches of this batch.

    Called in the AUGUstus environment after pixel/label reviews and native
    publication. This function cannot itself accept an image or a label.
    """
    import fcntl
    output = Path(output).resolve()
    with (output/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verified_plan(output)
        report = read_mapping(output/'report.json')
        old = [r['admission_report'] for r in report['rows'] if r.get('trainable')]
        refs = list(dict.fromkeys(old + [str(Path(p).resolve()) for p in sample_reports]))
        _admit_reports(output, report, refs)
        write(output/'report.json', report)
        return report


def _admit_reports(output, report, refs):
    from driversi_agents.augustus.branch_supervision import validate_sample
    by_report = {str((output/'tasks'/r['render_task']/'select/report.json').resolve()): r
                 for r in report['rows'] if r['rendered'] and r['distinct']}
    seen = set()
    for row in report['rows']:
        row.update(visual_admitted=False, trainable=False)
    for path in refs:
        sample = validate_sample(path)
        row = by_report.get(str(Path(sample['branch_report']).resolve()))
        if row is None or sample['sample_id'] in seen:
            raise ValueError('Foreign or duplicate published synthetic sample')
        _validate_batch_admission(output, row, sample)
        seen.add(sample['sample_id'])
        if row.get('sample_id') not in (None, sample['sample_id']):
            raise ValueError('One physical branch cannot inflate the sample quota')
        row.update(visual_admitted=True, trainable=True, sample_id=sample['sample_id'],
                   admission_report=str(Path(path).resolve()), admission_sha256=digest(path))
    n = sum(r['trainable'] for r in report['rows'])
    report.update(visually_admitted=n, trainable_samples=n,
                  qualified_sample_shortfall=max(0, report['target_qualified_samples']-n),
                  status='target_met' if n >= report['target_qualified_samples']
                  else 'completed_with_quota_shortfall')


def _validate_batch_admission(output, row, sample):
    state = HarnessStore(output/'harness.sqlite').get(row['render_task'])
    if state['status'] != 'completed':
        raise ValueError('Batch renderer must be completed and sealed')
    _verify_completed(state)
    artifacts = {s['id']: s['result']['artifacts'] for s in state['stages']}
    visual = read_mapping(sample['visual_admission'])
    expected = dict(branch_report=artifacts['select']['report'],
                    render_report=artifacts['render']['render_report'],
                    trajectory_window=artifacts['select']['trajectory_window'],
                    source_comparison=artifacts['source']['report'])
    if any(Path(visual[k]).resolve() != Path(v).resolve() for k, v in expected.items()):
        raise ValueError('Native sample visual proof belongs to another render attempt')


def _supervise(parent, registry, task_id):
    """Bind each independent Harness supervisor to its batch coordinator."""
    import ctypes
    import os
    import signal
    from driversi.rsi.workflow.engine import run as execute

    def interrupted(*_):
        raise KeyboardInterrupt('expansion coordinator stopped')

    signal.signal(signal.SIGTERM, interrupted)
    if ctypes.CDLL(None).prctl(1, signal.SIGTERM) != 0 or os.getppid() != parent:
        raise RuntimeError('Cannot bind expansion supervisor lifetime')
    state = execute(Path(registry), task_id)
    return 0 if state['status'] == 'completed' else 2


if __name__ == '__main__':
    if len(sys.argv) == 5 and sys.argv[1] == 'supervise':
        raise SystemExit(_supervise(int(sys.argv[2]), sys.argv[3], sys.argv[4]))
    raise SystemExit('Use scripts/rsi/run_branch_expansion.py for prepare/run/admit')
