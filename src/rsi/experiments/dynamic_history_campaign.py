"""Bounded same-branch dynamic assets → history RGB → native replay pair.

This is an adapter over the existing reconstruction, admission and Harness
training lifecycle. A stopped worker is never inferred to have failed safely.
Completed physical rollouts are reused explicitly; no new teacher score is
invented and no future image is rendered by this campaign.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import SimpleQueue
from threading import Lock
import argparse
import json
import math
import os
import signal
import subprocess
import time

from driversi.agents.backends.native import digest, project_root
from driversi.configuration import read_mapping
from driversi.rsi.experiments.continuation import _lock, frozen_json, task_state
from driversi.runtime.budget import write_json

IMAGE_CONTRACT = 'history_rgb_future_states_v1'
VISUAL_PROFILE = 'dynamic_history25_v1'
CONTRACTS = {
    IMAGE_CONTRACT:dict(profile=VISUAL_PROFILE,proof_version=3,history_images=16),
    'anchor_rgb_history_future_states_v1':dict(profile='dynamic_anchor25_v1',
                                              proof_version=4,history_images=4),
}


def validate_plan(plan):
    """Validate a finite campaign; runtime policy cannot lower admission gates."""
    if plan.get('schema_version') != 1 or plan.get('model') != 'chainflow_stage1':
        raise ValueError('Versioned native ChainFlow stage1 campaign required')
    from driversi.rsi.contracts import _identity
    _identity(plan['id'], 'dynamic history campaign')
    gates = plan['gates']
    contract=CONTRACTS.get(gates.get('image_contract'))
    if gates != dict(pdms_gt=.95, source_psnr_gt=25.0,
                     image_contract=gates.get('image_contract'),
                     visual_profile=contract['profile'] if contract else None) or not contract:
        raise ValueError('PDMS, PSNR, binding and history-only gates are fixed')
    sources = plan['sources']
    if not isinstance(sources, list) or not 2 <= len(sources) <= 3:
        raise ValueError('Small batch requires two or three declared navtrain sources')
    identifiers, branches = set(), set()
    for source in sources:
        _identity(source['id'], 'source id')
        if source['id'] in identifiers or not source['branches']:
            raise ValueError('Unique sources with explicit physical branches required')
        identifiers.add(source['id'])
        for branch in source['branches']:
            _identity(branch['id'], 'branch id')
            if branch['id'] in branches:
                raise ValueError('A physical branch cannot be repeated in this batch')
            branches.add(branch['id'])
            for key in ('generation_report', 'diversity_report'):
                if not Path(branch[key]).is_absolute():
                    raise ValueError('Physical evidence paths must be absolute')
        if not Path(source['recovery_job']).is_absolute():
            raise ValueError('Original capture recovery job must be absolute')
        if source.get('prebuilt_asset_job') and not Path(source['prebuilt_asset_job']).is_absolute():
            raise ValueError('Prebuilt asset job must be absolute')
        if 'adopt_existing_pid' in source:
            if (not source.get('prebuilt_asset_job') or type(source['adopt_existing_pid']) is not int
                    or source['adopt_existing_pid']<=0 or not source.get('adopt_existing_start')
                    or not source.get('adopt_host')):
                raise ValueError('Asset adoption requires exact job, host, PID and start identity')
    limits = plan['limits']
    if (not 2 <= len(branches) <= 12 or limits['max_branches'] != len(branches)
            or type(limits['minimum_samples']) is not int
            or not 2 <= limits['minimum_samples'] <= len(branches)
            or type(limits['minimum_sources']) is not int
            or not 2 <= limits['minimum_sources'] <= len(sources)):
        raise ValueError('Explicit small-batch count and distinct-source floor required')
    runtime, training = plan['runtime'], plan['training']
    for key in ('asset_gpus','render_gpus'):
        values=runtime[key]
        if (not isinstance(values,list) or not 1 <= len(values) <= 2
                or len(set(values)) != len(values)
                or any(type(v) is not int or not 0 <= v <= 7 for v in values)):
            raise ValueError('One or two distinct allocated GPUs per stage required')
    if (type(runtime['asset_timeout_seconds']) is not int
            or not 60 <= runtime['asset_timeout_seconds'] <= 86400
            or type(runtime['asset_max_iterations']) is not int
            or not 1 <= runtime['asset_max_iterations'] <= 30001):
        raise ValueError('Bounded allocated reconstruction resources required')
    if (type(training['synthetic_exposures']) is not int
            or not 1 <= training['synthetic_exposures'] <= 8
            or type(training['batch_size']) is not int
            or not 1 <= training['batch_size'] <= 8
            or type(training['max_updates_per_arm']) is not int
            or not 2 <= training['max_updates_per_arm'] <= 256
            or not isinstance(training['target_token'], str) or not training['target_token']):
        raise ValueError('Bounded replay exposures and one local target required')
    for mapping, keys in ((runtime, ('asset_python', 'render_python')),
                          (training, ('base_manifest', 'model_config'))):
        if any(not Path(mapping[k]).is_absolute() for k in keys):
            raise ValueError('Campaign runtime and training paths must be absolute')
    if runtime.get('gsplat_runtime_witness') and not Path(runtime['gsplat_runtime_witness']).is_absolute():
        raise ValueError('Verified native runtime witness must be absolute')
    return plan


def input_files(plan):
    """The manifest/model contents, branch receipts and capture jobs are frozen."""
    files = [plan['training'][k] for k in ('base_manifest', 'model_config')]
    if plan['runtime'].get('gsplat_runtime_witness'):files.append(plan['runtime']['gsplat_runtime_witness'])
    for source in plan['sources']:
        files.append(source['recovery_job'])
        if source.get('prebuilt_asset_job'):files.append(source['prebuilt_asset_job'])
        for branch in source['branches']:
            files.extend(branch[k] for k in ('generation_report', 'diversity_report'))
    return {str(Path(p).resolve(strict=True)): digest(p) for p in files}


def _verify_files(files):
    if any(digest(p) != sha for p, sha in files.items()):
        raise ValueError('Campaign input changed after preparation')


def _stage(folder, invoke, *, recover=None, retries=1):
    """Consume durable final receipts; never relaunch an unknown operation.

    Adapters own safe internal checkpoint recovery. The campaign only advances
    after explicit rejection/completion, or reconciles a completed final report.
    """
    folder.mkdir(parents=True, exist_ok=True)
    receipt, marker = folder/'stage-result.json', folder/'started.json'
    if receipt.exists():
        return read_mapping(receipt)
    def settle(value):
        if (value.get('status')=='retryable' and value.get('reason')=='known_transient_storage'
                and value.get('result_known') is True and value.get('no_inflight') is True):
            failure={k:value[k] for k in ('status','reason','result_known','no_inflight')}
            frozen_json(folder/'known-failure.json',failure)
            if retries:
                next_value=_stage(folder/'retry-1',invoke,retries=0)
                if next_value.get('status')=='unknown':return next_value
                value=dict(next_value,automatic_retries=1,prior_failure=failure)
            else:value=dict(value,status='failed',reason='bounded_storage_retry_exhausted')
        frozen_json(receipt,value)
        return value
    if marker.exists():
        final = folder/'execution/report.json'
        value = read_mapping(final) if final.exists() else (recover() if recover else None)
        if not value or value.get('status') in ('running', 'registered', 'unknown', 'blocked_unknown'):
            return dict(status='unknown', reason='unfinished_owned_operation', stage=str(folder))
        return settle(dict(**value,reconciled=True))
    frozen_json(marker, dict(pid=os.getpid(), started_at=time.time()))
    started = time.monotonic()
    try:
        value = invoke(folder/'execution')
    except Exception as exc:
        # Do not turn a Python exception into a known physical rejection.
        write_json(folder/'exception.json', dict(type=type(exc).__name__, message=str(exc),
                   wall_seconds=time.monotonic()-started, automatic_resend=False))
        return dict(status='unknown', reason='operation_exception', error_type=type(exc).__name__,
                    stage=str(folder))
    if not isinstance(value, dict) or 'status' not in value:
        raise ValueError('Adapter did not return an explicit result')
    value = dict(value, stage_wall_seconds=time.monotonic()-started)
    if value['status'] in ('running', 'registered', 'unknown', 'blocked_unknown'):
        return dict(value, status='unknown')
    return settle(value)


def _group_alive(pid):
    try:os.killpg(pid,0)
    except ProcessLookupError:return False
    # Dead orphan zombies may await init reaping but cannot keep doing GPU work.
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:fields=path.read_text().rsplit(')',1)[1].split()
        except (OSError,IndexError):continue
        if int(fields[2])==pid and fields[0]!='Z':return True
    return False


def _owned_process(command, output, timeout):
    """Reap only the child process group created here, including on timeout."""
    started_at=time.time()
    with (output/'reconstruction.log').open('ab') as log:
        child=subprocess.Popen(command,cwd=project_root(),stdout=log,stderr=subprocess.STDOUT,
                               start_new_session=True)
        stat=(Path('/proc')/str(child.pid)/'stat').read_text().rsplit(')',1)[1].split()
        write_json(output/'owned-process.json',dict(pid=child.pid,start=stat[19],
                   group=child.pid,host=os.uname().nodename,command=command))
        timed_out=False
        try:code=child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out=True
            os.killpg(child.pid,signal.SIGTERM)
            try:code=child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);code=child.wait(timeout=10)
        except BaseException:
            if _group_alive(child.pid):os.killpg(child.pid,signal.SIGKILL)
            child.wait(timeout=10)
            raise
        if _group_alive(child.pid):
            os.killpg(child.pid,signal.SIGKILL)
            deadline=time.monotonic()+10
            while _group_alive(child.pid) and time.monotonic()<deadline:time.sleep(.1)
        settled=not _group_alive(child.pid)
    receipt=dict(returncode=code,timed_out=timed_out,no_inflight=settled,
                 process_group=child.pid,started_at=started_at,finished_at=time.time())
    write_json(output/'process-exit.json',receipt)
    if not settled:raise RuntimeError('Owned descendant group did not settle')
    return receipt


def _known_asset_failure(exit_receipt,status_path):
    if not status_path.exists():return None
    status=read_mapping(status_path)
    if (exit_receipt.get('returncode') not in (None,0)
            and exit_receipt.get('no_inflight') is True and not exit_receipt.get('timed_out')
            and status.get('status')=='failed'
            and exit_receipt['started_at'] <= status.get('failed_at',-1) <= exit_receipt['finished_at']+1):
        return dict(status='failed',reason='native_asset_execution_failed',
                    error_type=status.get('error_type'),native_status=str(status_path),
                    result_known=True,no_inflight=True,automatic_resend=False)
    return None


def reconstruct(source, runtime, output):
    from driversi.rendering.gaussian.dynamic_reconstruction import prepare_dynamic_job
    output.mkdir(parents=True, exist_ok=True)
    if source.get('prebuilt_asset_job'):
        job=Path(source['prebuilt_asset_job']).resolve(strict=True)
        spec=read_mapping(job)
        if (spec.get('kind')!='native_mtgs_dynamic_v1'
                or Path(spec['recovery_job']).resolve()!=Path(source['recovery_job']).resolve()
                or spec['gpu'] not in runtime['asset_gpus']):
            raise ValueError('Prebuilt asset job belongs to another capture or allocation')
        status=_adopt_asset(source,job,output,runtime['asset_timeout_seconds'])
        if status is not None:return status
    else:
        job = prepare_dynamic_job(source['recovery_job'], output/'reconstruction',
                                  gpu=runtime['gpu'], max_iterations=runtime['asset_max_iterations'])
    command=asset_command(job,runtime)
    frozen_json(output/'native-invocation.json',dict(command=command))
    process=_owned_process(command,
        output,runtime['asset_timeout_seconds'])
    report_path = Path(read_mapping(job)['output'])/'report.json'
    if not report_path.exists():
        text=(output/'reconstruction.log').read_text(errors='replace')[-12000:]
        known=any(x in text for x in ('[Errno 39] Directory not empty','[Errno 116] Stale file handle'))
        if process['returncode']!=0 and known and not process['timed_out']:
            value=dict(status='retryable',reason='known_transient_storage',result_known=True,
                       no_inflight=True,process_exit=str(output/'process-exit.json'))
            write_json(output/'report.json',value);return value
        failed=_known_asset_failure(process,Path(read_mapping(job)['output'])/'status.json')
        if failed:
            write_json(output/'report.json',failed);return failed
        raise RuntimeError('Native asset worker has no final receipt; exit='+str(process['returncode']))
    report = read_mapping(report_path)
    if report['status'] == 'completed' and not Path(report['bundle_root']).is_dir():
        raise ValueError('Completed reconstruction lacks its actual bundle')
    write_json(output/'report.json', report)
    return report


def asset_command(job,runtime):
    witness=runtime.get('gsplat_runtime_witness')
    module=('driversi.rendering.gaussian.verified_gsplat_runtime' if witness else
            'driversi.rendering.gaussian.dynamic_reconstruction')
    return [runtime['asset_python'],'-u','-m',module,'--job',str(job)]+(
        ['--witness',witness] if witness else [])


def _job_processes(job):
    matches=[]
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args=p.read_bytes().decode().strip('\0').split('\0')
            stat=(p.parent/'stat').read_text().rsplit(')',1)[1].split()
        except (OSError,UnicodeError,IndexError):continue
        if stat[0]=='Z':continue
        if '--job' in args and args[args.index('--job')+1:args.index('--job')+2]==[str(job)]:
            matches.append(dict(pid=int(p.parent.name),group=int(stat[2]),start=stat[19]))
    return matches


def _adopt_asset(source,job,output,timeout):
    """Wait for an explicitly owned old asset group; never double-launch it."""
    adopted=source.get('adopt_existing_pid')
    if adopted and source['adopt_host']!=os.uname().nodename:
        raise ValueError('Adopted asset belongs to another execution host')
    started=time.monotonic()
    while True:
        active=_job_processes(job)
        if not active:break
        _validate_asset_claim(source,active)
        # Native training children stay in the verified parent group. Its leader
        # may exit after training while descendants settle; no restart occurs.
        write_json(output/'adoption.json',dict(status='waiting_existing_asset',job=str(job),
            pid=adopted,start=str(source['adopt_existing_start']),processes=active,
            waited_seconds=time.monotonic()-started,new_launch=False))
        if time.monotonic()-started>=timeout:
            return dict(status='unknown',reason='adopted_asset_still_running',automatic_resend=False)
        time.sleep(5)
    value=_completed_asset(job)
    if value is not None:
        write_json(output/'report.json',value)
        return value
    # No live job and no final receipt: the native job owns its checkpoint-aware
    # recovery and can continue without repeating completed optimization.
    return None


def _validate_asset_claim(source,active):
    adopted=source.get('adopt_existing_pid')
    if adopted and source['adopt_host']!=os.uname().nodename:
        raise ValueError('Adopted asset belongs to another execution host')
    if not adopted or any(row['group']!=adopted for row in active):
        raise RuntimeError('Existing asset job is not the declared owned process group')
    direct=next((row for row in active if row['pid']==adopted),None)
    if direct and direct['start']!=str(source['adopt_existing_start']):
        raise RuntimeError('Adopted asset PID was reused')


def _completed_asset(job):
    path=Path(read_mapping(job)['output'])/'report.json'
    if path.exists():
        report=read_mapping(path)
        if report.get('status') in ('completed','unsupported'):
            if report['status']=='completed':
                if not report.get('outputs'):raise ValueError('Completed asset has no sealed outputs')
                _verify_files(report['outputs'])
                if not Path(report['bundle_root']).is_dir():raise ValueError('Completed asset bundle missing')
            value=dict(report,adopted_existing_asset=True,new_asset_launch=False)
            return value
    return None


def reconcile_asset(source,runtime,output):
    """Read-only recovery after interrupted adoption; never relaunch a job."""
    if not source.get('prebuilt_asset_job'):return None
    if source.get('adopt_existing_pid') and source['adopt_host']!=os.uname().nodename:
        raise ValueError('Asset recovery must run on its recorded execution host')
    job=Path(source['prebuilt_asset_job']).resolve(strict=True);spec=read_mapping(job)
    if (spec.get('kind')!='native_mtgs_dynamic_v1'
            or Path(spec['recovery_job']).resolve()!=Path(source['recovery_job']).resolve()
            or spec['gpu'] not in runtime['asset_gpus']):
        raise ValueError('Recovery points to another asset job')
    active=_job_processes(job)
    if active:
        owned=Path(output)/'owned-process.json';claim=source
        if owned.exists():
            saved=read_mapping(owned);args=saved['command']
            if '--job' not in args or args[args.index('--job')+1]!=str(job):
                raise ValueError('Saved owned process belongs to another job')
            claim=dict(adopt_existing_pid=saved['pid'],adopt_existing_start=saved['start'],adopt_host=saved['host'])
        _validate_asset_claim(claim,active)
        return dict(status='unknown',reason='adopted_asset_still_running',automatic_resend=False)
    return _completed_asset(job)


def _parallel_jobs(jobs, gpus, invoke):
    """Bounded lanes: one operation per allocated device, never shared by accident."""
    available=SimpleQueue()
    for gpu in gpus:available.put(gpu)
    def work(job):
        gpu=available.get()
        try:return invoke(job,gpu)
        finally:available.put(gpu)
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        future_jobs={pool.submit(work,job):job for job in jobs}
        for future in as_completed(future_jobs):
            yield future_jobs[future],future.result()


def regenerate(branch, asset, runtime, output):
    from driversi.rendering.diagnostics.dynamic_history import regenerate as run
    value = run(branch['generation_report'], branch['diversity_report'],
                asset['bundle_root'], output, gpu=runtime['gpu'],
                render_python=runtime['render_python'],image_contract=runtime['image_contract'],
                gsplat_runtime_witness=runtime.get('gsplat_runtime_witness'))
    return value


def validate_sample(path, *, native_python=None, image_contract=IMAGE_CONTRACT):
    """Revalidate sealed native evidence; callback success alone is insufficient."""
    from driversi_agents.chainflow.synthetic import validate_admission
    sample = read_mapping(path)
    contract=CONTRACTS[image_contract]
    if (sample.get('kind') != 'chainflow_admitted_synthetic_sample'
            or sample.get('image_contract') != image_contract
            or sample.get('future_images_included') is not False
            or sample.get('image_counts') != dict(history=contract['history_images'], future=0)):
        raise ValueError('New history-only native package receipt required')
    proof = validate_admission(sample['row'], sample['files'])
    if proof.get('schema_version') != contract['proof_version'] or proof.get('image_contract') != image_contract:
        raise ValueError('Declared consumed-image native proof version required')
    from driversi.rendering.diagnostics.worldengine_admission import validate_admission as validate_visual
    visual = read_mapping(proof['visual_admission']['path'])
    checked=validate_visual(proof['visual_admission']['path'])
    if visual.get('visual_quality_profile') != contract['profile'] or checked.get('admitted') is not True:
        raise ValueError('Strict dynamic binding visual admission required')
    if native_python:
        code = ('import json,sys;from driversi_agents.chainflow.data import verify_records;'
                's=json.load(open(sys.argv[1]));verify_records([s["row"]])')
        subprocess.run([native_python,'-B','-c',code,str(path)],cwd=project_root(),
                       check=True,timeout=600,capture_output=True,text=True)
    return sample['row']['sample_id']


def update_budget(count, exposures, global_batch, ceiling):
    updates = math.ceil(count * exposures * 8 / global_batch)
    updates += updates % 2
    if not 2 <= updates <= ceiling:
        raise ValueError('Actual admitted population exceeds bounded native update budget')
    return updates


def train_pair(plan, samples, output):
    """Use the existing native compiler, engine, recovery and evaluation stages."""
    from functools import partial
    from driversi.rsi.experiments.chainflow_campaign import target_stop_guard
    from driversi.rsi.plugins.chainflow_synthetic import compile_paired_task
    from driversi_agents.chainflow.sampling import distributed_settings
    output.mkdir(parents=True, exist_ok=True)
    recipe = plan['training']; model = read_mapping(recipe['model_config'])
    settings = distributed_settings(model['settings'])
    if settings['synthetic_every'] != 8:
        raise ValueError('Replay treatment requires one synthetic per eight global samples')
    global_batch = recipe['batch_size'] * len(settings['devices'])
    updates = update_budget(len(samples), recipe['synthetic_exposures'], global_batch,
                            recipe['max_updates_per_arm'])
    frozen_json(output/'training-input.json', dict(sample_reports={p:digest(p) for p in samples},
                model_config_sha256=digest(recipe['model_config']), updates_per_arm=updates,
                requested_synthetic_exposures=recipe['synthetic_exposures'],
                global_batch_size=global_batch, synthetic_every=8, native_loss_unchanged=True))
    task_file = output/'paired/task.json'
    if not task_file.exists():
        compile_paired_task(recipe['base_manifest'], samples, recipe['model_config'], output/'paired',
            updates_per_arm=updates, epoch_training=True, batch_size=recipe['batch_size'],
            pre_evaluation_gate=True, training_population='mixed', target_token=recipe['target_token'])
    task = read_mapping(task_file)
    target = output/'tasks'/task['id']/'treatment_pre_evaluation/target-feedback.json'
    state = task_state(output/'harness.sqlite', task, task_file, output/'tasks', execute=True,
                       evidence_guard=partial(target_stop_guard, target))
    report = dict(status=state['status'], directory=state['directory'],
                  failure_reason=state.get('failure_reason'), optimizer_updates_per_arm=updates,
                  requested_synthetic_exposures=recipe['synthetic_exposures'],
                  model='chainflow_stage1', native_loss_unchanged=True,
                  target_feedback=str(target) if target.exists() else None,
                  promoted=False, full_navtest_required=12146,
                  sim_agent_steps=state.get('sim_agent_steps',0))
    if state.get('failure_reason') == 'local_target_still_zero':
        report.update(status='target_still_zero', reason='local_target_still_zero')
    elif state['status'] == 'completed':
        report['release'] = release_decision(Path(state['directory']), recipe['target_token'])
        report['promoted'] = report['release']['promoted']
    write_json(output/'report.json', report)
    return report


def release_decision(task, target):
    """The existing full-test decision applies; never activate a failed candidate."""
    from driversi.rsi.experiments.continual_curriculum import evaluation_from_native
    from driversi.rsi.decisions.continual_learning import decide, scores
    feedback = read_mapping(task/'paired_feedback/report.json')
    gate = read_mapping(task/'treatment_pre_evaluation/gate.json')
    local = read_mapping(task/'treatment_pre_evaluation/target-feedback.json')
    if (feedback.get('effective_recipe_verified') is not True
            or feedback.get('unconsumed_added_sample_ids') or not feedback.get('added_sample_ids')
            or gate.get('resume_exact') is not True or local.get('target_success') is not True):
        raise ValueError('Actual exposures, exact recovery and successful target required')
    values = {side:read_mapping(task/(side+'_evaluation/training-result.json'))
              for side in ('control','treatment')}
    expected = [r['sample_id'] for r in values['treatment']['paired_samples']]
    if len(expected) != 12146 or len(set(expected)) != 12146:
        raise ValueError('Full official navtest required for release')
    reports = {side:evaluation_from_native(value, expected) for side,value in values.items()}
    parent = evaluation_from_native(values['treatment'], expected, 'parent')
    if scores(parent, expected) != scores(evaluation_from_native(values['control'], expected, 'parent'),expected):
        raise ValueError('Parent evaluation parity differs')
    for side in ('control','treatment'):
        trained = read_mapping(task/(side+'_resume/training-result.json'))
        checkpoint = str(Path(trained['checkpoint']).resolve(strict=True))
        if (trained.get('native_loss_unchanged') is not True
                or trained['checkpoint_files'].get(checkpoint) != digest(checkpoint)
                or values[side].get('evaluated_checkpoint') != checkpoint):
            raise ValueError('Native checkpoint or loss provenance differs')
        if side == 'treatment' and (local['checkpoint'] != checkpoint
                or local['checkpoint_sha256'] != digest(checkpoint) or local['target_token'] != target):
            raise ValueError('Target evaluation belongs to another candidate')
    verdict = decide(parent,reports['treatment'],reports['control'],expected=expected,target=target,
                     retained=[],native_paired_benefit=feedback.get('observed_pilot_benefit'),
                     mastery=0,strict_target=True)
    verdict['activation_scope'] = 'campaign_incumbent_only'
    frozen_json(task/'campaign-release.json', verdict)
    # This bounded experiment owns only its own pointer, never global deployment.
    checkpoint = read_mapping(task/'treatment_resume/training-result.json')['checkpoint']
    if verdict['promoted']:
        frozen_json(task/'campaign-incumbent.json',dict(checkpoint=checkpoint,sha256=digest(checkpoint),
                    decision=str(task/'campaign-release.json')))
    return verdict


def activate_dataset(root, campaign, *, image_contract=IMAGE_CONTRACT):
    from driversi.data.datasets.simscale_collection import activate_history_only
    root,campaign=Path(root),Path(campaign)
    previous=root/'postprocessing.json'
    archive=campaign/'prior-postprocessing.json'
    if previous.exists() and not archive.exists():
        frozen_json(archive,read_mapping(previous))
    if image_contract=='anchor_rgb_history_future_states_v1':
        from driversi.data.datasets.simscale_collection import activate_anchor_only
        activate_anchor_only(root,str(campaign))
    else:activate_history_only(root,str(campaign))
    write_json(previous,dict(schema_version=1,mode='training_only',campaign=str(campaign),
        features='deferred',scene_caption='deferred',classification='deferred',
        training_admission='unchanged'))


def run(plan_path, output, *, asset_runner=None, branch_runner=None, pair_runner=None,
        sample_validator=None, dataset_activator=None):
    plan = validate_plan(read_mapping(plan_path)); output = Path(output).resolve()
    if not output.is_relative_to(project_root().resolve()/'exp'):
        raise ValueError('Campaign artifacts belong under project exp')
    asset_runner, branch_runner = asset_runner or reconstruct, branch_runner or regenerate
    pair_runner = pair_runner or train_pair
    if sample_validator is None:
        native_python=read_mapping(plan['training']['model_config'])['python']
        sample_validator=lambda path:validate_sample(path,native_python=native_python,
                                                     image_contract=plan['gates']['image_contract'])
    with _lock(output):
        frozen_json(output/'plan.json', plan)
        evidence = input_files(plan); frozen_json(output/'input-files.json', evidence)
        _verify_files(evidence)
        final = output/'report.json'
        if final.exists() and read_mapping(final)['status'] not in ('blocked_unknown','blocked_training'):
            return read_mapping(final)
        start = output/'started.json'
        frozen_json(start, read_mapping(start) if start.exists() else dict(started_at=time.time()))
        summary = dict(status='running', model='chainflow_stage1', native_loss_unchanged=True,
                       sources=[], samples=[], physical_rollouts_reused=0,new_teacher_rollouts=0,
                       admitted_samples=0, training_started=False,promoted=False,
                       image_contract=plan['gates']['image_contract'],future_rgb_generated=0)
        def finish(status, reason=None):
            summary.update(status=status,reason=reason,wall_seconds=time.time()-read_mapping(start)['started_at'])
            summary.setdefault('sim_agent_steps',40*summary['admitted_samples'])
            _verify_files(evidence)
            write_json(final, summary)
            return summary
        def progress():
            summary['wall_seconds']=time.time()-read_mapping(start)['started_at']
            write_json(output/'progress.json',summary)
        samples, ids, contributing = [],set(),set()
        assets, rows, unknown = {},{},[]
        asset_locks={gpu:Lock() for gpu in plan['runtime']['asset_gpus']}
        def asset_job(source,gpu):
            actual_gpu=read_mapping(source['prebuilt_asset_job'])['gpu'] if source.get('prebuilt_asset_job') else gpu
            if actual_gpu not in asset_locks:raise ValueError('Existing asset GPU is not allocated')
            with asset_locks[actual_gpu]:
                return _stage(output/'sources'/source['id']/'asset',
                    lambda dest:asset_runner(source,dict(plan['runtime'],gpu=actual_gpu),dest),
                    recover=lambda:reconcile_asset(source,plan['runtime'],
                        output/'sources'/source['id']/'asset/execution'))
        for source,asset in _parallel_jobs(plan['sources'],plan['runtime']['asset_gpus'],asset_job):
            row = dict(id=source['id'],asset_status=asset['status'],branches=[])
            rows[source['id']]=row;summary['sources'].append(row)
            if asset['status']=='unknown':
                unknown.append(asset.get('reason'))
            elif asset['status']=='completed':assets[source['id']]=asset
            else:row['reason']=asset.get('reason','asset_not_completed')
            progress()
        if unknown:return finish('blocked_unknown',unknown[0])
        if dataset_activator is None:
            dataset_activator=lambda root,campaign:activate_dataset(root,campaign,
                image_contract=plan['gates']['image_contract'])
        dataset_activator(project_root()/'dataset',str(output))
        jobs=[dict(source_id=s['id'],branch=b) for s in plan['sources'] if s['id'] in assets
              for b in s['branches']]
        def branch_job(job,gpu):
            branch=job['branch'];asset=assets[job['source_id']]
            return _stage(output/'branches'/branch['id'],lambda dest:branch_runner(
                branch,asset,dict(plan['runtime'],gpu=gpu,
                                 image_contract=plan['gates']['image_contract']),dest))
        for job,result in _parallel_jobs(jobs,plan['runtime']['render_gpus'],branch_job):
            source_id,branch=job['source_id'],job['branch']
            summary['physical_rollouts_reused']+=1
            rows[source_id]['branches'].append(dict(id=branch['id'],status=result['status'],reason=result.get('reason')))
            if result['status']=='unknown':unknown.append(result.get('reason'))
            elif result['status']=='training_sample_ready':
                path=str(Path(result['sample_report']).resolve(strict=True))
                identity=sample_validator(path)
                if identity in ids:
                    raise ValueError('Duplicate admitted package cannot fill cohort quota')
                ids.add(identity); samples.append(path);contributing.add(source_id)
                summary['samples'].append(dict(sample_id=identity,sample_report=path,
                                              source_id=source_id,branch_id=branch['id']))
                summary['admitted_samples']=len(samples)
            progress()
        if unknown:return finish('blocked_unknown',unknown[0])
        # Completion order must not alter the sampling permutation on restart.
        summary['samples'].sort(key=lambda row:(row['source_id'],row['branch_id']))
        samples=[row['sample_report'] for row in summary['samples']]
        if (len(samples)<plan['limits']['minimum_samples']
                or len(contributing)<plan['limits']['minimum_sources']):
            return finish('insufficient_admitted_data','bounded_batch_admission_floor_not_met')
        summary['training_started']=True;progress()
        training = _stage(output/'training',lambda dest:pair_runner(plan,samples,dest),
                          recover=lambda:pair_runner(plan,samples,output/'training/execution'))
        summary['training']=training;summary['promoted']=training.get('promoted',False)
        summary['sim_agent_steps']=training.get('sim_agent_steps',0)+40*len(samples)
        return finish('completed' if training['status']=='completed' else
                      'target_still_zero' if training['status']=='target_still_zero' else 'blocked_training',
                      training.get('reason') or training.get('failure_reason'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();plan=validate_plan(read_mapping(args.plan));root=Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    # The public entrypoint registers a real Harness task; nested native training
    # retains its own typed evidence and restart semantics.
    duration=len(plan['sources'])*plan['runtime']['asset_timeout_seconds']+200000
    task=dict(schema_version=2,id=plan['id'],label='Dynamic actors and history-only replay comparison',
        budget=dict(max_wall_seconds=duration+60,max_sim_agent_steps=7*12146*160+12*80),
        steps=[dict(id='campaign',provider='dynamic_history_replay_campaign',inputs={},
                    config=dict(plan=str(Path(args.plan).resolve())),max_wall_seconds=duration)])
    state=task_state(root/'harness.sqlite',task,root/'task.json',root/'tasks',execute=True)
    print(json.dumps(dict(status=state['status'],directory=state['directory'],
                         failure_reason=state.get('failure_reason')),ensure_ascii=False))


if __name__=='__main__':
    main()
