"""Resume settled production after an explicit retrieval-processing deferral."""
import argparse
import json
import os
import time
from pathlib import Path

from driversi.agents.backends.native import digest, project_root
from driversi.configuration import read_mapping
from driversi.runtime.budget import write_json
from .continuation import frozen_json


def known_arrival_rejection(result, teacher, native):
    """Only observed terminal physical rejection; never an arbitrary failed job."""
    events=native.get('arrival_events',{})
    physical_event=any(isinstance(events.get(k),list) and any(x is True for x in events[k])
                       for k in ('collision','offroad','goal','route_support_end'))
    return (result.get('status')=='teacher_rejected'
        and teacher.get('status')=='execution_or_admission_rejected'
        and teacher.get('harness_status')=='failed'
        and native.get('status')=='rejected' and native.get('reason')=='arrival_ego_terminated'
        and not native.get('error') and not native.get('error_type')
        and bool(native.get('input_files')) and physical_event)


def settled_generation(wave, *, allow_prior_completed=False):
    """An interrupted future may have a complete native receipt; verify it."""
    from driversi.rsi.plugins.synthesis import check_files
    wave = Path(wave).resolve()
    prefetch = read_mapping(wave/'physical-prefetch-status.json')
    if prefetch['submitted'] != prefetch['completed'] or any(not j['done'] for j in prefetch['jobs'].values()):
        raise ValueError('Physical jobs are still in flight')
    if any(value != 'KeyboardInterrupt' for value in prefetch.get('errors',{}).values()):
        raise ValueError('Unresolved physical execution error')
    plan_path=wave/'trajectory-family-plan.json';plan=read_mapping(plan_path)
    assignments={a['id']:a for a in plan['assignments']}
    started={p.parent.name for p in (wave/'candidates').glob('*/generation')}
    if (not set(prefetch['jobs'])<=started or not started<=set(assignments)
            or (not allow_prior_completed and started!=set(prefetch['jobs']))):
        raise ValueError('Started physical jobs differ from the settlement inventory')
    files={str(p):digest(p) for p in (plan_path,wave/'physical-prefetch-status.json')}
    rejected_reconciliations = {}
    for key in sorted(started):
        folder=wave/'candidates'/key/'generation';path=folder/'report.json'
        job=prefetch['jobs'].get(key,dict(output=str(folder)))
        if Path(job['output']).resolve()!=folder:
            raise ValueError('Physical job output changed')
        result=read_mapping(path);teacher=result.get('teacher',{});a=assignments[key]
        native_path=Path(teacher['teacher_report']).resolve(strict=True)
        if not native_path.is_relative_to(folder):raise ValueError('Foreign native result')
        native=read_mapping(native_path)
        early_rejected=known_arrival_rejection(result,teacher,native) if allow_prior_completed else False
        future_rejected=False
        if allow_prior_completed and result.get('status')=='teacher_rejected':
            from driversi.rsi.decisions.generation_outcomes import incomplete_physical_branch
            admission_receipt=native_path.parent.parent/'admission.receipt.json'
            teacher_receipt=native_path.parent.parent/'teacher.receipt.json'
            if (incomplete_physical_branch(native) and admission_receipt.exists()
                    and teacher_receipt.exists() and teacher.get('harness_status') in ('failed','outcome_unknown')):
                receipt=read_mapping(admission_receipt);published=read_mapping(teacher_receipt)
                legacy=(receipt.get('phase')=='failed' and receipt.get('failed_phase')=='executing'
                    and receipt.get('error_type')=='ValueError'
                    and receipt.get('reason')=='Incomplete branch cannot supply a complete RGB window')
                structured=False
                selection_path=admission_receipt.with_name('admission.result.json')
                if receipt.get('phase')=='published' and selection_path.exists():
                    selection=read_mapping(selection_path)
                    structured=(selection.get('status')=='failed'
                        and selection.get('reason')=='incomplete_physical_branch'
                        and selection.get('summary',{}).get('native_report_sha256')==digest(native_path))
                    if structured:files[str(selection_path)]=digest(selection_path)
                future_rejected=published.get('phase')=='published' and (legacy or structured)
                if future_rejected:
                    files.update({str(admission_receipt):digest(admission_receipt),str(teacher_receipt):digest(teacher_receipt)})
                    rejected_reconciliations[key]='explicit_future_physical_termination_not_training_data'
        if (result.get('status') not in ('teacher_rejected','trajectory_ready','rendered')
                or (teacher.get('harness_status') != 'completed' and not (early_rejected or future_rejected))
                or teacher.get('status') not in ('trajectory_admitted','execution_or_admission_rejected')
                or teacher.get('plan_sha256')!=digest(plan_path)
                or teacher.get('assignment_id')!=key
                or teacher.get('token')!=a['token'] or teacher.get('source_log')!=a['source_log']):
            raise ValueError('Physical outcome is unknown or belongs to a different assignment')
        native_path=Path(teacher['teacher_report']).resolve(strict=True)
        if not native_path.is_relative_to(folder):raise ValueError('Foreign native result')
        native=read_mapping(native_path)
        if native.get('status') not in ('completed','rejected'):raise ValueError('Unknown native execution')
        check_files(native['input_files'] if early_rejected else native['source_files'])
        for name in ('source_files','files'):
            if name in native:check_files(native[name])
        files.update({str(path):digest(path),str(native_path):digest(native_path)})
        physical=folder.parent/'physical-prefetch.json'
        if read_mapping(physical).get('status')!='completed':
            raise ValueError('Physical worker has no terminal completion receipt')
        files[str(physical)]=digest(physical)
    for path in (wave/'candidates').glob('*/outcome.json'):
        seal=path.with_name('outcome-seal.json')
        if not seal.exists() or read_mapping(seal)['sha256']!=digest(path):
            raise ValueError('Unsettled candidate outcome')
        result=read_mapping(path)
        check_files(result.get('files',{}));check_files(result.get('trajectory_files',{}))
        files[str(path)]=digest(path);files[str(seal)]=digest(seal)
    return dict(started=len(started),known_results=len(started),files=files,
                known_physical_rejections_reconciled=rejected_reconciliations,
                prior_completed_reconciled=sorted(started-set(prefetch['jobs'])),
                interrupted_waits_reconciled=list(prefetch.get('errors',{})),
                completed_work_reexecuted=False)


def run(root, stop_request, output):
    from . import continual_curriculum as cc
    from .continual_execution import gpu_snapshot, required_gpus, run as automatic
    from .reactive_autoloop import run as production
    from driversi.data.datasets.simscale_collection import active

    root, stop_request, out = Path(root).resolve(), Path(stop_request).resolve(), Path(output).resolve()
    if not root.is_relative_to(project_root()/'exp') or not out.is_relative_to(root):
        raise ValueError('Resume evidence belongs to the active experiment')
    with cc.locked(root):
        if (out/'launch-start.json').exists():raise ValueError('This authorized resume has already started')
        stop=read_mapping(stop_request)
        if (stop.get('reason')!='user_requested_no_new_features_or_captions_training_only'
                or stop.get('signal')!='SIGINT_parent_only_drain_existing_prefetch'):
            raise ValueError('Explicit processing-change stop receipt required')
        proc=Path('/proc')/str(stop['pid'])/'cmdline'
        if proc.exists() and proc.read_bytes():raise ValueError('Previous executor has not exited')
        # Exclude this dispatcher; any remaining owner of this campaign blocks recovery.
        for p in Path('/proc').glob('[0-9]*/cmdline'):
            if int(p.parent.name)==os.getpid():continue
            try:cmd=p.read_bytes().replace(b'\0',b' ').decode(errors='replace')
            except (FileNotFoundError,PermissionError):continue
            if str(root) in cmd and any(x in cmd for x in ('driversi.','navsim_teacher','resident_postprocess')):
                raise ValueError('Another campaign process still owns work')
        state=read_mapping(cc.state_path(root));case=next(c for c in state['cases'] if c['id']==state['active'])
        if case['status']!='unknown_execution':raise ValueError('Only the explicit interrupted production state can resume')
        folder=Path(case['current_round']);wave=folder/'waves/wave-0001'
        status=read_mapping(wave/'production-status.json')
        if status.get('status')!='failed' or status.get('error_type')!='KeyboardInterrupt':
            raise ValueError('This is not a settled operator interruption')
        settlement=settled_generation(wave)
        collection=active(project_root()/'dataset')
        if collection is None or not root.is_relative_to(Path(collection['campaign'])):
            raise ValueError('Wrong active dataset campaign')
        policy=dict(schema_version=1,mode='training_only',campaign=collection['campaign'],
                    features='deferred',scene_caption='deferred',classification='deferred',training_admission='unchanged')
        # Preserve prior derived state before resuming; native receipts are never rewritten.
        out.mkdir(parents=True,exist_ok=True)
        frozen_json(out/'state-before.json',state)
        frozen_json(out/'production-before.json',status)
        frozen_json(out/'prefetch-before.json',read_mapping(wave/'physical-prefetch-status.json'))
        frozen_json(out/'summary-before.json',read_mapping(wave/'trajectory-family-summary.json'))
        frozen_json(out/'settlement.json',settlement)
        frozen_json(out/'physical-execution-before.json',{
            p:read_mapping(p) for p in settlement['files'] if p.endswith('/physical-prefetch.json')})
        frozen_json(project_root()/'dataset/postprocessing.json',policy)
        wanted=required_gpus(case,'dispatch');available=gpu_snapshot()
        if any(available.get(g,100000)>=500 for g in wanted):raise ValueError('Rendering GPUs are not free')
        marker=root/'STOP_AUTOMATIC'
        if marker.exists():
            if marker.read_text().strip()!='owned_endpoint_tto_deployment_wait_for_current_batch':
                raise ValueError('Unrelated stop request must not be removed')
            (out/'prior-stop-marker.txt').write_bytes(marker.read_bytes());marker.unlink()
        frozen_json(out/'launch-start.json',dict(pid=os.getpid(),time=time.time(),
            stop_request=str(stop_request),stop_sha256=digest(stop_request),
            settlement_sha256=digest(out/'settlement.json'),required_gpus=wanted,
            preserved_qualified=read_mapping(wave/'trajectory-family-summary.json')['qualified'],
            recurring_monitor=False,postprocessing=policy))
        case['status']='dispatching';cc.save(root,state)
        try:production(folder/'generation-config.json','production')
        except BaseException:
            case['status']='unknown_execution';cc.save(root,state);raise
        result=read_mapping(wave/'production-status.json')
        case['qualified']=result.get('qualified_samples',0)
        case['status']=('data_ready' if result['status']=='generation_wave_completed' else
                        'blocked_assets' if result['status']=='production_budget_blocked' else 'generation_stopped')
        state['generation_started']=True;cc.save(root,state)
        write_json(out/'production-complete.json',dict(status=result['status'],qualified=case['qualified'],
                   elapsed_seconds=time.time()-read_mapping(out/'launch-start.json')['time']))
        return automatic(root,max_problems=3,max_transitions=18)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--stop-request',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.campaign,a.stop_request,a.output)


if __name__=='__main__':main()
