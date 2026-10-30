"""Apply an explicit operator protocol revision after known settled CPU work.

Old declarations/receipts are immutable. This is a bounded migration, not an
automatic retry for unknown GPU work, and it preserves cumulative attempts.
"""
import argparse
import copy
from pathlib import Path

from driversi.agents.backends.native import digest
from driversi.configuration import read_mapping
from driversi.runtime.budget import write_json
from .continuation import frozen_json

PROFILE='experimental_numeric25_spatial05_v1'
REVISION='source25-spatial-speed-v1'
STOP_TEXT='Pause at the next settled phase boundary for versioned bounded-grid prefix correction; retain in-flight work and all receipts.'


def verify_settled(round_path, pid):
    proc=Path('/proc')/str(pid)/'cmdline'
    if proc.exists() and proc.read_bytes():raise ValueError('Prior executor is still live')
    wave=Path(round_path)/'waves/wave-0001'
    status=read_mapping(wave/'production-status.json')
    prefetch=read_mapping(wave/'physical-prefetch-status.json')
    if (status.get('status')!='failed' or status.get('error_type')!='FileNotFoundError'
            or prefetch.get('status')!='settled_after_error'
            or prefetch.get('errors') or prefetch.get('unconsumed')
            or len({prefetch[k] for k in ('submitted','completed','consumed')})!=1):
        raise ValueError('Only the proven pre-render IO failure may be reconciled')
    if (wave/'resident-batches').exists() or list(wave.glob('candidates/*/generation/render-reservation.json')):
        raise ValueError('Rendering was reserved; explicit side-effect reconciliation required')
    plan_path=wave/'trajectory-family-plan.json'
    plan=read_mapping(plan_path);assigned={a['id']:a for a in plan['assignments']}
    jobs=prefetch.get('jobs',{})
    started={p.parent.name:p.resolve() for p in wave.glob('candidates/*/generation') if p.is_dir()}
    if set(started)!=set(jobs) or not set(jobs)<=set(assigned) or len(jobs)!=prefetch['completed']:
        raise ValueError('Unaccounted started task or missing prefetch identity')
    rows=[]
    for key,job in jobs.items():
        output=(wave/'candidates'/key/'generation').resolve()
        if (not job.get('done') or not job.get('consumed') or Path(job['output']).resolve()!=output
                or started[key]!=output or not (output/'report.json').exists()):
            raise ValueError('Every specific physical result must be durably settled')
        report=read_mapping(output/'report.json');teacher=report.get('teacher',{})
        if (report.get('token')!=assigned[key]['token'] or report.get('source_log')!=assigned[key]['source_log']
                or teacher.get('assignment_id')!=key or teacher.get('plan_sha256')!=digest(plan_path)
                or teacher.get('status') not in ('trajectory_admitted','execution_or_admission_rejected')):
            raise ValueError('Physical identity or terminal teacher state differs')
        rows.append(output/'report.json')
    if any(read_mapping(p)['status'] not in ('trajectory_ready','teacher_rejected') for p in rows):
        raise ValueError('Every physical result must be durably settled')
    return {str(p):digest(p) for p in [plan_path,wave/'production-status.json',wave/'physical-prefetch-status.json',*rows]}


def migrate(root, request):
    from . import continual_curriculum as cc
    from .generation_policy import started_attempts
    root=Path(root);request=read_mapping(request)
    if (request.get('revision')!=REVISION or request.get('source_psnr_strict_gt')!=25.
            or request.get('spatial_path_ade_m')!=.5 or not request.get('operator_authorized')):
        raise ValueError('Explicit operator 25dB/spatial/speed policy required')
    folder=root/'protocol-revisions'/REVISION
    done=folder/'completed.json'
    if done.exists():return read_mapping(done)
    folder.mkdir(parents=True,exist_ok=True)
    frozen_json(folder/'request.json',request)
    snapshot=folder/'before.json'
    if snapshot.exists():before=read_mapping(snapshot)
    else:
        state=read_mapping(cc.state_path(root))
        case=next(c for c in state['cases'] if c['id']==state['active'])
        if case['status']!='unknown_execution' or case['current_round']!=request['previous_round']:
            raise ValueError('Current state differs from the reviewed settled failure')
        proof=verify_settled(case['current_round'],request['prior_executor_pid'])
        before=dict(state=state,primary=read_mapping(root/'primary-generation-policy.json'),
            recovery=read_mapping(root/'asset-recovery-policy.json'),evidence=proof)
        frozen_json(snapshot,before)
    for p,sha in before['evidence'].items():
        if digest(p)!=sha:raise ValueError('Settled evidence changed during protocol migration')
    state=copy.deepcopy(before['state'])
    case=next(c for c in state['cases'] if c['id']==state['active'])
    charged=started_attempts(root,case['id'])
    cap=state['policy']['attempts_per_source']*state['policy']['max_rounds']
    if any(n>cap for n in charged.values()):raise ValueError('Prior source budget exceeded')
    state['policy'].update(visual_quality_profile=PROFILE,strict_source_psnr_gt=25.)
    state['protocol_revision']=REVISION
    case.update(status='needs_revision',asset_revision=case.get('asset_revision',0)+1,
        qualified=0,protocol_revision=REVISION,charged_attempts=charged)
    current=read_mapping(cc.state_path(root))
    prepared_path=folder/'prepared-state.json'
    prepared=read_mapping(prepared_path) if prepared_path.exists() else None
    if current not in (before['state'],state,prepared):
        raise ValueError('Concurrent curriculum update; do not overwrite or prepare it')
    primary=copy.deepcopy(before['primary'])
    primary.update(kind='spatial_balanced_history_native_tts_resident_v3',
        training_logs_root=before['recovery']['factory']['logs_root'])
    primary.pop('quality_and_novelty_gates_unchanged',None)
    recovery=copy.deepcopy(before['recovery'])
    recovery['strict_source_psnr_gt']=25.
    recovery['factory']['minimum_source_psnr_db']=25.
    for filename,old,value in [('primary-generation-policy.json',before['primary'],primary),
                                ('asset-recovery-policy.json',before['recovery'],recovery)]:
        actual=read_mapping(root/filename)
        if actual not in (old,value):raise ValueError('Concurrent policy update')
        write_json(root/filename,value)
    if current==before['state']:
        cc.save(root,state)
    current=read_mapping(cc.state_path(root))
    case=next(c for c in current['cases'] if c['id']==current['active'])
    if case['status']=='needs_revision':
        cc.prepare(root)
        frozen_json(prepared_path,read_mapping(cc.state_path(root)))
    current=read_mapping(cc.state_path(root))
    case=next(c for c in current['cases'] if c['id']==current['active'])
    result=dict(revision=REVISION,previous_round=request['previous_round'],new_round=case['current_round'],
        charged_attempts=charged,cumulative_cap_per_source=cap,old_receipts_unchanged=True,
        old_samples_auto_imported=False,training_started=False,producer_sha256=digest(__file__),
        spatial_and_speed_protocol=primary['kind'],source_psnr_strict_gt=25.,
        evidence=str(snapshot))
    frozen_json(done,result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True);parser.add_argument('--request',required=True)
    args=parser.parse_args()
    from .continual_curriculum import locked
    with locked(args.root):print(migrate(args.root,args.request))


if __name__=='__main__':main()
