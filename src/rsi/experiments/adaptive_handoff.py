"""One settled-boundary handoff into the existing automatic Harness executor."""
import argparse
import os
import time
from pathlib import Path

from driversi.agents.backends.native import digest,project_root
from driversi.configuration import read_mapping
from driversi.runtime.budget import write_json
from .continuation import frozen_json

STOP_TEXT='Pause only after the current native batch settles for operator-authorized adaptive policy deployment; preserve every receipt and in-flight job.'


def run(root, prior_pid, *, resume=False):
    from . import continual_curriculum as cc
    from .processing_resume import settled_generation
    from .continual_execution import run as automatic
    root=Path(root).resolve();out=root/'protocol-revisions/adaptive-diversity-v1/handoff'
    if not root.is_relative_to(project_root()/'exp'):raise ValueError('Owned experiment required')
    configured_context = read_mapping(root/'primary-generation-policy.json').get('adaptive_context', {})
    if not all(configured_context.get(k) for k in ('private_config', 'credential_file')):
        raise ValueError('Configure adaptive_context private_config and credential_file before handoff')
    out.mkdir(parents=True,exist_ok=True)
    if resume:
        original=read_mapping(out/'started.json');status=read_mapping(out/'status.json')
        proc=Path('/proc')/str(original['pid'])/'cmdline'
        if (status.get('status')!='needs_reconciliation' or original['prior_pid']!=prior_pid
                or (proc.exists() and proc.read_bytes()) or (out/'installed.json').exists()
                or (out/'resume-started.json').exists()):
            raise ValueError('Only one reconciled pre-install handoff resume is allowed')
        frozen_json(out/'pre-resume-status.json',status)
        frozen_json(out/'resume-started.json',dict(pid=os.getpid(),prior_pid=prior_pid,time=time.time(),
            original_start_sha256=digest(out/'started.json'),physical_or_api_retry=False))
    else:
        if (out/'started.json').exists():raise ValueError('Handoff already started; reconcile before another launch')
        frozen_json(out/'started.json',dict(pid=os.getpid(),prior_pid=prior_pid,started=time.time(),recurring_monitor=False))
    proc=Path('/proc')/str(prior_pid)/'cmdline';start=time.monotonic()
    while proc.exists() and proc.read_bytes():
        cmd=proc.read_bytes().replace(b'\0',b' ').decode(errors='replace')
        if str(root) not in cmd or 'driversi.rsi.experiments.' not in cmd:
            raise ValueError('Prior PID identity changed')
        if time.monotonic()-start>14400:raise TimeoutError('Prior batch did not settle in four hours')
        write_json(out/'status.json',dict(status='waiting_existing_batch_to_settle',pid=os.getpid(),updated=time.time()))
        time.sleep(10)
    with cc.locked(root):
        state=read_mapping(cc.state_path(root))
        case=next((x for x in state['cases'] if x['id']==state['active']),None)
        if case is None or case['status'] not in ('generation_stopped','data_ready','awaiting_release'):
            raise ValueError('No settled active problem for policy handoff')
        if case['status'] in ('generation_stopped','data_ready'):
            wave=Path(case['current_round'])/'waves/wave-0001'
            if read_mapping(wave/'physical-prefetch-status.json').get('errors'):
                raise ValueError('Prefetch errors require reconciliation')
            frozen_json(out/'settlement.json',settled_generation(wave,allow_prior_completed=True))
        marker=root/'STOP_AUTOMATIC'
        if not marker.exists() or marker.read_text()!=STOP_TEXT:raise ValueError('Unrelated stop request')
        primary=read_mapping(root/'primary-generation-policy.json')
        recovery=read_mapping(root/'generation-recovery-policy.json')
        context = primary.get('adaptive_context', {})
        if not all(context.get(k) for k in ('private_config', 'credential_file')):
            raise ValueError('Configure adaptive_context private_config and credential_file before handoff')
        frozen_json(out/'before.json',dict(state=state,primary=primary,recovery=recovery))
        primary.update(kind='adaptive_history_native_tts_resident_v1',adaptive_context=dict(
            context, advice_root=str(root/'adaptive-advice')))
        recovery.update(mode='adaptive_source_v1',max_revisions_per_learning_round=2)
        state['policy'].update(visual_quality_profile='experimental_numeric25_adaptive_v1',strict_source_psnr_gt=25.)
        state['protocol_revision']='source25-adaptive-diversity-v1'
        write_json(root/'primary-generation-policy.json',primary)
        write_json(root/'generation-recovery-policy.json',recovery)
        cc.save(root,state)
        frozen_json(out/'installed.json',dict(time=time.time(),producer=digest(__file__),
            changed=['diversity_policy','proposal_strategy','generation_recovery'],
            pdms_strict_gt=.95,source_psnr_strict_gt=25.,native_model_loss_unchanged=True))
        (out/'boundary-stop.txt').write_bytes(marker.read_bytes());marker.unlink()
        write_json(out/'status.json',dict(status='running_existing_harness',pid=os.getpid(),updated=time.time()))
        result=automatic(root,max_problems=3,max_transitions=18)
        write_json(out/'result.json',result)
        write_json(out/'status.json',dict(status='harness_returned',pid=os.getpid(),updated=time.time(),
            automatic_status=result.get('automatic_execution',{}).get('status')))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True);parser.add_argument('--prior-pid',type=int,required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    try:run(args.root,args.prior_pid,resume=args.resume)
    except Exception as exc:
        out=Path(args.root)/'protocol-revisions/adaptive-diversity-v1/handoff'
        write_json(out/'status.json',dict(status='needs_reconciliation',error_type=type(exc).__name__,updated=time.time()))
        raise


if __name__=='__main__':main()
