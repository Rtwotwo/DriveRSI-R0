"""Execute finite GS recovery recipes and register verified source windows.

This is an experiment service, not a monitor or an admission bypass. A source
certificate only permits trying new branches under the unchanged native gates.
"""
import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path

from driversi.agents.backends.native import digest, project_root
from driversi.configuration import read_mapping
from driversi.rsi.experiments.continuation import frozen_json
from driversi.runtime.budget import write_json


def policy(root):
    path=Path(root)/'asset-recovery-policy.json'
    if not path.exists():return None
    result=read_mapping(path)
    if result.get('schema_version')!=1 or result.get('strict_source_psnr_gt') not in (25.,27.):
        raise ValueError('Explicit strict25 or strict27 recovery policy required')
    if result.get('selection')!='available_within_frozen_visual_pool':
        raise ValueError('Recovery cannot introduce geographic selection')
    return result


def current_recovery_folder(folder):
    folder=Path(folder)
    if folder.parent.name=='remediations':folder=folder.parent.parent
    for name in ('depth-receipt-io-1','depth-receipt-1','depth-runtime-1','colmap-rigs-1','map-vis-1'):
        revised=folder/'remediations'/name
        if (revised/'launch.json').exists():return revised
    return folder


def ready_sources(root):
    p=policy(root)
    if not p:return {}
    result={}
    for name in p.get('existing_source_reports',[]):
        value=read_mapping(name);asset=value['asset']
        verified=verify_source(name,token=asset['token'],source_log=asset['source_log'],threshold=p['strict_source_psnr_gt'])
        if verified:result[asset['token']]=verified
    names=[*p.get('jobs',{}).values(),*sorted((Path(root)/'asset-recovery/sources').glob('*/recovery-job.json'))]
    for name in names:
        job=read_mapping(name);rp=current_recovery_folder(job['output'])/'report.json'
        if not rp.exists():continue
        receipt=read_mapping(rp)
        if receipt['status']=='source_ready':
            verified=verify_source(receipt['source_report'],token=job['token'],source_log=job['source_log'],threshold=p['strict_source_psnr_gt'])
            if not verified:raise ValueError('Registered recovery quality no longer passes')
            result[job['token']]=verified
    return result


def recover_current(root):
    """Dispatch declared recipes once, then let another ready problem progress."""
    from . import continual_curriculum as cc
    p=policy(root)
    if not p:raise ValueError('Register the recovery backend before automatic recovery')
    state=read_mapping(cc.state_path(root));case=next(c for c in state['cases'] if c['id']==state['active'])
    if case['status']!='blocked_assets':raise ValueError('Asset recovery requires an asset-blocked problem')
    frozen=read_mapping(Path(case['current_round'])/'round.json')
    ready=ready_sources(root);launches=[]
    from .continual_execution import gpu_snapshot
    observed=gpu_snapshot()
    occupied=set()
    for launch_path in (Path(root)/'asset-recovery/sources').glob('*/recovery/launch.json'):
        current_folder=current_recovery_folder(launch_path.parent)
        receipt=read_mapping(current_folder/'launch.json')
        terminal=current_folder/'report.json'
        if not terminal.exists() or read_mapping(terminal)['status']=='recovery_unknown':
            occupied.add(receipt['gpu'])
    slots=[gpu for gpu in p.get('builder_gpus',[]) if observed.get(gpu,999999)<500 and gpu not in occupied]
    for token in frozen['selected_training_tokens']:
        if token in ready:continue
        name=p.get('jobs',{}).get(token)
        existing=Path(root)/'asset-recovery/sources'/token/'recovery-job.json'
        if not name and existing.exists():name=str(existing)
        if not name and p.get('factory'):
            from driversi.rsi.services.reconstruction import prepare_recovery
            gpu=slots[0] if slots else None
            if gpu is not None:
                candidate=next(c for c in case['candidates'] if c['token']==token)
                try:
                    name=prepare_recovery(candidate,p['factory'],Path(root)/'asset-recovery/sources'/token,gpu)
                except (ValueError,FileNotFoundError) as exc:
                    launches.append(dict(token=token,status='source_preparation_failed',reason=str(exc)));continue
        if not name:
            launches.append(dict(token=token,status='deferred_resources' if not slots else 'source_preparation_incomplete'));continue
        job=read_mapping(name)
        if job['token']!=token or job['source_log'] not in {c['source_log'] for c in case['candidates'] if c['token']==token}:
            raise ValueError('Recovery recipe differs from selected training source')
        folder=Path(job['output']);folder.mkdir(parents=True,exist_ok=True)
        launch=folder/'launch.json'
        if launch.exists():
            from driversi.rsi.services.reconstruction_recovery import eligible, launch as repair_known_io
            terminal=folder/'report.json'
            if (terminal.exists() and read_mapping(terminal)['status']=='recovery_exhausted'
                    and job.get('reconstruction_job') and eligible(job)):
                if job['gpu'] in occupied or observed.get(job['gpu'],999999)>=500:
                    launches.append(dict(token=token,status='deferred_resources'));continue
                revised=repair_known_io(name,p['controller_python'])
                occupied.add(job['gpu']);slots=[g for g in slots if g!=job['gpu']]
                launches.append(dict(token=token,status='dispatched',receipt=str(revised)));continue
            folder=current_recovery_folder(folder);launch=folder/'launch.json'
            receipt=read_mapping(launch)
            if receipt['job_sha256']!=digest(name):raise ValueError('Launched recovery recipe changed')
            # Never relaunch a job with an unknown outcome, even if its PID exited.
            terminal=folder/'report.json'
            outcome=read_mapping(terminal)['status'] if terminal.exists() else 'running_or_unknown'
            launches.append(dict(token=token,status=outcome,receipt=str(launch)));continue
        if job['gpu'] in occupied or observed.get(job['gpu'],999999)>=500:
            launches.append(dict(token=token,status='deferred_resources'));continue
        cmd=[p['controller_python'],'-u','-m','driversi.rsi.experiments.asset_recovery','--job',name]
        with (folder/'service.log').open('ab') as log:
            child=subprocess.Popen(cmd,cwd=project_root(),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        frozen_json(launch,dict(pid=child.pid,command=cmd,job_sha256=digest(name),gpu=job['gpu']))
        occupied.add(job['gpu'])
        slots=[g for g in slots if g!=job['gpu']]
        launches.append(dict(token=token,status='dispatched',pid=child.pid,receipt=str(launch)))
    case['asset_recovery']=launches
    case['status']=('asset_recovery_pending' if any(x['status'] in
        ('dispatched','running_or_unknown','deferred_resources') for x in launches) else 'assets_unresolved')
    if not case.get('source_search'):
        state['active']=None
    cc.save(root,state)
    return cc.project(state)


def reconcile_pending(root):
    """Refresh evidence before scheduler decisions, including after process restart."""
    from . import continual_curriculum as cc
    if not policy(root):return
    state=read_mapping(cc.state_path(root));ready=ready_sources(root);changed=False
    for case in state['cases']:
        if case['status'] not in ('asset_recovery_pending','assets_unresolved'):continue
        from .source_expansion import readiness
        if readiness(case,ready):
            case['status']='queued';case['asset_revision']=case.get('asset_revision',0)+1
            changed=True;continue
        active=False;deferred=False;unknown=False
        for item in case.get('asset_recovery',[]):
            if item['status']=='deferred_resources':deferred=True;continue
            if not item.get('receipt'):continue
            receipt=Path(item['receipt'])
            receipt=current_recovery_folder(receipt.parent)/'launch.json'
            item['receipt']=str(receipt);report=receipt.parent/'report.json'
            if report.exists():
                item['status']=read_mapping(report)['status'];changed=True
                unknown|=item['status']=='recovery_unknown'
                if item['status'] in ('recovery_exhausted','recovery_failed'):
                    from driversi.rsi.services.reconstruction_recovery import eligible
                    launch=read_mapping(receipt)
                    recipe=read_mapping(launch['command'][-1])
                    if recipe.get('reconstruction_job') and eligible(recipe):deferred=True
            else:
                launch=read_mapping(receipt);proc=Path('/proc')/str(launch['pid'])/'cmdline'
                live=proc.read_bytes().replace(b'\0',b' ').decode() if proc.exists() else ''
                match=all(part in live for part in launch['command'][2:])
                active|=match;unknown|=not match
                item['status']='running' if match else 'unknown_execution';changed=True
        if unknown:case['status']='asset_recovery_unknown';changed=True
        elif not active and deferred:
            from .continual_execution import gpu_snapshot
            free=gpu_snapshot()
            if any(free.get(g,999999)<500 for g in policy(root).get('builder_gpus',[])):
                case['status']='queued';case['asset_revision']=case.get('asset_revision',0)+1;changed=True
        elif not active:
            case['status']='assets_unresolved';changed=True
            if case.get('source_search') and state['active'] is None:state['active']=case['id']
    if changed:cc.save(root,state)


def verify_source(path, *, token, source_log, threshold=27.):
    """Recompute the complete60-frame mean; never trust a passed boolean."""
    p=Path(path).resolve(strict=True);r=read_mapping(p)
    if (r.get('source_mode')!='canonical_source_pose_without_motion_v1'
        or r['asset']['token']!=token or r['asset']['source_log']!=source_log
        or r.get('operator')!='>' or r.get('threshold') not in (25.,27.)
        or threshold not in (25.,27.)):
        raise ValueError('Source identity and strict source-quality protocol must match')
    rows=r['frames'];expected={(i*5,c) for i in range(15)
        for c in ('CAM_F0','CAM_L0','CAM_R0','CAM_B0')}
    if len(rows)!=60 or {(x['tick'],x['camera']) for x in rows}!=expected:
        raise ValueError('Full four-camera15-time source evidence required')
    values=[x['measured_psnr_db'] for x in rows]
    if any(type(x) not in (int,float) or not math.isfinite(x) for x in values):
        raise ValueError('Finite measured PSNR required')
    mean=sum(values)/60
    if abs(mean-r['mean_psnr_db'])>1e-8:
        raise ValueError('Reported source mean differs from frame evidence')
    if not r.get('inputs') or not r.get('files'):
        raise ValueError('Bound asset inputs and measured images required')
    for row in rows:
        if any(not row.get(key) or row[key] not in r['files'] for key in
               ('rendered_source_image','real_source_image')):
            raise ValueError('Every measured comparison requires both bound images')
    for name,sha in {**r['inputs'],**r['files']}.items():
        if digest(name)!=sha:raise ValueError('Source evidence changed')
    if mean<=threshold:return None
    return dict(asset=r['asset'],source_report=str(p),source_sha256=digest(p),
        mean_psnr_db=mean,threshold=threshold,operator='>',trainable=False)


def execute(python, module, payload, folder, *, timeout, env):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    job=folder/'job.json';frozen_json(job,payload)
    receipt=folder/'execution.json'
    if receipt.exists():
        prior=read_mapping(receipt)
        if prior['job_sha256']!=digest(job):raise ValueError('Recovery job changed')
        if prior['status'] not in ('completed','failed'):raise RuntimeError('Unknown strategy outcome')
        return prior
    live=folder/'running.json'
    if live.exists():raise RuntimeError('Reconcile an unknown in-flight recovery before restarting')
    cmd=[python,'-u','-m',module,'--job',str(job)]
    start=time.monotonic()
    with (folder/'worker.log').open('ab') as log:
        child=subprocess.Popen(cmd,cwd=project_root(),env=env,stdout=log,
            stderr=subprocess.STDOUT,start_new_session=True)
        write_json(live,dict(pid=child.pid,command=cmd,job_sha256=digest(job)))
        try:rc=child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            import signal
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);child.wait()
            rc=124
    result=dict(status='completed' if rc==0 else 'failed',returncode=rc,
        job_sha256=digest(job),wall_seconds=time.monotonic()-start,command=cmd)
    write_json(receipt,result);live.unlink()
    return result


def repair_payload(stage,source,job,strategy,remaining):
    if remaining<120:return None
    frozen_job=Path(stage)/'execution/job.json'
    if frozen_job.exists():return read_mapping(frozen_job)
    return dict(source_report=source,asset=read_mapping(source)['asset'],sensor_root=job['sensor_root'],
        output=str(Path(stage)/'repair'),rounds=3 if strategy=='geometry' else 8,
        max_wall_seconds=min(3600,int(remaining)),minimum_source_psnr_db=job.get('minimum_source_psnr_db',27.),
        optimize_geometry=strategy=='geometry')


def run(job):
    token,source_log=job['token'],job['source_log']
    threshold=job.get('minimum_source_psnr_db',27.)
    if threshold not in (25.,27.):raise ValueError('Explicit source threshold required')
    out=Path(job['output']).resolve()
    if not out.is_relative_to(project_root()/'exp'):
        raise ValueError('Owned recovery output required')
    out.mkdir(parents=True,exist_ok=True)
    frozen_json(out/'declaration.json',dict(job=job,producer_sha256=digest(__file__)))
    if (out/'report.json').exists():
        prior=read_mapping(out/'report.json')
        if prior['status']=='source_ready':
            verify_source(prior['source_report'],token=token,source_log=source_log,threshold=threshold)
        return prior
    started=time.monotonic();attempts=[]
    source=job.get('source_report')
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(job['gpu'])
    env.update(job.get('runtime_environment',{}))
    if source:
        old=read_mapping(source)
        if old.get('threshold') not in (25.,27.) or old.get('operator')!='>':
            # Re-render with the requested protocol, preserving the old certificate.
            payload=dict(asset=old['asset'],sensor_root=job['sensor_root'],
                         output=str(out/f'strict{int(threshold)}-source'),minimum_source_psnr_db=threshold)
            checked=execute(job['gs_python'],'driversi.rendering.gaussian.source_preflight',
                payload,out/f'strict{int(threshold)}-execution',timeout=900,env=env)
            if checked['status']!='completed':raise RuntimeError('Fresh strict27 source preflight failed')
            source=str(out/f'strict{int(threshold)}-source/report.json')
        ready=verify_source(source,token=token,source_log=source_log,threshold=threshold)
    else:ready=None
    # Each method is used once. A failed command is recorded rather than being
    # mistaken for a low quality score or an absent output.
    strategies=['appearance','geometry','reconstruction'] if source else ['reconstruction','appearance','geometry']
    for strategy in strategies:
        if ready:break
        if (out/'STOP').exists():break
        if time.monotonic()-started>=job.get('max_wall_seconds',21600):break
        stage=out/strategy
        write_json(out/'progress.json',dict(status='recovering',strategy=strategy,
            token=token,attempts=attempts,wall_seconds=time.monotonic()-started,trainable=False))
        if strategy=='reconstruction':
            if not job.get('reconstruction_job'):
                attempts.append(dict(strategy=strategy,status='backend_not_configured'));continue
            witness=job.get('runtime_witness')
            while witness and not Path(witness).exists():
                write_json(out/'progress.json',dict(status='waiting_reconstruction_runtime',
                    strategy=strategy,wall_seconds=time.monotonic()-started,trainable=False))
                if (out/'STOP').exists() or time.monotonic()-started>=job.get('max_wall_seconds',21600):
                    raise TimeoutError('Declared reconstruction runtime did not become ready within the task budget')
                time.sleep(15)
            if witness and read_mapping(witness).get('ready') is not True:
                raise ValueError('Native reconstruction witness not verified')
            build=read_mapping(job['reconstruction_job'])
            remaining=job.get('max_wall_seconds',21600)-(time.monotonic()-started)
            result=execute(job['reconstruction_python'],'driversi.rendering.gaussian.mtgs_reconstruction',
                build,stage/'execution',timeout=min(remaining,job.get('reconstruction_timeout_seconds',18000)),env=env)
            attempts.append(dict(strategy=strategy,**result))
            if result['status']!='completed':continue
            rebuilt=read_mapping(Path(build['output'])/'report.json')
            if rebuilt['status']!='reconstructed_requires_source_preflight':continue
            asset={**job['asset_template'],**{k:rebuilt[k] for k in ('bundle_root','asset_id','metadata_sha256')}}
            payload=dict(asset=asset,sensor_root=job['sensor_root'],output=str(stage/'source'),minimum_source_psnr_db=threshold)
            remaining=job.get('max_wall_seconds',21600)-(time.monotonic()-started)
            if remaining<=0:break
            checked=execute(job['gs_python'],'driversi.rendering.gaussian.source_preflight',
                payload,stage/'preflight-execution',timeout=min(900,remaining),env=env)
            if checked['status']!='completed':continue
            source=str(stage/'source/report.json')
        else:
            if not source:continue
            remaining=job.get('max_wall_seconds',21600)-(time.monotonic()-started)
            payload=repair_payload(stage,source,job,strategy,remaining)
            if payload is None:break
            result=execute(job['gs_python'],'driversi.rendering.gaussian.source_repair_campaign',
                payload,stage/'execution',timeout=min(3700,remaining),env=env)
            attempts.append(dict(strategy=strategy,**result))
            if result['status']!='completed':continue
            source=read_mapping(stage/'repair/report.json')['source_report']
        ready=verify_source(source,token=token,source_log=source_log,threshold=threshold)
    result=dict(status='source_ready' if ready else 'recovery_exhausted',attempts=attempts,
        wall_seconds=time.monotonic()-started,trainable=False,
        next_action='fresh_branch_generation_and_admission' if ready else 'continue_other_source_keep_unresolved')
    if ready:result.update(ready)
    write_json(out/'report.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--job',required=True)
    args=parser.parse_args();job=read_mapping(args.job)
    try:result=run(job)
    except Exception as exc:
        unknown=any(Path(job['output']).rglob('running.json'))
        write_json(Path(job['output'])/'report.json',dict(status='recovery_unknown' if unknown else 'recovery_failed',
            error_type=type(exc).__name__,error=str(exc),trainable=False))
        raise
    print(json.dumps(result,ensure_ascii=False))
    if result['status']!='source_ready':raise SystemExit(2)


if __name__=='__main__':main()
