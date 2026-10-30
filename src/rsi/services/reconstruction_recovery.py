"""One registered repair for a known native diagnostic-directory failure."""
import argparse
import os
import subprocess
import time
from pathlib import Path

from driversi.agents.backends.native import digest, project_root
from driversi.configuration import read_mapping
from driversi.rsi.experiments.continuation import frozen_json
from driversi.runtime.budget import write_json


def remedy(job):
    native=read_mapping(job['reconstruction_job']);root=Path(native['output'])
    if not (root/'report.json').exists() or (root/'running.json').exists():return None
    result=read_mapping(root/'report.json');stage=result.get('stage')
    if result.get('status')!='stage_failed':return None
    log=root/(str(stage)+'.log')
    if not log.exists():return None
    text=log.read_text(errors='replace')
    if stage=='registration' and 'FileNotFoundError' in text and '/map_vis/registration.pdf' in text:
        return 'map-vis-1'
    if stage=='triangulation' and 'existing_rig.RefSensorId() == rig.RefSensorId()' in text:
        return 'colmap-rigs-1'
    if stage=='depth' and result.get('returncode')==0:
        previous=Path(job['output'])/'remediations/depth-receipt-1'
        report=previous/'report.json';failed_log=previous/'completion/worker.log'
        if report.exists() and failed_log.exists() and (root/'receipts/depth.json').exists():
            old=read_mapping(report);failure=failed_log.read_text(errors='replace')
            if (old.get('status')=='recovery_failed' and
                old.get('completion',{}).get('status')=='failed' and
                'FileNotFoundError' in failure and '/depth-completion/report.json.tmp' in failure):
                return 'depth-receipt-io-1'
        return 'depth-receipt-1'
    if stage=='depth' and "No module named 'timm.layers'" in text:
        return 'depth-runtime-1'
    return None


def directory(job):
    kind=remedy(job)
    if kind:return Path(job['output'])/'remediations'/kind
    for name in ('depth-receipt-io-1','depth-receipt-1','depth-runtime-1','colmap-rigs-1','map-vis-1'):
        candidate=Path(job['output'])/'remediations'/name
        if (candidate/'launch.json').exists():return candidate
    raise ValueError('No registered recovery kind')


def eligible(job):
    kind=remedy(job)
    if kind=='depth-runtime-1' and not (Path(job['runtime_witness']).parent/'ready-depth-v1.json').exists():return False
    return bool(kind and not (Path(job['output'])/'remediations'/kind/'launch.json').exists())


def launch(job_path, python):
    job=read_mapping(job_path);out=directory(job)
    if not eligible(job):raise ValueError('Only a settled registered native failure is repairable')
    out.mkdir(parents=True,exist_ok=True)
    command=[python,'-u','-m','driversi.rsi.services.reconstruction_recovery','--job',str(job_path)]
    with (out/'worker.log').open('ab') as log:
        child=subprocess.Popen(command,cwd=project_root(),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    frozen_json(out/'launch.json',dict(pid=child.pid,command=command,job_sha256=digest(job_path),gpu=job['gpu']))
    return out/'launch.json'


def run(job_path):
    import yaml
    from driversi.rsi.experiments.asset_recovery import execute, run as repair
    job=read_mapping(job_path);out=directory(job);native=read_mapping(job['reconstruction_job'])
    out.mkdir(parents=True,exist_ok=True)
    prior=read_mapping(Path(job['output'])/'report.json')
    if prior['status']!='recovery_exhausted':raise ValueError('Original recovery must have settled')
    if (out/'report.json').exists():return read_mapping(out/'report.json')
    failed=Path(native['output'])/'report.json'
    evidence=read_mapping(failed)
    kind=remedy(job)
    if kind not in ('map-vis-1','colmap-rigs-1','depth-runtime-1','depth-receipt-1','depth-receipt-io-1'):raise ValueError('Unknown failure')
    log=Path(native['output'])/(evidence['stage']+'.log')
    frozen_json(out/'prior-native-report.json',evidence)
    prior_log=out/('prior-'+evidence['stage']+'.log')
    if not prior_log.exists():prior_log.write_bytes(log.read_bytes())
    if digest(prior_log)!=digest(log):raise ValueError('Original failure log changed before repair')
    frozen_json(out/'declaration.json',dict(job_sha256=digest(job_path),failure_log_sha256=digest(log),
        failure_log_snapshot=str(prior_log),
        prior_recovery_sha256=digest(Path(job['output'])/'report.json'),
        remedy=kind,trainable=False))
    budget=int(job.get('max_wall_seconds',21600)-prior['wall_seconds'])
    if kind in ('colmap-rigs-1','depth-runtime-1','depth-receipt-1','depth-receipt-io-1'):
        previous=Path(job['output'])/'remediations/map-vis-1/native/execution.json'
        if previous.exists():budget-=int(read_mapping(previous)['wall_seconds'])
    if kind in ('depth-runtime-1','depth-receipt-1','depth-receipt-io-1'):
        for name in ('correction','native'):
            old=Path(job['output'])/'remediations/colmap-rigs-1'/name/'execution.json'
            if old.exists():budget-=int(read_mapping(old)['wall_seconds'])
    if kind in ('depth-receipt-1','depth-receipt-io-1'):
        previous=Path(job['output'])/'remediations/depth-runtime-1/native/execution.json'
        if previous.exists():budget-=int(read_mapping(previous)['wall_seconds'])
    if kind=='depth-receipt-io-1':
        previous=Path(job['output'])/'remediations/depth-receipt-1/completion/execution.json'
        if previous.exists():budget-=int(read_mapping(previous)['wall_seconds'])
    start=time.monotonic()
    if budget<120:raise ValueError('No remaining recovery budget')
    env=os.environ.copy();env.update(job.get('runtime_environment',{}));env['CUDA_VISIBLE_DEVICES']=str(job['gpu'])
    if kind=='map-vis-1':
        config=yaml.safe_load(Path(native['config']).read_text().split('\n',1)[1])
        target=(Path(config['data_root'])/config['road_block_name']/'map_vis').resolve()
        if not target.is_relative_to(project_root()/'exp'):raise ValueError('Owned directory required')
        target.mkdir(parents=True,exist_ok=True)
    elif kind=='colmap-rigs-1':
        corrected=execute(job['reconstruction_python'],'driversi.rendering.gaussian.colmap_resume',
            dict(native_job=job['reconstruction_job'],output=str(out/'rig-correction')),
            out/'correction',timeout=min(1800,budget),env=env)
        if corrected['status']!='completed':return dict(status='recovery_failed',correction=corrected,trainable=False)
        budget-=int(time.monotonic()-start);start=time.monotonic()
        if budget<120:raise ValueError('No remaining recovery budget')
    if kind=='depth-runtime-1':
        ready_path=Path(job['runtime_witness']).parent/'ready-depth-v1.json'
        ready=read_mapping(ready_path)
        if not ready['ready']:raise ValueError('Depth runtime lacks fresh-process witness')
        for name,sha in ready['files'].items():
            if digest(name)!=sha:raise ValueError('Depth runtime evidence changed')
        frozen_json(out/'runtime-binding.json',dict(path=str(ready_path),sha256=digest(ready_path)))
    if kind in ('depth-receipt-1','depth-receipt-io-1'):
        (out/'depth-completion').mkdir(parents=True,exist_ok=True)
        checked=execute(job['reconstruction_python'],'driversi.rendering.gaussian.depth_completion',
            dict(native_job=job['reconstruction_job'],output=str(out/'depth-completion')),
            out/'completion',timeout=min(900,budget),env=env)
        if checked['status']!='completed':return dict(status='recovery_failed',completion=checked,trainable=False)
        budget-=int(time.monotonic()-start);start=time.monotonic()
        if budget<120:raise ValueError('No remaining recovery budget')
    for receipt in (Path(native['output'])/'receipts').glob('*.json'):
        for name,sha in read_mapping(receipt).get('recovery_proofs',{}).items():
            if digest(name)!=sha:raise ValueError('Recovery proof changed')
            proof=read_mapping(name)
            if proof.get('backup') and digest(proof['backup'])!=proof['backup_sha256']:
                raise ValueError('Preserved failed database changed')
    result=execute(job['reconstruction_python'],'driversi.rendering.gaussian.mtgs_reconstruction',
        native,out/'native',timeout=min(budget,job.get('reconstruction_timeout_seconds',18000)),env=env)
    if result['status']!='completed':return dict(status='recovery_failed',native=result,trainable=False)
    built=read_mapping(Path(native['output'])/'report.json')
    asset={**job['asset_template'],**{k:built[k] for k in ('bundle_root','asset_id','metadata_sha256')}}
    payload=dict(asset=asset,sensor_root=job['sensor_root'],output=str(out/'source'),minimum_source_psnr_db=job.get('minimum_source_psnr_db',27.))
    remaining=budget-(time.monotonic()-start)
    if remaining<120:return dict(status='recovery_exhausted',trainable=False)
    checked=execute(job['gs_python'],'driversi.rendering.gaussian.source_preflight',payload,
        out/'preflight',timeout=min(900,remaining),env=env)
    if checked['status']!='completed':return dict(status='recovery_failed',preflight=checked,trainable=False)
    remaining=int(budget-(time.monotonic()-start))
    if remaining<120:return dict(status='recovery_exhausted',trainable=False)
    continuation=dict(job,output=str(out/'quality'),source_report=str(out/'source/report.json'),
                      max_wall_seconds=remaining)
    continuation.pop('reconstruction_job',None)
    result=repair(continuation)
    result['registered_io_remedy']=str(out/'declaration.json')
    result['prior_completed_stages_reused']=True
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--job',required=True)
    args=parser.parse_args();job=read_mapping(args.job);out=directory(job)
    try:result=run(args.job)
    except Exception as exc:
        result=dict(status='recovery_unknown' if any(out.rglob('running.json')) else 'recovery_failed',
                    error_type=type(exc).__name__,error=str(exc),trainable=False)
        write_json(out/'report.json',result);raise
    write_json(out/'report.json',result)
    if result['status']!='source_ready':raise SystemExit(2)


if __name__=='__main__':main()
