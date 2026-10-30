"""Recover a settled native pair through complete, checkpoint-bound navtest."""
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from driversi.agents.backends.native import digest, project_root
from driversi.configuration import read_mapping
from driversi.rsi.experiments.continuation import frozen_json
from driversi.runtime.budget import write_json


def compare_tables(tables):
    """Exact token joins; failed/missing observations cannot become gains."""
    keys = set(tables['parent'])
    if len(keys) != 12146 or any(set(v) != keys for v in tables.values()):
        raise ValueError('Complete identical official populations required')
    metrics = ('score', 'no_at_fault_collisions', 'drivable_area_compliance',
               'driving_direction_compliance', 'time_to_collision_within_bound',
               'ego_progress', 'comfort')
    results = {}
    for candidate, reference in [('control','parent'),('treatment','parent'),('treatment','control')]:
        detail = {}
        for metric in metrics:
            pairs = [(float(tables[candidate][k][metric]), float(tables[reference][k][metric])) for k in sorted(keys)]
            detail[metric] = dict(delta_percentage_points=100*sum(a-b for a,b in pairs)/len(pairs),
                improved=sum(a>b+1e-10 for a,b in pairs), regressed=sum(a<b-1e-10 for a,b in pairs),
                unchanged=sum(abs(a-b)<=1e-10 for a,b in pairs))
        detail['zero_score'] = dict(resolved=sum(float(tables[reference][k]['score'])==0 and float(tables[candidate][k]['score'])>0 for k in keys),
            newly_zero=sum(float(tables[reference][k]['score'])>0 and float(tables[candidate][k]['score'])==0 for k in keys))
        results[candidate+'_vs_'+reference] = detail
    return results


def validate_updates(declared, receipts):
    """Use the compiled pair budget, not a hard-coded pilot length."""
    if type(declared) is not int or not 8 <= declared <= 256:
        raise ValueError('Explicit bounded paired update count required')
    if set(receipts) != {'control', 'treatment'}:
        raise ValueError('Both completed training arms required')
    for receipt in receipts.values():
        if receipt.get('status') != 'completed' or receipt.get('total_updates') != declared:
            raise ValueError('Actual training updates differ from the compiled pair')
    return declared


def run(controller, output, *, gpus=(0,1,2,3,4,5), max_wall_seconds=21600):
    from driversi_agents.chainflow.benchmark import read_index, merge
    root=project_root(); source=Path(controller).resolve(strict=True); out=Path(output).resolve()
    if not out.is_relative_to(root/'exp') or len(gpus)!=6 or len(set(gpus))!=6:
        raise ValueError('Owned output and six distinct devices required')
    if not 60<=max_wall_seconds<=43200: raise ValueError('Bounded wall budget required')
    out.mkdir(parents=True,exist_ok=True)
    with (out/'run.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        existing=out/'report.json'
        if existing.exists() and read_mapping(existing)['status']=='completed':return read_mapping(existing)
        checkpoints={}; bindings={}; native=source/'tasks/paired'
        intervention=source/'paired/intervention.json'
        receipts={a:read_mapping(native/(a+'_resume/training-result.json')) for a in ('control','treatment')}
        updates=validate_updates(read_mapping(intervention)['optimizer_updates_per_arm'], receipts)
        bindings[str(intervention)]=digest(intervention)
        for arm in ('control','treatment'):
            gate=native/(arm+'_pre_evaluation/gate.json'); g=read_mapping(gate)
            if g['status']!='passed' or g['resume_exact'] is not True: raise ValueError('Exact recovery gate required')
            receipt=native/(arm+'_resume/training-result.json'); value=read_mapping(receipt)
            cp=Path(value['checkpoint']).resolve(strict=True)
            if digest(cp)!=value['checkpoint_files'][str(cp)]:raise ValueError('Trained checkpoint changed')
            checkpoints[arm]=str(cp)
            bindings.update({str(p):digest(p) for p in (gate,receipt,cp)})
        req=read_mapping(source/'paired/control/request.json'); model_path=Path(req['model_config']); model=read_mapping(model_path)
        checkpoints['parent']=str(root/model['weights']['checkpoint'])
        manifest_path=source/'paired/control/manifest.json'; manifest=read_mapping(manifest_path)
        index=manifest['benchmark']['test_index']; pop=read_index(index,'navtest')
        if len(pop['expected_tokens'])!=12146:raise ValueError('Full navtest required')
        bindings.update({str(p):digest(p) for p in (model_path,manifest_path,Path(index),Path(checkpoints['parent']))})
        bindings[str(root/'agents/chainflow/benchmark.py')]=digest(root/'agents/chainflow/benchmark.py')
        frozen_json(out/'inputs.json',dict(bindings=bindings,checkpoints=checkpoints,seed=42,
            shards_per_arm=2,gpus=list(gpus),gpu_memory_fraction=.20,training_unchanged=True,updates_per_arm=updates,
            scope='full_navtest_adaptive_checkpoint_pair',old_failed_evaluations_retained=True))
        jobs=[]
        for ai,arm in enumerate(('parent','control','treatment')):
            for shard in range(2):
                folder=out/arm/('shard-'+str(shard)); folder.mkdir(parents=True,exist_ok=True)
                config=dict(split='navtest',index=index,checkpoint=checkpoints[arm],
                    backbone=str(root/model['weights']['backbone']),maps_root=model['paths']['maps'],
                    seed=42,shards=2,shard=shard,gpu_memory_fraction=.20,output=str(folder))
                frozen_json(folder/'config.json',config)
                done=folder/'exit.json'
                if done.exists():
                    if read_mapping(done)['returncode']!=0:raise ValueError('Failed shard requires explicit reconciliation')
                    continue
                if (folder/'launch.json').exists():raise ValueError('Unsettled shard must be reconciled before resuming')
                jobs.append((folder,gpus[2*ai+shard]))
        snapshot=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
        used={int(row.split(',')[0]):int(row.split(',')[1]) for row in snapshot.splitlines()}
        if any(used[gpu]>=500 for _,gpu in jobs):raise ValueError('Selected GPU is occupied')
        processes=[]; start=time.monotonic()
        try:
            for folder,gpu in jobs:
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONUNBUFFERED='1',PYTHONHASHSEED='42',
                    OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
                command=[model['python'],str(root/'agents/chainflow/benchmark.py'),'evaluate',str(folder/'config.json')]
                with (folder/'worker.log').open('w') as log:
                    child=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                write_json(folder/'launch.json',dict(pid=child.pid,command=command,gpu=gpu,time=time.time()))
                processes.append((folder,child))
            while processes:
                running=[]; failed=[]
                for folder,child in processes:
                    code=child.poll()
                    if code is None:running.append((folder,child))
                    else:
                        write_json(folder/'exit.json',dict(returncode=code,time=time.time()))
                        if code:failed.append(str(folder))
                processes=running
                progress={str(f.relative_to(out)):read_mapping(f/'progress.json') if (f/'progress.json').exists() else {'recorded':0} for f in out.glob('*/shard-*')}
                write_json(out/'progress.json',dict(status='running',shards=progress,elapsed_seconds=time.monotonic()-start))
                if failed:raise RuntimeError('Native evaluation failed: '+','.join(failed))
                if time.monotonic()-start>max_wall_seconds:raise TimeoutError('Paired evaluation wall budget exhausted')
                if processes:time.sleep(10)
        finally:
            for folder,child in processes:
                if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
            for folder,child in processes:
                try:code=child.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait()
                write_json(folder/'exit.json',dict(returncode=code,interrupted=True,time=time.time()))
        summaries={}; tables={}
        for arm in ('parent','control','treatment'):
            summaries[arm]=merge(dict(split='navtest',index=index,shard_outputs=[str(out/arm/('shard-'+str(i))) for i in range(2)],output=str(out/arm/'merged')))
            if not summaries[arm]['complete'] or summaries[arm]['failed']!=0:raise ValueError('Incomplete native evaluation')
            with (out/arm/'merged/per-scene.csv').open() as stream:tables[arm]={row['token']:row for row in csv.DictReader(stream)}
        if any(digest(p)!=sha for p,sha in bindings.items()):raise ValueError('Evaluation bindings changed')
        report=dict(status='completed',scope='full_navtest_adaptive_checkpoint_pair',expected=12146,
            pdms={arm:summaries[arm]['pdms'] for arm in summaries},comparisons=compare_tables(tables),
            source_controller=str(source),bindings=bindings,elapsed_seconds=time.monotonic()-start,promoted=False)
        write_json(out/'report.json',report);return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--controller',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();print(json.dumps(run(args.controller,args.output)),flush=True)


if __name__=='__main__':main()
