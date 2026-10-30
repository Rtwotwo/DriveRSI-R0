"""Run bounded physical families, then existing admission and native paired training."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import subprocess

from driversi.configuration import read_mapping
from driversi.rsi.experiments.continuation import frozen_json
from driversi.runtime.budget import write_json


def prepare(config, selection):
    from driversi.rsi.experiments.reactive_cohort import expand_plan, validate_budget
    c, s = read_mapping(config), read_mapping(selection)
    original = read_mapping(s['plan'])
    if len(original['assignments']) != len(s['assets']):
        raise ValueError('One asset per selected training source required')
    root = Path(c['output']).resolve()
    root.mkdir(parents=True, exist_ok=True)
    plan = expand_plan(original, validate_budget(c), admission_profile=c['trajectory_admission_profile'],
                       route_support_profile=c.get('route_support_profile'))
    frozen_json(root/'trajectory-family-plan.json', plan)
    assets = {a['id']: asset for a,asset in zip(original['assignments'],s['assets'])}
    return c, s, root, plan, assets


def physical(config, selection, *, workers=4):
    from driversi.rsi.experiments.reactive_prefetch import physical_job
    from driversi.rsi.decisions.trajectory_diversity import compare, describe
    if not 1 <= workers <= 4:raise ValueError('Bounded physical workers required')
    c,s,root,plan,assets = prepare(config,selection)
    with (root/'physical-family.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        jobs=[]
        for a in plan['assignments']:
            dest=root/'candidates'/a['id']
            if (dest/'generation/report.json').exists():continue
            launch=dest/'physical-launch.json'
            if launch.exists() and Path('/proc') .joinpath(str(read_mapping(launch)['pid']),'cmdline').exists():
                raise ValueError('Independently launched physical job has not settled')
            jobs.append(dict(plan_path=str(root/'trajectory-family-plan.json'),assignment_id=a['id'],
                asset=assets[a['anchor_assignment_id']], database=c['train_catalog'],
                output=str(dest/'generation'),gpu=c['production_gpu'],render=False,runtime=c.get('runtime')))
        start=time.monotonic(); errors=[]
        def snapshot(status):
            rows=[];retained={}
            for a in plan['assignments']:
                p=root/'candidates'/a['id']/'generation/report.json'
                if not p.exists():continue
                value=read_mapping(p);row=dict(candidate=a['id'],anchor=a['anchor_assignment_id'],status=value['status'])
                if value['status']=='trajectory_ready':
                    desc=describe(value,a['id']); prior=retained.setdefault(a['anchor_assignment_id'],[])
                    row['diversity']=compare(desc,prior,minimum_ade_m=c['trajectory_family']['minimum_ade_m'])
                    if row['diversity']['distinct']:prior.append(desc)
                rows.append(row)
            result=dict(status=status,attempts_recorded=len(rows),planned=len(plan['assignments']),rows=rows,
                distinct_per_anchor={k:len(v) for k,v in retained.items()},errors=errors,
                elapsed_seconds=time.monotonic()-start,qualified_rendered_samples=0)
            write_json(root/'physical-family-summary.json',result);return result
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            pending={pool.submit(physical_job,j):j for j in jobs}
            snapshot('running')
            for future in as_completed(pending):
                try:future.result()
                except Exception as exc:errors.append(dict(candidate=pending[future]['assignment_id'],error=repr(exc)))
                snapshot('running')
        return snapshot('completed' if not errors else 'completed_with_errors')


def produce(config, selection, *, train=False, shared_gpu=None):
    from driversi.rsi.experiments.reactive_cohort import produce as produce_existing, family_training_gate
    c,s,root,plan,assets=prepare(config,selection)
    used=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total','--format=csv,noheader,nounits'],text=True)
    memory={int(l.split(',')[0]):tuple(int(v) for v in l.split(',')[1:]) for l in used.splitlines()}
    busy,total=memory[c['production_gpu']]
    if shared_gpu is not None:
        if shared_gpu!=c['production_gpu'] or total-busy<24576:raise ValueError('Authorized shared GPU needs >=24GiB free')
        frozen_json(root/'shared-gpu-authorization.json',dict(gpu=shared_gpu,source='explicit_operator_cli',
            gs_replay_pytorch_fraction=.10,gs_source_pytorch_fraction=.15,
            restriction='one_render_process_at_a_time_native_caps_unmodified'))
    elif busy>=500:raise ValueError('Rendering GPU occupied; physical results retained')
    if train:
        model=read_mapping(c['training_model_config'])
        fraction=model['settings'].get('gpu_memory_fraction',.12)
        if not 0<fraction<=.25:raise ValueError('Native family evaluation requires compatible memory fraction <=.25')
    def update(directory,stage,status,**extra):write_json(Path(directory)/'status.json',dict(stage=stage,status=status,**extra))
    samples=produce_existing(s,config,root,time.time()+c['max_wall_seconds'],update)
    summary=read_mapping(root/'trajectory-family-summary.json')
    gate=family_training_gate(summary,list(assets),c['trajectory_family']['target_per_anchor'])
    write_json(root/'training-gate.json',gate)
    if train and gate['passed']:
        from driversi.rsi.experiments.chainflow_campaign import train_pair
        return dict(gate=gate,training_completed=train_pair(config,samples,root/'chainflow-pair'))
    return dict(gate=gate,training_started=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--selection',required=True)
    p.add_argument('--phase',choices=('physical','produce'),required=True);p.add_argument('--train',action='store_true')
    p.add_argument('--shared-gpu',type=int)
    args=p.parse_args()
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
    fn=physical if args.phase=='physical' else produce
    kwargs={} if args.phase=='physical' else dict(train=args.train,shared_gpu=args.shared_gpu)
    print(json.dumps(fn(args.config,args.selection,**kwargs)),flush=True)


if __name__=='__main__':main()
