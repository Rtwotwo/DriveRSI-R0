"""Bounded retained/reset experience comparison over existing Harness tasks."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from driversi.agents.backends.native import project_root
from driversi.runtime.budget import write_json
from .frozen_skill_probe import read
from .skill_evolution import declare
from .skill_evolution_report import build


def compare(root):
    root=Path(root);reports={m:build(root/m) for m in ('retain','reset')}
    plans={m:read(root/m/'declaration/plan.json') for m in reports}
    for key in ('members','seeds','arms','skills','checkpoint_sha256','doctor_protocol','validation_scope','total_scoring_agent_steps'):
        if plans['retain'][key]!=plans['reset'][key]:raise ValueError('Comparison mismatch: '+key)
    runs={m:root/m/'runs'/plans[m]['task_id'] for m in reports}
    first={m:read(runs[m]/'proposal-0/doctor/response.json') for m in reports}
    if first['retain']!=first['reset']:raise ValueError('First Doctor reply not shared exactly')
    initial_predictions={}
    for m in reports:
        rows=read(runs[m]/'inference-0/predictions.json')
        initial_predictions[m]={(r['token'],r['seed'],r['arm']):(r['raw_trajectory'],r['prompt']) for r in rows}
    if initial_predictions['retain']!=initial_predictions['reset']:raise ValueError('Paired first-round inference differs')
    if [e['experience_read'] for e in reports['retain']['episodes']]!=[0,1]:raise ValueError('Retained experience not consumed')
    if [e['experience_read'] for e in reports['reset']['episodes']]!=[0,0]:raise ValueError('Reset experience leaked')
    result=dict(status='completed',conditions={},scored_predictions=sum(r['scored_predictions'] for r in reports.values()),
        optimizer_updates=0,new_api_requests=sum(r['new_api_requests'] for r in reports.values()),
        shared_first_reply=True,paired_first_inference_equal=True,
        validation_scope=plans['retain']['validation_scope'],memory_benefit_established=False,
        production_policy_changed=False)
    lines=['# Frozen conditional-skill memory comparison','',
        'Same initial skill, observations, paired seeds and first Doctor reply. No SFT. New anchors from previously used validation drives; not a pristine holdout.',
        '', '| Condition | Round | Experiences read | Candidate dev PDMS | Incumbent dev PDMS | Decision |',
        '|---|---:|---:|---:|---:|---|']
    for mode,r in reports.items():
        result['conditions'][mode]=dict(promoted_episodes=r['promoted_episodes'],wall_seconds=r['wall_seconds'],
            episodes=[dict(episode=e['episode']+1,experience_read=e['experience_read'],decision=e['decision'],
                diagnosis=e['diagnosis'],eligibility=e['eligibility'],native_fallback_predictions=e['native_fallback_predictions']) for e in r['episodes']])
        for e in r['episodes']:
            means=e['decision']['cohorts']['development']['means']
            lines.append(f"| {mode} | {e['episode']+1} | {e['experience_read']} | {100*means['candidate']:.4f} | {100*means['incumbent']:.4f} | {'local promotion' if e['decision']['accepted'] else 'rollback'} |")
    lines+=['','This short paired trial tests execution and immediate experience-conditioned changes; it does not establish general memory benefit or full-test improvement. Evidence references validate provenance, not causal truth.']
    out=root/'analysis';out.mkdir(exist_ok=True);write_json(out/'report.json',result);(out/'report.md').write_text('\n'.join(lines)+'\n')
    return result


def worker(campaign):
    from driversi.rsi.workflow.engine import register,run
    c=Path(campaign);plan=read(c/'declaration/plan.json')
    register(c/'harness.sqlite',c/'declaration/task.json',c/'runs')
    result=run(c/'harness.sqlite',plan['task_id'])
    print(json.dumps(dict(status=result['status'])),flush=True)
    return 0 if result['status']=='completed' else 1


def launch(root):
    root=Path(root).resolve();repo=project_root()
    if not root.is_relative_to(repo/'exp'):raise ValueError('Comparison outside experiment storage')
    if (root/'comparison-state.json').exists():raise ValueError('Comparison already launched; inspect receipts before recovery')
    config=read(root/'comparison-config.json');processes={};handles={}
    def start(mode,gpu):
        campaign=root/mode
        if not (campaign/'declaration/plan.json').exists():declare(campaign/'config.json',campaign/'declaration')
        env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'CUBLAS_WORKSPACE_CONFIG':':4096:8',
             'OMP_NUM_THREADS':'8','MKL_NUM_THREADS':'8','TOKENIZERS_PARALLELISM':'false','PYTHONUNBUFFERED':'1'}
        handles[mode]=(campaign/'harness.log').open('ab')
        p=subprocess.Popen([config['python'],'-m','driversi.rsi.experiments.skill_memory_compare','worker',str(campaign)],cwd=repo,env=env,
            stdout=handles[mode],stderr=subprocess.STDOUT,start_new_session=True)
        processes[mode]=p;(campaign/'harness.pid').write_text(str(p.pid))
    started=time.time();state=dict(status='running',started_at=started,conditions={},automatic_dispatch=True)
    write_json(root/'comparison-state.json',state)
    try:
        start('retain',config['gpus']['retain'])
        while True:
            first=root/'retain/runs'/read(root/'retain/declaration/plan.json')['task_id']/'proposal-0/report.json'
            serial=config['gpus']['retain']==config['gpus']['reset']
            if 'reset' not in processes and first.exists() and (not serial or processes['retain'].poll()==0):
                start('reset',config['gpus']['reset'])
            state['conditions']={m:dict(pid=p.pid,exit_code=p.poll()) for m,p in processes.items()}
            state['elapsed_seconds']=time.time()-started;write_json(root/'comparison-state.json',state)
            if any(p.poll() not in (None,0) for p in processes.values()):
                # Let the other already-started bounded task settle; never resend/restart.
                for p in processes.values():p.wait()
                raise RuntimeError('A bounded Harness condition failed; inspect known receipts')
            if len(processes)==2 and all(p.poll()==0 for p in processes.values()):break
            if len(processes)==1 and processes['retain'].poll()==0:raise RuntimeError('No first proposal to share')
            time.sleep(5)
        report=compare(root);state.update(status='completed',scored_predictions=report['scored_predictions'])
    except Exception as exc:
        state.update(status='failed',error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        state.update(finished_at=time.time(),elapsed_seconds=time.time()-started)
        write_json(root/'comparison-state.json',state)
        for f in handles.values():f.close()
    return state


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=['launch','worker','compare']);parser.add_argument('directory')
    args=parser.parse_args()
    if args.operation=='worker':raise SystemExit(worker(args.directory))
    result=launch(args.directory) if args.operation=='launch' else compare(args.directory)
    print(json.dumps({k:result[k] for k in ('status','scored_predictions') if k in result}))
