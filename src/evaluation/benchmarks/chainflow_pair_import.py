"""Verify recovered full benchmark shards and import real paired Harness feedback.

The original failed evaluator nodes are preserved. This is an explicit recovery
receipt; it does not promote a checkpoint or invent a completed old task.
"""
import argparse
import csv
import json
import fcntl
import math
from pathlib import Path
import time

from driversi.agents.backends.native import digest, project_root
from driversi.configuration import read_mapping
from driversi.rsi.experiments.continuation import frozen_json
from driversi.runtime.budget import write_json

METRICS=('score','no_at_fault_collisions','drivable_area_compliance',
         'driving_direction_compliance','time_to_collision_within_bound','ego_progress','comfort')


def shared_identity(identities, checkpoints):
    shared=None
    if set(identities)!={(a,s) for a in ('parent','control','treatment') for s in (0,1)}:
        raise ValueError('All six matched shards required')
    for (arm,shard),value in identities.items():
        if value.get('checkpoint_sha256')!=checkpoints[arm] or value.get('shard')!=shard:
            raise ValueError('Worker checkpoint or shard differs')
        normalized={k:v for k,v in value.items() if k not in ('checkpoint_sha256','shard')}
        if shared is None:shared=normalized
        if normalized!=shared:raise ValueError('Cross-arm inference identity differs')
    if shared.get('seed')!=42 or shared.get('shards')!=2 or shared.get('precision')!='32-true' or shared.get('tto') is not False:
        raise ValueError('Recovery inference contract differs')
    return shared


def read_table(path, known):
    rows={}
    with Path(path).open() as stream:
        for row in csv.DictReader(stream):
            token=row['token']
            values={k:float(row[k]) for k in METRICS}
            if (token in rows or token not in known or row['log_name']!=known[token]
                    or row['valid']!='True' or row['error']
                    or any(not math.isfinite(v) or not 0<=v<=1 for v in values.values())):
                raise ValueError('Incomplete, duplicated or invalid scored population')
            rows[token]=values
    if set(rows)!=set(known) or len(rows)!=12146:raise ValueError('Full official population required')
    return rows


def exact_changes(candidate, reference):
    if candidate.keys()!=reference.keys():raise ValueError('Matched population required')
    return {k:dict(improved=sum(candidate[t][k]>reference[t][k] for t in candidate),
                   regressed=sum(candidate[t][k]<reference[t][k] for t in candidate),
                   delta_percentage_points=100*sum(candidate[t][k]-reference[t][k] for t in candidate)/len(candidate))
            for k in METRICS}


def verify_shard_rows(rows, expected, merged):
    observed=set()
    for row in rows:
        token=row['token']
        if token in observed or token not in expected or row.get('valid') is not True or row.get('error'):
            raise ValueError('Invalid shard membership or result')
        if row['log_name']!=expected[token] or {k:row[k] for k in METRICS}!=merged[token]:
            raise ValueError('Merged table differs from identity-bound worker output')
        observed.add(token)
    if observed!=set(expected):raise ValueError('Shard coverage incomplete')


def recovery_overhead(source):
    """Keep verification and superseded work outside unique training lineage."""
    source=Path(source); result={}
    replay=source/'tasks/paired/treatment_replay/training-result.json'
    if replay.exists():
        row=read_mapping(replay)
        result['resume_verification']=dict(receipt=str(replay),sha256=digest(replay),
            status=row['status'],consumed=row.get('consumed'),
            included_in_unique_training_cost=False)
    final=source/'tasks/paired/report.json'
    if final.exists():
        row=read_mapping(final)
        stages=[s for s in row['stages'] if s['id'] in ('control_evaluation','treatment_evaluation')]
        result['superseded_serial_evaluation']=dict(receipt=str(final),sha256=digest(final),
            task_status=row['status'], stages=[dict(stage=s['id'],status=s['status'],
                elapsed_seconds=s.get('elapsed_seconds'),
                charged_agent_steps=s.get('charged_agent_steps'),cost_basis=s.get('cost_basis'))
                for s in stages],
            wall_seconds=sum(s['elapsed_seconds'] for s in stages if isinstance(s.get('elapsed_seconds'),(int,float))),
            wall_seconds_complete=all(isinstance(s.get('elapsed_seconds'),(int,float)) for s in stages),
            scoring_cost_may_be_reserved_upper_bound=True)
    return result


def run(evaluation):
    from driversi_agents.chainflow.benchmark import read_index
    from driversi.rsi.plugins.chainflow import SAFETY, run_feedback
    from driversi.rsi.plugins.capability_skills import prepare_outcome, execute_outcome
    root=project_root();ev=Path(evaluation).resolve(strict=True);out=ev/'verified-import'
    report=read_mapping(ev/'report.json');inputs=read_mapping(ev/'inputs.json')
    if report['status']!='completed':raise ValueError('Evaluation has not completed')
    out.mkdir(exist_ok=True)
    with (out/'run.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cache={}
        def sha(path):
            path=str(Path(path).resolve(strict=True))
            if path not in cache:cache[path]=digest(path)
            return cache[path]
        for path,value in inputs['bindings'].items():
            if sha(path)!=value:raise ValueError('Evaluation binding changed: '+path)
        source=Path(report['source_controller']);native=source/'tasks/paired'
        from driversi.evaluation.benchmarks.chainflow_pair import validate_updates
        from driversi.training.contracts.evaluation import resolve_context
        declared=read_mapping(source/'paired/intervention.json')['optimizer_updates_per_arm']
        models={a:read_mapping(native/(a+'_resume/training-result.json')) for a in ('control','treatment')}
        updates=validate_updates(declared,models)
        if inputs.get('updates_per_arm', updates)!=updates:
            raise ValueError('Evaluation update count differs from trained arms')
        plans={};evalplans={};bound={}
        for arm in ('control','treatment'):
            plans[arm]=read_mapping(native/(arm+'_resume/training-plan.json'))
            original_eval=native/(arm+'_evaluation/training-plan.json')
            evalplans[arm]=(read_mapping(original_eval) if original_eval.exists() else
                dict(evaluation_context=resolve_context(str(source/'paired'/arm/'evaluator.json'),
                    'chainflow_stage1',plans[arm]['source_groups']['development'])))
            if models[arm]['checkpoint']!=inputs['checkpoints'][arm]:
                raise ValueError('Recovery input is not the settled trained checkpoint')
            for p,h in plans[arm]['files'].items():
                if sha(p)!=h:raise ValueError('Original training/source input changed: '+p)
                bound[p]=h
            ec=evalplans[arm]['evaluation_context']
            if sha(ec['manifest'])!=ec['manifest_sha256']:raise ValueError('Original evaluator contract changed')
            for p,h in ec['value']['files'].items():
                if sha(p)!=h:raise ValueError('Held-out source changed')
        identities={};worker_seconds={}
        for arm in ('parent','control','treatment'):
            worker_seconds[arm]=0.
            for shard in range(2):
                folder=ev/arm/('shard-'+str(shard))
                if read_mapping(folder/'exit.json').get('returncode')!=0:raise ValueError('Worker did not complete')
                identities[arm,shard]=read_mapping(folder/'identity.json')
                duration=read_mapping(folder/'exit.json')['time']-read_mapping(folder/'launch.json')['time']
                if not math.isfinite(duration) or duration<=0:raise ValueError('Invalid worker process cost')
                worker_seconds[arm]+=duration
        common=shared_identity(identities,{a:sha(p) for a,p in inputs['checkpoints'].items()})
        model=plans['control']['model_config']
        config=read_mapping(ev/'parent/shard-0/config.json')
        yaml=root/'agents/chainflow/native/navsim/planning/script/config/common/agent/chainflow_vla_stage1.yaml'
        if common['model_config_sha256']!=sha(yaml) or common['backbone_sha256']!=sha(root/model['weights']['backbone']) or common['index_sha256']!=sha(config['index']):
            raise ValueError('Worker model, backbone or population changed')
        index=read_index(config['index'],'navtest');known={r['token']:r['source_log'] for r in index['records']}
        tables={arm:read_table(ev/arm/'merged/per-scene.csv',known) for arm in ('parent','control','treatment')}
        evidence_files={}
        for arm in tables:
            for shard in range(2):
                folder=ev/arm/('shard-'+str(shard))
                expected={r['token']:r['source_log'] for r in index['records'][shard::2]}
                with (folder/'scores.jsonl').open() as stream:
                    verify_shard_rows((json.loads(line) for line in stream),expected,tables[arm])
                for name in ('scores.jsonl','identity.json','progress.json','exit.json','launch.json','config.json'):
                    evidence_files[str(folder/name)]=sha(folder/name)
            for name in ('per-scene.csv','summary.json'):
                path=ev/arm/'merged'/name;evidence_files[str(path)]=sha(path)
            actual_pdms=100*sum(r['score'] for r in tables[arm].values())/len(known)
            if not math.isclose(actual_pdms,report['pdms'][arm],rel_tol=0,abs_tol=1e-8):
                raise ValueError('Coordinator aggregate differs from raw worker results')
        frozen_json(out/'input-verification.json',dict(shared_identity=common,training_bound_sources_verified=bound,
            evaluation_bindings=inputs['bindings'],verification_timing='post_run_against_worker_identity_and_original_training_seals',
            original_failed_nodes_unchanged=True,files={**evidence_files,str(ev/'report.json'):sha(ev/'report.json'),str(ev/'inputs.json'):sha(ev/'inputs.json')}))
        artifacts={}
        for arm in ('control','treatment'):
            student=models[arm]
            metrics={side:{k:sum(v[k] for v in tables[name].values())/len(known) for k in METRICS}
                     for side,name in [('parent','parent'),('candidate',arm)]}
            imported=dict(schema_version=1,model='chainflow_stage1',phase='evaluate',status='completed',reason=None,
                context=student['context'],evaluation_context=evalplans[arm]['evaluation_context'],source_groups=plans[arm]['source_groups'],
                metrics=metrics,directions={k:1 for k in METRICS},primary_metric='score',safety_higher=list(SAFETY),
                metric_scope='official_navtest_v1_full_coverage_adaptive_evaluation',
                evaluated_sample_ids=list(known),paired_samples=[dict(sample_id=t,source_log=known[t],parent=tables['parent'][t],candidate=tables[arm][t]) for t in known],
                evaluated_checkpoint=student['checkpoint'],promoted=False,
                consumed=dict(environment_steps=0,scoring_agent_steps=12146*80*1.5,
                              wall_seconds=worker_seconds[arm]+worker_seconds['parent']/2),
                recovery_import=dict(benchmark=str(ev),shared_parent_cost_fraction=.5,old_failed_node_retained=True,
                                     worker_wall_sum_not_parallel_elapsed=True,shared_identity=common))
            p=out/(arm+'-evaluation.json');frozen_json(p,imported)
            artifacts[arm+'_student']=dict(artifacts=dict(plan=str(native/(arm+'_resume/training-plan.json')),report=str(native/(arm+'_resume/training-result.json'))))
            artifacts[arm+'_evaluation']=dict(artifacts=dict(report=str(p)))
        task=read_mapping(source/'paired/task.json')
        cfg=next(s['config'] for s in task['steps'] if s['id']=='paired_feedback')
        feedback_path=out/'paired-feedback/report.json'
        if not feedback_path.exists():run_feedback(cfg,artifacts,out/'paired-feedback',None)
        feedback=read_mapping(feedback_path)
        mc=next(s['config'] for s in task['steps'] if s['id']=='skill_memory')
        prepared,_=prepare_outcome(mc,source,None)
        if not (out/'skill-memory/report.json').exists():
            execute_outcome(prepared,dict(feedback=dict(artifacts=dict(report=str(feedback_path)))),out/'skill-memory',None)
        changes={a+'_vs_'+b:exact_changes(tables[a],tables[b]) for a,b in [('control','parent'),('treatment','parent'),('treatment','control')]}
        result=dict(status='completed',full_scenes_per_arm=12146,pdms=report['pdms'],raw_exact_comparisons=changes,
            observed_pilot_benefit=feedback['observed_pilot_benefit'],next_action=feedback['next_action'],
            unique_training_cost=feedback['unique_training_cost'],evaluation_worker_wall_seconds=worker_seconds,
            parallel_evaluation_elapsed_seconds=report['elapsed_seconds'],training_steps_per_arm=updates,
            qualified_synthetic_training_samples=len(feedback['added_sample_ids']),
            family_target_met=(read_mapping(source.parent/'training-gate.json')['passed']
                               if (source.parent/'training-gate.json').exists() else False),
            original_failed_nodes_unchanged=True,promoted=False,memory_written=True,
            paired_feedback=str(feedback_path),verification=str(out/'input-verification.json'))
        result['verification_and_recovery_overhead']=recovery_overhead(source)
        result['cost_scope']='unique_training_plus_recovered_evaluation; verification_and_superseded_work_reported_separately; generation_cost_incomplete'
        frozen_json(out/'report.json',result)
        return result


def main():
    import json
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evaluation',required=True)
    p.add_argument('--wait-seconds',type=int,default=0);a=p.parse_args()
    if not 0<=a.wait_seconds<=21600:raise ValueError('Bounded wait required')
    deadline=time.monotonic()+a.wait_seconds
    while not (Path(a.evaluation)/'report.json').exists():
        if time.monotonic()>=deadline:raise TimeoutError('Full evaluation not yet complete; no feedback written')
        time.sleep(10)
    print(json.dumps(run(a.evaluation)),flush=True)


if __name__=='__main__':main()
