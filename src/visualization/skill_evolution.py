"""Aggregate current frozen-skill campaign; never export test identities or captions."""
import json
import errno
import sqlite3
import time
from pathlib import Path


def read(path):
    # NFS atomic replacement can invalidate a descriptor between open and read.
    # Reopen only this known transient error; never fabricate a fresh snapshot.
    for attempt in range(3):
        try:
            return json.loads(path.read_text())
        except FileNotFoundError:
            return {}
        except OSError as exc:
            if exc.errno!=getattr(errno,'ESTALE',116) or attempt==2:raise
            time.sleep(0.1*(attempt+1))


def project(directory):
    root=Path(directory)
    if (root/'comparison-config.json').exists():
        state=read(root/'comparison-state.json');analysis=read(root/'analysis/report.json')
        conditions=[dict(name=n,current=project(root/n)) for n in ('retain','reset')]
        return dict(kind='frozen_skill_evolution',status=state.get('status','preparing'),
            conditions=conditions,weight_updates=0,elapsed_seconds=state.get('elapsed_seconds'),
            shared_first_reply=analysis.get('shared_first_reply'),
            paired_first_inference_equal=analysis.get('paired_first_inference_equal'),
            memory_benefit_established=False,new_api_requests=analysis.get('new_api_requests'),
            planned_predictions=960,completed_predictions=sum(c['current'].get('completed_predictions',0) for c in conditions))
    plan=read(root/'declaration/plan.json')
    if not plan: return dict(kind='frozen_skill_evolution',status='preparing',episodes=[],stages=[])
    run=root/'runs'/plan['task_id']; state=read(run/'report.json')
    database=root/'harness.sqlite'
    if not state and database.exists():
        with sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True) as db:
            row=db.execute('SELECT payload FROM harness_tasks WHERE id=?',(plan['task_id'],)).fetchone()
            if row: state=json.loads(row[0])
    observations=read(run/'observations/report.json')
    episodes=[]; predictions=0; requests=0; input_tokens=0; output_tokens=0
    active=plan['skills']['version']; revision=plan.get('initial_memory_revision',0)
    for n in range(plan['episodes']):
        proposal=read(run/f'proposal-{n}/report.json')
        inference=read(run/f'inference-{n}/report.json')
        progress=read(run/f'inference-{n}/progress.json')
        decision=read(run/f'decision-{n}/report.json')
        response=read(run/f'proposal-{n}/doctor/response.json')
        d=decision.get('decision',{})
        usage=response.get('usage',{});requests+=bool(response)
        input_tokens+=usage.get('prompt_tokens',0);output_tokens+=usage.get('completion_tokens',0)
        count=inference.get('rows',progress.get('completed',0));predictions+=count
        means={};failures={}
        for role,c in d.get('cohorts',{}).items():
            means[role]={k:round(v*100,4) for k,v in c['means'].items()}
            pairs=[x for ref in c['comparisons'].values() for x in ref.get('new_zero',[])]
            safety=[x for ref in c['comparisons'].values() for x in ref.get('safety_regressions',[])]
            failures[role]=dict(new_zero_scenes=len({x[0] for x in pairs}),
                                safety_regression_scenes=len({x['token'] for x in safety}))
        if decision:
            active=decision['active_version'];revision=decision['memory_revision']
        episodes.append(dict(episode=n+1,skill=proposal.get('skill_id'),
            text=proposal.get('candidate',{}).get('skills',{}).get(proposal.get('skill_id'),''),
            experience_read=proposal.get('experience_count'),predictions=count,
            status=('promoted_locally' if d.get('accepted') else 'rolled_back') if decision else
                    (progress.get('stage','inference') if proposal else 'pending'),
            proposal_valid=proposal.get('valid'), means=means,failures=failures,
            eligibility=proposal.get('candidate',{}).get('eligibility',{}).get(proposal.get('skill_id')),
            diagnosis_reference_valid=proposal.get('diagnosis_reference_valid',proposal.get('valid') if plan.get('skill_repair_protocol') else None),
            reasons=d.get('reasons',[]),parameters_frozen=inference.get('parameters_frozen'),
            optimizer_updates=inference.get('optimizer_updates'), doctor_seconds=response.get('seconds')))
        if plan.get('skill_repair_protocol'):
            episodes[-1].update(modification_check=proposal.get('modification_check',{}),
                selected_variant=decision.get('selected_variant'),diagnostic_variant=decision.get('diagnostic_variant'),
                variants={k:dict(target_ttc_delta=v['target_ttc_delta'],pdms_delta=v['development_pdms_delta'],
                    eligible=v['development_eligible'],regression_passed=v['regression']['passed'],
                    reasons=v['development_reasons']) for k,v in decision.get('variants',{}).items()},
                validation_target_ttc_delta=decision.get('validation_target_ttc_delta'),
                regression_checks=len(proposal.get('regression_obligations',[])))
    status=state.get('status','prepared')
    # Shared storage can expose the terminal task receipt before all stage files.
    if status=='completed' and any(e['status'] not in ('promoted_locally','rolled_back') for e in episodes):
        status='finalizing_receipts'
    return dict(kind='frozen_skill_evolution',status=status,
        repair_protocol=plan.get('skill_repair_protocol'),
        repair_family_scenes=sum(m.get('repair_family',False) for m in plan['members']),
        model='AUGUstus',weight_updates=0,weight_update_policy='disabled',
        active_version=active,memory_revision=revision,
        inherited_experiences=plan.get('historical_experience_count',plan.get('initial_memory_revision',0)),
        memory_mode=plan.get('memory_mode','retain'),validation_scope=plan.get('validation_scope','fresh_acquisition_drives'),
        development_scenes=sum(m['role']=='development' for m in plan['members']),
        validation_drives_per_episode=plan.get('validation_drives_per_episode',8),
        planned_predictions=plan['total_scoring_agent_steps']//80,completed_predictions=predictions,
        doctor_requests=requests,max_doctor_requests=plan['max_doctor_requests'],
        input_tokens=input_tokens,output_tokens=output_tokens,
        elapsed_seconds=(time.time()-state['started_at'] if state.get('status')=='running' and state.get('started_at')
                         else state.get('elapsed_seconds')),episodes=episodes,
        evidence=observations.get('caption_summary',{}),
        stages=[dict(id=s['id'],status=s['status']) for s in state.get('stages',[])],
        production_policy_changed=False,scope='navtest_adaptation_small_cross_log_validation')


def publish_until_settled(config_path, max_seconds=3600):
    """Finite read-only UI projection, not an experiment scheduler or restart loop."""
    from datetime import datetime, timezone
    from .catalog import PROJECT
    from .current import current_campaign
    from .build import atomic
    config=read(Path(config_path))
    if not config.get('current_skill_campaign'): raise ValueError('Explicit skill campaign required')
    destination=Path(config['output'])/'platform.json'
    deadline=time.monotonic()+max_seconds
    while time.monotonic()<deadline:
        payload=read(destination)
        if payload.get('schema_version')!=1: raise ValueError('Build the platform before publishing current state')
        payload['current']=current_campaign(config,PROJECT)
        payload['generated_at']=datetime.now(timezone.utc).isoformat()
        atomic(destination,json.dumps(payload,ensure_ascii=False,allow_nan=False))
        if payload['current']['status'] in ('completed','failed','cancelled','budget_exhausted','outcome_unknown'):
            return payload['current']['status']
        time.sleep(10)
    return 'projection_time_limit'


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    args=parser.parse_args();print(publish_until_settled(args.config))
