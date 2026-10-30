"""Bounded targeted repair on the existing typed DriveHarness execution spine."""
import copy
import hashlib
import json
import math
import re
from pathlib import Path

from driversi.agents.backends.native import digest, project_root
from driversi.runtime.budget import write_json
from driversi.rsi.experiments import frozen_skill_probe as pilot
from driversi.rsi.experiments import skill_evolution as evolution
from driversi.rsi.decisions import grounded_skills as grounded
from driversi.rsi.decisions import skill_repair as repair
from driversi.rsi.decisions.frozen_skills import choose_skill, SAFETY
from driversi.rsi.decisions.skill_evolution import acquisition_group, identity, memory


def observed(row):
    velocity=[]
    for key in ('vx','vy'):
        m=re.search(r'\b'+key+r':\s*([-+0-9.eE]+)\s*m/s',row['ego_dynamic_state'])
        if m is None:raise ValueError('Native velocity unavailable')
        velocity.append(float(m.group(1)))
    return dict(navigation_command=row['driving_command'],speed_mps=math.hypot(*velocity))


def family(state,preset):
    return state['navigation_command'].upper()=='GO STRAIGHT' and choose_skill(state,preset)['id']==repair.NODE


def imported_regressions(paths,development):
    """Import only measured prior development safety losses; never online token rules."""
    obligations={}
    for path in paths:
        rows=pilot.read(path);index={(r['token'],r['seed'],r['arm']):r for r in rows}
        for (token,seed,arm),r in index.items():
            if token not in development or arm!='candidate':continue
            a=index[token,seed,'incumbent']['metrics'];b=r['metrics']
            failed=[k for k in SAFETY if b[k]<a[k]-1e-9]
            if a['PDMS']>0 and b['PDMS']==0:failed.append('new_zero')
            if failed:obligations[token,seed]=dict(token=token,seed=seed,origin=str(path),failed=failed)
    return list(obligations.values())


def declare(config,output):
    from driversi_agents.augustus.data import read_cache
    c=pilot.read(config);root=project_root();out=Path(output).resolve()
    if not out.is_relative_to(root/'exp') or c['episodes']!=2:raise ValueError('Invalid bounded experiment')
    initial=pilot.read(c['initial_prepared']);score=pilot.read(c['initial_scoring'])
    if score['prepared_sha256']!=digest(c['initial_prepared']) or digest(score['scores'])!=score['scores_sha256']:
        raise ValueError('Initial feedback unsealed')
    baseline=pilot.read(c['baseline_report']);bp=pilot.read(c['baseline_plan'])
    if baseline['plan_sha256']!=digest(c['baseline_plan']) or not baseline['complete']:raise ValueError('Baseline incomplete')
    if bp['inputs'][str(Path(initial['plan']['checkpoint'])/'model.safetensors')]!=initial['plan']['checkpoint_sha256']:
        raise ValueError('Parent checkpoint differs')
    initial_rows=pilot.read(score['scores']);members=[]
    for i,m in enumerate(initial['members']):
        metrics=[r['metrics']['TTC'] for r in initial_rows if r['token']==m['token'] and r['arm']=='matched_skill']
        target=family(m['skill']['evidence'],initial['plan']['skills']) and sum(metrics)/len(metrics)<1
        members.append({**m,'role':'development','case_id':f'case-{i+1:02d}','repair_family':target})
    used={acquisition_group(m['source_log']) for m in members};excluded={m['token'] for m in members}
    for p in c.get('exclude_prepared',[]):excluded.update(m['token'] for m in pilot.read(p)['members'])
    ordered=sorted(baseline['rows'],key=lambda r:hashlib.sha256((c['selection_salt']+'/'+r['token']).encode()).hexdigest())
    lookup={r['token']:i for i,chunk in enumerate(bp['chunks']) for r in chunk};loaded={}
    def chunk_for(token):return Path(c['baseline_plan']).parent/'chunks'/f'{lookup[token]:04d}'
    def state_for(token):
        ch=chunk_for(token)
        if ch not in loaded:
            receipt=pilot.read(ch/'data/receipt.json')
            ds=read_cache(receipt['block']).select_columns(['frame_id','driving_command','ego_dynamic_state'])
            loaded[ch]={r['frame_id']:r for r in ds}
        return observed(loaded[ch][token])
    selected=[]
    for group in ('family','control'):
        count=0
        for row in ordered:
            drive=acquisition_group(row['source_log']);token=row['token']
            if drive in used or token in excluded:continue
            if group=='family' and (row['metrics']['TTC']>=1 or not family(state_for(token),initial['plan']['skills'])):continue
            ch=chunk_for(token);done=pilot.read(ch/'completed.json')
            if digest(done['report'])!=done['sha256']:raise ValueError('Baseline metric receipt changed')
            files=pilot.read(done['report'])['inputs']['metric_files']
            metric=next(p for p in files if Path(p).parent.name==token)
            selected.append(dict(token=token,source_log=row['source_log'],role='validation',validation_group=group,
                selection_stratum='historical_TTC_failure_native_route' if group=='family' else 'score_blind_drive',
                cache_receipt=str(ch/'data/receipt.json'),metric_source=metric,metric_sha256=files[metric]))
            used.add(drive);count+=1
            if count==8:break
        if count!=8:raise ValueError('Insufficient independent '+group+' validation drives')
    members+=selected
    obligations=imported_regressions(c['regression_scores'],{m['token'] for m in members if m['role']=='development'})
    sources=list(initial['plan']['files'])+[str(config),c['initial_prepared'],c['initial_scoring'],score['scores'],c['baseline_report'],c['baseline_plan']]
    sources+=c.get('exclude_prepared',[])+c['regression_scores']+[c['prior_experience']]
    historical=pilot.read(c['prior_experience'])
    if historical['active']!=initial['plan']['skills']:raise ValueError('Historical skill has a different active parent')
    sources+=[str(root/p) for p in ('src/rsi/decisions/skill_repair.py','src/rsi/decisions/grounded_skills.py',
        'src/rsi/decisions/skill_evolution.py','src/rsi/experiments/skill_repair.py','src/rsi/experiments/skill_evolution.py',
        'src/rsi/plugins/skill_evolution.py','src/data/catalog/skill_context.py','agents/augustus/skill_inference.py')]
    plan=dict(schema_version=1,checkpoint=initial['plan']['checkpoint'],checkpoint_sha256=initial['plan']['checkpoint_sha256'],
        skills=initial['plan']['skills'],seeds=[42,43],arms=list(repair.ARMS),members=members,files={p:digest(p) for p in sources},
        source_receipts={m['cache_receipt']:digest(m['cache_receipt']) for m in members},initial_scores=score['scores'],
        private_config=c['private_config'],credential_file=c['credential_file'],memory=str(out.parent/'memory/state.sqlite'),
        episodes=2,scoring_agent_steps=len(members)*2*5*80,total_scoring_agent_steps=2*len(members)*2*5*80,
        max_doctor_requests=2,scope='retrospective_navtest_adaptation',no_optimizer=True,gpu=0,gpu_memory_fraction=0.25,
        validation_grouping='original_acquisition_drive',validation_drives_per_episode=16,task_id=c['task_id'],
        skill_repair_protocol=repair.PROTOCOL,doctor_protocol=repair.PROTOCOL,memory_mode='retain',
        prior_experience=c['prior_experience'],historical_experience_count=len(historical['events']),
        validation_scope='fixed_cross_log_family_and_controls_reused_across_rounds',initial_regressions=obligations,
        caption_library=c['caption_library'],excluded_validation_tokens=sorted(excluded),optimizer_updates=0)
    out.mkdir(parents=True,exist_ok=True);path=out/'plan.json'
    if path.exists() and pilot.read(path)!=plan:raise ValueError('Immutable declaration differs')
    write_json(path,plan)
    task=dict(schema_version=1,id=c['task_id'],label='Frozen AUGUstus targeted TTC skill repair',
        budget=dict(max_wall_seconds=14400,max_sim_agent_steps=plan['total_scoring_agent_steps']),steps=[])
    def add(name,provider,inputs,n,seconds):
        task['steps'].append(dict(id=name,provider=provider,inputs=inputs,config=dict(plan=str(path),episode=n),max_wall_seconds=seconds))
    add('observations','skill_evolution_observations',{},0,2400)
    for n in range(2):
        inputs={'observations':'observations'}
        if n:inputs['previous']='decision-'+str(n-1)
        add('proposal-'+str(n),'skill_evolution_propose_initial' if not n else 'skill_evolution_propose_next',inputs,n,600)
        add('inference-'+str(n),'native_model_skill_evolution',{'proposal':'proposal-'+str(n)},n,2400)
        add('scoring-'+str(n),'skill_evolution_scoring',{'proposal':'proposal-'+str(n),'predictions':'inference-'+str(n)},n,1200)
        add('decision-'+str(n),'skill_evolution_decision',{'proposal':'proposal-'+str(n),'scores':'scoring-'+str(n)},n,300)
    write_json(out/'task.json',task)
    write_json(out/'selection-audit.json',dict(development=len(members)-16,family_development=sum(m.get('repair_family',False) for m in members),
        family_validation=8,control_validation=8,regression_checks=len(obligations),acquisition_drives_disjoint=True,
        prior_validation_drives_may_be_reused=True,validation_fixed_before_new_predictions=True))
    return plan


def prepare(plan_path,output):return evolution.prepare(plan_path,output)


def obligations_for(plan,state):
    result={(r['token'],r['seed']):r for r in plan['initial_regressions']}
    for e in state['events']:
        for r in e.get('new_regressions',[]):result[r['token'],r['seed']]=r
    return list(result.values())


def proposal(observations,previous,episode,output):
    source=pilot.read(observations);plan=source['plan'];out=Path(output);out.mkdir(parents=True,exist_ok=True)
    state=memory(plan['memory'])
    if state['revision']!=episode:raise ValueError('Unexpected durable revision')
    if episode==0:
        if previous is not None:raise ValueError('Unexpected prior feedback')
        rows=pilot.read(plan['initial_scores']);arm='matched_skill'
    else:
        before=pilot.read(previous)
        if before['episode']!=episode-1 or before['memory_after_hash']!=identity(state):raise ValueError('Memory handoff mismatch')
        if digest(before['scores'])!=before['scores_sha256']:raise ValueError('Prior feedback changed')
        active=before['selected_variant'] if before['decision']['accepted'] else 'incumbent'
        rows=repair.mapped(pilot.read(before['scores']),before['diagnostic_variant'])
        arm='candidate' if active!='incumbent' else 'incumbent'
    development=grounded.feedback(rows,source['members'],arm,evolution.read_contexts(source),all_cases=True)
    obligations=obligations_for(plan,state);ids={m['token']:m.get('case_id') for m in source['members']}
    visible=[dict(case_id=ids[r['token']],seed=r['seed'],failed=r.get('failed',[])) for r in obligations]
    targets=[m['case_id'] for m in source['members'] if m.get('repair_family')]
    prior=pilot.read(plan['prior_experience'])['events']
    proposed=repair.propose(development,state,out/'doctor',plan['private_config'],plan['credential_file'],targets,visible,prior_events=prior)
    presets={'incumbent':proposed['incumbent'],**proposed['variants']};members=[]
    for original in source['members']:
        m=copy.deepcopy(original);obs=m['skill']['evidence']
        m['arm_skills']={a:grounded.apply_gate(choose_skill(obs,p),p) for a,p in presets.items()}
        m['skill']=m['arm_skills']['combined'];members.append(m)
    prepared={**source,'members':members}
    write_json(out/'prepared.json',prepared)
    proposed.update(episode=episode,prepared=str(out/'prepared.json'),prepared_sha256=digest(out/'prepared.json'),
        candidate_hash=identity(proposed['candidate']),sealed_before_inference=True,regression_obligations=obligations,
        development=development,diagnosis=(proposed.get('proposal') or {}).get('diagnosis'))
    write_json(out/'report.json',proposed);return proposed


def decide(proposal_path,score_path,output):
    proposed=pilot.read(proposal_path);scoring=pilot.read(score_path)
    if digest(proposed['prepared'])!=proposed['prepared_sha256'] or scoring['prepared_sha256']!=proposed['prepared_sha256']:
        raise ValueError('Prepared observations changed')
    if digest(scoring['scores'])!=scoring['scores_sha256']:raise ValueError('Scores changed')
    prepared=pilot.read(proposed['prepared']);rows=pilot.read(scoring['scores'])
    result=repair.decide(rows,prepared['members'],prepared['plan']['seeds'],proposed,proposed['regression_obligations'])
    arm=result['diagnostic_variant'];decision=result['decision'];candidate=proposed['variants'][arm]
    development=grounded.feedback(repair.mapped(rows,arm),prepared['members'],'candidate',evolution.read_contexts(prepared),all_cases=True)
    index={(r['token'],r['seed'],r['arm']):r for r in rows};new=[]
    for m in prepared['members']:
        if m['role']!='development':continue
        for seed in prepared['plan']['seeds']:
            a=index[m['token'],seed,'incumbent']['metrics']
            for variant in repair.VARIANTS:
                b=index[m['token'],seed,variant]['metrics'];bad=[k for k in SAFETY if b[k]<a[k]-1e-9]
                if a['PDMS']>0 and b['PDMS']==0:bad.append('new_zero')
                if bad:new.append(dict(token=m['token'],seed=seed,failed=bad,origin=prepared['plan']['task_id']+'/'+str(proposed['episode'])+'/'+variant))
    feedback={a:{k:v for k,v in d.items() if k not in ('full_decision','regression')} for a,d in result['variants'].items()}
    event=dict(id=prepared['plan']['task_id']+'/episode-'+str(proposed['episode']),skill_id=repair.NODE,
        incumbent_hash=identity(proposed['incumbent']),candidate=candidate,decision=decision,development=development,
        scores_sha256=scoring['scores_sha256'],proposal_hash=identity(proposed),modification_check=proposed['modification_check'],
        variant_feedback=feedback,new_regressions=new,diagnosis=proposed['diagnosis'])
    event.update(proposed_preset=proposed['candidate'],selected_variant=result['selected_variant'],diagnostic_variant=arm,
        variant_definitions={a:dict(text=p['skills'][repair.NODE],eligibility=p.get('eligibility',{}).get(repair.NODE))
            for a,p in proposed['variants'].items()})
    state=memory(prepared['plan']['memory'],event=event,expected_revision=proposed['memory_revision'])
    report=dict(status='completed',episode=proposed['episode'],**result,memory_after_hash=identity(state),
        memory_revision=state['revision'],active_version=state['active']['version'],experience_count=len(state['events']),
        proposal_valid=proposed['valid'],scores=scoring['scores'],scores_sha256=scoring['scores_sha256'],
        optimizer_updates=0,production_policy_changed=False,new_regressions=new)
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    write_json(out/'report.json',report);write_json(out/'memory-snapshot.json',state);return report


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('output');a=p.parse_args()
    result=declare(a.config,a.output)
    print(json.dumps(dict(members=len(result['members']),predictions=result['total_scoring_agent_steps']//80)))
