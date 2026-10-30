"""Evidence-grounded, staged repair inside the existing DriveHarness lifecycle."""
import copy
import gc
import json
import statistics
import time
from pathlib import Path

from driversi.agents.backends.native import digest, project_root
from driversi.runtime.budget import write_json
from driversi.rsi.experiments import frozen_skill_probe as pilot
from driversi.rsi.decisions import grounded_skills as grounded, skill_doctor
from driversi.rsi.decisions import hard_case_repair as rules
from driversi.rsi.decisions.frozen_skills import choose_skill, METRICS
from driversi.rsi.decisions.skill_evolution import identity, memory, acquisition_group

NODE='following_speed_check'
IDENTITY_GATE=dict(commands=list(grounded.COMMANDS),speed_min_mps=0,speed_max_mps=60,exclude_stop=False)
PROMPT=grounded.PROMPT+'''
This is LOCAL HARD-CASE REPAIR, not global stylistic rewriting. Real development history
images, saved native/incumbent trajectories and retrospective scorer conflict diagnostics
are supplied. Future geometry is privileged diagnostic evidence only: never copy it into
the student's prompt as an observed fact. Images do not establish an unseen future motion.
The student never receives the retrospective evidence bundle or scoring results.
Propose ONE distinct, minimal intervention: change skill_text OR eligibility, not both.
For a text edit, copy incumbent_eligibility exactly. For a gate edit, copy incumbent_text
exactly. Address one explicit, testable conflict hypothesis. Preserve all safe behavior.
Actual previous rejected text/gates and measured outcomes are binding negative experience:
do not repeat them with a different rationale or synonym-only rewrite. If past speed-control
advice failed, consider a genuinely different observation-grounded hypothesis, not a longer
list of generic rules. Use sensor images and geometry to distinguish traffic ahead from
oncoming, crossing or stationary objects. Never state a causal conclusion unsupported by
the evidence. Do not add case IDs, coordinates, future positions or numeric targets to skill_text.
Only development feedback is available. No validation membership or outcomes are supplied.
'''


def verify_evidence(source,score,evidence,prepared_path):
    expected=dict(prepared_sha256=digest(prepared_path),scores_sha256=score['scores_sha256'])
    if evidence.get('inputs')!=expected or evidence.get('scope')!='offline_development_diagnosis_only':
        raise ValueError('Diagnostic evidence source identity mismatch')
    members={m['token']:m for m in source['members']}
    if {c['token'] for c in evidence['cases']}!=set(evidence['target_tokens']):
        raise ValueError('Diagnostic target coverage mismatch')
    for case in evidence['cases']:
        m=members[case['token']]
        if (m['role']!='development' or case['case_id']!=m['case_id'] or
            case['source_log']!=m['source_log'] or case['observed_state']!=m['skill']['evidence'] or
            case['metric_cache_sha256']!=m['metric_sha256']):
            raise ValueError('Diagnostic case differs from sealed observation')


def declare(config,output):
    c=pilot.read(config);root=project_root();out=Path(output).resolve()
    if not out.is_relative_to(root/'exp'):raise ValueError('Artifacts must be under exp')
    source=pilot.read(c['source_prepared']);score=pilot.read(c['historical_scoring'])
    if score['prepared_sha256']!=digest(c['source_prepared']) or score['scores_sha256']!=digest(score['scores']):
        raise ValueError('Historical evidence unsealed')
    ev=pilot.read(c['evidence']);targets=ev['target_tokens']
    verify_evidence(source,score,ev,c['source_prepared'])
    if not 1<=len(targets)<=3 or len(set(targets))!=len(targets):raise ValueError('One to three targets required')
    by={m['token']:m for m in source['members']}
    if any(t not in by or by[t]['role']!='development' or by[t]['skill']['id']!=NODE for t in targets):
        raise ValueError('Invalid target cohort')
    for im in ev['images']:
        if im['case_id'] not in {by[t]['case_id'] for t in targets}:raise ValueError('Image from outside development target')
    # Preserve every known development regression, plus two high-score same-node controls.
    rows=pilot.read(score['scores']);selected=set(targets)
    selected.update(r['token'] for r in source['plan'].get('initial_regressions',[]))
    history=[]
    for path in c.get('prior_memories',[]):
        snapshot=pilot.read(path);history+=snapshot['events']
        for event in snapshot['events']:
            selected.update(r['token'] for r in event.get('new_regressions',[]) if r['token'] in by and by[r['token']]['role']=='development')
    controls=[m for m in source['members'] if m['role']=='development' and m['skill']['id']==NODE and m['token'] not in selected]
    def quality(m):return statistics.mean(r['metrics']['PDMS'] for r in rows if r['token']==m['token'] and r['arm']=='incumbent')
    selected.update(m['token'] for m in sorted(controls,key=lambda m:(-quality(m),m['token']))[:2])
    members=[{**m,'repair_family':m['token'] in targets} for m in source['members']
             if m['role']=='validation' or m['token'] in selected]
    dev={acquisition_group(m['source_log']) for m in members if m['role']=='development'}
    val={acquisition_group(m['source_log']) for m in members if m['role']=='validation'}
    if dev & val or not val:raise ValueError('Cross-log isolation failed')
    attempts=c.get('max_doctor_requests',6)
    if type(attempts) is not int or not 1<=attempts<=6:raise ValueError('Bounded Doctor budget required')
    files=[str(config),c['source_prepared'],c['historical_scoring'],score['scores'],c['evidence'],*c.get('prior_memories',[])]
    files += [x['path'] for x in ev['images']]
    files += [str(root/n) for n in ['src/rsi/experiments/hard_case_repair.py','src/rsi/experiments/hard_case_evidence.py',
        'src/rsi/decisions/hard_case_repair.py','src/rsi/decisions/skill_doctor.py','src/rsi/decisions/grounded_skills.py',
        'src/rsi/decisions/skill_evolution.py','src/rsi/plugins/skill_evolution.py','agents/augustus/skill_inference.py',
        'src/rsi/experiments/frozen_skill_probe.py']]
    # Snapshot the model/native preprocessing identities again without rewriting old receipts.
    files += [p for p in source['plan']['files'] if '/weights/' in p or '/native/AUGUstus_SFT/' in p]
    plan=dict(schema_version=1,protocol='grounded_hard_case_v1',task_id=c['task_id'],members=members,
        target_tokens=targets,checkpoint=source['plan']['checkpoint'],checkpoint_sha256=source['plan']['checkpoint_sha256'],
        skills=source['plan']['skills'],seeds=[42,43],arms=['native','incumbent','candidate'],no_optimizer=True,
        gpu=c.get('gpu',0),gpu_memory_fraction=0.25,source_prepared=c['source_prepared'],historical_scores=score['scores'],
        evidence=c['evidence'],private_config=source['plan']['private_config'],credential_file=source['plan']['credential_file'],
        memory=str(out.parent/'memory.sqlite'),prior_history=history,max_doctor_requests=attempts,
        files={p:digest(p) for p in files},source_receipts=source['plan']['source_receipts'],
        max_sim_agent_steps=attempts*len(members)*2*3*80,
        scope='retrospective_navtest_hard_case_adaptation',validation_scope='fixed_cross_acquisition_log_previous_exposure',
        validation_once_then_stop=True,optimizer_updates=0)
    out.mkdir(parents=True,exist_ok=True)
    path=out/'plan.json'
    if path.exists() and pilot.read(path)!=plan:raise ValueError('Immutable plan changed')
    write_json(path,plan)
    task=dict(schema_version=1,id=c['task_id'],label='Grounded frozen AUGUstus hard-case repair',
        budget=dict(max_wall_seconds=14400,max_sim_agent_steps=plan['max_sim_agent_steps']),steps=[
            dict(id='automatic-search',provider='native_model_hard_case_search',inputs={},
                 config=dict(plan=str(path)),max_wall_seconds=14000)])
    write_json(out/'task.json',task)
    return plan


def subset(source,plan,members,incumbent,candidate,out):
    """Select cache rows; native dataset and preprocessing remain unchanged."""
    from datasets import load_from_disk
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    for name,sha in source['cache_files'].items():
        if digest(name)!=sha:raise ValueError('Historical native cache changed')
    path=out/'prepared.json'
    expected=dict(plan_hash=identity(plan),tokens=[m['token'] for m in members],
                  incumbent=identity(incumbent),candidate=identity(candidate))
    if path.exists():
        p=pilot.read(path)
        if p['selection_identity']!=expected:raise ValueError('Stage identity changed')
        for name,sha in p['cache_files'].items():
            if digest(name)!=sha:raise ValueError('Stage cache changed')
        return path
    indices={m['token']:i for i,m in enumerate(source['members'])};copies=[]
    for m in members:
        m=copy.deepcopy(m);obs=m['skill']['evidence']
        m['arm_skills']={a:grounded.apply_gate(choose_skill(obs,p),p) for a,p in
                         [('incumbent',incumbent),('candidate',candidate)]}
        copies.append(m)
    cache=out/'cache'
    load_from_disk(str(Path(source['cache'])/'batch_0')).select([indices[m['token']] for m in members]).save_to_disk(str(cache/'batch_0'))
    prepared={**source,'plan':plan,'members':copies,'cache':str(cache),'selection_identity':expected,
              'cache_files':{str(p):digest(p) for p in cache.rglob('*') if p.is_file()}}
    write_json(path,prepared);return path


def evaluate_stage(prepared,out):
    """Only known complete stages are reused; unknown partial execution fails closed."""
    from driversi_agents.augustus.skill_inference import run as infer
    out=Path(out);inf=out/'inference';sc=out/'scoring';result=sc/'report.json'
    if result.exists():
        r=pilot.read(result)
        if r['prepared_sha256']!=digest(prepared) or digest(r['scores'])!=r['scores_sha256']:
            raise ValueError('Scored stage identity changed')
        return r
    intent=out/'evaluation-intent.json'
    if intent.exists() and not (inf/'report.json').exists():
        raise RuntimeError('inference_outcome_unknown_requires_reconciliation')
    if not intent.exists():write_json(intent,dict(prepared_sha256=digest(prepared),started_at=time.time()))
    if pilot.read(intent)['prepared_sha256']!=digest(prepared):raise ValueError('Evaluation intent changed')
    infer(prepared,inf)
    gc.collect()
    import torch
    torch.cuda.empty_cache()
    score_intent=out/'scoring-intent.json'
    if score_intent.exists():raise RuntimeError('scoring_outcome_unknown_requires_reconciliation')
    write_json(score_intent,dict(prepared_sha256=digest(prepared),started_at=time.time()))
    return pilot.score(prepared,inf/'predictions.json',inf/'report.json',sc)


def history_context(history):
    values=[]
    for e in history:
        candidate=e.get('proposed_preset',e.get('candidate',{}))
        if NODE not in candidate.get('skills',{}):continue
        values.append(dict(text=candidate['skills'][NODE],eligibility=candidate.get('eligibility',{}).get(NODE),
            reasons=[r for r in e.get('decision',{}).get('reasons',[])
                     if not any(word in r.lower() for word in ('validation','cross_log','subgroup','transfer'))],
            screen=e.get('screen'),precheck=e.get('precheck'),
            development_feedback=e.get('development_feedback'),diagnosis=e.get('diagnosis')))
    # All actual contents are in the durable ledger; bounded prompt retains recent failures.
    return values[-8:]


def request_proposal(plan,ev,state,history,out):
    facts=dict(selected_skill_id=NODE,allowed_commands=list(grounded.COMMANDS),
        incumbent_text=state['active']['skills'][NODE],
        incumbent_eligibility=state['active'].get('eligibility',{}).get(NODE,IDENTITY_GATE),
        development=dict(evidence=ev['evidence'],cases=ev['cases']),
        persisted_experience=history_context(history),validation_examples_available=False,
        diagnostic_geometry_limits='Object heading and velocity use first cached observation; they are not event-time measurements. Event positions and overlap times come from logged future occupancy replay.',
        retrospective_diagnostics_not_student_inputs=True)
    images=[{**im,'role':'development','sensor_image':True} for im in ev['images']]
    job=dict(prompt=PROMPT,evidence=images,visual_protocol='development_real_history_v1',
        diagnosis_evidence_ids=[e['id'] for e in ev['evidence']],response_max_tokens=1600,facts=facts)
    saved=skill_doctor.cached_request(job,out,plan['private_config'],plan['credential_file'])
    proposal=None;error=None;candidate=state['active']
    try:
        if saved['finish_reason']!='stop':raise ValueError('Incomplete answer')
        proposal=json.loads(saved['content'])
        candidate=grounded.validate(proposal,NODE,state['active'],ev['evidence'])
        if state['active'].get('eligibility',{}).get(NODE) is None and candidate['eligibility'][NODE]==IDENTITY_GATE:
            candidate['eligibility'].pop(NODE)
            if not candidate['eligibility']:candidate.pop('eligibility')
    except (ValueError,TypeError,KeyError) as exc:error=type(exc).__name__+': '+str(exc)
    return dict(candidate=candidate,skill_id=NODE,target_tokens=plan['target_tokens'],valid=error is None,
        error=error,proposal=proposal,usage=saved['usage'],doctor_seconds=saved['seconds'],job_hash=identity(job))


def development_feedback(rows,members):
    by={(r['token'],r['seed'],r['arm']):r for r in rows};result=[]
    for m in members:
        result.append(dict(case_id=m['case_id'],current_state=m['skill']['evidence'],
            metrics={a:{k:statistics.mean(r['metrics'][k] for r in rows if r['token']==m['token'] and r['arm']==a)
                       for k in METRICS} for a in ('native','incumbent','candidate')}))
    return result


def run(plan_path,output):
    plan=pilot.validate_plan(plan_path);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    source=pilot.read(plan['source_prepared']);ev=pilot.read(plan['evidence'])
    state=memory(plan['memory'],plan['skills']);start_path=out/'started.json'
    if not start_path.exists():write_json(start_path,dict(time=time.time(),plan_sha256=digest(plan_path)))
    first=pilot.read(start_path)
    if first['plan_sha256']!=digest(plan_path):raise ValueError('Run identity changed')
    started=first['time'];attempts=[];steps=0;stop='proposal_budget_exhausted'
    dev=[m for m in plan['members'] if m['role']=='development'];val=[m for m in plan['members'] if m['role']=='validation']
    for n in range(plan['max_doctor_requests']):
        folder=out/f'attempt-{n:02d}';folder.mkdir(parents=True,exist_ok=True);done=folder/'decision.json'
        committed=next((e for e in state['events'] if e['id']==plan['task_id']+'/attempt-'+str(n)),None)
        if committed and not done.exists():
            write_json(done,committed['attempt_receipt'])
            write_json(out/'memory-snapshot.json',state)
        if done.exists():
            rec=pilot.read(done)
            if rec['plan_sha256']!=digest(plan_path):raise ValueError('Attempt belongs to another plan')
            attempts.append(rec);steps+=rec['sim_agent_steps']
            if rec['stop_search']:stop=rec['stop_reason'];break
            continue
        if state['revision']!=n:raise ValueError('Memory/attempt reconciliation required')
        write_json(out/'progress.json',dict(stage='doctor',attempt=n+1,max_attempts=plan['max_doctor_requests'],
            completed_attempts=len(attempts),optimizer_updates=0))
        history=plan['prior_history']+state['events'];proposed=request_proposal(plan,ev,state,history,folder/'doctor')
        write_json(folder/'proposal.json',proposed)
        check=rules.precheck(proposed,state['active'],dev,history) if proposed['valid'] else dict(passed=False,reasons=['invalid_proposal:'+proposed['error']])
        decision=dict(accepted=False,reasons=check['reasons']);screen=None;scored=[];cost=0;stop_search=False;stop_reason=None
        if check['passed']:
            write_json(out/'progress.json',dict(stage='development_screen',attempt=n+1,optimizer_updates=0))
            prep=subset(source,plan,dev,state['active'],proposed['candidate'],folder/'development')
            sr=evaluate_stage(prep,folder/'development');cost+=sr['sim_agent_steps'];scored=pilot.read(sr['scores'])
            screen=rules.screen(scored,dev,plan['seeds'],plan['target_tokens']);decision=screen
            if screen['accepted']:
                # Seal candidate before a single fixed transfer test. Never feedback validation into another proposal.
                write_json(folder/'transfer-selection.json',dict(candidate_hash=identity(proposed['candidate']),
                    development_scores_sha256=sr['scores_sha256'],selected_without_validation=True))
                write_json(out/'progress.json',dict(stage='cross_log_validation',attempt=n+1,optimizer_updates=0))
                vp=subset(source,plan,val,state['active'],proposed['candidate'],folder/'validation')
                vr=evaluate_stage(vp,folder/'validation');cost+=vr['sim_agent_steps']
                decision=rules.final_decision(scored+pilot.read(vr['scores']),plan['members'],plan['seeds'],plan['target_tokens'])
                stop_search=True;stop_reason='skill_promoted' if decision['accepted'] else 'transfer_rejected_holdout_consumed'
        rec=dict(attempt=n,plan_sha256=digest(plan_path),precheck=check,screen=screen,decision=decision,
            sim_agent_steps=cost,usage=proposed['usage'],doctor_seconds=proposed['doctor_seconds'],
            stop_search=stop_search,stop_reason=stop_reason,memory_revision=n+1,
            effective_candidate_hash=rules.fingerprint(proposed['candidate'],NODE))
        event=dict(id=plan['task_id']+'/attempt-'+str(n),skill_id=NODE,incumbent_hash=identity(state['active']),
            candidate=proposed['candidate'],decision=decision,precheck=check,screen=screen,
            development_feedback=development_feedback(scored,dev) if scored else [],
            diagnosis=proposed['proposal'].get('diagnosis') if isinstance(proposed.get('proposal'),dict) else None,
            attempt_receipt=rec)
        state=memory(plan['memory'],event=event,expected_revision=n)
        write_json(done,rec);write_json(out/'memory-snapshot.json',state);attempts.append(rec);steps+=cost
        if stop_search:stop=stop_reason;break
    totals={k:sum(a['usage'].get(k,0) for a in attempts) for k in ('prompt_tokens','completion_tokens','total_tokens')}
    report=dict(status='completed',stop_reason=stop,attempts=attempts,doctor_calls=len(attempts),usage=totals,
        sim_agent_steps=steps,scored_predictions=steps//80,wall_seconds=time.time()-started,
        target_scenes=len(plan['target_tokens']),development_scenes=len(dev),validation_scenes=len(val),
        active_version=state['active']['version'],skill_promoted=stop=='skill_promoted',
        optimizer_updates=0,parameters_frozen=True,production_policy_changed=False,scope=plan['scope'],
        validation_scope=plan['validation_scope'],weights_sha256=plan['checkpoint_sha256'])
    write_json(out/'report.json',report)
    write_json(out/'progress.json',dict(stage='completed',**{k:report[k] for k in ('stop_reason','scored_predictions','skill_promoted')}))
    return report


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('output');a=p.parse_args()
    v=declare(a.config,a.output);print(json.dumps(dict(members=len(v['members']),targets=len(v['target_tokens']),max_requests=v['max_doctor_requests'])))
