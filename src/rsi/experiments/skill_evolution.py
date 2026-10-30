"""Prepare and execute evidence-separated, frozen-policy skill evolution episodes."""
import copy
import hashlib
import json
from pathlib import Path

from driversi.agents.backends.native import digest, project_root
from driversi.runtime.budget import write_json
from driversi.rsi.experiments import frozen_skill_probe as pilot
from driversi.rsi.decisions.frozen_skills import choose_skill
from driversi.rsi.decisions.skill_evolution import ARMS, acquisition_group, evaluate, identity, memory, seed_memory
from driversi.rsi.decisions import skill_doctor


def declare(config, output):
    c = pilot.read(config); root = project_root(); out = Path(output).resolve()
    if not out.is_relative_to(root/'exp'): raise ValueError('Experiment output must be under exp')
    validation_size = c['validation_drives_per_episode']
    if c['episodes'] != 2 or type(validation_size) is not int or not 3 <= validation_size <= 8:
        raise ValueError('This bounded pilot requires two episodes and three to eight fresh drives each')
    initial = pilot.read(c['initial_prepared'])
    score = pilot.read(c['initial_scoring'])
    if score['prepared_sha256'] != digest(c['initial_prepared']) or digest(score['scores']) != score['scores_sha256']:
        raise ValueError('Initial evidence is not sealed')
    baseline = pilot.read(c['baseline_report']); baseline_plan = pilot.read(c['baseline_plan'])
    if baseline['plan_sha256'] != digest(c['baseline_plan']) or not baseline['complete']:
        raise ValueError('Historical baseline identity mismatch')
    if baseline_plan['inputs'][str(Path(initial['plan']['checkpoint'])/'model.safetensors')] != initial['plan']['checkpoint_sha256']:
        raise ValueError('Parent model mismatch')
    members = [{**m, 'role':'development'} for m in initial['members']]
    used = {acquisition_group(m['source_log']) for m in members}
    excluded = set()
    excluded_tokens = {m['token'] for m in members}
    validation_scope = c.get('validation_scope','fresh_acquisition_drives')
    if validation_scope not in ('fresh_acquisition_drives','new_anchors_prior_validation_drives'):
        raise ValueError('Unknown validation scope')
    for path in c.get('exclude_prepared', []):
        prior_members=pilot.read(path)['members']
        excluded_tokens.update(m['token'] for m in prior_members)
        if validation_scope=='fresh_acquisition_drives':
            excluded.update(acquisition_group(m['source_log']) for m in prior_members)
    used.update(excluded)
    selection_salt = c.get('selection_salt', 'skill-evolution-v1')
    ordered = sorted(baseline['rows'], key=lambda r: hashlib.sha256((selection_salt+'/'+r['token']).encode()).hexdigest())
    selected = []
    # Neither score nor score rank participates in validation membership.
    for row in ordered:
        group = acquisition_group(row['source_log'])
        if group in used or row['token'] in excluded_tokens: continue
        used.add(group)
        selected.append(dict(token=row['token'], source_log=row['source_log'],
            role='validation-'+str(len(selected)//validation_size), selection_stratum='score_blind_acquisition_group'))
        if len(selected) == 2*validation_size: break
    if len(selected) != 2*validation_size: raise ValueError('Insufficient independent validation drives')
    lookup = {r['token']: i for i, rows in enumerate(baseline_plan['chunks']) for r in rows}
    receipts = {}
    for member in selected:
        chunk = Path(c['baseline_plan']).parent/'chunks'/f"{lookup[member['token']]:04d}"
        cache = chunk/'data/receipt.json'; done = pilot.read(chunk/'completed.json')
        if digest(done['report']) != done['sha256']: raise ValueError('Historical metric receipt changed')
        files = pilot.read(done['report'])['inputs']['metric_files']
        metric = next(p for p in files if Path(p).parent.name == member['token'])
        member.update(cache_receipt=str(cache), metric_source=metric, metric_sha256=files[metric])
    members += selected
    for m in members: receipts[m['cache_receipt']] = digest(m['cache_receipt'])
    sources = list(initial['plan']['files']) + [str(config), c['initial_prepared'], c['initial_scoring'], score['scores']]
    sources += [str(root/p) for p in ('src/rsi/decisions/skill_evolution.py',
        'src/rsi/decisions/skill_doctor.py', 'src/rsi/experiments/skill_evolution.py',
        'src/rsi/plugins/skill_evolution.py', 'src/config/rsi/policies/augustus-skill-evolution.json')]
    sources += c.get('exclude_prepared', [])
    inherited = pilot.read(c['initial_memory']) if c.get('initial_memory') else None
    if inherited:
        if inherited['active'] != initial['plan']['skills']:
            raise ValueError('Inherited active skill needs matching sealed initial scores')
        sources.append(c['initial_memory'])
    if c.get('caption_library'):
        sources.append(str(root/'src/data/catalog/skill_context.py'))
    protocol=c.get('doctor_protocol','text_v1')
    memory_mode=c.get('memory_mode','retain')
    if protocol not in ('text_v1','grounded_conditional_v1') or memory_mode not in ('retain','reset'):
        raise ValueError('Unsupported Doctor protocol or memory condition')
    if protocol=='grounded_conditional_v1':
        sources.append(str(root/'src/rsi/decisions/grounded_skills.py'))
    for directory in c.get('cached_doctor_replies', {}).values():
        sources += [str(Path(directory)/n) for n in ('job.json','request-intent.json','response.json')]
    plan = dict(schema_version=1, checkpoint=initial['plan']['checkpoint'],
        checkpoint_sha256=initial['plan']['checkpoint_sha256'], skills=initial['plan']['skills'],
        seeds=[42,43], arms=list(ARMS), members=members, files={p:digest(p) for p in sources},
        source_receipts=receipts, initial_scores=score['scores'], private_config=c['private_config'],
        credential_file=c['credential_file'], memory=str(out.parent/'memory/state.sqlite'), episodes=2,
        scoring_agent_steps=(len(initial['members'])+validation_size)*2*3*80,
        total_scoring_agent_steps=2*(len(initial['members'])+validation_size)*2*3*80,
        max_doctor_requests=2, scope='navtest_adaptation_pilot', no_optimizer=True,
        validation_grouping='original_acquisition_drive', gpu=0,
        task_id=c['task_id'], validation_drives_per_episode=validation_size, excluded_acquisition_groups=sorted(excluded),
        initial_memory=c.get('initial_memory'), initial_memory_revision=inherited['revision'] if inherited else 0,
        doctor_protocol=protocol,memory_mode=memory_mode,validation_scope=validation_scope,
        gpu_memory_fraction=c.get('gpu_memory_fraction',0.60),
        excluded_validation_tokens=sorted(excluded_tokens),
        caption_library=c.get('caption_library'),
        cached_doctor_replies=c.get('cached_doctor_replies', {}),
        doctor_context='unverified_current_anchor_captions' if c.get('caption_library') else 'scores_and_state')
    out.mkdir(parents=True, exist_ok=True); path=out/'plan.json'
    if path.exists() and pilot.read(path) != plan: raise ValueError('Declaration is immutable')
    write_json(path, plan)
    task = dict(schema_version=1, id=c['task_id'], label='Frozen AUGUstus automatic skill evolution',
                budget=dict(max_wall_seconds=14400, max_sim_agent_steps=plan['total_scoring_agent_steps']), steps=[])
    def add(name, provider, inputs, episode, seconds):
        task['steps'].append(dict(id=name, provider=provider, inputs=inputs,
            config=dict(plan=str(path), episode=episode), max_wall_seconds=seconds))
    add('observations','skill_evolution_observations',{},0,2400)
    for n in range(2):
        inputs={'observations':'observations'}
        if n: inputs['previous']='decision-'+str(n-1)
        add('proposal-'+str(n),'skill_evolution_propose_initial' if n==0 else 'skill_evolution_propose_next',inputs,n,600)
        add('inference-'+str(n),'native_model_skill_evolution',{'proposal':'proposal-'+str(n)},n,2400)
        add('scoring-'+str(n),'skill_evolution_scoring',{'proposal':'proposal-'+str(n),'predictions':'inference-'+str(n)},n,1200)
        add('decision-'+str(n),'skill_evolution_decision',{'proposal':'proposal-'+str(n),'scores':'scoring-'+str(n)},n,300)
    write_json(out/'task.json',task)
    return plan


def prepare(plan_path, output):
    p = pilot.prepare_data(plan_path, output)
    plan = p['plan']
    p['initial_memory'] = (seed_memory(plan['memory'], plan['skills'], pilot.read(plan['initial_memory']))
                           if plan.get('initial_memory') else memory(plan['memory'], plan['skills']))
    if p['initial_memory']['revision'] != plan.get('initial_memory_revision', 0):
        raise ValueError('New pilot requires the declared initial memory revision')
    if plan.get('caption_library'):
        from driversi.data.catalog.skill_context import collect
        evidence = collect(p['members'], plan['caption_library'])
        path = Path(output)/'caption-context.json'
        write_json(path, evidence)
        p.update(caption_context=str(path), caption_context_sha256=digest(path), caption_summary=evidence['summary'])
    write_json(Path(output)/'report.json',p)
    return p


def proposal(observations, previous, episode, output):
    from datasets import load_from_disk
    source = pilot.read(observations); plan=source['plan']; out=Path(output); out.mkdir(parents=True,exist_ok=True)
    state=memory(plan['memory'])
    if state['revision'] != plan.get('initial_memory_revision', 0)+episode: raise ValueError('Unexpected memory revision')
    if episode == 0:
        if previous is not None: raise ValueError('Unexpected prior decision')
        scored=pilot.read(plan['initial_scores']); arm='matched_skill'
    else:
        before=pilot.read(previous)
        if before['episode'] != episode-1 or before['memory_after_hash'] != identity(state):
            raise ValueError('Previous decision does not match durable memory')
        scored=pilot.read(before['scores'])
        if digest(before['scores']) != before['scores_sha256']: raise ValueError('Prior feedback changed')
        arm='candidate' if before['decision']['accepted'] else 'incumbent'
    contexts = read_contexts(source)
    grounded=plan.get('doctor_protocol')=='grounded_conditional_v1'
    if grounded:
        from driversi.rsi.decisions import grounded_skills
        scored,arm=grounded_skills.feedback_rows(scored,arm,plan['memory_mode'],episode)
        development=grounded_skills.feedback(scored,source['members'],arm,contexts)
        proposed=grounded_skills.propose(development,state,out/'doctor',plan['private_config'],plan['credential_file'],
            mode=plan['memory_mode'],reuse_from=plan.get('cached_doctor_replies',{}).get(str(episode)))
    else:
        development=skill_doctor.feedback(scored,source['members'],arm,contexts)
        proposed=skill_doctor.propose(development,state,out/'doctor',plan['private_config'],plan['credential_file'],
                                    reuse_from=plan.get('cached_doctor_replies',{}).get(str(episode)))
    indices=[i for i,m in enumerate(source['members']) if m['role'] in ('development','validation-'+str(episode))]
    members=[]
    for i in indices:
        m=copy.deepcopy(source['members'][i]); m['role']='development' if m['role']=='development' else 'validation'
        observed=m['skill']['evidence']
        m['arm_skills']={a:choose_skill(observed,proposed[a]) for a in ('incumbent','candidate')}
        if grounded:
            m['arm_skills']={a:grounded_skills.apply_gate(m['arm_skills'][a],proposed[a]) for a in ('incumbent','candidate')}
        m['skill']=m['arm_skills']['candidate'];members.append(m)
    cache=out/'cache'
    load_from_disk(str(Path(source['cache'])/'batch_0')).select(indices).save_to_disk(str(cache/'batch_0'))
    prepared={**source,'members':members,'cache':str(cache),
        'cache_files':{str(f):digest(f) for f in cache.rglob('*') if f.is_file()}}
    write_json(out/'prepared.json',prepared)
    proposed.update(episode=episode,prepared=str(out/'prepared.json'),prepared_sha256=digest(out/'prepared.json'),
                    candidate_hash=identity(proposed['candidate']),sealed_before_inference=True)
    write_json(out/'report.json',proposed)
    return proposed


def decide(proposal_path, score_path, output):
    proposed=pilot.read(proposal_path); scoring=pilot.read(score_path)
    if digest(proposed['prepared']) != proposed['prepared_sha256']: raise ValueError('Proposal observations changed')
    prepared=pilot.read(proposed['prepared'])
    if scoring['prepared_sha256'] != proposed['prepared_sha256'] or digest(scoring['scores']) != scoring['scores_sha256']:
        raise ValueError('Decision evidence mismatch')
    rows=pilot.read(scoring['scores'])
    decision=evaluate(rows,prepared['members'],prepared['plan']['seeds'])
    if not proposed['valid']:
        decision['accepted']=False;decision['reasons'].append('invalid_doctor_proposal')
    if prepared['plan'].get('doctor_protocol')=='grounded_conditional_v1':
        from driversi.rsi.decisions import grounded_skills
        development=grounded_skills.feedback(rows,prepared['members'],'candidate',read_contexts(prepared))
    else:
        development=skill_doctor.feedback(rows,prepared['members'],'candidate',read_contexts(prepared))
    event=dict(id=prepared['plan'].get('task_id', 'evolution')+'/episode-'+str(proposed['episode']),skill_id=proposed['skill_id'],
        incumbent_hash=identity(proposed['incumbent']),candidate=proposed['candidate'],decision=decision,
        development=development,scores_sha256=scoring['scores_sha256'],proposal_hash=identity(proposed))
    if prepared['plan'].get('doctor_protocol')=='grounded_conditional_v1':
        event.update(diagnosis=proposed.get('diagnosis'),memory_mode=prepared['plan']['memory_mode'])
    state=memory(prepared['plan']['memory'],event=event,expected_revision=proposed['memory_revision'])
    report=dict(status='completed',episode=proposed['episode'],decision=decision,
        memory_after_hash=identity(state),memory_revision=state['revision'],active_version=state['active']['version'],
        experience_count=len(state['events']),proposal_valid=proposed['valid'],
        scores=scoring['scores'],scores_sha256=scoring['scores_sha256'],
        optimizer_updates=0,production_policy_changed=False)
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    write_json(out/'report.json',report);write_json(out/'memory-snapshot.json',state)
    return report


def read_contexts(prepared):
    if not prepared.get('caption_context'): return None
    path = prepared['caption_context']
    if digest(path) != prepared['caption_context_sha256']: raise ValueError('Caption context changed after preparation')
    value = pilot.read(path)
    if set(value['contexts']) != {m['token'] for m in prepared['members'] if m['role'] == 'development'}:
        raise ValueError('Caption context must contain development scenes only')
    return value['contexts']


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('config');parser.add_argument('output')
    args=parser.parse_args();p=declare(args.config,args.output)
    print(json.dumps(dict(members=len(p['members']),episodes=p['episodes'],predictions=p['total_scoring_agent_steps']//80)))
