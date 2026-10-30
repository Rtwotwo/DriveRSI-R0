"""Offline receipt audit for the bounded frozen-skill evolution experiment."""
import json
from pathlib import Path

from driversi.agents.backends.native import digest
from driversi.runtime.budget import write_json
from driversi.rsi.decisions.skill_evolution import acquisition_group, identity


def build(directory):
    root=Path(directory);read=lambda p:json.loads(Path(p).read_text())
    declaration=read(root/'declaration/plan.json');task=read(root/'declaration/task.json')
    run=root/'runs'/task['id'];state=read(run/'report.json')
    if state['status']!='completed': raise ValueError('Cannot summarize an incomplete Harness run')
    groups={role:{acquisition_group(m['source_log']) for m in declaration['members'] if m['role']==role}
            for role in ['development','validation-0','validation-1']}
    assert not any(groups[a]&groups[b] for a,b in [('development','validation-0'),('development','validation-1'),('validation-0','validation-1')])
    episodes=[]
    for n in range(declaration['episodes']):
        proposal=read(run/f'proposal-{n}/report.json');inference=read(run/f'inference-{n}/report.json')
        scoring=read(run/f'scoring-{n}/report.json');decision=read(run/f'decision-{n}/report.json')
        doctor=read(run/f'proposal-{n}/doctor/job.json');response=read(run/f'proposal-{n}/doctor/response.json')
        snapshot=read(run/f'decision-{n}/memory-snapshot.json')
        assert decision['memory_after_hash']==identity(snapshot)
        prior = declaration.get('initial_memory_revision', 0)
        expected_experiences=0 if declaration.get('memory_mode')=='reset' else min(prior+n,3) if declaration.get('doctor_protocol')=='grounded_conditional_v1' else prior+n
        assert snapshot['revision']==prior+n+1 and proposal['experience_count']==expected_experiences
        assert identity(doctor)==proposal['request_job_hash']==response['job_hash']
        assert len(doctor['facts']['persisted_experience'])==expected_experiences
        assert not doctor['facts']['validation_examples_available']
        assert inference['parameters_frozen'] and inference['optimizer_updates']==0
        assert inference['checkpoint_sha256']==declaration['checkpoint_sha256']
        assert scoring['scores_sha256']==digest(scoring['scores'])
        prepared=read(proposal['prepared']);members={m['token']:m for m in prepared['members']}
        if declaration.get('caption_library'):
            from driversi.rsi.experiments.skill_evolution import read_contexts
            contexts = read_contexts(prepared)
            assert contexts and doctor['facts']['caption_assisted_diagnosis']
            assert not (groups['validation-'+str(n)] & set(declaration['excluded_acquisition_groups']))
        predictions=read(run/f'inference-{n}/predictions.json')
        by={(r['token'],r['seed'],r['arm']):r for r in predictions}
        native_metadata_annotations=0
        for m in members.values():
            for seed in declaration['seeds']:
                native=by[m['token'],seed,'native']
                for arm in ('incumbent','candidate'):
                    enabled=m['arm_skills'][arm].get('enabled',True)
                    expected=(native['prompt'].rstrip()+'\n\nPlanning checklist for this observation:\n'+m['arm_skills'][arm]['text']) if enabled else native['prompt']
                    assert by[m['token'],seed,arm]['prompt']==expected
                    if not enabled:
                        assert by[m['token'],seed,arm]['native_fallback_verified']
                        assert by[m['token'],seed,arm]['raw_trajectory']==native['raw_trajectory']
                if native.get('skill') is not None:native_metadata_annotations+=1
        if n:
            previous=read(run/f'decision-{n-1}/memory-snapshot.json')
            assert proposal['memory_hash']==identity(previous)
            assert proposal['incumbent']==previous['active']
        episodes.append(dict(episode=n,skill_id=proposal['skill_id'],proposal_valid=proposal['valid'],
            proposed_text=proposal['candidate']['skills'][proposal['skill_id']],
            rationale=(proposal.get('proposal') or {}).get('rationale'),
            experience_read=proposal['experience_count'],decision=decision['decision'],
            native_prompt_identity_verified=True,
            native_rows_with_legacy_unapplied_skill_annotation=native_metadata_annotations,
            active_version=decision['active_version'],scored_predictions=scoring['completed'],
            diagnosis=proposal.get('diagnosis'),
            eligibility=proposal['candidate'].get('eligibility',{}).get(proposal['skill_id']),
            native_fallback_predictions=sum(bool(r.get('native_fallback_verified')) for r in predictions),
            doctor_reply_reused=(run/f'proposal-{n}/doctor/cache-adoption.json').exists(),
            frozen_witness=inference['witness'],doctor_usage=response['usage'],
            seconds=dict(doctor=response['seconds'],inference=inference['wall_seconds'],scoring=scoring['wall_seconds'])))
    report=dict(status='completed',model='AUGUstus frozen original SFT',optimizer_updates=0,
        checkpoint_sha256=declaration['checkpoint_sha256'],episodes=episodes,
        acquisition_groups={k:len(v) for k,v in groups.items()},group_overlap=0,
        api_requests=len(episodes),input_tokens=sum(e['doctor_usage'].get('prompt_tokens',0) for e in episodes),
        output_tokens=sum(e['doctor_usage'].get('completion_tokens',0) for e in episodes),
        scored_predictions=sum(e['scored_predictions'] for e in episodes),
        promoted_episodes=sum(e['decision']['accepted'] for e in episodes),
        wall_seconds=state['finished_at']-state['started_at'],memory_protocol_verified=True,
        memory_reuse_verified=declaration.get('memory_mode','retain')=='retain',
        memory_reset_verified=declaration.get('memory_mode')=='reset',
        production_policy_changed=False,full_navtest_improvement_established=False,
        memory_benefit_over_reset_control_established=False,
        initial_experiences=declaration.get('initial_memory_revision', 0),
        doctor_context=declaration.get('doctor_context', 'scores_and_state'),
        caption_summary=prepared.get('caption_summary'), student_caption_injection=False)
    report.update(doctor_protocol=declaration.get('doctor_protocol','text_v1'),
        memory_mode=declaration.get('memory_mode','retain'),validation_scope=declaration.get('validation_scope','fresh_acquisition_drives'),
        new_api_requests=sum(not e['doctor_reply_reused'] for e in episodes))
    report.update(proposal_opportunities=len(episodes),api_requests=report['new_api_requests'],
        reply_input_tokens=report['input_tokens'],reply_output_tokens=report['output_tokens'],
        input_tokens=sum(e['doctor_usage'].get('prompt_tokens',0) for e in episodes if not e['doctor_reply_reused']),
        output_tokens=sum(e['doctor_usage'].get('completion_tokens',0) for e in episodes if not e['doctor_reply_reused']))
    out=root/'analysis';out.mkdir(exist_ok=True);write_json(out/'report.json',report)
    lines=['# AUGUstus automatic frozen-skill pilot','',
        'Two autonomous Harness episodes. No SFT, optimizer updates, model replacement or production release.',
        '', '| Episode | Doctor node | Prior experiences read | Decision | Development candidate − incumbent | Validation candidate − incumbent |',
        '|---|---|---:|---|---:|---:|']
    for e in episodes:
        c=e['decision']['cohorts']
        lines.append(f"| {e['episode']+1} | {e['skill_id']} | {e['experience_read']} | {'promote locally' if e['decision']['accepted'] else 'rollback'} | {100*c['development']['comparisons']['incumbent']['mean_delta']:.4f} | {100*c['validation']['comparisons']['incumbent']['mean_delta']:.4f} |")
    for e in episodes:
        lines += ['',f"## Episode {e['episode']+1}",'',e['proposed_text'],'',
                  'Decision reasons: '+(', '.join(e['decision']['reasons']) or 'all declared gates passed')]
    lines += ['','## Scope','',
        f"{report['scored_predictions']} predictions; {report['proposal_opportunities']} proposal opportunities, {report['new_api_requests']} new private Doctor requests; wall time {report['wall_seconds']/60:.2f} minutes.",
        'Validation scope: '+report['validation_scope']+'. Cohorts are disjoint from development acquisition drives and each other. Prior-drive reuse is not pristine holdout evaluation. The initial incumbent was experimental; local promotion is not generalization proof.',
        'Native prompt identity is checked explicitly. Caption-assisted diagnosis, where enabled, uses additional offline current-anchor context; student images/navigation and router remain unchanged.',
        'This verifies automatic edit→validation→decision and the declared '+report['memory_mode']+' experience condition. It does not establish that persistent memory outperforms equal-budget memory reset, or that the full test score improves.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    return report


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('directory');args=parser.parse_args()
    result=build(args.directory)
    print(json.dumps({k:result[k] for k in ['status','promoted_episodes','api_requests','scored_predictions','wall_seconds']}))
