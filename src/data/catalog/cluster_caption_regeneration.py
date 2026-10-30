"""Explicit failed-only regeneration and atomic primary-library publication."""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import contextlib
import datetime as dt
import fcntl
import json
import signal
import sqlite3
from pathlib import Path

from . import bailian_caption as client
from . import cluster_caption as captions
from . import cluster_caption_batch as batch
from . import cluster_caption_repair as repair
from driversi.rendering.whitebox.exporter import sha256


def read(path):
    return json.loads(Path(path).read_text())


def inventory(libraries, campaigns, output):
    """Freeze eligible failures once, so restart cannot expand the paid scope."""
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    path=output/'targets.json'
    if path.exists():
        plan=read(path)
        if plan['libraries']!={s:str(Path(p).resolve()) for s,p in libraries.items()}:
            raise ValueError('Existing regeneration belongs to different libraries')
        if plan['campaigns']!={s:str(Path(p).resolve()) for s,p in campaigns.items()}:
            raise ValueError('Existing regeneration belongs to different producer campaigns')
        return plan
    targets={};previous={};profiles={}
    for split,library in libraries.items():
        library=Path(library);manifest=read(library/'manifest.json')
        if manifest['split']!=split:raise ValueError('Library partition mismatch')
        audit=read(library/'finalization.json')
        if audit['status'] not in ('verified','verified_with_unresolved_failures'):
            raise ValueError('Original library needs completed audit before regeneration')
        if read(Path(campaigns[split])/'progress.json')['status'] not in ('complete','complete_with_failures'):
            raise ValueError('Original producer has not completed')
        previous[split]=audit;profiles[split]=read(Path(campaigns[split])/'profile.json')
        db=sqlite3.connect(f'file:{library/"captions.sqlite3"}?mode=ro',uri=True);db.row_factory=sqlite3.Row
        try:
            if db.execute("SELECT COUNT(*) FROM captions WHERE status='pending'").fetchone()[0]:
                raise ValueError('Original library still has pending members')
            targets[split]=[dict(row) for row in db.execute("SELECT token,annotation_key,record,receipt,error FROM captions WHERE status='response_failed' ORDER BY token")]
        finally:db.close()
        if any(not row['receipt'] for row in targets[split]):
            raise ValueError('Failure has no original receipt; inspect before regeneration')
    if profiles['navtrain']['prompt']!=profiles['navtest']['prompt']:
        raise ValueError('Source caption prompts differ')
    for key in ('model','image_max_side','jpeg_quality','max_output_tokens'):
        if profiles['navtrain']['policy'][key]!=profiles['navtest']['policy'][key]:
            raise ValueError('Source caption contracts differ')
    overlap={r['token'] for r in targets['navtrain']}&{r['token'] for r in targets['navtest']}
    if overlap:raise ValueError('Cross-split targets overlap')
    plan=dict(libraries={s:str(Path(p).resolve()) for s,p in libraries.items()},
        campaigns={s:str(Path(p).resolve()) for s,p in campaigns.items()},targets=targets,
        previous_finalization=previous,source_profiles=profiles,
        explicit_failed_regeneration=True,max_rounds=3,cost_cap_cny=20,
        original_successes_excluded=True,automatic_monitoring=False)
    client.write(path,plan)
    client.write(output/'inventory.json',dict(failed_targets={s:len(v) for s,v in targets.items()},new_api_requests=0))
    return plan


def original_job(row,split,profile):
    jobs=repair.batch_jobs(str(Path(row['receipt']).parents[2]/'jobs.json'))
    job=jobs[row['token']]
    if (job.get('split','navtrain')!=split or job['prompt']!=profile['prompt']
            or job['scope']!='current_anchor_only' or job['facts']['future_included'] is not False
            or [e['id'] for e in job['evidence']]!=[c+'_anchor' for c in captions.CAMERAS]
            or any(e['relative_time_s']!=0 for e in job['evidence'])):
        raise ValueError('Original job differs from frozen current-camera contract')
    # Evidence is reused, never replaced by generated images or other scenes.
    for evidence in job['evidence']:
        if sha256(evidence['file'])!=evidence['sha256']:
            raise ValueError('Original compressed camera evidence changed')
    return job


def apply_results(library, split, jobs, report, output):
    """Only failed rows can be updated; successful pre-existing rows are untouched."""
    by_token={j['token']:j for j in jobs};db=captions.connect(library)
    repaired=0;accepted=0;failed=0
    try:
        with db:
            for item in report['rows']:
                token=item['token'];job=by_token[token]
                row=db.execute('SELECT * FROM captions WHERE token=?',(token,)).fetchone()
                if not row or json.loads(row['record'])['kind']!='original_'+split:
                    raise ValueError('Reply does not belong to library')
                if row['status']=='machine_draft':continue
                if row['status']!='response_failed':raise ValueError('Only frozen failed rows may change')
                result=read(item['file']) if item.get('file') else item
                if item.get('file') and (result['token']!=token or result['input_identity']!=job['input_identity']):
                    raise ValueError('Regeneration receipt does not match frozen scene input')
                caption=None
                if result['status']=='format_valid_unreviewed':
                    caption=captions.extension(token,job,result,item['file'])
                    caption['evidence_file']=str(Path(item['file']).parents[2]/'jobs.json')
                elif result['status']=='rejected' and item.get('file'):
                    try:
                        fixed=repair.repair_record(dict(row,receipt=item['file'],error=result.get('error')))
                        caption=fixed['caption'];caption['offline_repair']['applied']=True;repaired+=1
                    except (ValueError,KeyError,TypeError,OSError):pass
                if caption:
                    if caption['split']!=split or caption['indexable'] or caption['trainable']:
                        raise ValueError('Regenerated caption violates partition/admission')
                    db.execute("UPDATE captions SET status='machine_draft',caption=?,receipt=?,error=NULL WHERE token=? AND status='response_failed'",
                        (json.dumps(caption,ensure_ascii=False),item['file'],token));accepted+=1
                else:
                    db.execute("UPDATE captions SET receipt=COALESCE(?,receipt),error=? WHERE token=? AND status='response_failed'",
                        (item.get('file'),result.get('error',result['status']),token));failed+=1
    finally:db.close()
    summary=dict(accepted=accepted,offline_repaired=repaired,failed=failed)
    client.write(Path(output)/'application.json',summary)
    return summary


def publish_shards(library, targets):
    """Replace existing shard paths; unrelated lines remain byte-for-byte unchanged."""
    library=Path(library);tokens={r['token'] for r in targets};changed=[]
    if not tokens:return changed
    db=captions.connect(library)
    try:
        for number,members in enumerate(read(library/'batches.json')):
            selected=tokens.intersection(members)
            if not selected:continue
            path=library/'shards'/f'batch-{number:05d}.jsonl'
            rows={row['annotation_key']:dict(row) for row in db.execute(
                'SELECT annotation_key,status,caption,error FROM captions WHERE token IN ('+','.join('?' for _ in selected)+')',tuple(selected))}
            lines=path.read_bytes().splitlines(keepends=True);replacement=[];seen=set()
            for line in lines:
                item=json.loads(line);key=item['annotation_key']
                if key not in rows:replacement.append(line);continue
                seen.add(key);row=rows[key]
                item['cluster_caption']=json.loads(row['caption']) if row['caption'] else dict(
                    status=row['status'],error=row['error'],indexable=False,trainable=False)
                replacement.append((json.dumps(item,ensure_ascii=False)+'\n').encode())
            if seen!=set(rows):raise ValueError('Affected primary shard membership mismatch')
            data=b''.join(replacement)
            if data!=b''.join(lines):
                temporary=path.with_suffix('.regeneration.tmp');temporary.write_bytes(data);temporary.replace(path);changed.append(number)
    finally:db.close()
    return changed


def pending_targets(library,targets):
    db=captions.connect(library)
    try:
        return [r for r in targets if db.execute('SELECT status FROM captions WHERE token=?',(r['token'],)).fetchone()[0]=='response_failed']
    finally:db.close()


def run_group(number,split,rows,plan,output,config,key,budget):
    directory=output/'batches'/f'batch-{number:05d}';directory.mkdir(parents=True,exist_ok=True)
    jobs=[original_job(r,split,plan['source_profiles'][split]) for r in rows]
    existing=directory/'jobs.json'
    if existing.exists() and read(existing)!=jobs:raise ValueError('Frozen retry group changed')
    client.write(existing,jobs);client.write(directory/'group.json',dict(split=split,tokens=[r['token'] for r in rows]))
    report=client.run(jobs,config,directory/'api',key=key,transport=budget,decode=captions.decode)
    return apply_results(plan['libraries'][split],split,jobs,report,directory)


def usage_by_split(output):
    result={s:dict(received_responses=0,input_tokens=0,output_tokens=0,estimated_cny=0.,unknown_outcomes=0,
                   http_rejected=0,new_api_requests=0,offpeak_responses=0,peak_responses=0) for s in ('navtrain','navtest')}
    for path in (Path(output)/'batches').glob('batch-*'):
        split=read(path/'group.json')['split'];s=result[split]
        for attempt in (path/'api/attempts').glob('*.json'):
            d=read(attempt)
            if d['status']!='not_sent':s['new_api_requests']+=1
            if d['status']=='response_received':
                cost=repair.response_cost(d['response']);s['received_responses']+=1
                s['offpeak_responses' if cost['offpeak'] else 'peak_responses']+=1
                for k in ('input_tokens','output_tokens','estimated_cny'):s[k]+=cost[k]
            elif d['status']=='api_failed' and str(d.get('error','')).endswith(('HTTP 401','HTTP 403','HTTP 429')):
                s['http_rejected']+=1
            elif d['status']!='not_sent':s['unknown_outcomes']+=1
    return result


def run(libraries,campaigns,output,private_config,credential_file):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    with contextlib.ExitStack() as stack:
        for path in [output/'run.lock']+[Path(campaigns[s])/'campaign.lock' for s in sorted(campaigns)]:
            lock=stack.enter_context(path.open('a'));fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=inventory(libraries,campaigns,output)
        policy=dict(plan['source_profiles']['navtrain']['policy'])
        policy.update(max_requests=max(1,sum(map(len,plan['targets'].values()))*3),cost_cap_cny=20)
        config=client.private_config(private_config)
        if config['models']!=[policy['model']]:raise ValueError('Model differs from frozen recipe')
        if any(client.identity(config['base_url'])!=p['endpoint_identity'] for p in plan['source_profiles'].values()):
            raise ValueError('Provider endpoint differs from original caption campaigns')
        config.update(concurrency=8,requests_per_minute=600,max_attempts=1,max_tokens=policy['max_output_tokens'],max_requests=64,enable_thinking=False)
        client.validate_config(config)
        profile=dict(policy=policy,model=policy['model'],prompt=plan['source_profiles']['navtrain']['prompt'],
            endpoint_identity=client.identity(config['base_url']),implementation_sha256=sha256(__file__),
            decoder_module_sha256=sha256(captions.__file__),client_sha256=sha256(client.__file__))
        if (output/'profile.json').exists() and read(output/'profile.json')!=profile:
            raise ValueError('Retry code or recipe changed; inspect instead of mixing paid calls')
        client.write(output/'profile.json',profile)
        key=client.credential(credential_file);budget=batch.SharedBudget(output,policy)
        previous={sig:signal.signal(sig,lambda *_:budget.stop.set()) for sig in (signal.SIGTERM,signal.SIGINT)}
        try:
            for split in libraries:
                client.write(Path(libraries[split])/'finalization.json',dict(status='regenerating',prior_audit=str(output/'targets.json'),automatic_monitoring=False))
                client.write(Path(libraries[split])/'progress.json',dict(status='regenerating',source_split=split,regeneration_run=str(output),automatic_monitoring=False))
            for round_number in range(3):
                group_file=output/f'round-{round_number+1}.json'
                if group_file.exists():groups=read(group_file)
                else:
                    groups=[]
                    for split_index,split in enumerate(('navtrain','navtest')):
                        remaining=pending_targets(libraries[split],plan['targets'][split])
                        for chunk in range(0,len(remaining),64):
                            groups.append(dict(number=round_number*10000+split_index*1000+chunk//64,split=split,rows=remaining[chunk:chunk+64]))
                    client.write(group_file,groups)
                if not groups or budget.stop.is_set():break
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                    futures=[]
                    for group in groups:
                        if pending_targets(libraries[group['split']],group['rows']):
                            futures.append(pool.submit(run_group,group['number'],group['split'],group['rows'],plan,output,config,key,budget))
                    for future in concurrent.futures.as_completed(futures):
                        result=future.result()
                        if result['failed']>max(3,(result['accepted']+result['failed'])*.2):budget.stop.set()
                for split in libraries:publish_shards(libraries[split],plan['targets'][split])
                remaining={s:len(pending_targets(libraries[s],plan['targets'][s])) for s in libraries}
                client.write(output/'progress.json',dict(status='running',completed_round=round_number+1,remaining=remaining,usage=budget.snapshot(),automatic_monitoring=False))
                if not any(remaining.values()):break
            costs=repair.cost_audit(output,output)
            if not costs['snapshot_complete']:raise ValueError('Received regeneration receipts do not reconcile')
            split_costs=usage_by_split(output);audits={}
            costs['new_api_requests']=sum(c['new_api_requests'] for c in split_costs.values())
            client.write(output/'cost-audit.json',costs)
            for split,library in libraries.items():
                publish_shards(library,plan['targets'][split]);captions.export(library)
                audit=repair.verify_library(library,campaigns[split])
                prior=plan['previous_finalization'][split];cost=split_costs[split]
                cumulative=dict(prior['cost'])
                for field in ('received_responses','input_tokens','output_tokens','estimated_cny','offpeak_responses','peak_responses'):
                    cumulative[field]=prior['cost'].get(field,0)+cost[field]
                states=dict(prior['cost']['spend_ledger_states'])
                for field,value in [('received',cost['received_responses']),('unknown',cost['unknown_outcomes']),('http_rejected',cost['http_rejected'])]:
                    if value:states[field]=states.get(field,0)+value
                cumulative.update(spend_ledger_states=states,invoice=False,unknown_outcomes_excluded=True,
                    source_reports=[str(output/'targets.json'),str(output/'cost-audit.json')],new_api_requests=cost['new_api_requests'])
                audit.update(status='verified' if not audit['counts'].get('response_failed',0) else 'verified_with_unresolved_failures',
                    completed_at_beijing=dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(),
                    prior_finalization=str(output/'targets.json'),regeneration_cost=cost,cost=cumulative,
                    estimated_cumulative_cny=cumulative['estimated_cny'],new_api_requests=cost['new_api_requests'],
                    raw_receipts_unchanged=True,automatic_monitoring=False)
                client.write(Path(library)/'finalization.json',audit)
                client.write(Path(library)/'progress.json',dict(status=audit['status'],counts=audit['counts'],stage='failed_caption_regeneration',audit_file=str(Path(library)/'finalization.json'),automatic_monitoring=False))
                client.write(output/f'{split}-verification.json',audit);audits[split]=audit
            report=dict(status='complete' if all(a['status']=='verified' for a in audits.values()) else 'complete_with_failures',
                audits=audits,new_cost=costs,primary_library_updated=True,separate_caption_dataset_created=False,
                automatic_monitoring=False,factual_approval=False)
            client.write(output/'report.json',report);client.write(output/'progress.json',dict(status=report['status'],counts={s:a['counts'] for s,a in audits.items()},usage=budget.snapshot(),automatic_monitoring=False))
            return report
        except BaseException as exc:
            budget.stop.set()
            client.write(output/'progress.json',dict(status='failed',error_type=type(exc).__name__,usage=budget.snapshot(),automatic_monitoring=False))
            raise
        finally:
            for sig,handler in previous.items():signal.signal(sig,handler)
            budget.db.close()


def apply_reviewed_descriptions(library, entries, audit_file):
    """Explicit offline visual authoring, never represented as a provider response."""
    library=Path(library);audit_file=Path(audit_file)
    split=read(library/'manifest.json')['split'];db=captions.connect(library);prepared=[];audited=[]
    try:
        for entry in entries:
            token=entry['job']['token'];job=entry['job'];value=entry['description']
            if job.get('split','navtrain')!=split or job['facts']['future_included'] is not False:
                raise ValueError('Reviewed scene partition or observation scope differs')
            if job['scope']!='current_anchor_only' or [x['id'] for x in job['evidence']]!=[c+'_anchor' for c in captions.CAMERAS]:
                raise ValueError('Reviewed caption requires the original four current cameras')
            for ev in job['evidence']:
                if ev['relative_time_s']!=0 or sha256(ev['file'])!=ev['sha256']:
                    raise ValueError('Reviewed image evidence changed')
            flags=captions.validate_description(job,value)
            row=db.execute('SELECT * FROM captions WHERE token=?',(token,)).fetchone()
            if not row or json.loads(row['record'])['kind']!='original_'+split:
                raise ValueError('Reviewed source differs from library')
            if row['status']!='response_failed':
                existing=json.loads(row['caption']) if row['caption'] else {}
                if (existing.get('description_origin')=='codex_visual_annotation_from_original_four_cameras'
                        and all(existing.get(k)==v for k,v in value.items())):
                    if existing.get('receipt_file')!=str(audit_file):raise ValueError('Reuse the original visual review audit path')
                    audited.append(token);continue
                raise ValueError('Only explicitly failed captions can receive visual authoring')
            if not str(row['error']).endswith('HTTP 400'):
                raise ValueError('This authoring path is only for the inspected persistent provider rejection')
            caption=captions.extension(token,job,dict(description=value,model='codex_visual_review',content_review_flags=flags),str(audit_file))
            caption.update(evidence_file=entry['evidence_file'],
                description_origin='codex_visual_annotation_from_original_four_cameras',
                review_status='assistant_visual_annotation_not_independently_reviewed',
                provider_generated=False,requested_provider_model='deepseek-v4.1-flash-1017',
                original_failed_receipt=row['receipt'],new_api_requests=0)
            prepared.append((token,caption));audited.append(token)
        if len(set(audited))!=len(audited):raise ValueError('Duplicate reviewed token')
        audit=dict(status='prepared',method='codex_visual_review',tokens=audited,
            count=len(audited),newly_applied=len(prepared),new_api_requests=0,provider_generated=False,
            factual_approval=False,indexable=False,trainable=False,automatic_monitoring=False)
        client.write(audit_file,audit)
        with db:
            for token,caption in prepared:
                db.execute("UPDATE captions SET status='machine_draft',caption=?,receipt=?,error=NULL WHERE token=? AND status='response_failed'",
                    (json.dumps(caption,ensure_ascii=False),str(audit_file),token))
        audit['status']='applied';client.write(audit_file,audit)
        return audit
    finally:db.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('operation',choices=('inventory','run'))
    for name in ('train-library','test-library','train-campaign','test-campaign','output','private-config','credential-file'):
        p.add_argument('--'+name,required=name not in ('private-config','credential-file'))
    a=p.parse_args();libraries={'navtrain':a.train_library,'navtest':a.test_library};campaigns={'navtrain':a.train_campaign,'navtest':a.test_campaign}
    result=inventory(libraries,campaigns,a.output) if a.operation=='inventory' else run(libraries,campaigns,a.output,a.private_config,a.credential_file)
    print(json.dumps(dict(status=result.get('status','inventoried'),targets={s:len(v) for s,v in result.get('targets',{}).items()}),ensure_ascii=False))


if __name__=='__main__':main()
