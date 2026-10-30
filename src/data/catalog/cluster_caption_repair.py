"""Offline, lossless caption repair and native-annotation coverage verification."""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import concurrent.futures
import ctypes
import fcntl
import functools
import json
import os
import select
import sqlite3
from pathlib import Path

from . import cluster_caption as captions
from .bailian_caption import write
from driversi.rendering.whitebox.exporter import sha256


def normalize_attributes(value):
    """Group original clauses, never rewrite/drop words or lower a validation limit."""
    value = copy.deepcopy(value)
    items = value['attributes']
    if not isinstance(items, list) or not all(isinstance(x, dict) for x in items):
        raise ValueError('Attributes are not objects')
    for item in items:
        if (set(item) != {'dimension','description','evidence_ids','certainty'}
                or item['dimension'] not in captions.DIMENSIONS
                or item['certainty'] not in ('visible','uncertain')
                or not isinstance(item['description'],str)
                or not 0 < len(item['description']) <= 120
                or not isinstance(item['evidence_ids'],list)
                or not item['evidence_ids']
                or not all(isinstance(x,str) for x in item['evidence_ids'])):
            raise ValueError('Cannot losslessly group invalid individual attribute')
    groups = [[i] for i in range(len(items))]

    def merged(indices):
        source = [items[i] for i in indices]
        equal_refs = all(set(x['evidence_ids']) == set(source[0]['evidence_ids']) for x in source)
        prose = '\n'.join(x['description'] if equal_refs else
                          '['+'/'.join(x['evidence_ids'])+'] '+x['description'] for x in source)
        return dict(dimension=source[0]['dimension'],certainty=source[0]['certainty'],description=prose,
                    evidence_ids=list(dict.fromkeys(e for x in source for e in x['evidence_ids'])))

    while len(groups) > 10:
        options = []
        for a in range(len(groups)):
            for b in range(a+1,len(groups)):
                x,y = items[groups[a][0]],items[groups[b][0]]
                if (x['dimension'],x['certainty']) != (y['dimension'],y['certainty']):continue
                indices = sorted(groups[a]+groups[b]);combined = merged(indices)
                if len(combined['description']) <= 120:
                    equal = set(x['evidence_ids']) == set(y['evidence_ids'])
                    options.append((not equal,len(combined['description']),a,b,indices))
        if not options:raise ValueError('Cannot reduce attribute count without dropping text or evidence')
        _,_,a,b,indices = min(options)
        groups[a] = indices;groups.pop(b)
    value['attributes'] = [merged(g) for g in groups]
    return value, groups


@functools.lru_cache(maxsize=32)
def batch_jobs(path):
    return {j['token']:j for j in json.loads(Path(path).read_text())}


def repair_record(row):
    result_file = Path(row['receipt'])
    result = json.loads(result_file.read_text())
    if result.get('status') != 'rejected':raise ValueError('Not a format-rejected response')
    batch = result_file.parents[2]
    job = batch_jobs(str(batch/'jobs.json'))[row['token']]
    if result['token'] != row['token'] or result['input_identity'] != job['input_identity']:
        raise ValueError('Receipt does not match the scene input')
    candidates = []
    for path in (batch/'api/attempts').glob(result['request_identity']+'-*.json'):
        attempt = json.loads(path.read_text())
        if (attempt.get('status') == 'response_received'
                and attempt['request_identity'] == result['request_identity']
                and attempt['input_identity'] == job['input_identity']):
            candidates.append((attempt['attempt'],path,attempt))
    if not candidates:raise ValueError('No received raw response to repair')
    _,attempt_file,attempt = max(candidates,key=lambda x:x[0])
    response = copy.deepcopy(attempt['response'])
    value = json.loads(response['choices'][0]['message']['content'])
    normalized,groups = normalize_attributes(value)
    if normalized == value:raise ValueError('No supported lossless repair applies')
    response['choices'][0]['message']['content'] = json.dumps(normalized,ensure_ascii=False)
    decoded = captions.decode(job,result['profile']['model'],response)
    caption = captions.extension(row['token'],job,decoded,str(result_file))
    caption['evidence_file'] = str(batch/'jobs.json')
    caption['offline_repair'] = dict(method='lossless_attribute_grouping_v1',original_receipt=str(result_file),
        original_attempt=str(attempt_file),original_error=row['error'],source_attribute_groups=groups,
        original_attribute_count=len(value['attributes']),new_attribute_count=len(normalized['attributes']),
        new_api_requests=0,factual_approval=False)
    return dict(token=row['token'],annotation_key=row['annotation_key'],original_receipt=str(result_file),
                caption=caption,original_response_retained=True,new_api_requests=0)


def prepare(library, output):
    library,output = Path(library),Path(output)
    db = sqlite3.connect(f'file:{library/"captions.sqlite3"}?mode=ro',uri=True)
    db.row_factory = sqlite3.Row
    try:rows = [dict(r) for r in db.execute("SELECT token,annotation_key,receipt,error FROM captions WHERE status='response_failed'")]
    finally:db.close()
    repaired=[];unresolved=[]
    for row in rows:
        try:
            if not str(row['error']).startswith('ValueError:'):raise ValueError('No usable format-rejected reply; do not retry API')
            repair=repair_record(row)
            write(library/'offline_repairs/v1'/f"{row['token']}.json",repair)
            repaired.append(row['token'])
        except (ValueError,KeyError,TypeError,OSError) as exc:
            unresolved.append(dict(token=row['token'],error=row['error'],reason=str(exc)))
    report=dict(scanned_failed=len(rows),repaired_sidecars=len(repaired),repaired_tokens=repaired,
        unresolved=unresolved,new_api_requests=0,applied=False,
        note='Only formatting changed; factual approval and production-index eligibility remain false.')
    write(output/'repair-report.json',report)
    return report


def response_cost(response):
    usage=response.get('usage') or {};p=usage.get('prompt_tokens');o=usage.get('completion_tokens')
    created=response.get('created')
    if type(p) is not int or type(o) is not int or type(created) is not int or min(p,o)<0:
        raise ValueError('Usage or provider timestamp missing')
    local=dt.datetime.fromtimestamp(created,dt.timezone(dt.timedelta(hours=8)))
    offpeak=local.hour>=22 or local.hour<8
    return dict(input_tokens=p,output_tokens=o,offpeak=offpeak,
                estimated_cny=(p+4*o)*(1 if offpeak else 2)/1e6,
                created_utc=created)


def cost_audit(campaign,output):
    """Read saved replies only; missing/ambiguous outcomes remain unpriced."""
    def one_batch(path):
        rows=[];unpriced=0
        for p in (path/'api/attempts').glob('*.json'):
            d=json.loads(p.read_text())
            if d.get('status')!='response_received':continue
            try:rows.append(response_cost(d['response']))
            except (ValueError,KeyError):unpriced+=1
        return rows,unpriced
    totals=dict(received_responses=0,input_tokens=0,output_tokens=0,offpeak_responses=0,
                peak_responses=0,estimated_cny=0.,unpriced_responses=0)
    earliest=None;latest=None
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for rows,unknown in pool.map(one_batch,sorted((Path(campaign)/'batches').glob('batch-*'))):
            totals['unpriced_responses']+=unknown
            for row in rows:
                totals['received_responses']+=1
                for k in ['input_tokens','output_tokens','estimated_cny']:totals[k]+=row[k]
                totals['offpeak_responses' if row['offpeak'] else 'peak_responses']+=1
                t=row['created_utc'];earliest=t if earliest is None else min(t,earliest);latest=t if latest is None else max(t,latest)
    db=sqlite3.connect(f'file:{Path(campaign)/"spend.sqlite3"}?mode=ro',uri=True)
    try:totals['spend_ledger_states']=dict(db.execute('SELECT status,COUNT(*) FROM calls GROUP BY status'))
    finally:db.close()
    totals['snapshot_complete']=(totals['received_responses']+totals['unpriced_responses'] ==
                                 totals['spend_ledger_states'].get('received',0))
    totals.update(first_response_timestamp=earliest,last_response_timestamp=latest,
        price_basis='Beijing public uncached input/output1/4CNY per million22:00-08:00;2/8otherwise',
        price_source='https://help.aliyun.com/zh/model-studio/model-pricing',
        invoice=False,unknown_outcomes_excluded=True,new_api_requests=0)
    write(Path(output)/'cost-audit.json',totals)
    return totals


def inventory(library, output):
    db = sqlite3.connect(f"file:{Path(library)/'captions.sqlite3'}?mode=ro", uri=True)
    try:
        counts = dict(db.execute('SELECT status,COUNT(*) FROM captions GROUP BY status'))
        errors = db.execute("SELECT error,COUNT(*) FROM captions WHERE status LIKE '%failed' GROUP BY error").fetchall()
    finally:
        db.close()
    report = dict(counts=counts, errors=errors, new_api_requests=0, read_only=True)
    write(Path(output)/'inventory.json', report)
    return report


def verify_library(library, campaign, expected_count=None):
    library,campaign=Path(library),Path(campaign)
    manifest=json.loads((library/'manifest.json').read_text())
    if expected_count is None:
        expected_count={'navtrain':103288,'navtest':12146}[manifest['split']]
    expected_sha=manifest['inputs']['annotation_sha256']
    if sha256(library/manifest['original_file'])!=expected_sha:
        raise ValueError('Original annotation copy changed')
    if sha256(manifest['annotation_source'])!=expected_sha:
        raise ValueError('Official annotation source changed')
    original=json.loads((library/manifest['original_file']).read_text())
    if len(original)!=expected_count or manifest['scene_count']!=expected_count:
        raise ValueError('Source population differs from expected count')
    db=captions.connect(library)
    try:rows={r['annotation_key']:dict(r) for r in db.execute('SELECT * FROM captions')}
    finally:db.close()
    if set(rows)!=set(original) or len({r['token'] for r in rows.values()})!=expected_count:
        raise ValueError('Database coverage is not one-to-one')
    if any(r['status']=='pending' for r in rows.values()):raise ValueError('Pending members remain')
    expected_tokens=set()
    for key,row in rows.items():
        record=json.loads(row['record']);expected_tokens.add(row['token'])
        if (record['token']!=row['token'] or record['kind']!='original_'+manifest['split']
                or key!=record['source_log']+'_'+row['token']):raise ValueError('Source lineage differs')
    batches=json.loads((library/'batches.json').read_text());flat=[t for b in batches for t in b]
    if len(flat)!=expected_count or set(flat)!=expected_tokens:raise ValueError('Batch coverage differs')
    extended=json.loads((library/manifest['extended_file']).read_text())
    if set(extended)!=set(original):raise ValueError('Extended JSON membership differs')
    for key,item in extended.items():
        if {k:v for k,v in item.items() if k!='cluster_caption'}!=original[key]:
            raise ValueError('Native training label changed in extended JSON')
        row=rows[key];extension=item['cluster_caption']
        if row['caption']:
            if extension!=json.loads(row['caption']):raise ValueError('Exported caption differs from database')
            if extension.get('split')!=manifest['split']:raise ValueError('Caption partition differs from source')
        elif extension['status']!=row['status']:raise ValueError('Exported failure status differs')
    seen=set()
    for path in sorted((library/'shards').glob('batch-*.jsonl')):
        for line in path.open():
            item=json.loads(line);key=item.pop('annotation_key')
            if key in seen or key not in original:raise ValueError('Shard duplicate or foreign member')
            seen.add(key)
            if {k:v for k,v in item.items() if k!='cluster_caption'}!=original[key]:
                raise ValueError('Native training label changed in shard')
            if rows[key]['caption']:
                if item['cluster_caption']!=json.loads(rows[key]['caption']):raise ValueError('Shard caption differs')
            elif item['cluster_caption']['status']!=rows[key]['status']:raise ValueError('Shard failure status differs')
    if seen!=set(original):raise ValueError('Shard coverage incomplete')
    counts={s:sum(r['status']==s for r in rows.values()) for s in {r['status'] for r in rows.values()}}
    return dict(expected_members=expected_count,source_members=len(original),database_members=len(rows),
        exported_members=len(extended),shard_members=len(seen),missing_members=0,duplicate_members=0,
        native_training_fields_unchanged=True,original_source_and_copy_unchanged=True,counts=counts,
        factual_approval=False,new_api_requests=0)


def finalize(library,campaign,output,expected_count=None):
    library,campaign,output=Path(library),Path(campaign),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    with (campaign/'campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=json.loads((campaign/'progress.json').read_text())
        if state['status'] not in ('complete','complete_with_failures'):
            raise ValueError('Producer has not completed; no database changes allowed')
        repair_report=prepare(library,output)
        db=captions.connect(library);applied=[]
        try:
            if db.execute("SELECT COUNT(*) FROM captions WHERE status='pending'").fetchone()[0]:
                raise ValueError('Producer still has pending members')
            for token in repair_report['repaired_tokens']:
                row=dict(db.execute('SELECT * FROM captions WHERE token=?',(token,)).fetchone())
                derived=json.loads((library/'offline_repairs/v1'/f'{token}.json').read_text())
                if row['status']!='response_failed' or row['receipt']!=derived['original_receipt']:
                    raise ValueError('Repair source status changed')
                caption=derived['caption'];caption['offline_repair']['applied']=True
                applied.append((json.dumps(caption,ensure_ascii=False),token))
            with db:db.executemany("UPDATE captions SET status='machine_draft',caption=?,error=NULL WHERE token=?",applied)
            # Include earlier applied repairs so a crash between DB commit and
            # shard/export replacement can be recovered without another API call.
            repaired_tokens=set()
            for path in (library/'offline_repairs/v1').glob('*.json'):
                row=db.execute('SELECT caption FROM captions WHERE token=?',(path.stem,)).fetchone()
                if row and row['caption'] and json.loads(row['caption']).get('offline_repair',{}).get('applied'):
                    repaired_tokens.add(path.stem)
            batches=json.loads((library/'batches.json').read_text())
            for number,tokens in enumerate(batches):
                if not repaired_tokens.intersection(tokens):continue
                path=library/'shards'/f'batch-{number:05d}.jsonl';items=[json.loads(s) for s in path.read_text().splitlines()]
                by_key={r['annotation_key']:r for r in db.execute('SELECT annotation_key,caption FROM captions WHERE token IN ('+','.join('?' for _ in tokens)+')',tokens)}
                for item in items:
                    row=by_key[item['annotation_key']]
                    if row['caption']:item['cluster_caption']=json.loads(row['caption'])
                temporary=path.with_suffix('.repair.tmp')
                temporary.write_text(''.join(json.dumps(i,ensure_ascii=False)+'\n' for i in items));temporary.replace(path)
        finally:db.close()
        captions.export(library)
        audit=verify_library(library,campaign,expected_count)
        cost=cost_audit(campaign,output)
        if not cost['snapshot_complete'] or cost['spend_ledger_states'].get('reserved',0):
            raise ValueError('Cost receipts and finished spend ledger disagree')
        audit.update(status='verified_with_unresolved_failures' if audit['counts'].get('response_failed',0) or audit['counts'].get('preparation_failed',0) else 'verified',
                     offline_repairs_applied_this_run=len(applied),cost=cost,
                     completed_at_beijing=dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(),
                     raw_receipts_unchanged=True,automatic_monitoring=False)
        write(output/'completion-audit.json',audit);write(library/'finalization.json',audit)
        return audit


def wait_for_producer(pid,campaign,output):
    """One kernel exit event, not periodic monitoring or automatic restarting."""
    campaign,output=Path(campaign).resolve(),Path(output)
    try:
        if hasattr(os,'pidfd_open'):
            fd=os.pidfd_open(pid)
        else:
            # This server's Python build omits the wrapper; Linux6.8 supports
            # the same exit event through syscall434 on x86_64/aarch64.
            if os.uname().machine not in ('x86_64','aarch64'):
                raise RuntimeError('No supported pidfd wrapper on this architecture')
            libc=ctypes.CDLL(None,use_errno=True);libc.syscall.restype=ctypes.c_long
            fd=libc.syscall(ctypes.c_long(434),ctypes.c_int(pid),ctypes.c_uint(0))
            if fd<0:
                error=ctypes.get_errno();raise OSError(error,os.strerror(error))
    except ProcessLookupError:return
    try:
        command=Path(f'/proc/{pid}/cmdline')
        if command.exists():
            args=command.read_bytes().decode().split('\0')
            if (not any(a.endswith('/annotate_cluster_captions.py') or a=='scripts/data/annotate_cluster_captions.py' for a in args)
                    or '--output' not in args or Path(args[args.index('--output')+1]).resolve()!=campaign):
                raise ValueError('PID does not match authorized producer')
        write(output/'finalizer-status.json',dict(status='waiting_for_producer_exit',pid=pid,
            mechanism='single_pidfd_exit_event',periodic_monitoring=False,new_api_requests=0))
        select.select([fd],[],[])
    finally:os.close(fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('inventory','prepare','cost','finalize'),default='inventory')
    parser.add_argument('--library', required=True)
    parser.add_argument('--campaign')
    parser.add_argument('--output', required=True)
    parser.add_argument('--wait-for-pid',type=int)
    args = parser.parse_args()
    if args.mode=='finalize':
        output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
        with (output/'finalization.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:
                if args.wait_for_pid:wait_for_producer(args.wait_for_pid,args.campaign,output)
                result=finalize(args.library,args.campaign,output)
                write(output/'finalizer-status.json',dict(status='complete',new_api_requests=0))
            except BaseException as exc:
                write(output/'finalizer-status.json',dict(status='blocked',error=type(exc).__name__+': '+str(exc),new_api_requests=0));raise
    elif args.mode=='prepare':result=prepare(args.library,args.output)
    elif args.mode=='cost':result=cost_audit(args.campaign,args.output)
    else:result=inventory(args.library,args.output)
    print(json.dumps({k:v for k,v in result.items() if k not in ('repaired_tokens','unresolved')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
