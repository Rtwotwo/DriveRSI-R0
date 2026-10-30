"""Bounded parallel corpus annotation with one shared spend and rate controller."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import signal
import sqlite3
import threading
import time
from pathlib import Path

from . import bailian_caption as client
from . import cluster_caption as captions
from .store import identity
from driversi.rendering.whitebox.exporter import sha256


class SharedBudget:
    def __init__(self, output, policy):
        self.policy=policy;self.lock=threading.Lock();self.stop=threading.Event();self.next_start=0.;self.backoff_until=0.
        self.db=sqlite3.connect(Path(output)/"spend.sqlite3",timeout=60,check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS calls(id INTEGER PRIMARY KEY, status TEXT, cost REAL, prompt INTEGER, completion INTEGER, seconds REAL)")
        self.db.commit()
        # Unknown outcomes retain the whole reservation, not a guessed zero cost.
        self.reserve_cost=(policy["cost_reserved_input_tokens"]*policy["busy_input_cny_per_million"]+
                           policy["max_output_tokens"]*policy["busy_output_cny_per_million"])/1e6

    def snapshot(self):
        with self.lock:
            n,c,p,o,s=self.db.execute("SELECT COUNT(*),COALESCE(SUM(cost),0),COALESCE(SUM(prompt),0),COALESCE(SUM(completion),0),COALESCE(SUM(seconds),0) FROM calls WHERE status!='not_sent'").fetchone()
            unknown=self.db.execute("SELECT COUNT(*) FROM calls WHERE status IN ('reserved','unknown')").fetchone()[0]
        return dict(requests_started=n,estimated_busy_cost_cny=c,input_tokens=p,output_tokens=o,
                    api_elapsed_seconds_sum=s,unknown_or_inflight=unknown,cost_cap_cny=self.policy["cost_cap_cny"])

    def reserve(self):
        with self.lock:
            n,c=self.db.execute("SELECT COUNT(*),COALESCE(SUM(cost),0) FROM calls WHERE status!='not_sent'").fetchone()
            if self.stop.is_set() or n>=self.policy["max_requests"] or c+self.reserve_cost>self.policy["cost_cap_cny"]:
                self.stop.set();raise client.NotSent("campaign_budget_or_stop")
            with self.db:
                number=self.db.execute("INSERT INTO calls(status,cost) VALUES('reserved',?)",(self.reserve_cost,)).lastrowid
        interval=max(60/self.policy["requests_per_minute"],
                     60*(self.policy["reserved_input_tokens"]+self.policy["max_output_tokens"])/self.policy["tokens_per_minute"])
        while True:
            with self.lock:
                if self.stop.is_set():
                    with self.db:self.db.execute("UPDATE calls SET status='not_sent',cost=0 WHERE id=?",(number,))
                    raise client.NotSent("campaign_stopped_while_waiting")
                now=time.monotonic();delay=max(self.next_start,self.backoff_until)-now
                if delay<=0:
                    self.next_start=now+interval
                    return number
            # Recheck shared stop/backoff before dispatch, including already queued calls.
            self.stop.wait(min(delay,1.))

    def __call__(self,body,key,config):
        number=self.reserve();started=time.monotonic()
        try:
            response=client.request(body,key,config)
        except BaseException as exc:
            with self.lock:
                known=isinstance(exc,client.APIError) and exc.status in (401,403,429)
                with self.db:
                    self.db.execute("UPDATE calls SET status=?,cost=?,seconds=? WHERE id=?",
                        ("http_rejected" if known else "unknown",0. if known else self.reserve_cost,time.monotonic()-started,number))
                if isinstance(exc,client.APIError) and exc.status in (401,403):self.stop.set()
                if isinstance(exc,client.APIError) and exc.status==429:
                    self.backoff_until=max(self.backoff_until,time.monotonic()+60)
            raise
        usage=response.get("usage") or {}
        p,o=usage.get("prompt_tokens"),usage.get("completion_tokens")
        with self.lock:
            if type(p) is int and type(o) is int and p>=0 and o>=0:
                cost=(p*self.policy["busy_input_cny_per_million"]+o*self.policy["busy_output_cny_per_million"])/1e6
                if p>self.policy["reserved_input_tokens"]:self.stop.set()
                with self.db:
                    self.db.execute("UPDATE calls SET status='received',cost=?,prompt=?,completion=?,seconds=? WHERE id=?",
                                    (cost,p,o,time.monotonic()-started,number))
            else:
                with self.db:self.db.execute("UPDATE calls SET status='unknown',seconds=? WHERE id=?",(time.monotonic()-started,number))
        return response


def run_batch(number,tokens,library,output,original,prompt,policy,config,key,budget):
    directory=output/"batches"/f"batch-{number:05d}"
    directory.mkdir(parents=True,exist_ok=True)
    db=captions.connect(library)
    try:
        records=[dict(db.execute("SELECT * FROM captions WHERE token=?",(token,)).fetchone()) for token in tokens]
        by_token={r["token"]:r for r in records}
        if (directory/"jobs.json").exists():
            jobs=json.loads((directory/"jobs.json").read_text())
            expected={r['token'] for r in records if r['status']!='preparation_failed'}
            if len(jobs)!=len(expected) or {j['token'] for j in jobs}!=expected:
                raise ValueError('Cached job membership differs from batch')
            for job in jobs:
                row=by_token[job['token']]
                if (job['prompt']!=prompt or job['scope']!='current_anchor_only'
                    or job.get('split','navtrain')!=json.loads(row['record'])['kind'].removeprefix('original_')
                    or set(job['facts'])!={'scope','future_included','anchor_speed_kmh','navigation_command'}
                    or job['facts']['scope']!='current_anchor_only' or job['facts']['future_included'] is not False
                    or job['facts']['navigation_command']!=original[row['annotation_key']]['driving_command']
                    or [e['id'] for e in job['evidence']] != [cam+'_anchor' for cam in captions.CAMERAS]
                    or any(e['relative_time_s']!=0 for e in job['evidence'])):
                    raise ValueError('Cached job prompt/scope/current-camera evidence differs')
        else:
            jobs=[]
            for row in records:
                if budget.stop.is_set():break
                try:
                    jobs.append(captions.prepare_job(json.loads(row["record"]),original[row["annotation_key"]],
                                output,prompt,policy))
                except (ValueError,OSError,KeyError) as exc:
                    with db:db.execute("UPDATE captions SET status='preparation_failed',error=? WHERE token=?",
                                       (type(exc).__name__+': '+str(exc)[:300],row["token"]))
            # Never freeze an interrupted partial job list as a complete preparation.
            if budget.stop.is_set():return dict(batch=number,finished=False)
            client.write(directory/"jobs.json",jobs)
        per_batch={**config,"max_requests":max(1,2*len(tokens))}
        report=client.run(jobs,per_batch,directory/"api",key=key,transport=budget,decode=captions.decode)
        jobs_by_token={j["token"]:j for j in jobs}
        updates=[]
        for row in report["rows"]:
            token=row["token"]
            receipt=json.loads(Path(row["file"]).read_text()) if row.get("file") else row
            if receipt["status"]=="format_valid_unreviewed":
                caption=captions.extension(token,jobs_by_token[token],receipt,row["file"])
                # The batch's exact jobs are two levels above the results file.
                caption["evidence_file"]=str(directory/"jobs.json")
                updates.append(('machine_draft',json.dumps(caption,ensure_ascii=False),row['file'],None,token))
            elif receipt['status']=='not_sent':
                updates.append(('pending',None,row.get('file'),receipt.get('error'),token))
            else:
                updates.append(('response_failed',None,row.get('file'),receipt.get('error',receipt['status']),token))
        # Responses are already durable. One atomic batch avoids a NFS fsync per scene;
        # interruption before commit replays those receipts without another paid call.
        with db:db.executemany('UPDATE captions SET status=?,caption=?,receipt=?,error=? WHERE token=?',updates)
        states=[dict(db.execute("SELECT * FROM captions WHERE token=?",(token,)).fetchone()) for token in tokens]
        shard=Path(library)/"shards"/f"batch-{number:05d}.jsonl";shard.parent.mkdir(parents=True,exist_ok=True)
        with shard.with_suffix('.tmp').open('w') as stream:
            for row in states:
                extension=json.loads(row['caption']) if row['caption'] else dict(status=row['status'],error=row['error'],indexable=False,trainable=False)
                stream.write(json.dumps(dict(annotation_key=row['annotation_key'],
                    **original[row['annotation_key']],cluster_caption=extension),ensure_ascii=False)+'\n')
        shard.with_suffix('.tmp').replace(shard)
        summary=dict(batch=number,finished=all(r['status']!='pending' for r in states),rows=len(states),
                     drafts=sum(r['status']=='machine_draft' for r in states),
                     failed=sum(r['status'].endswith('failed') for r in states),api_report=str(directory/'api/report.json'))
        client.write(directory/'summary.json',summary)
        return summary
    finally:db.close()


def run(library,output,private_file,credential_file,workers=8,limit_batches=None,policy_file=None):
    import fcntl
    library,output=Path(library).resolve(),Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    with (output/'campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        policy=json.loads((captions.CONFIG/'cluster-caption.json').read_text())
        if policy_file:
            custom=json.loads(Path(policy_file).read_text())
            if set(custom)!=set(policy) or any(custom[k]!=v for k,v in policy.items()
                    if k not in ('max_requests','cost_cap_cny')):
                raise ValueError('Partition policy may only change request and cost budgets')
            if (type(custom['max_requests']) is not int or not 0<custom['max_requests']<=policy['max_requests']
                    or not isinstance(custom['cost_cap_cny'],(int,float))
                    or not 0<custom['cost_cap_cny']<=policy['cost_cap_cny']):
                raise ValueError('Partition budgets must be positive and bounded by original policy')
            policy=custom
        if not 1<=workers<=policy['workers']<=8:raise ValueError('Use1..8shard workers')
        config=client.private_config(private_file)
        if config['models']!=[policy['model']]:raise ValueError('Configured model differs from declared campaign')
        config.update(concurrency=policy['calls_per_worker'],requests_per_minute=600,
                      max_attempts=2,max_tokens=policy['max_output_tokens'],max_requests=128,enable_thinking=False)
        client.validate_config(config)
        prompt=(captions.CONFIG/'cluster-caption-prompt.md').read_text()
        profile=dict(policy=policy,prompt=prompt,library_manifest_sha256=sha256(library/'manifest.json'),
                     adapter_sha256=sha256(captions.__file__),runner_sha256=sha256(__file__),
                     client_sha256=sha256(client.__file__),endpoint_identity=identity(config['base_url']))
        if (output/'profile.json').exists() and json.loads((output/'profile.json').read_text())!=profile:
            raise ValueError('Changed campaign code/input: do not mix paid results; inspect before a new run')
        client.write(output/'profile.json',profile)
        key=client.credential(credential_file)
        manifest=json.loads((library/'manifest.json').read_text())
        original=json.loads((library/manifest['original_file']).read_text())
        if sha256(library/manifest['original_file'])!=manifest['inputs']['annotation_sha256']:
            raise ValueError('Original annotation copy changed')
        batches=json.loads((library/'batches.json').read_text())
        flat=[t for b in batches for t in b]
        check_db=captions.connect(library)
        try:population={r[0] for r in check_db.execute('SELECT token FROM captions')}
        finally:check_db.close()
        if len(flat)!=len(population) or set(flat)!=population:
            raise ValueError('Batch partition does not cover each training member exactly once')
        if limit_batches is not None and limit_batches<1:raise ValueError('Positive pilot batch limit required')
        selected=list(range(len(batches)))[:limit_batches]
        pending=[]
        for number in selected:
            done=output/'batches'/f'batch-{number:05d}'/'summary.json'
            if not done.exists() or not json.loads(done.read_text())['finished']:pending.append(number)
        budget=SharedBudget(output,policy);previous={}
        for sig in (signal.SIGTERM,signal.SIGINT):previous[sig]=signal.signal(sig,lambda *_:budget.stop.set())
        started=time.monotonic();db=captions.connect(library)
        state=dict(status='running',requested_total=manifest['scene_count'],selected_batches=len(selected),
                   total_batches=len(batches),workers=workers,calls_per_worker=config['concurrency'],
                   format_valid_is_not_factual_approval=True,source_split=manifest['split'])
        def save():
            if (output/'STOP').exists():
                state['stop_reason']='operator_stop_file';budget.stop.set()
            counts={r[0]:r[1] for r in db.execute('SELECT status,COUNT(*) FROM captions GROUP BY status')}
            state.update(counts=counts,usage=budget.snapshot(),session_elapsed_seconds=time.monotonic()-started,
                         updated_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
            client.write(output/'progress.json',state);client.write(library/'progress.json',state)
        try:
            save();next_batch=iter(pending);active={}
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                def dispatch():
                    while len(active)<workers and not budget.stop.is_set():
                        number=next(next_batch,None)
                        if number is None:break
                        future=pool.submit(run_batch,number,batches[number],library,output,original,prompt,policy,config,key,budget)
                        active[future]=number
                dispatch()
                while active:
                    done,_=concurrent.futures.wait(active,timeout=5,return_when=concurrent.futures.FIRST_COMPLETED)
                    for future in done:
                        active.pop(future)
                        try:result=future.result()
                        except BaseException:
                            budget.stop.set();raise
                        if result.get('rows',0)>=16 and result.get('failed',0)/result['rows']>policy['failure_stop_fraction']:
                            state['stop_reason']='batch_failure_fraction_exceeded';budget.stop.set()
                    save();dispatch()
            state['status']='stopped' if budget.stop.is_set() else ('pilot_complete' if limit_batches else 'complete')
            save()
            if state['status']=='complete' and state['counts'].get('pending',0):state['status']='incomplete'
            if state['status']=='complete' and any(v for k,v in state['counts'].items() if k.endswith('failed')):
                state['status']='complete_with_failures'
            captions.export(library);save()
        except BaseException as exc:
            budget.stop.set();state.update(status='failed',error_type=type(exc).__name__);save();raise
        finally:
            for sig,handler in previous.items():signal.signal(sig,handler)
            db.close();budget.db.close()
        return state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('initialize','run','export'))
    parser.add_argument('--library',required=True)
    parser.add_argument('--index');parser.add_argument('--annotations');parser.add_argument('--output')
    parser.add_argument('--private-config');parser.add_argument('--credential-file')
    parser.add_argument('--workers',type=int,default=8);parser.add_argument('--limit-batches',type=int)
    args=parser.parse_args()
    if args.operation=='initialize':result=captions.initialize(args.index,args.annotations,args.library)
    elif args.operation=='export':result={'file':str(captions.export(args.library))}
    else:result=run(args.library,args.output,args.private_config,args.credential_file,args.workers,args.limit_batches)
    print(json.dumps(result,ensure_ascii=False))
    if result.get('status') in ('failed','stopped','incomplete'):raise SystemExit(2)


if __name__=='__main__':main()
