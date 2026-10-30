"""Independent native GPU encoders with a bounded, split-local single writer.

Scheduling never changes image batch size, native preprocessing or vector layout.
Workers receive history records, not catalog handles; only the coordinator writes.
"""

import multiprocessing as mp
import os
import queue
import signal
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path

from .corpus import load_evidence, write_report
from .features import VisionEncoder, feature_profile
from .store import Catalog, encode, identity
from .vectors import VectorStore


def _worker(device, config, tasks, results, stop, prefetch, backend_factory):
    # Spawned process: set device before the first torch/CUDA import.
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    os.environ.setdefault('MKL_NUM_THREADS', '4')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    backend = None
    def load(job):
        start = time.monotonic()
        evidence, images, error = load_evidence(job['scene'], config)
        return job, evidence, images, error, time.monotonic() - start
    try:
        with ThreadPoolExecutor(max_workers=prefetch) as pool:
            while not stop.is_set():
                batch = tasks.get()
                if batch is None:
                    return
                futures = deque(pool.submit(load, j) for j in batch)
                while futures and not stop.is_set():
                    job, evidence, images, error, load_seconds = futures.popleft().result()
                    values, cost, state = None, {'load_seconds': load_seconds}, 'failed'
                    if error is None:
                        if job['existing'] == [identity(evidence), 'ready']:
                            state = 'reused'
                        else:
                            if backend is None:
                                start = time.monotonic()
                                backend = backend_factory(config)
                                results.put({'kind':'worker_ready', 'device':device,
                                             'startup_seconds':time.monotonic()-start})
                            # Encoder/runtime failures stop the job, rather than
                            # silently marking thousands of valid scenes bad.
                            values, encoded = backend(images)
                            cost.update(encoded)
                            state = 'ready'
                    result = {'kind':'result', 'split':job['split'], 'token':job['token'],
                              'evidence':evidence, 'values':values, 'cost':cost,
                              'state':state, 'error':error, 'device':device}
                    while not stop.is_set():
                        try:
                            results.put(result, timeout=1)
                            break
                        except queue.Full:
                            pass
                for future in futures:
                    future.cancel()
    except BaseException as exc:
        results.put({'kind':'fatal', 'device':device,
                     'error':type(exc).__name__ + ': ' + str(exc)[:2000]})


def dump_parallel(databases, config, output, *, devices, workers=4, tokens=None,
                  backend_factory=VisionEncoder, profile_factory=feature_profile):
    """Account for every selected token, with durable writes and exact reuse.

    `tokens` is a split->explicit-token-list for bounded validation only; omission
    means both full catalog populations. The same native backend is used either way.
    """
    import fcntl

    if not databases or not set(databases) <= {'navtrain', 'navtest'}:
        raise ValueError('Explicit split-local databases required')
    if len({str(Path(p).resolve()) for p in databases.values()}) != len(databases):
        raise ValueError('Training and test databases must be distinct')
    if not devices or any(type(d) is not int or d < 0 for d in devices) or len(set(devices)) != len(devices):
        raise ValueError('Explicit unique nonnegative CUDA device indices required')
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError('Use 1..8 CPU prefetch workers per device')
    if tokens is not None and set(tokens) != set(databases):
        raise ValueError('Token selections must exactly match database splits')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    data = profile_factory(config)
    profile = identity(data)
    write_report(output/'profile.json', data)
    context = mp.get_context('spawn')
    stop = context.Event()
    tasks = context.Queue(maxsize=2*len(devices))
    results = context.Queue(maxsize=4*len(devices))
    processes, previous, stores, catalogs, logs = [], {}, {}, {}, {}
    state = {'profile':profile, 'status':'running', 'complete':False, 'devices':devices,
             'workers_per_device':workers, 'splits':{}, 'worker_startup':{},
             'trainable':False, 'training_started':False,
             'timings':{'write_seconds':0., 'result_wait_seconds':0., 'encode_seconds':0.,
                        'load_worker_seconds':0.}}
    def request_stop(signum, frame):
        stop.set()
    with ExitStack() as stack:
        populations, selections, existing = {}, {}, {}
        # Both locks are held for the whole run; old serial dumpers cannot race.
        for split in sorted(databases):
            path = Path(databases[split])
            lock = stack.enter_context(Path(str(path)+'.feature.lock').open('a'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            catalog = Catalog(path, split)
            stack.callback(catalog.close)
            catalogs[split] = catalog
            stores[split] = VectorStore(catalog, writable=True)
            stores[split].profile(data)
            populations[split] = [r[0] for r in catalog.db.execute(
                'SELECT token FROM scenes ORDER BY source_log,token')]
            choice = populations[split] if tokens is None else tokens[split]
            if (tokens is not None and not choice) or len(choice) != len(set(choice)) or not set(choice) <= set(populations[split]):
                raise ValueError('Explicit tokens must be unique members of their catalog')
            chosen = set(choice)
            selections[split] = deque(t for t in populations[split] if t in chosen)
            existing[split] = {r[0]:list(r[1:]) for r in catalog.db.execute(
                'SELECT token,input_id,status FROM vectors WHERE profile=?', (profile,))}
            state['splits'][split] = {'population':len(populations[split]), 'requested':len(choice),
                'processed':0, 'ready':0, 'reused':0, 'failed':0, 'complete':False}
            target = output/(split+'-features')
            target.mkdir(exist_ok=True)
            logs[split] = stack.enter_context((target/'attempts.jsonl').open('a',buffering=1))
        if len(populations) == 2 and set(populations['navtrain']) & set(populations['navtest']):
            raise ValueError('Training and test token populations overlap')
        active = deque(sorted(selections))
        submitted, received = set(), set()
        batch_size = workers*2
        def check_workers():
            if not stop.is_set() and any(p.exitcode is not None for p in processes):
                raise RuntimeError('Encoder worker exited before all results were committed')
        def dispatch():
            # Fair alternating split chunks; preserves log locality within split.
            while active and len(submitted)-len(received) < batch_size*2*len(devices) and not stop.is_set():
                check_workers()
                split = active.popleft()
                batch = []
                for _ in range(min(batch_size,len(selections[split]))):
                    token = selections[split].popleft()
                    submitted.add((split,token))
                    batch.append({'split':split,'token':token,'scene':catalogs[split].scene(token),
                                  'existing':existing[split].get(token)})
                if selections[split]:
                    active.append(split)
                if batch:
                    tasks.put(batch)
        def save():
            state['elapsed_seconds'] = time.monotonic()-start
            write_report(output/'progress.json',state)
            for split, report in state['splits'].items():
                write_report(output/(split+'-features')/'report.json',
                             {**report, 'split':split,'profile':profile,'status':state['status'],
                              'elapsed_seconds':state['elapsed_seconds']})
        try:
            for sig in (signal.SIGINT,signal.SIGTERM):
                previous[sig] = signal.signal(sig,request_stop)
            for device in devices:
                p = context.Process(target=_worker, args=(device,config,tasks,results,stop,workers,backend_factory))
                p.start()
                processes.append(p)
            state['worker_pids'] = [p.pid for p in processes]
            save()
            dispatch()
            last_save = time.monotonic()
            while len(received) < sum(r['requested'] for r in state['splits'].values()) and not stop.is_set():
                check_workers()
                waiting = time.monotonic()
                try:
                    result = results.get(timeout=1)
                except queue.Empty:
                    if stop.is_set():
                        break
                    check_workers()
                    continue
                finally:
                    state['timings']['result_wait_seconds'] += time.monotonic()-waiting
                if result['kind'] == 'fatal':
                    raise RuntimeError('Encoder worker failed: '+result['error'])
                if result['kind'] == 'worker_ready':
                    state['worker_startup'][str(result['device'])] = result['startup_seconds']
                    continue
                key = (result['split'],result['token'])
                if key not in submitted or key in received:
                    raise RuntimeError('Unknown or duplicate worker result')
                split, token = key
                store = stores[split]
                status = result['state']
                before = time.monotonic()
                if status == 'reused':
                    if not store.completed(token,profile,result['evidence']):
                        raise RuntimeError('Resume evidence changed while worker was running')
                else:
                    store.put(token,profile,result['evidence'],result['values'],result['cost'],result['error'])
                state['timings']['write_seconds'] += time.monotonic()-before
                state['timings']['encode_seconds'] += result['cost'].get('encode_seconds',0)
                state['timings']['load_worker_seconds'] += result['cost']['load_seconds']
                logs[split].write(encode({k:result[k] for k in ('token','state','cost','error','device')})+'\n')
                received.add(key)
                report = state['splits'][split]
                report[status] += 1
                report['processed'] += 1
                if status == 'ready':
                    if 'first_encoded_result_seconds' not in state:
                        state['first_encoded_result_seconds'] = time.monotonic()-start
                    state['last_encoded_result_seconds'] = time.monotonic()-start
                report['complete'] = report['processed'] == report['requested']
                dispatch()
                if time.monotonic()-last_save >= 2:
                    save()
                    last_save = time.monotonic()
            state['complete'] = all(r['complete'] for r in state['splits'].values())
            state['status'] = ('complete_with_failures' if any(r['failed'] for r in state['splits'].values()) else 'complete') if state['complete'] else 'interrupted'
        except BaseException as exc:
            state.update(status='failed',error=type(exc).__name__+': '+str(exc)[:2000])
            raise
        finally:
            stop.set()
            # Wake idle workers; busy workers see stop after their current sample.
            for _ in processes:
                try:
                    tasks.put_nowait(None)
                except queue.Full:
                    break
            for p in processes:
                p.join(timeout=3)
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=3)
                if p.is_alive():
                    p.kill()
                    p.join()
            for channel in (tasks,results):
                channel.cancel_join_thread()
                channel.close()
            for sig, handler in previous.items():
                signal.signal(sig,handler)
            for split,store in stores.items():
                state['splits'][split]['stats'] = store.stats(profile)
            save()
    return state
