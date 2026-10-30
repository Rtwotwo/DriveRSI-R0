import json
import os
import pickle
import signal
import threading
import time

import numpy as np
import pytest
from PIL import Image

from driversi.data.catalog.parallel import dump_parallel
from driversi.data.catalog.store import Catalog
from driversi.data.catalog.vectors import VectorStore

PROFILE = {'cameras':['CAM_F0'], 'dimension':2}
CONFIG = {'cameras':['CAM_F0'], 'max_side':128}


def fake_profile(config):
    return PROFILE


class FakeEncoder:
    def __init__(self, config):
        pass

    def __call__(self, images):
        return np.array([[float(np.asarray(im).mean())+1, 1] for im in images]), {'encode_seconds':0.01}


class ForbiddenEncoder:
    def __init__(self, config):
        raise RuntimeError('Resume must not instantiate encoder')


class DeadEncoder:
    def __init__(self, config):
        os._exit(3)


class SlowEncoder(FakeEncoder):
    def __call__(self, images):
        time.sleep(0.1)
        return super().__call__(images)


class OneDeadEncoder(FakeEncoder):
    def __call__(self, images):
        if os.environ['CUDA_VISIBLE_DEVICES']=='0':
            time.sleep(.3)
            os._exit(3)
        time.sleep(.01)
        return super().__call__(images)


def make_db(root, split, count=3):
    root.mkdir(parents=True,exist_ok=True)
    image = root/'image.png'
    Image.new('RGB',(128,128),(123,42,17)).save(image)
    frames = [{'token':f'{split}-{i}', 'log_name':split, 'timestamp':i*500000,
               'cams':{'CAM_F0':{'data_path':'image.png'}}} for i in range(count+3)]
    log = root/'log.pkl'
    log.write_bytes(pickle.dumps(frames))
    db = Catalog(root/'catalog.sqlite', split)
    rows = [{'token':f['token'], 'source_log':split, 'kind':'original_'+split,
             'anchor_index':i, 'log_path':str(log), 'sensor_root':str(root)}
            for i,f in enumerate(frames) if i>=3]
    desc = [{'token':f['token'], 'source_log':split, 'source_group':split,
             'history':{'continuous':True,'anchor_timestamp_us':f['timestamp']}}
            for f in frames[3:]]
    db.ingest(rows,desc,{'test':True})
    db.close()
    return root/'catalog.sqlite'


def run(databases, output, **kwargs):
    return dump_parallel(databases,CONFIG,output,devices=[0,1],workers=1,
                         profile_factory=fake_profile,**kwargs)


def test_spawn_split_isolation_parity_and_resume(tmp_path):
    dbs = {s:make_db(tmp_path/s,s) for s in ('navtrain','navtest')}
    first = run(dbs,tmp_path/'first',backend_factory=FakeEncoder)
    assert first['complete'] and all(v['ready']==3 for v in first['splits'].values())
    snapshots = {}
    for split,path in dbs.items():
        db = Catalog(path,split,readonly=True)
        snapshots[split] = list(db.db.execute('SELECT token,input_id,centroid,image_vectors FROM vectors ORDER BY token'))
        assert all(row[0].startswith(split) for row in snapshots[split])
        db.close()
    resumed = run(dbs,tmp_path/'resume',backend_factory=ForbiddenEncoder)
    assert all(v['reused']==3 for v in resumed['splits'].values())
    for split,path in dbs.items():
        db = Catalog(path,split,readonly=True)
        assert [tuple(r) for r in db.db.execute('SELECT token,input_id,centroid,image_vectors FROM vectors ORDER BY token')] == [tuple(r) for r in snapshots[split]]
        db.close()
    # Changed image bytes are not allowed to reuse an old ready embedding.
    Image.new('RGB',(128,128),(0,0,0)).save(tmp_path/'navtrain/image.png')
    changed = run(dbs,tmp_path/'changed',backend_factory=FakeEncoder)
    assert changed['splits']['navtrain']['ready']==3
    assert changed['splits']['navtest']['reused']==3


def test_worker_death_stops_and_recovers(tmp_path):
    dbs = {'navtrain':make_db(tmp_path/'train','navtrain')}
    with pytest.raises(RuntimeError,match='worker exited'):
        run(dbs,tmp_path/'dead',backend_factory=DeadEncoder)
    report = json.loads((tmp_path/'dead/progress.json').read_text())
    assert report['status']=='failed' and not report['complete']
    assert run(dbs,tmp_path/'retry',backend_factory=FakeEncoder)['complete']


def test_one_worker_death_detected_while_other_produces(tmp_path):
    dbs={'navtrain':make_db(tmp_path/'train','navtrain',100)}
    with pytest.raises(RuntimeError,match='worker exited'):
        run(dbs,tmp_path/'one-dead',backend_factory=OneDeadEncoder)
    report=json.loads((tmp_path/'one-dead/progress.json').read_text())
    assert report['status']=='failed'
    assert report['splits']['navtrain']['processed'] < 90


def test_signal_keeps_commits_and_resume(tmp_path):
    dbs = {'navtrain':make_db(tmp_path/'train','navtrain',80)}
    timer = threading.Timer(2, lambda:os.kill(os.getpid(),signal.SIGTERM))
    timer.start()
    try:
        partial = run(dbs,tmp_path/'partial',backend_factory=SlowEncoder)
    finally:
        timer.cancel()
    assert partial['status']=='interrupted' and not partial['complete']
    final = run(dbs,tmp_path/'resume',backend_factory=FakeEncoder)
    assert final['complete']
    assert final['splits']['navtrain']['reused'] == partial['splits']['navtrain']['ready']


def test_bad_input_and_locks(tmp_path):
    import fcntl
    train = make_db(tmp_path/'train','navtrain')
    with pytest.raises(ValueError,match='distinct'):
        run({'navtrain':train,'navtest':train},tmp_path/'same')
    with pytest.raises(ValueError,match='split differs'):
        run({'navtest':train},tmp_path/'wrong')
    with pytest.raises(ValueError,match='unique members'):
        run({'navtrain':train},tmp_path/'token',tokens={'navtrain':['navtest-3']})
    with open(str(train)+'.feature.lock','a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            run({'navtrain':train},tmp_path/'locked')


def test_invalid_history_excluded(tmp_path):
    path = make_db(tmp_path/'train','navtrain')
    db = Catalog(path,'navtrain')
    with db.db:
        db.db.execute("UPDATE scenes SET descriptor=json_set(descriptor,'$.history.continuous',json('false'))")
    db.close()
    report = run({'navtrain':path},tmp_path/'invalid',backend_factory=ForbiddenEncoder)
    assert report['complete'] and report['splits']['navtrain']['failed']==3


def test_coordinator_caption_order_and_quality_gate(tmp_path,monkeypatch):
    import importlib.util
    import sys
    from pathlib import Path
    from driversi.data.catalog import parallel
    path=Path(__file__).resolve().parents[3]/'src/data/catalog/corpus_cli.py'
    spec=importlib.util.spec_from_file_location('corpus_runner_test',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg=tmp_path/'config.json';cfg.write_text('{}')
    request={'databases':{'navtrain':'train.sqlite','navtest':'test.sqlite'},'devices':[0,1],
             'feature_config':str(cfg),'output':str(tmp_path/'out'),'workers':1,
             'caption_release':str(tmp_path/'no-release.json')}
    req=tmp_path/'request.json';req.write_text(json.dumps(request))
    monkeypatch.setattr(sys,'argv',['run_scene_corpus.py',str(req)])
    monkeypatch.setattr(module,'label_batch',lambda *a,**k:pytest.fail('Caption must not start'))
    complete={'complete':True,'profile':'p','splits':{s:{'complete':True} for s in request['databases']}}
    monkeypatch.setattr(parallel,'dump_parallel',lambda *a,**k:complete)
    module.main()
    state=json.loads((tmp_path/'out/progress.json').read_text())
    assert state['status']=='features_done_caption_quality_review_required'
    complete['complete']=False
    with pytest.raises(RuntimeError,match='coverage interrupted'):
        module.main()
    state=json.loads((tmp_path/'out/progress.json').read_text())
    assert state['status']=='stopped' and not state['caption_batches']
