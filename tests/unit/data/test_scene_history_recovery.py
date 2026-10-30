import json
import pickle
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from driversi.data.catalog import history_recovery as recovery
from driversi.data.catalog.observations import history_evidence
from driversi.data.catalog.store import Catalog
from driversi.data.catalog.vectors import VectorStore

CONFIG={'cameras':['CAM_F0','CAM_L0','CAM_R0'],'max_side':128}
BASE={'kind':'qwen35_native_vision_features_v1','cameras':CONFIG['cameras'],'dimension':2}
ERROR='ValueError: Irregular history is not silently padded'


def scene(root,token,split='navtrain',regular=False):
    root.mkdir(parents=True,exist_ok=True)
    times=[0,500000,1000000,1500000] if regular else [0,500000,1500000,2000000]
    frames=[]
    for i,t in enumerate(times+[2500000]):
        name=f'{token}-{i}.png'
        Image.new('RGB',(160,80),(30*i,10,80)).save(root/name)
        frames.append({'token':token if i==3 else f'{token}-{i}', 'log_name':token,
                       'timestamp':t,'cams':{c:{'data_path':name} for c in CONFIG['cameras']}})
    path=root/'log.pkl';path.write_bytes(pickle.dumps(frames))
    return {'record':{'token':token,'source_log':token,'kind':'original_'+split,
                       'log_path':str(path),'sensor_root':str(root),'anchor_index':3},
            'descriptor':{'token':token,'source_log':token,'source_group':token,
                          'history':{'continuous':regular,'anchor_timestamp_us':times[-1],
                                     'actual_intervals_us':[b-a for a,b in zip(times,times[1:])]}}}


class Encoder:
    def __init__(self,config):pass
    def __call__(self,images):
        return np.array([[np.asarray(i).mean()+1,1] for i in images]),{}


class Forbidden:
    def __init__(self,config):raise AssertionError('Unexpected inference')


def make_catalog(tmp_path,monkeypatch,split='navtrain'):
    monkeypatch.setattr(recovery,'feature_profile',lambda _:BASE)
    a=scene(tmp_path/'a','a',split,regular=True)
    b=scene(tmp_path/'b','b',split)
    db=Catalog(tmp_path/'catalog.sqlite',split)
    db.ingest([a['record'],b['record']],[a['descriptor'],b['descriptor']],{'complete':True})
    store=VectorStore(db,writable=True);base=store.profile(BASE)
    evidence,images=history_evidence(a,CONFIG['cameras'],128)
    store.put('a',base,evidence,Encoder(CONFIG)(images)[0],{})
    store.put('b',base,[],None,{},ERROR)
    return db,base,a,b


def test_exact_native_image_transform_and_real_times(tmp_path):
    a=scene(tmp_path/'a','a',regular=True)
    ev,ims=history_evidence(a,CONFIG['cameras'],128)
    new,images=recovery.observed_history(a,CONFIG['cameras'],128,require_irregular=False)
    assert all(np.array_equal(x,y) for x,y in zip(ims,images))
    assert all({k:n[k] for k in e}==e for e,n in zip(ev,new))
    b=scene(tmp_path/'b','b')
    ev,ims=recovery.observed_history(b,CONFIG['cameras'],128)
    assert len(ims)==12 and [e['relative_time_s'] for e in ev[::3]]==[-2.,-1.5,-.5,0.]
    assert all(not e['synthetic'] and not e['temporal_training_admitted'] for e in ev)
    assert all(e['timestamp_us']<=2000000 for e in ev)
    with pytest.raises(ValueError,match='restricted'):
        recovery.observed_history(a,CONFIG['cameras'],128)


def test_profile_separation_resume_and_missing_image_invalidation(tmp_path,monkeypatch):
    db,base,a,b=make_catalog(tmp_path,monkeypatch)
    original=tuple(VectorStore(db).get('a',base))
    first=recovery.recover_batch(db,CONFIG,['b'],tmp_path/'first',backend_factory=Encoder)
    assert first['ready']==1 and first['complete']
    assert first['profile']!=base
    assert tuple(VectorStore(db).get('a',base))==original
    assert db.db.execute('SELECT status FROM vectors WHERE profile=? AND token=?',(base,'b')).fetchone()[0]=='failed'
    resumed=recovery.recover_batch(db,CONFIG,['b'],tmp_path/'resume',backend_factory=Forbidden)
    assert resumed['reused']==1 and resumed['ready']==0
    (tmp_path/'b/b-1.png').unlink()
    invalid=recovery.recover_batch(db,CONFIG,['b'],tmp_path/'missing',backend_factory=Forbidden)
    assert invalid['failed']==1
    with pytest.raises(ValueError,match='No ready'):
        VectorStore(db).get('b',first['profile'])
    db.close()


def test_readonly_overlay_can_retrieve_both_profiles_without_persistent_mutation(tmp_path,monkeypatch):
    db,base,a,b=make_catalog(tmp_path,monkeypatch)
    r=recovery.recover_batch(db,CONFIG,['b'],tmp_path/'run',backend_factory=Encoder)
    path=db.path;db.close()
    db=Catalog(path,'navtrain',readonly=True)
    recovery.enable_recovery_view(db,base,r['profile'])
    assert VectorStore(db).stats(base)=={'ready':2}
    report=VectorStore(db).search('training','b',base,limit=1,shortlist=2)
    recovery.retrieval_provenance(db,report,base,r['profile'])
    assert report['results'][0]['token']=='a'
    assert report['query_feature_provenance']['storage_profile']==r['profile']
    assert not report['results'][0]['nominal_2hz_pair']
    with pytest.raises(ValueError,match='purpose'):
        VectorStore(db).search('evaluation','b',base,limit=1,shortlist=2)
    db.close();db=Catalog(path,'navtrain',readonly=True)
    assert VectorStore(db).stats(base)=={'failed':1,'ready':1}
    with pytest.raises(ValueError,match='No ready'):VectorStore(db).get('b',base)
    db.close()


def test_recovery_rejects_wrong_source_and_nonmonotonic(tmp_path,monkeypatch):
    db,base,a,b=make_catalog(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='prior irregular'):
        recovery.recover_batch(db,CONFIG,['a'],tmp_path/'no',backend_factory=Forbidden)
    with db.db:
        record={**b['record'],'kind':'original_navtest'}
        db.db.execute('UPDATE scenes SET record=? WHERE token=?',(json.dumps(record),'b'))
    with pytest.raises(ValueError,match='split mismatch'):
        recovery.recover_batch(db,CONFIG,['b'],tmp_path/'split',backend_factory=Forbidden)
    path=Path(b['record']['log_path']);frames=pickle.loads(path.read_bytes())
    frames[2]['timestamp']=0;path.write_bytes(pickle.dumps(frames))
    with pytest.raises(ValueError,match='increase strictly'):
        recovery.observed_history(b,CONFIG['cameras'],128)
    db.close()


def test_harness_keeps_irregular_candidate_out_of_training(tmp_path,monkeypatch):
    from driversi.rsi.plugins.scene_catalog import prepare
    from driversi_agents.chainflow import benchmark
    db,base,a,b=make_catalog(tmp_path,monkeypatch)
    r=recovery.recover_batch(db,CONFIG,['b'],tmp_path/'run',backend_factory=Encoder)
    path=db.path;db.close()
    monkeypatch.setattr(benchmark,'official_filter',lambda _: (None,{'tokens':['a','b'],'log_names':['a','b']}))
    cfg={'database':str(path),'structured_tags':[],'visual_tags':[],'profile':None,
         'text':None,'limit':1,'score_run':None,'score_below':None,
         'history_recovery_profile':r['profile'],
         'vector':{'query_token':'a','profile':base,'shortlist':2,'exclude_source':True,'diverse_sources':True}}
    prepared,_=prepare(cfg,tmp_path,None)
    candidate=prepared['report']['candidates'][0]
    assert candidate['token']=='b' and candidate['next_step']=='source_history_repair_required'
    assert not candidate['trainable']
    assert not prepared['report']['synthesis_authorized']


def test_encoder_failure_is_counted_before_abort(tmp_path,monkeypatch):
    class Broken:
        def __init__(self,config):pass
        def __call__(self,images):raise RuntimeError('encoder failed')
    db,base,a,b=make_catalog(tmp_path,monkeypatch)
    output=tmp_path/'broken'
    with pytest.raises(RuntimeError,match='encoder failed'):
        recovery.recover_batch(db,CONFIG,['b'],output,backend_factory=Broken)
    report=json.loads((output/'report.json').read_text())
    assert report['failed']==1 and not report['complete'] and report['status']=='failed'
    assert report['rows']==[{'token':'b','status':'failed','error':'RuntimeError: encoder failed'}]
    assert db.db.execute('SELECT status FROM vectors WHERE profile=? AND token=?',
                         (report['profile'],'b')).fetchone()[0]=='failed'
    db.close()


@pytest.mark.parametrize('missing_image',[False,True])
def test_storage_failure_cannot_erase_attempted_failure_report(tmp_path,monkeypatch,missing_image):
    db,base,a,b=make_catalog(tmp_path,monkeypatch)
    if missing_image:(tmp_path/'b/b-1.png').unlink()
    def broken_write(*args,**kwargs):raise OSError('storage unavailable')
    monkeypatch.setattr(VectorStore,'put',broken_write)
    output=tmp_path/'write-failed'
    with pytest.raises(OSError,match='storage unavailable'):
        recovery.recover_batch(db,CONFIG,['b'],output,backend_factory=Encoder)
    report=json.loads((output/'report.json').read_text())
    assert report['failed']==1 and not report['complete'] and report['status']=='failed'
    assert report['rows'][0]['token']=='b' and report['rows'][0]['status']=='failed'
    db.close()


@pytest.mark.parametrize('split',['navtrain','navtest'])
def test_registered_main_collection_survives_reopen(tmp_path,monkeypatch,split):
    from driversi.data.catalog.retrieval_collection import register_collection
    db,base,a,b=make_catalog(tmp_path,monkeypatch,split)
    r=recovery.recover_batch(db,CONFIG,['b'],tmp_path/'run',backend_factory=Encoder)
    raw=[tuple(x) for x in db.db.execute('SELECT * FROM vectors ORDER BY profile,token')]
    receipt=register_collection(db,base,r['profile'])
    assert register_collection(db,base,r['profile'])==receipt
    assert raw==[tuple(x) for x in db.db.execute('SELECT * FROM vectors ORDER BY profile,token')]
    path=db.path;db.close()
    reader=Catalog(path,split,readonly=True)
    assert VectorStore(reader).stats(base)=={'ready':2}
    # Existing explicit callers remain compatible with the registered default.
    recovery.enable_recovery_view(reader,base,r['profile'])
    result=VectorStore(reader).search('training' if split=='navtrain' else 'evaluation',
                                    'b',base,limit=1,shortlist=2)
    assert result['query_feature_provenance']['observed_irregular_history']
    assert result['query_feature_provenance']['storage_profile']==r['profile']
    assert not result['results'][0]['nominal_2hz_pair']
    assert reader.scene('b')['descriptor']['history']['continuous'] is False
    assert reader.db.execute('SELECT status FROM main.vectors WHERE profile=? AND token=?',
                             (base,'b')).fetchone()[0]=='failed'
    reader.close()
    # Extraction and maintenance still see raw native rows, not rewritten ones.
    writer=Catalog(path,split)
    assert VectorStore(writer).stats(base)=={'failed':1,'ready':1}
    writer.close()


@pytest.mark.parametrize('failure',['wrong_split','bad_bytes','synthetic'])
def test_main_collection_registration_fails_closed(tmp_path,monkeypatch,failure):
    from driversi.data.catalog.retrieval_collection import register_collection,KEY
    db,base,a,b=make_catalog(tmp_path,monkeypatch)
    r=recovery.recover_batch(db,CONFIG,['b'],tmp_path/'run',backend_factory=Encoder)
    with db.db:
        if failure=='wrong_split':
            db.db.execute('UPDATE scenes SET record=? WHERE token=?',
                          (json.dumps({**b['record'],'kind':'original_navtest'}),'b'))
        elif failure=='bad_bytes':
            db.db.execute('UPDATE vectors SET centroid=? WHERE profile=?',(b'bad',r['profile']))
        else:
            from driversi.data.catalog.store import identity
            evidence=json.loads(VectorStore(db).get('b',r['profile'])['evidence'])
            evidence[0]['synthetic']=True
            db.db.execute('UPDATE vectors SET evidence=?,input_id=? WHERE profile=?',
                          (json.dumps(evidence),identity(evidence),r['profile']))
    with pytest.raises(ValueError):register_collection(db,base,r['profile'])
    assert db.db.execute('SELECT value FROM metadata WHERE key=?',(KEY,)).fetchone() is None
    db.close()


def test_registered_main_collection_preserves_harness_pilot_scope(tmp_path,monkeypatch):
    from driversi.data.catalog.retrieval_collection import register_collection
    from driversi.rsi.plugins.scene_catalog import _training_catalog
    from driversi_agents.chainflow import benchmark
    db,base,a,b=make_catalog(tmp_path,monkeypatch)
    r=recovery.recover_batch(db,CONFIG,['b'],tmp_path/'run',backend_factory=Encoder)
    register_collection(db,base,r['profile'])
    path=db.path;db.close()
    monkeypatch.setattr(benchmark,'official_filter',lambda _: (None,{'tokens':['a','b'],'log_names':['a','b']}))
    pilot=_training_catalog({'database':str(path),'pilot_tokens':['b']},tmp_path)
    assert [x[0] for x in pilot.db.execute('SELECT token FROM scenes')]==['b']
    assert VectorStore(pilot).get('b',base)['status']=='ready'
    assert VectorStore(pilot).search('training','b',base,limit=1,shortlist=2)['results']==[]
    pilot.close()
