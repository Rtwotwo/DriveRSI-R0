import copy
import datetime as dt
import json
import sqlite3

import pytest

from driversi.data.catalog import cluster_caption_repair as repair
from driversi.data.catalog import cluster_caption as captions


def value(n=13):
    return dict(scene_description_zh='前方为城市道路，路边可见车辆。'*10,
        attributes=[dict(dimension='road_layout',description=f'原文{i}：道路可见。',
            evidence_ids=[('CAM_F0_anchor' if i%2 else 'CAM_B0_anchor')],certainty='visible') for i in range(n)],
        retrieval_tags=['路侧车辆','前方道路','路侧设施'],uncertainties=[])


def test_lossless_grouping_preserves_every_clause_and_reference():
    original=value();before=copy.deepcopy(original)
    fixed,groups=repair.normalize_attributes(original)
    assert original==before and 4<=len(fixed['attributes'])<=10
    assert sorted(i for g in groups for i in g)==list(range(13))
    for item,indices in zip(fixed['attributes'],groups):
        source=[original['attributes'][i] for i in indices]
        assert all(x['description'] in item['description'] for x in source)
        assert set(item['evidence_ids'])=={e for x in source for e in x['evidence_ids']}
        assert {x['certainty'] for x in source}=={item['certainty']}
    response={'choices':[{'finish_reason':'stop','message':{'content':json.dumps(fixed)}}]}
    decoded=captions.decode({'evidence':[{'id':'CAM_F0_anchor'},{'id':'CAM_B0_anchor'}]},'model',response)
    assert not decoded['indexable'] and not decoded['trainable']


def test_cross_camera_merge_has_per_clause_labels():
    original=value(11)
    for i,item in enumerate(original['attributes']):
        item['dimension']=sorted(captions.DIMENSIONS)[i//2]
    fixed,groups=repair.normalize_attributes(original)
    group=next((a,g) for a,g in zip(fixed['attributes'],groups) if len(g)>1)
    assert '[CAM_F0_anchor]' in group[0]['description']
    assert '[CAM_B0_anchor]' in group[0]['description']


def test_does_not_truncate_or_weaken_certainty_or_length():
    original=value(11)
    for i,item in enumerate(original['attributes']):
        item['description']='原始细节'*28
    with pytest.raises(ValueError,match='dropping text'):repair.normalize_attributes(original)
    original=value();original['attributes'][0]['description']='长'*121
    with pytest.raises(ValueError,match='invalid individual'):repair.normalize_attributes(original)


@pytest.mark.parametrize('hour,factor',[(7,1),(8,2),(21,2),(22,1)])
def test_price_boundary_uses_beijing_provider_timestamp(hour,factor):
    stamp=int(dt.datetime(2026,10,27,hour,tzinfo=dt.timezone(dt.timedelta(hours=8))).timestamp())
    result=repair.response_cost({'created':stamp,'usage':{'prompt_tokens':1000000,'completion_tokens':1000000}})
    assert result['estimated_cny']==5*factor


def test_missing_timestamp_is_not_misreported_as_free():
    with pytest.raises(ValueError):repair.response_cost({'usage':{'prompt_tokens':10,'completion_tokens':2}})


def finished_fixture(tmp_path):
    library=tmp_path/'library';library.mkdir();campaign=tmp_path/'campaign';campaign.mkdir()
    source=tmp_path/'source.json';source.write_text(json.dumps({'log_token':{'solution':{'prediction':'OFFICIAL EXACT TARGET'},'id':1}}))
    index=tmp_path/'index.json';index.write_text(json.dumps({'split':'navtrain','complete':True,'expected_tokens':['token'],
        'records':[dict(token='token',source_log='log',kind='original_navtrain',log_path='unused',sensor_root='unused',anchor_index=4,image_files=[])]}))
    captions.initialize(index,source,library)
    b=campaign/'batches/batch-00000';(b/'api/results').mkdir(parents=True);(b/'api/attempts').mkdir()
    job={'token':'token','input_identity':'same-input','facts':{'anchor_speed_kmh':20,'navigation_command':'GO STRAIGHT'},
        'evidence':[{'id':'CAM_F0_anchor'},{'id':'CAM_B0_anchor'}]}
    (b/'jobs.json').write_text(json.dumps([job]))
    result=b/'api/results/request.json'
    result.write_text(json.dumps(dict(status='rejected',token='token',input_identity='same-input',request_identity='request',profile={'model':'test-model'})))
    raw={'choices':[{'finish_reason':'stop','message':{'content':json.dumps(value())}}],
         'created':1793041151,'usage':{'prompt_tokens':100,'completion_tokens':50}}
    (b/'api/attempts/request-0.json').write_text(json.dumps(dict(status='response_received',request_identity='request',input_identity='same-input',attempt=0,response=raw)))
    db=captions.connect(library)
    with db:db.execute("UPDATE captions SET status='response_failed',receipt=?,error='ValueError: Invalid attributes'",(str(result),))
    db.close()
    (library/'shards').mkdir();original=json.loads(source.read_text())
    (library/'shards/batch-00000.jsonl').write_text(json.dumps(dict(annotation_key='log_token',**original['log_token'],cluster_caption={'status':'response_failed'}))+'\n')
    captions.export(library)
    (campaign/'progress.json').write_text(json.dumps({'status':'complete_with_failures'}))
    db=sqlite3.connect(campaign/'spend.sqlite3');db.execute('CREATE TABLE calls(status TEXT)');db.execute("INSERT INTO calls VALUES('received')");db.commit();db.close()
    return library,campaign,source,result


def test_real_entry_repairs_without_api_and_preserves_original_targets(tmp_path,monkeypatch):
    from driversi.data.catalog import bailian_caption as client
    monkeypatch.setattr(client,'request',lambda *_:pytest.fail('Offline repair must not call API'))
    library,campaign,source,receipt=finished_fixture(tmp_path)
    original=source.read_bytes();raw=receipt.read_bytes()
    first=repair.finalize(library,campaign,tmp_path/'audit',expected_count=1)
    second=repair.finalize(library,campaign,tmp_path/'audit2',expected_count=1)
    assert first['offline_repairs_applied_this_run']==1 and second['offline_repairs_applied_this_run']==0
    assert first['native_training_fields_unchanged'] and first['shard_members']==1
    assert source.read_bytes()==original and receipt.read_bytes()==raw
    assert first['cost']['snapshot_complete'] and first['cost']['received_responses']==1


def test_finalizer_recovers_after_db_commit_before_export(tmp_path,monkeypatch):
    library,campaign,_,_=finished_fixture(tmp_path);original_export=captions.export
    monkeypatch.setattr(captions,'export',lambda *_:(_ for _ in ()).throw(OSError('simulated export interruption')))
    with pytest.raises(OSError):repair.finalize(library,campaign,tmp_path/'audit',expected_count=1)
    monkeypatch.setattr(captions,'export',original_export)
    result=repair.finalize(library,campaign,tmp_path/'audit',expected_count=1)
    assert result['counts']=={'machine_draft':1} and result['native_training_fields_unchanged']


def test_active_producer_blocks_apply_and_native_mutation_is_detected(tmp_path):
    library,campaign,_,_=finished_fixture(tmp_path)
    (campaign/'progress.json').write_text(json.dumps({'status':'running'}))
    with pytest.raises(ValueError,match='not completed'):repair.finalize(library,campaign,tmp_path/'audit',expected_count=1)
    extended=json.loads((library/'autodrive-p3.with-cluster-caption.json').read_text())
    extended['log_token']['solution']['prediction']='CORRUPTED'
    (library/'autodrive-p3.with-cluster-caption.json').write_text(json.dumps(extended))
    with pytest.raises(ValueError,match='Native training label'):repair.verify_library(library,campaign,expected_count=1)


def test_one_shot_wait_uses_matching_process_exit(tmp_path):
    import subprocess
    import sys
    import os
    campaign=tmp_path/'campaign';campaign.mkdir()
    script=tmp_path/'annotate_cluster_captions.py'
    script.write_text('import time\ntime.sleep(0.5)\n')
    process=subprocess.Popen([sys.executable,str(script),'run','--output',str(campaign)])
    try:
        repair.wait_for_producer(process.pid,campaign,tmp_path/'audit')
        assert process.wait(timeout=2)==0
        state=json.loads((tmp_path/'audit/finalizer-status.json').read_text())
        assert state['mechanism']=='single_pidfd_exit_event' and state['periodic_monitoring'] is False
    finally:
        if process.poll() is None:process.terminate();process.wait(timeout=2)
