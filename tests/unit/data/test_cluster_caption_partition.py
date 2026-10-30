import copy
import json
import sqlite3

import pytest

from driversi.data.catalog import cluster_caption as captions
from driversi.data.catalog import cluster_caption_batch as batch
from driversi.data.catalog import cluster_caption_partition as partition
from driversi.data.catalog import cluster_caption_repair as repair


def audit():
    return dict(status='verified_with_unresolved_failures', expected_members=103288,
        source_members=103288,database_members=103288,exported_members=103288,shard_members=103288,
        missing_members=0,duplicate_members=0,native_training_fields_unchanged=True,
        original_source_and_copy_unchanged=True, counts={'machine_draft':103088,'response_failed':200},
        cost={'snapshot_complete':True,'spend_ledger_states':{'received':103088,'unknown':200}})


@pytest.mark.parametrize('field,value',[
    ('status','running'),('native_training_fields_unchanged',False),('missing_members',1),
    ('original_source_and_copy_unchanged',False),('database_members',103287),
    ('counts',{'machine_draft':103287,'pending':1}),
    ('counts',{'machine_draft':101288,'response_failed':2000}),
    ('cost',{'snapshot_complete':False,'spend_ledger_states':{}}),
    ('cost',{'snapshot_complete':True,'spend_ledger_states':{'reserved':1}})])
def test_prerequisite_blocks_incomplete_or_invalid_audit(tmp_path,field,value):
    data=audit();data[field]=value;p=tmp_path/'audit.json';p.write_text(json.dumps(data))
    with pytest.raises(ValueError):partition.prerequisite(p)


def test_audited_failures_remain_explicit_but_do_not_require_paid_retry(tmp_path):
    p=tmp_path/'audit.json';p.write_text(json.dumps(audit()))
    assert partition.prerequisite(p)['counts']['response_failed']==200


def test_real_test_preparation_never_invents_native_targets(tmp_path,monkeypatch):
    train=tmp_path/'train';train.mkdir();db=sqlite3.connect(train/'captions.sqlite3')
    db.execute('CREATE TABLE captions(token TEXT)');db.execute("INSERT INTO captions VALUES('training-only')")
    db.commit();db.close()
    sensors=tmp_path/'sensors';sensors.mkdir();camera_data={};files=[]
    for cam in captions.CAMERAS:
        image=sensors/(cam+'.jpg');image.write_bytes(b'image placeholder for file existence check')
        files.append(str(image));camera_data[cam]={'data_path':image.name}
    frame=dict(token='test-only',log_name='test-log',driving_command=[1,0,0,0],
        ego_dynamic_state=[2,0,0,0],cams=camera_data)
    monkeypatch.setattr(captions,'source_log',lambda _:([frame],'log-hash'))
    record=dict(token='test-only',source_log='test-log',kind='original_navtest',
        log_path='test.pkl',sensor_root=str(sensors),anchor_index=0,image_files=files)
    index=tmp_path/'index.json';index.write_text(json.dumps(dict(split='navtest',complete=True,
        records=[record],expected_tokens=['test-only'])))
    library=tmp_path/'test';output=tmp_path/'out'
    manifest,report=partition.initialize_navtest(index,train,library,output,expected_count=1)
    assert report['new_api_requests']==0 and report['checked_current_images']==4
    assert manifest['split']=='navtest' and not manifest['native_training_labels_available']
    original=json.loads((library/'navsim.original.json').read_text())
    row=next(iter(original.values()));assert row['solution'] is None and row['historical_trajectory'] is None
    assert row['driving_command']=='TURN LEFT'
    assert not (library/'autodrive-p3.original.json').exists()
    assert partition.initialize_navtest(index,train,library,output,expected_count=1)[0]==manifest
    db=sqlite3.connect(train/'captions.sqlite3');db.execute("INSERT INTO captions VALUES('test-only')")
    db.commit();db.close()
    with pytest.raises(ValueError,match='overlaps'):
        partition.initialize_navtest(index,train,library,output,expected_count=1)


def test_test_extension_retains_common_contract_and_partition():
    job=dict(split='navtest',facts={'anchor_speed_kmh':20,'navigation_command':'GO FORWARD'})
    receipt=dict(description={'scene_description_zh':'scene','attributes':[],'retrieval_tags':[],
        'uncertainties':[]},model='model',content_review_flags=[])
    result=captions.extension('test-token',job,receipt,'/batch/api/results/file.json')
    assert result['split']=='navtest' and not result['trainable'] and not result['indexable']
    assert result['native_prediction_target']=='not_available_in_test_metadata'
    train=captions.extension('train-token',{**job,'split':'navtrain'},receipt,'/batch/api/results/file.json')
    assert set(result)==set(train)


def test_partition_policy_cannot_change_model_or_image_contract(tmp_path):
    policy=json.loads((captions.CONFIG/'cluster-caption.json').read_text())
    policy['image_max_side']=384;p=tmp_path/'policy.json';p.write_text(json.dumps(policy))
    with pytest.raises(ValueError,match='only change request and cost'):
        batch.run(tmp_path/'library',tmp_path/'run','unused','unused',policy_file=p)


def test_navtest_offline_finalize_preserves_partition_and_metadata(tmp_path):
    library=tmp_path/'library';campaign=tmp_path/'campaign';campaign.mkdir()
    index=tmp_path/'index.json';index.write_text(json.dumps(dict(split='navtest',complete=True,
        expected_tokens=['token'],records=[dict(token='token',source_log='log',kind='original_navtest',
            log_path='unused',sensor_root='unused',anchor_index=0,image_files=[])])))
    source=tmp_path/'source.json';source.write_text(json.dumps({'log_token':{'solution':None,'driving_command':'GO FORWARD'}}))
    captions.initialize(index,source,library)
    job=dict(split='navtest',facts={},evidence=[])
    receipt=dict(model='m',description={'scene_description_zh':'scene','attributes':[],
        'retrieval_tags':[],'uncertainties':[]},content_review_flags=[])
    extension=captions.extension('token',job,receipt,'/a/b/c/file.json')
    db=captions.connect(library)
    with db:db.execute("UPDATE captions SET status='machine_draft',caption=?",(json.dumps(extension),))
    db.close()
    (library/'shards').mkdir();(library/'shards/batch-00000.jsonl').write_text(json.dumps(dict(
        annotation_key='log_token',solution=None,driving_command='GO FORWARD',cluster_caption=extension))+'\n')
    (campaign/'progress.json').write_text(json.dumps({'status':'complete'}))
    db=sqlite3.connect(campaign/'spend.sqlite3');db.execute('CREATE TABLE calls(status TEXT)');db.close()
    result=repair.finalize(library,campaign,tmp_path/'audit',expected_count=1)
    assert result['native_training_fields_unchanged'] and result['counts']=={'machine_draft':1}
    final=json.loads((library/'navsim.with-cluster-caption.json').read_text())
    assert final['log_token']['solution'] is None and final['log_token']['cluster_caption']['split']=='navtest'


def test_cached_train_job_cannot_be_sent_as_navtest(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from threading import Event
    from driversi.data.catalog import bailian_caption as client
    index=tmp_path/'index.json';source=tmp_path/'source.json';library=tmp_path/'library';output=tmp_path/'run'
    record=dict(token='t',source_log='l',kind='original_navtest',log_path='unused',
                sensor_root='unused',anchor_index=0,image_files=[])
    index.write_text(json.dumps(dict(split='navtest',complete=True,expected_tokens=['t'],records=[record])))
    original={'l_t':{'driving_command':'GO FORWARD','solution':None}}
    source.write_text(json.dumps(original));captions.initialize(index,source,library)
    directory=output/'batches/batch-00000';directory.mkdir(parents=True)
    job=dict(token='t',split='navtrain',scope='current_anchor_only',prompt='p',
        facts=dict(scope='current_anchor_only',future_included=False,anchor_speed_kmh=10,navigation_command='GO FORWARD'),
        evidence=[dict(id=cam+'_anchor',relative_time_s=0) for cam in captions.CAMERAS])
    (directory/'jobs.json').write_text(json.dumps([job]))
    monkeypatch.setattr(client,'run',lambda *a,**kw:pytest.fail('Wrong split must not reach API'))
    with pytest.raises(ValueError,match='Cached job'):
        batch.run_batch(0,['t'],library,output,original,'p',{}, {},'unused',SimpleNamespace(stop=Event()))
