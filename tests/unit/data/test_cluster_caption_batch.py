import copy
import hashlib
import json

import pytest

from driversi.data.catalog import cluster_caption as captions
from driversi.data.catalog import cluster_caption_batch as batch
from driversi.data.catalog import bailian_caption as client


def corpus(tmp_path, count=2):
    records=[];annotations={}
    for i in range(count):
        token=f'token{i}';log=f'log{i}'
        records.append(dict(token=token,source_log=log,kind='original_navtrain',
            log_path=str(tmp_path/f'{log}.pkl'),sensor_root=str(tmp_path),anchor_index=4,image_files=[]))
        annotations[log+'_'+token]=dict(id=i,driving_command='GO STRAIGHT',solution={'prediction':'ORIGINAL EXACT TEXT'})
    index=tmp_path/'index.json';source=tmp_path/'source.json'
    index.write_text(json.dumps(dict(split='navtrain',complete=True,records=records,expected_tokens=[r['token'] for r in records])))
    source.write_text(json.dumps(annotations))
    return index,source,annotations


def policy():
    return json.loads((captions.CONFIG/'cluster-caption.json').read_text())


def response():
    return dict(choices=[dict(finish_reason='stop',message=dict(content=json.dumps({
        'scene_description_zh':'前方为道路，路边可见车辆。'*10,
        'attributes':[dict(dimension='road_layout',description='前方道路。',evidence_ids=['CAM_F0_anchor'],certainty='visible') for _ in range(4)],
        'retrieval_tags':['路侧车辆','前方道路','路侧设施'],'uncertainties':[]})))],
        usage=dict(prompt_tokens=100,completion_tokens=50),model='deepseek-v4.1-flash-1017',id='test-response')


def test_copy_and_extension_preserve_original_fields(tmp_path):
    index,source,annotations=corpus(tmp_path)
    original=source.read_bytes();library=tmp_path/'lib'
    result=captions.initialize(index,source,library)
    assert result['scene_count']==2
    assert source.read_bytes()==original==(library/'autodrive-p3.original.json').read_bytes()
    extended=json.loads((library/'autodrive-p3.with-cluster-caption.json').read_text())
    for key,value in extended.items():
        extension=value.pop('cluster_caption')
        assert value==annotations[key]
        assert extension['status']=='pending' and extension['trainable'] is False
    assert captions.initialize(index,source,library)==result


def test_split_and_member_mismatch_fail_closed(tmp_path):
    index,source,_=corpus(tmp_path)
    data=json.loads(index.read_text());data['split']='navtest';index.write_text(json.dumps(data))
    with pytest.raises(ValueError):captions.initialize(index,source,tmp_path/'lib')


def test_caption_camera_check_and_no_auto_approval():
    job=dict(evidence=[dict(id='CAM_F0_anchor')])
    decoded=captions.decode(job,'model',response())
    assert decoded['status']=='format_valid_unreviewed'
    assert decoded['indexable'] is False and decoded['trainable'] is False
    bad=response();value=json.loads(bad['choices'][0]['message']['content'])
    value['attributes'][0]['evidence_ids']=['CAM_B0_anchor']
    bad['choices'][0]['message']['content']=json.dumps(value)
    with pytest.raises(ValueError):captions.decode(job,'model',bad)


def test_shared_budget_charges_usage_and_persists_resume(tmp_path,monkeypatch):
    conf=policy();conf['requests_per_minute']=1000000;conf['tokens_per_minute']=100000000
    conf['max_requests']=1
    monkeypatch.setattr(client,'request',lambda *_:response())
    guard=batch.SharedBudget(tmp_path,conf)
    guard({},'test',{})
    assert guard.snapshot()['estimated_busy_cost_cny']==pytest.approx(.0006)
    guard.db.close()
    resumed=batch.SharedBudget(tmp_path,conf)
    with pytest.raises(client.APIError):resumed({},'test',{})
    assert resumed.snapshot()['requests_started']==1
    resumed.db.close()


def test_unknown_outcome_keeps_reserved_cost(tmp_path,monkeypatch):
    def fail(*_):raise client.APIError('network_outcome_unknown_no_automatic_retry')
    monkeypatch.setattr(client,'request',fail)
    guard=batch.SharedBudget(tmp_path,policy())
    with pytest.raises(client.APIError):guard({},'test',{})
    assert guard.snapshot()['unknown_or_inflight']==1
    assert guard.snapshot()['estimated_busy_cost_cny']==guard.reserve_cost
    guard.db.close()


def test_server_error_keeps_unknown_cost(tmp_path,monkeypatch):
    def fail(*_):raise client.APIError(500)
    monkeypatch.setattr(client,'request',fail)
    guard=batch.SharedBudget(tmp_path,policy())
    with pytest.raises(client.APIError):guard({},'test',{})
    assert guard.snapshot()['estimated_busy_cost_cny']==guard.reserve_cost
    guard.db.close()


def test_stopped_pacer_never_calls_transport(tmp_path,monkeypatch):
    import concurrent.futures
    import time
    calls=[];monkeypatch.setattr(client,'request',lambda *_:calls.append(1))
    guard=batch.SharedBudget(tmp_path,policy());guard.next_start=time.monotonic()+10
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future=pool.submit(guard,{},'test',{})
        time.sleep(.03);guard.stop.set()
        with pytest.raises(client.NotSent):future.result(timeout=1)
    assert not calls and guard.snapshot()['requests_started']==0
    guard.db.close()


def test_complete_run_and_resume_never_reissues_paid_jobs(tmp_path,monkeypatch):
    index,source,original=corpus(tmp_path,1);library=tmp_path/'lib';output=tmp_path/'run'
    captions.initialize(index,source,library)
    image=tmp_path/'photo.jpg';image.write_bytes(b'test compressed evidence')
    def prepare(record,annotation,out,prompt,config):
        assert annotation['solution']['prediction']=='ORIGINAL EXACT TEXT'
        return dict(token=record['token'],scope='current_anchor_only',input_identity='test-input',prompt='test',
            facts=dict(navigation_command='GO STRAIGHT',anchor_speed_kmh=10,future_included=False),
            evidence=[dict(id='CAM_F0_anchor',relative_time_s=0,file=str(image),sha256=hashlib.sha256(image.read_bytes()).hexdigest())])
    monkeypatch.setattr(captions,'prepare_job',prepare)
    config=dict(base_url='https://example.test/compatible-mode/v1',models=['deepseek-v4.1-flash-1017'],
        concurrency=1,requests_per_minute=600,max_requests=10,max_attempts=1,timeout_seconds=10,max_tokens=1800,
        image_max_side=768,jpeg_quality=80,enable_thinking=False)
    monkeypatch.setattr(client,'private_config',lambda _:copy.deepcopy(config))
    monkeypatch.setattr(client,'credential',lambda _:'test-not-a-real-key')
    calls=[]
    def request(*args):calls.append(1);return response()
    monkeypatch.setattr(client,'request',request)
    assert batch.run(library,output,'private','credential',workers=1)['status']=='complete'
    assert batch.run(library,output,'private','credential',workers=1)['status']=='complete'
    assert len(calls)==1
    extended=json.loads((library/'autodrive-p3.with-cluster-caption.json').read_text())
    key=next(iter(original));assert extended[key]['solution']==original[key]['solution']
    assert extended[key]['cluster_caption']['status']=='machine_draft'
    assert not extended[key]['cluster_caption']['indexable']


def test_client_not_sent_attempt_resumes_without_losing_evidence(tmp_path):
    image=tmp_path/'image.jpg';image.write_bytes(b'prepared test bytes')
    job=dict(token='t',scope='current_anchor_only',input_identity='frozen',prompt='p',facts={},
             evidence=[dict(id='CAM_F0_anchor',relative_time_s=0,file=str(image),sha256=hashlib.sha256(image.read_bytes()).hexdigest())])
    config=dict(base_url='https://example.test/compatible-mode/v1',models=['deepseek-v4.1-flash-1017'],
        concurrency=1,requests_per_minute=600,max_requests=1,max_attempts=1,timeout_seconds=10,max_tokens=1800,
        image_max_side=768,jpeg_quality=80,enable_thinking=False)
    def stopped(*_):raise client.NotSent('operator_stopped_before_send')
    first=client.run([job],config,tmp_path/'api',key='test',transport=stopped,decode=captions.decode)
    assert first['rows'][0]['status']=='not_sent'
    called=[]
    def real(*_):called.append(1);return response()
    second=client.run([job],config,tmp_path/'api',key='test',transport=real,decode=captions.decode)
    third=client.run([job],config,tmp_path/'api',key='test',transport=real,decode=captions.decode)
    assert second['complete'] and third['reused']==1 and len(called)==1
    receipts=[json.loads(p.read_text()) for p in (tmp_path/'api/attempts').glob('*.json')]
    assert len(receipts)==2 and {r['status'] for r in receipts}=={'not_sent','response_received'}


def test_batch_not_sent_stays_pending_then_resumes(tmp_path,monkeypatch):
    import threading
    index,source,original=corpus(tmp_path,1);library=tmp_path/'lib';output=tmp_path/'run'
    captions.initialize(index,source,library)
    image=tmp_path/'image.jpg';image.write_bytes(b'test image')
    def prepare(record,annotation,out,prompt,config):
        return dict(token=record['token'],scope='current_anchor_only',input_identity='frozen',prompt=prompt,
            facts=dict(scope='current_anchor_only',future_included=False,anchor_speed_kmh=10,navigation_command='GO STRAIGHT'),
            evidence=[dict(id=cam+'_anchor',relative_time_s=0,file=str(image),sha256=hashlib.sha256(image.read_bytes()).hexdigest()) for cam in captions.CAMERAS])
    monkeypatch.setattr(captions,'prepare_job',prepare)
    config=dict(base_url='https://example.test/compatible-mode/v1',models=['deepseek-v4.1-flash-1017'],
        concurrency=1,requests_per_minute=600,max_requests=2,max_attempts=1,timeout_seconds=10,max_tokens=1800,
        image_max_side=768,jpeg_quality=80,enable_thinking=False)
    class Transport:
        stop=threading.Event()
        stopped=True
        sent=0
        def __call__(self,*_):
            if self.stopped:raise client.NotSent('operator_stopped')
            self.sent+=1;return response()
    transport=Transport()
    first=batch.run_batch(0,['token0'],library,output,original,'prompt',policy(),config,'test',transport)
    assert not first['finished'] and first['failed']==0
    db=captions.connect(library)
    assert db.execute('SELECT status FROM captions').fetchone()[0]=='pending'
    db.close()
    transport.stopped=False
    second=batch.run_batch(0,['token0'],library,output,original,'prompt',policy(),config,'test',transport)
    assert second['finished'] and second['drafts']==1 and transport.sent==1


def test_missing_provider_usage_retains_reserved_cost(tmp_path,monkeypatch):
    result=response();result.pop('usage')
    monkeypatch.setattr(client,'request',lambda *_:result)
    guard=batch.SharedBudget(tmp_path,policy())
    assert guard({},'test',{})==result
    assert guard.snapshot()['unknown_or_inflight']==1
    assert guard.snapshot()['estimated_busy_cost_cny']==guard.reserve_cost
    guard.db.close()


def test_retry_budget_counts_paid_attempts_across_not_sent_resume(tmp_path,monkeypatch):
    image=tmp_path/'image.jpg';image.write_bytes(b'test')
    job=dict(token='t',scope='current_anchor_only',input_identity='frozen',prompt='p',facts={},
        evidence=[dict(id='CAM_F0_anchor',relative_time_s=0,file=str(image),sha256=hashlib.sha256(image.read_bytes()).hexdigest())])
    config=dict(base_url='https://example.test/compatible-mode/v1',models=['deepseek-v4.1-flash-1017'],
        concurrency=1,requests_per_minute=600,max_requests=2,max_attempts=2,timeout_seconds=10,max_tokens=1800,
        image_max_side=768,jpeg_quality=80,enable_thinking=False)
    steps=[]
    monkeypatch.setattr(client.time,'sleep',lambda _:None)
    def interrupted(*_):
        steps.append(1)
        if len(steps)==1:raise client.APIError(429,True)
        raise client.NotSent('paused')
    first=client.run([job],config,tmp_path/'api',key='test',transport=interrupted,decode=captions.decode)
    assert first['rows'][0]['status']=='not_sent'
    calls=[]
    def rejected(*_):calls.append(1);raise client.APIError(429,True)
    second=client.run([job],config,tmp_path/'api',key='test',transport=rejected,decode=captions.decode)
    assert len(calls)==1 and second['rows'][0]['status']=='api_failed'


def test_64_concurrent_callers_share_one_request_budget(tmp_path,monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    conf=policy();conf.update(requests_per_minute=10000000,tokens_per_minute=1000000000,max_requests=40)
    monkeypatch.setattr(client,'request',lambda *_:response())
    guard=batch.SharedBudget(tmp_path,conf)
    def call(_):
        try:guard({},'test',{});return True
        except client.NotSent:return False
    with ThreadPoolExecutor(max_workers=64) as pool:results=list(pool.map(call,range(64)))
    snapshot=guard.snapshot()
    assert snapshot['requests_started']==sum(results)<=40
    assert snapshot['unknown_or_inflight']==0
    assert snapshot['estimated_busy_cost_cny']==pytest.approx(sum(results)*.0006)
    guard.db.close()
