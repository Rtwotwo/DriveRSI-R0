import copy
import hashlib
import json
from pathlib import Path

import pytest

from driversi.data.catalog import cluster_caption as captions
from driversi.data.catalog import cluster_caption_batch as batch
from driversi.data.catalog import cluster_caption_regeneration as regen
from driversi.data.catalog import cluster_caption_repair as repair
from driversi.data.catalog import bailian_caption as client


def response(attribute_count=4):
    body=dict(scene_description_zh='前方为城市道路，路侧可见停车与建筑。'*8,
        attributes=[dict(dimension='road_layout',description=f'第{i}项原有道路细节。',
            evidence_ids=['CAM_F0_anchor'],certainty='visible') for i in range(attribute_count)],
        retrieval_tags=['道路','路侧停车','建筑'],uncertainties=[])
    return dict(choices=[dict(finish_reason='stop',message=dict(content=json.dumps(body,ensure_ascii=False)))],
        usage=dict(prompt_tokens=100,completion_tokens=50),created=1793073354,model='deepseek-v4.1-flash-1017',id='test-reply')


def fixture(tmp_path,monkeypatch):
    libraries={};campaigns={};old_good=None
    config=dict(base_url='https://example.test/compatible-mode/v1',models=['deepseek-v4.1-flash-1017'],
        concurrency=8,requests_per_minute=600,max_requests=128,max_attempts=1,
        timeout_seconds=10,max_tokens=1800,image_max_side=768,jpeg_quality=80,enable_thinking=False)
    policy=json.loads((captions.CONFIG/'cluster-caption.json').read_text())
    for split in ('navtrain','navtest'):
        library=tmp_path/split;campaign=tmp_path/(split+'-campaign');campaign.mkdir()
        libraries[split]=library;campaigns[split]=campaign;tokens=[split+'-failed']+(['good'] if split=='navtrain' else [])
        image=tmp_path/(split+'.jpg');image.write_bytes(b'isolated fixture '+split.encode())
        records=[];original={};jobs=[]
        for i,token in enumerate(tokens):
            log=split+'-log';record=dict(token=token,source_log=log,kind='original_'+split,
                log_path='unused',sensor_root='unused',anchor_index=i,image_files=[])
            records.append(record);original[log+'_'+token]=dict(solution={'prediction':'EXACT NATIVE LABEL'} if split=='navtrain' else None,driving_command='GO FORWARD',id=token)
            job=dict(token=token,split=split,scope='current_anchor_only',input_identity='input-'+token,prompt='frozen prompt',
                facts=dict(scope='current_anchor_only',future_included=False,anchor_speed_kmh=10,navigation_command='GO FORWARD'),
                evidence=[dict(id=cam+'_anchor',relative_time_s=0,file=str(image),sha256=hashlib.sha256(image.read_bytes()).hexdigest()) for cam in captions.CAMERAS])
            jobs.append(job)
        index=tmp_path/(split+'-index.json');index.write_text(json.dumps(dict(split=split,complete=True,records=records,expected_tokens=tokens)))
        source=tmp_path/(split+'-source.json');source.write_text(json.dumps(original))
        captions.initialize(index,source,library)
        directory=campaign/'batches/batch-00000';(directory/'api/results').mkdir(parents=True)
        (directory/'jobs.json').write_text(json.dumps(jobs))
        db=captions.connect(library)
        for token,job in zip(tokens,jobs):
            receipt=directory/'api/results'/f'{token}.json';receipt.write_text(json.dumps(dict(status='api_failed',token=token,input_identity=job['input_identity'])))
            if token=='good':
                caption=captions.extension(token,job,captions.decode(job,policy['model'],response()),str(receipt))
                db.execute("UPDATE captions SET status='machine_draft',caption=?,receipt=? WHERE token=?",(json.dumps(caption),str(receipt),token))
                old_good=dict(caption=caption,receipt=receipt.read_bytes())
            else:db.execute("UPDATE captions SET status='response_failed',receipt=?,error='Private provider HTTP 503' WHERE token=?",(str(receipt),token))
        db.commit();rows={r['annotation_key']:dict(r) for r in db.execute('SELECT * FROM captions')};db.close()
        (library/'shards').mkdir();lines=[]
        # One fixture batch; production batches are preserved by the runner.
        (library/'batches.json').write_text(json.dumps([tokens]))
        for key,value in original.items():
            row=rows[key];extension=json.loads(row['caption']) if row['caption'] else {'status':'response_failed'}
            lines.append(json.dumps(dict(annotation_key=key,**value,cluster_caption=extension),ensure_ascii=False)+'\n')
        (library/'shards/batch-00000.jsonl').write_text(''.join(lines))
        captions.export(library)
        (campaign/'profile.json').write_text(json.dumps(dict(policy=policy,prompt='frozen prompt',endpoint_identity=client.identity(config['base_url']))))
        (campaign/'progress.json').write_text(json.dumps(dict(status='complete_with_failures')))
        (library/'finalization.json').write_text(json.dumps(dict(status='verified_with_unresolved_failures',cost=dict(
            estimated_cny=1.,received_responses=1,input_tokens=1,output_tokens=1,offpeak_responses=0,peak_responses=1,
            snapshot_complete=True,spend_ledger_states={'received':1,'unknown':1})) ))
    monkeypatch.setattr(client,'private_config',lambda _:copy.deepcopy(config));monkeypatch.setattr(client,'credential',lambda _:'fixture-not-secret')
    real_verify=repair.verify_library
    monkeypatch.setattr(repair,'verify_library',lambda library,campaign:real_verify(library,campaign,expected_count=json.loads((Path(library)/'manifest.json').read_text())['scene_count']))
    return libraries,campaigns,tmp_path/'retry',old_good


def test_inventory_only_selects_failures_and_freezes_population(tmp_path,monkeypatch):
    libraries,campaigns,out,_=fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(client,'request',lambda *_:pytest.fail('Inventory cannot call API'))
    first=regen.inventory(libraries,campaigns,out)
    assert {s:len(v) for s,v in first['targets'].items()}=={'navtrain':1,'navtest':1}
    assert all(r['token']!='good' for rows in first['targets'].values() for r in rows)
    assert regen.inventory(libraries,campaigns,out)==first


def test_real_runner_updates_primary_files_without_success_overwrite_or_paid_resume(tmp_path,monkeypatch):
    libraries,campaigns,out,old_good=fixture(tmp_path,monkeypatch)
    before=(libraries['navtrain']/'shards/batch-00000.jsonl').read_bytes().splitlines(keepends=True)
    good_line=next(x for x in before if json.loads(x)['id']=='good')
    called=[]
    def request(*_):called.append(1);return response(13)
    monkeypatch.setattr(client,'request',request)
    first=regen.run(libraries,campaigns,out,'private','credential')
    assert first['status']=='complete' and len(called)==2
    second=regen.run(libraries,campaigns,out,'private','credential')
    assert second['status']=='complete' and len(called)==2
    assert first['new_cost']['estimated_cny']==pytest.approx(.0012)
    assert first['new_cost']['new_api_requests']==2
    for split,library in libraries.items():
        manifest=json.loads((library/'manifest.json').read_text());full=json.loads((library/manifest['extended_file']).read_text())
        assert all(v['cluster_caption']['status']=='machine_draft' for v in full.values())
        assert all(v['cluster_caption']['split']==split for v in full.values())
        assert not (library/'offline_repairs').exists()
        row=full[split+'-log_'+split+'-failed']
        assert row['cluster_caption']['offline_repair']['applied'] and not row['cluster_caption']['trainable']
    after=(libraries['navtrain']/'shards/batch-00000.jsonl').read_bytes().splitlines(keepends=True)
    assert good_line in after


def test_received_rows_recover_after_publication_crash_without_new_api(tmp_path,monkeypatch):
    libraries,campaigns,out,_=fixture(tmp_path,monkeypatch);called=[]
    monkeypatch.setattr(client,'request',lambda *_:(called.append(1) or response()))
    publish=regen.publish_shards
    monkeypatch.setattr(regen,'publish_shards',lambda *_:(_ for _ in ()).throw(OSError('publication interrupted')))
    with pytest.raises(OSError):regen.run(libraries,campaigns,out,'private','credential')
    assert len(called)==2
    monkeypatch.setattr(regen,'publish_shards',publish)
    assert regen.run(libraries,campaigns,out,'private','credential')['status']=='complete'
    assert len(called)==2


def test_only_failed_scene_gets_next_authorized_round(tmp_path,monkeypatch):
    libraries,campaigns,out,_=fixture(tmp_path,monkeypatch);calls=[]
    def request(*_):
        calls.append(1)
        if len(calls)==1:raise client.APIError(503)
        return response()
    monkeypatch.setattr(client,'request',request)
    report=regen.run(libraries,campaigns,out,'private','credential')
    assert report['status']=='complete' and len(calls)==3
    assert report['new_cost']['spend_ledger_states']['unknown']==1


def test_changed_original_camera_is_rejected_before_paid_request(tmp_path,monkeypatch):
    libraries,campaigns,out,_=fixture(tmp_path,monkeypatch);plan=regen.inventory(libraries,campaigns,out)
    (tmp_path/'navtrain.jpg').write_bytes(b'changed')
    with pytest.raises(ValueError,match='camera evidence'):
        regen.original_job(plan['targets']['navtrain'][0],'navtrain',plan['source_profiles']['navtrain'])


def test_visual_authoring_has_explicit_non_provider_provenance(tmp_path,monkeypatch):
    libraries,campaigns,out,_=fixture(tmp_path,monkeypatch)
    plan=regen.inventory(libraries,campaigns,out);row=plan['targets']['navtrain'][0]
    job=regen.original_job(row,'navtrain',plan['source_profiles']['navtrain'])
    value=json.loads(response()['choices'][0]['message']['content'])
    db=captions.connect(libraries['navtrain']);db.execute("UPDATE captions SET error='Private provider HTTP 400' WHERE token=?",(job['token'],));db.commit();db.close()
    monkeypatch.setattr(client,'request',lambda *_:pytest.fail('Offline visual authoring cannot call API'))
    entries=[dict(job=job,description=value,evidence_file='review-inputs.json')]
    audit=regen.apply_reviewed_descriptions(libraries['navtrain'],entries,out/'review.json')
    assert audit['count']==1 and audit['new_api_requests']==0
    db=captions.connect(libraries['navtrain']);caption=json.loads(db.execute('SELECT caption FROM captions WHERE token=?',(job['token'],)).fetchone()[0]);db.close()
    assert not caption['provider_generated'] and caption['model']=='codex_visual_review'
    assert not caption['indexable'] and not caption['trainable']
    assert caption['review_status']=='assistant_visual_annotation_not_independently_reviewed'
    again=regen.apply_reviewed_descriptions(libraries['navtrain'],entries,out/'review.json')
    assert again['count']==1 and again['newly_applied']==0
    assert json.loads((out/'review.json').read_text())['tokens']==[job['token']]
    bad=dict(value,attributes=[])
    with pytest.raises(ValueError,match='Invalid attributes'):captions.validate_description(job,bad)
