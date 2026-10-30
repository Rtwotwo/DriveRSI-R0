import copy
import json
import sqlite3

import pytest

from driversi.data.catalog.skill_context import context, collect, CAMERAS
from driversi.rsi.decisions.skill_doctor import feedback
from driversi.rsi.decisions.skill_evolution import identity, memory, seed_memory


def caption():
    return dict(schema_version='driversi.cluster_caption.v1', token='test', split='navtest',
        observation_scope='current_anchor_four_cameras', future_images_included=False,
        navigation_command='GO FORWARD', anchor_speed_kmh=9,
        scene_description_zh='DO NOT COPY navigation or future targets', retrieval_tags=['DO NOT COPY'],
        solution_prediction='future label sentinel', model='fixture', review_status='machine_draft',
        attributes=[dict(dimension='road_layout', description='Intersection with visible crosswalk.',
                         evidence_ids=['CAM_F0_anchor'], certainty='visible'),
                    dict(dimension='traffic_control', description='Signal color unknown.',
                         evidence_ids=['CAM_R0_anchor'], certainty='uncertain'),
                    dict(dimension='road_layout', description='导航指令直行',
                         evidence_ids=['CAM_F0_anchor'], certainty='visible')],
        uncertainties=['Vehicle motion not established.'])


def test_allowlist_preserves_uncertainty_and_conflict_not_caption_state():
    c=caption();out=context(c,dict(navigation_command='STOP',speed_mps=2.5))
    assert out['navigation_conflict'] and out['excluded_state_attributes']==1
    assert out['attributes'][1]['certainty']=='uncertain'
    assert out['temporal_interaction_verified'] is False
    serialized=json.dumps(out,ensure_ascii=False)
    assert 'DO NOT COPY' not in serialized and 'future label sentinel' not in serialized
    assert 'GO FORWARD' not in serialized and '导航' not in serialized
    assert c==caption()


@pytest.mark.parametrize('mutation',[
    {'future_images_included':True}, {'observation_scope':'full_clip'},
    {'attributes':[dict(dimension='road_layout',description='claim',certainty='visible',evidence_ids=['future'])]}])
def test_wrong_scope_or_evidence_fails(mutation):
    with pytest.raises(ValueError):context({**caption(),**mutation},dict(navigation_command='STOP'))


def test_collect_reads_only_development_with_timestamp_and_identity(tmp_path):
    c=caption();evidence=tmp_path/'job.json';receipt=tmp_path/'cache.json'
    evidence.write_text(json.dumps([dict(token='test',split='navtest',scope='current_anchor_only',
        facts=dict(scope='current_anchor_only',future_included=False,anchor_speed_kmh=9,navigation_command='GO FORWARD'),
        evidence=[dict(id=k,relative_time_s=0,source_timestamp_us=123) for k in CAMERAS])]))
    receipt.write_text(json.dumps(dict(block=dict(records=[dict(frame_id='test',log_name='log',timestamp_us=123)]))))
    c['evidence_file']=str(evidence)
    lib=tmp_path/'library';(lib/'navtest').mkdir(parents=True)
    with sqlite3.connect(lib/'navtest/captions.sqlite3') as db:
        db.execute('CREATE TABLE captions (token TEXT PRIMARY KEY, caption TEXT)')
        db.execute('INSERT INTO captions VALUES (?,?)',('test',json.dumps(c)))
    member=dict(token='test',source_log='log',role='development',cache_receipt=str(receipt),
                skill=dict(evidence=dict(navigation_command='STOP',speed_mps=2.5)))
    val=dict(token='MISSING_HELDOUT_SENTINEL',role='validation-0')
    out=collect([member,val],lib)
    assert set(out['contexts'])=={'test'} and out['summary']['validation_captions_read']==0
    member['source_log']='wrong'
    with pytest.raises(ValueError,match='log mismatch'):collect([member],lib)
    member['source_log']='log'
    job=json.loads(evidence.read_text());job[0]['evidence'][0]['source_timestamp_us']=999
    evidence.write_text(json.dumps(job))
    with pytest.raises(ValueError,match='timestamp'):collect([member],lib)


def test_caption_feedback_groups_seeds_and_excludes_validation():
    members=[dict(token=t,role=role,skill=dict(id='following',evidence=dict(navigation_command='STOP',speed_mps=2)))
             for t,role in [('dev','development'),('val','validation')]]
    rows=[dict(token=t,seed=s,arm=a,metrics={k:(0.1 if t=='dev' else 0.987654321) for k in ['PDMS','NC','DAC','DDC','EP','TTC','C']})
          for t in ['dev','val'] for s in [42,43] for a in ['incumbent','candidate']]
    result=feedback(rows,members,'candidate',{'dev':{'attributes':['visible road']}})
    cases=result['following']['difficult_observations']
    assert len(cases)==1 and cases[0]['paired_seed_count']==2
    assert cases[0]['delta_vs_incumbent']['PDMS']==0
    assert cases[0]['scene_context']=={'attributes':['visible road']}
    assert '0.987654321' not in json.dumps(result) and 'token' not in json.dumps(result)


def test_inherited_memory_retains_rejections_and_is_idempotent(tmp_path):
    initial={'version':'base','skills':{'x':'initial'}}
    old=tmp_path/'old.sqlite';new=tmp_path/'new.sqlite'
    memory(old,initial)
    event=dict(id='old/episode-0',incumbent_hash=identity(initial),candidate=initial,
               decision={'accepted':False,'reasons':['safety']})
    snapshot=memory(old,event=event,expected_revision=0)
    before=old.read_bytes()
    assert seed_memory(new,initial,snapshot)==snapshot
    assert seed_memory(new,initial,snapshot)==snapshot and old.read_bytes()==before
    next_event={**event,'id':'new/episode-0'}
    assert memory(new,event=next_event,expected_revision=1)['revision']==2
    with pytest.raises(ValueError):seed_memory(tmp_path/'bad.sqlite',initial,{**snapshot,'revision':99})


def test_context_receipt_cannot_include_validation_or_change(tmp_path):
    from driversi.rsi.experiments.skill_evolution import read_contexts
    from driversi.agents.backends.native import digest
    path=tmp_path/'context.json';path.write_text(json.dumps({'contexts':{'dev':{}}}))
    prepared=dict(caption_context=str(path),caption_context_sha256=digest(path),
                  members=[dict(token='dev',role='development'),dict(token='val',role='validation')])
    assert read_contexts(prepared)=={'dev':{}}
    path.write_text(json.dumps({'contexts':{'val':{}}}))
    with pytest.raises(ValueError,match='changed'):read_contexts(prepared)
    prepared['caption_context_sha256']=digest(path)
    with pytest.raises(ValueError,match='development'):read_contexts(prepared)


def test_known_doctor_result_adoption_does_not_call_transport(tmp_path):
    from driversi.rsi.decisions.skill_doctor import cached_request
    job={'prompt':'fixture','facts':{},'evidence':[]};old=tmp_path/'old';old.mkdir()
    saved={'job_hash':identity(job),'content':'known reply'}
    for name,value in [('job.json',job),('request-intent.json',{'job_hash':identity(job)}),('response.json',saved)]:
        (old/name).write_text(json.dumps(value))
    assert cached_request(job,tmp_path/'new','must-not-read','must-not-read',reuse_from=old)==saved
    assert json.loads((tmp_path/'new/cache-adoption.json').read_text())['new_api_call'] is False
    with pytest.raises(ValueError,match='same job'):
        cached_request({**job,'prompt':'different'},tmp_path/'bad','must-not-read','must-not-read',reuse_from=old)
