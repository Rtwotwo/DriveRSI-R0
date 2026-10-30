"""Multimodal evidence contract tests with a real tiny PNG; no network."""
import copy
import json

import pytest
from PIL import Image

from driversi.rsi.decisions import skill_doctor as doctor
from driversi.agents.backends.native import digest
from driversi.rsi.decisions.skill_evolution import identity


def image_job(tmp_path):
    path=tmp_path/'current.png';Image.new('RGB',(4,4),(30,60,90)).save(path)
    return dict(prompt='test diagnostic protocol',visual_protocol='development_real_history_v1',
        evidence=[dict(path=str(path),sha256=digest(path),role='development',sensor_image=True,
                       label='front anchor',case_id='case-01',time_offset_s=0)],
        diagnosis_evidence_ids=['case-01.metric'],
        facts=dict(development=dict(evidence=[dict(id='case-01.metric',kind='measured_metric',value=0)])))


def test_real_png_payload_and_hash_revalidation(tmp_path):
    job=image_job(tmp_path)
    payload=doctor.request_payload(job,dict(max_tokens=1600,enable_thinking=False))
    content=payload['messages'][0]['content']
    assert len(content)==3 and content[-1]['image_url']['url'].startswith('data:image/png;base64,')
    assert payload['temperature']==0
    Image.new('RGB',(4,4),(90,60,30)).save(job['evidence'][0]['path'])
    with pytest.raises(ValueError,match='evidence changed'):doctor.validated_images(job)


@pytest.mark.parametrize('field,value',[('role','validation'),('sensor_image',False),('time_offset_s',0.5),
                                      ('time_offset_s',-2.5),('time_offset_s',float('nan'))])
def test_future_validation_or_non_sensor_images_rejected(tmp_path,field,value):
    job=image_job(tmp_path);job['evidence'][0][field]=value
    with pytest.raises(ValueError):doctor.validated_images(job)


def test_images_require_explicit_protocol_and_matching_diagnosis_catalog(tmp_path):
    job=image_job(tmp_path);job.pop('visual_protocol')
    with pytest.raises(ValueError,match='Undeclared'):doctor.request_payload(job,dict(max_tokens=800,enable_thinking=False))
    job=image_job(tmp_path);job['diagnosis_evidence_ids']=['unavailable']
    with pytest.raises(ValueError,match='catalog mismatch'):doctor.request_payload(job,dict(max_tokens=800,enable_thinking=False))


def test_unknown_request_never_resends_and_completed_request_is_cached(tmp_path):
    job=image_job(tmp_path);out=tmp_path/'request';out.mkdir()
    intent=dict(job_hash=identity(job),model=doctor.MODEL,at=1)
    (out/'request-intent.json').write_text(json.dumps(intent))
    called=[]
    transport=lambda *args:called.append(args)
    with pytest.raises(RuntimeError,match='outcome_unknown_do_not_resend'):
        doctor.cached_request(job,out,'not-read','not-read',transport=transport)
    assert called==[]
    reply=dict(job_hash=identity(job),model=doctor.MODEL,content='{}',finish_reason='stop',usage={},seconds=1)
    (out/'response.json').write_text(json.dumps(reply))
    assert doctor.cached_request(job,out,'not-read','not-read',transport=transport)==reply
    assert called==[]
    changed=copy.deepcopy(job);changed['prompt']='changed'
    with pytest.raises(ValueError,match='identity changed'):
        doctor.cached_request(changed,out,'not-read','not-read',transport=transport)
