import copy
import importlib.util
import json
from pathlib import Path

import pytest

from driversi.data.catalog import bailian_caption as bc

spec = importlib.util.spec_from_file_location('scene_fixture', Path(__file__).with_name('test_scene_description_v2.py'))
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


def config():
    root = Path(bc.__file__).resolve().parents[3]
    return json.loads((root/'src/config/data/semantic_conditioning/bailian_private.example.json').read_text())


def job(tmp_path, scope='retrieval_history'):
    _, bundle = fixture.raw_and_bundle(tmp_path)
    file = tmp_path/'source.json'
    bc.write(file, bundle)
    return dict(token=bundle['token'], scope=scope, bundle=str(file),
                source_sha256=bc.sha256(file), input_identity=scope,
                prompt='Fixed prompt', facts={}, evidence=[dict(id='CAM_F0_f+0', relative_time_s=0)])


def response(text='A tree beside the road.'):
    obs = fixture.observations()
    obs['overview'] = dict(text=text, evidence_ids=['CAM_F0_f+0'])
    return dict(model='test', usage={'prompt_tokens':10, 'completion_tokens':20},
                choices=[dict(message={'content':json.dumps(obs)}, finish_reason='stop')])


@pytest.mark.parametrize('field,value', [('base_url','http://private-provider.example.invalid'),
    ('base_url','https://secret@private-provider.example.invalid/compatible-mode/v1'), ('api_key','dummy'),
    ('concurrency',0), ('max_requests',10001), ('enable_thinking','false')])
def test_explicit_private_bounded_config(field,value):
    c=config();c[field]=value
    with pytest.raises(ValueError):bc.validate_config(c)


def test_credential_file_private_and_outside_repo(tmp_path,monkeypatch):
    monkeypatch.setattr(bc,'PRIVATE_ROOT',tmp_path)
    p=tmp_path/'private.env';p.write_text('DASHSCOPE_API_KEY=not-a-real-key\n');p.chmod(0o600)
    assert bc.credential(p)=='not-a-real-key'
    p.chmod(0o644)
    with pytest.raises(ValueError):bc.credential(p)


@pytest.mark.parametrize('scope',['retrieval_history','video_clip'])
def test_draft_claims_not_silently_released(tmp_path,scope):
    j=job(tmp_path,scope);out=bc.result_from_response(j,'test',response())
    assert out['status']=='format_valid_unreviewed'
    assert out['caption']['appearance']['approved_paths']==[]
    assert out['caption']['scene_overview']['reviewed_visual'] is None
    assert not out['trainable']
    d=out['draft_prompts']
    assert (d['retrieval_text'] is None)==(scope=='video_clip')
    assert (d['video_positive_prompt'] is None)==(scope=='retrieval_history')


def test_future_evidence_and_truncated_caption_rejected(tmp_path):
    j=job(tmp_path);j['evidence'][0]['relative_time_s']=1
    with pytest.raises(ValueError,match='Future'):bc.result_from_response(j,'test',response())
    r=response();r['choices'][0]['finish_reason']='length'
    with pytest.raises(ValueError,match='Incomplete'):bc.result_from_response(job(tmp_path),'test',r)


def test_resume_no_new_calls_and_secret_echo_redacted(tmp_path,monkeypatch):
    monkeypatch.setattr(bc,'payload',lambda j,*args:{'input':j['input_identity']})
    monkeypatch.setattr(bc.time,'sleep',lambda *_:None)
    c=config();c['models']=['test'];j=job(tmp_path);calls=[]
    def transport(*args):
        calls.append(1)
        return response('A tree. credential-for-test')
    first=bc.run([j],c,tmp_path/'out',key='credential-for-test',transport=transport)
    assert first['format_valid']==1
    second=bc.run([j],c,tmp_path/'out',key='credential-for-test',transport=transport)
    assert len(calls)==1 and second['reused']==1
    assert second['lifetime_requests_started']==1
    assert all('credential-for-test' not in p.read_text() for p in (tmp_path/'out').rglob('*.json'))


def test_retry_and_lifetime_budget(tmp_path,monkeypatch):
    monkeypatch.setattr(bc,'payload',lambda j,*args:{'input':j['input_identity']})
    monkeypatch.setattr(bc.time,'sleep',lambda *_:None)
    c=config();c['models']=['test'];c['max_requests']=2;j=job(tmp_path);calls=[]
    def transport(*args):
        calls.append(1)
        if len(calls)==1:raise bc.APIError(429,True)
        return response()
    first=bc.run([j],c,tmp_path/'out',key='not-real',transport=transport)
    assert first['format_valid']==1 and len(calls)==2
    j2=copy.deepcopy(j);j2['input_identity']='another'
    second=bc.run([j2],c,tmp_path/'out',key='not-real',transport=transport)
    assert len(calls)==2 and second['rows'][0]['status']=='budget_exhausted'


def test_uncertain_attempt_not_automatically_rebilled(tmp_path,monkeypatch):
    monkeypatch.setattr(bc,'payload',lambda *args:{})
    monkeypatch.setattr(bc.time,'sleep',lambda *_:None)
    c=config();c['models']=['test'];j=job(tmp_path);calls=[]
    def transport(*args):
        calls.append(1);raise bc.APIError('network_outcome_unknown_no_automatic_retry')
    first=bc.run([j],c,tmp_path/'out',key='not-real',transport=transport)
    Path(first['rows'][0]['file']).unlink()  # Simulate failure before final result write.
    second=bc.run([j],c,tmp_path/'out',key='not-real',transport=transport)
    assert len(calls)==1
    assert second['rows'][0]['status']=='uncertain_previous_attempt_requires_review'


def test_public_dns_and_redirect_rejected(monkeypatch):
    monkeypatch.setattr(bc.socket,'getaddrinfo',lambda *args:[(None,None,None,None,('8.8.8.8',443))])
    with pytest.raises(ValueError):bc.private_endpoint(config()['base_url'])
    with pytest.raises(bc.APIError):bc.NoRedirect().redirect_request(None,302,'',{},'https://public.example')


def test_received_response_recovered_without_rebilling(tmp_path,monkeypatch):
    monkeypatch.setattr(bc,'payload',lambda *args:{})
    monkeypatch.setattr(bc.time,'sleep',lambda *_:None)
    c=config();c['models']=['test'];j=job(tmp_path);calls=[]
    def transport(*args):calls.append(1);return response()
    first=bc.run([j],c,tmp_path/'out',key='not-real',transport=transport)
    Path(first['rows'][0]['file']).unlink()
    second=bc.run([j],c,tmp_path/'out',key='not-real',transport=transport)
    assert len(calls)==1 and second['rows'][0]['recovered_received_response']
    assert second['format_valid']==1


def test_adapter_provenance_does_not_change_payable_identity(tmp_path,monkeypatch):
    monkeypatch.setattr(bc,'payload',lambda *args:{})
    monkeypatch.setattr(bc.time,'sleep',lambda *_:None)
    c=config();c['models']=['test'];j=job(tmp_path);calls=[]
    def transport(*args):calls.append(1);return response()
    bc.run([j],c,tmp_path/'out',key='not-real',transport=transport)
    real=bc.sha256
    monkeypatch.setattr(bc,'sha256',lambda p:'changed-adapter-only' if str(p)==bc.__file__ else real(p))
    second=bc.run([j],c,tmp_path/'out',key='not-real',transport=transport)
    assert len(calls)==1 and second['reused']==1


def test_legacy_batch_never_rebilled(tmp_path):
    c=config();bc.write(tmp_path/'out/results/old.json',{'profile':{'adapter':'legacy'}})
    with pytest.raises(ValueError,match='read-only'):bc.run([],c,tmp_path/'out',key='not-real')


def test_private_config_rejects_world_readable_file(tmp_path,monkeypatch):
    monkeypatch.setattr(bc,'PRIVATE_ROOT',tmp_path)
    p=tmp_path/'config.json';bc.write(p,config());p.chmod(0o600)
    assert bc.private_config(p)==config()
    p.chmod(0o644)
    with pytest.raises(ValueError,match='0600'):bc.private_config(p)


def test_legacy_attempt_only_batch_never_rebilled(tmp_path):
    bc.write(tmp_path/'attempts/legacy-0.json',{'status':'inflight'})
    with pytest.raises(ValueError,match='attempt-only'):bc.run([],config(),tmp_path,key='not-real')


def test_credentials_outside_authorized_root_and_env_rejected(tmp_path,monkeypatch):
    p=tmp_path/'key.env';p.write_text('DASHSCOPE_API_KEY=fake-test-key\n');p.chmod(0o600)
    monkeypatch.setattr(bc,'PRIVATE_ROOT',tmp_path/'private')
    monkeypatch.setenv('DASHSCOPE_API_KEY','fake-test-key')
    with pytest.raises(ValueError,match='authorized'):bc.credential(p)
    with pytest.raises(ValueError,match='fallback'):bc.credential()
