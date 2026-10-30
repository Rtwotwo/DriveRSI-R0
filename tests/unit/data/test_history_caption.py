import copy
import json
import math
import pickle

import pytest
from PIL import Image
from driversi.data.catalog.history_caption import (
    ego_context, load_history, public_context, parse_camera, parse_bundle, FIELDS,
)
from driversi.data.catalog.store import identity


def fixture_scene(tmp_path):
    frames=[]
    for i in range(6):
        path=tmp_path/f'{i}.png'; Image.new('RGB',(320,180),(30*i,0,0)).save(path)
        frames.append({'token':str(i),'log_name':'source','timestamp':1_000_000+500_000*i,
            'cams':{cam:{'data_path':path.name} for cam in ['CAM_F0','CAM_L0','CAM_R0']},
            'ego2global_translation':[10.,20.+i,5.],
            'ego2global_rotation':[math.sqrt(.5),0,0,math.sqrt(.5)],
            'ego_dynamic_state':[i,0.,0.,2.],
            'future_trajectory':'must_not_appear','driving_command':[1,0,0]})
    log=tmp_path/'source.pkl';log.write_bytes(pickle.dumps(frames))
    return {'record':{'token':'3','source_log':'source','kind':'original_navtrain','log_path':str(log),
                     'anchor_index':3,'sensor_root':str(tmp_path)},
            'descriptor':{'history':{'continuous':True,'anchor_timestamp_us':2_500_000}}},frames,log


def config():
    return {'cameras':['CAM_F0','CAM_L0','CAM_R0'],'max_side':256,'prompt_version':'history_state_v8'}


def camera_value():
    return {'pavement':'Wide paved corridor.', 'road_paint':'unknown',
            'traffic_entities':'People are visible in image-center.',
            'roadside_context':'Glass entrance and pillars.', 'image_change':'unknown'}


def bundle(context,conf=None):
    return json.dumps({'ego_context':context,'camera_responses':{
        cam:json.dumps(camera_value()) for cam in (conf or config())['cameras']}})


def test_ego_coordinates_and_causal_inputs(tmp_path):
    scene,frames,log=fixture_scene(tmp_path)
    evidence,_,ctx=load_history(scene,config())
    assert ctx['frames'][0]['x_m']==pytest.approx(-3)
    assert ctx['frames'][0]['y_m']==pytest.approx(0)
    assert ctx['frames'][0]['heading_deg']==pytest.approx(0)
    assert ctx['frames'][-1]['speed_mps']==3
    assert ctx['frames'][-1]['acceleration_magnitude_mps2']==2
    assert ctx['measured_change']['speed_delta_mps']==3
    assert ctx['frames'][0]['relative_time_s']==-1.5
    assert 'ego_speed_mps' in public_context(ctx)['frames'][0]
    assert 'speed_mps' not in public_context(ctx)['frames'][0]
    raw=json.dumps(public_context(ctx))
    for forbidden in ['source','future_trajectory','driving_command','score','token','navtrain']:
        assert forbidden not in raw
    frames[4]={};frames[5]=None;log.write_bytes(pickle.dumps(frames))
    assert load_history(scene,config())[0]==evidence
    assert ego_context(scene)==ctx
    frames[2]['ego_dynamic_state'][0]=10.;log.write_bytes(pickle.dumps(frames))
    changed,_,newctx=load_history(scene,config())
    assert identity(changed)!=identity(evidence)
    assert newctx!=ctx


def test_irregular_history_retains_real_time(tmp_path):
    scene,frames,log=fixture_scene(tmp_path)
    for row in frames[2:]:row['timestamp']+=500000
    scene['descriptor']['history'].update(anchor_timestamp_us=3_000_000,continuous=False)
    log.write_bytes(pickle.dumps(frames))
    evidence,images,ctx=load_history(scene,config())
    assert ctx['actual_intervals_s']==[.5,1.,.5]
    assert ctx['sampling']=='observed_irregular'
    assert ctx['frames'][0]['relative_time_s']==-2.
    assert len(images)==12 and len(evidence)==12
    assert not ctx['temporal_training_admitted']


@pytest.mark.parametrize('field,value',[('ego_dynamic_state',[1,2,3]),
    ('ego_dynamic_state',[math.nan,0,0,0]),('ego2global_rotation',[0,0,0,0])])
def test_invalid_native_state_refused(tmp_path,field,value):
    scene,frames,log=fixture_scene(tmp_path);frames[1][field]=value;log.write_bytes(pickle.dumps(frames))
    with pytest.raises(ValueError):ego_context(scene)


@pytest.mark.parametrize('field,value',[
    ('road_paint','Bollards and signs form the markings.'),
    ('road_paint','A dashed line.'),
    ('traffic_entities','Car parked ahead.'),
    ('pavement','Curved glass roof.'),
    ('roadside_context','CAM_L0 is another view.'),
    ('roadside_context','A hotel labeled PARK MGM.'),
    ('pavement','Floor with flower beds and bollards.'),
    ('image_change','The silver car moved from left to center.'),
    ('road_paint','The road surface is gray asphalt.'),
    ('traffic_entities',' '.join(['word']*26)),
    ('image_change',{'text':'unknown'}),
])
def test_unsupported_categories_and_schema_are_rejected(field,value):
    obj=camera_value();obj[field]=value
    with pytest.raises(ValueError):parse_camera(json.dumps(obj))


def test_missing_extra_and_duplicate_fields_rejected():
    for mutation in [dict(camera_value(),score=100), {k:v for k,v in camera_value().items() if k!='image_change'}]:
        with pytest.raises(ValueError):parse_camera(json.dumps(mutation))
    raw=json.dumps(camera_value())[:-1]+',"pavement":"different"}'
    with pytest.raises(ValueError,match='Duplicate'):parse_camera(raw)


def test_code_binds_camera_references_and_structured_context(tmp_path):
    scene,_,_=fixture_scene(tmp_path);e,_,ctx=load_history(scene,config())
    canonical,structured=parse_bundle(bundle(ctx),e,config())
    assert len(canonical['caption'])<1600
    assert canonical['observations']==[]  # no invented categorical tags
    assert not structured['trainable'] and structured['assessment']=='unreviewed'
    for part in structured['camera_observations']:
        for obs in part['observations']:
            expected=[f'h3_{part["camera"]}'] if obs['field']!='image_change' else [f'h0_{part["camera"]}',f'h3_{part["camera"]}']
            assert obs['evidence']==expected
    bad=copy.deepcopy(ctx);bad['frames'][0]['speed_mps']+=1
    with pytest.raises(ValueError,match='frozen'):parse_bundle(bundle(bad),e,config())
    obj=json.loads(bundle(ctx));obj['camera_responses'].pop('CAM_R0')
    with pytest.raises(ValueError,match='coverage'):parse_bundle(json.dumps(obj),e,config())


def test_existing_batch_persists_reuses_and_invalidates_state(tmp_path,monkeypatch):
    from driversi.data.catalog import labeling,history_caption
    from driversi.data.catalog.store import Catalog
    scene,frames,log=fixture_scene(tmp_path)
    scene['descriptor'].update(source_log='source',source_group='source',token='3',conditions=[])
    c=Catalog(tmp_path/'train.sqlite','navtrain');c.ingest([scene['record']],[scene['descriptor']],{})
    monkeypatch.setattr(labeling,'annotator_profile',lambda config:{'fixed_test_profile':1,'config':{'prompt_version':'history_state_v8'}})
    calls=[]
    class Backend:
        def __init__(self,config):pass
        def __call__(self,evidence,images,context):
            calls.append(context);return bundle(context),{'images':len(images)}
    monkeypatch.setattr(history_caption,'HistoryAnnotator',Backend)
    out=tmp_path/'out'
    first=labeling.label_batch(c,config(),['3'],out,max_scenes=1,max_seconds=30)
    assert first['rows'][0]['status']=='validated'
    assert (out/f'attempt-{first["rows"][0]["attempt"]}.json').exists()
    again=labeling.label_batch(c,config(),['3'],out/'resume',max_scenes=1,max_seconds=30)
    assert again['rows'][0]['status']=='reused' and len(calls)==1
    sidecar=out/'resume'/f'attempt-{first["rows"][0]["attempt"]}.json'
    assert sidecar.exists()
    sidecar.unlink()
    labeling.label_batch(c,config(),['3'],out/'resume',max_scenes=1,max_seconds=30)
    assert sidecar.exists() and len(calls)==1
    assert c.query('training',profile=first['profile'],text='People')==[]
    c.review(first['rows'][0]['attempt'],'unit-reviewer','spot_check_pass','checked image evidence')
    assert len(c.query('training',profile=first['profile'],text='People'))==1
    c.review(first['rows'][0]['attempt'],'unit-reviewer','rejected','incorrect visual fact')
    cached=labeling.label_batch(c,config(),['3'],out/'rejected',max_scenes=1,max_seconds=30)
    assert cached['rows'][0]['status']=='cached_rejected' and len(calls)==1
    report=json.loads((out/'rejected'/f'attempt-{first["rows"][0]["attempt"]}.json').read_text())
    assert report['structured']['assessment']=='rejected'
    assert c.query('training',profile=first['profile'],text='People')==[]
    frames[0]['ego_dynamic_state'][0]=5;log.write_bytes(pickle.dumps(frames))
    second=labeling.label_batch(c,config(),['3'],out/'changed',max_scenes=1,max_seconds=30)
    assert second['rows'][0]['status']=='validated' and len(calls)==2
    assert c.annotation('3',second['profile'])['review'] is None
    c.close()


def test_resume_export_failure_does_not_poison_authoritative_attempt(tmp_path,monkeypatch):
    from driversi.data.catalog import labeling,history_caption
    from driversi.data.catalog.store import Catalog
    scene,_,_=fixture_scene(tmp_path)
    scene['descriptor'].update(source_log='source',source_group='source',token='3',conditions=[])
    c=Catalog(tmp_path/'train.sqlite','navtrain');c.ingest([scene['record']],[scene['descriptor']],{})
    monkeypatch.setattr(labeling,'annotator_profile',lambda config:{'config':{'prompt_version':'history_state_v8'}})
    calls=[]
    class Backend:
        def __init__(self,config):pass
        def __call__(self,evidence,images,context):calls.append(1);return bundle(context),{}
    monkeypatch.setattr(history_caption,'HistoryAnnotator',Backend)
    out=tmp_path/'out'
    first=labeling.label_batch(c,config(),['3'],out,max_scenes=1,max_seconds=30)
    attempt=first['rows'][0]['attempt'];c.review(attempt,'unit','spot_check_pass','checked')
    original=labeling.export_history_sidecar
    def broken(*args,**kwargs):raise OSError('simulated export unavailable')
    monkeypatch.setattr(labeling,'export_history_sidecar',broken)
    with pytest.raises(OSError,match='export unavailable'):
        labeling.label_batch(c,config(),['3'],out/'failed-export',max_scenes=1,max_seconds=30)
    assert c.db.execute('SELECT count(*) FROM attempts').fetchone()[0]==1
    assert c.annotation('3',first['profile'])['id']==attempt
    monkeypatch.setattr(labeling,'export_history_sidecar',original)
    resumed=labeling.label_batch(c,config(),['3'],out/'resume',max_scenes=1,max_seconds=30)
    assert resumed['rows'][0]['status']=='reused' and len(calls)==1
    c.close()
