import copy
import json
import numpy as np
import pytest

from driversi.data.catalog.scene_description import (VISUAL_FIELDS, build_description,
    description_schema, parse_observations, parse_generation, scoped_measurements)
from driversi.rendering.whitebox.conditioning import build_source
from driversi.rendering.whitebox.motion_audit import audit_source_motion


def raw_and_bundle(tmp_path):
    raw=[]
    for i in range(17):
        ego=np.eye(4);ego[0,3]=i
        cams={k:dict(sensor2lidar_rotation=np.eye(3),sensor2lidar_translation=[0,0,1],
                    cam_intrinsic=np.eye(3),distortion=np.zeros(5),data_path='test.jpg')
              for k in ('CAM_F0','CAM_L0','CAM_R0','CAM_B0')}
        objects=[('a','vehicle',[20-i,0,1,4,2,2,np.deg2rad(179+i*.2)])]
        if i>=8:objects.append(('b','pedestrian',[2,3,1,1,1,2,0]))
        raw.append(dict(token=str(i),log_name='log',map_location='map',timestamp=i*500000,
            ego2global=ego,ego2global_rotation=[1,0,0,0],ego2global_translation=[i,0,0],
            lidar2ego=np.eye(4),cams=cams,
            anns=dict(track_tokens=[o[0] for o in objects],gt_names=[o[1] for o in objects],gt_boxes=[o[2] for o in objects])))
    file=tmp_path/'source.pkl';file.write_bytes(b'fixture')
    record=dict(kind='original_navtrain',anchor_index=4,token='4',source_log='log',log_path=str(file))
    return raw,build_source(record,frames=raw)


def observations():
    return {**{k:dict(text='unknown',evidence_ids=[]) for k in VISUAL_FIELDS},'distinctive_features':[]}


def test_fixed_schema_and_future_isolation(tmp_path):
    _,b=raw_and_bundle(tmp_path)
    a=build_description(b,'retrieval_history')
    b2=copy.deepcopy(b)
    for f in b2['frames'][21:]:f['ego_rear_axle_to_local'][0][3]+=200;f['objects']=[]
    assert build_description(b2,'retrieval_history')==a
    assert build_description(b2,'video_clip')!=build_description(b,'video_clip')
    assert a['video_generation'] is None
    assert build_description(b,'video_clip')['retrieval'] is None
    try:import jsonschema
    except ImportError:return
    jsonschema.validate(a,description_schema())
    jsonschema.validate(build_description(b,'video_clip'),description_schema())


def test_unknown_evidence_and_duplicate_keys_rejected():
    obs=observations();assert parse_observations(json.dumps(obs),[])==obs
    obs['overview']={'text':'Cars','evidence_ids':['CAM_F0_f+1']}
    with pytest.raises(ValueError,match='Unknown'):parse_observations(json.dumps(obs),['CAM_F0_f+0'])
    obs['overview']={'text':'Cars','evidence_ids':[]}
    with pytest.raises(ValueError,match='every actual'):parse_observations(json.dumps(obs),[])
    with pytest.raises(ValueError,match='Duplicate'):parse_observations('{"overview":1,"overview":2}',[])


def test_only_single_root_syntax_repair_without_content_invention():
    expected=observations();raw=json.dumps(expected)
    parsed,repair=parse_generation(raw[:-1],[])
    assert parsed==expected and repair=='append_single_final_root_brace'
    assert parse_generation(raw,[])[1] is None
    with pytest.raises(ValueError):parse_generation(raw[:-3],[])
    broken=json.dumps({'overview':expected['overview'],'distinctive_features':[]})[:-1]
    with pytest.raises(ValueError):parse_generation(broken,[])


def test_review_is_per_claim_and_video_scope_not_retrieval(tmp_path):
    _,b=raw_and_bundle(tmp_path);obs=observations()
    obs['overview']=dict(text='Visible trees beside the road.',evidence_ids=['CAM_F0_f+0'])
    obs['road_surface']=dict(text='Wet asphalt.',evidence_ids=['CAM_F0_f+0'])
    ev=[dict(id='CAM_F0_f+0',relative_time_s=0)]
    draft=build_description(b,'retrieval_history',observations=obs,evidence=ev)
    assert 'trees' not in draft['retrieval']['text']
    reviewed=build_description(b,'retrieval_history',observations=obs,evidence=ev,
           approved_paths=['overview'],review=dict(reviewer='test',note='Reviewed only overview'))
    assert 'trees' in reviewed['retrieval']['text'] and 'Wet asphalt' not in reviewed['retrieval']['text']
    with pytest.raises(ValueError,match='review'):build_description(b,'video_clip',observations=obs,evidence=ev,approved_paths=['overview'])
    with pytest.raises(ValueError,match='Future'):build_description(b,'retrieval_history',evidence=[dict(id='future',relative_time_s=1)])


def test_multicamera_claim_not_mistaken_for_camera_specific_appearance(tmp_path):
    _,b=raw_and_bundle(tmp_path);obs=observations()
    obs['overview']=dict(text='A rear building and a front road.',evidence_ids=['CAM_F0_f+0','CAM_B0_f+0'])
    ev=[dict(id=x,relative_time_s=0) for x in obs['overview']['evidence_ids']]
    value=build_description(b,'video_clip',observations=obs,evidence=ev,approved_paths=['overview'],review=dict(reviewer='test',note='Synthetic test'))
    assert 'rear building' in value['video_generation']['positive_prompt']
    assert all('rear building' not in s for s in value['video_generation']['camera_prompts'].values())


def test_source_motion_and_stationary_global_object(tmp_path):
    raw,b=raw_and_bundle(tmp_path);report=audit_source_motion(b,raw)
    assert report['passed'],report
    assert report['actor_states_checked']>71
    assert report['maximum_errors']['object_position_m']<1e-6
    for f in b['frames']:assert np.allclose(f['objects'][0]['pose_xyz_yaw'][:3],[16,0,1])
    assert b['frames'][39]['objects'][0]['actor_id']=='a'
    assert len(b['frames'][39]['objects'])==1
    assert len(b['frames'][40]['objects'])==2


@pytest.mark.parametrize('fault',['position','heading','identity','camera','clock','nan','intrinsics','distortion','missing_camera','duplicate_id'])
def test_motion_audit_detects_corrupt_single_frame(tmp_path,fault):
    raw,b=raw_and_bundle(tmp_path);f=b['frames'][31]
    if fault=='position':f['objects'][0]['pose_xyz_yaw'][0]+=1
    elif fault=='heading':f['objects'][0]['pose_xyz_yaw'][3]+=.2
    elif fault=='identity':f['objects'][0]['instance_id']=77
    elif fault=='camera':f['cameras'][0]['camera_to_world'][0][3]+=1
    elif fault=='nan':f['objects'][0]['pose_xyz_yaw'][0]=float('nan')
    elif fault=='intrinsics':f['cameras'][0]['K'][0][0]*=2
    elif fault=='distortion':f['cameras'][0]['distortion'][0]+=10
    elif fault=='missing_camera':f['cameras']=[]
    elif fault=='duplicate_id':
        for row in b['frames']:
            for o in row['objects']:o['instance_id']=1
    else:f['timestamp_us']+=1
    assert not audit_source_motion(b,raw)['passed']


def test_branch_cannot_be_certified_as_original(tmp_path):
    raw,b=raw_and_bundle(tmp_path);b['role']='executed_branch'
    with pytest.raises(ValueError,match='physical branch'):audit_source_motion(b,raw)


def test_wrong_valid_camera_mount_cannot_pass_source_audit(tmp_path):
    raw,bundle=raw_and_bundle(tmp_path)
    assert audit_source_motion(bundle,raw)['passed']
    bundle['frames'][10]['cameras'][0]['camera_mount_to_ego'][2][3]+=.2
    result=audit_source_motion(bundle,raw)
    assert not result['passed']
    assert any(x.get('field')=='camera_pose' for x in result['issues'])


def test_independent_dimension_audit_and_legacy_hold(tmp_path):
    raw, _ = raw_and_bundle(tmp_path)
    for i, frame in enumerate(raw):
        frame['anns']['gt_boxes'][0][3:6] = [4+i*.1, 2+i*.02, 1.5+i*.03]
    record = dict(kind='original_navtrain', anchor_index=4, token='4',
                  source_log='log', log_path=str(tmp_path/'source.pkl'))
    bundle = build_source(record, frames=raw)
    assert audit_source_motion(bundle, raw)['passed']
    np.testing.assert_allclose(bundle['frames'][3]['objects'][0]['dimensions_lwh'], [4.06, 2.012, 1.518])
    bad = copy.deepcopy(bundle)
    bad['frames'][3]['objects'][0]['dimensions_lwh'] = [4, 2, 1.5]
    assert not audit_source_motion(bad, raw)['passed']
    legacy = copy.deepcopy(bundle); legacy.pop('dimension_sampling')
    for i, frame in enumerate(legacy['frames']):
        for obj in frame['objects']:
            index = raw[i//5]['anns']['track_tokens'].index(obj['actor_id'])
            obj['dimensions_lwh'] = raw[i//5]['anns']['gt_boxes'][index][3:6]
    assert audit_source_motion(legacy, raw)['passed']
    legacy['dimension_sampling'] = 'unsupported_guess'
    with pytest.raises(ValueError, match='dimension sampling'):
        audit_source_motion(legacy, raw)


def test_executed_branch_keeps_world_dimensions_and_truthful_metadata(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from driversi.rendering.whitebox.conditioning import branch_from_source, describe
    import driversi.rendering.whitebox.exporter as exporter
    import driversi.rendering.bev.replay as replay
    _, source = raw_and_bundle(tmp_path)
    original = copy.deepcopy(source)
    rows = [dict(time_seconds=i/10, lifecycle={}, signals={}, objects=[
        dict(actor_id='ego', pose_xyz_yaw=[i/10, 0, 0, 0], dimensions_lwh=[5,2,2]),
        dict(actor_id='a', pose_xyz_yaw=[20, 0, 0, 0], dimensions_lwh=[4+i*.01,2,1.5])]) for i in range(71)]
    monkeypatch.setattr(exporter, 'interaction_frames', lambda *args: rows)
    monkeypatch.setattr(replay, '_load_run', lambda *args: {})
    monkeypatch.setattr(replay, '_scene_key', lambda *args: 'test')
    scene = SimpleNamespace(actor_ids=['ego','a'], metadata=dict(source_start_timestamp_us=0,
        origin_xyz=source['origin_xyz'], source_actor_kinds={'a':'vehicle'}))
    file=tmp_path/'rollout.json'; file.write_text('{}')
    branch=branch_from_source(source,scene,file)
    assert source == original
    assert branch['dimension_sampling']=='executed_physics_unmodified'
    for i, frame in enumerate(branch['frames']):
        assert frame['objects'][0]['dimensions_lwh']==rows[i]['objects'][1]['dimensions_lwh']
        assert frame['objects'][0]['dimension_basis']=='executed_physics'
    assert not any('bracket-interpolated annotations' in x for x in describe(branch)['uncertainty'])
    assert any('without temporal interpolation' in x for x in describe(branch)['uncertainty'])
