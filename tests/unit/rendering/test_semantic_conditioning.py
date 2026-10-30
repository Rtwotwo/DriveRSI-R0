import copy
import json

import numpy as np
import pytest

from driversi.data.catalog.condition_caption import assemble, parse
from driversi.rendering.whitebox.conditioning import bracket, describe, palette, resample_objects, interpolate_transform


def test_rotation_interpolation_preserves_endpoints_and_shortest_arc():
    def matrix(theta):
        c,s=np.cos(theta),np.sin(theta)
        return np.array([[c,-s,0,0],[s,c,0,0],[0,0,1,0],[0,0,0,1.]])
    a,b=matrix(np.deg2rad(179)),matrix(np.deg2rad(-179));b[0,3]=2
    assert np.allclose(interpolate_transform(a,b,0),a)
    assert np.allclose(interpolate_transform(a,b,1),b)
    middle=interpolate_transform(a,b,.5)
    assert np.allclose(middle[:3,:3],matrix(np.pi)[:3,:3])
    assert middle[0,3]==1
    with pytest.raises(ValueError):interpolate_transform(a,b,1.1)


def obj(yaw=0):
    return {'actor_id':'a','kind':'vehicle','pose_xyz_yaw':[0,0,0,yaw],'dimensions_lwh':[4,2,1.5]}


def test_no_extrapolation_or_hidden_sampling_gap():
    assert bracket([0,500000,1000000],100000)==(0,1,.2)
    with pytest.raises(ValueError,match='extrapolation'):bracket([0,500000],700000)
    with pytest.raises(ValueError,match='sampling gap'):bracket([0,1000000],100000)


def test_interpolation_wraps_heading_without_inventing_vehicle_type():
    a=obj(np.deg2rad(179));b=obj(np.deg2rad(-179));b['pose_xyz_yaw'][0]=1
    row=resample_objects({'a':a},{'a':b},.5,.5)[0]
    assert abs(abs(row['pose_xyz_yaw'][3])-np.pi)<1e-8
    assert row['pose_xyz_yaw'][0]==.5
    assert row['subtype']=='unknown' and row['velocity_xyz_mps']==[2.,0.,0.]


def test_missing_identity_is_not_interpolated_and_late_birth_not_early():
    b=obj();b['actor_id']='b'
    rows=resample_objects({'a':obj()},{'b':b},.4,.5)
    assert [x['actor_id'] for x in rows]==['a']
    assert rows[0]['sample_basis']=='previous_observation_held_presence_uncertain'
    assert rows[0]['velocity_xyz_mps'] is None


def test_class_changes_are_rejected():
    b=obj();b['kind']='pedestrian'
    with pytest.raises(ValueError,match='class changed'):resample_objects({'a':obj()},{'a':b},.4,.5)


@pytest.mark.parametrize('kind', ['vehicle', 'pedestrian', 'bicycle', 'traffic_cone',
                                'barrier', 'construction_zone_sign', 'generic_object'])
def test_box_size_uses_pose_clock_without_boundary_step_or_mutation(kind):
    a, b = obj(), obj()
    a['kind'] = b['kind'] = kind
    b['dimensions_lwh'] = [5, 2.5, 2.0]
    original = copy.deepcopy((a, b))
    rows = [resample_objects({'a':a}, {'a':b}, t, .5)[0] for t in np.linspace(0, 1, 6)]
    sizes = np.array([row['dimensions_lwh'] for row in rows])
    np.testing.assert_allclose(sizes[0], a['dimensions_lwh'])
    np.testing.assert_allclose(sizes[-1], b['dimensions_lwh'])
    np.testing.assert_allclose(np.diff(sizes, axis=0), np.tile([.2, .1, .1], (5, 1)))
    assert all(row['dimension_basis'] == 'bracket_interpolation' for row in rows)
    assert (a, b) == original
    exact = resample_objects({'a':b}, {'a':b}, 0, 0, exact=True)[0]
    assert exact['dimensions_lwh'] == b['dimensions_lwh']
    assert exact['dimension_basis'] == 'observed_source'


def test_missing_identity_does_not_borrow_other_object_size():
    a, b = obj(), obj(); b.update(actor_id='other', dimensions_lwh=[20, 5, 4])
    row = resample_objects({'a':a}, {'other':b}, .9, .5)[0]
    assert row['dimensions_lwh'] == a['dimensions_lwh']
    assert row['dimension_basis'] == 'previous_observation_held_presence_uncertain'


@pytest.mark.parametrize('size', [[0, 2, 1], [4, float('nan'), 1], [4, 2]])
def test_invalid_interpolated_dimensions_rejected(size):
    b = obj(); b['dimensions_lwh'] = size
    with pytest.raises(ValueError, match='dimensions'):
        resample_objects({'a':obj()}, {'a':b}, .5, .5)


def test_palette_colors_never_claim_paint_or_subtype():
    d=palette();assert 'not physical paint' in d['encoding']
    assert d['colors']['vehicle']!=d['colors']['pedestrian']
    assert 'unspecified' in d['vehicle_subtypes']


def fixture_bundle():
    rows=[]
    for i in range(71):
        o=obj();o.update(instance_id=1,sample_basis='observed_source')
        rows.append(dict(time_seconds=i/10,ego_rear_axle_to_local=np.eye(4).tolist(),objects=[o]))
    return dict(frames=rows,actors={'a':{'instance_id':1,'kind':'vehicle'}},anchor_tick=20,limitations=[],map_layers={})


def test_retrieval_facts_invariant_to_future_actor_changes():
    a=fixture_bundle();first=describe(a)
    b=copy.deepcopy(a);o=obj();o.update(actor_id='future',instance_id=2,sample_basis='observed_source')
    b['actors']['future']={'instance_id':2,'kind':'vehicle'}
    for f in b['frames'][21:]:f['objects'].append(o);f['ego_rear_axle_to_local'][0][3]=100
    assert describe(b)==first
    assert describe(b,'video_clip')!=describe(a,'video_clip')


def test_unreviewed_vlm_does_not_overwrite_facts_or_promote_training():
    facts=describe(fixture_bundle(),'video_clip')
    visual={k:'unknown' for k in ['road_appearance','surroundings','lighting','occluders','distinctive_details']}
    visual['distinctive_details']='An unsupported red bus.'
    result=assemble(facts,visual)
    assert 'red bus' not in result['generation_prompt']
    assert not result['trainable']
    assert 'red bus' in result['vlm_observations']['distinctive_details']
    assert result['retrieval_text'] is None
    assert 'net displacement' in result['structured_text']


def test_facts_are_plain_json_with_explicit_presence_basis():
    facts=describe(fixture_bundle())
    restored=json.loads(json.dumps(facts,allow_nan=False))
    assert restored==facts
    assert 'observed_times_seconds' not in restored['actor_presence'][0]
    assert restored['actor_presence'][0]['sample_bases']==['observed_source']


def test_caption_schema_unknown_supported_and_duplicates_rejected():
    d={k:'unknown' for k in ['road_appearance','surroundings','lighting','occluders','distinctive_details']}
    assert parse(json.dumps(d))==d
    with pytest.raises(ValueError):parse('{"road_appearance":"a","road_appearance":"b"}')
    with pytest.raises(ValueError):parse(json.dumps({**d,'invented_actor':'bus'}))
