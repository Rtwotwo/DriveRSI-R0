import numpy as np
import pytest
from driversi.rendering.diagnostics.scene_quality import hole_fractions,frame_allowed,small_static_appearance_allowed,projected_box_fraction
from driversi.rendering.diagnostics.worldengine_admission import visual_policy,photometric_rejection

def review():
    return dict(verdict='experimental_pass',issues=['limited_texture_blur'],dynamic_objects_consistent=True,
                multiview_temporal_consistent=True,no_severe_geometry_misalignment=True)

def test_unchanged_default_and_explicit_experiment():
    assert photometric_rejection(26.,'source27_v1') is not None
    assert photometric_rejection(26.,'experimental_scene24_v1') is None
    assert not visual_policy()['limited_appearance_experiment']
    with pytest.raises(ValueError):visual_policy('anything')

def test_large_holes_and_false_dynamic_evidence_rejected():
    m=np.zeros((100,100),np.uint8);m[:15,:15]=255
    q=hole_fractions(m);assert q['largest_component_fraction']==.0225
    assert not frame_allowed(review(),q)
    assert frame_allowed(review(),hole_fractions(np.zeros_like(m)))
    assert not frame_allowed({**review(),'dynamic_objects_consistent':False},hole_fractions(np.zeros_like(m)))
    assert not frame_allowed({**review(),'issues':['missing_car']},hole_fractions(np.zeros_like(m)))

def test_small_static_uncertainty_never_waives_geometry_or_vehicles():
    e=dict(physical_immobile=True,physical_background_geometry={'gaussian_count':5})
    b=dict(physical_box_alignment_confirmed=True,appearance_uncertainty_accepted_for_experiment=True)
    assert small_static_appearance_allowed('traffic_cone',e,b,.001)
    for kind,size in [('vehicle',.001),('pedestrian',.001),('traffic_cone',.1)]:
        assert not small_static_appearance_allowed(kind,e,b,size)
    assert not small_static_appearance_allowed('traffic_cone',e,{**b,'physical_box_alignment_confirmed':False},.001)

def test_void_mask_is_binary():
    with pytest.raises(ValueError):hole_fractions(np.ones((4,4)))

def test_projection_bound_rejects_near_plane_and_contains_distorted_edges():
    from itertools import product
    points=np.array(list(product([-.1,.1],[-.1,.1],[9.9,10.1])))
    camera=dict(camera_to_world=np.eye(4)[:3].tolist(),distortion=[-.3,.1,0,0,0],
                distorted_intrinsic=[[500,0,320],[0,500,180],[0,0,1]],native_calibration_size=[640,360])
    assert 0<projected_box_fraction(points,camera)<.005
    points[:,2]=.1
    assert projected_box_fraction(points,camera)==float('inf')


def test_source25_keeps_critical_road_checks_and_old_defaults():
    from driversi.rendering.diagnostics.scene_quality import CRITICAL_CHECKS
    assert photometric_rejection(24.99, 'experimental_scene25_v1')
    assert photometric_rejection(25., 'experimental_scene25_v1') is None
    assert photometric_rejection(26., 'source27_v1')
    holes = dict(total_fraction=0., largest_component_fraction=0.)
    v = {**review(), **dict.fromkeys(CRITICAL_CHECKS, True)}
    assert frame_allowed(v, holes, profile='experimental_scene25_v1')
    for key in CRITICAL_CHECKS:
        assert not frame_allowed({**v, key: False}, holes, profile='experimental_scene25_v1')
    assert not frame_allowed({**v, 'issues': ['missing_pedestrian']}, holes, profile='experimental_scene25_v1')


def test_projected_fraction_uses_intrinsic_output_resolution():
    from itertools import product
    points = np.array(list(product([-.1,.1],[-.1,.1],[9.9,10.1])))
    camera = dict(camera_to_world=np.eye(4)[:3].tolist(), distortion=[0]*5,
                  distorted_intrinsic=[[500,0,320],[0,500,180],[0,0,1]],
                  native_calibration_size=[1280,720])
    full = projected_box_fraction(points, camera)
    actual = projected_box_fraction(points, {**camera, 'output_image_size': [640,360]})
    assert actual == pytest.approx(4*full)


def test_review_preparation_seals_masks_and_supporting_evidence(tmp_path):
    import json
    from driversi.rsi.plugins.worldengine import prepare_admission
    from driversi.agents.backends.native import digest
    mask=tmp_path/'mask.png';mask.write_bytes(b'bound-mask')
    evidence=tmp_path/'scope.json';evidence.write_text('{}')
    path=tmp_path/'review.json';path.write_text(json.dumps(dict(
        frames=[dict(void_mask=str(mask))], supporting_evidence=[str(evidence)])))
    config,_=prepare_admission(dict(visual_review=str(path),
        visual_quality_profile='experimental_scene25_v1'),tmp_path,None)
    assert config['files'][str(mask)] == digest(mask)
    assert config['files'][str(evidence)] == digest(evidence)
    mask.write_bytes(b'changed');assert config['files'][str(mask)] != digest(mask)
