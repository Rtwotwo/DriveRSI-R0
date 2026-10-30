import copy
import pytest
from driversi.rsi.experiments.readiness import analyze_render

CAMS=('CAM_F0','CAM_L0','CAM_R0','CAM_B0')

def data():
    ticks=list(range(0,71,5))
    report=dict(source_log='train',asset_id='asset',history_ticks=[5,10,15,20],
        future_ticks=ticks[5:],terrain_fully_supported=True,
        frames=[dict(tick=t,physical_missing_appearance=['small-static'],
            static_background_candidates=[dict(actor_id='small-static',physical_immobile=True,
                physical_background_geometry={'gaussian_count':3})],
            cameras={c:dict(terrain_fully_supported=True) for c in CAMS})for t in ticks])
    source=dict(source_mode='canonical_source_pose_without_motion_v1',
        frames=[dict(tick=t,camera=c,measured_psnr_db=28.)for t in ticks for c in CAMS])
    return dict(render_report=report,native_future_points=10),source


def test_no_automatic_admission_or_model_failure_from_good_source_psnr():
    r,s=data();out=analyze_render(r,s,'source27_v1')
    assert out['status']=='independent_visual_review_required'
    assert out['unresolved_appearance_tokens']==['small-static']
    assert not out['trainable'] and not out['student_failure_established']


def test_complete_last_second_required():
    r,s=data();s['frames']=s['frames'][:-8]
    with pytest.raises(ValueError,match='missing'):analyze_render(r,s,'source27_v1')


def test_quality_and_terrain_failures_remain_separate():
    r,s=data();r['render_report']['terrain_fully_supported']=False
    r['render_report']['frames'][-1]['cameras']['CAM_B0']['terrain_fully_supported']=False
    for row in s['frames']:row['measured_psnr_db']=22.
    out=analyze_render(r,s,'source27_v1')
    assert out['blockers']==['finite_terrain_support_missing','source_reconstruction_psnr_below_27db']
    assert out['status']=='repair_required'


def test_existing_profile_is_not_silently_changed():
    r,s=data()
    for row in s['frames']:row['measured_psnr_db']=25.
    assert analyze_render(r,s,'source27_v1')['blockers']
    assert not analyze_render(r,s,'experimental_source24_v1')['blockers']
    assert not analyze_render(r,s,'experimental_source24_v1')['trainable']


def test_missing_frame_summary_cannot_override_actual_terrain():
    r,s=data();r['render_report']['frames'][-1]['cameras']['CAM_B0']['terrain_fully_supported']=False
    with pytest.raises(ValueError,match='summary'):analyze_render(r,s,'source27_v1')


def test_visible_missing_static_candidate_is_mechanical_repair():
    r,s=data();frame=r['render_report']['frames'][-1]
    frame['static_background_candidates']=[]
    out=analyze_render(r,s,'source27_v1')
    assert out['blockers']==['physical_object_appearance_unresolved']
    assert len(out['missing_appearance_prerequisites'])==4
    assert all(x['tick']==70 for x in out['missing_appearance_prerequisites'])


def test_box_behind_camera_is_not_an_appearance_prerequisite():
    import itertools
    r,s=data();frame=r['render_report']['frames'][-1]
    frame['static_background_candidates']=[]
    frame['unresolved_physical_boxes']={'small-static':list(itertools.product([-1.,1.],[-1.,1.],[-3.,-2.]))}
    for cam in frame['cameras'].values():
        cam.update(camera_to_world=[[1,0,0,0],[0,1,0,0],[0,0,1,0]],output_ray_bounds=[-1,1,-1,1])
    assert analyze_render(r,s,'source27_v1')['missing_appearance_prerequisites']==[]
    frame['unresolved_physical_boxes']['small-static']=[(x,y,-z) for x,y,z in frame['unresolved_physical_boxes']['small-static']]
    assert len(analyze_render(r,s,'source27_v1')['missing_appearance_prerequisites'])==4


def test_unused_binding_attempt_is_not_a_physical_object_failure():
    r,s=data();r['render_report']['static_background_binding_errors']={'absent':'never present'}
    out=analyze_render(r,s,'source27_v1')
    assert out['unused_static_binding_attempt_errors']=={'absent':'never present'}
    assert out['active_static_binding_errors']=={}
    assert 'audit_static_lifecycle_and_geometry_before_background_binding' not in out['repair_actions']


@pytest.mark.parametrize('mutation',[{'physical_immobile':False},{'physical_background_geometry':{'gaussian_count':0}}])
@pytest.mark.parametrize('profile',['source27_v1','experimental_source24_v1'])
def test_static_candidate_geometry_cannot_be_fixed_by_a_review(mutation,profile):
    r,s=data();r['render_report']['frames'][-1]['static_background_candidates'][0].update(mutation)
    out=analyze_render(r,s,profile)
    assert out['blockers']==['static_physical_registration_unresolved']
    assert len(out['missing_appearance_prerequisites'])==4
