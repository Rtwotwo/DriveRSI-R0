"""Source and executed-branch camera coordinate regressions."""
import copy
import numpy as np
import pytest
from driversi.rendering.whitebox.conditioning import frame_camera, object_map
from driversi.rendering.whitebox.projection_profiles import display_geometry, PLANAR


def rotation_y(angle):
    c,s=np.cos(angle),np.sin(angle)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])


def fixture(pitch=0.0, z=0.0):
    pose=np.eye(4);pose[:3,:3]=rotation_y(pitch);pose[2,3]=z
    mount=np.eye(4);mount[:3,3]=[1.3,.1,1.6]
    lidar=np.eye(4);lidar[:3,3]=[.9,.2,1.8]
    raw={'lidar2ego':lidar,'cams':{'CAM_F0':{
        'sensor2lidar_rotation':mount[:3,:3], 'sensor2lidar_translation':mount[:3,3],
        'cam_intrinsic':np.eye(3),'distortion':np.zeros(5),'data_path':'image.jpg'}},'token':'x'}
    return raw,pose,mount


def test_camera_mount_does_not_apply_nonidentity_lidar_twice():
    raw,pose,mount=fixture(.2,3)
    cam=frame_camera(raw,'CAM_F0',pose)
    expected=pose@mount@np.diag([1,-1,-1,1])
    np.testing.assert_allclose(cam['camera_to_world'],expected[:3])
    np.testing.assert_allclose(cam['camera_mount_to_ego'],mount@np.diag([1,-1,-1,1]))


def test_source_centers_invert_upstream_full_se3_not_yaw_only():
    raw,pose,_=fixture(.15,5)
    world_center=np.array([15.,4.,6.])
    native=pose[:3,:3].T@(world_center-pose[:3,3])
    raw.update(ego2global=pose,ego2global_rotation=[np.cos(.075),0,np.sin(.075),0],
               anns={'track_tokens':['a'],'gt_names':['vehicle'],'gt_boxes':[np.r_[native,4,2,1.5,.3]]})
    row=object_map(raw,np.array([1.,2.,3.]))['a']
    np.testing.assert_allclose(row['pose_xyz_yaw'][:3],world_center-[1,2,3])
    assert row['pose_xyz_yaw'][3]==pytest.approx(.3)


def test_branch_planar_camera_uses_mount_even_if_source_body_tilts():
    outputs=[]
    for pitch,z in [(.1,2),(-.08,4)]:
        raw,source_pose,mount=fixture(pitch,z)
        camera=frame_camera(raw,'CAM_F0',source_pose)
        executed=np.eye(4);executed[2,3]=z
        frame={'ego_rear_axle_to_local':executed.tolist(),'sampling':'executed_physics_state',
               'objects':[{'actor_id':'a','pose_xyz_yaw':[8,1,z+3,.1],'dimensions_lwh':[4,2,1.5]}]}
        before=copy.deepcopy(frame);original=copy.deepcopy(camera)
        out,objects=display_geometry(frame,camera,PLANAR)
        outputs.append(out['camera_to_world'])
        np.testing.assert_allclose(out['camera_to_world'],(mount@np.diag([1,-1,-1,1]))[:3])
        assert objects[0]['pose_xyz_yaw'][2]==.75
        assert frame==before and camera==original
    np.testing.assert_allclose(*outputs)


def test_branch_without_mount_cannot_silently_keep_source_bounce():
    raw,pose,_=fixture(.1,3);camera=frame_camera(raw,'CAM_F0',pose)
    del camera['camera_mount_to_ego']
    frame={'ego_rear_axle_to_local':np.eye(4).tolist(),'sampling':'executed_physics_state','objects':[]}
    with pytest.raises(ValueError,match='explicit native camera mount'):display_geometry(frame,camera,PLANAR)


@pytest.mark.parametrize('bad', [[[1]], np.full((4,4),np.nan).tolist(), np.zeros((4,4)).tolist()])
def test_bad_mount_fails_instead_of_relocating_camera(bad):
    raw,pose,_=fixture();camera=frame_camera(raw,'CAM_F0',pose);camera['camera_mount_to_ego']=bad
    frame={'ego_rear_axle_to_local':pose.tolist(),'objects':[]}
    with pytest.raises(ValueError,match='Invalid camera mount'):display_geometry(frame,camera,PLANAR)


def test_render_provenance_does_not_call_executed_motion_measured_source():
    from driversi.rendering.whitebox.projection_profiles import profile_description
    source=profile_description(PLANAR, state_role="source_log")
    branch=profile_description(PLANAR, state_role="executed_branch")
    assert "measured/interpolated source" in source["position_basis"]
    assert "executed physics" in branch["position_basis"]
    assert "measured/interpolated" not in branch["position_basis"]
    assert branch["state_role"]=="executed_branch"
    with pytest.raises(ValueError):profile_description(PLANAR, state_role="unknown")


def test_all_profiles_distinguish_executed_camera_provenance():
    from driversi.rendering.whitebox.projection_profiles import profile_description, MEASURED, LEGACY
    for profile in (PLANAR, MEASURED, LEGACY):
        d=profile_description(profile, "executed_branch")
        assert d["state_role"]=="executed_branch"
        assert "executed" in d["camera_basis"]
        assert "measured/interpolated" not in d["camera_basis"]
