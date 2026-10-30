"""Regression against producer semantics and explicit planar display invariants."""
import copy
import numpy as np
import pytest

from driversi.rendering.whitebox.conditioning import object_map, frame_camera
from driversi.rendering.whitebox.projection_profiles import display_geometry, PLANAR, MEASURED
from driversi.rendering.camera_review.exporter import corners


def transform(q, xyz):
    w,x,y,z = np.array(q)/np.linalg.norm(q)
    t = np.eye(4)
    t[:3,:3] = [[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
               [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
               [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]
    t[:3,3] = xyz
    return t


def test_producer_inverse_center_transform_is_undone_with_full_rotation():
    q = np.array([.8,.1,-.2,.3]); q /= np.linalg.norm(q)
    t = transform(q,[10,20,30]);world = np.array([52,-15,31.4])
    w,x,y,z = q;source_yaw = np.arctan2(2*(w*z-x*y),1-2*(y*y+z*z))
    box = np.r_[t[:3,:3].T@(world-t[:3,3]),4,2,1.8,.4-source_yaw]
    raw = dict(ego2global=t,ego2global_rotation=q,ego2global_translation=t[:3,3],
               anns=dict(track_tokens=['car'],gt_names=['vehicle'],gt_boxes=[box]))
    result = object_map(raw,np.array([1,2,3]))['car']
    np.testing.assert_allclose(result['pose_xyz_yaw'],[* (world-[1,2,3]),.4],atol=1e-12)


def test_legacy_camera_field_is_camera_to_ego_not_camera_to_lidar():
    pose = transform([.9,.1,.1,.2],[3,4,5])
    extra = transform([1,0,0,0],[1,0,3])
    raw=dict(lidar2ego=extra,cams={'CAM_F0':dict(sensor2lidar_rotation=np.eye(3),
            sensor2lidar_translation=[1,0,1.5],cam_intrinsic=np.eye(3),distortion=np.zeros(5),data_path='a')},token='a')
    actual=frame_camera(raw,'CAM_F0',pose)
    mount=np.eye(4);mount[:3,3]=[1,0,1.5]
    np.testing.assert_allclose(actual['camera_to_world'],(pose@mount@np.diag([1,-1,-1,1]))[:3])


def test_planar_view_does_not_mutate_measured_geometry_or_invent_motion():
    pose=transform([.98,.08,.12,.1],[10,20,1.8])
    mount=np.eye(4);mount[:3,3]=[1,0,1.5]
    camera=dict(camera_to_world=(pose@mount)[:3].tolist())
    obj=dict(actor_id='car',kind='vehicle',pose_xyz_yaw=[30,20,2.2,.6],dimensions_lwh=[4,2,1.7])
    frame=dict(ego_rear_axle_to_local=pose.tolist(),objects=[obj]);before=copy.deepcopy(frame)
    actual,objects=display_geometry(frame,camera,PLANAR)
    assert frame==before
    assert objects[0]['pose_xyz_yaw'][:2]==obj['pose_xyz_yaw'][:2]
    assert objects[0]['pose_xyz_yaw'][3]==obj['pose_xyz_yaw'][3]
    assert min(corners(objects[0]['pose_xyz_yaw'][:3],obj['dimensions_lwh'],.6)[:,2])==0
    assert actual['camera_to_world'][2][3]==pytest.approx(1.5)
    measured,original=display_geometry(frame,camera,MEASURED)
    assert measured['camera_to_world']==camera['camera_to_world'] and original==frame['objects']
    with pytest.raises(ValueError):display_geometry(frame,camera,'auto_guess')
