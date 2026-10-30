import importlib.util
from pathlib import Path

import numpy as np
import pytest

path = Path(__file__).resolve().parents[1] / "src/rendering/video/streetcrafter/gs_conditions.py"
spec = importlib.util.spec_from_file_location("gs_conditions", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_dense_matching_keeps_unique_actual_timestamps():
    frames = [{"token":str(i),"timestamp":1000000+i*100000+123} for i in range(25)]
    actual = module.match_sources(frames, 1000000)
    assert [f["timestamp"] for f in actual] == [f["timestamp"] for f in frames]
    with pytest.raises(ValueError,match="Repeated"):
        module.match_sources(frames[::2],1000000)
    with pytest.raises(ValueError,match="tolerance"):
        module.match_sources(frames,1000000,tolerance_us=100)
    with pytest.raises(ValueError,match="another physical tick"):
        module.match_sources([{"token":"shifted","timestamp":1055000}],1000000,count=1)


def test_same_camera_depth_and_black_point_occupancy():
    calibration={"camera_to_world":np.c_[np.eye(3),[10,20,30]].tolist(),
                 "distorted_intrinsic":[[1,0,2],[0,1,2],[0,0,1]],"distortion":[0]*5,
                 "output_calibration_size":[5,5]}
    points=np.array([[10,20,32],[10,20,31],[10,20,29]])
    colors=np.array([[1,0,0],[0,0,0],[0,1,0]])
    rgb,mask=module.project_cloud(points,colors,calibration)
    assert mask.sum()==255
    assert mask[2,2]==255
    assert np.array_equal(rgb[2,2],[0,0,0])


def test_rigid_transport_preserves_native_pivot_and_shape():
    points=np.array([[2.,1,0],[3,1,0]])
    pivot=np.array([2.,1,0]); rotation=np.array([[0,-1,0],[1,0,0],[0,0,1]])
    shifted=module.transport(points,pivot,rotation,np.array([4,5,6]))
    assert np.array_equal(shifted,[[6,6,6],[6,7,6]])


def test_ego_reference_translates_origins_and_uses_center():
    from pyquaternion import Quaternion
    pose=np.eye(4);pose[:3,:3]=Quaternion(axis=[0,0,1],angle=.4).rotation_matrix
    pose[:3,3]=[2,3,4]
    ref=module.ego_reference({"ego2global_original":pose},np.array([100,200,10]),np.array([101,202,12]))
    assert np.allclose(ref,[1+1.461*np.cos(.4),1+1.461*np.sin(.4),2,.4])

