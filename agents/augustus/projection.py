"""Conservative crop-frustum exclusion for branch language annotations."""
from functools import lru_cache
import numpy as np

@lru_cache(maxsize=48)
def _rays(intrinsic, distortion, size, name):
    import cv2
    k=np.asarray(intrinsic,float).reshape(3,3)
    d=np.asarray(distortion,float)
    # All output pixel rays in native crop, plus a conservative angular margin.
    w,h=size
    left,top,right,bottom=(0,0,w,h*1024/1080) if name=='CAM_F0' else (w*416/1920,h*28/1080,w*1504/1920,h*1052/1080)
    yy,xx=np.mgrid[int(np.floor(top)):int(np.ceil(bottom))+1,int(np.floor(left)):int(np.ceil(right))+1]
    pix=np.stack([xx.ravel(),yy.ravel()],1).astype(np.float64)
    rays=cv2.undistortPointsIter(pix[:,None],k,d,None,None,
        (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,50,1e-10)).reshape(-1,2)
    if not np.isfinite(rays).all():
        raise ValueError('Nonfinite crop rays')
    low,high=rays.min(0)-0.01,rays.max(0)+0.01
    return [float(low[0]),float(high[0]),float(low[1]),float(high[1])]

def outside_crop_frustum(corners,camera,name,size):
    from driversi.rendering.gaussian.worldengine_geometry import box_frustum_support
    if name not in ('CAM_L0','CAM_F0','CAM_R0'):
        raise ValueError('Unknown native camera')
    bounds=_rays(tuple(np.asarray(camera['cam_intrinsic'],float).ravel()),tuple(np.asarray(camera['distortion'],float).ravel()),tuple(size),name)
    pose=np.column_stack([camera['sensor2lidar_rotation'],camera['sensor2lidar_translation']])
    return box_frustum_support(corners,pose,bounds)['outside_frustum']
