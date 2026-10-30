"""Audit whitebox source motion against raw NAVSIM annotations, independently of resampling."""
import bisect
import math
from pathlib import Path

import numpy as np

from driversi.data.sources.navsim import load_log


def _angle(x):
    return math.atan2(math.sin(x), math.cos(x))


def _world_objects(frame, origin, full_centers=False):
    w,x,y,z = frame['ego2global_rotation']
    yaw = math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
    c,s = math.cos(yaw),math.sin(yaw)
    rotation = np.array([[c,-s,0],[s,c,0],[0,0,1]])
    if full_centers:
        q = np.asarray([w,x,y,z], float);w,x,y,z = q/np.linalg.norm(q)
        rotation = np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                             [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                             [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
        yaw = math.atan2(2*(w*z-x*y),1-2*(y*y+z*z))
    offset = np.asarray(frame['ego2global_translation'])-origin
    ann = frame['anns']
    result = {}
    for token,kind,box in zip(ann['track_tokens'],ann['gt_names'],ann['gt_boxes'],strict=True):
        if str(token) in result:
            raise ValueError('Duplicate raw identity')
        result[str(token)] = {'xyz':rotation@np.asarray(box[:3])+offset,
                              'yaw':float(box[6])+yaw,'dimensions':np.asarray(box[3:6]),'kind':str(kind)}
    return result


def _rotation_between(a,b,t):
    relative = a.T@b
    theta = math.acos(float(np.clip((np.trace(relative)-1)/2,-1,1)))
    if theta < 1e-9:
        return a
    if abs(math.pi-theta) < 1e-6:
        raise ValueError('Near-pi rotation requires separate audit')
    axis = np.array([relative[2,1]-relative[1,2],relative[0,2]-relative[2,0],relative[1,0]-relative[0,1]])/(2*math.sin(theta))
    x,y,z = axis
    skew = np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    return a@(np.eye(3)+math.sin(t*theta)*skew+(1-math.cos(t*theta))*(skew@skew))


def audit_source_motion(bundle, raw_frames=None):
    if bundle['role'] != 'source_log':
        raise ValueError('Original-log audit cannot certify a physical branch')
    raw = load_log(Path(bundle['source_log_file']),trusted=True) if raw_frames is None else raw_frames
    timestamps = [int(f['timestamp']) for f in raw]
    if any(b<=a for a,b in zip(timestamps,timestamps[1:])):
        raise ValueError('Nonmonotonic raw clock')
    origin = np.asarray(bundle['origin_xyz'])
    convention = bundle.get('geometry_convention', 'legacy_yaw_only_v1')
    if convention not in ('legacy_yaw_only_v1', 'openscene_center_se3_scalar_yaw_v2'):
        raise ValueError('Unknown source geometry convention')
    full_centers = convention == 'openscene_center_se3_scalar_yaw_v2'
    dimension_sampling = bundle.get('dimension_sampling', 'previous_annotation_hold_v1')
    if dimension_sampling not in ('previous_annotation_hold_v1', 'source_bracket_linear_v1'):
        raise ValueError('Unknown source dimension sampling')
    camera_convention = bundle.get('camera_convention', 'legacy_lidar_composition_v1')
    if camera_convention not in ('legacy_lidar_composition_v1', 'openscene_camera_to_ego_v2'):
        raise ValueError('Unknown camera convention')
    issues=[]; maxima={'object_position_m':0.,'object_heading_rad':0.,'ego_pose':0.,'camera_pose':0.,'camera_intrinsics':0.,'camera_distortion':0.,'dimension_m':0.}
    compared=0; observed=0; held=0; steps=0; previous={}; identity={}; object_cache={}
    def check(name,error,tick,actor=None):
        if not math.isfinite(float(error)):
            issues.append({'tick':tick,'actor':actor,'field':name,'error':'nonfinite'})
            return
        maxima[name]=max(maxima[name],float(error))
        if error>1e-6:
            issues.append({'tick':tick,'actor':actor,'field':name,'error':float(error)})
    for i,frame in enumerate(bundle['frames']):
        stamp=bundle['anchor_timestamp_us']-2000000+i*100000
        if frame['tick']!=i or frame['timestamp_us']!=stamp or abs(frame['time_seconds']-i/10)>1e-8:
            issues.append({'tick':i,'field':'clock'})
        right=bisect.bisect_left(timestamps,stamp)
        if right>=len(raw) or stamp<timestamps[0]:raise ValueError('Window exceeds raw support')
        left=right if timestamps[right]==stamp else right-1
        alpha=0. if left==right else (stamp-timestamps[left])/(timestamps[right]-timestamps[left])
        if left!=right and timestamps[right]-timestamps[left]>750000:raise ValueError('Uncovered source gap')
        for j in (left,right):
            if j not in object_cache:object_cache[j]=_world_objects(raw[j],origin,full_centers)
        before,after=object_cache[left],object_cache[right]
        actual={o['actor_id']:o for o in frame['objects']}
        if len(actual)!=len(frame['objects']) or set(actual)!=set(before):
            issues.append({'tick':i,'field':'presence_or_duplicate_identity'})
        if len({o['instance_id'] for o in frame['objects']})!=len(frame['objects']):
            issues.append({'tick':i,'field':'instance_id_collision'})
        ego_a=np.asarray(raw[left]['ego2global'],float).copy();ego_b=np.asarray(raw[right]['ego2global'],float).copy()
        ego_a[:3,3]-=origin;ego_b[:3,3]-=origin
        expected_ego=np.eye(4);expected_ego[:3,:3]=_rotation_between(ego_a[:3,:3],ego_b[:3,:3],alpha)
        expected_ego[:3,3]=ego_a[:3,3]*(1-alpha)+ego_b[:3,3]*alpha
        check('ego_pose',np.max(np.abs(np.asarray(frame['ego_rear_axle_to_local'])-expected_ego)),i)
        for token,obj in actual.items():
            if token not in before:continue
            a=before[token];b=after.get(token,a);xyz=a['xyz']+(b['xyz']-a['xyz'])*alpha
            yaw=a['yaw']+_angle(b['yaw']-a['yaw'])*alpha
            check('object_position_m',np.linalg.norm(np.asarray(obj['pose_xyz_yaw'][:3])-xyz),i,token)
            check('object_heading_rad',abs(_angle(obj['pose_xyz_yaw'][3]-yaw)),i,token)
            expected_dimensions = a['dimensions']
            if dimension_sampling == 'source_bracket_linear_v1':
                expected_dimensions = (1-alpha)*a['dimensions'] + alpha*b['dimensions']
            check('dimension_m',np.max(np.abs(np.asarray(obj['dimensions_lwh'])-expected_dimensions)),i,token)
            basis='observed_source' if left==right else ('bracket_interpolation' if token in after else 'previous_observation_held_presence_uncertain')
            if dimension_sampling == 'source_bracket_linear_v1' and obj.get('dimension_basis') != basis:
                issues.append({'tick':i,'actor':token,'field':'dimension_basis'})
            if obj['kind']!=a['kind'] or obj['sample_basis']!=basis:issues.append({'tick':i,'actor':token,'field':'class_or_basis'})
            if token in identity and identity[token]!=obj['instance_id']:issues.append({'tick':i,'actor':token,'field':'instance_changed'})
            if token not in bundle['actors'] or bundle['actors'][token]['instance_id']!=obj['instance_id']:
                issues.append({'tick':i,'actor':token,'field':'instance_metadata_mismatch'})
            identity[token]=obj['instance_id']; compared+=1;observed+=basis=='observed_source';held+='held' in basis
            if token in previous:steps+=1
        previous=actual
        if len(frame['cameras'])!=4 or {c['camera_id'] for c in frame['cameras']}!={'CAM_F0','CAM_L0','CAM_R0','CAM_B0'}:
            issues.append({'tick':i,'field':'camera_completeness'})
        for camera in frame['cameras']:
            raw_camera=raw[left]['cams'][camera['camera_id']]
            extrinsic=np.eye(4);extrinsic[:3,:3]=raw_camera['sensor2lidar_rotation'];extrinsic[:3,3]=raw_camera['sensor2lidar_translation']
            transport = np.eye(4) if camera_convention == 'openscene_camera_to_ego_v2' else np.asarray(raw[left]['lidar2ego'])
            expected=(expected_ego@transport@extrinsic@np.diag([1,-1,-1,1]))[:3]
            check('camera_pose',np.max(np.abs(expected-np.asarray(camera['camera_to_world']))),i)
            if 'camera_mount_to_ego' in camera:
                expected_mount = transport @ extrinsic @ np.diag([1,-1,-1,1])
                mount = np.asarray(camera['camera_mount_to_ego'],float)
                if mount.shape != (4,4):
                    issues.append({'tick':i,'field':'camera_mount_shape'})
                else:
                    check('camera_pose',np.max(np.abs(expected_mount-mount)),i)

            intrinsic=np.asarray(raw_camera['cam_intrinsic'],float).copy()
            intrinsic[0]*=camera['width']/1920;intrinsic[1]*=camera['height']/1080
            check('camera_intrinsics',np.max(np.abs(intrinsic-np.asarray(camera['K']))),i)
            distortion=np.asarray(raw_camera['distortion'],float);actual_distortion=np.asarray(camera['distortion'],float)
            if distortion.shape!=actual_distortion.shape:issues.append({'tick':i,'field':'distortion_shape'})
            elif distortion.size:check('camera_distortion',np.max(np.abs(distortion-actual_distortion)),i)
    if len(bundle['frames'])!=71:issues.append({'field':'incomplete_window'})
    return {'kind':'raw_log_motion_audit_v1','passed':not issues,'token':bundle['token'],
            'geometry_convention':convention,'camera_convention':camera_convention,
            'dimension_sampling':dimension_sampling,
            'frames_checked':len(bundle['frames']),'actor_states_checked':compared,
            'consecutive_identity_pairs':steps,'observed_actor_states':observed,'uncertain_held_actor_states':held,
            'maximum_errors':maxima,'issues':issues,
            'limits':['Checks fidelity to source annotations and declared interpolation, not physical correctness of annotations.',
                      'Sub-0.5s true motion is unobserved; intermediate states are offline interpolation.',
                      'Missing annotation is not proof of physical disappearance.']}
