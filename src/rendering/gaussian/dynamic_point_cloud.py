"""Measured foreground initialization for pinned MTGS, without invented geometry.

The upstream preparation uses only the top LiDAR. This adapter uses all recorded
LiDAR returns, upstream box transforms and calibrated real-image RGB projection.
Background, masks and calibration remain the previously prepared native inputs.
"""
from collections import defaultdict
from pathlib import Path


def build_instance_clouds(job, scene):
    import cv2
    import numpy as np
    import open3d as o3d
    from pyquaternion import Quaternion
    from nuplan_scripts.utils.camera_utils import undistort_image_with_cam_info
    from nuplan_scripts.utils.nuplan_utils_custom import load_lidar, get_rgb_point_cloud, adjust_brightness
    from nuplan_scripts.utils.stack_point_cloud_utils import extract_frame_background_instance_lidar
    from driversi.runtime.budget import write_json

    accumulated = defaultdict(lambda: [[], []])
    classes, frames_seen, frame_records = {}, defaultdict(list), []
    data = scene.load_pickle(scene.pickle_path)
    for video_token, video in data.items():
        for info in video['frame_infos']:
            if info.get('skipped') == 'low_velocity':
                continue
            points = load_lidar(str(Path(job['sensor_root']) / info['lidar_path']),
                                remove_close=False, only_top=False)
            objects = extract_frame_background_instance_lidar(info, l2g=False, points=points)['instance']
            eligible = [(token, obj) for token, obj in objects.items() if len(obj.points)]
            if not eligible:
                continue
            lengths = [len(obj.points) for _, obj in eligible]
            world_points = np.concatenate([obj.points_ego for _, obj in eligible])
            rgb_sum = np.zeros((len(world_points), 3))
            votes = np.zeros(len(world_points), dtype=np.int32)
            for cam in info['cams'].values():
                image = cv2.imread(str(Path(job['sensor_root']) / cam['data_path']))
                if image is None:
                    raise ValueError('Missing measured source camera')
                image = undistort_image_with_cam_info(image, cam, interpolation='linear', mode='keep_focal_length')
                if 'v_adjust' in cam:
                    image = adjust_brightness(image, cam['v_adjust'])
                ego2cam = np.eye(4)
                ego2cam[:3, :3] = Quaternion(cam['sensor2ego_rotation']).rotation_matrix.T
                ego2cam[:3, 3] = -ego2cam[:3, :3] @ cam['sensor2ego_translation']
                intr = cam.get('colmap_param', {}).get('cam_intrinsic', cam['cam_intrinsic'])
                view = np.eye(4)
                view[:3, :3] = intr
                projection = view @ ego2cam @ info['lidar2ego']
                rgb, visible = get_rgb_point_cloud(world_points, projection[None], image[None])
                rgb_sum[visible] += rgb[visible]
                votes[visible] += 1
            offset = 0
            for (token, obj), length in zip(eligible, lengths):
                key = (video_token, token)
                span = slice(offset, offset + length)
                valid = votes[span] > 0
                color = rgb_sum[span][valid] / votes[span][valid, None]
                accumulated[key][0].append(obj.points[valid])
                accumulated[key][1].append(color)
                classes[key] = obj.class_name
                frames_seen[key].append(int(info['frame_idx']))
                offset += length
            frame_records.append({'video_token': video_token, 'frame_idx': int(info['frame_idx']),
                                  'measured_lidar_returns': len(points), 'colored_object_returns': int((votes > 0).sum())})
            write_json(Path(job['output']) / 'point-cloud-progress.json',
                       {'completed_frames': len(frame_records), 'last_frame': frame_records[-1]})
    rows = []
    for (video_token, token), (point_parts, color_parts) in sorted(accumulated.items()):
        points, colors = np.concatenate(point_parts), np.concatenate(color_parts)
        # Exact repeated returns cannot manufacture the native 100-point support.
        if len(points):
            _, indices = np.unique(np.round(points, 4), axis=0, return_index=True)
            points, colors = points[indices], colors[indices]
        path = Path(scene.instance_point_cloud_path) / video_token / f'{token}.pcd'
        path.parent.mkdir(parents=True, exist_ok=True)
        if len(points):
            cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
            cloud.colors = o3d.utility.Vector3dVector(colors)
            if not o3d.io.write_point_cloud(str(path), cloud):
                raise IOError('Native point cloud writer failed')
        rows.append({'video_token': video_token, 'token': token, 'class_name': classes[(video_token, token)],
                     'measured_colored_points': len(points), 'frame_indices': frames_seen[(video_token, token)],
                     'path': str(path) if len(points) else None})
    return {'schema_version': 1, 'method': 'all_recorded_lidar_calibrated_real_rgb_v1',
            'synthetic_points': 0, 'native_minimum_points': 100, 'objects': rows, 'frames': frame_records}
