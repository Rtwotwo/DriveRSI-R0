"""Structured NAVSIM/physical-world conditions; no inferred sensor measurements."""

from __future__ import annotations

import bisect
import copy
import json
from pathlib import Path

import numpy as np

from driversi.data.catalog.store import identity
from driversi.data.sources.navsim import load_log
from driversi.rendering.whitebox.exporter import sha256

CAMERAS = ('CAM_F0', 'CAM_L0', 'CAM_R0', 'CAM_B0')
DIMENSION_SAMPLING = 'source_bracket_linear_v1'
PALETTE = {
    'ego': '#FFFFFF', 'vehicle': '#4785E8', 'pedestrian': '#F073AE',
    'bicycle': '#43C6B5', 'traffic_cone': '#F29B38', 'barrier': '#9C73D8',
    'construction_zone_sign': '#F2D45A', 'czone_sign': '#F2D45A',
    'generic_object': '#A2ADB8', 'other': '#A2ADB8',
}


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(',', ':')))


def rigid(matrix):
    value = np.asarray(matrix, dtype=float)
    if (value.shape != (4, 4) or not np.isfinite(value).all()
            or not np.allclose(value[3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(value[:3, :3].T @ value[:3, :3], np.eye(3), atol=2e-5)
            or not np.isclose(np.linalg.det(value[:3, :3]), 1, atol=2e-5)):
        raise ValueError('Finite proper rigid transform required')
    return value


def interpolate_transform(a, b, fraction):
    a, b = rigid(a), rigid(b)
    if not np.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError('Interpolation fraction must be in [0,1]')
    def quaternion(m):
        # Eigenvector construction is stable at rotations close to 180 degrees.
        x, y, z = m[:, 0], m[:, 1], m[:, 2]
        k = np.array([[x[0]-y[1]-z[2], x[1]+y[0], x[2]+z[0], y[2]-z[1]],
                      [x[1]+y[0], y[1]-x[0]-z[2], y[2]+z[1], z[0]-x[2]],
                      [x[2]+z[0], y[2]+z[1], z[2]-x[0]-y[1], x[1]-y[0]],
                      [y[2]-z[1], z[0]-x[2], x[1]-y[0], x[0]+y[1]+z[2]]]) / 3
        return np.linalg.eigh(k)[1][:, -1]
    qa, qb = quaternion(a[:3, :3]), quaternion(b[:3, :3])
    dot = float(np.dot(qa, qb))
    if dot < 0:
        qb, dot = -qb, -dot
    if dot > .9995:
        q = qa * (1-fraction) + qb * fraction
    else:
        theta = np.arccos(np.clip(dot, -1., 1.))
        q = (qa*np.sin((1-fraction)*theta)+qb*np.sin(fraction*theta))/np.sin(theta)
    x, y, z, w = q / np.linalg.norm(q)
    result = np.eye(4)
    result[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                    [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                    [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    result[:3, 3] = a[:3, 3] * (1 - fraction) + b[:3, 3] * fraction
    return result


def bracket(times, stamp, maximum_gap_us=750000):
    if stamp < times[0] or stamp > times[-1]:
        raise ValueError('Source does not cover complete seven-second window; no extrapolation')
    right = bisect.bisect_left(times, stamp)
    if times[right] == stamp:
        return right, right, 0.
    left = right - 1
    gap = times[right] - times[left]
    if gap <= 0 or gap > maximum_gap_us:
        raise ValueError('Source sampling gap exceeds declared interpolation support')
    return left, right, (stamp - times[left]) / gap


def object_map(frame, origin):
    ann = frame['anns']
    result = {}
    for token, kind, box in zip(ann['track_tokens'], ann['gt_names'], ann['gt_boxes'], strict=True):
        token, kind, box = str(token), str(kind), np.asarray(box, float)
        if token in result or token == 'ego' or not token or box.shape != (7,) or not np.isfinite(box).all() or np.any(box[3:6] <= 0):
            raise ValueError('Invalid source identity or object box')
        # OpenScene producer inverse-transforms centers with the full ego rotation,
        # but subtracts a scalar quaternion yaw from headings. Undo these separately.
        transform = rigid(frame['ego2global'])
        w, x, y, z = np.asarray(frame['ego2global_rotation'], float) / np.linalg.norm(frame['ego2global_rotation'])
        source_yaw = np.arctan2(2*(w*z-x*y), 1-2*(y*y+z*z))
        center = transform[:3, :3] @ box[:3] + transform[:3, 3] - origin
        pose = np.r_[center, np.arctan2(np.sin(box[6]+source_yaw), np.cos(box[6]+source_yaw))]
        result[token] = dict(actor_id=token, kind=kind, pose_xyz_yaw=pose.tolist(), dimensions_lwh=box[3:6].tolist())
    return result


def resample_objects(before, after, fraction, seconds, exact=False):
    """Offline bracket interpolation, including shape; no early birth or gap bridge.

    Source box dimensions are noisy measurements, not changing physical sizes.
    Sampling shape on the same clock as pose avoids artificial 2 Hz silhouette
    steps. This does not stabilize annotations or resample executed physics.
    """
    if not np.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError('Interpolation fraction must be in [0,1]')
    if not exact and (not np.isfinite(seconds) or seconds <= 0):
        raise ValueError('Positive interpolation interval required')
    result = []
    for token, a in before.items():
        row = copy.deepcopy(a)
        if exact:
            row.update(sample_basis='observed_source', velocity_xyz_mps=None)
        elif token in after:
            b = after[token]
            if a['kind'] != b['kind']:
                raise ValueError('Tracked semantic class changed')
            pa, pb = np.asarray(a['pose_xyz_yaw']), np.asarray(b['pose_xyz_yaw'])
            pose = pa + fraction * (pb - pa)
            delta = np.arctan2(np.sin(pb[3] - pa[3]), np.cos(pb[3] - pa[3]))
            pose[3] = np.arctan2(np.sin(pa[3] + fraction * delta), np.cos(pa[3] + fraction * delta))
            size_a, size_b = np.asarray(a['dimensions_lwh'], float), np.asarray(b['dimensions_lwh'], float)
            if (size_a.shape != (3,) or size_b.shape != (3,)
                    or not np.isfinite([size_a, size_b]).all()
                    or np.any(size_a <= 0) or np.any(size_b <= 0)):
                raise ValueError('Finite positive source box dimensions required')
            row.update(pose_xyz_yaw=pose.tolist(), sample_basis='bracket_interpolation',
                       dimensions_lwh=(size_a + fraction * (size_b-size_a)).tolist(),
                       velocity_xyz_mps=((pb[:3] - pa[:3]) / seconds).tolist())
        else:
            row.update(sample_basis='previous_observation_held_presence_uncertain', velocity_xyz_mps=None)
        row['velocity_basis'] = 'annotation_finite_difference_not_behavior_or_measured_velocity'
        row['dimension_basis'] = row['sample_basis']
        row['subtype'] = 'unknown'
        row['semantic_color'] = PALETTE.get(row['kind'], PALETTE['other'])
        result.append(row)
    return result


def frame_camera(raw, cam, pose, width=640, height=360):
    c = raw['cams'][cam]
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = c['sensor2lidar_rotation']
    extrinsic[:3, 3] = c['sensor2lidar_translation']
    # OpenScene retains legacy field names for camera-to-ego extrinsics.
    # lidar2ego must not be applied a second time.
    optical = pose @ rigid(extrinsic)
    k = np.asarray(c['cam_intrinsic'], float).copy()
    # NAVSIM original camera geometry; dimensions verified by the exporter.
    k[0] *= width / 1920
    k[1] *= height / 1080
    return dict(camera_id=cam, owner_actor_id='ego', width=width, height=height,
                camera_mount_to_ego=(rigid(extrinsic) @ np.diag([1., -1., -1., 1.])).tolist(),
                camera_to_world=(optical @ np.diag([1., -1., -1., 1.]))[:3].tolist(),
                K=k.tolist(), distortion=np.asarray(c['distortion'], float).tolist(),
                coordinate_convention='local map; camera right-up-back; native OpenCV distorted lens',
                source_calibration_token=raw['token'], source_image_relative_path=c['data_path'])


def build_source(record, *, frames=None, width=640, height=360):
    if record['kind'] != 'original_navtrain':
        raise ValueError('Conditioning synthesis requires training-owned original navtrain')
    frames = load_log(Path(record['log_path']), trusted=True) if frames is None else frames
    anchor = record['anchor_index']
    if frames[anchor]['token'] != record['token']:
        raise ValueError('Catalog anchor identity changed')
    stamp = int(frames[anchor]['timestamp'])
    start = stamp - 2000000
    times = [int(f['timestamp']) for f in frames]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError('Source clock is nonmonotonic')
    brackets = [bracket(times, start + i * 100000) for i in range(71)]
    used = sorted({j for a, b, _ in brackets for j in (a, b)})
    if any(frames[j]['log_name'] != record['source_log'] or frames[j]['map_location'] != frames[anchor]['map_location'] for j in used):
        raise ValueError('Source log or map changed')
    origin = np.asarray(frames[anchor]['ego2global_translation'], float)
    poses, objects = {}, {}
    for j in used:
        poses[j] = rigid(frames[j]['ego2global']).copy()
        poses[j][:3, 3] -= origin
        objects[j] = object_map(frames[j], origin)
    rows = []
    for i, (a, b, alpha) in enumerate(brackets):
        pose = poses[a] if a == b else interpolate_transform(poses[a], poses[b], alpha)
        os = resample_objects(objects[a], objects[b], alpha, (times[b] - times[a]) / 1e6, exact=a == b)
        cameras = [frame_camera(frames[a], c, pose, width, height) for c in CAMERAS]
        rows.append(dict(tick=i, time_seconds=i / 10, relative_to_anchor_seconds=i / 10 - 2,
                         timestamp_us=start + i * 100000, ego_rear_axle_to_local=pose.tolist(),
                         source_bracket=dict(left_timestamp_us=times[a], right_timestamp_us=times[b],
                                             left_token=frames[a]['token'], right_token=frames[b]['token'], fraction=alpha),
                         sampling='observed_source' if a == b else 'offline_bracket_interpolation_not_sensor_observation',
                         objects=os, cameras=cameras,
                         navigation_command=np.asarray(frames[a].get('driving_command',[])).tolist(),
                         source_roadblock_ids=[str(x) for x in frames[a].get('roadblock_ids',[])],
                         signals=dict(records=[[str(c), bool(red)] for c, red in frames[a].get('traffic_lights', [])],
                                      timestamp_us=times[a], semantics='connector_red_boolean; false is not proven green',
                                      sampling='previous_source_observation')))
    # Chronological allocation: future arrivals never renumber past instances.
    ids = list(dict.fromkeys(o['actor_id'] for row in rows for o in row['objects']))
    id_map = {key: i + 1 for i, key in enumerate(ids)}
    for row in rows:
        for obj in row['objects']:
            obj['instance_id'] = id_map[obj['actor_id']]
    actors = {key: dict(instance_id=num, kind=next(o['kind'] for f in rows for o in f['objects'] if o['actor_id'] == key)) for key, num in id_map.items()}
    return dict(schema_version=1, kind='semantic_whitebox_conditioning', role='source_log',
                geometry_convention='openscene_center_se3_scalar_yaw_v2',
                camera_convention='openscene_camera_to_ego_v2',
                dimension_sampling=DIMENSION_SAMPLING,
                display_profile='planar_consistent_v1',
                token=record['token'], source_log=record['source_log'], source_split='navtrain',
                map_name=frames[anchor]['map_location'], origin_xyz=origin.tolist(),
                source_log_file=record['log_path'], source_log_sha256=sha256(record['log_path']),
                anchor_timestamp_us=stamp, anchor_tick=20, history_ticks=[5, 10, 15, 20], future_ticks=list(range(25, 71, 5)),
                timing=dict(interval_seconds=.1, duration_seconds=7., frame_count=71, endpoints_included=True,
                            observations='interpolated source is offline video condition, not causal online sensor history',
                            original_intervals_seconds=[(times[j+1]-times[j])/1e6 for j in used[:-1]]),
                actors=actors, palette=palette(), frames=rows,
                limitations=['Planar map ground is a display proxy, not measured road elevation.',
                             'Boxes do not encode body articulation, meshes or true visible-pixel masks.',
                             'Entry/exit is annotation visibility, not physical birth/death.',
                             'Intermediate box dimensions are bracket-interpolated annotations, not measured size changes.',
                             'Vehicle subtype is unknown unless separately reviewed from source imagery.'],
                trainable=False, data_use='offline_generation_condition_not_training_admission')


def palette():
    return dict(version='semantic_classes_v1', encoding='class color, not physical paint color',
                colors=PALETTE, instance_identity='stable integer instance_id and source actor_id, not color',
                vehicle_subtypes='vehicle means unspecified vehicle; do not convert all blue boxes into same make/body type',
                map_semantics='baseline_paths are navigation centerlines, not observed lane paint',
                unknown='unknown is retained; no synthetic appearance labels implied')


def branch_from_source(source, scene, rollout):
    """New trajectories must retain executed state; never interpolate the actor pool."""
    from driversi.rendering.whitebox.exporter import interaction_frames
    from driversi.rendering.bev.replay import _load_run, _scene_key
    rollout=Path(rollout)
    run = _load_run(scene, rollout, _scene_key(scene))
    state_rows = interaction_frames(scene, run)
    if len(state_rows) < 71 or abs(scene.metadata['source_start_timestamp_us'] - (source['anchor_timestamp_us'] - 2000000)) > 1000:
        raise ValueError('Branch and source must cover identical physical seven-second clock')
    result = copy.deepcopy(source)
    result['role'] = 'executed_branch'
    result['source_geometry_convention'] = source.get('geometry_convention', 'legacy_source')
    result['geometry_convention'] = 'executed_physics_state_not_source_se3_audit'
    result['actor_state_basis'] = 'recorded_planar_physics_unmodified'
    result['dimension_sampling'] = 'executed_physics_unmodified'
    result['limitations'] = [text for text in result['limitations'] if text !=
        'Intermediate box dimensions are bracket-interpolated annotations, not measured size changes.']
    result['limitations'].append('Executed box dimensions are copied from world states without temporal interpolation.')
    result['branch_sha256'] = sha256(rollout)
    result['clock_alignment_delta_us'] = int(scene.metadata['source_start_timestamp_us']-(source['anchor_timestamp_us']-2000000))
    result['source_anchor_timestamp_us'] = source['anchor_timestamp_us']
    result['anchor_timestamp_us'] = int(scene.metadata['source_start_timestamp_us'])+2000000
    result['frames'] = []
    origin_delta = np.asarray(scene.metadata['origin_xyz']) - np.asarray(source['origin_xyz'])
    extra = sorted(set(scene.actor_ids) - set(source['actors']) - {'ego'})
    for actor in extra:
        result['actors'][actor] = dict(instance_id=len(result['actors']) + 1, kind=scene.metadata.get('source_actor_kinds', {}).get(actor, 'other'))
    for i, row in enumerate(state_rows[:71]):
        if abs(row['time_seconds'] - i / 10) > 1e-6:
            raise ValueError('Branch is not measured at0.1s simulation steps')
        f = copy.deepcopy(source['frames'][i])
        f['source_reference_timestamp_us'] = f['timestamp_us']
        f['timestamp_us'] = int(scene.metadata['source_start_timestamp_us'])+i*100000
        f['sampling'] = 'executed_physics_state'
        f['objects'] = []
        ego = next(o for o in row['objects'] if o['actor_id'] == 'ego')
        yaw = ego['pose_xyz_yaw'][3]
        rear = np.asarray(ego['pose_xyz_yaw'][:3]) + origin_delta
        rear[:2] -= 1.461 * np.array([np.cos(yaw), np.sin(yaw)])
        measured_source_pose = np.asarray(source['frames'][i]['ego_rear_axle_to_local'])
        # Physics is planar; transport source calibration rigidly in XY/yaw only.
        c, s = np.cos(yaw), np.sin(yaw)
        new_pose = np.array([[c,-s,0,rear[0]],[s,c,0,rear[1]],[0,0,1,measured_source_pose[2,3]],[0,0,0,1.]])
        source_yaw = np.arctan2(measured_source_pose[1,0], measured_source_pose[0,0])
        d = yaw-source_yaw
        transform = np.eye(4);transform[:2,:2] = [[np.cos(d),-np.sin(d)],[np.sin(d),np.cos(d)]]
        transform[:3,3] = new_pose[:3,3] - transform[:3,:3] @ measured_source_pose[:3,3]
        for cam in f['cameras']:
            pose = np.eye(4);pose[:3] = cam['camera_to_world']
            cam['camera_to_world'] = (transform @ pose)[:3].tolist()
            cam['pose_basis'] = 'source_calibration_planar_executed_ego_transport_height_unverified'
        f['ego_rear_axle_to_local'] = new_pose.tolist()
        f['lifecycle'] = row['lifecycle']
        f['signals'] = row['signals']
        for obj in row['objects']:
            if obj['actor_id'] == 'ego':
                continue
            o = copy.deepcopy(obj);o['pose_xyz_yaw'][:3] = (np.asarray(o['pose_xyz_yaw'][:3]) + origin_delta).tolist()
            o['kind'] = result['actors'][o['actor_id']]['kind']
            o.update(instance_id=result['actors'][o['actor_id']]['instance_id'], semantic_color=PALETTE.get(o['kind'], PALETTE['other']),
                     sample_basis='executed_physics', subtype='unknown', velocity_xyz_mps=None, velocity_basis='not_exported')
            o['dimension_basis'] = 'executed_physics'
            f['objects'].append(o)
        result['frames'].append(f)
    result['limitations'].append('Branch source camera height is a proxy; no GS terrain certificate asserted.')
    return result


def describe(bundle, scope='retrieval_history'):
    if scope not in ('retrieval_history', 'video_clip'):
        raise ValueError('Explicit description scope required')
    rows = bundle['frames'][:21] if scope == 'retrieval_history' else bundle['frames']
    anchor = rows[-1] if scope == 'retrieval_history' else rows[bundle['anchor_tick']]
    pose = np.asarray(anchor['ego_rear_axle_to_local'])
    inverse = np.linalg.inv(pose)
    observations = []
    for obj in anchor['objects']:
        xyz = (inverse @ np.r_[obj['pose_xyz_yaw'][:3], 1])[:3]
        velocity=obj.get('velocity_xyz_mps')
        observations.append(dict(instance_id=obj['instance_id'], semantic_class=obj['kind'],
                                 relative_ego_xyz_m=np.round(xyz, 2).tolist(), dimensions_lwh=np.round(obj['dimensions_lwh'], 2).tolist(),
                                 velocity_in_anchor_ego_axes_mps=np.round(inverse[:3,:3]@velocity,2).tolist() if velocity is not None else None,
                                 velocity_semantics='finite_difference_in_rotated_axes_not_relative_velocity_or_intent',
                                 subtype='unknown', sampling=obj['sample_basis']))
    return dict(scope=scope, source='structured measurements/interpolation; not VLM perception',
                duration_seconds=rows[-1]['time_seconds']-rows[0]['time_seconds'],
                anchor_objects=sorted(observations,key=lambda o:sum(v*v for v in o['relative_ego_xyz_m'][:2])),
                class_counts={kind:sum(o['kind']==kind for o in anchor['objects']) for kind in sorted({o['kind'] for o in anchor['objects']})},
                nearby_class_counts_30m={kind:int(sum(o['semantic_class']==kind and np.linalg.norm(o['relative_ego_xyz_m'][:2])<=30 for o in observations)) for kind in sorted({o['kind'] for o in anchor['objects']})},
                navigation_command_native=anchor.get('navigation_command',[]),
                signals=anchor.get('signals',{}),
                ego_displacement_m=round(float(np.linalg.norm(np.asarray(rows[-1]['ego_rear_axle_to_local'])[:2,3]-np.asarray(rows[0]['ego_rear_axle_to_local'])[:2,3])),3),
                actor_presence=[dict(instance_id=num['instance_id'], semantic_class=num['kind'],
                                     present_sample_times_seconds=[f['time_seconds'] for f in rows if any(o['actor_id']==actor for o in f['objects'])],
                                     sample_bases=sorted({o['sample_basis'] for f in rows for o in f['objects'] if o['actor_id']==actor}))
                                for actor,num in bundle['actors'].items() if any(any(o['actor_id']==actor for o in f['objects']) for f in rows)],
                future_included=scope=='video_clip', class_colors=palette(),
                map_layer_counts={k:v['feature_count'] for k,v in bundle.get('map_layers',{}).items() if v['feature_count']},
                uncertainty=bundle['limitations'])


def export_bundle(record, maps_root, output, *, frames=None, scene=None, rollout=None):
    from driversi.data.scenes.context import read_map_context
    from PIL import Image
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    data = build_source(record, frames=frames)
    for cam in CAMERAS:
        with Image.open(Path(record['sensor_root']) / next(c for c in data['frames'][20]['cameras'] if c['camera_id']==cam)['source_image_relative_path']) as im:
            if im.size != (1920,1080):
                raise ValueError('Source camera dimensions differ from declared calibration')
    # Static map footprint independent of the future trajectory.
    bounds=np.array([-250.,-250.,250.,250.])
    if any(np.max(np.abs(np.asarray(f['ego_rear_axle_to_local'])[:2,3])) > 200 for f in data['frames']):
        raise ValueError('Seven-second path leaves fixed map support')
    features,layers,provenance=read_map_context(data['map_name'],Path(maps_root),data['origin_xyz'],bounds)
    data['map_layers']=layers
    data['map_reference']='map.json'
    data['map_bounds_local']=bounds.tolist()
    output.mkdir(parents=True)
    write_json(output/'source.json',data)
    if scene is not None:
        branch=branch_from_source(data,scene,rollout)
        write_json(output/'branch.json',branch)
        write_json(output/'branch-retrieval-facts.json',describe(branch,'retrieval_history'))
        write_json(output/'branch-video-facts.json',describe(branch,'video_clip'))
    write_json(output/'map.json',dict(features=features,layers=layers,provenance=provenance,
                                    elevation='flattened local z=0 proxy; not measured surface'))
    write_json(output/'palette.json',palette())
    write_json(output/'retrieval-facts.json',describe(data,'retrieval_history'))
    write_json(output/'video-facts.json',describe(data,'video_clip'))
    files={p.name:sha256(p) for p in output.iterdir() if p.is_file()}
    write_json(output/'manifest.json',dict(kind='semantic_whitebox_conditions',files=files,
                                          identity=identity(files),producer_sha256=sha256(__file__),
                                          trainable=False,frame_count=71,step_seconds=.1))
    return data
