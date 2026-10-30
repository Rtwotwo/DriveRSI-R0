import copy
import numpy as np
from driversi.data.sources.navsim import compile_frames
from driversi.data.participants.fleet import configure_vehicle_fleet


def frames():
    return [dict(timestamp=1000000+i*500000, token=str(i), log_name='log', map_location='map',
        ego2global_translation=[i*2.,0,0],ego2global_rotation=[1,0,0,0],
        anns=dict(gt_boxes=np.array([[20-i*2.+i*.2,8,0,4,2,1.5,0.]]),
                  gt_names=['vehicle'],track_tokens=['parked'],gt_velocity_3d=np.zeros((1,3))))
        for i in range(5)]


def fleet(source):
    return configure_vehicle_fleet(compile_frames(source,start=0,duration=2,split='train',max_actors=1))


def test_zero_measured_velocity_prevents_drift_becoming_departure():
    scene=fleet(frames());role=scene.metadata['vehicle_fleet']['roles']['parked']
    assert role['mode']=='stationary_hold'
    assert role['stationary_evidence']['accepted']
    evidence=scene.metadata['source_motion_evidence']['actors']['parked']
    assert evidence['samples']==evidence['finite_velocity_samples']==5
    assert abs(evidence['max_position_excursion_m']-.8)<1e-8
    assert scene.actor_ids==('ego','parked')


def test_later_measured_departure_is_not_frozen():
    source=frames();source[-1]['anns']['gt_velocity_3d'][0,0]=1.
    scene=fleet(source)
    assert not scene.metadata['vehicle_fleet']['roles']['parked']['stationary_evidence']['accepted']
    assert scene.metadata['source_motion_evidence']['actors']['parked']['max_reported_speed_mps']==1.


def test_missing_velocity_does_not_certify_stationary():
    source=frames();del source[2]['anns']['gt_velocity_3d']
    scene=fleet(source)
    assert scene.metadata['source_motion_evidence']['actors']['parked']['finite_velocity_samples']==4
    assert not scene.metadata['vehicle_fleet']['roles']['parked']['stationary_evidence']['accepted']


def test_disappearance_ends_first_privileged_run():
    source=frames();source[2]['anns']=dict(gt_boxes=np.empty((0,7)),gt_names=[],track_tokens=[])
    source[3]['anns']['gt_velocity_3d'][0,0]=3.
    scene=fleet(source);e=scene.metadata['source_motion_evidence']['actors']['parked']
    assert e['samples']==2 and e['max_reported_speed_mps']==0
    assert not scene.metadata['reactive_background']['source_presence'][10][1]
