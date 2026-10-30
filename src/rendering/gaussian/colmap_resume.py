"""Resume matched legacy captures after COLMAP4 trivial-rig ID mismatch.

The camera calibration, image IDs, measured poses and matches are unchanged.
Only the independently assigned trivial-rig IDs are made consistent.
"""
import argparse
import sqlite3
from pathlib import Path

from driversi.agents.backends.native import digest
from driversi.configuration import read_mapping
from driversi.rsi.experiments.continuation import frozen_json
from driversi.runtime.budget import write_json


def synchronize(database, reconstruction):
    import numpy as np
    images = {i.image_id: i for i in database.read_all_images()}
    cameras = {c.camera_id: c for c in database.read_all_cameras()}
    if set(images) != set(reconstruction.images) or set(cameras) != set(reconstruction.cameras):
        raise ValueError('Image and camera membership must be identical')
    if database.num_frames() != 0:
        raise ValueError('Only the legacy database without explicit frames is supported')
    for key, image in reconstruction.images.items():
        if (image.name, image.camera_id) != (images[key].name, images[key].camera_id):
            raise ValueError('Measured image identity changed')
    for key, camera in reconstruction.cameras.items():
        other = cameras[key]
        if (camera.model, camera.width, camera.height) != (other.model, other.width, other.height):
            raise ValueError('Camera model changed')
        if not np.array_equal(camera.params, other.params):
            raise ValueError('Camera intrinsics changed')
    before = database.read_all_rigs()
    after = list(reconstruction.rigs.values())
    if any(r.num_sensors() != 1 for r in [*before, *after]):
        raise ValueError('Nontrivial rigs require a separate calibrated migration')
    if {r.ref_sensor_id for r in before} != {r.ref_sensor_id for r in after}:
        raise ValueError('Rig sensor membership differs')
    database.clear_rigs()
    for rig in after:
        database.write_rig(rig, use_rig_id=True)
    return dict(before=[str(r) for r in before], after=[str(r) for r in after],
                cameras=len(cameras), images=len(images), geometry_modified=False)


def run(payload):
    import pycolmap
    from driversi.rendering.gaussian.mtgs_reconstruction import (
        activate, configure_runtime, stage_bindings)
    configure_runtime()
    job = read_mapping(payload['native_job'])
    root = Path(job['output']); out = Path(payload['output'])
    out.mkdir(parents=True, exist_ok=True)
    if (root/'running.json').exists():
        raise ValueError('Settle the previous stage before recovery')
    if (out/'report.json').exists():
        return read_mapping(out/'report.json')
    cfg, scene = activate(job)
    from nuplan_scripts.ba_multi_traversal import align_model
    data = scene.load_pickle(scene.pickle_path_registered)
    result, proofs = {}, []
    for video_id, video in data.items():
        folder = Path(scene.colmap_path)/video_id
        database_path = folder/'database.db'
        backup = out/(video_id+'-failed.db')
        if not backup.exists():
            with sqlite3.connect(str(database_path)) as src, sqlite3.connect(str(backup)) as dst:
                src.backup(dst)
        before_hash = digest(backup)
        reconstruction = pycolmap.Reconstruction(str(folder/'sparse_model'))
        db = pycolmap.Database.open(database_path)
        try:
            proof = synchronize(db, reconstruction)
        finally:
            db.close()
        frozen_json(out/(video_id+'-rigs.json'), dict(proof, backup=str(backup),
            backup_sha256=before_hash, input_job_sha256=digest(payload['native_job'])))
        options = pycolmap.IncrementalPipelineOptions()
        options.num_threads = job.get('cpu_workers', 8)
        options.ba_global_function_tolerance = .000001
        options.ba_global_max_num_iterations = 30
        options.ba_global_max_refinements = 3
        dst = folder/'sfm_model/0'; dst.mkdir(parents=True, exist_ok=True)
        pycolmap.triangulate_points(reconstruction, str(database_path),
            str(folder/'images'), str(dst), clear_points=False,
            options=options, refine_intrinsics=True)
        result.update(align_model(scene, {video_id: video}, str(folder)))
        proofs.append(str(out/(video_id+'-rigs.json')))
    scene.video_scene_dict = result
    scene.dump_pickle(scene.pickle_path_colmap)
    scene.update_pickle_link(scene.pickle_path_colmap)
    receipt = dict(stage='triangulation', status='completed', inputs=stage_bindings(job,'triangulation'),
        outputs={str(scene.pickle_path_colmap):digest(scene.pickle_path_colmap)}, trainable=False,
        recovery_producer_sha256=digest(__file__), recovery_proofs={p:digest(p) for p in proofs},
        matches_reused=True, calibrated_geometry_replaced=False)
    frozen_json(root/'receipts/triangulation.json', receipt)
    write_json(out/'report.json', dict(status='completed', receipt=str(root/'receipts/triangulation.json'),
                                     trainable=False))
    return receipt


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--job',required=True)
    run(read_mapping(p.parse_args().job))
