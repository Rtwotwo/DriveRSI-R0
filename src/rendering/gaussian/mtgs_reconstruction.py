"""Owned, receipt-based execution of pinned MTGS reconstruction stages.

Only selected navtrain captures are consumed. This produces a candidate asset;
source PSNR and subsequent same-branch admission remain separate operations.
"""
import argparse
import json
import os
import pickle
import runpy
import shutil
import subprocess
import sys
import time
from pathlib import Path

from driversi.agents.backends.native import digest, project_root
from driversi.configuration import read_mapping
from driversi.runtime.budget import write_json

STAGES = ('video', 'semantic', 'registration', 'triangulation', 'depth',
          'point_cloud', 'training', 'export', 'terrain')


def configure_runtime():
    import ninja
    cuda=Path(os.environ['CUDA_HOME']).resolve(strict=True)
    if not (cuda/'bin/nvcc').is_file():raise ValueError('Declared CUDA compiler missing')
    bins=[str(cuda/'bin'),str(Path(sys.executable).parent),str(ninja.BIN_DIR)]
    os.environ['PATH']=os.pathsep.join(bins)+os.pathsep+os.environ.get('PATH','')
    os.environ['LD_LIBRARY_PATH']=str(cuda/'lib64')+os.pathsep+os.environ.get('LD_LIBRARY_PATH','')
    os.environ['WANDB_MODE']='disabled'
    os.environ['WANDB_DISABLED']='true'
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TRANSFORMERS_OFFLINE']='1'


def effective_inputs(job):
    paths={Path(job[k]).resolve(strict=True) for k in ('config','sensors','database')}
    for row in read_mapping(job['sensors'])['rows']:
        paths.add(Path(row['source_path']).resolve(strict=True))
    for directory in job.get('model_roots',[]):
        paths.update(p.resolve() for p in Path(directory).rglob('*') if p.is_file())
    from driversi.vendor import path
    for package in ('mtgs','nuplan-devkit'):
        paths.update(p for p in path(package).rglob('*.py') if '__pycache__' not in p.parts)
    return {str(p):digest(p) for p in sorted(paths)}


def stage_bindings(job,stage):
    out=Path(job['output']);paths=[out/'inputs.json']
    paths.extend(out/'receipts'/f'{prior}.json' for prior in STAGES[:STAGES.index(stage)])
    return {str(p):digest(p) for p in paths}


def activate(job):
    from driversi.agents.backends.sources import activate_packages
    from driversi.vendor import path
    vendor = path('mtgs')
    activate_packages({'mtgs': vendor/'mtgs', 'nuplan_scripts': vendor/'nuplan_scripts',
                       'nuplan': path('nuplan-devkit')/'nuplan'})
    from nuplan_scripts.utils import constants
    constants.NUPLAN_DATA_ROOT = str(Path(job['database']).parent)
    constants.NUPLAN_DB_FILES = str(Path(job['database']).parent)
    constants.NUPLAN_SENSOR_ROOT = job['sensor_root']
    constants.NUPLAN_MAPS_ROOT = job['maps_root']
    constants.NUPLAN_MAP_VERSION = 'nuplan-maps-v1.0'
    from nuplan_scripts.utils.config import load_config
    from nuplan_scripts.utils.video_scene_dict_tools import VideoScene
    cfg = load_config(job['config'])
    return cfg, VideoScene(cfg)


def native(module, job, extra=()):
    sys.argv = [module, '--config', job['config'], *extra]
    try:
        runpy.run_module('nuplan_scripts.'+module, run_name='__main__')
    except SystemExit as exc:
        if exc.code not in (None,0):raise
        # A successful CLI exit must return to the stage receipt writer.


def video_stage(job, cfg, scene):
    import numpy as np
    from nuplan.database.nuplan_db_orm.nuplandb import NuPlanDB
    from nuplan_scripts.nuplan_video_processing import ProduceVideoSceneDict
    sensors = read_mapping(job['sensors'])['rows']
    tokens = list(dict.fromkeys(row['token'] for row in sensors))
    db = NuPlanDB(str(Path(job['database']).parent), job['database'], None, verbose=False)
    pcs = [db.lidar_pc[token] for token in tokens]
    pcs.sort(key=lambda p: p.timestamp)
    video = dict(video_token=cfg.road_block_name+'-0', video_idx=0,
                 log_token=db.log.token, log_name=db.log.logfile,
                 start_ts=pcs[0].timestamp, lidar_pcs=pcs,
                 trajectory=np.asarray([[p.ego_pose.x,p.ego_pose.y] for p in pcs]))
    data = ProduceVideoSceneDict.run(cfg, {db.log.token:db}, [video], num_workers=0)
    actual = next(iter(data.values()))['frame_infos']
    if {f['token'] for f in actual} != set(tokens):
        raise ValueError('Native preprocessing lost measured source captures')
    for f in actual:
        for cam in f['cams'].values():
            if not (Path(job['sensor_root'])/cam['data_path']).is_file():
                raise ValueError('Native matched camera capture is missing')
    Path(scene.sub_data_root).mkdir(parents=True,exist_ok=True)
    scene.video_scene_dict=data
    scene.dump_pickle(scene.pickle_path_raw)
    scene.update_pickle_link(scene.pickle_path_raw)
    db.session.close()


def triangulation_stage(job, cfg, scene):
    import pycolmap
    from nuplan_scripts.ba_multi_traversal import (copy_ego_masks, create_colmap_model,
                                                  align_model)
    data = scene.load_pickle(scene.pickle_path)
    copy_ego_masks(scene)
    result = {}
    for video_id, video in data.items():
        folder = Path(scene.colmap_path)/video_id
        part = {video_id:video}
        create_colmap_model(scene,part,str(folder))
        reader = pycolmap.ImageReaderOptions()
        reader.mask_path = str(folder/'masks')
        sift = pycolmap.FeatureExtractionOptions()
        sift.num_threads = job.get('cpu_workers',8)
        sift.max_image_size = 3200
        pycolmap.extract_features(database_path=str(folder/'database.db'),
            image_path=str(folder/'images'), reader_options=reader,
            extraction_options=sift, device=pycolmap.Device.cpu)
        match = pycolmap.FeatureMatchingOptions()
        match.num_threads = job.get('cpu_workers',8)
        pairs = pycolmap.ImportedPairingOptions()
        pairs.match_list_path = folder/'image_pairs.txt'
        pycolmap.match_image_pairs(database_path=str(folder/'database.db'),
            pairing_options=pairs, matching_options=match,
            device=pycolmap.Device.cpu)
        options = pycolmap.IncrementalPipelineOptions()
        options.num_threads = job.get('cpu_workers',8)
        options.ba_global_function_tolerance = .000001
        options.ba_global_max_num_iterations = 30
        options.ba_global_max_refinements = 3
        dst=folder/'sfm_model/0';dst.mkdir(parents=True,exist_ok=True)
        pycolmap.triangulate_points(pycolmap.Reconstruction(str(folder/'sparse_model')),
            str(folder/'database.db'),str(folder/'images'),str(dst),
            clear_points=False, options=options, refine_intrinsics=True)
        result.update(align_model(scene,part,str(folder)))
    scene.video_scene_dict=result
    scene.dump_pickle(scene.pickle_path_colmap)
    scene.update_pickle_link(scene.pickle_path_colmap)


def terrain_stage(job, cfg, scene):
    """Measured road cells only; unknown/ambiguous cells stay NaN."""
    import numpy as np
    import open3d as o3d
    cloud=Path(scene.rgb_point_cloud_path)/(cfg.road_block_name+'-0.pcd')
    labels=cloud.with_name(cloud.stem+'_sem_labels.npy')
    xyz=np.asarray(o3d.io.read_point_cloud(str(cloud)).points)
    semantic=np.load(labels,allow_pickle=False)
    if semantic.shape != (len(xyz),) or not np.isfinite(xyz).all():
        raise ValueError('Measured road point correspondence required')
    road=xyz[semantic==0]
    if len(road)<16:raise ValueError('Insufficient measured road returns')
    low=np.floor(road[:,:2].min(0)*4)/4
    ij=np.floor((road[:,:2]-low)*4).astype(int)
    shape=ij.max(0)+2
    if int(np.prod(shape))>4000000:raise ValueError('Bounded road grid exceeded')
    grid=np.full(tuple(shape[::-1]),np.nan,dtype=np.float32)
    cell=ij[:,1]*shape[0]+ij[:,0]
    order=np.argsort(cell,kind='stable')
    for ids in np.split(order,np.flatnonzero(np.diff(cell[order]))+1):
        z=road[ids,2]
        if len(z)>=3 and np.ptp(z)<=.08:
            x,y=ij[ids[0]];grid[y,x]=np.mean(z)
    folder=Path(job['asset_root'])/cfg.road_block_name/'road_height_map'
    folder.mkdir(parents=True,exist_ok=True)
    np.save(folder/'road_height_map.npy',grid)
    # RoadHeightMap maps reconstruction metres to pixels via s*(R@xy+t).
    write_json(folder/'sim2.json',dict(R=np.eye(2).tolist(),t=(-low).tolist(),s=4.))
    write_json(folder/'provenance.json',dict(method='measured_road_cells_v1',
        source={str(cloud):digest(cloud),str(labels):digest(labels)},
        finite_cells=int(np.isfinite(grid).sum()),unknown_cells=int(np.isnan(grid).sum()),
        cell_width_m=.25,minimum_returns=3,max_vertical_spread_m=.08,
        interpolation=False,trainable=False))


def execute_stage(job, stage):
    cfg,scene=activate(job)
    if stage=='video':video_stage(job,cfg,scene)
    elif stage=='semantic':native('generate_semantic_mask',job)
    elif stage=='registration':native('lidar_registration_multi_traversal',job)
    elif stage=='triangulation':triangulation_stage(job,cfg,scene)
    elif stage=='depth':native('generate_dense_depth',job,('--num_workers','0'))
    elif stage=='point_cloud':native('stack_RGB_point_cloud',job,('--num_workers','0'))
    elif stage=='training':native('auto_reconstruction.background_reconstruction',job)
    elif stage=='export':native('auto_reconstruction.export_reconstruction',job,
                               ('--output_dir',job['asset_root']))
    elif stage=='terrain':terrain_stage(job,cfg,scene)
    else:raise ValueError('Unregistered reconstruction stage')
    # A stage receipt records real outputs, never a nominal success.
    paths = {
        'video':[scene.pickle_path_raw], 'semantic':list(Path(scene.raw_mask_path).rglob('*.png')),
        'registration':[scene.pickle_path_registered], 'triangulation':[scene.pickle_path_colmap],
        'depth':list(Path(scene.optimal_undistorted_depth_path).rglob('*.png')),
        'point_cloud':[scene.pickle_path_final,*Path(scene.rgb_point_cloud_path).glob('*')],
        'training':list((Path(cfg.data_root)/'background_reconstruction'/cfg.road_block_name).rglob('*.ckpt')),
        'export':list((Path(job['asset_root'])/cfg.road_block_name).rglob('*.ckpt')),
        'terrain':list((Path(job['asset_root'])/cfg.road_block_name/'road_height_map').glob('*')),
    }[stage]
    paths=[Path(p) for p in paths if Path(p).is_file()]
    if not paths:raise ValueError('Stage produced no required artifact')
    write_json(Path(job['output'])/'receipts'/f'{stage}.json',dict(stage=stage,status='completed',
        inputs=stage_bindings(job,stage),outputs={str(p):digest(p) for p in paths},trainable=False,
        execution_producer_sha256=digest(__file__),
        producer_amendment_sha256=digest(Path(job['output'])/'producer-amendment.json') if (Path(job['output'])/'producer-amendment.json').exists() else None))


def run(job_path):
    job_path=Path(job_path).resolve(strict=True);job=read_mapping(job_path)
    out=Path(job['output']).resolve()
    if not out.is_relative_to(project_root()/'exp'):
        raise ValueError('Owned reconstruction output required')
    out.mkdir(parents=True,exist_ok=True);(out/'receipts').mkdir(exist_ok=True)
    from driversi.rsi.experiments.continuation import frozen_json
    expected=dict(job_sha256=digest(job_path),producer_sha256=digest(__file__),
        effective_inputs=effective_inputs(job),minimum_source_psnr_db=27.,native_student_and_loss_unchanged=True)
    original=out/'inputs.json'
    if original.exists() and read_mapping(original).get('producer_sha256')!=expected['producer_sha256']:
        prior=read_mapping(original);amendment=read_mapping(out/'producer-amendment.json')
        if ({k:v for k,v in prior.items() if k!='producer_sha256'} !=
                {k:v for k,v in expected.items() if k!='producer_sha256'}
                or amendment.get('original_inputs_sha256')!=digest(original)
                or amendment.get('old_producer_sha256')!=prior['producer_sha256']
                or amendment.get('new_producer_sha256')!=expected['producer_sha256']
                or amendment.get('scope')!='normal_systemexit_zero_no_geometry_change'):
            raise ValueError('Unverified reconstruction producer change')
    else:frozen_json(original,expected)
    started=time.monotonic()
    for stage in STAGES:
        receipt=out/'receipts'/f'{stage}.json'
        if receipt.exists():
            saved=read_mapping(receipt)
            if (saved['status']!='completed' or saved.get('inputs')!=stage_bindings(job,stage)
                or any(digest(p)!=h for p,h in saved['outputs'].items())):
                raise ValueError('Completed reconstruction evidence changed')
            continue
        running=out/'running.json'
        if running.exists():
            # A killed supervisor must reconcile the exact owned child first.
            raise RuntimeError('Unknown in-flight stage requires reconciliation')
        log=(out/f'{stage}.log').open('ab')
        cmd=[sys.executable,'-u','-m',__spec__.name,'--job',str(job_path),'--stage',stage]
        child=subprocess.Popen(cmd,cwd=job['working_directory'],stdout=log,stderr=subprocess.STDOUT)
        write_json(running,dict(stage=stage,pid=child.pid,command=cmd))
        write_json(out/'progress.json',dict(status='running',stage=stage,pid=child.pid,
            completed_stages=STAGES.index(stage),wall_seconds=time.monotonic()-started))
        try:
            rc=child.wait(timeout=job.get('stage_timeout_seconds',7200))
        except subprocess.TimeoutExpired:
            import psutil
            owned=psutil.Process(child.pid).children(recursive=True)
            for process in reversed(owned):
                try:process.terminate()
                except psutil.NoSuchProcess:pass
            child.terminate()
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                for process in reversed(owned):
                    try:process.kill()
                    except psutil.NoSuchProcess:pass
                child.kill();child.wait()
            rc=124
        finally:log.close()
        running.unlink()
        if rc or not receipt.exists():
            write_json(out/'report.json',dict(status='stage_failed',stage=stage,returncode=rc,
                wall_seconds=time.monotonic()-started,trainable=False))
            return False
    cfg,scene=activate(job)
    bundle=Path(job['asset_root'])/cfg.road_block_name
    write_json(out/'report.json',dict(status='reconstructed_requires_source_preflight',
        bundle_root=str(bundle),asset_id=cfg.road_block_name,
        metadata_sha256=digest(bundle/'video_scene_dict.pkl'),
        wall_seconds=time.monotonic()-started,trainable=False))
    return True


def main():
    configure_runtime()
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job',required=True);p.add_argument('--stage',choices=STAGES)
    a=p.parse_args()
    if a.stage:execute_stage(read_mapping(a.job),a.stage)
    elif not run(a.job):raise SystemExit(2)


if __name__=='__main__':main()
