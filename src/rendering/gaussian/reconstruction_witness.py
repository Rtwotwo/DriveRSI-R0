"""Measured isolated-runtime witness; does not grant synthetic-data admission."""
import argparse
import json
from pathlib import Path

from driversi.agents.backends.native import digest, project_root
from driversi.configuration import read_mapping
from driversi.runtime.budget import write_json


def run(job_path):
    from driversi.rendering.gaussian.mtgs_reconstruction import configure_runtime
    configure_runtime()
    import numpy as np
    import torch
    import pycolmap
    from gsplat.rendering import rasterization
    from unidepth.models import UniDepthV2
    from transformers import Mask2FormerForUniversalSegmentation
    from kiss_icp.kiss_icp import KissICP
    from driversi.rendering.gaussian.mtgs_reconstruction import activate, video_stage
    from driversi.rsi.experiments.continuation import frozen_json
    job=read_mapping(job_path);out=Path(job['output'])
    if not out.resolve().is_relative_to(project_root()/'exp'):raise ValueError('Owned witness output required')
    out.mkdir(parents=True,exist_ok=True)
    frozen_json(out/'job.json',job)
    torch.manual_seed(42);np.random.seed(42)
    torch.cuda.set_per_process_memory_fraction(.3)
    means=torch.tensor([[0.,0.,3.],[.2,.1,4.]],device='cuda',requires_grad=True)
    quat=torch.tensor([[1.,0.,0.,0.]]*2,device='cuda')
    scale=torch.full((2,3),.1,device='cuda')
    opacity=torch.full((2,),.8,device='cuda')
    colors=torch.rand((2,3),device='cuda')
    view=torch.eye(4,device='cuda')[None]
    K=torch.tensor([[[32.,0.,16.],[0.,32.,16.],[0.,0.,1.]]],device='cuda')
    image,alpha,_=rasterization(means,quat,scale,opacity,colors,view,K,32,32,packed=False)
    assert image.shape==(1,32,32,3) and torch.isfinite(image).all()
    image.sum().backward();assert means.grad is not None and torch.isfinite(means.grad).all()
    torch.cuda.synchronize()
    import importlib
    cfg,scene=activate(job['native_job'])
    for module in ['generate_semantic_mask','lidar_registration_multi_traversal',
                   'generate_dense_depth','stack_RGB_point_cloud',
                   'auto_reconstruction.background_reconstruction','auto_reconstruction.export_reconstruction']:
        importlib.import_module('nuplan_scripts.'+module)
    if Path(scene.pickle_path_raw).exists():
        data=scene.load_pickle(scene.pickle_path_raw)
    else:
        video_stage(job['native_job'],cfg,scene);data=scene.video_scene_dict
    frames=next(iter(data.values()))['frame_infos']
    expected={x['token'] for x in read_mapping(job['native_job']['sensors'])['rows']}
    assert {x['token'] for x in frames}==expected
    result=dict(ready=False,checks_passed=True,seed=42,torch=torch.__version__,numpy=np.__version__,
        pycolmap=pycolmap.__version__,raster_sum=float(image.sum()),gradient_sum=float(means.grad.sum()),
        source_frames=len(frames),camera_records=sum(len(f['cams']) for f in frames),
        measured_source_metadata={str(scene.pickle_path_raw):digest(scene.pickle_path_raw)},
        producer_sha256=digest(__file__),job_sha256=digest(job_path),
        pending='fresh_process_witness_verification',trainable=False)
    write_json(out/'report.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--job',required=True)
    args=parser.parse_args();print(json.dumps(run(args.job)))


if __name__=='__main__':main()
