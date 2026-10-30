"""Recover native depth outputs after its successful CLI exit skipped a receipt."""
import argparse
from pathlib import Path
from driversi.configuration import read_mapping
from driversi.agents.backends.native import digest
from driversi.rsi.experiments.continuation import frozen_json
from driversi.runtime.budget import write_json


def validate_native_depth(array, size):
    import numpy as np
    if array.shape != (size[1], size[0], 3) or array.dtype != np.uint8:
        raise ValueError('Depth output shape/encoding differs from native source')
    # Native OpenCV output uses B/G for centimetres and leaves R unused.
    if array[:, :, 0].any():
        raise ValueError('Native depth unused channel must be zero')
    depth = array[:, :, 2].astype(np.uint16) + 256 * array[:, :, 1].astype(np.uint16)
    nonzero = depth[depth > 0]
    if not nonzero.size or nonzero.min() < 10 or nonzero.max() > 8000:
        raise ValueError('Invalid native centimetre depth range')


def run(payload):
    import numpy as np
    from PIL import Image
    from driversi.rendering.gaussian.mtgs_reconstruction import activate, stage_bindings
    job=read_mapping(payload['native_job']);root=Path(job['output']);out=Path(payload['output'])
    failure=read_mapping(root/'report.json')
    if failure.get('status')!='stage_failed' or failure.get('stage')!='depth' or failure.get('returncode')!=0:
        raise ValueError('Only successful native depth exit without receipt may be recovered')
    if (root/'running.json').exists():raise ValueError('Unsettled native process')
    cfg,scene=activate(job);data=scene.load_pickle(scene.pickle_path_colmap)
    cameras=[cam for v in data.values() for f in v['frame_infos'] if not f.get('skipped',False)
             for cam in f['cams'].values()]
    if not cameras:raise ValueError('No measured cameras')
    if f'{len(cameras)}/{len(cameras)}' not in (root/'depth.log').read_text(errors='replace'):
        raise ValueError('Native depth completion counter missing')
    files={}
    for cam in cameras:
        p=Path(scene.optimal_undistorted_depth_path)/cam['data_path'].replace('.jpg','.png')
        source=Path(scene.raw_image_path)/cam['data_path']
        with Image.open(source) as im:size=im.size
        with Image.open(p) as im:array=np.array(im)
        validate_native_depth(array,size)
        files[str(p)]=digest(p)
    if len(files)!=len(cameras):raise ValueError('Duplicate measured camera output')
    receipt=dict(stage='depth',status='completed',inputs=stage_bindings(job,'depth'),outputs=files,
        recovered_native_normal_exit=True,recovery_producer_sha256=digest(__file__),
        native_failure_sha256=digest(root/'report.json'),native_log_sha256=digest(root/'depth.log'),trainable=False)
    frozen_json(root/'receipts/depth.json',receipt)
    write_json(out/'report.json',dict(status='completed',images=len(files),reused_depth=True,trainable=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--job',required=True)
    run(read_mapping(parser.parse_args().job))
