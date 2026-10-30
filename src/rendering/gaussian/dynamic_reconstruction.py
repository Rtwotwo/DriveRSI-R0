"""Isolated native MTGS joint reconstruction with actual token-bound vehicles.

No object shape/appearance is invented. A completed asset is not an admission:
source PSNR and per-render visible-object completeness remain separate gates.
"""
import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from driversi.agents.backends.native import digest, project_root
from driversi.configuration import read_mapping
from driversi.runtime.budget import write_json

RUNTIME_KEYS = ('CUDA_HOME', 'TORCH_EXTENSIONS_DIR', 'TORCH_CUDA_ARCH_LIST', 'MAX_JOBS', 'OMP_NUM_THREADS')
MODULE = 'driversi.rendering.gaussian.dynamic_reconstruction'


def _owned(path):
    path = Path(path).resolve()
    root = (project_root() / 'exp').resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError('Dynamic reconstruction outputs must be below project exp/')
    return path


def classify_objects(metadata, point_inventory):
    """Native rigid support is narrower than all annotated actors."""
    point_rows = {(r['video_token'], r['token']): r for r in point_inventory['objects']}
    objects = {}
    for video_token, video in metadata.items():
        for frame in video['frame_infos']:
            for token, name in zip(frame['track_tokens'], frame['gt_names']):
                key = (video_token, str(token))
                if key in objects and objects[key]['class_name'] != str(name):
                    raise ValueError('Object token changes class within source')
                count = int(point_rows.get(key, {}).get('measured_colored_points', 0))
                supported = str(name) == 'vehicle' and count >= 100
                objects[key] = {'video_token': video_token, 'token': str(token), 'class_name': str(name),
                                'measured_colored_points': count, 'supported': supported,
                                'reason': None if supported else ('unsupported_actor_class' if str(name) != 'vehicle' else 'insufficient_measured_points')}
    return list(objects.values())


def prepare_dynamic_job(recovery_job, output, gpu, max_iterations=30001):
    """Create an immutable small asset job; no training or GPU allocation here."""
    from driversi.rendering.gaussian.mtgs_reconstruction import activate
    from driversi.vendor import path as vendor_path
    recovery_path = Path(recovery_job).resolve(strict=True)
    recovery = read_mapping(recovery_path)
    # The Harness CPU environment intentionally does not contain MTGS packages.
    # Native YAML objects must be loaded in the pinned reconstruction runtime.
    runtime = Path(recovery['reconstruction_python']).absolute()
    if Path(sys.executable).absolute() != runtime:
        subprocess.run([str(runtime), '-m', MODULE, '--prepare-recovery', str(recovery_path),
                        '--output', str(_owned(output)), '--gpu', str(gpu),
                        '--max-iterations', str(max_iterations)], cwd=project_root(),
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return _owned(output) / 'job.json'
    original_path = Path(recovery['reconstruction_job']).resolve(strict=True)
    original = read_mapping(original_path)
    cfg, scene = activate(original)
    output = _owned(output)
    if not 1 <= int(max_iterations) <= 30001:
        raise ValueError('Bounded native iteration count required')
    job = {'schema_version': 1, 'kind': 'native_mtgs_dynamic_v1', 'output': str(output),
           'recovery_job': str(recovery_path), 'source_job': str(original_path),
           'gpu': int(gpu), 'max_iterations': int(max_iterations), 'source_token': recovery['token'],
           'source_log': recovery['source_log'], 'native_minimum_points': 100,
           'source_processed_root': str(Path(scene.sub_data_root).resolve(strict=True)),
           'source_metadata': str(Path(scene.pickle_path_final).resolve(strict=True)),
           'config': str(output / 'mtgs.yaml'), 'asset_root': str(output / 'assets'),
           'database': original['database'], 'maps_root': original['maps_root'],
           'sensor_root': original['sensor_root'], 'cpu_workers': min(8, original.get('cpu_workers', 8)),
           'python': recovery['reconstruction_python'], 'runtime_witness': recovery.get('runtime_witness'),
           'source_quality_pending': True, 'memory_fraction': 0.15}
    source_files = [recovery_path, original_path, Path(original['config']), Path(job['source_metadata'])]
    road_cloud = Path(scene.rgb_point_cloud_path) / (cfg.road_block_name + '-0.pcd')
    source_files += [road_cloud, road_cloud.with_name(road_cloud.stem + '_sem_labels.npy')]
    source_files += [p for p in (vendor_path('mtgs')).rglob('*.py') if '__pycache__' not in p.parts]
    source_files += list((vendor_path('mtgs')).glob('*LICENSE*'))
    source_files += [Path(__file__), Path(__file__).with_name('dynamic_point_cloud.py')]
    job['inputs'] = {str(p.resolve()): digest(p) for p in source_files}
    job_path = output / 'job.json'
    if job_path.exists():
        if read_mapping(job_path) != job:
            raise ValueError('Existing dynamic job differs; use a new versioned output')
        return job_path
    output.mkdir(parents=True, exist_ok=True)
    cfg = copy.deepcopy(cfg)
    cfg.data_root = str(output / 'processed')
    cfg.save_config(job['config'])
    dst = Path(cfg.data_root) / cfg.road_block_name
    dst.mkdir(parents=True, exist_ok=True)
    # Native parsing reads these completed products. Instance point clouds alone
    # are regenerated into a private directory, leaving old receipts untouched.
    for src in Path(job['source_processed_root']).iterdir():
        if src.name == 'instance_point_cloud':
            continue
        target = dst / src.name
        if target.is_symlink() and target.resolve() == src.resolve():
            continue
        if target.exists() or target.is_symlink():
            raise ValueError('Existing preparation path differs from source')
        target.symlink_to(src.resolve(), target_is_directory=src.is_dir())
    (dst / 'instance_point_cloud').mkdir(exist_ok=True)
    write_json(job_path, job)
    return job_path


def _verify(job):
    for name, sha in job['inputs'].items():
        if digest(Path(name)) != sha:
            raise ValueError(f'Immutable input changed: {name}')
    _owned(job['output'])
    if job['native_minimum_points'] != 100:
        raise ValueError('Do not relax native measured-point eligibility')


def native_training_complete(checkpoint_dir, max_iterations):
    """A finished checkpoint is usable even if our post-training receipt was lost."""
    import torch
    path = Path(checkpoint_dir) / f'step-{max_iterations - 1:09d}.ckpt'
    if not path.exists():
        return False
    state = torch.load(path, map_location='cpu', weights_only=False)
    return (state.get('step') == max_iterations - 1 and bool(state.get('pipeline'))
            and 'optimizers' not in state)


def _train(job):
    from driversi.rendering.gaussian.mtgs_reconstruction import activate, configure_runtime
    configure_runtime()
    cfg, _ = activate(job)
    os.environ['NERFSTUDIO_DATAPARSER_CONFIGS'] = 'nuplan=mtgs.config.nuplan_dataparser:nuplan_dataparser'
    os.environ['NERFSTUDIO_METHOD_CONFIGS'] = 'mtgs_st=mtgs.config.MTGS_ST:method'
    import torch
    torch.set_num_threads(job['cpu_workers'])
    torch.cuda.set_per_process_memory_fraction(job['memory_fraction'])
    from mtgs.config.MTGS_ST import config
    from nerfstudio.scripts.train import main
    config = copy.deepcopy(config)
    train_dir = Path(cfg.data_root) / 'background_reconstruction' / cfg.road_block_name
    train_dir.mkdir(parents=True, exist_ok=True)
    config.experiment_name = cfg.road_block_name
    config.pipeline.datamanager.cache_strategy = 'prefetch'
    parser = config.pipeline.datamanager.dataparser
    parser.road_block_config = job['config']
    parser.train_scene_travels = (0,)
    parser.eval_scene_travels = (0,)
    parser.only_moving = False
    config.pipeline.model.color_corrected_metrics = False
    config.pipeline.model.lpips_metric = False
    config.pipeline.model.model_config['background'].verbose = False
    config.max_num_iterations = job['max_iterations']
    config.output_dir = train_dir
    config.set_base_dir(str(train_dir))
    # Restart only from native state that includes optimizers; otherwise restart
    # this new isolated asset. Completed old background assets are never edited.
    load_dir = train_dir / 'nerfstudio_models'
    resume = sorted(load_dir.glob('step-*.ckpt')) if load_dir.exists() else []
    if resume:
        state = torch.load(resume[-1], map_location='cpu', weights_only=False)
        if state.get('optimizers'):
            config.load_dir = load_dir
    main(config)


def validate_portable(checkpoint, objects):
    import torch
    data = torch.load(checkpoint, map_location='cpu', weights_only=False)
    expected = {row['token'] for row in objects if row['supported']}
    actual = {key.removeprefix('rigid_object_') for key in data if key.startswith('rigid_object_')}
    if not expected or actual != expected or not {'background', 'skybox'} <= set(data):
        raise ValueError('Exported real rigid nodes do not match measured object eligibility')
    rows = []
    for token in sorted(expected):
        node = data['rigid_object_' + token]
        if node['config']['type'] not in ('RigidSubModel', 'MirroredRigidSubModel'):
            raise ValueError('Unexpected rigid object implementation')
        tensors = [value for value in node['state_dict'].values() if torch.is_tensor(value)]
        if not tensors or not all(torch.isfinite(v).all() for v in tensors):
            raise ValueError('Nonfinite or empty object asset')
        means = next((v for k, v in node['state_dict'].items() if k.endswith('means')), None)
        if means is None or len(means) == 0:
            raise ValueError('Empty reconstructed object Gaussian field')
        rows.append({'token': token, 'node': 'rigid_object_' + token, 'gaussians': len(means),
                     'native_type': node['config']['type']})
    return rows


def terrain_outputs(bundle):
    """The portable replay contract requires measured terrain, not only RGB."""
    import numpy as np
    folder = Path(bundle) / 'road_height_map'
    paths = [folder / name for name in ('road_height_map.npy', 'sim2.json', 'provenance.json')]
    if not all(path.is_file() for path in paths):
        raise ValueError('Dynamic bundle lacks measured terrain outputs')
    grid = np.load(paths[0], allow_pickle=False)
    transform = read_mapping(paths[1])
    provenance = read_mapping(paths[2])
    if (grid.ndim != 2 or not np.isfinite(grid).any() or np.isinf(grid).any()
            or provenance.get('method') != 'measured_road_cells_v1'
            or provenance.get('interpolation') is not False
            or transform.get('s') != 4.):
        raise ValueError('Invalid measured terrain contract')
    for name, sha in provenance['source'].items():
        if digest(name) != sha:
            raise ValueError('Measured terrain source changed')
    return paths


def run_dynamic_job(job_path):
    import fcntl
    from driversi.rendering.gaussian.mtgs_reconstruction import activate, native
    from driversi.rendering.gaussian.dynamic_point_cloud import build_instance_clouds
    job_path = Path(job_path).resolve(strict=True)
    job = read_mapping(job_path)
    _verify(job)
    output = Path(job['output'])
    with (output / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report_path = output / 'report.json'
        if report_path.exists():
            report = read_mapping(report_path)
            if report.get('status') == 'completed':
                for name, sha in report['outputs'].items():
                    if digest(Path(name)) != sha:
                        raise ValueError('Completed dynamic asset has changed')
                return report
        recovery = read_mapping(job['recovery_job'])
        for key in RUNTIME_KEYS:
            if key in recovery.get('runtime_environment', {}):
                os.environ[key] = str(recovery['runtime_environment'][key])
        os.environ['CUDA_VISIBLE_DEVICES'] = str(job['gpu'])
        os.environ['OMP_NUM_THREADS'] = str(job['cpu_workers'])
        import torch
        torch.set_num_threads(job['cpu_workers'])
        start = time.time()
        cfg, scene = activate(job)
        point_path = output / 'point-cloud-inventory.json'
        if not point_path.exists():
            write_json(output / 'status.json', {'status': 'preparing_measured_object_clouds', 'started_at': start})
            points = build_instance_clouds(job, scene)
            points['files'] = {r['path']: digest(Path(r['path'])) for r in points['objects'] if r['path']}
            write_json(point_path, points)
        else:
            points = read_mapping(point_path)
            for name, sha in points['files'].items():
                if digest(Path(name)) != sha:
                    raise ValueError('Prepared measured object cloud changed')
        objects = classify_objects(scene.load_pickle(scene.pickle_path), points)
        write_json(output / 'object-eligibility.json', {'objects': objects})
        if not any(row['supported'] for row in objects):
            report = {'status': 'unsupported', 'reason': 'no_rigid_object_with_native_measured_support',
                      'unsupported_objects': objects, 'elapsed_seconds': time.time() - start}
            write_json(report_path, report)
            return report
        train_receipt = output / 'training-receipt.json'
        native_checkpoints = Path(cfg.data_root) / 'background_reconstruction' / cfg.road_block_name / 'nerfstudio_models'
        if not train_receipt.exists() and native_training_complete(native_checkpoints, job['max_iterations']):
            write_json(train_receipt, {'elapsed_seconds': None, 'iterations': job['max_iterations'],
                                      'only_moving': False, 'joint_background_skybox_real_rigid': True,
                                      'recovered_completed_native_checkpoint_without_retraining': True})
        if not train_receipt.exists():
            write_json(output / 'status.json', {'status': 'joint_native_training', 'started_at': start,
                       'eligible_objects': sum(row['supported'] for row in objects)})
            t0 = time.time()
            with (output / 'training.log').open('a') as log:
                subprocess.run([sys.executable, '-m', MODULE, '--job', str(job_path), '--stage', 'train'],
                               cwd=project_root(), stdout=log, stderr=subprocess.STDOUT, check=True)
            write_json(train_receipt, {'elapsed_seconds': time.time() - t0, 'iterations': job['max_iterations'],
                                      'only_moving': False, 'joint_background_skybox_real_rigid': True})
        from driversi.rendering.gaussian.mtgs_reconstruction import configure_runtime
        configure_runtime()
        torch.cuda.set_per_process_memory_fraction(job['memory_fraction'])
        native('auto_reconstruction.export_reconstruction', job, ['--output_dir', job['asset_root']])
        bundle = Path(job['asset_root']) / cfg.road_block_name
        checkpoint = bundle / 'background' / f'{cfg.road_block_name}.ckpt'
        metadata = bundle / 'video_scene_dict.pkl'
        nodes = validate_portable(checkpoint, objects)
        from driversi.rendering.gaussian.mtgs_reconstruction import terrain_stage
        terrain_stage(job, cfg, scene)
        terrain_files = terrain_outputs(bundle)
        binding = bundle / 'dynamic-binding-manifest.json'
        write_json(binding, {'schema_version': 1, 'source_token': job['source_token'], 'source_log': job['source_log'],
                            'checkpoint_sha256': digest(checkpoint), 'metadata_sha256': digest(metadata),
                            'point_inventory_sha256': digest(point_path), 'object_nodes': nodes,
                            'terrain_files': {str(p): digest(p) for p in terrain_files},
                            'unsupported_objects': [r for r in objects if not r['supported']],
                            'source_quality_pending': True, 'novel_view_pixel_truth_available': False,
                            'construction': 'native_MTGS_ST_joint_all_measured_lidar_only_moving_false'})
        report = {'status': 'completed', 'bundle_root': str(bundle), 'checkpoint': str(checkpoint),
                  'metadata': str(metadata), 'binding_manifest': str(binding), 'object_tokens': [r['token'] for r in nodes],
                  'terrain': str(bundle / 'road_height_map'),
                  'unsupported_objects': [r for r in objects if not r['supported']], 'bindings_validated': True,
                  'source_quality_pending': True, 'elapsed_seconds': time.time() - start,
                  'outputs': {str(p): digest(p) for p in [checkpoint, metadata, binding, point_path, train_receipt, *terrain_files]}}
        write_json(report_path, report)
        write_json(output / 'status.json', report)
        return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job')
    parser.add_argument('--prepare-recovery')
    parser.add_argument('--output')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--max-iterations', type=int, default=30001)
    parser.add_argument('--stage', choices=['train'])
    args = parser.parse_args()
    if args.prepare_recovery:
        print(prepare_dynamic_job(args.prepare_recovery, args.output, args.gpu, args.max_iterations))
    elif args.stage:
        job = read_mapping(args.job)
        _verify(job)
        _train(job)
    else:
        try:
            report = run_dynamic_job(args.job)
        except Exception as exc:
            if args.job:
                output = _owned(read_mapping(args.job)['output'])
                write_json(output / 'status.json', {'status': 'failed', 'error_type': type(exc).__name__,
                                                  'error': str(exc), 'failed_at': time.time()})
            raise
        print(json.dumps({key: report[key] for key in ('status', 'elapsed_seconds')}))


if __name__ == '__main__':
    main()
