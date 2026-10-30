"""Export future scene-token labels only. Never generates trajectory pseudo-labels."""
import argparse
from datetime import timedelta
import hashlib
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SensorConfig
from navsim.planning.training.log_partitions import resolve_training_log_names
from navsim.agents.chainflow_fpd_future_predictor.builders.feature_builder import camera_tensor
from navsim.agents.chainflow_fpd_future_predictor.target_cache import (
    SCHEMA, cache_file, file_sha256, load_record, motion_targets,
)


class FutureImages(Dataset):
    def __init__(self, loader, tokens, image_size):
        self.loader, self.tokens, self.image_size = loader, tokens, image_size

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        token = self.tokens[index]
        scene = self.loader.get_scene_from_token(token)
        targets = motion_targets(scene)
        image = camera_tensor(scene.frames[7].cameras, self.image_size)
        return image, targets, token


def write_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--training-config')
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--expected-scenes', type=int, default=103288)
    parser.add_argument('--precision', choices=('bf16-mixed', '32-true'), default='bf16-mixed')
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0 or args.expected_scenes < 1:
        raise ValueError('Invalid export configuration')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    rank, world = int(os.getenv('RANK', 0)), int(os.getenv('WORLD_SIZE', 1))
    local = int(os.getenv('LOCAL_RANK', 0))
    if world > 1:
        dist.init_process_group('gloo', timeout=timedelta(hours=24))
    torch.cuda.set_device(local)
    torch.manual_seed(0)
    checkpoint = Path(args.checkpoint).resolve()
    teacher_hash = file_sha256(checkpoint)
    saved_path = Path(args.training_config) if args.training_config else checkpoint.parents[3] / 'resolved_config.yaml'
    source_cfg = OmegaConf.load(saved_path)
    if source_cfg.train_test_split.get('synthetic_sources'):
        raise ValueError('Only real recorded training data is supported')
    config_dir = Path(__file__).parent / 'config/common/agent'
    with initialize_config_dir(config_dir=str(config_dir.resolve()), version_base=None):
        cfg = compose(config_name='chainflow_fpd_future_predictor')
    cfg.batch_size = args.batch_size
    cfg.initialization_checkpoint = str(checkpoint)
    cfg.enable_ray = False
    # Explicit alternative checkpoints require an explicit matching hash in the
    # training configuration too; the default export is the selected best teacher.
    if str(cfg.config.teacher_sha256) != teacher_hash:
        raise ValueError('Export checkpoint is not the configured best teacher')
    agent = instantiate(cfg)
    agent.initialize()
    agent.eval().requires_grad_(False).cuda(local)
    model = agent._chainflow_vla_model
    scene_filter = instantiate(source_cfg.train_test_split.scene_filter)
    train_logs = set(resolve_training_log_names(source_cfg))
    scene_filter.log_names = sorted(train_logs if scene_filter.log_names is None else train_logs.intersection(scene_filter.log_names))
    sensors = SensorConfig(cam_f0=[7], cam_b0=[7], cam_l0=[7], cam_r0=[7],
                           cam_l1=[], cam_l2=[], cam_r1=[], cam_r2=[], lidar_pc=[])
    loader = SceneLoader(data_path=Path(source_cfg.navsim_log_path), sensor_blobs_path=Path(source_cfg.sensor_blobs_path),
                         scene_filter=scene_filter, sensor_config=sensors)
    tokens = sorted(loader.tokens)
    if len(tokens) != args.expected_scenes or len(set(tokens)) != len(tokens):
        raise ValueError(f'Expected {args.expected_scenes} unique training tokens, got {len(tokens)}')
    metadata = torch.load(checkpoint, map_location='cpu', weights_only=False)['native_teacher_training']
    identity = {'schema': SCHEMA, 'teacher_checkpoint': str(checkpoint), 'teacher_sha256': teacher_hash,
                'teacher_config': metadata['contract']['config'], 'future_offset': 4,
                'query_protocol': 'actual_recorded_timestamp', 'image_size': [1148, 672],
                'precision': args.precision, 'expected_scenes': len(tokens),
                'tokens_sha256': hashlib.sha256('\n'.join(tokens).encode()).hexdigest(),
                'navsim_log_path': str(Path(source_cfg.navsim_log_path).resolve()),
                'sensor_blobs_path': str(Path(source_cfg.sensor_blobs_path).resolve())}
    del metadata
    root = Path(args.output).resolve()
    if rank == 0:
        root.mkdir(parents=True, exist_ok=True)
        if (root / 'manifest.json').exists():
            old = json.loads((root / 'manifest.json').read_text())
            if any(old.get(key) != value for key, value in identity.items()):
                raise ValueError('Existing latent cache belongs to another teacher/data/preprocessing setup')
        write_json(root / 'manifest.json', {**identity, 'complete': False})
        (root / 'tokens.txt').write_text('\n'.join(tokens) + '\n')
    if world > 1:
        dist.barrier()
    owned = tokens[rank::world]
    pending = []
    for token in owned:
        if cache_file(root, token).exists():
            load_record(root, token, teacher_hash)
        else:
            pending.append(token)
    dl = DataLoader(FutureImages(loader, pending, cfg.config.image_size), batch_size=args.batch_size,
                    num_workers=args.num_workers, pin_memory=True, shuffle=False, drop_last=False)
    completed = len(owned) - len(pending)
    for images, targets, batch_tokens in dl:
        images = images.cuda(local, non_blocking=True)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.precision == 'bf16-mixed'):
            latent = model.image_backbone(images, model.scene_embeds.expand(len(images), -1, -1, -1))
        latent = latent.float().cpu().numpy()
        for i, token in enumerate(batch_tokens):
            path = cache_file(root, token)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix('.npz.tmp')
            with temporary.open('wb') as stream:
                np.savez_compressed(stream, token=np.asarray(token), teacher_sha256=np.asarray(teacher_hash),
                                    future_tokens=latent[i], **{k: v[i].numpy() for k, v in targets.items()})
            temporary.replace(path)
            load_record(root, token, teacher_hash)
        completed += len(batch_tokens)
        logging.info('rank %d: %d/%d future labels exported', rank, completed, len(owned))
    if world > 1:
        dist.barrier()
    if rank == 0:
        if any(not cache_file(root, token).exists() for token in tokens):
            raise RuntimeError('Incomplete future target export')
        write_json(root / 'manifest.json', {**identity, 'complete': True, 'completed_scenes': len(tokens)})
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
