"""Prepare measured, training-only MTGS source windows without GPU execution.

The result is a reconstruction job, not an asset or training qualification.
Source coordinates only bound the reconstruction after retrieval has finished.
"""
import argparse
import json
import math
import sqlite3
from pathlib import Path

from driversi.configuration import read_mapping
from driversi.data.sources.navsim import load_log
from driversi.runtime.budget import write_json


CAMERAS = ("CAM_F0", "CAM_L0", "CAM_L1", "CAM_L2", "CAM_R0", "CAM_R1", "CAM_R2", "CAM_B0")


def resolve(relative, roots):
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Relative sensor path required')
    for root in roots:
        path = (Path(root) / relative).resolve()
        if path.is_file():
            return str(path)
    return None


def prepare(config, output):
    """Inspect the chosen source anchors, preserving missing measurements explicitly."""
    config = read_mapping(config)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    # A separately recorded official train membership, never a test query list.
    index = read_mapping(config['train_index'])
    expected = set(index['expected_tokens'])
    if index['split'] != 'navtrain':
        raise ValueError('Reconstruction sources must use official navtrain membership')
    candidates = config['candidates']
    if not 1 <= len(candidates) <= 20 or len({r['token'] for r in candidates}) != len(candidates):
        raise ValueError('One bounded problem with unique training candidates required')
    results = []
    for row in candidates:
        token, name = row['token'], row['source_log']
        if token not in expected or Path(name).name != name:
            raise ValueError('Selected reconstruction source is not an official training member')
        folder = output / token
        folder.mkdir(exist_ok=True)
        log = Path(config['logs_root']) / (name + '.pkl')
        frames = load_log(log, trusted=True)
        anchors = [i for i, f in enumerate(frames) if f['token'] == token]
        if len(anchors) != 1 or any(f['log_name'] != name for f in frames):
            raise ValueError('Unique token and declared source log must agree')
        anchor = anchors[0]
        begin, end = anchor - 6, anchor + 17
        window = frames[max(0, begin):end]
        missing, sensors = [], []
        if begin < 0 or end > len(frames):
            missing.append(dict(reason='source_window_truncated'))
        if any(abs(b['timestamp'] - a['timestamp'] - 500000) > 50000 for a, b in zip(window, window[1:])):
            missing.append(dict(reason='source_window_discontinuous'))
        for frame in window:
            items = [('lidar', frame['lidar_path'])] + [('camera', frame['cams'][c]['data_path']) for c in CAMERAS]
            for modality, relative in items:
                path = resolve(relative, config['sensor_roots'][modality]) if relative else None
                item = dict(token=frame['token'], timestamp_us=frame['timestamp'],
                            modality=modality, relative_path=relative, source_path=path)
                sensors.append(item)
                if path is None:
                    missing.append(dict(reason='missing_' + modality, **item))
        database = Path(config['db_root']) / (name + '.db')
        db_token = False
        if database.is_file():
            with sqlite3.connect(database.as_uri() + '?mode=ro', uri=True) as conn:
                db_token = conn.execute('SELECT 1 FROM lidar_pc WHERE token=?', (bytes.fromhex(token),)).fetchone() is not None
        if not db_token:
            missing.append(dict(reason='raw_database_or_anchor_missing'))
        coords = [f['ego2global_translation'][:2] for f in window]
        bounds = [math.floor(min(x[d] for x in coords) - 20) for d in (0, 1)] + [math.ceil(max(x[d] for x in coords) + 20) for d in (0, 1)]
        native = dict(road_block_name=name + '-' + token, road_block=bounds,
                      city=frames[anchor]['map_location'], data_root=str(output / 'processed'),
                      interval=5, expand_buffer=0, reconstruct_buffer=0,
                      selected_videos=[], split='trainval', collect_raw=False,
                      exclude_bad_registration=True, use_colmap_ba=False,
                      central_log=name, central_tokens=[token], multi_traversal_mode='off')
        # Exact upstream class fields; no unavailable 10 Hz frames are fabricated.
        import yaml
        native_text = '!!python/object:nuplan_scripts.utils.config.FrameCentralConfig\n' + yaml.safe_dump(native, sort_keys=False)
        p = folder / 'mtgs.yaml'
        if p.exists() and p.read_text() != native_text:
            raise ValueError('Reconstruction configuration changed; use a new preparation directory')
        p.write_text(native_text)
        write_json(folder / 'sensors.json', dict(rows=sensors, missing=missing, source_rate_hz=2))
        results.append(dict(token=token, source_log=name, config=str(p), database=str(database),
                            source_frames=len(window), camera_files=sum(x['modality']=='camera' and x['source_path'] is not None for x in sensors),
                            lidar_files=sum(x['modality']=='lidar' and x['source_path'] is not None for x in sensors),
                            missing_count=len(missing), source_window_complete=not missing,
                            status='source_window_ready' if not missing else 'missing_source_measurements',
                            asset_ready=False, reconstruction_started=False))
    value = dict(schema_version=1, kind='mtgs_train_source_preparation', candidates=results,
                 source_window_ready=sum(r['source_window_complete'] for r in results),
                 source_rate_hz=2, temporal_scope_s=[-3, 8],
                 full_preprocessor_readiness=False, reconstruction_started=False,
                 remaining=['native_preprocessor_runtime', 'whole_reconstruction_region_sensor_coverage',
                            'free_gpu', 'reconstruction_and_export', 'registration_and_visual_admission'])
    write_json(output / 'report.json', value)
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    print(json.dumps(prepare(a.config, a.output)))


if __name__ == '__main__':
    main()
