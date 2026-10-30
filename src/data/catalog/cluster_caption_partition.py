"""Prepare isolated NAVSIM test captions and run after a verified train campaign."""
from __future__ import annotations

import argparse
import collections
import json
import sqlite3
from pathlib import Path

import numpy as np

from . import cluster_caption as captions
from . import cluster_caption_batch as batch
from . import cluster_caption_repair as repair
from .bailian_caption import write


def prerequisite(path):
    audit = json.loads(Path(path).read_text())
    if audit.get('status') not in ('verified', 'verified_with_unresolved_failures'):
        raise ValueError('Train final audit has not passed')
    for field in ('expected_members','source_members','database_members','exported_members','shard_members'):
        if audit.get(field) != 103288:
            raise ValueError('Train membership audit is incomplete')
    if (audit.get('missing_members') != 0 or audit.get('duplicate_members') != 0
            or audit.get('native_training_fields_unchanged') is not True
            or audit.get('original_source_and_copy_unchanged') is not True):
        raise ValueError('Train original source or labels failed audit')
    counts = audit['counts']
    if (sum(counts.values()) != 103288 or counts.get('pending', 0)
            or set(counts)-{'machine_draft','response_failed','preparation_failed'}):
        raise ValueError('Train campaign still has incomplete members')
    if sum(v for k,v in counts.items() if k.endswith('failed')) / 103288 > .01:
        raise ValueError('More than one percent unresolved train failures require inspection')
    cost = audit['cost']
    if not cost.get('snapshot_complete') or cost['spend_ledger_states'].get('reserved', 0):
        raise ValueError('Train receipt ledger has not reconciled')
    return audit


def initialize_navtest(index_file, train_library, library, output, expected_count=12146):
    """No future supervision or inference: derive only anchor metadata from logs."""
    index_file, library, output = Path(index_file), Path(library), Path(output)
    index = json.loads(index_file.read_text())
    records = index['records']
    expected = set(index['expected_tokens'])
    if (index['split'] != 'navtest' or index['complete'] is not True
            or len(records) != expected_count or len(expected) != expected_count
            or {r['token'] for r in records} != expected
            or any(r['kind'] != 'original_navtest' for r in records)):
        raise ValueError('Complete official navtest membership is required')
    db = sqlite3.connect(f'file:{Path(train_library)/"captions.sqlite3"}?mode=ro', uri=True)
    try:
        train_tokens = {r[0] for r in db.execute('SELECT token FROM captions')}
    finally:
        db.close()
    if expected & train_tokens:
        raise ValueError('Train and test membership overlaps')
    metadata = {}; commands = collections.Counter(); checked_images = 0
    command_map = {(1,0,0,0):'TURN LEFT', (0,1,0,0):'GO FORWARD', (0,0,1,0):'TURN RIGHT'}
    for record in sorted(records, key=lambda r:(r['source_log'],r['anchor_index'])):
        frames, _ = captions.source_log(record['log_path'])
        frame = frames[record['anchor_index']]
        if frame['token'] != record['token'] or frame['log_name'] != record['source_log']:
            raise ValueError('Indexed test anchor differs from source log')
        command = command_map.get(tuple(frame['driving_command']), 'UNKNOWN')
        dynamic = np.asarray(frame['ego_dynamic_state'], dtype=float)
        if dynamic.shape != (4,) or not np.isfinite(dynamic).all():
            raise ValueError('Invalid current ego state')
        for camera in captions.CAMERAS:
            image = Path(record['sensor_root']) / frame['cams'][camera]['data_path']
            if str(image) not in record['image_files'] or not image.is_file():
                raise ValueError('Missing or unindexed current test camera')
            checked_images += 1
        key = record['source_log']+'_'+record['token']
        # Same outer keys, but unavailable native targets are explicit nulls.
        # This is NAVSIM metadata, never an alleged official AutoDrive-P3 label.
        metadata[key] = dict(id=record['token'], scene_name=record['source_log'],
            frame_id=str(record['anchor_index']), video_path=None, driving_command=command,
            ego_dynamic_state=f'vx: {dynamic[0]:.2f} m/s, vy: {dynamic[1]:.2f} m/s, '
                              f'ax: {dynamic[2]:.2f} m/s², ay: {dynamic[3]:.2f} m/s²',
            historical_trajectory=None, solution=None)
        commands[command] += 1
    output.mkdir(parents=True, exist_ok=True)
    source = output/'current-anchor-metadata.json'
    if source.exists() and json.loads(source.read_text()) != metadata:
        raise ValueError('Current-anchor source changed; do not overwrite')
    if not source.exists():
        write(source, metadata)
    manifest = captions.initialize(index_file, source, library)
    report = dict(status='prepared', scene_count=len(metadata), checked_current_images=checked_images,
        train_test_token_overlap=0, source_split='navtest', navigation_counts=dict(commands),
        native_targets_available=False, future_images_included=False,
        new_api_requests=0, automatic_monitoring=False)
    write(output/'preparation.json', report)
    return manifest, report


def run_partition(library, output, train_audit, private_config, credential_file, policy_file):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((Path(library)/'manifest.json').read_text())
    if manifest['split'] != 'navtest' or manifest['scene_count'] != 12146:
        raise ValueError('Only complete private navtest library is authorized')
    prerequisite(train_audit)
    # The explicit audited campaign profile is supplied in the adjacent contract file.
    contract = json.loads((output/'contract.json').read_text())
    train_profile_path = Path(contract['train_profile'])
    train_profile = json.loads(train_profile_path.read_text())
    if (train_profile['prompt'] != (captions.CONFIG/'cluster-caption-prompt.md').read_text()
            or any(train_profile['policy'][k] != json.loads(Path(policy_file).read_text())[k]
                   for k in train_profile['policy'] if k not in ('cost_cap_cny','max_requests'))):
        raise ValueError('Test prompt/model/image/throughput contract differs from train')
    write(output/'continuation-status.json', dict(status='running', automatic_monitoring=False,
        prerequisite_audit=str(train_audit), source_split='navtest'))
    campaign = output/'campaign'
    try:
        result = batch.run(library, campaign, private_config, credential_file,
                           workers=8, policy_file=policy_file)
        if result['status'] not in ('complete','complete_with_failures'):
            raise ValueError('Caption production stopped; no automatic retry')
        audit = repair.finalize(library, campaign, output/'final', expected_count=12146)
        write(output/'continuation-status.json', dict(status=audit['status'], counts=audit['counts'],
            estimated_cny=audit['cost']['estimated_cny'], automatic_monitoring=False))
        return audit
    except BaseException as exc:
        write(output/'continuation-status.json', dict(status='blocked',error_type=type(exc).__name__,
            automatic_monitoring=False))
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation',choices=('initialize','run'))
    for name in ('library','output','index','train-library','train-audit','private-config','credential-file','policy'):
        p.add_argument('--'+name,required=name in ('library','output'))
    a=p.parse_args()
    if a.operation=='initialize':
        _,result=initialize_navtest(a.index,a.train_library,a.library,a.output)
    else:
        result=run_partition(a.library,a.output,a.train_audit,a.private_config,a.credential_file,a.policy)
    print(json.dumps({k:result[k] for k in ('status','scene_count','counts') if k in result},ensure_ascii=False))


if __name__=='__main__':
    main()
