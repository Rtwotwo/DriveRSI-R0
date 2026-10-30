#!/usr/bin/env python3
"""Bounded real-source appearance pilot; no test source, no training or publishing."""
from pathlib import Path
import argparse
import collections
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from driversi.data.catalog import bailian_caption as client
from driversi.data.catalog import geometry_caption as caption
from driversi.data.catalog.store import Catalog
from driversi.data.sources.navsim import load_log
from driversi.rendering.whitebox.conditioning import build_source, CAMERAS
from driversi.rendering.whitebox.motion_audit import audit_source_motion
from driversi.rendering.whitebox.exporter import sha256

CITIES = ('us-ma-boston', 'us-nv-las-vegas-strip', 'us-pa-pittsburgh-hazelwood', 'sg-one-north')
QUOTAS = (16, 18, 16, 14)


def order(x):
    return hashlib.sha256(('caption64-v1:'+x).encode()).hexdigest()


def select(args):
    out = Path(args.output)
    if (out/'selection.json').exists():
        return json.loads((out/'selection.json').read_text())
    index = Path(args.index)
    groups = collections.defaultdict(list)
    for line in index.open():
        row = json.loads(line)
        if row['split'] != 'navtrain':
            raise ValueError('Only navtrain work index allowed')
        groups[row['source_group']].append(row)
    by_city = collections.defaultdict(list)
    hints = []
    logs_root = Path(args.logs_root)
    for group, rows in groups.items():
        with (logs_root/(rows[0]['source_log']+'.pkl')).open('rb') as f:
            head = f.read(1048576)
        cities = [city for city in CITIES if city.encode() in head]
        hints.append({'source_group': group, 'candidate_cities': cities})
        if len(cities) == 1:
            by_city[cities[0]].append(group)
    client.write(out/'city-hints.json', hints)
    cat = Catalog(args.database, 'navtrain', readonly=True)
    chosen, failures = [], []
    try:
        for city, quota in zip(CITIES, QUOTAS):
            count = 0
            for group in sorted(by_city[city], key=order):
                if count == quota:
                    break
                for row in sorted(groups[group], key=lambda x: order(x['token']))[:8]:
                    record = cat.scene(row['token'])['record']
                    target = out/'bundles'/row['token']
                    try:
                        if (target/'audit.json').exists():
                            bundle = json.loads((target/'source.json').read_text())
                            audit = json.loads((target/'audit.json').read_text())
                            if sha256(record['log_path']) != bundle['source_log_sha256']:
                                raise ValueError('Resumed source changed')
                        else:
                            raw = load_log(Path(record['log_path']), trusted=True)
                            bundle = build_source(record, frames=raw)
                            if bundle['map_name'] != city:
                                raise ValueError('Header city disagrees with decoded scene')
                            a = record['anchor_index']
                            for j in sorted({a-4,a-3,a-2,a-1,a,a+5,a+10}):
                                for cam in CAMERAS:
                                    if not (Path(record['sensor_root'])/raw[j]['cams'][cam]['data_path']).is_file():
                                        raise ValueError('Missing photographed evidence')
                            audit = audit_source_motion(bundle, raw)
                            if not audit['passed']:
                                raise ValueError('Source-motion audit failed')
                            bundle['map_geometry_status'] = 'not_loaded_for_caption_pilot; no lane topology claims'
                            client.write(target/'source.json', bundle)
                            client.write(target/'audit.json', audit)
                        if not audit['passed']:
                            raise ValueError('Prior audit rejected')
                        chosen.append({'token': row['token'], 'source_group': group,
                            'source_log': record['source_log'], 'city': city, 'bundle': str(target.resolve()),
                            'source_sha256': sha256(target/'source.json'), 'descriptor': cat.scene(row['token'])['descriptor']})
                        count += 1
                        print(json.dumps({'selected': len(chosen), 'city': city, 'token': row['token']}), flush=True)
                        client.write(out/'selection-progress.json', {'rows': chosen, 'failures': failures})
                        break
                    except (ValueError, IndexError, FileNotFoundError) as exc:
                        failures.append({'token': row['token'], 'reason': type(exc).__name__+': '+str(exc)[:300]})
                        client.write(out/'preflight-failures.json',failures)
            if count != quota:
                raise ValueError(f'Insufficient preflight-valid source groups in {city}: {count}')
    finally:
        cat.close()
    result = {'version': 1, 'source_split': 'navtrain', 'index_sha256': sha256(index),
        'selection': 'fixed hash order; Boston16,LasVegas18,Pittsburgh16,Singapore14 (14 source groups available); one per original source group; source/evidence preflight only',
        'scene_count': len(chosen), 'rows': chosen, 'preflight_failures': failures,
        'frozen_before_main_batch_calls': True,
        'spine_calls_before_final_selection': len(list((out/'pilot'/'attempts').glob('*.json'))),
        'trainable': False}
    client.write(out/'selection.json', result)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['select', 'prepare', 'run'])
    p.add_argument('--output', required=True)
    p.add_argument('--database')
    p.add_argument('--index')
    p.add_argument('--logs-root')
    p.add_argument('--private-config')
    p.add_argument('--credential-file')
    p.add_argument('--first-scenes', type=int)
    args = p.parse_args()
    out = Path(args.output)
    if args.command == 'select':
        select(args)
    elif args.command == 'prepare':
        selected = json.loads((out/'selection.json').read_text())
        config = client.private_config(args.private_config)
        jobs = caption.prepare(args.database, [r['bundle'] for r in selected['rows']], config, out/'pilot-v3_1')
        client.write(out/'jobs.json', jobs)
        print(json.dumps({'prepared_scenes':len(selected['rows']), 'jobs': len(jobs)}))
    else:
        config = client.private_config(args.private_config)
        jobs = json.loads((out/'jobs.json').read_text())
        if args.first_scenes:
            tokens = [r['token'] for r in json.loads((out/'selection.json').read_text())['rows'][:args.first_scenes]]
            jobs = [j for j in jobs if j['token'] in tokens]
        result = client.run(jobs, config, out/'pilot-v3_1', key=client.credential(args.credential_file),
                            decode=caption.result_from_response)
        print(json.dumps({k:v for k,v in result.items() if k != 'rows'}))


if __name__ == '__main__':
    main()
