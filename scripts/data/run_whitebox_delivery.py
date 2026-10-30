"""Bounded, resumable NAVSIM whitebox delivery; full planning uses export_semantic_conditions.py."""
import argparse
import concurrent.futures
import json
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--batch-file', type=Path, help='One bounded plan shard; otherwise config.items')
    p.add_argument('--workers', type=int, default=1)
    args = p.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    if config.get('schema_version') != 1:
        raise ValueError('Expected schema_version=1')
    if config.get('prompt_profile', 'generic_four_camera_v1') not in ('generic_four_camera_v1', 'minimax_h3_ref2va_v1'):
        raise ValueError('Unknown prompt profile')
    for name in ('database', 'maps_root', 'output', 'reviews_root'):
        if not Path(config[name]).is_absolute():
            raise ValueError('Use an absolute path for ' + name)
    # The validated server keeps QuickJS/ffmpeg in a dedicated rendering environment.
    if config.get('renderer_site_packages'):
        site = Path(config['renderer_site_packages'])
        if not site.is_absolute() or not site.is_dir():
            raise ValueError('Renderer environment missing: ' + str(site))
        sys.path.insert(0, str(site))
    rows = json.loads(args.batch_file.read_text()) if args.batch_file else config['items']
    if not 1 <= len(rows) <= 128 or len({r['token'] for r in rows}) != len(rows):
        raise ValueError('Require 1..128 unique tokens in one shard')
    if not 1 <= args.workers <= 4:
        raise ValueError('Require 1..4 CPU workers per bounded job')
    if any(len(r['token']) != 16 or any(c not in '0123456789abcdef' for c in r['token']) for r in rows):
        raise ValueError('NAVSIM token must be 16 lower-case hexadecimal characters')
    for row in rows:
        for name in ('bundle', 'review'):
            if name in row and not Path(row[name]).is_absolute():
                raise ValueError('Per-item ' + name + ' must be an absolute path')
    from driversi.rendering.whitebox.delivery import atomic_json, run_scene
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=True)
    label = args.batch_file.stem if args.batch_file else 'configured-samples'
    report = dict(schema_version=1, complete=False, planned=len(rows), rows=[], new_api_calls=0)
    report_path = output / ('report-' + label + '.json')
    # Individual receipts are atomic; different shards can share one output root.
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_scene, config, row) for row in rows]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            report['rows'].append(result)
            atomic_json(report_path, report)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    report['complete'] = True
    report['successful'] = sum(r['status'] in ('completed', 'reused') for r in report['rows'])
    report['all_delivered'] = report['successful'] == len(rows)
    atomic_json(report_path, report)
    return 0 if report['all_delivered'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
