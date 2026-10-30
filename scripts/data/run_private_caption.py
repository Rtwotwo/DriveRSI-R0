"""Prepare or run bounded private-network VLM comparisons for whitebox bundles."""
import argparse
import json
from pathlib import Path

from driversi.data.catalog.bailian_caption import (
    credential, prepare, private_endpoint, private_config, run, write)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['prepare', 'run'])
    p.add_argument('--manifest', required=True, type=Path)
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--env-file', type=Path)
    args = p.parse_args()
    config = private_config(args.config)
    manifest = json.loads(args.manifest.read_text())
    if set(manifest) != {'database', 'bundles'}:
        raise ValueError('Use database and existing source bundles')
    jobs = prepare(manifest['database'], manifest['bundles'], config, args.output)
    if args.command == 'run':
        key = credential(args.env_file)
        addresses = private_endpoint(config['base_url'])
        write(args.output / 'transport.json', {'transport': 'private_https',
              'private_resolution_verified': bool(addresses), 'public_fallback': False, 'proxy': False})
        report = run(jobs, config, args.output, key=key)
        print(json.dumps({k: v for k, v in report.items() if k != 'rows'}))
    else:
        print((args.output / 'preparation.json').read_text())


if __name__ == '__main__':
    main()
