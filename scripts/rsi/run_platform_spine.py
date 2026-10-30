"""Prepare or resume existing DriveHarness services; never launch103kimplicitly."""
import argparse
import json
from pathlib import Path
from driversi.rsi.experiments.interfaces import interfaces,write_pool
from driversi.rsi.experiments.continuation import run
from driversi.runtime.budget import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    i=sub.add_parser('inventory');i.add_argument('--output',required=True)
    c=sub.add_parser('pool');c.add_argument('--database',required=True);c.add_argument('--output',required=True);c.add_argument('--shards',type=int,default=8)
    c=sub.add_parser('continue');c.add_argument('--config',required=True);c.add_argument('--execute',action='store_true');c.add_argument('--execute-training',action='store_true')
    a=p.parse_args()
    if a.command=='inventory':
        value=interfaces();write_json(Path(a.output),value)
        print(json.dumps({'models':list(value['models']),'providers':len(value['providers']),'full_RSI_validated':False}))
    elif a.command=='pool':print(json.dumps(write_pool(a.database,a.output,shards=a.shards)))
    else:
        value=run(a.config,execute=a.execute,execute_training=a.execute_training)
        print(json.dumps({k:value.get(k) for k in ('status','reason','training_executed','promoted')}))
        # Prepared/waiting are pending work, not scientific success. Keep failure
        # exit codes visible to batch callers, as the underlying Harness does.
        return 0 if value['status'] in {
            'teacher_assignment_prepared','waiting_visual_review','admitted_dataset_available',
            'render_prepared','render_registered','render_running',
            'export_prepared','export_registered','export_running',
            'paired_prepared','paired_registered','paired_running','paired_completed'} else 2
    return 0


if __name__=='__main__':raise SystemExit(main())
