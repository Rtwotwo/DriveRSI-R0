"""Cover exact split-membership reads without scanning large scene JSON blobs."""
import argparse
import json
import sqlite3
import time
from pathlib import Path


def prepare(path, split):
    if split not in ('navtrain','navtest'):
        raise ValueError('Explicit catalog split required')
    path=Path(path).resolve(strict=True);started=time.monotonic()
    db=sqlite3.connect(path.as_uri()+'?mode=rw',uri=True,timeout=30)
    try:
        if db.execute("SELECT value FROM metadata WHERE key='split'").fetchone() != (split,):
            raise ValueError('Catalog split differs')
        before=db.execute('SELECT count(*) FROM scenes').fetchone()[0]
        with db:
            db.execute('CREATE INDEX IF NOT EXISTS scene_membership_lookup ON scenes(token,source_log)')
        plan=db.execute('EXPLAIN QUERY PLAN SELECT token,source_log FROM scenes').fetchall()
        if not any('COVERING INDEX scene_membership_lookup' in row[-1] for row in plan):
            raise ValueError('Covering membership index not selected')
        began=time.monotonic()
        rows=db.execute('SELECT token,source_log FROM scenes').fetchall()
        if len(rows)!=before or len({r[0] for r in rows})!=before:
            raise ValueError('Scene membership changed during index preparation')
        return dict(split=split,count=before,index='scene_membership_lookup',
            query_plan=plan,membership_read_seconds=time.monotonic()-began,
            total_seconds=time.monotonic()-started,logical_records_modified=False)
    finally:
        db.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--database',required=True)
    parser.add_argument('--split',required=True);parser.add_argument('--report',required=True)
    args=parser.parse_args();report=prepare(args.database,args.split)
    from driversi.runtime.budget import write_json
    write_json(Path(args.report),report);print(json.dumps(report))


if __name__=='__main__':main()
