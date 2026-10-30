"""Bounded native teacher expansion; no student scoring or training launch."""
import argparse
import json
from pathlib import Path
from driversi.rsi.experiments.branch_expansion import prepare, run, attach_admissions

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    q = sub.add_parser('prepare')
    q.add_argument('--teacher-template', type=Path, required=True)
    q.add_argument('--render-template', type=Path, required=True)
    q.add_argument('--output', type=Path, required=True)
    q.add_argument('--existing-rollout', action='append', type=Path, default=[])
    q.add_argument('--target', type=int, default=20)
    q.add_argument('--max-renders', type=int, default=4)
    q.add_argument('--workers', type=int, default=2)
    q = sub.add_parser('run'); q.add_argument('--output', type=Path, required=True)
    q = sub.add_parser('admit'); q.add_argument('--output', type=Path, required=True)
    q.add_argument('--sample-report', action='append', type=Path, required=True)
    a = p.parse_args()
    if a.command == 'prepare':
        result = prepare(a.teacher_template, a.render_template, a.output, target=a.target,
                         max_renders=a.max_renders, workers=a.workers,
                         existing_rollouts=a.existing_rollout)
    elif a.command == 'admit':
        result = attach_admissions(a.output, a.sample_report)
    else:
        result = run(a.output)
    print(json.dumps({k:v for k,v in result.items()
                      if k not in ('rows','files','asset_files','teacher_template','render_template')},
                     ensure_ascii=False, indent=2))
