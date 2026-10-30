"""Observed generation costs, with acceptance and duplicate rates kept separate."""

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np

from driversi.data.catalog.scene_hierarchy import digest, save_json


def describe(values):
    return dict(count=len(values), total_seconds=sum(values),
                median_seconds=float(np.median(values)) if values else None,
                p10_seconds=float(np.quantile(values, .1)) if values else None,
                p90_seconds=float(np.quantile(values, .9)) if values else None)


def audit(campaign, output):
    root = Path(campaign).resolve()
    path = root / 'trajectory-family-summary.json'
    family = json.loads(path.read_text())
    teacher, render, source, total_render = {}, {}, {}, {}
    counts = Counter(row['status'] for row in family['rows'])
    rows = []
    for item in family['rows']:
        report = root / 'candidates' / item['candidate'] / 'generation/report.json'
        if not report.exists():
            continue
        generation = json.loads(report.read_text())
        t = generation.get('teacher', {})
        key = t.get('teacher_report', str(report))
        if t.get('wall_seconds') is not None:
            teacher[key] = t['wall_seconds']
        if not generation.get('render_report'):
            continue
        # Reused rendering must be counted once at its actual origin.
        original = Path(generation['render_report']).parents[3]
        database = original / 'harness.sqlite'
        if not database.exists():
            rows.append(dict(report=str(report), timing_missing=True))
            continue
        db = sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)
        try:
            result = db.execute("SELECT payload FROM harness_tasks WHERE id='generation-render'").fetchone()
        finally:
            db.close()
        if not result:
            continue
        task = json.loads(result[0])
        times = {s['id']: s.get('elapsed_seconds') for s in task['stages']}
        key = str(database)
        if times.get('render') is not None:
            render[key] = times['render']
        if times.get('source') is not None:
            source[key] = times['source']
        if task.get('elapsed_seconds') is not None:
            total_render[key] = task['elapsed_seconds']
        rows.append(dict(candidate=item['candidate'], database=key, times=times,
                         status=task['status'], report=str(report)))
    total_measured = sum(teacher.values()) + sum(total_render.values())
    summary = dict(source=str(path), source_sha256=digest(path), attempts=len(family['rows']),
        status_counts=dict(counts), qualified=family['qualified'],
        registered_render_attempts=family['rendered'],
        teacher=describe(list(teacher.values())), render=describe(list(render.values())),
        source_reconstruction=describe(list(source.values())),
        render_task=describe(list(total_render.values())),
        total_counted_service_seconds=total_measured,
        counted_service_seconds_per_qualified=total_measured / family['qualified'] if family['qualified'] else None,
        caveats=['phase_times_are_service_work_not_campaign_wall_time',
                 'some_teacher_and_render_receipts_reused_from_prior_campaign',
                 'excludes_asset_reconstruction_queue_export_and_training',
                 'four_qualified_samples_from_experimental_24db_profile_not_25db',
                 'small_biased_batch_cannot_guarantee_twenty_distinct_samples_per_anchor'],
        rows=rows)
    save_json(output, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.campaign, args.output)
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    main()
