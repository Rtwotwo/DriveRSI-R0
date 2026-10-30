import json
import sqlite3
from pathlib import Path

from driversi.rsi.experiments.synthesis_cost_audit import audit


def test_reused_receipts_are_counted_once_and_cost_is_not_wall_time(tmp_path):
    generation = tmp_path / 'origin' / 'generation'
    report = generation / 'tasks' / 'generation-render' / 'render' / 'report.json'
    report.parent.mkdir(parents=True)
    report.write_text('{}')
    db = sqlite3.connect(generation / 'harness.sqlite')
    db.execute('CREATE TABLE harness_tasks (id TEXT, payload TEXT)')
    db.execute('INSERT INTO harness_tasks VALUES (?,?)', ('generation-render', json.dumps({
        'status': 'completed', 'elapsed_seconds': 15,
        'stages': [{'id': 'render', 'elapsed_seconds': 10}, {'id': 'source', 'elapsed_seconds': 4}]})))
    db.commit()
    db.close()
    root = tmp_path / 'campaign'
    rows = []
    for candidate in ['a', 'b']:
        p = root / 'candidates' / candidate / 'generation' / 'report.json'
        p.parent.mkdir(parents=True)
        p.write_text(json.dumps({'teacher': {'teacher_report': '/shared/teacher.json', 'wall_seconds': 20},
                                 'render_report': str(report)}))
        rows.append({'candidate': candidate, 'status': 'training_sample_ready'})
    (root / 'trajectory-family-summary.json').write_text(json.dumps({
        'rows': rows, 'qualified': 2, 'rendered': 2}))
    result = audit(root, tmp_path / 'audit.json')
    assert result['teacher']['count'] == 1
    assert result['render']['count'] == 1
    assert result['total_counted_service_seconds'] == 35
    assert 'phase_times_are_service_work_not_campaign_wall_time' in result['caveats']
