"""Draft reuse must not mutate input evidence or bypass current validation."""
import json
from pathlib import Path
import pytest

from driversi.data.catalog import scene_label_jobs as jobs


def test_changed_contract_rejected_before_preparation_or_old_artifact_writes(tmp_path,monkeypatch):
    bundle=tmp_path/'bundle';bundle.mkdir()
    (bundle/'source.json').write_text(json.dumps(dict(dimension_sampling=jobs.DIMENSION_SAMPLING)))
    out=tmp_path/'run';out.mkdir()
    job=dict(source_sha256=jobs.sha256(bundle/'source.json'),bundle=str(bundle/'source.json'),
        prompt='old prompt',facts={},image_max_side=960,jpeg_quality=85)
    (out/'job.json').write_text(json.dumps(job))
    (out/'execution-plan.json').write_text(json.dumps(dict(model='model')))
    (out/'sample.json').write_text('original draft')
    (out/'context.json').write_text('original context')
    before={p.name:p.read_bytes() for p in out.iterdir()}
    monkeypatch.setattr(jobs,'audit_source_motion',lambda s:dict(passed=True))
    monkeypatch.setattr(jobs.labels,'measurements',lambda s:dict(actor_tracks=[]))
    def forbidden(*a,**kw):raise AssertionError('Preparation must not run on a changed contract')
    monkeypatch.setattr(jobs.client,'prepare',forbidden)
    with pytest.raises(ValueError,match='Different scene/contract'):
        jobs.prepare('unused',bundle,dict(image_max_side=960,jpeg_quality=85,models=['model']),out)
    assert {p.name:p.read_bytes() for p in out.iterdir()}==before


def test_reused_raw_result_is_checked_again_by_current_decoder(tmp_path,monkeypatch):
    out=tmp_path
    receipt=out/'cached.json';receipt.write_text(json.dumps(dict(raw='old malformed content')))
    monkeypatch.setattr(jobs,'pilot_config',lambda p:dict(models=['model']))
    monkeypatch.setattr(jobs,'prepare',lambda *a:dict())
    monkeypatch.setattr(jobs.client,'credential',lambda p:'not-a-real-key')
    monkeypatch.setattr(jobs.client,'run',lambda *a,**kw:dict(complete=True,rows=[dict(file=str(receipt))]))
    def reject(*a,**kw):raise ValueError('Current decoder rejected prior raw output')
    monkeypatch.setattr(jobs,'decode',reject)
    from types import SimpleNamespace
    args=SimpleNamespace(private_config='unused',database='unused',bundle='unused',output=out,execute=True,credential_file='unused')
    with pytest.raises(ValueError,match='Current decoder rejected'):
        jobs.execute(args)
    assert json.loads((out/'sample.json').read_text())['review']['format_valid'] is False
    assert json.loads((out/'publication.json').read_text())['status']=='current_validation_rejected'


def test_rejected_revalidation_preserves_old_draft_but_never_leaves_it_current(tmp_path,monkeypatch):
    original=json.dumps(dict(review=dict(format_valid=True),labels=['stale']))
    (tmp_path/'sample.json').write_text(original)
    def reject(*a,**kw):raise ValueError('rejected')
    monkeypatch.setattr(jobs,'decode',reject)
    with pytest.raises(ValueError):jobs.publish({},'model',dict(raw='old'),tmp_path)
    current=json.loads((tmp_path/'sample.json').read_text())
    assert 'labels' not in current and current['review']['format_valid'] is False
    assert (tmp_path/current['review']['previous_sample']).read_text()==original
