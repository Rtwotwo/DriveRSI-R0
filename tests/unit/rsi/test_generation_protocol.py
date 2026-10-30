import json
import pytest
from driversi.rsi.experiments.generation_protocol import verify_settled
from driversi.agents.backends.native import digest


def prepared(tmp_path):
    wave=tmp_path/'waves/wave-0001';wave.mkdir(parents=True)
    (wave/'production-status.json').write_text(json.dumps(dict(status='failed',error_type='FileNotFoundError')))
    (wave/'trajectory-family-plan.json').write_text(json.dumps(dict(assignments=[dict(id='a',token='t',source_log='s')])))
    (wave/'physical-prefetch-status.json').write_text(json.dumps(dict(status='settled_after_error',
        submitted=1,completed=1,consumed=1,errors={},unconsumed=[],jobs={'a':dict(done=True,consumed=True,output=str(wave/'candidates/a/generation'))})))
    gen=wave/'candidates/a/generation';gen.mkdir(parents=True)
    (gen/'report.json').write_text(json.dumps(dict(status='trajectory_ready',token='t',source_log='s',teacher=dict(assignment_id='a',plan_sha256=digest(wave/'trajectory-family-plan.json'),status='trajectory_admitted'))))
    return wave,gen


def test_pre_render_io_failure_is_reconciled_without_replay(tmp_path):
    prepared(tmp_path)
    assert len(verify_settled(tmp_path,999999999))==4


def test_reserved_gpu_work_cannot_be_restarted_by_io_migration(tmp_path):
    wave,gen=prepared(tmp_path)
    (gen/'render-reservation.json').write_text('{}')
    with pytest.raises(ValueError,match='reserved'):verify_settled(tmp_path,999999999)


def test_missing_physical_receipt_remains_unknown(tmp_path):
    wave,gen=prepared(tmp_path)
    (gen/'report.json').unlink()
    with pytest.raises(ValueError,match='durably'):verify_settled(tmp_path,999999999)


def test_extra_started_task_is_not_silently_dropped(tmp_path):
    wave,gen=prepared(tmp_path)
    (wave/'candidates/unknown/generation').mkdir(parents=True)
    with pytest.raises(ValueError,match='Unaccounted'):verify_settled(tmp_path,999999999)


def test_revision_name_does_not_authorize_overwriting_advanced_state(tmp_path):
    import copy
    from driversi.rsi.experiments.generation_protocol import migrate,REVISION
    state=dict(active='p1',policy=dict(attempts_per_source=32,max_rounds=3),
        cases=[dict(id='p1',status='unknown_execution',asset_revision=2,qualified=2)])
    folder=tmp_path/'protocol-revisions'/REVISION;folder.mkdir(parents=True)
    (folder/'before.json').write_text(json.dumps(dict(state=state,evidence={})))
    advanced=copy.deepcopy(state);advanced.update(active='p2',protocol_revision=REVISION,training_started=True)
    private=tmp_path/'private';private.mkdir()
    (private/'state.json').write_text(json.dumps(advanced))
    request=tmp_path/'request.json';request.write_text(json.dumps(dict(revision=REVISION,
        source_psnr_strict_gt=25.,spatial_path_ade_m=.5,operator_authorized=True)))
    with pytest.raises(ValueError,match='Concurrent curriculum'):migrate(tmp_path,request)
    assert json.loads((private/'state.json').read_text())==advanced
