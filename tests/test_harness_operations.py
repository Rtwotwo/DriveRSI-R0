import json
import time
from pathlib import Path

import pytest

from driversi.agents.backends.native import digest
from driversi.rsi.experiments import storage_recovery as recovery
from driversi.rsi.decisions.training_recipe import choose


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def failed_batch(tmp_path):
    folder=tmp_path/'batch';out=folder/'execution'
    config=put(tmp_path/'config.json',{'protocol':'unchanged'})
    generation=put(tmp_path/'generation.json',{'status':'trajectory_ready'})
    diversity=put(tmp_path/'diversity.json',{'distinct':True})
    job=dict(generation=str(generation),sha256=digest(generation),
        diversity=str(diversity),diversity_sha256=digest(diversity))
    reservation=put(folder/'reservation.json',dict(jobs=[job],config_sha256=digest(config)))
    manifest=put(folder/'manifest.json',dict(jobs=[job],gpus=[4],config=str(config),
        output=str(out),reservation=str(reservation),reservation_sha256=digest(reservation),max_seconds=120))
    put(out/'request.json',dict(manifest_sha256=digest(manifest)))
    put(out/'status.json',dict(status='failed'))
    put(out/'final-service-settlement.json',dict(rows=[],errors=[]))
    put(out/'launch.json',dict(started=time.time()-2))
    lane=out/'lane-0';lane.mkdir()
    (lane/'failure.txt').write_text('image_store.py with tempfile.TemporaryDirectory driversi-png-abc OSError: Directory not empty')
    return folder


def completed_run(manifest):
    spec=json.loads(Path(manifest).read_text());out=Path(spec['output'])
    result=dict(status='completed',rows=[dict(prepared_generation=j['generation'],outcome={'files':{}})
        for j in spec['jobs']],wall_seconds=1,production_increment=1)
    put(out/'request.json',dict(manifest_sha256=digest(manifest)))
    put(out/'report.json',result);put(out/'status.json',dict(status='completed'))
    put(out/'final-service-settlement.json',dict(rows=[],errors=[]))
    return result


def test_known_recovery_and_idempotent(tmp_path,monkeypatch):
    folder=failed_batch(tmp_path);calls=[]
    from driversi.rsi.experiments import resident_batch
    def run(m):calls.append(m);return completed_run(m)
    monkeypatch.setattr(resident_batch,'run',run)
    recovery.recover_batch(folder)
    recovery.recover_batch(folder)
    assert len(calls)==1
    manifest,report=recovery.result_paths(folder)
    assert report.exists() and 'storage-recovery-01' in str(manifest)
    assert (folder/'execution/lane-0/failure.txt').exists()
    attempt=json.loads((folder/'storage-recovery-01/attempt.json').read_text())
    assert attempt['max_seconds']<120 and not attempt['physical_reexecution']


def test_completed_without_seal_is_recovered_without_rerun(tmp_path,monkeypatch):
    folder=failed_batch(tmp_path)
    completed_run(folder/'manifest.json')
    from driversi.rsi.experiments import resident_batch
    monkeypatch.setattr(resident_batch,'run',lambda _:pytest.fail('must not rerender'))
    recovery.recover_batch(folder)
    assert (folder/'completion.json').exists()


@pytest.mark.parametrize('known,expected_calls', [(True,2),(False,1)])
def test_second_attempt_is_automatic_only_for_settled_known_fault(tmp_path,monkeypatch,known,expected_calls):
    folder=failed_batch(tmp_path);calls=[]
    from driversi.rsi.experiments import resident_batch
    def run(manifest):
        calls.append(manifest)
        if len(calls)>1:return completed_run(manifest)
        spec=json.loads(Path(manifest).read_text());out=Path(spec['output'])
        put(out/'request.json',dict(manifest_sha256=digest(manifest)))
        put(out/'status.json',dict(status='failed'))
        put(out/'final-service-settlement.json',dict(rows=[],errors=[]))
        put(out/'launch.json',dict(started=time.time()-2))
        lane=out/'lane-0';lane.mkdir()
        (lane/'failure.txt').write_text('image_store.py TemporaryDirectory driversi-png- Directory not empty'
                                      if known else 'unknown renderer failure')
        raise RuntimeError('terminal worker failed')
    monkeypatch.setattr(resident_batch,'run',run)
    if known:recovery.recover_batch(folder)
    else:
        with pytest.raises(ValueError,match='Unregistered'):recovery.recover_batch(folder)
    assert len(calls)==expected_calls
    assert (folder/'completion.json').exists()==known


@pytest.mark.parametrize('mutation',['unknown','admitted','unsettled','changed'])
def test_unsafe_failures_do_not_retry(tmp_path,mutation):
    folder=failed_batch(tmp_path);out=folder/'execution'
    if mutation=='unknown':(out/'lane-0/failure.txt').write_text('unknown exception')
    if mutation=='admitted':(out/'lane-0/branch-000/admission').mkdir(parents=True)
    if mutation=='unsettled':put(out/'final-service-settlement.json',dict(rows=[],errors=['live']))
    if mutation=='changed':put(tmp_path/'generation.json',dict(changed=True))
    with pytest.raises(ValueError):recovery.recover_batch(folder)
    assert not list(folder.glob('storage-recovery-*'))


def test_no_refresh_of_batch_budget(tmp_path):
    folder=failed_batch(tmp_path)
    put(folder/'execution/launch.json',dict(started=time.time()-200))
    with pytest.raises(ValueError,match='budget exhausted'):recovery.recover_batch(folder)


def test_recipe_uses_actual_population_and_bounded_exposures():
    a=choose(30,0,[6]);b=choose(30,1,[6])
    assert a['epochs']==4 and a['optimizer_updates_per_arm']==60
    assert a['devices']==[6] and a['global_batch']==2
    assert b['epochs']==6 and b['optimizer_updates_per_arm']==90
    assert not a['learned_optimal']
    with pytest.raises(ValueError):choose(1000,0,[6])


def test_png_publication_does_not_need_temporary_directory(tmp_path,monkeypatch):
    import tempfile
    import numpy as np
    from PIL import Image
    from driversi.rendering.gaussian.image_store import PngWriter
    monkeypatch.setattr(tempfile,'TemporaryDirectory',lambda *a,**k:pytest.fail('NFS temp directory'))
    pixels=np.arange(768,dtype=np.uint8).reshape(16,16,3)
    out=tmp_path/'frame.png';writer=PngWriter();writer(pixels,out)
    assert np.array_equal(np.asarray(Image.open(out)),pixels)
    with pytest.raises(ValueError):writer(pixels,out)
    assert len(writer.rows)==1


def test_public_lineage_does_not_publish_private_details():
    from driversi.data.datasets.curriculum_lineage import public
    value=dict(generated_at=1,active_problem='problem-001',total_preserved=1,preserved_prior=0,
        split_counts={'synthetic':1},sources=[],training=[],recoveries=[dict(rule='known',problem='problem-001',
        status='recovered',started=1,batches=['secret-path'],test_token='secret-test')])
    text=json.dumps(public(value))
    assert 'secret-path' not in text and 'secret-test' not in text


def test_prior_pool_is_capped_per_source(tmp_path):
    from driversi.rsi.experiments.problem_cohort import seed
    evidence=put(tmp_path/'outcome.json',{})
    rows=[dict(source_token=t,original_outcome=str(evidence),original_outcome_sha256=digest(evidence))
        for t,n in [('A',11),('B',3),('C',3)] for i in range(n)]
    path=put(tmp_path/'pool.json',dict(rows=rows))
    config=dict(prior_cohort=dict(path=str(path),sha256=digest(path)),trajectory_family=dict(target_per_anchor=10))
    plan=dict(assignments=[dict(token=t,anchor_assignment_id=t) for t in 'ABC'])
    selected=seed(config,plan)
    assert len(selected)==16 and sum(x['source_token']=='A' for x in selected)==10


def test_old_success_does_not_change_new_rejection_diagnosis():
    from driversi.rsi.experiments.adaptive_generation import source_actions
    old=dict(candidate='old-v999',status='training_sample_ready',trainable=True,reused_prior_cohort=True)
    current=dict(candidate='new',status='teacher_rejected',trainable=False)
    actions=source_actions(dict(rows=[old,current]),[dict(id='new',token='A')],0)
    assert actions=={'A':'reduce_history_perturbation'}


def test_recipe_replays_native_step_rounding():
    from driversi.rsi.decisions.problem_learning import sft_updates
    r=choose(31,0,[6]);updates=sft_updates(31,4,2)
    assert r['optimizer_updates_per_arm']==updates+(updates%2)
