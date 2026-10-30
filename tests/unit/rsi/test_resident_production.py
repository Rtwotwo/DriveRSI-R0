import json
from pathlib import Path
import pytest
from driversi.agents.backends.native import digest
from driversi.rsi.experiments.resident_production import recover_completed


def save(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def test_partial_two_job_writeback_recovers_original_without_launch(tmp_path):
    batch=tmp_path/'batch';rows=[];assignments=[]
    reservation=batch/'reservation.json';jobs=[]
    config=save(batch/'config.json',{'fixed':True})
    for i in range(2):
        d=tmp_path/'candidates'/str(i)
        assignments.append(dict(id=str(i),anchor_assignment_id='anchor',sampling_seed=i))
        generation=save(d/'generation/report.json',dict(status='trajectory_ready'))
        diversity=save(d/'diversity.json',dict(description=dict(files={},behavior_types=['test']),decision=dict(distinct=True)))
        jobs.append(dict(generation=str(generation),sha256=digest(generation),diversity=str(diversity),diversity_sha256=digest(diversity)))
        rows.append(dict(prepared_generation=str(d/'generation/report.json'),outcome=dict(status='visual_preflight_rejected',trainable=False)))
    save(reservation,dict(config_sha256=digest(config),jobs=jobs))
    for i in range(2):save(tmp_path/f'candidates/{i}/generation/render-reservation.json',dict(batch=str(reservation),sha256=digest(reservation)))
    manifest=save(batch/'manifest.json',dict(config=str(config),reservation=str(reservation),reservation_sha256=digest(reservation)))
    report=save(batch/'execution/report.json',dict(rows=rows))
    save(batch/'completion.json',dict(report_sha256=digest(report),manifest_sha256=digest(manifest)))
    # Crash after the first candidate has been settled, before the second.
    save(tmp_path/'candidates/0/outcome.json',dict(previous=True))
    save(tmp_path/'candidates/0/outcome-seal.json',dict(previous=True))
    recover_completed(tmp_path,dict(assignments=assignments))
    second=tmp_path/'candidates/1/outcome.json'
    assert json.loads(second.read_text())['status']=='visual_preflight_rejected'
    assert json.loads((second.parent/'outcome-seal.json').read_text())['sha256']==digest(second)
    assert json.loads((tmp_path/'candidates/0/outcome.json').read_text())==dict(previous=True)
    before=second.read_bytes();recover_completed(tmp_path,dict(assignments=assignments))
    assert second.read_bytes()==before


def test_unsettled_reservation_does_not_regroup_or_repeat(tmp_path):
    d=tmp_path/'candidates/a';reservation=save(tmp_path/'batch/reservation.json',{})
    save(d/'generation/render-reservation.json',dict(batch=str(reservation),sha256=digest(reservation)))
    with pytest.raises(RuntimeError,match='no regrouping'):
        recover_completed(tmp_path,dict(assignments=[dict(id='a')]))
    assert not (d/'outcome.json').exists()


@pytest.mark.parametrize('extra_pid',[False,True])
def test_only_exact_owned_feature_process_can_reuse_busy_gpu(tmp_path,monkeypatch,extra_pid):
    from types import SimpleNamespace
    from driversi.rsi.experiments import continual_execution as auto
    from driversi.data.catalog import feature_worker
    config=save(tmp_path/'generation-config.json',dict(generated_catalog=str(tmp_path/'features.sqlite'),resident_postprocessing={'gpus':[4,5],'feature_gpu':0}))
    descriptor=save(tmp_path/'features.sqlite.feature-worker.json',dict(directory=str(tmp_path/'service'),gpu=0))
    save(tmp_path/'service/active.json',dict(pid=123,birth='same'))
    command=['python',str(Path(feature_worker.__file__).resolve()),'--config',str(descriptor)]
    save(tmp_path/'service/launch.json',dict(pid=123,config_sha256=digest(descriptor),command=command))
    monkeypatch.setattr(feature_worker,'process_birth',lambda pid:'same')
    original_exists=Path.exists;original_read=Path.read_bytes
    monkeypatch.setattr(Path,'exists',lambda p:True if str(p)=='/proc/123/cmdline' else original_exists(p))
    monkeypatch.setattr(Path,'read_bytes',lambda p:'\0'.join(command).encode() if str(p)=='/proc/123/cmdline' else original_read(p))
    def inventory(args,**kwargs):
        text='0, GPU-0\n4, GPU-4\n5, GPU-5\n' if '--query-gpu=index,uuid' in args else 'GPU-0, 123\n'+('GPU-0, 456\n' if extra_pid else '')
        return SimpleNamespace(returncode=0,stdout=text)
    monkeypatch.setattr(auto.subprocess,'run',inventory)
    assert auto.owned_feature_gpu(dict(current_round=str(tmp_path)))==(None if extra_pid else 0)


def test_changed_diversity_rejected_before_recovering_outcome(tmp_path):
    test_partial_two_job_writeback_recovers_original_without_launch(tmp_path)
    directory=tmp_path/'candidates/1'
    (directory/'outcome.json').unlink();(directory/'outcome-seal.json').unlink()
    save(directory/'diversity.json',dict(changed=True))
    with pytest.raises(ValueError,match='diversity changed'):
        recover_completed(tmp_path,dict(assignments=[dict(id='0'),dict(id='1')]))
    assert not (directory/'outcome.json').exists()


def test_first_production_batch_creates_owned_directory_and_reuses_completion(tmp_path,monkeypatch):
    import time
    from driversi.rsi.experiments import resident_batch, resident_production
    cfg=save(tmp_path/'config.json',dict(visual_quality_profile='experimental_numeric27_strict_ade05_v1',resident_postprocessing={'gpus':[4,5],'feature_gpu':0},generated_catalog=str(tmp_path/'features.sqlite')))
    d=tmp_path/'wave/candidates/a'
    generation=save(d/'generation/report.json',{'status':'trajectory_ready'})
    save(d/'diversity.json',{'distinct':True})
    calls=[]
    def run(manifest):
        calls.append(manifest);spec=json.loads(Path(manifest).read_text())
        assert spec['gpus']==[4]
        assert Path(spec['reservation']).is_file()
        result={'status':'completed','rows':[{'prepared_generation':str(generation),'outcome':{'status':'training_sample_ready','files':{}}}]}
        save(Path(spec['output'])/'report.json',result)
        return result
    monkeypatch.setattr(resident_batch,'run',run)
    pending=[({'id':'a'},d,{})]
    first=resident_production.dispatch(cfg,tmp_path/'wave',pending,time.time()+120)
    assert first==resident_production.dispatch(cfg,tmp_path/'wave',pending,time.time()+120)
    assert len(calls)==1
