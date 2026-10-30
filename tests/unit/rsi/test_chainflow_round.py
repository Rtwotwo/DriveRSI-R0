import json

import pytest

from driversi.rsi.experiments import chainflow_round as module


def config(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "project_root", lambda: tmp_path)
    for name in ("base.json", "sample.json", "model.json"):
        (tmp_path / name).write_text("{}")
    value = dict(
        base_manifest="base.json",
        sample_reports=["sample.json"],
        model_config="model.json",
        output="exp/round",
        updates_per_arm=8,
    )
    path = tmp_path / "config.json"
    path.write_text(json.dumps(value))
    return path


def test_rejected_data_never_registers_or_launches_training(tmp_path, monkeypatch):
    path = config(tmp_path, monkeypatch)

    def reject(*a, **kw):
        raise ValueError("visual admission rejected")

    monkeypatch.setattr(module, "compile_paired_task", reject)
    monkeypatch.setattr(
        module.engine, "register", lambda *a: pytest.fail("registered rejected data")
    )
    report = module.run(path, execute=True)
    assert report["status"] == "blocked" and report["training_executed"] is False
    assert (tmp_path / "exp/round/report.json").is_file()


def test_partial_harness_with_feedback_cannot_claim_complete_round(tmp_path, monkeypatch):
    path = config(tmp_path, monkeypatch)

    def compile_(*args, **kwargs):
        task = tmp_path / "task.json"
        task.write_text(json.dumps({"id": "pair"}))
        return task

    monkeypatch.setattr(module, "compile_paired_task", compile_)
    monkeypatch.setattr(module.engine, "register", lambda *a: None)

    def partial(*args):
        feedback = tmp_path / "exp/round/tasks/pair/paired_feedback/report.json"
        feedback.parent.mkdir(parents=True)
        feedback.write_text(
            json.dumps({"observed_pilot_benefit": True, "next_action": "replicate"})
        )
        return {"status": "failed", "failure_reason": "resume replay differs"}

    monkeypatch.setattr(module.engine, "run", partial)
    report = module.run(path, execute=True)
    assert report["status"] == "incomplete_execution"
    assert report["training_executed"] is None and report["promoted"] is False


def test_compile_only_never_runs_and_reused_output_refused(tmp_path, monkeypatch):
    path = config(tmp_path, monkeypatch)
    task = tmp_path / "task.json"
    task.write_text(json.dumps({"id": "pair"}))
    monkeypatch.setattr(module, "compile_paired_task", lambda *a, **kw: task)
    monkeypatch.setattr(
        module.engine, "register", lambda *a: pytest.fail("compile started training")
    )
    assert module.run(path)["status"] == "prepared"
    with pytest.raises(ValueError, match="new exp"):
        module.run(path)


@pytest.mark.parametrize(
    "failure", [ValueError("late validation"), FileNotFoundError("missing output")]
)
def test_execution_validation_error_preserves_unknown_partial_work(tmp_path, monkeypatch, failure):
    path = config(tmp_path, monkeypatch)
    task = tmp_path / "task.json"
    task.write_text(json.dumps({"id": "pair"}))
    monkeypatch.setattr(module, "compile_paired_task", lambda *a, **kw: task)
    monkeypatch.setattr(module.engine, "register", lambda *a: None)

    def started(*a):
        raise failure

    monkeypatch.setattr(module.engine, "run", started)
    result = module.run(path, execute=True)
    assert result["status"] == "execution_error" and result["training_executed"] is None


@pytest.mark.parametrize("different_pids,same_state", [(True, False), (False, True), (True, True)])
def test_completed_harness_requires_actual_resume_verification(
    tmp_path, monkeypatch, different_pids, same_state
):
    path = config(tmp_path, monkeypatch)
    task = tmp_path / "task.json"
    task.write_text(json.dumps({"id": "pair"}))
    monkeypatch.setattr(module, "compile_paired_task", lambda *a, **kw: task)
    monkeypatch.setattr(module.engine, "register", lambda *a: None)

    def completed(*a):
        root = tmp_path / "exp/round/tasks/pair"
        feedback = root / "paired_feedback/report.json"
        feedback.parent.mkdir(parents=True)
        feedback.write_text(
            json.dumps({"observed_pilot_benefit": True, "next_action": "replicate"})
        )
        resume = root / "treatment_feedback/feedback.json"
        resume.parent.mkdir()
        resume.write_text(
            json.dumps(
                {
                    "cross_process_resume": {
                        "different_pids": different_pids,
                        "same_compared_state": same_state,
                    }
                }
            )
        )
        return {"status": "completed"}

    monkeypatch.setattr(module.engine, "run", completed)
    result = module.run(path, execute=True)
    assert result["training_executed"] is True
    assert result["observed_pilot_benefit"] is (different_pids and same_state)
    assert result["status"] == (
        "feedback_available" if different_pids and same_state else "resume_verification_failed"
    )
    assert result["promoted"] is False


def test_epoch_controls_reach_existing_pair_compiler(tmp_path, monkeypatch):
    path = config(tmp_path, monkeypatch)
    value = json.loads(path.read_text()); value.update(epoch_training=True, batch_size=3)
    path.write_text(json.dumps(value)); calls=[]
    task = tmp_path / "task.json"; task.write_text(json.dumps({"id":"pair"}))
    def compile_(*a, **kw):
        calls.append(kw); return task
    monkeypatch.setattr(module,"compile_paired_task",compile_)
    assert module.run(path)["status"] == "prepared"
    assert calls[0]["epoch_training"] is True and calls[0]["batch_size"] == 3


def test_batch_without_epoch_flag_cannot_be_silently_dropped(tmp_path, monkeypatch):
    path=config(tmp_path,monkeypatch);value=json.loads(path.read_text());value['batch_size']=2;path.write_text(json.dumps(value))
    def compile_(*args,**kwargs):
        assert kwargs['epoch_training'] is False and kwargs['batch_size']==2
        raise ValueError('Legacy paired recipe uses batch size one')
    monkeypatch.setattr(module,'compile_paired_task',compile_)
    report=module.run(path)
    assert report['status']=='blocked' and 'batch size one' in report['reason']
