import json

import pytest

from driversi.agents.backends.native import digest
from driversi.rsi.experiments import augustus_campaign as campaign


def fixture_config(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign.socket, "gethostname", lambda: "target")
    plan = tmp_path / "plan.json"
    plan.write_text('{"seed":42,"chunks":[[1],[2]]}')
    stopped = tmp_path / "stopped.json"
    stopped.write_text(json.dumps(dict(plan_sha256=digest(plan), all_stopped=True)))
    config = dict(
        host="target",
        plan_sha256=digest(plan),
        gpus=list(range(8)),
        output=str(tmp_path / "exp/dispatch"),
        predecessor_stop_receipt=str(stopped),
    )
    path = tmp_path / "execution.json"
    path.write_text(json.dumps(config))
    return path, config, plan, stopped


def test_placement_does_not_change_frozen_plan(tmp_path, monkeypatch):
    path, _, plan, _ = fixture_config(tmp_path, monkeypatch)
    before = plan.read_bytes()
    value, output = campaign.execution_settings(path, plan, tmp_path)
    assert value["gpus"] == list(range(8))
    assert output == tmp_path / "exp/dispatch"
    assert plan.read_bytes() == before


@pytest.mark.parametrize(
    "update",
    [
        {"host": "other"},
        {"plan_sha256": "bad"},
        {"gpus": [0, 0]},
        {"gpus": []},
        {"gpus": [True]},
        {"gpus": [-1]},
        {"output": "/tmp/outside"},
    ],
)
def test_wrong_placement_rejected(tmp_path, monkeypatch, update):
    path, config, plan, _ = fixture_config(tmp_path, monkeypatch)
    config.update(update)
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        campaign.execution_settings(path, plan, tmp_path)


@pytest.mark.parametrize(
    "receipt", [{"all_stopped": False}, {"all_stopped": True, "plan_sha256": "other"}]
)
def test_repartition_requires_stopped_matching_predecessor(tmp_path, monkeypatch, receipt):
    path, _, plan, stopped = fixture_config(tmp_path, monkeypatch)
    stopped.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        campaign.execution_settings(path, plan, tmp_path)


def test_relabel_guard_does_not_allow_implicit_fallback(tmp_path):
    from driversi.rsi.experiments.navval_loop import require_relabel_route

    (tmp_path / "continuation-policy.json").write_text(
        json.dumps({"production_route": "driverl_new_history_gs"})
    )
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"output": str(tmp_path)}))
    with pytest.raises(ValueError, match="relabel fallback"):
        require_relabel_route(config)


def test_failed_worker_leader_still_cleans_its_process_group(monkeypatch):
    from types import SimpleNamespace

    calls = []
    waited = []
    monkeypatch.setattr(campaign.os, "killpg", lambda pid, sig: calls.append((pid, sig)))
    process = SimpleNamespace(pid=123, poll=lambda: 1, wait=lambda **kw: waited.append(kw))
    campaign.stop_owned_workers([process])
    assert calls == [(123, campaign.signal.SIGTERM), (123, campaign.signal.SIGKILL)]
    assert len(waited) == 2


def test_supervisor_sigterm_enters_exception_cleanup():
    with pytest.raises(InterruptedError):
        campaign.interrupt_dispatch(campaign.signal.SIGTERM, None)
