"""Fault-injection checks for the planning adapter's non-interference contract."""
import copy
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from driversi.evaluation.search.driverl_planning import capture_pair, forecast


class Session:
    def __init__(self, fail_at=None):
        self.tick = 20
        self.history = list(range(21))
        self.parity = [dict(marker="original")]
        self.last_decision = {"marker": "original"}
        self.configuration = {"test_fixture": True}
        self.calls = 0
        self.fail_at = fail_at
        self.world = SimpleNamespace(state=lambda: np.array([self.tick]),
                                     observe=lambda: SimpleNamespace(alive=[True]))

    @property
    def ego(self):
        return SimpleNamespace(time_us=self.tick * 100000)

    def snapshot(self):
        return dict(ego=[self.ego.time_us], tick=self.tick, history=copy.deepcopy(self.history))

    def restore(self, value):
        self.tick = value["tick"]
        self.history = copy.deepcopy(value["history"])

    def step(self):
        self.calls += 1
        random.random()
        np.random.random()
        torch.rand(1)
        self.tick += 1
        self.history.append(self.tick)
        self.last_decision = {"mutated": True}
        self.parity.append(dict(repair_active=False, action_origin="native_policy",
                                raw_action=[0., 0.], native_search_active=True))
        if self.calls == self.fail_at:
            raise RuntimeError("injected native failure")
        return SimpleNamespace(observation=SimpleNamespace(alive=[True]))


@pytest.fixture
def adapter_dependencies(monkeypatch):
    monkeypatch.setitem(sys.modules, "driversi_agents.driverl.closed_loop",
                        SimpleNamespace(ego_vector=lambda ego: [ego.time_us]))
    monkeypatch.setitem(sys.modules,
        "navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation",
        SimpleNamespace(ego_states_to_state_array=lambda states:
                        np.array([[s.time_us] for s in states])))


@pytest.mark.parametrize("failure", [None, 4])
def test_forecast_restores_world_history_diagnostics_and_rng(adapter_dependencies, failure):
    s = Session(failure)
    before = s.snapshot()
    random.seed(12); np.random.seed(12); torch.manual_seed(12)
    expected = (random.random(), np.random.random(), torch.rand(1))
    random.seed(12); np.random.seed(12); torch.manual_seed(12)
    if failure:
        with pytest.raises(RuntimeError, match="injected"):
            forecast(s, before, deadline=time.monotonic()+30)
    else:
        result = forecast(s, before, deadline=time.monotonic()+30)
        assert result["policy_calls"] == 40
        assert result["simulated_states"][-1] == [6000000]
    assert s.snapshot() == before
    assert s.parity == [{"marker": "original"}]
    assert s.last_decision == {"marker": "original"}
    assert random.random() == expected[0]
    assert np.random.random() == expected[1]
    assert torch.equal(torch.rand(1), expected[2])


def test_both_anchors_recompute_and_existing_evidence_is_not_overwritten(adapter_dependencies, tmp_path):
    s = Session()
    now = s.snapshot()
    previous = dict(ego=[1500000], tick=15, history=list(range(16)))
    path = tmp_path / "pair.json"
    pair = capture_pair(s, previous, now, path, deadline=time.monotonic()+30)
    assert s.calls == 80
    assert s.snapshot() == now
    assert pair["previous"]["anchor_timestamp_us"] == 1500000
    assert pair["current"]["anchor_timestamp_us"] == 2000000
    with pytest.raises(FileExistsError):
        capture_pair(s, previous, now, path, deadline=time.monotonic()+30)
    assert s.calls == 80


def test_timeout_does_not_change_main_branch(adapter_dependencies):
    s = Session(); before = s.snapshot()
    with pytest.raises(TimeoutError):
        forecast(s, before, deadline=time.monotonic()-1)
    assert s.calls == 0
    assert s.snapshot() == before
