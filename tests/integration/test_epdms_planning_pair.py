"""Run separately in the owned NAVSIM v2 runtime, never alongside loaded v1."""
import copy
import json

import numpy as np
import pytest

from driversi.evaluation.epdms.planning_pair import extended_comfort
from driversi.evaluation.epdms.state_codec import unpack_ego
from driversi.agents.backends.native import digest
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import ego_states_to_state_array
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex


def fixture(tmp_path):
    anchor = 10_000_000

    def packed(t):
        return [(t - anchor) / 1e6 * 5, 0., 0., 5., 0., 0., 0., 0., 0., 0., 0., t]

    def states(start):
        return ego_states_to_state_array([unpack_ego(packed(start + i * 100000)) for i in range(41)])

    current, previous = states(anchor), states(anchor - 500000)
    pair = dict(kind="independently_generated_native_plans", uses_dummy_trajectory=False,
                copied_from_executed_rollout=False, sampling_interval_s=0.1,
                current=dict(anchor_timestamp_us=anchor, simulated_states=current.tolist()),
                previous=dict(anchor_timestamp_us=anchor - 500000, simulated_states=previous.tolist()))
    data = dict(context=dict(anchor_timestamp_us=anchor), adjacent_native_plans=pair,
                prefix_history=[dict(origin="world", ego=packed(anchor + i * 100000)) for i in range(-20, 1)])
    reseal(data, tmp_path)
    return data, current, TrajectorySampling(num_poses=40, interval_length=0.1)


def reseal(data, tmp_path):
    pair = data["adjacent_native_plans"]
    capture = tmp_path / "capture.json"
    capture.write_text(json.dumps({k: v for k, v in pair.items() if k != "capture_file"}))
    pair["capture_file"] = dict(path=str(capture), sha256=digest(capture))


def test_native_comfort_pass_and_independent_planning_disagreement(tmp_path):
    data, actual, sampling = fixture(tmp_path)
    assert extended_comfort(data, actual, sampling) == 1.0
    previous = np.asarray(data["adjacent_native_plans"]["previous"]["simulated_states"])
    # Same reached state, conflicting future acceleration: native EC must fail.
    previous[1:, StateIndex.ACCELERATION_X] = 3.0
    data["adjacent_native_plans"]["previous"]["simulated_states"] = previous.tolist()
    reseal(data, tmp_path)
    assert extended_comfort(data, actual, sampling) == 0.0


def test_previous_plan_from_different_world_rejected_even_when_sealed(tmp_path):
    data, actual, sampling = fixture(tmp_path)
    data["adjacent_native_plans"]["previous"]["simulated_states"][0][0] += 1
    reseal(data, tmp_path)
    with pytest.raises(ValueError, match="differs from actual generated history"):
        extended_comfort(data, actual, sampling)


@pytest.mark.parametrize("key", ["uses_dummy_trajectory", "copied_from_executed_rollout"])
def test_dummy_or_rollout_slice_cannot_establish_ec(tmp_path, key):
    data, actual, sampling = fixture(tmp_path)
    data["adjacent_native_plans"][key] = True
    reseal(data, tmp_path)
    with pytest.raises(ValueError, match="Dummy trajectories"):
        extended_comfort(data, actual, sampling)


def test_missing_or_changed_capture_never_defaults_to_comfortable(tmp_path):
    data, actual, sampling = fixture(tmp_path)
    missing = copy.deepcopy(data)
    del missing["adjacent_native_plans"]
    with pytest.raises(ValueError, match="cannot be omitted"):
        extended_comfort(missing, actual, sampling)
    data["adjacent_native_plans"]["previous"]["simulated_states"][1][0] += 1
    with pytest.raises(ValueError, match="sealed independent capture"):
        extended_comfort(data, actual, sampling)


def test_missing_history_or_changed_current_plan_rejected(tmp_path):
    data, actual, sampling = fixture(tmp_path)
    missing = copy.deepcopy(data)
    missing["prefix_history"] = missing["prefix_history"][-1:]
    with pytest.raises(ValueError, match="exactly one"):
        extended_comfort(missing, actual, sampling)
    wrong = actual.copy()
    wrong[10, 0] += 1
    with pytest.raises(ValueError, match="differs from the captured current plan"):
        extended_comfort(data, wrong, sampling)
