import numpy as np

from driversi.agents.traffic_control.controllers import ConstantJerk, JointController
from driversi.contracts.transition import Observation


def test_per_actor_policies_receive_one_common_time():
    observed_steps = []

    class RecordingPolicy:
        def act(self, observation):
            observed_steps.append(observation.step)
            observation.features[:] = 999
            return np.full(len(observation.actor_ids), 10, dtype=np.int32)

    observation = Observation(("a", "b", "c"), np.zeros((3, 1124)), np.ones(3, bool), 4)
    shared = RecordingPolicy()
    controls = JointController({"a": shared, "b": ConstantJerk(1), "c": shared})
    assert controls.act(observation) == {"a": 10, "b": 1, "c": 10}
    assert observed_steps == [4]
    assert not observation.features.any()
