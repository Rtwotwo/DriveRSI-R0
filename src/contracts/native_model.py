"""Native policy boundary: preserve upstream inputs, sampling and outputs."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DecisionIdentity:
    session: str
    episode: str
    generation: int
    tick: int
    model_version: str

    def __post_init__(self):
        if any(
            not isinstance(v, str) or not v
            for v in (self.session, self.episode, self.model_version)
        ):
            raise ValueError("Nonempty decision identity required")
        if any(type(v) is not int or v < 0 for v in (self.generation, self.tick)):
            raise ValueError("Nonnegative integer generation/tick required")


@dataclass(frozen=True)
class NativeResult:
    identity: DecisionIdentity
    output: object
    representation: str
    authority: str = "proposal_only"
    likelihood: object = None


class NavsimPolicyAdapter:
    """An already initialized official agent, in its own native environment.

    AgentInput is passed verbatim. No resizing, extra history, future scene,
    scorer substitution, autocast, reseeding or trajectory resampling is added.
    This adapter does not advance a world and does not turn trajectories into PPO actions.
    """

    def __init__(self, native_agent, model_version, *, num_poses=8, interval_s=0.5):
        self.native_agent = native_agent
        self.model_version = model_version
        self.num_poses = num_poses
        self.interval_s = interval_s

    def predict(self, agent_input, identity, *, expected_identity):
        if identity != expected_identity or identity.model_version != self.model_version:
            raise ValueError("Stale or foreign decision identity")
        trajectory = self.native_agent.compute_trajectory(agent_input)
        poses = trajectory.poses
        if (
            not isinstance(poses, np.ndarray)
            or poses.shape != (self.num_poses, 3)
            or not np.isfinite(poses).all()
        ):
            raise ValueError("Native trajectory must contain finite rear-axle x/y/heading poses")
        sampling = trajectory.trajectory_sampling
        if (
            sampling.num_poses != self.num_poses
            or abs(sampling.interval_length - self.interval_s) > 1e-9
        ):
            raise ValueError("Native trajectory clock differs; explicit execution adapter required")
        return NativeResult(identity, trajectory, "navsim_rear_axle_relative_se2_m_rad")


class NuplanPlannerAdapter:
    """Keep official nuPlan history, initialization and AbstractTrajectory native."""

    def __init__(self, native_planner, model_version):
        self.native_planner = native_planner
        self.model_version = model_version

    def initialize(self, initialization):
        self.native_planner.initialize(initialization)

    def predict(self, planner_input, identity, *, expected_identity):
        if identity != expected_identity or identity.model_version != self.model_version:
            raise ValueError("Stale or foreign decision identity")
        trajectory = self.native_planner.compute_planner_trajectory(planner_input)
        if not callable(getattr(trajectory, "get_state_at_time", None)):
            raise ValueError("Expected native nuPlan AbstractTrajectory")
        return NativeResult(identity, trajectory, "nuplan_absolute_ego_state_trajectory")
