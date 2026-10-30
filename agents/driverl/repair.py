"""Explicit Harness correction backend, separate from native DriveRL.

Replans from actual current/history inputs. Planning state is reset on each call;
only immutable map caches persist. Final qualification uses executed nuPlan metrics,
not the proposal scorer. The native model, official scorer and actors are unchanged.
"""
import gc
import hashlib
from pathlib import Path

import numpy as np
import yaml


def bounded_tracking_control(previous_acceleration, desired_acceleration, steering_rate, dt):
    """Physical limits for the separate correction tracker, not score tolerances."""
    values = [previous_acceleration, desired_acceleration, steering_rate, dt]
    if not np.isfinite(values).all() or abs(dt - 0.1) > 1e-8:
        raise ValueError("Correction requires finite controls at 10 Hz")
    if not -6.5 <= previous_acceleration <= 4.0:
        raise ValueError("Actual acceleration outside correction support")
    acceleration = float(np.clip(desired_acceleration,
        max(-6.5, previous_acceleration - 8.0 * dt),
        min(4.0, previous_acceleration + 5.0 * dt)))
    rate = float(np.clip(steering_rate, -0.8, 0.8))
    return acceleration, rate


def nonreversing_tracking_control(previous, desired, rate, speed, dt):
    """Respect the pinned bicycle's 0.2s acceleration lag without changing state."""
    if not np.isfinite(speed) or speed < 0.0:
        raise ValueError("Correction requires a nonnegative current forward speed")
    # Ease braking before zero speed; the exact one-step guard below remains authoritative.
    acceleration, steering = bounded_tracking_control(previous, max(desired, -speed), rate, dt)
    alpha = dt / (dt + 0.2)
    required = (-speed / dt - (1 - alpha) * previous) / alpha
    ceiling = min(4.0, previous + 5.0 * dt)
    if required > ceiling + 1e-12:
        raise ValueError("No nonreversing correction within command limits; anchor unsupported")
    acceleration = max(acceleration, min(required, ceiling))
    return acceleration, steering


def select_admissible_proposal(scores, multiplicative, direction, ttc, progress=None):
    """Reject invalid forecast candidates; this is not official execution scoring."""
    scores, multiplicative, direction, ttc = map(np.asarray, (scores, multiplicative, direction, ttc))
    if scores.ndim != 1 or not len(scores) or multiplicative.shape != (2, len(scores)) or direction.shape != scores.shape or ttc.shape != scores.shape:
        raise ValueError("Unexpected pinned PDM proposal evidence shape")
    valid = (np.isfinite(scores) & np.isfinite(multiplicative).all(axis=0)
             & (multiplicative == 1.0).all(axis=0) & (direction == 1.0) & (ttc == 1.0))
    indices = np.flatnonzero(valid)
    if not len(indices):
        return None, valid
    if progress is None:
        progress = np.zeros_like(scores)
    progress = np.asarray(progress)
    if progress.shape != scores.shape or not np.isfinite(progress).all():
        raise ValueError("Invalid proposal progress evidence")
    # PDM gives all sub-threshold progress proposals the same score. Resolve
    # exact ties by actual predicted progress, without modifying any score.
    tied = indices[scores[indices] == scores[indices].max()]
    return int(tied[np.argmax(progress[tied])]), valid


class PDMRouteRepair:
    def __init__(self, root, initialization):
        from driversi.agents.backends.sources import activate_packages

        activate_packages({"navsim": Path(root) / "src/vendor/driving/navsim-v1.1/navsim"})
        from navsim.planning.simulation.planner.pdm_planner import pdm_closed_planner
        from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
        from nuplan.planning.simulation.controller.motion_model.kinematic_bicycle import (
            KinematicBicycleModel,
        )
        from nuplan.planning.simulation.controller.tracker.lqr import LQRTracker

        repo = Path(root) / "src/vendor/driving/navsim-v1.1"
        pin = "3e8291bfa89ff247231e0227778840cd0a036896"
        from driversi.agents.backends.sources import require_source

        require_source(repo, pin)
        if not Path(pdm_closed_planner.__file__).resolve().is_relative_to(repo.resolve()):
            raise ValueError("Correction requires the owned NAVSIM v1.1 runtime")
        self.initialization = initialization
        self.planner_type = pdm_closed_planner.PDMClosedPlanner
        path = Path(root) / "agents/driverl/native/DriveRL/nuplan-devkit/nuplan/planning/script/config/simulation/ego_controller/tracker/lqr_tracker.yaml"
        cfg = {k: v for k, v in yaml.safe_load(path.read_text()).items() if not k.startswith("_")}
        cfg = {k: (np.asarray(v, dtype=float) if isinstance(v, list) else
                   int(v) if k == "tracking_horizon" else float(v)) for k, v in cfg.items()}
        self.tracker = LQRTracker(**cfg)
        self.motion = KinematicBicycleModel(get_pacifica_parameters())
        self.description = dict(navsim_revision=pin, lateral_offsets_m=[-2.0, -1.0, 1.0, 2.0],
                                speed_limit_fractions=[0.6, 0.8, 1.0], proposals=15,
                                proposal_horizon_s=4.0, trajectory_horizon_s=8.0,
                                stateful=False, score_tie_break="predicted_progress_metres", nonreversing_command_guard=True, proposal_guard="no_fault_collision_drivable_direction_ttc",
                                implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                qualification="independent_executed_metrics_required")

    def step(self, planner_input, next_iteration, ego):
        from navsim.planning.simulation.planner.pdm_planner.proposal.batch_idm_policy import (
            BatchIDMPolicy,
        )
        from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
            WeightedMetricIndex,
        )
        from nuplan.common.actor_state.dynamic_car_state import DynamicCarState
        from nuplan.common.actor_state.state_representation import StateVector2D
        from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

        dt = (next_iteration.time_point.time_us - ego.time_us) / 1e6
        if abs(dt - 0.1) > 1e-8:
            raise ValueError("Correction tracker requires 10 Hz actual-world input")
        policies = BatchIDMPolicy(10.0, [0.6, 0.8, 1.0], 2.0, 1.5, 1.5, 3.0)
        planner = self.planner_type(TrajectorySampling(num_poses=80, interval_length=0.1),
                                    TrajectorySampling(num_poses=40, interval_length=0.1),
                                    policies, self.description["lateral_offsets_m"], 100.0)
        was_enabled = gc.isenabled()
        try:
            planner.initialize(self.initialization)
            trajectory = planner.compute_planner_trajectory(planner_input)
        finally:
            if was_enabled:
                gc.enable()
        scorer = planner._scorer
        scores = scorer._aggregate_scores()
        direction = scorer._weighted_metrics[WeightedMetricIndex.DRIVING_DIRECTION]
        ttc = scorer._weighted_metrics[WeightedMetricIndex.TTC]
        selected, valid = select_admissible_proposal(scores, scorer._multi_metrics, direction, ttc, scorer._progress_raw)
        if selected is None:
            desired_acceleration, desired_rate = -3.0, 0.0
        else:
            # compute_planner_trajectory already extended its chosen proposal to
            # eight seconds. Reset through the public generator API before
            # extending our admissible choice; never reuse the extended cache.
            planner._generator.generate_proposals(ego, planner._observation, planner._proposal_manager)
            trajectory = planner._generator.generate_trajectory(selected)
            desired = self.tracker.track_trajectory(planner_input.iteration, next_iteration, ego, trajectory)
            desired_acceleration, desired_rate = desired.rear_axle_acceleration_2d.x, desired.tire_steering_rate
        previous = ego.dynamic_car_state.rear_axle_acceleration_2d.x
        acceleration, steering_rate = nonreversing_tracking_control(previous, desired_acceleration, desired_rate, ego.dynamic_car_state.rear_axle_velocity_2d.x, dt)
        dynamic = DynamicCarState.build_from_rear_axle(
            rear_axle_to_center_dist=ego.car_footprint.rear_axle_to_center_dist,
            rear_axle_velocity_2d=ego.dynamic_car_state.rear_axle_velocity_2d,
            rear_axle_acceleration_2d=StateVector2D(acceleration, 0.0),
            tire_steering_rate=steering_rate)
        predicted = self.motion.propagate_state(ego, dynamic, next_iteration.time_point - ego.time_point)
        actions = [(acceleration - previous) / dt, steering_rate]
        return predicted, actions, dict(backend="pdm_route_v1", proposals=15,
            selected_proposal=selected, admissible_proposals=int(valid.sum()),
            proposal_scores=np.where(np.isfinite(scores), scores, -1.0).tolist(),
            proposal_direction=direction.tolist(), proposal_ttc=ttc.tolist(),
            proposal_progress_m=scorer._progress_raw.tolist(),
            fallback_reason="no_admissible_forecast_bounded_brake" if selected is None else None,
            desired_acceleration=float(desired_acceleration),
            bounded_acceleration=acceleration, bounded_steering_rate=steering_rate,
            qualification="not_admitted")
