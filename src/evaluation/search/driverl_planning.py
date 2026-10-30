"""Forecast two plans from saved worlds for an action-only native teacher.

This is a privileged world-rollout adapter, not a new DriveRL trajectory head.
Each forecast runs policy inference afresh from its own causal history. Nothing
is read from the later executed branch. Restore both runtime and random state.
"""
import copy
import random
import time

import numpy as np
import torch


def forecast(session, anchor, *, deadline):
    from driversi_agents.driverl.closed_loop import ego_vector
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import ego_states_to_state_array

    original = session.snapshot()
    original_state = session.world.state().copy()
    original_decision = copy.deepcopy(session.last_decision)
    original_parity = list(session.parity)
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
           torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)
    started = time.monotonic()
    try:
        session.restore(anchor)
        first = int(session.ego.time_us)
        states, decisions = [copy.deepcopy(session.ego)], []
        for _ in range(40):
            if time.monotonic() >= deadline:
                raise TimeoutError("Independent planning forecast exceeded teacher budget")
            if not session.world.observe().alive[0]:
                raise ValueError("Independent planning forecast terminated before four seconds")
            transition = session.step()
            states.append(copy.deepcopy(session.ego))
            step = session.parity[-1]
            if step.get("repair_active") or step.get("action_origin") != "native_policy":
                raise ValueError("Native forecast cannot silently substitute another generator")
            decisions.append(dict(timestamp_us=int(states[-2].time_us),
                                  raw_action=step["raw_action"],
                                  native_search_active=step["native_search_active"]))
            if not transition.observation.alive[0]:
                raise ValueError("Independent planning forecast terminated")
        if [int(s.time_us) for s in states] != [first + i * 100000 for i in range(41)]:
            raise ValueError("Independent forecast clock is discontinuous")
        return dict(anchor_timestamp_us=first,
                    simulated_states=ego_states_to_state_array(states).tolist(),
                    decisions=decisions, policy_calls=len(decisions),
                    wall_seconds=time.monotonic() - started)
    finally:
        session.restore(original)
        session.parity[:] = original_parity
        session.last_decision = original_decision
        random.setstate(rng[0])
        np.random.set_state(rng[1])
        torch.set_rng_state(rng[2])
        if rng[3] is not None:
            torch.cuda.set_rng_state_all(rng[3])
        np.testing.assert_array_equal(session.world.state(), original_state)
        np.testing.assert_array_equal(ego_vector(session.ego), original["ego"])
        if session.history != original["history"]:
            raise AssertionError("Planning capture changed causal world history")


def capture_pair(session, previous, current, output, *, deadline):
    import json
    from pathlib import Path
    from driversi.agents.backends.native import digest

    if current["ego"][-1] - previous["ego"][-1] != 500000:
        raise ValueError("Two actual planning snapshots must be 0.5 seconds apart")
    path = Path(output)
    if path.exists():
        raise FileExistsError("Independent planning evidence cannot be overwritten")
    pair = dict(kind="independently_generated_native_plans",
                producer="driverl_action_policy_world_forecast_v1",
                producer_file=dict(path=str(Path(__file__).resolve()), sha256=digest(__file__)),
                generator=session.configuration,
                uses_dummy_trajectory=False, copied_from_executed_rollout=False,
                sampling_interval_s=0.1, privileged_world_forecast=True,
                previous=forecast(session, previous, deadline=deadline),
                current=forecast(session, current, deadline=deadline),
                restored_main_branch=True)
    path.write_text(json.dumps(pair, indent=2, allow_nan=False))
    return {**pair, "capture_file": dict(path=str(path.resolve()), sha256=digest(path))}
