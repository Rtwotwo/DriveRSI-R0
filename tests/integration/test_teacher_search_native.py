"""Native dynamics and reference construction; not a learned-performance test."""

import numpy as np
import unittest
import importlib.util
from driversi.agents.backends.native import project_root
from driversi_agents.driverl.repair import PDMRouteRepair
from driversi_agents.driverl.control_lattice import rollout_feedback, rollout_reference


def check_forecast(family, parameters):
    from driversi.agents.backends.sources import activate_packages

    activate_packages(
        {"nuplan": project_root() / "agents/driverl/native/DriveRL/nuplan-devkit/nuplan"}
    )
    rep = PDMRouteRepair(project_root(), None)
    from nuplan.common.actor_state.ego_state import EgoState
    from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
        ego_state_to_state_array,
    )

    ego = EgoState.build_from_rear_axle(
        StateSE2(5.0, 0.5, 0.03),
        StateVector2D(8.0, 0.0),
        StateVector2D(0.0, 0.0),
        0.0,
        TimePoint(1000000),
        get_pacifica_parameters(),
    )
    path = PDMPath([StateSE2(float(x), 0.0, 0.0) for x in range(0, 201)])
    initial = ego_state_to_state_array(ego).copy()
    fn = rollout_feedback if family == "feedback" else rollout_reference
    a, controls, first = fn(rep.motion, ego, path, parameters, rep.tracker)
    b, repeated, again = fn(rep.motion, ego, path, parameters, rep.tracker)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(controls, repeated)
    np.testing.assert_array_equal(a[0], initial)
    np.testing.assert_array_equal(ego_state_to_state_array(ego), initial)
    np.testing.assert_array_equal(a[1], ego_state_to_state_array(first))
    assert a.shape == (41, 11) and controls.shape == (40, 2) and np.isfinite(a).all()
    assert (
        (a[:, 3] >= 0).all()
        and (controls[:, 0] >= -8.0 - 1e-8).all()
        and (controls[:, 0] <= 5.0 + 1e-8).all()
    )
    assert np.max(np.abs(controls[:, 1])) <= 0.8


@unittest.skipUnless(
    importlib.util.find_spec("torch") and importlib.util.find_spec("pyquaternion"),
    "native nuPlan environment required",
)
class NativeForecastTest(unittest.TestCase):
    def test_feedback(self):
        check_forecast("feedback", (1.0, 8.0, 0.0))

    def test_continuous_reference(self):
        check_forecast("reference", (1.5, 0.0, 0.0))

    def test_teacher_execution_whitelist_and_native_separation(self):
        from types import SimpleNamespace
        from driversi_agents.driverl.closed_loop import DriveRLSession
        from driversi_agents.driverl.control_lattice import ControlLatticeRepair

        class ReachedSnapshot(Exception):
            pass

        def snapshot():
            raise ReachedSnapshot()

        session = object.__new__(DriveRLSession)
        session.world = SimpleNamespace(episode_done=False)
        session.repair = session.search = session.reference = None
        session.snapshot = snapshot
        for cls in (PDMRouteRepair, ControlLatticeRepair):
            teacher = object.__new__(cls)
            with self.assertRaises(ReachedSnapshot):
                session.step(teacher=teacher)
            with self.assertRaises(ValueError):
                session.step(teacher=teacher, native_action=[0.0, 0.0])
            for flag in ("repair", "search", "reference"):
                setattr(session, flag, True)
                with self.assertRaises(ValueError):
                    session.step(teacher=teacher)
                setattr(session, flag, None)

        class UnregisteredTeacher(ControlLatticeRepair):
            pass

        with self.assertRaises(ValueError):
            session.step(teacher=object.__new__(UnregisteredTeacher))
