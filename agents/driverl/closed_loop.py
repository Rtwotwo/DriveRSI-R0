"""DriveRL ego executor. Imports the pinned upstream planner/controller unchanged.

No training, future detections or action quantization. Optional explicitly labelled correction.
The world owns collisions/termination; official nuPlan owns ego integration.
"""

# ruff: noqa: E402 -- select the owned native package roots before upstream imports

import copy
import hashlib
from pathlib import Path

import numpy as np
import torch
import yaml

from driversi.agents.backends.sources import activate_driverl, require_source

activate_driverl()

from driverl.nuplan.config import DriveRLNuPlanPlannerConfig
from driverl.nuplan.controller import DriveRLOneStageController
from driverl.nuplan.planner import DriveRLNuPlanPlanner
from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.scene_object import SceneObjectMetadata
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.static_object import StaticObject
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.maps.maps_datatypes import TrafficLightStatusData, TrafficLightStatusType
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import (
    SimulationIteration,
)

from driversi.sim.backends.pufferdrive import PufferDriveWorld


def ego_vector(e):
    d = e.dynamic_car_state
    return [
        e.rear_axle.x,
        e.rear_axle.y,
        e.rear_axle.heading,
        d.rear_axle_velocity_2d.x,
        d.rear_axle_velocity_2d.y,
        d.rear_axle_acceleration_2d.x,
        d.rear_axle_acceleration_2d.y,
        e.tire_steering_angle,
        d.angular_velocity,
        d.angular_acceleration,
        d.tire_steering_rate,
        e.time_us,
    ]


def unpack_ego(v):
    return EgoState.build_from_rear_axle(
        StateSE2(*v[:3]),
        StateVector2D(*v[3:5]),
        StateVector2D(*v[5:7]),
        v[7],
        TimePoint(int(v[11])),
        get_pacifica_parameters(),
        angular_vel=v[8],
        angular_accel=v[9],
        tire_steering_rate=v[10],
    )


def native_tensor_equal(a, b):
    """Exact finite values and identical missing-value positions."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if not a.is_floating_point():
        return torch.equal(a, b)
    return bool(torch.all((a == b) | (torch.isnan(a) & torch.isnan(b))).item())


def pack_objects(obs):
    result = []
    for a in obs.tracked_objects.tracked_objects:
        b = a.box
        vel = getattr(a, "velocity", StateVector2D(0.0, 0.0))
        angular = getattr(a, "angular_velocity", None)
        if angular is not None:
            if np.isnan(angular):
                angular = "nan"  # Preserve upstream unknown, using valid JSON.
            elif not np.isfinite(angular):
                raise ValueError("Infinite native angular velocity")
        result.append(
            dict(
                token=a.metadata.token,
                track_token=a.track_token,
                kind=a.tracked_object_type.name,
                pose=[b.center.x, b.center.y, b.center.heading],
                size=[b.length, b.width, b.height],
                velocity=[vel.x, vel.y],
                angular_velocity=angular,
            )
        )
    return result


def unpack_objects(rows, stamp):
    result = []
    for v in rows:
        kind = TrackedObjectType[v["kind"]]
        b = OrientedBox(StateSE2(*v["pose"]), *v["size"])
        meta = SceneObjectMetadata(stamp, v["token"], None, v["track_token"])
        if kind in (
            TrackedObjectType.VEHICLE,
            TrackedObjectType.PEDESTRIAN,
            TrackedObjectType.BICYCLE,
        ):
            a = Agent(
                kind,
                b,
                StateVector2D(*v["velocity"]),
                meta,
                angular_velocity=None
                if v.get("angular_velocity") is None
                else float(v["angular_velocity"]),
            )
        else:
            a = StaticObject(kind, b, meta)
        result.append(a)
    return DetectionsTracks(TrackedObjects(result))


def native_scenario(scene, maps_root, *, duration=None):
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import (
        ScenarioExtractionInfo,
    )

    if "navsim_source" in scene.metadata:
        from driversi_agents.driverl.navsim_context import scenario_from_scene

        return scenario_from_scene(scene, maps_root)
    m = scene.metadata["nuplan_source"]
    extraction = (
        None
        if duration is None
        else ScenarioExtractionInfo(
            scenario_duration=duration, extraction_offset=0.0, subsample_ratio=0.5
        )
    )
    return NuPlanScenario(
        str(Path(m["db_path"]).parent),
        m["db_path"],
        m["anchor_token"],
        m["anchor_timestamp_us"],
        "driversi_closed_loop",
        str(maps_root),
        "nuplan-maps-v1.0",
        m["map_name"],
        extraction,
        get_pacifica_parameters(),
    )


def native_initialized_scene(scene, scenario):
    """Explicit new branch initializer, never overwrite the source scene/log."""
    result = copy.deepcopy(scene)
    e = scenario.initial_ego_state
    d = e.dynamic_car_state
    c = e.center
    i = result.actor_ids.index("ego")
    origin = np.asarray(result.metadata["origin_xyz"][:2])
    rotation = np.array(
        [[np.cos(c.heading), -np.sin(c.heading)], [np.sin(c.heading), np.cos(c.heading)]]
    )
    old = dict(pose=result.poses[0, i].tolist(), velocity=result.velocities[0, i].tolist())
    result.poses[0, i, :2] = np.array([c.x, c.y]) - origin
    result.poses[0, i, 3] = c.heading
    result.velocities[0, i, :2] = rotation @ np.array(
        [d.center_velocity_2d.x, d.center_velocity_2d.y]
    )
    result.metadata["native_initial_motion"] = {
        "ego": [
            float(d.center_acceleration_2d.x),
            float(d.center_acceleration_2d.y),
            float(e.tire_steering_angle),
        ]
    }
    result.metadata["native_initialization"] = dict(
        source="official_nuplan_initial_ego_state",
        previous=old,
        current=dict(pose=result.poses[0, i].tolist(), velocity=result.velocities[0, i].tolist()),
        ego=[float(v) for v in ego_vector(e)[:-1]] + [int(e.time_us)],
        note="SDK heading and zero initial yaw-rate; differs from causal derivative compiler convention",
    )
    result.validate()
    return result


def make_planner(config, initialization, scenario=None):
    p = DriveRLNuPlanPlanner(DriveRLNuPlanPlannerConfig(**config), scenario=scenario)
    p.initialize(initialization)
    if getattr(p._loaded_agent.agent_config, "future_steps", 0):
        raise ValueError("Oracle future models are not supported in closed loop")
    return p


class DriveRLSession:
    """Single ego, multi-object reactive world; chunk boundaries never reset it."""

    def __init__(
        self,
        scene,
        *,
        repository,
        checkpoint,
        maps_root,
        device="cuda:0",
        parity=False,
        sampling_method="argmax",
        search_candidates=0,
        search_start=None,
        search_stop=None,
        search_seed=42,
        repair_backend=None,
        repair_start=None,
        repair_stop=None,
    ):
        if tuple(a for a, c in zip(scene.actor_ids, scene.controlled) if c) != ("ego",):
            raise ValueError("DriveRL native adapter currently requires ego as sole policy slot")
        from driversi_agents.driverl.random_state import validate_sampling

        validate_sampling(sampling_method, search_candidates=search_candidates, parity=parity)
        repository = Path(repository).resolve()
        pinned = "24959547edac549d64c34b2aab62d5c52599d078"
        actual = require_source(repository, pinned)["actual"]
        self.origin = np.asarray(scene.metadata["origin_xyz"][:2])
        self.scenario = native_scenario(scene, maps_root)
        self.source_context = scene.metadata.get(
            "nuplan_source", scene.metadata.get("navsim_source")
        )
        self.source_interval = getattr(self.scenario, "driversi_history_interval", 0.05)
        self.scene = scene = native_initialized_scene(scene, self.scenario)
        self.initialization = PlannerInitialization(
            self.scenario.get_route_roadblock_ids(),
            self.scenario.get_mission_goal(),
            self.scenario.map_api,
        )
        from driversi_agents.driverl.search import repair_window, search_window

        self.search = search_window(search_candidates, search_start, search_stop, seed=search_seed)
        self.repair = repair_window(repair_backend, repair_start, repair_stop)
        if self.search and self.repair:
            raise ValueError("Native TTS and separate correction are distinct experiments")
        self.config = dict(
            config_path=str(Path(repository) / "release/configs/driverl_teacher.yaml"),
            checkpoint_path=str(checkpoint),
            device=device,
            strict_checkpoint=True,
            sampling_method=sampling_method,
            trajectory_mode="action",
            compile_agent=False,
            tts_enabled=bool(self.search),
            tts_num_candidates=search_candidates or 8,
            tts_seed=search_seed,
        )
        self.configuration = dict(
            version="driverl-continuous-v1",
            planner=self.config,
            checkpoint_sha256=hashlib.file_digest(open(checkpoint, "rb"), "sha256").hexdigest(),
            action_space="driverl_native_jerk_steering_rate",
            navigation="official_source_route_roadblocks_and_mission_goal",
            history="source_past_then_actual_world",
            signal="latest_causal_log_signal",
            upstream_revision=actual,
            correction_search=self.search,
            correction_backend=self.repair,
        )
        if hasattr(self.scenario, "driversi_history_evidence"):
            self.configuration["source_history_bootstrap"] = self.scenario.driversi_history_evidence
        self.planner = make_planner(self.config, self.initialization, self.scenario)
        self.reference = (
            make_planner(self.config, self.initialization, self.scenario) if parity else None
        )
        self._search_selectors = [(p, p._tts_selector) for p in (self.planner, self.reference) if p]
        self._set_search_tick(0)
        self.repair_controller = None
        if self.repair:
            from driversi_agents.driverl.repair import PDMRouteRepair

            self.repair_controller = PDMRouteRepair(
                Path(__file__).resolve().parents[2], self.initialization
            )
            self.configuration["correction_implementation"] = self.repair_controller.description
        self.world = PufferDriveWorld(scene)
        try:
            self.initial = self.world.snapshot()
            controller_path = (
                Path(repository)
                / "nuplan-devkit/nuplan/planning/script/config/simulation/ego_controller/driverl_one_stage_controller.yaml"
            )
            controller_config = {
                k: v
                for k, v in yaml.safe_load(controller_path.read_text()).items()
                if not k.startswith("_")
            }
            controller_config["device"] = device
            self.configuration["controller"] = controller_config
            self.controller = DriveRLOneStageController(self.scenario, **controller_config)
            self.ref_controller = (
                DriveRLOneStageController(self.scenario, **controller_config) if parity else None
            )
            # Source native history from pinned nuPlan SQL queries (no model-side approximation).
            egos = list(self.scenario.get_ego_past_trajectory(0, time_horizon=1.0, num_samples=20))
            obs = list(self.scenario.get_past_tracked_objects(0, time_horizon=1.0, num_samples=20))
            self.history = [
                dict(
                    ego=[float(v) for v in ego_vector(e)[:-1]] + [int(e.time_us)],
                    objects=pack_objects(o),
                    origin="source",
                )
                for e, o in zip(egos, obs)
            ]
            if len(egos) != len(obs) or not egos:
                raise ValueError("Incomplete native source history")
            self.clock0 = int(self.source_context["anchor_timestamp_us"])
            self.ego = self._world_ego(self.scenario.initial_ego_state, self.clock0)
            self.history = [v for v in self.history if v["ego"][-1] < self.clock0]
            self._append_world()
            self.last_decision = None
            self.parity = []
            self.initial_conversion = dict(
                native_ego=ego_vector(self.scenario.initial_ego_state),
                world_ego=ego_vector(self.ego),
                max_pose_error=float(
                    np.max(
                        np.abs(
                            np.asarray(ego_vector(self.ego)[:3])
                            - np.asarray(ego_vector(self.scenario.initial_ego_state)[:3])
                        )
                    )
                ),
                max_velocity_error=float(
                    np.max(
                        np.abs(
                            np.asarray(ego_vector(self.ego)[3:5])
                            - np.asarray(ego_vector(self.scenario.initial_ego_state)[3:5])
                        )
                    )
                ),
            )
            self.source_parity = self._source_parity(egos, obs) if parity else None
        except BaseException:
            self.world.close()
            raise

    def _source_parity(self, egos, observations):
        """Independent official raw-log objects versus serialized adapter inputs."""
        stamp = self.scenario.initial_ego_state.time_us
        egos = [*egos, self.scenario.initial_ego_state]
        observations = [*observations, self.scenario.initial_tracked_objects]
        lights = list(self.scenario.get_traffic_light_status_at_iteration(0))
        native = PlannerInput(
            SimulationIteration(TimePoint(stamp), 0),
            SimulationHistoryBuffer.initialize_from_list(
                len(egos), egos, observations, self.source_interval
            ),
            lights,
        )
        adapted = PlannerInput(
            native.iteration,
            SimulationHistoryBuffer.initialize_from_list(
                len(egos),
                [unpack_ego(ego_vector(e)) for e in egos],
                [unpack_objects(pack_objects(o), e.time_us) for o, e in zip(observations, egos)],
                self.source_interval,
            ),
            lights,
        )
        before_a = self._runtime_memory(self.planner, self.controller)
        before_b = self._runtime_memory(self.reference, self.ref_controller)
        try:
            a = self.reference._feature_builder.build(native, self.initialization)
            b = self.planner._feature_builder.build(adapted, self.initialization)
            values = 0
            fields = []
            for k, v in vars(a).items():
                if isinstance(v, torch.Tensor):
                    if not native_tensor_equal(v, getattr(b, k)):
                        other = getattr(b, k)
                        indices = torch.nonzero(v != other)
                        raise AssertionError(
                            f"Source conversion mismatch: {k}; indices={indices[:8].tolist()}; "
                            f"native={v[v != other][:8].tolist()}; adapted={other[v != other][:8].tolist()}"
                        )
                    values += v.numel()
                    fields.append(k)
            ta = self.reference.compute_planner_trajectory(native)
            tb = self.planner.compute_planner_trajectory(adapted)
            if (
                not np.isfinite(ta.raw_action.numpy()).all()
                or not np.isfinite(tb.raw_action.numpy()).all()
            ):
                raise AssertionError("Non-finite native source action")
            np.testing.assert_array_equal(ta.raw_action.numpy(), tb.raw_action.numpy())
            next_iteration = SimulationIteration(TimePoint(stamp + 100000), 1)
            self.ref_controller.update_state(native.iteration, next_iteration, egos[-1], ta)
            self.controller.update_state(
                adapted.iteration, next_iteration, adapted.history.ego_states[-1], tb
            )
            np.testing.assert_array_equal(
                ego_vector(self.controller.get_state()), ego_vector(self.ref_controller.get_state())
            )
            return dict(
                source=getattr(
                    self.scenario, "driversi_history_scope", "official_nuplan_sql_history"
                ),
                compared_tensor_fields=fields,
                compared_values=values,
                feature_max_abs_error=0.0,
                action_max_abs_error=0.0,
                state_max_abs_error=0.0,
            )
        finally:
            self._restore_runtime(self.planner, self.controller, before_a)
            self._restore_runtime(self.reference, self.ref_controller, before_b)

    def _set_search_tick(self, tick):
        active = (
            self.search is not None and self.search["start_tick"] <= tick < self.search["stop_tick"]
        )
        for planner, selector in self._search_selectors:
            planner._tts_selector = selector if active else None
        return active

    def close(self):
        self.world.close()

    def _world_ego(self, prototype, stamp):
        i = self.scene.actor_ids.index("ego")
        s = self.world.state()[i]
        yaw = float(s[3])
        co, si = np.cos(yaw), np.sin(yaw)
        rot = np.array([[co, -si], [si, co]])
        d = get_pacifica_parameters().rear_axle_to_center
        rear = s[:2] + self.origin - d * np.array([co, si])
        omega = prototype.dynamic_car_state.angular_velocity
        vel = rot.T @ s[4:6] - np.array([0.0, omega * d])
        # Keep official rear-axle acceleration and angular dynamics. Native center
        # storage is used for pose/velocity; quantify its float32 conversion loss.
        v = ego_vector(prototype)
        v[:5] = [*rear, yaw, *vel]
        v[-1] = stamp
        if s[9]:
            v[3:11] = [0.0] * 8
        return unpack_ego(v)

    def _objects(self):
        rows = []
        state = self.world.state()
        kinds = {
            1: "VEHICLE",
            2: "PEDESTRIAN",
            3: "BICYCLE",
            4: "TRAFFIC_CONE",
            10: "GENERIC_OBJECT",
        }
        for i in np.flatnonzero(self.world.presence()):
            actor = self.scene.actor_ids[i]
            if actor == "ego":
                continue
            s = state[i]
            rows.append(
                dict(
                    token=actor,
                    track_token=actor,
                    kind=kinds.get(int(self.scene.kinds[i]), "GENERIC_OBJECT"),
                    pose=[*(s[:2] + self.origin), float(s[3])],
                    size=self.scene.dimensions[i].tolist(),
                    velocity=s[4:6].tolist(),
                )
            )
        return rows

    def _append_world(self):
        self.history.append(dict(ego=ego_vector(self.ego), objects=self._objects(), origin="world"))
        stamp = self.ego.time_us
        self.history = [v for v in self.history if v["ego"][-1] >= stamp - 2_000_000]

    def planner_input(self):
        egos = [unpack_ego(v["ego"]) for v in self.history]
        observations = [unpack_objects(v["objects"], int(v["ego"][-1])) for v in self.history]
        history = SimulationHistoryBuffer.initialize_from_list(len(egos), egos, observations, 0.1)
        signals = []
        for entry in self.source_context["signals"]:
            if entry["timestamp_us"] > self.ego.time_us:
                break
            signals = entry["states"]
        lights = [
            TrafficLightStatusData(
                TrafficLightStatusType[x["status"].upper()],
                int(x["lane_connector_id"]),
                self.ego.time_us,
            )
            for x in signals
        ]
        return PlannerInput(
            SimulationIteration(TimePoint(self.ego.time_us), self.world.observe().step),
            history,
            lights,
        )

    def _runtime_memory(self, p, c):
        names = [
            "_last_policy_time_us",
            "_last_action",
            "_last_goal_points",
            "_last_model_input_debug",
            "_cached_policy_acceleration_target",
        ]
        return dict(
            planner={k: copy.deepcopy(getattr(p, k)) for k in names},
            route=copy.deepcopy(p._feature_builder._corrected_route_roadblock_ids),
            controller=copy.deepcopy(c._cached_policy_acceleration_target),
            controller_state=ego_vector(c._current_state) if c._current_state is not None else None,
        )

    def _restore_runtime(self, p, c, v):
        for k, x in v["planner"].items():
            setattr(p, k, copy.deepcopy(x))
        p._feature_builder._corrected_route_roadblock_ids = copy.deepcopy(v["route"])
        c._cached_policy_acceleration_target = copy.deepcopy(v["controller"])
        c._current_state = (
            unpack_ego(v["controller_state"]) if v["controller_state"] is not None else None
        )

    def snapshot(self):
        # Preserve legacy snapshots; stochastic branches also own their RNG.
        from driversi_agents.driverl import random_state

        return dict(
            **({"random_state": random_state.capture()} if self.config["sampling_method"] == "sample" else {}),
            version="driverl-continuous-v1",
            configuration=copy.deepcopy(self.configuration),
            world=self.world.snapshot(),
            history=copy.deepcopy(self.history),
            ego=ego_vector(self.ego),
            runtime=self._runtime_memory(self.planner, self.controller),
            reference=self._runtime_memory(self.reference, self.ref_controller)
            if self.reference
            else None,
        )

    def restore(self, v):
        expected = {"version", "configuration", "world", "history", "ego", "runtime", "reference"}
        from driversi_agents.driverl import random_state

        if self.config["sampling_method"] == "sample":
            expected.add("random_state")
        if (
            not isinstance(v, dict)
            or set(v) != expected
            or v["version"] != "driverl-continuous-v1"
            or v["configuration"] != self.configuration
        ):
            raise ValueError("DriveRL snapshot model/configuration mismatch")
        history = copy.deepcopy(v["history"])
        ego = unpack_ego(v["ego"])
        if not history or history[-1]["ego"] != v["ego"] or history[-1]["origin"] != "world":
            raise ValueError("Native history must end at current physical ego")
        stamps = []
        for row in history:
            ev = row["ego"]
            stamp = int(ev[-1])
            if (
                len(ev) != 12
                or not np.isfinite(ev).all()
                or row["origin"] not in ("source", "world")
            ):
                raise ValueError("Invalid native ego history")
            if row["origin"] == "source" and stamp >= self.clock0:
                raise ValueError("Source future cannot enter native history")
            stamps.append(stamp)
            tokens = [x["track_token"] for x in row["objects"]]
            if len(set(tokens)) != len(tokens):
                raise ValueError("Duplicate native track")
            for x in row["objects"]:
                if (
                    not np.isfinite([*x["pose"], *x["size"], *x["velocity"]]).all()
                    or min(x["size"]) <= 0
                ):
                    raise ValueError("Invalid native object history")
            unpack_objects(row["objects"], stamp)
        if any(a >= b for a, b in zip(stamps, stamps[1:])) or stamps[-1] - stamps[0] > 2_000_000:
            raise ValueError("Invalid native history clock")
        before = self.snapshot()
        try:
            if "random_state" in v:
                random_state.restore(v["random_state"])
            self.world.restore(v["world"])
            tick = self.world.observe().step
            if ego.time_us != self.clock0 + tick * 100000:
                raise ValueError("Native ego and world clock differ")
            actual = self._world_ego(ego, ego.time_us)
            if not np.allclose(ego_vector(actual)[:5], ego_vector(ego)[:5], rtol=0, atol=1e-7):
                raise ValueError("Native ego differs from physical world")
            if history[-1]["objects"] != self._objects():
                raise ValueError("Native object history differs from physical world")
            self._restore_runtime(self.planner, self.controller, v["runtime"])
            if self.reference:
                self._restore_runtime(self.reference, self.ref_controller, v["reference"])
        except BaseException:
            if "random_state" in before:
                random_state.restore(before["random_state"])
            self.world.restore(before["world"])
            self._restore_runtime(self.planner, self.controller, before["runtime"])
            if self.reference:
                self._restore_runtime(self.reference, self.ref_controller, before["reference"])
            raise
        self.history = history
        self.ego = ego
        self.last_decision = None

    def step(
        self,
        *,
        native_action=None,
        policy_forward=False,
        teacher=None,
        action_origin="native_learner",
    ):
        if action_origin not in ("native_learner", "bounded_perturbation") or (
            action_origin == "bounded_perturbation"
            and (native_action is None or teacher is not None)
        ):
            raise ValueError("Invalid external action provenance")
        if self.world.episode_done:
            raise RuntimeError("Episode done; no implicit reset")
        if teacher is not None:
            from driversi.rsi.correction.history_perturbation import arrival_search_compatible
            from driversi_agents.driverl.control_lattice import ControlLatticeRepair
            from driversi_agents.driverl.history_perturbation import HistoryPerturbation
            from driversi_agents.driverl.repair import PDMRouteRepair
            from driversi_agents.driverl.source_corridor import SourceCorridorRepair

            history_teacher = type(teacher) is HistoryPerturbation
            disjoint_search = history_teacher and arrival_search_compatible(
                self.world.observe().step, self.search
            )
            if (
                (history_teacher and not disjoint_search)
                or type(teacher)
                not in (
                    PDMRouteRepair,
                    ControlLatticeRepair,
                    SourceCorridorRepair,
                    HistoryPerturbation,
                )
                or native_action is not None
                or self.repair
                or (self.search and not disjoint_search)
                or self.reference
            ):
                raise ValueError(
                    "Explicit same-anchor teacher cannot mix native learning/search/repair"
                )
        if "navsim_source" in self.scene.metadata and native_action is None:
            from driversi_agents.driverl.navsim_context import require_dense_history

            require_dense_history(self.history, self.ego.time_us)
        before = self.snapshot()
        tick = self.world.observe().step
        targets = {}
        action = [0.0, 0.0]
        report = {
            "tick": tick,
            "action_origin": action_origin if native_action is not None else "native_policy",
            "inference": False,
            "native_search_active": self._set_search_tick(tick),
        }
        try:
            if self.world.observe().alive[0]:
                inp = self.planner_input()
                if self.reference and tick % 90 == 0:
                    left = self.planner._feature_builder.build(inp, self.initialization)
                    right = self.reference._feature_builder.build(
                        copy.deepcopy(inp), self.initialization
                    )
                    compared = 0
                    for key, value in vars(left).items():
                        if isinstance(value, torch.Tensor):
                            if not native_tensor_equal(value, getattr(right, key)):
                                raise AssertionError("Native feature mismatch: " + key)
                            compared += value.numel()
                    report["native_feature_values_compared"] = compared
                if native_action is None:
                    trajectory = self.planner.compute_planner_trajectory(inp)
                else:
                    # Explicit learner action uses the official continuous trajectory/controller.
                    # No search/repair or reference-policy substitution is admissible for on-policy RL.
                    if self.reference is not None or self.repair is not None:
                        raise ValueError(
                            "Native training action cannot mix parity/repair controllers"
                        )
                    from driverl.nuplan.planner import DriveRLActionTrajectory

                    raw = torch.as_tensor(
                        native_action, dtype=torch.float32, device=self.planner._device
                    ).reshape(1, 1, 2)
                    agent = self.planner._agent
                    if (
                        not torch.isfinite(raw).all()
                        or (raw < agent._action_low).any()
                        or (raw > agent._action_high).any()
                    ):
                        raise ValueError("Invalid native continuous action")
                    jerk, steer = self.planner._decode_action(raw)
                    cfg = self.planner.config
                    trajectory = DriveRLActionTrajectory(
                        ego_state=self.ego,
                        jerk_long=jerk,
                        lat_command=steer,
                        raw_action=raw.detach().cpu().clone(),
                        trajectory_sampling=self.planner._trajectory_sampling,
                        goal_points=None,
                        debug_info={
                            "policy_forward": bool(policy_forward),
                            "policy_interval_s": cfg.policy_interval_s,
                            "engine_action_config": {
                                k: getattr(cfg, k)
                                for k in (
                                    "min_jerk_long",
                                    "max_jerk_long",
                                    "positive_jerk_limit_when_nonnegative_acc",
                                    "nuplan_bicycle_accel_time_constant",
                                    "nuplan_bicycle_max_acceleration",
                                    "nuplan_bicycle_max_steering_rate",
                                )
                            },
                        },
                    )
                nxt = SimulationIteration(TimePoint(self.ego.time_us + 100000), tick + 1)
                self.controller.update_state(inp.iteration, nxt, self.ego, trajectory)
                predicted = self.controller.get_state()
                action = [float(trajectory.jerk_long), float(trajectory.lat_command)]
                if not np.isfinite([*action, *ego_vector(predicted)]).all():
                    raise AssertionError("Non-finite native control output")
                report.update(
                    inference=True,
                    raw_action=trajectory.raw_action.reshape(-1, 2)[0].tolist(),
                    all_actions_sha256=hashlib.sha256(
                        trajectory.raw_action.numpy().tobytes()
                    ).hexdigest(),
                )
                if self.reference:
                    ref = self.reference.compute_planner_trajectory(copy.deepcopy(inp))
                    self.ref_controller.update_state(inp.iteration, nxt, self.ego, ref)
                    np.testing.assert_array_equal(
                        trajectory.raw_action.numpy(), ref.raw_action.numpy()
                    )
                    np.testing.assert_array_equal(action, [ref.jerk_long, ref.lat_command])
                    np.testing.assert_array_equal(
                        ego_vector(predicted), ego_vector(self.ref_controller.get_state())
                    )
                    report["native_output_max_abs_error"] = 0.0
                report["native_action"] = list(action)
                repair_active = teacher is not None or bool(
                    self.repair and self.repair["start_tick"] <= tick < self.repair["stop_tick"]
                )
                report["repair_active"] = repair_active
                if repair_active:
                    predicted, action, repair_evidence = (teacher or self.repair_controller).step(
                        inp, nxt, self.ego
                    )
                    if not np.isfinite([*action, *ego_vector(predicted)]).all():
                        raise AssertionError("Non-finite correction output")
                    report["repair_evidence"] = repair_evidence
                    report["native_output_is_shadow_only"] = True
                report["executed_action"] = list(action)
                center = predicted.center
                d = predicted.dynamic_car_state
                vel = d.center_velocity_2d
                rot = np.array(
                    [
                        [np.cos(center.heading), -np.sin(center.heading)],
                        [np.sin(center.heading), np.cos(center.heading)],
                    ]
                )
                velocity = rot @ np.array([vel.x, vel.y])
                acc = d.center_acceleration_2d
                targets["ego"] = [
                    center.x - self.origin[0],
                    center.y - self.origin[1],
                    float(self.world.state()[0, 2]),
                    center.heading,
                    *velocity,
                    0.0,
                    acc.x,
                    acc.y,
                    predicted.tire_steering_angle,
                ]
            else:
                predicted = self.ego
            report["native_targets"] = copy.deepcopy(targets)
            tr = self.world.step_native(targets, observation_tick=tick)
            self.ego = self._world_ego(predicted, self.clock0 + (tick + 1) * 100000)
            if targets:
                report["world_position_rounding_m"] = float(
                    np.linalg.norm(self.world.state()[0, :2] - np.asarray(targets["ego"][:2]))
                )
            self._append_world()
            self.last_decision = {
                "tick": tick,
                "executed_actions": {"ego": action},
                "diagnostics": [
                    dict(
                        actor="ego",
                        action=action,
                        controller=(
                            "Harness-bounded-perturbation"
                            if action_origin == "bounded_perturbation"
                            else (
                                "Harness-native-recovery-search-v1"
                                if report.get("repair_evidence", {}).get("backend")
                                == "native_recovery_search_v1"
                                else "Harness-privileged-source-corridor-LQR"
                                if report.get("repair_evidence", {}).get("backend")
                                == "source_corridor_lqr_v1"
                                else "Harness-source-history-perturbation-LQR"
                                if report.get("repair_evidence", {}).get("backend")
                                == "source_history_perturbation_v1"
                                else "Harness-PDM-route-repair"
                            )
                            if report.get("repair_active")
                            else "DriveRL-official-native"
                        ),
                        raw_action=report.get("raw_action"),
                        policy_executed=bool(report["inference"])
                        and not report.get("repair_active", False)
                        and action_origin != "bounded_perturbation",
                        action_origin=report["action_origin"],
                        repair_executed=bool(report.get("repair_active")),
                    )
                ],
            }
            self.parity.append(report)
            return tr
        except BaseException:
            self.restore(before)
            raise
