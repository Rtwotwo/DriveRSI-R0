"""Local, sequential whole-world JSON worker for isolated renderer runtimes.

Run with the supported core interpreter. Stdout is exclusively the protocol;
native/Python diagnostics are redirected to stderr. This is a trusted local
process boundary, not a network service or a learned driving policy.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

from driversi.agents.traffic_control.population import PopulationController
from driversi.agents.traffic_control.routing import resolve_routes
from driversi.contracts.checkpoint import decode_world, encode_world
from driversi.contracts.population import EpisodePopulation
from driversi.contracts.road import RoadGraph
from driversi.contracts.route_tasks import RouteTasks
from driversi.contracts.scene import Scene
from driversi.sim.backends.pufferdrive import PufferDriveWorld


class WorldWorker:
    """One world, fixed stateless policy assignments, and an original reset snapshot."""

    def __init__(
        self,
        scene: Scene,
        graph: RoadGraph,
        tasks: RouteTasks | None = None,
        *,
        desired_speed: float = 8.0,
        population: EpisodePopulation | None = None,
    ):
        routes, routing = resolve_routes(scene, graph, tasks)
        self.controller = PopulationController(routes, population, desired_speed=desired_speed)
        self.configuration = self.controller.describe()
        self.configuration["routing"] = routing
        self.scene = scene
        self.world = PufferDriveWorld(scene)
        self.initial = self.world.snapshot()
        self.closed = False

    def close(self) -> None:
        self.world.close()
        self.closed = True

    def state(self) -> dict:
        observation = self.world.observe()
        return {
            "tick": observation.step,
            "all_actor_ids": list(self.scene.actor_ids),
            "controlled_ids": list(self.world.actor_ids),
            "state": self.world.state().tolist(),
            "motion": self.world.motion().tolist(),
            "presence": self.world.presence().tolist(),
            "lifecycle": self.world.lifecycle(),
            "alive": observation.alive.tolist(),
            "done": self.world.episode_done,
            "build_id": self.initial.build_id,
            "dt_seconds": 0.1,
            "behavior_policies": self.controller.behavior_policies,
            "external_actor_ids": list(self.controller.external_actor_ids),
        }

    def ready(self) -> dict:
        return {
            "schema_version": 1,
            "status": "ready",
            "request_id": None,
            "operation": "ready",
            **self.state(),
            "scene_metadata": self.scene.metadata,
            "dimensions": self.scene.dimensions.tolist(),
            "kinds": self.scene.kinds.tolist(),
            "controller_configuration": self.configuration,
            "population_configuration": self.controller.population.to_dict(),
            "behavior_policies": self.controller.behavior_policies,
            "external_actor_ids": list(self.controller.external_actor_ids),
            "action_space": "jerk_discrete_12",
            "trainable": False,
            "state_columns": [
                "x",
                "y",
                "z",
                "yaw",
                "vx",
                "vy",
                "vz",
                "goal_x",
                "goal_y",
                "stopped",
                "collision",
                "offroad",
                "reached_goal",
            ],
            "motion_columns": ["a_long", "a_lat", "steering"],
        }

    def _validate(self, request: dict) -> str:
        if not isinstance(request, dict):
            raise ValueError("Request must be a JSON object")
        identity = request.get("request_id")
        if type(identity) not in (str, int) or identity == "":
            raise ValueError("request_id must be a nonempty string or integer")
        operation = request.get("operation")
        if operation not in {
            "observe",
            "step",
            "reset",
            "close",
            "snapshot",
            "restore",
            "privileged_observe",
            "native_step",
        }:
            raise ValueError("Unknown operation")
        allowed = {"request_id", "operation"}
        if operation == "step":
            allowed.update(("expected_tick", "overrides", "external_actions"))
        if operation == "native_step":
            allowed.update(("expected_tick", "native_targets", "native_actions"))
        if operation == "restore":
            allowed.update(("expected_tick", "snapshot"))
        if set(request) - allowed:
            raise ValueError("Unexpected request fields")
        if self.closed:
            raise RuntimeError("Worker is closed")
        if operation == "step":
            expected = request.get("expected_tick")
            if type(expected) is not int or expected != self.world.observe().step:
                raise ValueError("expected_tick does not match current world tick")
            self.controller.validate_external_actions(request.get("external_actions"))
            overrides = request.get("overrides", {})
            if not isinstance(overrides, dict) or set(overrides) - set(self.world.actor_ids):
                raise ValueError("overrides must map controlled actor IDs to actions")
            if any(
                type(action) is not int or not 0 <= action < 12 for action in overrides.values()
            ):
                raise ValueError("Override actions must be integers in [0, 11]")
            if self.world.episode_done:
                raise RuntimeError("Episode is done; reset explicitly")
        if operation == "native_step":
            if (
                type(request.get("expected_tick")) is not int
                or request["expected_tick"] != self.world.observe().step
            ):
                raise ValueError("Native proposal clock is stale")
            if set(self.controller.native_actor_ids) != set(self.world.actor_ids):
                raise ValueError(
                    "Native port requires explicit native policy ownership for all policy actors"
                )
            if self.world.episode_done:
                raise RuntimeError("Episode is done; reset explicitly")
            live = {a for a, flag in zip(self.world.actor_ids, self.world.observe().alive) if flag}
            for field, keys, length in (
                ("native_targets", live, 10),
                ("native_actions", set(self.world.actor_ids), 2),
            ):
                value = request.get(field)
                if (
                    not isinstance(value, dict)
                    or set(value) != keys
                    or any(
                        not isinstance(v, list)
                        or len(v) != length
                        or any(type(x) not in (int, float) or not math.isfinite(x) for x in v)
                        for v in value.values()
                    )
                ):
                    raise ValueError("Invalid native control payload: " + field)
        if operation == "restore":
            if (
                type(request.get("expected_tick")) is not int
                or request["expected_tick"] != self.world.observe().step
            ):
                raise ValueError("expected_tick does not match current world tick")
            decode_world(
                request.get("snapshot"),
                initial=self.initial,
                configuration=self.configuration,
                actor_ids=self.scene.actor_ids,
            )
        return operation

    def handle(self, request: dict) -> dict:
        """Validate before mutation; restore a step/reset that fails to finish."""
        operation = self._validate(request)
        reply = {
            "schema_version": 1,
            "status": "ok",
            "operation": operation,
            "request_id": request["request_id"],
        }
        before = (
            self.world.snapshot()
            if operation in {"step", "native_step", "reset", "restore"}
            else None
        )
        try:
            if operation == "step":
                # Every decision sees this exact frozen observation. Overrides
                # affect only the joint action, never another actor's decision.
                commands = self.controller.commands(
                    self.world.observe(), request.get("external_actions")
                )
                policy = {command["actor"]: command["action"] for command in commands}
                executed = {**policy, **request.get("overrides", {})}
                transition = self.world.step(executed)
                origins = {
                    actor: identity["controller"]
                    for actor, identity in self.controller.behavior_policies.items()
                }
                overrides = request.get("overrides", {})
                origins.update(dict.fromkeys(overrides, "override"))
                reply.update(
                    policy_actions=policy,
                    executed_actions=executed,
                    behavior_policies=self.controller.behavior_policies,
                    action_origins=origins,
                    override_actor_ids=sorted(overrides),
                    diagnostics=commands,
                    valid=transition.valid.tolist(),
                    terminated=transition.terminated.tolist(),
                    truncated=transition.truncated.tolist(),
                    events={key: value.tolist() for key, value in transition.events.items()},
                )
            elif operation == "native_step":
                transition = self.world.step_native(
                    request["native_targets"], observation_tick=request["expected_tick"]
                )
                actions = request["native_actions"]
                reply.update(
                    policy_actions=actions,
                    executed_actions=actions,
                    action_space="native_continuous_state",
                    native_targets=request["native_targets"],
                    action_origins={a: "external-native-state-v1" for a in actions},
                    override_actor_ids=[],
                    diagnostics=[
                        {"actor": a, "action": v, "controller": "external-native-state-v1"}
                        for a, v in actions.items()
                    ],
                    valid=transition.valid.tolist(),
                    terminated=transition.terminated.tolist(),
                    truncated=transition.truncated.tolist(),
                    events={k: v.tolist() for k, v in transition.events.items()},
                )
            elif operation == "privileged_observe":
                from driversi.training.sampling.environment import OBSERVATION_VERSION, features

                reply["privileged_features"] = features(
                    self.world.observe(), self.world.max_steps
                ).tolist()
                reply["observation_version"] = OBSERVATION_VERSION
            elif operation == "snapshot":
                reply["snapshot"] = encode_world(
                    self.world.snapshot(),
                    tick=self.world.observe().step,
                    configuration=self.configuration,
                    actor_ids=self.scene.actor_ids,
                )
            elif operation == "restore":
                payload = request["snapshot"]
                self.world.restore(
                    decode_world(
                        payload,
                        initial=self.initial,
                        configuration=self.configuration,
                        actor_ids=self.scene.actor_ids,
                    )
                )
                if self.world.observe().step != payload["tick"]:
                    raise ValueError("Native checkpoint tick differs from declared tick")
            elif operation == "reset":
                self.world.restore(self.initial)
            reply.update(self.state())
            # Reject non-finite values before considering a transition committed.
            json.dumps(reply, allow_nan=False)
        except Exception:
            if before is not None:
                try:
                    self.world.restore(before)
                except Exception as rollback_error:
                    self.close()
                    raise RuntimeError(
                        "Fatal world rollback failure; worker closed"
                    ) from rollback_error
            raise
        if operation == "close":
            self.close()
        return reply


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--road-graph", type=Path, required=True)
    parser.add_argument("--route-tasks", type=Path)
    parser.add_argument("--desired-speed", type=float, default=8.0)
    parser.add_argument("--population-config", type=Path)
    args = parser.parse_args()
    sys.stdout.flush()
    # Retain a private protocol descriptor before rerouting fd1 itself; C code
    # using printf must not corrupt line-delimited JSON either.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1, encoding="utf-8")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    worker = None

    def emit(reply: dict) -> None:
        protocol.write(json.dumps(reply, allow_nan=False) + "\n")

    def reject_constant(value: str):
        raise ValueError(f"Non-finite JSON constant: {value}")

    try:
        worker = WorldWorker(
            Scene.load(args.scene),
            RoadGraph.load(args.road_graph),
            RouteTasks.load(args.route_tasks) if args.route_tasks else None,
            desired_speed=args.desired_speed,
            population=EpisodePopulation.load(args.population_config)
            if args.population_config
            else None,
        )
        emit(worker.ready())
        for line in sys.stdin:
            request = None
            try:
                request = json.loads(line, parse_constant=reject_constant)
                reply = worker.handle(request)
            except Exception as error:
                reply = {
                    "schema_version": 1,
                    "status": "error",
                    "request_id": request.get("request_id") if isinstance(request, dict) else None,
                    "operation": request.get("operation") if isinstance(request, dict) else None,
                    "tick": worker.world.observe().step if not worker.closed else None,
                    "error": {"type": type(error).__name__, "message": str(error)},
                }
            emit(reply)
            if worker.closed:
                return 0 if reply["status"] == "ok" else 1
        return 0
    except Exception as error:
        print(
            json.dumps(
                {
                    "operation": "startup_or_protocol",
                    "error": type(error).__name__,
                    "message": str(error),
                }
            ),
            file=sys.stderr,
            flush=True,
        )
        return 1
    finally:
        if worker is not None:
            worker.close()
        protocol.close()


if __name__ == "__main__":
    raise SystemExit(main())
