"""Collect an engineering episode, preserving masks and the actual final state."""

from pathlib import Path

from driversi.agents.traffic_control.controllers import ConstantJerk, JointController
from driversi.agents.traffic_control.following import RouteFollower
from driversi.agents.traffic_control.routing import resolve_routes
from driversi.agents.traffic_control.traffic import TrafficController
from driversi.contracts.road import RoadGraph
from driversi.contracts.route_tasks import RouteTasks
from driversi.contracts.scene import Scene
from driversi.sim.backends.pufferdrive import PufferDriveWorld
from driversi.sim.persistence.recording import RolloutRecord, write_rollout


def rollout(scene: Scene, output: Path, *, policy: str = "coast", road_graph: RoadGraph | None = None,
            route_tasks: RouteTasks | None = None) -> dict:
    if policy not in {"coast", "brake", "follow", "traffic"}:
        raise ValueError("Unknown engineering controller")
    configuration = {"controller": policy}
    if policy in {"follow", "traffic"}:
        if road_graph is None:
            raise ValueError("Route controllers require an explicitly compiled road graph")
        routes, routing = resolve_routes(scene, road_graph, route_tasks)
        controller = TrafficController(routes, cooperative_ids=routes) if policy == "traffic" else RouteFollower(routes)
        configuration = controller.describe()
        configuration["routing"] = routing
    else:
        if route_tasks:
            raise ValueError("Route tasks apply only to route controllers")
        controller = ConstantJerk(7 if policy == "coast" else 1)
    with PufferDriveWorld(scene) as world:
        controllers = JointController(dict.fromkeys(world.actor_ids, controller))
        initial_snapshot = world.snapshot()
        record = RolloutRecord.start(world)
        while not world.episode_done:
            observation = world.observe()
            commands = None
            if isinstance(controller, RouteFollower):
                commands = controller.commands(observation)
                actions = {command["actor"]: command["action"] for command in commands}
            else:
                actions = controllers.act(observation)
            transition = world.step(actions)
            record.append(world, actions, transition, commands)
        return write_rollout(scene, output, record=record, snapshot=initial_snapshot,
                             controlled_ids=world.actor_ids, policy=policy,
                             controller_configuration=configuration)
