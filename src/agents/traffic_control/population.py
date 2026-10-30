"""Stateless rule populations and explicit external joint-action dispatch."""

from dataclasses import asdict

from driversi.agents.traffic_control.following import FollowingConfig, RouteFollower
from driversi.agents.traffic_control.traffic import TrafficController
from driversi.contracts.population import ActorPolicyBinding, EpisodePopulation, PopulationPolicy
from driversi.contracts.road import Route
from driversi.contracts.transition import Observation


class PopulationController:
    """Immutable episode assignments, grouped calls on copies of one observation."""

    def __init__(
        self,
        routes: dict[str, Route],
        population: EpisodePopulation | None = None,
        *,
        desired_speed: float = 8.0,
    ):
        self._actor_ids = tuple(routes)
        if population is None:
            population = EpisodePopulation(
                (
                    PopulationPolicy(
                        "default",
                        "route-idm-v2",
                        "route-idm-v2",
                        (("desired_speed", desired_speed),),
                    ),
                ),
                tuple(ActorPolicyBinding(actor, "default") for actor in routes),
            )
        population.validate_actors(self._actor_ids)
        self._controllers = {}
        policies = []
        cooperative = {
            b.actor_id
            for b in population.assignments
            if next(p for p in population.policies if p.policy_id == b.policy_id).controller
            == "route-conflict-v1"
        }
        for policy in population.policies:
            if policy.controller in {"route-idm-v2", "route-conflict-v1"}:
                try:
                    config = FollowingConfig(**dict(policy.parameters))
                except TypeError as error:
                    raise ValueError("Unknown route following parameter") from error
                policy = PopulationPolicy(
                    policy.policy_id,
                    policy.version,
                    policy.controller,
                    tuple(asdict(config).items()),
                )
                self._controllers[policy.policy_id] = (
                    TrafficController(routes, config, cooperative_ids=cooperative)
                    if policy.controller == "route-conflict-v1"
                    else RouteFollower(routes, config)
                )
            policies.append(policy)
        self._population = EpisodePopulation(tuple(policies), population.assignments)
        self._bindings = {item.actor_id: item for item in self._population.assignments}
        self._policies = {item.policy_id: item for item in self._population.policies}
        self._routes = dict(routes)

    @property
    def population(self) -> EpisodePopulation:
        return self._population

    @property
    def external_actor_ids(self) -> tuple[str, ...]:
        return tuple(
            actor
            for actor in self._actor_ids
            if self._policies[self._bindings[actor].policy_id].controller == "external-joint-v1"
        )

    @property
    def native_actor_ids(self) -> tuple[str, ...]:
        return tuple(
            actor
            for actor in self._actor_ids
            if self._policies[self._bindings[actor].policy_id].controller
            == "external-native-state-v1"
        )

    @property
    def behavior_policies(self) -> dict:
        return {
            actor: {
                "policy_id": binding.policy_id,
                "version": policy.version,
                "controller": policy.controller,
                "role": binding.role,
                "observation_version": policy.observation_version,
                "trainable": False,
            }
            for actor in self._actor_ids
            for binding in (self._bindings[actor],)
            for policy in (self._policies[binding.policy_id],)
        }

    def describe(self) -> dict:
        return {
            "controller": "episode-population-v1",
            "stateful": False,
            "population_configuration": self.population.to_dict(),
            "routes": {actor: route.to_dict() for actor, route in self._routes.items()},
            "external_state_owner": "caller_unrecorded",
            "action_space": "native_continuous_state"
            if self.native_actor_ids
            else "jerk_discrete_12",
            "trainable": False,
        }

    def validate_external_actions(self, actions: dict | None) -> dict[str, int]:
        if self.native_actor_ids:
            raise ValueError("Native state policies require the native_step port")
        actions = {} if actions is None else actions
        if not isinstance(actions, dict) or set(actions) != set(self.external_actor_ids):
            raise ValueError("external_actions must cover exactly the external policy actors")
        if any(type(action) is not int or not 0 <= action < 12 for action in actions.values()):
            raise ValueError("External actions must be integers in [0, 11]")
        return dict(actions)

    def commands(
        self, observation: Observation, external_actions: dict | None = None
    ) -> list[dict]:
        self.population.validate_actors(observation.actor_ids)
        external = self.validate_external_actions(external_actions)
        groups = {}
        for index, actor in enumerate(observation.actor_ids):
            policy = self._policies[self._bindings[actor].policy_id]
            if policy.controller in {"route-idm-v2", "route-conflict-v1"}:
                groups.setdefault(policy.policy_id, []).append(index)
        commands = {}
        for policy_id, indices in groups.items():
            part = Observation(
                tuple(observation.actor_ids[i] for i in indices),
                observation.features[indices].copy(),
                observation.alive[indices].copy(),
                observation.step,
                None if observation.metric is None else observation.metric.select(indices),
            )
            for command in self._controllers[policy_id].commands(part):
                commands[command["actor"]] = command
        commands.update(
            {
                actor: {"actor": actor, "action": action, "external_observation_verified": False}
                for actor, action in external.items()
            }
        )
        return [commands[actor] for actor in observation.actor_ids]
