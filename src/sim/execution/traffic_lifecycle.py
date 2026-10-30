"""Transactional admission/departure state; source support is not physical membership."""

import copy

import numpy as np

from driversi.agents.traffic_control.traffic import swept_contact
from driversi.contracts.road import Route
from driversi.evaluation.metrics.geometry import rectangle_corners, rectangles_overlap


def admission_neighbors(metric, positions, present, dimensions, horizon, margin):
    """Conservative distance bound; exact original checks decide every nearby pair.

    For swept SAT, sum of the two ego-axis extents is bounded above by
    the sum of both boxes' length/width plus 2*margin. Relative travel
    enlarges that bound by horizon*|relative_velocity|. The following
    test independently has a longitudinal + lateral L1 bound below.
    A pair beyond BOTH bounds cannot block admission. Order is unchanged.
    """
    ids = np.flatnonzero(present)
    other = positions[ids]
    size = dimensions[ids, :2]
    distance = np.linalg.norm(other[:, :2] - metric[:2], axis=1)
    speeds = np.linalg.norm(other[:, 4:6], axis=1) + np.linalg.norm(metric[3:5])
    following = .5 * (metric[5] + metric[6] + size.sum(axis=1)) + 2.5 + 1.5 * speeds
    swept = metric[5] + metric[6] + size.sum(axis=1) + 2 * margin + horizon * np.linalg.norm(other[:, 4:6] - metric[3:5], axis=1)
    bound = np.maximum(following, swept)
    return ids[(distance <= bound + 1e-6) | ~np.isfinite(distance) | ~np.isfinite(bound)]


class TrafficLifecycle:
    def __init__(self, scene):
        self.scene = scene
        self.config = scene.metadata["traffic_continuity"]
        if self.config.get("version") != 1 or self.config.get("forward_only") is not True:
            raise ValueError("Unsupported traffic continuity")
        if self.config.get("initial_overlap_policy", "reject") not in {
            "reject",
            "retain_and_report",
        }:
            raise ValueError("Unsupported initial overlap policy")
        self.domain = np.asarray(self.config["domain"], float)
        if (
            self.domain.shape != (4,)
            or not np.isfinite(self.domain).all()
            or np.any(self.domain[2:] <= self.domain[:2])
        ):
            raise ValueError("Invalid traffic domain")
        for key, low, high in (
            ("admission_horizon_seconds", 0.1, 5.0),
            ("admission_margin_m", 0.0, 2.0),
        ):
            value = self.config.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
                or not low <= value <= high
            ):
                raise ValueError("Invalid traffic admission configuration")
        for actor, evidence in self.config.get("navigation", {}).items():
            if actor not in scene.actor_ids:
                raise ValueError("Unknown exit identity")
            if evidence.get("departure_allowed"):
                sides = evidence.get("exit_sides")
                if (
                    not isinstance(sides, list)
                    or not sides
                    or any(type(k) is not int or k not in range(4) for k in sides)
                ):
                    raise ValueError("Departure requires designated exit sides")
        support = np.asarray(scene.metadata["reactive_background"]["source_presence"], bool)
        self.spawn = np.array(
            [
                np.flatnonzero(support[:, i])[0] if support[:, i].any() else len(support)
                for i in range(len(scene.actor_ids))
            ]
        )
        from driversi.data.scenes.static_context import static_prior_actors

        for actor in static_prior_actors(scene):
            self.spawn[scene.actor_ids.index(actor)] = 0
        self.routes = {
            a: Route(t["points"])
            for a, t in scene.metadata["reactive_background"]["assignments"].items()
        }
        self.reset()

    def reset(self):
        present = self.scene.presence_at(0)
        self.state = {
            "tick": 0,
            "status": ["active" if p else "unseen" for p in present],
            "previous": [False] * len(present),
            "present": present.tolist(),
            "events": [],
            "pending": [],
            "admissions": {
                a: {"tick": 0, "source_tick": 0} for a, p in zip(self.scene.actor_ids, present) if p
            },
            "departures": {},
        }
        for i in np.flatnonzero(present):
            for j in np.flatnonzero(present):
                if j <= i:
                    continue
                boxes = [
                    rectangle_corners(
                        *self.scene.poses[0, k, :2],
                        self.scene.poses[0, k, 3],
                        *self.scene.dimensions[k, :2],
                    )
                    for k in (i, j)
                ]
                if rectangles_overlap(*boxes):
                    if self.config.get("initial_overlap_policy", "reject") == "reject":
                        raise ValueError("Continuity rejects initial overlapping source objects")
                    self.state["events"].append(
                        {
                            "kind": "source_initial_contact",
                            "actors": [self.scene.actor_ids[i], self.scene.actor_ids[j]],
                            "tick": 0,
                            "collision_exempt": False,
                            "trainable": False,
                        }
                    )
        return copy.deepcopy(self.state)

    def restore(self, value):
        if not isinstance(value, dict) or set(value) != set(self.state):
            raise ValueError("Invalid lifecycle checkpoint")
        if type(value["tick"]) is not int or not 0 <= value["tick"] < len(self.scene.times):
            raise ValueError("Invalid lifecycle clock")
        n = len(self.scene.actor_ids)
        if any(len(value[k]) != n for k in ("status", "previous", "present")):
            raise ValueError("Lifecycle checkpoint identity mismatch")
        if any(x not in ("unseen", "pending", "active", "departed") for x in value["status"]):
            raise ValueError("Invalid participant status")
        if any(type(v) is not bool for k in ("previous", "present") for v in value[k]):
            raise ValueError("Invalid lifecycle mask")
        if value["present"] != [x == "active" for x in value["status"]]:
            raise ValueError("Lifecycle status/mask mismatch")
        if any(value["status"][i] != "active" for i in np.flatnonzero(self.scene.controlled)):
            raise ValueError("Cannot remove policy actors")
        admissions, departures = value["admissions"], value["departures"]
        if not isinstance(admissions, dict) or not isinstance(departures, dict):
            raise ValueError("Missing participant lifecycle evidence")
        known = set(self.scene.actor_ids)
        if not set(admissions) <= known or not set(departures) <= set(admissions):
            raise ValueError("Lifecycle evidence has unknown or unadmitted identities")
        for i, actor in enumerate(self.scene.actor_ids):
            status = value["status"][i]
            admission = admissions.get(actor)
            departure = departures.get(actor)
            if status in ("active", "departed"):
                if (
                    not isinstance(admission, dict)
                    or type(admission.get("tick")) is not int
                    or not self.spawn[i] <= admission["tick"] <= value["tick"]
                    or admission.get("source_tick") != int(self.spawn[i])
                ):
                    raise ValueError("Participant lacks valid admission evidence")
            elif admission is not None or departure is not None:
                raise ValueError("Unadmitted participant has an impossible history")
            if (
                status == "unseen"
                and self.spawn[i] <= value["tick"]
                or status == "pending"
                and self.spawn[i] > value["tick"]
            ):
                raise ValueError("Participant status precedes or skips source admission")
            if status == "departed":
                if (
                    not isinstance(departure, dict)
                    or departure.get("kind") != "supported_domain_exit"
                    or type(departure.get("tick")) is not int
                    or not admission["tick"] < departure["tick"] <= value["tick"]
                    or not self.config["navigation"].get(actor, {}).get("departure_allowed")
                ):
                    raise ValueError("Participant lacks supported departure evidence")
            elif departure is not None:
                raise ValueError("Active participant cannot have departure evidence")
        if {p.get("actor") for p in value["pending"]} != {
            a for a, s in zip(self.scene.actor_ids, value["status"]) if s == "pending"
        }:
            raise ValueError("Pending participants lack retained evidence")
        self.state = copy.deepcopy(value)

    def propose(self, tick, state, next_states):
        if tick != self.state["tick"]:
            raise ValueError("Lifecycle proposal clock mismatch")
        result = copy.deepcopy(self.state)
        result["tick"] = tick + 1
        present = np.asarray(result["present"], bool)
        result["previous"] = present.tolist()
        result["events"], result["pending"] = [], []
        positions = state.copy()
        for row in next_states:
            positions[int(row[0]), :7] = row[1:]
        # Policy next state isn't known before joint submission. Admission uses
        # the current velocity envelope, not policy future actions.
        for i in np.flatnonzero(self.scene.controlled):
            positions[i, :2] += 0.1 * positions[i, 4:6]
        for i in np.flatnonzero(present & ~self.scene.controlled):
            actor = self.scene.actor_ids[i]
            if not self.config["navigation"].get(actor, {}).get("departure_allowed"):
                continue
            pose = state[i]
            corners = rectangle_corners(*pose[:2], pose[3], *self.scene.dimensions[i, :2])
            navigation = self.config["navigation"][actor]
            sides = navigation.get("exit_sides", [])
            _, distance, tangent = self.routes[actor].project(pose[:2])
            forward = pose[4:6] @ np.array([np.cos(tangent), np.sin(tangent)]) > 0
            outward = (
                distance <= navigation.get("exit_route_max_distance_m", 2.0)
                and forward
                and any(
                    (np.all(corners[:, k % 2] < self.domain[k]) and pose[4 + k % 2] < 0)
                    if k < 2
                    else (np.all(corners[:, k % 2] > self.domain[k]) and pose[4 + k % 2] > 0)
                    for k in sides
                )
            )
            if outward:
                present[i] = False
                result["status"][i] = "departed"
                result["events"].append(
                    {
                        "actor": actor,
                        "kind": "supported_domain_exit",
                        "tick": tick + 1,
                        "last_state": pose[:7].tolist(),
                        "domain": self.domain.tolist(),
                    }
                )
                result["departures"][actor] = copy.deepcopy(result["events"][-1])
        entrants = []
        eligible = [
            i
            for i in range(len(present))
            if not self.scene.controlled[i]
            and result["status"][i] in ("unseen", "pending")
            and self.spawn[i] <= tick + 1
        ]
        for i in sorted(eligible, key=lambda i: (self.spawn[i], self.scene.actor_ids[i])):
            source_tick = int(self.spawn[i])
            candidate = np.r_[
                self.scene.poses[source_tick, i], self.scene.velocities[source_tick, i]
            ]
            metric = np.r_[
                candidate[:2], candidate[3], candidate[4:6], self.scene.dimensions[i, :2]
            ]
            blockers = []
            for j in admission_neighbors(metric, positions, present, self.scene.dimensions,
                    self.config["admission_horizon_seconds"], self.config["admission_margin_m"]):
                other = np.r_[
                    positions[j, :2],
                    positions[j, 3],
                    positions[j, 4:6],
                    self.scene.dimensions[j, :2],
                ]
                # Include t=0 explicitly; swept_contact starts at t=0 as well.
                direction = np.array([np.cos(metric[2]), np.sin(metric[2])])
                delta = other[:2] - metric[:2]
                aligned = np.cos(other[2] - metric[2]) > 0.94
                lateral = abs(delta @ np.array([-direction[1], direction[0]]))
                ahead = float(delta @ direction)
                gap = abs(ahead) - 0.5 * (metric[5] + other[5])
                trailing_speed = max(
                    0.0, float((metric[3:5] if ahead > 0 else other[3:5]) @ direction)
                )
                if (
                    aligned
                    and lateral < 0.5 * (metric[6] + other[6]) + 0.5
                    and gap < 2.0 + 1.5 * trailing_speed
                ):
                    blockers.append(self.scene.actor_ids[j])
                    continue
                if (
                    swept_contact(
                        metric,
                        other,
                        metric[3:5],
                        self.config["admission_horizon_seconds"],
                        self.config["admission_margin_m"],
                    )
                    is not None
                ):
                    blockers.append(self.scene.actor_ids[j])
            if blockers:
                result["status"][i] = "pending"
                result["pending"].append(
                    {
                        "actor": self.scene.actor_ids[i],
                        "source_tick": source_tick,
                        "delay_steps": tick + 1 - source_tick,
                        "blocking_actors": blockers,
                        "source_pose": candidate[:4].tolist(),
                    }
                )
            else:
                present[i] = True
                result["status"][i] = "active"
                positions[i, :7] = candidate
                entrants.append(np.r_[i, candidate])
                result["events"].append(
                    {
                        "actor": self.scene.actor_ids[i],
                        "kind": "admitted",
                        "tick": tick + 1,
                        "source_tick": source_tick,
                        "delay_steps": tick + 1 - source_tick,
                        "initial_state": candidate.tolist(),
                    }
                )
                result["admissions"][self.scene.actor_ids[i]] = copy.deepcopy(result["events"][-1])
        result["present"] = present.tolist()
        return result, np.asarray(entrants, float).reshape(-1, 8)


def validate_recorded_lifecycle(scene, states, presence, lifecycle):
    """Recompute every admission/departure from recorded, current geometry."""
    manager = TrafficLifecycle(scene)
    if len(lifecycle) != len(states) or len(presence) != len(states):
        raise ValueError("Continuity records must cover every state")
    if lifecycle[0].get("continuity") != manager.state:
        raise ValueError("Invalid initial continuity evidence")
    for tick in range(len(states)):
        if np.asarray(presence[tick]).tolist() != manager.state["present"]:
            raise ValueError("Continuity presence differs from execution evidence")
        if tick == len(states) - 1:
            break
        controls = lifecycle[tick].get("executed_traffic_controls", [])
        rows = np.asarray(
            [[scene.actor_ids.index(d["actor"]), *d["next_state"]] for d in controls], float
        ).reshape(-1, 8)
        derived, entrants = manager.propose(tick, states[tick], rows)
        if derived != lifecycle[tick + 1].get("continuity"):
            raise ValueError("Admission/departure geometry does not match evidence")
        for row in entrants:
            actual = np.asarray(states[tick + 1])[int(row[0]), :7]
            if not np.allclose(actual, row[1:], atol=2e-3, rtol=0):
                raise ValueError("Admitted actor differs from retained spawn state")
        manager.restore(derived)
