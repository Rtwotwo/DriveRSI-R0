"""Build offline hard-case evidence from sealed predictions and real native images.

This module never calls a policy or edits its trajectory. Native PDM scoring is
replayed only to expose the conflict records that the original scalar report
does not serialize. Future occupancy is explicitly diagnostic-only evidence.
"""
import hashlib
import io
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

from driversi.agents.backends.native import digest, project_root
from driversi.runtime.budget import write_json

SHORT_METRICS = {
    "score": "PDMS", "no_at_fault_collisions": "NC",
    "drivable_area_compliance": "DAC", "driving_direction_compliance": "DDC",
    "ego_progress": "EP", "time_to_collision_within_bound": "TTC", "comfort": "C",
}


def read(path):
    return json.loads(Path(path).read_text())


def eligible_members(prepared, rows):
    """Select by old measured development failure, never by held-out outcomes."""
    result = []
    for member in prepared["members"]:
        if member.get("role") != "development":
            continue
        skill = member.get("skill", {})
        if (skill.get("id") != "following_speed_check" or
                skill.get("evidence", {}).get("navigation_command") != "GO STRAIGHT"):
            continue
        matches = [r for r in rows if r["token"] == member["token"] and r["arm"] == "incumbent"]
        if matches and any(r["metrics"]["TTC"] < 1.0 for r in matches):
            result.append(member)
    return sorted(result, key=lambda m: m["case_id"])


def check_replay(saved, actual):
    for name in SHORT_METRICS.values():
        if not math.isfinite(float(actual[name])) or not math.isclose(
                float(saved[name]), float(actual[name]), rel_tol=0, abs_tol=1e-9):
            raise ValueError("Native replay differs from sealed metric: " + name)


def relative_xy(x, y, origin):
    dx, dy = float(x) - origin[0], float(y) - origin[1]
    c, s = math.cos(origin[2]), math.sin(origin[2])
    return [dx * c + dy * s, -dx * s + dy * c]


def _native():
    root = project_root() / "agents/augustus/native/AUGUstus_SFT"
    native = root / "third_party/navsim"
    nuplan = root / "third_party/nuplan-devkit"
    for path in (native, nuplan):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from navsim.evaluate.pdm_score import pdm_score
    from navsim.common.dataclasses import Trajectory
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    # Refuse an already-imported, different NAVSIM scoring implementation.
    import inspect
    if not Path(inspect.getfile(pdm_score)).resolve().is_relative_to(native.resolve()):
        raise ValueError("Conflicting NAVSIM import; use a fresh native process")
    sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
    config = PDMScorerConfig(progress_weight=5.0, ttc_weight=5.0, comfortable_weight=2.0,
        driving_direction_horizon=1.0, driving_direction_compliance_threshold=2.0,
        driving_direction_violation_threshold=6.0, stopped_speed_threshold=5e-3,
        progress_distance_threshold=5.0)
    return pdm_score, Trajectory, PDMSimulator(proposal_sampling=sampling), PDMScorer(sampling, config)


def conflict_events(scorer, initial_ego):
    """Attribute native first-infraction times to the same occupancy predicates.

    The scalar/native score remains authoritative. This diagnostic pass mirrors
    its stopped/red-light/past-collision and ahead/lateral filters, and verifies
    its first event time against the scorer. It does not create a new metric.
    """
    import numpy as np
    from shapely import affinity
    from nuplan.common.actor_state.state_representation import StateSE2
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    from nuplan.planning.metrics.utils.collision_utils import CollisionType
    from nuplan.planning.simulation.observation.idm.utils import is_agent_ahead, is_agent_behind
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer_utils import get_collision_type
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex, EgoAreaIndex

    obs = scorer._observation
    step = scorer.proposal_sampling.interval_length
    origin = [initial_ego.rear_axle.x, initial_ego.rear_axle.y, initial_ego.rear_axle.heading]
    collision_seen, ttc_seen = set(obs.collided_track_ids), set(obs.collided_track_ids)
    collision, ttc = [], []

    def detail(token, ti, projection, kind, collision_type=None):
        state = scorer._states[1, ti]
        at = ti + projection
        obj = obs.unique_objects[token]
        center = obs[at][token].centroid
        velocity = getattr(obj, "velocity", None)
        heading = float(state[StateIndex.HEADING])
        current = [float(state[StateIndex.X]), float(state[StateIndex.Y]), heading]
        return dict(kind=kind, time_s=float(ti * step), projection_s=float(projection * step),
            object_token=token, object_type=obj.tracked_object_type.name,
            collision_type=collision_type,
            object_center_anchor_m=relative_xy(center.x, center.y, origin),
            object_center_ego_m=relative_xy(center.x, center.y, current),
            object_heading_relative_rad=float((obj.box.center.heading - heading + math.pi) % (2 * math.pi) - math.pi),
            object_velocity_world_mps=[float(velocity.x), float(velocity.y)] if velocity is not None else None,
            object_velocity_scope="first_cached_object_observation_not_event_velocity",
            ego_speed_mps=float(math.hypot(state[StateIndex.VELOCITY_X], state[StateIndex.VELOCITY_Y])),
            ego_multiple_lanes=bool(scorer._ego_areas[1, ti, EgoAreaIndex.MULTIPLE_LANES]),
            ego_nondrivable=bool(scorer._ego_areas[1, ti, EgoAreaIndex.NON_DRIVABLE_AREA]),
            geometry_intersects=True, evidence_scope="offline_replay_future_occupancy",
            causal_diagnosis_verified=False)

    for ti in range(scorer.proposal_sampling.num_poses + 1):
        state = scorer._states[1, ti]
        polygon = scorer._ego_polygons[1, ti]
        lateral = bool(scorer._ego_areas[1, ti, EgoAreaIndex.MULTIPLE_LANES] or
                       scorer._ego_areas[1, ti, EgoAreaIndex.NON_DRIVABLE_AREA])
        if not collision:
            for token in obs[ti].intersects(polygon):
                if obs.red_light_token in token or token in collision_seen:
                    continue
                kind = get_collision_type(state, polygon, obs.unique_objects[token], obs[ti][token])
                fault = kind in (CollisionType.ACTIVE_FRONT_COLLISION, CollisionType.STOPPED_TRACK_COLLISION)
                fault = fault or (lateral and kind == CollisionType.ACTIVE_LATERAL_COLLISION)
                if fault:
                    collision.append(detail(token, ti, 0, "at_fault_collision", kind.name))
                else:
                    collision_seen.add(token)
        if not ttc:
            speed = math.hypot(state[StateIndex.VELOCITY_X], state[StateIndex.VELOCITY_Y])
            if speed < scorer._config.stopped_speed_threshold:
                continue
            ego = StateSE2(*state[StateIndex.STATE_SE2])
            intersection = scorer._drivable_area_map.is_in_layer(ego.point, layer=SemanticMapLayer.INTERSECTION)
            for projection in (0, 3, 6, 9):
                dt = projection * step
                predicted = affinity.translate(polygon,
                    math.cos(ego.heading) * speed * dt, math.sin(ego.heading) * speed * dt)
                for token in obs[ti + projection].intersects(predicted):
                    if obs.red_light_token in token or token in ttc_seen:
                        continue
                    center = obs[ti + projection][token].centroid
                    other = StateSE2(center.x, center.y, obs.unique_objects[token].box.center.heading)
                    fault = is_agent_ahead(ego, other) or ((lateral or intersection) and not is_agent_behind(ego, other))
                    if fault:
                        ttc.append(detail(token, ti, projection, "ttc_projection_overlap"))
                    else:
                        ttc_seen.add(token)
        if collision and ttc:
            break
    for label, events, native_time in (
        ("collision", collision, scorer.time_to_at_fault_collision(1)),
        ("ttc", ttc, scorer.time_to_ttc_infraction(1)),
    ):
        if bool(events) != math.isfinite(native_time):
            raise ValueError("Diagnostic event disagrees with native " + label)
        if events and not math.isclose(min(e["time_s"] for e in events), native_time, abs_tol=1e-9):
            raise ValueError("Diagnostic first event differs from native " + label)
    return dict(collision=collision[:4], ttc=ttc[:4],
                object_trajectory_is_logged_replay=True, diagnostic_not_policy_input=True)


def replay(member, row, native):
    import lzma
    import pickle
    import numpy as np
    path = Path(member["metric_source"])
    if digest(path) != member["metric_sha256"]:
        raise ValueError("Metric cache changed")
    with lzma.open(path, "rb") as handle:
        cache = pickle.load(handle)
    pdm_score, Trajectory, simulator, scorer = native
    poses = np.asarray(row["trajectory"], dtype=np.float32)
    result = pdm_score(cache, Trajectory(poses), simulator.proposal_sampling, simulator, scorer)
    metrics = {SHORT_METRICS[k]: float(v) for k, v in asdict(result).items()}
    check_replay(row["metrics"], metrics)
    return dict(arm=row["arm"], seed=row["seed"], metrics=metrics,
                trajectory_ego_xyh=row["trajectory"], raw_trajectory_ego_xyh=row["raw_trajectory"],
                interval_s=0.5, trajectory_scope="sealed_model_output_not_expert_target",
                conflicts=conflict_events(scorer, cache.ego_state))


def extract_images(row, member, output):
    """Native decoded frames are real captures; no generated or future image."""
    import av
    from PIL import Image
    with av.open(io.BytesIO(row["video"])) as container:
        frames = [frame.to_image().convert("RGB") for frame in container.decode(video=0)]
    if len(frames) != 4 or any(im.size != (2048, 512) for im in frames):
        raise ValueError("Native four-frame stitched video contract differs")
    result = []
    # Exact source timestamp need not equal ideal 2 Hz: retain frame index and
    # mark the time offset nominal. The cache builder verified the source order.
    for index, nominal in ((0, -1.5), (3, 0.0)):
        image = frames[index]
        image.thumbnail((1536, 1536), Image.Resampling.LANCZOS)
        path = Path(output) / (member["case_id"] + f"-history-{index}.png")
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
        result.append(dict(path=str(path), sha256=digest(path), case_id=member["case_id"],
            label=member["case_id"] + (" historical first frame" if index == 0 else " current anchor") +
                  "; native real-image stitch LEFT / FRONT / RIGHT, no rear camera",
            time_offset_s=nominal, time_offset_is_nominal=True, native_frame_index=index,
            cameras=["CAM_L0", "CAM_F0", "CAM_R0"], future_image=False,
            source_video_sha256=hashlib.sha256(row["video"]).hexdigest()))
    return result


def build(prepared_path, historical_scores_path, output):
    """Return sealed development evidence; held-out images/events are never read."""
    start = time.monotonic()
    prepared, rows = read(prepared_path), read(historical_scores_path)
    out = Path(output).resolve()
    expected = dict(prepared_sha256=digest(prepared_path), scores_sha256=digest(historical_scores_path))
    if (out / "report.json").exists():
        old = read(out / "report.json")
        if old["inputs"] != expected:
            raise ValueError("Evidence inputs changed; immutable output required")
        for item in old["images"]:
            if digest(item["path"]) != item["sha256"]:
                raise ValueError("Cached real image changed")
        return old
    score_receipt = read(Path(historical_scores_path).parent / "report.json")
    if score_receipt.get("scores_sha256") != expected["scores_sha256"]:
        raise ValueError("Historical score file is not sealed")
    if score_receipt.get("prepared_sha256") != expected["prepared_sha256"]:
        raise ValueError("Historical scores reference a different prepared observation")
    members = eligible_members(prepared, rows)
    if not members:
        raise ValueError("No incumbent TTC failures in the declared development family")
    native = _native()
    cases = []
    for member in members:
        seeds = sorted(r["seed"] for r in rows if r["token"] == member["token"] and
                       r["arm"] == "incumbent" and r["metrics"]["TTC"] < 1.0)
        seed = seeds[0]
        arms = []
        for arm in ("native", "incumbent"):
            matches = [r for r in rows if r["token"] == member["token"] and r["seed"] == seed and r["arm"] == arm]
            if len(matches) != 1:
                raise ValueError("Historical paired prediction coverage is not unique")
            arms.append(replay(member, matches[0], native))
        if not arms[1]["conflicts"]["ttc"]:
            continue
        cases.append(dict(case_id=member["case_id"], token=member["token"], source_log=member["source_log"],
            observed_state=member["skill"]["evidence"], roles=["development", "diagnostic_target"],
            metric_cache_sha256=member["metric_sha256"], comparisons=arms,
            causal_diagnosis_verified=False))
        if len(cases) == 3:
            break
    if not cases:
        raise ValueError("No failure with a verified native conflict event")
    from driversi_agents.augustus.data import read_cache
    wanted = {c["token"] for c in cases}
    block = dict(cache=prepared["cache"], records=[dict(frame_id=m["token"]) for m in prepared["members"]])
    # Verify persisted native cache payload before decoding any observation.
    for name, sha in prepared.get("cache_files", {}).items():
        if digest(name) != sha:
            raise ValueError("Prepared native image cache changed")
    ds = read_cache(block)
    selected_rows = {r["frame_id"]: r for r in ds if r["frame_id"] in wanted}
    if set(selected_rows) != wanted:
        raise ValueError("Missing selected real observation")
    images, evidence = [], []
    by_token = {m["token"]: m for m in members}
    for case in cases:
        row, member = selected_rows[case["token"]], by_token[case["token"]]
        if row["scene_token"] != case["source_log"]:
            raise ValueError("Video source log differs from metric source")
        images.extend(extract_images(row, member, out / "images"))
        prefix = case["case_id"]
        evidence.append(dict(id=prefix + ".observed_state", kind="native_current_state", value=case["observed_state"]))
        for arm in case["comparisons"]:
            label = prefix + "." + arm["arm"]
            for metric, value in arm["metrics"].items():
                evidence.append(dict(id=label + "." + metric, kind="measured_metric", value=value))
            evidence.append(dict(id=label + ".trajectory", kind="sealed_planning_output", value={
                "poses_ego_xyh": arm["trajectory_ego_xyh"], "interval_s": 0.5, "offline_diagnostic_only": True}))
            evidence.append(dict(id=label + ".conflicts", kind="native_replayed_conflict", value=arm["conflicts"]))
    for item in images:
        evidence.append(dict(id=item["case_id"] + ".image." + str(item["native_frame_index"]),
            kind="real_historical_image", value={k: v for k, v in item.items() if k != "path"}))
    report = dict(schema_version=1, status="completed", inputs=expected,
        target_tokens=[c["token"] for c in cases], cases=cases, evidence=evidence, images=images,
        development=evidence, scope="offline_development_diagnosis_only",
        future_images_included=False, future_occupancy_is_offline_diagnostic=True,
        held_out_examples_exposed=False, policy_input_changed=False, new_policy_calls=0,
        replayed_predictions=len(cases) * 2, wall_seconds=time.monotonic() - start)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "report.json", report)
    return report
