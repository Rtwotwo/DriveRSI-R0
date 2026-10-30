"""Measured trajectory diversity; behavior hypotheses are not causal skill claims."""

import json
from pathlib import Path

import numpy as np

from driversi.agents.backends.native import digest
from driversi.configuration import read_mapping
from driversi.contracts.scene import Scene


def distances(first, second):
    a, b = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if a.shape != (71, 2) or b.shape != a.shape or not np.isfinite([a, b]).all():
        raise ValueError("Diversity requires two complete actual 0-7 s XY trajectories")
    d = np.linalg.norm(a - b, axis=1)
    return dict(ade_7s_m=float(d.mean()), ade_future_5s_m=float(d[20:].mean()), fde_m=float(d[-1]))


def path_distances(first, second):
    """Compare actual XY at equal travelled distance from the shared origin.

    Neither time warping nor independent anchor translation can create novelty.
    Stops are removed only from arc interpolation; short support stays unresolved.
    """
    curves=[]
    for points in (first,second):
        xy=np.asarray(points,dtype=float)
        if xy.ndim!=2 or xy.shape[1]!=2 or len(xy)<2 or not np.isfinite(xy).all():raise ValueError('Actual XY sequence required')
        arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))]
        keep=np.r_[True,np.diff(arc)>1e-9]
        curves.append((arc[keep],xy[keep]))
    common=float(min(curves[0][0][-1],curves[1][0][-1]))
    if common<2.:
        return dict(common_path_m=common,path_ade_m=None,path_max_m=None)
    sample=np.linspace(0.,common,101)
    a,b=[np.column_stack([np.interp(sample,s,xy[:,j]) for j in range(2)]) for s,xy in curves]
    delta=np.linalg.norm(a-b,axis=1)
    return dict(common_path_m=common,path_ade_m=float(delta.mean()),path_max_m=float(delta.max()))


def compare(candidate, retained, *, minimum_ade_m=0.5, minimum_path_ade_m=None):
    if not np.isfinite(minimum_ade_m) or minimum_ade_m <= 0:
        raise ValueError("Positive declared diversity distance required")
    comparisons = []
    if minimum_path_ade_m is not None and (not np.isfinite(minimum_path_ade_m) or minimum_path_ade_m<=0):
        raise ValueError('Positive declared path distance required')
    supported=True
    if minimum_path_ade_m is not None:
        supported=(path_distances(candidate['xy'],candidate['xy'])['common_path_m']>=2.
                   and path_distances(candidate['xy'][20:],candidate['xy'][20:])['common_path_m']>=2.)
    for other in retained:
        if candidate["comparison_key"] != other["comparison_key"]:
            raise ValueError("Different source states cannot share an ADE deduplication bucket")
        row = dict(candidate_id=other["candidate_id"], **distances(candidate["xy"], other["xy"]))
        row["duplicate"] = min(row["ade_7s_m"], row["ade_future_5s_m"]) < minimum_ade_m
        if minimum_path_ade_m is not None:
            row.update(path_distances(candidate['xy'],other['xy']))
            future=path_distances(candidate['xy'][20:],other['xy'][20:])
            row['future_path_ade_m']=future['path_ade_m']
            row['future_common_path_m']=future['common_path_m']
            row['duplicate'] |= any(v is None or v<minimum_path_ade_m for v in (row['path_ade_m'],row['future_path_ade_m']))
        if candidate.get('environment_speed_policy'):
            from .environment_speed import pair_speed
            if other.get('environment_speed_policy')!=candidate['environment_speed_policy']:
                raise ValueError('Source speed context differs within the family')
            row.update(pair_speed(candidate['xy'],other['xy'],candidate['environment_speed_policy']))
        comparisons.append(row)
    return dict(
        distinct=supported and not any(x["duplicate"] for x in comparisons),
        minimum_ade_m=minimum_ade_m,
        comparisons=comparisons,
        rule=("full_future_time_ADE_and_common_arc_path_ADE_v1" if minimum_path_ade_m is not None
              else "both_full_7s_and_future_5s_ADE_must_reach_minimum"),
        **({'minimum_path_ade_m':minimum_path_ade_m,'minimum_common_path_m':2.,'path_support_sufficient':supported}
           if minimum_path_ade_m is not None else {}),
    )


def describe(generation, candidate_id):
    """Read the actual sealed teacher branch, never its desired waypoints."""
    attempt = generation["teacher"]
    report_path = Path(attempt["teacher_report"])
    report = read_mapping(report_path)
    if (
        generation["status"] not in ("trajectory_ready", "rendered")
        or attempt["status"] != "trajectory_admitted"
        or report.get("restore_exact") is not True
        or report.get("arrival_backend")
        not in ("driverl_native_causal_history", "source_history_perturbation_v1")
        or report.get("teacher_backend") != "driverl_native"
    ):
        raise ValueError("A complete admitted native DriveRL branch is required")
    execution = report_path.parent / "execution"
    path, scene_path = execution / "teacher.npz", execution / "scene.npz"
    scene = Scene.load(scene_path)
    with np.load(path, allow_pickle=False) as z:
        states, presence = z["states"], z["presence"].astype(bool)
        metadata = json.loads(str(z["metadata"]))
        ids = z["actor_ids"].tolist()
        ego = ids.index("ego")
        if (
            states.shape[:2] != (71, len(ids))
            or presence.shape != (71, len(ids))
            or z["actions"].shape != (70, 1, 2)
            or metadata["dt_seconds"] != 0.1
            or not presence[:, ego].all()
        ):
            raise ValueError("Actual contiguous 70-action rollout required")
        xy = states[:, ego, :2].astype(float)
        if not np.isfinite(xy).all():
            raise ValueError("Nonfinite physical trajectory")
        source = scene.poses[:, scene.actor_ids.index("ego"), :2]
        # Distance to the observed source corridor, not time-aligned progress error.
        start, segment = source[:-1], source[1:] - source[:-1]
        length2 = np.sum(segment * segment, axis=1)
        rel = xy[:, None] - start
        u = np.clip(np.sum(rel * segment, axis=2) / np.maximum(length2, 1e-12), 0, 1)
        corridor = np.linalg.norm(rel - u[:, :, None] * segment, axis=2).min(axis=1)
        speed = np.linalg.norm(np.diff(xy, axis=0), axis=1) / 0.1
        neighbors = np.linalg.norm(states[:, :, :2] - xy[:, None], axis=2)
        neighbors[~presence] = np.inf
        neighbors[:, ego] = np.inf
        nearest = neighbors.min(axis=1)
        heading = states[:, ego, 3]
        delta = states[:, :, :2] - xy[:, None]
        longitudinal = delta[:, :, 0] * np.cos(heading[:, None]) + delta[:, :, 1] * np.sin(
            heading[:, None]
        )
        lateral = -delta[:, :, 0] * np.sin(heading[:, None]) + delta[:, :, 1] * np.cos(
            heading[:, None]
        )
        ahead = presence & (longitudinal > 0) & (longitudinal < 15) & (np.abs(lateral) < 3)
        ahead[:, ego] = False
        speed_change = speed[10:] - speed[:-10]
        response = bool(np.any(ahead[10:70].any(axis=1) & (speed_change < -1.0)))
        recovery = bool(corridor[20] >= 0.5 and corridor[20] - corridor[-1] >= 0.25)
        types = (["Reaction_candidate"] if response else []) + (
            ["Recovery_candidate"] if recovery else []
        )
        if not types:
            types = ["unclassified"]
        origin = scene.metadata["source_train_membership"]
        comparison_key = dict(
            source_log=origin["source_log"],
            token=origin["token"],
            origin_xyz=scene.metadata["origin_xyz"],
            timestamp_us=scene.metadata["navsim_source"]["anchor_timestamp_us"],
            initial_xy=xy[0].tolist(),
        )
        result = dict(
            candidate_id=candidate_id,
            comparison_key=comparison_key,
            xy=xy.tolist(),
            initial_heading_rad=float(heading[0]),
            behavior_types=types,
            classification="kinematic_hypothesis_not_causal",
            evidence=dict(
                anchor_corridor_distance_m=float(corridor[20]),
                final_corridor_distance_m=float(corridor[-1]),
                min_object_center_distance_m=float(nearest.min())
                if np.isfinite(nearest.min())
                else None,
                max_one_second_deceleration_mps=float(-speed_change.min()),
                nearby_forward_object_with_deceleration=response,
            ),
            files={str(p): digest(p) for p in (path, scene_path, report_path)},
            sampling_seed=report.get("sampling_seed", 42),
        )
        plan_path=attempt.get('plan_path')
        if plan_path:
            plan=read_mapping(plan_path)
            chosen=next(a for a in plan['assignments'] if a['id']==attempt['assignment_id'])
            if chosen.get('environment_speed_policy'):
                from .environment_speed import policy_from_facts
                value=chosen['environment_speed_policy']
                if value!=policy_from_facts(value['facts']) or digest(plan_path)!=attempt['plan_sha256']:
                    raise ValueError('Source speed context or plan changed')
                result['environment_speed_policy']=value
                result['files'][str(plan_path)]=digest(plan_path)
                result['files'][value['facts']['source_file']]=value['facts']['source_sha256']
            if chosen.get('adaptive_diversity_policy'):
                from .adaptive_diversity import validate
                value=chosen['adaptive_diversity_policy']
                validate(value,result['environment_speed_policy'])
                if digest(plan_path)!=attempt['plan_sha256']:raise ValueError('Adaptive plan changed')
                result['adaptive_diversity_policy']=value
        return result
