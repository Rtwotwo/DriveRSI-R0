"""Bounded training-state generation; constraints do not modify driving scores."""
import math
import numpy as np
from driversi.contracts.road import Route


def validate_perturbation(value, total_steps):
    required={"start_tick","duration_ticks","jerk_delta","lateral_delta"}
    if not isinstance(value,dict) or set(value)!=required:
        raise ValueError("Perturbation requires exact timing and native action deltas")
    start,duration=value["start_tick"],value["duration_ticks"]
    if type(start) is not int or not 20<=start<=200 or type(duration) is not int or not 1<=duration<=30 or start+duration>total_steps-100:
        raise ValueError("Perturbation needs actual history and at least ten seconds of suffix")
    for key,bound in (("jerk_delta",2.),("lateral_delta",.8)):
        x=value[key]
        if type(x) not in (int,float) or not math.isfinite(x) or abs(x)>bound:
            raise ValueError("Perturbation action outside declared bounds")
    if value["jerk_delta"]==value["lateral_delta"]==0:
        raise ValueError("Zero intervention is a control, not generated difficulty")
    return dict(value)


def perturbed_action(raw, value, low, high):
    raw=np.asarray(raw,dtype=float).reshape(2)
    delta=np.array([value["jerk_delta"],value["lateral_delta"]])
    return np.clip(raw+delta,np.asarray(low).reshape(2),np.asarray(high).reshape(2))


def arrival_constraints(scene, state, transition, contacts):
    route=Route(**scene.metadata["driving_tasks"]["actors"]["ego"]["route"])
    q=np.asarray(state)[scene.actor_ids.index("ego")]
    arc,distance,_=route.project(q[:2])
    tangent=route.at(min(route.distances[-1],arc+1))-route.at(max(0.,arc-1))
    heading=float(abs(np.arctan2(np.sin(q[3]-np.arctan2(tangent[1],tangent[0])),np.cos(q[3]-np.arctan2(tangent[1],tangent[0])))))
    speed=float(q[4]*np.cos(q[3])+q[5]*np.sin(q[3]))
    reasons=[]
    if not np.isfinite([distance,heading,speed]).all():reasons.append("nonfinite")
    if distance>2.:reasons.append("lateral_support_exceeded")
    if heading>math.radians(20):reasons.append("heading_bound_exceeded")
    if not 0.<=speed<=25.:reasons.append("speed_bound_exceeded")
    if contacts:reasons.append("world_contact_in_arrival")
    if not transition.valid.all() or transition.terminated.any() or transition.truncated.any():reasons.append("invalid_arrival_transition")
    if any(transition.events.get("route_support_end",())):reasons.append("route_support_exhausted")
    return dict(accepted=not reasons,reasons=reasons,lateral_m=float(distance),heading_rad=heading,forward_speed_mps=speed)
