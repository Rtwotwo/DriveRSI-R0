"""Generate a continuous training-only arrival, then let the native student recover.

No pose edits, object deletions or teacher actions enter the arrival. Each attempt
is a separately budgeted Harness task; safe and rejected attempts remain evidence.
"""
import copy
from pathlib import Path
import numpy as np
import torch
from driversi.rsi.correction.contact_origin import source_contacts
from driversi.rsi.correction.curriculum import failed_driving, route_support_failure
from driversi.rsi.correction.perturbation import validate_perturbation, perturbed_action, arrival_constraints
from driversi.rsi.correction.reached_state import verify_reached_state
from driversi.runtime.budget import write_json
from driversi.sim.persistence.recording import RolloutRecord
from driversi_agents.driverl.collection import session


def perturb_step(s, cfg, tick):
    from driversi_agents.driverl.closed_loop import ego_vector
    from nuplan.common.actor_state.state_representation import TimePoint
    from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
    inp=s.planner_input()
    trajectory=s.planner.compute_planner_trajectory(inp)
    raw=trajectory.raw_action.reshape(-1,2)[0].detach().cpu().numpy()
    agent=s.planner._agent
    commanded=perturbed_action(raw,cfg,agent._action_low.detach().cpu().numpy(),agent._action_high.detach().cpu().numpy())
    memory=s._runtime_memory(s.planner,s.controller)
    try:
        s.controller.update_state(inp.iteration,SimulationIteration(TimePoint(s.ego.time_us+100000),tick+1),s.ego,trajectory)
        native_prediction=np.asarray(ego_vector(s.controller.get_state()),float)
    finally:s._restore_runtime(s.planner,s.controller,memory)
    tr=s.step(native_action=commanded,policy_forward=True,action_origin="bounded_perturbation")
    actual_prediction=np.asarray(ego_vector(s.controller.get_state()),float)
    effect=float(np.max(np.abs(actual_prediction[:-1]-native_prediction[:-1])))
    return tr,dict(tick=tick,native=raw.tolist(),commanded=commanded.tolist(),
        executed=s.last_decision["executed_actions"]["ego"],native_predicted_ego=native_prediction.tolist(),
        perturbed_predicted_ego=actual_prediction.tolist(),physical_effect_max_abs=effect,
        comparison="same_input_native_controller_next_state_before_world_rounding")


def collect_active(plan):
    from driversi_agents.driverl.recovery import row, risk, save_branch, official_score
    cfg=validate_perturbation(plan["perturbation"],plan["scout_steps"])
    anchor_tick=cfg["start_tick"]+cfg["duration_ticks"]
    out=Path(plan["output"]);s=session(plan,plan["checkpoint"])
    record,rows=RolloutRecord.start(s.world),[row(s)]
    contacts,arrival,actions=[],[],[]
    control=anchor=chosen=None
    reject=None
    try:
        origin=source_contacts(s.scene);write_json(out/"contact-origin.json",origin)
        if origin["quarantined"]:
            write_json(out/"selection.json",dict(status="source_quarantined",reason=origin["reason"],
                source_contacts=origin,selection=None,candidates=[],physics_steps=0,all_world_slots=len(s.scene.actor_ids),trainable=False))
            return
        error=None
        try:
            for tick in range(plan["scout_steps"]):
                if s.world.episode_done:break
                intervention=cfg["start_tick"]<=tick<anchor_tick
                if intervention:
                    tr,command=perturb_step(s,cfg,tick)
                    actions.append(command)
                else:tr=s.step()
                record.append(s.world,s.last_decision["executed_actions"],tr,s.last_decision["diagnostics"])
                rows.append(row(s));now=int(s.world.observe().step)
                pairs=s.world.contacts()
                if pairs:contacts.append(dict(tick=now,pairs=pairs))
                if now<=anchor_tick:
                    check=arrival_constraints(s.scene,s.world.state(),tr,pairs);arrival.append(dict(tick=now,**check))
                    if not check["accepted"]:
                        reject="arrival_constraints_failed";break
                if now==20:control=dict(runtime=s.snapshot(),prefix=copy.deepcopy(record),rows=copy.deepcopy(rows))
                if now==anchor_tick:
                    if not any(a["physical_effect_max_abs"]>1e-9 for a in actions):
                        reject="intervention_had_no_action_effect";break
                    chosen=risk(s)
                    anchor=dict(runtime=s.snapshot(),prefix=copy.deepcopy(record),rows=copy.deepcopy(rows))
                if now%20==0:write_json(out/"progress.json",dict(tick=now,target=plan["scout_steps"],generation="bounded_physical_perturbation"))
        except Exception as exc:
            reject="generation_execution_failed"
            error=repr(exc)
        torch.save(dict(runtime=s.snapshot(),record=record,rows=rows,anchor=anchor,control=control),out/"scout-execution.pt")
        generation=dict(kind="bounded_physical_perturbation",configuration=cfg,anchor_tick=anchor_tick,
            bounds=arrival,commands=actions,execution_error=error,physical_effect_verified=any(a["physical_effect_max_abs"]>1e-9 for a in actions),teleported=False,learned_self_play=False,
            traffic="unchanged_reactive_rule_controllers",teacher_actions_are_on_policy=False)
        write_json(out/"generation.json",generation)
        if record.actions:
            save_branch(s,record,out/"scout",plan,"Perturbed arrival then unassisted native student")
            write_json(out/"scout/history.json",rows)
        else:
            write_json(out/"initial-state.json",dict(actor_ids=list(s.scene.actor_ids),state=s.world.state().tolist(),error=error))
        report=dict(status="arrival_rejected" if reject else "no_driving_failure",
            reason=reject or "student_did_not_fail_fixed_driving_checks",selection=None,
            candidates=[chosen] if chosen else [],contacts=contacts,generation=generation,
            physics_steps=int(s.world.observe().step),recorded_steps=len(record.actions),all_world_slots=len(s.scene.actor_ids),
            complete=len(record.actions)==plan["scout_steps"],hard_failure_required=True,trainable=False)
        support=route_support_failure([t.events.get("route_support_end",[False]) for t in record.transitions]);report["support_diagnosis"]=support
        if not reject:
            score=official_score(s,rows,out/"scout/official-metrics",plan,"DriveRL-after-bounded-arrival")
            report["score"]=score
            if not report["complete"]:report.update(status="incomplete_scout",reason="incomplete_long_window")
            elif support["blocked"]:report.update(status="platform_support_failure",reason=support["reason"])
            elif failed_driving(score):
                if anchor is None or control is None or not chosen["eligible"]:report.update(status="no_anchor",reason="failure_without_clean_reached_state")
                else:
                    torch.save(anchor,out/"treatment-anchor.pt");torch.save(control,out/"control-anchor.pt")
                    s.restore(anchor["runtime"])
                    reached=verify_reached_state(s.scene,out/"scout/rollout.npz",s.world,anchor_tick)
                    report.update(status="selected",reason=None,selection=chosen,reached=reached,
                        correction_steps=plan["scout_steps"]-anchor_tick,automatic_selection=True,
                        causal_diagnosis="native_failure_after_bounded_arrival_requires_teacher_control")
        write_json(out/"selection.json",report)
    finally:s.close()
