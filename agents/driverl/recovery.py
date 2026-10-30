"""Actual-world hard-anchor mining and same-state teacher execution worker."""
import argparse
import copy
from pathlib import Path

import numpy as np
import torch

from driversi.agents.backends.native import project_root
from driversi.configuration import read_mapping
from driversi.contracts.road import RoadGraph
from driversi.evaluation.metrics.nuplan_metrics import score_record
from driversi.rendering.bev.replay import export_replay
from driversi.rsi.correction.curriculum import admit, select_anchor, failed_driving, correction_verdict, valid_recovery_transition
from driversi.rsi.correction.contact_origin import source_contacts
from driversi.rsi.correction.reached_state import verify_reached_state
from driversi.rsi.plugins.synthesis import check_files
from driversi.runtime.budget import write_json
from driversi.sim.persistence.recording import RolloutRecord, write_rollout
from driversi_agents.driverl.collection import session
from driversi_agents.driverl.closed_loop import ego_vector, unpack_objects
from driversi_agents.driverl.learning.learner import deterministic_training
from driversi_agents.driverl.repair import PDMRouteRepair


def row(s):
    return dict(ego=ego_vector(s.ego), objects=s._objects(), lights=[
        dict(status=x.status.name, lane_connector_id=x.lane_connector_id, timestamp=x.timestamp)
        for x in s.planner_input().traffic_light_data])


def risk(s):
    e = s.ego
    h = e.center.heading
    v = e.dynamic_car_state.center_velocity_2d
    ev = np.array([v.x*np.cos(h)-v.y*np.sin(h), v.x*np.sin(h)+v.y*np.cos(h)])
    near = []
    for obj in unpack_objects(s._objects(), e.time_us).tracked_objects.tracked_objects:
        gap = float(e.car_footprint.geometry.distance(obj.box.geometry))
        delta = np.array([obj.center.x-e.center.x, obj.center.y-e.center.y])
        ov = getattr(obj, "velocity", None)
        vel = np.array([ov.x, ov.y]) if ov is not None else np.zeros(2)
        closing = float(np.dot(ev-vel, delta)/max(float(np.linalg.norm(delta)), 1e-6))
        near.append(dict(actor=obj.track_token, clearance_m=gap,
                         closing_ttc_s=gap/closing if closing > .1 else None))
    candidate = select_anchor([dict(x, eligible=True, tick=0) for x in near]) if near else {}
    tick = int(s.world.observe().step)
    return dict(tick=tick, eligible=bool(s.world.observe().alive[0] and not s.world.contacts()),
                clearance_m=candidate.get("clearance_m"), closing_ttc_s=candidate.get("closing_ttc_s"),
                actor=candidate.get("actor"), meaning="risk_proxy_not_causal_failure_diagnosis")


def save_branch(s, record, path, plan, title):
    path.mkdir(parents=True, exist_ok=True)
    s.scene.save(path / "scene.npz")
    protocol = dict(kind="counterfactual_recovery", title=title, trainable=False,
                    note="Actual reactive world. Teacher execution is separate from native student. No visual admission.")
    write_rollout(s.scene, path/"rollout.npz", record=record, snapshot=s.initial,
                  controlled_ids=s.world.actor_ids, policy=title,
                  controller_configuration=s.configuration, protocol=protocol)
    export_replay(s.scene, path/"rollout.npz", path/"viewer.html", inspection=protocol,
                  road_graph=RoadGraph.load(Path(plan["road_graph"])), compress_payload=True)


def collect(plan):
    if plan.get("perturbation"):
        from driversi_agents.driverl.active_recovery import collect_active
        return collect_active(plan)
    if plan.get("hard_failure_required"):
        return collect_hard(plan)
    out = Path(plan["output"])
    s = session(plan, plan["checkpoint"])
    record = RolloutRecord.start(s.world)
    rows = [row(s)]
    candidates, saved, contacts = [], {}, []
    try:
        for _ in range(plan["scout_steps"]):
            if s.world.episode_done:
                break
            tr = s.step()
            record.append(s.world, s.last_decision["executed_actions"], tr, s.last_decision["diagnostics"])
            rows.append(row(s))
            tick = int(s.world.observe().step)
            if s.world.contacts():
                contacts.append(dict(tick=tick, pairs=s.world.contacts()))
            if tick >= 20 and tick % 5 == 0:
                x = risk(s)
                x["eligible"] = x["eligible"] and tick+40 < len(s.scene.times)
                candidates.append(x)
                if x["eligible"]:
                    saved[tick] = dict(runtime=s.snapshot(), prefix=copy.deepcopy(record), rows=copy.deepcopy(rows))
        save_branch(s, record, out/"scout", plan, "Current student: automatic risk discovery")
        rejection = dict(candidates=candidates, contacts=contacts,
            physics_steps=len(record.actions), all_world_slots=len(s.scene.actor_ids),
            selection=None, status="no_anchor", reason="no_eligible_reached_state", trainable=False)
        if not any(x["eligible"] for x in candidates):
            write_json(out/"selection.json", rejection)
            return
        chosen = select_anchor(candidates)
        if 20 not in saved:
            rejection.update(status="control_anchor_unavailable", reason="paired_control_not_available")
            write_json(out/"selection.json", rejection)
            return
        for name, tick in (("treatment", chosen["tick"]), ("control", 20)):
            torch.save(saved[tick], out/(name+"-anchor.pt"))
        s.restore(saved[chosen["tick"]]["runtime"])
        reached = verify_reached_state(s.scene, out/"scout/rollout.npz", s.world, chosen["tick"])
        write_json(out/"selection.json", dict(status="selected", selection=chosen, candidates=candidates,
            contacts=contacts, reached=reached, physics_steps=len(record.actions),
            all_world_slots=len(s.scene.actor_ids), current_checkpoint=plan["checkpoint"],
            automatic_selection=True, learned_self_play=False,
            causal_diagnosis="unknown_until_same_anchor_controls", trainable=False))
    finally:
        s.close()


def correct(plan):
    out = Path(plan["output"])
    bundle = torch.load(plan["anchor"], map_location="cpu", weights_only=False)
    s = session(plan, plan["checkpoint"])
    reports = {}
    try:
        for name in ("student", "teacher"):
            path = out/name
            path.mkdir()
            s.restore(bundle["runtime"])
            same = s.snapshot()["ego"] == bundle["runtime"]["ego"] and s.history == bundle["runtime"]["history"]
            record, rows = copy.deepcopy(bundle["prefix"]), copy.deepcopy(bundle["rows"])
            teacher_type = PDMRouteRepair
            if plan.get("backend") == "native_recovery_search_v1":
                from driversi_agents.driverl.control_lattice import ControlLatticeRepair
                teacher_type = ControlLatticeRepair
            teacher = teacher_type(project_root(), s.initialization) if name == "teacher" else None
            diagnostics, contacts, valid, error = [], [], True, None
            start = len(record.actions)
            try:
                for _ in range(plan.get("correction_steps", 40)):
                    if s.world.episode_done:
                        break
                    tr = s.step(teacher=teacher)
                    record.append(s.world, s.last_decision["executed_actions"], tr, s.last_decision["diagnostics"])
                    rows.append(row(s))
                    diagnostics.append(copy.deepcopy(s.parity[-1]))
                    if teacher is not None and getattr(teacher, "last_forecast", None) is not None and len(diagnostics) == 1:
                        np.savez_compressed(path/"candidate-forecast.npz", **teacher.last_forecast)
                    valid = valid and valid_recovery_transition(tr, final_step=len(record.actions)-start == plan.get("correction_steps",40))
                    if s.world.contacts():
                        contacts.append(dict(tick=int(s.world.observe().step), pairs=s.world.contacts()))
                    if teacher is not None and plan.get("backend") == "native_recovery_search_v1" and diagnostics[-1].get("repair_evidence", {}).get("selected_proposal") is None:
                        # An executed fallback cannot satisfy admission. Preserve that one bounded
                        # action and its state, then spend no more budget on this rejected branch.
                        error = "teacher_search_exhausted_without_admissible_forecast"
                        break
            except Exception as exc:
                error = repr(exc)
            save_branch(s, record, path, plan, "Same anchor: " + name)
            write_json(path/"history.json", rows)
            write_json(path/"decisions.json", diagnostics)
            execution = dict(same_anchor=same, complete=len(record.actions)-start == plan.get("correction_steps", 40) and error is None,
                valid=bool(valid), teacher_no_fallback=name=="teacher" and bool(diagnostics) and
                all(x.get("repair_evidence", {}).get("selected_proposal") is not None for x in diagnostics),
                contact_states=len(contacts), contacts=contacts, error=error,
                anchor_tick=start, suffix_steps=len(record.actions)-start,
                requested_suffix_steps=plan.get("correction_steps", 40),
                score_scope="complete_window" if len(record.actions)-start == plan.get("correction_steps", 40) and error is None else "partial_execution_diagnostic_only",
                prefix_scope="identical_actual_arrival_history", arrival_perturbation=plan.get("perturbation"), all_world_slots=len(s.scene.actor_ids))
            try:
                score = score_record(s.scene, rows, path/"official-metrics",
                    project_root()/plan["model_config"]["repository"]/"DriveRL", plan["model_config"]["paths"]["maps"],
                    planner_name="DriveHarness-"+name,
                    execution_policy=dict(kind=name, anchor_tick=start, branch=str(path/"rollout.npz"),
                        observation="actual_reactive_world", expert_reference="source_log_future_for_evaluation_only"))
            except Exception as exc:
                score = dict(status="failed", reason=repr(exc), official_benchmark=False)
            report = dict(execution=execution, score=score, admission=admit(execution, score),
                          teacher=teacher.description if teacher else None)
            write_json(path/"report.json", report)
            reports[name] = report
        write_json(out/"correction.json", dict(branches=reports, selected="teacher" if reports["teacher"]["admission"]["accepted"] else None,
            physics_steps=sum(x["execution"]["suffix_steps"] for x in reports.values()),
            all_world_slots=len(s.scene.actor_ids),
            attribution="same_anchor_intervention_evidence_not_proof_of_model_causality",
            teacher_actions_are_on_policy=False, visual_trainable=False, promoted=False,
            genuine=correction_verdict(reports, plan.get("hard_failure_required", False))))
    finally:
        s.close()



def official_score(s, rows, path, plan, role):
    return score_record(s.scene, rows, path, project_root()/plan["model_config"]["repository"]/"DriveRL",
        plan["model_config"]["paths"]["maps"], planner_name=role,
        execution_policy=dict(kind=role, actual_reactive_world=True, harness_override=bool(plan.get("perturbation")),
            arrival_perturbation=plan.get("perturbation"),native_student_suffix=True))


def collect_hard(plan):
    """Full-window evidence first; risk only chooses an anchor, not a success label."""
    out = Path(plan["output"])
    s = session(plan, plan["checkpoint"])
    record, rows = RolloutRecord.start(s.world), [row(s)]
    candidates, saved, contacts = [], {}, []
    try:
        origin = source_contacts(s.scene)
        write_json(out/"contact-origin.json", origin)
        if origin["quarantined"]:
            write_json(out/"initial-state.json", dict(state=s.world.state().tolist(), actor_ids=list(s.scene.actor_ids),
                source_contacts=origin, note="Static initial state; no policy action executed"))
            write_json(out/"selection.json", dict(status="source_quarantined",reason=origin["reason"],
                source_contacts=origin, selection=None, candidates=[], physics_steps=0,
                all_world_slots=len(s.scene.actor_ids), trainable=False))
            return
        for _ in range(plan["scout_steps"]):
            if s.world.episode_done: break
            tr = s.step()
            record.append(s.world, s.last_decision["executed_actions"], tr, s.last_decision["diagnostics"])
            rows.append(row(s))
            tick = int(s.world.observe().step)
            if tick % 20 == 0:
                write_json(out/"progress.json",dict(tick=tick,target=plan["scout_steps"]))
            pairs = s.world.contacts()
            if pairs: contacts.append(dict(tick=tick,pairs=pairs))
            if tick >= 20 and tick % 20 == 0 and tick <= plan["scout_steps"]-100:
                x = risk(s)
                # Neither past world contacts nor a short suffix may enter training.
                x["eligible"] = x["eligible"] and not contacts
                candidates.append(x)
                if x["eligible"]:
                    saved[tick] = dict(runtime=s.snapshot(), prefix=copy.deepcopy(record), rows=copy.deepcopy(rows))
        # Persist executed evidence before expensive independent validation/rendering.
        # This is a diagnostic checkpoint, never an admitted training publication.
        keep = {20}
        if any(x["eligible"] for x in candidates):keep.add(select_anchor(candidates)["tick"])
        torch.save(dict(plan=plan, runtime=s.snapshot(), record=record, rows=rows,
            candidates=candidates, contacts=contacts,
            anchors={tick:saved[tick] for tick in keep if tick in saved}), out/"scout-execution.pt")
        save_branch(s,record,out/"scout",plan,"Long native student scout: no correction override")
        write_json(out/"scout/history.json",rows)
        score = official_score(s,rows,out/"scout/official-metrics",plan,"DriveRL-native-hard-scout")
        complete = len(record.actions)==plan["scout_steps"]
        report = dict(status="no_driving_failure",reason="student_did_not_fail_fixed_driving_checks",
            selection=None,candidates=candidates,contacts=contacts,score=score,
            physics_steps=len(record.actions),all_world_slots=len(s.scene.actor_ids),
            complete=complete,hard_failure_required=True,trainable=False)
        from driversi.rsi.correction.curriculum import route_support_failure
        support = route_support_failure([t.events.get("route_support_end", [False]) for t in record.transitions])
        report["support_diagnosis"] = support
        if not complete:
            report.update(status="incomplete_scout",reason="incomplete_long_window")
        elif support["blocked"]:
            report.update(status="platform_support_failure",reason=support["reason"])
        elif failed_driving(score):
            if 20 not in saved or not any(x["eligible"] for x in candidates):
                report.update(status="no_anchor",reason="failure_without_clean_supported_history")
            else:
                chosen = select_anchor(candidates)
                for name,tick in (("treatment",chosen["tick"]),("control",20)):
                    torch.save(saved[tick],out/(name+"-anchor.pt"))
                s.restore(saved[chosen["tick"]]["runtime"])
                reached = verify_reached_state(s.scene,out/"scout/rollout.npz",s.world,chosen["tick"])
                report.update(status="selected",reason=None,selection=chosen,reached=reached,
                    correction_steps=plan["scout_steps"]-chosen["tick"],automatic_selection=True,
                    learned_self_play=False,causal_diagnosis="unresolved_until_same_anchor_control")
        write_json(out/"selection.json",report)
    finally:
        s.close()

def main():
    p=argparse.ArgumentParser();p.add_argument("plan");p.add_argument("--mode",choices=("collect","correct"),required=True)
    a=p.parse_args();plan=read_mapping(a.plan);check_files(plan["files"])
    deterministic_training();torch.set_num_threads(4);torch.manual_seed(42)
    (collect if a.mode=="collect" else correct)(plan)


if __name__ == "__main__":
    main()
