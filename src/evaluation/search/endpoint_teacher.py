"""Bounded actual-history NAVSIM teacher branches with independent native PDMS.

An arrival perturbation is not learned self-play. No model/visual qualification
is inferred from successful execution; every failed branch remains inspectable.
"""

import argparse
import copy
import json
import pickle
import time
from pathlib import Path

import numpy as np


def step_arguments(action, teacher=None, *, force_native=False):
    if force_native:
        if action is not None:
            raise ValueError("Native arrival cannot silently ignore a supplied raw action")
        return {}
    if action is not None:
        return {"native_action": action, "action_origin": "bounded_perturbation"}
    return {"teacher": teacher} if teacher is not None else {}


def arrival_control(tick, backend, raw_action):
    """Build dense *executed* history before the native policy's first call."""
    if backend == "driverl_native_causal_history":
        return None, True
    if backend == "driverl_native":
        return ([0.0, 0.0], False) if tick < 8 else (None, True)
    return raw_action, False


def run(
    scene_path,
    graph_path,
    output,
    *,
    checkpoint,
    maps_root,
    arrival_action=(0.0, 0.0),
    arrival_backend="bounded_control",
    device="cuda:0",
    search_candidates=0,
    sampling_method="argmax",
    teacher_backend="driverl_native",
    recovery_seconds=2.0,
    position_gain=0.0,
    seed=42,
    admission_profile="strict_world95_v1",
    history_perturbation=None,
    max_wall_seconds=900,
):
    if teacher_backend != "source_endpoint_tto_v1":
        raise ValueError("This versioned runner only supports endpoint TTO")
    import torch

    from driversi.rsi.correction.trajectory_admission import (
        trajectory_policy,
        trajectory_quality_pass,
    )

    from driversi_agents.driverl.random_state import validate_sampling
    validate_sampling(sampling_method, search_candidates=search_candidates)
    if sampling_method == "sample" and (arrival_backend != "driverl_native_causal_history" or teacher_backend != "driverl_native"):
        raise ValueError("Stochastic native generation requires full-window native control")
    trajectory_policy(admission_profile)
    if type(max_wall_seconds) is not int or not 300 <= max_wall_seconds <= 1800:
        raise ValueError("Teacher task budget must be between 300 and 1800 seconds")
    from driversi.rsi.correction.epdms_admission import required_for
    capture_planning = required_for(output)
    if capture_planning and (arrival_backend != "driverl_native_causal_history"
                             or teacher_backend != "driverl_native"):
        raise ValueError("Complete EC capture currently requires a full-window native DriveRL generator")

    from driversi.agents.backends.native import digest, project_root
    from driversi.contracts.road import RoadGraph
    from driversi.contracts.scene import Scene
    from driversi.evaluation.metrics.counterfactual_pdm import prepare_reference, score_execution
    from driversi.rendering.bev.replay import export_replay
    from driversi.sim.persistence.recording import RolloutRecord, write_rollout
    from driversi_agents.driverl.closed_loop import DriveRLSession, ego_vector, unpack_objects
    from driversi_agents.driverl.navsim_context import verify_current_input

    if arrival_backend not in (
        "bounded_control",
        "driverl_native",
        "driverl_native_causal_history",
        "source_history_perturbation_v1",
    ):
        raise ValueError("Unknown arrival backend")
    perturbed_history = arrival_backend == "source_history_perturbation_v1"
    if perturbed_history:
        from driversi.rsi.correction.history_perturbation import validate_perturbation

        history_perturbation = validate_perturbation(history_perturbation)
        if not ((teacher_backend == "driverl_native" and search_candidates in (8, 64)) or (teacher_backend == "source_endpoint_tto_v1" and search_candidates == 0)):
            raise ValueError("Perturbed history requires native future TTS")
    elif history_perturbation is not None:
        raise ValueError("Historical proposal supplied to a different backend")
    if teacher_backend == "source_endpoint_tto_v1" and not perturbed_history:
        raise ValueError("Endpoint TTO requires a bounded actual historical proposal")
    native_arrival = arrival_backend in ("driverl_native", "driverl_native_causal_history")
    full_native = arrival_backend == "driverl_native_causal_history"
    if (native_arrival or perturbed_history) and any(v != 0 for v in arrival_action):
        raise ValueError("Native arrival requires an unused zero raw-action placeholder")
    if teacher_backend not in (
        "driverl_native",
        "pdm_route_v1",
        "bounded_native_search_v1",
        "source_corridor_lqr_v1",
        "source_endpoint_tto_v1",
    ) or (teacher_backend != "driverl_native" and search_candidates != 0):
        raise ValueError("Invalid teacher/search combination")
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError("Invalid native sampling seed")
    started = time.monotonic()
    root = project_root()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    scene = Scene.load(Path(scene_path))
    graph = RoadGraph.load(Path(graph_path))
    if (full_native or perturbed_history) and scene.metadata.get(
        "driverl_history_mode"
    ) != "causal_resampled_10hz_v1":
        raise ValueError("Full native control requires sealed causal source-history initialization")
    if full_native and teacher_backend != "driverl_native":
        raise ValueError("All-window DriveRL cannot use a substitute future controller")
    report = dict(
        schema_version=1,
        kind="navsim_native_teacher_pair",
        status="running",
        source_log=scene.metadata["source_log"],
        token=scene.metadata["scenario_id"],
        arrival_tick=20,
        sampling_seed=seed,
        sampling_method=sampling_method,
        admission_profile=admission_profile,
        history_ticks=[5, 10, 15, 20],
        label_ticks=list(range(25, 61, 5)),
        native_training_future_ticks=list(range(25, 71, 5)),
        arrival_method="source_history_C2_reference_executed_native_LQR"
        if perturbed_history
        else "native_DriveRL_from_t0_with_declared_past_resampling"
        if full_native
        else "physical_warmup_then_native_DriveRL"
        if native_arrival
        else "bounded_continuous_control",
        arrival_backend=arrival_backend,
        arrival_warmup_ticks=8 if arrival_backend == "driverl_native" else 0,
        native_arrival_policy_ticks=20 if full_native else 12 if native_arrival else 0,
        learned_self_play=False,
        teacher=(
            "bounded_native_control_rollout_search"
            if teacher_backend == "bounded_native_search_v1"
            else "bounded_training_source_endpoint_TTO_native_LQR"
            if teacher_backend == "source_endpoint_tto_v1"
            else "privileged_training_source_corridor_LQR"
            if teacher_backend == "source_corridor_lqr_v1"
            else "PDM_closed_route_LQR"
            if teacher_backend == "pdm_route_v1"
            else "DriveRL_native_tts"
            if search_candidates
            else "DriveRL_native_sample" if sampling_method == "sample"
            else "DriveRL_native_argmax"
        ),
        teacher_backend=teacher_backend,
        student_evaluated=False,
        baseline_role="native_DriveRL_continuation_not_student"
        if native_arrival or perturbed_history
        else "continued_arrival_perturbation_not_student",
        history_change_rule="any_history_xy_deviation_gt_0.05m",
        trainable=False,
        branches={},
        charged_joint_steps=0,
    )

    def dump():
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))

    dump()
    session = None
    try:
        torch.set_num_threads(2)
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        session = DriveRLSession(
            scene,
            repository=root / "agents/driverl/native/DriveRL",
            checkpoint=checkpoint,
            maps_root=maps_root,
            device=device,
            search_candidates=search_candidates,
            sampling_method=sampling_method,
            search_seed=seed,
            search_start=(0 if full_native else 20) if search_candidates else None,
            search_stop=70 if search_candidates else None,
        )
        if full_native or perturbed_history:
            report["initial_native_parity"] = verify_current_input(session)
            report["source_history_bootstrap"] = session.scenario.driversi_history_evidence
        arrival_controller = None
        if perturbed_history:
            from driversi_agents.driverl.history_perturbation import HistoryPerturbation

            arrival_controller = HistoryPerturbation(
                root, session.initialization, session.scene, session.ego, history_perturbation
            )
            report["arrival_configuration"] = arrival_controller.description
            (output / "history-reference.json").write_text(
                json.dumps(arrival_controller.reference, allow_nan=False)
            )
        teacher_controller = None
        if teacher_backend == "pdm_route_v1":
            from driversi_agents.driverl.repair import PDMRouteRepair

            teacher_controller = PDMRouteRepair(root, session.initialization)
        session.scene.save(output / "scene.npz")
        graph.save(output / "road-graph.json")
        initial = session.world.snapshot()
        initial.save(output / "initial-world.npz")
        record = RolloutRecord.start(session.world)
        targets = []
        prefix_contacts = [{"tick": 0, "actors": list(pair)} for pair in session.world.contacts()]

        def advance(action=None, *, force_native=False, arrival=False):
            if time.monotonic() - started > max_wall_seconds:
                raise TimeoutError("Bounded teacher task exceeded its declared wall budget")
            if not session.world.observe().alive[0]:
                raise ValueError("Ego already terminated; no silent stationary continuation")
            tr = session.step(
                **step_arguments(
                    action,
                    arrival_controller if arrival else teacher_controller,
                    force_native=force_native,
                )
            )
            report["charged_joint_steps"] += 1
            record.append(
                session.world,
                session.last_decision["executed_actions"],
                tr,
                session.last_decision["diagnostics"],
            )
            targets.append(copy.deepcopy(session.parity[-1]["native_targets"]))
            return tr

        def save(name):
            write_rollout(
                session.scene,
                output / (name + ".npz"),
                record=record,
                snapshot=initial,
                controlled_ids=("ego",),
                policy=(
                    "bounded-control-search-candidate"
                    if name.startswith("candidate-")
                    else report["arrival_method"]
                    if name != "teacher"
                    else report["arrival_method"] + "_then_" + report["teacher"]
                ),
                controller_configuration={
                    **session.configuration,
                    "arrival_configuration": arrival_controller.description
                    if arrival_controller
                    else None,
                    "executed_teacher_backend": teacher_backend
                    if name == "teacher" or name.startswith("candidate-")
                    else None,
                    "teacher_configuration": teacher_controller.description
                    if (name == "teacher" or name.startswith("candidate-")) and teacher_controller
                    else None,
                    "native_model_shadow_only_after_anchor": bool(
                        name == "teacher" and teacher_controller
                    ),
                },
                protocol={
                    "kind": report["kind"],
                    "branch": name,
                    "trainable": False,
                    "arrival_controller": report["arrival_method"],
                    "arrival_ticks": [0, 20],
                    "future_controller": report["teacher"]
                    if name == "teacher" or name.startswith("candidate-")
                    else report["arrival_method"],
                    "future_ticks": [20, len(record.actions)],
                },
            )
            (output / (name + "-targets.json")).write_text(json.dumps(targets, allow_nan=False))
            export_replay(
                session.scene, output / (name + ".npz"), output / (name + ".html"), road_graph=graph
            )

        previous_planning_anchor = None
        for arrival_tick in range(20):
            action, native = arrival_control(arrival_tick, arrival_backend, arrival_action)
            tr = advance(
                None if perturbed_history else action,
                force_native=False if perturbed_history else native,
                arrival=perturbed_history,
            )
            prefix_contacts.extend(
                {"tick": session.world.observe().step, "actors": list(pair)}
                for pair in session.world.contacts()
            )
            if not tr.observation.alive[0]:
                report.update(
                    status="rejected",
                    reason="arrival_ego_terminated",
                    arrival_events={k: v.tolist() for k, v in tr.events.items()},
                )
                save("arrival-failure")
                return report
            if capture_planning and arrival_tick == 14:
                previous_planning_anchor = session.snapshot()
        if perturbed_history:
            report["actual_anchor"] = dict(
                time_us=session.ego.time_us,
                rear_axle_xy=session.ego.rear_axle.array[:2].tolist(),
                speed_mps=float(session.ego.dynamic_car_state.rear_axle_velocity_2d.x),
                endpoint_error_m=float(
                    np.linalg.norm(
                        session.ego.rear_axle.array[:2]
                        - np.asarray(arrival_controller.description["requested_endpoint_xy"])
                    )
                ),
                tts_start_tick=20,
                no_state_reset_at_handoff=True,
            )
        report["arrival_contacts"] = prefix_contacts
        report["parity"] = verify_current_input(session)
        anchor = session.snapshot()
        planning_pair = None
        if capture_planning:
            from driversi.evaluation.search.driverl_planning import capture_pair
            planning_pair = capture_pair(
                session, previous_planning_anchor, anchor, output / "adjacent-native-plans.json",
                deadline=started + max_wall_seconds,
            )
            report["charged_joint_steps"] += 80
            report["independent_planning"] = dict(
                producer=planning_pair["producer"], policy_steps=80,
                anchors_us=[planning_pair[k]["anchor_timestamp_us"] for k in ("previous", "current")],
                capture=planning_pair["capture_file"], main_branch_restored=True,
            )
        anchor["world"].save(output / "anchor-world.npz")
        prefix = copy.deepcopy(record)
        prefix_targets = copy.deepcopy(targets)
        history = np.stack(record.states)[report["history_ticks"], 0, :4]
        reference = session.scene.poses[report["history_ticks"], 0, :4]
        report["history_xy_deviation_m"] = np.linalg.norm(
            history[:, :2] - reference[:, :2], axis=1
        ).tolist()
        report["history_changed"] = bool(
            np.any(np.linalg.norm(history[:, :2] - reference[:, :2], axis=1) > 0.05)
        )
        if teacher_backend == "source_corridor_lqr_v1":
            from driversi_agents.driverl.source_corridor import SourceCorridorRepair

            teacher_controller = SourceCorridorRepair(
                root,
                session.initialization,
                session.scene,
                session.ego,
                recovery_seconds,
                position_gain,
            )
            report["teacher_configuration"] = teacher_controller.description
        native_prefix_history = copy.deepcopy(session.history)
        context = prepare_reference(session.planner_input(), session.initialization)
        truth = []
        candidate_actions = {}
        if teacher_backend == "bounded_native_search_v1":
            candidate_actions = {
                f"candidate-{i:02d}": [jerk, steering]
                for i, (jerk, steering) in enumerate(
                    (j, s) for j in (-0.15, 0.0, 0.15) for s in (-0.02, 0.0, 0.02)
                )
            }
        endpoint_candidates = {}
        if teacher_backend == "source_endpoint_tto_v1":
            from driversi.rsi.correction.endpoint_tto import endpoint_grid
            endpoint_candidates = {f"candidate-{i:02d}": offset
                for i, offset in enumerate(endpoint_grid(history_perturbation))}
        report["search_endpoint_candidates"] = endpoint_candidates
        report["search_candidate_actions"] = candidate_actions
        selected_action = None
        search_keys = list(endpoint_candidates or candidate_actions)
        for name in ("baseline", *search_keys, "teacher"):
            if name == "teacher" and search_keys:
                valid = [
                    key
                    for key in search_keys
                    if report["branches"][key]["trajectory_admitted"]
                ]
                if not valid:
                    report.update(
                        status="rejected",
                        reason="no_admitted_executed_control_candidate",
                        qualified_teacher_branch=False,
                    )
                    return report
                selected = max(valid, key=lambda key: report["branches"][key]["score"]["score"])
                report["selected_candidate"] = selected
                selected_action = candidate_actions.get(selected)
            session.restore(anchor)
            if endpoint_candidates and name != "baseline":
                from driversi.evaluation.search.source_endpoint_tto import SourceEndpointTTO
                teacher_controller = SourceEndpointTTO(root, session.initialization,
                    session.scene, session.ego, endpoint_candidates[selected if name == "teacher" else name])
                if name == "teacher":
                    report["teacher_configuration"] = teacher_controller.description
            record = copy.deepcopy(prefix)
            targets = copy.deepcopy(prefix_targets)
            egos = [session.ego]
            objects = [unpack_objects(session._objects(), session.ego.time_us)]
            lights = [list(session.planner_input().traffic_light_data)]
            events = []
            contacts = []
            reason = None
            for i in range(50):
                action = (
                    (None if native_arrival or perturbed_history else arrival_action)
                    if name == "baseline"
                    else candidate_actions[name]
                    if name in candidate_actions
                    else selected_action
                )
                tr = advance(
                    action,
                    force_native=name == "baseline" and (native_arrival or perturbed_history),
                )
                egos.append(session.ego)
                objects.append(unpack_objects(session._objects(), session.ego.time_us))
                lights.append(list(session.planner_input().traffic_light_data))
                events.append({k: bool(v[0]) for k, v in tr.events.items()})
                contacts.extend(
                    {"tick": session.world.observe().step, "actors": list(pair)}
                    for pair in session.world.contacts()
                )
                if name == "teacher" and i < 5:
                    truth.append((session.world.state().copy(), ego_vector(session.ego)))
                if not tr.observation.alive[0] and i < 49:
                    reason = "ego_terminated_before_complete_score_window"
                    break
            row = {
                "completed_steps": len(events),
                "failure": reason,
                "object_contacts": contacts,
                "events": {k: sum(e.get(k, False) for e in events) for k in events[0]},
                "score": None,
            }
            if reason is None:
                row["score"] = score_execution(context, egos, objects)
            row["admission_profile"] = admission_profile
            row["trajectory_admitted"] = bool(
                reason is None
                and len(events) == 50
                and report["history_changed"]
                and trajectory_quality_pass(
                    row["score"],
                    profile=admission_profile,
                    arrival_contacts=len(prefix_contacts),
                    future_contacts=len(contacts),
                    events=row["events"],
                )
            )
            if name == "teacher" and teacher_backend == "source_corridor_lqr_v1":
                from driversi_agents.driverl.source_corridor import corridor_evidence

                row["source_corridor"] = corridor_evidence(
                    [[e.rear_axle.x, e.rear_axle.y] for e in egos],
                    teacher_controller.source_xy(egos),
                )
                if admission_profile == "strict_world95_v1":
                    row["trajectory_admitted"] = (
                        row["trajectory_admitted"] and row["source_corridor"]["passed"]
                    )
            report["branches"][name] = row
            save(name)
            if name == "teacher" and reason is None:
                # Internal trusted artifact, sealed by Harness with the matching
                # rollout. Native training must never reuse a logged cache here.
                cache_context = output / "teacher-counterfactual-context.pkl"
                cache_context.write_bytes(
                    pickle.dumps(
                        dict(
                            schema_version=1,
                            epdms_context_version=1,
                            kind="executed_counterfactual_context",
                            context=context,
                            egos=egos,
                            observations=objects,
                            traffic_lights=lights,
                            prefix_history=native_prefix_history,
                            adjacent_native_plans=planning_pair,
                            anchor_tick=20,
                            branch_files={
                                str(output / "scene.npz"): digest(output / "scene.npz"),
                                str(output / "teacher.npz"): digest(output / "teacher.npz"),
                            },
                        ),
                        protocol=4,
                    )
                )
                row["counterfactual_context"] = dict(
                    path=str(cache_context), sha256=digest(cache_context)
                )
                from driversi.rsi.correction.epdms_admission import required_for, run as run_epdms

                if required_for(output):
                    pdms_admitted = row["trajectory_admitted"]
                    row["trajectory_admitted"] = False
                    row["epdms_status"] = "awaiting_complete_native_score"
                    epdms = run_epdms(cache_context, output / "teacher-epdms.json")
                    row["score"]["epdms"] = epdms
                    row["epdms_status"] = "passed" if epdms["passed"] else "rejected"
                    row["trajectory_admitted"] = pdms_admitted and epdms["passed"]

            dump()
        session.restore(anchor)
        for state, ego in truth:
            session.step(
                **(
                    {"native_action": selected_action, "action_origin": "bounded_perturbation"}
                    if selected_action is not None
                    else {"teacher": teacher_controller}
                    if teacher_controller
                    else {}
                )
            )
            report["charged_joint_steps"] += 1
            np.testing.assert_array_equal(session.world.state(), state)
            np.testing.assert_array_equal(ego_vector(session.ego), ego)
        report["restore_exact"] = True
        baseline = report["branches"]["baseline"]
        teacher = report["branches"]["teacher"]
        report["perturbation_recovery_demonstrated"] = bool(
            arrival_backend == "bounded_control"
            and teacher["trajectory_admitted"]
            and (baseline["failure"] or not baseline["trajectory_admitted"])
        )
        report["correction_demonstrated"] = False
        report.update(
            status="completed",
            qualified_teacher_branch=teacher["trajectory_admitted"],
            visual_admission="not_run",
        )
        return report
    except Exception as exc:
        report.update(status="failed", reason=type(exc).__name__ + ": " + str(exc))
        if session and "record" in locals() and len(record.actions):
            try:
                save("partial-failure")
            except Exception as save_error:
                report["recording_error"] = str(save_error)
        return report
    finally:
        report["wall_seconds"] = time.monotonic() - started
        report["input_files"] = {
            str(Path(p).resolve()): digest(p) for p in (scene_path, graph_path, checkpoint)
        }
        if session:
            session.close()
        dump()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--road-graph", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--maps-root", required=True)
    p.add_argument("--arrival-action", type=float, nargs=2, default=[0.0, 0.0])
    p.add_argument(
        "--arrival-backend",
        choices=(
            "bounded_control",
            "driverl_native",
            "driverl_native_causal_history",
            "source_history_perturbation_v1",
        ),
        default="bounded_control",
    )
    p.add_argument("--history-perturbation", type=json.loads)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-wall-seconds", type=int, default=900)
    p.add_argument("--search-candidates", type=int, default=0)
    p.add_argument("--sampling-method", choices=("argmax", "sample"), default="argmax")
    p.add_argument(
        "--teacher-backend",
        choices=(
            "driverl_native",
            "pdm_route_v1",
            "bounded_native_search_v1",
            "source_corridor_lqr_v1",
        "source_endpoint_tto_v1",
        ),
        default="driverl_native",
    )
    p.add_argument("--recovery-seconds", type=float, choices=(2.0, 3.0), default=2.0)
    p.add_argument("--position-gain", type=float, choices=(0.0, 0.5, 1.0), default=0.0)
    p.add_argument(
        "--admission-profile",
        default="strict_world95_v1",
        choices=("strict_world95_v1", "pdms_gt95_v1"),
    )
    a = p.parse_args()
    result = run(
        a.scene,
        a.road_graph,
        a.output,
        checkpoint=a.checkpoint,
        maps_root=a.maps_root,
        arrival_action=a.arrival_action,
        arrival_backend=a.arrival_backend,
        history_perturbation=a.history_perturbation,
        device=a.device,
        search_candidates=a.search_candidates,
        sampling_method=a.sampling_method,
        teacher_backend=a.teacher_backend,
        recovery_seconds=a.recovery_seconds,
        position_gain=a.position_gain,
        seed=a.seed,
        admission_profile=a.admission_profile,
        max_wall_seconds=a.max_wall_seconds,
    )
    print(json.dumps(result, indent=2))
    return 0 if result["status"] in ("completed", "rejected") else 1


if __name__ == "__main__":
    raise SystemExit(main())
