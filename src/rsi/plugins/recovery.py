"""Typed DriveHarness path from mined anchors to admitted online curricula."""
import copy
from pathlib import Path

from driversi.agents.backends.native import digest, project_root
from driversi.configuration import read_mapping
from driversi.rsi.contracts import _keys
from driversi.rsi.plugins import collection
from driversi.rsi.workflow.results import result
from driversi.rsi.plugins.synthesis import check_files, protect, run_process
from driversi.runtime.budget import write_json
from driversi.training.contracts.job import resolve_request
from driversi.training.orchestration import lifecycle


def prepare_selection(config, base, registry):
    hard = config.get("hard_failure_required", False)
    if type(hard) is not bool: raise ValueError("Invalid hard failure mode")
    _keys({k:v for k,v in config.items() if k not in ("hard_failure_required", "perturbation")}, ("scene", "road_graph", "source_registry", "scout_steps", "max_wall_seconds"), "automatic recovery selection")
    if type(config["scout_steps"]) is not int or not ((300 <= config["scout_steps"] <= 600) if hard else (25 <= config["scout_steps"] <= 80)):
        raise ValueError("Hard mining requires 300-600 steps; short compatibility mode requires 25-80")
    compatible = {k:v for k,v in config.items() if k not in ("scout_steps", "hard_failure_required", "perturbation")}
    wall=config["max_wall_seconds"]
    if type(wall) not in (int,float) or not 0<wall<=(1800 if hard else 900):
        raise ValueError("Bounded native scouting time required")
    compatible["max_wall_seconds"] = min(wall,900)
    compatible["anchor_tick"] = 20
    p, _ = collection.prepare(compatible, base, registry)
    from driversi.contracts.scene import Scene
    scene = Scene.load(Path(p["scene"]))
    if len(scene.times) <= config["scout_steps"]+(0 if hard else 40):
        raise ValueError("Mining requires full future correction support")
    p["max_wall_seconds"] = wall
    p["scout_steps"] = config["scout_steps"]
    p["hard_failure_required"] = hard
    if "perturbation" in config:
        if not hard: raise ValueError("Active generation requires genuine failure verification")
        from driversi.rsi.correction.perturbation import validate_perturbation
        p["perturbation"] = validate_perturbation(config["perturbation"], config["scout_steps"])
        # Freeze generation, native wrapper, teacher and admission code across stages.
        root=project_root()
        for name in ("agents/driverl/active_recovery.py","agents/driverl/recovery.py",
                     "agents/driverl/closed_loop.py","agents/driverl/repair.py",
                     "src/rsi/correction/perturbation.py","src/rsi/correction/curriculum.py",
                     "src/evaluation/metrics/nuplan_metrics.py", "src/rsi/plugins/recovery.py",
                     "src/rsi/plugins/long_recovery.py", "agents/driverl/learning/learner.py"):
            path=str(root/name);p["files"][path]=digest(path)
    p["sim_cap"] = config["scout_steps"] * len(scene.actor_ids)
    return p, p["sim_cap"]


def _worker(plan, output, mode):
    write_json(output/"worker-plan.json", plan)
    cfg=plan["model_config"]
    run_process([cfg["python"], "-B", "-m", "driversi_agents.driverl.recovery", str(output/"worker-plan.json"), "--mode", mode],
        output, plan["max_wall_seconds"], environment={"CUDA_VISIBLE_DEVICES":str(cfg["settings"]["gpu"]),"CUBLAS_WORKSPACE_CONFIG":":4096:8"})
    check_files(plan["files"])


def _artifacts(output):
    return {str(p.relative_to(output)):str(p) for p in output.rglob("*") if p.is_file()}


def run_selection(config, inputs, output, registry):
    check_files(config["files"])
    student=read_mapping(inputs["student"]["artifacts"]["report"])
    training=read_mapping(inputs["student"]["artifacts"]["plan"])
    collection.validate_student(student, training, config["source_groups"][0])
    protect(config["source_groups"], registry, extra=student["source_groups"]["development"])
    check_files(training["files"])
    output.mkdir(parents=True)
    plan={**config,"output":str(output),"checkpoint":student["checkpoint"],
          "model_config":training["model_config"], "source_groups":student["source_groups"],
          "files":{**config["files"], **student["checkpoint_files"]},
          "student_identity":inputs["student"]["summary"]["artifact_reference"]["identity"],
          "mined_source_log":config["source_groups"][0],
          "parent_source_groups":student["source_groups"]}
    _worker(plan,output,"collect")
    return selection_result(output, read_mapping(output/"selection.json"))


def selection_result(output, report):
    artifacts={**_artifacts(output), "plan":str(output/"worker-plan.json"),
               "selection":str(output/"selection.json")}
    steps=report["physics_steps"]*report["all_world_slots"]
    if report["status"] in ("no_anchor", "control_anchor_unavailable", "source_quarantined", "no_driving_failure", "incomplete_scout", "platform_support_failure", "arrival_rejected"):
        return result(output,report,status="rejected",reason=report["reason"],steps=steps,artifacts=artifacts)
    if report["status"] != "selected" or not report.get("selection"):
        raise ValueError("Unexpected selection outcome")
    artifacts.update(anchor=str(output/"treatment-anchor.pt"),control_anchor=str(output/"control-anchor.pt"))
    return result(output,report,steps=steps,artifacts=artifacts)


def prepare_correction(config,base,registry):
    _keys({k:v for k,v in config.items() if k != "backend"},("max_wall_seconds",),"same-anchor correction")
    backend=config.get("backend", "pdm_route_v1")
    if backend not in ("pdm_route_v1", "native_recovery_search_v1"):
        raise ValueError("Unknown recovery teacher backend")
    if type(config["max_wall_seconds"]) not in (int,float) or not 0<config["max_wall_seconds"]<=1200:
        raise ValueError("Bounded teacher budget required")
    prepared={**config,"backend":backend}
    if backend == "native_recovery_search_v1":
        path=str(project_root()/"agents/driverl/control_lattice.py")
        prepared["teacher_files"]={path:digest(path)}
    return prepared,1200*1024


def run_correction(config,inputs,output,registry):
    check_files(config.get("teacher_files", {}))
    evidence=inputs["selection"];old=read_mapping(evidence["artifacts"]["plan"])
    protect(old["source_groups"]["train"],registry,extra=old["source_groups"]["development"])
    output.mkdir(parents=True)
    if old.get("hard_failure_required"):
        # Recheck raw executed masks, including artifacts produced before this guard.
        import numpy as np
        from driversi.rsi.correction.curriculum import route_support_failure
        with np.load(evidence["artifacts"]["scout/rollout.npz"], allow_pickle=False) as record:
            support = route_support_failure(record["route_support_end"])
        if support["blocked"]:
            return result(output,{"support_diagnosis":support,"teacher_executed":False},
                status="rejected",reason=support["reason"])
    plan={**old,**config,"output":str(output),"anchor":evidence["artifacts"]["anchor"],
          "files":{**old["files"],**config.get("teacher_files", {}),**{p:digest(p) for p in evidence["artifacts"].values()}},
          "selection_identity":evidence["summary"]["artifact_reference"]["identity"]}
    selection_report=read_mapping(evidence["artifacts"]["selection"])
    if old.get("hard_failure_required"):
        plan["correction_steps"]=selection_report["correction_steps"]
    _worker(plan,output,"correct")
    report=read_mapping(output/"correction.json")
    return result(output,report,steps=report["physics_steps"]*report["all_world_slots"],
        artifacts={**_artifacts(output),"plan":str(output/"worker-plan.json"),
                   "teacher_report":str(output/"teacher/report.json"),"teacher_rollout":str(output/"teacher/rollout.npz")})


def prepare_dataset(config,base,registry):
    _keys(config,("request",),"curriculum dataset")
    resolved=resolve_request((base/config["request"]).resolve())
    if resolved["model"]!="driverl" or resolved["request"]["mode"]!="online_rl":
        raise ValueError("Only native online curriculum is supported")
    data=read_mapping((Path(resolved["request_path"]).parent/resolved["request"]["data"]).resolve())
    groups=lifecycle.sources(data)
    return {"request":resolved["request_path"],"files":{resolved["request_path"]:resolved["request_sha256"]},
            "source_groups":groups["train"]+groups["development"]},0


def inherited_files(*plans):
    """Carry original expected hashes, not freshly trusted post-correction content."""
    files={}
    for plan in plans:
        check_files(plan["files"])
        for path,expected in plan["files"].items():
            if path in files and files[path]!=expected:
                raise ValueError("Conflicting inherited file version: "+path)
            files[path]=expected
    return files


def run_dataset(config,inputs,output,registry):
    from driversi.rsi.correction.curriculum import admit
    check_files(config["files"])
    selection, correction=inputs["selection"],inputs["correction"]
    plan=read_mapping(selection["artifacts"]["plan"])
    corrected=read_mapping(correction["artifacts"]["plan"])
    frozen=inherited_files(plan,corrected)
    if corrected["selection_identity"]!=selection["summary"]["artifact_reference"]["identity"]:
        raise ValueError("Correction came from another reached state")
    teacher=read_mapping(correction["artifacts"]["teacher_report"])
    admission=admit(teacher["execution"],teacher["score"])
    output.mkdir(parents=True)
    if plan.get("perturbation"):
        generation=read_mapping(selection["artifacts"]["generation.json"])
        if not generation.get("physical_effect_verified") or generation.get("execution_error"):
            return result(output,{"generation":generation},status="rejected",reason="unverified_physical_generation")
    outcome=read_mapping(correction["artifacts"]["report"])
    genuine=outcome.get("genuine", {})
    if corrected.get("hard_failure_required") and not genuine.get("accepted"):
        return result(output,{"admission":admission,"genuine":genuine},status="rejected",reason="genuine_failure_not_corrected")
    if not admission["accepted"]:
        return result(output,admission,status="rejected",reason="counterfactual_teacher_not_admitted")
    resolved=resolve_request(config["request"])
    data=read_mapping((Path(resolved["request_path"]).parent/resolved["request"]["data"]).resolve())
    groups=lifecycle.sources(data)
    from driversi.training.sampling.environment import source_log_group
    mined=source_log_group(plan.get("mined_source_log",data["train"]["source_logs"][0]))
    parents=plan.get("parent_source_groups",plan["source_groups"])
    required_train=set(parents["train"])|{mined}
    if (set(groups["train"])!=required_train or not set(parents["development"]).issubset(groups["development"])
        or digest(data["train"]["scene"])!=digest(plan["scene"])):
        raise ValueError("Curriculum and learner source groups differ")
    from driversi.contracts.scene import Scene
    source_log=Scene.load(Path(plan["scene"])).metadata["source_log"]
    if data["train"]["source_logs"][0]!=source_log:raise ValueError("Primary training source differs from reset scene")
    protect(groups["train"],registry,extra=groups["development"])
    chosen=read_mapping(selection["artifacts"]["selection"])["selection"]
    manifest=dict(schema_version=1,purpose="feasible_online_reset_curriculum",admission=admission,
        source_log=data["train"]["source_logs"][0],scene_sha256=digest(plan["scene"]),
        parent_checkpoint=plan["checkpoint"],parent_checkpoint_sha256=digest(plan["checkpoint"]),
        correction_report=correction["artifacts"]["teacher_report"],
        selected_branch=correction["artifacts"]["teacher_rollout"],
        hard_failure_required=corrected.get("hard_failure_required",False),
        paired_correction_report=correction["artifacts"]["report"],
        selection_identity=corrected["selection_identity"],
        correction_identity=correction["summary"]["artifact_reference"]["identity"],
        starts={"control":{"snapshot":selection["artifacts"]["control_anchor"],"tick":20},
                "treatment":{"snapshot":selection["artifacts"]["anchor"],"tick":chosen["tick"]}},
        teacher_labels_use="feasibility_reference_not_PPO_transitions",
        render={"status":"not_requested_for_structured_RL","visual_trainable":False},
        files={**{p:digest(p) for evidence in (selection,correction) for p in evidence["artifacts"].values()},**frozen})
    write_json(output/"curriculum.json",manifest)
    model=copy.deepcopy(plan["model_config"]);model["weights"]["checkpoint"]=plan["checkpoint"]
    write_json(output/"model.json",model)
    artifacts={"manifest":str(output/"curriculum.json")}
    for arm in ("control","treatment"):
        d=copy.deepcopy(data);d["train"].update(curriculum=str(output/"curriculum.json"),curriculum_arm=arm)
        write_json(output/(arm+"-data.json"),d)
        request=copy.deepcopy(resolved["request"])
        request.update(id="recovery-"+arm,model_config=str(output/"model.json"),data=str(output/(arm+"-data.json")))
        write_json(output/(arm+"-request.json"),request)
        artifacts[arm+"_request"]=str(output/(arm+"-request.json"))
    return result(output,manifest,artifacts={**_artifacts(output),**artifacts})


def prepare_train(config,base,registry):
    _keys(config,("arm",),"curriculum learner")
    if config["arm"] not in ("control","treatment"):
        raise ValueError("Unknown paired arm")
    return config,512*1024


def _learning_result(output,report):
    return dict(status=report["status"],reason=report.get("reason"),
        sim_agent_steps=report["consumed"].get("sim_agent_steps",report["consumed"].get("environment_steps",0)),
        artifacts={"report":str(output/"training-result.json"),"plan":str(output/"training-plan.json")},
        summary=report,admission={"trainable":False,"promoted":False})


def run_train(config,inputs,output,registry):
    evidence=inputs["dataset"];manifest=read_mapping(evidence["artifacts"]["manifest"])
    if not manifest["admission"]["accepted"]:
        raise ValueError("Rejected teacher cannot enter training")
    resolved=resolve_request(evidence["artifacts"][config["arm"]+"_request"])
    plan=lifecycle.prepare(resolved,output)
    protect(plan["source_groups"]["train"],registry,extra=plan["source_groups"]["development"])
    return _learning_result(output,lifecycle.execute(plan,own_process_group=False))


def prepare_empty(config,base,registry):
    if config:
        raise ValueError("No runtime score or loss overrides")
    return {},512*1024


def _continue(inputs,output,registry,phase):
    previous=inputs["student"]["artifacts"]["report"]
    old=read_mapping(inputs["student"]["artifacts"]["plan"])
    check_files(old["files"])
    protect(old["source_groups"]["train"],registry,extra=old["source_groups"]["development"])
    plan=lifecycle.prepare(resolve_request(old["resolved"]["request_path"]),output,phase=phase,previous=previous)
    return _learning_result(output,lifecycle.execute(plan,own_process_group=False))


def run_resume(config,inputs,output,registry):
    return _continue(inputs,output,registry,"resume")


def run_evaluation(config,inputs,output,registry):
    return _continue(inputs,output,registry,"evaluate")


def validate_arms(control, treatment):
    for name, arm in (("control", control), ("treatment", treatment)):
        if arm["plan"]["data"]["train"].get("curriculum_arm") != name:
            raise ValueError("Paired arm does not match input port: " + name)
    if control["plan"]["data"]["development"] != treatment["plan"]["data"]["development"]:
        raise ValueError("Paired development scene differs")


def run_feedback(config,inputs,output,registry):
    from driversi.rsi.decisions.weaknesses import review
    arms={}
    for name in ("control","treatment"):
        s=read_mapping(inputs[name+"_student"]["artifacts"]["report"])
        e=read_mapping(inputs[name+"_evaluation"]["artifacts"]["report"])
        p=read_mapping(inputs[name+"_student"]["artifacts"]["plan"])
        f=lifecycle.feedback(s,e)
        arms[name]=dict(student=s,evaluation=e,plan=p,feedback=f,weaknesses=review(s,e,f))
    a,b=arms["control"],arms["treatment"]
    validate_arms(a,b)
    for key in ("model_config","source_groups"):
        if a["plan"][key]!=b["plan"][key]:raise ValueError("Paired "+key+" differs")
    for key in ("pilot","budget"):
        if a["plan"]["resolved"]["request"][key]!=b["plan"]["resolved"]["request"][key]:raise ValueError("Paired budget differs")
    if a["student"]["total_updates"]!=b["student"]["total_updates"] or a["evaluation"]["metrics"]["parent"]!=b["evaluation"]["metrics"]["parent"]:
        raise ValueError("Paired updates/baseline differ")
    replay=read_mapping(inputs["replay"]["artifacts"]["report"])
    trained=b["student"]
    if replay["context"]!=trained["context"] or replay["resume_from"]!=trained["resume_from"]:
        raise ValueError("Replay from a different training boundary")
    resume=dict(different_processes=replay["worker_pid"]!=trained["worker_pid"],same_state=replay["state_fingerprint"]==trained["state_fingerprint"])
    deltas={k:b["evaluation"]["metrics"]["candidate"][k]-a["evaluation"]["metrics"]["candidate"][k] for k in b["evaluation"]["directions"]}
    manifest=read_mapping(inputs["dataset"]["artifacts"]["manifest"])
    for arm in arms.values():
        if arm["plan"]["data"]["train"]["curriculum"]!=inputs["dataset"]["artifacts"]["manifest"]:
            raise ValueError("Feedback dataset not consumed by learner")
    if b["evaluation"].get("metric_scope")=="official_nuplan_metrics_custom_long_logs_not_benchmark":
        from driversi.rsi.plugins.long_recovery import benefit, verify_evaluated_checkpoint
        for arm in arms.values():
            verify_evaluated_checkpoint(arm["student"],arm["evaluation"])
        gate=benefit(a["evaluation"],b["evaluation"],resume)
        if not manifest.get("hard_failure_required"):
            gate["accepted"]=False;gate["reasons"].append("not_a_genuine_hard_curriculum")
        report=dict(kind="long_recovery_benefit",dataset=manifest,arms=arms,resume=resume,
            benefit=gate,promoted=gate["accepted"],observed_generalization_benefit=gate["accepted"],
            promoted_checkpoint=trained["checkpoint"] if gate["accepted"] else None,
            checkpoint_sha256=digest(trained["checkpoint"]),
            development_groups=b["evaluation"]["development_groups"],
            scope="bounded_research_candidate_not_benchmark_or_deployment",
            costs={name:lifecycle.training_lineage(inputs[name+"_student"]["artifacts"]["report"])["cost"] for name in arms})
        output.mkdir(parents=True)
        return result(output,report)
    report=dict(kind="recovery_curriculum_feedback",dataset=manifest,arms=arms,resume=resume,
        treatment_minus_control=deltas,observed_generalization_benefit=False,promoted=False,
        automatic_next_training=False,reason="small_short_horizon_pilot_no_promotion",
        evaluation_scope=b["evaluation"]["metric_scope"],
        evaluation_physics_steps=b["plan"]["resolved"]["request"]["pilot"]["evaluation_steps"],
        replay_cost=replay["consumed"],
        next_action="expand_independent_development" if all(resume.values()) and b["feedback"]["pilot_metric_improved"] and deltas["travelled_m"]>0 and deltas["contact_free_fraction"]>=0 else "diagnose_before_more_training",
        costs={name:lifecycle.training_lineage(inputs[name+"_student"]["artifacts"]["report"])["cost"] for name in arms})
    output.mkdir(parents=True)
    return result(output,report)


def register(registry,Provider):
    from driversi.rsi.plugins import long_recovery
    long_recovery.register(registry,Provider)
    for p in (
        Provider("automatic_recovery_selection","1",(("student","unified_training_result"),),"recovery_selection",prepare_selection,run_selection,("report","plan","selection","anchor","control_anchor")),
        Provider("same_anchor_counterfactual_correction","1",(("selection","recovery_selection"),),"counterfactual_correction",prepare_correction,run_correction,("report","plan","teacher_report","teacher_rollout")),
        Provider("online_recovery_curriculum","1",(("selection","recovery_selection"),("correction","counterfactual_correction")),"online_recovery_curriculum",prepare_dataset,run_dataset,("report","manifest","control_request","treatment_request")),
        Provider("curriculum_training","1",(("dataset","online_recovery_curriculum"),),"unified_training_result",prepare_train,run_train,("report","plan")),
        Provider("curriculum_resume","1",(("student","unified_training_result"),),"unified_training_result",prepare_empty,run_resume,("report","plan")),
        Provider("curriculum_evaluation","1",(("student","unified_training_result"),),"unified_evaluation_result",prepare_empty,run_evaluation,("report","plan")),
        Provider("curriculum_feedback","1",(("control_student","unified_training_result"),("treatment_student","unified_training_result"),("control_evaluation","unified_evaluation_result"),("treatment_evaluation","unified_evaluation_result"),("replay","unified_training_result"),("dataset","online_recovery_curriculum")),"curriculum_feedback",prepare_empty,run_feedback,("report",)),
    ):registry.register(p)
