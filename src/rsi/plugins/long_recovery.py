"""Independent long evaluation and fail-closed benefit decisions for recovery training."""
import math
from pathlib import Path
from driversi.agents.backends.native import digest, signature
from driversi.configuration import read_mapping
from driversi.contracts.scene import Scene
from driversi.rsi.contracts import _keys
from driversi.rsi.workflow.results import result
from driversi.rsi.plugins.collection import validate_student
from driversi.rsi.plugins.synthesis import check_files, run_process
from driversi.runtime.budget import write_json
from driversi.training.sampling.environment import source_log_group
from driversi.rsi.correction.curriculum import SAFETY


def prepare(config,base,registry):
    _keys(config,("suite","max_wall_seconds"),"long evaluation")
    if type(config["max_wall_seconds"]) not in (int,float) or not 0<config["max_wall_seconds"]<=3600:
        raise ValueError("Bounded long evaluation required")
    path=(base/config["suite"]).resolve(strict=True);suite=read_mapping(path)
    if set(suite)!={"schema_version","protocol","cases","search_source_logs"} or suite["schema_version"]!=1 or suite["protocol"]!="nuplan_custom_long_v1":
        raise ValueError("Unknown evaluation protocol")
    if not 2<=len(suite["cases"])<=4:raise ValueError("Two to four long evaluation cases required")
    from driversi.agents.backends.native import project_root
    official=read_mapping(project_root()/"src/vendor/driving/navsim-v1.1/navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml")
    protected={source_log_group(x) for x in official["log_names"]}
    groups=[];files={str(path):digest(path)};cap=0
    for row in suite["cases"]:
        _keys(row,("scene","road_graph","source_log","steps"),"evaluation case")
        if type(row["steps"]) is not int or not 300<=row["steps"]<=600:raise ValueError("30-60 seconds required")
        for key in ("scene","road_graph"):
            p=Path(row[key]);files[str(p)]=digest(p)
        s=Scene.load(Path(row["scene"]))
        if s.metadata["source_log"]!=row["source_log"] or len(s.times)<=row["steps"] or abs(float(s.times[1]-s.times[0])-.1)>1e-6:
            raise ValueError("Evaluation source/duration mismatch")
        db=Path(s.metadata["nuplan_source"]["db_path"])
        if (s.metadata["split"] not in ("train","unassigned") or any(x in ("val","test") for x in db.parts)
            or db.stem!=row["source_log"] or source_log_group(row["source_log"]) in protected):
            raise ValueError("Official test source cannot enter iterative development")
        groups.append(source_log_group(row["source_log"]));cap+=2*row["steps"]*len(s.actor_ids)
    if len(set(groups))<2:raise ValueError("Need distinct original development logs")
    search={source_log_group(x) for x in suite["search_source_logs"]}
    if not search or search & set(groups):raise ValueError("Search source leaked into evaluation")
    return {**config,"suite":str(path),"evaluation_suite":suite,"evaluation_suite_sha256":signature(suite),
        "development_groups":sorted(set(groups)),"files":files},cap


def execute(config,inputs,output,registry):
    check_files(config["files"])
    item=inputs["student"];student=read_mapping(item["artifacts"]["report"]);old=read_mapping(item["artifacts"]["plan"])
    check_files(old["files"]);check_files(student["checkpoint_files"])
    if student["status"]!="completed" or student["model"]!="driverl" or student["context"]!=old["context"]:
        raise ValueError("Invalid student evidence")
    known=set(student["source_groups"]["train"])
    declared={source_log_group(x) for x in config["evaluation_suite"]["search_source_logs"]}
    if not known.issubset(declared):raise ValueError("Evaluator omitted known model training lineage")
    if set(config["development_groups"]) & known:
        raise ValueError("Training source leaked into long evaluation")
    output.mkdir(parents=True)
    cfg=old["model_config"]
    from driversi.agents.backends.native import project_root
    from driversi.agents.backends.sources import require_source
    require_source(project_root()/cfg["repository"],cfg["revision"])
    from driversi.agents.backends.native import project_root,path_at
    parent=str(path_at(project_root(),cfg["weights"]["checkpoint"]))
    plan={**config,"output":str(output),"model_config":cfg,"checkpoint":student["checkpoint"],
        "parent_checkpoint":parent,"context":student["context"],"seed":old["resolved"]["request"]["pilot"]["seed"],
        "files":{**config["files"],**student["checkpoint_files"],parent:digest(parent)}}
    write_json(output/"training-plan.json",plan)
    run_process([cfg["python"],"-B","-m","driversi_agents.driverl.long_evaluation",str(output/"training-plan.json")],
        output,config["max_wall_seconds"],environment={"CUDA_VISIBLE_DEVICES":str(cfg["settings"]["gpu"]),"CUBLAS_WORKSPACE_CONFIG":":4096:8"})
    check_files(plan["files"]);report=read_mapping(output/"training-result.json")
    return result(output,report,status=report["status"],reason=report["reason"],steps=report["consumed"]["sim_agent_steps"],
        artifacts={**{str(p.relative_to(output)):str(p) for p in output.rglob("*") if p.is_file()},
            "report":str(output/"training-result.json"),"plan":str(output/"training-plan.json")})


def verify_evaluated_checkpoint(student, evaluation):
    checkpoint = student["checkpoint"]
    expected = student["checkpoint_files"].get(checkpoint)
    if (not expected or evaluation.get("evaluated_checkpoint") != checkpoint
        or evaluation.get("evaluated_checkpoint_sha256") != expected
        or digest(checkpoint) != expected):
        raise ValueError("Evaluated checkpoint differs from published/current student")


def benefit(control,treatment,resume):
    """Fixed independent acceptance; never modifies raw official scores."""
    reasons=[]
    if not all(resume.values()):reasons.append("resume_not_reproduced")
    if control.get("status")!="completed" or treatment.get("status")!="completed":reasons.append("incomplete_evaluation")
    if control.get("evaluation_suite_sha256")!=treatment.get("evaluation_suite_sha256"):reasons.append("different_evaluators")
    cases_a, cases_b=control.get("cases",[]),treatment.get("cases",[])
    if len(cases_a)!=len(cases_b) or len(cases_b)<2:reasons.append("incomplete_case_pairing")
    groups={source_log_group(c["source_log"]) for c in cases_b}
    if len(groups)<2:reasons.append("insufficient_independent_logs")
    gains=[]
    for a,b in zip(cases_a,cases_b):
        if a["source_log"]!=b["source_log"] or a["steps"]!=b["steps"] or b["steps"]<300:
            reasons.append("mismatched_or_short_case");continue
        if a["branches"]["parent"]["score"] .get("metric_scores")!=b["branches"]["parent"]["score"].get("metric_scores"):
            reasons.append("different_parent_replay")
        x,y=a["branches"]["candidate"],b["branches"]["candidate"]
        for branch in (*a["branches"].values(),*b["branches"].values()):
            if not branch["complete"] or branch["source_contacts"]["quarantined"]:
                reasons.append("execution_or_source_contact")
            if branch.get("support_diagnosis", {}).get("blocked") is not False:
                reasons.append("source_support_unverified_or_exhausted")
        if y["contacts"]:reasons.append("candidate_world_contact")
        xm,ym=x["score"].get("metric_scores",{}),y["score"].get("metric_scores",{})
        for k in SAFETY:
            if ym.get(k)!=1.0 or ym.get(k,-1)<xm.get(k,1):reasons.append("safety_not_passed:"+k)
        scores=[z["score"].get("score") for z in (x,y,b["branches"]["parent"])]
        if not all(type(v) in (int,float) and math.isfinite(v) for v in scores):
            reasons.append("invalid_score");continue
        if scores[1]<max(scores[0],scores[2]):reasons.append("per_log_score_regression")
        gains.append(scores[1]-scores[0])
    mean=sum(gains)/len(gains) if gains else None
    if mean is None or mean<.01:reasons.append("less_than_one_point_over_equal_budget_control")
    return dict(accepted=not reasons,reasons=sorted(set(reasons)),treatment_minus_control=mean,
        aggregation_unit="equal_window_fixed_suite",promoted=False,promotion_scope="bounded_research_only_not_deployment",official_benchmark=False)


def collect_promoted(config,inputs,output,registry):
    from driversi.rsi.plugins import collection
    gate=read_mapping(inputs["feedback"]["artifacts"]["report"])
    student=read_mapping(inputs["student"]["artifacts"]["report"])
    if gate.get("promoted") is not True:
        output.mkdir(parents=True)
        return result(output,{"promoted":False,"collection_executed":False},status="rejected",reason="promotion_not_accepted")
    if (gate["promoted_checkpoint"]!=student["checkpoint"] or
        gate["checkpoint_sha256"]!=digest(student["checkpoint"]) or
        not gate["benefit"]["accepted"]):raise ValueError("Wrong promoted model")
    group=source_log_group(Scene.load(Path(config["scene"])).metadata["source_log"])
    if group in gate["development_groups"]:raise ValueError("Recollection would mine development data")
    return collection.execute(config,{"student":inputs["student"]},output,registry)



def select_promoted(config,inputs,output,registry):
    from driversi.rsi.plugins import recovery
    gate=read_mapping(inputs["feedback"]["artifacts"]["report"])
    student=read_mapping(inputs["student"]["artifacts"]["report"])
    if gate.get("promoted") is not True:
        output.mkdir(parents=True)
        return result(output,{"promoted":False,"collection_executed":False},status="rejected",reason="promotion_not_accepted")
    if (gate["promoted_checkpoint"]!=student["checkpoint"] or gate["checkpoint_sha256"]!=digest(student["checkpoint"]) or not gate["benefit"]["accepted"]):
        raise ValueError("Wrong promoted checkpoint")
    source=source_log_group(Scene.load(Path(config["scene"])).metadata["source_log"])
    if source in gate["development_groups"]:raise ValueError("Recollection would mine development data")
    if config.get("hard_failure_required") is not True:raise ValueError("Long actual-failure collection required")
    return recovery.run_selection(config,{"student":inputs["student"]},output,registry)

def register(registry,Provider):
    from driversi.rsi.plugins import collection, recovery
    registry.register(Provider("promoted_hard_recovery_selection","1",(("student","unified_training_result"),("feedback","curriculum_feedback")),
        "recovery_selection",recovery.prepare_selection,select_promoted,("report","plan","selection","anchor","control_anchor")))
    registry.register(Provider("promoted_curriculum_collection","1",(("student","unified_training_result"),("feedback","curriculum_feedback")),
        "learned_collection",collection.prepare,collect_promoted,("report",)))

    registry.register(Provider("curriculum_long_evaluation","1",(("student","unified_training_result"),),
        "unified_evaluation_result",prepare,execute,("report","plan")))
