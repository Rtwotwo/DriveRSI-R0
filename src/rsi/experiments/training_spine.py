"""A single admitted synthetic sample through native paired training/evaluation.

An engineering pilot, never a completed trajectory family or model promotion.
Uses the existing admission, export, compiler and durable Harness task engine.
"""
import argparse
import fcntl
import json
import time
from pathlib import Path

from driversi.agents.backends.native import digest, project_root
from driversi.configuration import read_mapping
from driversi.rsi.experiments.continuation import frozen_json, task_state
from driversi.runtime.budget import write_json


def validate_scope(plan, config, manifest):
    if plan.get("trajectory_family") or config.get("trajectory_family"):
        raise ValueError("A family plan cannot bypass its minimum-sample gate")
    if len(plan.get("assignments", [])) != 1 or plan["context"]["model"] != "chainflow_stage1":
        raise ValueError("One explicit ChainFlow training-spine assignment required")
    if (manifest.get("schema_version") != 2 or len(manifest["splits"]["development"]) != 12146
            or manifest.get("benchmark",{}).get("protocol") != "official_navtest_v1"):
        raise ValueError("Full official navtest manifest required, not a subset")
    if len(manifest["splits"]["train"]) != 103288:
        raise ValueError("Original full navtrain replay pool required")
    if config.get("visual_quality_profile") != "experimental_numeric25_v1":
        raise ValueError("The declared 25dB profile must remain unchanged")


def run(config, generation, plan, base_manifest, output):
    from driversi.rsi.decisions.capability_skills import validate_plan
    from driversi.rsi.decisions.trajectory_diversity import compare, describe
    from driversi.rsi.experiments.chainflow_campaign import process
    from driversi.rsi.plugins.chainflow_synthetic import compile_paired_task
    from driversi.rsi.correction.epdms_admission import required_for
    from driversi.rsi.plugins.synthesis import check_files

    paths = [Path(p).resolve(strict=True) for p in (config,generation,plan,base_manifest)]
    cp,gp,pp,mp = paths
    c,g,p,m = map(read_mapping,paths)
    validate_plan(p)
    validate_scope(p,c,m)
    out = Path(output).resolve()
    if not out.is_relative_to(project_root()/"exp") or not required_for(gp):
        raise ValueError("Owned complete-EPDMS campaign required")
    if g["status"] != "rendered" or g["teacher"]["plan_sha256"] != digest(pp):
        raise ValueError("Fresh rendered execution of this exact single-assignment plan required")
    check_files(g["files"])
    out.mkdir(parents=True,exist_ok=True)
    with (out/"run.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        frozen_json(out/"inputs.json",dict(files={str(p):digest(p) for p in paths},
            scope="single_sample_training_spine_validation", updates_per_arm=8,
            full_navtest_per_arm=12146, family_completed=False, promoted=False))
        def status(state,**extra):
            write_json(out/"status.json",dict(status=state,time=time.time(),
                scope="single_sample_training_spine_validation",family_completed=False,
                promoted=False,**extra))
        try:
            status("checking_admission_and_export")
            description=describe(g,p["assignments"][0]["id"])
            novelty=compare(description,[],minimum_ade_m=.5)
            frozen_json(gp.parent.parent/"diversity.json",dict(description=description,
                decision=novelty,retained=[]))
            frozen_json(out/"diversity-scope.json",dict(
                kind="singleton_no_pairwise_diversity_result", family_target_achieved=False))
            sample=process(gp,cp,out/"postprocess")
            if sample.get("status") != "training_sample_ready":
                status("admission_rejected",details=sample)
                return sample
            status("compiling_native_pair",qualified_samples=1)
            task=out/"paired/task.json"
            if not task.exists():
                compile_paired_task(mp,[sample["sample_report"]],c["training_model_config"],
                    out/"paired",updates_per_arm=8,capability_plan=pp,
                    skill_memory=out/"skill-memory.sqlite",epoch_training=True,
                    batch_size=2,pre_evaluation_gate=True)
            status("paired_task_dispatched",qualified_samples=1,updates_per_arm=8)
            result=task_state(out/"harness.sqlite",read_mapping(task),task,out/"tasks",execute=True)
            status("completed" if result["status"]=="completed" else "paired_task_failed",
                   task_status=result["status"],qualified_samples=1,updates_per_arm=8)
            return result
        except BaseException as exc:
            status("failed",error_type=type(exc).__name__,error=str(exc))
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("config","generation","plan","base-manifest","output"):
        parser.add_argument("--"+name,required=True)
    print(json.dumps(run(**vars(parser.parse_args()))),flush=True)


if __name__=="__main__":
    main()
