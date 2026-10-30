"""Replay archived real nuPlan evidence through the RSI decision provider.
No new rollout, score query, parameter update or promotion is performed.
"""
import argparse
import json
from pathlib import Path

from driversi.agents.backends.native import project_root
from driversi.contracts.scene import Scene
from driversi.rsi.workflow import engine as harness
from driversi.runtime.budget import write_json
from driversi.rsi.decisions.learning_decisions import CHECKS, COSTS


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(); source=a.source.resolve(); output=a.output.resolve()
    if not output.is_relative_to(project_root()/"outputs"):
        raise ValueError("Evidence output belongs under project outputs")
    output.mkdir(parents=True,exist_ok=False)
    cases=[];audits=[]
    comparison=json.loads((source/"comparison.json").read_text())
    for row in comparison:
        folder=source/row["correction"]
        raw=json.loads((folder/"report.json").read_text())
        decisions=json.loads((folder/"decisions.json").read_text())
        scene=Scene.load(folder/"scene.npz")
        checks={k:"unknown" for k in CHECKS}
        checks["runtime_parity"]="pass" if raw["native_output_max_abs_error"]==0 and raw["source_input_parity"]["feature_max_abs_error"]==0 else "fail"
        # Absence of a collision does not certify world semantics, traffic rules,
        # training visual quality or success of the unassisted native policy.
        cases.append(dict(id=row["correction"],node="planning",failure_class="navigation_recovery",
                          source_log=scene.metadata["source_log"],split="development",
                          parent_version=raw["configuration"]["checkpoint_sha256"],
                          evaluator_version=row["official"]["protocol"],input_kind="counterfactual",
                          checks=checks,evidence=[str(folder/"report.json"),str(folder/"execution-evidence.json"),
                                                  str(folder/"decisions.json"),str(folder/"official-score.json")]))
        interventions=sum(bool(x.get("repair_active")) for x in decisions)
        missing_prob=sum("log_probability" not in x for x in decisions)
        audits.append(dict(case_id=row["correction"],steps=len(decisions),external_intervention_steps=interventions,
                           missing_native_behavior_likelihood_steps=missing_prob,online_update_admitted=False,
                           score=row["official"]["score"],score_scope=row["official"]["protocol"],
                           reason="archived_inference_and_harness_execution_is_not_an_on_policy_training_batch",
                           source_split=scene.metadata.get("split"),decision_role="engineering_development_only"))
    document=dict(schema_version=1,cases=cases,recipes=[dict(id="same-anchor-probe-v1",action="probe",node="planning",
                     execution=None,cost=dict(wall_seconds=60,gpu_seconds=60,score_queries=8))],
                  memory=[],studies=[],budget=dict(wall_seconds=120,gpu_seconds=120,score_queries=16),
                  protected_logs=[],max_failed_attempts=3)
    write_json(output/"decision-input.json",document)
    write_json(output/"online-audit.json",audits)
    task=dict(schema_version=2,id="real-nuplan-learning-routing",label="Archived nuPlan evidence routing; no training",
              budget=dict(max_wall_seconds=60,max_sim_agent_steps=0),
              steps=[dict(id="decision",provider="learning_decision",inputs={},
                          config={"evidence":"decision-input.json"},max_wall_seconds=30)])
    write_json(output/"task.json",task)
    registry=output/"registry"
    state=harness.register(registry,output/"task.json",output/"tasks")
    state=harness.run(registry,state["id"])
    report=dict(status=state["status"],task_id=state["id"],
                cases=len(cases),archived_decisions=sum(x["steps"] for x in audits),
                external_intervention_steps=sum(x["external_intervention_steps"] for x in audits),
                source="existing_real_nuplan_execution_evidence",new_rollouts=0,new_score_queries=0,
                optimizer_updates=0,policy_promoted=False,online_audit=audits,
                decision_artifact=str(Path(state["directory"])/"decision/report.json"))
    write_json(output/"report.json",report)
    print(json.dumps(report,ensure_ascii=False))
    if state["status"]!="completed":raise SystemExit(1)

if __name__=="__main__":main()
