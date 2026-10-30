"""Long native, uncorrected evaluation through the shared Harness artifact boundary."""
import argparse
from pathlib import Path
import torch

from driversi.agents.backends.native import digest
from driversi.configuration import read_mapping
from driversi.rsi.correction.contact_origin import source_contacts
from driversi.rsi.correction.curriculum import SAFETY, route_support_failure
from driversi.contracts.scene import Scene
from driversi.runtime.budget import write_json
from driversi.sim.persistence.recording import RolloutRecord
from driversi.rsi.plugins.synthesis import check_files
from driversi_agents.driverl.collection import session
from driversi_agents.driverl.recovery import row, save_branch, official_score
from driversi_agents.driverl.learning.learner import deterministic_training


def evaluate(plan):
    out=Path(plan["output"]); cases=[]; cost=0
    keys=(*SAFETY,"ego_progress_along_expert_route","speed_limit_compliance","ego_is_comfortable")
    for index, item in enumerate(plan["evaluation_suite"]["cases"]):
        case=dict(source_log=item["source_log"], steps=item["steps"], branches={})
        origin=source_contacts(Scene.load(Path(item["scene"])))
        for role, checkpoint in (("parent",plan["parent_checkpoint"]),("candidate",plan["checkpoint"])):
            torch.manual_seed(plan["seed"])
            path=out/f"case-{index:02}"/role
            if origin["quarantined"]:
                report=dict(complete=False,recorded_steps=0,score=dict(status="not_scored",reason=origin["reason"]),
                    error=None,source_contacts=origin,contacts=[],checkpoint=checkpoint,
                    checkpoint_sha256=digest(checkpoint),harness_override=False,policy_executed=False,
                    support_diagnosis=dict(blocked=True,reason="source_initial_overlap"))
                path.mkdir(parents=True,exist_ok=True);write_json(path/"execution.json",report)
                case["branches"][role]=report
                continue
            p={**plan,**item};s=session(p,checkpoint)
            record=RolloutRecord.start(s.world);rows=[row(s)];contacts=[];error=None
            try:
                for _ in range(item["steps"]):
                    if s.world.episode_done:break
                    tr=s.step()
                    record.append(s.world,s.last_decision["executed_actions"],tr,s.last_decision["diagnostics"])
                    rows.append(row(s))
                    if len(record.actions)%20==0:write_json(out/"progress.json",dict(case=index,role=role,tick=len(record.actions),target=item["steps"]))
                    if s.world.contacts():contacts.append(dict(tick=int(s.world.observe().step),pairs=s.world.contacts()))
                complete=len(record.actions)==item["steps"]
                path.mkdir(parents=True,exist_ok=True)
                torch.save(dict(record=record,rows=rows,runtime=s.snapshot(),contacts=contacts),path/"execution-state.pt")
                save_branch(s,record,path,p,"Independent long evaluation: "+role)
                write_json(path/"history.json",rows)
                score=official_score(s,rows,path/"official-metrics",p,"DriveRL-native-"+role)
            except Exception as exc:
                complete=False;error=repr(exc);score=dict(status="failed",reason=error)
            finally:
                cost+=len(record.actions)*len(s.scene.actor_ids)
                s.close()
            report=dict(complete=complete,recorded_steps=len(record.actions),score=score,error=error,
                source_contacts=origin,contacts=contacts,checkpoint=checkpoint,checkpoint_sha256=digest(checkpoint),
                harness_override=False,policy_executed=True,
                support_diagnosis=route_support_failure([t.events.get("route_support_end",[False]) for t in record.transitions]))
            path.mkdir(parents=True,exist_ok=True);write_json(path/"execution.json",report)
            case["branches"][role]=report
        cases.append(case)
    valid=all(b["complete"] and b["score"].get("status")=="scored" for c in cases for b in c["branches"].values())
    metrics={}
    if valid:
        for role in ("parent","candidate"):
            metrics[role]={k:sum(c["branches"][role]["score"]["metric_scores"][k] for c in cases)/len(cases) for k in keys}
            metrics[role]["nuplan_score"]=sum(c["branches"][role]["score"]["score"] for c in cases)/len(cases)
    report=dict(status="completed" if valid else "failed",reason=None if valid else "incomplete_long_evaluation",
        model="driverl",phase="evaluate",context=plan["context"],cases=cases,metrics=metrics,
        directions={k:1 for k in (*keys,"nuplan_score")},primary_metric="nuplan_score",safety_higher=list(SAFETY),
        evaluated_checkpoint=plan["checkpoint"],evaluated_checkpoint_sha256=digest(plan["checkpoint"]),
        evaluation_suite_sha256=plan["evaluation_suite_sha256"],development_groups=plan["development_groups"],
        metric_scope="official_nuplan_metrics_custom_long_logs_not_benchmark",
        consumed=dict(sim_agent_steps=cost),promoted=False,trainable=False)
    write_json(out/"training-result.json",report)


def main():
    p=argparse.ArgumentParser();p.add_argument("plan");a=p.parse_args();plan=read_mapping(a.plan)
    check_files(plan["files"]);deterministic_training();torch.set_num_threads(4)
    evaluate(plan);check_files(plan["files"])

if __name__=="__main__":main()
