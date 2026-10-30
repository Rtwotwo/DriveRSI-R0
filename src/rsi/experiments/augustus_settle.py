"""Settle a service experiment after evaluation-only recovery, never retrain.

Execute against the original frozen runtime so sample admission and native
implementations retain their sealed identity. This continuation has its own ID.
"""
import json
from pathlib import Path
from driversi.configuration import read_mapping
from driversi.agents.backends.native import digest, project_root
from driversi.runtime.budget import write_json
from driversi.rsi.decisions.service_memory import review_learning, identity
from driversi.rsi.experiments.augustus_round import verify_program, _finalize, _verify, _files
from driversi.rsi.experiments.augustus_pair import compare


def settle(program, evaluation_reports, output, *, execution_receipt):
    import fcntl
    original = Path(program).resolve()
    out = Path(output).resolve()
    if (out == original or out.is_relative_to(original)
            or not out.is_relative_to(project_root() / "exp")):
        raise ValueError("Evaluation continuation needs a separate exp directory")
    plan = verify_program(original)
    failure = read_mapping(original / "review.json")
    if (failure.get("node") != "independent_evaluation"
            or failure.get("training_executed") is not True):
        raise ValueError("Only completed training with blocked evaluation can settle here")
    if set(evaluation_reports) != {"control", "treatment"}:
        raise ValueError("Both new evaluation receipts are required")
    execution = read_mapping(execution_receipt)
    if execution.get("optimizer_updates") != 0 or execution.get("parent_program_id") != plan["program_id"]:
        raise ValueError("Evaluation recovery cannot contain new optimizer updates")
    for arm, path in evaluation_reports.items():
        row = execution["rows"][arm]
        if (Path(row["report"]).resolve() != Path(path).resolve()
                or row["sha256"] != digest(path) or row["status"] != "completed"):
            raise ValueError("Recovery receipt does not bind both completed evaluations")
    roots = {arm: original/"tasks"/(plan["id"]+"-"+arm) for arm in ("control", "treatment")}
    provenance = _files([original/"program.json", original/"review.json", execution_receipt,
                         *evaluation_reports.values(), __file__])
    from driversi.rsi.experiments import augustus_pair
    provenance.update(_files([augustus_pair.__file__]))
    key = identity(dict(kind="evaluation_recovery", parent=plan["program_id"], files=provenance))
    out.mkdir(parents=True, exist_ok=True)
    with (out/"settlement.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out/"program.json").exists():
            prior=read_mapping(out/"program.json")
            if prior["program_id"] != key:
                raise ValueError("Existing continuation has different evidence")
        if (out/"review.json").exists():
            review=read_mapping(out/"review.json")
            _verify(review["execution_evidence"])
            if not (out/"next-iteration.json").exists():
                return _finalize(read_mapping(out/"program.json"), review)
            return review
        feedback = compare(roots["control"], roots["treatment"],
                           generation_costs=plan.get("generation_costs"),
                           evaluation_reports=evaluation_reports)
        checks = [read_mapping(root/"feedback/feedback.json")["cross_process_resume"]
                  for root in roots.values()]
        review = review_learning(feedback, checks)
        write_json(out/"paired-feedback.json", feedback)
        continuation = dict(plan, output=str(out), program_id=key,
                            parent_program=str(original), parent_program_id=plan["program_id"],
                            continuation_kind="evaluation_only_no_optimizer_updates",
                            continuation_files=provenance)
        write_json(out/"program.json", continuation)
        tasks = {arm:read_mapping(root/"report.json") for arm,root in roots.items()}
        evidence = {**provenance, **feedback["evidence_files"],
                    **_files([out/"paired-feedback.json", *[root/"feedback/feedback.json" for root in roots.values()],
                              *[root/"report.json" for root in roots.values()]])}
        review.update(training_executed=True, new_optimizer_updates=0, admitted_samples=feedback["synthetic_rows"],
                      evaluation_scope=feedback["evaluation_scope"], evaluation_samples=feedback["evaluation_samples"],
                      resume_checks=checks, execution_evidence=evidence,
                      paired_feedback=_files([out/"paired-feedback.json"]),
                      recovered_from=plan["program_id"], generation_policy_causal_credit=False,
                      partial_cost=dict(native_learning=feedback["training_evaluation_costs"],
                          original_tasks_including_replay_and_old_evaluation={arm:dict(wall_seconds=t.get("elapsed_seconds"),
                              sim_agent_steps=t.get("sim_agent_steps")) for arm,t in tasks.items()},
                          generation=plan.get("generation_costs"),
                          evaluation_recovery_wall_seconds=execution["wall_seconds"],
                          components_overlap_do_not_sum=True),complete_total_cost=False)
        return _finalize(continuation, review)
