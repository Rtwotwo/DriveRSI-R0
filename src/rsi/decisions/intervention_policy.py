"""Bounded service-skill scheduling, not driving policy optimization.

Only aggregate, independently checked experiment reviews enter this policy.
No scene tokens, captions, predictions or arbitrary generated code are inputs.
"""


def select_intervention(candidates, history, remaining_trials):
    """Explore declared alternatives after no-benefit; stop at unresolved gates.

    Priority is a predeclared hypothesis, not a learned reward estimate. A pilot
    improvement requests larger validation and cannot promote a driving skill.
    """
    if history:
        last = history[-1]
        if last["node"] != "learning_review":
            return {"action": "repair", "reason": "unresolved_execution_gate", "node": last["node"]}
        if last["status"] == "pilot_signal":
            return {"action": "validate", "reason": "expand_independent_validation"}
        if last["status"] != "no_benefit":
            return {"action": "repair", "reason": "unrecognized_review"}
    if remaining_trials <= 0:
        return {"action": "stop", "reason": "trial_budget_exhausted"}
    tried = {h.get("intervention_id") for h in history}
    options = sorted(
        (c for c in candidates if c["id"] not in tried), key=lambda c: (c["priority"], c["id"])
    )
    if not options:
        return {"action": "stop", "reason": "declared_interventions_exhausted"}
    candidate = options[0]
    return {
        "action": "execute",
        "intervention_id": candidate["id"],
        "reason": "test_declared_revision_after_no_benefit"
        if history
        else "initial_declared_trial",
        "hypothesis": candidate["hypothesis"],
        "policy": "bounded_priority_v1",
        "policy_effectiveness_established": False,
    }
