"""Observed-condition prefix skills and paired evidence; no model or scorer imports."""

import math
import statistics

ARMS = ("native", "matched_skill", "neutral_control")
METRICS = ("PDMS", "NC", "DAC", "DDC", "EP", "TTC", "C")
SAFETY = ("NC", "DAC", "DDC", "TTC", "C")


def choose_skill(observation, preset):
    """Allowlist the router input: never accept caption, scores, or future targets."""
    if set(observation) != {"navigation_command", "speed_mps"}:
        raise ValueError("Skill routing accepts navigation and current speed only")
    speed = observation["speed_mps"]
    if type(speed) not in (int, float) or not math.isfinite(speed) or speed < 0:
        raise ValueError("Invalid current speed")
    command = observation["navigation_command"]
    if not isinstance(command, str):
        raise ValueError("Explicit navigation text required")
    command = command.upper()
    if "LEFT" in command or "RIGHT" in command:
        key = "turn_conflict_check"
    elif speed <= preset["low_speed_mps"]:
        key = "low_speed_progress_check"
    else:
        key = "following_speed_check"
    skill = preset["skills"][key]
    return dict(id=key, version=preset["version"], text=skill,
                evidence=dict(observation), trigger_is_failure_diagnosis=False)


def compile_prompts(base, skill, tokenizer):
    """Match the full raw text token count, without changing the native template."""
    matched = base.rstrip() + "\n\nPlanning checklist for this observation:\n" + skill["text"]
    count = lambda s: len(tokenizer.encode(s, add_special_tokens=False))
    target = count(matched)
    stem = base.rstrip() + "\n\nArchive formatting note: headings and paragraph spacing."
    # Repeated benign words are an explicit length control, not another driving instruction.
    for n in range(max(0, target - count(stem) - 8), target + 8):
        neutral = stem + " section" * n
        if count(neutral) == target:
            return {"native": base, "matched_skill": matched, "neutral_control": neutral}
    raise ValueError("Unable to produce exact tokenizer-length control")


def analyze_pairs(rows, expected_keys):
    expected_keys = set(expected_keys)
    by = {}
    for row in rows:
        key = (row["token"], row["seed"], row["arm"])
        if key in by or key not in expected_keys:
            raise ValueError("Duplicate or unexpected scored member")
        if any(type(row["metrics"].get(k)) not in (float, int)
               or not math.isfinite(row["metrics"][k])
               or not 0 <= row["metrics"][k] <= 1 for k in METRICS):
            raise ValueError("Incomplete native metric record")
        by[key] = row
    if set(by) != expected_keys:
        raise ValueError("Cannot analyze partial paired coverage")
    summary = {}
    cases = sorted({(t, s) for t, s, _ in expected_keys})
    for arm in ARMS:
        group = [by[(t, s, arm)] for t, s in cases]
        summary[arm] = {
            "count": len(group),
            "metrics": {m: statistics.mean(r["metrics"][m] for r in group) for m in METRICS},
            "zero_count": sum(r["metrics"]["PDMS"] == 0 for r in group),
            "mean_inference_seconds": statistics.mean(r["inference_seconds"] for r in group),
        }
    comparisons = {}
    for arm in ARMS[1:]:
        pairs = []
        for token, seed in cases:
            a, b = by[(token, seed, "native")], by[(token, seed, arm)]
            pairs.append(dict(token=token, seed=seed, source_log=a["source_log"],
                              pdms_delta=b["metrics"]["PDMS"]-a["metrics"]["PDMS"],
                              zero_escaped=a["metrics"]["PDMS"] == 0 < b["metrics"]["PDMS"],
                              new_zero=b["metrics"]["PDMS"] == 0 < a["metrics"]["PDMS"],
                              safety_regressions=[m for m in SAFETY if b["metrics"][m] < a["metrics"][m]],
                              trajectory_xy_ade_m=b["delta_from_native"]["xy_ade_m"]))
        per_log = {}
        for p in pairs:
            per_log.setdefault(p["source_log"], []).append(p["pdms_delta"])
        comparisons[arm] = dict(
            mean_pdms_delta=statistics.mean(p["pdms_delta"] for p in pairs),
            improved=sum(p["pdms_delta"] > 1e-8 for p in pairs),
            worsened=sum(p["pdms_delta"] < -1e-8 for p in pairs),
            escaped_zero_pairs=sum(p["zero_escaped"] for p in pairs),
            new_zero_pairs=sum(p["new_zero"] for p in pairs),
            safety_regression_pairs=sum(bool(p["safety_regressions"]) for p in pairs),
            changed_trajectory_pairs=sum(p["trajectory_xy_ade_m"] > 1e-8 for p in pairs),
            per_log_mean_delta={k: statistics.mean(v) for k, v in per_log.items()}, pairs=pairs)
    skill, neutral = (comparisons[a] for a in ARMS[1:])
    positive = (skill["mean_pdms_delta"] > 1e-8
                and skill["mean_pdms_delta"] > neutral["mean_pdms_delta"]
                and skill["new_zero_pairs"] == 0 and skill["safety_regression_pairs"] == 0)
    return dict(arms=summary, comparisons=comparisons,
                disposition="candidate_for_independent_validation" if positive else "retain_native_default",
                promoted=False, optimizer_updates=0,
                scope="retrospective_navtest_development_fixed_skill_pilot",
                recursive_improvement_demonstrated=False,
                seeds_are_paired_repeats_not_independent_logs=True)
