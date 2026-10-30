import copy
import pytest
from driversi.rsi.decisions.frozen_skills import ARMS, METRICS, analyze_pairs, choose_skill, compile_prompts


def preset():
    return dict(version="v1", low_speed_mps=2.0, skills=dict(
        turn_conflict_check="Inspect the visible conflict area.",
        low_speed_progress_check="Inspect the available gap.",
        following_speed_check="Inspect visible relative motion."))


def test_router_rejects_future_information():
    with pytest.raises(ValueError):
        choose_skill(dict(navigation_command="LEFT", speed_mps=5, future_score=0), preset())


@pytest.mark.parametrize("nav,speed,want", [("TURN LEFT", 8, "turn_conflict_check"),
    ("STRAIGHT", 0, "low_speed_progress_check"), ("STRAIGHT", 10, "following_speed_check")])
def test_observed_router(nav, speed, want):
    got = choose_skill(dict(navigation_command=nav, speed_mps=speed), preset())
    assert got["id"] == want and not got["trigger_is_failure_diagnosis"]


def test_length_control_and_original_preserved():
    class Tokenizer:
        def encode(self, text, **kw): return text.split()
    p = compile_prompts("Native driving task.\n", dict(text="Check the visible lead vehicle."), Tokenizer())
    assert p["native"] == "Native driving task.\n"
    assert len(p["matched_skill"].split()) == len(p["neutral_control"].split())


def rows():
    return [dict(token="a", source_log="log", seed=42, arm=a,
                 inference_seconds=1, delta_from_native=dict(xy_ade_m=0.2),
                 metrics={m: 1.0 for m in METRICS}) for a in ARMS]


def test_new_zero_and_safety_regression_prevent_promotion():
    r = rows(); r[1]["metrics"].update(PDMS=0.0, NC=0.0)
    out = analyze_pairs(r, {(x["token"], x["seed"], x["arm"]) for x in r})
    assert out["comparisons"]["matched_skill"]["new_zero_pairs"] == 1
    assert out["disposition"] == "retain_native_default" and not out["promoted"]


def test_missing_duplicate_and_invalid_scores_fail_closed():
    r = rows(); keys = {(x["token"], x["seed"], x["arm"]) for x in r}
    for bad in [r[:-1], r + [copy.deepcopy(r[0])]]:
        with pytest.raises(ValueError): analyze_pairs(bad, keys)
    r[0]["metrics"]["PDMS"] = float("nan")
    with pytest.raises(ValueError): analyze_pairs(r, keys)
