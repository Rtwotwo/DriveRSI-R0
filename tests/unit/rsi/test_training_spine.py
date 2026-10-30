import pytest
from driversi.rsi.experiments.training_spine import validate_scope


def test_singleton_cannot_bypass_family_or_replace_full_evaluation():
    p={"assignments":[{}],"context":{"model":"chainflow_stage1"}}
    c={"visual_quality_profile":"experimental_numeric25_v1"}
    m={"schema_version":2,"benchmark":{"protocol":"official_navtest_v1"},"splits":{"development":[None]*12146,"train":[None]*103288}}
    validate_scope(p,c,m)
    with pytest.raises(ValueError,match="family"):
        validate_scope({**p,"trajectory_family":{"target_per_anchor":10}},c,m)
    with pytest.raises(ValueError,match="family"):
        validate_scope(p,{**c,"trajectory_family":{"target_per_anchor":10}},m)
    with pytest.raises(ValueError,match="subset"):
        validate_scope(p,c,{**m,"schema_version":3})
    with pytest.raises(ValueError,match="25dB"):
        validate_scope(p,{**c,"visual_quality_profile":"experimental_numeric24_ade03_v1"},m)
