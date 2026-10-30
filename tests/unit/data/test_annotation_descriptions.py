import copy
import json
import pytest

from driversi.data.catalog.annotation_descriptions import annotation_facts, describe, solution
from driversi.data.catalog.annotation_description_jobs import decode, text_jobs


def row():
    return {
        "driving_command": "TURN LEFT",
        "solution": {
            "perception": json.dumps([{"bbox_2d": [0, 0, 10, 10], "label": "vehicle"}]),
            "prediction": json.dumps([{"bbox_2d": [0, 0, 10, 10], "future_position": "near, front"}]),
            "planning": json.dumps([{"x_y_radian": [i, 0, 0], "label": str(i/2)+"s"} for i in range(1, 9)]),
        },
    }


def test_projection_does_not_modify_training_target():
    source = row()
    before = copy.deepcopy(source)
    facts = annotation_facts(source)
    result = describe("train-token", "train-log", facts)
    assert source == before
    assert facts["prediction_future_bins"] == {"near, front": 1}
    assert "导航指令：左转" in result["template_text_zh"]
    assert result["trainable"] is False and result["online_policy_input"] is False
    assert not result["review"]["indexable"]


def test_empty_key_objects_does_not_mean_empty_scene():
    source = row()
    source["solution"]["prediction"] = '[{"bbox_2d": [], "future_position": []}]'
    facts = annotation_facts(source)
    text = describe("a", "b", facts)["template_text_zh"]
    assert facts["prediction_object_count"] == 0
    assert "无已标注关键对象" in text
    assert "无车" not in text


def test_unknown_map_is_not_negative_evidence():
    facts = annotation_facts(row())
    facts["road"] = {"intersection_samples": None}
    text = describe("a", "b", facts)["template_text_zh"]
    assert "路口地图覆盖未知" in text
    assert "未落入" not in text


@pytest.mark.parametrize("value", [[], [{"x_y_radian": [1, 2, 3]}]*7,
                                  [{"x_y_radian": [1, 2, float("nan")]}]*8])
def test_native_future_contract_not_silently_repaired(value):
    source = row()
    source["solution"]["planning"] = json.dumps(value)
    with pytest.raises(ValueError):
        annotation_facts(source)


def test_escaped_official_fence_parses():
    assert solution('```json\\n[{"a":1}]\\n```') == [{"a": 1}]


def response(value):
    return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(value)}}]}


def test_unknown_llm_evidence_rejected():
    with pytest.raises(ValueError):
        decode({"facts": {"motion": {}}}, "model",
               response({"summary_zh": "红灯停车", "evidence_fields": ["signal_phase"]}))


def test_format_valid_paraphrase_never_auto_approved():
    receipt = decode({"facts": {"motion": {}}}, "model",
                     response({"summary_zh": "速度变化。", "evidence_fields": ["motion"]}))
    assert receipt["status"] == "format_valid_unreviewed"
    assert receipt["indexable"] is False and receipt["trainable"] is False


def test_actual_straight_command_is_preserved():
    source = row()
    source["driving_command"] = "GO STRAIGHT"
    assert "导航指令：直行" in describe("a", "b", annotation_facts(source))["template_text_zh"]


def test_exact_request_inputs_retained_without_private_source_paths(tmp_path):
    scenes = tmp_path/"scenes"
    scenes.mkdir()
    scene = describe("train-token", "/private/log-name", annotation_facts(row()))
    scene["facts"]["road"] = {"map_name": "private-city", "map_version": "v1",
                               "intersection_samples": 1}
    (scenes/"a.json").write_text(json.dumps(scene))
    job = text_jobs(tmp_path)[0]
    saved = list((tmp_path/"api/prepared").glob("*.json"))
    assert len(saved) == 1
    assert json.loads(saved[0].read_text()) == job
    assert "private-city" not in json.dumps(job["facts"])
    assert "/private/log-name" not in job["prompt"] + json.dumps(job["facts"])
    assert "独立统计" in job["prompt"]
    assert text_jobs(tmp_path)[0] == job


@pytest.mark.parametrize("limit", [0, 4])
def test_text_pilot_limit_enforced(tmp_path, limit):
    with pytest.raises(ValueError):
        text_jobs(tmp_path, limit=limit)


def test_non_string_evidence_is_rejected():
    with pytest.raises(ValueError):
        decode({"facts": {"motion": {}}}, "model",
               response({"summary_zh": "速度变化。", "evidence_fields": [["motion"]]}))


def test_anchor_camera_references_are_enforced():
    from driversi.data.catalog.annotation_anchor_views import decode as anchor_decode
    job = {"evidence": [{"id": "CAM_F0_anchor"}]}
    value = {"summary_zh": "前方道路。", "features": [
        {"field": "road_layout", "text": "前方道路。", "evidence_ids": ["CAM_R0_anchor"]}]}
    with pytest.raises(ValueError):
        anchor_decode(job, "model", response(value))
    value["features"][0]["evidence_ids"] = ["CAM_F0_anchor"]
    result = anchor_decode(job, "model", response(value))
    assert result["scope"] == "current_anchor_only"
    assert result["indexable"] is False


def test_anchor_prompt_excludes_temporal_inference():
    from driversi.data.catalog.annotation_anchor_views import PROMPT
    text = PROMPT.read_text()
    assert "只描述当前可见内容" in text
    assert "不能证明切入" in text
    assert "当前速度不是速度变化" in text


def test_anchor_can_describe_same_feature_type_in_different_cameras():
    from driversi.data.catalog.annotation_anchor_views import decode as anchor_decode
    job = {"evidence": [{"id": "CAM_F0_anchor"}, {"id": "CAM_L0_anchor"}]}
    value = {"summary_zh": "两路图片可见不同车辆。", "features": [
        {"field": "visible_actors", "text": "前视图车辆。", "evidence_ids": ["CAM_F0_anchor"]},
        {"field": "visible_actors", "text": "左视图车辆。", "evidence_ids": ["CAM_L0_anchor"]}]}
    result = anchor_decode(job, "model", response(value))
    assert len(result["description"]["features"]) == 2
    assert result["indexable"] is False and result["trainable"] is False

