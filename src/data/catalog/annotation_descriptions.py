"""Compact offline retrieval descriptors derived from official AUGUstus annotations."""
from __future__ import annotations

import json
from collections import Counter

import numpy as np

VERSION = "driversi.annotation_description.v1"


def solution(raw):
    """Decode official fenced JSON without rewriting the original annotation."""
    if not isinstance(raw, str):
        raise ValueError("Official solution must be a string")
    text = raw.replace("\\n", "\n").replace("\\t", "\t").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    value = json.loads(text)
    if not isinstance(value, list):
        raise ValueError("Official solution must contain a list")
    return value


def annotation_facts(row):
    prediction = solution(row["solution"]["prediction"])
    planning = solution(row["solution"]["planning"])
    perception = solution(row["solution"]["perception"])
    future = np.asarray([x["x_y_radian"] for x in planning], dtype=float)
    if future.shape != (8, 3) or not np.isfinite(future).all():
        raise ValueError("Require native eight-pose future target")
    bins = Counter()
    for item in prediction:
        if not item.get("bbox_2d") and not item.get("future_position"):
            continue  # The official empty-object sentinel is not missing annotation.
        box = np.asarray(item["bbox_2d"], dtype=float)
        label = item["future_position"]
        if box.shape != (4,) or not np.isfinite(box).all() or not isinstance(label, str):
            raise ValueError("Invalid official prediction object")
        bins[label] += 1
    classes = Counter(x["label"] for x in perception if isinstance(x.get("label"), str))
    return {
        "navigation_command": row["driving_command"],
        "key_object_classes": dict(sorted(classes.items())),
        "prediction_future_bins": dict(sorted(bins.items())),
        "prediction_horizon_s": 4.0,
        "prediction_reference": "official_P3_relative_position_categories",
        "prediction_object_count": sum(bins.values()),
        "planning_horizon_s": 4.0,
        "planning_point_count": 8,
        "annotation_semantics": "salient annotated objects, not full scene population",
    }


def describe(token, source_log, facts):
    result = {
        "schema_version": VERSION, "token": token, "source_log": source_log,
        "split": "navtrain", "scope": "retrospective_full_clip",
        "online_policy_input": False, "facts": facts,
        "native_training_target": "original solution_prediction unchanged",
        "review": {"status": "unreviewed_derivation", "indexable": False},
        "trainable": False,
    }
    commands = {"TURN LEFT": "左转", "GO FORWARD": "直行", "GO STRAIGHT": "直行", "TURN RIGHT": "右转"}
    text = ["导航指令：" + commands.get(facts["navigation_command"], "未知")]
    if "motion" in facts:
        m = facts["motion"]
        text += [f'片段速度 {m["start_kmh"]}→{m["end_kmh"]} km/h，'
                 f'范围 {m["range_kmh"][0]}–{m["range_kmh"][1]} km/h',
                 f'片段净航向变化 {m["net_heading_change_deg"]}°（不自动判定转弯或变道）']
    if "road" in facts:
        count = facts["road"]["intersection_samples"]
        text += ["路口地图覆盖未知" if count is None else "路径采样点" + (
            "经过路口区域" if count else "未落入已加载路口区域")]
    text += ["关键对象未来位置标注：" + (
        "、".join(f"{k}×{v}" for k, v in facts["prediction_future_bins"].items())
        or "无已标注关键对象")]
    result["template_text_zh"] = "；".join(text) + "。"
    return result

