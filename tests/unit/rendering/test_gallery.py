import copy
import re
from html.parser import HTMLParser

import pytest

from driversi.rendering.inspection.gallery import export_gallery


def report():
    return {
        "schema_version": 1, "kind": "interaction_gallery", "status": "complete",
        "requested_seconds": 9,
        "real_cases": [
            {"id": "working", "source_scene": "first/scene.npz", "map_name": "boston",
             "status": "complete", "requested_seconds": 9, "actual_seconds": 9,
             "viewer": "real/working/viewer.html",
             "details": "real/working/case.json", "route_warnings": ["short-route-car"],
             "summary": {"valid_agent_steps": 73,
                         "events": {"collision": 1, "offroad": 0, "goal": 2},
                         "actors": {"car-a": {"valid_steps": 23, "route_progress_m": 12.5,
                                               "events": {"collision": True}}}},
             "interaction": {"any_controlled_seconds": 5, "two_controlled_seconds": 2.3,
                             "per_actor_active_seconds": {"car-a": 2.3}}},
            *[{"id": status, "source_scene": "second/scene.npz", "status": status,
               "error": f"reason: {status}"} for status in (
                   "unresolved_task", "compile_rejected", "runtime_error", "contract_failure")],
        ],
        "following": {
            "report_path": "following/report.json", "case_count": 25, "checks_total": 550,
            "checks_passed": 550, "passed": True, "independent_templates": 1,
            "selected_cases": [{"case_index": 0, "speed_mps": 4, "gap_m": 6,
                                "passed": True, "checks_total": 22, "checks_passed": 22,
                                "metrics": {"reactive": {"desired_response_delay_s": .1,
                                                         "actual_response_delay_s": .3,
                                                         "brake": {"minimum_bumper_gap_m": 2.1,
                                                                   "minimum_ttc_seconds": None,
                                                                   "geometric_contact_any": False}}},
                                "views": [{"label": "前车干预", "path": "following/a.html",
                                           "kind": "lead"}]}],
        },
    }


class Elements(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.elements = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        self.elements.append((tag, dict(attrs)))


def test_gallery_keeps_every_failure_visible_and_separates_active_duration(tmp_path):
    output = tmp_path / "index.html"
    export_gallery(report(), output)
    text = output.read_text(encoding="utf-8")
    cards = [attrs for tag, attrs in Elements(text).elements
             if tag == "article" and attrs.get("class") == "case-card"]
    assert len(cards) == 5
    assert {item["data-status"] for item in cards} == {
        "complete", "unresolved_task", "compile_rejected", "runtime_error", "contract_failure"}
    assert not any("hidden" in item for item in cards)
    for label in ("请求时域", "实际记录时域", "≥1 辆受控车有效", "≥2 辆受控车有效"):
        assert label in text
    assert "9 s" in text and "5 s" in text and "2.3 s" in text
    assert "执行完成 / 非资格通过" in text
    assert "真实数据视频本身不证明因果交互" in text
    assert "一个合成跟驰模板" in text
    assert "550 / 550" in text
    assert 'href="real/working/viewer.html"' in text
    assert 'href="following/a.html"' in text
    assert 'href="real/working/case.json"' in text
    assert "初始路线长度不足以满足制动距离下界" in text and "short-route-car" in text


def test_gallery_escapes_untrusted_labels_and_uses_no_remote_resources(tmp_path):
    data = report()
    attack = '</script><img src="https://invalid.example/" onerror="alert(1)">'
    data["real_cases"][0].update(id=attack, map_name=attack, source_scene=attack)
    data["real_cases"][1]["error"] = attack
    data["following"]["selected_cases"][0]["views"][0]["label"] = attack
    output = tmp_path / "index.html"
    export_gallery(data, output)
    text = output.read_text(encoding="utf-8")
    assert attack not in text
    assert "&lt;/script&gt;&lt;img" in text
    assert not re.search(r'<(?:script|link|img)\b[^>]*(?:src|href)\s*=', text)
    assert "fetch(" not in text and "XMLHttpRequest" not in text


@pytest.mark.parametrize("path", ["../outside.html", "/absolute.html", "https://example.com/a",
                                 "javascript:alert(1)", "//example.com/a", "a\\..\\outside.html",
                                 "%2e%2e/outside.html", "viewer.html#jump", "viewer.html?q=x"])
def test_gallery_rejects_nonlocal_or_escaping_asset_links(tmp_path, path):
    data = report()
    data["real_cases"][0]["viewer"] = path
    with pytest.raises(ValueError, match="relative asset|inside the output"):
        export_gallery(data, tmp_path / "index.html")
    assert not (tmp_path / "index.html").exists()


def test_running_report_does_not_turn_absence_into_passing_evidence(tmp_path):
    data = copy.deepcopy(report())
    data.update(status="running", real_cases=[], following={})
    output = tmp_path / "index.html"
    export_gallery(data, output)
    text = output.read_text(encoding="utf-8")
    assert "当前报告未完成" in text
    assert "总体检查 <b>未验证</b>" in text
    assert "总体检查 <b>通过</b>" not in text


def test_gallery_rejects_unknown_case_status_instead_of_hiding_it(tmp_path):
    data = report()
    data["real_cases"][0]["status"] = "unrecognized"
    with pytest.raises(ValueError, match="case status"):
        export_gallery(data, tmp_path / "index.html")


def test_failed_collection_preserves_partial_evidence_and_explains_incompletion(tmp_path):
    data = report()
    data.update(status="error", error="OSError: <unavailable>")
    output = tmp_path / "index.html"
    export_gallery(data, output)
    text = output.read_text(encoding="utf-8")
    assert "样例集生成失败 · 当前结果不完整" in text
    assert "OSError: &lt;unavailable&gt;" in text
    assert "不能将部分结果作为完整验收结论" in text
    assert len([attrs for tag, attrs in Elements(text).elements
                if tag == "article" and attrs.get("class") == "case-card"]) == 5


def test_initial_collision_is_visible_even_without_valid_transitions(tmp_path):
    data = report()
    data["real_cases"] = [data["real_cases"][0]]
    data["real_cases"][0]["summary"] = {
        "valid_agent_steps": 0, "initial_inactive_actors": 1,
        "initial_events": {"collision": 1, "offroad": 0, "goal": 0},
        "events": {"collision": 0, "offroad": 0, "goal": 0},
        "actors": {"reset-contact-car": {"valid_steps": 0, "initial_inactive": True,
                                          "initial_events": {"collision": True},
                                          "events": {"collision": False}}},
    }
    output = tmp_path / "index.html"
    export_gallery(data, output)
    text = output.read_text(encoding="utf-8")
    assert "初始化 / reset · 受控车失效 <b>1</b>" in text
    assert "碰撞标记 <b>1</b>" in text
    assert "推进事件 · 碰撞 <b>0</b>" in text
    assert "受控 actor-episodes；每车每类最多计一次" in text
    assert "初始失效；collision" in text
    assert "<td>无记录事件</td>" not in text


def test_selected_configuration_caption_uses_actual_report_indices(tmp_path):
    data = report()
    data["following"]["selected_cases"][0]["case_index"] = 17
    output = tmp_path / "index.html"
    export_gallery(data, output)
    text = output.read_text(encoding="utf-8")
    assert "下列预先固定的配置编号：17（从 0 计数）。" in text
    assert "第 0、12、24 组" not in text


def test_adaptive_gallery_exposes_plan_source_limit_and_unverified_probe(tmp_path):
    data = report()
    data["schema_version"] = 2
    case = data["real_cases"][0]
    case.update(requested_seconds=22., planned_seconds=6., actual_seconds=4.7,
                horizon_plan={"mode": "fit_source", "available_future_seconds": 6.04,
                              "backend_max_duration_seconds": 9.,
                              "limiting_factors": ["source_exhaustion"],
                              "quantization_loss_seconds": .04},
                population={"previous_duration_seconds": 1.5, "previous_world": 5, "world": 3},
                interaction_probe={"status": "evidence_insufficient", "reason": "Pair stopped before response"})
    output = tmp_path / "index.html"
    export_gallery(data, output)
    text = output.read_text(encoding="utf-8")
    assert "计划时域" in text and "22 s" in text and "6 s" in text and "4.7 s" in text
    assert "6.04 s" in text and "source_exhaustion" in text and "后端上限" in text
    assert "交互证据不足" in text and "Pair stopped before response" in text
    assert "基准窗口 1.5 s → 本窗口" in text
    assert "人口 · 3 s →" not in text
    assert "真实跟驰探针已验证" in text


def test_probe_links_are_local_and_unknown_probe_status_is_rejected(tmp_path):
    data = report()
    data["real_cases"][0]["interaction_probe"] = {
        "status": "verified", "metrics": {"desired_response_delay_s": .1, "actual_response_delay_s": .3},
        "comparison_notes": ["分支提前结束时刻不同，分别展示，未补帧或裁剪。"],
        "report_path": "real/working/probe/report.json",
        "views": [{"label": "真实跟驰配对", "path": "real/working/probe/paired.html"}]}
    output = tmp_path / "index.html"
    export_gallery(data, output)
    text = output.read_text(encoding="utf-8")
    assert 'href="real/working/probe/paired.html"' in text
    assert "选定跟驰对响应已验证" in text and "目标响应延迟" in text
    assert "未补帧或裁剪" in text and 'href="real/working/probe/report.json"' in text
    data["real_cases"][0]["interaction_probe"]["views"][0]["path"] = "../outside.html"
    with pytest.raises(ValueError, match="inside the output"):
        export_gallery(data, tmp_path / "other.html")
    data["real_cases"][0]["interaction_probe"]["status"] = "looks_good"
    with pytest.raises(ValueError, match="probe status"):
        export_gallery(data, tmp_path / "other.html")
