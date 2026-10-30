#!/usr/bin/env python3
"""Render the DriveRSI platform progress spec as deterministic, editable SVG.

Only standard-library vector primitives and text are emitted.  The ten stage
identities are stable so a containing page can select them with data-stage.
"""
import argparse
import json
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path

NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", NS)
STAGE_IDS = ("source", "selfplay", "mining", "teacher", "verify", "render",
             "dataset", "train", "evaluate", "feedback")
LAYOUT = ((56, 112), (440, 112), (824, 112), (1208, 112), (1208, 366),
          (1208, 632), (1208, 900), (824, 900), (440, 900), (56, 900))
WIDTH, HEIGHT, CARD_WIDTH, CARD_HEIGHT = 1600, 1120, 320, 180
TONES = {"blue": ("#3265B7", "#EFF5FF"), "amber": ("#9A6314", "#FFF7E9"),
         "violet": ("#7450A9", "#F5F0FD"), "teal": ("#157C70", "#EBF8F4")}
STATES = {"supported": ("有运行证据", "#176C57", "#E8F6EE"),
          "partial": ("部分接通", "#885B14", "#FFF3D7"),
          "planned": ("待接入", "#5D6479", "#EFF1F6")}
STAGE_TONES = ("blue", "blue", "amber", "amber", "amber", "violet", "violet",
               "teal", "teal", "teal")


def validate_spec(spec):
    """Raise ValueError for invalid required fields; leave evidence fields intact."""
    if not isinstance(spec, dict):
        raise ValueError("spec must be an object")
    for key in ("title", "subtitle", "updated_at", "evidence_scope"):
        if not isinstance(spec.get(key), str) or not spec[key].strip():
            raise ValueError(f"{key} must be a non-empty string")
    version = spec.get("schema_version")
    if not ((isinstance(version, int) and not isinstance(version, bool) and version > 0)
            or (isinstance(version, str) and version.strip())):
        raise ValueError("schema_version must be a positive integer or non-empty string")
    stages = spec.get("stages")
    if not isinstance(stages, list) or len(stages) != len(STAGE_IDS):
        raise ValueError("stages must contain exactly ten ordered stage objects")
    for index, (stage, stage_id) in enumerate(zip(stages, STAGE_IDS), 1):
        if not isinstance(stage, dict) or stage.get("id") != stage_id:
            raise ValueError(f"stage {index:02d} must have id {stage_id!r}")
        if str(stage.get("number", "")).zfill(2) != f"{index:02d}":
            raise ValueError(f"{stage_id}: number must match {index:02d}")
        for key in ("title", "summary"):
            if not isinstance(stage.get(key), str) or not stage[key].strip():
                raise ValueError(f"{stage_id}: {key} must be a non-empty string")
        tech = stage.get("tech")
        if not isinstance(tech, list) or not 2 <= len(tech) <= 3:
            raise ValueError(f"{stage_id}: tech must contain two or three short labels")
        if any(not isinstance(item, str) or not item.strip() for item in tech):
            raise ValueError(f"{stage_id}: each tech label must be a non-empty string")
        if stage.get("status") not in STATES:
            raise ValueError(f"{stage_id}: unknown evidence status")


def _units(text):
    return sum(1.0 if unicodedata.east_asian_width(char) in "WF" else
               0.28 if char in " .,;:!|'ilI" else 0.62 for char in text)


def _wrap(text, size, width):
    """Conservative Unicode-aware wrapping without dropping content."""
    lines = []
    for paragraph in text.splitlines() or [""]:
        line = ""
        for char in paragraph:
            if line and _units(line + char) * size > width:
                # Keep English tokens together when the preceding space is near.
                split = line.rfind(" ")
                if split > len(line) * 0.55 and ord(char) < 128 and char != " ":
                    lines.append(line[:split].rstrip())
                    line = line[split + 1:] + char
                else:
                    lines.append(line.rstrip())
                    line = char.lstrip()
            else:
                line += char
        if line:
            lines.append(line.rstrip())
    return lines or [""]


def _fit(text, size, width, max_lines, minimum):
    while True:
        lines = _wrap(text, size, width)
        if len(lines) <= max_lines:
            return lines, round(size, 2)
        size = round(size - 0.2, 2)
        if size < minimum:
            raise ValueError(f"Display text is too long; shorten this field: {text!r}")


def render(spec):
    """Return a deterministic SVG string.  Does not mutate spec or write files."""
    validate_spec(spec)
    svg = ET.Element(f"{{{NS}}}svg", {
        "viewBox": f"0 0 {WIDTH} {HEIGHT}", "width": str(WIDTH), "height": str(HEIGHT),
        "font-family": "'Noto Sans CJK SC','Microsoft YaHei','PingFang SC',Arial,sans-serif",
        "role": "group", "aria-roledescription": "交互流程图", "aria-labelledby": "progress-title progress-description",
        "data-schema-version": str(spec["schema_version"]),
    })

    def el(tag, attrs=None, value=None, parent=None):
        node = ET.SubElement(svg if parent is None else parent, f"{{{NS}}}{tag}",
                             {key: str(val) for key, val in (attrs or {}).items()})
        if value is not None:
            node.text = value
        return node

    def text(x, y, value, size=14, fill="#53657B", weight=400, parent=None,
             anchor="start", extra=None):
        attrs = {"x": x, "y": y, "font-size": size, "fill": fill,
                 "font-weight": weight, "text-anchor": anchor}
        attrs.update(extra or {})
        return el("text", attrs, value, parent)

    def box(x, y, width, height, fill, stroke="none", radius=14, parent=None, extra=None):
        attrs = {"x": x, "y": y, "width": width, "height": height, "rx": radius,
                 "fill": fill, "stroke": stroke}
        attrs.update(extra or {})
        return el("rect", attrs, parent=parent)

    def path(d, fill="none", stroke="#7890A5", width=2, parent=None, extra=None):
        attrs = {"d": d, "fill": fill, "stroke": stroke, "stroke-width": width,
                 "stroke-linecap": "round", "stroke-linejoin": "round"}
        attrs.update(extra or {})
        return el("path", attrs, parent=parent)

    def wrapped(x, y, value, size, width, maximum, minimum=10.5, leading=18,
                fill="#53657B", weight=400, parent=None):
        lines, fitted_size = _fit(value, size, width, maximum, minimum)
        node = text(x, y, None, fitted_size, fill, weight, parent)
        for index, line in enumerate(lines):
            el("tspan", {"x": x, "dy": 0 if index == 0 else leading}, line, node)
        return node

    def status_symbol(status, cx, cy, color, parent=None):
        el("circle", {"cx": cx, "cy": cy, "r": 6.2, "fill": "none",
                      "stroke": color, "stroke-width": 1.5}, parent=parent)
        if status == "supported":
            path(f"M {cx-3.3} {cy} l 2.2 2.4 l 4.4 -4.6", stroke=color,
                 width=1.7, parent=parent)
        elif status == "partial":
            path(f"M {cx} {cy-4.5} A 4.5 4.5 0 0 0 {cx} {cy+4.5} Z",
                 fill=color, stroke="none", parent=parent)
        else:
            path(f"M {cx} {cy-3.2} v 3.5 h 2.6", stroke=color,
                 width=1.4, parent=parent)

    el("title", {"id": "progress-title"}, spec["title"])
    el("desc", {"id": "progress-description"},
       spec["subtitle"] + "。" + spec["evidence_scope"] +
       "。十个可选择阶段顺时针排列；虚线将阶段十反馈到阶段二。"
       "在线 RL 从二直接到八；无需新 RGB 时从五直接到七。"
       "中心展示同一个世界的观察、策略与联合推进，推进后返回下一时刻观察；另列原位姿标签增强和反事实合成。"
       "阶段状态描述当前证据，不表示完成百分比；流程连通不证明学习型 SelfPlay、GS 质量或 RSI。")
    el("metadata", value=json.dumps({key: spec[key] for key in
        ("schema_version", "updated_at", "evidence_scope")}, ensure_ascii=False,
        sort_keys=True, separators=(",", ":")))
    defs = el("defs")
    for name, color in {"main": "#526D88", "teal": "#158374", "blue": "#3B74C5",
                        "violet": "#8361AF"}.items():
        marker = el("marker", {"id": "progress-arrow-" + name,
            "viewBox": "0 0 10 10", "refX": 8.5, "refY": 5, "markerWidth": 7,
            "markerHeight": 7, "orient": "auto", "markerUnits": "strokeWidth"}, parent=defs)
        el("path", {"d": "M 1 1 L 9 5 L 1 9 Z", "fill": color}, parent=marker)
    shadow = el("filter", {"id": "progress-shadow", "x": "-10%", "y": "-15%",
                           "width": "120%", "height": "135%"}, parent=defs)
    el("feDropShadow", {"dx": 0, "dy": 4, "stdDeviation": 6,
                        "flood-color": "#1D3A5A", "flood-opacity": ".065"}, parent=shadow)
    el("style", value="""
.stage-node { cursor:pointer; outline:none; }
.stage-node.is-muted,[data-edge].is-muted { opacity:.2; }
[data-edge] { transition:opacity .15s ease; }
.stage-node .stage-card { transition:stroke .15s ease,stroke-width .15s ease; }
.stage-node:hover .stage-card { stroke:#6B8EAF; stroke-width:2; }
.stage-node:focus .stage-card,.stage-node:focus-visible .stage-card,
.stage-node.selected .stage-card,.stage-node.is-selected .stage-card,
.stage-node[aria-pressed='true'] .stage-card { stroke:#173F68; stroke-width:3; }
.stage-node .selection-mark { opacity:0; }
.stage-node.selected .selection-mark,.stage-node.is-selected .selection-mark,
.stage-node[aria-pressed='true'] .selection-mark { opacity:1; }
text { text-rendering:optimizeLegibility; }
@media (prefers-reduced-motion:reduce) { .stage-node .stage-card { transition:none; } }
""")
    box(0, 0, WIDTH, HEIGHT, "#F5F8FC", radius=0)
    wrapped(56, 29, spec["title"], 20, 760, 1, 16, fill="#18344F", weight=700)
    for index, status in enumerate(STATES):
        label, color, _ = STATES[status]
        xx = 1150 + index * 128
        status_symbol(status, xx, 24, color)
        text(xx + 13, 29, label, 12.5, color, 600)
    text(1528, 57, "顺时针主线  /  按任务分流", 12.5, "#667F95", 500, anchor="end")
    text(1528, 80, "流程连通 ≠ 学习型 SelfPlay / GS 质量 / RSI 已验收", 12.5,
         "#667F95", anchor="end")

    # Draw connectors before panels. Their geometry stays outside every card.
    connectors = el("g", {"id": "progress-connectors", "aria-hidden": "true"})
    main_paths = (
        ("source", "selfplay", "M 378 202 H 432"),
        ("selfplay", "mining", "M 762 202 H 816"),
        ("mining", "teacher", "M 1146 202 H 1200"),
        ("teacher", "verify", "M 1368 294 V 358"),
        ("verify", "render", "M 1368 548 V 624"),
        ("render", "dataset", "M 1368 814 V 892"),
        ("dataset", "train", "M 1206 990 H 1152"),
        ("train", "evaluate", "M 822 990 H 768"),
        ("evaluate", "feedback", "M 438 990 H 384"),
    )
    for source, target, d in main_paths:
        path(d, stroke="#526D88", width=2.5, parent=connectors, extra={
            "id": f"edge-{source}-{target}", "data-edge": f"{source}-{target}", "data-from": source, "data-to": target,
            "marker-end": "url(#progress-arrow-main)"})
    path("M 54 990 H 24 V 64 H 600 V 104", stroke="#158374", width=2.6,
         parent=connectors, extra={"id": "edge-feedback-selfplay", "data-edge": "feedback-selfplay", "data-from": "feedback",
         "data-to": "selfplay", "stroke-dasharray": "7 6",
         "marker-end": "url(#progress-arrow-teal)"})
    text(56, 54, "晋级版本 / 指定对手 → 下一轮采样", 12.5, "#157C70", 600)
    path("M 540 294 V 328 H 408 V 856 H 984 V 892", stroke="#3B74C5", width=2.5,
         parent=connectors, extra={"id": "edge-selfplay-train", "data-edge": "selfplay-train", "data-from": "selfplay",
         "data-to": "train", "marker-end": "url(#progress-arrow-blue)"})
    text(696, 845, "在线 RL · 02 → 08 · 跳过教师与合成渲染", 14, "#3265B7", 600,
         anchor="middle")
    path("M 1530 458 H 1560 V 958 H 1536", stroke="#8361AF", width=2.3,
         parent=connectors, extra={"id": "edge-verify-dataset", "data-edge": "verify-dataset", "data-from": "verify",
         "data-to": "dataset", "marker-end": "url(#progress-arrow-violet)"})
    no_rgb = el("g", {"transform": "translate(1580 708) rotate(90)"})
    text(0, 0, "05 → 07  无需新 RGB · 复用原图 / 结构化数据", 12.5,
         "#7450A9", 500, no_rgb, anchor="middle")
    # Labels use their own small backing, preserving arrow visibility around them.
    for y, label in ((329, "候选与执行证据"), (590, "需要新图像时"), (858, "图像按任务准入")):
        width = _units(label) * 12.5 + 22
        box(1368 - width / 2, y - 15, width, 23, "#F5F8FC", radius=5)
        text(1368, y + 1, label, 12.5, "#5D7289", anchor="middle")

    support = el("g", {"id": "progress-support"})
    box(56, 366, 320, 442, "#EAF0F7", "#D9E3ED", 18, support)
    text(78, 402, "共同底座", 22, "#203F5B", 700, support)
    text(78, 426, "BUDGET  /  EVIDENCE", 11.5, "#6E849C", 600, support,
         extra={"letter-spacing": 1.2})
    box(76, 446, 280, 139, "#FFFFFF", radius=12, parent=support)
    text(94, 476, "预算与改进器 ψ", 18, "#294E6D", 700, support)
    for index, line in enumerate(("方案选择 → 执行 → 成本入账", "预算、时限、取消与恢复留痕", "ψ 的提升需要独立评测")):
        text(94, 505 + index * 25, line, 13.2, "#536B82", parent=support)
    box(76, 603, 280, 147, "#FFFFFF", radius=12, parent=support)
    text(94, 633, "证据与场景工作台", 18, "#294E6D", 700, support)
    for index, line in enumerate(("模型、数据、分支、资产可追溯", "成功、失败与拒绝原因都保留", "回放、规则与学习型角色分开")):
        text(94, 662 + index * 25, line, 13.2, "#536B82", parent=support)
    text(216, 781, "支持模块 · 不作为额外阶段", 12.5, "#71869B", parent=support,
         anchor="middle")

    center = el("g", {"id": "progress-common-world"})
    box(440, 366, 704, 442, "#FFFFFF", "#D9E3ED", 18, center)
    text(464, 403, "同一个世界，持续闭环", 25, "#183B59", 700, center)
    text(464, 429, "状态、动作、观察与分支身份始终对齐", 14, "#6B8095", parent=center)
    for index, (label, english) in enumerate((("同刻观察", "OBSERVE"), ("逐车决策", "POLICY"), ("联合推进", "JOINT ADVANCE"))):
        xx = 464 + index * 222
        box(xx, 451, 188, 66, "#F0F5FA", "#DFE7EF", 11, center)
        text(xx + 94, 477, label, 18, "#294E6D", 700, center, anchor="middle")
        text(xx + 94, 499, english, 10.5, "#7A8FA3", 600, center, anchor="middle",
             extra={"letter-spacing": 1})
        if index < 2:
            path(f"M {xx+195} 484 H {xx+214}", stroke="#526D88", width=2,
                 parent=center, extra={"marker-end": "url(#progress-arrow-main)"})
    # Return above the three boxes: joint advance produces the next observation.
    return_path = path("M 1002 450 V 442 H 558 V 450", stroke="#526D88", width=1.6,
                       parent=center, extra={"id": "progress-world-return",
                       "marker-end": "url(#progress-arrow-main)"})
    el("title", value="联合推进后，回到下一时刻的同刻观察", parent=return_path)
    box(464, 528, 656, 29, "#F6F2FC", radius=7, parent=center)
    text(792, 548, "视觉策略：当前世界先渲染再决策  ·  结构化策略可不渲染", 13.2,
         "#7655A0", 500, center, anchor="middle")
    box(464, 572, 656, 96, "#ECF8F4", radius=12, parent=center)
    box(480, 588, 28, 28, "#D4ECE2", radius=8, parent=center)
    text(494, 608, "A", 15, "#157C70", 700, center, anchor="middle")
    text(520, 607, "原位姿标签增强", 18, "#157C70", 700, center)
    text(1098, 607, "当前状态与观察历史不变", 12.5, "#557D72", parent=center,
         anchor="end")
    text(480, 635, "同一历史  →  复核轨迹  →  新的未来监督标签", 15, "#315F56", 500, center)
    text(480, 655, "可复用原图；未来图像不进入当前策略输入。", 12.5, "#648478", parent=center)
    box(464, 684, 656, 101, "#F5F0FD", radius=12, parent=center)
    box(480, 700, 28, 28, "#E8DDF5", radius=8, parent=center)
    text(494, 720, "B", 15, "#7450A9", 700, center, anchor="middle")
    text(520, 719, "反事实交互合成", 18, "#7450A9", 700, center)
    text(1098, 719, "新状态 / 机位 / 他车 → 新观察", 12.5, "#827096", parent=center,
         anchor="end")
    text(480, 747, "联合分支  →  按新状态 GS 渲染  →  按任务双准入", 15, "#675183", 500, center)
    text(480, 770, "可离线合成；图像失效阻断视觉用途，结构化资格另行检查。", 12.5,
         "#867597", parent=center)

    for index, (stage, (x, y), tone) in enumerate(zip(spec["stages"], LAYOUT, STAGE_TONES), 1):
        color, tint = TONES[tone]
        state_label, state_color, state_tint = STATES[stage["status"]]
        group = el("g", {"id": "stage-" + stage["id"], "class": "stage-node",
            "data-stage": stage["id"], "data-status": stage["status"], "tabindex": 0,
            "role": "button", "aria-pressed": "false",
            "aria-label": f"{index:02d} {stage['title']}，{state_label}。{stage['summary']}",
            "data-x": x, "data-y": y, "data-width": CARD_WIDTH, "data-height": CARD_HEIGHT})
        el("title", value=f"{index:02d} {stage['title']} · {state_label}", parent=group)
        box(x, y, CARD_WIDTH, CARD_HEIGHT, "#FFFFFF", "#D8E3EE", 16, group,
            {"class": "stage-card", "filter": "url(#progress-shadow)"})
        box(x + 20, y + 16, 38, 28, tint, radius=8, parent=group)
        text(x + 39, y + 36, f"{index:02d}", 16, color, 700, group, anchor="middle")
        box(x + 204, y + 17, 96, 25, state_tint, radius=12, parent=group)
        status_symbol(stage["status"], x + 217, y + 29.5, state_color, group)
        text(x + 229, y + 34, state_label, 11.5, state_color, 600, group)
        el("circle", {"class": "selection-mark", "cx": x + 72, "cy": y + 30,
                      "r": 3, "fill": "#173F68"}, parent=group)
        wrapped(x + 20, y + 71, stage["title"], 21, 280, 1, 15,
                fill="#203D58", weight=700, parent=group)
        wrapped(x + 20, y + 96, stage["summary"], 13.6, 280, 2, 11.2,
                leading=18, fill="#62778D", parent=group)
        path(f"M {x+20} {y+126} H {x+300}", stroke="#E9EFF5", width=1, parent=group)
        wrapped(x + 20, y + 147, " · ".join(stage["tech"]), 12.2, 280, 2, 10.5,
                leading=17, fill=color, weight=500, parent=group)

    text(56, 1105, "状态仅对应有界证据  ·  不代表全量验收  ·  选择阶段查看交付、边界与下一步", 12,
         "#71849A")
    text(1528, 1105, "DriveRSI  /  " + spec["updated_at"], 12, "#71849A", anchor="end")
    ET.indent(svg, space="  ")
    return ET.tostring(svg, encoding="unicode")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path, help="Versioned progress JSON")
    parser.add_argument("--output", type=Path, required=True, help="Output SVG file")
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    result = render(spec)
    ET.fromstring(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result, encoding="utf-8")
    print(json.dumps({"svg": str(args.output), "stages": len(STAGE_IDS),
                      "bytes": len(result.encode("utf-8"))}))


if __name__ == "__main__":
    main()