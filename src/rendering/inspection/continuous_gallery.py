"""Evidence-first index for continuous physical execution, not stitched source playback."""

import html
import json
from pathlib import Path


def export_gallery(root):
    root = Path(root)
    rows = []
    for path in sorted(root.glob("*/report.json")):
        v = json.loads(path.read_text())
        name = path.parent.name
        diagnosis = json.loads((path.parent / "diagnosis.json").read_text())
        bounds = " ".join(
            f'<a href="viewer.html?tick={b["tick"]}">{b["tick"] / 10:g}s</a>'
            for b in v["boundaries"]
        )
        checkpoints = "".join(
            "<tr><td>"
            + str(b["tick"])
            + "</td><td>"
            + str(b["present"])
            + "</td><td>"
            + str(b["ego_alive"])
            + "</td><td>"
            + html.escape(str(b["history_ticks"]))
            + "</td><td>"
            + html.escape(str(b["policy_memory"]))
            + "</td></tr>"
            for b in v["boundaries"]
        )
        contact_rows = "".join(
            f'<tr><td>{html.escape(str(p["actors"]))}</td><td><a href="anchor.html?tick={p["first_tick"]}">{p["first_tick"] / 10:g}s</a></td><td>{html.escape(p["onset"])}</td><td>{p["involves_owner"]}</td></tr>'
            for p in diagnosis["branches"]["baseline"]["contacts"]["pairs"]
        )
        page = f"""<h1>连续世界与同锚点诊断</h1><a href="../index.html">← 全部场景</a>
<p>{html.escape(v["source_log"])} · {html.escape(v["city"])}</p>
<p>{v["seconds"]} 秒真实物理推进，{v["actors"]} 个窗口内对象；{v["feedback_actors"]} 个交通反馈对象，其中行人 {v["feedback_pedestrians"]}。策略车辆的有效决策步数：{v["ego_valid_steps"]} / {v["steps"]}。</p>
<p>使用明确声明的日志路线作为特权导航及规则控制；非模型训练、非 PDM 验收。未接管的对象仍按源标注回放，接管对象在支持耗尽后可能停留；不能把本次连续性验收当成交通真实性通过。</p>
<nav><a href="viewer.html">连续执行回放</a> · <a href="source.html">同时间范围源日志</a> · <a href="anchor.html?tick={diagnosis["anchor_tick"]}">同锚点原策略／制动对照</a> · <a href="report.json">完整结果</a> · <a href="execution-evidence.json">全程接触／地图证据</a> · <a href="diagnosis.json">诊断证据</a></nav>
<h2>记录边界，不是重置点</h2><p>{bounds}</p><table><tr><th>Tick</th><th>当前对象</th><th>Ego仍有效</th><th>实际历史Tick</th><th>策略记忆</th></tr>{checkpoints}</table>
<p>检查点恢复：{v["restore"]["start_tick"]} → {v["restore"]["end_tick"]}；状态、动作、观测、有效掩码与奖励精确一致。规则控制器记忆仅为决策计数；递归记忆与随机数另由状态型策略回归验证。</p>
<h2>同锚点诊断</h2><p>锚点 {diagnosis["anchor_tick"] / 10:g}s；重复执行一致：{diagnosis["repeat_exact"]}；干预动作不同：{diagnosis["action_changed"]}。</p><p>证据分类：{html.escape(", ".join(diagnosis["evidence_flags"]))}。事故责任：未确定。制动改变接触结果不等于模型负责。</p>
<table><tr><th>接触对象</th><th>首次时刻</th><th>起始类型</th><th>包含Ego</th></tr>{contact_rows or '<tr><td colspan="4">该锚点窗口未记录接触</td></tr>'}</table>
<h2>完整来源对照</h2><p>逐对接触同时保留源时刻是否仍有标注、源几何是否重叠及反事实控制归属。源标注缺失或接触差异都不是事故责任结论，详见诊断证据。</p><h2>全程地图支持</h2><p>道路从真实同城 GPKG 提取，覆盖全段源参与者范围，外扩80米。每步检查车身轮廓与地图裁切范围；范围内不等于交通规则已验证。Ego边界警告：{len(v["map_crop_warning_ticks"])} 个Tick；全程Ego接触：{v["owner_contact_ticks"]} 个Tick。</p>
"""
        (path.parent / "evidence.html").write_text(shell(page), encoding="utf-8")
        if (path.parent / "blockage.html").exists():
            page += '<p><a href="blockage.html">源框重叠与控制器等待链诊断</a></p>'
            (path.parent / "evidence.html").write_text(shell(page), encoding="utf-8")
        rows.append(
            f'<article><h2>{html.escape(v["city"])} · {html.escape(name)}</h2><p>{v["seconds"]}s · {v["actors"]}对象 · {v["feedback_actors"]}交通反馈 · Ego有效{v["ego_valid_steps"]}/{v["steps"]}步</p><a href="{name}/evidence.html">连续性与诊断证据</a> · <a href="{name}/viewer.html">播放连续物理世界</a> · <a href="{name}/anchor.html?tick={diagnosis["anchor_tick"]}">同锚点对照</a></article>'
        )
    content = (
        """<h1>nuPlan 连续物理世界与失败诊断</h1><p>同一个世界持续联合推进。9秒仅为记录和调度分块，不重置对象、观测历史或策略状态。</p><p>当前是有界预编译连续段：最多600秒、全段累计1024对象、地图宽高各不超过2公里；不拼接不连续DB，不声称无界流式仿真。数据仍为 trainable=false。</p><p><a href="../nuplan-capacity-v1/logs/log-07/viewer.html">Las Vegas完整198.95秒源日志：已补齐连续道路</a> · <a href="../nuplan-capacity-v1/full-map-report.json">8份日志地图覆盖报告</a></p>"""
        + "".join(rows)
    )
    if (root / "log-03-motion-v2/blockage.html").exists():
        content = '<p><a href="log-03-motion-v2/blockage.html">当前阻塞：源日志与控制器仿真对照</a></p>' + content
    if (root / "vehicle12-comparison.html").exists():
        content = '<p><a href="vehicle12-comparison.html">12号车误移动：修复前后与剩余接触问题</a></p>' + content
    (root / "index.html").write_text(shell(content), encoding="utf-8")
    return {"cases": len(rows), "index": str(root / "index.html")}


def shell(content):
    return (
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DriveRSI · 连续世界验收</title><style>body{max-width:1200px;margin:auto;padding:28px;font:16px/1.7 system-ui;background:#f0f4f6;color:#193444}h1{font-size:30px}h2{font-size:21px}a{color:#087c84;margin-right:10px}article,nav,table{background:white;border:1px solid #cbd9df;border-radius:8px;padding:18px;margin:20px 0}table{border-collapse:collapse;display:block;overflow:auto}td,th{border:1px solid #d5dfe5;padding:8px;text-align:left}p{max-width:1000px}</style>'
        + content
        + "</html>"
    )
