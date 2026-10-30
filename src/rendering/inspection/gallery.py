"""Standalone evidence index; rendering never upgrades a result into qualification."""

import html
import math
from pathlib import Path, PurePosixPath
from urllib.parse import quote, unquote, urlsplit

_STATUSES = {
    "complete": "执行完成",
    "unresolved_task": "任务未解析",
    "compile_rejected": "编译拒绝",
    "runtime_error": "执行异常",
    "contract_failure": "执行契约失败",
}
_PROBE_STATUSES = {
    "not_run": "未验证",
    "verified": "选定跟驰对响应已验证",
    "evidence_insufficient": "交互证据不足",
    "not_applicable": "不适用当前跟驰探针",
    "failed": "探针检查失败",
}


def _text(value) -> str:
    return html.escape(str(value), quote=True)


def _number(value, suffix: str = "") -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return "—"
    return f"{value:,.2f}".rstrip("0").rstrip(".") + suffix


def _href(value: str, root: Path) -> str:
    """Accept only local descendant assets, including when a parent is a symlink."""
    if not isinstance(value, str) or not value or unquote(value) != value:
        raise ValueError("Gallery links require an unencoded relative asset path")
    url = urlsplit(value)
    path = PurePosixPath(value)
    if (url.scheme or url.netloc or url.query or url.fragment or path.is_absolute()
            or "\\" in value or ":" in value or any(ord(c) < 32 for c in value)
            or ".." in path.parts or not (root / value).resolve().is_relative_to(root.resolve())):
        raise ValueError("Gallery links must remain inside the output directory")
    return _text(quote(value, safe="/"))


def _link(path, label: str, root: Path, *, secondary=False) -> str:
    if not path:
        return ""
    style = "link secondary" if secondary else "link"
    return f'<a class="{style}" href="{_href(path, root)}">{_text(label)} ↗</a>'


def _metric(label: str, value, suffix="") -> str:
    return f'<div class="metric"><span>{_text(label)}</span><strong>{_number(value, suffix)}</strong></div>'


def _verification(value) -> str:
    return "通过" if value is True else "失败" if value is False else "未验证"


def _real_card(case: dict, root: Path, requested) -> str:
    status = case["status"]
    summary = case.get("summary") or {}
    interaction = case.get("interaction") or {}
    population = case.get("population") or {}
    events = summary.get("events") or {}
    verification = case.get("verification") or {}
    horizon = case.get("horizon_plan") or {}
    probe = case.get("interaction_probe") or {"status": "not_run"}
    city = str(case.get("map_name") or "地图未解析")
    metrics = "".join([
        _metric("请求时域", case.get("requested_seconds", requested), " s"),
        _metric("计划时域", case.get("planned_seconds"), " s"),
        _metric("实际记录时域", case.get("actual_seconds"), " s"),
        _metric("≥2 辆受控车有效", interaction.get("two_controlled_seconds"), " s"),
    ])
    horizon_html = ""
    if horizon:
        factors = ", ".join(horizon.get("limiting_factors") or []) or "无缩短限制"
        horizon_html = (
            '<p class="facts">时域模式 <b>' + _text(horizon.get("mode", "未提供"))
            + '</b> · 连续源未来 <b>' + _number(horizon.get("available_future_seconds"), " s")
            + '</b> · 后端上限 <b>' + _number(horizon.get("backend_max_duration_seconds"), " s")
            + '</b><br>限制：' + _text(factors)
            + ' · 向下量化损失 ' + _number(horizon.get("quantization_loss_seconds"), " s") + '</p>'
        )
    exposure_html = ('<p class="facts">≥1 辆受控车有效 <b>'
                     + _number(interaction.get("any_controlled_seconds"), " s") + '</b></p>')
    probe_status = probe.get("status", "not_run")
    probe_metrics = probe.get("metrics") or {}
    probe_links = "".join(_link(view["path"], view["label"], root, secondary=True)
                           for view in probe.get("views", []))
    probe_links += _link(probe.get("report_path"), "完整探针结果", root, secondary=True)
    probe_html = (
        '<div class="probe"><p class="facts">真实场景交互探针：<b>'
        + _text(_PROBE_STATUSES[probe_status]) + '</b></p>'
        + ('<p class="small">' + _text(probe["reason"]) + '</p>' if probe.get("reason") else "")
        + ('<p class="facts">目标响应延迟 <b>'
           + _number(probe_metrics.get("desired_response_delay_s"), " s")
           + '</b> · 执行响应延迟 <b>' + _number(probe_metrics.get("actual_response_delay_s"), " s")
           + '</b></p>' if probe_metrics else "")
        + "".join('<p class="small">' + _text(note) + '</p>'
                  for note in probe.get("comparison_notes", []))
        + ('<div class="actions">' + probe_links + '</div>' if probe_links else "") + '</div>'
    )
    rows = []
    for actor, item in (summary.get("actors") or {}).items():
        active = (interaction.get("per_actor_active_seconds") or {}).get(actor)
        actor_events = ", ".join(name for name, happened in (item.get("events") or {}).items() if happened)
        initial_events = ", ".join(name for name, happened in (item.get("initial_events") or {}).items()
                                   if happened)
        initial_status = ("初始失效" if item.get("initial_inactive") is True else
                          "初始有效" if item.get("initial_inactive") is False else "初始状态未提供")
        initial_text = initial_status + "；" + (
            initial_events or ("无初始事件" if "initial_events" in item else "初始事件未提供"))
        rows.append(f'<tr><th scope="row">{_text(actor)}</th><td>{_number(active, " s")}</td>'
                    f'<td>{_number(item.get("valid_steps"))}</td>'
                    f'<td>{_number(item.get("route_progress_m"), " m")}</td>'
                    f'<td>{_text(initial_text)}</td>'
                    f'<td>{_text(actor_events or "推进中无记录事件")}</td></tr>')
    actors = ("<details><summary>逐车有效时长与路线进度</summary><div class=table-wrap><table>"
              "<thead><tr><th>受控 ID</th><th>有效时长</th><th>有效步</th><th>路线进度</th>"
              "<th>初始化状态 / 事件</th><th>推进事件</th></tr></thead><tbody>"
              + "".join(rows) + "</tbody></table></div></details>"
              if rows else "")
    population_html = ""
    if population:
        retained = population.get("retained_previous_ids")
        lost = population.get("lost_previous_ids")
        added = population.get("new_world_ids")
        def counts(value):
            return len(value) if isinstance(value, list) else value
        population_html = (
            '<div class="population"><span>物理世界人口 · 基准窗口 '
            + _number(population.get("previous_duration_seconds"), " s") + ' → 本窗口</span>'
            f'<strong>{_number(population.get("previous_world"))} → {_number(population.get("world"))}</strong>'
            f'<span>其中受控 {_number(population.get("controlled"))}；保留 {_number(counts(retained))}'
            f' / 丢失 {_number(counts(lost))} / 新纳入 {_number(counts(added))}</span></div>'
        )
    initial_events = summary.get("initial_events") or {}
    initial_html = (f'<p class="facts">初始化 / reset · 受控车失效 <b>{_number(summary.get("initial_inactive_actors"))}</b>'
                    f' · 碰撞标记 <b>{_number(initial_events.get("collision"))}</b>'
                    f' · 越界标记 <b>{_number(initial_events.get("offroad"))}</b>'
                    f' · goal 标记 <b>{_number(initial_events.get("goal"))}</b>'
                    ' <span class="small">（车辆数；不计作推进事件）</span></p>' if summary else "")
    event_html = (f'<p class="facts">有效受控 transitions <b>{_number(summary.get("valid_agent_steps"))}</b></p>'
                  f'<p class="facts">推进事件 · 碰撞 <b>{_number(events.get("collision"))}</b>'
                  f' · 越界 <b>{_number(events.get("offroad"))}</b>'
                  f' · goal <b>{_number(events.get("goal"))}</b>'
                  ' <span class="small">（受控 actor-episodes；每车每类最多计一次）</span></p>' if summary else "")
    checks = (f'<p class="small">相同联合动作重放：{_verification(verification.get("joint_action_replay_exact"))}'
              f' · 中段恢复：{_verification(verification.get("mid_episode_restore_exact"))}</p>'
              if verification else "")
    error = f'<pre class="error">{_text(case["error"])}</pre>' if case.get("error") else ""
    warning = (f'<div class="error"><b>初始路线长度不足以满足制动距离下界</b><br>'
               f'{_text(", ".join(case["route_warnings"]))}<br>'
               '该警告不因执行完成而消失；运行结果须结合终止事件与路线进度解释。</div>'
               if case.get("route_warnings") else "")
    return (
        f'<article class="case-card" data-status="{status}" data-map="{_text(city)}">'
        f'<div class="card-top"><span class="eyebrow">{_text(city)}</span>'
        f'<span class="badge {"ok" if status == "complete" else "warn"}">{_STATUSES[status]}</span></div>'
        f'<h3>{_text(case["id"])}</h3><p class="source">{_text(case.get("source_scene", "来源未记录"))}</p>'
        f'<div class="metrics">{metrics}</div>{horizon_html}{exposure_html}{population_html}'
        f'{initial_html}{event_html}{probe_html}{error}{warning}{checks}{actors}'
        f'<div class="actions">{_link(case.get("viewer"), "打开交互回放", root)}'
        f'{_link(case.get("details"), "完整场景结果", root, secondary=True)}</div></article>'
    )


def _following_card(case: dict, root: Path) -> str:
    metrics = case.get("metrics") or {}
    reactive = metrics.get("reactive") or {}
    rows = []
    labels = (("reactive", "keep", "反应式 · 前车维持"),
              ("reactive", "brake", "反应式 · 前车制动"),
              ("reactive", "far", "反应式 · 远车制动"),
              ("replay", "keep", "回放 · 前车维持"),
              ("replay", "brake", "回放 · 前车制动"))
    for mode, branch, label in labels:
        values = (metrics.get(mode) or {}).get(branch) or {}
        contact = values.get("geometric_contact_any")
        rows.append(f'<tr><th scope="row">{label}</th>'
                    f'<td>{_number(values.get("minimum_bumper_gap_m"), " m")}</td>'
                    f'<td>{_number(values.get("minimum_ttc_seconds"), " s")}</td>'
                    f'<td>{"有" if contact is True else "无" if contact is False else "—"}</td></tr>')
    views = "".join(_link(view["path"], view["label"], root, secondary=True)
                    for view in case.get("views", []))
    passed = case.get("passed")
    badge = "规范检查通过" if passed is True else "规范检查失败" if passed is False else "未完成"
    return (
        '<article class="paired-card">'
        f'<div class="card-top"><span class="eyebrow">预先固定 · 配置 {_text(case["case_index"])}</span>'
        f'<span class="badge {"ok" if passed is True else "warn"}">{badge}</span></div>'
        f'<h3>{_number(case.get("speed_mps"))} m/s <span>初速 /</span> {_number(case.get("gap_m"))} m <span>初始车距</span></h3>'
        '<div class="metrics paired-metrics">'
        + _metric("目标加速度响应", reactive.get("desired_response_delay_s"), " s")
        + _metric("执行加速度响应", reactive.get("actual_response_delay_s"), " s")
        + '</div><div class="table-wrap"><table><thead><tr><th>分支</th><th>最小净车距</th>'
        '<th>最小 TTC</th><th>几何接触</th></tr></thead><tbody>' + "".join(rows)
        + '</tbody></table></div><p class="small">TTC 的 — 表示没有定义值或未提供；无碰撞不等于无风险。'
        f'本配置检查 {_number(case.get("checks_passed"))} / {_number(case.get("checks_total"))}。</p>'
        f'<div class="actions vertical">{views}</div></article>'
    )


def export_gallery(report: dict, output: Path) -> None:
    """Render a versioned report without fetching data or hiding failure cases."""
    if (type(report.get("schema_version")) is not int or report["schema_version"] not in (1, 2)
            or report.get("kind") != "interaction_gallery"
            or report.get("status") not in ("running", "complete", "error")):
        raise ValueError("Unsupported interaction gallery report")
    cases = report.get("real_cases", [])
    if not isinstance(cases, list) or any(case.get("status") not in _STATUSES for case in cases):
        raise ValueError("Unsupported gallery case status")
    if any((case.get("interaction_probe") or {}).get("status", "not_run") not in _PROBE_STATUSES
           for case in cases):
        raise ValueError("Unsupported gallery interaction probe status")
    output = Path(output)
    root = output.parent
    requested = report.get("requested_seconds")
    completed = sum(case["status"] == "complete" for case in cases)
    cities = sorted({str(case.get("map_name") or "地图未解析") for case in cases})
    city_options = "".join(f'<option value="{_text(city)}">{_text(city)}</option>' for city in cities)
    following = report.get("following") or {}
    selected_cases = following.get("selected_cases", [])
    following_cards = "".join(_following_card(case, root) for case in selected_cases)
    selection_text = ("下列预先固定的配置编号：" + "、".join(_text(case["case_index"]) for case in selected_cases)
                      + "（从 0 计数）。" if selected_cases else "尚未生成所选配对展示。")
    real_cards = "".join(_real_card(case, root, requested) for case in cases)
    summary = "".join([
        _metric("真实来源起点", len(cases)),
        _metric("执行完成 / 非资格通过", completed),
        _metric("保留的失败与未解析", len(cases) - completed),
        (_metric("真实跟驰探针已验证", sum((case.get("interaction_probe") or {}).get("status") == "verified"
                                          for case in cases))
         if report["schema_version"] == 2 else _metric("请求回合长度", requested, " s")),
    ])
    pending = ('<div class="pending" role="status">生成仍在进行 · 当前报告未完成，不能作为最终验收结果。</div>'
               if report["status"] == "running" else "")
    if report["status"] == "error":
        pending = ('<div class="pending" role="alert"><b>样例集生成失败 · 当前结果不完整</b>'
                   f'<pre class="error">{_text(report.get("error", "错误原因未提供"))}</pre>'
                   '以下保留已完成的来源结果与配对证据；不能将部分结果作为完整验收结论。</div>')
    html_body = (
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<title>DriveRSI · 交互证据与逐场景时域</title><style>' + _CSS + '</style></head><body>'
        '<header><a class="brand" href="#top">DriveRSI <span>INTERACTION EVIDENCE</span></a>'
        '<nav><a href="#paired">配对干预</a><a href="#real">真实场景</a><a href="#limits">验收边界</a></nav></header>'
        '<main id="top"><section class="hero"><p class="eyebrow">多智能体交互 · 证据可追溯</p>'
        '<h1>让交互可见，<br><span>让有效时长可核查。</span></h1>'
        '<p class="intro">更多样例用于检查可重复的行为与边界。视频播放多久、车辆真正交互多久，分别记录。</p>'
        '<div class="hero-note"><b>时长按场景记录</b><span>请求是预算，计划来自已声明的时域策略，'
        '实际时长来自世界推进。既有后端上限仍为 9 秒，尚未支持长期对象进出场；'
        '真实窗口不必与合成配对协议同长，执行完成也不等于交互已验证。</span></div>'
        f'{pending}<div class="summary metrics">{summary}</div></section>'
        '<section id="paired"><div class="section-heading"><div><p class="eyebrow">01 · 因果对照</p>'
        '<h2>前车改变，后车如何响应？</h2></div><span class="tag">一个合成跟驰模板</span></div>'
        '<p class="section-copy">同一模式内共享干预前缀，在同一快照分叉：前车制动作为正干预，'
        '远车制动与回放后车作为负对照。此合成协议计划时域 '
        + _number(following.get("duration_seconds"), " s") + '，干预时刻 '
        + _number(following.get("intervention_seconds"), " s")
        + '；同一对照分支共享时域，提前停止不补帧。指标按既定干预后区间计算。</p>'
        '<div class="protocol"><span>全部配置 <b>' + _number(following.get("case_count"))
        + '</b></span><span>规范检查 <b>' + _number(following.get("checks_passed"))
        + ' / ' + _number(following.get("checks_total")) + '</b></span><span>总体检查 <b>'
        + _verification(following.get("passed")) + '</b></span>'
        + _link(following.get("report_path"), "完整协议结果", root, secondary=True) + '</div>'
        f'<p class="small">{selection_text}速度和车距变体不是独立交通类型，'
        '三种双分支视图也不是三类驾驶能力。</p><div class="paired-grid">' + following_cards + '</div></section>'
        '<section id="real"><div class="section-heading"><div><p class="eyebrow">02 · 真实数据覆盖</p>'
        '<h2>每个起点，都保留结果。</h2></div><span class="tag">开发回归集 · 非持留测试集</span></div>'
        '<p class="section-copy">物理对象仍需在整个编译窗口存在。窗口加长可能减少交通人口；'
        '日志参考对象只补充显示，不补充物理交互。执行完成只表示流程完成，事件与有效时长仍须逐项阅读。</p>'
        '<div class="filters"><label>执行结果<select id="status-filter"><option value="all">全部结果</option>'
        '<option value="complete">执行完成</option><option value="issues">失败 / 未解析</option></select></label>'
        '<label>地图区域<select id="map-filter"><option value="all">全部区域</option>' + city_options
        + '</select></label><span id="visible-count" aria-live="polite">' + str(len(cases))
        + ' / ' + str(len(cases)) + ' 个来源起点</span></div><div class="real-grid">' + real_cards
        + '</div><p id="empty" hidden>当前筛选没有结果；切换“全部结果”可查看所有失败与成功项。</p></section>'
        '<section id="limits" class="limits"><p class="eyebrow">03 · 解释边界</p><h2>这些证据能说明什么？</h2>'
        '<div class="limits-grid"><div><h3>可检查</h3><p>短时同刻观测与联合推进、前车制动响应、'
        '无关干预负对照、固定动作重放和中段恢复。真实场景可检查路线、事件、人口损失及提前停止。</p></div>'
        '<div><h3>尚不能推出</h3><p>真实数据视频本身不证明因果交互；有效受控车时长也不代表始终存在交互。'
        '本集合不证明合流、路权、长期稳定、并发吞吐或正式训练资格。</p></div>'
        '<div><h3>合成数据分层</h3><p>世界状态、动作和事件来自仿真执行；源日志 Context 只是诊断参考，'
        '车辆偏离后不能成为反事实世界真值。goal 事件也不自动等于合法路线完成。</p></div></div></section>'
        '</main><footer>DriveRSI · 独立静态索引 · 所有页面与资产位于同一产物目录</footer>'
        '<script>' + _SCRIPT + '</script></body></html>'
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html_body, encoding="utf-8")


_SCRIPT = """
const cards = [...document.querySelectorAll('.case-card')];
const statusFilter = document.getElementById('status-filter');
const mapFilter = document.getElementById('map-filter');
function applyFilter() {
  let count = 0;
  for (const card of cards) {
    const state = statusFilter.value;
    const selected = state === 'all' || (state === 'issues' ? card.dataset.status !== 'complete' : card.dataset.status === state);
    card.hidden = !(selected && (mapFilter.value === 'all' || card.dataset.map === mapFilter.value));
    if (!card.hidden) count++;
  }
  document.getElementById('visible-count').textContent = `${count} / ${cards.length} 个来源起点`;
  document.getElementById('empty').hidden = count !== 0;
}
statusFilter.addEventListener('change', applyFilter);
mapFilter.addEventListener('change', applyFilter);
"""

_CSS = """
:root{color-scheme:light;--ink:#172d3b;--muted:#647582;--line:#dce5e7;--teal:#147b78;--paper:#f3f6f5;--amber:#805413}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:80px}body{margin:0;background:var(--paper);color:var(--ink);font:14px/1.7 system-ui,-apple-system,'Segoe UI','Microsoft YaHei',sans-serif}a{color:inherit}header{max-width:1456px;margin:auto;padding:22px 40px;display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid var(--line)}.brand{text-decoration:none;font-size:22px;font-weight:750;letter-spacing:-1px}.brand span{font-size:10px;letter-spacing:2px;margin-left:20px;color:var(--teal)}nav{display:flex;gap:26px}nav a{color:var(--muted);text-decoration:none;font-size:13px}nav a:hover{color:var(--teal)}main{max-width:1456px;margin:auto;padding:0 40px}section{padding:45px 0;scroll-margin-top:20px}.hero{padding-top:56px}.eyebrow{font-size:11px;letter-spacing:1.6px;text-transform:uppercase;color:var(--teal);font-weight:650;margin:0 0 10px}h1{font-size:clamp(36px,4vw,60px);letter-spacing:-2px;line-height:1.17;margin:15px 0 20px}h1 span{color:#617c86;font-weight:450}h2{font-size:28px;letter-spacing:-.7px;line-height:1.3;margin:0}h3{font-size:17px;line-height:1.5;margin:14px 0 7px;overflow-wrap:anywhere}h3 span{font-size:12px;color:var(--muted);font-weight:450}.intro,.section-copy{max-width:950px;color:var(--muted);font-size:15px}.hero-note{display:flex;gap:18px;align-items:flex-start;max-width:1120px;padding:17px 20px;background:#e7efed;border-left:3px solid var(--teal);margin:26px 0}.hero-note b{flex-shrink:0}.hero-note span{color:#4d666e}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.summary{margin-top:32px;gap:0;border:1px solid var(--line);border-radius:12px;overflow:hidden;background:white}.metric{display:flex;flex-direction:column;gap:5px;min-width:0}.metric>span{font-size:11px;color:var(--muted)}.metric strong{font-size:21px;font-weight:620;font-variant-numeric:tabular-nums}.summary .metric{padding:19px 23px;border-right:1px solid var(--line)}.summary .metric:last-child{border:0}.summary strong{font-size:30px}.section-heading{display:flex;align-items:center;justify-content:space-between;gap:16px}.tag{font-size:11px;border:1px solid #becfce;border-radius:30px;padding:6px 11px;white-space:nowrap;color:#536e72}.protocol{display:flex;align-items:center;gap:26px;flex-wrap:wrap;padding:15px 18px;border-radius:10px;background:#e7eeec;margin:22px 0 13px;color:#4d666e}.protocol b{color:var(--ink);margin-left:5px}.protocol a{margin-left:auto}.paired-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px;margin-top:20px}.paired-card,.case-card{background:#fff;border:1px solid var(--line);padding:22px;border-radius:13px;min-width:0;box-shadow:0 4px 15px #243d4b04}.card-top{display:flex;justify-content:space-between;gap:8px;align-items:center}.card-top .eyebrow{margin:0;font-size:10px;letter-spacing:.7px}.badge{border-radius:5px;padding:3px 7px;white-space:nowrap;font-size:10px;font-weight:600}.ok{background:#e6f1ec;color:#286650}.warn{background:#fff0d9;color:var(--amber)}.paired-metrics{grid-template-columns:repeat(2,1fr);background:#f5f8f7;border-radius:7px;padding:13px;margin:18px 0}.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:11px;text-align:left}th,td{padding:9px 5px;border-bottom:1px solid #e8edee}thead th{font-weight:500;color:var(--muted);white-space:nowrap}tbody th{font-weight:450;overflow-wrap:anywhere;max-width:150px}td{white-space:nowrap;font-variant-numeric:tabular-nums}.small{font-size:11px;color:var(--muted);margin:13px 0 0}.actions{display:flex;flex-wrap:wrap;gap:9px;margin-top:17px}.vertical{flex-direction:column}.link{display:inline-flex;text-decoration:none;font-size:12px;align-items:center;justify-content:center;background:var(--teal);color:white;border:1px solid var(--teal);border-radius:6px;padding:7px 12px;font-weight:550}.link:hover{filter:brightness(.95)}.secondary{background:#f6f9f8;color:var(--teal);border-color:#cddedc}.filters{display:flex;gap:20px;align-items:center;background:white;border:1px solid var(--line);border-radius:9px;padding:13px 16px;margin:23px 0 18px}.filters label{display:flex;align-items:center;gap:10px;font-size:12px;color:var(--muted)}select{background:#f6f8f7;border:1px solid #d7e1e2;border-radius:5px;padding:7px 28px 7px 9px;color:var(--ink);max-width:320px;font:inherit}.filters>span{font-size:12px;color:var(--muted);margin-left:auto}.real-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.case-card>.metrics{padding:15px 0;border-top:1px solid #edf1f1;border-bottom:1px solid #edf1f1;margin-top:16px}.case-card .metric strong{font-size:20px}.source{font-family:ui-monospace,Consolas,monospace;font-size:10px;color:var(--muted);overflow-wrap:anywhere;margin:0}.population{display:flex;gap:6px 12px;flex-wrap:wrap;align-items:center;font-size:11px;margin:15px 0;color:var(--muted)}.population strong{color:var(--ink);font-size:14px}.population span:last-child{width:100%}.facts{font-size:12px;color:var(--muted)}.facts b{color:var(--ink)}details{margin-top:13px;font-size:12px;border-top:1px solid #edf1f1;padding-top:11px}summary{cursor:pointer;color:var(--teal)}.error{white-space:pre-wrap;overflow-wrap:anywhere;font-family:ui-monospace,Consolas,monospace;font-size:11px;background:#fff8ee;border:1px solid #f0dfc4;border-radius:7px;padding:12px;color:#795932;max-height:180px;overflow:auto}.limits{border-top:1px solid var(--line);margin-top:28px}.limits-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:35px;margin-top:18px}.limits-grid p{color:var(--muted);font-size:13px}.limits-grid h3{font-size:15px}footer{padding:25px 40px;text-align:center;font-size:11px;color:var(--muted);border-top:1px solid var(--line);margin-top:20px}.pending{background:#fff0d9;color:var(--amber);padding:12px 17px;border-radius:6px}.case-card[hidden],[hidden]{display:none!important}a:focus-visible,select:focus-visible,summary:focus-visible{outline:3px solid #88b9b5;outline-offset:3px}#empty{color:var(--muted);padding:30px;text-align:center}@media(max-width:1050px){.paired-grid{grid-template-columns:1fr}.paired-card .table-wrap{max-width:none}.vertical{flex-direction:row}.real-grid{grid-template-columns:1fr}.paired-card h3{font-size:20px}.paired-metrics{max-width:450px}.brand span{display:none}}@media(max-width:640px){header,main{padding-left:18px;padding-right:18px}header{padding-top:17px;padding-bottom:17px}nav{gap:15px}nav a{font-size:11px}.hero{padding-top:35px}h1{letter-spacing:-1px}.hero-note{flex-direction:column;gap:5px}.summary{grid-template-columns:repeat(2,1fr)}.summary .metric{border-bottom:1px solid var(--line)}.summary strong{font-size:25px}.section-heading{align-items:flex-start;flex-direction:column}h2{font-size:24px}.paired-card,.case-card{padding:16px}.filters{flex-wrap:wrap;gap:10px}.filters label{width:100%;justify-content:space-between}.filters select{max-width:72%}.filters>span{margin-left:0}.limits-grid{grid-template-columns:1fr;gap:6px}.case-card>.metrics{grid-template-columns:repeat(2,1fr);gap:16px}.protocol{gap:10px 20px}.protocol a{margin-left:0}}
"""
