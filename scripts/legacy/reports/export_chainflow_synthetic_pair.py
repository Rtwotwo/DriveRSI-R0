"""Publish sealed synthetic-pair progress without turning blocked stages into gains."""

import argparse
import json
import os
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def export(root, experiment, destination, render):
    root, experiment, destination, render = [
        Path(x).resolve() for x in (root, experiment, destination, render)
    ]
    if (
        not experiment.is_relative_to(root / "exp")
        or not render.is_relative_to(experiment)
        or not destination.is_relative_to(root / "outputs/demonstrations")
    ):
        raise ValueError("Gallery inputs must belong to this experiment")
    data = read(render / "report.json")
    if data["native_future_points"] != 10 or data["image_count"] != 60:
        raise ValueError("Complete actual ten-point render required")
    destination.mkdir(parents=True, exist_ok=True)
    images = destination / "images"
    images.mkdir(exist_ok=True)

    def link(path, name):
        path = Path(path).resolve(strict=True)
        target = images / name
        if target.is_symlink():
            if target.resolve() != path:
                raise ValueError("Gallery evidence identity changed")
        elif target.exists():
            raise ValueError("Gallery evidence cannot be replaced")
        else:
            os.symlink(path, target)
        return "images/" + name

    frames = []
    for frame in data["frames"]:
        for camera, item in frame["cameras"].items():
            source = (
                Path("/data/xinyao/navsim_workspace/dataset/sensor_blobs/trainval_all")
                / item["source_image_relative_path"]
            )
            frames.append(
                dict(
                    tick=frame["tick"],
                    camera=camera,
                    original=link(source, "source-" + item["file"] + ".jpg"),
                    novel=link(render / item["file"], item["file"]),
                )
            )
    branches = []
    for path in sorted((experiment / "tasks").glob("*/teacher/execution/report.json")):
        value = read(path)
        for name, row in value.get("branches", {}).items():
            page = path.parent / (name + ".html")
            branches.append(
                dict(
                    task=path.parents[2].name,
                    branch=name,
                    source_log=value["source_log"],
                    completed_steps=row["completed_steps"],
                    failure=row.get("failure"),
                    score=row.get("score"),
                    admitted=row.get("trajectory_admitted", False),
                    object_contacts=len(row.get("object_contacts", [])),
                    replay=link(page, path.parents[2].name + "-" + name + ".html")
                    if page.exists()
                    else None,
                )
            )
    baseline = read(experiment / "development/parent-baseline-v2/report.json")
    training = []
    for path in sorted(experiment.rglob("training-result.json")):
        row = read(path)
        training.append(
            {
                "receipt": str(path.relative_to(experiment)),
                **{
                    k: row.get(k)
                    for k in ("status", "phase", "updates", "total_updates", "metrics", "consumed")
                },
            }
        )
    feedback = []
    for path in sorted((experiment / "tasks").glob("*/paired_feedback/report.json")):
        feedback.append(read(path))
    report = dict(
        scope="real_synthetic_intervention_pilot_not_benchmark",
        frames=frames,
        branches=branches,
        baseline=baseline,
        training=training,
        feedback=feedback,
        bridge=read(experiment / "bridge/verification.json"),
        terrain=read(experiment / "asset-repair/terrain-repair-report.json"),
        render={
            k: data.get(k)
            for k in (
                "source_log",
                "image_count",
                "history_ticks",
                "future_ticks",
                "terrain_fully_supported",
                "missing_appearance_tokens",
            )
        },
        visual_admission="not_asserted_by_gallery",
        promoted=any(f.get("promoted") is True for f in feedback),
    )
    (destination / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    page = """<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ChainFlow 合成增量与配对训练</title>
<style>body{font:16px/1.65 system-ui;background:#edf2f6;color:#183047;max-width:1250px;margin:auto;padding:24px}section{background:white;border:1px solid #d5e0e8;border-radius:12px;padding:20px;margin:16px 0}h1{font-size:28px}.warn{background:#fff1d6;padding:14px;border-left:4px solid #b77912}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}img{width:100%;background:#0c2131}td,th{padding:8px;text-align:left;border-bottom:1px solid #dde5eb}table{border-collapse:collapse;width:100%}a{color:#006a82}button,select{font-size:16px;padding:6px;margin:5px}input{width:45%}pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:360px;overflow:auto}</style>
<h1>4 帧新历史＋10 个真实未来点</h1><p>ChainFlow 官方 stage1 9308 权重 · navtrain 来源 · 实际反事实分支 · 原生模型和 loss</p>
<p class="warn" id="status">读取实际回执…</p><p><a href="report.json">完整结果</a> · <a href="/chainflow-rsi-readiness-v1/index.html">此前相同数据训练验证</a></p>
<p><a href="asset-checks/index.html">完整资产：源图、车辆绑定与新加坡关键帧对照</a></p><section><h2>训练窗口与新视角</h2><p>历史为 0.5–2.0 秒的 4 帧；未来为 2.5–7.0 秒的 10 帧。0 秒是额外上下文。原生 PDM 评分保持 4 秒，实际执行与训练数据覆盖 5 秒未来；二者不能混写。</p><p>左图是原日志照片，不是新轨迹真值；右图沿实际分支渲染。图像能显示不等于通过视觉准入。</p><button id="play">播放</button><button id="reset">回到开始</button><select id="camera"><option>CAM_F0</option><option>CAM_L0</option><option>CAM_R0</option><option>CAM_B0</option></select><input id="time" type="range" min="0" max="14" value="0"><output id="stamp"></output><div class="grid"><div>原日志照片<img id="original"></div><div>实际分支 GS<img id="novel"></div></div></section>
<section><h2>同锚点纠正与保留失败</h2><p>对照是扰动延续，不是 ChainFlow 学生。教师使用 PDM 路线规划与规则交通，本轮不宣称学习型自博弈。</p><div id="branches"></div></section>
<section><h2>固定父模型：8 份独立来源日志</h2><p>选样在评分前固定；日志与本轮训练来源隔离。父模型是否曾预训练这些来源仍未知，此表不是正式 navtest 或泛化复现。</p><div id="baseline"></div></section>
<section><h2>实际训练和收益</h2><div id="training"></div><div id="feedback"></div></section>
<section><h2>数据与视觉证据</h2><div id="evidence"></div></section>
<script>
const esc=x=>String(x??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));let d,timer=null;const time=document.querySelector('#time'),cam=document.querySelector('#camera');
function draw(){const tick=Number(time.value)*5,row=d.frames.find(x=>x.tick===tick&&x.camera===cam.value);for(const k of ['original','novel'])document.getElementById(k).src=row[k];document.querySelector('#stamp').textContent=(tick/10).toFixed(1)+' 秒 · '+(tick===0?'额外上下文':tick<=20?'新历史':'纠正未来')}
function stop(){clearInterval(timer);timer=null;document.querySelector('#play').textContent='播放'}
time.oninput=draw;cam.onchange=draw;document.querySelector('#reset').onclick=()=>{stop();time.value=0;draw()};document.querySelector('#play').onclick=()=>{if(timer){stop();return}document.querySelector('#play').textContent='暂停';timer=setInterval(()=>{if(+time.value>=14){stop();return}time.value=+time.value+1;draw()},500)};
const detail=(label,x)=>'<details><summary>'+esc(label)+'</summary><pre>'+esc(JSON.stringify(x,null,2))+'</pre></details>';
fetch('report.json',{cache:'no-store'}).then(r=>r.json()).then(v=>{d=v;draw();document.querySelector('#status').textContent=d.training.length?'已有训练回执，收益与晋级以独立反馈为准。':'尚未启动合成配对训练：视觉准入未通过。不会将相同数据流程测试记为合成收益。';document.querySelector('#branches').innerHTML='<table><tr><th>任务／分支</th><th>真实未来步数</th><th>反事实 PDMS</th><th>接触状态数</th><th>失败／准入</th></tr>'+d.branches.map(x=>'<tr><td>'+(x.replay?'<a href="'+esc(x.replay)+'">'+esc(x.task+'/'+x.branch)+'</a>':esc(x.task+'/'+x.branch))+'</td><td>'+esc(x.completed_steps)+'</td><td>'+(x.score?(100*x.score.score).toFixed(4):'未完整评分')+'</td><td>'+x.object_contacts+'</td><td>'+esc(x.failure||(x.admitted?'轨迹通过':'轨迹未通过'))+'</td></tr>').join('')+'</table>';document.querySelector('#baseline').innerHTML='<table><tr><th>原始日志</th><th>PDMS</th><th>进度子项</th></tr>'+d.baseline.rows.map(x=>'<tr><td>'+esc(x.source_log)+'</td><td>'+(x.metrics.score*100).toFixed(4)+'</td><td>'+(x.metrics.ego_progress*100).toFixed(4)+'</td></tr>').join('')+'</table>';document.querySelector('#training').innerHTML=d.training.length?d.training.map(x=>detail(x.receipt,x)).join(''):'优化器更新：0；配对学习收益：未评测；晋级和重新采集：未执行。';document.querySelector('#feedback').innerHTML=d.feedback.map(x=>detail('配对反馈',x)).join('');document.querySelector('#evidence').innerHTML=detail('原生14帧、双目标与51步缓存验证',d.bridge)+detail('高程离线修复，拒绝不可靠填充',d.terrain)+detail('渲染记录',d.render)}).catch(e=>document.querySelector('#status').textContent='读取结果失败：'+e.message);
</script></html>"""
    (destination / "index.html").write_text(page)
    export_asset_checks(experiment, destination / "asset-checks")
    return {"branches": len(branches), "images": len(frames), "training_receipts": len(training)}


def export_asset_checks(experiment, destination):
    """Publish actual diagnostic pixels; never convert probe results into admission."""
    import html

    destination.mkdir(parents=True, exist_ok=True)
    items = destination / "evidence"
    items.mkdir(exist_ok=True)

    def link(path, name):
        path = Path(path).resolve(strict=True)
        target = items / name
        if target.is_symlink():
            if target.resolve() != path:
                raise ValueError("Asset check identity changed")
        elif target.exists():
            raise ValueError("Cannot replace asset check evidence")
        else:
            target.symlink_to(path)
        return "evidence/" + name

    sections = []
    for name, title in (
        ("singapore-source-four-cameras-v1", "Singapore：同位姿源图重建"),
        ("singapore-branch-keyframes-v1", "Singapore：同一合格轨迹的锚点与末帧"),
    ):
        path = experiment / "gs" / name / "report.json"
        if not path.exists():
            continue
        report = read(path)
        rows = []
        for index, row in enumerate(report["rows"]):
            image = link(row["image"], name + "-" + str(index) + ".png")
            original = link(row["real"], name + "-" + str(index) + ".jpg") if "real" in row else None
            evidence = row.get("evidence", {})
            nearest = row.get("nearest_same_traversal_source_cameras", [])
            rows.append(dict(
                camera=row["camera"], tick=row.get("tick"), image=image, original=original,
                psnr_db=row.get("psnr_db"), nearest_observation=nearest[:1],
                object_bindings=evidence.get("object_bindings", {}),
                terrain=evidence.get("terrain_evidence", {}),
            ))
        sections.append(dict(title=title, rows=rows, report=link(path, name + ".json"),
                             mean_source_psnr_db=report.get("mean_source_psnr_db"),
                             scope=report.get("scope"), admitted=False))
    extra = []
    for name, title in (
        ("asset-repair/singapore-source-window-ground-screen.json", "固定地面规则下的候选窗口筛查"),
        ("asset-repair/singapore-branch-measured-road-summary.json", "实际分支：地面支持与失败"),
        ("gs/native-binding-repair-ledger.json", "此前车辆缺标后的真实绑定修复"),
        ("gs/singapore-source-window-v1/report.json", "Singapore 完整60张源重建检查：未通过"),
        ("gs/singapore-source-obb-coordinate-audit.json", "平面源框与完整3D源框的真实坐标对照"),
        ("source-membership/implementation.json", "训练来源凭证与实际锚点分离"),
        ("video-generation-selection.json", "视频生成候选调研：尚未集成或推理"),
        ("gs/singapore-obb-fix-v1/report.json", "源物体框坐标修复与原始像素不变核验"),
        ("asset-repair/gs-warm-start-v1/report.json", "有界原生GS资产优化前后验证"),
    ):
        path = experiment / name
        if path.exists():
            extra.append(dict(title=title, href=link(path, name.replace("/", "--"))))
    result = dict(sections=sections, evidence=extra, trainable=False,
                  scope="diagnostic source and keyframes, not complete visual admission")
    (destination / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    blocks = []
    research_path = experiment / "video-generation-selection.json"
    if research_path.exists():
        research = read(research_path)
        rows = []
        for item in research.get("decisions", []):
            urls = item.get("sources", [])
            title = html.escape(item["model"])
            if urls:
                title = "<a href='" + html.escape(urls[0], quote=True) + "'>" + title + "</a>"
            role = html.escape(item.get("role", "待核查"))
            limits = "；".join(item.get("limitations", []))
            rows.append("<tr><td>" + title + "</td><td>" + role + "</td><td>" + html.escape(limits) + "</td></tr>")
        blocks.append("<section><h2>视频生成替代路径：公开能力调研</h2><p>以下均未在本项目运行。轨迹分数不能替代生成图像的车辆位置、相机与时序一致性验证。4帧历史加10帧未来按2Hz采样，首末帧跨度为6.5秒；修改播放帧率不能补足物理时间覆盖。</p><table><thead><tr><th>模型</th><th>可能用途</th><th>尚需解决</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></section>")
    for section in sections:
        cards = []
        for row in section["rows"]:
            label = row["camera"] + (" · tick " + str(row["tick"]) if row["tick"] is not None else "")
            if row["psnr_db"] is not None:
                label += " · 源重建 PSNR " + format(row["psnr_db"], ".2f") + " dB"
            near = row["nearest_observation"]
            if near:
                label += " · 最近真实相机 " + format(near[0]["translation_m"], ".2f") + " m"
            photos = ("<figure>原始实拍<img loading='lazy' src='" + row["original"] + "'></figure>") if row["original"] else ""
            photos += "<figure>实际生成<img loading='lazy' src='" + row["image"] + "'></figure>"
            details = html.escape(json.dumps({k: row[k] for k in ("object_bindings", "terrain")}, ensure_ascii=False, indent=2))
            cards.append("<article><h3>" + html.escape(label) + "</h3><div class='photos'>" + photos + "</div><details><summary>对象与高程依据</summary><pre>" + details + "</pre></details></article>")
        blocks.append("<section><h2>" + html.escape(section["title"]) + "</h2><p><a href='" + section["report"] + "'>原始报告</a></p>" + "".join(cards) + "</section>")
    links = "".join("<li><a href='" + x["href"] + "'>" + html.escape(x["title"]) + "</a></li>" for x in extra)
    page = """<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GS 完整资产与几何核查</title>
<style>body{font:16px/1.7 system-ui;background:#edf2f6;color:#183047;max-width:1250px;margin:auto;padding:24px}section,article{background:white;border:1px solid #d5e0e8;border-radius:12px;padding:16px;margin:16px 0}.photos{display:flex;gap:12px}figure{margin:0;flex:1;min-width:0}img{width:100%}pre{white-space:pre-wrap;max-height:400px;overflow:auto}a{color:#006a82}.warn{background:#fff1d6;padding:16px}</style>
<h1>完整资产仍须核查实际图像与物理绑定</h1><p><a href="../index.html">返回完整训练链路</a> · <a href="report.json">页面数据</a></p>
<p class="warn">本页全部是诊断证据，不授予视觉准入。合成配对训练尚未开始，优化器更新为 0。原始实拍仅用于同位姿重建对照，不能当作新轨迹真值。</p>
<p>Singapore 教师分支的反事实 PDMS 为 95.1801，扰动延续对照为 92.6474。两者均保留完整执行与恢复证据；这是 PDM 路线教师与规则交通，不是学习型自博弈，也未证明 ChainFlow 的能力短板。</p>
<p>末帧距同一时刻源位姿约 9.1m，但附近存在其他时刻的真实相机。按实际空间覆盖核查，不能仅按同时间位移判断越界。关键帧尚未施加合格地面修正，仍需完整时序视觉验收。</p>
""" + "".join(blocks) + "<section><h2>保留的检查与失败依据</h2><ul>" + links + "</ul></section></html>"
    (destination / "index.html").write_text(page)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("root", "experiment", "destination", "render"):
        parser.add_argument("--" + key, type=Path, required=True)
    print(json.dumps(export(**vars(parser.parse_args()))))
