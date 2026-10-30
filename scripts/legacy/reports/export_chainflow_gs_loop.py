"""Export the small ChainFlow learning/GS experiment without inferring success."""

import argparse
import json
import os
import pickle
from pathlib import Path


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def export(root, experiment, destination):
    root, experiment, destination = map(
        lambda p: Path(p).resolve(), (root, experiment, destination)
    )
    if not experiment.is_relative_to(root / "exp") or not destination.is_relative_to(
        root / "outputs/demonstrations"
    ):
        raise ValueError("Experiment and gallery must stay within DriveRSI")
    destination.mkdir(parents=True, exist_ok=True)
    runs, feedback, failures = [], [], []
    receipt_names = {"training-result.json", "feedback.json", "paired-feedback.json"}
    for path in sorted(experiment.rglob("*.json")):
        if path.name not in receipt_names and path.name != "report.json":
            continue
        data = read(path)
        if path.name == "report.json" and (
            not isinstance(data, dict) or data.get("kind") != "chainflow_paired_feedback"
        ):
            continue
        if not isinstance(data, dict):
            failures.append(
                {"path": str(path.relative_to(experiment)), "reason": "unreadable_receipt"}
            )
            continue
        identity = str(path.relative_to(experiment))
        if path.name == "training-result.json":
            keep = (
                "status",
                "reason",
                "phase",
                "model",
                "updates",
                "total_updates",
                "worker_pid",
                "metrics",
                "losses",
                "source_groups",
                "consumed",
                "evaluation_scope",
                "promoted",
                "policy_activated",
                "evaluated_sample_ids",
                "paired_samples",
                "state_fingerprint",
                "sample_trace",
                "native_recipe",
                "optimizer_schedule",
            )
            runs.append({"receipt": identity, **{k: data[k] for k in keep if k in data}})
        else:
            keep = (
                "kind",
                "status",
                "reason",
                "model",
                "pilot_metric_improved",
                "observed_benefit",
                "promoted",
                "cross_process_resume",
                "control",
                "treatment",
                "deltas",
                "paired_deltas",
                "metrics",
                "weakness_review",
                "scope",
                "observed_pilot_benefit",
                "synthetic_gain_demonstrated",
                "treatment_minus_continued_training",
                "added_sample_ids",
                "training_cost",
                "replay_cost",
                "evaluation_scope",
                "exposed_added_sample_ids",
                "sample_exposures",
                "unique_training_cost",
                "cost_scope",
                "regressions_against_control",
            )
            feedback.append({"receipt": identity, **{k: data[k] for k in keep if k in data}})
    attempts = []
    for path in sorted((experiment / "tasks").glob("*/*.receipt.json")):
        data = read(path)
        if isinstance(data, dict):
            attempts.append({"receipt": str(path.relative_to(experiment)), **{
                k: data[k] for k in ("task_id", "stage_id", "status", "reason", "error_type", "phase", "failed_phase", "consumed", "outcome", "started_at", "ended_at") if k in data
            }})
    weakness = read(experiment / "weakness/analysis-summary.json")
    # Pick the latest completed rerender; failed attempts remain separately visible.
    renders = []
    for path in sorted(experiment.rglob("report.json")):
        data = read(path)
        if not isinstance(data, dict) or data.get("kind") != "worldengine_physical_branch_render":
            continue
        renders.append(
            {
                "receipt": str(path.relative_to(experiment)),
                **{
                    k: data[k]
                    for k in (
                        "status",
                        "reason",
                        "image_count",
                        "quality_admitted",
                        "trainable",
                        "missing_appearance_tokens",
                        "limitations",
                    )
                    if k in data
                },
            }
        )
    report = {
        "schema_version": 1,
        "scope": "small_real_data_training_workflow_and_separate_synthetic_visual_validation",
        "runs": runs,
        "feedback": feedback,
        "renders": renders,
        "weakness": weakness,
        "read_failures": failures,
        "execution_attempts": attempts,
        "synthetic_benefit_claimed": False,
        "automatic_promotion": False,
        "asset_height": read(experiment / "asset-height-ledger.json"),
        "feedback_analysis": read(experiment / "learning-feedback-analysis-v2.json"),
        "source_exposure": read(experiment / "source-exposure-audit.json"),
        "training_validation": read(experiment / "training-implementation-ledger-v3.json"),
    }
    (destination / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    template = r"""<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ChainFlow 实际训练与 GS 验证</title>
<style>body{font:16px/1.65 system-ui,sans-serif;background:#f4f7fb;color:#183047;max-width:1250px;margin:auto;padding:28px}h1{font-size:28px}section{background:white;border:1px solid #dce5ed;border-radius:12px;padding:22px;margin:18px 0}a{color:#006a82}table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:8px;border-bottom:1px solid #dde4eb}pre{overflow:auto;max-height:420px;background:#edf3f7;padding:12px;white-space:pre-wrap;overflow-wrap:anywhere}.note{background:#fff2d8;padding:12px;border-left:4px solid #bc7b00}.muted{color:#587086}.status{font-weight:700}</style>
<h1>ChainFlow stage1：实际训练、恢复、评测与反馈</h1>
<p class="note">先用少量真实 navtrain 数据验证训练流程。GS 分支只有通过轨迹与独立视觉准入后，才能作为合成增强；训练完成、平均得分上升都不自动代表 RSI 收益或模型晋级。</p>
<p><a href="report.json">机器可读结果</a> · <a href="gs-comparison.html">高程修复连续对照</a> · <a href="../worldengine-rsi-v1/pittsburgh-viewer.html">已有 GS 同位姿与新轨迹对照</a> · <a href="../worldengine-rsi-v1/pittsburgh-teacher.html">98.28 分教师物理分支</a></p>
<section><h2>本次验证结果</h2><div id="summary"></div></section>
<section><h2>真实训练执行</h2><div id="runs">读取回执…</div></section>
<section><h2>跨进程恢复与收益反馈</h2><div id="feedback"></div></section>
<section><h2>难例诊断与下一步</h2><p>同一来源锚点比较学生、真实人类轨迹、PDM 参考和时序干预。区别原状态补标签、交互世界修复与生成新历史。</p><div id="weakness"></div></section>
<section><h2>执行记录与保留失败</h2><div id="attempts"></div></section>
<section><h2>高斯物理分支</h2><p>高程改正、静态背景外观和新视角画质分别检查。静态路障没有独立节点不等于没有外观；背景点数量也不能证明路障形状正确。</p><div id="renders"></div></section>
<script>
const esc=x=>String(x??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const detail=(x)=>'<details><summary>'+esc(x.receipt||'查看证据')+'</summary><pre>'+esc(JSON.stringify(x,null,2))+'</pre></details>';
fetch('report.json',{cache:'no-store'}).then(r=>{if(!r.ok)throw Error(r.status);return r.json()}).then(d=>{
 const v=d.training_validation; document.querySelector('#summary').innerHTML=v?'<p><strong>统一训练到反馈已执行；合成增强闭环尚未完成。</strong></p><table><tr><th>最终学生更新</th><td>'+esc(v.execution.final_model_optimizer_updates)+' 次；另含一次独立恢复重放</td></tr><tr><th>开发数据</th><td>'+esc(v.source_scope.development_original_logs)+' 份与本轮训练隔离的原始日志；不是正式测试集成绩</td></tr><tr><th>原生 PDMS</th><td>'+(v.metrics.parent.score*100).toFixed(6)+' → '+(v.metrics.candidate.score*100).toFixed(6)+'，未认定收益或晋级</td></tr><tr><th>配对验证</th><td>两臂相同真实数据，全部指标一致；不代表合成数据收益</td></tr><tr><th>GS 数据准入</th><td>未通过，未进入训练</td></tr></table>':'尚无最终验收回执';
 document.querySelector('#runs').innerHTML=d.runs.length?'<table><thead><tr><th>阶段</th><th>状态</th><th>本阶段更新</th><th>耗时 / s</th><th>证据</th></tr></thead><tbody>'+d.runs.map(r=>'<tr><td>'+esc(r.phase)+'</td><td class="status">'+esc(r.status)+'</td><td>'+esc(r.updates)+'</td><td>'+esc(r.consumed?.wall_seconds?.toFixed(1))+'</td><td>'+detail(r)+'</td></tr>').join('')+'</tbody></table>':'尚无完成或失败的训练回执；不能把准备工作计作优化器更新。';
 document.querySelector('#feedback').innerHTML=d.feedback.length?d.feedback.map(detail).join('')+(d.feedback_analysis?detail({receipt:'训练后分析：保留原始差异，区分数值尺度',...d.feedback_analysis}):''):'反馈阶段尚未发布结果。';
 document.querySelector('#weakness').innerHTML=d.weakness?'<pre>'+esc(JSON.stringify(d.weakness,null,2))+'</pre>':'同锚点分析正在核对，尚未发布决策回执。';
 document.querySelector('#attempts').innerHTML=(d.execution_attempts||[]).map(detail).join('');
 document.querySelector('#renders').innerHTML=d.renders.length?d.renders.map(detail).join('')+(d.asset_height?detail({receipt:'地形支持及路障绑定',...d.asset_height}):''):'新高程渲染尚未发布回执；已有失败和历史图像见上方对照。';
}).catch(e=>document.querySelector('#runs').textContent='无法读取结果：'+e.message);
</script></html>"""
    (destination / "index.html").write_text(template)
    return report


def export_images(root, experiment, destination, render_path):
    from urllib.parse import quote

    root, experiment, destination, render_path = map(
        lambda p: Path(p).resolve(), (root, experiment, destination, render_path)
    )
    if not render_path.is_relative_to(experiment):
        raise ValueError("Only this experiment's physical render can be published")
    report = read(render_path / "report.json")
    if report.get("status") != "rendered_diagnostic" or report.get("image_count") != 52:
        raise ValueError("Complete fixed-window render required")
    source = root / "exp/worldengine-rsi-v1/pittsburgh-matched-source-v2"
    before = root / "exp/worldengine-rsi-v1/tasks/worldengine-sealed-render-v2/render/render"
    prior = read(before / "report.json")
    if report["source_log"] != prior["source_log"]:
        raise ValueError("Before/after source differs")
    current_hashes = {Path(p).name: sha for p, sha in report["inputs"].items()}
    old_hashes = {Path(p).name: sha for p, sha in prior["inputs"].items()}
    if any(
        current_hashes.get(k) != old_hashes.get(k)
        for k in ("scene.npz", "teacher.npz", "video_scene_dict.pkl")
    ):
        raise ValueError("Before/after physical branch differs")
    log_path = next(
        Path(p) for p in report["inputs"] if Path(p).name == report["source_log"] + ".pkl"
    )
    with log_path.open("rb") as stream:
        raw = {f["token"]: f for f in pickle.load(stream)}
    image_dir = destination / "images"
    image_dir.mkdir(exist_ok=True)

    def link(path, name):
        target = image_dir / name
        if target.is_symlink():
            if target.resolve() != path.resolve():
                target.unlink()
        elif target.exists():
            raise ValueError("Refuse replacing existing gallery image")
        if not target.exists():
            os.symlink(path.resolve(), target)
        return "images/" + quote(name)

    items = []
    for row in report["frames"]:
        tick = row["tick"]
        for cam in ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0"):
            name = row["cameras"][cam]["file"]
            source_path = (
                Path("/data/xinyao/navsim_workspace/dataset/sensor_blobs/trainval_all")
                / raw[row["source_token"]]["cams"][cam]["data_path"]
            )
            if not all(
                p.is_file() for p in (source_path, render_path / name, before / name, source / name)
            ):
                raise ValueError("Missing before/after or source frame")
            items.append(
                {
                    "tick": tick,
                    "camera": cam,
                    "source_token": row["source_token"],
                    "source": link(source_path, "source-" + name + ".jpg"),
                    "matched": link(source / name, "matched-" + name),
                    "before": link(before / name, "before-" + name),
                    "after": link(render_path / name, "after-" + name),
                }
            )
    (destination / "images.json").write_text(json.dumps(items, indent=2))
    page = r"""<!doctype html><html lang="zh"><meta charset="utf-8"><title>GS 高程修复连续对照</title><style>body{font:16px/1.6 system-ui;background:#132634;color:#e7f1f6;margin:20px}a{color:#87d1e3}.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}img{width:100%;background:black}label,button,select{font-size:16px;margin:8px}input{width:50%}.warn{color:#ffce7f}</style><h1>同一物理分支：高程修复前后</h1><p class="warn">实拍与原位姿 GS 用于检查重建；新轨迹没有实拍真值。下方两张新轨迹图只改变高程处理，不能把差异自动当作画质改善。</p><p><a href="index.html">返回训练与反馈</a></p><button id="play">播放</button><button id="reset">回到开始</button><label>相机<select id="camera"><option>CAM_F0</option><option>CAM_L0</option><option>CAM_R0</option><option>CAM_B0</option></select></label><input type="range" id="time" min="0" max="12" step="1" value="0"><output id="stamp"></output><div class="grid"><div>原日志实拍<img id="source"></div><div>原位姿 GS（未做新轨迹扰动）<img id="matched"></div><div>新轨迹 GS · 高程修复前<img id="before"></div><div>同一新轨迹 GS · 地形高程<img id="after"></div></div><script>
let rows=[],timer=null;const time=document.querySelector('#time'),camera=document.querySelector('#camera');function draw(){const tick=Number(time.value)*5,row=rows.find(x=>x.tick===tick&&x.camera===camera.value);if(!row)return;for(const key of ['source','matched','before','after'])document.getElementById(key).src=row[key];document.querySelector('#stamp').textContent=(tick/10).toFixed(1)+'秒 · '+(tick<=20?'到达历史':'纠正未来')+' · '+row.source_token;}time.oninput=draw;camera.onchange=draw;document.querySelector('#reset').onclick=()=>{clearInterval(timer);timer=null;time.value=0;document.querySelector('#play').textContent='播放';draw()};document.querySelector('#play').onclick=()=>{if(timer){clearInterval(timer);timer=null;document.querySelector('#play').textContent='播放';return}document.querySelector('#play').textContent='暂停';timer=setInterval(()=>{if(Number(time.value)>=12){clearInterval(timer);timer=null;document.querySelector('#play').textContent='播放';return}time.value=Number(time.value)+1;draw()},500)};fetch('images.json').then(r=>r.json()).then(x=>{rows=x;draw()});</script></html>"""
    (destination / "gs-comparison.html").write_text(page)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--render", type=Path)
    args = parser.parse_args()
    report = export(args.root, args.experiment, args.destination)
    if args.render:
        export_images(args.root, args.experiment, args.destination, args.render)
    print(
        json.dumps(
            {
                "runs": len(report["runs"]),
                "feedback": len(report["feedback"]),
                "renders": len(report["renders"]),
            }
        )
    )
