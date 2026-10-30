"""Publish bounded training/GS evidence without test-scene identities."""
import json
import shutil
from pathlib import Path

ROOT = Path('/data/xinyao/navsim_workspace/DriveRSI')
EXP = ROOT / 'exp/gs-multiview-training-v2'
OUT = ROOT / 'outputs/demonstrations/gs-multiview-training-v2'


def publish():
    OUT.mkdir(parents=True, exist_ok=True)
    cases = []
    for name, folder, status in (
        ('failed-window', 'inspection-diagnostic-v2', '已拒绝：后半段高程证据冲突，不能训练'),
        ('training-window', '../portable-july16/inspection-v3', '新场景四相机窗口：以独立准入回执为准'),
    ):
        src = EXP / 'render' / folder
        if not (src / 'viewer.html').exists():
            continue
        dest = OUT / name
        dest.mkdir(exist_ok=True)
        for path in src.iterdir():
            if path.name == 'viewer.html' or path.name.endswith(('-gs.png', '-boxes.jpg')):
                shutil.copy2(path, dest / path.name)
        # Exact evidence remains in exp; public overview contains no test tokens.
        (dest / 'report.json').write_text(json.dumps({
            'scope': status, 'detailed_evidence': '保存在32341的exp目录',
            'test_scene_details': 'not_published',
        }, ensure_ascii=False))
        acceptance = src / 'acceptance.json'
        if acceptance.exists():
            verdict = json.loads(acceptance.read_text()).get('admitted') is True
            (dest / 'acceptance.json').write_text(json.dumps({
                'admitted': verdict,
                'scope': '仅适用本次固定四相机窗口与24dB试验组；逐样本证据仅在服务器exp中保存',
            }, ensure_ascii=False))
        else:
            (dest / 'acceptance.json').write_text(json.dumps({
                'admitted': False, 'scope': status,
            }, ensure_ascii=False))
        cases.append((name, status))
    for path in (EXP / 'video-causality-v1').glob('*.png'):
        shutil.copy2(path, OUT / path.name)
    decision = json.loads((EXP / 'renderer-decision.json').read_text())
    (OUT / 'generation-probe.json').write_text(json.dumps({
        'scope': '非真实场景的输入敏感性诊断，仅用于判断历史是否依赖未来；不是训练样本',
        'per_history_tick': decision['per_history_tick'],
        'selected_renderer': decision['selected_renderer'],
    }, ensure_ascii=False, indent=2))
    # Only synthetic GS pixels are published; private real-source crops stay in exp.
    hi = EXP / 'portable-july16/cone-native-resolution-v1'
    hirows = []
    for tick in (0, 20, 30, 40):
        hp = hi / f'{tick:04d}-CAM_L0-novel-1920.png'
        lp = EXP / 'portable-july16/tasks/portable-july16-gs-v3/render/render' / f'{tick:04d}-CAM_L0.png'
        if not hp.exists() or not lp.exists():
            continue
        hname, lname = f'cone-{tick:04d}-1920.png', f'cone-{tick:04d}-960.png'
        shutil.copy2(hp, OUT / hname)
        shutil.copy2(lp, OUT / lname)
        hirows.append(f'<h2>{tick / 10:.1f}秒 · CAM_L0</h2><div class="pair"><figure><img src="{lname}"><figcaption>960×540 原始GS</figcaption></figure><figure><img src="{hname}"><figcaption>同资产重新光栅化1920×1080；非放大/修图</figcaption></figure></div>')
    if hirows:
        (OUT / 'resolution-check.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>小物体原生分辨率复核</title><style>body{font:16px/1.6 system-ui;background:#eef3f6;max-width:1400px;margin:auto;padding:24px}.pair{display:grid;grid-template-columns:1fr 1fr}img{width:100%}figure{margin:8px}</style><h1>分辨率复核：仍未取得视觉准入</h1><p>同一物理状态、同一GS资产及相机畸变，内参随分辨率成比例缩放。路锥轮廓仍无法独立追踪；没有删除物体，也没有用高清图冒充合格。</p><p>此处只显示合成GS；源照片与私有裁图留在服务器。</p>' + ''.join(hirows) + '</html>')
    refinement = EXP / 'portable-july16/cone-refinement-v1'
    refinement_report = refinement / 'report.json'
    if refinement_report.exists():
        rr = json.loads(refinement_report.read_text())
        cards = []
        for before, after in zip(rr['before_holdout'], rr['after_holdout'], strict=True):
            assert (before['raw_index'], before['camera']) == (after['raw_index'], after['camera'])
            names = []
            for label, row in (('before', before), ('after', after)):
                source = Path(row['image']).resolve()
                if source.parent != refinement.resolve() or source.name.endswith('-real-private.png'):
                    raise ValueError('Only synthetic local GS crops may be published')
                name = f"refine-{row['raw_index']:03d}-{row['camera']}-{label}.png"
                shutil.copy2(source, OUT / name)
                names.append(name)
            cards.append(f'<h2>来源帧 {before["raw_index"]} · {before["camera"]}</h2><div class="pair"><figure><img src="{names[0]}"><figcaption>原GS合成裁图</figcaption></figure><figure><img src="{names[1]}"><figcaption>200步局部外观优化后合成裁图</figcaption></figure></div>')
        (OUT / 'asset-refinement.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>GS局部资产修复对照</title><style>body{font:16px/1.6 system-ui;background:#eef3f6;max-width:1100px;margin:auto;padding:24px}.pair{display:grid;grid-template-columns:1fr 1fr}img{width:100%;image-rendering:auto}figure{margin:8px}</style><h1>实际200步局部GS优化：仍未取得视觉准入</h1><p>只更新物体原始三维框内6个既有Gaussian的颜色和透明度。全部几何、其他参数及物理分支不变。这不是ChainFlow训练。</p><p>预先留出5个时刻的全部相机，其中4个时刻有目标投影，共6组检查。平均局部PSNR变化仅+0.00271dB，目标轮廓与遮挡仍不确定。留出仅针对本次局部优化，不代表原资产从未见过这些图像。</p><p>实测25个LiDAR点位于目标上部及车／植被边界，不能独立确认目标身份，其中6点来自本次留出时刻。没有据此填框造点。这里只公开合成图；原照片留在私有审计目录。</p>' + ''.join(cards) + '</html>')
    summary = EXP / 'public-summary.json'
    data = json.loads(summary.read_text()) if summary.exists() else {
        'status': '正在筛选另一条有完整高程支撑的高分分支；尚未启动训练',
        'training': None,
    }
    # Public training summary is a fixed allow-list of scalar aggregate values.
    allowed = ('updates_per_arm', 'replay_updates', 'test_count', 'test_log_count',
               'parent_score', 'control_score', 'treatment_score', 'delta_control',
               'delta_parent', 'resume_exact', 'synthetic_exposures', 'promoted')
    training = data.get('training')
    if training is not None:
        data['training'] = {k: training[k] for k in allowed if k in training}
        if any(type(v) not in (int, float, bool) for v in data['training'].values()):
            raise ValueError('Public metrics must be numeric aggregates only')
    statuses = {
        'pending': '本轮完整窗口未通过准入；配对训练适配已准备，未开始训练',
        'running': '合格分支已进入配对训练，结果尚未完成',
        'completed': '配对训练与固定navtest子集评测已完成；结果只用于小规模验证',
        'rejected': '本次分支未通过准入，已保留失败，没有训练',
    }
    data = {'status': statuses[data.get('state', 'pending')], 'training': data.get('training')}
    (OUT / 'summary.json').write_text(json.dumps(data, ensure_ascii=False, indent=2))
    links = ''.join(f'<li><a href="{n}/viewer.html">{s}</a></li>' for n, s in cases)
    rows = ''.join(
        f'<tr><td>{r["tick"]/10:.1f}s</td><td>{r["repeat_mae_255"]:.4f}</td>'
        f'<td>{r["future_change_mae_255"]:.4f}</td></tr>'
        for r in decision['per_history_tick']
    )
    html = '''<!doctype html><html lang="zh"><meta charset="utf-8">
<title>DriveRSI · 四相机 GS 与配对训练</title><style>
body{margin:0;background:#eef3f6;color:#183443;font:16px/1.7 system-ui}main{max-width:1120px;margin:auto;padding:28px}section{background:white;border-radius:14px;padding:24px;margin:22px 0}h1{font-size:30px}a{color:#007d84}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #dde4e9;text-align:left}.warn{border-left:5px solid #d5972b;padding:14px;background:#fff6df}.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}img{width:100%}small{color:#557080}pre{white-space:pre-wrap}</style>
<main><h1>四相机 GS → 数据准入 → ChainFlow stage1 配对实验</h1>
<p>同一物理分支保留 4 帧新历史＋10 帧真实未来。完整失败保留，准入不由画面观感代替。</p>
<section><h2>当前实际结果</h2><p id="status"></p><div id="results"></div>
<p>训练原生相机顺序 F0、B0、L0、R0。四帧历史全部归档，原生 stage1 使用当前时刻四相机；未来图片不进入当前策略。</p>
<p>评测预先固定32个navtest场景，来自不同原始日志；父9405基线已完成32/32，94.0358分，零运行异常。配对训练结果以本页正式回执为准。这不是全量12,146场景复现，页面不发布测试token或逐场景明细。</p></section>
<section><h2>新场景的物理与静态配准</h2><p>从现有便携GS资产重新筛出的NAVSIM训练来源中，实际原生DriveRL到达＋PDM路线教师分支达到99.8366分（原生4秒评分），归档包含真实5秒未来。15个采样时刻已完成资产高程图支持检查（非完整三维坡度证明），60张新视角＋60张源位姿对照已完成。两辆停车和电气箱的固定三维绑定及遮挡检查通过；路锥在960×540输出中无法独立确认，8张1920×1080原生重渲染也未解决。另外实际完成200步局部GS颜色／透明度优化，几何完全不变，但目标轮廓仍未恢复。正式Harness导出已返回视觉拒绝，配对训练编译器同样拒绝该回执；本分支没有进入训练。</p></section><section><h2>四相机同步回放</h2><ul>LINKS</ul><p><a href="resolution-check.html">原生1920×1080与960×540对照（仍未通过）</a></p>
<p class="warn">原窗口的高程歧义仍未消除。此前47/71是旧版检查的历史结果；新增残差检查发现，少量但可能组成第二表面的测量不能当作离群点丢弃。三条旧合成分支按最新证明重新检查，各0/15图像时刻通过；旧资格不能复用。更换场景不代表修好了原高程。歧义也不等于已证实存在两条道路。</p>
<p>沿用先前授权的24dB源位姿重建试验组；不是27dB新视角质量证明。本分支源位姿平均26.4207dB，最低帧22.2632dB；这是均值24dB试验组，不能表述为每帧都超过24dB。侧视画质单独报告。</p></section>
<section><h2>实际局部资产优化</h2><p>原生GS RGB损失项完成200次更新，用时12.70秒，峰值显存0.90GB。固定留出视角的平均局部PSNR仅变化+0.00271dB，视觉审核仍拒绝。已独立核对模型参数和训练暴露，更新真实发生，未使用留出帧；没有执行ChainFlow优化更新。</p><p><a href="asset-refinement.html">局部优化前后对照（合成GS裁图）</a></p></section><section><h2>替代资产与补建检查</h2><p>另一份navtrain日志中找到41个完整7秒道路支撑窗口，但同一个GS资产缺少画面中可见的2名行人；原生训练对这类对象使用了排除掩码。18辆汽车有对应资产，并不能替代行人覆盖检查。</p><p>已通过原始数据库核对48帧行人标注所需LiDAR：缺少对应文件。另有168份同日志扫描，但与两名行人的有效标注时段均不匹配，不能用于补建。此失败记录保留；没有删除行人或替换身份。</p><p>原始人物图片留在私有实验审计目录；此处只展示计数与缺口。没有从该替代窗口启动重建或训练。</p></section><section><h2>为何本轮退回原始 GS</h2><p>同输入、同种子重复完全一致；只改未来条件，生成的历史仍发生变化。本次原生 StreetCrafter 配方不具备训练历史所需的因果隔离，也尚未证明联合多视角一致性。</p>
<table><thead><tr><th>历史时刻</th><th>重复误差 MAE</th><th>仅改未来后的 MAE</th></tr></thead><tbody>ROWS</tbody></table>
<small>像素范围0–255；干预把未来条件置黑，仅作敏感性探测，不是有效物理场景，不用于训练。</small>
<div class="pair"><figure><img src="baseline-0020.png"><figcaption>原条件生成的2.0秒历史</figcaption></figure><figure><img src="future_changed-0020.png"><figcaption>历史条件不变，只改变后续条件</figcaption></figure></div></section>
<p>代码、初始权重与原始数据在32341；新训练和检查点均在 exp/gs-multiview-training-v2/。原生模型、loss、评分保持不变。</p></main>
<script>fetch('summary.json').then(r=>r.json()).then(d=>{document.getElementById('status').textContent=d.status;let e=document.getElementById('results');if(d.training){const p=document.createElement('pre');p.textContent=JSON.stringify(d.training,null,2);e.appendChild(p);}});</script></html>'''
    (OUT / 'index.html').write_text(html.replace('LINKS', links).replace('ROWS', rows))
    print(OUT / 'index.html')


if __name__ == '__main__':
    publish()
