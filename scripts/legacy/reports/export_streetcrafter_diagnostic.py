"""Export actual bounded StreetCrafter outputs for physical/visual inspection."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def publish(reports, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'images').mkdir()
    runs = []
    for number, report in enumerate(reports):
        value = json.loads(Path(report).read_text())
        if value.get('status') != 'generated_diagnostic_partial' or value.get('trainable'):
            raise ValueError('Completed diagnostic-only real output required')
        if value['generated_ticks'] != list(range(25)):
            raise ValueError('All 25 actual generated frames required')
        inputs = {x['tick']: x for x in value['guide_inputs']}
        gs = {x['tick']: x for x in value.get('gs_inputs', [])}
        rows = []

        def copy(path, sha, label):
            if digest(path) != sha:
                raise ValueError('Artifact changed: ' + str(path))
            name = f'images/run{number}-{label}{Path(path).suffix}'
            shutil.copyfile(path, output / name)
            return name

        for row in value['frames']:
            tick = row['tick']
            item = dict(tick=tick, timestamp_us=row['timestamp_us'], paths={})
            item['paths']['generated'] = copy(row['path'], row['sha256'], f'{tick:04d}-generated')
            guide = inputs[tick]
            item['paths']['guide'] = copy(guide['path'], guide['sha256'], f'{tick:04d}-guide')
            if tick in gs:
                source = gs[tick]['gs']
                item['paths']['gs'] = copy(source['path'], source['sha256'], f'{tick:04d}-gs')
            rows.append(item)
        ref = value['initial_reference']
        reference = copy(ref['native_copy'], ref['native_copy_sha256'], 'initial-reference')
        runs.append(dict(
            mode='GS 辅助去噪' if value.get('gs_refinement_performed') else '彩色点云条件生成',
            camera=value['selected_camera'], frames=rows, reference=reference,
            recipe=value['native_recipe'], stage_seconds=value['stage_seconds'],
            branch_identity=value['branch_identity'],
            weight_sha256=value['checkpoint_sha256'],
            limitations=value['limitations'],
            trainable=False, visual_admitted=False,
        ))
    payload = dict(runs=runs, status='diagnostic_only', quality_passed=False,
                   note='This page is not evidence of visual admission or model improvement.')
    (output / 'manifest.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    data = json.dumps(payload, ensure_ascii=False).replace('</', '<\\/')
    (output / 'index.html').write_text(HTML.replace('__DATA__', data), encoding='utf-8')
    return dict(output=str(output), runs=len(runs), generated_frames=sum(len(x['frames']) for x in runs))


HTML = '''<!doctype html><html lang="zh"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>StreetCrafter · 结构化条件辅助 GS</title>
<style>body{font:16px system-ui;background:#eef3f5;color:#17303d;max-width:1450px;margin:28px auto;padding:0 24px}h1{font-size:28px}p{line-height:1.7}.notice{background:#fff0d3;border-left:5px solid #bd7a12;padding:14px 20px;border-radius:8px}.controls,section{background:white;border-radius:10px;padding:16px;margin:16px 0}button,select,input{font:inherit;margin:6px;padding:8px}button{cursor:pointer}#seek{width:55%}#grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.tile{background:#122b39;border-radius:8px;color:white;overflow:hidden}.tile img{display:block;width:100%;aspect-ratio:16/9}.tile b{display:block;padding:10px}table{border-collapse:collapse;width:100%}td{padding:8px;border-bottom:1px solid #ddd}code{word-break:break-all}#reference{max-width:640px;width:100%}@media(max-width:900px){#grid{grid-template-columns:1fr}}</style>
<h1>结构化条件辅助 3DGS：StreetCrafter 实际输出</h1>
<p>同一 NAVSIM 训练来源物理分支 · 单相机 25 个连续时刻 · 10 Hz · 1024×576。图像生成不改变车辆状态、物理接触或轨迹分数。</p>
<div class="notice"><b>这是已执行的小规模生成对照，尚未取得视觉训练资格。</b><br>新视角没有对应实拍真值；画面更清晰也不能证明车辆位置、遮挡和道路几何正确。单相机不能证明多视角一致，2.4 秒片段也不等于完整“4 帧历史＋10 帧未来”训练窗口。</div>
<div class="controls"><select id="run"></select><button id="play">播放</button><button id="reset">回到开始</button><br><input id="seek" type="range" min="0" max="24" value="0"><b id="stamp"></b></div>
<div id="grid"></div><section><table id="metrics"></table></section>
<details><summary>原生采样的首帧条件</summary><p>点云生成使用初始真实图像；GS 辅助去噪使用同标定的 GS 首帧。它参与了生成，不能作为独立质量验证。</p><img id="reference"></details>
<section><b>仍需验证</b><p>动态对象数量、位置和遮挡；跨相机同一时刻的一致性；连续帧漂移；历史像素是否受未来条件影响。上游掩码混合约束未启用，不能把“提供了物理条件”当作“输出必然遵守物理”。所有结果保留为 diagnostic，未进入训练。</p><a href="manifest.json">可复核图像清单</a></section>
<script>const data=__DATA__;let ri=0,fi=0,timer=null;const select=document.querySelector('#run'),seek=document.querySelector('#seek');data.runs.forEach((r,i)=>{const o=document.createElement('option');o.value=i;o.textContent=r.mode+' · '+r.camera;select.append(o)});const labels={gs:'输入：实际 GS 渲染',guide:'条件：同相机彩色点云',generated:'输出：StreetCrafter'};function draw(){const r=data.runs[ri],f=r.frames[fi];seek.value=fi;document.querySelector('#stamp').textContent=`${fi+1}/25 · t=${(f.tick/10).toFixed(1)} s`;const g=document.querySelector('#grid');g.innerHTML='';for(const key of ['gs','guide','generated']){if(!f.paths[key])continue;const d=document.createElement('div');d.className='tile';const b=document.createElement('b');b.textContent=labels[key];const im=new Image();im.src=f.paths[key];d.append(b,im);g.append(d)}document.querySelector('#reference').src=r.reference;const table=document.querySelector('#metrics');table.innerHTML='';for(const [k,v] of [['模式',r.mode],['采样',JSON.stringify(r.recipe)],['实际分项耗时（秒）',JSON.stringify(r.stage_seconds)],['物理分支',r.branch_identity],['权重 SHA-256',r.weight_sha256],['视觉准入／训练', '未准入；没有启动训练']]){const tr=document.createElement('tr');for(const t of [k,v]){const td=document.createElement('td');td.textContent=t;tr.append(td)}table.append(tr)}}function stop(){clearInterval(timer);timer=null;document.querySelector('#play').textContent='播放'}document.querySelector('#play').onclick=()=>{if(timer)return stop();document.querySelector('#play').textContent='暂停';timer=setInterval(()=>{if(fi===24)return stop();fi++;draw()},100)};document.querySelector('#reset').onclick=()=>{stop();fi=0;draw()};select.onchange=()=>{stop();ri=Number(select.value);fi=0;draw()};seek.oninput=()=>{stop();fi=Number(seek.value);draw()};draw();</script></html>'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reports', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(publish(args.reports, args.output), ensure_ascii=False))
