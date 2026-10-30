"""Server-hosted, self-contained evidence report for the engineering PPO pilot."""
import argparse
import html
import json
from pathlib import Path


def write_report(root: Path, render_directory="render-v2"):
    if Path(render_directory).name != render_directory:
        raise ValueError("Render directory must be a direct child")
    report = json.loads((root / "run/report.json").read_text())
    render = json.loads((root / render_directory / "verification.json").read_text())
    metrics = [json.loads(line) for line in (root / "run/metrics.jsonl").read_text().splitlines()]
    rows = []
    for name, data in [("保持当前控制 coast", report["baseline"]["coast"]),
                       ("初始网络", report["baseline"]["initial"]), ("PPO候选", report["evaluation"])]:
        rows.append(f"<tr><td>{name}</td><td>{data['actor_episodes']}</td>" + "".join(
            f"<td>{100 * data[k]:.1f}%</td>" for k in
            ("goal_per_actor", "collision_per_actor", "offroad_per_actor")) + "</tr>")
    payload = json.dumps(metrics, allow_nan=False).replace("<", "\u003c")
    overview = {k: report[k] for k in ("valid_transitions", "updates", "seconds_including_eval_recording")}
    page = """<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>DriveRSI · 学习型 Self-Play 试点</title><style>
body{background:#0d1823;color:#dce9f2;font:16px system-ui;max-width:1120px;margin:30px auto;padding:0 24px;line-height:1.65}
h1,h2{color:#fff}a{color:#7bcaff}section{background:#162735;border:1px solid #2e495c;border-radius:12px;padding:20px;margin:20px 0}
table{border-collapse:collapse;width:100%}td,th{text-align:left;border-bottom:1px solid #365366;padding:10px}.warning{color:#ffd48c}
canvas{width:100%;height:260px}select{background:#243c50;color:white;padding:8px}pre{white-space:pre-wrap;font-size:13px}nav{display:flex;gap:25px;flex-wrap:wrap}
</style><h1>DriveRSI · 学习型 Self-Play 试点</h1>
<p>同一共享策略控制多辆车 → 联合推进 → 有效样本 → PPO更新 → 冻结候选 → 写实渲染。</p>
<section><h2>当前资格：教师候选，未批准为专家监督</h2>
<p class="warning">本轮单种子、NAVSIM mini开发划分。短窗口目标不等于完整驾驶任务；需同时比较安全与完成指标，训练奖励增长不能证明超过coast。画面仍有重建伪影，RGB用于合成展示，未进入该特权策略。</p>
<p>模型输入为当前自车、邻车、地图、任务终点及剩余时长。动作空间为12类jerk；没有读取日志未来轨迹。</p><pre>OVERVIEW</pre></section>
<section><h2>相同开发场景评测</h2><p>SCENECOUNT个场景，ACTORCOUNT个车辆回合。原始采集日志与训练分组隔离；这不是官方测试或真实道路泛化认证。</p>
<table><thead><tr><th>策略</th><th>车辆回合</th><th>到达率</th><th>碰撞率</th><th>越界率</th></tr></thead><tbody>ROWS</tbody></table></section>
<section><h2>学习记录</h2><label>曲线 <select id="metric"><option value="mean_reward">每批平均奖励（非评测得分）</option><option value="entropy">动作熵</option><option value="value_loss">Value loss</option><option value="approx_kl">Approx KL</option></select></label>
<canvas id="curve" width="1000" height="280"></canvas><p id="range"></p></section>
<section><h2>学习模型驱动的真实 NAVSIM 写实交互</h2><nav>LINKS</nav>
<p>三个分支使用相同冻结模型；1.1秒前为共同前缀。图像没有作为策略输入，不以“画面写实”认证驾驶正确。</p><pre>RENDER</pre></section>
<p><a href="run/report.json">完整评测JSON</a> · <a href="run/config.json">训练配置与数据划分</a> · <a href="run/metrics.jsonl">全部更新记录</a></p>
<script>const data=PAYLOAD;const select=document.getElementById('metric');
function draw(){const c=document.getElementById('curve'),g=c.getContext('2d'),key=select.value,ys=data.map(x=>x[key]),lo=Math.min(...ys),hi=Math.max(...ys),span=hi-lo||1;
g.clearRect(0,0,c.width,c.height);g.strokeStyle='#365366';g.beginPath();g.moveTo(55,10);g.lineTo(55,240);g.lineTo(980,240);g.stroke();g.fillStyle='#aac2d5';g.font='13px system-ui';g.fillText(hi.toPrecision(4),0,20);g.fillText(lo.toPrecision(4),0,240);g.fillText('有效车辆 transitions',780,275);
g.strokeStyle='#65c9ff';g.beginPath();data.forEach((d,i)=>{let x=55+925*d.valid_transitions/data.at(-1).valid_transitions,y=230-210*(d[key]-lo)/span;i?g.lineTo(x,y):g.moveTo(x,y)});g.stroke();document.getElementById('range').textContent=`${data.length} 次更新 · 最后值 ${ys.at(-1).toPrecision(5)}`;}
select.addEventListener('change',draw);draw();</script></html>"""
    links = " ".join(f'<a href="{html.escape(render_directory)}/{name}/viewer.html">{label}</a>'
                     for name, label in (("candidate", "候选原分支"), ("replay", "恢复重放"), ("brake", "前车制动干预")))
    for key, value in (("SCENECOUNT", str(len(report["evaluation"]["episodes"]))),
                       ("ACTORCOUNT", str(report["evaluation"]["actor_episodes"])), ("OVERVIEW", html.escape(json.dumps(overview, indent=2))), ("ROWS", "".join(rows)),
                       ("LINKS", links), ("RENDER", html.escape(json.dumps(render, indent=2))), ("PAYLOAD", payload)):
        page = page.replace(key, value)
    (root / "index.html").write_text(page, encoding="utf-8")
    return root / "index.html"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--render-directory", default="render-v2")
    args = parser.parse_args()
    print(write_report(args.root, args.render_directory))


if __name__ == "__main__":
    main()
