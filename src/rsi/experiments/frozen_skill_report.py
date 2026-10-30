"""Offline interpretation of sealed frozen-skill evidence; no inference or selection."""
import argparse
import json
import statistics
from pathlib import Path

from driversi.agents.backends.native import digest
from driversi.runtime.budget import write_json
from driversi.rsi.decisions.frozen_skills import ARMS, analyze_pairs


def build(run, output):
    run, output = Path(run).resolve(), Path(output).resolve()
    scores = run / "scoring/scores.json"
    receipt = json.loads((run / "scoring/report.json").read_text())
    if digest(scores) != receipt["scores_sha256"]:
        raise ValueError("Scored artifact changed")
    rows = json.loads(scores.read_text())
    prepared = json.loads((run / "prepare/report.json").read_text())
    expected = {(m["token"], s, a) for m in prepared["members"]
                for s in prepared["plan"]["seeds"] for a in ARMS}
    summary = analyze_pairs(rows, expected)
    by = {(x["token"], x["seed"], x["arm"]): x for x in rows}
    scenes = []
    for member in prepared["members"]:
        t = member["token"]
        values = {arm: [by[t, s, arm] for s in prepared["plan"]["seeds"]] for arm in ARMS}
        means = {a: statistics.mean(x["metrics"]["PDMS"] for x in g) for a, g in values.items()}
        scenes.append(dict(token=t, source_log=member["source_log"],
            stratum=member["selection_stratum"], skill=member["skill"]["id"], pdms=means,
            delta=means["matched_skill"]-means["native"],
            zero_resolved_in_both_seeds=all(x["metrics"]["PDMS"] == 0 for x in values["native"])
                and all(x["metrics"]["PDMS"] > 0 for x in values["matched_skill"]),
            skill_xy_ade_m=statistics.mean(x["delta_from_native"]["xy_ade_m"] for x in values["matched_skill"])))
    groups = {}
    for dimension in ("stratum", "skill"):
        groups[dimension] = {}
        for key in sorted({x[dimension] for x in scenes}):
            subset = [x for x in scenes if x[dimension] == key]
            groups[dimension][key] = dict(scenes=len(subset),
                pdms={arm: statistics.mean(x["pdms"][arm] for x in subset) for arm in ARMS})
    gains = sorted(scenes, key=lambda x: x["delta"], reverse=True)
    report = dict(source_scores_sha256=digest(scores), summary=summary, scenes=scenes, groups=groups,
                  strongest_gains=gains[:3], largest_regression=min(scenes, key=lambda x:x["delta"]),
                  zero_resolved_scenes=sum(x["zero_resolved_in_both_seeds"] for x in scenes),
                  improved_scenes=sum(x["delta"]>1e-8 for x in scenes),
                  worsened_scenes=sum(x["delta"]< -1e-8 for x in scenes),
                  interpretation="Exploratory selected-log evidence; threshold changes can amplify small path changes. No automatic promotion or recursive-learning claim.")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "interpretation.json", report)
    lines = ["# AUGUstus前置技能：首轮冻结权重实测", "",
             "32个不同日志标识的场景，12个既有零分、12个非零低分、8个其他场景；每场景2个配对种子、3种提示，共192次规划。样本特意偏向难例，以下均不是全量navtest分数。", "",
             "| 提示 | PDMS（百分制） | 零分场景—种子对/64 |", "|---|---:|---:|"]
    for arm in ARMS:
        a = summary["arms"][arm]
        lines.append(f"| {arm} | {100*a['metrics']['PDMS']:.4f} | {a['zero_count']} |")
    lines += ["", f"两种子均脱离零分：{report['zero_resolved_scenes']}个场景。按两个种子的场景均分，{report['improved_scenes']}个提高，{report['worsened_scenes']}个下降，其余评分不变。全部下降也保留在清单中。", "",
              "## 分层结果", "", "| 原评测分组 | 场景数 | 原生 | 技能 | 中性文本 |", "|---|---:|---:|---:|---:|"]
    for name, g in groups["stratum"].items():
        lines.append(f"| {name} | {g['scenes']} | " + " | ".join(f"{g['pdms'][a]*100:.4f}" for a in ARMS) + " |")
    lines += ["", "## 解释边界", "",
              "本轮权重、动作解码和原生评分均未修改，无优化器更新。首条真实输入的重复动作和开关事后解释动作逐值相同。所有轨迹先封存后评分，不依据真实PDMS选择输出。", "",
              "明显收益集中在少数场景：道路准入或TTC等离散子项改变会放大小幅轨迹变化。不能将这些分数跃升直接解释为掌握了抽象驾驶技能。正常组的进度分也可能轻微下降。", "",
              "路由目前只是依据当前导航与速度选择3个固定、条件式检查表，并未自动发明或修订技能。完整RSI仍需执行反馈驱动的修订、跨日志迁移、持久经验对照，以及晋级/回滚。", "",
              "2个种子不是独立的新场景。当前结果来自已选取的开发日志，不能作全库收益或未见场景泛化结论。保持原生默认，记录候选技能证据。", "",
              "![逐场景增益及路径变化](scene-effects.png)", "",
              "图中横轴编号仅用于本地报告，按技能收益排序，包含全部32个场景；配对种子先在场景内求均值。"]
    (output / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    ordered = sorted(scenes, key=lambda x:x["delta"])
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True, layout="constrained")
    xs = range(1, len(ordered)+1)
    colors = ["#1c8f87" if x["delta"]>1e-8 else "#ca6e45" if x["delta"]< -1e-8 else "#a8b3bd" for x in ordered]
    axes[0].bar(xs, [100*x["delta"] for x in ordered], color=colors)
    axes[0].axhline(0, color="#53616c", linewidth=.8)
    axes[0].set_ylabel("Skill - native PDMS (points)")
    axes[0].set_title("Frozen AUGUstus: 32 selected scenes, paired-seed means")
    unchanged = len(scenes)-report['improved_scenes']-report['worsened_scenes']
    axes[0].legend(handles=[Patch(color="#1c8f87", label=f"Improved score ({report['improved_scenes']})"),
                           Patch(color="#a8b3bd", label=f"Unchanged score ({unchanged})"),
                           Patch(color="#ca6e45", label=f"Lower score ({report['worsened_scenes']})")], loc="upper left", frameon=False)
    axes[1].bar(xs, [100*x["skill_xy_ade_m"] for x in ordered], color=colors)
    axes[1].set_ylabel("Path change from native (cm ADE)")
    axes[1].set_xlabel("Scene index, sorted by score delta (not temporal order)")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.2);ax.set_axisbelow(True)
    fig.savefig(output/"scene-effects.png", dpi=160)
    fig.savefig(output/"scene-effects.svg")
    plt.close(fig)
    return report


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run");parser.add_argument("output")
    args=parser.parse_args()
    result=build(args.run,args.output)
    print(json.dumps({k:result[k] for k in ["zero_resolved_scenes","improved_scenes","worsened_scenes"]}))
