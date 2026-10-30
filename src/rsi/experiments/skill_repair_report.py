"""Audit real targeted repair receipts; never turn fixture output into driving evidence."""
import json
from pathlib import Path
from driversi.agents.backends.native import digest
from driversi.runtime.budget import write_json
from driversi.rsi.experiments.frozen_skill_probe import read
from driversi.rsi.decisions.skill_evolution import acquisition_group
from driversi.rsi.decisions.skill_repair import ARMS,VARIANTS


def build(directory):
    root=Path(directory);plan=read(root/'declaration/plan.json');run=root/'runs'/plan['task_id']
    task=read(run/'report.json')
    if task['status']!='completed':raise ValueError('Full task not complete')
    dev={acquisition_group(m['source_log']) for m in plan['members'] if m['role']=='development'}
    val=[acquisition_group(m['source_log']) for m in plan['members'] if m['role']=='validation']
    if dev.intersection(val) or len(val)!=len(set(val)):raise ValueError('Cross-log isolation failed')
    episodes=[];total=0;requests=0;prior_decision=None;first_native=None;previous_intervention=None
    for n in range(plan['episodes']):
        proposal=read(run/f'proposal-{n}/report.json');inf=read(run/f'inference-{n}/report.json')
        scoring=read(run/f'scoring-{n}/report.json');decision=read(run/f'decision-{n}/report.json')
        if not inf['parameters_frozen'] or inf['optimizer_updates']!=0:raise ValueError('Unexpected parameter update')
        if digest(scoring['scores'])!=scoring['scores_sha256']:raise ValueError('Scoring receipt changed')
        rows=read(scoring['scores'])
        expected={(m['token'],s,a) for m in plan['members'] for s in plan['seeds'] for a in ARMS}
        actual=[(r['token'],r['seed'],r['arm']) for r in rows]
        if len(actual)!=len(expected) or set(actual)!=expected:raise ValueError('Incomplete factorial coverage')
        if decision['memory_revision']!=n+1:raise ValueError('Memory did not advance')
        if n and proposal['memory_revision']!=n:raise ValueError('Next round not inherited')
        if n and proposal['memory_hash']!=prior_decision['memory_after_hash']:raise ValueError('Next Doctor did not consume committed memory')
        if decision['selection_uses_validation']:raise ValueError('Validation used in variant selection')
        native={(r['token'],r['seed']):(r['raw_trajectory'],r['metrics']) for r in rows if r['arm']=='native'}
        if first_native is None:first_native=native
        elif native!=first_native:raise ValueError('Paired native output drifted across rounds')
        prior_decision=decision
        total+=len(rows);requests+=(run/f'proposal-{n}/doctor/response.json').exists()
        intervention={k:proposal['candidate'].get(k) for k in ('skills','eligibility')}
        repeated=intervention==previous_intervention
        previous_intervention=intervention
        variants={}
        for arm in VARIANTS:
            v=decision['variants'][arm];d=v['full_decision'];variants[arm]=dict(
                target_ttc_delta=v['target_ttc_delta'],declared_target_ttc_delta=v['declared_target_ttc_delta'],
                raw_action_changed=v['raw_action_changed'],development_eligible=v['development_eligible'],
                development_reasons=v['development_reasons'],regression=v['regression'],
                harm_by_cohort={role:dict(new_zero_scenes=len({x[0] for x in c['comparisons']['incumbent']['new_zero']}),
                    safety_regression_scenes=len({x['token'] for x in c['comparisons']['incumbent']['safety_regressions']}))
                    for role,c in d['cohorts'].items()},
                means={role:{k:100*x for k,x in c['means'].items()} for role,c in d['cohorts'].items()})
        episodes.append(dict(episode=n+1,proposal_valid=proposal['valid'],proposal_error=proposal.get('proposal_error'),
            modification_check=proposal['modification_check'],experience_read=proposal['experience_count'],
            same_intervention_as_previous=repeated,
            selected_variant=decision['selected_variant'],diagnostic_variant=decision['diagnostic_variant'],
            accepted=decision['decision']['accepted'],reasons=decision['decision']['reasons'],
            validation_target_ttc_delta=decision['validation_target_ttc_delta'],variants=variants))
    report=dict(status='completed',scored_predictions=total,distinct_scenes=len(plan['members']),
        elapsed_seconds=task['elapsed_seconds'],doctor_requests=requests,optimizer_updates=0,
        cross_acquisition_log_isolation=True,validation_fixed_before_new_scores=True,
        previous_validation_drives_may_be_reused=True,scope=plan['scope'],
        automatic_round_transition=True,paired_native_replay_exact=True,episodes=episodes,production_policy_changed=False,
        stage_elapsed_seconds={s['id']:s.get('elapsed_seconds') for s in task['stages']},
        source_receipts={str(run/'report.json'):digest(run/'report.json')})
    out=root/'analysis';out.mkdir(exist_ok=True);write_json(out/'report.json',report)
    lines=['# 冻结AUGUstus：针对性技能修复实测','',
        f'完成{total}条规划，{len(plan["members"])}个不同锚点，两个种子、五组、两轮；耗时{task["elapsed_seconds"]/60:.2f}分钟。权重更新0，Doctor请求{requests}次。',
        '', '具体问题为直行规划TTC失败，不推断未经证实的跟车原因。8个开发目标、24个开发保护场景；8个同类跨日志验证＋8个广泛保持验证。验证与开发采集日志隔离，但验证日志可能用于旧实验，属于适应试验。',
        '', '|轮次|修改|开发目标TTC变化|开发PDMS|验证PDMS|历史回归|开发资格|', '|---|---|---:|---:|---:|---|---|']
    names={'text_only':'只改文本','gate_only':'只改条件','combined':'联合修改'}
    notes=[]
    for e in episodes:
        for a,v in e['variants'].items():
            lines.append(f'|{e["episode"]}|{names[a]}|{100*v["target_ttc_delta"]:+.2f}分|{v["means"]["development"]["candidate"]:.4f}|{v["means"]["validation"]["candidate"]:.4f}|{"通过" if v["regression"]["passed"] else "失败"}|{"通过" if v["development_eligible"] else "失败"}|')
        notes+=['',f'第{e["episode"]}轮读取{e["experience_read"]}条经验；'+('晋级' if e['accepted'] else '回滚')+'：'+ '; '.join(e['reasons']), '']
        if e['same_intervention_as_previous']:notes+=['实际技能文本与条件重复上一轮；读取经验不等于生成了新的有效干预。','']
    base=episodes[0]['variants']['text_only']['means']
    lines+=['',f'既有技能基线：开发PDMS {base["development"]["incumbent"]:.4f}，验证PDMS {base["validation"]["incumbent"]:.4f}。',*notes]
    lines+=['所有候选先封存规划再评分；开发结果选修改，验证不用于倒选。无合格修改时仍保存失败经验并自动开始下一轮。',
        '', '本轮没有长度/错配提示新对照，不据此主张文本语义因果或广泛RSI收益。采样有限，严格原安全门槛保留，没有事后放宽。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    return report


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('directory');args=p.parse_args();r=build(args.directory)
    print(json.dumps({k:r[k] for k in ('status','scored_predictions','elapsed_seconds','optimizer_updates')}))
