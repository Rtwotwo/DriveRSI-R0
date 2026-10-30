# VLA 困难场景的可执行服务记忆

本模块在 DriveHarness 中提供“实际历史条件 → 经验检索 → 执行固定技能 → 核验当前证据 → 回写”的最小链路。它服务于 AUGUstus 的离线诊断和数据改进，不向驾驶模型注入 prompt，不接管方向盘，也不把教师分数记作学生收益。

## 参考与取舍

RoboRSI 的公开设计使用任务—技能层次、执行与复审分工、稳定流程的代码固化，以及 evolve/frozen 模式。本模块借鉴这些边界，复用 DriveHarness 的预算、封存产物和失败记录，没有复制它的机器人原语或宣称复现其性能。当前代码不自动生成或安装新技能，也没有多 Agent 自主改代码。

来源：[RoboRSI](https://github.com/nssmd/RoboRSI)、[冻结与进化协议](https://github.com/nssmd/RoboRSI/blob/main/docs/EVALUATION.md)。

## 三层组织

| 层次 | 内容 | 当前执行 |
|---|---|---|
| task | augustus/hard_scene_service | 绑定学生父权重、教师版本和真实来源 |
| atomic | world_validity / teacher_qualification / joint_supervision | 按顺序报告最早未通过的门槛 |
| base | history_signature / contact_trace_audit / native_readiness_audit | 真实历史提取、条件触发的接触明细检查、既有原生准入审计 |

“最早未通过的门槛”不是已经证明的根因。例如世界中有接触，仍需进一步区分原始标注、接管、控制器和模型行为；当前保持 causal_attribution=unresolved。

## 记忆内容与检索

SQLite `vla_scene_cases` 保存实际历史条件、父学生/教师上下文、源日志组、封存执行身份、未通过检查和证据适用范围。经验限定为 model-service diagnostic，student_gain_credited=false；它不进入原有 SkillMemory 的训练收益账目。

查询只读取实际四帧历史，要求2 Hz、末帧为当前锚点：

- 用后轴位姿计算1.5秒平均路径速度和累计航向变化。
- 用当前实际在场对象的位置统计30米内对象数，不推测类别或未来行为。
- 首版分箱：速度低于0.2/2 m/s，转角大于0.1 rad，附近对象至少8个；使用集合 Jaccard，0.25 起召回，最多3条。

这些是尚未校准的粗粒度检索条件，不是驾驶失败、视觉质量或轨迹准入阈值。没有为真实验证的未命中调低阈值。后续可增加经验证的地图、对象类别和审核后的 caption 条件；不能用未来轨迹、测试明细或学生低分反推当前视觉观察。

当前只匹配相同学生模型名、父权重绝对路径、既有教师检查点摘要和技能版本，排除同一源日志组，避免把同一个案例重放当作跨场景迁移。这里的父模型隔离是路径标识，不是完整学生配置或权重内容等价证明；不要在原路径替换权重后仍按同一父模型解释记忆。跨父模型经验迁移尚未启用。源目录只能是官方 navtrain；不要求 score_run，不读取或生成 navtrain 学生评分。

物理预热起点与官方场景token可能不同。训练成员身份从封存 scene 的 `source_train_membership` 读取，与官方目录的token、日志和anchor_index核对；实际历史来自执行窗口。教师报告和窗口的起点token分别保留，不能拿它们替换官方成员身份。

## 执行与模式

`vla_memory_skill_episode` 输入是同一 Harness 台账中的选中教师分支和教师执行证据。检索结果中至少一条具有实际正接触记录时（单纯恢复/历史检查失败不会触发），它调用真实 `contact_trace_audit`，统计对象对、首次接触tick、跨锚点持续和新增接触。计数单位是pair/tick记录，不是独立事故数。旧案例只决定“检查什么”，当前事实决定结论。

`native_readiness_audit` 始终运行，不能被记忆跳过。它继续执行既有95分门槛及物理/监督检查；记忆不会赋予数据训练资格。

- `evolve`：追加诊断案例；同一证据重复写入不会增加案例数或成功次数。
- `frozen`：只读已有记忆。即使记忆文件不存在，也不创建它；允许生成本次任务的输出回执，但不修改记忆库。

本次审计耗时由报告与 Harness 回执记录，scope 为本次审计。复用旧物理分支不会重新收费为一次新仿真；这些数值不是端到端训练成本，更不能用它们计算训练收益/成本。

## 可执行入口

从 DriveRSI 根目录运行，`--registry` 必须是保存来源任务的原 Harness 台账：

```bash
.venv/bin/python scripts/rsi/run_vla_memory_episode.py \
  --registry exp/harness-capability-skills-v1/native-attempt-02/harness.sqlite \
  --source-task native-attempt-02 --branch-stage admission \
  --database exp/scene-catalog-v1/navtrain.sqlite \
  --memory exp/my-vla-memory/cases.sqlite \
  --output-root exp/my-vla-memory/tasks \
  --task-id audit-001 --mode evolve
```

教师/学生默认配置沿用 `src/config/rsi/skills/augustus-teacher-student.json`。更换 `--mode frozen` 并使用新的task-id可复核冻结状态；所有任务回执都保留，不覆盖旧任务。

## 实际验证与限制

三份既有真实 navtrain 日志，在独立进程中完成写入、检索、执行与回写。第一份近乎静止、周围密集，第二份行驶较快、周围较疏，两者未匹配。为验证调用路径，随后有目的地选取第三份“低速、近直行、周围密集”的既有日志：它以0.5的条件相似度命中第一份经验并执行接触检查。这不是随机检索准确率实验。

第三份的实际历史/未来均没有接触，流程没有照搬旧案例的“接触”结论，而是继续报教师轨迹准入未通过（既有反事实PDMS约78.11）。前两份仍保留原来的100/95.35分及接触拒绝。这些均为旧教师分数，本轮没有新增模型评分或重新推进物理世界。

冻结重放后数据库字节不变；再次以evolve重试同一证据，案例数仍为3。最初一次因源token语义混淆而失败的任务已保留；修复使用现有封存成员字段，没有放宽来源验证。

当前证明的是“记忆确实能影响可执行诊断程序，并且不替代当前验收”。它没有证明诊断准确率、节省计算、AUGUstus学会新技能、合成数据合格或持续训练收益。当前原生联合监督和AUGUstus技能配对反馈仍未接通。

下一步应以新日志检验适用条件，记录误召回/漏召回和真实耗时，比较无记忆、固定程序与有记忆三组；只有独立实验支持后才升级技能的可靠性等级。将其作为在线辅助时，必须分别报告原生AUGUstus与AUGUstus＋辅助系统的表现。
