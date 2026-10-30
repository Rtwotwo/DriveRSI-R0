# 多策略轨迹族与原生配对训练

本入口服务于一个训练来源的有限生成实验。物理候选、视觉合格数据与学生训练收益分别记录；不能把通过PDMS的执行轨迹数量当成训练样本数量。

## 生成与准入

`python -m driversi.rsi.experiments.trajectory_family --config CONFIG --selection SELECTION --phase physical` 使用最多4个CPU工作进程，复用已完成的物理回执。`--phase produce --shared-gpu 6` 按原计划完成顺序渲染、既有视觉准入、原生导出和真实图像特征提取。共享GPU参数必须对应用户明确授权；渲染前检查剩余显存，沿用GS新视角10%、源视角15%的PyTorch上限。

当前两类生成器独立留证：原生DriveRL随机闭环全7秒；前2秒C2受限历史控制、后5秒原生DriveRL纠正的组合生成器。不能把后一类写成纯DriveRL。固定当前来源的物理总预算为64次：初始8次组合控制、32次原生随机、24次修订组合控制；不是无限重试。

用户已授权新数据采用严格PDMS>0.95，EC/EPDMS保留诊断用途。`dataset/collection.json` 指向新分区 `synthetic_driverl_pdms_v2`；旧完整EPDMS分区及其封存证据不变。物理连续性、对象一致性、源视角PSNR>25以及原生监督检查继续执行。当前视觉配置为`experimental_numeric25_v1`，源视角重建达到门槛不能冒充独立新视角画质认证。

## 合并与训练

```bash
.venv/bin/python -m driversi.rsi.experiments.trajectory_family_merge \
  --config exp/continual-family-pdms-v1/merged/config.json \
  --family exp/continual-family-pdms-v1/native-sampling/production \
  --family exp/continual-family-pdms-v1/controlled-history/production \
  --wait-seconds 10800 --train --shared-gpu 6
```

此命令是有限一次性任务依赖等待，不创建定时监控。只有原生产者已结束才能合并；重新验证原准入、执行文件、学生父模型与生成器血缘。相同来源的完整7秒和后5秒ADE必须同时达到0.5米；跨策略重复包或近重复轨迹排除。0.5米是本试点既有工程门槛，不是论文给定的通用技能差异标准。

合并不移动或改写原分支，用已有`reused_executions`契约保留原计划、技能身份、控制参数和场景证据。每个来源至少10条真正完成渲染和导出的合格数据才能训练；总数够但某个来源不足仍不通过。合并报告保存原族报告与排除原因。

当前新配对配置为GPU6单卡、原生FP32、batch2、40次更新（20次训练+20次恢复），每8个全局样本插入一条合成数据，使10条新数据均进入优化器。控制组与合成组共享父权重、预算、原生模型/loss、普通训练池和评分协议。设置20%显存上限，保持与原生评测入口兼容。两臂和额外恢复重放验证之后，分别进行全量12146个navtest评测与反馈；没有navtrain学生评分。

## 已训练小试验的评测恢复

`driversi.evaluation.benchmarks.chainflow_pair` 对此前已完成的8更新检查点进行父模型/控制/合成组三臂全量评测。`chainflow_pair_import --evaluation PATH --wait-seconds 10800` 核对六份原始分片、逐场景完整性、父权重和评价身份，建立新的恢复回执，随后调用既有配对反馈和技能记忆。它不会覆盖旧失败节点、重复训练或自动晋级。

新40更新实验与旧8更新单样本实验必须分别报告。生成成功不记为学生收益，平均分上升不等同于所有安全子项无退化；正式晋级仍须通过持续学习控制器的保持与回滚条件。
