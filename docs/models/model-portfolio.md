# 模型范围与代码归属

当前驾驶模型仅保留 `agents/driverl`、`agents/augustus`、`agents/fpd_drive`、`agents/chainflow`。DriveRL 为特权教师和闭环 RL 学习对象；AUGUstus 为主 VLA 学生；FPD-Drive 与 ChainFlow stage1 为测试对照。权威清单：`src/config/model-selection.json`，`driversi agents` 会同时显示可执行原生适配与该分工。

Drive-JEPA 是计划接入的独立项目，尚未配置训练器或权重，不是 WA-JEPA 的别名，也不等同于可生成图像的世界模拟器。WA-JEPA 的实现、配置和专属权重已退役。

## 渲染属于平台服务

- MTGS 适配与物理绑定：`src/rendering/gaussian/`；上游实现：`src/vendor/rendering/mtgs/`。
- StreetCrafter 适配：`src/rendering/video/streetcrafter/`；上游实现：`src/vendor/rendering/streetcrafter/`。
- WorldDreamer 小规模对照适配：`src/rendering/video/worlddreamer/`；上游实现：`src/vendor/rendering/worlddreamer/`。

这些渲染器没有 `agents/` 别名。独立进程通过各自 `runtime.py` 启动，Harness 绑定当前适配与第三方源码。上游许可、版本及既有补丁说明保留。渲染权重、GS 重建资产、SAM 与视觉标注 backbone 继续保留。

## 训练能力边界

DriveRL 当前是平台按论文实现的有界在线学习器，不是官方完整训练代码；AUGUstus 已有小规模原生 SFT 配对训练、恢复和反馈证据，但没有稳定收益或 GRPO 在线接入结论。ChainFlow 原生 loss 配对训练保留；FPD-Drive 当前仅有推理和评测适配。此轮仅整理代码归属，不改变原生模型、loss、评分及准入。

旧 MLP/线性模型不再安装为产品模型；通用采样器测试的小网络仅在 `tests/support/diagnostic_agents.py`，测试配置在 `tests/fixtures/legacy/`。历史 NPZ 读取器位于 `src/training/legacy/frozen.py`。通用 legacy 训练/采样基准必须显式指定 agent 配置，不会默认加载旧 MLP。

下载权重保存在 `weights/`；新训练产物保存在 `exp/`。已有 `weights/runs/augustus` 是历史训练产物，本轮保留，避免破坏当前回执依赖。历史封存实验不改写、不刷新哈希；迁移后创建新任务需重新准备，旧证书不会自动变成当前版本合格证明。

## 本轮执行验证（2026-10-25）

核心回归 3,148 通过、55 跳过；版本锁修复后的相关回归 67 通过；独立 Torch 环境的接口测试 83 通过，数量存在覆盖，不能相加。MTGS 已在迁移后真实加载 Boston 资产并输出 RGB、深度和累积图；StreetCrafter/WorldDreamer 的六个独立入口及原生环境导入通过。此项验证不证明画质或驾驶指标提升。

实际启动计划检查中，DriveRL 预检依赖齐备。FPD 默认评测配置仍缺独立环境、DINOv2 backbone 和 v1 metric cache；AUGUstus 默认评测示例仍缺对应 dataset.json 与评分缓存，运行时应使用实际实验的数据清单。本轮没有修改这两个默认配置或删除这些依赖。Drive-JEPA 仍须独立完成模型与训练器接入。
