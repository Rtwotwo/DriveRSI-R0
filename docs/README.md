# DriveRSI documentation

- [Installation and configuration](getting-started.md)
- [Run and inspect](running.md)
- [Training and evaluation](train-eval.md)
- [Release verification](release-verification.md)
- [Complete workflow and task control](workflow.md)
- [Data and image contracts](data-contract.md)
- [Code ownership](code-map.md)
- [Release status and known limitations](release-status.md)

The reference material below includes historical experiments and optional backends. Those results and machine-local artifacts are not distributed with this snapshot.

## Technical reference

|目录|内容|
|---|---|
|architecture/|源码与配置组织、统一可视化|
|models/|模型范围及VLA服务设计|
|training/|原生训练和小规模实验说明|
|rsi/|Harness、技能、记忆、服务流程和扩充分支|
|data/|数据库、检索、标注、语义及白膜交付|
|rendering/|GS多视角、资产、配准和生成辅助|
|design/|既有专题设计资料|
|research/|既有调研与研究边界|

优先入口：[总体架构](architecture/system-overview.md) · [项目分层](architecture/repository-layout.md) · [命令入口](../scripts/README.md) · [统一工作台](architecture/unified-visualization.md) · [Harness服务](rsi/harness-service-program.md)。
文档中的历史实验数量和结论保留其原始范围，不代表当前全平台已通过验收。

- [DriveHarness技能、自迭代与10月调研](rsi/harness-evolution.md)

- [场景簇池与训练候选扩展](data/scene-cluster-pools.md)

- [场景语义标签、检索与聚类池](data/scene-labels.md)

- [实际历史扰动与原生TTS生成](rsi/history-perturbation-tts.md)

- [常驻合成后处理与存储](rendering/postprocess-throughput.md)

- [历史扰动合成与加速源码入口](rsi/synthesis-source-entrypoints.md)：代码归属、规范命令和隔离运行边界。

- [测试短板到训练簇映射与检索交集](rsi/cluster-failure-mapping.md)

- [检索地理距离与私有图像对照](rsi/retrieval-geography.md)

- [按场景调整生成策略与自动恢复](rsi/harness-adaptive-generation.md)

- [逐问题自动执行与安全交接](rsi/controller-continuation.md)

- [冻结模型的条件化技能与经验对照](rsi/conditional-skill-memory.md)
