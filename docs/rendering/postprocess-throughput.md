# 合成后处理的常驻执行与存储

本接口用于在固定物理分支上验证加速，当前为显式隔离入口，尚未替换自动生产调度。使用相同原生 ChainFlow 评分、导出、加载和特征校验；不更改模型、loss、准入阈值。实测数据及限制见 `exp/full-synthesis-throughput-v1/report.md`。

## 组件职责

- `rendering.gaussian.resident_session`：单进程绑定一张 GPU，保留场景资产，逐分支重新设置对象和相机状态；分支与原日志对照使用不同渲染实例。
- `rendering.gaussian.source_session`：缓存源位姿对照。缓存绑定资产、相机、源图、时刻、分辨率及代码，并核验缓存文件；每次仍签发绑定当前分支渲染产物的正式对照回执。
- `rendering.gaussian.image_store`：可选 PNG 压缩级别1及本地临时编码。写入共享盘临时文件后核验字节、同步、原子发布；既有不同文件不会被覆盖。低压缩不改变像素，但文件更大。
- `rsi.services.native_export_worker`：常驻原生 Python 环境执行已有 `chainflow_export`，继续生成原生缓存、重评分和加载校验。特征由已有常驻特征服务重新编码。
- `rsi.experiments.resident_postprocess`：准备好的物理分支进入原有 WorldEngine 与 ChainFlow 准入流程，实验数据进入隔离索引。
- `runtime.parent_exec`、`runtime.experiment_guard`：将子任务与拥有它的进程绑定；异常退出后按明确路径、命令及进程身份清理本实验任务。

## 显式入口

使用项目环境执行 `python -m driversi.rsi.experiments.resident_postprocess --spec <lane.json> --parent-pid <supervisor-pid> --max-seconds 2400`。启动者必须设置对应的 `CUDA_VISIBLE_DEVICES`。该接口要求新的 `exp/` 输出目录，单 lane 最多64个明确输入作业。

spec 包含 `mode`（`baseline` 或 `resident`）、`gpu`、`output`、`config`、`bundle`、可选本地 `staging` 和 `jobs`。每个 job 引用完成的 `generation`、其 `sha256`、`diversity` 及其哈希。配置和输入应由已完成的 Harness 产物生成，不能用手工标记替代物理执行和 ADE 证据。

`baseline` 维持逐条子进程；`resident` 使用常驻渲染与原生导出。两者均执行原有校验并在计时结束前同步输出、结算本阶段拥有的服务。常驻对象不是跨用户共享的全局状态。

## 完整性与恢复

1. 执行前核对物理分支及去重回执；资产和代码在常驻会话期间必须保持不变。
2. 原日志缓存命中仍检查文件完整性及当前渲染绑定，不能直接复用旧分支的准入回执。
3. 原生作业有显式数量、空闲、生命周期和单作业时限。失败或结果未知的作业不自动重复执行。
4. 完成回执保留输入、输出和结果哈希；重入时必须匹配原身份，拒绝覆盖不同输入。
5. 本地临时编码文件不是权威数据。共享盘原子提交成功并通过校验后才可使用。
6. 实验入口不追加生产合格数、不触发训练。接入生产前还需将 lane 调度接到原 Harness 逐场景预算、唯一性和恢复状态机。

## 性能口径

分别记录物理搜索、渲染/导出/新特征/存储、最终合格数。已保存控制的重放不计作新的数据发现；低于画质条件的10条渲染不能算成10条合格样本。后处理加速比不得直接乘到物理搜索或资产重建工期上。一次仅启用明确空闲的 GPU，不与未知任务争抢资源。

验证至少包括四相机完整输出逐像素一致、原生标签/窗口一致、71状态/70动作一致、重新编码的向量一致、源位姿PSNR一致，以及子进程退出和提交失败处理。长期吞吐还须覆盖更多资产及共享盘负载。

多卡正式源码入口已整理为 `src/rsi/experiments/resident_batch.py`，薄入口 `scripts/rsi/benchmark_synthesis_postprocess.py`，参数与范围见[源码入口](../rsi/synthesis-source-entrypoints.md)。默认只读预检；原实验固定启动器保留审计，不再作为新批次入口。
