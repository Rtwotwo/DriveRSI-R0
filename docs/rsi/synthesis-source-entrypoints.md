# 历史扰动合成与加速源码入口

本页是维护入口，实验目录只保存输入快照、启动回执和结果。归档整理不启动数据生产或训练；聚类库接入另行处理。

## 方法及代码职责

|环节|维护代码|
|---|---|
|两秒历史参考扰动、终点/速度网格与五次多项式|`src/rsi/correction/history_perturbation.py`|
|原生LQR物理跟踪适配|`agents/driverl/history_perturbation.py`（按仓库规范属于支持模型适配，不复制进src）|
|教师配置、连续执行和未来搜索衔接|`src/rsi/plugins/navsim_teacher.py`、`src/rsi/experiments/skill_execution.py`|
|逐场景候选、实际ADE及配额/恢复|`src/rsi/experiments/reactive_cohort.py`、`src/rsi/decisions/trajectory_diversity.py`|
|统一生产/训练/复评入口|`src/rsi/experiments/reactive_autoloop.py`|
|GS资产常驻及多相机渲染|`src/rendering/gaussian/resident_session.py`、`src/rendering/gaussian/worldengine_batch.py`|
|源位姿对照缓存|`src/rendering/gaussian/source_session.py`|
|PNG无损快速编码、本地暂存及原子提交|`src/rendering/gaussian/image_store.py`|
|原生导出常驻环境|`src/rsi/services/native_export_worker.py`|
|新历史图像常驻编码与独立向量库|`src/data/catalog/feature_worker.py`、`src/data/catalog/derived_features.py`|
|单卡常驻后处理、多卡隔离编排|`src/rsi/experiments/resident_postprocess.py`、`src/rsi/experiments/resident_batch.py`|
|已有逐作业并行入口|`src/rsi/experiments/postprocess_pool.py`|

实际算法为0～2秒扰动历史参考的LQR跟踪，2～7秒从实际锚点使用DriveRL原生TTS8候选搜索。它借鉴SimScale的到达状态扩充思路，并非完整复现SimScale；也不是梯度TTO。保存实际71状态/70动作，不能用参考曲线代替实际执行。模型、loss、PDMS和已声明准入规则均保持原实现。

## 统一合成入口

薄入口为 `scripts/rsi/run_synthesis_campaign.py`，调用现有 `reactive_autoloop.main`。先检查命令帮助：

```bash
.venv/bin/python scripts/rsi/run_synthesis_campaign.py --help
```

后续明确恢复生产时才执行：

```bash
.venv/bin/python scripts/rsi/run_synthesis_campaign.py --config /absolute/path/to/campaign.json --phase production
```

这是完整原有自动流程，达标后可能进入配对训练；不是只读命令。配置使用 `generation_skill: perturbed_history_tts`、`trajectory_family.method: history_endpoint_grid_tts_v1`。完整要求见[算法协议](history-perturbation-tts.md)。新增可选 `physical_prefetch_workers: 1..4`，默认1，控制CPU物理任务预取；不改候选顺序、ADE结算或逐场景10条要求，也不等于后处理已经并行。无需再使用实验目录中的cohort-prefetch-runner.py等临时启动器来启用这项参数。

## 常驻多卡后处理入口

薄入口 `scripts/rsi/benchmark_synthesis_postprocess.py` 现在调用维护源码 `resident_batch.main`。它取代固定实验的run-batch-benchmark.py中的通用单批调度；原实验脚本和回执保留作审计，不作为新运行依赖。

默认命令仅做只读预检：

```bash
.venv/bin/python scripts/rsi/benchmark_synthesis_postprocess.py --manifest /absolute/path/to/batch.json
```

明确允许渲染测试时加 `--execute`。当前仍为隔离验证入口：不推进物理世界、不写canonical合成训练索引、不启动训练。每批1～64条已执行分支，同一GS资产；1～4张显式渲染GPU，另配一张独立特征GPU。不能自动占用其他任务显卡。

输入JSON字段：

```json
{
  "schema_version": 1,
  "output": "/absolute/project/exp/new-run/batch",
  "config": "/absolute/path/to/prepared-campaign.json",
  "feature_worker_config": "/absolute/path/to/verified-feature-worker.json",
  "mode": "resident",
  "gpus": [1, 2],
  "feature_gpu": 3,
  "max_seconds": 2400,
  "staging": "/tmp/driversi-png-stage",
  "jobs": [
    {
      "generation": "/absolute/path/to/completed-generation/report.json",
      "sha256": "existing-report-sha256",
      "diversity": "/absolute/path/to/diversity.json",
      "diversity_sha256": "existing-diversity-sha256"
    }
  ]
}
```

示例为字段说明，实际jobs数必须不少于GPU数。引用完整物理执行及ADE证据，不能手填通过标记。配置来自已准备的原生ChainFlow stage1任务；服务模板必须对应同一个训练来源库与编码器profile。编排在新输出目录写独立config、feature-worker描述和features.sqlite，不覆盖来源配置或原始库。`mode: baseline`用于相同输入的逐分支渲染/原生导出冷启动对照；两种mode都使用同一个常驻特征服务方案，因此不能称为整条流水线全部冷启动。

同源分支按原顺序交错分配至各lane，每个lane保持资产、标定和原生导出服务；源对照缓存只在已验证相同输入下复用。特征写入仍通过已有串行服务。正常和异常退出都会结算本批拥有的子进程；启动后结果未知的目录禁止直接重跑，应先检查回执。

## 状态与限制

`request.json`、`launch.json`、`status.json`、逐lane回执和`report.json`位于新exp输出；`verified`仅是隔离验证条数，`production_increment`恒为0。图像质量不足的分支仍会被原准入拒绝。

已验证的加速组件已保存，不代表常驻多卡后处理已经接入生产cohort的配额状态机。这里不把隔离benchmark冒充生产加速，也不修改已有统一可视化。真实合成产量、跨资产吞吐和训练收益需后续明确启动后另行验证。
