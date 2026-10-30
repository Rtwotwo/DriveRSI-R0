# 命令入口

scripts 是运维和实验的启动入口：读取参数、调用所属 src 模块、返回退出码。模型实现不放这里，实验结果也不放这里。
每个现存 Python 入口的用途、实现位置和状态见 [catalog.json](catalog.json)。未列为历史不等于可直接运行，仍须检查环境、输入、预算和输出位置。

|目录|用途|
|---|---|
|build/|原生扩展和软件包构建|
|data/|场景索引、向量、caption、数据校验和白膜交付|
|rsi/|技能、记忆、分支扩充、配对实验和服务流程|
|training/|原生训练入口及有界训练验证|
|evaluation/|正式基准审计、聚合和结果导出|
|simulation/|连续日志、交通交互及官方运行时对照|
|rendering/|GS窗口、相机/白膜产物和本地发布入口|
|diagnostics/|明确对象和输入的工程诊断|
|legacy/reports/|保留旧报告生成程序，供历史审计，不作为新实验入口|

## 常用入口

以下命令仅展示帮助或只读目录；真实执行须按帮助显式提供参数：
```bash
.venv/bin/driversi agents
.venv/bin/driversi services
.venv/bin/python scripts/rsi/run_platform_spine.py --help
.venv/bin/python scripts/data/manage_scene_catalog.py --help
.venv/bin/python scripts/data/run_scene_corpus.py --help
.venv/bin/python scripts/rendering/run_multiview_gs_window.py --help
.venv/bin/python scripts/training/run_chainflow_epoch.py --help
.venv/bin/python -m driversi.visualization --help
```

统一工作台通过 `python -m driversi.visualization build/publish/watch` 更新，新实验只增加数据登记。历史报告工具会创建其原有页面，不应用它们替换统一站点入口。

## 兼容与退役

26 个原 scripts 根目录别名已移除；规范路径见 src/layout.json 的 retired_script_aliases。旧 exp 回执不改写，不能保证其中旧命令直接重放。
已移除停用的 navtrain 全量学生评分、依赖旧评分确认的 campaign 和教师启动脚本；这不删除原生评测器，也不改变教师反事实评分。
新业务不再从另一个脚本动态导入。公共数据编排、发布校验和GS窗口检查已归入 src/data、src/visualization、src/rendering；部分构建或验证脚本仍负责自身的命令编排。

历史扰动/合成使用 `rsi/run_synthesis_campaign.py`；常驻多卡后处理预检及隔离验证使用 `rsi/benchmark_synthesis_postprocess.py`。算法与编排归属src，详见 `docs/rsi/synthesis-source-entrypoints.md`。后者默认只读预检，`--execute`才运行GPU任务。

发布版入口与顺序以 [安装说明](../docs/getting-started.md) 和 [工作流](../docs/workflow.md) 为准。旧 Pages 名称仅保留 Python/命令兼容，导出不再包含 GitHub 工作流。配置显式副本使用 `python scripts/build/materialize_config.py --input PRESET --output src/config/local/job.json`。
