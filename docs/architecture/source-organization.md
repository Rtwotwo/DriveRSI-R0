# 配置、服务与第三方源码组织

当前代码按职责组织。参考 RSIGym 的服务与研究任务分离，不复制其实现，也不引入 Tinker/E2B。

```text
DriveRSI/
├── agents/                         四类保留模型的适配与原生实现
│   ├── driverl/                    特权教师、论文衍生闭环 RL
│   ├── augustus/                   主 VLA 学生
│   ├── chainflow/                  stage1 对照
│   └── fpd_drive/                  推理、评测参考
├── src/
│   ├── config/                     可版本化的平台参数和实验模板
│   │   ├── rsi/                    Harness 任务、技能、LLM 建议、服务分类
│   │   ├── agents/                 模型启动、能力、环境与权重路径
│   │   ├── training/               训练任务、预算、数据清单模板
│   │   ├── evaluation/             固定评测协议和对照
│   │   ├── data/                   数据读入、检索、标注与提示词
│   │   ├── rendering/              GS、视频与白膜参数
│   │   └── visualization/          预览与发布参数
│   ├── rsi/
│   │   ├── services/               从实际注册器生成服务目录
│   │   ├── workflow/               DAG、预算、进程、产物、恢复
│   │   ├── plugins/                具体服务的 prepare/execute 适配
│   │   ├── decisions/              选题、能力分析、干预建议与收益判断
│   │   ├── correction/             技能与纠正记忆
│   │   └── experiments/            配对训练与多轮实验组织
│   ├── data/ training/ evaluation/ sim/ rendering/   各领域实现
│   └── vendor/
│       ├── driving/                NAVSIM 两版本、nuPlan、DrivoR
│       └── rendering/              MTGS、StreetCrafter、WorldDreamer
├── weights/                        初始权重与共享 backbone
└── exp/                            新实验数据、检查点、日志、评测及反馈
```

平台配置的唯一入口是 `driversi.config`：`path(id)` 定位文件，`load(id)` 读取 JSON/YAML 结构化数据（含数组）、TOML 映射或 TXT/Markdown 文本；任务执行仍使用各自的严格参数校验，`catalog()` 列出预设。`rsi/` 专门放 RSI 任务参数；训练、渲染等服务参数按所属领域保存，不混进同一个大配置文件。相对文件引用以该配置的目录为基准；模型配置原有的 repository/weights/paths 字段仍按项目根目录解释。迁移已保持两种原有语义。

```bash
cd <repository>
.venv/bin/driversi config list --group rsi
.venv/bin/driversi config path rsi/tasks/preflight/driverl.json
.venv/bin/driversi config show rsi/registry/services.json
.venv/bin/driversi services --group training
.venv/bin/driversi agents
```

任务执行仍走现有 Harness：

```bash
.venv/bin/python -m driversi.rsi.workflow.engine --registry exp/my-run/harness.sqlite \
  register src/config/rsi/tasks/preflight/driverl.json --output-root exp/my-run/tasks
.venv/bin/python -m driversi.rsi.workflow.engine --registry exp/my-run/harness.sqlite \
  run native-driverl-v3
```

这个示例只执行 DriveRL 清单与严格权重加载。它不采样、不训练、不证明闭环成绩。其余模板运行前仍须准备对应输入。

第三方依赖用 `driversi.vendor.path(name)` 定位，`verify(name)` 校验既有版本及源码清单。移动不改原生算法和许可证；各上游内部的 `configs/`、`third_party/` 是其源码的一部分，保留原样。两个 NAVSIM、不同 diffusers/nuPlan 版本通过原有独立进程加载，不能全局加入同一个 sys.path。平台适配器与原生源码的边界保持明确。

本次没有保留项目根目录 `configs/`、`third_party/` 的软链接或重复副本。新命令必须使用新路径；旧 exp/outputs 中的封存文件、哈希和证书不修改。已有绝对旧路径的任务须重新准备，不能宣称旧回执自动兼容当前实现。运行仍面向完整源码 checkout，不宣称一个 wheel 就包含全部原生环境、资产与权重。

凭据只从安全环境配置读取，不放入 src/config。服务目录显示的是已注册能力，不等于环境、权重或数据已就绪，也不改变现有预算、准入和评分约束。

RSIGym 的启发是把研究任务和执行服务分开，集中管理预算，并返回足够证据指导下一次实验；不是靠目录整理证明递归改进。本轮复用已有预算与产物机制，未新增 HTTP 微服务，也未证明训练收益。

参考：[RSIGym 源码与服务划分](https://github.com/evolvent-ai/RSIGym)、[论文第 3 节](https://arxiv.org/html/2610.10310v1#S3)、[训练服务协议](https://github.com/evolvent-ai/RSIGym/blob/main/train_server/README.md)。
