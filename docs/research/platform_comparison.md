# 论文比较与平台选择依据

核查日期：2026-10-12。目的：支持 DriveRSI 的需求和选型，而非宣称全面复现。GigaFlow/GigaPixel 使用本地 PDF 正文与附录，其他工作针对方法、系统与边界核查；DriveZero 详读记录在此前本地研究材料中。本文的实现建议是分析，不是论文原有平台保证具备的能力。

## 1. 名称与职责

| 标准名称 | 含义与本项目用途 |
|---|---|
| GigaFlow | *Robust Autonomy Emerges from Self-Play* 的仿真及训练系统；策略 self-play 的主要方法参考 |
| GigaPixel | *Scaling Self-Play for End-to-End Driving* 的视觉仿真平台；学生闭环与教师分叉参考 |
| PufferDrive | 可检查源码的仿真软件底座；不等同于基于它实现的各论文扩展 |
| self-play | 多个参与者在共同环境中交互并改进策略；不要求每车一个独立大模型 |
| RSI | 本项目暂指 Recursive Self-Improvement；需单独定义与检验改进器能力 |

## 2. 各工作解决的问题

| 工作与原始来源 | 核查到的关键机制 | 对 DriveRSI 的建议 | 适用边界 |
|---|---|---|---|
| [GigaFlow](https://proceedings.mlr.press/v267/cusumano-towner25a.html) | 合成地图、大量并行世界、共享且逐车条件化的策略；每车独立目标与奖励偏好 | 首先做共享策略、逐车条件和高效批量采样 | 结构化特权输入；其局部近邻筛选不等于遮挡建模；无需把历史联赛设为必需条件 |
| [GigaPixel](https://arxiv.org/abs/2606.19641) | 结构化教师 RL → 视觉学生 self-play DAgger → 真实图像适配；从学生状态分叉教师世界 | 快照/分叉、学生状态采样、模态和控制适配应提前设计 | 主车 LQR 跟踪轨迹，其他车用学生动作头；仍需真实/合成配对数据；教师可能看到被遮挡对象 |
| [DriveZero](https://github.com/XiaomiAutoL3/DriveZero) | DriveRL 的混合 provider 与可选多车 self-play；向视觉规划器蒸馏 | 保留逐车控制器路由；使用已有推理版本作为外部基线 | 主实验默认 ego-only；视觉蒸馏不同于学生闭环 DAgger；公开推理不代表训练完整开放 |
| [GPUDrive](https://arxiv.org/html/2408.01584v3) | Madrona/C++ 加速多世界，多种观测与 Python 接口 | GPU 仿真备选；关注稀疏对象、批量形状和可控 actor 掩码 | simulator FPS 不能当作完整 PPO 吞吐；功能与编译需按当前提交核查 |
| [Waymax](https://arxiv.org/abs/2310.08710) | JAX 对象级状态转移、可替换动力学和行为模型；以 WOMD 场景为主要入口 | 参考显式状态与可组合接口，可作为独立对象级对照 | 非相机仿真；不能直接承担后续视觉训练，也不等于现成 self-play 实验管理系统 |
| [TerraZero](https://arxiv.org/html/2607.13028v1) | C 侧仿真配合 GPU 学习；地图、初始化、类型、动力学、信号、规则交通可组合 | 场景应是可组合配置；真实地图可以承载新交互；CPU 仿真也值得验证 | 是方法及工程参考，未在本项目确认可部署的官方完整训练源码 |
| [Pictura](https://arxiv.org/html/2607.26005v1) | 简化透视渲染后直接像素 PPO；显式研究遮挡 | 视觉路线保留直接 RL，对照教师蒸馏；可见性进入核心契约 | WOMD 迁移使用同一渲染器重新渲染布局，并非真实相机迁移完成 |
| [SPICED](https://arxiv.org/html/2606.19370v1) | 冻结人类行为克隆策略，以 KL 正则约束 self-play | 人类风格锚点是可选实验；分开评估本群体、规则交通与 replay | human-replay 是非反应代理，不能证明与真实人类的双向交互能力 |

以上八项的共同启示是：仿真、控制、观测、训练和评测需要分离。没有一项论文单独提供了本项目全部 RSI 实验能力。

## 3. GigaFlow：重点借鉴条件化和场景生成

依据：本地 28 页 PDF（arXiv v1），PDF p. 3 Figure 2；pp. 13–19 附录；p. 27 Figure A3。

建议在统一策略的输入中明确加入该车自身目标、风格、动力学条件；训练时随机化，评测时固定并公开。邻居的隐藏目标与偏好不能因共享模型而泄漏。历史策略池可用于检验泛化和遗忘，其收益要与条件化本身分开消融。

论文的吞吐是特定硬件和配置下的系统结果；原文约 1.2M transitions/s 对应 8 GPU 的训练配置。长期低事故展示还调整了决策频率及环境条件，不能当作任意交通或真实道路的可靠性保证（PDF pp. 22–24）。

## 4. GigaPixel：重点借鉴学生到达状态的教师分叉

依据：本地 26 页 PDF（封面为 arXiv v2），PDF p. 4 Figure 2/Equation 1，p. 8 Table 3，p. 10 limitations，pp. 23–25 Tables 5–6 与实现附录。网页 v1 仅作为检索入口，具体实现以所读 PDF 为准。

实现上要区分三类对象：继续演化的学生世界、同一状态创建的教师分叉、用于监督的轨迹/动作标签。标签必须绑定教师版本、控制条件、观察协议与有效窗口；失败、提前终止或不完整目标不能悄悄进入正常监督。

附录的轨迹/动作双头执行尤其影响平台设计：不能假定所有体型的车都可直接使用主车 LQR 参数。教师分叉也不能只改一个 ego 而不记录周围车采用什么控制器。后续可对比全教师联合 rollout 与固定周车的反事实标签，但应把它们定义成不同实验。

仓库截至检查时主要为项目说明和素材，仍列出未来发布时间：renderer 预计 10 月 23 日，教师 10 月 30 日，学生 11 月 6 日，适配/评测 11 月 13 日。这是维护者计划，不是可依赖的交付承诺；目前不把它设为平台起步的阻塞依赖。[官方仓库](https://github.com/montrealrobotics/gigapixel)

## 5. 可以复用什么，应该自己建设什么

| 分类 | 建议 |
|---|---|
| 优先复用 | 一个对象级仿真内核、现有 PPO 训练实现、地图读取工具、已有 NAVSIM 评测组件、DriveRL 推理基线 |
| 平台自己持有 | 统一场景/状态/动作契约、数据划分、逐车控制器分配、完整快照、实验版本、跨对手评测与失败归档 |
| 先验证再承诺 | nuPlan/NAVSIM 转换器、信号语义、原生 jerk、车辆跟踪器、CPU/GPU 传输、后端快照可扩展性 |
| 稍后实现 | 批量透视渲染、可见性受限教师、世界模型想象、VLM 控制、自主实验改进器 |

[PufferDrive 官方仓库](https://github.com/Emerge-Lab/PufferDrive) 已有训练入口与地图转换说明。[2.0 Python 接口](https://github.com/Emerge-Lab/PufferDrive/blob/2.0/pufferlib/ocean/drive/drive.py) 可见 jerk/classic 和多种控制模式；因此不能照抄较早论文比较表中“只有一种动力学”等描述来判断当前版本。接口存在仍需编译和行为验证。首次选型锁定具体提交，不把多个项目各自依赖的 PufferLib 版本混装。

## 6. 对上一版方案的修订

| 原先倾向 | 本次结论 |
|---|---|
| GPUDrive 优先 | PufferDrive 先做工程验证；GPUDrive 保留备用；决定因素是现有数据、扩展成本和实测结果 |
| 较早建设策略池 | 先共享策略与逐车条件，历史池后续作为有对照的研究变量 |
| 视觉可从离线教师蒸馏开始 | 保留这一基线，同时把学生闭环、教师分叉与可见性匹配作为长期能力 |
| 统一到 DriveRL 的动作/频率 | 初期保留后端原生行为，再通过显式 adapter 对齐；不能仅统一张量形状 |
| 以高吞吐作为主要起点 | 先验证世界与数据语义，再测完整训练吞吐；稀有错误和失败可复现性同样决定平台价值 |

## 7. 需要防止的三种误判

**速度误判。** 同一张图可能报告 render-only、simulation-only 或 training throughput。车辆数、停用掩码、相机数、模型和硬件不同，不能直接排名。我们的统一 profiling 协议见需求文档。

**交互误判。** self-play 高分可能依赖共同形成的惯例；replay 又缺少双向反应。应把相同策略、冻结异策略、规则交通和 replay 分开报告，再做混合群体与未见场景评测。

**递归改进误判。** 模型在自身经验上训练与改进器本身变得更善于改进，是两个不同待检验命题。平台优先提供公平预算、稳定评测和完整实验血缘，使后续研究能够区分它们。

## 8. SimScale、GigaPixel 与可控渲染

核查时间：2026-10-13。这里的 SimScale 指 OpenDriveLab 的驾驶合成数据工作，不是同名工程 CFD 产品。参考：[SimScale 论文 v2](https://arxiv.org/html/2511.23369v2)、[官方仓库](https://github.com/OpenDriveLab/SimScale)。

论文 §2.2–2.3 将背景交通响应、ego 轨迹执行与传感器渲染分开：LQR 执行 ego 轨迹，IDM 背景车产生响应；静态 3DGS 与动态对象分离，以相机内外参和车辆位姿渲染。先扰动后恢复的两段 rollout 需要可行性和画质过滤。发布数据的 6 s 样本由 2 s 历史和 4 s 未来组成，不是整个模拟世界只能运行 6 s。对 DriveRSI 的启示是权威状态和渲染解耦，先验证有界区域的离线生成；这并不证明任意视角、任意新车或任意扰动都有可信图像。

公开仓库已有数据、模型和训练/评测入口；本次核查未找到可直接运行的完整 SimScale 3DGS 数据生产入口，不能将发布规模当作本项目可复用的生成系统。其 [IDM 实现](https://github.com/OpenDriveLab/SimScale/blob/main/navsim/traffic_agents_policies/navsim_IDM_traffic_agents.py) 围绕给定 ego 轨迹计算反应背景；仍需核验与本项目每 tick 全车共同观测、联合决策语义的适配。

GigaPixel §3.1 / Appendix A 使用 PufferDrive 世界状态和 Madrona 批量渲染，车辆/障碍物为简化几何，道路与信号也采用简单资产。它更适合启发快速、可控、带遮挡的多视图接口，不应归类为写实数字孪生。其计划发布状态见第 4 节；当前不能依赖尚未发布的 renderer 完成工程交付。[论文 v2](https://arxiv.org/html/2606.19641v2)

MTGS 是可进一步试点的公开重建路径，提供数据准备、重建、评测、查看和 checkpoint 导出工具。先使用一个官方处理片段与匹配权重，再核对世界坐标和动态资产支持；自动重建需要匹配的原始 nuPlan 数据及标定，不只是 NAVSIM pickle。文档中的 60 Hz 是插值姿态输出配置，不能引用为渲染速度实测。[仓库](https://github.com/OpenDriveLab/MTGS)、[自动重建](https://github.com/OpenDriveLab/MTGS/blob/main/docs/auto_reconstruction.md)、[数据准备](https://github.com/OpenDriveLab/MTGS/blob/main/docs/prepare_dataset.md)、[运行说明](https://github.com/OpenDriveLab/MTGS/blob/main/docs/running.md)

| 阶段 | 可借鉴路径 | 必须独立验收 |
|---|---|---|
| 在线 BEV 与记录 | 自有结构化世界 + 只读可视化 Session | 暂停/单步/reset、慢客户端隔离、逐 tick 无损记录；实时显示不影响物理 |
| 简化透视视图 | GigaPixel 的几何与批量相机思路 | 遮挡、相机投影、位姿与 tick 对齐、批量吞吐；标明 primitive_rgb |
| 写实离线合成 | SimScale 的行为/渲染解耦、MTGS 的重建与导出 | 地图/资产坐标、动态车替换、有效视域、画质与物理过滤；标明 reconstructed_rgb |
| 写实在线闭环 | 上一阶段合格资产 + 受控渲染 worker | 多相机延迟、队列背压、GPU 内存、控制超时；不以离线渲染 FPS 代替完整闭环吞吐 |

原始图像的存在只能支持日志视角；自定义路线偏离后要重新生成对应图像和遮挡。当前平面地图不能保证完整 SE(3) 相机姿态，坡道和高架需额外几何。上述仍为分阶段设计，具体契约、性能分母和数据准入维护在需求文档第 16 节。

## 9. DriveZero / DriveRL 的日志与交互时域

核查时间：2026-10-13。依据 [DriveZero 论文](https://arxiv.org/html/2609.06055v1) §3.1、Appendix A.1/A.2/A.4、Table A12，以及本地 PDF 第 9、24–26、30 页。应参考其结构化交互平台 DriveRL，不把视觉学生的一次预测长度当成整个模拟世界时长。

| 时间概念 | 论文配置 | 解释 |
|---|---|---|
| 导出的训练日志片段 | 10 Hz，41 history + 200 future frames | 约 4 s 历史、20 s 未来，合计约 24 s；秒数是按帧数/频率换算，不是原始 nuPlan 整份日志长度 |
| 世界初始化 | 11 frames / 2 s | 初始化历史，与单次策略输入历史不同 |
| PPO rollout | 110 steps at 5 Hz | 名义推进预算 22 s；硬事件可提前终止，不表示每个有效 episode 都运行满 22 s |
| 策略输入历史 | t−0.8 s 到 t，5 frames at 5 Hz | 仅 0.8 s 历史跨度，不是回合时长 |
| nuPlan NR/R 闭环评测 | 每场 15 s | 实际报告评测的场景时域 |
| 视觉学生轨迹输出 | 20 steps at 5 Hz | 4 s 预测/教师监督时域，不是 DriveRL 交互上限 |

按常规含当前帧的计数，241 个 10 Hz 状态下采样为 121 个 5 Hz 状态，初始化 11 帧后剩 110 次推进；这说明约 24 s 数据、2 s 初始化与 22 s rollout 在数量上相容。精确起始索引和锚点时间仍需训练数据加载器核实，不能把这个推算当作源码验证；尤其不能宣称从日志锚点 t=0 开始有 22 s 已记录未来。原始整份日志长度和回合的理论最大可配置长度不是这些数值。

代码交叉核查基于官方 commit `24959547edac549d64c34b2aab62d5c52599d078`：[teacher release 配置](https://github.com/XiaomiAutoL3/DriveZero/blob/24959547edac549d64c34b2aab62d5c52599d078/DriveRL/release/configs/driverl_teacher.yaml) 与 selfplay 配置均保留 110 步和 5 Hz。不过公开范围为 [nuPlan 推理运行时](https://github.com/XiaomiAutoL3/DriveZero/blob/main/DriveRL/README.md)，不是完整训练 reset/loader；[TTS 代码](https://github.com/XiaomiAutoL3/DriveZero/blob/24959547edac549d64c34b2aab62d5c52599d078/DriveRL/src/driverl/nuplan/tts.py#L156-L168) 明确 110 也用于 reward normalization，实际搜索时域独立。因此训练时长主要依据论文，不能仅凭配置字段证明每个 episode 的完整实现。[nuPlan 配置](https://github.com/XiaomiAutoL3/DriveZero/blob/24959547edac549d64c34b2aab62d5c52599d078/DriveRL/src/driverl/nuplan/config.py#L57-L112) 的 80 poses × 0.1 s 是 8 s 输出轨迹，12 s 是路线前瞻参数，都不是评测回合长度；评测 tick 为 0.1 s，不应把整个系统统称为 5 Hz。

论文 §2.1.2 / A.4 规定 goal arrival 只产生一次事件/奖励，不终止 rollout；硬安全事件与时间预算另行处理。晚进入的日志对象有场景一致性门控（当前源帧存在、ego 距源位姿不超过 5 m / 0.35 rad，且不与可见对象碰撞），并非只保留全窗口始终存在的对象。两点都是延长本平台有效交互的直接参考；其特定 ego 门控不能直接视为多辆受控车共同偏离日志时的充分条件。

按用户修订，DriveRSI 应根据实际日志窗口、任务和后端支持选择时长，不统一到 DriveRL 的 15 或 22 s；上述时长仅作可选比较档。当前物理步长 0.1 s 可保留，无须为了模仿论文而先更改频率。历史缓存、源未来覆盖、episode 预算、后续训练采样段和标签切片分别配置、分别记账。到达中间导航锚点应与完成最终驾驶任务分开；仍须显式定义终止策略，不能直接删去终止条件。不等长场景均以共同观测/联合推进和配对因果响应验收，单纯动画时长不能作为通过证据。具体数据驱动准入规则见需求文档第 16.1 节，当前 9 s 后端支持上限未扩大。

NAVSIM 的约 6 s 样本不应被解释为日志通用上限：[官方加载器](https://github.com/autonomousvision/navsim/blob/main/navsim/common/dataloader.py) 从日志中按 SceneFilter 截取样本，[当前 navtrain 配置](https://github.com/autonomousvision/navsim/blob/main/navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml) 为 4 个历史/当前帧 + 10 个未来帧，说明帧数取决于具体配置。已裁剪样本的历史与未来要按决策锚点分别计算，整份日志则审计连续时间戳、身份和所需字段；不能给 NAVSIM、OpenScene 或 nuPlan 名称硬编码一个交互秒数。日志支撑时域以外的合成延伸是另一项能力，必须另行验证和标记。

### 高程与场景支持的实现参考

HUGSIM根据当前水平位置查询局部参考平面，并将车辆高程更新和图像生成分离：[ego实现](https://github.com/hyzhou404/HUGSIM/blob/main/sim/hugsim_env/envs/hug_sim.py)、[其他车辆实现](https://github.com/hyzhou404/HUGSIM/blob/main/sim/utils/plan.py)。其第一前相机OpenCV坐标及相机平面原点不能直接移植到DriveRSI的z-up/box-center。当前平台先实现同actor参考轨迹的有限高度差代理，不将它称为地面重建。

AlpaSim的[grounding模块](https://github.com/NVlabs/alpasim/blob/main/src/physics/alpasim_physics/backend.py)利用环境mesh与包围盒底面采样射线拟合接触平面，并记录点数不足、平移/旋转过大的失败；修正保留原XY，适合后续接入可信地面mesh。当前MTGS资产缺少已验证同坐标道路mesh，不能宣称已实现该贴地能力。

SimScale的[方法与数据筛选](https://arxiv.org/html/2511.23369v3#S2.SS2)区分静态背景、车辆轨迹与图像生成，并限制扰动与过滤重建质量；它不保证任意其他车辆机位都可靠。有限视角外观、缺失ego/行人及新视图黑块仍须独立资产补齐和质量验收。
