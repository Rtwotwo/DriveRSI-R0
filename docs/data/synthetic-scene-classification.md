# 合成数据的场景描述与簇归属

正式合成包由 `src/data/datasets/synthetic_publish.py` 发布完成后登记持久任务；Harness 的原生 features 后处理完成时，通过 `src/rsi/services/synthetic_classification.py` 在基础运行时执行标注和分类。AUGUstus 与 ChainFlow 共用此入口；AUGUstus统一实验程序中的直接publication_jobs路径也在验证发布后调用同一服务。未达到原生准入的包保留队列，可显式处理作诊断。没有启动自动监控。

## 数据和源码归属

- `src/data/catalog/synthetic_classification.py`：实际新锚点的四相机描述、状态/拓扑、冻结簇映射、JSONL发布及训练读取验证。
- `src/rsi/plugins/synthetic_classification.py`：`synthetic_scene_classification` Harness provider。
- `src/rsi/services/synthetic_classification.py`：受监督执行、队列补处理。
- `src/config/data/synthetic-scene-classification.json`：模型、图片合同、预算、地图和冻结层次路径。
- `scripts/data/classify_synthetic_scenes.py`：薄入口。
- `lib/cluster_caption/synthetic/jobs/`：每个生成包的请求、响应、地图证据和失败状态。
- `lib/cluster_caption/synthetic/records/*.jsonl`：每个生成场景的当前扩展记录。
- `lib/cluster_caption/synthetic/members.jsonl`、`pools.json`：合成库视图，始终不混写原始 navtrain/navtest。
- `exp/synthetic-scene-classification-v1/`：本轮验证，源图像和训练数据仍由原生导出包管理，不重复复制全套未来图像。

## 语义

`cluster_caption` 复用 `driversi.cluster_caption.v1` 四相机格式，split 明确为 `synthetic_navtrain`，保留实际父日志和官方来源成员。当前锚点为导出窗口第四帧；只提交这一时刻的图像、速度、导航指令。地图在实际新位置查询。不把导航指令说成已执行行为，不以源日志文本代替新场景描述，不把未来训练答案送入caption。

分类调用既有 scene_hierarchy 的 transform 和存储的二叉分裂中心，训练侧拟合的模型只读。没有读取 lib/retrieval 图像向量。输出粗、中、细、叶簇以及半径外提示、最近备选簇。簇名表示训练组特征，不代表新场景具备名字中的每项属性。

caption依然是待审核机器草稿，`indexable=false/trainable=false` 指文本没有成为正式语言监督。新场景可以作为探索聚类成员；原生训练资格独立保存在 `native_admission`。只有位于正式dataset/synthetic_navsim/samples的封存数据包，且有当前有效原生资格的记录标为 qualified；benchmark/诊断输出即使内部样本标为trainable也保持隔离。其余为 quarantine。`pools.json` 包含两者，不能直接当训练清单。训练消费者调用 `training_members(leaf=None)`，重新核验原生资格、实际图像/日志、文本证据和聚类版本；分类不能授予原生准入。原生轨迹、视觉、语言和模型loss均保持不变。

## 执行与恢复

```bash
.venv/bin/python scripts/data/classify_synthetic_scenes.py --export /absolute/export.json --sample-report /absolute/native-admission.json
.venv/bin/python scripts/data/classify_synthetic_scenes.py --drain --limit 16
```

不提供 sample-report 可作诊断，但不会进入训练读取。队列 drain 仅处理尚无结果的任务；失败需显式选job检查/重放。收到的回答复用，未知在途结果不自动重发，执行合同变更会拒绝。重新发布同一个包不会产生重复记录或重复API请求；补充原生资格可以重放并更新资格。

每次最多一次请求，全合成库共享请求/费用账本，初始上限1000次和按已有公开忙时单价估算30元。未用实际账户账单替代成本估计。当前串行请求，跨进程加锁保护预算；限速时间、429退避和认证阻断跨场景持久化，后续扩容需显式调整；无需修改原来的全量caption库。私有配置只引用既有授权配置名，密钥在运行时由私有文件读取，代码和产物不保存密钥。

## 当前验证边界

一个真实 Boston 生成分支已通过实际Harness任务完成caption、地图事实和簇分配，原始数据库未写入。该样本没有当前准入凭证，所以为隔离诊断。这个验证不代表新增合格训练样本或训练收益。
