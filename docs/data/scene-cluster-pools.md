# 场景簇池与训练候选扩展

两个成员池共用仅由 navtrain 拟合的模型，navtest 只做变换和归属。不是分别聚类后按编号硬配对。该接口复用既有图像特征与历史结构化事实，不重新推理模型、生成caption或修改原始数据库。

## 当前语义边界

本版是“历史场景上下文分组”：速度、历史速度/航向变化、车辆/行人/自行车/静态对象数量、邻近对象数量、四向导航指令，结合4历史帧×3相机的既有中心向量。没有完整道路拓扑、7秒交互事件序列和因果归因，缺项明确保留未知。现有condition标签作为可重叠属性；condition标签组合子组不表示完整数值或时序相同。

数值事实使用log1p，缺失值置0并加独立缺失标记；导航指令0..3及未知各占一维。视觉向量以种子42投影到32维，各模态仅用训练侧均值/标准差归一化（标准差下限1e-4），截断至正负5，按维数缩放使两个模态拥有相同总权重。默认在训练侧8192个样本上拟合128个k-means中心、20次迭代，再对完整两个划分分配簇。99%训练分配平方距离阈值仅标注离群点，不删除或自动判为无效。

这不等同于复现某篇LLM聚类论文。没有LLM参与命名或重新标注，簇描述是训练侧已有事实统计。场景类别与学生失败指标分开；没有相关性标注时语义Recall为null。

## 缓存入口

```bash
.venv/bin/python -m driversi.data.catalog.centroid_cache \
  --catalog exp/scene-catalog-v1/navtrain.sqlite --split navtrain \
  --profile <existing-profile-id> --output exp/<run>/navtrain/cache \
  --scratch /dev/shm
```

navtest使用相同命令但独立catalog、split和output。SQLite backup生成一致节点本地快照后提取连续float32矩阵，保留补充向量集合身份、原图input_id、来源组和不规则历史标记。scratch可改成节点本地磁盘。进程正常结束时临时快照清理；缓存与清单保存到exp。初次构建仍有读盘成本。

同目录重跑校验来源签名和输出完整性后复用；源库或profile变化需新缓存版本。数据库非空WAL参与来源签名，空WAL不表示数据改变。中断未完成构建不会冒充完整缓存。固定快照不会自动包含之后新增合成样本，需显式生成下一版本。

## 聚类与映射入口

```bash
.venv/bin/python -m driversi.rsi.experiments.scene_cluster_mapping \
  --config exp/<run>/config.json --output exp/<run>/mapping
```

配置包含train_cache、test_cache、baseline_config、priority_queue，以及clusters、fit_samples、iterations、seed、probes、expansion_per_query、benchmark_queries。完整官方成员、相同编码器、无split交集、父模型全量评测与计划、0/<0.6完整失败集合都会检查。

输出：

- `model.npz` / `model.json`：训练侧模型、转换参数与语义范围。
- `navtrain/pool.json`、`navtest/pool.json`：独立成员与同一model_id。
- `navtrain/clusters.json`：训练侧代表、边界样本与事实统计。
- `private-mapping.json`：内部查询证据，不发布测试token。
- `production/training-production-queue.json`：只含训练token，供现有Harness选择器消费。
- `summary.json`：可公开的聚合计数及实验范围。
- `complete.json`：完成与恢复检查；已完成回执不覆盖。

固定生产配方探测8个相关簇，每查询扩充最多64个候选、扩充分支每来源组最多3个；并保留原始中心向量精确余弦检索的16个来源多样化候选（每来源组1个）。两路去重后最多80个/查询，某组最多4个。此兜底不等于旧的12视角精排top16。生产队列按候选排名轮流遍历各查询，再去重，避免先处理的查询占满预算。

## 速度与召回口径

64个固定随机查询只用于算法比较，8簇生产配方预先固定。Approximate centroid Recall@20对照同一全库原始中心向量精确余弦近邻，不是驾驶语义相关性、旧多视角精排召回或模型收益。比较4/8/16/32簇，记录均值、最差查询和扫描数量。当前计时`seconds_including_reference_sort`包括精确参考排序，不能用它冒充纯检索延迟；额外的query-only计时必须另行明确测量。

实际生产仍计算全库精确批量分数用于兜底，因此速度改善首先来自缓存和批量矩阵计算。场景簇用于控制扩展范围、组织样本和保留长尾；不能把它的价值混同于ANN必然更快。

## Harness与发布

`reactive_autoloop`支持`selection_mode=full_training_clusters`，其`mining_root`指向上述production目录。现有selector继续检查父模型、官方训练成员、资产库存、排除历史失败和预算，并记录场景簇选择角色。选择完成不等于启动教师或训练；PDMS、接触、GS视觉、语言监督等既有准入保持。

新试验通过`src/config/visualization/platform.json`登记聚合字段，复用统一工作台。不发布测试成员、测试描述、预测、私人路径或密钥。没有单独HTML模板。
