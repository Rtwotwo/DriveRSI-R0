# 已保存检索结果的地理对照与私有图像核查

实现入口：`src/rsi/experiments/retrieval_geography.py`；固定平台组件：
`src/visualization/retrieval_comparison.py` 与 `static/retrieval-comparison.js`。

## 数据与含义

复用已冻结的527个ChainFlow stage1 9405零分/非零低于0.6查询，
读取已保存的视觉、旧融合和文本/状态/拓扑聚类候选，均截取原排名前16。
不重跑检索、不运行navtrain学生评分、不调用VLM、不修改原始caption。
旧融合有部分查询仅15个候选；旧方法来源多样化约束未重排统一，因而属于现有生产输出对照。

用冻结成员清单的token和源日志定位真实NAVSIM帧，检查身份唯一，
提取`ego2global_translation[:2]`及原生wxyz四元数航向。
读取对应nuPlan `map.gpkg` 的 `projectedCoordSystem`，要求投影坐标且轴单位为米。
只有相同map名称才计算平面直线距离与环绕角差，跨map返回null。
这些坐标表示地图位置，不是邮政地址、道路测地距离或楼层相同证明。

记录Top1位置距离、Top16内至少一个候选落入25/50/100/500/1000米范围的查询数，
另加航向差不超过30度。指标分母包括未命中和跨地图的查询。
同地图距离中位数仅在同地图子集计算，须同时报告该子集大小。
全训练库最近距离只作为地理可用性参照，不是语义检索基线。
地理接近不能证明相同交互或相同失败原因；不得命名为语义Precision/Recall。

## 运行

```bash
.venv/bin/python -m driversi.rsi.experiments.retrieval_geography \
  --config exp/<run>/config.json
.venv/bin/python -m driversi.rsi.experiments.retrieval_geographic_recall \
  --result exp/<run>
.venv/bin/python -m driversi.visualization.retrieval_comparison \
  --project <repository> --result exp/<run>
.venv/bin/python -m driversi.visualization build \
  --config src/config/visualization/platform.json
```

配置包括绝对`project`、`output`和`workers`。输出保存摘要、地图版本、来源哈希、
私有坐标缓存与逐对距离；缓存复用要求代码、索引及原日志内容不变。
首次按源日志顺序读取，避免在大SQLite库内按token排序产生随机磁盘读。

## 图像与隐私

固定选择每城市两例：视觉Top1地理距离的中位案例，以及聚类/视觉Top1地理差异最大例。
因此8例是说明性展示，不是随机准确率样本。每例可切换原Top1和前16中地理最近候选，
后者必须标出原排名。相机均来自各自实际锚点；缩略图等比例，不裁剪、拼造或调用图像生成。
保留图片内容哈希和token/时间/相机来源，caption仍标为未经事实认证的机器草稿。

所有逐测试场景资料留在`exp/<run>/private/`，不写入发布仓库或公开clusters.json。
仅已绑定127.0.0.1的预览服务可通过显式私有目录链接提供素材。
平台前端只在localhost/loopback主机请求该数据。前端主机检查不是安全认证，
真正边界为loopback监听、SSH隧道、与公开发布树隔离，以及现有Pages关闭流程。
禁止把该私有目录挂载到公网服务。此模块不会恢复自动监控或公网发布。

## 全库地理Recall与交集

用户指定以日志位置衡量后，新增精确地理真值计算：每个测试查询在103288个训练锚点中，
筛选同地图、给定半径内（25/50/100/500m），可另限制朝向差≤30°。
默认展示100m＋30°，其他阈值同时输出，避免只报告单一阈值。
对每个查询分别取视觉Top16、聚类Top16，以及两者交集和并集；
全局去重token交集不可代替逐查询交集。

Geographic Recall = 命中相关锚点数 / 该查询在全训练库的地理相关锚点总数。
逐查询均值为macro，所有命中对/所有相关对为micro，分别保存。
零真值查询从macro Recall中排除并显式计数；Hit以全部527查询为分母。
返回对的合格比例为micro Precision；逐查询Precision对空返回记0，单独输出。
交集可能更纯但覆盖更低，并集最多32条，不能当作同预算Top16融合。
计算并保留逐查询集合包含关系校验，源数据、原排名不修改。
地理真值是空间条件定义，并未认证车道、楼层、交通动态或失败原因相同。
