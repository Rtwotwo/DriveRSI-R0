# 从官方 AUGUstus 标注派生检索描述

入口：`scripts/data/describe_annotations.py`，实现：`src/data/catalog/annotation_description_jobs.py`。
输出为 `driversi.annotation_description.v1`，单独保存，不写回官方 `solution_prediction`。

## 输入与范围
原始 navtrain 成员索引、官方 P3 标注、原日志和地图。当前为最多六个不同来源日志的试点，左右转/直行导航指令各最多两个；不是代表性抽样，也不依赖学生模型评分。
源日志要求完整的锚点前2秒、后5秒原采样；prediction标签仍保持原始4秒范围。
地图描述是采样点落入多边形的关系，不是完整车道拓扑匹配或变道分类。
对象类别与未来方位目前仅保留各自的计数，不能将它们交叉配对成逐对象交互事实。原始标注仍保留框与对应标签；此试点没有完成其对象关联派生。

## 两种文本
- template_text_zh：根据官方类别、源状态数值和地图区域关系确定性拼接。
- 可选语言整理：受限的短中文描述，最多三次请求、每次一个场景，只发送精简事实，不再发送逐点坐标或重复照片。新视觉内容不能在这里获得证据。

发布配置：
```bash
.venv/bin/python scripts/data/describe_annotations.py \
  --index <official-navtrain-index.json> --annotations <official-train.json> \
  --maps <map-root> --output exp/<pilot>/prepared --limit 6
```
只有显式 `--execute --private-config <private-config> --credential-file <private-env>` 才调用既有私有API。安全配置目录与权限由现有客户端校验；命令行不传密钥。
断点复用准备结果和API回执。改变输入或代码必须另用输出目录，不能把旧示范当作当前结果。调用完成不等于事实审核通过，所有新描述初始 `indexable=false`、`trainable=false`。
每条文本请求的精确提示词与事实保存到 `api/prepared/`，与回复分开留存，不包含凭据。

## 原生训练兼容性的意义
原始官方预测文本通过AUGUstus原生训练加载器生成语言token监督，同时配套四帧三相机视频和8x3未来动作目标。新检索描述不是原生目标，不应直接替换它。
小样本验证覆盖解码、预处理与监督字段，不证明全部标注正确，也不等于新增优化器更新或训练收益。

## 本轮实际试点
`exp/scene-annotation-reuse-pilot-v1/pilot-v2/` 保存六份日志的事实与模板，`native-validation.json` 保存六条原生加载验证。三个真实文本请求平均约640输入、135输出tokens；缓存恢复三条全部复用，新增请求为零。尚未测量检索/聚类收益。
复核后提示词增加了独立直方图不能交叉配对的约束；已执行请求的原提示词完整留存，未将它们重标为新提示词结果，也没有为此重复付费调用。

## 当前帧多视角补充
用户后续指定减少历史图像后，新增 `src/data/catalog/annotation_anchor_views.py`，在已验证的训练成员中任选一个token，提取当前锚点四相机图片（长边768、JPEG80），只附当前速度和导航指令，不发送未来预测标签或历史/未来图片。视觉观察与离线全片日志运动保持分层。
```bash
.venv/bin/python -m driversi.data.catalog.annotation_anchor_views \
  --prepared exp/<pilot>/prepared --token <training-token> --output exp/<pilot>/anchor-view
```
显式追加同样的私有配置参数和 `--execute` 才执行最多一次请求。输出保留最多五项带相机证据的静态特征和120字内摘要；单帧不能证明切入/横穿/变道等时序过程。

## 当前未完成
完整道路连接/车道关联、时序交互事件识别、历史专用描述、全量聚类及检索效果对照。不能从导航指令推断实际变道，从未来方位类别推断切入，或把未标注对象当成空路。

