# 主视觉检索集合

主数据库仍为各自隔离的 navtrain.sqlite 与 navtest.sqlite。原始特征和不规则历史补充特征使用不同的存储 profile，不复制或改写原始向量。

`metadata.visual_retrieval_collection` 持久注册同编码器的基础与补充配置。`Catalog(..., readonly=True)` 自动启用连接级合并视图，后续 VectorStore、混合检索以及批量检索默认获得完整集合。`main.vectors` 始终保留真实存储记录，写入与提取入口不使用合并视图。直接用 sqlite3 访问原始表的审计程序，应区分基础配置数量与合并集合覆盖。

注册由 `driversi.data.catalog.retrieval_collection.register_collection` 完成：逐条检查来源 split、历史时间、图像证据、维度、完整有限向量及原始历史失败记录。重复注册必须完全一致。训练和测试分别注册，没有跨库拷贝或暴露测试明细。

本次补充的是原来已经提取好的 244 条训练和 15 条测试视觉向量。原图存在，历史时间间隔不规则；没有视频生图、插值、重复帧、修改时间戳或新模型推理。用于视觉检索时比较真实图像的顺序位置，不能将其解释为等间隔时序匹配。检索结果保留实际存储 profile 和 observed_irregular_history；它不授予时序训练或合成资格。

旧的 64 查询检索回执仍对应原先 103044 条基础向量，不追改或冒称使用了新集合。新打开的检索连接会包括 244 条补充向量；在途进程和旧结果不会被此次注册强制重启或覆盖。使用新集合重新执行实验，应建立新运行目录并记录注册身份。

验收记录：`exp/vector-main-collection-v1/registration.json`。字节级核对全部 259 条补充向量；基础失败记录和 scene.history.continuous=false 保持不变。
