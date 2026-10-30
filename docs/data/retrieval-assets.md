# 检索资产管理

原始检索库统一位于 `lib/retrieval/navtrain/catalog.sqlite` 与 `lib/retrieval/navtest/catalog.sqlite`。两库独立；244/15条历史不规则样本的已恢复原图向量仍保留其标记，不因此获得时序训练资格。

同目录 `centroids-v1/` 保存连续的场景中心矩阵和成员清单，用于粗排；原库中的多视角向量用于精排。`assets.json` 记录数量、集合身份与矩阵哈希。`location-binding.json` 绑定迁移后的数据库签名；数据库修改后应生成新版本缓存，不覆盖旧清单。

`src/config/data/retrieval_assets.json` 是部署路径声明。数据库、矩阵、生成向量及私有成员清单不入Git。新合成图像的特征保存在 `lib/retrieval/generated/` 的独立实验分区，只有有效原生准入和新像素编码才可检索。测试图像、预测及标签不进入合成训练库。

历史 `exp/scene-catalog-v1/*.sqlite` 与 `exp/scene-cluster-pools-v1/*/cache` 仅为兼容软链接；新配置使用正式路径。迁移不重新提取特征，不复用旧模型的失败查询选择。
