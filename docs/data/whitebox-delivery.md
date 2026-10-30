# 白膜视频与提示词批处理接口

本接口固化已展示的六份真实 navtrain 样例：原日志 → 结构化白膜 → 原始运动审计 → 四相机视频 → 已审核外观描述 → 五文件下载包。此入口不调用 VLM，不读取 API 密钥，不启动训练，不上传 GitHub。

## 固定交付约定

每个场景的 ZIP 仅有 `CAM_F0.mp4`、`CAM_L0.mp4`、`CAM_R0.mp4`、`CAM_B0.mp4`、`video-generation-prompt.txt`。前、左、右、后视角分别存储，640×360，10 fps，70 帧，播放区间 `[0,7)`。包含 7 秒端点的 71 个结构化状态、284 张 PNG、地图、审计和回执都留在内部目录，不进入下载包。

10 Hz 是约 2 Hz 日志的离线插值，不是新的传感器实测；白膜道路为平面代理，立方体遮挡不是像素级真值。入口复用原日志运动审计，不修改对象运动、删除行人或补造缺失观测。当前入口只接受原始 navtrain 日志；反事实分支要使用其物理执行和恢复验收，不能冒用本入口的原日志审计。

提示词模板固定在 `src/rendering/whitebox/delivery_prompt.py`。几何、运动和方位由四路白膜视频约束；经过审核的 VLM 文本只补充环境、材料和光照。语义颜色不是车漆，步行道颜色不是草地，粗类别不能推断车辆具体型号。此产物是视频生成条件，不表示生成的视频已经合格或获准训练。

## 执行环境与配置

在 32341 的 `<repository>` 执行。正式配置为 `src/config/rendering/whitebox_delivery.json`，字段均使用绝对路径：

- `database`：只读 navtrain 场景库。
- `maps_root`：源地图。
- `output`：产物根目录，默认 `exp/whitebox-delivery-v1`。
- `reviews_root`：按 `<token>.json` 保存的外观审核结果。
- `renderer_site_packages`：已验证的 QuickJS、Pillow、ffmpeg 环境路径。

外观文件必须包含匹配的 `token`、`status`、`claims`。允许的状态是 `assistant_reviewed_subset_for_generation_trial` 或 `human_reviewed_for_generation_trial`。每条 claim 包含 `text` 和真实来源图片的 `evidence_ids`。状态由实际审核产生；不得为了跑通批处理而把未检查的 VLM 标签改为已审核。审核只是外观文本的审核，不能替代生成视频质量验证。

```json
{
  "token": "实际navtrain token",
  "status": "human_reviewed_for_generation_trial",
  "claims": [{"text": "经过图片核对的静态环境描述", "evidence_ids": ["来源图片标识"]}]
}
```

## 全量计划与有界执行

先创建清单。该命令只读取索引并输出分片，不渲染、不调用模型：

```bash
.venv/bin/python scripts/data/export_semantic_conditions.py plan \
  --database exp/scene-catalog-v1/navtrain.sqlite \
  --output exp/whitebox-plan-v1 --batch-size 128
```

再显式指定一个分片：

```bash
.venv/bin/python scripts/data/run_whitebox_delivery.py \
  --config src/config/rendering/whitebox_delivery.json \
  --batch-file exp/whitebox-plan-v1/batch-00000.json --workers 3
```

每批 1–128 个唯一 token，1–4 个 CPU worker。不同分片可共用 output；同一 token 有进程锁。**本轮只核对全量清单，未运行 103k 全量渲染，也未验证全量吞吐/磁盘容量。** 后续扩大前应先核对地图/图片覆盖、待审核比例、单场景耗时与磁盘预算，不要直接把所有分片同时提交。

使用完全相同的命令即可跨进程续跑：完整数据、审计、渲染和下载包检查通过后复用，已有 ZIP 不重写。中断且未完成的导出/渲染目录保留为 `.incomplete-*`，仅该未完成阶段重新执行；目前不承诺逐帧续跑。已有完整产物若被修改则拒绝，使用新输出版本重新生成，不静默覆盖。

缺少审核时记录 `waiting_for_appearance_review`，补齐审核后重跑即可。所有场景都有 `receipt.json` 和追加式 `attempts.jsonl`；分片报告的 `complete` 只表示所有场景处理结束，必须检查 `all_delivered`。退出码 0 表示全部交付；2 表示有失败、待审核或忙碌场景。失败会保留并继续处理其他场景。

## 复用已有样例

配置可带 `items`，每项为 `token` 与可选的 `bundle`、`review` 绝对路径。`bundle` 用于复用已有完整、封存的原始日志白膜及渲染；入口不会修改外部 bundle，外部产物缺失时直接拒绝。

```bash
.venv/bin/python scripts/data/run_whitebox_delivery.py \
  --config exp/whitebox-delivery-interface-v1/six-sample-config.json --workers 3
```

源文件和每帧渲染哈希、四路 MP4 实际解码的帧数/尺寸/帧率/时长、下载包内容都检查。已符合当前坐标约定的产物再次执行应为 `reused`、`rendered=false`、`exported=false`、`new_api_calls=0`。上述历史六样例配置仅保留复现用途；旧坐标产物会被当前批处理拒绝，需要选择新输出目录重新导出，不能据旧回执宣布已修复。

## 后续全量标注边界

此前 caption 接口可独立生成候选文本；本入口消费经过审核的结果，不把“JSON 格式通过”当作外观正确。后续若启用批量模型标注，仍仅从 `/path/to/private` 按需读取密钥，成本与调用记录留在私有实验目录。视频生成文本可使用整段内容；检索历史标签必须保持自身时间范围，不能把未来描述注入当前驾驶输入。

六份现有公开演示与下载链接保持不变。本入口只写实验产物，不自动部署页面，也不自动上传仓库。


MiniMax-H3 官方本地 Ref2VA 使用独立的 `minimax_h3_ref2va_v1` profile，生成六段式英文提示词和每相机请求JSON，保持原五文件下载约定。见[H3适配说明](minimax-h3-whitebox.md)。

## 三维记录与平面条件视频（坐标修复）

新导出的 `source.json` 使用 `openscene_center_se3_scalar_yaw_v2`：物体中心按完整 ego 旋转恢复，物体航向按 OpenScene 的标量 yaw 定义恢复；相机 legacy `sensor2lidar` 字段实际是 camera-to-ego，不能重复乘 LiDAR 外参。原始实测 Z、pitch、roll 均保存在结构化数据中。

新视频默认 `planar_consistent_v1` 显示投影：道路统一平面，对象底面落在该平面，相机保留同一安装外参并沿日志 XY/yaw 运动。它没有修改真实运动记录，但不表达坡度、路缘高差、立交或真实三维遮挡，不能作为 GS 高程配准证明。`measured_world_v2` 可用于原姿态诊断；若没有独立道路高程，平面地图与实测物体不能当作完整三维真值。旧产物保留旧约定，不静默重写；源记录审计通过不代表旧渲染的地面一致。

修复依据：[OpenScene 原始标注生成](https://github.com/OpenDriveLab/OpenScene/blob/main/DriveEngine/process_data/create_openscene_metadata.py) 与[原始相机外参](https://github.com/OpenDriveLab/OpenScene/blob/main/DriveEngine/process_data/helpers/nuplan_cameras_utils.py)。

维护模块职责见 [src/rendering/whitebox/README.md](../../src/rendering/whitebox/README.md)。源码与回归测试进入版本管理，实验视频与验收回执保存在 `exp/`。

### Source box temporal sampling

`source_bracket_linear_v1` interpolates source box sizes on the same bounded adjacent-observation bracket as center and heading. Original timestamps retain their original dimensions; identity gaps are not bridged and new identities are never introduced early. Intermediate sizes are annotation interpolation, not measured body deformation. Executed branches use `executed_physics_unmodified`. Old `previous_annotation_hold_v1` exports remain auditable, but the current delivery runner requires a new export instead of reusing those stepped silhouettes. Source and render receipts carry the sampling convention. This addresses source-video shape discontinuities and does not certify GS alignment, road height, visibility, or training eligibility.
