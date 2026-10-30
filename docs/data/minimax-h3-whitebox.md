# MiniMax-H3 本地 Ref2VA 白膜提示词接口

本入口在已有白膜交付流程上增加 `minimax_h3_ref2va_v1` profile；保留原版 `generic_four_camera_v1`。适配的是官方 Ref2VA **提示词和本地请求格式**，不是官方托管 H3-Context-IR 的复现，也不表示已验证生成画面质量。

官方源码位于 `clones/MiniMax-H3`，用于参考和审查；DriveRSI 运行时不会从该目录 import。没有下载模型权重，也没有调用收费 API。提示词构建代码为 `src/rendering/whitebox/minimax_h3.py`。

## 原生规范

依据：[官方六段式规范](https://github.com/MiniMax-AI/MiniMax-H3/blob/main/skills/h3-prompt-writing/references/ref-en.txt)、[本地Ref2VA请求示例](https://github.com/MiniMax-AI/MiniMax-H3/blob/main/scripts/readme/reproducible-768p-ref2va-request.sh)、[输入输出规格](https://github.com/MiniMax-AI/MiniMax-H3#system-overview)。

每条原生 prompt 使用六个英文段落，顺序固定：

1. `subject_definitions`：当前相机视频、交通对象和道路结构的引用。
2. `summary`：声明 video editing，并指定 `<Video 1>`。
3. `retention_analysis`：保持运动、布局与相机；语义颜色和方块外观转为照片风格。
4. `detailed_description`：单镜头连续7秒；外观描述按当前相机的证据筛选。
5. `overall_soundscape`：请求静音，不虚构鸣笛或语音。
6. `non_diegetic_music`：无背景音乐。

静音是文字指令，不是已验证的原生禁音开关；原生模型联合生成音视频。位置、速度、朝向、对象数量和遮挡关系以白膜视频为约束，VLM只提供已审核的环境与材质线索；没有相机对应的审核描述时明确保留未知，不借用其他视角的地标。包含多个相机证据的混合描述整条暂扣，必须先拆为有单相机证据的条目；不会用“任一证据匹配”把全部地标复制到每个视角。内部JSON保存暂扣原因。

原生输入最多三个参考视频且总时长最多15秒。四路7秒白膜不能一起当作一个请求输入，也不能假设一次返回四路。因此本适配器产生**四个独立任务**，每次只绑定当前相机视频到 `<Video 1>`。默认seed=0，可在配置中显式更改；共享源时间轴、外观约定和seed不能保证生成结果跨相机一致，仍需验收。

## JSON与下载包

内部JSON格式固化在 `src/config/rendering/minimax_h3_condition.schema.json`；它约束本项目使用的单视频、7秒子集，不声称覆盖H3所有模式。

内部每场景保存：

- `h3/condition.json`：`driversi.minimax_h3_whitebox.v1`；包含源场景、颜色语义、结构化轨迹/地图引用、相机、审核证据、时间轴和四个任务。
- `h3/CAM_*.request.json`：可交给官方示例采用的本地 SGLang `/v1/videos` 接口。只含 `task`、`prompt`、`conditions`、`target`、`seed`。
- `receipt.json`、原始运动审计等：仍在实验目录。

下载包依然只有四个独立相机MP4与一个 `video-generation-prompt.txt`。TXT汇总四段带起止标记的任务提示词，**每次只复制当前相机的一段六节原生prompt，不要整份TXT一次提交**。内部JSON不放进下载包。

`condition.json` 是 DriveRSI 元数据，不能整份作为H3请求。相机、对象ID、轨迹等结构化信息通过白膜像素约束H3；模型不会因为存在本地JSON就自动读取几何。已有检索标签和历史时间范围保持不变，视频全片描述不进入驾驶模型的当前观测。

## 时间约定

输入仍是7秒、10fps、70帧、640×360；它来自约2Hz日志离线插值。内部71个状态含7秒端点。原生目标是7秒、24fps、短边768、16:9，不添加未经上游确认的 `fps=10` 请求字段。

生成后按真实输出PTS与源时间戳对齐；不能简单把24fps重标成10fps，也不能用原帧号直接对应。输入分辨率不是生成分辨率，几何验收需要按实际输出尺寸缩放投影。原生实现可能对帧数做时域对齐，必须检查实际时长和首末帧，不预先声称所有输出恰有某一帧数。

## 执行

沿用同一个可恢复入口，使用H3配置：

```bash
.venv/bin/python scripts/data/run_whitebox_delivery.py \
  --config src/config/rendering/whitebox_delivery_h3.json \
  --batch-file exp/whitebox-delivery-interface-v1/full-plan/batch-00000.json \
  --workers 3
```

此命令只完成条件导出、白膜渲染/复用和提示词打包，不调用H3生成。没有外观审核时等待，不自动增加VLM费用。原配置和原产物不覆盖；换profile时使用新output。

真实六场景复用示例：

```bash
.venv/bin/python scripts/data/run_whitebox_delivery.py \
  --config exp/minimax-h3-prompts-v1/six-sample-config-v2.json --workers 3
```

用户已有的本地 SGLang Ref2VA 服务可显式提交单相机请求，例如：

```bash
curl --fail-with-body --request POST http://127.0.0.1:30011/v1/videos \
  --header 'Content-Type: application/json' \
  --data-binary @exp/minimax-h3-prompts-v1/deliveries-v2/86f4b571f4ae5e3f/h3/CAM_F0.request.json
```

`conditions[].uri` 是 `file:///` 绝对路径，由**运行模型的服务进程**读取；如果服务在另一台主机或容器，先挂载/复制视频并显式改为该服务可见路径。不要把本机文件路径当成对远端自动上传。

## 验收与训练用途

本轮检查的是原始白膜运动、视频与prompt/JSON绑定及跨进程复用；没有运行H3权重，不能声称图像更好或多视角已一致。后续实际生成需检查对象数量/位置、道路内车辆、遮挡、时序、相机方向及跨视角外观。

MiniMax H3公开[许可第V.3条](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)限制使用H3或其输出改进其他AI模型。因此不能仅凭格式和画质合格就将其输出用于ChainFlow训练；需要确认适用授权。此处只保存提示词与输入条件，生成结果的 `training_admitted` 保持false。
