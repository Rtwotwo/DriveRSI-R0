# AutoDrive-P3 聚类描述扩展库

维护代码：`src/data/catalog/cluster_caption.py` 与 `cluster_caption_batch.py`；薄入口：`scripts/data/annotate_cluster_captions.py`。提示词与公开运行参数在 `src/config/data/scene-labels/cluster-caption*`。凭据继续从已有私有目录运行时读取，不写入这些文件。

## 数据位置与用途

本轮库为 `<repository>/lib/cluster_caption/navtrain/`。

- `autodrive-p3.original.json`：原始训练标注的逐字节副本。
- `autodrive-p3.with-cluster-caption.json`：原键和原字段保持不变，仅新增 `cluster_caption`。在初始化、小批检查点和全量结束时导出；运行期间以数据库和分片为准。
- `captions.sqlite3`：逐场景状态、描述、相机证据引用、回执及失败原因，持续更新。
- `shards/batch-*.jsonl`：完成批次的完整扩展记录，可增量消费。
- `progress.json`：累计完成、失败、剩余数量与保守费用用量。

只处理官方完整103288条navtrain成员；不读取navtest明细、学生分数。每个场景只发送当前锚点四路源图、当前速度及导航指令，照片长边768，JPEG质量80。所有原生prediction/planning等监督完整保留，但不发送未来标签让VLM倒推画面。

新描述推荐150–300中文字，硬格式范围100–400字、4–10项相机绑定属性、3–12个检索标签及最多4项未知。道路拓扑、运动事件不由静态图片臆测。格式合格存为`machine_draft`，不等于事实已审核；不会自动替换训练监督或发布到正式检索库。

## 执行与恢复

```bash
.venv/bin/python scripts/data/annotate_cluster_captions.py initialize \
  --index <official-navtrain-index.json> --annotations <original-train.json> \
  --library lib/cluster_caption/navtrain

.venv/bin/python scripts/data/annotate_cluster_captions.py run \
  --library lib/cluster_caption/navtrain --output exp/cluster-caption-batch-v1/campaign \
  --private-config <private-service-config> --credential-file <private-env> \
  --workers 8
```

首次以`--workers 1 --limit-batches 1`运行16个不同日志的试点。完整任务复用这16个已完成结果；不重新收费。八个批次各最多八路调用，合计最多64路；共享限速和消费控制，不是八份独立额度。按2500输入+1800最大输出预留吞吐，1M TPM下理论上限约233次/分钟，全量单纯限速下限约7.4小时，实际还受API延迟与文件IO影响。

费用按北京忙时公开标价每百万输入2元、输出8元计算，仅用于保守估算而非私有账户账单。运行前有1600元估算停止阈值；每个在途调用保留16384输入+1800输出的费用。只有明确的401/403/429拒绝估为零，网络和5xx未知结果保留预留额。实际输入超过吞吐预留量时停止核查，不能宣称绝对账单上限。

结果不确定的已发送请求不会自动重发；仅429最多两次真实尝试，跨进程仍计算总数。调度器确认尚未发送的取消请求保持pending，恢复时可重新发送，旧回执保留。收到回复后即使解析失败，也保留原回复而不重新请求。

可创建运行目录的`STOP`文件或发送SIGTERM进行有记录的暂停；恢复前移除`STOP`并执行同一命令。代码、提示词或库版本改变会拒绝混用，应先检查已有回执，不能直接建立新输出目录重跑全部付费任务。

## 验证范围

当前流程验证的是副本不破坏原字段、当前图像输入边界、并发预算和恢复，以及小批描述质量。没有完成全库事实审核、聚类效果/Recall评测或模型收益验证。缺图、未知和错误必须在数量与记录中保留，不能静默删去后声称全量成功。


## 离线修复与一次性收尾

`python scripts/data/repair_cluster_captions.py --mode prepare --library lib/cluster_caption/navtrain --output exp/cluster-caption-offline-repair-v1/initial`只读取已保存回答，生成独立修复文件，不写正在生产的数据库。按相同语义维度及确定性合并属性，保留每段原文及来源索引；跨相机合并会给每段加相机ID，原有长度、条数与证据校验全部保留。不能无损合并的继续失败，不发新请求。

`--mode finalize --campaign exp/cluster-caption-batch-v1/campaign --wait-for-pid <producer-pid>`只等待该进程的一次退出事件，随后核对完成状态、获取生产锁并执行修复/导出/核验。它不是定时监控，不重启生产器、不调用模型、不发送通知。生产未完成则停止并记录阻塞。`--mode cost`从现存响应时间与token核算北京峰谷公开价，未知请求单列，不能代替账户账单。

最终结果在`lib/cluster_caption/navtrain/finalization.json`以及指定exp目录的`completion-audit.json`。原API回执与生产progress不改写；应结合finalization查看修复后的数量。核验涵盖103288原始成员、数据库/分片/扩展JSON覆盖和原生字段等值。格式修复不等于事实审核，描述继续不可自动进入正式检索或原生训练目标。

## Private navtest continuation

`python scripts/data/annotate_caption_partition.py initialize --index <complete-navtest-index> --train-library lib/cluster_caption/navtrain --library lib/cluster_caption/navtest --output exp/<test-run>` validates12146members, current four-camera files and disjoint tokens. It constructs only current-anchor NAVSIM metadata; solution, historical trajectory and video path are null. It does not invent AutoDrive-P3 test labels. The original current metadata and caption extension use navsim.original.json and navsim.with-cluster-caption.json. The cluster_caption schema is identical to train; split and native-label availability are explicit.

`run` additionally requires --train-audit, --private-config, --credential-file and --policy. Private credentials are read only at runtime by the existing client. The test run's contract.json references the frozen training campaign profile. All policy settings except lower request/cost caps must match; prompt, model and image preprocessing stay unchanged.

The gate requires a terminal103288-member audit with no missing/duplicate/pending records, original source/native fields unchanged and reconciled received receipts. Recorded unresolved failures are preserved and may not exceed1%. It checks only technical completion, not factual caption correctness. No model response of unknown outcome is automatically retried.

The run is one ordinary batch process followed by offline repair and full12146-member verification. There is no heartbeat, polling monitor, notification, or automatic restart. Results remain private, unreviewed, non-indexable and non-trainable. The default test preset reduces the request cap to13000 and the public busy-equivalent guard to200CNY while retaining8x8calls and1MTPM. API fees are estimates, not the provider invoice.
