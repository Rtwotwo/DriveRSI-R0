# Synthetic training storage without retrieval processing

The active dataset can defer feature extraction, scene captions and clustering while preserving native training admission. Canonical packages remain under `dataset/synthetic_navsim/samples/`; admitted SimScale/OpenScene views remain under `dataset/navsim_logs`, `sensor_blobs`, `sensor_blobs_hist`, `sensor_blobs_fut` and `manifests` in the synthetic split. They are not added to navtrain.

`dataset/postprocessing.json` is a campaign-scoped scheduling policy (`mode: training_only`). It does not change `collection.json`, native models, losses, trajectory thresholds or visual admission. A matching production config resolves this policy even when its prior config is immutable. Independent quarantine benchmarks retain their default mode. New curriculum rounds explicitly record `postprocessing_mode: training_only`.

The publisher stores a version-2 `admitted_synthetic_training_reference` with `trainable: true`, `indexable: false` and explicit deferred retrieval fields. No feature report is fabricated. Native sample validation, all exported file hashes, anchor identity, log/sensor path binding and the current trajectory metric remain required. All history/future pixels are checked, including images outside the anchor-only policy input.

The resident renderer keeps its two GPU lanes and native exporter but does not require a feature worker or feature GPU in this mode. It does not enqueue classification. Existing feature/caption receipts are retained. Native training supervision remains required; this mode defers retrieval descriptions rather than model targets.

`driversi.rsi.experiments.processing_resume` provides a one-use recovery after an explicitly recorded processing-change interruption. It requires every physical job and native receipt to be terminal and verified, owns the curriculum lock, archives prior statuses and refuses unknown/in-flight work. Completed physical jobs reuse their receipts without changing original timing/PID evidence. It then continues the same bounded campaign. It does not create a recurring monitor.

Current transition evidence: `exp/continual-family-pdms-v1/adaptive-problem-27db-v1/protocol-revisions/training-only-endpoint-v1/implementation-ledger.json`.
