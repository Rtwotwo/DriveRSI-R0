# Benchmark failure to training-cluster mapping

Maintained entry points: `src/rsi/experiments/cluster_failure_mapping.py` and `cluster_retrieval_overlap.py`. Run from the repository with `.venv/bin/python -m driversi.rsi.experiments.cluster_failure_mapping --config exp/<run>/config.json`; compare saved methods with the analogous `cluster_retrieval_overlap` module.

Required config: cluster_root (canonical text/state/map hierarchy), score_csv, score_summary, checkpoint_sha256, output; threads defaults8. Comparison adds comparison_mapping (saved legacy mapping folder) and optional requested_comparison_count for audit. Do not point to image-vector caches.

Only complete normalized NAVSIM_v1_PDMS navtest rows are accepted. Select score==0 and0<score<0.6; require matching checkpoint, score mean, source logs and official train/test populations. Use frozen primary leaf plus three closest training leaf centers, retain uncertainty, and rank at most20 training scenes per query with at most2 per source_log. No refit, paid caption call, student scoring or admission changes.

Timing separately measures loading/integrity, stored-ID lookup, routing, nearby centers, pool expansion and ranking. Complete query stages are timed together three times; stages also have repeats. File-system cache is uncontrolled; representation and original clustering costs are outside scope. Module total excludes interpreter startup and final summary serialization.

Overlap compares saved candidate unions AND per-query agreement. Pure visual Top16 and historical visual8+structured-text8 RRF union are distinct. The latter can contain15 rather than16 due deduplication. Both-capped16 is a diagnostic, not equal-cardinality validation. Set agreement is not semantic precision or recall.

Register aggregate cluster-focus.json and optional overlap/summary.json under `clustering.failure_mapping` / `clustering.overlap_summary` in src/config/visualization/platform.json. Fixed view=clusters filters zero/low/expanded groups. Only training examples and aggregate test counts are projected. The GitHub Pages closed-site workflow must remain closed until user explicitly reopens it.

A verified run lives at exp/chainflow9405-cluster-mapping-v1/report.md. Captions remain machine drafts and candidates only express related scene conditions; they do not prove equal failure causes or training value.
