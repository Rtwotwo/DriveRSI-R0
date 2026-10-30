# Maintained source organization

Apply to new and edited maintained source beneath src/. Preserve native upstream code, license notices and pinned metadata under vendor/; do not rename its public import API for style.

- Long-lived behavior lives in its owning domain: rsi/{services,workflow,decisions,experiments}, data/, sim/, training/, evaluation/, rendering/, visualization/.
- Python modules use lowercase snake_case and concrete responsibilities. Avoid experiment IDs, dates, vague temporary names or result claims in module names. Protocol version identifiers are allowed when required for compatibility.
- src/visualization owns the single application shell, navigation, aggregate projection and shared view components. New runs register data; do not generate another dashboard template per experiment.
- Config presets use lowercase kebab-case and a role directory. RSI presets must be recorded in config/rsi/catalog.json with purpose and preparation status. Historical examples belong in examples/. No absolute secret paths or API keys in public configuration.
- Experiment outputs, logs, checkpoints and generated reports belong under exp/<run-id>/, not maintained source or config/. Use source-backed runtime parameters rather than copying an old experiment folder.
- scripts/<domain>/ contains thin verb_noun.py entry points; move reusable logic into src/<domain>/. Filesystem script aliases have been retired; use the canonical command catalog. Python module compatibility for serialized objects remains separate.
- agents/ is reserved for supported model implementations; rendering/data SDKs belong in the owned adapters or pinned vendor tree.
- View projections may export allowlisted aggregate metrics, curated repo-relative preset identifiers and explicitly published training/engineering replay assets. Never publish private machine paths, credentials, test tokens, predictions or per-test captions.
