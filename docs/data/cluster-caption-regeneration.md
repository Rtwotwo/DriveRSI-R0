# Failed caption regeneration

`python scripts/data/regenerate_failed_captions.py inventory|run` accepts paired train/test libraries and original producer campaigns, an experiment output directory, and (run only) private configuration/credential file paths.

This command requires explicit user authorization to regenerate failed paid requests. Inventory freezes only response_failed rows after original production and audits have completed. It excludes all previously successful rows. It preserves the original prompt, model, provider, four current camera images and decoder. A run is bounded to three rounds and 20 CNY public busy-price reservation; systemic failures or authentication rejection stop execution. Unknown prior outcomes remain recorded and may have incurred charges.

Each new response is first decoded, then any losslessly repairable format error is normalized with the existing repair code. No content is dropped and no schema threshold is relaxed. Successful captions replace the failed row in captions.sqlite3, the existing shards/batch-*.jsonl and the canonical extended JSON. Other shard lines are unchanged. Operational API receipts are stored under exp; no alternate caption dataset is created. Atomic per-file publication is recoverable but is not a multi-file transaction; consumers must require the final verified audit before reading a completed generation.

The original training fields remain intact. Test metadata retains absent native supervision as null. Captions remain unreviewed machine drafts with indexable=false and trainable=false. This command does not install automatic monitoring, publish test details, score students or train models.

Restart the same run directory after an interrupted publication to reuse saved responses without sending successful paid requests again. A failed response may be retried only in a subsequent explicitly bounded round; immutable previous receipts remain unchanged.

## Explicit visual completion after provider rejection

`apply_reviewed_descriptions` is a separate explicit offline authoring operation, never an automatic paid retry fallback. It accepts original current four-camera evidence and a description authored after inspecting that evidence; the unchanged common validator checks every field. Its provenance is `codex_visual_review`, provider_generated=false, with a link to the original failed provider receipt. It does not label these descriptions as DeepSeek replies or independently human-approved text. Only still-failed HTTP400 rows may receive this operation; successful provider captions cannot be overwritten. Resume using the same review audit path preserves the complete receipt membership. The primary export and full verification must run afterwards.
