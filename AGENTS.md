# DriveRSI repository organization

Read README.md and docs/README.md for domain ownership. Keep long-lived implementation in src, supported native models in agents, pinned upstream code in src/vendor, presets in src/config, and explicit command entry points in scripts. Nested instructions apply to those folders.

Use exp/<run-id> for new run artifacts and checkpoints; weights is for downloaded initial models, assets for authored reusable assets. outputs retains historical artifacts and preview links used by recorded evidence. Do not move or delete referenced runtime data, environments or initial weights merely to simplify directory counts.

clones is source/build/download staging, not a new runtime Python import root. The remaining native build inputs and data preparation dependencies must be audited before migration. Preserve upstream licensing and public native APIs.

Use the fixed src/visualization interface and explicit evidence catalog for new experiment display. Do not generate another complete dashboard per run. Do not introduce root-level temporary scripts or configurations.

Document a changed command or configuration path and update its tracked callers. Preserve immutable historical receipts; do not silently re-sign them. Keep Python module aliases needed by saved objects; filesystem script aliases may be retired with an explicit migration record.

No credentials or test-scene details in published artifacts. No navtrain student scoring unless the user explicitly reverses their stop request. Model/loss/scoring/admission changes are separate from organization work.
