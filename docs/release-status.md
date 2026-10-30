# Release status and known limitations

This snapshot is based on source commit `2725bed62f70e3fefa6b3880a341412425cbce8b`, followed by release-only documentation, portable-configuration and local-view changes. `release-manifest.json` describes the final payload. Original experiment outputs and private checkpoint identities are not included.

## What is preserved

- Retrieval, caption and independent text/state/topology clustering implementations.
- Bounded history perturbation, correction, rendering, qualification and export components.
- Persistent Harness task ownership, budgets, explicit recovery and experience records.
- Native model adapters, paired continuation training and evaluation gates.
- Frozen AUGUstus skill proposal/validation/rollback code.
- Tests and upstream source/provenance/license files.

## What is not established

- Sustained recursive improvement across repeated driving tasks.
- Generalization benefit from the frozen-skill experiments; recent automatic revisions were rolled back.
- Complete admission or learning benefit from the latest dynamic-asset history/anchor batch.
- A portable one-command GPU installation or fully reproduced paper benchmark from this source-only snapshot.
- Pixel-level correctness of arbitrary novel views.

## Paused dynamic-asset experiment

At the pause, both assets completed 30,001 native reconstruction iterations. No sample in that **new batch** had completed all of PDMS, source PSNR and branch-binding admission; its real-replay paired training had not started.

Two concrete issues remain in the inherited source:

1. A canonical metadata path resolves to `video_scene_dict_final.pkl`, but some comparison/admission readers expect the basename `video_scene_dict.pkl`.
2. A reconstructed static rigid node can retain an earlier observation while being absent from current anchor annotations. The current binding checker demands a unique current source observation. Earlier observations need explicit immutable-object evidence, not fabricated current boxes or silent omission.

These are documented rather than bypassed in the release. Asset completion alone does not establish PSNR or binding quality. Existing source clouds also leave some sparse actors without supported appearance assets; whether a selected branch is usable depends on its actual visible objects.

The original experiment remains paused. Packaging, installation and inspection never authorize resuming it. No original assets, datasets, checkpoints or historical receipts were moved into this code release.

## Recovery scope

Known settled transient failures have bounded recovery paths. Unrecognized program failures and ambiguous in-flight results require reconciliation; the Harness does not repair arbitrary code defects on its own. In a research implementation this is a necessary distinction between unattended workflow execution and unverified autonomous code repair.

## Reading validation results

Release checks cover source installation, command discovery, local visualization, contracts and selected CPU regression tests. They do not execute native GPU training or certify scientific claims. Optional native tests can skip when a compiled simulator or model-specific environment is absent. The final verification record is separate from this status description and reports actual pass/skip/failure counts.

Historical documentation is retained as technical reference. When it mentions an old result path or threshold, it describes that experiment, not the current paused profile. Consult a model adapter and the explicit plan for the actual operation being launched.
