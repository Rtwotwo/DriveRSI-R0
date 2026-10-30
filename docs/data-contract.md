# Data and supervision contracts

## One physical branch

A generated package ties together the training source, executed history, future correction, actor lifecycle, camera calibration, rendered images, scoring context and final training target. A source log token, branch identity and timestamp are different identifiers; preserve all of them.

Same-branch checks use actual camera transforms, actor states, geometry, source observations and model-specific decoding. A caption, a successful process exit or an asset filename is not a substitute for these checks.

## Images versus numerical states

| Contract | Rendered RGB | Numerical record | Intended use |
|---|---|---|---|
| `history_rgb_future_states_v1` | Four cameras at four history/anchor timestamps: 16 images | Executed history and future states | Student requiring the four-image history |
| `anchor_rgb_history_future_states_v1` | Four cameras at the current anchor: 4 images | Same history and future numerical states | Native ChainFlow stage1 anchor-image features |

Both profiles generate **zero future RGB**. ChainFlow exports retain four history and ten future states at 0.5 s spacing. The current teacher PDMS horizon is four seconds, whereas the executed future can be five seconds; the two must not be reported as identical coverage. AUGUstus's native target is eight future poses and its image input is different: use its own adapter, not an implicit contract conversion.

Changing only future labels while retaining real current/history images is a distinct same-observation supervision experiment. Changing history poses requires corresponding new history observations, even if future RGB is omitted.

## Generated training storage

```text
dataset/
├── sensor_blobs/        # Exporter-owned rendered input images
├── navsim_logs/         # Exporter-owned native scene/log records
└── ...                 # Exporter-owned manifests, registry and admission receipts
```

Let the exporter choose native filenames and identifiers. Do not manually create a log row pointing at images from another branch. Full generation, render, comparison, rejection and timing evidence belongs under `exp/<run-id>/`; the admitted package points back to this evidence. Raw navtrain/navtest data remain separate, and a generated library is not merged into the original navtrain annotations.

## Current admission profile

- Teacher **PDMS >0.95**, strict inequality; record the original score unit and convert explicitly when a report uses 0–100.
- Mean source-view **PSNR >25 dB**, strict inequality. Report the minimum and distribution separately where available; mean >25 does not mean every image >25.
- Recompute binding between images, camera motion, dynamic objects and the physical branch.
- Validate the declared native input/target contract before publishing a training sample.
- Keep trajectory difference criteria explicit and versioned. Adaptive criteria must still reject identical or near-identical repeats.

Source-view PSNR compares reconstruction at measured camera poses with corresponding photographs. Novel-view images generally have no paired ground truth; neither this metric nor a teacher trajectory score guarantees absence of artifacts. Missing dynamic-object coverage remains a source-specific limitation. Independent visual review, if performed, must be reported separately.

Historical profiles in the repository retain their original thresholds and cannot be relabeled as passing the current profile. EPDMS has separate code and input requirements; PDMS admission is not an EPDMS certificate.
