# Current generated dataset

The active experiment uses `dataset/collection.json`; initialization requires
empty sample, staging, admitted and compatibility directories. It never clears
or restores data. Execution receipts, caches, metrics and model checkpoints stay
under the declared `exp/` campaign. Other campaigns cannot publish into this
active dataset.

```text
dataset/
  collection.json
  navsim_logs/synthetic_driverl_v1/<branch>.pkl
  sensor_blobs_hist/synthetic_driverl_v1/<branch>/<camera>/<frame>.png
  sensor_blobs_fut/synthetic_driverl_v1/<branch>/<camera>/<frame>.png
  sensor_blobs/synthetic_driverl_v1/<branch>/<camera>/<frame>.png
  manifests/synthetic_driverl_v1/{index.json,<package>.json}
  synthetic_navsim/{samples,admitted,.staging,.locks}/
```

This follows [SimScale's published OpenScene layout](https://github.com/OpenDriveLab/SimScale#-data-preparation).
Unlike its released four-second future window, this experiment archives ten
actual future frames at 2 Hz, together with four history frames. The complete
continuous teacher rollout remains in experiment evidence. Native ChainFlow
model, target horizon and loss remain unchanged. Existing admission still applies,
with the explicitly requested additional complete-EPDMS gate described below.

Canonical `samples/<package>/` holds one sealed native log, calibration/object
annotations and the generated pixels. History/future/combined sensor views link
those pixels; there are no duplicate RGB copies. The compatibility PKL changes
only `cams[*].data_path` by removing the history/future prefix. Every other
native field remains equal, including timestamps, ego poses and object labels.
The combined sensor root resolves every frame from that PKL. History-only
consumers must continue selecting the first four frames; future images are not
policy inputs.

Publication is hooked to `index_admitted_package`, after native sample admission
and fresh-feature validation. Export completion alone cannot add a collection
member. Consumers use `manifests/.../index.json` and revalidate the referenced
native receipt; never glob loose PKLs from an interrupted publication. A crash
before index commit may leave verified links, which the same publication reuses.
Caption/cluster records remain in the isolated generated library.

Initialize with:

```bash
.venv/bin/python -m driversi.data.datasets.simscale_collection \
  --campaign <repository>/exp/continual-scene-curriculum-v2
```

Set `dataset_collection` in the versioned generation config to the resulting
`dataset/collection.json`, then initialize the existing continual curriculum.
This only prepares storage and ownership; it does not certify missing assets or
launch a GPU task.

The current collection additionally requires **each trajectory EPDMS > 0.95**,
including actual two-frame extended comfort with its native weight 2. It does
not substitute the student's average navtest score, v1 PDMS or a score with
missing comfort. This policy is enforced before both canonical admitted-index
insertion and SimScale collection publication, and again on training consumption.
See [the native EPDMS contract](../evaluation/epdms.md). No synthetic sample has
yet passed this new complete contract; the current DriveRL producer still lacks
an independently captured adjacent native planning pair.
