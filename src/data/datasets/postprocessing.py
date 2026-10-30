"""Campaign-scoped scheduling policy, independent of training admission."""
import json
from pathlib import Path

from driversi.data.datasets.simscale_collection import active

TRAINING_ONLY = "training_only"


def training_only(root):
    root = Path(root)
    path = root / "postprocessing.json"
    if not path.exists():
        return False
    policy = json.loads(path.read_text())
    collection = active(root)
    expected = dict(schema_version=1, mode=TRAINING_ONLY,
                    campaign=collection["campaign"] if collection else None,
                    features="deferred", scene_caption="deferred",
                    classification="deferred", training_admission="unchanged")
    if collection is None or policy != expected:
        raise ValueError("Dataset postprocessing policy does not match its active campaign")
    return True


def mode(config, *, production=True):
    value = config.get("postprocessing_mode", "features_and_classification")
    if value not in (TRAINING_ONLY, "features_and_classification"):
        raise ValueError("Unknown synthetic postprocessing mode")
    if value == TRAINING_ONLY:
        return value
    from driversi.data.datasets.synthetic_publish import dataset_root
    # A new policy can defer pending work from an immutable older plan. It
    # cannot grant admission or retroactively relabel completed receipts.
    root = dataset_root().parent
    if production and training_only(root):
        policy = json.loads((root / "postprocessing.json").read_text())
        output = Path(config.get("output", "")).resolve()
        if output.is_relative_to(Path(policy["campaign"]).resolve()):
            return TRAINING_ONLY
    return value
