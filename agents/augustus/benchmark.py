"""Locked official navtest membership for the unchanged AUGUstus SFT worker.

The shared NAVSIM index is transported by the ChainFlow benchmark module; its
model inference and scores are never imported. Test annotations stay isolated.
"""

from pathlib import Path

from driversi.agents.backends.native import digest
from driversi_agents.chainflow.benchmark import official_filter, read_index, read_pilot_subset


def validate_membership(value):
    spec = value["benchmark"]
    if set(spec) != {"protocol", "test_subset", "test_subset_sha256"} or spec["protocol"] not in (
        "official_navtest_v1_locked_pilot",
        "official_navtest_v1",
    ):
        raise ValueError("Explicit AUGUstus locked navtest protocol required")
    if digest(spec["test_subset"]) != spec["test_subset_sha256"]:
        raise ValueError("Locked evaluation subset changed")
    if spec["protocol"] == "official_navtest_v1":
        pilot = read_index(spec["test_subset"], "navtest")
        if len(pilot["records"]) != 12146:
            raise ValueError("Full official 12146 population required")
    else:
        pilot = read_pilot_subset(spec["test_subset"])
    expected = {(r["token"], r["source_log"]) for r in pilot["records"]}
    actual = [(r["frame_id"], r["log_name"]) for r in value["splits"]["development"]["records"]]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Native evaluation cache differs from locked official population")
    _, train = official_filter("navtrain")
    train_tokens, train_logs = set(train["tokens"]), set(train["log_names"])
    # In joint data, synthetic identities are validated by the sealed branch
    # contract; only the original source block is tested for official tokens.
    block = value["splits"]["train"]
    original = block.get("joint_source_blocks", [block])[0]
    for record in original["records"] + value["splits"]["validation"]["records"]:
        if record["frame_id"] not in train_tokens or record["log_name"] not in train_logs:
            raise ValueError("Non-navtrain record in AUGUstus optimization/validation inputs")
    return pilot


def sealed_files(value):
    """Every immutable source that the lifecycle must recheck before launch."""
    pilot = validate_membership(value)
    subset = value["benchmark"]["test_subset"]
    return {
        str(Path(subset).resolve()): digest(subset),
        **({pilot["test_index"]: pilot["test_index_sha256"]} if "test_index" in pilot else {}),
        **{
            str(path): digest(path)
            for path, _ in (official_filter("navtrain"), official_filter("navtest"))
        },
    }
