from __future__ import annotations

from typing import Any, Dict, List, Sequence


_SUPPORTED_PARTITIONS = ("train", "val")


def _deduplicate_preserving_order(values: Sequence[str]) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


def resolve_training_log_names(cfg: Any) -> List[str]:
    """Resolve which official log partitions form the training population.

    ``training_log_partitions`` is the explicit interface used by new splits.
    If absent, the historical ``include_val_logs_in_train`` behavior is kept
    byte-for-byte: navtrain selects train logs and navtrainval selects both.
    """

    split_cfg = cfg.train_test_split
    configured = split_cfg.get("training_log_partitions")
    if configured is None:
        partitions = (
            ["train", "val"]
            if bool(split_cfg.get("include_val_logs_in_train", False))
            else ["train"]
        )
    else:
        partitions = [str(value) for value in configured]

    if not partitions:
        raise ValueError("training_log_partitions must not be empty")
    if len(partitions) != len(set(partitions)):
        raise ValueError(
            f"training_log_partitions contains duplicates: {partitions}"
        )
    unknown = sorted(set(partitions) - set(_SUPPORTED_PARTITIONS))
    if unknown:
        raise ValueError(
            "training_log_partitions contains unsupported values "
            f"{unknown}; expected a subset of {_SUPPORTED_PARTITIONS}"
        )

    available: Dict[str, Sequence[str]] = {
        "train": cfg.train_logs,
        "val": cfg.val_logs,
    }
    selected: List[str] = []
    for partition in partitions:
        selected.extend(str(value) for value in available[partition])
    selected = _deduplicate_preserving_order(selected)
    if not selected:
        raise ValueError(
            f"Resolved an empty training log population from {partitions}"
        )
    return selected


def resolve_cache_training_log_names(cfg: Any) -> List[str]:
    """Resolve cache-only logs without changing historical implicit splits.

    The legacy cache-only branch always selected ``cfg.train_logs``, even when
    ``include_val_logs_in_train`` was true.  Only the new explicit partition
    interface opts into the unified resolver so existing jobs stay unchanged.
    """

    if cfg.train_test_split.get("training_log_partitions") is None:
        return list(cfg.train_logs)
    return resolve_training_log_names(cfg)
