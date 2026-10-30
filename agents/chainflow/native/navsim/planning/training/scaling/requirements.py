from __future__ import annotations

from typing import Any, Mapping, Optional

import torch

from .contracts import CommonSupervision


def resolve_supervision_mask(
    targets: Mapping[str, Any],
    requirement: CommonSupervision,
    *,
    runtime_validity: Optional[torch.Tensor] = None,
    missing: str = "error",
) -> torch.Tensor:
    """Resolve a named data requirement without knowing the consuming model.

    A future model chooses the requirement it needs. The returned mask is the
    intersection of the source's static capability and optional per-sample or
    per-proposal runtime validity. No source role or model class is special.
    """

    if missing not in {"error", "mask"}:
        raise ValueError("missing must be 'error' or 'mask'")
    source_index = targets.get("_data_source_index")
    if not isinstance(source_index, torch.Tensor):
        raise KeyError("Capability resolution requires _data_source_index")
    capabilities = targets.get("_data_capabilities")
    if not isinstance(capabilities, Mapping):
        raise KeyError("Capability resolution requires _data_capabilities")
    record = capabilities.get(requirement.capability)
    if not isinstance(record, Mapping):
        if missing == "error":
            raise KeyError(f"Capability is absent: {requirement.capability}")
        mask = torch.zeros_like(source_index, dtype=torch.bool)
    else:
        required_fields = {
            "available",
            "exact",
            "max_horizon_s",
            "max_num_poses",
            "interval_s",
        }
        absent_fields = required_fields - set(record)
        if absent_fields:
            raise KeyError(
                f"Capability {requirement.capability} is missing fields: {sorted(absent_fields)}"
            )
        mask = record["available"].to(dtype=torch.bool)
        if requirement.required_quality == "exact":
            mask = mask & record["exact"].to(dtype=torch.bool)
        if requirement.min_horizon_s is not None:
            mask = mask & (
                record["max_horizon_s"].to(dtype=torch.float32)
                >= requirement.min_horizon_s - 1e-6
            )
        if requirement.min_num_poses is not None:
            mask = mask & (
                record["max_num_poses"].to(dtype=torch.long)
                >= requirement.min_num_poses
            )
        if requirement.interval_s is not None:
            expected_interval = torch.as_tensor(
                requirement.interval_s,
                device=record["interval_s"].device,
                dtype=torch.float32,
            )
            mask = mask & torch.isclose(
                record["interval_s"].to(dtype=torch.float32),
                expected_interval,
                atol=1e-6,
                rtol=0.0,
            )

    if runtime_validity is not None:
        runtime_mask = runtime_validity.to(device=mask.device, dtype=torch.bool)
        while mask.ndim < runtime_mask.ndim:
            mask = mask.unsqueeze(-1)
        mask = torch.broadcast_to(mask, runtime_mask.shape) & runtime_mask
    return mask
