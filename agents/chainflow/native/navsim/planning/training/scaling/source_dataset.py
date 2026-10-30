from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from .contracts import DataCapability, ScaleTrainingPlan, SourceContract


def _capability_tensor_record(
    capability: Optional[DataCapability],
) -> Dict[str, torch.Tensor]:
    return {
        "available": torch.tensor(
            capability is not None and capability.available, dtype=torch.bool
        ),
        "exact": torch.tensor(
            capability is not None and capability.quality == "exact", dtype=torch.bool
        ),
        "max_horizon_s": torch.tensor(
            capability.max_horizon_s
            if capability is not None and capability.max_horizon_s is not None
            else -1.0,
            dtype=torch.float32,
        ),
        "max_num_poses": torch.tensor(
            capability.max_num_poses
            if capability is not None and capability.max_num_poses is not None
            else -1,
            dtype=torch.long,
        ),
        "interval_s": torch.tensor(
            capability.interval_s
            if capability is not None and capability.interval_s is not None
            else -1.0,
            dtype=torch.float32,
        ),
    }


class SourceAnnotatedDataset(torch.utils.data.Dataset):
    """Attach physical data capabilities without knowing the consuming model."""

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        contract: SourceContract,
        plan: ScaleTrainingPlan,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.contract = contract
        self.plan = plan
        self._capabilities = {
            name: _capability_tensor_record(contract.capability(name))
            for name in plan.capability_names
        }

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[Any, ...]:
        item = self.dataset[index]
        if not isinstance(item, tuple) or len(item) not in (2, 3):
            raise TypeError("SourceAnnotatedDataset expects a 2- or 3-tuple Dataset item")
        features, original_targets = item[:2]
        if not isinstance(original_targets, dict):
            raise TypeError("SourceAnnotatedDataset expects targets to be a dictionary")

        targets = dict(original_targets)
        targets["_data_source_index"] = torch.tensor(
            self.contract.source_index, dtype=torch.long
        )
        targets["_data_is_synthetic"] = torch.tensor(
            self.contract.is_synthetic, dtype=torch.bool
        )
        targets["_data_future_horizon_s"] = torch.tensor(
            self.contract.future_trajectory_horizon_s, dtype=torch.float32
        )
        targets["_data_observation_horizon_s"] = torch.tensor(
            self.contract.observation_horizon_s, dtype=torch.float32
        )
        targets["_data_metric_cache_name"] = self.contract.metric_cache_name
        targets["_data_metric_protocol"] = self.contract.metric_protocol
        targets["_data_scale_protocol"] = self.plan.protocol
        targets["_data_capabilities"] = {
            name: {field: value.clone() for field, value in record.items()}
            for name, record in self._capabilities.items()
        }

        if len(item) == 2:
            return features, targets
        return features, targets, item[2]
