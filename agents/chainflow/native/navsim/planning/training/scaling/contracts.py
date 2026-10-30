from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from omegaconf import DictConfig, OmegaConf


ALLOWED_QUALITIES = {"exact", "approximate", "unavailable"}


def _plain_mapping(value: Any, name: str) -> Dict[str, Any]:
    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=True)
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return dict(value)


def _optional_positive_float(value: Any, name: str) -> Optional[float]:
    if value is None:
        return None
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return parsed


def _optional_positive_int(value: Any, name: str) -> Optional[int]:
    if value is None:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


@dataclass(frozen=True)
class DataCapability:
    """One model-independent fact provided by a physical data source."""

    name: str
    quality: str
    max_horizon_s: Optional[float] = None
    max_num_poses: Optional[int] = None
    interval_s: Optional[float] = None

    @property
    def available(self) -> bool:
        return self.quality != "unavailable"


@dataclass(frozen=True)
class CommonSupervision:
    """A source capability required by the shared scaling baseline."""

    name: str
    capability: str
    required_quality: str
    min_horizon_s: Optional[float] = None
    min_num_poses: Optional[int] = None
    interval_s: Optional[float] = None


@dataclass(frozen=True)
class SourceContract:
    """Physical source metadata; it never names a model head or loss."""

    name: str
    source_index: int
    dataset_role: str
    expected_samples: int
    metric_cache_name: str
    metric_protocol: str
    observation_horizon_s: float
    future_trajectory_horizon_s: float
    capabilities: Mapping[str, DataCapability]

    @property
    def is_synthetic(self) -> bool:
        return self.dataset_role == "synthetic"

    def capability(self, name: str) -> Optional[DataCapability]:
        return self.capabilities.get(name)

    def satisfies(self, requirement: CommonSupervision) -> bool:
        capability = self.capability(requirement.capability)
        if capability is None or not capability.available:
            return False
        if requirement.required_quality == "exact" and capability.quality != "exact":
            return False
        if requirement.required_quality == "approximate" and capability.quality not in {
            "exact",
            "approximate",
        }:
            return False
        if requirement.min_horizon_s is not None and (
            capability.max_horizon_s is None
            or capability.max_horizon_s + 1e-6 < requirement.min_horizon_s
        ):
            return False
        if requirement.min_num_poses is not None and (
            capability.max_num_poses is None
            or capability.max_num_poses < requirement.min_num_poses
        ):
            return False
        if requirement.interval_s is not None and (
            capability.interval_s is None
            or abs(capability.interval_s - requirement.interval_s) > 1e-6
        ):
            return False
        return True


@dataclass(frozen=True)
class ScaleTrainingPlan:
    """Validated data-only plan for a mixed-source scaling experiment."""

    protocol: str
    schema_version: str
    contracts: Mapping[str, SourceContract]
    common_supervision: Mapping[str, CommonSupervision]
    expected_total_samples: int

    def source(self, name: str) -> SourceContract:
        try:
            return self.contracts[name]
        except KeyError as exc:
            raise KeyError(
                f"Scale source '{name}' is not declared in protocol {self.protocol}"
            ) from exc

    @property
    def capability_names(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    capability_name
                    for contract in self.contracts.values()
                    for capability_name in contract.capabilities
                }
            )
        )

    def manifest(self, actual_counts: Optional[Mapping[str, int]] = None) -> Dict[str, Any]:
        counts = {
            name: int(actual_counts[name]) if actual_counts is not None else contract.expected_samples
            for name, contract in self.contracts.items()
        }
        total = sum(counts.values())
        coverage = {}
        for name, requirement in self.common_supervision.items():
            exact_samples = sum(
                counts[source_name]
                for source_name, contract in self.contracts.items()
                if contract.satisfies(requirement)
            )
            coverage[name] = {
                "satisfied_samples": exact_samples,
                "total_samples": total,
                "satisfied_rate": exact_samples / total if total else 0.0,
            }

        body: Dict[str, Any] = {
            "schema_version": self.schema_version,
            "protocol": self.protocol,
            "expected_total_samples": self.expected_total_samples,
            "actual_total_samples": total,
            "common_supervision": {
                name: {
                    "capability": requirement.capability,
                    "required_quality": requirement.required_quality,
                    "min_horizon_s": requirement.min_horizon_s,
                    "min_num_poses": requirement.min_num_poses,
                    "interval_s": requirement.interval_s,
                    **coverage[name],
                }
                for name, requirement in self.common_supervision.items()
            },
            "sources": {
                name: {
                    "source_index": contract.source_index,
                    "dataset_role": contract.dataset_role,
                    "expected_samples": contract.expected_samples,
                    "actual_samples": counts[name],
                    "metric_cache_name": contract.metric_cache_name,
                    "metric_protocol": contract.metric_protocol,
                    "observation_horizon_s": contract.observation_horizon_s,
                    "future_trajectory_horizon_s": contract.future_trajectory_horizon_s,
                    "capabilities": {
                        capability_name: {
                            "quality": capability.quality,
                            "max_horizon_s": capability.max_horizon_s,
                            "max_num_poses": capability.max_num_poses,
                            "interval_s": capability.interval_s,
                        }
                        for capability_name, capability in contract.capabilities.items()
                    },
                }
                for name, contract in self.contracts.items()
            },
        }
        canonical = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
        body["contract_sha256"] = hashlib.sha256(canonical).hexdigest()
        return body

    def validate_actual_counts(self, actual_counts: Mapping[str, int]) -> None:
        unknown = set(actual_counts) - set(self.contracts)
        missing = set(self.contracts) - set(actual_counts)
        if unknown or missing:
            raise RuntimeError(
                "Scale dataset sources differ from contract: "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        for name, contract in self.contracts.items():
            actual = int(actual_counts[name])
            if actual != contract.expected_samples:
                raise RuntimeError(
                    f"{name}: scale sample count changed: {actual} != {contract.expected_samples}"
                )
        actual_total = sum(int(value) for value in actual_counts.values())
        if actual_total != self.expected_total_samples:
            raise RuntimeError(
                f"Scale total sample count changed: {actual_total} != {self.expected_total_samples}"
            )

    def write_manifest(self, output_dir: Path, actual_counts: Mapping[str, int]) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        destination = output_dir / "scale_manifest.json"
        temporary = output_dir / ".scale_manifest.json.tmp"
        temporary.write_text(
            json.dumps(self.manifest(actual_counts), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
        return destination


def _parse_capability(source_name: str, name: str, raw: Any) -> DataCapability:
    cfg = _plain_mapping(raw, f"scale_training.sources.{source_name}.capabilities.{name}")
    quality = str(cfg.get("quality", "unavailable"))
    if quality not in ALLOWED_QUALITIES:
        raise ValueError(f"{source_name}.{name}: unsupported quality {quality!r}")
    capability = DataCapability(
        name=name,
        quality=quality,
        max_horizon_s=_optional_positive_float(
            cfg.get("max_horizon_s"), f"{source_name}.{name}.max_horizon_s"
        ),
        max_num_poses=_optional_positive_int(
            cfg.get("max_num_poses"), f"{source_name}.{name}.max_num_poses"
        ),
        interval_s=_optional_positive_float(
            cfg.get("interval_s"), f"{source_name}.{name}.interval_s"
        ),
    )
    if quality == "unavailable" and any(
        value is not None
        for value in (
            capability.max_horizon_s,
            capability.max_num_poses,
            capability.interval_s,
        )
    ):
        raise ValueError(f"{source_name}.{name}: unavailable capability cannot claim support")
    return capability


def _parse_source(name: str, raw: Any) -> SourceContract:
    cfg = _plain_mapping(raw, f"scale_training.sources.{name}")
    role = str(cfg.get("dataset_role"))
    if role not in {"real", "synthetic"}:
        raise ValueError(f"{name}: dataset_role must be 'real' or 'synthetic', got {role!r}")
    source_index = int(cfg.get("source_index"))
    expected_samples = int(cfg.get("expected_samples"))
    observation_horizon_s = float(cfg.get("observation_horizon_s"))
    future_horizon_s = float(cfg.get("future_trajectory_horizon_s"))
    metric_cache_name = str(cfg.get("metric_cache_name", ""))
    metric_protocol = str(cfg.get("metric_protocol", ""))
    if source_index < 0 or expected_samples <= 0:
        raise ValueError(f"{name}: source_index must be non-negative and expected_samples positive")
    if (
        not math.isfinite(observation_horizon_s)
        or observation_horizon_s <= 0.0
        or not math.isfinite(future_horizon_s)
        or future_horizon_s <= 0.0
        or not metric_cache_name
        or not metric_protocol
    ):
        raise ValueError(
            f"{name}: horizons, metric_cache_name and metric_protocol must be valid"
        )
    capabilities_cfg = _plain_mapping(
        cfg.get("capabilities"), f"scale_training.sources.{name}.capabilities"
    )
    if not capabilities_cfg:
        raise ValueError(f"{name}: at least one data capability is required")
    capabilities = {
        capability_name: _parse_capability(name, capability_name, value)
        for capability_name, value in capabilities_cfg.items()
    }
    return SourceContract(
        name=name,
        source_index=source_index,
        dataset_role=role,
        expected_samples=expected_samples,
        metric_cache_name=metric_cache_name,
        metric_protocol=metric_protocol,
        observation_horizon_s=observation_horizon_s,
        future_trajectory_horizon_s=future_horizon_s,
        capabilities=capabilities,
    )


def _parse_requirement(name: str, raw: Any) -> CommonSupervision:
    cfg = _plain_mapping(raw, f"scale_training.common_supervision.{name}")
    capability = str(cfg.get("capability", ""))
    required_quality = str(cfg.get("required_quality", "exact"))
    if not capability:
        raise ValueError(f"{name}: common supervision must reference a capability")
    if required_quality not in {"exact", "approximate"}:
        raise ValueError(f"{name}: unsupported required_quality {required_quality!r}")
    return CommonSupervision(
        name=name,
        capability=capability,
        required_quality=required_quality,
        min_horizon_s=_optional_positive_float(
            cfg.get("min_horizon_s"), f"common_supervision.{name}.min_horizon_s"
        ),
        min_num_poses=_optional_positive_int(
            cfg.get("min_num_poses"), f"common_supervision.{name}.min_num_poses"
        ),
        interval_s=_optional_positive_float(
            cfg.get("interval_s"), f"common_supervision.{name}.interval_s"
        ),
    )


def build_scale_training_plan(cfg: Any) -> Optional[ScaleTrainingPlan]:
    """Build a data-only scale plan; ordinary splits return ``None`` unchanged."""

    enabled = bool(OmegaConf.select(cfg, "scale_training.enabled", default=False))
    if not enabled:
        return None
    scale_cfg = _plain_mapping(OmegaConf.select(cfg, "scale_training"), "scale_training")
    sources_cfg = _plain_mapping(scale_cfg.get("sources"), "scale_training.sources")
    contracts = {name: _parse_source(name, value) for name, value in sources_cfg.items()}
    if not contracts:
        raise ValueError("An enabled scale-data protocol must declare at least one source")

    common_cfg = _plain_mapping(
        scale_cfg.get("common_supervision"), "scale_training.common_supervision"
    )
    common_supervision = {
        name: _parse_requirement(name, value) for name, value in common_cfg.items()
    }
    if not common_supervision:
        raise ValueError("An enabled scale-data protocol must declare common supervision")

    indices = [contract.source_index for contract in contracts.values()]
    if len(indices) != len(set(indices)):
        raise ValueError("scale_training source_index values must be unique")
    real_sources = [name for name, contract in contracts.items() if contract.dataset_role == "real"]
    if real_sources != ["real_train"]:
        raise ValueError(f"The scale protocol must declare exactly one real_train source, got {real_sources}")

    for source_name, contract in contracts.items():
        missing_requirements = [
            name
            for name, requirement in common_supervision.items()
            if not contract.satisfies(requirement)
        ]
        if missing_requirements:
            raise ValueError(
                f"{source_name}: source does not satisfy common supervision {missing_requirements}"
            )

    split_sources = OmegaConf.select(cfg, "train_test_split.synthetic_sources", default={})
    split_source_names = set(_plain_mapping(split_sources, "train_test_split.synthetic_sources"))
    contract_synthetic_names = {
        name for name, contract in contracts.items() if contract.dataset_role == "synthetic"
    }
    if split_source_names != contract_synthetic_names:
        raise ValueError(
            "Synthetic split sources differ from scale contract: "
            f"split={sorted(split_source_names)}, contract={sorted(contract_synthetic_names)}"
        )

    expected_real = OmegaConf.select(
        cfg, "train_test_split.expected_real_train_samples", default=None
    )
    if expected_real is None or int(expected_real) != contracts["real_train"].expected_samples:
        raise ValueError(
            "train_test_split.expected_real_train_samples must equal the real_train contract count"
        )
    for name in split_source_names:
        split_count = OmegaConf.select(
            cfg, f"train_test_split.synthetic_sources.{name}.expected_num_samples", default=None
        )
        if split_count is None or int(split_count) != contracts[name].expected_samples:
            raise ValueError(f"{name}: split expected_num_samples differs from scale contract")

    expected_total = int(scale_cfg.get("expected_total_samples"))
    declared_sum = sum(contract.expected_samples for contract in contracts.values())
    split_total = OmegaConf.select(
        cfg, "train_test_split.expected_total_train_samples", default=None
    )
    if declared_sum != expected_total or split_total is None or int(split_total) != expected_total:
        raise ValueError(
            "Scale sample totals disagree: "
            f"source_sum={declared_sum}, protocol={expected_total}, split={split_total}"
        )

    return ScaleTrainingPlan(
        protocol=str(scale_cfg.get("protocol", "unnamed")),
        schema_version=str(scale_cfg.get("schema_version", "scale-data/v2")),
        contracts=contracts,
        common_supervision=common_supervision,
        expected_total_samples=expected_total,
    )
