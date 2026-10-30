"""Build the native AUGUstus PDM context without executing any model training.

Run using the AUGUstus environment. Inputs are a validated supervision manifest,
original NAVSIM logs and maps. All new cache and provenance belong to exp.
"""

import argparse
import os
import sys
import time
from pathlib import Path

from driversi.agents.backends.native import digest, project_root
from driversi.runtime.budget import write_json
from driversi_agents.augustus.data import manifest
from driversi_agents.augustus.metric_cache import scoring_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--logs", required=True, type=Path)
    parser.add_argument("--maps", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = project_root()
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(root / "exp"):
        raise ValueError("Use a new output directory under exp")
    dataset = manifest(args.manifest, allow_smoke=True)
    rows = dataset["splits"]["development"]["records"]
    tokens = [row["frame_id"] for row in rows]
    if not tokens or len(tokens) != len(set(tokens)):
        raise ValueError("Nonempty unique development tokens required")
    sft = root / "agents/augustus/native/AUGUstus_SFT"
    sys.path[:0] = [
        str(sft),
        str(sft / "third_party/navsim"),
        str(sft / "third_party/nuplan-devkit"),
    ]
    os.environ["NUPLAN_MAPS_ROOT"] = str(args.maps.resolve(strict=True))
    from navsim.common.dataclasses import SceneFilter, SensorConfig
    from navsim.common.dataloader import MetricCacheLoader, SceneLoader
    from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor
    from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
    from nuplan.planning.training.experiments.cache_metadata_entry import save_cache_metadata

    started = time.monotonic()
    loader = SceneLoader(
        args.logs.resolve(strict=True),
        None,
        SceneFilter(
            num_history_frames=4,
            num_future_frames=10,
            frame_interval=1,
            log_names=sorted({row["log_name"] for row in rows}),
            tokens=tokens,
        ),
        SensorConfig.build_no_sensors(),
    )
    if set(loader.tokens) != set(tokens):
        raise ValueError("Original logs lack required full native scoring windows")
    output.mkdir(parents=True, exist_ok=False)
    cache_path = output / "metric_cache"
    processor = MetricCacheProcessor(str(cache_path), False)
    entries = []
    for token in loader.tokens:
        scenario = NavSimScenario(
            loader.get_scene_from_token(token),
            map_root=os.environ["NUPLAN_MAPS_ROOT"],
            map_version="nuplan-maps-v1.0",
        )
        entry = processor.compute_metric_cache(scenario)
        if entry is None:
            raise ValueError("Native cache creation failed: " + token)
        entries.append(entry)
    save_cache_metadata(entries, cache_path, 0)
    loader = MetricCacheLoader(cache_path)
    for token in tokens:
        item = loader.get_from_token(token)
        if not hasattr(item, "trajectory") or not hasattr(item.centerline, "project"):
            raise ValueError("Not a native evaluation cache: " + token)
    files = scoring_files(cache_path, rows)
    value = dict(
        schema_version=1,
        model="augustus",
        protocol="bundled_navsim_pdm_v1",
        development_source_logs=sorted({row["log_name"] for row in rows}),
        dataset_manifest_sha256=digest(args.manifest),
        metric_cache=str(cache_path),
        files={str(path): digest(path) for path in files},
    )
    write_json(output / "evaluator.json", value)
    write_json(
        output / "build.json",
        dict(
            tokens=tokens,
            wall_seconds=time.monotonic() - started,
            native_processor="owned AUGUstus bundled MetricCacheProcessor; unchanged parameters",
            source_manifest=str(args.manifest.resolve()),
        ),
    )
    print(output / "evaluator.json")


if __name__ == "__main__":
    main()
