"""AUGUstus native SFT/evaluation in an isolated environment, with explicit data contracts."""

import copy
import json
import struct
from pathlib import Path

import yaml

from driversi.agents.backends.native import asset_files
from driversi.training.sampling.environment import source_log_group
from driversi_agents.augustus.data import cache_files, manifest


def safetensors_extent_complete(path):
    """Cheap structural rejection before launch; runtime strict_load remains required.

    A header may be readable while hundreds of MB of payload are still missing.
    This is not a checksum or proof that tensor values are correct.
    """
    try:
        path = Path(path)
        size = path.stat().st_size
        with path.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                return False
            header_size = struct.unpack("<Q", prefix)[0]
            if not 2 <= header_size <= min(16 << 20, size - 8):
                return False
            header = json.loads(stream.read(header_size))
        if not isinstance(header, dict):
            return False
        ranges = []
        for key, value in header.items():
            if key == "__metadata__":
                continue
            offsets = value["data_offsets"]
            if not isinstance(offsets, list) or len(offsets) != 2:
                return False
            start, end = offsets
            if type(start) is not int or type(end) is not int or not 0 <= start <= end:
                return False
            ranges.append((start, end))
        cursor = 0
        for start, end in sorted(ranges):
            if start != cursor:
                return False
            cursor = end
        return bool(ranges) and 8 + header_size + cursor == size
    except (OSError, ValueError, KeyError, TypeError, struct.error):
        return False


def complete_assets(path):
    # aria2 preallocates filenames; other interrupted downloads leave no marker.
    base = Path(path)
    markers = list(base.rglob("*.aria2")) if base.is_dir() else []
    if markers:
        return [str(p) + ".PENDING_DOWNLOAD" for p in markers]
    files = asset_files(path)
    return [
        p + ".INCOMPLETE_SAFETENSORS"
        if p.endswith(".safetensors") and Path(p).is_file() and not safetensors_extent_complete(p)
        else p
        for p in files
    ]


def merge_recipe(base, published):
    result = copy.deepcopy(base)
    for key, value in published.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_recipe(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def build(cfg, root, repo, python, weights, paths, output, operation):
    settings = cfg["settings"]
    if set(settings) - {"gpu", "gpus", "max_sim_agent_steps", "max_optimizer_steps"} or not {"gpu", "gpus", "max_sim_agent_steps"} <= set(settings):
        raise ValueError("Explicit GPUs and native PDM budget required")
    gpus = settings["gpus"]
    if (
        not isinstance(gpus, list)
        or not 1 <= len(gpus) <= 8
        or any(type(x) is not int or x < 0 for x in gpus)
        or len(set(gpus)) != len(gpus)
        or settings["gpu"] != gpus[0]
        or type(settings["max_sim_agent_steps"]) is not int
        or settings["max_sim_agent_steps"] < 0
    ):
        raise ValueError("Invalid native allocation")
    if operation not in ("train", "evaluate") or (operation == "evaluate" and len(gpus) != 1):
        raise ValueError("Train supports 1-8 GPUs; paired evaluation uses one fixed GPU")
    if set(weights) not in ({"parent"}, {"parent", "backbone"}) or set(paths) != {
        "dataset_manifest",
        "validation_metric_cache",
        "development_metric_cache",
    }:
        raise ValueError(
            "Explicit full parent, dataset and separate scoring contexts required"
        )
    sft = repo / "AUGUstus_SFT"
    parent = Path(weights["parent"])
    recipe = parent / "config.yaml"
    # The inventory remains inspectable before download, but missing recipe is a
    # required file, so this fallback can never execute as an alternative recipe.
    source = sft / "configs/navsim_qwen3p5.yaml"
    native = yaml.safe_load(source.read_text())
    if recipe.is_file():
        native = merge_recipe(native, yaml.safe_load(recipe.read_text()))
    if (
        native["trajectory"] != {"action_dim": 3, "action_repr": "absolute", "traj_token": 8}
        or native["cot"]["cot_components"] != ["prediction"]
        or native["data"]["data_format"] != "hf_cache"
        or native["data"]["video_fps"] != 2
        or native["data"]["video_max_frames"] != 4
        or native.get("resume", {}).get("enabled", False)
        or not native["normalization"]["action_norm"]
    ):
        raise ValueError("Unsupported native recipe: no implicit action/CoT/resume conversion")
    native = copy.deepcopy(native)
    if "max_optimizer_steps" in settings:
        cap = settings["max_optimizer_steps"]
        if type(cap) is not int or cap <= 0:
            raise ValueError("Positive native optimizer-step cap required")
        native["training"]["max_steps"] = cap
        # Keep num_epochs / schedule / loss / optimizer identical to the parent recipe.
    data_path = Path(paths["dataset_manifest"])
    dataset = manifest(data_path) if data_path.is_file() else None
    stats = parent / "action_norm_stats.json"
    if stats.is_file():
        from driversi_agents.augustus.normalization import read_stats

        read_stats(stats)
    checkpoint_output = output / "checkpoints" if output.is_relative_to(root / "exp") else root / "exp" / output.relative_to(root / "outputs") / "checkpoints"
    bundled = "backbone" not in weights
    metadata = parent
    if not (metadata / "config.json").is_file() and (parent / "pytorch_model/config.json").is_file():
        metadata = parent / "pytorch_model"
    native["model"]["qwen_model_name"] = str(metadata) if bundled else weights["backbone"]
    # This is derived from the chosen weight source, including saved candidates.
    # runner strictly restores every tensor before the model can be used.
    native["model"]["skip_backbone_weight_load"] = bundled
    native["normalization"]["stats_file"] = str(stats)
    native["data"]["prompt_file"] = str(sft / "prompts/default.txt")
    native["data"]["train_data_path"] = (
        dataset["splits"]["train"]["cache"] if dataset else str(data_path.parent / "train")
    )
    native["data"]["val_data_path"] = (
        dataset["splits"]["validation" if operation == "train" else "development"]["cache"]
        if dataset
        else str(data_path.parent / "validation")
    )
    native["data"]["metric_cache_test"] = paths[
        "validation_metric_cache" if operation == "train" else "development_metric_cache"
    ]
    native["data"]["metric_cache_train"] = None
    native["training"]["output_dir"] = str(checkpoint_output)
    native["training"]["logger"] = ["console"]
    native["training"]["wandb_token"] = ""
    native["resume"] = {"enabled": False, "resume_path": None}
    official_test = (
        root
        / "src/vendor/driving/navsim-v1.1/navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml"
    )
    if dataset:
        protected = set(yaml.safe_load(official_test.read_text())["log_names"])
        all_logs = {x["log_name"] for b in dataset["splits"].values() for x in b["records"]}
        if {source_log_group(x) for x in protected} & {source_log_group(x) for x in all_logs}:
            raise ValueError("Official navtest cannot enter RSI training/feedback")
        if operation == "train":
            minimum = native["training"]["per_device_train_batch_size"] * len(gpus)
            if len(dataset["splits"]["train"]["records"]) < minimum:
                raise ValueError(
                    "Dataset too small for native drop_last batch; no updates would occur"
                )
    wrapper = root / "agents/augustus/runner.py"
    argv = [python]
    if operation == "train":
        argv += [
            "-m",
            "accelerate.commands.launch",
            "--config_file",
            str(sft / "configs/accelerate_deepspeed.yaml"),
            "--num_processes",
            str(len(gpus)),
            "--num_machines",
            "1",
            "--main_process_port",
            "0",
        ]
    argv += [
        str(wrapper),
        "--repo",
        str(repo),
        "--operation",
        operation,
        "--config",
        str(output / "native.yaml"),
        "--checkpoint",
        str(parent),
        "--output",
        str(output),
        "--dataset",
        str(data_path),
        "--max-sim-agent-steps",
        str(settings["max_sim_agent_steps"]),
    ]
    env = {
        "CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus)),
        "PYTHONPATH": ":".join(
            [str(sft), str(sft / "third_party/navsim"), str(sft / "third_party/nuplan-devkit")]
        ),
        "SWANLAB_MODE": "disabled",
        "WANDB_MODE": "disabled",
    }
    files = [
        str(recipe),
        str(stats),
        str(source),
        str(wrapper),
        str(wrapper.with_name("budget.py")),
        str(wrapper.with_name("normalization.py")),
        str(wrapper.with_name("metric_cache.py")),
        str(wrapper.with_name("data.py")),
        str(data_path),
        str(official_test),
        str(sft / "train_qwen3p5.py"),
        str(sft / "test_qwen3p5.py"),
        str(sft / "configs/accelerate_deepspeed.yaml"),
        str(sft / "prompts/default.txt"),
        *complete_assets(parent),
        *(complete_assets(weights["backbone"]) if not bundled else [
            str(metadata / name) for name in (
                "config.json", "tokenizer.json", "tokenizer_config.json", "processor_config.json"
            )
        ]),
    ]
    if dataset:
        files += cache_files(dataset)
        from driversi_agents.augustus.metric_cache import scoring_files

        for split in ("validation", "development"):
            files += scoring_files(
                paths[split + "_metric_cache"], dataset["splits"][split]["records"]
            )
    launch = dict(
        argv=argv,
        env=env,
        files=sorted(set(files)),
        directories=[
            str(sft / "third_party/navsim"),
            str(sft / "third_party/nuplan-devkit"),
            paths["validation_metric_cache"],
            paths["development_metric_cache"],
        ],
        protocol="augustus_native_sft"
        if operation == "train"
        else "augustus_native_paired_development",
        generated_files={"native.yaml": yaml.safe_dump(native, sort_keys=False)},
    )
    if operation == "train":
        launch["checkpoint_output"] = str(checkpoint_output)
    return launch
