"""Thin executable boundary; native algorithms remain in the pinned upstream tree."""

import argparse
import importlib
import json
import sys
from pathlib import Path

import yaml

from driversi_agents.augustus.budget import metered


def strict_load(model, checkpoint):
    from safetensors.torch import load_file

    path = Path(checkpoint)
    candidates = [path / "model.safetensors", path / "pytorch_model/model.safetensors"]
    file = next((p for p in candidates if p.is_file()), None)
    if file is None:
        raise ValueError("Complete single-file native safetensors checkpoint required")
    state = load_file(str(file), device="cpu")
    if state and all(k.startswith("module.") for k in state):
        state = {k[len("module.") :]: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model


def strict_native_evaluation(native, cfg, checkpoint, device="cuda", *, loader=None):
    """Keep native eval construction/precision, but allow a self-contained parent.

    The native evaluation entry point does not forward its constructor's
    config-only flag. Scope that one argument to this isolated load; no inference
    is permitted until the complete checkpoint passes strict restoration.
    """
    load = loader or native.load_model
    constructor = native.TransfusionQwen3p5

    def bundled_constructor(*args, **kwargs):
        kwargs["skip_backbone_weight_load"] = True
        return constructor(*args, **kwargs)

    if cfg["model"].get("skip_backbone_weight_load", False):
        native.TransfusionQwen3p5 = bundled_constructor
    try:
        model = load(cfg, checkpoint, device=device)
        return strict_load(model, checkpoint)
    finally:
        native.TransfusionQwen3p5 = constructor


def main():
    p = argparse.ArgumentParser()
    for name in ("repo", "operation", "config", "checkpoint", "output", "dataset"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--max-sim-agent-steps", type=int, required=True)
    a = p.parse_args()
    from driversi.agents.backends.sources import require_model_root

    require_model_root(a.repo, "augustus")
    root = Path(a.repo) / "AUGUstus_SFT"
    sys.path.insert(0, str(root))
    # No raw shell launcher: no global Ray cleanup, automatic environment changes
    # or automatic latest-checkpoint resume.
    from driversi_agents.augustus.data import manifest, read_cache

    value = manifest(a.dataset)
    for split in ("train", "validation", "development"):
        read_cache(value["splits"][split])
    cfg = yaml.safe_load(Path(a.config).read_text())
    stats = Path(cfg["normalization"]["stats_file"])
    if not stats.is_file():
        raise ValueError("Normalization is mandatory; never recompute on new data")
    utils = importlib.import_module("model.utils")
    from driversi_agents.augustus.normalization import frozen_loader

    utils.load_or_compute_action_norm_stats = frozen_loader(stats)
    utils.compute_pdm_scores_batch = metered(
        utils.compute_pdm_scores_batch,
        Path(a.output) / "scoring-budget.json",
        a.max_sim_agent_steps,
    )
    if a.operation == "train":
        native = importlib.import_module("train_qwen3p5")
        original = native.build_model

        def build(config):
            model = original(config)
            strict_load(model, a.checkpoint)
            print("DRIVERSI_PARENT_STRICT_LOAD_OK", flush=True)
            return model

        native.build_model = build
        sys.argv = [str(root / "train_qwen3p5.py"), "--config", a.config]
        native.main()
        import os

        import torch

        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        if int(os.environ.get("RANK", "0")) != 0:
            return
        # Native saves final under a run timestamp. Copy only the immutable norm
        # file into the new self-contained candidate bundle, never into the parent.
        save_root = Path(cfg["training"]["output_dir"])
        finals = list(save_root.glob("*/final"))
        if len(finals) != 1:
            raise ValueError("Expected exactly one completed final checkpoint")
        training_state = json.loads((finals[0] / "training_state.json").read_text())
        updates = training_state.get("global_step")
        if type(updates) is not int or updates <= 0:
            raise ValueError("Native training produced no optimizer updates")
        cap = cfg["training"].get("max_steps")
        if cap and updates > cap:
            raise ValueError("Native global_step exceeded declared cap")
        import shutil

        shutil.copy2(stats, finals[0] / "action_norm_stats.json")
        for dest in (finals[0] / "pytorch_model",):
            if dest.is_dir():
                shutil.copy2(stats, dest / "action_norm_stats.json")
                shutil.copy2(finals[0] / "config.yaml", dest / "config.yaml")
        (Path(a.output) / "candidate.json").write_text(
            json.dumps(
                {
                    "checkpoint": str(finals[0]),
                    "parent": a.checkpoint,
                    "dataset": a.dataset,
                    "initialization": "strict_full_parent_fresh_optimizer",
                    "updates": updates,
                    "online_self_play": False,
                    "promoted": False,
                },
                indent=2,
            )
        )
    elif a.operation == "evaluate":
        native = importlib.import_module("test_qwen3p5")
        original = native.load_model

        def load(config, checkpoint, device="cuda"):
            return strict_native_evaluation(
                native, config, checkpoint, device, loader=original
            )

        native.load_model = load
        sys.argv = [
            str(root / "test_qwen3p5.py"),
            "--config",
            a.config,
            "--checkpoint",
            a.checkpoint,
            "--batch_eval",
            "--test_data",
            cfg["data"]["val_data_path"],
            "--output_dir",
            a.output,
            "--seed",
            str(cfg["training"]["seed"]),
        ]
        native.main()
    else:
        raise ValueError("Unsupported operation")


if __name__ == "__main__":
    main()
