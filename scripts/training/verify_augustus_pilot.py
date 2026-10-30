"""Bounded AUGUstus native-component update/recovery test; never a benchmark.

This deliberately does not relax the production dataset provenance gate. It uses
four real training examples, two optimizer calls and one replayed suffix. Native
DeepSpeed launch and shuffled sampler continuation are outside this test's scope.
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Require deterministic GPU kernels for exact replay diagnostics",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    if not output.is_relative_to(root / "outputs") or args.gpu < 0:
        raise ValueError("Explicit project output and nonnegative GPU required")
    output.mkdir(parents=True, exist_ok=False)
    checkpoint_root = root / "weights/runs/augustus" / output.relative_to(root / "outputs")
    checkpoint_root.mkdir(parents=True, exist_ok=False)
    os.environ.update(
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        WANDB_MODE="disabled",
        SWANLAB_MODE="disabled",
    )
    if args.deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import numpy as np
    import torch
    import yaml
    from accelerate import Accelerator
    from accelerate.utils import send_to_device

    from driversi.agents.backends.sources import require_model_root, require_source
    from driversi_agents.augustus.adapter import complete_assets, merge_recipe
    from driversi_agents.augustus.data import manifest, read_cache, validate_row
    from driversi_agents.augustus.normalization import frozen_loader
    from driversi_agents.augustus.runner import strict_load

    repo = root / "agents/augustus/native"
    require_model_root(repo, "augustus")
    require_source(repo)
    sft = repo / "AUGUstus_SFT"
    sys.path.insert(0, str(sft))
    parent = root / "weights/augustus/SFT"
    missing = [p for p in complete_assets(parent) if not Path(p).is_file()]
    if missing:
        raise ValueError("Incomplete parent: " + repr(missing))
    data = manifest(args.dataset, allow_smoke=True)
    records = data["splits"]["train"]["records"]
    if len(records) < 4:
        raise ValueError("Four distinct real training samples required")
    raw = read_cache(data["splits"]["train"])
    for index in range(4):
        validate_row(raw[index])
    import model.utils as utils

    utils.load_or_compute_action_norm_stats = frozen_loader(parent / "action_norm_stats.json")
    import train_qwen3p5 as native
    from model.dataset import TransfusionCollator, TransfusionHFCacheDataset

    cfg = merge_recipe(
        yaml.safe_load((sft / "configs/navsim_qwen3p5.yaml").read_text()),
        yaml.safe_load((parent / "config.yaml").read_text()),
    )
    cfg["model"].update(qwen_model_name=str(parent), skip_backbone_weight_load=True)
    cfg["normalization"]["stats_file"] = str(parent / "action_norm_stats.json")
    cfg["data"].update(
        prompt_file=str(sft / "prompts/default.txt"),
        train_data_path=data["splits"]["train"]["cache"],
    )
    cfg["training"].update(
        output_dir=str(checkpoint_root),
        per_device_train_batch_size=2,
        dataloader_num_workers=0,
        max_steps=2,
    )
    (checkpoint_root / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (output / "native.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    report = {
        "scope": "native_component_training_and_fixed_batch_recovery_smoke",
        "production_entrypoint_exercised": False,
        "deterministic_algorithms": args.deterministic,
        "recovery_process_scope": "in_process_same_objects_fixed_batches",
        "recovery_criterion": "byte_exact_state_and_exact_loss_same_environment",
        "parent_training_provenance": data.get("parent_provenance", "declared"),
        "independent_evaluation": False,
        "promoted": False,
        "trainable": False,
        "optimizer_calls": 0,
        "replayed_optimizer_calls": 0,
        "unique_samples": 4,
        "batch_size": 2,
        "checkpoint_root": str(checkpoint_root),
        "passed": False,
    }
    started = time.monotonic()

    def save_report():
        report["wall_seconds"] = time.monotonic() - started
        (output / "report.json").write_text(json.dumps(report, indent=2))

    def fingerprint(value):
        if torch.is_tensor(value):
            tensor = value.detach().contiguous().cpu()
            return {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "sha256": hashlib.sha256(
                    memoryview(tensor.reshape(-1).view(torch.uint8).numpy())
                ).hexdigest(),
            }
        if isinstance(value, np.ndarray):
            return {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
            }
        if isinstance(value, dict):
            return {str(k): fingerprint(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [fingerprint(v) for v in value]
        if isinstance(value, np.generic):
            return value.item()
        return value

    try:
        torch.set_num_threads(8)
        if args.deterministic:
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        accelerator = Accelerator(
            mixed_precision=cfg["training"]["mixed_precision"], gradient_accumulation_steps=1
        )
        if accelerator.num_processes != 1:
            raise ValueError("This bounded probe requires exactly one process")
        native.set_seed(cfg["training"]["seed"])
        model = strict_load(native.build_model(cfg), parent)
        print("STRICT_PARENT_LOADED", flush=True)
        model.qwen_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if getattr(model.processor, "video_processor", None) is not None:
            model.processor.video_processor.do_sample_frames = False
        dataset = TransfusionHFCacheDataset(
            data["splits"]["train"]["cache"],
            model.processor,
            cfg["data"]["prompt_file"],
            traj_token=8,
            action_dim=3,
            is_training=True,
            enable_cot=True,
            cot_components=["prediction"],
        )
        collate = TransfusionCollator(model.processor, traj_token=8, action_dim=3)
        batches = []
        for first in (0, 2):
            for index in (first, first + 1):
                prompt = dataset._build_prompt(raw[index])
                assert raw[index]["solution_prediction"] not in prompt
                assert raw[index]["solution_planning"] not in prompt
            batches.append(collate([dataset[first], dataset[first + 1]]))
        report["tokens"] = [raw[i]["frame_id"] for i in range(4)]
        report["future_in_prompt"] = False
        tcfg = cfg["training"]
        optimizer = torch.optim.AdamW(
            model.get_param_groups(
                tcfg["weight_decay"],
                backbone_lr_scale=tcfg.get("backbone_lr_scale", 1.0),
                base_lr=tcfg["learning_rate"],
            ),
            lr=tcfg["learning_rate"],
            betas=(0.9, 0.95),
            eps=1e-8,
        )
        scheduler = native.get_cosine_schedule_with_warmup(
            optimizer,
            tcfg["warmup_steps"],
            (len(dataset) // 2) * tcfg["num_epochs"],
            tcfg["min_lr"] / tcfg["learning_rate"],
        )
        model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
        batches = [send_to_device(batch, accelerator.device) for batch in batches]
        model.train()

        def state():
            return {
                "model": fingerprint(accelerator.unwrap_model(model).state_dict()),
                "optimizer": fingerprint(optimizer.state_dict()),
                "scheduler": fingerprint(scheduler.state_dict()),
                "torch_rng": fingerprint(torch.get_rng_state()),
                "cuda_rng": fingerprint(torch.cuda.get_rng_state_all()),
                "numpy_rng": fingerprint(np.random.get_state()),
                "python_rng": fingerprint(random.getstate()),
            }

        def update(batch):
            rates = [g["lr"] for g in optimizer.param_groups]
            with accelerator.accumulate(model):
                prediction = model(**batch)
                if not torch.isfinite(prediction.loss):
                    raise ValueError("Nonfinite native joint loss")
                accelerator.backward(prediction.loss)
                norm = accelerator.clip_grad_norm_(model.parameters(), tcfg["max_grad_norm"])
                if not torch.isfinite(norm):
                    raise ValueError("Nonfinite gradient norm")
                optimizer.step()
                if accelerator.optimizer_step_was_skipped:
                    raise ValueError("Optimizer update was skipped")
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            item = {
                "loss": float(prediction.loss.detach()),
                "diffusion_loss": float(prediction.diffusion_loss.detach()),
                "language_loss": float(prediction.lm_loss.detach()),
                "gradient_norm": float(norm),
                "learning_rates": rates,
            }
            print("UPDATE", json.dumps(item), flush=True)
            return item

        parameter_names = set(dict(accelerator.unwrap_model(model).named_parameters()))
        initial = fingerprint(accelerator.unwrap_model(model).state_dict())
        first = update(batches[0])
        report["optimizer_calls"] = 1
        save_report()
        snap = state()
        native.save_checkpoint(
            accelerator,
            tcfg,
            1,
            model=accelerator.unwrap_model(model),
            processor=accelerator.unwrap_model(model).processor,
            save_optimizer=True,
        )
        print("FULL_CHECKPOINT_SAVED", flush=True)
        second = update(batches[1])
        report["optimizer_calls"] = 2
        save_report()
        uninterrupted = state()
        native.save_checkpoint(
            accelerator,
            tcfg,
            2,
            tag="uninterrupted",
            model=accelerator.unwrap_model(model),
            processor=accelerator.unwrap_model(model).processor,
            save_optimizer=False,
        )
        accelerator.load_state(str(checkpoint_root / "checkpoint-1"))
        restored = state()
        report["snapshot_restore"] = {k: snap[k] == restored[k] for k in snap}
        if not all(report["snapshot_restore"].values()):
            raise ValueError("Checkpoint restoration differs: " + repr(report["snapshot_restore"]))
        print("EXACT_SNAPSHOT_RESTORED", flush=True)
        replayed = update(batches[1])
        report["replayed_optimizer_calls"] = 1
        actual = state()
        report["suffix_exact"] = {k: actual[k] == uninterrupted[k] for k in actual}
        report["loss_suffix_exact"] = second == replayed
        report["model_tensors_checked"] = len(actual["model"])
        report["changed_model_tensors"] = sum(initial[k] != actual["model"][k] for k in initial)
        report["updates"] = [first, second]
        report["replayed_update"] = replayed
        report["changed_parameter_tensors"] = sum(
            initial[k] != actual["model"][k] for k in parameter_names
        )
        report["parameter_update_observed"] = report["changed_parameter_tensors"] > 0
        report["max_gpu_memory_bytes"] = torch.cuda.max_memory_allocated()
        report["runtime_origins"] = {
            name: str(Path(module.__file__).resolve())
            for name, module in list(sys.modules.items())
            if getattr(module, "__file__", None)
            and (name == "train_qwen3p5" or name.split(".")[0] in {"model", "navsim", "nuplan"})
        }
        if any(
            Path(path).is_relative_to(root / "clones")
            for path in report["runtime_origins"].values()
        ):
            raise ValueError("Runtime source imported from clones")
        (output / "state-fingerprints.json").write_text(
            json.dumps(
                {
                    "snapshot": snap,
                    "uninterrupted": uninterrupted,
                    "restored": restored,
                    "replayed": actual,
                },
                indent=2,
            )
        )
        if not all(report["suffix_exact"].values()) or not report["loss_suffix_exact"]:
            raise ValueError("Fixed-batch replay differs; exact recovery not established")
        if not report["parameter_update_observed"]:
            raise ValueError("No parameter change observed")
        report["passed"] = True
        print("BOUNDED_NATIVE_RECOVERY_OK", flush=True)
    except Exception:
        report["error"] = traceback.format_exc()
        print(report["error"], flush=True)
        raise
    finally:
        save_report()


if __name__ == "__main__":
    main()
