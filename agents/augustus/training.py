"""AUGUstus native main() adapter: no rewritten model, forward loss or optimizer."""

import gc
import json
import shutil
import sys
from pathlib import Path

from driversi.agents.backends.native import asset_files, digest, path_at, project_root
from driversi.configuration import read_mapping
from driversi.training.orchestration.state import (
    EpochSampler,
    deterministic,
    fingerprint,
    rng_state,
)
from driversi.training.sampling.environment import source_log_group


def run(plan):
    import torch
    import yaml
    from accelerate import Accelerator
    from torch.utils.data import DataLoader

    from driversi_agents.augustus.adapter import merge_recipe
    from driversi_agents.augustus.budget import metered
    from driversi_agents.augustus.data import manifest, read_cache
    from driversi_agents.augustus.normalization import frozen_loader
    from driversi_agents.augustus.runner import strict_load, strict_native_evaluation

    root, out = project_root(), Path(plan["output"])
    cfg_model, req, data = plan["model_config"], plan["resolved"]["request"], plan["data"]
    sft = root / cfg_model["repository"] / "AUGUstus_SFT"
    sys.path[:0] = [
        str(sft),
        str(sft / "third_party/navsim"),
        str(sft / "third_party/nuplan-devkit"),
    ]
    import model.utils as utils
    import train_qwen3p5 as native

    # Existing cache smoke provenance is retained. This pilot can measure the
    # current run's log-heldout change, never certify parent-data independence.
    dataset = manifest(data["manifest"], allow_smoke=True)
    for name in ("train", "development"):
        groups = {source_log_group(x["log_name"]) for x in dataset["splits"][name]["records"]}
        if groups != set(plan["source_groups"][name]):
            raise ValueError("AUGUstus native cache log declaration mismatch")
        read_cache(dataset["splits"][name])
    previous = read_mapping(plan["previous"]) if plan["previous"] else None
    metric_cache = data["metric_cache"]
    evaluator = plan.get("evaluation_context")
    if evaluator:
        value = evaluator["value"]
        if digest(data["manifest"]) != value["dataset_manifest_sha256"]:
            raise ValueError("Evaluator refers to a different supervision manifest")
        metric_cache = value["metric_cache"]
    parent = path_at(root, cfg_model["weights"]["parent"])
    cfg = merge_recipe(
        yaml.safe_load((sft / "configs/navsim_qwen3p5.yaml").read_text()),
        yaml.safe_load((parent / "config.yaml").read_text()),
    )
    cfg["model"].update(qwen_model_name=str(parent), skip_backbone_weight_load=True)
    cfg["normalization"]["stats_file"] = str(parent / "action_norm_stats.json")
    cfg["resume"] = {"enabled": False, "resume_path": None}
    cfg["data"].update(
        prompt_file=str(sft / "prompts/default.txt"),
        train_data_path=dataset["splits"]["train"]["cache"],
        val_data_path=dataset["splits"]["development"]["cache"],
        metric_cache_train=None,
        metric_cache_test=metric_cache,
    )
    cap = req["budget"]["max_optimizer_steps"]
    start = previous["total_updates"] if previous and plan["phase"] == "resume" else 0
    cfg["training"].update(
        output_dir=str(out / "checkpoints"),
        per_device_train_batch_size=req["pilot"]["batch_size"],
        per_device_eval_batch_size=req["pilot"]["batch_size"],
        dataloader_num_workers=0,
        max_steps=start + cap,
        save_steps=0,
        eval_steps=0,
        logger=["console"],
        wandb_token="",
        seed=req["pilot"]["seed"],
    )
    if cfg.get("ema", {}).get("enabled", False):
        raise ValueError("This single-process pilot does not support EMA recipe recovery")
    deterministic(req["pilot"]["seed"])
    (out / "native.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    utils.load_or_compute_action_norm_stats = frozen_loader(parent / "action_norm_stats.json")
    budget_file = out / "scoring-budget.json"
    native.compute_pdm_scores_batch = metered(
        utils.compute_pdm_scores_batch, budget_file, req["budget"]["max_scoring_agent_steps"]
    )
    original_build = native.build_model

    def build_model(c):
        model = strict_load(original_build(c), parent)
        if plan["phase"] != "evaluate":

            def observed_forward(module, args, kwargs):
                ids = kwargs.get("_frame_id")
                if not isinstance(ids, list) or not ids or any(not isinstance(x, str) for x in ids):
                    raise ValueError("Native training frame identities missing")
                captured.setdefault("pending_samples", []).extend(ids)

            model.register_forward_pre_hook(observed_forward, with_kwargs=True)
        return model

    native.build_model = build_model

    if plan["phase"] == "evaluate" and data.get("full_evaluation_plan"):
        from driversi_agents.augustus.full_pair_evaluation import run as full_evaluate

        return full_evaluate(plan, previous)

    if plan["phase"] == "evaluate":
        import test_qwen3p5 as native_eval

        from driversi_agents.augustus.metric_cache import scoring_files

        files = scoring_files(metric_cache, dataset["splits"]["development"]["records"])
        if evaluator and set(files) != set(evaluator["value"]["files"]):
            raise ValueError("Evaluator did not seal exactly the consumed scoring files")
        if not utils._init_pdm(metric_cache):
            raise RuntimeError("Native PDM dependencies unavailable before model evaluation")
        acc = Accelerator(mixed_precision=cfg["training"]["mixed_precision"])
        metrics = {}
        paired_metrics = {}
        source_logs = {
            row["frame_id"]: row["log_name"] for row in dataset["splits"]["development"]["records"]
        }
        for name, checkpoint in (("parent", parent), ("candidate", previous["checkpoint"])):
            deterministic(req["pilot"]["seed"])
            model = strict_native_evaluation(native_eval, cfg, checkpoint, device="cuda")
            if hasattr(model.processor, "video_processor"):
                model.processor.video_processor.do_sample_frames = False
            _, raw_loader = native.build_loaders(cfg, model.processor)
            loader = DataLoader(
                raw_loader.dataset,
                batch_size=raw_loader.batch_size,
                shuffle=False,
                collate_fn=raw_loader.collate_fn,
                num_workers=0,
                pin_memory=True,
                generator=torch.Generator().manual_seed(req["pilot"]["seed"]),
            )
            loader = acc.prepare(loader)
            # Preserve predictions before expensive scoring, including failed attempts.
            scorer = metered(
                utils.compute_pdm_scores_batch,
                budget_file,
                req["budget"]["max_scoring_agent_steps"],
            )

            def score_and_record(predictions, frame_ids, **kwargs):
                import numpy as np

                np.savez_compressed(
                    out / (name + "-pdm-inputs.npz"),
                    trajectories=predictions,
                    tokens=np.asarray(frame_ids, dtype=str),
                )
                wants_samples = kwargs.get("return_per_sample", False)
                kwargs["return_per_sample"] = True
                result = scorer(predictions, frame_ids, **kwargs)
                samples = result["per_sample"]
                if len(samples) != len(frame_ids) or any(row is None for row in samples):
                    raise ValueError("Incomplete per-sample native evaluation evidence")
                if set(frame_ids) != set(source_logs) or len(frame_ids) != len(source_logs):
                    raise ValueError("Evaluation samples differ from heldout supervision")
                paired_metrics[name] = dict(zip(frame_ids, samples, strict=True))
                (out / (name + "-pdm-samples.json")).write_text(
                    json.dumps(
                        [
                            {"sample_id": token, "source_log": source_logs[token], "metrics": row}
                            for token, row in paired_metrics[name].items()
                        ]
                    )
                )
                if not wants_samples:
                    result.pop("per_sample")
                return result

            native.compute_pdm_scores_batch = score_and_record
            # Loading and loader initialization must not consume the evaluation seed.
            deterministic(req["pilot"]["seed"])
            with torch.no_grad():
                metrics[name] = native.evaluate(
                    model,
                    loader,
                    acc,
                    None,
                    model.processor,
                    step=0,
                    log_num_samples=0,
                    enable_cot=cfg["cot"]["enable_cot"],
                    metric_cache_path=metric_cache,
                )
            del model, loader
            gc.collect()
            torch.cuda.empty_cache()
        if not all("PDMS" in x for x in metrics.values()):
            raise ValueError("Official PDM result missing; no reward fallback allowed")
        spent = read_mapping(budget_file)
        return {
            "status": "completed",
            "reason": None,
            "metrics": metrics,
            "directions": {k: 1 for k in ("PDMS", "NC", "DAC", "DDC", "TTC", "C")},
            "primary_metric": "PDMS",
            "safety_higher": ["NC", "DAC", "DDC", "TTC"],
            "metric_scope": (
                "native_navsim_pdm_on_locked_official_navtest_pilot"
                if dataset.get("benchmark", {}).get("protocol")
                == "official_navtest_v1_locked_pilot"
                else "native_navsim_pdm_on_current_run_log_heldout"
            ),
            "evaluated_sample_ids": list(source_logs),
            "paired_samples": [
                {
                    "sample_id": token,
                    "source_log": source_logs[token],
                    "parent": paired_metrics["parent"][token],
                    "candidate": paired_metrics["candidate"][token],
                }
                for token in source_logs
            ],
            "evaluated_checkpoint": previous["checkpoint"],
            "consumed": {
                "environment_steps": 0,
                "scoring_agent_steps": spent["reserved_agent_steps"],
            },
        }

    captured = {}
    original_loaders = native.build_loaders

    def loaders(c, processor):
        train, val = original_loaders(c, processor)
        sampler = EpochSampler(train.dataset, req["pilot"]["seed"])
        gen = torch.Generator().manual_seed(req["pilot"]["seed"])
        train = DataLoader(
            train.dataset,
            batch_size=train.batch_size,
            sampler=sampler,
            generator=gen,
            collate_fn=train.collate_fn,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
        )
        captured["sampler"] = sampler
        return train, val

    native.build_loaders = loaders
    original_accelerator = native.Accelerator

    def accelerator(*args, **kwargs):
        acc = original_accelerator(*args, **kwargs)
        original_prepare = acc.prepare

        def prepare(*objects, **kw):
            for obj in objects:
                if isinstance(obj, torch.optim.Optimizer):
                    captured["optimizer"] = obj
                    captured["updates"] = 0

                    def post(*a, **k):
                        captured["updates"] += 1
                        row = {
                            "phase_update": captured["updates"],
                            "global_update": start + captured["updates"],
                            "sample_ids": captured.pop("pending_samples", []),
                            "learning_rates": [g["lr"] for g in captured["optimizer"].param_groups],
                            "scope": "forwards_since_previous_actual_optimizer_update",
                        }
                        with (out / "consumed-training-samples.jsonl").open("a") as stream:
                            stream.write(json.dumps(row) + "\n")

                    obj.register_step_post_hook(post)
            result = original_prepare(*objects, **kw)
            captured["acc"] = acc
            return result

        acc.prepare = prepare
        return acc

    native.Accelerator = accelerator
    original_resume = native.try_resume

    def resume(acc, args, tcfg):
        if previous:
            checkpoint = Path(previous["checkpoint"])
            extra = read_mapping(checkpoint / "recovery.json")
            if extra["context"] != plan["context"] or extra["global_step"] != start:
                raise ValueError("AUGUstus recovery context/cursor mismatch")
            for name in ("optimizer.bin", "scheduler.bin", "random_states_0.pkl"):
                if not (checkpoint / name).is_file():
                    raise ValueError("Incomplete full training state: " + name)
            acc.load_state(str(checkpoint))
            if acc._dataloaders:
                acc._dataloaders[0].set_epoch(start // len(acc._dataloaders[0]))
            return start
        return original_resume(acc, args, tcfg)

    native.try_resume = resume
    original_save = native.save_checkpoint

    def save(acc, tcfg, global_step, tag="", **kwargs):
        # The native loop's final artifact normally omits optimizer state.
        # Make it resumable through the existing native save function.
        kwargs["save_optimizer"] = True
        original_save(acc, tcfg, global_step, tag=tag, **kwargs)
        checkpoint = Path(tcfg["output_dir"]) / (tag or f"checkpoint-{global_step}")
        shutil.copy2(parent / "action_norm_stats.json", checkpoint / "action_norm_stats.json")
        (checkpoint / "recovery.json").write_text(
            json.dumps(
                {
                    "context": plan["context"],
                    "global_step": global_step,
                    "cursor": "native_global_step_epoch_offset",
                    "sampler": "epoch_seeded_permutation",
                }
            )
        )
        model = kwargs["model"]
        state = {
            "model": model.state_dict(),
            "optimizer": captured["optimizer"].state_dict(),
            "scheduler": acc._schedulers[0].state_dict(),
            "rng": rng_state(),
        }
        captured.update(
            checkpoint=str(checkpoint),
            total_updates=global_step,
            state_fingerprint=fingerprint(state),
        )

    native.save_checkpoint = save
    sys.argv = [str(sft / "train_qwen3p5.py"), "--config", str(out / "native.yaml")]
    native.main()
    return {
        "status": "completed",
        "reason": None,
        "checkpoint": captured["checkpoint"],
        "recovery_checkpoint": captured["checkpoint"],
        "checkpoint_files": {p: digest(p) for p in asset_files(captured["checkpoint"])},
        "updates": captured["updates"],
        "total_updates": captured["total_updates"],
        "state_fingerprint": captured["state_fingerprint"],
        "resume_from": previous["checkpoint"] if previous else None,
        "checkpoint_state": [
            "model",
            "optimizer",
            "scheduler",
            "rng",
            "native_step_cursor",
            "epoch_seeded_sampler",
        ],
        "consumed": {"environment_steps": 0, "scoring_agent_steps": 0},
    }
