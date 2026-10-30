"""Strict parent or learned-checkpoint evaluation using the unchanged native evaluator."""

import argparse
import json
import sys
import time
from pathlib import Path

from driversi.agents.backends.native import digest, project_root
from driversi.runtime.budget import write_json

_RUNTIME = {}
_DIGESTS = {}


def cached_digest(path):
    path = Path(path)
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    if key not in _DIGESTS:
        _DIGESTS[key] = digest(path)
    return _DIGESTS[key]


def evaluate(cache, checkpoint, metric_cache, output, *, seed=42, batch_size=2, reuse_model=False):
    import numpy as np
    import torch
    import yaml
    from accelerate import Accelerator
    from torch.utils.data import DataLoader

    from driversi.training.orchestration.state import deterministic
    from driversi_agents.augustus.adapter import merge_recipe
    from driversi_agents.augustus.budget import metered
    from driversi_agents.augustus.data import read_cache
    from driversi_agents.augustus.normalization import frozen_loader
    from driversi_agents.augustus.runner import strict_native_evaluation

    root = project_root()
    sft = root / "agents/augustus/native/AUGUstus_SFT"
    sys.path[:0] = [
        str(sft),
        str(sft / "third_party/navsim"),
        str(sft / "third_party/nuplan-devkit"),
    ]
    import model.utils as utils
    import test_qwen3p5 as native_eval
    import train_qwen3p5 as native

    parent = root / "weights/augustus/SFT"
    output = Path(output)
    block = json.loads(Path(cache).read_text())["block"]
    read_cache(block)
    cfg = merge_recipe(
        yaml.safe_load((sft / "configs/navsim_qwen3p5.yaml").read_text()),
        yaml.safe_load((parent / "config.yaml").read_text()),
    )
    cfg["model"].update(qwen_model_name=str(parent), skip_backbone_weight_load=True)
    cfg["normalization"]["stats_file"] = str(parent / "action_norm_stats.json")
    cfg["data"].update(
        train_data_path=block["cache"],
        val_data_path=block["cache"],
        prompt_file=str(sft / "prompts/default.txt"),
        metric_cache_test=str(metric_cache),
        metric_cache_train=None,
    )
    cfg["training"].update(per_device_eval_batch_size=batch_size, dataloader_num_workers=0)
    identity = dict(
        checkpoint=str(Path(checkpoint).resolve()),
        checkpoint_sha256=cached_digest(Path(checkpoint) / "model.safetensors"),
        cache_sha256=digest(cache),
        seed=seed,
        batch_size=batch_size,
        implementation_sha256=digest(__file__),
        metric_cache=str(metric_cache),
    )
    from driversi_agents.augustus.metric_cache import scoring_files

    identity["metric_files"] = {
        str(p): digest(p) for p in scoring_files(metric_cache, block["records"])
    }
    report = output / "report.json"
    if report.exists():
        old = json.loads(report.read_text())
        if old["inputs"] != identity:
            raise ValueError("Evaluation resume inputs changed")
        for p, h in old["files"].items():
            if digest(p) != h:
                raise ValueError("Evaluation output changed")
        return old
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "inputs.json", identity)
    utils.load_or_compute_action_norm_stats = frozen_loader(parent / "action_norm_stats.json")
    if not utils._init_pdm(metric_cache):
        raise RuntimeError("Native PDM unavailable")
    deterministic(seed)
    started = time.monotonic()
    runtime_key = (identity["checkpoint_sha256"], str(parent))
    if reuse_model and _RUNTIME.get("key") == runtime_key:
        model = _RUNTIME["model"]
    else:
        model = strict_native_evaluation(native_eval, cfg, checkpoint, device="cuda")
        if reuse_model:
            _RUNTIME.update(key=runtime_key, model=model)
    model.processor.video_processor.do_sample_frames = False
    _, raw = native.build_loaders(cfg, model.processor)
    loader = DataLoader(
        raw.dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=raw.collate_fn,
        num_workers=0,
        pin_memory=True,
        generator=torch.Generator().manual_seed(seed),
    )
    acc = Accelerator(mixed_precision=cfg["training"]["mixed_precision"])
    loader = acc.prepare(loader)
    captured = {}
    expected = {x["frame_id"]: x["log_name"] for x in block["records"]}
    scorer = metered(
        utils.compute_pdm_scores_batch, output / "scoring-budget.json", 80 * len(expected)
    )

    def score(predictions, frame_ids, **kwargs):
        if len(frame_ids) != len(expected) or set(frame_ids) != set(expected):
            raise ValueError("Incomplete locked evaluation")
        np.savez_compressed(
            output / "predictions.npz",
            trajectories=predictions,
            tokens=np.asarray(frame_ids, dtype=str),
        )
        kwargs["return_per_sample"] = True
        result = scorer(predictions, frame_ids, **kwargs)
        samples = result["per_sample"]
        if len(samples) != len(expected) or any(x is None for x in samples):
            raise ValueError("Native scoring failure")
        captured["rows"] = [
            dict(token=t, source_log=expected[t], metrics=m)
            for t, m in zip(frame_ids, samples, strict=True)
        ]
        result.pop("per_sample")
        return result

    native.compute_pdm_scores_batch = score
    # Construction/loading must not affect the stochastic action sampler's seed.
    deterministic(seed)
    with torch.no_grad():
        metrics = native.evaluate(
            model,
            loader,
            acc,
            None,
            model.processor,
            step=0,
            log_num_samples=0,
            enable_cot=cfg["cot"]["enable_cot"],
            metric_cache_path=str(metric_cache),
        )
    write_json(output / "samples.json", captured["rows"])
    result = dict(
        status="completed",
        inputs=identity,
        metrics=metrics,
        rows=captured["rows"],
        wall_seconds=time.monotonic() - started,
        full_benchmark=False,
        optimizer_updates=0,
        files={str(output / n): digest(output / n) for n in ("predictions.npz", "samples.json")},
    )
    write_json(report, result)
    return result


def main():
    p = argparse.ArgumentParser()
    for name in ("cache", "checkpoint", "metric-cache", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=2)
    a = p.parse_args()
    v = evaluate(
        a.cache, a.checkpoint, a.metric_cache, a.output, seed=a.seed, batch_size=a.batch_size
    )
    print(json.dumps(dict(status=v["status"], metrics=v["metrics"])))


if __name__ == "__main__":
    main()
