"""Explicit diagnostic behavior cloning through replaceable ActorCritic models."""
import importlib
import inspect
import json
import math
import time
from pathlib import Path

import numpy as np

from driversi.agents.interfaces.config import build_agent, load_inference, read_agent_config
from driversi.data.assets.mtgs_assets import file_sha256
from driversi.runtime.budget import write_json
from driversi.rsi.contracts import _keys
from driversi.rsi.plugins.dataset import PURPOSE, result


def prepare_student(config, base, registry):
    _keys(config, ("purpose", "agent_config", "parent", "seed", "epochs", "batch_size", "learning_rate"), "student config")
    if config["purpose"] != PURPOSE:
        raise ValueError("This adapter only supports explicit diagnostic imitation")
    for key, lo, hi in (("seed", 0, 2**31-1), ("epochs", 1, 1000), ("batch_size", 1, 65536)):
        if type(config[key]) is not int or not lo <= config[key] <= hi:
            raise ValueError("Invalid student " + key)
    rate = config["learning_rate"]
    if type(rate) not in (int, float) or not math.isfinite(rate) or not 0 < rate <= .1:
        raise ValueError("Invalid learning rate")
    cfg = dict(config)
    cfg["agent"] = read_agent_config((base / config["agent_config"]).resolve(strict=True))
    cfg["implementation_files"] = {}
    for key in ("model_target", "inference_target"):
        module_name, name = cfg["agent"][key].split(":")
        module = importlib.import_module(module_name)
        target = getattr(module, name)
        if not isinstance(target, type):
            raise ValueError("Model and inference targets must be classes")
        # A compatibility module may only re-export a class defined elsewhere.
        # Include its real definition and inherited implementations, not just
        # the configured import facade. This is not a transitive environment lock.
        paths = [module.__file__]
        for cls in inspect.getmro(target):
            if cls is object:
                continue
            try:
                paths.append(inspect.getsourcefile(cls))
            except TypeError as exc:
                raise ValueError("Model implementation must have versionable source") from exc
        if any(not path or not Path(path).is_file() for path in paths):
            raise ValueError("Model implementation must have a versionable source file")
        for path in paths:
            cfg["implementation_files"][str(Path(path).resolve())] = file_sha256(path)
    cfg["parent_source_groups"] = []
    if cfg["parent"] is not None:
        cfg["parent"] = str((base / cfg["parent"]).resolve(strict=True))
        cfg["parent_sha256"] = file_sha256(cfg["parent"])
        load_inference(Path(cfg["parent"]), cfg["agent"])
        with np.load(cfg["parent"], allow_pickle=False) as parent:
            metadata = json.loads(str(parent["metadata"]))
        groups = metadata.get("training_source_groups")
        if not isinstance(groups, list) or any(not isinstance(g, str) or not g for g in groups):
            raise ValueError("Parent training provenance required for independent evaluation")
        from driversi.training.sampling.environment import source_log_group
        cfg["parent_source_groups"] = sorted({source_log_group(g) for g in groups})
    return cfg, 0


def run_student(config, inputs, output, registry):
    import torch
    from torch.nn import functional as F

    if any(file_sha256(p) != h for p, h in config["implementation_files"].items()):
        raise ValueError("Model implementation changed since registration")
    source = inputs["dataset"]
    manifest = json.loads(Path(source["artifacts"]["report"]).read_text())
    if config["purpose"] not in manifest.get("allowed_uses", []):
        raise ValueError("Dataset not admitted for the requested learning task")
    from driversi.rsi.execution.registry import ExperimentRegistry
    if (set(manifest["source_groups"]) | set(config["parent_source_groups"])) & ExperimentRegistry(registry).protected_groups():
        raise ValueError("Protected source cannot enter student training")
    if file_sha256(source["artifacts"]["samples"]) != manifest["samples_sha256"]:
        raise ValueError("Labels changed after dataset admission")
    with np.load(source["artifacts"]["samples"], allow_pickle=False) as data:
        x, y = data["observations"].copy(), data["actions"].copy()
    if (x.shape != (manifest["samples"], 1125) or y.shape != (len(x),)
            or not len(x) or not np.isfinite(x).all() or y.dtype.kind not in "iu"
            or (y < 0).any() or (y >= 12).any()):
        raise ValueError("Invalid current observations or executed labels")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(config["seed"])
    model = build_agent(config["agent"]).cpu()
    if config["parent"] is not None:
        if file_sha256(config["parent"]) != config["parent_sha256"]:
            raise ValueError("Parent changed since registration")
        with np.load(config["parent"], allow_pickle=False) as data:
            model.load_state_dict({k: torch.from_numpy(data[k].copy()) for k in data.files if k != "metadata"}, strict=True)
    initial_weights = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def export(path, version, trained=False):
        model.export(path, version)
        with np.load(path, allow_pickle=False) as raw:
            arrays = {k: raw[k].copy() for k in raw.files if k != "metadata"}
            meta = json.loads(str(raw["metadata"]))
        meta.update(training_method="diagnostic_behavior_cloning" if trained else "unmodified_parent", approved_teacher=False,
                    critic_trained=False, harness_dataset=source["summary"]["artifact_reference"]["identity"] if trained else None,
                    training_source_groups=sorted(set(config["parent_source_groups"]) | (set(manifest["source_groups"]) if trained else set())))
        np.savez_compressed(path, metadata=json.dumps(meta), **arrays)

    export(output / "parent.npz", "harness-parent")
    tx, ty = torch.from_numpy(x), torch.from_numpy(y).long()
    model.eval()
    with torch.no_grad():
        initial_loss = float(F.cross_entropy(model(tx)[0], ty))
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    rng = torch.Generator().manual_seed(config["seed"])
    updates, exposures, start = 0, 0, time.monotonic()
    model.train()
    for epoch in range(config["epochs"]):
        permutation = torch.randperm(len(x), generator=rng)
        for indices in permutation.split(config["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(tx[indices])[0], ty[indices])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite student loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            updates += 1
            exposures += len(indices)
            write_json(output / "progress.json", {"updates": updates, "sample_exposures": exposures, "epoch": epoch + 1})
    model.eval()
    with torch.no_grad():
        final_loss = float(F.cross_entropy(model(tx)[0], ty))
    export(output / "candidate.npz", "harness-candidate", trained=True)
    write_json(output / "agent.json", config["agent"])
    report = {"purpose": PURPOSE, "training_method": "behavior_cloning_not_ppo", "device": "cpu",
        "dataset_identity": source["summary"]["artifact_reference"]["identity"],
        "dataset_sha256": manifest["samples_sha256"], "source_groups": sorted(set(manifest["source_groups"]) | set(config["parent_source_groups"])),
        "parent_source_groups": config["parent_source_groups"], "implementation_files": config["implementation_files"],
        "recipe": manifest["recipe"], "samples_consumed": len(x), "sample_exposures": exposures,
        "updates": updates, "initial_loss": initial_loss, "final_loss": final_loss,
        "wall_seconds": time.monotonic() - start, "teacher_qualified": False, "promoted": False,
        "weights_changed": any(not torch.equal(v, initial_weights[k]) for k, v in model.state_dict().items()),
        "parent_sha256": file_sha256(output / "parent.npz"), "candidate_sha256": file_sha256(output / "candidate.npz"),
        "initialization": "provided_checkpoint" if config["parent"] else "seeded_random_diagnostic"}
    write_json(output / "report.json", report)
    return result(output, report, artifacts={name: str(output / filename) for name, filename in {
        "report": "report.json", "parent": "parent.npz", "candidate": "candidate.npz", "agent": "agent.json"}.items()})
