"""Native Qwen visual feature dump: no text decoder, future images or scores."""

import os
import time
from pathlib import Path

from .observations import file_hash, history_evidence
from .store import encode
from .vectors import VectorStore


def feature_profile(config):
    import torch
    import transformers

    if set(config) != {"model_path", "cameras", "max_side", "image_batch_size"}:
        raise ValueError("Unexpected feature configuration")
    if type(config["image_batch_size"]) is not int or not 1 <= config["image_batch_size"] <= 12:
        raise ValueError("image_batch_size must be 1..12")
    model = Path(config["model_path"]).resolve()
    files = sorted(p for p in model.iterdir() if p.suffix in (".json", ".safetensors"))
    if not any(p.suffix == ".safetensors" for p in files):
        raise ValueError("Local model weights required")
    return {
        "kind": "qwen35_native_vision_features_v1", "config": config,
        "cameras": config["cameras"], "dimension": 2048,
        "checkpoint_files": {p.name: file_hash(p) for p in files},
        "runtime": {"torch": torch.__version__, "transformers": transformers.__version__},
        "source": {name: file_hash(Path(__file__).with_name(name))
                   for name in ("features.py", "vectors.py", "observations.py")},
        "pooling": "post_merger_patch_mean_l2_per_image_then_mean_l2_centroid",
        "storage": {"centroid": "float32", "image_vectors": "float16"},
        "layout": "history_oldest_first_then_configured_camera_order",
        "text_alignment": False, "input_scope": "four_history_frames_only",
    }


class VisionEncoder:
    def __init__(self, config):
        import torch
        from safetensors import safe_open
        from transformers import AutoConfig, AutoProcessor
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

        if not torch.cuda.is_available():
            raise RuntimeError("Allocate an explicit free CUDA device")
        self.config = config
        model_path = Path(config["model_path"]).resolve()
        cfg = AutoConfig.from_pretrained(model_path, local_files_only=True)
        cfg.vision_config._attn_implementation = "sdpa"
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True).image_processor
        # Load only the native visual module; no random feature fallback.
        self.model = Qwen3_5VisionModel(cfg.vision_config)
        state = {}
        prefix = "model.visual."
        for path in sorted(model_path.glob("*.safetensors")):
            with safe_open(path, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    if key.startswith(prefix):
                        name = key[len(prefix):]
                        if name in state:
                            raise ValueError("Duplicate visual checkpoint key")
                        state[name] = handle.get_tensor(key).to(dtype=torch.bfloat16)
        # Assign parameter tensors without casting native nonpersistent buffers:
        # vision rotary frequencies must stay float32, as in from_pretrained.
        self.model.load_state_dict(state, strict=True, assign=True)
        self.model = self.model.eval().to("cuda")

    def __call__(self, images):
        import torch

        outputs = []
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.monotonic()
        for start in range(0, len(images), self.config["image_batch_size"]):
            batch = images[start:start + self.config["image_batch_size"]]
            inputs = self.processor(images=batch, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                result = self.model(
                    inputs["pixel_values"].to(torch.bfloat16),
                    grid_thw=inputs["image_grid_thw"], return_dict=True,
                ).pooler_output
                sizes = (inputs["image_grid_thw"].prod(-1) // self.model.spatial_merge_size**2).tolist()
                outputs.extend(x.float().mean(dim=0) for x in torch.split(result, sizes))
        values = torch.stack(outputs).cpu().numpy()
        torch.cuda.synchronize()
        return values, {"encode_seconds": time.monotonic() - started, "images": len(images),
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated()}


def dump_batch(catalog, config, tokens, output, *, max_scenes=1000, max_seconds=3600):
    import fcntl

    if not 1 <= len(tokens) <= max_scenes <= 10000 or len(tokens) != len(set(tokens)):
        raise ValueError("Explicit nonempty unique bounded token batch required")
    if not 1 <= max_seconds <= 14400:
        raise ValueError("Bounded feature job required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with Path(str(catalog.path) + ".feature.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.monotonic()
        store = VectorStore(catalog, writable=True)
        profile_data = feature_profile(config)
        profile = store.profile(profile_data)
        (output / "profile.json").write_text(encode(profile_data))
        report = {"split": catalog.split, "profile": profile, "requested": len(tokens),
                  "rows": [], "trainable": False, "complete": False}
        backend = None
        for token in tokens:
            if time.monotonic() - started >= max_seconds:
                break
            scene = catalog.scene(token)  # Wrong split/token is a job error, not an inference failure.
            evidence, values, cost, error = [], None, {}, None
            step = time.monotonic()
            try:
                evidence, images = history_evidence(scene, config["cameras"], config["max_side"])
                if store.completed(token, profile, evidence):
                    report["rows"].append({"token": token, "status": "reused"})
                    continue
                if backend is None:
                    backend = VisionEncoder(config)
                values, cost = backend(images)
            except Exception as exc:
                error = type(exc).__name__ + ": " + str(exc)[:2000]
            cost["total_seconds"] = time.monotonic() - step
            try:
                store.put(token, profile, evidence, values, cost, error)
            except ValueError as exc:
                # Invalid native outputs must invalidate a prior ready snapshot,
                # just like inference or image-decoding failures.
                if values is None:
                    raise
                error = type(exc).__name__ + ": " + str(exc)[:2000]
                values = None
                store.put(token, profile, evidence, None, cost, error)
            report["rows"].append({"token": token, "status": "ready" if values is not None else "failed",
                                   "cost": cost, "error": error})
            report["elapsed_seconds"] = time.monotonic() - started
            temporary = output / "report.tmp"
            temporary.write_text(encode(report))
            os.replace(temporary, output / "report.json")
        report["complete"] = len(report["rows"]) == len(tokens)
        report["elapsed_seconds"] = time.monotonic() - started
        report["stats"] = store.stats(profile)
        (output / "report.json").write_text(encode(report))
        return report
