"""Native StreetCrafter sampling; no training or image-admission authority."""

import argparse
import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    import types

    root = Path(__file__).resolve().parents[4]
    for name, folder in (("driversi", "src"),):
        if name in sys.modules:
            continue
        module = types.ModuleType(name)
        module.__path__ = [str(root / folder)]
        sys.modules[name] = module


def load_native(checkpoint):
    import torch
    from omegaconf import OmegaConf
    from safetensors.torch import load_file

    from driversi.vendor import path as dependency_path
    vendor = dependency_path("streetcrafter") / "video_diffusion"
    from driversi.agents.backends.sources import require_source
    require_source(vendor.parent)
    sys.path.insert(0, str(vendor))
    import sample_condition as native

    def strict_loader(config, ckpt):
        config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
        # The complete official checkpoint must supply CLIP too. Never download
        # an unrecorded replacement backbone or accept random missing tensors.
        for entry in config.model.params.conditioner_config.params.emb_models:
            if entry.input_key == "cond_frames_without_noise":
                entry.params.open_clip_embedding_config.params.version = None
        model = native.instantiate_from_config(config.model)
        state = (
            load_file(ckpt)
            if ckpt.endswith(".safetensors")
            else torch.load(ckpt, map_location="cpu", weights_only=True)["state_dict"]
        )
        model.load_state_dict(state, strict=True)
        return native.initial_model_load(model).eval()

    original = native.load_model_from_config
    try:
        native.load_model_from_config = strict_loader
        return native.VideoDiffusionModel(
            config_path=str(vendor / "configs/inference/waymo_high_res.yaml"),
            ckpt_path=str(checkpoint),
            height=576,
            width=1024,
            n_steps=50,
            cfg_scale=2.5,
            seed=23,
        )
    finally:
        native.load_model_from_config = original


def run(prepared, checkpoint, output):
    import numpy as np
    import torch
    from PIL import Image

    from driversi.agents.backends.native import digest
    from driversi.rendering.video.bundle import clip_ranges, validate_bundle
    from driversi.rsi.plugins.synthesis import check_files

    spec = json.loads(Path(prepared).read_text())
    check_files(spec["files"])
    bundle = validate_bundle(spec["bundle"])
    if (
        spec["kind"] != "streetcrafter_native_conditions"
        or spec["branch_identity"] != bundle["branch_identity"]
    ):
        raise ValueError("Sealed same-branch conditions required")
    if [row["tick"] for row in spec["frames"]] != list(range(71)):
        raise ValueError("All actual 10Hz ticks required; no padding")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(
        kind="conditioned_video_output",
        schema_version=1,
        backend="streetcrafter",
        status="loading",
        source_control=False,
        branch_identity=spec["branch_identity"],
        conditioning_report=str(Path(prepared).resolve()),
        conditioning_sha256=digest(prepared),
        camera_order=spec["camera_order"],
        history_ticks=bundle["history_ticks"],
        future_ticks=bundle["future_ticks"],
        frames=[],
        overlap_checks=[],
        native_recipe=dict(
            steps=50,
            guidance_scale=2.5,
            seed=23,
            fps=10,
            frames_per_clip=25,
            resolution=[576, 1024],
            joint_multiview=False,
        ),
        trainable=False,
        visual_admitted=False,
    )

    def save():
        (output / "report.json").write_text(json.dumps(report, indent=2))

    start = time.monotonic()
    save()
    try:
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError("Official StreetCrafter checkpoint missing: " + str(checkpoint))
        report["checkpoint_sha256"] = digest(checkpoint)
        model = load_native(checkpoint)
        rows = {
            row["tick"]: dict(tick=row["tick"], timestamp_us=row["timestamp_us"], cameras={})
            for row in spec["frames"]
        }
        for cam in spec["camera_order"]:
            generated = {}
            initial = (
                np.asarray(
                    Image.open(spec["initial_reference_images"][cam])
                    .convert("RGB")
                    .resize((1024, 576)),
                    np.float32,
                )
                / 255
            )
            for ticks in clip_ranges(range(71)):
                images = np.zeros((25, 3, 576, 1024), np.float32)
                images[0] = (initial if ticks[0] == 0 else generated[ticks[0]]).transpose(
                    2, 0, 1
                ) * 2 - 1
                guides = np.stack(
                    [
                        np.asarray(
                            Image.open(spec["frames"][t]["cameras"][cam]["guide"]), np.float32
                        ).transpose(2, 0, 1)
                        / 127.5
                        - 1
                        for t in ticks
                    ]
                )
                batch = dict(
                    img_seq=torch.from_numpy(images),
                    guide_seq=torch.from_numpy(guides),
                    training_free_guidance=False,
                )
                samples = (
                    model.forward(batch, scale=1.0, cond_indices=[0])
                    .detach()
                    .float()
                    .cpu()
                    .permute(0, 2, 3, 1)
                    .numpy()
                )
                if samples.shape != (25, 576, 1024, 3) or not np.isfinite(samples).all():
                    raise ValueError("Native clip output malformed")
                for tick, pixels in zip(ticks, samples):
                    if tick in generated:
                        report["overlap_checks"].append(
                            dict(
                                camera=cam,
                                tick=tick,
                                mean_abs_difference=float(np.abs(generated[tick] - pixels).mean()),
                            )
                        )
                        continue
                    generated[tick] = pixels
                    path = output / f"{tick:04d}-{cam}.png"
                    Image.fromarray((pixels * 255).round().clip(0, 255).astype(np.uint8)).save(path)
                    rows[tick]["cameras"][cam] = dict(
                        path=str(path), sha256=digest(path), size=[1024, 576]
                    )
                print("generated", cam, ticks[0], ticks[-1], flush=True)
                report["frames"] = [
                    rows[t] for t in bundle["history_ticks"] + bundle["future_ticks"]
                ]
                save()
        report["status"] = "generated_diagnostic"
    except BaseException as exc:
        report.update(status="failed", failure=type(exc).__name__ + ": " + str(exc))
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - start
        save()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("prepared", "checkpoint", "output"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    run(args.prepared, args.checkpoint, args.output)
