"""Bounded WorldDreamer inference on sealed DriveRSI physical conditions."""

import argparse
import json
import sys
import time
from pathlib import Path

# Native Python3.10 process bridge, matching existing training workers.
if __package__ in (None, ""):
    import sys
    import types

    _root = Path(__file__).resolve().parents[4]
    for _name, _folder in (("driversi", "src"),):
        if _name in sys.modules:
            continue
        _module = types.ModuleType(_name)
        _module.__path__ = [str(_root / _folder)]
        sys.modules[_name] = _module
import numpy as np


def activate():
    from driversi.vendor import path as dependency_path
    root = dependency_path("worlddreamer") / "WorldDreamer"
    from driversi.agents.backends.sources import require_source
    require_source(root.parent)
    sys.path[:0] = [
        str(root),
        str(root / "third_party/diffusers/src"),
        str(root / "third_party/nuplan-devkit"),
    ]
    return root


def load_pipeline(weights):
    activate()
    import torch
    from diffusers import UniPCMultistepScheduler
    from projects.dreamer.networks.clip_embedder import FrozenOpenCLIPImageEmbedderV2
    from projects.dreamer.networks.unet_2d_condition_multiview import UNet2DConditionModelMultiview
    from projects.dreamer.networks.unet_addon_single_ref import SingleRefControlNetModel
    from projects.dreamer.pipeline.pipeline_controlnet_single_ref import (
        StableDiffusionSingleRefControlNetPipeline,
    )

    weights = Path(weights)
    checkpoint = weights / "SDv1.5_mv_single_ref_nus_nup/weight-S200000"
    loaded = {}
    for name, cls in [
        ("unet", UNet2DConditionModelMultiview),
        ("controlnet", SingleRefControlNetModel),
    ]:
        model, info = cls.from_pretrained(
            checkpoint / name,
            torch_dtype=torch.float16,
            output_loading_info=True,
            local_files_only=True,
        )
        if any(
            info.get(key)
            for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
        ):
            raise ValueError("Native checkpoint load mismatch: " + name + " " + str(info))
        loaded[name] = model.eval()
    projector = torch.nn.Linear(768, 768)
    projector.load_state_dict(
        torch.load(
            checkpoint / "image_proj/image_proj_model.bin", map_location="cpu", weights_only=True
        ),
        strict=True,
    )
    embedder = FrozenOpenCLIPImageEmbedderV2(
        arch="ViT-B-32", model_path=str(weights / "clip-b32/open_clip_pytorch_model.bin")
    )
    pipe = StableDiffusionSingleRefControlNetPipeline.from_pretrained(
        weights / "base-sd15",
        **loaded,
        image_proj_model=projector.half().eval().cuda(),
        embedder=embedder.half().eval().cuda(),
        safety_checker=None,
        feature_extractor=None,
        torch_dtype=torch.float16,
        local_files_only=True,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    return pipe.to("cuda")


def run(prepared, weights, output):
    activate()
    import torch

    from driversi.agents.backends.native import digest
    from driversi.rsi.plugins.synthesis import check_files

    spec = json.loads(Path(prepared).read_text())
    check_files(spec["files"])
    if spec["kind"] != "worlddreamer_native_conditions":
        raise ValueError("Native conditions required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(
        schema_version=1,
        kind="conditioned_video_output",
        backend="worlddreamer",
        status="loading",
        branch_identity=spec["branch_identity"],
        conditioning_report=str(Path(prepared).resolve()),
        conditioning_sha256=digest(prepared),
        source_control=spec["source_control"],
        history_ticks=[5, 10, 15, 20],
        future_ticks=list(range(25, 71, 5)),
        camera_order=spec["camera_order"],
        frames=[],
        native_recipe=dict(
            steps=50, guidance_scale=2, eta=0, seed=42, resolution=[224, 400], autoregressive=True
        ),
        trainable=False,
        visual_admitted=False,
    )

    def save():
        (output / "report.json").write_text(json.dumps(report, indent=2))

    save()
    start = time.monotonic()
    try:
        pipe = load_pipeline(weights)
        report["status"] = "generating"
        save()
        previous = None
        for frame in spec["frames"]:
            with np.load(frame["conditions"], allow_pickle=False) as data:
                kwargs = {
                    key: torch.as_tensor(data[key], device="cuda", dtype=torch.float16).unsqueeze(0)
                    for key in ["bev_hdmap", "camera_param", "relative_pose", "layout_canvas"]
                }
                kwargs["rel_pose"] = kwargs.pop("relative_pose")
                boxes = {
                    key: torch.as_tensor(data["bbox_" + key], device="cuda").unsqueeze(0)
                    for key in ["bboxes", "classes", "masks"]
                }
                boxes["bboxes"] = boxes["bboxes"].half()
                reference = (
                    torch.as_tensor(
                        data["ref_images"], device="cuda", dtype=torch.float16
                    ).unsqueeze(0)
                    if previous is None
                    else previous
                )
            with torch.inference_mode():
                images = pipe(
                    prompt=[spec["caption"]],
                    **kwargs,
                    ref_images=reference,
                    bev_controlnet_kwargs={"bboxes_3d_data": boxes},
                    height=224,
                    width=400,
                    generator=torch.Generator(device="cuda").manual_seed(42),
                    guidance_scale=2,
                    num_inference_steps=50,
                    eta=0.0,
                    controlnet_conditioning_scale=1.0,
                    guess_mode=False,
                    use_zero_map_as_unconditional=False,
                    bbox_max_length=None,
                ).images[0]
            if len(images) != 6:
                raise ValueError("Native rig output incomplete")
            row = {"tick": frame["tick"], "timestamp_us": frame["timestamp_us"], "cameras": {}}
            for cam, img in zip(spec["camera_order"], images):
                path = output / f"{frame['tick']:04d}-{cam}.png"
                img.save(path)
                row["cameras"][cam] = {
                    "path": str(path),
                    "sha256": digest(path),
                    "size": list(img.size),
                }
            previous = (
                torch.from_numpy(np.stack([np.asarray(img).transpose(2, 0, 1) for img in images]))
                .cuda()
                .half()
                .unsqueeze(0)
                / 127.5
                - 1
            )
            report["frames"].append(row)
            save()
            print("generated", frame["tick"], flush=True)
        report["status"] = "generated_diagnostic"
    except BaseException as ex:
        report.update(status="failed", failure=type(ex).__name__ + ": " + str(ex))
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - start
        save()
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ["prepared", "weights", "output"]:
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    run(a.prepared, a.weights, a.output)
