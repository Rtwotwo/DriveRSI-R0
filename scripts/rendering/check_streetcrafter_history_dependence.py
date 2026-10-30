"""Native same-seed future-conditioning probe; outputs can never train a student."""

import argparse
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
for name, folder in (("driversi", "src"), ("driversi_agents", "agents")):
    module = types.ModuleType(name)
    module.__path__ = [str(ROOT / folder)]
    sys.modules[name] = module



def run(receipt_path, checkpoint, output):
    import torch

    from driversi.agents.backends.native import digest
    from driversi.rendering.video.streetcrafter.runtime import load_native

    receipt = json.loads(Path(receipt_path).read_text())
    if not receipt.get("gs_refinement_performed") or receipt["generated_ticks"] != list(range(25)):
        raise ValueError("Completed actual 25-frame GS refinement required")
    if digest(checkpoint) != receipt["checkpoint_sha256"]:
        raise ValueError("Checkpoint mismatch")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    report = dict(kind="native_history_future_dependence_probe", trainable=False,
                  physical_branch_changed=False, generated_probe_is_valid_scene=False,
                  source_receipt=str(receipt_path), source_sha256=digest(receipt_path),
                  checkpoint_sha256=receipt["checkpoint_sha256"], status="preparing",
                  native_recipe=receipt["native_recipe"], rounds=[],
                  intervention="Only ticks21..24 GS and colored guides set to black; all input masks unchanged",
                  history_ticks=[5, 10, 15, 20], changed_future_ticks=[21, 22, 23, 24])

    def save():
        (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))

    save()

    def read(row, mode):
        if digest(row["path"]) != row["sha256"]:
            raise ValueError("Changed saved input")
        with Image.open(row["path"]) as image:
            if image.mode != mode or image.size != (1024, 576):
                raise ValueError("Native input dimensions required")
            return np.asarray(image).copy()

    gs = np.stack([read(x["gs"], "RGB") for x in receipt["gs_inputs"]])
    guides = np.stack([read(x, "RGB") for x in receipt["guide_inputs"]])
    render_masks = np.stack([read(x["render_mask"], "L") for x in receipt["gs_inputs"]])
    guide_masks = np.stack([read(x["guide_mask"], "L") for x in receipt["gs_inputs"]])
    report["input_sha256"] = {
        row["path"]: row["sha256"] for item in receipt["gs_inputs"]
        for row in [item["gs"], item["render_mask"], item["guide_mask"]]
    }
    report["input_sha256"].update({x["path"]: x["sha256"] for x in receipt["guide_inputs"]})
    report["status"] = "load_native"
    save()
    model = load_native(Path(checkpoint).resolve())
    results = {}
    try:
        for name in ("baseline", "repeat", "future_changed"):
            img, guide = gs.copy(), guides.copy()
            if name == "future_changed":
                img[21:] = 0
                guide[21:] = 0
            assert np.array_equal(img[:21], gs[:21]) and np.array_equal(guide[:21], guides[:21])
            batch = dict(
                img_seq=torch.from_numpy(img.transpose(0, 3, 1, 2).astype(np.float32) / 127.5 - 1),
                guide_seq=torch.from_numpy(guide.transpose(0, 3, 1, 2).astype(np.float32) / 127.5 - 1),
                img_mask_seq=torch.from_numpy(render_masks[:, None].astype(np.float32) / 255),
                guide_mask_seq=torch.from_numpy(guide_masks[:, None].astype(np.float32) / 255),
                training_free_guidance=True,
            )
            report["status"] = name
            save()
            start = time.monotonic()
            # Native forward resets all random seeds to model.seed on every invocation.
            values = model.forward(batch, scale=0.3, cond_indices=[0]).detach().float().cpu()
            results[name] = values.numpy()
            report["rounds"].append(dict(name=name, seconds=time.monotonic() - start))
            for tick in report["history_ticks"]:
                pixels = (results[name][tick].transpose(1, 2, 0) * 255).round().astype(np.uint8)
                Image.fromarray(pixels).save(out / f"{name}-{tick:04d}.png")
            save()
        floor = np.abs(results["baseline"] - results["repeat"])
        effect = np.abs(results["baseline"] - results["future_changed"])
        report["per_history_tick"] = [dict(
            tick=tick, repeat_mae_255=float(floor[tick].mean() * 255),
            future_change_mae_255=float(effect[tick].mean() * 255),
            repeat_max_255=float(floor[tick].max() * 255),
            future_change_max_255=float(effect[tick].max() * 255),
        ) for tick in report["history_ticks"]]
        report["history_depends_on_future_observed"] = any(
            x["future_change_mae_255"] > x["repeat_mae_255"] + 1e-6
            for x in report["per_history_tick"]
        )
        report["status"] = "completed"
        report["limitations"] = [
            "Probe input intervention is not a realistic scene or training example",
            "Single front-camera recipe; detects dependence, not general video-model causality",
            "Terrain ambiguity in original diagnostic inputs does not change identical-history test",
        ]
    except BaseException as exc:
        report.update(status="failed", failure=repr(exc))
        raise
    finally:
        save()
    print(json.dumps(report["per_history_tick"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("receipt", "checkpoint", "output"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    run(args.receipt, args.checkpoint, args.output)
