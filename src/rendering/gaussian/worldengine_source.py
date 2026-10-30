"""Render canonical source poses independently of a counterfactual branch."""

import argparse
import json
from pathlib import Path


def run(render_report, output, *, session=None):
    if session is not None:
        return run_cached(render_report, output, session)

    from PIL import Image

    from driversi.agents.backends.native import digest
    from driversi.rendering.diagnostics.worldengine_admission import source_view_psnr
    from driversi.rendering.gaussian.worldengine_portable import PortableRenderer

    render_report, output = Path(render_report), Path(output)
    report = json.loads(render_report.read_text())
    metadata = [Path(p) for p in report["inputs"] if Path(p).name == "video_scene_dict.pkl"]
    checkpoints = [Path(p) for p in report["inputs"] if p.endswith(".ckpt")]
    if len(metadata) != 1 or len(checkpoints) != 1:
        raise ValueError("One bound reconstruction and checkpoint required")
    for path, sha in report["inputs"].items():
        if digest(path) != sha:
            raise ValueError("Selected branch input changed")
    from driversi.contracts.scene import Scene

    scenes = [Path(p) for p in report["inputs"] if Path(p).name == "scene.npz"]
    if len(scenes) != 1:
        raise ValueError("One bound physical scene required")
    scene = Scene.load(scenes[0])
    sensor_root = Path(scene.metadata["navsim_source"]["sensor_root"])
    renderer = PortableRenderer(checkpoints[0], metadata=metadata[0])
    output.mkdir(parents=True, exist_ok=False)
    width, height = report.get("output_image_size", [960, 540])
    if type(width) is not int or width not in (960, 1920) or height != round(width * 1080 / 1920):
        raise ValueError("Bound reconstruction output size required")
    rows, files = [], {}
    inputs = {
        str(render_report): digest(render_report),
        str(metadata[0]): digest(metadata[0]),
        str(checkpoints[0]): digest(checkpoints[0]),
    }
    code = Path(__file__).resolve()
    for frame in report["frames"]:
        source = renderer.binding.resolve(frame["source_token"])
        for camera in ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0"):
            image, evidence = renderer.render(source, camera, width=width)
            path = output / f"{frame['tick']:04d}-{camera}-source-render.png"
            Image.fromarray(image).save(path)
            real = sensor_root / source["cams"][camera]["data_path"]
            factor = float(source["cams"][camera].get("v_adjust", 1.0))
            files[str(path)] = digest(path)
            files[str(real)] = digest(real)
            rows.append(
                dict(
                    tick=frame["tick"],
                    camera=camera,
                    source_token=source["token"],
                    rendered_source_image=str(path),
                    real_source_image=str(real),
                    exposure_factor=factor,
                    measured_psnr_db=source_view_psnr(path, real, factor),
                    source_camera_to_world=evidence["camera_to_world"],
                    source_motions={},
                    source_appearance_timestamp_us=int(source["timestamp"]),
                )
            )
        print("source rendered", frame["tick"], flush=True)
    value = dict(
        schema_version=1,
        output_image_size=[width, height],
        kind="worldengine_source_comparison",
        render_report_sha256=digest(render_report),
        inputs=inputs,
        files=files,
        producer_code={
            str(code): digest(code),
            str(Path(__file__).with_name("worldengine_portable.py")): digest(
                Path(__file__).with_name("worldengine_portable.py")
            ),
        },
        source_mode="canonical_source_pose_without_motion_v1",
        frames=rows,
    )
    (output / "report.json").write_text(json.dumps(value, indent=2))
    return value



def run_cached(render_report, output, session):
    """Same canonical comparison with a verified one-asset session."""
    from driversi.agents.backends.native import digest
    from driversi.rendering.gaussian.source_session import SourceSession
    from driversi.runtime.budget import write_json

    if not isinstance(session, SourceSession):
        raise ValueError("A verified canonical source session is required")
    render_report, output = Path(render_report), Path(output)
    current = json.loads(render_report.read_text())
    for name, sha in current["inputs"].items():
        if digest(name) != sha:
            raise ValueError("Current physical render input changed")
    assets = {name: sha for name, sha in current["inputs"].items()
              if name.endswith(".ckpt") or Path(name).name == "video_scene_dict.pkl"}
    if len(assets) != 2 or any(session.inputs.get(k) != v for k, v in assets.items()):
        raise ValueError("Cached canonical renderer uses another asset")
    path, hit = session.run(render_report, output / "canonical")
    cached = json.loads(path.read_text())
    code = {name: sha for name, sha in session.inputs.items() if name not in assets}
    code[str(Path(__file__).resolve())] = digest(Path(__file__).resolve())
    value = dict(schema_version=1, kind="worldengine_source_comparison",
        output_image_size=current["output_image_size"], render_report_sha256=digest(render_report),
        inputs={str(render_report): digest(render_report), **assets},
        files=cached["files"], producer_code=code,
        source_mode="canonical_source_pose_without_motion_v1", frames=cached["frames"],
        canonical_cache=dict(hit=hit, report=str(path), sha256=digest(path)))
    write_json(output / "report.json", value)
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    import torch

    torch.cuda.set_per_process_memory_fraction(0.15, 0)
    run(args.render_report, args.output)
