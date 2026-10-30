"""Asset-bound canonical source reuse; pixels and current input hashes are verified."""

import json
from pathlib import Path

from driversi.runtime.budget import write_json as write

CAMERAS = ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0")


def replay_adapter(renderer):
    """Adapt the existing four-call replay interface without rewriting it."""

    class Adapter:
        def __init__(self):
            self._key = None
            self._pending = {}

        def __getattr__(self, name):
            return getattr(renderer, name)

        def render(self, source, camera_id, **kwargs):
            source = renderer.binding.resolve(source)
            key = json.dumps([source["token"], kwargs], sort_keys=True)
            if key != self._key or camera_id not in self._pending:
                self._pending = renderer.render_many(source, CAMERAS, **kwargs)
                self._key = key
            return self._pending.pop(camera_id)

    return Adapter()


class SourceSession:
    """Verified canonical-source cache, scoped to one immutable renderer asset."""

    def __init__(self, renderer, checkpoint, metadata, code_files, image_writer=None):
        from driversi.agents.backends.native import digest

        self.renderer = renderer
        self.image_writer = image_writer
        self.inputs = {str(p): digest(p) for p in [checkpoint, metadata, *code_files]}
        self.entries = {}

    def cached_report(self, key):
        from driversi.agents.backends.native import digest

        path, original_digest = self.entries[key]
        if digest(path) != original_digest:
            raise ValueError("Cached source report changed")
        return path, json.loads(path.read_text())

    def contract(self, report, sensor_root):
        from driversi.agents.backends.native import digest

        for path, sha in self.inputs.items():
            if digest(path) != sha:
                raise ValueError("Source session asset/code changed")
        requests, real_files = [], {}
        for frame in report["frames"]:
            source = self.renderer.binding.resolve(frame["source_token"])
            for camera in CAMERAS:
                real = sensor_root / source["cams"][camera]["data_path"]
                real_files[str(real)] = digest(real)
                requests.append(
                    [
                        frame["tick"],
                        source["token"],
                        int(source["timestamp"]),
                        camera,
                        float(source["cams"][camera].get("v_adjust", 1.0)),
                    ]
                )
        return dict(
            inputs=self.inputs,
            source_files=real_files,
            requests=requests,
            size=report["output_image_size"],
            source_mode="canonical_source_pose_without_motion_v1",
        )

    def run(self, render_report, output):
        from PIL import Image

        from driversi.agents.backends.native import digest, signature
        from driversi.contracts.scene import Scene
        from driversi.rendering.diagnostics.worldengine_admission import source_view_psnr

        report = json.loads(Path(render_report).read_text())
        scene = Scene.load(next(Path(p) for p in report["inputs"] if Path(p).name == "scene.npz"))
        sensor_root = Path(scene.metadata["navsim_source"]["sensor_root"])
        contract = self.contract(report, sensor_root)
        key = signature(contract)
        if key in self.entries:
            path, cached = self.cached_report(key)
            for name, sha in cached["files"].items():
                if digest(name) != sha:
                    raise ValueError("Cached source file changed")
            output.mkdir(parents=True, exist_ok=False)
            write(
                output / "cache-reference.json",
                dict(
                    cache_key=key,
                    report=str(path),
                    report_sha256=digest(path),
                    cache_hit=True,
                    contract=contract,
                ),
            )
            return path, True
        output.mkdir(parents=True, exist_ok=False)
        rows, files = [], {}
        width = report["output_image_size"][0]
        for frame in report["frames"]:
            source = self.renderer.binding.resolve(frame["source_token"])
            images = self.renderer.render_many(source, CAMERAS, width=width)
            for camera in CAMERAS:
                image, evidence = images[camera]
                path = output / f"{frame['tick']:04d}-{camera}-source-render.png"
                if self.image_writer is None:
                    Image.fromarray(image).save(path)
                else:
                    self.image_writer(image, path)
                real = sensor_root / source["cams"][camera]["data_path"]
                factor = float(source["cams"][camera].get("v_adjust", 1.0))
                files[str(path)], files[str(real)] = digest(path), digest(real)
                rows.append(
                    dict(
                        tick=frame["tick"],
                        camera=camera,
                        source_token=source["token"],
                        rendered_source_image=str(path),
                        real_source_image=str(real),
                        exposure_factor=factor,
                        measured_psnr_db=source_view_psnr(path, real, factor),
                        source_motions={},
                        source_camera_to_world=evidence["camera_to_world"],
                        source_appearance_timestamp_us=int(source["timestamp"]),
                    )
                )
        path = output / "report.json"
        write(
            path,
            dict(
                kind="benchmark_canonical_source",
                contract=contract,
                cache_key=key,
                frames=rows,
                files=files,
                trainable=False,
            ),
        )
        self.entries[key] = (path, digest(path))
        return path, False
