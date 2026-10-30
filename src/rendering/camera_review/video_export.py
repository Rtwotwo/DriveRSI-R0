"""Export auditable H.264 whitebox clips using the browser's projection/drawing code.

Optional tool environment: quickjs==1.19.4 Pillow==11.3.0 imageio-ffmpeg==0.6.0.
No GS inference, world advancement, or synthesized source-camera viewpoints.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path

DEFAULT_LAYERS = (
    "lanes_polygons",
    "crosswalks",
    "walkways",
    "baseline_paths",
    "traffic_lights",
    "traffic_signs",
    "boundaries",
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def video_plan(data, fps=25):
    """Causal zero-order hold on a half-open time interval, never time-warped."""
    frames = data["frames"]
    times = [float(f["time_seconds"]) for f in frames]
    if not isinstance(fps, int) or not 1 <= fps <= 60:
        raise ValueError("Output fps must be an integer from 1 to 60")
    if len(times) < 2 or not all(math.isfinite(t) for t in times):
        raise ValueError("Need at least two finite timestamps")
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("Timestamps must be strictly increasing")
    if len({f["generation"] for f in frames}) != 1:
        raise ValueError("A video must not cross episode resets")
    if any(b["tick"] <= a["tick"] for a, b in zip(frames, frames[1:])):
        raise ValueError("Ticks must be strictly increasing")
    ids = set(data["camera_owners"])
    if not ids or any(not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", name) for name in ids):
        raise ValueError("Unsafe or empty camera IDs")
    for f in frames:
        if len(f["cameras"]) != len(ids) or {c["camera_id"] for c in f["cameras"]} != ids:
            raise ValueError("All video frames must have the same camera coverage")
        for c in f["cameras"]:
            if c["owner_actor_id"] != data["camera_owners"][c["camera_id"]]:
                raise ValueError("Video camera owner changed")
    duration = times[-1] - times[0]
    count = math.ceil(duration * fps - 1e-8)
    indices = [bisect.bisect_right(times, times[0] + i / fps + 1e-9) - 1 for i in range(count)]
    clips = []
    for camera_id in sorted(ids):
        cameras = [next(c for c in f["cameras"] if c["camera_id"] == camera_id) for f in frames]
        if any(c["virtual"] != cameras[0]["virtual"] for c in cameras):
            raise ValueError("Camera type changed")
        variants = ["branch"]
        if not cameras[0]["virtual"]:
            if not all(c.get("source_whitebox") for c in cameras):
                raise ValueError("Missing original whitebox for a source-camera video")
            variants = ["source", "branch", "comparison"]
        clips.extend(
            {
                "camera_id": camera_id,
                "owner_actor_id": data["camera_owners"][camera_id],
                "variant": v,
            }
            for v in variants
        )
    return {
        "scene_duration_seconds": duration,
        "output_fps": fps,
        "frame_count": count,
        "encoded_duration_seconds": count / fps,
        "frame_indices": indices,
        "clips": clips,
    }


class WhiteboxPainter:
    """Capture shared Canvas polygon commands in QuickJS, rasterize with Pillow."""

    def __init__(self, data):
        import quickjs
        from PIL import ImageFont

        self.context = quickjs.Context()
        self.context.set_memory_limit(256 * 1024 * 1024)
        self.context.set_time_limit(30)
        root = Path(__file__).parent
        for name in ("projection.js", "semantic.js"):
            self.context.eval((root / name).read_text())
        self.context.eval(
            "const videoFeatures=" + json.dumps(data["map_features"], allow_nan=False) + ";"
        )
        layers = [
            n
            for n in DEFAULT_LAYERS
            if data["map_layers"].get(n, {}).get("availability") == "available"
        ]
        self.context.eval("const videoLayers=new Set(" + json.dumps(layers) + ");")
        self.draw_commands = self.context.eval(r"""(function(raw){
          const c=JSON.parse(raw),commands=[];let points=[];
          const ctx={fillStyle:'',strokeStyle:'',lineWidth:1,
            fillRect(x,y,w,h){commands.push({type:'rect',xy:[x,y,x+w,y+h],color:this.fillStyle});},
            beginPath(){points=[];},moveTo(x,y){points.push([x,y]);},lineTo(x,y){points.push([x,y]);},closePath(){},
            fill(){commands.push({type:'fill',points:points.slice(),color:this.fillStyle});},
            stroke(){commands.push({type:'stroke',points:points.slice(),color:this.strokeStyle,width:this.lineWidth});}
          };
          SemanticWhitebox.draw({width:c.width,height:c.height,getContext(){return ctx;}},c,{features:videoFeatures,layers:videoLayers});
          return JSON.stringify(commands);
        })""")
        self.font = ImageFont.load_default(size=19)
        self.small = ImageFont.load_default(size=14)
        self.layers = layers

    def draw(self, camera):
        from PIL import Image, ImageDraw

        width, height = camera["width"], camera["height"]
        scale = 2
        image = Image.new("RGB", (width * scale, height * scale))
        draw = ImageDraw.Draw(image)
        for command in json.loads(self.draw_commands(json.dumps(camera, allow_nan=False))):
            if command["type"] == "rect":
                draw.rectangle([v * scale for v in command["xy"]], fill=command["color"])
            else:
                points = [tuple(v * scale for v in p) for p in command["points"]]
                if command["type"] == "fill":
                    draw.polygon(points, fill=command["color"])
                elif points:
                    draw.line(
                        points + points[:1], fill=command["color"], width=scale, joint="curve"
                    )
        return image.resize((width, height), Image.Resampling.LANCZOS)

    def panel(self, image, camera, frame, variant):
        from PIL import Image, ImageDraw

        width, height = image.size
        result = Image.new("RGB", (width, height + 100), "#11253b")
        result.paste(image, (0, 60))
        draw = ImageDraw.Draw(result)
        title = "ORIGINAL NAVSIM LOG" if variant == "source" else "NEW TRAJECTORY"
        draw.text((14, 9), title + " | SEMANTIC WHITEBOX", font=self.font, fill="white")
        draw.text(
            (14, 34),
            camera["camera_id"] + " | owner: " + camera["owner_actor_id"],
            font=self.small,
            fill="#9de4d5",
        )
        stamp = (
            camera.get("source_whitebox", {}).get("source_timestamp_us")
            if variant == "source"
            else frame["simulation_timestamp_us"]
        )
        label = f"t={frame['time_seconds']:.2f}s | tick {frame['tick']} | observed us: {stamp}"
        draw.text((14, height + 65), label, font=self.small, fill="white")
        draw.text(
            (14, height + 83),
            "Annotation geometry; pink = missing GS asset. Not photorealistic or certified training labels.",
            font=self.small,
            fill="#b8ccdf",
        )
        return result


def encode(path, frames, indices, fps):
    import imageio_ffmpeg

    first = frames[indices[0]]
    width, height = first.size
    if width % 2 or height % 2:
        raise ValueError("H.264 yuv420p needs even dimensions")
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        str(fps),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-threads",
        "2",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-n",
        str(path),
    ]
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=errors)
        try:
            for index in indices:
                process.stdin.write(frames[index].tobytes())
            process.stdin.close()
            code = process.wait(timeout=120)
            if code:
                errors.seek(0)
                raise RuntimeError(errors.read().decode(errors="replace"))
        except BaseException:
            process.kill()
            process.wait()
            raise
    count, duration = imageio_ffmpeg.count_frames_and_secs(str(path))
    if count != len(indices) or abs(duration - count / fps) > 1 / fps:
        raise ValueError("Encoded timing/frame count differs from simulation timeline")
    reader = imageio_ffmpeg.read_frames(str(path))
    try:
        metadata = next(reader)
    finally:
        reader.close()
    if tuple(metadata["size"]) != (width, height) or metadata["codec"] != "h264":
        raise ValueError("Unexpected encoded video geometry or codec")
    return {
        "width": width,
        "height": height,
        "encoded_frame_count": count,
        "duration_seconds": count / fps,
        "codec": "h264",
        "pixel_format": "yuv420p",
    }


def export_videos(bundle, fps=25):
    from PIL import Image

    from .publication import bundle_files

    root = Path(bundle).resolve()
    bundle_files(root)
    if (root / "videos").exists() or (root / "videos.json").exists():
        raise FileExistsError(
            "Video export already exists; preserve it and use a new bundle/version"
        )
    data_path = root / "data.json"
    data = json.loads(data_path.read_text())
    plan = video_plan(data, fps)
    painter = WhiteboxPainter(data)
    manifest = {k: v for k, v in plan.items() if k not in ("clips", "frame_indices")}
    manifest.update(
        schema_version=1,
        kind="camera_whitebox_videos",
        trainable=False,
        session_id=data["session_id"],
        scenario_id=data["scenario_id"],
        data_sha256=sha256(data_path),
        simulation_sample_times=[f["time_seconds"] for f in data["frames"]],
        encoded_sample_indices=plan["frame_indices"],
        timing="Causal sample hold; [start,end), terminal state has zero dwell and remains in data.json. Encoded frames are not new sensor observations.",
        renderer="Shared projection.js + semantic.js; Pillow 2x antialias; lossy H.264 encoding",
        renderer_sha256={
            n: sha256(Path(__file__).with_name(n)) for n in ["projection.js", "semantic.js"]
        },
        map_layers=painter.layers,
        selection=None,
        videos=[],
    )
    with tempfile.TemporaryDirectory(prefix=".whitebox-export-", dir=root) as temporary:
        temporary = Path(temporary)
        folder = temporary / "videos"
        folder.mkdir()
        for camera_id in sorted(data["camera_owners"]):
            cameras = [
                next(c for c in f["cameras"] if c["camera_id"] == camera_id) for f in data["frames"]
            ]
            panels = {"source": {}, "branch": {}}
            source_cache = {}
            for index in sorted(set(plan["frame_indices"])):
                camera, frame = cameras[index], data["frames"][index]
                panels["branch"][index] = painter.panel(
                    painter.draw(camera), camera, frame, "branch"
                )
                if not camera["virtual"]:
                    original = camera["source_whitebox"]
                    key = hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest()
                    if key not in source_cache:
                        source_cache[key] = painter.draw(original)
                    panels["source"][index] = painter.panel(
                        source_cache[key], camera, frame, "source"
                    )
            for clip in [c for c in plan["clips"] if c["camera_id"] == camera_id]:
                variant = clip["variant"]
                if variant == "comparison":
                    frames = {}
                    for index in panels["branch"]:
                        left, right = panels["source"][index], panels["branch"][index]
                        if left.size != right.size:
                            raise ValueError("Original/new camera frame sizes differ")
                        image = Image.new("RGB", (left.width * 2, left.height))
                        image.paste(left, (0, 0))
                        image.paste(right, (left.width, 0))
                        frames[index] = image
                else:
                    frames = panels[variant]
                filename = f"{camera_id}--{variant}.mp4"
                file = folder / filename
                info = encode(file, frames, plan["frame_indices"], fps)
                manifest["videos"].append(
                    {
                        **clip,
                        **info,
                        "path": "videos/" + filename,
                        "bytes": file.stat().st_size,
                        "sha256": sha256(file),
                    }
                )
                print(json.dumps({"exported": filename, "bytes": file.stat().st_size}), flush=True)
        (folder / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        archive = folder / "all-whitebox-videos.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as z:
            for file in sorted(folder.iterdir()):
                if file != archive:
                    z.write(file, file.name)
        manifest["archive"] = {
            "path": "videos/" + archive.name,
            "bytes": archive.stat().st_size,
            "sha256": sha256(archive),
        }
        (temporary / "videos.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        if sha256(data_path) != manifest["data_sha256"]:
            raise ValueError("Scene changed during video export")
        os.rename(folder, root / "videos")
        os.rename(temporary / "videos.json", root / "videos.json")
    bundle_files(root)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=25)
    args = parser.parse_args()
    result = export_videos(args.bundle, args.fps)
    print(
        json.dumps({"videos": len(result["videos"]), "archive_bytes": result["archive"]["bytes"]})
    )


if __name__ == "__main__":
    main()
