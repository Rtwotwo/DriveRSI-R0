"""Encode the gallery's calibrated wireframe style as downloadable MP4s."""

from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path

from .gallery import digest, dump
from .video_export import WhiteboxPainter, encode, video_plan


class WireframePainter(WhiteboxPainter):
    def __init__(self, data):
        super().__init__(data)
        self.context.eval(Path(__file__).with_name("wireframe.js").read_text())
        self.draw_commands = self.context.eval("""(function(raw){
          const c=JSON.parse(raw),commands=[];let points=[],closed=false;
          const ctx={fillStyle:'',strokeStyle:'',lineWidth:1,
            fillRect(x,y,w,h){commands.push({type:'rect',xy:[x,y,x+w,y+h],color:this.fillStyle});},
            beginPath(){points=[];closed=false;},moveTo(x,y){points.push([x,y]);},lineTo(x,y){points.push([x,y]);},closePath(){closed=true;},
            fill(){commands.push({type:'fill',points:points.slice(),color:this.fillStyle});},
            stroke(){commands.push({type:'stroke',points:points.slice(),closed,width:this.lineWidth,color:this.strokeStyle});}
          };
          WireWhitebox.draw({width:c.width,height:c.height,getContext(){return ctx;}},c,{features:videoFeatures,layers:videoLayers});
          return JSON.stringify(commands);
        })""")

    def draw(self, camera):
        from PIL import Image, ImageDraw

        scale = 2
        image = Image.new("RGB", (camera["width"] * scale, camera["height"] * scale))
        draw = ImageDraw.Draw(image)
        for command in json.loads(self.draw_commands(json.dumps(camera, allow_nan=False))):
            if command["type"] == "rect":
                draw.rectangle([x * scale for x in command["xy"]], fill=command["color"])
                continue
            points = [tuple(x * scale for x in p) for p in command["points"]]
            if command["type"] == "fill" and len(points) > 2:
                draw.polygon(points, fill=command["color"])
            elif command["type"] == "stroke" and len(points) > 1:
                draw.line(
                    points + (points[:1] if command["closed"] else []),
                    fill=command["color"],
                    width=max(1, round(command["width"] * scale)),
                    joint="curve",
                )
        return image.resize((camera["width"], camera["height"]), Image.Resampling.LANCZOS)


def export(bundle, gallery):
    from PIL import Image

    from .publication import bundle_files

    bundle, gallery = Path(bundle), Path(gallery)
    bundle_files(bundle)
    target = gallery / "wireframe-videos"
    if target.exists():
        raise FileExistsError(target)
    data = json.loads((bundle / "data.json").read_text())
    plan = video_plan(data)
    painter = WireframePainter(data)
    manifest = {k: v for k, v in plan.items() if k not in ("clips", "frame_indices")}
    manifest.update(
        kind="camera_wireframe_videos",
        schema_version=1,
        trainable=False,
        session_id=data["session_id"],
        source_data_sha256=digest(bundle / "data.json"),
        encoded_sample_indices=plan["frame_indices"],
        style="calibrated semantic wireframe",
        map_layers=painter.layers,
        renderer_sha256={
            n: digest(Path(__file__).with_name(n)) for n in ("projection.js", "wireframe.js")
        },
        videos=[],
    )
    with tempfile.TemporaryDirectory(prefix=".wireframe-export-", dir=gallery) as tmp:
        folder = Path(tmp) / "wireframe-videos"
        folder.mkdir()
        for cid in sorted(data["camera_owners"]):
            cameras = [
                next(c for c in f["cameras"] if c["camera_id"] == cid) for f in data["frames"]
            ]
            panels = {"branch": {}, "source": {}}
            for i in sorted(set(plan["frame_indices"])):
                cam, f = cameras[i], data["frames"][i]
                panels["branch"][i] = painter.panel(painter.draw(cam), cam, f, "branch")
                if not cam["virtual"]:
                    panels["source"][i] = painter.panel(
                        painter.draw(cam["source_whitebox"]), cam, f, "source"
                    )
            for clip in (x for x in plan["clips"] if x["camera_id"] == cid):
                variant = clip["variant"]
                if variant == "comparison":
                    frames = {}
                    for i, right in panels["branch"].items():
                        left = panels["source"][i]
                        out = Image.new("RGB", (left.width + right.width, left.height))
                        out.paste(left, (0, 0))
                        out.paste(right, (left.width, 0))
                        frames[i] = out
                else:
                    frames = panels[variant]
                name = f"{cid}--{variant}.mp4"
                file = folder / name
                info = encode(file, frames, plan["frame_indices"], plan["output_fps"])
                manifest["videos"].append(
                    {
                        **clip,
                        **info,
                        "path": "wireframe-videos/" + name,
                        "bytes": file.stat().st_size,
                        "sha256": digest(file),
                    }
                )
                print("Encoded", name, flush=True)
        dump(folder / "manifest.json", manifest)
        archive = folder / "all-wireframe-videos.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as z:
            for p in sorted(folder.iterdir()):
                if p != archive:
                    z.write(p, p.name)
        manifest["archive"] = {
            "path": "wireframe-videos/" + archive.name,
            "bytes": archive.stat().st_size,
            "sha256": digest(archive),
        }
        folder.rename(target)
    dump(gallery / "wireframe-videos.json", manifest)
    entry = json.loads((gallery / "gallery-entry.json").read_text())
    for p in [gallery / "wireframe-videos.json", *target.iterdir()]:
        entry["files"][p.relative_to(gallery).as_posix()] = digest(p)
    dump(gallery / "gallery-entry.json", entry)
    return manifest


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--gallery", type=Path, required=True)
    args = p.parse_args()
    export(args.bundle, args.gallery)


if __name__ == "__main__":
    main()
