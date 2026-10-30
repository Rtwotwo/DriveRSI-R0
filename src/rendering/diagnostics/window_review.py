"""Build exact-pixel GS sequence review material without assigning admission."""

import argparse
import itertools
import json
import shutil
from pathlib import Path

import numpy as np

CAMERAS = ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0")
COLORS = ("#00ff80", "#ffb000", "#00ddff", "#ff77cc", "#aaff22")


def project_camera(points, evidence):
    """Use the actual optimized source lens, including its distortion."""
    import cv2

    pose = np.asarray(evidence["camera_to_world"], dtype=float)
    xyz = (np.asarray(points) - pose[:3, 3]) @ pose[:3, :3]
    pixels = np.full((len(xyz), 2), np.nan)
    valid = xyz[:, 2] > 0.1
    if valid.any():
        pixels[valid] = cv2.projectPoints(
            xyz[valid],
            np.zeros(3),
            np.zeros(3),
            np.asarray(evidence["distorted_intrinsic"], dtype=float),
            np.asarray(evidence["distortion"], dtype=float),
        )[0].reshape(-1, 2)
    return pixels, xyz


def clipped_box_projection(corners, evidence, *, near=0.1):
    """Project the actual box edges after camera-near-plane clipping.

    A partially clipped box is still a required actor. This only supplies its
    geometric overlay; it never certifies appearance, occlusion or admission.
    """
    import cv2

    pose = np.asarray(evidence["camera_to_world"], dtype=float)
    corners = np.asarray(corners, dtype=float)
    if corners.shape != (8, 3) or not np.isfinite(corners).all():
        raise ValueError("Finite eight-corner physical box required")
    xyz = (corners - pose[:3, 3]) @ pose[:3, :3]
    if not np.isfinite(xyz).all():
        raise ValueError("Nonfinite camera-space box")
    segments, clip_count = [], 0
    for i, j in itertools.combinations(range(8), 2):
        if (i ^ j) not in (1, 2, 4):
            continue
        a, b = xyz[i].copy(), xyz[j].copy()
        if max(a[2], b[2]) < near:
            continue
        if min(a[2], b[2]) < near:
            cut = a + (b - a) * ((near - a[2]) / (b[2] - a[2]))
            if a[2] < near:
                a = cut
            else:
                b = cut
            clip_count += 1
        # Sample the distorted edge rather than drawing one undistorted chord.
        samples = a + np.linspace(0, 1, 17)[:, None] * (b - a)
        pixels = cv2.projectPoints(
            samples,
            np.zeros(3),
            np.zeros(3),
            np.asarray(evidence["distorted_intrinsic"], dtype=float),
            np.asarray(evidence["distortion"], dtype=float),
        )[0].reshape(-1, 2)
        if not np.isfinite(pixels).all():
            raise ValueError("Nonfinite clipped lens projection")
        segments.append(pixels.tolist())
    return dict(
        method="physical_box_near_plane_clipped_edges_v1",
        near_m=near,
        clipped_edges=clip_count,
        pixel_edges=segments,
        original_camera_depth=xyz[:, 2].tolist(),
        projection_supported=bool(segments),
        appearance_certified=False,
    )


def check_window(report, comparison):
    future = report.get("future_ticks", [])
    if future not in (list(range(25, 61, 5)), list(range(25, 71, 5))):
        raise ValueError("Native eight-point or ten-point future required")
    ticks = list(range(0, future[-1] + 1, 5))
    if (
        report.get("status") != "rendered_diagnostic"
        or [f["tick"] for f in report["frames"]] != ticks
    ):
        raise ValueError("Complete actual native history/future window required")
    if report.get("history_ticks") != [5, 10, 15, 20] or report.get("future_ticks") != ticks[5:]:
        raise ValueError("History/future timeline differs")
    source = {(r["tick"], r["camera"]): r for r in comparison["frames"]}
    if len(comparison["frames"]) != len(ticks) * len(CAMERAS) or set(source) != {
        (t, c) for t in ticks for c in CAMERAS
    }:
        raise ValueError("Complete unique canonical-source frames required")
    for frame in report["frames"]:
        if set(frame["cameras"]) != set(CAMERAS):
            raise ValueError("Four-camera coverage differs")
        bindings = [item.get("object_bindings", {}) for item in frame["cameras"].values()]
        if any(item != bindings[0] for item in bindings[1:]):
            raise ValueError("Different physical object binding between cameras")
        for camera, item in frame["cameras"].items():
            if (
                item["source_token"] != frame["source_token"]
                or source[(frame["tick"], camera)]["source_token"] != frame["source_token"]
            ):
                raise ValueError("Camera/reference source synchronization differs")
            if any(
                b["target_tick"] != frame["tick"] for b in item.get("object_bindings", {}).values()
            ):
                raise ValueError("Object binding belongs to another tick")
    return source


def portable_actor_corners(frame, item, token, scene, states, canonical):
    """Independent observed 3D box transported by the executed physical delta."""
    from driversi.rendering.gaussian.mtgs_bindings import observed_3d_box_corners
    from driversi.rendering.gaussian.mtgs_registration import motion_in_source
    from driversi.rendering.gaussian.worldengine_replay import physical_delta

    anchor = canonical[frame["source_token"]]
    j, tick = scene.actor_ids.index(token), frame["tick"]
    corners, annotation = observed_3d_box_corners(anchor, token, scene.dimensions[j])
    delta = physical_delta(states[tick, j], scene.poses[tick, j])
    if not np.allclose(delta, frame["motions"][token], rtol=0, atol=1e-5):
        raise ValueError("Portable audit motion differs from executed physical branch")
    terrain = item["terrain_evidence"][token]
    if terrain["status"] != "supported":
        raise ValueError("Portable bound actor has no supported 3D height")
    translation, rotation, _ = motion_in_source(anchor, delta[:2], delta[2])
    translation[2] = terrain["vertical_displacement_m"]
    center = corners.mean(0)
    return (corners - center) @ rotation.T + center + translation, annotation, terrain


def required_portable_actors(
    frame, item, scene, states, presence, canonical, *, allow_unknown_height=False
):
    """Recompute the required review set at final consumption, not from VLM text."""
    from driversi.rendering.gaussian.worldengine_geometry import box_frustum_support

    moved = set(frame["motions"]) - {"ego"}
    missing = set(frame.get("unresolved_physical_boxes", {}))
    expected = {
        t for j, t in enumerate(scene.actor_ids) if t != "ego" and presence[frame["tick"], j]
    }
    if moved | missing != expected or not moved <= set(item["active_object_tokens"]):
        raise ValueError("Portable audit physical population is incomplete")
    required = set()
    for token in sorted(expected):
        if (
            allow_unknown_height
            and token in moved
            and item["terrain_evidence"][token]["status"] != "supported"
        ):
            required.add(token)  # Potentially visible; no invented Z or projected box.
            continue
        corners = (
            portable_actor_corners(frame, item, token, scene, states, canonical)[0]
            if token in moved
            else np.asarray(frame["unresolved_physical_boxes"][token], dtype=float)
        )
        if box_frustum_support(corners, item["camera_to_world"], item["output_ray_bounds"])[
            "outside_frustum"
        ]:
            continue
        projection = clipped_box_projection(corners, item)
        if not projection["projection_supported"]:
            raise ValueError("Portable actor has no supported clipped box projection")
        required.add(token)
    return required


def run(render_path, source_path, output, *, visual_quality_profile="experimental_scene25_v1"):
    from driversi.rendering.diagnostics.worldengine_admission import visual_policy

    allow_unknown_height = not visual_policy(visual_quality_profile)["require_terrain_support"]
    import pickle

    from PIL import Image, ImageDraw

    from driversi.agents.backends.native import digest
    from driversi.contracts.scene import Scene
    from driversi.rendering.gaussian.mtgs_bindings import observed_3d_box_corners
    from driversi.rendering.gaussian.mtgs_registration import motion_in_source
    from driversi.rsi.plugins.synthesis import check_files

    render_path, source_path = render_path.resolve(strict=True), source_path.resolve(strict=True)
    report, comparison = json.loads(render_path.read_text()), json.loads(source_path.read_text())
    if comparison["render_report_sha256"] != digest(render_path):
        raise ValueError("Source comparison belongs to another rendering")
    check_files({**report["inputs"], **comparison["inputs"], **comparison["files"]})
    sources = check_window(report, comparison)
    scene = Scene.load(next(Path(p) for p in report["inputs"] if Path(p).name == "scene.npz"))
    metadata = next(Path(p) for p in report["inputs"] if Path(p).name == "video_scene_dict.pkl")
    with metadata.open("rb") as stream:
        video = pickle.load(stream)[report["asset_id"] + "-" + str(report["travel_id"])]
    canonical = {frame["token"]: frame for frame in video["frame_infos"]}
    portable = report.get("renderer_backend") == "worldengine_portable"
    if portable:
        rollouts = [
            Path(p) for p in report["inputs"] if Path(p).name in ("teacher.npz", "baseline.npz")
        ]
        if len(rollouts) != 1:
            raise ValueError("One exact portable physical branch required for pixel inspection")
        with np.load(rollouts[0], allow_pickle=False) as data:
            states, present = data["states"], data["presence"]
            if tuple(data["actor_ids"].tolist()) != scene.actor_ids:
                raise ValueError("Portable inspection actor population differs")
    actor_ids = sorted(
        {
            token
            for f in report["frames"]
            for token in (
                set(f["cameras"][CAMERAS[0]].get("object_bindings", {}))
                | (set(f.get("motions", {})) - {"ego"})
                | set(f.get("unresolved_physical_boxes", {}))
            )
        }
    )
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    files, projections = {}, []
    for camera in CAMERAS:
        sheets = {
            kind: Image.new("RGB", (1440, 1450), "#18212b")
            for kind in ("novel", "boxes", "source", "real")
        }
        for i, frame in enumerate(report["frames"]):
            item, tick = frame["cameras"][camera], frame["tick"]
            path = render_path.parent / item["file"]
            image = Image.open(path).convert("RGB")
            copy_path = output / f"{tick:04d}-{camera}-gs.png"
            shutil.copyfile(path, copy_path)
            files[str(copy_path)] = digest(copy_path)
            overlay = image.copy()
            draw = ImageDraw.Draw(overlay)
            projected = []
            native_bindings = item.get("object_bindings", {})
            portable_tokens = (set(frame.get("motions", {})) - {"ego"}) if portable else set()
            geometry_tokens = sorted(
                set(native_bindings)
                | portable_tokens
                | set(frame.get("unresolved_physical_boxes", {}))
            )
            if portable:
                expected_tokens = {
                    token
                    for j, token in enumerate(scene.actor_ids)
                    if token != "ego" and present[tick, j]
                }
                if set(geometry_tokens) != expected_tokens:
                    raise ValueError("Portable pixel audit omits or invents physical actors")
                if not portable_tokens <= set(item["active_object_tokens"]):
                    raise ValueError("Portable rendered actor missing from active assets")
            for token in geometry_tokens:
                if (
                    allow_unknown_height
                    and token in portable_tokens
                    and item["terrain_evidence"][token]["status"] != "supported"
                ):
                    # Preserve identity and unknown status without drawing fabricated geometry.
                    projected.append(
                        dict(
                            actor_id=token,
                            projected_box_overlaps_image=True,
                            projection_status="unknown_height_no_overlay",
                            terrain_status=item["terrain_evidence"][token]["status"],
                            pixel_visibility="unknown_requires_ego_camera_review",
                            corners_reconstruction=None,
                        )
                    )
                    continue
                binding = native_bindings.get(token)
                if token in portable_tokens:
                    corners, annotation, terrain = portable_actor_corners(
                        frame, item, token, scene, states, canonical
                    )
                elif binding is None:
                    corners = np.asarray(frame["unresolved_physical_boxes"][token], dtype=float)
                    annotation = {
                        "contract": "recorded_physical_background_box_requires_independent_binding"
                    }
                    terrain = {"status": "background_registration_requires_review"}
                else:
                    anchor = canonical[binding["source_token"]]
                    corners, annotation = observed_3d_box_corners(
                        anchor, token, scene.dimensions[scene.actor_ids.index(token)]
                    )
                    terrain = item["terrain_evidence"][token]
                    if terrain["status"] != "source_identity":
                        delta = binding["planar_delta"]
                        translation, rotation, _ = motion_in_source(anchor, delta[:2], delta[2])
                        if terrain["status"] == "supported":
                            translation[2] = terrain["vertical_displacement_m"]
                        center = corners.mean(0)
                        corners = (corners - center) @ rotation.T + center + translation
                uv, xyz = project_camera(corners, item)
                finite = np.isfinite(uv).all(1)
                from driversi.rendering.gaussian.worldengine_geometry import box_frustum_support

                outside = box_frustum_support(
                    corners, item["camera_to_world"], item["output_ray_bounds"]
                )["outside_frustum"]
                # Exclude only geometry proven outside the camera frustum.
                overlap = not outside
                clipped = clipped_box_projection(corners, item) if overlap else None
                projection_status = (
                    "outside_frustum"
                    if outside
                    else "verified_clipped_near_plane"
                    if clipped["projection_supported"] and not finite.all()
                    else "verified"
                    if clipped["projection_supported"]
                    else "unresolved_near_plane_projection"
                )
                color = COLORS[actor_ids.index(token) % len(COLORS)]
                if overlap and clipped["projection_supported"]:
                    import cv2

                    for edge in clipped["pixel_edges"]:
                        for a, b in zip(edge[:-1], edge[1:], strict=True):
                            # clipLine accepts int32; clamp only display coordinates.
                            a, b = np.clip([a, b], -1e8, 1e8).astype(int)
                            visible, a, b = cv2.clipLine(
                                (0, 0, image.width, image.height), tuple(a), tuple(b)
                            )
                            if visible:
                                draw.line([a, b], fill=color, width=3)
                    points = np.asarray(clipped["pixel_edges"]).reshape(-1, 2)
                    label = np.clip(
                        np.median(points, axis=0),
                        [0, 25],
                        [max(0, image.width - 160), max(25, image.height - 25)],
                    )
                    draw.text(tuple(label), token[:8], fill=color)
                projected.append(
                    {
                        "actor_id": token,
                        "color": color,
                        "projected_box_overlaps_image": overlap,
                        "projection_status": projection_status,
                        "pixel_visibility": "requires_review_occlusion_not_inferred_from_box",
                        "corners_reconstruction": corners.tolist(),
                        "camera_depth": xyz[:, 2].tolist(),
                        "pixel_corners": [p.tolist() if np.isfinite(p).all() else None for p in uv],
                        "clipped_projection": clipped,
                        "source_annotation": annotation,
                        "terrain_status": terrain["status"],
                    }
                )
            overlay_path = output / f"{tick:04d}-{camera}-boxes.jpg"
            overlay.save(overlay_path, quality=95)
            files[str(overlay_path)] = digest(overlay_path)
            current = sources[(tick, camera)]
            views = {
                "novel": image,
                "boxes": overlay,
                "source": Image.open(current["rendered_source_image"]).convert("RGB"),
                "real": Image.open(current["real_source_image"]).convert("RGB"),
            }
            for kind, pixels in views.items():
                pixels.thumbnail((480, 270))
                x, y = i % 3 * 480, i // 3 * 290
                sheets[kind].paste(pixels, (x, y + 20))
                ImageDraw.Draw(sheets[kind]).text(
                    (x + 4, y + 3), f"{camera} tick={tick} {kind}", fill="white"
                )
            projections.append(
                {
                    "tick": tick,
                    "camera": camera,
                    "image": str(path),
                    "image_sha256": digest(path),
                    "overlay": str(overlay_path),
                    "actors": projected,
                }
            )
        for kind, pixels in sheets.items():
            path = output / f"{camera}-{kind}.jpg"
            pixels.save(path, quality=95)
            files[str(path)] = digest(path)
    receipt = {
        "kind": "multiview_gs_exact_pixel_inspection_material",
        "producer_code": {str(Path(__file__).resolve()): digest(__file__)},
        "trainable": False,
        "render_report": str(render_path),
        "render_report_sha256": digest(render_path),
        "source_report": str(source_path),
        "source_report_sha256": digest(source_path),
        "mechanical_checks": {
            "actual_frame_camera_pairs": len(report["frames"]) * len(CAMERAS),
            "source_frame_camera_pairs": len(comparison["frames"]),
            "same_actor_binding_all_cameras": True,
            "portable_physical_actor_coverage_checked": portable,
            "source_camera_synchronization": True,
        },
        "actor_colors": {token: COLORS[i % len(COLORS)] for i, token in enumerate(actor_ids)},
        "frames": projections,
        "files": files,
        "human_review": "pending_no_script_assigned_pass",
        "limitations": [
            "Projected 3D boxes do not prove visible-pixel geometry or resolve occlusion",
            "Measured actor entries and source-identity bindings require per-frame review; this pilot does not establish dynamic-interaction generalization",
        ],
    }
    (output / "report.json").write_text(json.dumps(receipt, indent=2))
    (output / "viewer.html").write_text(
        viewer_html(actor_ids, future_points=len(report["future_ticks"]))
    )
    return receipt


def viewer_html(actor_ids, *, future_points=10):
    from html import escape

    if type(future_points) is not int or future_points not in (8, 10):
        raise ValueError("Native eight or ten future points required")
    actors = " · ".join(escape(token) for token in actor_ids)
    html = (
        """<!doctype html><html lang="zh"><meta charset="utf-8"><title>GS 四相机连续分支核查</title>
<style>body{font:16px/1.6 system-ui;margin:24px;background:#edf3f6;color:#18313e}header,section{background:white;padding:18px;border-radius:12px;margin-bottom:18px}aside{background:#fff0cc;padding:12px}button,select{padding:9px;margin:4px}input{width:100%}.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}figure{margin:0;background:#14202a;color:white}img{width:100%;display:block}figcaption{padding:8px}a{color:#00717d}.status{font-weight:bold}</style>
<header><h1>GS 四相机连续分支核查</h1><p>同一条 navtrain 物理分支 · 原始 3DGS · 4 帧新历史＋10 帧实际未来 · 单帧渲染不读取未来相机条件</p>
<aside>当前页面展示待独立验收图像。投影框一致不等于像素位置、遮挡或画质自动通过。tick 0 是额外验收帧，不进入四帧历史。</aside>
<p id="acceptance" class="status">视觉准入：等待独立像素检查与正式回执。</p><p>对象身份："""
        + actors
        + """</p><p>对象包括保留的源身份对象及实际入场对象；此结果不代表动态多车交互的视觉泛化。</p><a href="report.json">机械检查与逐帧证据</a></header>
<section><strong id="label"></strong><input id="tick" type="range" min="0" max="14" value="0" step="1"><button id="play">播放</button><button id="prev">上一帧</button><button id="next">下一帧</button><select id="mode"><option value="gs.png">原始 GS</option><option value="boxes.jpg">物理源框投影核查</option></select><p>历史：tick 5、10、15、20（2 Hz）；未来：tick 25–70（2 Hz，5 秒）。训练模型原生图像输入顺序由加载器保持为 F0、B0、L0、R0。</p></section>
<div class="grid" id="grid"></div><script>
const cams=['CAM_F0','CAM_L0','CAM_R0','CAM_B0'],ticks=Array.from({length:15},(_,i)=>i*5);let timer=null;
const slider=document.getElementById('tick'),mode=document.getElementById('mode');
document.getElementById('grid').innerHTML=cams.map(c=>`<figure><figcaption>${c}</figcaption><img id="${c}" alt="${c}"></figure>`).join('');
function draw(){const t=ticks[+slider.value],role=t===0?'额外验收帧（不用于训练）':t<=20?'新历史':'纠正未来';document.getElementById('label').textContent=`tick ${t} · t=${(t/10).toFixed(1)} s · ${role}`;cams.forEach(c=>document.getElementById(c).src=`${String(t).padStart(4,'0')}-${c}-${mode.value}`);}
function stop(){clearInterval(timer);timer=null;document.getElementById('play').textContent='播放';}
slider.oninput=draw;mode.onchange=draw;document.getElementById('prev').onclick=()=>{stop();slider.value=Math.max(0,+slider.value-1);draw();};document.getElementById('next').onclick=()=>{stop();slider.value=Math.min(14,+slider.value+1);draw();};document.getElementById('play').onclick=()=>{if(timer){stop();return;}if(+slider.value===14)slider.value=0;draw();document.getElementById('play').textContent='暂停';timer=setInterval(()=>{if(+slider.value===14){stop();return;}slider.value=+slider.value+1;draw();},500);};
fetch('acceptance.json').then(r=>r.ok?r.json():null).then(d=>{if(d)document.getElementById('acceptance').textContent='外部正式验收：'+(d.admitted===true?'已通过指定试验组准入；不代表泛化收益':'未通过')+'；'+(d.scope||'详见回执');}).catch(()=>{});draw();
</script></html>"""
    )

    if future_points == 8:
        html = (
            html.replace("＋10 帧实际未来", "＋8 帧实际未来")
            .replace('max="14"', 'max="12"')
            .replace("length:15", "length:13")
            .replace("Math.min(14,", "Math.min(12,")
            .replace("===14", "===12")
            .replace("25–70（2 Hz，5 秒）", "25–60（2 Hz，4 秒）")
            .replace(
                "训练模型原生图像输入顺序由加载器保持为 F0、B0、L0、R0。",
                "AUGUstus 原生输入为 L0、F0、R0 三相机拼接；B0 仅用于本页面核查。",
            )
        )
    return html


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("render-report", "source-report", "output"):
        parser.add_argument("--" + key, required=True, type=Path)
    args = parser.parse_args()
    run(args.render_report, args.source_report, args.output)


if __name__ == "__main__":
    main()
