"""Native WorldDreamer tensors from one calibrated, sealed physical branch."""

import argparse
import json
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

from driversi.rendering.video.worlddreamer.runtime import activate


def native_box_corners(actors):
    # Reproduce nuplan_converter -> LiDARInstance3DBoxes corner convention.
    # Native token uses bottom-origin; layout separately shifts to centre.
    signs = np.asarray(
        [
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, 1],
            [-1, 1, -1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, 1],
            [1, 1, -1],
        ],
        np.float32,
    )
    q = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], np.float32)
    layout = []
    tokens = []
    for actor in actors:
        corners = np.asarray(actor["corners_ego"], np.float32)
        centre = corners.mean(axis=0) @ q.T
        axis = corners[4] - corners[0]
        yaw = -np.arctan2(axis[1], axis[0]) - np.pi
        c, s = np.cos(yaw), np.sin(yaw)
        dims = np.asarray(actor["dimensions"], np.float32)[[1, 0, 2]]
        rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float32)
        points = (signs * dims / 2) @ rotation + centre
        layout.append(points)
        tokens.append(points + np.array([0, 0, dims[2] / 2], np.float32))
    return np.asarray(layout, np.float32).reshape(-1, 8, 3), np.asarray(tokens, np.float32).reshape(
        -1, 8, 3
    )


def prepare(geometry_path, output):
    activate()
    import torch
    from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
    from PIL import Image
    from shapely.ops import unary_union
    from shapely.vectorized import contains

    from driversi.agents.backends.native import digest
    from driversi.rendering.video.bundle import validate_bundle
    from driversi.rsi.plugins.synthesis import check_files
    from driversi.rendering.video.worlddreamer.native_map import (
        calculate_rotation,
        extract_map_features,
        project_box_to_image,
        project_map_to_image_nuplan,
        rotate_features,
        visualize_bev_hdmap,
    )

    g = json.loads(Path(geometry_path).read_text())
    check_files(g["files"])
    validate_bundle(g["bundle"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    maps = Path(g["map_file"]).parents[2]
    api = get_maps_api(str(maps), "nuplan-maps-v1.0", g["map_location"])
    polygons = []
    for layer in ["intersections", "road_segments"]:
        polygons.extend(api._maps_db.load_vector_layer(g["map_location"], layer).geometry)
    drivable = unary_union(polygons)
    classes = [
        "car",
        "truck",
        "construction_vehicle",
        "bus",
        "trailer",
        "barrier",
        "motorcycle",
        "bicycle",
        "pedestrian",
        "traffic_cone",
    ]
    kind_mapping = {
        "vehicle": 0,
        "pedestrian": 8,
        "bicycle": 7,
        "cyclist": 7,
        "motorcycle": 6,
        "traffic_cone": 9,
        "traffic_barrier": 5,
        "barrier": 5,
    }
    refs = []
    for cam in g["camera_order"]:
        im = Image.open(g["initial_reference_images"][cam]).convert("RGB")
        w, h = im.size
        nw, nh = int(w * 0.21), int(h * 0.21)
        im = im.resize((nw, nh), Image.Resampling.BILINEAR)
        left = int((nw - 400) / 2)
        top = nh - 224
        im = im.crop((left, top, left + 400, top + 224))
        refs.append(np.asarray(im).transpose(2, 0, 1) / 127.5 - 1)
    refs = np.asarray(refs, np.float32)
    previous = np.asarray(g["initial_reference_ego_to_world"])
    files = {str(Path(geometry_path).resolve()): digest(geometry_path)}
    rows = []
    for frame in g["frames"]:
        if frame["tick"] == 0:
            continue
        ego = np.asarray(frame["ego_to_world"])
        features = extract_map_features(api, ego[:2, 3], radius=50)
        vectors = rotate_features(
            features,
            -calculate_rotation(ego[:3, :3]),
            {"ROAD_LINE_SOLID_SINGLE_WHITE": 2, "ROAD_LINE_BROKEN_SINGLE_WHITE": 0, "CROSSWALK": 1},
            [(-50, 50), (50, 50), (50, -50), (-50, -50)],
        )
        lines = [np.asarray(line.coords) for line in vectors["gt_vecs_pts_loc"]]
        labels = vectors["gt_vecs_label"]
        grid = np.arange(-49.75, 50, 0.5)
        gx, gy = np.meshgrid(grid, grid, indexing="ij")
        yaw = np.arctan2(ego[1, 0], ego[0, 0])
        c, s = np.cos(yaw), np.sin(yaw)
        ex, ey = gx, gy
        wx = c * ex - s * ey + ego[0, 3]
        wy = s * ex + c * ey + ego[1, 3]
        mask = contains(drivable, wx, wy)
        bev = visualize_bev_hdmap(
            lines, labels, [200, 200], drivable_mask=mask, nuplan=True
        ).transpose(2, 0, 1)
        corners = np.asarray([a["corners_ego"] for a in frame["actors"]], np.float32).reshape(
            -1, 8, 3
        )
        native_rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], np.float32)
        corners, token_corners = native_box_corners(frame["actors"])
        labels_box = np.asarray([kind_mapping[a["kind"]] for a in frame["actors"]], np.int64)

        class Boxes:
            def __init__(self, x):
                self.corners = x

            def __len__(self):
                return len(self.corners)

        bbox_data = []
        bbox_cls = []
        bbox_mask = []
        camera_param = []
        layout = []
        rot = torch.tensor([[0.0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        for cam in g["camera_order"]:
            data = frame["cameras"][cam]
            k = np.asarray(data["intrinsic"], np.float32)
            extr = np.asarray(data["camera_to_ego"], np.float32)
            extr = extr.copy()
            extr[:3] = native_rotation @ extr[:3]
            camera_param.append(
                np.concatenate([np.diag([0.8333, 0.8333, 1]) @ k, extr[:3]], axis=-1)
            )
            inv = np.linalg.inv(extr)
            campts = token_corners @ inv[:3, :3].T + inv[:3, 3]
            valid = (campts[..., 2] > 0).any(axis=1)
            bbox_data.append(token_corners.copy())
            bbox_cls.append(labels_box.copy())
            bbox_mask.append(valid)
            # The original helper rasterizes900x1600 then slices[::4][1:].
            # Lift the actual224x400 reference crop calibration into that canvas.
            with Image.open(data["source_image"]) as source_image:
                sw, sh = source_image.size
            nw, nh = int(sw * 0.21), int(sh * 0.21)
            left = int((nw - 400) / 2)
            top = nh - 224
            crop = np.array([[nw / sw, 0, -left], [0, nh / sh, -top], [0, 0, 1]], np.float32)
            canvas = np.array([[4, 0, 0], [0, 4, 4], [0, 0, 1]], np.float32) @ crop @ k
            kk = torch.from_numpy(canvas)
            ee = torch.from_numpy(extr)
            map_canvas = project_map_to_image_nuplan(
                lines, labels, kk, rot @ ee, drivable_mask=mask
            )
            projection = np.eye(4, dtype=np.float32)
            projection[:3, :3] = canvas
            projection = projection @ inv
            box_canvas = project_box_to_image(
                Boxes(torch.from_numpy(corners)),
                torch.from_numpy(labels_box),
                torch.from_numpy(projection),
                classes,
            )
            layout.append(np.concatenate([map_canvas, box_canvas], axis=-1).transpose(2, 0, 1))
        path = output / f"{frame['tick']:04d}.npz"
        np.savez_compressed(
            path,
            bev_hdmap=bev.astype(np.float32),
            camera_param=np.asarray(camera_param),
            relative_pose=(np.linalg.inv(ego) @ previous).astype(np.float32),
            layout_canvas=np.asarray(layout, np.float32),
            ref_images=refs,
            bbox_bboxes=np.asarray(bbox_data),
            bbox_classes=np.asarray(bbox_cls),
            bbox_masks=np.asarray(bbox_mask),
        )
        files[str(path)] = digest(path)
        rows.append(
            dict(
                tick=frame["tick"],
                timestamp_us=frame["timestamp_us"],
                conditions=str(path),
                actor_ids=[a["actor_id"] for a in frame["actors"]],
                map_vector_count=len(lines),
            )
        )
        previous = ego
        print("conditions", frame["tick"], len(lines), len(corners), flush=True)
    report = dict(
        kind="worlddreamer_native_conditions",
        schema_version=1,
        branch_identity=g["branch_identity"],
        geometry=str(Path(geometry_path).resolve()),
        source_control=g["source_control"],
        camera_order=g["camera_order"],
        caption="A driving scene image at " + g["map_location"] + ". Daytime urban road.",
        frames=rows,
        files=files,
        trainable=False,
        limitations=g["geometry_limitations"]
        + [
            "Layout adapter v5: native bottom-origin box tokens; official+90deg camera/box basis, raw-ego map, exact crop projection; native raster functions unchanged; pinhole layout omits raw lens distortion, terrain/registration/pixels require independent admission"
        ],
    )
    (output / "report.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--geometry", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    prepare(a.geometry, a.output)
