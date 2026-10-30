"""Full read-only nuPlan DB inspection; never a physics rollout or training sample."""

import hashlib
import json
from pathlib import Path

import numpy as np

from driversi.data.sources.navsim import yaw_from_quaternion
from driversi.data.sources.nuplan_db import CATEGORIES, log_header, open_db
from driversi.rendering.inspection.nuplan_log_template import LOG_HTML


def dump(path, value):
    content = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def export_log(db_path, output, *, scenes=(), chunk_size=200, max_frames=20000):
    """Preserve all raw lidar timestamps and boxes. Refuse overflow, never truncate.

    Geometry is map-axis XY translated by the first measured ego rear axle.
    Object centers retain raw dimensions; ego center uses nuPlan Pacifica offset.
    Map patches are only the supplied compiled windows, explicitly incomplete.
    """
    if not 1 <= chunk_size <= 200 or not 2 <= max_frames <= 20000:
        raise ValueError("Invalid bounded export budget")
    path, output = Path(db_path).resolve(strict=True), Path(output).resolve()
    before = path.stat()
    output.mkdir(parents=True, exist_ok=False)
    with open_db(path) as conn:
        header = log_header(conn)
        streams = conn.execute("SELECT DISTINCT lidar_token FROM lidar_pc").fetchall()
        if len(streams) != 1:
            raise ValueError("Require one lidar stream per full DB playback")
        owner = conn.execute(
            "SELECT log_token FROM lidar WHERE token=?", (streams[0][0],)
        ).fetchone()
        if owner is None or owner[0].hex() != header["log_token"]:
            raise ValueError("Lidar stream/log mismatch")
        rows = conn.execute(
            """SELECT p.token AS pc_token,p.timestamp AS pc_us,
            p.scene_token,e.* FROM lidar_pc p LEFT JOIN ego_pose e ON e.token=p.ego_pose_token
            ORDER BY p.timestamp LIMIT ?""",
            (max_frames + 1,),
        ).fetchall()
        if not 2 <= len(rows) <= max_frames:
            raise ValueError("Full DB frame budget exceeded or insufficient frames")
        stamps = [int(row["pc_us"]) for row in rows]
        if any(b <= a for a, b in zip(stamps, stamps[1:])):
            raise ValueError("Non-monotonic or duplicate source timestamps")
        if any(row["token"] is None for row in rows):
            raise ValueError("Missing ego pose")
        origin = [float(rows[0]["x"]), float(rows[0]["y"])]
        actors = [{"id": "ego", "kind": "ego", "intervals": [[0, len(rows) - 1]]}]
        lookup = {"ego": 0}
        chunks, counts, ego_path = [], [], []
        gaps = [i for i in range(1, len(rows)) if stamps[i] - stamps[i - 1] > 150000]
        gap_set = set(gaps)
        annotations = 0
        for start in range(0, len(rows), chunk_size):
            batch = rows[start : start + chunk_size]
            tokens = [row["pc_token"] for row in batch]
            slots = {token: [] for token in tokens}
            marks = ",".join("?" for _ in tokens)
            for box in conn.execute(
                f"""SELECT b.*,c.name AS category_name FROM lidar_box b
                LEFT JOIN track t ON t.token=b.track_token
                LEFT JOIN category c ON c.token=t.category_token
                WHERE b.lidar_pc_token IN ({marks}) ORDER BY b.lidar_pc_token,b.track_token""",
                tokens,
            ):
                records = slots[box["lidar_pc_token"]]
                if len(records) >= 8192:
                    raise ValueError("Source frame object budget exceeded")
                records.append(box)
            lights = {token: [] for token in tokens}
            for light in conn.execute(
                f"SELECT * FROM traffic_light_status WHERE lidar_pc_token IN ({marks})", tokens
            ):
                lights[light["lidar_pc_token"]].append(
                    [str(light["lane_connector_id"]), str(light["status"])]
                )
            frames = []
            for offset, row in enumerate(batch):
                frame = start + offset
                yaw = yaw_from_quaternion([row[k] for k in ("qw", "qx", "qy", "qz")])
                ex, ey = float(row["x"]) - origin[0], float(row["y"]) - origin[1]
                ego_path.append([ex, ey])
                values = [
                    [
                        0,
                        ex + 1.461 * np.cos(yaw),
                        ey + 1.461 * np.sin(yaw),
                        yaw,
                        5.176,
                        2.297,
                        float(row["z"]),
                        1.777,
                        float(row["vx"]),
                        float(row["vy"]),
                    ]
                ]
                seen = set()
                for box in slots[row["pc_token"]]:
                    if not box["track_token"] or box["category_name"] not in CATEGORIES:
                        raise ValueError("Unknown source category/identity")
                    identity, kind = box["track_token"].hex(), box["category_name"]
                    if identity in seen:
                        raise ValueError("Duplicate track in source frame")
                    seen.add(identity)
                    if identity not in lookup:
                        if len(actors) >= 20000:
                            raise ValueError("Source unique actor budget exceeded")
                        lookup[identity] = len(actors)
                        actors.append({"id": identity, "kind": kind, "intervals": []})
                    index = lookup[identity]
                    actor = actors[index]
                    if actor["kind"] != kind:
                        raise ValueError("Track category changed")
                    spans = actor["intervals"]
                    if spans and spans[-1][1] == frame - 1 and frame not in gap_set:
                        spans[-1][1] = frame
                    else:
                        spans.append([frame, frame])
                    geometry = [
                        float(box[k]) for k in ("x", "y", "yaw", "length", "width", "z", "height")
                    ]
                    if (
                        not np.isfinite(geometry).all()
                        or min(geometry[3], geometry[4], geometry[6]) <= 0
                    ):
                        raise ValueError("Invalid source object geometry")
                    geometry[0] -= origin[0]
                    geometry[1] -= origin[1]
                    velocity = [box["vx"], box["vy"]]
                    velocity = [
                        float(v) if v is not None and np.isfinite(v) else None for v in velocity
                    ]
                    values.append([index, *geometry, *velocity])
                annotations += len(values) - 1
                counts.append(len(values))
                frames.append(
                    {
                        "index": frame,
                        "timestamp_us": int(row["pc_us"]),
                        "token": row["pc_token"].hex(),
                        "ego_pose_timestamp_us": int(row["timestamp"]),
                        "scene_token": row["scene_token"].hex(),
                        "objects": values,
                        "signals": lights[row["pc_token"]],
                    }
                )
            name = f"chunk-{start // chunk_size:04d}.json"
            digest = dump(output / name, {"start": start, "frames": frames})
            chunks.append({"path": name, "start": start, "count": len(frames), "sha256": digest})
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("Source DB changed during export")
        map_lines, anchors, keys = [], [], set()
        for key, scene in scenes:
            source = scene.metadata["nuplan_source"]
            if Path(source["db_path"]).resolve() != path:
                raise ValueError("Compiled map patch belongs to another DB")
            anchor = int(np.searchsorted(stamps, source["anchor_timestamp_us"]))
            anchors.append(
                {
                    "id": key,
                    "frame": anchor,
                    "diagnostic": f"../../{key}/diagnostic.html",
                    "source": f"../../{key}/source.html",
                }
            )
            translation = np.asarray(scene.metadata["origin_xyz"][:2]) - origin
            for i, kind in enumerate(scene.road_types):
                points = (
                    scene.road_points[scene.road_offsets[i] : scene.road_offsets[i + 1], :2]
                    + translation
                )
                signature = hashlib.sha256(points.tobytes()).hexdigest()
                if signature not in keys:
                    keys.add(signature)
                    map_lines.append({"kind": int(kind), "points": points.tolist()})
        report = {
            "schema_version": 1,
            "mode": "raw_source_only",
            "status": "complete",
            **header,
            "db_path": str(path),
            "source_stat": {"size": before.st_size, "mtime_ns": before.st_mtime_ns},
            "frame_count": len(rows),
            "duration_seconds": (stamps[-1] - stamps[0]) / 1e6,
            "timestamps_us": stamps,
            "gaps_before_frames": gaps,
            "origin_xy": origin,
            "actors": actors,
            "counts": counts,
            "ego_path": ego_path,
            "chunks": chunks,
            "chunk_size": chunk_size,
            "annotation_count": annotations,
            "velocity_scope": "Raw DB columns, not a uniform projected map velocity contract for ego and boxes",
            "map_lines": map_lines,
            "map_scope": "compiled_window_patches_only",
            "anchors": anchors,
            "collision_assessed": False,
            "simulation": False,
            "trainable": False,
            "fields": [
                "actor_index",
                "x",
                "y",
                "yaw",
                "length",
                "width",
                "z_global",
                "height",
                "vx_source_db",
                "vy_source_db",
            ],
        }
        dump(output / "manifest.json", report)
        (output / "viewer.html").write_text(LOG_HTML)
        return report


def build_full_logs(gallery):
    from driversi.contracts.scene import Scene
    from driversi.rendering.inspection.nuplan_gallery import render_index, save

    root = Path(gallery).resolve()
    report = json.loads((root / "report.json").read_text())
    grouped = {}
    for case in report["cases"]:
        if case["status"] != "complete":
            continue
        scene = Scene.load(root / case["scene"])
        grouped.setdefault(scene.metadata["nuplan_source"]["db_path"], []).append(
            (case["id"], scene)
        )
    report["full_logs"] = []
    for i, (path, scenes) in enumerate(grouped.items()):
        folder = root / "logs" / f"log-{i + 1:02d}"
        result = export_log(path, folder, scenes=scenes)
        entry = {
            k: result[k]
            for k in (
                "source_log",
                "map_name",
                "frame_count",
                "duration_seconds",
                "annotation_count",
            )
        }
        entry.update(
            id=folder.name,
            viewer=str(folder.relative_to(root) / "viewer.html"),
            actors=len(result["actors"]),
            peak_present=max(result["counts"]),
            gaps=len(result["gaps_before_frames"]),
        )
        report["full_logs"].append(entry)
        for case in report["cases"]:
            for anchor in result["anchors"]:
                if case["id"] == anchor["id"]:
                    case["full_log_viewer"] = entry["viewer"] + "?frame=" + str(anchor["frame"])
        save(root / "report.json", report)
        render_index(root, report)
        print(json.dumps(entry), flush=True)
    return report


def refresh_full_maps(gallery, maps_root):
    """Use actual full-log object extents; never substitute an ego polyline for roads."""
    from driversi.data.road_network.maps import extract_map
    reports = []
    for path in sorted(Path(gallery).glob("logs/*/manifest.json")):
        manifest = json.loads(path.read_text())
        bounds = []
        for chunk in manifest["chunks"]:
            data = (path.parent / chunk["path"]).read_bytes()
            if hashlib.sha256(data).hexdigest() != chunk["sha256"]:
                raise ValueError("Source chunk changed")
            xy = np.asarray([v[1:3] for f in json.loads(data)["frames"] for v in f["objects"]])
            bounds.extend([xy.min(axis=0), xy.max(axis=0)])
        points, offsets, kinds, metadata = extract_map(
            manifest["map_name"], bounds, manifest["origin_xy"], Path(maps_root),
            margin=80, max_extent_m=10000)
        manifest["map_lines"] = [
            {"kind": int(kind), "points": points[offsets[i]:offsets[i+1], :2].tolist()}
            for i, kind in enumerate(kinds)]
        manifest["map_scope"] = "full_log_object_extent_plus80m"
        manifest["map_evidence"] = metadata
        # Replace only once the complete extraction succeeded; source chunks unchanged.
        temporary = path.with_suffix(".new.json")
        dump(temporary, manifest)
        temporary.replace(path)
        (path.parent / "viewer.html").write_text(LOG_HTML)
        row = {"log": path.parent.name, "map_lines": len(kinds), "map_points": len(points),
               "bounds": metadata["map_bounds_global"], "scope": manifest["map_scope"]}
        reports.append(row)
        print(json.dumps(row), flush=True)
    dump(Path(gallery) / "full-map-report.json", reports)
    return reports


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gallery", type=Path, required=True)
    parser.add_argument("--refresh-maps", action="store_true")
    parser.add_argument("--maps-root", type=Path)
    args = parser.parse_args()
    if args.refresh_maps:
        if args.maps_root is None:
            parser.error("--refresh-maps requires --maps-root")
        refresh_full_maps(args.gallery, args.maps_root)
    else:
        build_full_logs(args.gallery)
