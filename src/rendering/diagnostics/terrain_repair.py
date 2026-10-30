"""Derive bounded terrain repairs from any sealed portable physical render.

The source world, Gaussian assets and admission rules are unchanged. This is a
reusable repair operation, not a planar display conversion or training approval.
"""

import argparse
from pathlib import Path

import numpy as np

from driversi.agents.backends.native import digest
from driversi.configuration import read_mapping
from driversi.rendering.gaussian.worldengine_geometry import RoadHeightMap, UnsupportedTerrain
from driversi.rendering.gaussian.worldengine_terrain_repair import repair_road_height_holes
from driversi.runtime.budget import write_json


def repair_queries(grid, transform, queries):
    """Use original measured cells only; also recheck previously supported queries."""
    terrain = RoadHeightMap(grid, transform)
    requested, before = set(), []
    for query in queries:
        center = np.asarray(query["center"], float)
        translation = np.asarray(query["translation"], float)
        if (
            center.shape != (3,)
            or translation.shape != (3,)
            or not np.isfinite([center, translation]).all()
        ):
            raise ValueError("Finite actual source/target positions required")
        try:
            terrain.lift(center, translation)
            before.append(True)
        except UnsupportedTerrain:
            before.append(False)
            for xy in (center[:2], (center + translation)[:2]):
                uv = terrain.scale * (terrain.rotation @ xy + terrain.translation)
                col, row = map(int, np.floor(uv))
                for y, x in ((row, col), (row, col + 1), (row + 1, col), (row + 1, col + 1)):
                    if 0 <= y < terrain.grid.shape[0] and 0 <= x < terrain.grid.shape[1]:
                        if np.isnan(terrain.grid[y, x]):
                            requested.add((y, x))
    fixed, evidence = repair_road_height_holes(grid, transform, sorted(requested))
    derived, rows = RoadHeightMap(fixed, transform), []
    for query, was_supported in zip(queries, before, strict=True):
        row = dict(query, supported_before=was_supported)
        try:
            _, support = derived.lift(query["center"], query["translation"])
            row.update(supported_after=True, support=support)
        except UnsupportedTerrain as error:
            row.update(supported_after=False, remaining_reason=str(error))
        rows.append(row)
    return fixed, dict(
        repair=evidence,
        queries=rows,
        failures_before=sum(not x for x in before),
        failures_after=sum(not x["supported_after"] for x in rows),
        all_queries_supported=bool(rows) and all(x["supported_after"] for x in rows),
        quality_admitted=False,
        trainable=False,
    )


def run(render_report, output):
    from driversi.rendering.gaussian.mtgs_registration import motion_in_source
    from driversi.rendering.gaussian.worldengine_portable import (
        PortableRenderer,
        ego_center_pivot,
        supported_log_pose,
    )

    render_report = Path(render_report).resolve(strict=True)
    report = read_mapping(render_report)
    if (
        report.get("renderer_backend") != "worldengine_portable"
        or report.get("status") != "rendered_diagnostic"
    ):
        raise ValueError("Complete portable physical render required")
    for name, checksum in report["inputs"].items():
        if digest(name) != checksum:
            raise ValueError("Source render input changed")

    def one_file(predicate):
        found = [Path(p) for p in report["inputs"] if predicate(Path(p))]
        if len(found) != 1:
            raise ValueError("Ambiguous bound terrain/metadata/checkpoint input")
        return found[0]

    height = one_file(lambda p: p.name == "road_height_map.npy")
    transform = one_file(lambda p: p.name == "sim2.json")
    metadata = one_file(lambda p: p.name == "video_scene_dict.pkl")
    checkpoint = one_file(lambda p: p.suffix == ".ckpt")
    if height.parent != transform.parent:
        raise ValueError("Terrain grid and transform must belong to the same bundle")
    output = Path(output).resolve()
    if output.exists():
        previous = read_mapping(output / "report.json")
        if previous["render_report_sha256"] != digest(render_report):
            raise ValueError("Repair directory belongs to another render")
        for name, checksum in previous["files"].items():
            if digest(name) != checksum:
                raise ValueError("Saved derived terrain changed")
        return previous
    renderer = PortableRenderer(checkpoint, metadata=metadata, terrain=height.parent, device="cpu")
    if (
        renderer.binding.asset_id != report["asset_id"]
        or renderer.binding.source_log != report["source_log"]
    ):
        raise ValueError("Bound source differs from rendered asset")
    queries = []
    for frame in report["frames"]:
        source = renderer.binding.resolve(frame["source_token"])
        for actor, motion in frame["motions"].items():
            if actor == "ego":
                center = ego_center_pivot(source)
            else:
                model = renderer.models["rigid_object_" + actor]
                pose = supported_log_pose(model, source["timestamp"])
                if pose is None:
                    raise ValueError("Rendered actor lacks a supported source pose")
                center = pose[1].detach().cpu().numpy()
            translation = motion_in_source(source, motion[:2], motion[2])[0]
            queries.append(
                dict(
                    tick=frame["tick"],
                    actor=actor,
                    center=center.tolist(),
                    translation=translation.tolist(),
                )
            )
    grid = np.load(height, allow_pickle=False)
    transform_value = read_mapping(transform)
    fixed, evidence = repair_queries(grid, transform_value, queries)
    output.mkdir(parents=True, exist_ok=False)
    terrain = output / "terrain"
    terrain.mkdir()
    np.save(terrain / "road_height_map.npy", fixed)
    write_json(terrain / "sim2.json", transform_value)
    result = dict(
        schema_version=1,
        kind="bounded_render_terrain_repair",
        status="ready_for_rerender"
        if evidence["all_queries_supported"]
        else "additional_measured_support_required",
        render_report=str(render_report),
        render_report_sha256=digest(render_report),
        asset_id=report["asset_id"],
        source_log=report["source_log"],
        source_inputs=report["inputs"],
        derived_terrain=str(terrain),
        files={str(p): digest(p) for p in (terrain / "road_height_map.npy", terrain / "sim2.json")},
        retry_requires="rerender_same_branch_then_original_visual_admission",
        **evidence,
    )
    write_json(output / "report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.render_report, args.output)
    print({k: result[k] for k in ("status", "failures_before", "failures_after", "trainable")})


if __name__ == "__main__":
    main()
