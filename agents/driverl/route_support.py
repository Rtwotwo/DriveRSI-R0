"""Use native DriveRL navigation correction for simulator support, not policy updates."""

import argparse
from pathlib import Path


def correct(scene_path, maps_root, output):
    from driversi.agents.backends.sources import activate_driverl

    activate_driverl()
    from driverl.nuplan.route_correction import correct_route_roadblock_ids
    from driversi.agents.backends.native import digest
    from driversi.contracts.scene import Scene
    from driversi.runtime.budget import write_json
    from driversi_agents.driverl.navsim_context import scenario_from_scene

    scene = Scene.load(Path(scene_path))
    # Only current rear-axle/map/navigation are used. Dense bootstrap is irrelevant here.
    scene.metadata.pop("driverl_history_mode", None)
    context = scenario_from_scene(scene, Path(maps_root))
    raw = context.get_route_roadblock_ids()
    corrected = correct_route_roadblock_ids(
        context.get_ego_state_at_iteration(0), context.map_api, raw
    )
    if not corrected:
        raise ValueError("Native map correction found no supported navigation")
    result = dict(
        schema_version=1,
        kind="native_driverl_navigation_correction",
        scene_sha256=digest(scene_path),
        raw=raw,
        corrected=corrected,
        scope="current_pose_and_directed_map_not_executed_future",
    )
    write_json(output, result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", required=True, type=Path)
    p.add_argument("--maps-root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    correct(a.scene, a.maps_root, a.output)


if __name__ == "__main__":
    main()
