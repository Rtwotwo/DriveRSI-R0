"""Re-render a recorded joint trajectory with all configured source cameras.

Reads saved states without executing a native simulator. Rejects state changes.
This cannot produce new decisions or training interactions.
"""

import argparse
import json
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--world-python", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--memory-fraction", type=float, default=0.10)
    parser.add_argument("--trusted-assets", action="store_true")
    args = parser.parse_args()
    if not args.trusted_assets or not 0 < args.memory_fraction <= 0.5:
        raise ValueError("Explicit trusted assets and bounded GPU memory required")
    if args.output.exists():
        raise FileExistsError(args.output)
    import torch

    free, total = torch.cuda.mem_get_info()
    if free < args.memory_fraction * total + 1024**3:
        raise RuntimeError("Insufficient free memory for the reserved bounded run")
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    from driversi.rendering.gaussian.mtgs_runtime import MTGSRuntime
    from driversi.rendering.gaussian.mtgs_session import MTGSWorldSession

    old = json.loads(args.manifest.read_text())
    if len({x["generation"] for x in old["frames"]}) != 1:
        raise ValueError("Exactly one recorded generation is required")
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    assets = old["assets"]
    inputs = old["inputs"]
    runtime = MTGSRuntime(
        config_path=Path(assets["config"]["path"]),
        checkpoint_path=Path(assets["checkpoint"]["path"]),
        road_block_config=Path(assets["road_block_config"]["path"]),
        scale=0.5,
        trusted_assets=True,
    )
    print("Runtime loaded", flush=True)
    session = MTGSWorldSession(
        runtime=runtime,
        scene=Path(inputs["scene"]["path"]),
        road_graph=Path(inputs["road_graph"]["path"]),
        binding=Path(inputs["binding"]["path"]),
        world_python=args.world_python,
        native_path=args.native,
        output=args.output / "session",
        cameras=("CAM_F0", "CAM_L0", "CAM_L1", "CAM_L2", "CAM_R0", "CAM_R1", "CAM_R2", "CAM_B0"),
        sensor_stride=2,
        virtual_camera_actors=tuple(old["virtual_camera_actors"]),
        height_mode="reference-track",
        population_config=Path(inputs["population"]["path"]),
        recorded_manifest=args.manifest,
        image_write_workers=2,
    )
    try:
        if session.current["build_id"] != old["physics_build_id"]:
            raise ValueError("Physics build mismatch")
        initial = next(iter(old["initial_world_by_generation"].values()))
        for key in ("state", "motion", "presence"):
            if session.current[key] != initial[key]:
                raise ValueError("Initial state mismatch: " + key)
        for step in old["steps"]:
            session.advance(
                expected_tick=session.current["tick"],
                decision_identity=session.decision_identity(),
            )
            actual = session.manifest["steps"][-1]
            for key in (
                "state",
                "motion",
                "presence",
                "executed_actions",
                "valid",
                "terminated",
                "truncated",
                "events",
                "alive",
                "done",
            ):
                if key in step and actual.get(key) != step[key]:
                    raise ValueError(f"Replay mismatch: {key} at {session.current['tick']}")
            if session.current["tick"] % 10 == 0:
                print(
                    "tick",
                    session.current["tick"],
                    "elapsed",
                    round(time.perf_counter() - started, 2),
                    flush=True,
                )
        report = dict(
            status="completed",
            physics_steps=len(old["steps"]),
            replay_exact=True,
            world_source="recorded_states_no_new_simulation",
            source_manifest_sha256=__import__("hashlib")
            .sha256(args.manifest.read_bytes())
            .hexdigest(),
            camera_ids=session.manifest["camera_ids"],
            camera_frames=sum(len(f["cameras"]) for f in session.manifest["frames"]),
            wall_seconds=time.perf_counter() - started,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            trainable=False,
            learned_self_play=False,
        )
        (args.output / "verification.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
    finally:
        session.close()


if __name__ == "__main__":
    main()
