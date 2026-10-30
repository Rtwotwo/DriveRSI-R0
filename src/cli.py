"""Stable command-line entry point. Commands delegate to the owning modules."""

import argparse
import json
from pathlib import Path

from driversi import __version__


def write_json(value: dict, output: Path | None) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if output is None:
        print(text, end="")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="driversi")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    from driversi.training.legacy.config import add_training_arguments

    training = commands.add_parser("legacy-train", help="Retired diagnostic PPO, explicit compatibility use only")
    add_training_arguments(training)
    agents_command = commands.add_parser("legacy-agents", help="List retired diagnostic model configs")
    agents_command.add_argument("--agent-dir", type=Path)
    agents_command.add_argument("--manifest", type=Path, help="Inspect a model capability JSON/YAML without importing it")
    agents_command.add_argument("--for-runtime", choices=("structured_ppo", "visual_sampling"))
    commands.add_parser("agents", help="List native model integrations and capability boundaries")
    presets = commands.add_parser("config", help="Inspect versioned platform configuration presets")
    preset_actions = presets.add_subparsers(dest="config_action", required=True)
    list_presets = preset_actions.add_parser("list")
    list_presets.add_argument("--group")
    for action in ("path", "show"):
        preset_actions.add_parser(action).add_argument("preset")
    services = commands.add_parser("services", help="List actual DriveHarness services by responsibility")
    services.add_argument("--group")
    commands.add_parser("train", help="Retired implicit training entry; select a native model explicitly")
    native = commands.add_parser("model", help="Plan or run a pinned native model job")
    native.add_argument("--config", type=Path, required=True)
    native.add_argument("--operation", choices=("preflight", "inference", "evaluate", "train"), required=True)
    native.add_argument("--output", type=Path, required=True)
    native.add_argument("--execute", action="store_true")
    native.add_argument("--max-wall-seconds", type=float)
    sampling_benchmark = commands.add_parser("benchmark-sampling", help="Measure CPU closed-loop sampling without learning")
    sampling_benchmark.add_argument("--agent-config", type=Path, required=True)
    sampling_benchmark.add_argument("--suite", type=Path, required=True)
    sampling_benchmark.add_argument("--output", type=Path, required=True)
    for key, default in (("worlds", 32), ("workers", 0), ("steps", 16), ("batches", 5),
                         ("seed", 0), ("scene-cache-size", 32), ("max-rollout-mib", 1024)):
        sampling_benchmark.add_argument("--" + key, type=int, default=default)
    sampling_benchmark.add_argument("--worker-timeout-seconds", type=float, default=30.)
    sampling_benchmark.add_argument("--record", action="store_true")
    sampling_benchmark.add_argument("--min-seconds", type=float, default=0.,
                                    help="Minimum probe wall time after startup; batches is also a minimum")
    fleet = commands.add_parser("configure-vehicle-fleet", help="Assign feedback or bounded hold/brake to every source-present vehicle; learner slots stay separate")
    fleet.add_argument("--scene", type=Path, required=True)
    fleet.add_argument("--output", type=Path, required=True)
    background = commands.add_parser("reactive-background", help="Assign/audit background paths and run paired real-scene interactions")
    background.add_argument("--scene", type=Path, required=True)
    background.add_argument("--road-graph", type=Path, required=True)
    background.add_argument("--context", type=Path)
    background.add_argument("--population-config", type=Path)
    background.add_argument("--brake-actor")
    background.add_argument("--observation-memory", action=argparse.BooleanOptionalAction, default=True)
    background.add_argument("--output", type=Path, required=True)
    crossing = commands.add_parser("reactive-crossing", help="Decision-level pedestrian/vehicle counterfactual audit")
    crossing.add_argument("--scene", type=Path, required=True)
    crossing.add_argument("--context", type=Path)
    crossing.add_argument("--vehicle", required=True)
    crossing.add_argument("--pedestrian", required=True)
    crossing.add_argument("--output", type=Path, required=True)
    describe = commands.add_parser("describe", help="Describe implemented platform capabilities")
    describe.add_argument("--output", type=Path)
    mtgs_assets = commands.add_parser("prepare-mtgs-assets", help="Verify and optionally extract the predownloaded, pinned MTGS pilot bundle")
    mtgs_assets.add_argument("--asset-root", type=Path, required=True)
    mtgs_assets.add_argument("--extract", action="store_true")
    mtgs_assets.add_argument("--output", type=Path)
    mtgs_metadata = commands.add_parser("inspect-mtgs-metadata", help="Inspect trusted source metadata; does not certify simulator coverage")
    mtgs_metadata.add_argument("--metadata", type=Path, required=True)
    mtgs_metadata.add_argument("--trusted-pickle", action="store_true")
    mtgs_metadata.add_argument("--output", type=Path)
    mtgs_bind = commands.add_parser("bind-mtgs-source", help="Audit exact NAVSIM and MTGS source identity for a compiled scene")
    for option in ("log-file", "metadata", "scene", "output"):
        mtgs_bind.add_argument("--"+option, type=Path, required=True)
    mtgs_bind.add_argument("--travel-id", type=int, required=True)
    mtgs_bind.add_argument("--camera", default="CAM_F0")
    mtgs_bind.add_argument("--trusted-pickle", action="store_true")
    mtgs_render = commands.add_parser("render-mtgs-probe", help="Render a matched MTGS checkpoint with isolated object translation and restoration")
    mtgs_render.add_argument("--config", type=Path, required=True)
    mtgs_render.add_argument("--checkpoint", type=Path, required=True)
    mtgs_render.add_argument("--road-block-config", type=Path, required=True)
    mtgs_render.add_argument("--output-dir", type=Path, required=True)
    mtgs_render.add_argument("--camera", default="CAM_F0")
    mtgs_render.add_argument("--frames", type=int, default=3)
    mtgs_render.add_argument("--scale", type=float, default=.5)
    mtgs_render.add_argument("--translation", nargs=3, type=float, default=[0., 2., 0.])
    mtgs_render.add_argument("--start-index", type=int, default=0)
    mtgs_render.add_argument("--frame-stride", type=int, default=5)
    mtgs_render.add_argument("--trusted-assets", action="store_true")
    mtgs_session = commands.add_parser("run-mtgs-session", help="Run matched NAVSIM world-driven rendering and optional loopback controls")
    for option in ("config", "checkpoint", "road-block-config", "scene", "road-graph", "binding",
                   "world-python", "native-path", "output-dir"):
        mtgs_session.add_argument("--"+option, type=Path, required=True)
    mtgs_session.add_argument("--scale", type=float, default=.5)
    mtgs_session.add_argument("--sensor-stride", type=int, default=2)
    mtgs_session.add_argument("--cameras", nargs="+", default=["CAM_F0", "CAM_R0"])
    mtgs_session.add_argument("--virtual-camera-actors", nargs="*", default=[],
                              help="Controlled reconstructed vehicles receiving a diagnostic roof camera")
    mtgs_session.add_argument("--height-mode", choices=["source-time", "reference-track"],
                              default="source-time", help="Explicit bounded display height correction; not terrain")
    mtgs_session.add_argument("--population-config", type=Path, help="Fixed per-agent policy population JSON")
    mtgs_session.add_argument("--visual-policy", action=argparse.BooleanOptionalAction, default=False,
                              help="Managed RGB diagnostic decisions; independent of rendering/recording")
    mtgs_session.add_argument("--visual-actors", nargs="*", default=[], help="Visual actor subset; default controlled camera owners")
    mtgs_session.add_argument("--visual-history", type=int, default=2)
    mtgs_session.add_argument("--desired-speed", type=float, default=13.)
    mtgs_session.add_argument("--paired-brake-actor")
    mtgs_session.add_argument("--serve-port", type=int)
    mtgs_session.add_argument("--trusted-assets", action="store_true")
    mtgs_export = commands.add_parser("export-mtgs-episode", help="Validate and export recorded world and camera sequences; not a PPO batch")
    mtgs_export.add_argument("--manifest", type=Path, required=True)
    mtgs_export.add_argument("--output-dir", type=Path, required=True)
    visual_export = commands.add_parser("export-visual-samples", help="Validate actual visual decisions and full physical action intervals")
    visual_export.add_argument("--manifest", type=Path, required=True)
    visual_export.add_argument("--output-dir", type=Path, required=True)
    mtgs_mine = commands.add_parser("mine-mtgs-episode", help="Mine auditable event and render-quality candidate clips, never training labels")
    mtgs_mine.add_argument("--episode", type=Path, required=True)
    mtgs_mine.add_argument("--output-dir", type=Path, required=True)
    mtgs_mine.add_argument("--context-before-seconds", type=float, default=1.)
    mtgs_mine.add_argument("--context-after-seconds", type=float, default=1.)
    mtgs_mine.add_argument("--braking-threshold-mps2", type=float, default=-3.)
    mtgs_mine.add_argument("--minimum-speed-mps", type=float, default=1.)
    mtgs_mine.add_argument("--viewer", action="store_true")
    mtgs_mine.add_argument("--web-root", type=Path)
    sensors_command = commands.add_parser("audit-sensors", help="Audit source NAVSIM camera files and full calibration")
    sensors_command.add_argument("--log-file", type=Path, required=True)
    sensors_command.add_argument("--frame-index", type=int, default=0)
    sensors_command.add_argument("--sensor-root", type=Path, required=True)
    sensors_command.add_argument("--trusted-pickle", action="store_true")
    sensors_command.add_argument("--output", type=Path)
    source_gallery_command = commands.add_parser("build-source-gallery", help="Inspect original NAVSIM photos, source boxes and maps")
    source_gallery_command.add_argument("--config", type=Path, required=True)
    source_gallery_command.add_argument("--dataset-root", type=Path, required=True)
    source_gallery_command.add_argument("--output-dir", type=Path, required=True)
    source_gallery_command.add_argument("--trusted-pickle", action="store_true")
    source_gallery_command.add_argument("--diagnostic-record", type=Path,
                                        help="Optional primitive RGB illustration; not a paired image metric")
    visual_command = commands.add_parser("rollout-visual", help="Run online primitive cameras and an RGB-only diagnostic policy")
    visual_command.add_argument("--scene", type=Path, required=True)
    visual_command.add_argument("--output-dir", type=Path, required=True)
    visual_command.add_argument("--width", type=int, default=160)
    visual_command.add_argument("--height", type=int, default=96)
    visual_command.add_argument("--sensor-stride", type=int, default=1)
    visual_command.add_argument("--history", type=int, default=2)
    visual_command.add_argument("--seed", type=int, default=0)
    visual_command.add_argument("--desired-speed", type=float, default=4.)
    visual_command.add_argument("--allow-unverified-heading", action="store_true",
                                help="Explicitly allow virtual box-axis cameras; does not certify vehicle-front direction")
    visual_test_command = commands.add_parser("evaluate-visual", help="Run paired RGB interaction checks and publish visual evidence")
    visual_test_command.add_argument("--config", type=Path, required=True)
    visual_test_command.add_argument("--output-dir", type=Path, required=True)
    visual_test_command.add_argument("--real-scene", type=Path)
    visual_test_command.add_argument("--allow-unverified-heading", action="store_true")
    nuplan_inventory = commands.add_parser("inventory-nuplan", help="Inspect nuPlan v1.1 DB partitions without reading sensors")
    nuplan_inventory.add_argument("--dataset-root", type=Path, required=True)
    nuplan_inventory.add_argument("--output", type=Path)
    nuplan_index = commands.add_parser("index-nuplan", help="Index bounded readonly nuPlan DB windows")
    nuplan_index.add_argument("--dataset-root", type=Path, required=True)
    nuplan_index.add_argument("--partitions", nargs="+", required=True)
    nuplan_index.add_argument("--max-logs", type=int, default=1)
    nuplan_index.add_argument("--windows-per-log", type=int, default=2)
    nuplan_index.add_argument("--duration", type=float, default=9.)
    nuplan_index.add_argument("--history-seconds", type=float, default=1.5)
    nuplan_index.add_argument("--max-actors", type=int, default=32)
    nuplan_index.add_argument("--split-policy", type=Path)
    nuplan_index.add_argument("--protected-logs", type=Path, nargs="*", default=[])
    nuplan_index.add_argument("--output", type=Path, required=True)
    inventory_command = commands.add_parser("inventory-navsim", help="Inspect public NAVSIM data partitions")
    inventory_command.add_argument("--dataset-root", type=Path, required=True)
    inventory_command.add_argument("--output", type=Path)
    index_command = commands.add_parser("index-navsim", help="Index bounded NAVSIM scene windows")
    index_command.add_argument("--dataset-root", type=Path, required=True)
    index_command.add_argument("--partitions", nargs="+", choices=["mini", "trainval", "test"], default=["mini"])
    index_command.add_argument("--max-logs", type=int, default=3)
    index_command.add_argument("--windows-per-log", type=int, default=2)
    index_command.add_argument("--duration", type=float, default=3.0)
    index_command.add_argument("--duration-mode", choices=["exact", "fit_source"], default="exact")
    index_command.add_argument("--min-duration", type=float, default=0.1)
    index_command.add_argument("--max-actors", type=int, default=32)
    index_command.add_argument("--trusted-pickle", action="store_true")
    index_command.add_argument("--output", type=Path, required=True)
    query_command = commands.add_parser("list-scenes", help="Query a source-checked scene catalogue")
    query_command.add_argument("--catalog", type=Path, required=True)
    query_command.add_argument("--limit", type=int, default=20)
    query_command.add_argument("--offset", type=int, default=0)
    query_command.add_argument("--status", choices=["all", "geometry_pending", "rejected", "conflict"], default="geometry_pending")
    query_command.add_argument("--map-name")
    query_command.add_argument("--min-controlled", type=int, default=0)
    query_command.add_argument("--output", type=Path)
    library_command = commands.add_parser("compile-catalog", help="Compile selected indexed scene assets")
    library_command.add_argument("--catalog", type=Path, required=True)
    library_command.add_argument("--dataset-root", type=Path)
    library_command.add_argument("--maps-root", type=Path)
    library_command.add_argument("--limit", type=int, default=8)
    library_command.add_argument("--offset", type=int, default=0)
    library_command.add_argument("--map-name")
    library_command.add_argument("--min-controlled", type=int, default=1)
    library_command.add_argument("--trusted-pickle", action="store_true")
    library_command.add_argument("--workers", type=int, default=0, help="Source-log processes (0=serial, maximum 64)")
    library_command.add_argument("--output-dir", type=Path, required=True)
    audit = commands.add_parser("audit-navsim", help="Audit trusted local NAVSIM log schemas")
    audit.add_argument("--log-dir", type=Path, required=True)
    audit.add_argument("--max-files", type=int, default=3)
    audit.add_argument("--trusted-pickle", action="store_true")
    audit.add_argument("--output", type=Path)
    horizons_command = commands.add_parser("audit-horizons", help="Inspect source continuity and population loss without compiling")
    horizons_command.add_argument("--log-file", type=Path, required=True)
    horizons_command.add_argument("--start-frame", type=int, default=0)
    horizons_command.add_argument("--durations", nargs="+", type=float, default=[3.0, 9.0, 30.0, 60.0])
    horizons_command.add_argument("--trusted-pickle", action="store_true")
    horizons_command.add_argument("--output", type=Path)
    compile_command = commands.add_parser("compile-navsim", help="Compile one scene from trusted NAVSIM logs")
    compile_command.add_argument("--log-file", type=Path, required=True)
    compile_command.add_argument("--maps-root", type=Path, required=True)
    compile_command.add_argument("--start-frame", type=int, default=0)
    compile_command.add_argument("--duration", type=float, default=9.0)
    compile_command.add_argument("--duration-mode", choices=["exact", "fit_source"], default="exact")
    compile_command.add_argument("--min-duration", type=float, default=0.1)
    compile_command.add_argument("--max-actors", type=int, default=32)
    compile_command.add_argument("--split", choices=["mini", "train", "development", "heldout"], required=True)
    compile_command.add_argument("--trusted-pickle", action="store_true")
    compile_command.add_argument("--output", type=Path, required=True)
    rollout_command = commands.add_parser("rollout", help="Run an engineering controller in a compiled scene")
    rollout_command.add_argument("--scene", type=Path, required=True)
    rollout_command.add_argument("--policy", choices=["coast", "brake", "follow", "traffic"], default="coast")
    rollout_command.add_argument("--road-graph", type=Path)
    rollout_command.add_argument("--route-tasks", type=Path)
    rollout_command.add_argument("--output", type=Path, required=True)
    graph_command = commands.add_parser("compile-lanes", help="Compile a versioned initial-position lane graph")
    graph_command.add_argument("--scene", type=Path, required=True)
    graph_command.add_argument("--maps-root", type=Path, required=True)
    graph_command.add_argument("--margin", type=float, default=120)
    graph_command.add_argument("--output", type=Path, required=True)
    following_command = commands.add_parser("evaluate-following", help="Run paired braking and negative controls")
    following_command.add_argument("--config", type=Path, required=True)
    following_command.add_argument("--output-dir", type=Path, required=True)
    probe_command = commands.add_parser("probe-interaction", help="Test a controlled following pair in an existing scene")
    probe_command.add_argument("--scene", type=Path, required=True)
    probe_command.add_argument("--road-graph", type=Path, required=True)
    probe_command.add_argument("--route-tasks", type=Path)
    probe_command.add_argument("--intervention-seconds", type=float, default=.5)
    probe_command.add_argument("--response-seconds", type=float, default=2.)
    probe_command.add_argument("--output-dir", type=Path, required=True)
    routes_command = commands.add_parser("inspect-routes", help="Explain current-pose lane candidates and unresolved intentions")
    routes_command.add_argument("--scene", type=Path, required=True)
    routes_command.add_argument("--road-graph", type=Path, required=True)
    routes_command.add_argument("--route-tasks", type=Path)
    routes_command.add_argument("--output", type=Path)
    replay_command = commands.add_parser("export-replay", help="Export a standalone interactive bird's-eye inspector")
    replay_command.add_argument("--scene", type=Path, required=True)
    replay_command.add_argument("--rollout", type=Path, required=True)
    replay_command.add_argument("--road-graph", type=Path)
    replay_command.add_argument("--compare", type=Path)
    replay_command.add_argument("--context", type=Path)
    replay_command.add_argument("--output", type=Path, required=True)
    context_command = commands.add_parser("compile-context", help="Preserve source objects and semantic map layers for inspection")
    context_command.add_argument("--scene", type=Path, required=True)
    context_command.add_argument("--log-file", type=Path, required=True)
    context_command.add_argument("--maps-root", type=Path, required=True)
    context_command.add_argument("--margin", type=float, default=40)
    context_command.add_argument("--trusted-pickle", action="store_true")
    context_command.add_argument("--output", type=Path, required=True)
    real_command = commands.add_parser("evaluate-scenes", help="Audit a frozen real-scene set; report coverage and behavior separately")
    real_command.add_argument("--selection", type=Path, required=True)
    real_command.add_argument("--scene-dir", type=Path, required=True)
    real_command.add_argument("--maps-root", type=Path, required=True)
    real_command.add_argument("--output-dir", type=Path, required=True)
    gallery_command = commands.add_parser("build-interaction-gallery", help="Build paired and source-bounded real-scene evidence")
    gallery_command.add_argument("--config", type=Path, required=True)
    gallery_command.add_argument("--scene-dir", type=Path, required=True)
    gallery_command.add_argument("--dataset-root", type=Path, required=True)
    gallery_command.add_argument("--trusted-pickle", action="store_true")
    gallery_command.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "config":
        from driversi.config import catalog, path, load
        if args.config_action == "path":
            print(path(args.preset))
        else:
            write_json(catalog(args.group) if args.config_action == "list" else load(args.preset), None)
        return 0
    if args.command == "services":
        from driversi.rsi.services.catalog import describe
        write_json({"services": describe(args.group)}, None)
        return 0
    if args.command == "train":
        parser.error("Implicit PPO training is retired. Use model --config ... for native jobs; legacy-train is for archived diagnostic runs only.")
    if args.command == "agents":
        from driversi.agents.backends.native import catalog
        from driversi.agents.backends.native import project_root
        from driversi.configuration import read_mapping
        write_json({"agents": catalog(), "portfolio": read_mapping(project_root() / "src/config/model-selection.json"), "native_parity": "not_measured"}, None)
        return 0
    if args.command == "model":
        from driversi.agents.backends.native import execute, prepare
        try:
            plan = prepare(args.config, args.operation, args.output)
            if args.execute:
                if args.max_wall_seconds is None:
                    parser.error("--execute requires --max-wall-seconds")
                result = execute(plan, timeout_seconds=args.max_wall_seconds)
                write_json(result, None)
                return 0 if result["status"] == "completed" else 2
            write_json(plan, None)
            return 0 if plan["ready_to_attempt"] else 2
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
    if args.command == "reactive-crossing":
        from driversi.evaluation.benchmarks.crossing import evaluate_crossing
        write_json(evaluate_crossing(args.scene, args.output, vehicle_id=args.vehicle,
                                     pedestrian_id=args.pedestrian, context_path=args.context), None)
        return 0

    if args.command == "configure-vehicle-fleet":
        from driversi.contracts.scene import Scene
        from driversi.data.participants.fleet import configure_vehicle_fleet

        scene = configure_vehicle_fleet(Scene.load(args.scene))
        scene.save(args.output)
        write_json({"scene": str(args.output), **scene.metadata["vehicle_fleet"]}, None)
        return 0

    if args.command == "reactive-background":
        from driversi.evaluation.benchmarks.reactive import evaluate_background
        write_json(evaluate_background(args.scene, args.road_graph, args.output,
                                      context_path=args.context, population_path=args.population_config,
                                      brake_actor=args.brake_actor, observation_memory=args.observation_memory), None)
        return 0

    if args.command == "benchmark-sampling":
        from driversi.evaluation.benchmarks.sampling import benchmark_sampling

        result = benchmark_sampling(args.suite, args.output, worlds=args.worlds, workers=args.workers,
                                    steps=args.steps, batches=args.batches, seed=args.seed,
                                    scene_cache_size=args.scene_cache_size, max_rollout_mib=args.max_rollout_mib,
                                    worker_timeout_seconds=args.worker_timeout_seconds, record=args.record,
                                    min_seconds=args.min_seconds, agent_config=args.agent_config)
        write_json({k: result[k] for k in ("status", "worlds", "workers", "valid_agent_transitions",
                                         "valid_transitions_per_second", "resets", "total_wall_seconds")}, None)
        return 0
    if args.command == "legacy-agents":
        from driversi.agents.interfaces.catalog import list_agents

        if args.manifest is not None:
            from driversi.configuration import read_mapping
            from driversi.contracts.model_capability import inspect_capability

            if args.agent_dir is not None:
                parser.error("--manifest and --agent-dir are mutually exclusive")
            report = inspect_capability(read_mapping(args.manifest))
            write_json(report, None)
            return 2 if args.for_runtime and not next(
                r["compatible"] for r in report["runtimes"] if r["runtime"] == args.for_runtime) else 0
        if args.for_runtime is not None:
            parser.error("--for-runtime requires --manifest")
        write_json({"agents": list_agents(args.agent_dir)}, None)
        return 0
    if args.command == "legacy-train":
        from driversi.training.ppo import train

        train(args)
        return 0

    if args.command == "bind-mtgs-source":
        from driversi.data.assets.mtgs_binding import audit_source_files

        if args.output.exists():
            parser.error("--output must be a new file")
        report = audit_source_files(log_path=args.log_file, metadata_path=args.metadata,
                                   scene_path=args.scene, travel_id=args.travel_id,
                                   camera_name=args.camera, trusted_pickle=args.trusted_pickle)
        write_json(report, args.output)
        return 0
    if args.command == "export-mtgs-episode":
        from driversi.data.assets.mtgs_episode import export_mtgs_episode

        result = export_mtgs_episode(args.manifest, args.output_dir)
        print(json.dumps({"output": str(args.output_dir), "status": result["status"],
                          "trainable": result["trainable"], "summary": result["summary"],
                          "eligibility": result["eligibility"]}, ensure_ascii=False))
        return 0
    if args.command == "mine-mtgs-episode":
        from driversi.data.scenes.episode_mining import mine_mtgs_episode

        if args.viewer and args.web_root is None:
            parser.error("--viewer requires --web-root")
        result = mine_mtgs_episode(args.episode, args.output_dir,
                                  context_before_seconds=args.context_before_seconds,
                                  context_after_seconds=args.context_after_seconds,
                                  braking_threshold_mps2=args.braking_threshold_mps2,
                                  minimum_speed_mps=args.minimum_speed_mps)
        if args.viewer:
            from driversi.rendering.inspection.case_gallery import write_candidate_viewer

            write_candidate_viewer(args.output_dir / "manifest.json", args.web_root)
        print(json.dumps({"output": str(args.output_dir), "summary": result["summary"],
                          "trainable": result["trainable"]}, ensure_ascii=False))
        return 0
    if args.command == "export-visual-samples":
        from driversi.data.datasets.visual_samples import export_visual_samples

        write_json(export_visual_samples(args.manifest, args.output_dir), None)
        return 0
    if args.command == "run-mtgs-session":
        from driversi.rendering.gaussian.mtgs_session import run_matched_mtgs

        run_matched_mtgs(config_path=args.config, checkpoint_path=args.checkpoint,
                         road_block_config=args.road_block_config, scene=args.scene,
                         road_graph=args.road_graph, binding=args.binding,
                         world_python=args.world_python, native_path=args.native_path,
                         output=args.output_dir, scale=args.scale, sensor_stride=args.sensor_stride,
                         cameras=tuple(args.cameras), desired_speed=args.desired_speed,
                         virtual_camera_actors=tuple(args.virtual_camera_actors), height_mode=args.height_mode,
                         paired_brake_actor=args.paired_brake_actor, serve_port=args.serve_port,
                         trusted_assets=args.trusted_assets, population_config=args.population_config,
                         visual_policy=args.visual_policy, visual_actors=tuple(args.visual_actors),
                         visual_history=args.visual_history)
        return 0
    if args.command == "render-mtgs-probe":
        from driversi.rendering.gaussian.mtgs import render_mtgs_probe
        from driversi.rendering.gaussian.mtgs_viewer import export_mtgs_viewer

        report = render_mtgs_probe(
            config_path=args.config, checkpoint_path=args.checkpoint,
            road_block_config=args.road_block_config, output=args.output_dir,
            camera_name=args.camera, frame_count=args.frames, scale=args.scale,
            displacement=args.translation, start_index=args.start_index,
            frame_stride=args.frame_stride, trusted_assets=args.trusted_assets)
        viewer = export_mtgs_viewer(report, args.output_dir / "viewer.html")
        write_json({"kind": report["kind"], "trainable": report["trainable"],
                    "frames": len(report["frames"]), "viewer": str(viewer),
                    "diagnostics": [frame["diagnostics"] for frame in report["frames"]]}, None)
        return 0
    if args.command == "prepare-mtgs-assets":
        from driversi.data.assets.mtgs_assets import extract_assets, inspect_assets

        if args.output is not None and args.output.exists():
            parser.error("--output must be a new file")
        report = inspect_assets(args.asset_root)
        if args.extract and report["all_verified"]:
            report["extraction"] = extract_assets(args.asset_root)
            extracted = {item["name"]: item["status"] for item in report["extraction"]["assets"]}
            for asset in report["assets"]:
                if asset["name"] in extracted:
                    asset["status"].update(extracted[asset["name"]])
        report["all_extracted"] = all(
            not asset["is_archive"] or asset["status"].get("extracted", False)
            for asset in report["assets"])
        write_json(report, args.output)
        return 0 if report["all_verified"] and (not args.extract or report["all_extracted"]) else 2
    if args.command == "inspect-mtgs-metadata":
        from driversi.data.assets.mtgs_assets import inspect_block_metadata

        if args.output is not None and args.output.exists():
            parser.error("--output must be a new file")
        report = inspect_block_metadata(args.metadata, trusted=args.trusted_pickle)
        write_json(report, args.output)
        return 0
    if args.command == "build-source-gallery":
        from driversi.data.scenes.source_gallery import build_source_gallery

        report = build_source_gallery(args.config, args.dataset_root, args.output_dir,
                                      trusted=args.trusted_pickle,
                                      diagnostic_record=args.diagnostic_record)
        write_json({"status": report["status"], "scenes": [
            {key: value for key, value in scene.items() if key != "assets"}
            for scene in report["scenes"]]}, None)
        return 0
    if args.command == "rollout-visual":
        from driversi.contracts.scene import Scene
        from driversi.sim.sensors.visual_rollout import rollout_visual

        report = rollout_visual(Scene.load(args.scene), args.output_dir, width=args.width,
                                height=args.height, sensor_stride=args.sensor_stride,
                                history=args.history, seed=args.seed, desired_speed=args.desired_speed,
                                allow_unverified_heading=args.allow_unverified_heading)
        write_json(report, None)
        return 0
    if args.command == "evaluate-visual":
        from driversi.evaluation.benchmarks.visual import evaluate_visual

        report = evaluate_visual(args.config, args.output_dir, real_scene=args.real_scene,
                                 allow_unverified_heading=args.allow_unverified_heading)
        write_json(report, None)
        return int(not report["passed"])
    if args.command == "audit-sensors":
        from driversi.data.sources.navsim import read_log
        from driversi.data.assets.sensors import extract_sensor_manifest

        if args.output is not None and args.output.exists():
            parser.error("--output must be a new file")
        frames = read_log(args.log_file, trusted=args.trusted_pickle, allow_single_frame=True)
        if not 0 <= args.frame_index < len(frames):
            parser.error("--frame-index must identify a source frame")
        try:
            manifest = extract_sensor_manifest(frames[args.frame_index], args.sensor_root)
        except ValueError as error:
            parser.error(str(error))
        manifest["source_file"] = str(args.log_file.resolve())
        manifest["source_frame_index"] = args.frame_index
        write_json(manifest, args.output)
        return 0
    if args.command == "probe-interaction":
        from driversi.contracts.road import RoadGraph
        from driversi.contracts.route_tasks import RouteTasks
        from driversi.contracts.scene import Scene
        from driversi.evaluation.benchmarks.interaction import probe_interaction

        report = probe_interaction(
            Scene.load(args.scene), RoadGraph.load(args.road_graph), args.output_dir,
            route_tasks=RouteTasks.load(args.route_tasks) if args.route_tasks else None,
            intervention_seconds=args.intervention_seconds, response_seconds=args.response_seconds)
        write_json(report, None)
        return 0 if report["status"] == "verified" else 1 if report["status"] == "failed" else 2
    if args.command == "build-interaction-gallery":
        from driversi.evaluation.benchmarks.gallery import build_interaction_gallery

        report = build_interaction_gallery(args.config, args.scene_dir, args.dataset_root,
                                           args.output_dir, trusted=args.trusted_pickle)
        write_json(report, None)
        if (not report["following_passed"] or report.get("probe_failures", 0)
                or report["stats"]["runtime_error"]
                or report["stats"]["contract_failure"]):
            return 1
        return 2 if report["stats"]["compile_rejected"] or report["stats"]["unresolved_task"] else 0
    if args.command == "compile-context":
        from driversi.contracts.scene import Scene
        from driversi.data.scenes.context import compile_context

        if args.output.exists():
            parser.error("--output must be a new file")
        context = compile_context(Scene.load(args.scene), args.log_file, args.maps_root,
                                  trusted=args.trusted_pickle, margin=args.margin)
        context.save(args.output)
        write_json({"kind": "source_scene_context", "path": str(args.output),
                    "frames": len(context.frames), "source_track_ids": len({
                        obj["id"] for frame in context.frames for obj in frame["objects"]}),
                    "object_kinds": sorted({obj["kind"] for frame in context.frames for obj in frame["objects"]}),
                    "map_features": len(context.map_features), "map_layers": context.map_layers}, None)
        return 0
    if args.command == "audit-horizons":
        from driversi.data.sources.navsim import audit_horizons, read_log

        if args.output is not None and args.output.exists():
            parser.error("--output must be a new file; existing files are not overwritten")
        try:
            report = audit_horizons(
                read_log(args.log_file, trusted=args.trusted_pickle, allow_single_frame=True),
                start=args.start_frame, durations=args.durations,
            )
        except ValueError as error:
            parser.error(str(error))
        report["source_file"] = str(args.log_file.resolve())
        if args.output is None:
            write_json(report, None)
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
        return 0
    if args.command == "evaluate-scenes":
        from driversi.evaluation.benchmarks.real_scenes import evaluate_scenes

        result = evaluate_scenes(args.selection, args.scene_dir, args.maps_root, args.output_dir)
        write_json(result, None)
        return (1 if result["runtime_errors"] or result["contract_failures"] else
                2 if result["unresolved_tasks"] or result["unexercised"] else 0)
    if args.command == "export-replay":
        from driversi.contracts.context import SceneContext
        from driversi.contracts.road import RoadGraph
        from driversi.contracts.scene import Scene
        from driversi.rendering.bev.replay import export_replay

        write_json(export_replay(Scene.load(args.scene), args.rollout, args.output,
                                 road_graph=RoadGraph.load(args.road_graph) if args.road_graph else None,
                                 scene_context=SceneContext.load(args.context) if args.context else None,
                                 compare_path=args.compare), None)
        return 0
    if args.command == "inspect-routes":
        from driversi.agents.traffic_control.routing import inspect_routes
        from driversi.contracts.road import RoadGraph
        from driversi.contracts.route_tasks import RouteTasks
        from driversi.contracts.scene import Scene

        report = inspect_routes(Scene.load(args.scene), RoadGraph.load(args.road_graph),
                                RouteTasks.load(args.route_tasks) if args.route_tasks else None)
        write_json(report, args.output)
        return 0 if report["all_resolved"] else 2
    if args.command == "evaluate-following":
        from driversi.evaluation.benchmarks.following import evaluate_following

        report = evaluate_following(args.config, args.output_dir)
        write_json(report, None)
        return int(not report["passed"])
    if args.command == "compile-lanes":
        from driversi.contracts.scene import Scene
        from driversi.data.road_network.lane_graph import compile_lane_graph

        graph = compile_lane_graph(Scene.load(args.scene), args.maps_root, margin=args.margin)
        graph.save(args.output)
        write_json({"nodes": len(graph.nodes), "unloaded_edges": sum(
            child not in graph.nodes for node in graph.nodes.values() for child in node["outgoing"]
        ), "output": str(args.output)}, None)
        return 0
    if args.command == "compile-catalog":
        from driversi.data.scenes.library import compile_catalog

        report = compile_catalog(
            args.catalog, args.output_dir, dataset_root=args.dataset_root, limit=args.limit,
            offset=args.offset, map_name=args.map_name, min_controlled=args.min_controlled,
            trusted=args.trusted_pickle, workers=args.workers, maps_root=args.maps_root,
        )
        write_json(report, None)
        return int(report["failed"] > 0)
    if args.command == "inventory-nuplan":
        from driversi.data.sources.nuplan_catalog import inventory
        write_json(inventory(args.dataset_root), args.output)
        return 0
    if args.command == "index-nuplan":
        from driversi.data.sources.nuplan_catalog import build_catalog
        write_json(build_catalog(args.dataset_root, args.output, partitions=args.partitions,
            max_logs=args.max_logs, windows_per_log=args.windows_per_log, duration=args.duration,
            history_seconds=args.history_seconds, max_actors=args.max_actors,
            split_policy=args.split_policy, protected_logs=args.protected_logs), None)
        return 0
    if args.command == "index-navsim":
        from driversi.data.scenes.catalog import build_catalog

        write_json(build_catalog(
            args.dataset_root, args.output, partitions=args.partitions, max_logs=args.max_logs,
            windows_per_log=args.windows_per_log, duration=args.duration,
            duration_mode=args.duration_mode, min_duration=args.min_duration,
            max_actors=args.max_actors, trusted=args.trusted_pickle,
        ), None)
        return 0
    if args.command == "list-scenes":
        from driversi.data.scenes.catalog import list_scenes

        write_json({"scenes": list_scenes(
            args.catalog, limit=args.limit, offset=args.offset, status=args.status,
            map_name=args.map_name, min_controlled=args.min_controlled,
        )}, args.output)
        return 0
    if args.command == "inventory-navsim":
        from driversi.data.scenes.catalog import inventory

        write_json(inventory(args.dataset_root), args.output)
        return 0
    if args.command == "rollout":
        from driversi.agents.traffic_control.routing import RouteResolutionError
        from driversi.contracts.road import RoadGraph
        from driversi.contracts.route_tasks import RouteTasks
        from driversi.contracts.scene import Scene
        from driversi.sim.execution.rollout import rollout

        try:
            report = rollout(Scene.load(args.scene), args.output, policy=args.policy,
                             road_graph=RoadGraph.load(args.road_graph) if args.road_graph else None,
                             route_tasks=RouteTasks.load(args.route_tasks) if args.route_tasks else None)
        except RouteResolutionError as error:
            write_json(error.report, None)
            return 2
        write_json(report, None)
        return 0
    if args.command == "audit-navsim":
        from driversi.data.sources.navsim import audit_logs

        if args.max_files < 1:
            parser.error("--max-files must be positive")
        paths = sorted(args.log_dir.glob("*.pkl"))[:args.max_files]
        if not paths:
            parser.error("No NAVSIM .pkl files found")
        report = audit_logs(paths, trusted=args.trusted_pickle)
        write_json(report, args.output)
        return int(bool(report["errors"]))
    if args.command == "compile-navsim":
        from driversi.data.road_network.maps import attach_nuplan_map
        from driversi.data.sources.navsim import HorizonPlanError, compile_frames, load_log

        if args.output.exists():
            parser.error("--output must be a new file")
        try:
            scene = compile_frames(
                load_log(args.log_file, trusted=args.trusted_pickle), start=args.start_frame,
                duration=args.duration, split=args.split, max_actors=args.max_actors,
                duration_mode=args.duration_mode, min_duration=args.min_duration,
            )
        except HorizonPlanError as error:
            write_json({"status": "rejected", "horizon_plan": error.plan}, None)
            return 2
        attach_nuplan_map(scene, args.maps_root)
        scene.save(args.output)
        write_json({"scene": str(args.output), "actors": len(scene.actor_ids),
                    "controlled": int(scene.controlled.sum()), "steps": len(scene.times) - 1,
                    "horizon_plan": scene.metadata["horizon_plan"]}, None)
        return 0
    write_json({"version": __version__, "implemented": ["package-cli", "navsim-audit", "scene-compiler",
                "navsim-inventory", "source-scene-catalog", "catalog-scene-library",
                "nuplan-readonly-db-input", "nuplan-provenance-catalog",
                "pufferdrive-cpu-adapter", "joint-control", "snapshot-fork", "engineering-rollout",
                "metric-current-observation", "initial-position-lane-graph", "bounded-unambiguous-routes",
                "stateless-idm-route-controller", "paired-following-acceptance",
                "explicit-initial-route-tasks", "lane-match-diagnostics", "frozen-real-scene-audit",
                "standalone-replay-inspector", "navsim-horizon-audit", "source-scene-context",
                "per-window-horizon-planning", "real-scene-following-probe", "navsim-sensor-manifest",
                "primitive-camera-rendering", "visual-diagnostic-session", "visual-snapshot-fork",
                "rgb-interaction-probe", "mtgs-world-rendering", "live-session-inspector",
                "identified-learning-api", "shared-policy-ppo", "configured-agent-training",
                "process-sharded-sampling", "bounded-scene-cache", "sampling-capacity-probe",
                "frozen-policy-audit", "action-conditioned-sequence-export",
                "frozen-history-opponent-sampling", "learner-only-mixed-ppo",
                "fixed-opponent-matchup-evaluation"],
                "pending": ["traffic-signals-and-right-of-way", "merging-and-junction-controllers",
                            "adaptive-opponent-league-pfsp", "independent-driving-evaluation",
                            "actor-lifecycle", "long-episodes",
                            "photorealistic-quality-and-realtime-throughput", "learned-visual-self-play", "world-model-training", "rsi"]}, args.output)
    return 0
