"""Unmodified upstream nuPlan metrics on an explicitly declared execution history.

Custom long logs are not Val14 or NAVSIM PDMS. Any failed metric fails the score.
"""

import json
from pathlib import Path

import numpy as np


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def score_history(history, scenario, output, repository, *, world_protocol, planner_name="DriveRL", execution_policy=None):
    import pandas as pd
    import yaml
    from hydra.utils import instantiate
    from nuplan.planning.metrics.metric_dataframe import MetricStatisticsDataFrame
    from nuplan.planning.script.builders.metric_builder import build_metrics_engines
    from omegaconf import OmegaConf

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    config_root = Path(repository) / "nuplan-devkit/nuplan/planning/script/config"
    root = config_root / "common/simulation_metric"
    protocol = yaml.safe_load((root / "simulation_closed_loop_reactive_agents.yaml").read_text())
    metrics = {}
    for group in protocol["defaults"]:
        for level, names in group.items():
            metrics[level] = {}
            for name in names:
                metrics[level].update(yaml.safe_load((root / level / (name + ".yaml")).read_text()))
    cfg = OmegaConf.create(
        dict(
            output_dir=str(output),
            metric_dir="raw",
            selected_simulation_metrics=None,
            simulation_metric=metrics,
        )
    )
    engine = build_metrics_engines(cfg, [scenario])[scenario.scenario_type]
    results = engine.compute_metric_results(history, scenario)
    raw = {name: [x.serialize() for x in values] for name, values in results.items()}
    (output / "metric-results.json").write_text(
        json.dumps(json_safe(raw), indent=2, allow_nan=False)
    )
    frames = {}
    scores = {}
    for values in results.values():
        if not values:
            raise ValueError("Official metric returned no results")
        for item in values:
            name = item.name
            if name in frames:
                raise ValueError("Duplicate official statistic: " + name)
            columns = item.serialize_dataframe()
            columns.update(
                scenario_name=scenario.scenario_name,
                scenario_type=scenario.scenario_type,
                log_name=scenario.log_name,
                planner_name=planner_name,
                metric_statistics_name=name,
                metric_computator=item.metric_computator,
            )
            frame = pd.DataFrame(columns)
            frame.to_parquet(output / (name + ".parquet"))
            frames[name] = MetricStatisticsDataFrame(name, frame)
            scores[name] = item.metric_score
    agg_cfg = yaml.safe_load(
        (
            config_root
            / "simulation/metric_aggregator/closed_loop_reactive_agents_weighted_average.yaml"
        ).read_text()
    )["closed_loop_reactive_agents_weighted_average"]
    required = set(agg_cfg["multiple_metrics"]) | (set(agg_cfg["metric_weights"]) - {"default"})
    if any(scores.get(k) is None or not np.isfinite(scores[k]) for k in required):
        raise ValueError("Incomplete official score components")
    agg_cfg["file_name"] = "official_weighted_average"
    aggregator = instantiate(OmegaConf.create(agg_cfg), aggregator_save_path=output)
    aggregator(frames)
    result = dict(
        status="scored",
        protocol="nuplan_closed_loop_reactive_metric_suite_on_custom_long_log",
        world_protocol=world_protocol,
        planner_name=planner_name,
        execution_policy=execution_policy,
        score=float(aggregator.final_metric_score),
        metric_scores=json_safe(scores),
        samples=len(history.data),
        first_timestamp_us=history.data[0].ego_state.time_us,
        last_timestamp_us=history.data[-1].ego_state.time_us,
        official_benchmark=False,
        navsim_pdms=None,
        trainable=False,
        raw_nonfinite_timeseries="serialized_as_null; metric scores required finite",
        aggregation=str(aggregator.parquet_file),
    )
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    return result


def score_record(scene, rows, output, repository, maps_root, *, planner_name="DriveRL", execution_policy=None):
    from nuplan.common.actor_state.state_representation import TimePoint
    from nuplan.common.maps.maps_datatypes import TrafficLightStatusData, TrafficLightStatusType
    from nuplan.planning.simulation.history.simulation_history import (
        SimulationHistory,
        SimulationHistorySample,
    )
    from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import (
        SimulationIteration,
    )

    from driversi_agents.driverl.closed_loop import native_scenario, unpack_ego, unpack_objects

    if len(rows) < 3:
        raise ValueError("Incomplete execution history")
    stamps = np.asarray([v["ego"][-1] for v in rows], dtype=np.int64)
    if np.any(np.diff(stamps) != 100000):
        raise ValueError("Expected contiguous 10Hz actual history")
    duration = (len(rows) + 0.5) * 0.1
    scenario = native_scenario(scene, maps_root, duration=duration)
    if scenario.get_number_of_iterations() != len(rows) + 1:
        raise ValueError("Expert/evaluation windows do not align")
    source_stamps = np.array([scenario.get_time_point(i).time_us for i in range(len(rows))])
    jitter = int(np.max(np.abs(source_stamps - stamps)))
    if jitter > 5000:
        raise ValueError("Source clock drifts more than 5ms from fixed evaluation grid")
    history = SimulationHistory(scenario.map_api, scenario.get_mission_goal())
    for i, row in enumerate(rows):
        e = unpack_ego(row["ego"])
        lights = [
            TrafficLightStatusData(
                TrafficLightStatusType[x["status"]], x["lane_connector_id"], x["timestamp"]
            )
            for x in row["lights"]
        ]
        history.add_sample(
            SimulationHistorySample(
                SimulationIteration(TimePoint(e.time_us), i),
                e,
                None,
                unpack_objects(row["objects"], e.time_us),
                lights,
            )
        )
    result = score_history(
        history,
        scenario,
        output,
        repository,
        world_protocol="DriveRSI_reactive_full_world; platform_terminal_holds_retained",
        planner_name=planner_name, execution_policy=execution_policy,
    )
    result["source_clock_max_error_us"] = jitter
    (Path(output) / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    return result
