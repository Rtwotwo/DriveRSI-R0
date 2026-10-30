"""Reconstruct current observations from executed correction branches."""
import json
import shutil
from pathlib import Path

import numpy as np

from driversi.contracts.scene import Scene
from driversi.data.assets.mtgs_assets import file_sha256
from driversi.runtime.budget import StepBudget, write_json
from driversi.rsi.contracts import _keys
from driversi.rsi.correction.recipes import check_suffix
from driversi.sim.backends.pufferdrive import PufferDriveWorld
from driversi.sim.persistence.recording import RolloutRecord
from driversi.training.sampling.environment import features, source_group

PURPOSE = "structured_imitation_diagnostic"


def result(output, summary, *, steps=0, status="completed", reason=None, artifacts=None):
    return {"status": status, "reason": reason, "sim_agent_steps": steps,
            "artifacts": artifacts or {"report": str(output / "report.json")},
            "summary": summary, "admission": {"trainable": False, "teacher_qualified": False,
            "purpose": PURPOSE, "promoted": False}}


def prepare_dataset(config, base, registry):
    _keys(config, ("purpose", "max_sim_agent_steps"), "correction dataset config")
    if config["purpose"] != PURPOSE or type(config["max_sim_agent_steps"]) is not int or config["max_sim_agent_steps"] < 1:
        raise ValueError("Explicit diagnostic purpose and replay budget required")
    return config, config["max_sim_agent_steps"]


def run_dataset(config, inputs, output, registry):
    repair, gate = inputs["repair"], inputs["admission"]
    identity = repair["summary"]["artifact_reference"]["identity"]
    if gate["summary"]["input_references"].get("evidence") != identity:
        raise ValueError("Admission belongs to another correction branch")
    if not gate["summary"].get("engineering_pass"):
        raise ValueError("Missing engineering admission")
    evidence = json.loads(Path(repair["artifacts"]["report"]).read_text())
    spec, selected = evidence["spec"], evidence["selected_recipe"]
    scene = Scene.load(Path(repair["artifacts"]["scene"]))
    from driversi.rsi.execution.registry import ExperimentRegistry
    if source_group(scene) in ExperimentRegistry(registry).protected_groups():
        raise ValueError("Correction source became protected")
    source = Path(repair["artifacts"]["rollout:" + selected])
    output.mkdir(parents=True, exist_ok=False)
    budget = StepBudget(output / "budget.json", config["max_sim_agent_steps"], "dataset_replay")
    anchor, horizon = spec["case"]["anchor_tick"], spec["case"]["future_steps"]
    observations, labels, ticks = [], [], []
    with np.load(source, allow_pickle=False) as data, PufferDriveWorld(scene) as world:
        meta = json.loads(str(data["metadata"]))
        snapshot = world.snapshot()
        if meta["backend_build"] != snapshot.build_id or meta["scene_key"] != snapshot.scene_key:
            raise ValueError("Replay requires the original scene and native build")
        if tuple(data["controlled_ids"]) != world.actor_ids or len(data["actions"]) != anchor + horizon:
            raise ValueError("Correction actor order or full horizon differs")
        slot = world.actor_ids.index(spec["case"]["actor"])
        record = RolloutRecord.start(world)
        for tick in range(anchor + horizon + 1):
            for name, actual in (("states", world.state()), ("motion", world.motion()), ("presence", world.presence())):
                if not np.array_equal(actual, data[name][tick]):
                    raise ValueError("Correction replay differs at tick " + str(tick) + ": " + name)
            if tick == anchor + horizon:
                break
            observation = world.observe()
            if tick >= anchor:
                if not observation.alive[slot] or not data["valid"][tick, slot]:
                    raise ValueError("Invalid correction owner; partial suffix is not admitted")
                observations.append(features(observation, world.max_steps)[slot].copy())
                labels.append(int(data["actions"][tick, slot]))
                ticks.append(tick)
            actions = dict(zip(world.actor_ids, map(int, data["actions"][tick])))
            budget.before_step(int(observation.alive.sum()))
            record.append(world, actions, world.step(actions), [])
        checks = check_suffix(scene, record, anchor, spec["case"]["actor"], horizon, spec["verifier"])
        if not checks["engineering_pass"]:
            raise ValueError("Independent replay rejected correction suffix")
    np.savez_compressed(output / "samples.npz", observations=np.asarray(observations),
                        actions=np.asarray(labels, dtype=np.int64), ticks=np.asarray(ticks))
    shutil.copyfile(repair["artifacts"]["scene"], output / "scene.npz")
    manifest = {"schema_version": 1, "purpose": PURPOSE, "allowed_uses": [PURPOSE],
        "trainable": False, "teacher_qualified": False, "samples": len(labels),
        "source_groups": [source_group(scene)], "recipe": selected,
        "repair_agent_steps": repair["sim_agent_steps"],
        "actor": spec["case"]["actor"], "anchor_tick": anchor, "future_steps": horizon,
        "observation_timing": "before_executed_action", "label_kind": "executed_discrete_action",
        "qualified_trajectory_labels": False, "repair_identity": identity,
        "rollout_sha256": file_sha256(source), "samples_sha256": file_sha256(output / "samples.npz"),
        "checks": checks, "failure_type": spec["case"]["failure_type"]}
    write_json(output / "report.json", manifest)
    return result(output, manifest, steps=budget.state["admitted_agent_steps"], artifacts={
        "report": str(output / "report.json"), "samples": str(output / "samples.npz"),
        "scene": str(output / "scene.npz")})
