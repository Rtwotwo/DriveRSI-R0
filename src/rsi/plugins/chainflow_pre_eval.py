"""Reject incomplete sample consumption or replay before full benchmark cost."""

import math
import shutil

from driversi.configuration import read_mapping
from driversi.runtime.budget import write_json


def verify_pair(control, treatment, replay, expected):
    for value in (control, treatment, replay):
        if (
            value.get("model") != "chainflow_stage1"
            or value.get("status") != "completed"
            or value.get("total_updates", 0) <= 0
        ):
            raise ValueError("Completed native ChainFlow optimizer receipts required")
    if control["total_updates"] != treatment["total_updates"]:
        raise ValueError("Pair update budgets differ")
    for key in ("context", "resume_from", "state_fingerprint", "total_updates"):
        if treatment.get(key) is None or treatment.get(key) != replay.get(key):
            raise ValueError("Cross-process recovery mismatch: " + key)
    if treatment["worker_pid"] == replay["worker_pid"]:
        raise ValueError("Recovery must use a separate process")
    trace = treatment.get("sample_trace", [])
    if [x["optimizer_step"] for x in trace] != list(range(1, treatment["total_updates"] + 1)):
        raise ValueError("Incomplete optimizer trace")
    exposed = set()
    for row in trace:
        rates = row.get("learning_rates_before", [])
        if not rates or any(not math.isfinite(x) or x < 0 for x in rates):
            raise ValueError("Actual optimizer learning rate evidence missing")
        if any(x > 0 for x in rates):
            exposed.update(row["sample_ids"])
    missing = set(expected) - exposed
    if not expected or missing:
        raise ValueError("Admitted synthetic samples lack positive-LR updates")
    return dict(
        kind="chainflow_pair_pre_evaluation_gate",
        status="passed",
        resume_exact=True,
        positive_lr_exposed=len(set(expected)),
        total_updates=treatment["total_updates"],
        full_navtest_started=False,
    )


def prepare(config, base, registry):
    if (set(config) not in ({'side'}, {'side','target_token'})
            or config["side"] not in ("control", "treatment")):
        raise ValueError("Explicit paired side required")
    if 'target_token' in config and (not isinstance(config['target_token'], str) or not config['target_token']):
        raise ValueError('Explicit private failure target required')
    return config, 80 if 'target_token' in config else 0


def evaluate_target(token, report, plan, output):
    """Run the exact native scorer on the failed query before the full benchmark."""
    import json
    import os
    import subprocess
    from pathlib import Path
    from driversi.agents.backends.native import project_root, digest, path_at
    root = project_root()
    checkpoint = str(Path(report['checkpoint']).resolve(strict=True))
    if report['checkpoint_files'].get(checkpoint) != digest(checkpoint):
        raise ValueError('Target test checkpoint differs from trained weights')
    model = plan['model_config']; dataset = read_mapping(plan['data']['manifest'])
    config = dict(index=dataset['benchmark']['test_index'], target_token=token,
        checkpoint=checkpoint, backbone=str(path_at(root, model['weights']['backbone'])),
        maps_root=model['paths']['maps'], seed=42, shards=1, shard=0,
        gpu_memory_fraction=min(.20, model['settings'].get('gpu_memory_fraction', .20)),
        output=str(output/'target-evaluation'))
    write_json(output/'target-config.json', config)
    env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(model['settings']['gpu']),
             OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    with (output/'target-evaluation.log').open('wb') as log:
        subprocess.run([plan['python'],'-u','-m','driversi_agents.chainflow.benchmark',
                        'evaluate',str(output/'target-config.json')], cwd=root,env=env,
                       stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
    records=[json.loads(l) for l in (output/'target-evaluation/scores.jsonl').read_text().splitlines()]
    if len(records)!=1 or records[0]['token']!=token or not records[0]['valid']:
        raise ValueError('Single target native evaluation incomplete')
    score=records[0]['score']
    from driversi.rsi.decisions.problem_learning import target_success
    verdict=dict(target_success=target_success(score,0,strict=True),pdms=score*100,
        target_token=token,checkpoint=checkpoint,checkpoint_sha256=digest(checkpoint),
        criterion='PDMS>0',promoted=False,adaptive_evaluation=True,
        full_benchmark=False,reason='local objective; global release remains separate')
    write_json(output/'target-feedback.json',verdict)
    return verdict


def execute(config, inputs, output, registry):
    reports = {k: read_mapping(v["artifacts"]["report"]) for k, v in inputs.items()}
    plan = read_mapping(inputs["treatment"]["artifacts"]["plan"])
    dataset = read_mapping(plan["data"]["manifest"])
    expected = [
        x["sample_id"] for x in dataset["splits"]["train"] if x["kind"] == "synthetic_navtrain"
    ]
    proof = verify_pair(reports["control"], reports["treatment"], reports["replay"], expected)
    output.mkdir(parents=True, exist_ok=False)
    source = inputs[config["side"]]
    for role, name in (("report", "training-result.json"), ("plan", "training-plan.json")):
        shutil.copyfile(source["artifacts"][role], output / name)
    if 'target_token' in config:
        proof['target_evaluation']=evaluate_target(config['target_token'],reports[config['side']],
            read_mapping(source['artifacts']['plan']),output)
    write_json(output / "gate.json", proof)
    return dict(
        status="completed",
        reason=None,
        sim_agent_steps=80 if 'target_token' in config else 0,
        artifacts=dict(
            report=str(output / "training-result.json"),
            plan=str(output / "training-plan.json"),
            validation=str(output / "gate.json"),
        ),
        summary=proof,
        admission=dict(trainable=False, promoted=False),
    )


def register(registry, Provider):
    registry.register(
        Provider(
            "chainflow_pair_pre_evaluation_gate",
            "1",
            (
                ("control", "unified_training_result"),
                ("treatment", "unified_training_result"),
                ("replay", "unified_training_result"),
            ),
            "unified_training_result",
            prepare,
            execute,
            ("report", "plan", "validation"),
        )
    )
