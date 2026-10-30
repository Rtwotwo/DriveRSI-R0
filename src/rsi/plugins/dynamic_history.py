"""Registered bounded dynamic-history production and native replay comparison."""
from pathlib import Path

from driversi.agents.backends.native import digest
from driversi.configuration import read_mapping
from driversi.rsi.experiments.dynamic_history_campaign import input_files, validate_plan
from driversi.rsi.workflow.results import result


def prepare(config, base, registry):
    if set(config) != {'plan'}:
        raise ValueError('One explicit immutable dynamic-history plan required')
    path = (Path(base)/config['plan']).resolve(strict=True)
    plan = validate_plan(read_mapping(path))
    # Native full evaluation includes parent/candidate; compiler retains its
    # stricter phase budgets and exact-resume verification independently.
    cap = 7*12146*160 + 12*80
    return dict(plan=str(path),sha256=digest(path),files=input_files(plan)),cap


def execute(config, inputs, output, registry):
    from driversi.rsi.experiments.dynamic_history_campaign import _verify_files, run
    if digest(config['plan']) != config['sha256']:
        raise ValueError('Dynamic-history plan changed after registration')
    _verify_files(config['files'])
    from driversi.visualization.dynamic_history import publishing
    with publishing(Path(output)/'campaign'):
        report=run(config['plan'],Path(output)/'campaign')
    status='completed' if report['status'] in ('completed','target_still_zero','insufficient_admitted_data') else 'failed'
    return result(output,report,status=status,reason=report.get('reason'),
                  steps=report.get('sim_agent_steps',0),
                  artifacts=dict(campaign=str(Path(output)/'campaign/report.json')))


def register(registry, Provider):
    registry.register(Provider('dynamic_history_replay_campaign','1',(),
        'dynamic_history_replay_comparison',prepare,execute,('report','campaign')))
