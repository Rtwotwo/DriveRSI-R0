"""Typed frozen-policy skill pilot services in the existing DriveHarness engine."""
from pathlib import Path
from driversi.agents.backends.native import digest
from driversi.rsi.contracts import _keys
from driversi.rsi.workflow.results import result
from driversi.rsi.experiments import frozen_skill_probe as pilot


def prepare(config, base, registry):
    _keys(config, ('plan',), 'frozen skill provider')
    path=(base/config['plan']).resolve(strict=True)
    plan = pilot.validate_plan(path)
    return dict(plan=str(path), sha256=digest(path),
                source_groups=sorted({m['source_log'] for m in plan['members']})), 0


def prepare_score(config, base, registry):
    prepared, _ = prepare(config, base, registry)
    return prepared, pilot.read(prepared['plan'])['scoring_agent_steps']


def check(config):
    if digest(config['plan']) != config['sha256']: raise ValueError('Pilot declaration changed')
    pilot.validate_plan(config['plan'])


def data(config, inputs, output, registry):
    check(config)
    report=pilot.prepare_data(config['plan'], output)
    return result(output, report)


def infer(config, inputs, output, registry):
    check(config)
    from driversi_agents.augustus.skill_inference import run
    report=run(inputs['observations']['artifacts']['report'], output)
    return result(output, report, artifacts={'predictions':str(output/'predictions.json'),
                                           'witness':str(output/'witness.json')})


def score(config, inputs, output, registry):
    check(config)
    report=pilot.score(inputs['observations']['artifacts']['report'],
                       inputs['predictions']['artifacts']['predictions'],
                       inputs['predictions']['artifacts']['report'], output)
    return result(output, report, steps=report['sim_agent_steps'], artifacts={'scores':report['scores']})


def analyze(config, inputs, output, registry):
    check(config)
    report=pilot.analyze(inputs['observations']['artifacts']['report'], inputs['scores']['artifacts']['report'], output)
    return result(output, report, artifacts={'experience':str(output/'skill-experience.json'),
                                           'analysis':str(output/'report.md')})


def register(registry, Provider):
    registry.register(Provider('frozen_skill_observations','1',(),'frozen_skill_observations',prepare,data,('report',)))
    registry.register(Provider('native_model_frozen_skill_inference','1',
        (('observations','frozen_skill_observations'),),'frozen_skill_predictions',prepare,infer,('report','predictions','witness')))
    registry.register(Provider('frozen_skill_native_scoring','1',
        (('observations','frozen_skill_observations'),('predictions','frozen_skill_predictions')),
        'frozen_skill_scores',prepare_score,score,('report','scores')))
    registry.register(Provider('frozen_skill_paired_analysis','1',
        (('observations','frozen_skill_observations'),('scores','frozen_skill_scores')),
        'frozen_skill_analysis',prepare,analyze,('report','experience','analysis')))
