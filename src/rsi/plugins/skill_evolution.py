"""Typed DriveHarness stages: no conversational intervention between episodes."""
from driversi.agents.backends.native import digest
from driversi.rsi.contracts import _keys
from driversi.rsi.workflow.results import result
from driversi.rsi.experiments import frozen_skill_probe as pilot
from driversi.rsi.experiments import skill_evolution as evolution


def prepare(config, base, registry):
    _keys(config,('plan','episode'),'skill evolution')
    path=(base/config['plan']).resolve(strict=True);p=pilot.validate_plan(path)
    if type(config['episode']) is not int or config['episode'] not in range(p['episodes']):
        raise ValueError('Undeclared episode')
    return dict(plan=str(path),sha256=digest(path),episode=config['episode'],
                source_groups=sorted({m['source_log'] for m in p['members']})),0


def prepare_score(config, base, registry):
    value,_=prepare(config,base,registry)
    return value,pilot.read(value['plan'])['scoring_agent_steps']


def check(config):
    if digest(config['plan']) != config['sha256']: raise ValueError('Evolution plan changed')
    pilot.validate_plan(config['plan'])


def data(config, inputs, output, registry):
    check(config)
    return result(output,implementation(config).prepare(config['plan'],output))


def implementation(config):
    if pilot.read(config['plan']).get('skill_repair_protocol')=='targeted_ablation_v1':
        from driversi.rsi.experiments import skill_repair
        return skill_repair
    return evolution


def propose(config, inputs, output, registry):
    check(config)
    previous=inputs.get('previous',{}).get('artifacts',{}).get('report')
    report=implementation(config).proposal(inputs['observations']['artifacts']['report'],previous,config['episode'],output)
    return result(output,report,artifacts={'prepared':report['prepared']})


def infer(config, inputs, output, registry):
    check(config)
    from driversi_agents.augustus.skill_inference import run
    p=pilot.read(inputs['proposal']['artifacts']['report'])
    if digest(p['prepared']) != p['prepared_sha256']: raise ValueError('Unsealed proposal')
    report=run(p['prepared'],output)
    return result(output,report,artifacts={'predictions':str(output/'predictions.json')})


def score(config, inputs, output, registry):
    check(config)
    report=pilot.score(inputs['proposal']['artifacts']['prepared'],inputs['predictions']['artifacts']['predictions'],
                       inputs['predictions']['artifacts']['report'],output)
    return result(output,report,steps=report['sim_agent_steps'],artifacts={'scores':report['scores']})


def decide(config, inputs, output, registry):
    check(config)
    report=implementation(config).decide(inputs['proposal']['artifacts']['report'],inputs['scores']['artifacts']['report'],output)
    # A rejected skill is a completed decision, so the next episode still runs.
    return result(output,report,artifacts={'memory':str(output/'memory-snapshot.json')})


def register(registry, Provider):
    registry.register(Provider('native_model_hard_case_search','1',(),'hard_case_search',
        prepare_hard_case,run_hard_case,('report','memory')))
    registry.register(Provider('skill_evolution_observations','1',(),'evolution_observations',prepare,data,('report',)))
    registry.register(Provider('skill_evolution_propose_initial','1',(('observations','evolution_observations'),),
        'evolution_proposal',prepare,propose,('report','prepared')))
    registry.register(Provider('skill_evolution_propose_next','1',(('observations','evolution_observations'),('previous','evolution_decision')),
        'evolution_proposal',prepare,propose,('report','prepared')))
    registry.register(Provider('native_model_skill_evolution','1',(('proposal','evolution_proposal'),),
        'evolution_predictions',prepare,infer,('report','predictions')))
    registry.register(Provider('skill_evolution_scoring','1',(('proposal','evolution_proposal'),('predictions','evolution_predictions')),
        'evolution_scores',prepare_score,score,('report','scores')))
    registry.register(Provider('skill_evolution_decision','1',(('proposal','evolution_proposal'),('scores','evolution_scores')),
        'evolution_decision',prepare,decide,('report','memory')))


def prepare_hard_case(config,base,registry):
    _keys(config,('plan',),'hard case search')
    path=(base/config['plan']).resolve(strict=True);plan=pilot.validate_plan(path)
    if plan.get('protocol')!='grounded_hard_case_v1':raise ValueError('Wrong hard case protocol')
    return dict(plan=str(path),sha256=digest(path),source_groups=sorted({m['source_log'] for m in plan['members']})),plan['max_sim_agent_steps']


def run_hard_case(config,inputs,output,registry):
    from driversi.rsi.experiments.hard_case_repair import run
    if digest(config['plan'])!=config['sha256']:raise ValueError('Hard case declaration changed')
    report=run(config['plan'],output)
    return result(output,report,steps=report['sim_agent_steps'],artifacts={'memory':str(output/'memory-snapshot.json')})
