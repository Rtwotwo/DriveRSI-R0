"""Versioned primary history-perturbation/TTS policy and execution placement."""
from pathlib import Path
from driversi.configuration import read_mapping
from driversi.agents.backends.native import digest
from driversi.rsi.experiments.continuation import frozen_json


def symmetric_grid():
    # Start along the lane, then explore both sides symmetrically. These are
    # proposals only; actual physics and unchanged admission decide validity.
    offsets = [(-2.,0.), (2.,0.), (-4.,0.), (4.,0.),
               (0.,-.5), (0.,.5), (-2.,-.5), (-2.,.5),
               (2.,-.5), (2.,.5), (0.,-1.), (0.,1.),
               (-2.,-1.), (-2.,1.), (2.,-1.), (2.,1.)]
    return [dict(longitudinal_m=x,lateral_m=y,speed_delta_mps=speed)
            for speed in (0.,-1.,1.,2.) for x,y in offsets]


def balanced_grid():
    """Cover all speed signs before a bounded budget truncates the grid."""
    offsets=symmetric_grid()[:16]
    return [dict(row,speed_delta_mps=speed)
            for row in offsets for speed in (0.,1.,2.,-1.)]


def spatial_grid():
    offsets=symmetric_grid()[:16]
    offsets.sort(key=lambda r:(r['lateral_m']==0,abs(r['lateral_m']),abs(r['longitudinal_m'])))
    return [dict(row,speed_delta_mps=speed) for row in offsets for speed in (0.,1.,2.,-1.)]


def apply(root, config, folder):
    path = Path(root)/'primary-generation-policy.json'
    if not path.exists():return config
    policy=read_mapping(path)
    kinds={'symmetric_history_native_tts_resident_v1':symmetric_grid,
           'budget_balanced_history_native_tts_resident_v2':balanced_grid,
           'spatial_balanced_history_native_tts_resident_v3':spatial_grid,
           'adaptive_history_native_tts_resident_v1':spatial_grid}
    if policy.get('kind') not in kinds:
        raise ValueError('Unknown primary generation policy')
    attempts=config['trajectory_family']['attempts_per_anchor']
    minimum=1 if config['trajectory_family'].get('cohort_mode')=='global_50_min1_v1' else 10
    if type(attempts) is not int or not minimum<=attempts<=64:raise ValueError('Finite existing attempt budget required')
    for name in ('stop_after_consecutive_duplicates','stop_after_visual_failures','stop_after_execution_failures'):
        if config['trajectory_family'].get(name) is not None:
            config['trajectory_family'][name]=min(config['trajectory_family'][name],attempts)
    frozen_json(Path(folder)/'primary-generation-binding.json',dict(policy=policy,sha256=digest(path)))
    config['generation_skill']='perturbed_history_tts'
    config['trajectory_family'].update(method='history_endpoint_grid_tts_v1',
        proposal_grid=kinds[policy['kind']]() if policy['kind'] in ('spatial_balanced_history_native_tts_resident_v3','adaptive_history_native_tts_resident_v1') else kinds[policy['kind']]()[:attempts])
    config['trajectory_family'].pop('proposal_seeds',None)
    config['physical_prefetch_workers']=policy['physical_prefetch_workers']
    config['resident_postprocessing']=policy['resident_postprocessing']
    if policy.get('scene_diversity'):
        config['trajectory_family']['scene_diversity']=policy['scene_diversity']
        config['trajectory_family']['stop_on_saturated_tail']=True
    if policy['kind'] in ('spatial_balanced_history_native_tts_resident_v3','adaptive_history_native_tts_resident_v1'):
        config['trajectory_family']['environment_speed_context']={
            'version':'source_history_speed_envelope_v1',
            'logs_root':policy['training_logs_root']}
    if policy['kind']=='adaptive_history_native_tts_resident_v1':
        config['visual_quality_profile']='experimental_numeric25_adaptive_v1'
        config['trajectory_family']['minimum_ade_m']=.20
        from . import continual_curriculum as cc
        from .adaptive_generation import context
        state=read_mapping(cc.state_path(root))
        case=next(c for c in state['cases'] if c['id']==state['active'])
        config['trajectory_family']['adaptive_context']=dict(policy['adaptive_context'],
            database=config['train_catalog'],**context(root,case))
        config['trajectory_family']['seed_start']+=sum(started_attempts(root,case['id']).values())
    return config



def started_attempts(root, case_id):
    charged = {}
    for plan in (Path(root)/'rounds').glob(case_id+'-*/waves/wave-0001/trajectory-family-plan.json'):
        assignments={x['id']:x for x in read_mapping(plan)['assignments']}
        for directory in (plan.parent/'candidates').glob('*'):
            if (directory/'generation').exists():
                token=assignments[directory.name]['token']
                charged[token]=charged.get(token,0)+1
    return charged

def resume_after_authorized_stop(root, stop_report):
    """One explicit policy migration; terminated work remains charged evidence.

    Caller holds the curriculum lock. This cannot be used to retry unknown work.
    """
    from . import continual_curriculum as cc
    root=Path(root); proof=read_mapping(stop_report)
    marker=root/'primary-generation-migration.json'
    if marker.exists():return read_mapping(marker)
    if proof.get('settled') is not True or proof.get('remaining'):
        raise ValueError('Every owned process must have settled')
    for process in proof['owned']:
        # A live PID is conservatively unresolved, even if the OS reused it.
        if (Path('/proc')/str(process['pid'])/'cmdline').exists():
            if (Path('/proc')/str(process['pid'])/'cmdline').read_bytes():
                raise ValueError('Stopped process identity still live')
    state=read_mapping(cc.state_path(root))
    case=next(x for x in state['cases'] if x['id']==state['active'])
    if case['status']!='dispatching':raise ValueError('Only interrupted dispatch migration supported')
    previous=Path(case['current_round'])
    charged={}; evidence={str(stop_report):digest(stop_report)}
    for folder in (root/'rounds').glob(case['id']+'-*'):
        plan=folder/'waves/wave-0001/trajectory-family-plan.json'
        if not plan.exists():continue
        mapping=read_mapping(plan);evidence[str(plan)]=digest(plan)
        assignments={x['id']:x for x in mapping['assignments']}
        for directory in (plan.parent/'candidates').glob('*'):
            if not (directory/'generation').exists():continue
            row=assignments[directory.name]
            charged[row['token']]=charged.get(row['token'],0)+1
    cap=state['policy']['attempts_per_source']*state['policy']['max_rounds']
    allocation=state['policy']['attempts_per_source']
    if any(n+allocation>cap for n in charged.values()):
        raise ValueError('No remaining declared multi-round problem budget')
    frozen_json(previous/'interruption.json',dict(reason=proof['reason'],
        evidence=evidence,charged_attempts=charged,unknown_results_reused=False))
    case.update(status='needs_revision',asset_revision=case.get('asset_revision',0)+1)
    cc.save(root,state)
    cc.prepare(root)
    state=read_mapping(cc.state_path(root));case=next(x for x in state['cases'] if x['id']==state['active'])
    result=dict(previous_round=str(previous),current_round=case['current_round'],
        charged_attempts=charged,new_allocation_per_source=allocation,
        cumulative_cap_per_source=cap,training_round_advanced=False,
        reason='user_authorized_primary_skill_correction',evidence=evidence)
    frozen_json(Path(case['current_round'])/'cumulative-generation-budget.json',result)
    frozen_json(marker,result)
    return result
