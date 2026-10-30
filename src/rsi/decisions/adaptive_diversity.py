"""Bounded contextual diversity, separate from immutable native safety scoring."""
import numpy as np

VERSION = 'source_adaptive_diversity_v1'
PROFILE = 'experimental_numeric25_adaptive_v1'
PROPOSAL_VERSION = 'bounded_endpoint_coverage_v2'


def policy(environment, preference='conservative', revision=0, action='initial'):
    from .environment_speed import policy_from_facts
    if environment != policy_from_facts(environment['facts']):
        raise ValueError('Frozen measured source context required')
    if preference not in ('spatial', 'speed', 'mixed', 'conservative'):
        raise ValueError('Unknown advice preference')
    if type(revision) is not int or not 0 <= revision <= 2:
        raise ValueError('Bounded adaptation revision required')
    if action not in ('initial','reduce_history_perturbation','adapt_spatial_or_speed_diversity','expand_bounded_untried_proposals'):
        raise ValueError('Unknown bounded action')
    speed = environment['facts']['anchor_speed_mps']
    return dict(version=VERSION, preference=preference, revision=revision, action=action,
        minimum_ade_m=.20,
        minimum_path_ade_m=.30 if preference=='spatial' and revision==0 else .20,
        minimum_speed_mae_mps=float(np.clip(.04*speed,.15,.50)),
        minimum_speed_duration_s=1., rule='temporal_floor_and_spatial_or_speed_v1',
        pdms_strict_gt=.95, native_safety_override=False)


def validate(value, environment):
    if value.get('version') == 'scene_conditioned_diversity_v2':
        from .scene_diversity_skill import compile_policy
        expected=compile_policy(environment,value['scene_context'],value['preference'],value['revision'],value['action'])
        if value!=expected:raise ValueError('Scene-conditioned thresholds must replay')
        return value
    if value != policy(environment, value['preference'], value['revision'],value['action']):
        raise ValueError('Adaptive thresholds must replay from source context')
    return value


def _speed(a, b, threshold):
    speeds=[np.linalg.norm(np.diff(np.asarray(x,dtype=float),axis=0),axis=1)/.1 for x in (a,b)]
    delta=np.abs(speeds[0]-speeds[1]);run=longest=0
    for above in delta>=threshold:
        run=run+1 if above else 0;longest=max(longest,run)
    return float(delta.mean()),float(longest*.1)


def compare(candidate, retained, settings):
    from .trajectory_diversity import distances,path_distances
    validate(settings,candidate['environment_speed_policy'])
    rows=[]
    for other in retained:
        if candidate['comparison_key']!=other['comparison_key']:
            raise ValueError('Different source states cannot share diversity bucket')
        if candidate['environment_speed_policy']!=other['environment_speed_policy']:
            raise ValueError('Measured speed context differs')
        a,b=candidate['xy'],other['xy']
        row=dict(candidate_id=other['candidate_id'],**distances(a,b))
        full=path_distances(a,b);future=path_distances(a[20:],b[20:])
        row.update(full);row['future_path_ade_m']=future['path_ade_m']
        row['future_common_path_m']=future['common_path_m']
        full_speed,full_duration=_speed(a,b,settings['minimum_speed_mae_mps'])
        future_speed,future_duration=_speed(a[20:],b[20:],settings['minimum_speed_mae_mps'])
        spatial=all(x is not None and x>=settings['minimum_path_ade_m']
                    for x in (full['path_ade_m'],future['path_ade_m']))
        speed=(min(full_speed,future_speed)>=settings['minimum_speed_mae_mps']
               and min(full_duration,future_duration)>=settings['minimum_speed_duration_s'])
        temporal=min(row['ade_7s_m'],row['ade_future_5s_m'])>=settings['minimum_ade_m']
        row.update(speed_mae_mps=full_speed,future_speed_mae_mps=future_speed,
            speed_duration_s=full_duration,future_speed_duration_s=future_duration,
            spatial_distinct=spatial,speed_distinct=speed,
            novelty_basis='both' if spatial and speed else 'spatial' if spatial else 'speed' if speed else 'none',
            duplicate=not (temporal and (spatial or speed)))
        rows.append(row)
    return dict(distinct=not any(x['duplicate'] for x in rows),
        minimum_ade_m=settings['minimum_ade_m'],comparisons=rows,
        rule=settings['rule'],adaptive_policy=settings)


def proposal_grid(environment, settings, *, used=()):
    """Finite measured-envelope proposals, not a promise of executed novelty."""
    validate(settings,environment)
    scale=(1.,.65,.4)[settings['revision']] if settings['action']=='reduce_history_perturbation' else 1.
    lateral=(0.,-.2,.2,-.4,.4) if settings['preference'] in ('speed','conservative') else (0.,-.3,.3,-.6,.6)
    positive=environment['max_positive_delta_mps'];negative=environment['max_negative_delta_mps']
    excluded={tuple(round(float(r[k]),6) for k in ('longitudinal_m','lateral_m','speed_delta_mps')) for r in used}
    seen=set(excluded);rows=[]
    # Build the whole finite pool before choosing a budget prefix. Nested-loop
    # truncation otherwise spends most attempts at the same endpoint.
    for x in (0.,.5,-.5,1.,-1.,2.,-2.):
        for y in lateral:
            for speed in (0.,.5*positive,-.5*negative,positive,-negative):
                values=tuple(round(v*scale,6) for v in (x,y,speed))
                if values in seen or values==(0.,0.,0.):continue
                seen.add(values)
                rows.append(dict(zip(('longitudinal_m','lateral_m','speed_delta_mps'),values)))
    if not rows:return rows
    values=np.array([[r[k] for k in ('longitudinal_m','lateral_m','speed_delta_mps')] for r in rows])
    # SimScale's 5m/0.5m spacing is a coverage reference, not our safety gate.
    # Speed is an additional axis; normalizing it does not certify feasibility.
    normalized=values/np.array([5.*scale,.5*scale,max(positive,negative,.15)*scale])
    selected=[int(np.argmin(np.linalg.norm(normalized,axis=1)))]
    distance=np.linalg.norm(normalized-normalized[selected[0]],axis=1)
    while len(selected)<min(64,len(rows)):
        distance[selected]=-1.
        index=int(np.argmax(distance));selected.append(index)
        distance=np.minimum(distance,np.linalg.norm(normalized-normalized[index],axis=1))
    return [rows[i] for i in selected]
