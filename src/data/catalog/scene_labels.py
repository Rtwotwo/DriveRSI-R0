"""Scene-level retrospective labels: evidence, retrieval and clustering stay separate."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / 'src/config/data/scene-labels'
VERSION = 'driversi.scene_labels.v1'
SCOPE = 'retrospective_full_clip'


def taxonomy():
    return json.loads((CONFIG/'taxonomy.json').read_text())


def response_schema():
    tax = taxonomy()
    def array(items, maximum=12):
        return dict(type='array', items=items, maxItems=maximum)
    string = dict(type='string', minLength=1, maxLength=600)
    def obj(properties):
        return dict(type='object', additionalProperties=False,
                    required=list(properties), properties=properties)
    claim = dict(text_zh=string,
                 evidence_ids={**array(string,16),'uniqueItems':True},
                 fact_ids={**array(string,16),'uniqueItems':True},
                 certainty={'enum':['supported','tentative']})
    tag = obj(dict(axis={'enum':list(tax['facets'])}, label=string, **claim))
    event = obj(dict(event_type={'enum':tax['event_types']},
        actors={**array(string,8),'minItems':1,'uniqueItems':True},
        time_span_s=dict(type='array',items=dict(type='number'),minItems=2,maxItems=2), **claim))
    return {'$schema':'https://json-schema.org/draft/2020-12/schema', **obj(dict(
        summary=obj(claim), facets=array(tag,24), events=array(event,8),
        distinctive_features=array(obj(claim),6),
        unknowns=array(obj(dict(field=string,reason_zh=string)),12),
        proposed_new_tags=array(obj(dict(axis={'enum':list(tax['facets'])},
            proposed_name_zh=string, reason_zh=string)),4)))}


def measurements(bundle):
    """One full scene; aliases and numbers from logs, never inferred by the VLM."""
    if bundle['role'] != 'source_log' or bundle['source_split'] != 'navtrain':
        raise ValueError('Pilot requires original navtrain source; test/branches are not accepted')
    frames = bundle['frames']
    if len(frames)!=71 or [f['tick'] for f in frames]!=list(range(71)):
        raise ValueError('Complete seven-second source sequence required')
    times=np.array([f['time_seconds'] for f in frames],float)
    if not np.allclose(times,np.arange(71)/10):
        raise ValueError('Unexpected scene clock')
    poses=np.array([f['ego_rear_axle_to_local'] for f in frames],float)
    headings=np.unwrap(np.arctan2(poses[:,1,0],poses[:,0,0]))
    speed=np.linalg.norm(np.diff(poses[:,:2,3],axis=0),axis=1)/np.diff(times)
    actor_names={a:f'actor_{int(v["instance_id"]):04d}' for a,v in bundle['actors'].items()}
    tracks={}
    # At most 15 knots per track; retain exact sample basis, missing knots and presence intervals.
    for i in range(0,71,5):
        frame=frames[i];pose=poses[i];yaw=headings[i]
        c,s=np.cos(yaw),np.sin(yaw);rotation=np.array([[c,s],[-s,c]])
        for o in frame['objects']:
            alias=actor_names[o['actor_id']]
            track=tracks.setdefault(alias,dict(actor=alias,semantic_class=o['kind'],samples=[]))
            xy=np.array(o['pose_xyz_yaw'][:2],float)
            track['samples'].append(dict(t_s=float(times[i]),
                ego_relative_xy_m=np.round(rotation@(xy-pose[:2,3]),2).tolist(),
                world_xy_m=np.round(xy,2).tolist(),basis=o['sample_basis']))
    for actor,track in tracks.items():
        present=[f['time_seconds'] for f in frames
                 if any(actor_names[o['actor_id']]==actor for o in f['objects'])]
        intervals=[]
        for t in present:
            if not intervals or t-intervals[-1][1]>.100001:intervals.append([t,t])
            else:intervals[-1][1]=t
        track['represented_intervals_s']=intervals
        track['sampled_nearest_ego_distance_m']=round(min(np.linalg.norm(x['ego_relative_xy_m']) for x in track['samples']),2)
    result=dict(scope=SCOPE,window_s=[0.,7.],anchor_s=2.,
        fact_definitions={'ego_motion':'Source-pose finite differences, not measured velocity or inferred intent.',
            'actor_tracks':'Each track uses its own stable source identity. Ego-relative x forward/y left at that sample; world XY uses one fixed map origin. Annotation noise/holds are retained.',
            'map_context':'Loaded map inventory only. Counts do not prove ego lane membership, topology or right-of-way.'},
        ego_motion=dict(path_length_xy_m=round(float(np.linalg.norm(np.diff(poses[:,:2,3],axis=0),axis=1).sum()),3),
            net_heading_change_deg=round(float(np.rad2deg(headings[-1]-headings[0])),3),
            annotation_speed_range_mps=np.round([speed.min(),speed.max()],3).tolist(),
            timeline=[dict(t_s=float(times[i]),world_xy_m=np.round(poses[i,:2,3],2).tolist(),
                           heading_change_deg=round(float(np.rad2deg(headings[i]-headings[0])),2)) for i in range(0,71,5)]),
        actor_tracks=list(tracks.values()),
        map_context=dict(available_layers=[k for k,v in bundle.get('map_layers',{}).items() if v.get('availability')=='available'],
            ego_lane_association='not_computed',road_elevation='planar_display_not_metric_terrain'),
        source_quality=dict(state_count=71,unique_annotated_actors=len(bundle['actors']),
            summarized_actors=len(tracks),dimension_sampling=bundle.get('dimension_sampling'),
            sampling='Offline source bracket interpolation; not new 10Hz sensor observations',
            unobserved_actor_is_absent=False))
    return result


def fact_ids(context):
    return {'ego_motion','map_context','source_quality'} | {
        'track:'+t['actor'] for t in context['actor_tracks']}


def _check_shape(value, schema):
    """Validate only the closed types used by our exported response contract."""
    if 'enum' in schema:
        if value not in schema['enum']:raise ValueError('Unsupported enum value')
        return
    kind=schema['type']
    if kind=='object':
        if not isinstance(value,dict) or set(value)!=set(schema['required']):
            raise ValueError('Fixed scene-label fields required')
        for key,child in schema['properties'].items():_check_shape(value[key],child)
    elif kind=='array':
        if not isinstance(value,list) or not schema.get('minItems',0)<=len(value)<=schema['maxItems']:
            raise ValueError('Invalid list length/type')
        if schema.get('uniqueItems') and len({json.dumps(x,sort_keys=True) for x in value})!=len(value):
            raise ValueError('Duplicate list values')
        for child in value:_check_shape(child,schema['items'])
    elif kind=='string':
        if not isinstance(value,str) or not value.strip() or not schema.get('minLength',0)<=len(value)<=schema['maxLength']:
            raise ValueError('Bounded nonempty string required')
    elif kind=='number':
        if type(value) not in (float,int) or not np.isfinite(value):
            raise ValueError('Finite numeric value required')
    else:raise ValueError('Unsupported contract type')


def parse(raw, context, evidence):
    def unique(pairs):
        out={}
        for k,v in pairs:
            if k in out:raise ValueError('Duplicate label key')
            out[k]=v
        return out
    value=json.loads(raw,object_pairs_hook=unique,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite label value')))
    _check_shape(value,response_schema())
    images={e['id'] for e in evidence};facts=fact_ids(context)
    actors={t['actor']:t for t in context['actor_tracks']}
    tax=taxonomy();seen=set()
    claims=[value['summary'],*value['facets'],*value['events'],*value['distinctive_features']]
    for claim in claims:
        if not claim['evidence_ids'] and not claim['fact_ids']:
            raise ValueError('Each claim needs image or structured evidence')
        if not set(claim['evidence_ids'])<=images or not set(claim['fact_ids'])<=facts:
            raise ValueError('Unknown evidence or fact reference')
    for tag in value['facets']:
        key=(tag['axis'],tag['label'])
        if tag['label'] not in tax['facets'][tag['axis']] or key in seen:
            raise ValueError('Unknown or repeated taxonomy label')
        seen.add(key)
    for event in value['events']:
        lo,hi=event['time_span_s']
        if not 0<=lo<hi<=7:raise ValueError('Event outside scene time range')
        if event['certainty']=='supported' and event['evidence_ids']:
            cited=[e['relative_time_s']+context.get('anchor_s',2) for e in evidence
                   if e['id'] in event['evidence_ids']]
            if not any(lo-.01<=t<=hi+.01 for t in cited):
                raise ValueError('Event images do not overlap the event interval')
        for actor in event['actors']:
            if actor!='ego' and actor not in actors:raise ValueError('Unknown actor alias')
            expected='ego_motion' if actor=='ego' else 'track:'+actor
            if expected not in event['fact_ids']:
                raise ValueError('An event actor needs its own structured evidence')
            if actor!='ego':
                # Do not certify a continuous event across an absent annotation interval.
                if not any(lo>=a-1e-6 and hi<=b+1e-6 for a,b in actors[actor]['represented_intervals_s']):
                    raise ValueError('Event bridges missing actor support')
                if event['certainty']=='supported':
                    samples=actors[actor].get('samples',[])
                    left=[s for s in samples if s['t_s']<=lo]
                    right=[s for s in samples if s['t_s']>=hi]
                    if not left or not right:
                        raise ValueError('Supported event lacks temporal track samples')
                    lower=max(s['t_s'] for s in left);upper=min(s['t_s'] for s in right)
                    support=[s for s in samples if lower<=s['t_s']<=upper]
                    if len(support)<2 or any('uncertain' in s['basis'] for s in support):
                        raise ValueError('Supported event relies on uncertain/single track observation')
    return value


def assemble(bundle, context, observations, evidence, model):
    if bundle.get('source_split') != 'navtrain' or bundle.get('role') != 'source_log':
        raise ValueError('Pilot publication requires original navtrain evidence')
    observations=parse(json.dumps(observations,ensure_ascii=False),context,evidence)
    tags=[t['axis']+':'+t['label'] for t in observations['facets'] if t['certainty']=='supported']
    tentative_tags=[t['axis']+':'+t['label'] for t in observations['facets'] if t['certainty']=='tentative']
    claims=[observations['summary'],*observations['facets'],*observations['events'],*observations['distinctive_features']]
    prose=[x['text_zh'] for x in claims if x['certainty']=='supported']
    registry=[dict(id=e['id'],camera=e['camera'],time_relative_to_anchor_s=e['relative_time_s']) for e in evidence]
    return dict(schema_version=VERSION,taxonomy_version=taxonomy()['version'],
        scene=dict(token=bundle['token'],source_log=bundle['source_log'],split='navtrain',unit='one_scene',window_s=[0,7],anchor_s=2),
        information_scope=dict(scope=SCOPE,future_included=True,online_policy_input=False),
        event_time_contract=dict(time_span_s='proposed_observation_or_encounter_window',
            action_onset_and_end='not_estimated',event_order='support_window_start_only',
            duration_as_clustering_feature=False),
        review=dict(status='awaiting_user_review',approved_for_batch=False,
                    factual_status='vlm_claims_unverified',format_valid=True),
        producer=dict(model=model),
        measured_summary={k:context[k] for k in ['ego_motion','map_context','source_quality']},
        labels=observations,evidence_registry=registry,
        support_files=dict(structured_evidence='context.json',source_photos='source-evidence-overview.jpg'),
        draft_retrieval=dict(text_zh=' '.join(dict.fromkeys(prose)),tag_keys=tags,
                             tentative_observations=[x for x in claims if x['certainty']=='tentative'],
                             scope=SCOPE,indexable=False),
        draft_clustering=dict(multi_membership=True,semantic_tag_keys=tags,
            tentative_tag_keys=tentative_tags,
            event_sequence=[dict(type=x['event_type'],actors=x['actors'],observation_window_s=x['time_span_s'],
                                 action_interval_s=None)
                            for x in sorted(observations['events'],key=lambda x:x['time_span_s'][0])
                            if x['certainty']=='supported'],
            tentative_events=[x for x in observations['events'] if x['certainty']=='tentative'],
            cluster_id=None,fit_on='navtrain_only_after_review',
            missing_modalities_are_negative_evidence=False,pool_write_allowed=False),
        failure_diagnosis=dict(included=False,reason='Scene semantics are separate from model score and causal diagnosis'),
        limitations=['Structured measurements are annotation-derived, not perfect physical truth.',
            'Image references validate provenance, not claim correctness.',
            'No verified image-to-track identity binding: no actor appearance inferred from track class.',
            'One scene does not validate retrieval accuracy, clustering quality or class coverage.'],trainable=False)
