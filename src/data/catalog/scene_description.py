"""Versioned, evidence-scoped scene descriptions and deterministic prompt compilation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from driversi.data.catalog.store import identity
from driversi.rendering.whitebox.conditioning import describe

VERSION = 'driversi.scene_description.v2'
VISUAL_FIELDS = ('overview', 'road_layout', 'surroundings', 'lighting',
                 'road_surface', 'traffic_configuration', 'temporal_changes')


def observation_schema():
    claim = {'type': 'object', 'additionalProperties': False,
             'required': ['text', 'evidence_ids'], 'properties': {
                 'text': {'type': 'string', 'minLength': 1, 'maxLength': 500},
                 'evidence_ids': {'type': 'array', 'maxItems': 16, 'uniqueItems': True,
                                  'items': {'type': 'string'}}}}
    properties = {field: claim for field in VISUAL_FIELDS}
    properties['distinctive_features'] = {'type': 'array', 'maxItems': 4, 'items': claim}
    return {'$schema': 'https://json-schema.org/draft/2020-12/schema',
            'title': 'DriveRSI observed appearance v2', 'type': 'object',
            'additionalProperties': False, 'required': list(properties), 'properties': properties}


def parse_observations(raw, evidence_ids):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate observation key')
            result[key] = value
        return result
    text = raw.strip()
    if text.startswith('```json') and text.endswith('```'):
        text = text[7:-3].strip()
    value = json.loads(text, object_pairs_hook=unique)
    if not isinstance(value, dict) or set(value) != set(VISUAL_FIELDS) | {'distinctive_features'}:
        raise ValueError('Fixed v2 observation fields required')
    if not isinstance(value['distinctive_features'], list) or len(value['distinctive_features']) > 4:
        raise ValueError('At most four distinctive claims')
    for claim in [value[k] for k in VISUAL_FIELDS] + value['distinctive_features']:
        if not isinstance(claim, dict) or set(claim) != {'text', 'evidence_ids'}:
            raise ValueError('Every claim requires text and evidence_ids')
        t, ids = claim['text'], claim['evidence_ids']
        if not isinstance(t, str) or not t.strip() or len(t) > 500 or len(t.split()) > 60:
            raise ValueError('Bounded nonempty claim required')
        if not isinstance(ids, list) or any(not isinstance(x, str) for x in ids):
            raise ValueError('Evidence IDs must be a string array')
        if len(ids) > 16 or len(ids) != len(set(ids)) or not set(ids) <= set(evidence_ids):
            raise ValueError('Unknown or duplicate image evidence')
        if (t.lower().strip() == 'unknown') != (len(ids) == 0):
            raise ValueError('Unknown must have no evidence; every actual claim needs evidence')
    return value


def parse_generation(raw, evidence_ids):
    """Permit only one missing final root brace; never create a claim or evidence ID."""
    try:
        return parse_observations(raw, evidence_ids), None
    except json.JSONDecodeError:
        if not raw.rstrip().endswith(']'):
            raise
        # The exact fixed schema must parse after this single syntax-only edit.
        return parse_observations(raw.rstrip()+'}', evidence_ids), 'append_single_final_root_brace'


def prompt_path_v2(scope):
    if scope not in ('retrieval_history', 'video_clip'):
        raise ValueError('Unknown information scope')
    return Path(__file__).resolve().parents[3] / 'src/config/data/semantic_conditioning' / (scope + '_v2.md')


def description_schema():
    """Stable exported envelope; measured facts retain their documented native keys."""
    names = ['schema_version','scope','scene_identity','information_scope','scene_overview',
             'structured_facts','appearance','distinctive_features','retrieval','video_generation',
             'evidence','facts_identity','limitations','trainable']
    properties = {k: {'type':'object'} for k in names}
    properties.update(schema_version={'const':VERSION}, scope={'enum':['retrieval_history','video_clip']},
                      distinctive_features={'type':'array','items':{'type':'string'}},
                      retrieval={'type':['object','null']},video_generation={'type':['object','null']},
                      evidence={'type':'array'}, facts_identity={'type':'string'},
                      limitations={'type':'array','items':{'type':'string'}},trainable={'const':False})
    def fixed(fields):
        return {'type':'object','additionalProperties':False,'required':list(fields),'properties':fields}
    string={'type':'string'};strings={'type':'array','items':string}
    properties['scene_identity']=fixed({'token':string,'source_split':{'const':'navtrain'},
                                       'role':{'enum':['source_log','executed_branch']}})
    properties['information_scope']=fixed({'future_included':{'type':'boolean'},
                                          'start_relative_s':{'const':-2},'end_relative_s':{'enum':[0,5]},
                                          'online_policy_input':{'const':False}})
    claim=observation_schema()['properties']['overview']
    properties['scene_overview']=fixed({'structured':string,'reviewed_visual':{'anyOf':[{'type':'null'},claim]}})
    properties['appearance']=fixed({'observations':{'anyOf':[{'type':'null'},observation_schema()]},
                                   'approved_paths':strings,'review':{'type':['object','null']},
                                   'status':{'enum':['partially_reviewed','draft_not_released']}})
    retrieval=fixed({'text':string,'facets':fixed({'nearby_30m_counts':{'type':'object','additionalProperties':{'type':'integer'}},
                                                 'heading_change_degrees':{'type':'number'},'path_length_m':{'type':'number'}})})
    video=fixed({'positive_prompt':string,'negative_prompt':string,
                 'camera_prompts':fixed({cam:string for cam in ('CAM_F0','CAM_L0','CAM_R0','CAM_B0')}),
                 'geometry_constraints':strings,'status':{'const':'conditioning_ready_not_generated_video_quality_approval'}})
    properties['retrieval']={'anyOf':[{'type':'null'},retrieval]}
    properties['video_generation']={'anyOf':[{'type':'null'},video]}
    return {'$schema':'https://json-schema.org/draft/2020-12/schema','title':VERSION,
            'type':'object','additionalProperties':False,'required':names,'properties':properties,
            'allOf':[{'if':{'properties':{'scope':{'const':'retrieval_history'}}},
                      'then':{'properties':{'video_generation':{'type':'null'},'retrieval':{'type':'object'}}},
                      'else':{'properties':{'retrieval':{'type':'null'},'video_generation':{'type':'object'}}}}]}


def scoped_measurements(bundle, scope):
    facts = describe(bundle, scope)
    stop = bundle['anchor_tick'] + 1 if scope == 'retrieval_history' else len(bundle['frames'])
    rows = bundle['frames'][:stop]
    positions = np.asarray([f['ego_rear_axle_to_local'] for f in rows])[:, :3, 3]
    headings = np.unwrap([np.arctan2(f['ego_rear_axle_to_local'][1][0], f['ego_rear_axle_to_local'][0][0]) for f in rows])
    snapshots = []
    for i in sorted(set(range(0, len(rows), 10)) | {len(rows)-1}):
        f = rows[i]
        snapshots.append({'time_relative_to_anchor_s': round(f['time_seconds']-2, 3),
                          'annotated_class_counts': {k: sum(o['kind']==k for o in f['objects'])
                                                     for k in sorted({o['kind'] for o in f['objects']})},
                          'sampling': f.get('sampling', 'unspecified')})
    facts['ego_path_length_xy_m'] = round(float(np.linalg.norm(np.diff(positions[:, :2], axis=0), axis=1).sum()), 3)
    facts['ego_heading_change_degrees'] = round(float(np.rad2deg(headings[-1]-headings[0])), 3)
    facts['timeline'] = snapshots
    facts['motion_semantics'] = 'Annotation/physics displacement, not inferred intention or measured behavior classification'
    return facts


def _claim_paths(observations):
    if observations is None:
        return {}
    return {**{k: observations[k] for k in VISUAL_FIELDS},
            **{f'distinctive_features.{i}': c for i, c in enumerate(observations['distinctive_features'])}}


def build_description(bundle, scope, *, observations=None, evidence=(), approved_paths=(), review=None):
    if bundle.get('source_split') != 'navtrain':
        raise ValueError('This training generation entry accepts navtrain only')
    facts = scoped_measurements(bundle, scope)
    evidence = list(evidence)
    if len({x['id'] for x in evidence}) != len(evidence):
        raise ValueError('Duplicate evidence ID')
    if scope == 'retrieval_history' and any(x['relative_time_s'] > 0 for x in evidence):
        raise ValueError('Future image evidence in retrieval')
    if observations is not None:
        observations = parse_observations(json.dumps(observations), [x['id'] for x in evidence])
    claims = _claim_paths(observations)
    approved_paths = sorted(set(approved_paths))
    if not set(approved_paths) <= set(claims):
        raise ValueError('Review references absent claim')
    if approved_paths and (not isinstance(review, dict) or not review.get('reviewer') or not review.get('note')):
        raise ValueError('Explicit factual review needed before releasing appearance')
    released = {k: claims[k] for k in approved_paths if claims[k]['text'].strip().lower() != 'unknown'}
    counts = {k: v for k, v in facts['nearby_class_counts_30m'].items() if v}
    count_text = ', '.join(f'{v} {k}' for k, v in counts.items()) or 'no annotated objects'
    scope_text = 'past/current' if scope == 'retrieval_history' else 'offline full-clip'
    overview = (f"A {facts['duration_seconds']:.1f}-second {scope_text} driving window. "
                f"At the anchor, the 30m neighborhood contains {count_text}. "
                f"Ego path length is {facts['ego_path_length_xy_m']:.2f}m, with a net heading change of "
                f"{facts['ego_heading_change_degrees']:.1f} degrees. Counts are annotations, not guaranteed visible pixels.")
    nearby = facts['anchor_objects'][:8]
    spatial = '; '.join(f"instance {o['instance_id']} {o['semantic_class']} at ego-frame xyz "
                        f"{o['relative_ego_xyz_m']}m, dimensions {o['dimensions_lwh']}m" for o in nearby)
    distinctions = [f"{v} annotated {k} within30m" for k, v in counts.items()]
    distinctions += [c['text'] for k,c in released.items() if k.startswith('distinctive_features.')]
    appearance_text = ' '.join(c['text'] for c in released.values())
    timeline_text = '; '.join(f"t={x['time_relative_to_anchor_s']:+.1f}s: " +
                             ', '.join(f'{v} {k}' for k,v in x['annotated_class_counts'].items())
                             for x in facts['timeline'])
    text = overview + (' ' + spatial if spatial else '') + (' ' + appearance_text if appearance_text else '')
    constraints = [
        'Use the supplied per-frame object identities, dimensions, trajectories and calibrated camera poses.',
        'Semantic colors identify classes, never paint colors; unspecified vehicle subtype remains unknown.',
        'Keep one consistent appearance for each instance across time and cameras; do not merge different identities.',
        'Preserve roads, object count, motion and visibility implied by the supplied geometry; text cannot change geometry.',
        'Reference-view photographs support appearance only, not the pose of a changed physical branch.',
        'Cuboid occlusion and planar road elevation are proxies requiring generated-image validation.'
    ]
    camera_prompts = {}
    for cam in ('CAM_F0','CAM_L0','CAM_R0','CAM_B0'):
        camera_prompts[cam] = ' '.join(c['text'] for c in released.values()
                                     if {i.split('_f')[0] for i in c['evidence_ids']}=={cam}) or 'No reviewed camera-specific appearance; follow geometry and valid reference images.'
    generation = None if scope != 'video_clip' else {
        'positive_prompt': 'Generate a synchronized multi-camera driving video from the supplied conditions. '+text+' Temporal annotation summary: '+timeline_text+' '+' '.join(constraints),
        'negative_prompt': 'No invented vehicles, pedestrians, lane markings or signal states. No object duplication, identity switches, teleportation, geometry changes, flicker, inconsistent paint across cameras, or semantic-color paint substitution.',
        'camera_prompts': camera_prompts,
        'geometry_constraints': constraints,
        'status': 'conditioning_ready_not_generated_video_quality_approval'
    }
    result = {'schema_version': VERSION, 'scope': scope,
              'scene_identity': {k: bundle[k] for k in ('token','source_split','role')},
              'information_scope': {'future_included': facts['future_included'], 'start_relative_s': -2,
                                    'end_relative_s': 0 if scope=='retrieval_history' else 5,
                                    'online_policy_input': False},
              'scene_overview': {'structured': overview, 'reviewed_visual': released.get('overview')},
              'structured_facts': facts,
              'appearance': {'observations': observations, 'approved_paths': approved_paths,
                             'review': review, 'status': 'partially_reviewed' if approved_paths else 'draft_not_released'},
              'distinctive_features': distinctions,
              'retrieval': {'text': text, 'facets': {'nearby_30m_counts': counts,
                              'heading_change_degrees': facts['ego_heading_change_degrees'],
                              'path_length_m': facts['ego_path_length_xy_m']}} if scope=='retrieval_history' else None,
              'video_generation': generation,
              'evidence': evidence, 'facts_identity': identity(facts),
              'limitations': facts['uncertainty'] + ['VLM evidence is sparse; linked evidence is not a correctness certificate.'],
              'trainable': False}
    return result
