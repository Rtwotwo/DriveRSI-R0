"""Geometry is compiled; a visual model may only propose scene appearance."""
from __future__ import annotations

import json
import math
from pathlib import Path
import re

import numpy as np

from . import bailian_caption as client
from . import scene_description as description
from .store import identity
from driversi.rendering.whitebox.exporter import sha256

FIELDS = ('setting', 'road_appearance', 'built_environment', 'vegetation', 'illumination')
VERSION = 'geometry_bound_appearance_v3_1'
# This conservative screen rejects candidate prose; it never edits world states.
UNBOUND = re.compile(r'\b(car|cars|vehicle|vehicles|bus|buses|truck|trucks|van|vans|'
    r'pedestrian|pedestrians|cyclist|cyclists|bicycle|bicycles|motorcycle|ego|'
    r'left|right|ahead|behind|approaching|receding|overtaking|turning|moving|'
    r'parked|crossing|people|person|sedan|sedans|suv|suvs|hatchback|coach|accelerat\w*|runs|running|stops|stopping|merg\w*|lanes?|signal|signals)\b', re.I)


def prompt(scope):
    timing = ('Only current and past photographs are supplied. Never describe a future event.'
              if scope == 'retrieval_history' else
              'Photographs sample a complete offline clip. Do not infer actor motion between them.')
    return '''You annotate photographed appearance for a structured driving scene.
All object identities, counts, positions, orientation, trajectories, occlusion order,
are supplied separately by authoritative whitebox data. Lane topology is NOT supplied
in this caption pilot; do not infer lane count, topology or signal states.
DO NOT describe or infer these in your answer. Do not mention cars, buses, trucks,
people, bicycles, ego actions, motion, image-left/right or relative directions.
No actor paint color or subtype: photographs have no verified 2D-to-3D identity matching.
Semantic whitebox colors encode classes and are NOT real paint colors.
Describe only static visible environment and appearance: road surface texture,
marking appearance (not lane count/topology), architectural materials, vegetation,
visible sky and illumination. No inferred weather, city names, text/plate/brand reading.
Distinguish scenes with concrete visible details; avoid generic safe-driving instructions,
invented intentions, decorative adjectives, inferred history and unsupported specificity.
Use English, at most 35 words per claim. Copy evidence_ids exactly from image_id.
Each non-unknown claim needs 1-4 supporting image IDs. Unknown is {"text":"unknown","evidence_ids":[]}.
Return exactly these JSON keys: setting, road_appearance, built_environment,
vegetation, illumination, distinctive_features. First five values are claim objects
{"text":"...","evidence_ids":["..."]}. distinctive_features is a list of at most
four nonredundant claim objects. No coordinates or kinematic fields allowed.
''' + timing + '\nRequired template (all five scalar fields MUST be claim objects, never strings):\n' + json.dumps({**{k: {'text':'unknown','evidence_ids':[]} for k in FIELDS},'distinctive_features':[]})


def compile_geometry(bundle, scope):
    if scope not in ('retrieval_history', 'video_clip'):
        raise ValueError('Unknown information scope')
    anchor = bundle['anchor_tick']
    frames = bundle['frames'][:anchor+1] if scope == 'retrieval_history' else bundle['frames']
    ego_anchor = np.asarray(bundle['frames'][anchor]['ego_rear_axle_to_local'], float)
    inverse = np.linalg.inv(ego_anchor)
    anchor_yaw = math.atan2(ego_anchor[1, 0], ego_anchor[0, 0])
    tracks = {}
    for frame in frames:
        for obj in frame['objects']:
            pose = obj['pose_xyz_yaw']
            xyz = (inverse @ np.r_[pose[:3], 1.])[:3]
            yaw = math.degrees(math.atan2(math.sin(pose[3]-anchor_yaw), math.cos(pose[3]-anchor_yaw)))
            tracks.setdefault(obj['actor_id'], []).append({
                'time_relative_s': frame['relative_to_anchor_seconds'],
                'position_in_anchor_ego_m': np.round(xyz, 4).tolist(),
                'heading_in_anchor_ego_degrees': round(yaw, 4),
                'presence_evidence': obj['sample_basis'],
                'semantic_class': obj['kind'], 'dimensions_lwh_m': obj['dimensions_lwh'],
                'semantic_color': obj['semantic_color']})
    summaries = []
    for actor, samples in tracks.items():
        at_anchor = next((s for s in samples if abs(s['time_relative_s']) < 1e-8), None)
        displacement = np.array(samples[-1]['position_in_anchor_ego_m']) - samples[0]['position_in_anchor_ego_m']
        summaries.append({'actor_id': actor, 'semantic_class': samples[0]['semantic_class'],
            'at_anchor': at_anchor, 'first_represented_sample_s': samples[0]['time_relative_s'],
            'last_represented_sample_s': samples[-1]['time_relative_s'],
            'terminal_presence_evidence': samples[-1]['presence_evidence'],
            'uncertain_held_sample_count': sum('uncertain' in x['presence_evidence'] for x in samples),
            'annotation_displacement_m': np.round(displacement, 4).tolist(),
            'sample_count': len(samples), 'physical_appearance': 'unknown_unbound',
            'missing_support_is_not_physical_exit': True})
    summaries.sort(key=lambda x: (x['at_anchor'] is None,
        np.linalg.norm(x['at_anchor']['position_in_anchor_ego_m'][:2]) if x['at_anchor'] else float('inf'), x['actor_id']))
    facts = description.scoped_measurements(bundle, scope)
    # Only the scoped frames are exported. History consumers never get future tracks.
    return {'schema_version': VERSION, 'scope': scope, 'token': bundle['token'],
        'source_split': bundle['source_split'], 'coordinate_frame': 'fixed ego at anchor: x forward, y left, z up',
        'geometry_authority': 'source_log_whitebox; VLM cannot override',
        'motion_semantics': 'annotation displacement, not measured speed, relative velocity or intent',
        'sampling': '0.1s offline bracket interpolation; NOT 10Hz sensor ground truth',
        'frame_count': len(frames), 'frame_interval_s': bundle['timing']['interval_seconds'],
        'appearance_binding': 'unknown; never infer paint/subtype from semantic class color',
        'map_geometry_status': bundle.get('map_geometry_status', 'not_included_in_scoped_caption_geometry'),
        'complete_video_condition': False,
        'actor_summaries': summaries, 'actor_tracks': tracks,
        'ego_frames': [{'time_relative_s': f['relative_to_anchor_seconds'],
                       'ego_rear_axle_to_local': f['ego_rear_axle_to_local'],
                       'cameras': f['cameras'], 'signals': f['signals']} for f in frames],
        'measured_facts': facts, 'class_palette': bundle['palette'], 'trainable': False}


def geometry_text(geometry):
    facts = geometry['measured_facts']
    lines = [f"Structured scope={geometry['scope']}; "
        f"nearby semantic class counts within 30m={json.dumps(facts['nearby_class_counts_30m'], sort_keys=True)}.",
        f"Ego annotation path length={facts['ego_path_length_xy_m']}m; "
        f"heading change={facts['ego_heading_change_degrees']} degrees.",
        'Object coordinates below are in the FIXED anchor ego frame: x forward, y left, z up.']
    for a in geometry['actor_summaries'][:12]:
        if a['at_anchor']:
            x = a['at_anchor']
            lines.append(f"ID={a['actor_id']} class={a['semantic_class']} "
                f"anchor_xyz_m={x['position_in_anchor_ego_m']} heading_deg={x['heading_in_anchor_ego_degrees']} "
                f"annotation_displacement_m={a['annotation_displacement_m']} "
                f"represented_interval_s=[{a['first_represented_sample_s']},{a['last_represented_sample_s']}] "
                f"terminal_basis={a['terminal_presence_evidence']} uncertain_held_samples={a['uncertain_held_sample_count']}.")
    lines += ['All IDs, full tracks, dimensions, camera calibration and class colors come from the scoped geometry JSON.',
              'Annotation displacement is not measured velocity or driving intent. Missing annotation is not physical exit.']
    return '\n'.join(lines)


def prepare(database, bundles, config, output):
    output = Path(output)
    jobs = client.prepare(database, bundles, config, output)
    for job in jobs:
        geometry = compile_geometry(json.loads(Path(job['bundle']).read_text()), job['scope'])
        target = output/'geometry'/f"{job['token']}-{job['scope']}.json"
        client.write(target, geometry)
        job.update(prompt=prompt(job['scope']),
                   facts={'scope': job['scope'], 'task': 'appearance_only_no_actor_identity_binding',
                          'geometry_supplied_separately': True},
                   geometry_file=str(target.resolve()), geometry_sha256=sha256(target),
                   compiler_sha256=sha256(__file__), schema_version=VERSION)
        job['input_identity'] = identity({k:v for k,v in job.items() if k != 'input_identity'})
        client.write(output/'prepared'/f"{job['token']}-{job['scope']}.json", job)
    return jobs


def parse(raw, allowed):
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != set(FIELDS) | {'distinctive_features'}:
        raise ValueError('Appearance-only schema keys required; kinematics forbidden')
    if not isinstance(value['distinctive_features'], list) or len(value['distinctive_features']) > 4:
        raise ValueError('At most four visual distinctions')
    warnings = []
    paths = [(k, value[k]) for k in FIELDS] + [(f'distinctive_features.{i}', c) for i,c in enumerate(value['distinctive_features'])]
    accepted = {}
    for path, claim in paths:
        if not isinstance(claim, dict) or set(claim) != {'text', 'evidence_ids'}:
            raise ValueError('Claim must contain text and evidence_ids')
        text, ids = claim['text'], claim['evidence_ids']
        if not isinstance(text, str) or not text.strip() or len(text) > 500 or len(text.split()) > 40:
            raise ValueError('Invalid appearance text')
        if not isinstance(ids, list) or len(ids) > 4 or any(x not in allowed for x in ids):
            raise ValueError('Invalid evidence IDs')
        if (text.lower() == 'unknown') != (len(ids) == 0):
            raise ValueError('Unknown/evidence mismatch')
        if UNBOUND.search(text):
            warnings.append({'path': path, 'reason': 'actor_or_direction_claim_outside_visual_scope', 'claim': claim})
        elif text.lower() != 'unknown':
            accepted[path] = claim
    return value, accepted, warnings


def result_from_response(job, model, response):
    choice = response['choices'][0]
    if choice.get('finish_reason') != 'stop' or not isinstance(choice['message']['content'], str):
        raise ValueError('Incomplete appearance response')
    raw = choice['message']['content']
    value, accepted, warnings = parse(raw, [e['id'] for e in job['evidence']])
    if sha256(job['geometry_file']) != job['geometry_sha256'] or sha256(job['bundle']) != job['source_sha256']:
        raise ValueError('Geometry/source changed since request preparation')
    geometry = json.loads(Path(job['geometry_file']).read_text())
    structured = geometry_text(geometry)
    visual = ' '.join(c['text'] for c in accepted.values())
    constraints = ('Render the supplied synchronized camera trajectories and per-frame semantic geometry. '
        'Preserve object identities, class, dimensions, position, heading, trajectories, road topology, '
        'projected depth order and occlusion where geometry supports them. No added/deleted actors, no teleportation, no time reversal. '
        'Keep appearance stable across cameras and time. Class colors are labels, not paint. '
        'Appearance text is an unverified candidate; conflicting geometry always wins. '
        'Sparse source photographs cannot establish per-frame visibility or hidden-surface appearance. '
        'This caption artifact has no lane polygons or road topology: pair a verified map and '
        'visibility conditions before using it as complete video-generation input.')
    draft = {'status': 'unreviewed_not_automatic_generation_admission',
        'retrieval_text': structured+'\nAppearance hypotheses: '+visual if job['scope']=='retrieval_history' else None,
        'video_positive_prompt': constraints+'\n'+structured+'\nAppearance hypotheses: '+visual if job['scope']=='video_clip' else None,
        'negative_prompt': 'hallucinated actors, changed identity, wrong class-color interpretation, '
            'altered road geometry, flicker, motion drift, wrong occlusion, camera-inconsistent appearance',
        'geometry_file': job['geometry_file'], 'geometry_sha256': job['geometry_sha256']}
    return {'schema_version': VERSION, 'status': 'format_valid_unreviewed', 'model': model,
        'returned_model': response.get('model'), 'request_id': response.get('id'), 'usage': response.get('usage'),
        'raw': raw, 'appearance_observations': value, 'candidate_claims': accepted,
        'quarantined_claims': warnings, 'geometry': {'file': job['geometry_file'], 'sha256': job['geometry_sha256'],
        'actor_count': len(geometry['actor_summaries']), 'frame_count': geometry['frame_count']},
        'draft_prompts': draft,
        'released_prompts': {'retrieval_text': structured if job['scope']=='retrieval_history' else None,
            'video_positive_prompt': constraints+'\n'+structured if job['scope']=='video_clip' else None,
            'appearance_status':'awaiting_explicit_evidence_review', 'appearance_included':False},
        'trainable': False}

