"""Compact retrieval-facing scene view; complete evidence stays in a linked audit."""
from __future__ import annotations

import json
import math
from pathlib import Path

from .scene_labels import CONFIG, VERSION, SCOPE

PROFILE = CONFIG / 'summary-profile.json'


def compact(full, audit_reference):
    """Project validated labels without treating them as factually approved."""
    if full.get('schema_version') != VERSION or not full['review']['format_valid']:
        raise ValueError('A validated full scene-label record is required')
    scope = full['information_scope']
    if scope != dict(scope=SCOPE, future_included=True, online_policy_input=False):
        raise ValueError('Preserve retrospective information scope')
    profile = json.loads(PROFILE.read_text())
    obs = full['labels']
    facets = [x for x in obs['facets'] if x['certainty'] == 'supported']

    def tags(*axes):
        return list(dict.fromkeys(x['label'] for x in facets if x['axis'] in axes))

    m = full['measured_summary']['ego_motion']
    start, end = full['scene']['window_s']
    distance = m['path_length_xy_m']
    low, high = m['annotation_speed_range_mps']
    if (not all(type(x) in (int, float) and math.isfinite(x)
                for x in (start, end, distance, low, high)) or
            end <= start or distance < 0 or not 0 <= low <= high):
        raise ValueError('Valid measured time, distance and speed range are required')
    speed = dict(mean=round(distance / (end-start) * 3.6, 2),
                 range=[round(low*3.6, 2), round(high*3.6, 2)],
                 basis='annotation_derived')
    events = list(dict.fromkeys(x['event_type'] for x in obs['events']
                 if x['certainty'] == 'supported' and
                 x['event_type'] not in profile['ego_only_event_types']))
    visual = tags('visibility', 'road_markings')
    visual += [x for x in tags('environment') if x in profile['visual_environment_labels']]
    uncertain = [dict(axis=x['axis'], label=x['label']) for x in obs['facets']
                 if x['certainty']=='tentative']
    uncertain += [dict(kind='event', label=x['event_type']) for x in obs['events']
                  if x['certainty']=='tentative']
    if obs['summary']['certainty']=='tentative':
        uncertain.append(dict(kind='caption', text=obs['summary']['text_zh']))
    return dict(
        schema_version=profile['version'],
        token=full['scene']['token'], split=full['scene']['split'],
        scope=SCOPE, window_s=[start, end], online_policy_input=False,
        caption=obs['summary']['text_zh'] if obs['summary']['certainty']=='supported' else None,
        road=dict(labels=tags('road_context', 'road_geometry'),
                  lane_association=full['measured_summary']['map_context']['ego_lane_association'],
                  signal_control='unknown', signal_state='unknown'),
        ego_behavior=tags('ego_motion'),
        speed_kmh=speed,
        interaction=dict(participants=tags('traffic_configuration'),
                         labels=tags('interaction'), events=events),
        visual_context=list(dict.fromkeys(visual)),
        uncertain=uncertain,
        unknown=[x['field'] for x in obs['unknowns']],
        review=dict(status=full['review']['status'], format_valid=True,
                    factual_status=full['review']['factual_status'],
                    approved_for_batch=False, indexable=False),
        evidence_ref=audit_reference, trainable=False)
