"""Source-caption advisory skills grounded in measured history; never safety gates."""
import hashlib
import json
import sqlite3
from pathlib import Path
from functools import lru_cache
import numpy as np

VERSION = 'scene_conditioned_diversity_v2'
CURVE_TAGS = {'左弯道路', '右弯道路', '弯道', '急弯', '高架桥下', '狭窄道路', '窄路', '道路收窄'}


@lru_cache(maxsize=2)
def _cluster_records(root, stamp):
    root=Path(root)
    with (root/'navtrain/members.jsonl').open() as f:
        members={r['token']:r for line in f if (r:=json.loads(line))}
    nodes={n['id']:n for n in json.loads((root/'clusters.json').read_text())['nodes']}
    return members,nodes


def source_context(database, environment, feedback=None):
    facts = environment['facts']
    with sqlite3.connect(Path(database).resolve().as_uri()+'?mode=ro', uri=True) as conn:
        row = conn.execute('select caption from captions where token=?', (facts['token'],)).fetchone()
    if not row or not row[0]:
        return dict(tags=[], status='missing_rules_only', source_token=facts['token'])
    caption = json.loads(row[0])
    if (caption['token'] != facts['token'] or caption['split'] != 'navtrain'
            or caption.get('future_images_included') is not False
            or caption.get('observation_scope') != 'current_anchor_four_cameras'):
        raise ValueError('Current training-source caption required')
    tags = caption.get('retrieval_tags', [])
    if not isinstance(tags, list) or any(not isinstance(t, str) for t in tags):
        raise ValueError('Caption tags must be strings')
    cluster_root=Path(database).parent.parent/'clusters'
    stamp=tuple((p.stat().st_mtime_ns,p.stat().st_size) for p in
        (cluster_root/'clusters.json',cluster_root/'navtrain/members.jsonl'))
    members,nodes=_cluster_records(str(cluster_root),stamp)
    member=members[facts['token']]
    if member['split']!='navtrain':raise ValueError('Only train cluster membership allowed')
    node=nodes[member['leaf']]
    cluster=dict(leaf=member['leaf'],label=node['label'],tags=node.get('tags',[]),
        map_labels=member.get('map_labels',[]),speed_mean_kmh=node['speed_mean_kmh'],
        speed_range_kmh=node['speed_range_kmh'],out_of_train_radius=member['out_of_train_radius'],
        member_sha256=hashlib.sha256(json.dumps(member,sort_keys=True).encode()).hexdigest())
    return dict(tags=sorted(set(tags)), status=caption.get('review_status', 'unverified'),cluster=cluster,
        feedback=feedback or {},
        source_token=facts['token'], caption_sha256=hashlib.sha256(row[0].encode()).hexdigest(),
        role='advisory_not_geometry_or_motion_truth')


def compile_policy(environment, context, preference, revision, action):
    from .environment_speed import policy_from_facts
    if environment != policy_from_facts(environment['facts']):
        raise ValueError('Frozen measured context required')
    if context['source_token'] != environment['facts']['token']:
        raise ValueError('Source context mismatch')
    if preference not in ('spatial','speed','mixed','conservative') or type(revision) is not int or not 0 <= revision <= 2:
        raise ValueError('Bounded source strategy required')
    if action not in ('initial','reduce_history_perturbation','adapt_spatial_or_speed_diversity','expand_bounded_untried_proposals'):
        raise ValueError('Unknown source action')
    speed = environment['facts']['anchor_speed_mps']
    slow = speed <= 2.
    curved = (environment['facts']['observed_curvature_inv_m'] or 0.) >= .01
    restricted_hint = bool(set(context['tags']) & CURVE_TAGS)
    cluster=context.get('cluster',{})
    cluster_hint=any(t['text'] in CURVE_TAGS for t in cluster.get('tags',[])) and not cluster.get('out_of_train_radius',True)
    speed_first = slow or curved or restricted_hint or cluster_hint or preference == 'speed'
    skill = 'slow_source_speed' if slow else 'constrained_source_speed' if speed_first else 'mixed_source'
    # Continuous source scales. Cluster speed is a prior, not the current speed.
    cluster_speed=float(cluster.get('speed_mean_kmh',speed*3.6))/3.6
    if not np.isfinite(cluster_speed) or cluster_speed<0:
        raise ValueError('Finite nonnegative cluster speed required')
    effective=.85*speed+.15*min(cluster_speed,2.*speed+.5)
    gap=environment['facts']['forward_box_gap_m']
    pressure=1. if gap is None else min(1.,max(.25,gap/max(2.,2.*speed)))
    curve_factor=1./(1.+20.*(environment['facts']['observed_curvature_inv_m'] or 0.))
    feedback=context.get('feedback',{})
    if any(type(n) is not int or n<0 for n in feedback.values()):
        raise ValueError('Nonnegative integer terminal counts required')
    attempts=max(1,sum(feedback.get(k,0) for k in ('teacher_rejected','trajectory_duplicate_before_render','training_sample_ready')))
    duplicate_fraction=feedback.get('trajectory_duplicate_before_render',0)/attempts
    scale=1./(1.+revision*duplicate_fraction) if action=='adapt_spatial_or_speed_diversity' else 1.
    threshold=float(np.clip((.02 if speed_first else .04)*effective*pressure*scale,.06,.50))
    temporal=float(np.clip(.5*threshold,.03,.25))
    path=float(np.clip(.02*effective*curve_factor*pressure*scale,.06,.35))
    return dict(version=VERSION, scene_context=context, skill_id=skill,
        preference='speed' if speed_first else preference, revision=revision, action=action,
        minimum_ade_m=temporal, minimum_path_ade_m=path,
        minimum_speed_mae_mps=threshold, minimum_speed_duration_s=1.,
        rule='temporal_floor_and_spatial_or_speed_v2', pdms_strict_gt=.95,
        native_safety_override=False, learned_optimal=False,
        feedback_scope='all_prior_problem_rounds_per_source',
        feedback_denominator=['teacher_rejected','trajectory_duplicate_before_render','training_sample_ready'],
        reasons=dict(measured_slow=slow, measured_curve=curved, advisory_restricted_hint=restricted_hint,
            cluster_hint=cluster_hint,pressure=pressure,curve_factor=curve_factor,duplicate_fraction=duplicate_fraction))


def saturated(rows):
    """A finite unproductive tail yields to the existing recovery controller."""
    tail = rows[-8:]
    return (len(tail) == 8 and all(r['status'] in {'teacher_rejected','trajectory_duplicate_before_render'} for r in tail)
            and sum(r['status']=='trajectory_duplicate_before_render' for r in tail)>=6)
