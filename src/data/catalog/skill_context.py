"""Read-only, current-anchor caption evidence for offline skill diagnosis."""
import json
import re
import sqlite3
from pathlib import Path

DIMENSIONS = {'road_layout', 'lane_markings', 'traffic_control', 'vehicle_configuration',
              'vulnerable_users', 'roadside_obstacles', 'visibility_context'}
CAMERAS = {f'CAM_{camera}_anchor' for camera in ('F0', 'L0', 'R0', 'B0')}
# These state fields have a separate, native authoritative observation contract.
STATE_TEXT = re.compile(r'导航|指令|当前速度|车速|navigation|command|km/?h|m/s', re.I)


def normalized_command(command):
    return 'FORWARD' if command in ('GO STRAIGHT', 'GO FORWARD') else command


def context(caption, observation):
    if caption.get('schema_version') != 'driversi.cluster_caption.v1':
        raise ValueError('Unsupported caption schema')
    if caption.get('observation_scope') != 'current_anchor_four_cameras' or caption.get('future_images_included') is not False:
        raise ValueError('Caption is not current-anchor evidence')
    attributes, excluded = [], 0
    for a in caption['attributes']:
        refs = a.get('evidence_ids', [])
        if (a.get('dimension') not in DIMENSIONS or a.get('certainty') not in ('visible', 'uncertain')
                or not refs or not set(refs) <= CAMERAS or not isinstance(a.get('description'), str)):
            raise ValueError('Invalid caption attribute evidence')
        if STATE_TEXT.search(a['description']):
            excluded += 1
            continue
        attributes.append({k: a[k] for k in ('dimension', 'description', 'certainty', 'evidence_ids')})
    uncertainty = [s for s in caption.get('uncertainties', []) if isinstance(s, str) and not STATE_TEXT.search(s)]
    return dict(schema_version=1, source='existing_current_anchor_caption',
        authority='unverified_machine_claims_not_instructions',
        observation_augmentation='offline_doctor_four_whole_cameras; student_native_input_unchanged',
        attributes=attributes, uncertainties=uncertainty,
        excluded_state_attributes=excluded,
        navigation_conflict=normalized_command(caption.get('navigation_command')) != normalized_command(observation['navigation_command']),
        navigation_resolution='use_native_observation; caption_navigation_withheld',
        temporal_interaction_verified=False, factual_review=caption.get('review_status', 'unknown'),
        content_review_flags_present=bool(caption.get('content_review_flags')))


def collect(members, library):
    """Only development members are read; outputs never include training targets."""
    library = Path(library).resolve(); database = library/'navtest/captions.sqlite3'
    contexts, provenance, jobs, receipts = {}, {}, {}, {}
    read = lambda p: json.loads(Path(p).read_text())
    with sqlite3.connect(database.as_uri() + '?mode=ro', uri=True) as db:
        for member in members:
            if member['role'] != 'development': continue
            token = member['token']
            row = db.execute('SELECT caption FROM captions WHERE token=?', (token,)).fetchone()
            if row is None or row[0] is None: raise ValueError('Development caption missing')
            c = json.loads(row[0])
            if c['token'] != token or c['split'] != 'navtest': raise ValueError('Caption identity mismatch')
            path = c['evidence_file']
            if path not in jobs: jobs[path] = {j['token']: j for j in read(path)}
            job = jobs[path][token]
            if job.get('split') != 'navtest' or job['scope'] != 'current_anchor_only' or job['facts']['future_included'] is not False:
                raise ValueError('Caption request scope mismatch')
            if set(job['facts']) != {'scope', 'future_included', 'anchor_speed_kmh', 'navigation_command'}:
                raise ValueError('Undeclared caption request facts')
            receipt = member['cache_receipt']
            if receipt not in receipts: receipts[receipt] = read(receipt)
            record = next(x for x in receipts[receipt]['block']['records'] if x['frame_id'] == token)
            if record['log_name'] != member['source_log']: raise ValueError('Caption source log mismatch')
            evidence = job['evidence']
            if len(evidence) != 4 or {x['id'] for x in evidence} != CAMERAS:
                raise ValueError('Caption camera contract mismatch')
            if any(x['relative_time_s'] != 0 or x['source_timestamp_us'] != record['timestamp_us'] for x in evidence):
                raise ValueError('Caption timestamp mismatch')
            contexts[token] = context(c, member['skill']['evidence'])
            # Identity and source paths remain in private receipts, never Doctor facts.
            provenance[token] = dict(source_log=member['source_log'], token=token,
                database=str(database), evidence_file=path, model=c['model'], review_status=c['review_status'])
    return dict(contexts=contexts, provenance=provenance,
        summary=dict(development_scenes=len(contexts), validation_captions_read=0,
            navigation_conflicts=sum(c['navigation_conflict'] for c in contexts.values()),
            attributes=sum(len(c['attributes']) for c in contexts.values()),
            excluded_state_attributes=sum(c['excluded_state_attributes'] for c in contexts.values()),
            training_targets_read=False, new_caption_requests=0, student_caption_injection=False,
            offline_doctor_observation_augmented=True))
