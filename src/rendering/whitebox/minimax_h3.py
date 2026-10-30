"""MiniMax-H3 Ref2VA prompt profile for existing whitebox videos (no inference)."""
from __future__ import annotations

import json
from pathlib import Path
import re

PROFILE = 'minimax_h3_ref2va_v1'
CAMERAS = {'CAM_F0': 'front-facing', 'CAM_L0': 'left-facing',
           'CAM_R0': 'right-facing', 'CAM_B0': 'rear-facing'}
FIELDS = ('subject_definitions', 'summary', 'retention_analysis',
          'detailed_description', 'overall_soundscape', 'non_diegetic_music')
GUIDE = 'https://github.com/MiniMax-AI/MiniMax-H3/blob/main/skills/h3-prompt-writing/references/ref-en.txt'
REQUEST_EXAMPLE = 'https://github.com/MiniMax-AI/MiniMax-H3/blob/main/scripts/readme/reproducible-768p-ref2va-request.sh'
LICENSE = 'https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE'

# Render palette meanings, not observed body paint or reconstructed materials.
PALETTE_TEXT = (
    'Decode blue #4785E8 as vehicle, pink #F073AE as pedestrian, teal #43C6B5 as the coarse bicycle class, '
    'orange #F29B38 as traffic cone, purple #9C73D8 as barrier, yellow #F2D45A as construction sign, '
    'and gray #A2ADB8 as generic or unknown object. These are semantic codes, not paint colors. '
    'The coarse bicycle class does not determine pedal-cycle versus motorcycle subtype. '
    'Road surfaces use #374355 for lanes, #293440 for drivable area, #303d4b for intersections '
    'and #334344 for parking. Green #295346 indicates a walkway, not grass; #6b5784 indicates '
    'a crosswalk, not purple pavement. #738190 marks map boundaries, #bdc9d7 other map surfaces, '
    'and #dbe4ee unmodeled background, not evidence of sky color. Compression may alter colors slightly.'
)


def fields_for_camera(camera, claims):
    if camera not in CAMERAS:
        raise ValueError('Unknown camera')
    appearance = ' '.join(c['text'] for c in claims) or (
        'No camera-specific appearance claim was verified. Use restrained plausible materials; '
        'do not import distinctive landmarks from another camera.')
    direction = CAMERAS[camera]
    return dict(
        subject_definitions=(
            '<Video 1> is the supplied seven-second ' + direction + ' semantic whitebox video, '
            'providing the complete source timeline and camera projection for this edit.\n'
            '<Subject 1> is the collection of visible traffic actors represented by colored cuboids '
            'in <Video 1>; their spatial arrangement, category and motion follow the source video.\n'
            '<Subject 2> is the visible road layout and static scene structure represented by '
            'the ground polygons and boundaries in <Video 1>.'),
        summary=(
            '[video editing] The target video is an edited version of <Video 1>. '
            'Convert the semantic whitebox appearance into realistic driving-camera footage while '
            'retaining the original seven-second camera path, actor layout and event timing. '
            'This is one continuous ' + direction + ' view, not a montage or a new driving maneuver.'),
        retention_analysis=(
            '<Video 1> (source timeline and camera projection): fully_preserved - preserve its '
            'seven-second real-time ordering, camera-facing direction, framing and projected motion.\n'
            '<Subject 1> (appears in [Shot 1]): partially_preserved - keep every visible actor category, '
            'identity, projected size, location, heading, ordering and occlusion relationship; replace '
            'semantic fills and cuboid surfaces with plausible physical appearance. Unverified paint '
            'and subtype are synthesized choices, not restored ground truth.\n'
            '<Subject 2> (appears in [Shot 1]): partially_preserved - preserve the road, sidewalk and '
            'intersection arrangement while replacing semantic fills with realistic surface textures. '
            'Map edges are geometry proxies, not a license to invent lane paint.'),
        detailed_description=(
            'The target is realistic, neutral driving-camera footage with stable exposure and ordinary '
            'roadside detail. It is a controlled visual translation, not cinematic reinterpretation.\n'
            '[Shot 1] The clip opens with the ' + direction + ' composition of <Video 1>. '
            '<Subject 1> occupies exactly the corresponding image regions from the reference, '
            'and <Subject 2> retains its road layout. The camera is rigidly attached to the vehicle: '
            'follow only the projection changes and motion already visible in the source. Do not add '
            'pan, zoom, orbit, tilt, cuts, stabilization drift, mirrored views or a new viewpoint. '
            + PALETTE_TEXT + '\n'
            'Camera-specific reviewed appearance: ' + appearance + '\n'
            'Treat this prose as guidance for materials, background and illumination only. It does '
            'not override the source geometry or determine which actor is moving. At the opening, '
            'through the middle of the shot and through its final moment, use the same reference '
            'timestamp for actor placement and camera pose. Keep stopped actors stopped and moving '
            'actors on the reference paths. Do not invent overtaking, braking, lane changes or '
            'turns from the scene description. Keep each chosen appearance stable when objects '
            'overlap, become partially hidden and reappear. Do not add, remove, duplicate or teleport '
            'actors, or reveal hidden actors through occluders. Do not create readable license plates, '
            'signage wording, logos or subtitles. Keep static structures stationary in the world. '
            'No flash transitions, flicker, texture swimming, border growth or scene replacement. '
            'End at the same seven-second scene time; do not extend the clip, accelerate its motion '
            'or converge to an invented final frame. Silence continues throughout.'),
        overall_soundscape='Silence. No dialogue, narration, invented horns, engine effects or other added sound.',
        non_diegetic_music='None. No background music.'
    )


def serialize_fields(fields):
    if tuple(fields) != FIELDS or any(not isinstance(v, str) or not v.strip() for v in fields.values()):
        raise ValueError('Six non-empty Ref2VA sections in official order required')
    return '\n\n'.join(name + ':\n' + fields[name] for name in FIELDS)


def validate_request(request):
    """Conservative subset of the upstream local SGLang Ref2VA example."""
    if set(request) != {'task', 'prompt', 'conditions', 'target', 'seed'} or request['task'] != 'ref2va':
        raise ValueError('Unexpected native request keys/task')
    prompt = request['prompt']
    headings = re.findall(r'^([a-z_]+):\s*$', prompt, flags=re.M)
    if tuple(headings) != FIELDS:
        raise ValueError('Incorrect Ref2VA section order')
    if 'summary:\n[video editing] The target video is an edited version of <Video 1>.' not in prompt:
        raise ValueError('Missing official video-editing summary')
    if set(re.findall(r'<(?:Video|Picture|Audio) \d+>', prompt)) != {'<Video 1>'}:
        raise ValueError('Unbound media reference')
    if set(re.findall(r'<Subject \d+>', prompt)) != {'<Subject 1>', '<Subject 2>'}:
        raise ValueError('Unexpected subject reference')
    conditions = request['conditions']
    if len(conditions) != 1 or set(conditions[0]) != {'type', 'uri', 'role'}:
        raise ValueError('One explicitly bound source video required per view')
    if conditions[0]['type'] != 'video' or conditions[0]['role'] != 'reference' or not conditions[0]['uri'].startswith('file:///'):
        raise ValueError('Expected server-readable local reference video URI')
    if request['target'] != {'short_edge': 768, 'aspect_ratio': '16:9', 'duration_seconds': 7}:
        raise ValueError('Preserve native resolution/duration contract; source 10fps is not output fps')
    if not isinstance(request['seed'], int) or isinstance(request['seed'], bool) or request['seed'] < 0:
        raise ValueError('Nonnegative integer seed required')


def scoped_claims(review, camera):
    """A joint evidence list does not attribute each clause to each camera."""
    accepted, withheld = [], []
    for claim in review['claims']:
        evidence_cameras = set()
        for evidence in claim['evidence_ids']:
            match = re.fullmatch(r'(CAM_[FLRB]0)_f[+-]?\d+', evidence)
            if not match:
                raise ValueError('Unrecognized camera/frame appearance evidence ID')
            evidence_cameras.add(match.group(1))
        if evidence_cameras == {camera}:
            accepted.append(claim)
        elif camera in evidence_cameras:
            withheld.append(dict(claim=claim, reason='compound_multicamera_claim_requires_reviewed_split'))
    return accepted, withheld


def build_document(bundle, review_path, token, seed=0):
    from driversi.rendering.whitebox.delivery import read_json, reviewed_prompt
    from driversi.rendering.whitebox.conditioning import PALETTE
    bundle = Path(bundle).resolve()
    reviewed_prompt(review_path, token)  # Reuse the established approval contract.
    source, review = read_json(bundle / 'source.json'), read_json(review_path)
    if source['token'] != token or source['role'] != 'source_log' or source['source_split'] != 'navtrain':
        raise ValueError('Only matched original navtrain conditions supported')
    if source['palette']['colors'] != PALETTE:
        raise ValueError('Prompt palette does not match rendered semantic colors')
    jobs = []
    for camera, direction in CAMERAS.items():
        video = bundle / 'source-renders' / (camera + '.mp4')
        if not video.is_file():
            raise FileNotFoundError(video)
        claims, withheld = scoped_claims(review, camera)
        fields = fields_for_camera(camera, claims)
        if source.get('display_profile') == 'planar_consistent_v1':
            fields['detailed_description'] += ('\nThe reference is an explicitly planar condition: '
                'measured XY paths, headings and timing are retained, while road grade and ego Z/pitch/roll '
                'are not represented. Do not infer metric 3D road heights or overpass layers from this proxy.')
        request = dict(task='ref2va', prompt=serialize_fields(fields),
                       conditions=[dict(type='video', uri=video.as_uri(), role='reference')],
                       target=dict(short_edge=768, aspect_ratio='16:9', duration_seconds=7), seed=seed)
        validate_request(request)
        jobs.append(dict(camera_id=camera, facing=direction,
                         media_binding={'<Video 1>': camera + '.mp4'}, appearance_claims=claims,
                         withheld_appearance_claims=withheld,
                         prompt_fields=fields, request=request))
    return dict(schema='driversi.minimax_h3_whitebox.v1', profile=PROFILE, token=token,
                provenance=dict(prompt_guide=GUIDE, request_example=REQUEST_EXAMPLE),
                source=dict(bundle=str(bundle), role='source_log', split='navtrain',
                            geometry=str(bundle / 'source.json'), maps=str(bundle / 'map.json'),
                            **({'display_profile': source['display_profile']} if 'display_profile' in source else {}),
                            palette=source['palette'], appearance_review=str(Path(review_path).resolve()),
                            width=640, height=360, fps=10, frames=70, duration_seconds=7,
                            video_interval='[0,7)', structured_endpoint_seconds=7),
                target=dict(model='MiniMax-H3', variant='Ref2VA', native_fps=24,
                            short_edge=768, aspect_ratio='16:9', duration_seconds=7),
                timeline=dict(source_timestamps_seconds=[i/10 for i in range(70)],
                              alignment='Use actual decoded output PTS; do not relabel 24fps as 10fps or change playback speed.'),
                multi_view=dict(execution='four_independent_jobs', joint_consistency_verified=False,
                                note='Common source timestamps and seed do not enforce cross-view appearance identity.'),
                use=dict(generation_executed=False, generated_video_admitted=False, training_admitted=False,
                         license_url=LICENSE, restriction='Community license V.3 restricts improving other AI models using H3 outputs; applicable authorization must be established separately.'),
                limitations=['Source 10Hz is offline interpolation from approximately 2Hz annotations.',
                             'Roads and cuboid occlusion are proxies, not measured dense surfaces.',
                             'Prompt adherence, timing and cross-view consistency require generated-output verification.',
                             'Whitebox colors encode category, not actor appearance or subtype.'], jobs=jobs)


def download_text(document):
    introduction = (
        'MiniMax-H3 / Ref2VA — FOUR SEPARATE VIDEO EDITING TASKS\n'
        'Upload only the camera MP4 named for each task as <Video 1>. Copy only that task\'s six-section prompt, '
        'not this whole file. Use target duration 7 seconds, aspect ratio 16:9, native 768 short edge / 24fps. '
        'The source videos remain 7 seconds / 10fps; do not change their playback speed. '
        'Each task produces one view. Four separate outputs are not guaranteed to agree across views.\n'
        'These are original-log conditioning videos, not H3-generated results or approved training data.\n\n')
    return introduction + '\n\n'.join(
        '===== BEGIN ' + j['camera_id'] + ' | input ' + j['camera_id'] + '.mp4 =====\n'
        + j['request']['prompt'] + '\n===== END ' + j['camera_id'] + ' ====='
        for j in document['jobs']) + '\n'


def export_document(bundle, output, token, review_path, seed=0):
    from driversi.rendering.whitebox.delivery import atomic_json, read_json
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    document = build_document(bundle, review_path, token, seed)
    path = output / 'condition.json'
    if path.exists() and read_json(path) != document:
        raise ValueError('Existing H3 condition differs; choose a new output version')
    if not path.exists():
        atomic_json(path, document)
    for job in document['jobs']:
        request_path = output / (job['camera_id'] + '.request.json')
        if request_path.exists() and read_json(request_path) != job['request']:
            raise ValueError('Changed H3 request; choose a new output version')
        if not request_path.exists():
            atomic_json(request_path, job['request'])
    return document, download_text(document)
