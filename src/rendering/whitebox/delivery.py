"""Verified, resumable five-file whitebox deliveries; no VLM/network calls."""
from __future__ import annotations

import json
from pathlib import Path
import uuid
import zipfile

from driversi.rendering.whitebox.conditioning import CAMERAS, DIMENSION_SAMPLING
from driversi.rendering.whitebox.exporter import sha256
from driversi.rendering.whitebox.delivery_prompt import generation_prompt

MEMBERS = tuple(c + '.mp4' for c in CAMERAS) + ('video-generation-prompt.txt',)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def stage_finished(path, required):
    """Legacy producers write manifests directly; a killed write is incomplete."""
    try:
        value = read_json(path)
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
        return False
    return isinstance(value, dict) and set(required).issubset(value)


def verify_bundle(bundle, token):
    """Reuse only complete exports still matching their existing manifest."""
    bundle = Path(bundle)
    manifest = read_json(bundle / 'manifest.json')
    for name, digest in manifest['files'].items():
        if Path(name).name != name or sha256(bundle / name) != digest:
            raise ValueError('Changed bundle file: ' + name)
    source = read_json(bundle / 'source.json')
    if (source['token'] != token or source['source_split'] != 'navtrain'
            or source['role'] != 'source_log' or len(source['frames']) != 71
            or [f['tick'] for f in source['frames']] != list(range(71))
            or source['timing']['interval_seconds'] != .1
            or source['timing']['duration_seconds'] != 7.0):
        raise ValueError('Requires matching original navtrain token and 71 states at 0.1s')
    for frame in source['frames']:
        if {c['camera_id'] for c in frame['cameras']} != set(CAMERAS):
            raise ValueError('Four calibrated cameras required at every timestamp')
        if any((c['width'], c['height']) != (640, 360) for c in frame['cameras']):
            raise ValueError('Current delivery contract requires 640x360 cameras')
    return source


def verify_renders(bundle):
    """Check every frame receipt and actually decode all four delivered videos."""
    import imageio_ffmpeg
    bundle = Path(bundle)
    root = bundle / 'source-renders'
    manifest = read_json(root / 'manifest.json')
    if (manifest['role'] != 'source' or manifest['frame_count'] != 71
            or manifest['video_frame_count'] != 70 or manifest['fps'] != 10):
        raise ValueError('Wrong render interval or role')
    for field, name in [('source_sha256', 'source.json'), ('map_sha256', 'map.json'), ('palette_sha256', 'palette.json')]:
        if manifest[field] != sha256(bundle / name):
            raise ValueError('Stale render conditions: ' + name)
    source = read_json(bundle / 'source.json')
    if source.get('display_profile') and manifest.get('projection', {}).get('profile') != source['display_profile']:
        raise ValueError('Render projection differs from declared delivery profile')
    if (manifest.get('dimension_sampling', 'previous_annotation_hold_v1')
            != source.get('dimension_sampling', 'previous_annotation_hold_v1')):
        raise ValueError('Render box-size sampling differs from source conditions')
    rows = manifest['images']
    if len(rows) != 284 or {(r['camera'], r['tick']) for r in rows} != {(c, t) for c in CAMERAS for t in range(71)}:
        raise ValueError('Missing/duplicate render frame receipts')
    for row in rows:
        name = f"{row['tick']:04d}-{row['camera']}.png"
        if row['file'] != name or sha256(root / name) != row['sha256']:
            raise ValueError('Changed rendered frame: ' + name)
    videos = {}
    for cam in CAMERAS:
        path = root / (cam + '.mp4')
        reader = imageio_ffmpeg.read_frames(str(path), pix_fmt='rgb24')
        try:
            metadata = next(reader)
            count = sum(1 for _ in reader)
        finally:
            reader.close()
        if (count != 70 or abs(metadata['fps'] - 10) > 1e-6
                or abs(metadata['duration'] - 7) > .01 or tuple(metadata['size']) != (640, 360)):
            raise ValueError('Wrong MP4 dimensions, duration or frame count: ' + cam)
        videos[cam + '.mp4'] = sha256(path)
    return videos


def reviewed_prompt(review_path, token):
    review = read_json(review_path)
    if review.get('token') != token:
        raise ValueError('Appearance review must explicitly identify this token')
    if review.get('status') not in ('assistant_reviewed_subset_for_generation_trial', 'human_reviewed_for_generation_trial'):
        raise ValueError('Unreviewed VLM text cannot become a delivered appearance claim')
    claims = review.get('claims', [])
    if not claims or any(not isinstance(c.get('text'), str) or not c['text'].strip() or not c.get('evidence_ids') for c in claims):
        raise ValueError('Reviewed appearance claims need text and source-image evidence IDs')
    return generation_prompt(' '.join(c['text'].strip() for c in claims))


def package_delivery(bundle, output, token, review_path, *, prompt_text=None):
    """Called after motion/render validation; compare archive content on resume."""
    bundle, output = Path(bundle), Path(output)
    default_prompt = reviewed_prompt(review_path, token)
    prompt = default_prompt if prompt_text is None else prompt_text
    source = read_json(bundle / 'source.json')
    if source.get('display_profile') == 'planar_consistent_v1':
        prompt += ('\nDISPLAY GEOMETRY: This condition is a consistent planar map projection. '
                   'Actor XY motion, headings, identities and timing follow the log; road grade and ego '
                   'vertical/pitch/roll motion are not represented. Do not treat it as metric 3D, '
                   'a reconstruction of ramps/overpasses, or photometric ground truth.\n')
    prompt = prompt.encode('utf-8')
    output.mkdir(parents=True, exist_ok=True)
    archive = output / 'whitebox-7s10hz-with-prompts.zip'
    inputs = {c + '.mp4': bundle / 'source-renders' / (c + '.mp4') for c in CAMERAS}

    def matches(path):
        with zipfile.ZipFile(path) as z:
            return (sorted(z.namelist()) == sorted(MEMBERS) and z.testzip() is None
                    and z.read(MEMBERS[-1]) == prompt
                    and all(z.read(n) == p.read_bytes() for n, p in inputs.items()))

    reused = archive.exists()
    if reused:
        if not matches(archive):
            raise ValueError('Existing delivery differs; use a new output version')
    else:
        temporary = archive.with_name(archive.name + '.' + uuid.uuid4().hex + '.partial')
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_STORED) as z:
            for name, path in inputs.items():
                z.write(path, name)
            z.writestr(MEMBERS[-1], prompt)
        if not matches(temporary):
            raise ValueError('Archive verification failed')
        temporary.replace(archive)
    text = output / MEMBERS[-1]
    if not text.exists() or text.read_bytes() != prompt:
        temporary = text.with_suffix('.txt.partial')
        temporary.write_bytes(prompt)
        temporary.replace(text)
    return dict(status='reused' if reused else 'completed', token=token,
                archive=str(archive), archive_sha256=sha256(archive), bytes=archive.stat().st_size,
                members=list(MEMBERS), new_api_calls=0,
                display_profile=source.get('display_profile', 'legacy_mixed_v1'),
                geometry_convention=source.get('geometry_convention', 'legacy_yaw_only_v1'),
                dimension_sampling=source.get('dimension_sampling', 'previous_annotation_hold_v1'),
                coordinate_repair_present=(source.get('geometry_convention') == 'openscene_center_se3_scalar_yaw_v2'
                    and source.get('camera_convention') == 'openscene_camera_to_ego_v2'
                    and source.get('display_profile') == 'planar_consistent_v1'))


def run_scene(config, row):
    """Scene-stage checkpoints. Incomplete attempts are preserved, never deleted."""
    from driversi.data.catalog.store import Catalog
    from driversi.rendering.whitebox.conditioning import export_bundle
    from driversi.rendering.whitebox.motion_audit import audit_source_motion

    token = row['token']
    output = Path(config['output']) / token
    output.mkdir(parents=True, exist_ok=True)
    # POSIX flock releases after process death, unlike stale PID lock files.
    import fcntl
    with (output / '.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return dict(token=token, status='busy', reason='Another worker owns this token')
        result = dict(token=token, rendered=False, exported=False, new_api_calls=0)
        try:
            bundle = Path(row.get('bundle', output / 'bundle'))
            external = 'bundle' in row
            with_catalog = Catalog(config['database'], 'navtrain', readonly=True)
            try:
                record = with_catalog.scene(token)['record']
            finally:
                with_catalog.close()
            if not stage_finished(bundle / 'manifest.json', ('files', 'producer_sha256', 'frame_count')):
                if external:
                    raise ValueError('External reusable bundle is incomplete')
                if bundle.exists():
                    bundle.rename(bundle.with_name('bundle.incomplete-' + uuid.uuid4().hex))
                export_bundle(record, config['maps_root'], bundle)
                result['exported'] = True
            source = verify_bundle(bundle, token)
            result['geometry_convention'] = source.get('geometry_convention', 'legacy_yaw_only_v1')
            result['display_profile'] = source.get('display_profile', 'legacy_mixed_v1')
            result['coordinate_repair_present'] = (
                source.get('geometry_convention') == 'openscene_center_se3_scalar_yaw_v2'
                and source.get('camera_convention') == 'openscene_camera_to_ego_v2'
                and source.get('display_profile') == 'planar_consistent_v1')
            if not result['coordinate_repair_present']:
                raise ValueError('Legacy coordinate bundle requires separate re-export; original preserved')
            result['dimension_sampling'] = source.get('dimension_sampling', 'previous_annotation_hold_v1')
            if result['dimension_sampling'] != DIMENSION_SAMPLING:
                raise ValueError('Legacy stepped box dimensions require separate re-export; original preserved')
            if (Path(source['source_log_file']).resolve() != Path(record['log_path']).resolve()
                    or source['source_log_sha256'] != sha256(record['log_path'])):
                raise ValueError('Source differs from current navtrain catalog/log')
            audit = audit_source_motion(source)
            atomic_json(output / 'motion-audit.json', audit)
            if not audit['passed']:
                raise ValueError('Raw-log motion audit failed')
            result['actor_states_checked'] = audit['actor_states_checked']
            # Missing review is a resumable wait, not a reason to spend render time.
            review = Path(row.get('review', Path(config['reviews_root']) / (token + '.json')))
            if not review.exists():
                result.update(status='waiting_for_appearance_review', review=str(review))
            else:
                reviewed_prompt(review, token)
                renders = bundle / 'source-renders'
                if not stage_finished(renders / 'manifest.json', ('images', 'source_sha256', 'map_sha256', 'palette_sha256')):
                    if external:
                        raise ValueError('External reusable renders are incomplete')
                    if renders.exists():
                        renders.rename(renders.with_name('source-renders.incomplete-' + uuid.uuid4().hex))
                    from driversi.rendering.whitebox.render_conditions import render
                    render(bundle)
                    result['rendered'] = True
                result['video_sha256'] = verify_renders(bundle)
                prompt_text = None
                profile = config.get('prompt_profile', 'generic_four_camera_v1')
                if profile == 'minimax_h3_ref2va_v1':
                    from driversi.rendering.whitebox.minimax_h3 import export_document
                    document, prompt_text = export_document(bundle, output / 'h3', token, review,
                                                            config.get('seed', 0))
                    result['h3_condition'] = str(output / 'h3/condition.json')
                    result['native_request_count'] = len(document['jobs'])
                elif profile != 'generic_four_camera_v1':
                    raise ValueError('Unknown prompt profile')
                result.update(package_delivery(bundle, output / 'download', token, review,
                                               prompt_text=prompt_text))
                result['prompt_profile'] = profile
                result['bundle'] = str(bundle)
        except Exception as exc:
            result.update(status='failed', reason=type(exc).__name__ + ': ' + str(exc))
        atomic_json(output / 'receipt.json', result)
        with (output / 'attempts.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(result, ensure_ascii=False) + '\n')
        return result
