"""Private-network, bounded visual caption jobs using the existing v2 contract."""
from __future__ import annotations

import base64
import concurrent.futures
import hashlib
import io
import ipaddress
import json
import os
from pathlib import Path
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from . import scene_description as description
from .store import Catalog, identity
from driversi.data.sources.navsim import load_log
from driversi.rendering.whitebox.conditioning import CAMERAS
from driversi.rendering.whitebox.exporter import sha256

PRIVATE_ROOT = Path(os.environ.get('DRIVERSI_PRIVATE_ROOT',
                                  str(Path.home() / '.config/driversi/private'))).expanduser()

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    tmp.replace(path)


def validate_config(value):
    allowed = {'base_url', 'models', 'concurrency', 'requests_per_minute',
               'max_requests', 'max_attempts', 'timeout_seconds', 'max_tokens',
               'image_max_side', 'jpeg_quality', 'enable_thinking'}
    if set(value) != allowed:
        raise ValueError('Use the documented config keys; credentials are forbidden')
    u = urllib.parse.urlparse(value['base_url'])
    if (u.scheme != 'https' or not u.hostname or u.username or u.password or u.query
            or u.fragment or not u.path.endswith('/compatible-mode/v1')):
        raise ValueError('Explicit private HTTPS endpoint required; no public fallback')
    models = value['models']
    if (not isinstance(models, list) or not 1 <= len(models) <= 8 or
        any(not isinstance(x, str) or not x or len(x) > 150 for x in models) or
        len(set(models)) != len(models)):
        raise ValueError('Explicit unique deployed model IDs required')
    limits = {'concurrency': (1, 8), 'requests_per_minute': (1, 600),
              'max_requests': (1, 10000), 'max_attempts': (1, 3),
              'timeout_seconds': (1, 300), 'max_tokens': (256, 8192),
              'image_max_side': (384, 1280), 'jpeg_quality': (50, 95)}
    for key, (low, high) in limits.items():
        if type(value[key]) is not int or not low <= value[key] <= high:
            raise ValueError('Out-of-bounds configuration: ' + key)
    if type(value['enable_thinking']) is not bool:
        raise ValueError('Explicit thinking mode required')
    return value


def private_endpoint(base_url):
    host = urllib.parse.urlparse(base_url).hostname
    addresses = sorted({x[4][0] for x in socket.getaddrinfo(host, 443)})
    if not addresses or any(not ipaddress.ip_address(x).is_private or
                            ipaddress.ip_address(x).is_loopback for x in addresses):
        raise ValueError('Private endpoint did not resolve to private cluster addresses')
    return addresses


def private_config(path):
    path=Path(path).resolve(strict=True)
    if not path.is_relative_to(PRIVATE_ROOT.resolve()) or path.stat().st_mode & 0o077:
        raise ValueError('Private service configuration must be0600 inside the authorized private API directory')
    return validate_config(json.loads(path.read_text()))


def credential(env_file=None):
    if env_file:
        path = Path(env_file).resolve(strict=True)
        if not path.is_relative_to(PRIVATE_ROOT.resolve()) or path.stat().st_mode & 0o077:
            raise ValueError('Credential file must be private and inside the authorized private API directory')
        values = [x.strip().split('=', 1)[1].strip().strip('\"\'')
                  for x in path.read_text().splitlines()
                  if x.strip().startswith('DASHSCOPE_API_KEY=')]
        if len(values) != 1 or not values[0]:
            raise ValueError('Credential file must define exactly one DASHSCOPE_API_KEY')
        return values[0]
    raise ValueError('Explicit private credential file required; no environment fallback')


def _prompt(scope):
    original = description.prompt_path_v2(scope).read_text()
    original = original.replace('Return at most TWO distinctive_features for this pilot.',
                                'Return up to FOUR non-redundant distinctive_features.')
    return original + ('\nPrioritize concrete road geometry, actor spatial arrangement, '
                       'persistent surroundings and supported temporal differences. '
                       'Do not force a difference that is not visible. '
                       'Copy evidence IDs exactly from image_id, excluding the time field. '
                       'Camera-image left/right is NOT ego-frame left/right, especially in rear views. '
                       'Tie each paint color to the correct photographed object; do not transfer a '
                       'nearby object color. If uncertain omit color and subtype. Never transcribe '
                       'vehicle numbers, signs or brands. Describe sky/illumination directly rather '
                       'than infer weather. A bridge with railings is not necessarily enclosed. '
                       'Use unknown for motion unless matched timestamps in the same camera show it. '
                       'Return JSON only.')


def prepare(database, bundles, config, output):
    """Prepare only existing verified navtrain source bundles; no API calls."""
    from PIL import Image
    validate_config(config)
    if not 1 <= len(bundles) <= 128 or len(set(map(str, bundles))) != len(bundles):
        raise ValueError('Use shards of 1..128 unique existing bundles')
    output = Path(output)
    jobs, tokens = [], set()
    cat = Catalog(database, 'navtrain', readonly=True)
    try:
        for bundle_path in bundles:
            bundle_path = Path(bundle_path).resolve()
            source_file = bundle_path / 'source.json'
            source = json.loads(source_file.read_text())
            if source['source_split'] != 'navtrain' or source['role'] != 'source_log':
                raise ValueError('This source-label batch accepts original navtrain only')
            token = source['token']
            if token in tokens:
                raise ValueError('Duplicate source token')
            tokens.add(token)
            record = cat.scene(token)['record']
            if sha256(record['log_path']) != source['source_log_sha256']:
                raise ValueError('Bundle and catalog source log differ')
            raw = load_log(Path(record['log_path']), trusted=True)
            a = record['anchor_index']
            if raw[a]['token'] != token or int(raw[a]['timestamp']) != source['anchor_timestamp_us']:
                raise ValueError('Raw anchor identity/time mismatch')
            for scope, indices in [('retrieval_history', [a-3, a-2, a-1, a]),
                                   ('video_clip', [a-4, a, a+5, a+10])]:
                if min(indices) < 0 or max(indices) >= len(raw):
                    raise ValueError('Missing source history/future; no extrapolation')
                facts = description.scoped_measurements(source, scope)
                context = {k: facts[k] for k in ('scope', 'duration_seconds', 'future_included',
                    'nearby_class_counts_30m', 'ego_path_length_xy_m', 'ego_heading_change_degrees',
                    'timeline', 'motion_semantics', 'class_colors', 'uncertainty', 'map_layer_counts')}
                context['nearest_annotated_objects'] = facts['anchor_objects'][:12]
                evidence = []
                for j in indices:
                    row = raw[j]
                    relative = (int(row['timestamp']) - source['anchor_timestamp_us']) / 1e6
                    if scope == 'retrieval_history' and relative > 0:
                        raise ValueError('Future image in retrieval')
                    for cam in CAMERAS:
                        path = Path(record['sensor_root']) / row['cams'][cam]['data_path']
                        with Image.open(path) as image:
                            image = image.convert('RGB')
                            image.thumbnail((config['image_max_side'], config['image_max_side']))
                            stream = io.BytesIO()
                            image.save(stream, format='JPEG', quality=config['jpeg_quality'])
                        data = stream.getvalue()
                        digest = hashlib.sha256(data).hexdigest()
                        target = output / 'images' / (digest + '.jpg')
                        target.parent.mkdir(parents=True, exist_ok=True)
                        if not target.exists():
                            target.write_bytes(data)
                        elif sha256(target) != digest:
                            raise ValueError('Changed compressed evidence')
                        evidence.append({'id': f'{cam}_f{j-a:+d}', 'camera': cam,
                            'relative_time_s': relative, 'source_file': str(path),
                            'source_sha256': sha256(path), 'file': str(target.resolve()),
                            'sha256': digest, 'bytes': len(data), 'source_bytes': path.stat().st_size})
                profile = {'scope': scope, 'prompt': _prompt(scope), 'facts': context,
                    'evidence': evidence, 'source_sha256': sha256(source_file),
                    'compiler_sha256': sha256(description.__file__),
                    'image_max_side': config['image_max_side'], 'jpeg_quality': config['jpeg_quality']}
                job = {'token': token, 'scope': scope, 'bundle': str(source_file),
                       'input_identity': identity(profile), **profile}
                write(output / 'prepared' / f'{token}-{scope}.json', job)
                jobs.append(job)
    finally:
        cat.close()
    write(output / 'preparation.json', {'status': 'prepared_not_called',
          'scenes': len(tokens), 'scopes': len(jobs), 'model_count': len(config['models']),
          'expected_successful_requests': len(jobs) * len(config['models']),
          'jpeg_bytes_per_model': sum(x['bytes'] for j in jobs for x in j['evidence']),
          'original_bytes_per_model': sum(x['source_bytes'] for j in jobs for x in j['evidence']),
          'payload': 'compressed source photos plus whitebox measured context; no future in retrieval',
          'trainable': False})
    return jobs


def payload(job, model, config):
    content = [{'type': 'text', 'text': job['prompt'] + '\nMeasured context:\n' + json.dumps(job['facts'])}]
    for row in job['evidence']:
        if sha256(row['file']) != row['sha256']:
            raise ValueError('Evidence changed after preparation')
        content += [{'type': 'text', 'text': json.dumps({'image_id': row['id'], 'time_relative_s': row['relative_time_s']})},
                    {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' +
                        base64.b64encode(Path(row['file']).read_bytes()).decode()}}]
    content.append({'type':'text', 'text':'Allowed evidence_ids (copy these strings exactly): '+json.dumps([x['id'] for x in job['evidence']])+'. Return the fixed JSON only.'})
    return {'model': model, 'messages': [{'role': 'user', 'content': content}],
            'response_format': {'type': 'json_object'}, 'temperature': 0,
            'max_tokens': config['max_tokens'], 'enable_thinking': config['enable_thinking'],
            'stream': False}


class APIError(Exception):
    def __init__(self, status, retryable=False):
        self.status, self.retryable = status, retryable
        super().__init__('Private provider HTTP ' + str(status))


class NotSent(APIError):
    """A local scheduler refused BEFORE transport; safe to resume, no API outcome."""


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise APIError('redirect_rejected')


def request(body, key, config):
    # No proxy/public fallback; credentials are only in the Authorization header.
    private_endpoint(config['base_url'])
    data = json.dumps(body).encode()
    if len(data) > 20 * 1024 * 1024:
        raise ValueError('Bounded caption request exceeds20MiB')
    req = urllib.request.Request(config['base_url'] + '/chat/completions', data=data,
          headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    try:
        with opener.open(req, timeout=config['timeout_seconds']) as response:
            raw = response.read(4 * 1024 * 1024 + 1)
            if len(raw) > 4 * 1024 * 1024:
                raise APIError('oversize_response')
            return json.loads(raw)
    except urllib.error.HTTPError as exc:
        # Error bodies may echo request credentials or media. Never persist them.
        raise APIError(exc.code, exc.code == 429) from None
    except (urllib.error.URLError, TimeoutError, OSError):
        raise APIError('network_outcome_unknown_no_automatic_retry') from None


def result_from_response(job, model, response):
    choice = response['choices'][0]
    raw = choice['message']['content']
    if choice.get('finish_reason') != 'stop' or not isinstance(raw, str):
        raise ValueError('Incomplete response; do not release truncated observations')
    observed, repair = description.parse_generation(raw, [x['id'] for x in job['evidence']])
    if sha256(job['bundle']) != job['source_sha256']:
        raise ValueError('Source bundle changed during request')
    bundle = json.loads(Path(job['bundle']).read_text())
    caption = description.build_description(bundle, job['scope'], observations=observed,
                                           evidence=job['evidence'])
    claims = [observed[k] for k in description.VISUAL_FIELDS] + observed['distinctive_features']
    prose = ' '.join(x['text'] for x in claims if x['text'].lower() != 'unknown')
    draft = {'factual_status': 'unreviewed_candidate_not_ground_truth',
             'retrieval_text': caption['scene_overview']['structured'] + ' ' + prose
                              if job['scope'] == 'retrieval_history' else None,
             'video_positive_prompt': caption['video_generation']['positive_prompt'] +
                 '\nUnreviewed appearance hypotheses: ' + prose if job['scope'] == 'video_clip' else None}
    return {'status': 'format_valid_unreviewed', 'model': model,
            'returned_model': response.get('model'), 'request_id': response.get('id'),
            'usage': response.get('usage'), 'raw': raw, 'format_repair': repair,
            'caption': caption, 'draft_prompts': draft, 'trainable': False}


class Budget:
    def __init__(self, config, used):
        self.config, self.used, self.next = config, used, 0.
        self.lock = threading.Lock()

    def take(self):
        with self.lock:
            if self.used >= self.config['max_requests']:
                return False
            self.used += 1
            now = time.monotonic()
            start = max(now, self.next)
            self.next = start + 60 / self.config['requests_per_minute']
        time.sleep(max(0, start - now))
        return True


def run(jobs, config, output, *, key, transport=request, decode=result_from_response):
    """Single runner, bounded threads, durable attempts; completed/failed jobs are reused."""
    import fcntl
    validate_config(config)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'batch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for old_path in (output / 'results').glob('*.json'):
            if 'request_identity' not in json.loads(old_path.read_text()):
                raise ValueError('Legacy adapter-hash batch is read-only; retain it and use an explicit new batch for changed requests')
        for old_path in (output / 'attempts').glob('*.json'):
            if json.loads(old_path.read_text()).get('request_identity')!=old_path.stem.rsplit('-',1)[0]:
                raise ValueError('Legacy attempt-only batch is read-only; reconcile before any new request')
        budget = Budget(config, sum(json.loads(p.read_text()).get('status') != 'not_sent'
                                    for p in (output / 'attempts').glob('*.json')))
        def one(job, model):
            body = payload(job, model, config)
            signature = identity({'base_url':config['base_url'], 'payload':body})
            profile = {'input': job['input_identity'], 'model': model, 'endpoint_identity': identity(config['base_url']),
                       'max_tokens': config['max_tokens'], 'enable_thinking': config['enable_thinking'],
                       'adapter': sha256(__file__)}
            path = output / 'results' / (signature + '.json')
            if path.exists():
                previous = json.loads(path.read_text())
                if previous['status']!='not_sent' and previous['input_identity']==job['input_identity'] and previous['token']==job['token'] and previous['scope']==job['scope']:
                    return {'file': str(path), 'status': previous['status'], 'reused': True,
                            'model': model, 'scope': job['scope'], 'token': job['token']}
            prior = sorted((output / 'attempts').glob(signature + '-*.json'),
                           key=lambda p:int(p.stem.rsplit('-',1)[1]))
            def finalize(result, elapsed):
                result.update(profile=profile, token=job['token'], scope=job['scope'],
                              elapsed_seconds=elapsed, input_identity=job['input_identity'],
                              request_identity=signature)
                write(path,result)
                return {'file':str(path),'status':result['status'],'reused':False,
                        'model':model,'scope':job['scope'],'token':job['token']}
            if prior:
                receipt = json.loads(prior[-1].read_text())
                if receipt['status']=='response_received':
                    try:
                        result=decode(job,model,receipt['response'])
                    except Exception as exc:
                        result={'status':'rejected','error':type(exc).__name__+': '+str(exc).replace(key,'[REDACTED]')[:500],
                                'usage':receipt['response'].get('usage'),'trainable':False}
                    row=finalize(result,receipt.get('request_elapsed_seconds'))
                    row.update(reused=True,recovered_received_response=True)
                    return row
                if receipt['status']!='not_sent':
                    return {'status': 'uncertain_previous_attempt_requires_review', 'reused': False,
                            'model': model, 'scope': job['scope'], 'token': job['token']}
            started = time.monotonic()
            result = None
            used_attempts = sum(json.loads(p.read_text()).get('status') != 'not_sent' for p in prior)
            remaining_attempts = max(0, config['max_attempts'] - used_attempts)
            if not remaining_attempts:
                return {'status': 'attempt_budget_exhausted', 'model': model,
                        'scope': job['scope'], 'token': job['token']}
            for attempt in range(remaining_attempts):
                if not budget.take():
                    return {'status': 'budget_exhausted', 'model': model, 'scope': job['scope'], 'token': job['token']}
                prior_numbers=[int(p.stem.rsplit('-',1)[1]) for p in prior]
                attempt_number=(max(prior_numbers)+1 if prior_numbers else 0)+attempt
                attempt_path = output / 'attempts' / (signature + f'-{attempt_number}.json')
                receipt = {'status': 'inflight', 'model': model, 'attempt': attempt_number,
                           'input_identity': job['input_identity'], 'request_identity': signature}
                write(attempt_path, receipt)
                try:
                    response = transport(body, key, config)
                    # Defensively redact any credential echo before persisting raw model text.
                    response = json.loads(json.dumps(response).replace(key, '[REDACTED]'))
                    receipt.update(status='response_received', response=response,
                                   request_elapsed_seconds=time.monotonic()-started)
                    write(attempt_path, receipt)
                    result = decode(job, model, response)
                    break
                except NotSent as exc:
                    receipt.update(status='not_sent', error=str(exc))
                    write(attempt_path, receipt)
                    result = {'status': 'not_sent', 'error': str(exc), 'trainable': False}
                    break
                except APIError as exc:
                    receipt.update(status='api_failed', error=str(exc))
                    write(attempt_path, receipt)
                    result = {'status': 'api_failed', 'error': str(exc), 'trainable': False}
                    if exc.retryable and attempt + 1 < remaining_attempts:
                        time.sleep(min(30, 2 ** attempt))
                        continue
                    break
                except Exception as exc:
                    result = {'status': 'rejected', 'error': type(exc).__name__ + ': ' + str(exc).replace(key, '[REDACTED]')[:500], 'trainable': False}
                    if 'response' in receipt:result['usage']=receipt['response'].get('usage')
                    break
            return finalize(result,time.monotonic()-started)
        rows = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=config['concurrency']) as pool:
            futures = [pool.submit(one, job, model) for job in jobs for model in config['models']]
            for future in concurrent.futures.as_completed(futures):
                rows.append(future.result())
                write(output / 'report.json', {'complete': False, 'rows': rows,
                      'lifetime_requests_started': budget.used, 'trainable': False})
        report = {'complete': all(x['status'] == 'format_valid_unreviewed' for x in rows),
                  'rows': rows, 'lifetime_requests_started': budget.used,
                  'format_valid': sum(x['status'] == 'format_valid_unreviewed' for x in rows),
                  'reused': sum(x.get('reused', False) for x in rows),
                  'factual_quality': 'not_yet_reviewed', 'trainable': False}
        write(output / 'report.json', report)
        return report
