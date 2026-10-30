"""Cached current-source visual advice. No authority over safety or release gates."""
import io
import json
import time
from pathlib import Path

from driversi.agents.backends.native import digest
from driversi.configuration import read_mapping
from driversi.rsi.experiments.continuation import frozen_json
from driversi.runtime.budget import write_json

MODEL='deepseek-v4.1-flash-1017'
PROMPT='''Inspect the four simultaneous current images of a training driving scene.
Suggest which bounded data-generation variation deserves priority. These are
single frames: never infer actor motion, intention, yield/cut-in events or exact
metric free space. Image left/right is camera-local, especially for the rear.
Measured source speed and history curvature are supplied separately. You cannot
change safety scoring, admission, speed limits or PDMS. Return JSON only with
exact keys: preference (spatial|speed|mixed|conservative), observations (1..4
objects each with camera and short observation_zh), uncertainty_zh (string).
Choose conservative if evidence is unclear. The controller will independently
measure trajectories and bound all parameters. Do not output numerical thresholds.'''


def validate_advice(value):
    if set(value)!={'preference','observations','uncertainty_zh'}:
        raise ValueError('Unexpected advisory keys')
    if value['preference'] not in ('spatial','speed','mixed','conservative'):
        raise ValueError('Unknown preference')
    if not isinstance(value['uncertainty_zh'],str) or len(value['uncertainty_zh'])>800:
        raise ValueError('Bounded uncertainty required')
    if not isinstance(value['observations'],list) or not 1<=len(value['observations'])<=4:
        raise ValueError('Bounded observations required')
    for row in value['observations']:
        if set(row)!={'camera','observation_zh'} or row['camera'] not in ('CAM_F0','CAM_L0','CAM_R0','CAM_B0'):
            raise ValueError('Exact current camera evidence required')
        if not isinstance(row['observation_zh'],str) or not 1<=len(row['observation_zh'])<=600:
            raise ValueError('Bounded visual observation required')
    return value


def advise(context, environment):
    from PIL import Image
    from driversi.data.catalog import bailian_caption as api
    from driversi.data.catalog.store import Catalog,identity
    from driversi.data.sources.navsim import load_log
    facts=environment['facts'];root=Path(context['advice_root'])
    root.mkdir(parents=True,exist_ok=True);folder=root/facts['token'];folder.mkdir(exist_ok=True)
    result=folder/'result.json'
    if result.exists():
        value=read_mapping(result)
        if value['source_sha256']!=facts['source_sha256'] or value['model']!=MODEL:
            raise ValueError('Cached advice belongs to different source/model')
        job=read_mapping(folder/'job.json')
        if job['prompt']!=PROMPT:raise ValueError('Cached advisory prompt changed; do not silently replay or repay')
        for path,sha in value.get('evidence',{}).items():
            if digest(path)!=sha:raise ValueError('Cached advisory evidence changed')
        return value
    common=dict(model=MODEL,source_sha256=facts['source_sha256'],preference='conservative',
                safety_authority=False,indexable=False,trainable=False)
    if (folder/'request-start.json').exists():
        return dict(common,status='unknown_or_unparsed_prior_request_no_retry')
    if len(list(root.glob('*/request-start.json')))>=3:
        return dict(common,status='advice_budget_exhausted_rules_only')
    config=api.private_config(context['private_config'])
    if config['models']!=[MODEL]:raise ValueError('Only confirmed1017 configuration allowed')
    config=dict(config,max_tokens=700,enable_thinking=False,timeout_seconds=90)
    cat=Catalog(context['database'],'navtrain',readonly=True)
    try:record=cat.scene(facts['token'])['record']
    finally:cat.close()
    if digest(record['log_path'])!=facts['source_sha256']:
        raise ValueError('Catalog and measured source differ')
    frames=load_log(Path(record['log_path']),trusted=True);frame=frames[record['anchor_index']]
    if frame['token']!=facts['token'] or int(frame['timestamp'])!=facts['anchor_timestamp_us']:
        raise ValueError('Current source anchor mismatch')
    evidence=[]
    for camera in ('CAM_F0','CAM_L0','CAM_R0','CAM_B0'):
        source=Path(record['sensor_root'])/frame['cams'][camera]['data_path']
        with Image.open(source) as image:
            image=image.convert('RGB');image.thumbnail((768,768))
            buffer=io.BytesIO();image.save(buffer,format='JPEG',quality=85)
        target=folder/(camera+'.jpg');target.write_bytes(buffer.getvalue())
        evidence.append(dict(id=camera,camera=camera,relative_time_s=0.,file=str(target),
            sha256=digest(target),source_file=str(source),source_sha256=digest(source)))
    supplied={k:facts[k] for k in ('anchor_speed_mps','observed_curvature_inv_m','forward_box_gap_m','objects_observed')}
    job=dict(prompt=PROMPT,facts=supplied,evidence=evidence)
    frozen_json(folder/'job.json',job)
    # All local validation precedes a write-ahead request record. No credentials
    # enter the receipt or exception; the existing private transport owns them.
    body=api.payload(job,MODEL,config);key=api.credential(context['credential_file'])
    api.private_endpoint(config['base_url']);started=time.time()
    frozen_json(folder/'request-start.json',dict(model=MODEL,job=identity(job),started_at=started))
    try:
        response=api.request(body,key,config)
        choice=response['choices'][0]
        frozen_json(folder/'response.json',dict(content=choice['message']['content'],
            finish_reason=choice.get('finish_reason'),usage=response.get('usage',{})))
        if choice.get('finish_reason')!='stop':raise ValueError('Incomplete advice')
        advice=validate_advice(json.loads(choice['message']['content']))
        value=dict({**common,**advice},status='advisory_received',
            usage=response.get('usage',{}),elapsed_seconds=time.time()-started)
    except Exception as exc:
        value=dict(common,status='advice_failed_rules_only',error_type=type(exc).__name__,
                   elapsed_seconds=time.time()-started)
    finally:del key
    value['evidence']={str(p):digest(p) for p in (folder/'job.json',folder/'request-start.json')}
    if (folder/'response.json').exists():value['evidence'][str(folder/'response.json')]=digest(folder/'response.json')
    write_json(result,value)
    return value
