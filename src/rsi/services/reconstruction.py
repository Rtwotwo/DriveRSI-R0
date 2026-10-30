"""Translate a frozen retrieved training source into an owned MTGS recovery job."""
import json
import pickle
import tarfile
from pathlib import Path

from driversi.agents.backends.native import digest, project_root
from driversi.configuration import read_mapping
from driversi.rsi.experiments.continuation import frozen_json


def prepare_recovery(candidate, factory, output, gpu):
    from driversi.data.assets.reconstruction_preparation import prepare
    import yaml
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=True)
    if not root.is_relative_to(project_root()/'exp'):
        raise ValueError('Owned per-source recovery directory required')
    recipe=root/'recovery-job.json'
    frozen_json(root/'factory-binding.json',dict(candidate=candidate,factory=factory,gpu=gpu))
    if recipe.exists():return str(recipe)
    name,token=candidate['source_log'],candidate['token']
    if Path(name).name!=name or not name or len(token)!=16:
        raise ValueError('Explicit retrieved source identity required')
    databases=[Path(base)/(name+'.db') for base in factory['db_roots']]
    databases=[p for p in databases if p.is_file()]
    if len(databases)!=1:raise ValueError('One exact raw training database required')
    prep_config=dict(train_index=factory['train_index'],candidates=[dict(token=token,source_log=name)],
        logs_root=factory['logs_root'],db_root=str(databases[0].parent),sensor_roots=factory['sensor_roots'])
    frozen_json(root/'preparation-config.json',prep_config)
    report=prepare(root/'preparation-config.json',root/'preparation')
    row=report['candidates'][0]
    if not row['source_window_complete']:return None
    original_config=Path(row['config'])
    values=yaml.safe_load(original_config.read_text().split('\n',1)[1])
    values['data_root']=str(root/'processed')
    values['collect_raw']=False
    text='!!python/object:nuplan_scripts.utils.config.FrameCentralConfig\n'+yaml.safe_dump(values,sort_keys=False)
    native_config=root/'mtgs.yaml'
    if native_config.exists() and native_config.read_text()!=text:
        raise ValueError('Reconstruction config changed')
    native_config.write_text(text)
    sensors=root/'preparation'/token/'sensors.json'
    unified=root/'sensors';unified.mkdir(exist_ok=True)
    for item in read_mapping(sensors)['rows']:
        relative=Path(item['relative_path']);source=Path(item['source_path']).resolve(strict=True)
        if relative.is_absolute() or '..' in relative.parts or relative.parts[0]!=name:
            raise ValueError('Sensor must belong to this training source')
        destination=unified/relative;destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists() and destination.resolve()!=source:
            raise ValueError('Conflicting measured sensor binding')
        if not destination.exists():destination.symlink_to(source)
    # Native preprocessing can refer to nearby camera timestamps; expose only
    # source-owned camera directories, without copying or altering raw captures.
    for camera in factory['sensor_roots']['camera']:
        source_log=Path(camera)/name
        for folder in source_log.glob('CAM_*'):
            for source in folder.glob('*.jpg'):
                dst=unified/name/folder.name/source.name
                if not dst.exists():dst.symlink_to(source.resolve())
    processed=Path(values['data_root']);processed.mkdir(exist_ok=True)
    (processed/values['road_block_name']/'map_vis').mkdir(parents=True,exist_ok=True)
    archive=Path(factory['ego_masks_archive']).resolve(strict=True)
    with tarfile.open(archive) as stream:
        for member in stream.getmembers():
            dst=(processed/member.name).resolve()
            if not dst.is_relative_to(processed) or member.issym() or member.islnk():
                raise ValueError('Unsafe public mask archive member')
            if member.isdir():dst.mkdir(parents=True,exist_ok=True)
            elif member.isfile():
                dst.parent.mkdir(parents=True,exist_ok=True)
                value=stream.extractfile(member).read()
                if dst.exists() and dst.read_bytes()!=value:raise ValueError('Mask archive changed')
                dst.write_bytes(value)
    work=root/'work';work.mkdir(exist_ok=True)
    aliases={'facebook/mask2former-swin-large-cityscapes-semantic':factory['semantic_model'],
             'lpiccinelli/unidepth-v2-vitl14':factory['depth_model']}
    for alias,destination in aliases.items():
        link=work/'ckpts/huggingface'/alias;link.parent.mkdir(parents=True,exist_ok=True)
        if not link.exists():link.symlink_to(Path(destination).resolve(),target_is_directory=True)
    with (Path(factory['logs_root'])/(name+'.pkl')).open('rb') as stream:frames=pickle.load(stream)
    anchor=[i for i,f in enumerate(frames) if f['token']==token]
    if len(anchor)!=1:raise ValueError('Unique native training anchor required')
    native=dict(token=token,source_log=name,config=str(native_config),sensors=str(sensors),
        model_roots=[factory['semantic_model'],factory['depth_model'],str(processed/'ego_masks')],
        database=str(databases[0]),sensor_root=str(unified),maps_root=factory['maps_root'],
        output=str(root/'reconstruction'),working_directory=str(work),
        asset_root=str(root/'assets'),stage_timeout_seconds=7200,cpu_workers=8)
    frozen_json(root/'native-job.json',native)
    asset=dict(token=token,source_log=name,anchor_index=anchor[0],source_group=candidate['source_group'],
        travel_id=0,cameras=['CAM_F0','CAM_L0','CAM_R0','CAM_B0'],history_frames=4,
        future_frames=10,render_times=15,trainable=False)
    job=dict(token=token,source_log=name,output=str(root/'recovery'),gpu=gpu,
        minimum_source_psnr_db=factory.get('minimum_source_psnr_db',27.),
        reconstruction_job=str(root/'native-job.json'),asset_template=asset,
        gs_python=factory['gs_python'],reconstruction_python=factory['reconstruction_python'],
        runtime_environment=factory['runtime_environment'],runtime_witness=factory['runtime_witness'],
        sensor_root=factory['sensor_roots']['camera'][0],max_wall_seconds=21600,
        reconstruction_timeout_seconds=18000)
    frozen_json(recipe,job)
    return str(recipe)
