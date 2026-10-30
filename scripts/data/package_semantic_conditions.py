"""Create a complete, checked source-whitebox archive (no real photos)."""
import argparse
import json
import zipfile
from pathlib import Path

from driversi.rendering.whitebox.exporter import sha256
from driversi.rendering.whitebox.conditioning import CAMERAS


def package(bundle, output):
    bundle,output=Path(bundle),Path(output)
    if output.exists():raise FileExistsError(output)
    data=json.loads((bundle/'source.json').read_text())
    audit=json.loads((bundle/'motion-audit.json').read_text())
    manifest=json.loads((bundle/'source-renders/manifest.json').read_text())
    if not audit['passed'] or audit['token']!=data['token']:raise ValueError('Original-log motion audit required')
    if manifest['source_sha256']!=sha256(bundle/'source.json'):raise ValueError('Rendered source changed')
    if manifest['map_sha256']!=sha256(bundle/'map.json') or manifest['palette_sha256']!=sha256(bundle/'palette.json'):
        raise ValueError('Map or palette changed')
    for scope in ['retrieval_history','video_clip']:
        path=bundle/f'source-{scope}-description.json'
        if not path.exists():raise ValueError('Both v2 descriptions required before packaging')
        caption=json.loads(path.read_text())
        if (caption['schema_version']!='driversi.scene_description.v2' or caption['scope']!=scope
            or caption['scene_identity']['token']!=data['token'] or caption['scene_identity']['role']!='source_log'):
            raise ValueError('Description does not match source bundle')
        if (scope=='video_clip' and caption['retrieval'] is not None) or (scope=='retrieval_history' and caption['video_generation'] is not None):
            raise ValueError('Description scope leakage')
    expected={(i,cam) for i in range(71) for cam in CAMERAS}
    if len(manifest['images'])!=284 or {(x['tick'],x['camera']) for x in manifest['images']}!=expected:
        raise ValueError('Complete four-camera PNG set required')
    files=[]
    for row in manifest['images']:
        p=bundle/'source-renders'/row['file']
        if p.name!=row['file'] or sha256(p)!=row['sha256']:raise ValueError('Rendered image mismatch')
        files.append(p)
    for cam in CAMERAS:
        p=bundle/'source-renders'/(cam+'.mp4')
        if not p.is_file():raise ValueError('Camera video missing')
        files.append(p)
    files += [bundle/'source-renders/manifest.json'] + sorted(bundle.glob('*.json'))
    checksums={str(p.relative_to(bundle)):sha256(p) for p in files}
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_suffix(output.suffix+'.partial')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for p in files:z.write(p,str(p.relative_to(bundle)),compress_type=zipfile.ZIP_STORED if p.suffix in ('.png','.mp4') else zipfile.ZIP_DEFLATED)
        z.writestr('DOWNLOAD_MANIFEST.json',json.dumps({'token':data['token'],'files':checksums,
            'real_photos_included':False,'source_sampling':'2Hz annotations;0.1s explicit interpolation',
            'contains':'71states;284semantic PNG;4MP4;map;palette;captions;motion audit'},indent=2))
    with zipfile.ZipFile(temporary) as z:
        if z.testzip() is not None:raise ValueError('Archive CRC failure')
    temporary.rename(output)
    return {'archive':str(output),'bytes':output.stat().st_size,'sha256':sha256(output),'files':len(files)+1,'crc_verified':True}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();print(json.dumps(package(args.bundle,args.output)))
