"""Build training-side semantic condition bundles or an exhaustive batch plan."""
import argparse
import json
from pathlib import Path

from driversi.data.catalog.store import Catalog
from driversi.rendering.whitebox.conditioning import export_bundle, write_json
from driversi.rendering.whitebox.exporter import sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['plan','export'])
    p.add_argument('--database',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--token',action='append',default=[])
    p.add_argument('--batch-file',type=Path)
    p.add_argument('--maps-root',default='/data/xinyao/navsim_workspace/dataset/maps')
    p.add_argument('--batch-size',type=int,default=128)
    args=p.parse_args()
    cat=Catalog(args.database,'navtrain',readonly=True)
    try:
        if args.command=='plan':
            if args.output.exists():raise FileExistsError(args.output)
            if not 1<=args.batch_size<=128:raise ValueError('Batch size must be 1..128, matching export limit')
            args.output.mkdir(parents=True)
            rows=cat.db.execute('SELECT token,source_group FROM scenes ORDER BY source_group,token').fetchall()
            batches=[]
            for i in range(0,len(rows),args.batch_size):
                items=[{'token':r['token'],'source_group':r['source_group']} for r in rows[i:i+args.batch_size]]
                name=f'batch-{i//args.batch_size:05d}.json'
                write_json(args.output/name,items);batches.append(name)
            report=dict(kind='full_navtrain_condition_plan',count=len(rows),batches=batches,
                        database=str(cat.path),executed=False,full_coverage_requires_all_receipts=True,
                        output_contract='71timestamps; four calibrated cameras; structured conditions; captions separate',
                        rejection_policy='retain source coverage/gap/map/calibration failures; never silently extrapolate')
            write_json(args.output/'plan.json',report)
        else:
            if args.batch_file:
                if args.token:raise ValueError('Choose tokens or one bounded batch file')
                args.token=[row['token'] for row in json.loads(args.batch_file.read_text())]
            if not 1<=len(args.token)<=128 or len(set(args.token))!=len(args.token):raise ValueError('1..128 unique batch tokens required')
            args.output.mkdir(parents=True,exist_ok=True);report={'rows':[],'complete':False}
            for token in args.token:
                try:
                    target=args.output/token
                    if (target/'manifest.json').exists():
                        manifest=json.loads((target/'manifest.json').read_text())
                        import driversi.rendering.whitebox.conditioning as producer
                        if manifest.get('producer_sha256')!=sha256(producer.__file__):raise ValueError('Changed exporter version; use new output version')
                        for name,expected in manifest['files'].items():
                            if Path(name).name!=name or sha256(target/name)!=expected:raise ValueError('Changed cached output')
                        source=json.loads((target/'source.json').read_text());record=cat.scene(token)['record']
                        if source['token']!=token or source['source_log_sha256']!=sha256(record['log_path']):raise ValueError('Changed cached source')
                        row=dict(token=token,status='reused',frame_count=len(source['frames']),actor_count=len(source['actors']))
                        report['rows'].append(row);write_json(args.output/'report.json',report);continue
                    data=export_bundle(cat.scene(token)['record'],args.maps_root,args.output/token)
                    row=dict(token=token,status='completed',frame_count=len(data['frames']),actor_count=len(data['actors']))
                except Exception as exc:row=dict(token=token,status='rejected',reason=type(exc).__name__+': '+str(exc))
                report['rows'].append(row);write_json(args.output/'report.json',report)
            report['complete']=True;report['successful']=sum(x['status'] in ('completed','reused') for x in report['rows']);write_json(args.output/'report.json',report)
        print(json.dumps({k:v for k,v in report.items() if k!='batches'}))
    finally:cat.close()


if __name__=='__main__':main()
