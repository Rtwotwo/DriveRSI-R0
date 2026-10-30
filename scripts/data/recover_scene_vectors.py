"""Recover visual vectors from explicitly selected irregular real histories."""

import argparse
import json
from pathlib import Path

from driversi.data.catalog.corpus import write_report
from driversi.data.catalog.history_recovery import recover_batch
from driversi.data.catalog.features import VisionEncoder
from driversi.data.catalog.store import Catalog


class NoNewInference:
    def __init__(self,config):
        raise RuntimeError('Expected complete independent-process feature reuse')


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('request')
    parser.add_argument('--verify-resume',action='store_true')
    args=parser.parse_args()
    request=json.loads(Path(args.request).read_text())
    if set(request['databases'])!=set(request['tokens']) or not set(request['databases'])<={'navtrain','navtest'}:
        raise ValueError('Explicit matching split-local databases and tokens required')
    if len({str(Path(p).resolve()) for p in request['databases'].values()})!=len(request['databases']):
        raise ValueError('Recovery splits require distinct databases')
    cfg=json.loads(Path(request['feature_config']).read_text())
    out=Path(request['output'])/('resume' if args.verify_resume else 'extraction')
    out.mkdir(parents=True,exist_ok=True)
    state={'complete':False,'splits':{},'synthetic_images':0,'training_history_repaired':False}
    for split,path in request['databases'].items():
        catalog=Catalog(path,split)
        try:
            report=recover_batch(catalog,cfg,request['tokens'][split],out/split,
                backend_factory=NoNewInference if args.verify_resume else VisionEncoder)
            if args.verify_resume and (report['ready'] or report['reused']!=report['requested']):
                raise RuntimeError('Resume did not reuse every feature')
            state['splits'][split]={k:v for k,v in report.items() if k!='rows'}
            write_report(out/'summary.json',state)
            print(split,report['status'],report['ready'],report['reused'],flush=True)
        finally:catalog.close()
    state['complete']=True
    write_report(out/'summary.json',state)


if __name__=='__main__':main()
