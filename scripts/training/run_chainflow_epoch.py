"""Use the unified lifecycle for an explicitly budgeted ChainFlow epoch attempt."""
import argparse
import json
from pathlib import Path
from driversi.training.contracts.job import resolve_request
from driversi.training.orchestration.lifecycle import prepare, execute


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--phase',choices=('train','resume','evaluate'),default='train')
    parser.add_argument('--previous',type=Path)
    parser.add_argument('--evaluator',type=Path)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args();resolved=resolve_request(args.request)
    if resolved['request']['schema_version']!=4:
        parser.error('This entry requires explicit schema_version 4')
    plan=prepare(resolved,args.output,phase=args.phase,previous=args.previous,evaluator=args.evaluator)
    result=execute(plan) if args.execute else dict(status='prepared',plan=str(args.output/'training-plan.json'))
    print(json.dumps({k:result.get(k) for k in ('status','reason','plan','checkpoint','recoverable_checkpoint')},indent=2))
    return 0 if result['status'] in ('completed','prepared') else 2


if __name__=='__main__':
    raise SystemExit(main())
