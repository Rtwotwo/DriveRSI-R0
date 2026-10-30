"""Run a bounded VLA memory/diagnosis episode on an existing native branch."""
import argparse
import json
from pathlib import Path

from driversi.agents.backends.native import project_root
from driversi.configuration import read_mapping
from driversi.rsi.workflow import engine
from driversi.runtime.budget import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('registry','database','memory','output-root'):
        parser.add_argument('--'+name,type=Path,required=True)
    for name in ('task-id','source-task','branch-stage'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--teacher-stage',default='teacher')
    parser.add_argument('--mode',choices=('evolve','frozen'),required=True)
    parser.add_argument('--roles',type=Path,default=project_root()/'src/config/rsi/skills/augustus-teacher-student.json')
    args=parser.parse_args()
    output=args.output_root.resolve()
    if not output.is_relative_to(project_root().resolve()/'exp'):
        raise ValueError('Episode outputs must live under exp')
    if not args.task_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in args.task_id):
        raise ValueError('Simple unique task identity required')
    output.mkdir(parents=True,exist_ok=True)
    request=output/(args.task_id+'.task.json')
    if request.exists():raise ValueError('Task exists; use a new id, preserve previous evidence')
    roles={k:str((args.roles.resolve().parent/v).resolve(strict=True)) for k,v in read_mapping(args.roles).items()}
    task={'schema_version':2,'id':args.task_id,'label':'History-conditioned VLA service memory',
        'budget':{'max_wall_seconds':300,'max_sim_agent_steps':0},'steps':[
            {'id':'branch','provider':'native_branch_reference','inputs':{},'max_wall_seconds':90,
             'config':{'source_task':args.source_task,'source_stage':args.branch_stage}},
            {'id':'memory','provider':'vla_memory_skill_episode','inputs':{'branch':'branch'},'max_wall_seconds':150,
             'config':{**roles,'database':str(args.database.resolve()),'memory':str(args.memory.resolve()),
                'mode':args.mode,'source_teacher':{'source_task':args.source_task,'source_stage':args.teacher_stage}}}]}
    write_json(request,task)
    engine.register(args.registry,request,output)
    state=engine.run(args.registry,args.task_id)
    print(json.dumps({'task':args.task_id,'status':state['status'],'directory':state['directory'],
        'stages':[{'id':s['id'],'status':s['status'],'reason':s.get('reason')} for s in state['stages']]}))
    return 0 if state['status']=='completed' else 2


if __name__=='__main__':raise SystemExit(main())
