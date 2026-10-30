"""Retrieve and execute reviewed VLA service skills over sealed physical evidence."""
import time
from pathlib import Path

from driversi.agents.backends.native import digest,project_root
from driversi.configuration import read_mapping
from driversi.data.catalog.store import Catalog
from driversi.rsi.contracts import _keys
from driversi.rsi.decisions.vla_memory import (
    LIBRARY_VERSION,SKILL_TREE,SceneSkillMemory,history_signature,contact_trace,
    earliest_unpassed_gate,make_case,membership_reference,
)
from driversi.rsi.plugins import model_services
from driversi.rsi.workflow.results import result
from driversi.rsi.plugins.chainflow_synthetic import _stored_stage
from driversi.rsi.plugins.synthesis import protect
from driversi.runtime.budget import write_json


def prepare(config,base,registry):
    _keys(config,('database','memory','mode','source_teacher','teacher_config','student_config'),'VLA memory service')
    if config['mode'] not in ('evolve','frozen'):raise ValueError('Explicit evolve/frozen mode required')
    memory=(base/config['memory']).resolve()
    if not memory.is_relative_to(project_root().resolve()/'exp'):raise ValueError('Memory must live under exp')
    database=(base/config['database']).resolve(strict=True)
    catalog=Catalog(database,'navtrain',readonly=True)
    catalog.close()
    state,_,ref=_stored_stage(config['source_teacher'],'navsim_teacher_execution',registry)
    roles,_=model_services.prepare({k:config[k] for k in ('teacher_config','student_config')},base,registry)
    return {'database':str(database),'memory':str(memory),'mode':config['mode'],'roles':roles,
        'source_teacher':{**config['source_teacher'],'identity':ref['identity']},
        'source_groups':state['task']['source_groups']},0


def execute(config,inputs,output,registry):
    started=time.monotonic()
    from driversi_agents.chainflow.benchmark import official_filter
    from driversi.contracts.scene import Scene

    reference=config['source_teacher']
    _,world,ref=_stored_stage({k:reference[k] for k in ('source_task','source_stage')},'navsim_teacher_execution',registry)
    if ref['identity']!=reference['identity']:raise ValueError('Native teacher receipt changed')
    teacher=read_mapping(world['artifacts']['report'])
    branch=read_mapping(inputs['branch']['artifacts']['report'])
    window_path=inputs['branch']['artifacts']['trajectory_window']
    if (branch['execution_identity']!=ref['identity'] or branch['branch']!='teacher'
            or branch['source_log']!=teacher['source_log']
            or digest(window_path)!=branch['trajectory_window_sha256']):
        raise ValueError('Memory diagnosis requires the same selected teacher execution')
    scene_path=str(Path(branch['execution'])/'scene.npz')
    if branch['files'].get(scene_path)!=digest(scene_path):
        raise ValueError('Sealed scene membership changed')
    window=read_mapping(window_path)
    physical=Scene.load(Path(scene_path))
    member=membership_reference(physical.metadata,teacher,window)
    catalog=Catalog(config['database'],'navtrain',readonly=True)
    try:scene=catalog.scene(member['token'])
    finally:catalog.close()
    record,desc=scene['record'],scene['descriptor']
    _,official=official_filter('navtrain')
    if (record['kind']!='original_navtrain' or record['source_log']!=teacher['source_log']
            or record['anchor_index']!=member['anchor_index']
            or record['token'] not in set(official['tokens']) or record['source_log'] not in set(official['log_names'])):
        raise ValueError('Only verified official training sources may write model-service memory')
    protect([record['source_log'],desc['source_group']],registry)
    if window['source_log']!=record['source_log'] or window['source_split']!='train':
        raise ValueError('Executed historical source mismatch')
    signature=history_signature(window)
    role=config['roles']
    for path, sha in role['student_parent_files'].items():
        if digest(path) != sha:
            raise ValueError('Student parent bytes changed after memory episode registration')
    context={'student_model':role['roles']['student'],'parent_checkpoint':role['roles']['student_parent'],
        'student_parent_assets_sha256':role['student_parent_assets_sha256'],
        'teacher_checkpoint_sha256':role['files'][role['roles']['teacher_checkpoint']],
        'skill_library_version':LIBRARY_VERSION,'scope':'offline_privileged_model_service'}
    memory=SceneSkillMemory(config['memory'],config['mode'])
    output.mkdir(parents=True)
    try:
        matches=memory.retrieve(context,signature,desc['source_group'])
        detailed=any(m['suggested_base_skill']=='contact_trace_audit' for m in matches)
        trace=[{'base_skill':'history_signature','status':'completed'}]
        contact=None
        if detailed:
            contact=contact_trace(teacher)
            write_json(output/'contact-trace.json',contact)
            trace.append({'base_skill':'contact_trace_audit','status':'completed',
                'selected_by':'relevant_prior_contact_evidence','cause_established':False})
        audited=model_services.execute(role,inputs,output/'readiness',registry)
        diagnosis=read_mapping(audited['artifacts']['report'])
        trace.append({'base_skill':'native_readiness_audit','status':'completed'})
        arrival=teacher.get('arrival_contacts');future=teacher.get('branches',{}).get('teacher',{}).get('object_contacts')
        if (not isinstance(arrival,list) or not isinstance(future,list)
                or len(arrival)!=branch['trajectory_evidence']['arrival_contact_records']
                or len(future)!=branch['trajectory_evidence']['future_contact_records']):
            raise ValueError('Branch counts and native contact evidence disagree')
        source={'source_split':'navtrain','source_log':record['source_log'],
                'source_group':desc['source_group'],'training_token':record['token'],
                'compiled_window_source_token':window['source_token']}
        case=make_case(context,signature,source,diagnosis,ref['identity'],contacts_present=bool(arrival or future))
        write=memory.record(case)
        count=len(memory.cases())
    finally:memory.close()
    report={'kind':'vla_memory_skill_episode','schema_version':1,'mode':config['mode'],
        'skill_tree':SKILL_TREE,'context':context,'source':source,'query_history':signature,
        'retrieved_cases':matches,'memory_influenced_procedure':detailed,'execution_trace':trace,
        'earliest_unpassed_gate':earliest_unpassed_gate(diagnosis),
        'next_action':diagnosis['next_action'],'causal_attribution':'unresolved','memory_update':write,
        'memory_case_count':count,'case_id':case['case_id'],
        'execution_identity':ref['identity'],'physics_reexecuted':False,'student_evaluated':False,
        'student_gain_credited':False,'training_executed':False,'policy_prompt_modified':False,
        'reliable_skill_promoted':False,'trainable':False,'score_threshold_changed':False,
        'cost':{'audit_execution_seconds':time.monotonic()-started,'scope':'this_audit_only_not_total_training_cost'},
        'claim':'persistent_procedure_reuse_not_student_learning_or_proven_diagnosis_accuracy'}
    write_json(output/'case.json',case)
    artifacts={'case':str(output/'case.json'),'diagnosis':audited['artifacts']['report']}
    if contact is not None:artifacts['contact_trace']=str(output/'contact-trace.json')
    return result(output,report,artifacts=artifacts)


def register(registry,Provider):
    registry.register(Provider('vla_memory_skill_episode','2',(('branch','selected_native_branch'),),
        'vla_memory_skill_episode',prepare,execute,('report','case','diagnosis')))
