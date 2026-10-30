import copy
import json
from pathlib import Path

import pytest

from driversi.rsi.decisions.vla_memory import (
    SceneSkillMemory,history_signature,contact_trace,earliest_unpassed_gate,make_case,membership_reference,
)


def history():
    return {'input_scope':'history_only','history':[
        {'relative_seconds':t,'ego_rear_axle_pose':[i*.5,0,0,0],
         'objects':[{'actor_id':'ego','center_state':[i*.5,0]},
                    *[{'actor_id':str(j),'center_state':[j+2,0]} for j in range(10)]]}
        for i,t in enumerate([-1.5,-1.,-.5,0.])]}


def diagnosis():
    return {'failed_physical_checks':['arrival_contacts_absent','producer_trajectory_admitted'],
            'learning_blockers':['joint_cache_missing']}


def case(group='log-a',context=None):
    return make_case(context or {'student':'augustus','parent':'parent'},history_signature(history()),
        {'source_split':'navtrain','source_group':group},diagnosis(),'execution-'+group,contacts_present=True)


def test_history_reads_only_past_and_uses_measurements():
    w=history();s=history_signature(w)
    w['future']=[{'malformed':'not historical input'}]
    assert history_signature(w)==s
    assert s['history_mean_path_speed_mps']==1 and s['objects_within_30m']==10
    assert s['conditions']==['ego_low_speed','history_near_straight','nearby_objects_dense']
    w['history'][-1]['relative_seconds']=1
    with pytest.raises(ValueError,match='timeline'):history_signature(w)


def test_memory_reopen_context_source_isolation_and_idempotence(tmp_path):
    p=tmp_path/'m.sqlite';m=SceneSkillMemory(p,'evolve');c=case()
    assert m.record(c)['recorded'];m.close()
    m=SceneSkillMemory(p,'evolve')
    assert not m.record(c)['recorded']
    assert len(m.cases())==1
    assert not m.retrieve(c['context'],c['history_signature'],'log-a')
    found=m.retrieve(c['context'],c['history_signature'],'log-b')
    assert len(found)==1 and found[0]['suggested_base_skill']=='contact_trace_audit'
    assert not found[0]['relevance_calibrated']
    assert not m.retrieve({'student':'augustus','parent':'another'},c['history_signature'],'log-b')
    wrong=copy.deepcopy(c);wrong['student_gain_credited']=True
    with pytest.raises(ValueError):m.record(wrong)
    wrong=copy.deepcopy(c);wrong['source_split']='navtest'
    with pytest.raises(ValueError):m.record(wrong)
    wrong=copy.deepcopy(c);wrong['causal_attribution']='invented'
    with pytest.raises(ValueError,match='Conflicting'):m.record(wrong)
    m.close()


def test_frozen_is_read_only_and_missing_memory_is_not_created(tmp_path):
    p=tmp_path/'m.sqlite';m=SceneSkillMemory(p,'evolve');m.record(case());m.close()
    before=p.read_bytes();stamp=p.stat().st_mtime_ns
    m=SceneSkillMemory(p,'frozen');assert len(m.cases())==1
    assert m.record(case('log-b'))=={'recorded':False,'reason':'frozen_no_write'}
    m.close();assert p.read_bytes()==before and p.stat().st_mtime_ns==stamp
    p2=tmp_path/'absent.sqlite';m=SceneSkillMemory(p2,'frozen')
    assert m.cases()==[];m.record(case());m.close();assert not p2.exists()


def test_contact_trace_counts_pair_ticks_not_accidents():
    t={'arrival_tick':2,'arrival_contacts':[{'tick':0,'actors':['b','a']},{'tick':1,'actors':['a','b']}],
       'branches':{'teacher':{'object_contacts':[{'tick':3,'actors':['b','a']},{'tick':3,'actors':['ego','c']}]}}}
    r=contact_trace(t)
    assert r['distinct_pairs']==2 and r['present_at_initial_tick_pairs']==1
    assert r['pairs_persisting_across_anchor']==1 and r['new_pairs_after_anchor']==1
    assert r['ego_pairs']==1 and r['causal_attribution']=='unresolved'
    t['branches']['teacher']['object_contacts'][0]['tick']=2
    with pytest.raises(ValueError,match='phase'):contact_trace(t)


def test_contact_trace_cannot_infer_no_contacts_from_missing_data():
    with pytest.raises(ValueError,match='Complete'):contact_trace({'arrival_tick':0,'branches':{}})


def test_routing_reports_first_gate_not_root_cause():
    assert earliest_unpassed_gate(diagnosis())=='world_validity'
    assert earliest_unpassed_gate({'failed_physical_checks':['native_teacher_pdms_ge_95'],'learning_blockers':['cache']})=='teacher_qualification'
    assert earliest_unpassed_gate({'failed_physical_checks':[],'learning_blockers':['cache']})=='joint_supervision'
    assert earliest_unpassed_gate({'failed_physical_checks':[],'learning_blockers':[]}) is None


def test_official_member_can_differ_from_physical_history_start():
    m={'source_train_membership':{'token':'official','source_log':'train-log','anchor_index':24,
          'reference_role':'official_split_membership_not_physical_anchor'},
       'split':'train','source_log':'train-log','scenario_id':'earlier-start'}
    t={'token':'earlier-start','source_log':'train-log'}
    w={'source_split':'train','source_log':'train-log','source_token':'earlier-start'}
    assert membership_reference(m,t,w)['token']=='official'
    t['token']='official'
    with pytest.raises(ValueError,match='identities'):membership_reference(m,t,w)
