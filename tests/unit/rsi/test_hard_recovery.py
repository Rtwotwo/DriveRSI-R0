import copy
import numpy as np
import pytest
from driversi.rsi.correction.contact_origin import source_contacts
from driversi.rsi.correction.curriculum import SAFETY,admit,correction_verdict,failed_driving
from driversi.rsi.plugins.long_recovery import benefit


def score(value=1.):
    return dict(status="scored",official_benchmark=False,score=value,metric_scores=dict.fromkeys(SAFETY,1.))


def test_source_static_overlap_is_quarantined_not_exempted(straight_scene):
    s=straight_scene;s.poses[0,1,:]=s.poses[0,0,:];s.metadata["source_actor_kinds"]={"a":"barrier","b":"barrier"}
    original=s.poses.copy();r=source_contacts(s)
    assert r["quarantined"] and r["contacts"][0]["static_annotation_pair"]
    assert not r["contacts"][0]["collision_exempt"]
    np.testing.assert_array_equal(s.poses,original)
    s.metadata["source_actor_kinds"]["a"]="vehicle"
    assert not source_contacts(s)["contacts"][0]["static_annotation_pair"]


def test_safe_or_missing_scores_cannot_count_as_corrected_failure():
    e=dict(same_anchor=True,complete=True,valid=True,teacher_no_fallback=True,contact_states=0)
    reports={x:dict(execution=copy.deepcopy(e),score=score()) for x in ("student","teacher")}
    assert not correction_verdict(reports,True)["accepted"]
    reports["student"]["score"]["metric_scores"][SAFETY[0]]=0
    assert correction_verdict(reports,True)["accepted"]
    reports["student"]["execution"]["complete"]=False
    assert not correction_verdict(reports,True)["accepted"]
    reports["student"]["score"]["metric_scores"].pop(SAFETY[1])
    assert not failed_driving(reports["student"]["score"])


def evaluation(candidate):
    return dict(status="completed",evaluation_suite_sha256="fixed",cases=[dict(source_log=x,steps=600,
        branches={role:dict(complete=True,source_contacts=dict(quarantined=False),support_diagnosis=dict(blocked=False),contacts=[],
                    score=score(candidate if role=="candidate" else .95)) for role in ("parent","candidate")})
        for x in ("log-a","log-b")])


def test_long_gate_rejects_local_regression_even_when_average_improves():
    a,b=evaluation(.96),evaluation(.99)
    assert benefit(a,b,dict(different_processes=True,same_state=True))["accepted"]
    b["cases"][1]["branches"]["candidate"]["score"]["metric_scores"][SAFETY[0]]=0
    assert not benefit(a,b,dict(different_processes=True,same_state=True))["accepted"]


@pytest.mark.parametrize("change",["same_log","short","source_contact","resume","tiny_gain","scorer"])
def test_no_promotion_for_insufficient_or_contaminated_evidence(change):
    a,b=evaluation(.96),evaluation(.99);resume=dict(different_processes=True,same_state=True)
    if change=="same_log":
        for v in (a,b):v["cases"][1]["source_log"]="log-a"
    elif change=="short":
        for v in (a,b):v["cases"][0]["steps"]=20
    elif change=="source_contact":b["cases"][0]["branches"]["candidate"]["source_contacts"]["quarantined"]=True
    elif change=="resume":resume["same_state"]=False
    elif change=="tiny_gain":b=evaluation(.9601)
    else:b["evaluation_suite_sha256"]="other"
    assert not benefit(a,b,resume)["accepted"]


def test_expected_last_time_truncation_only():
    from types import SimpleNamespace
    from driversi.rsi.correction.curriculum import valid_recovery_transition
    tr=SimpleNamespace(valid=np.array([True]),terminated=np.array([False]),
        truncated=np.array([True]),episode_done=True,events={})
    assert valid_recovery_transition(tr,final_step=True)
    assert not valid_recovery_transition(tr,final_step=False)
    tr.events={"route_support_end":np.array([True])}
    assert not valid_recovery_transition(tr,final_step=True)
    tr.events={};tr.terminated[0]=True
    assert not valid_recovery_transition(tr,final_step=True)
    tr.terminated[0]=False;tr.valid[0]=False
    assert not valid_recovery_transition(tr,final_step=True)


def test_checkpoint_promotion_binds_evaluated_publication(tmp_path):
    from driversi.agents.backends.native import digest
    from driversi.rsi.plugins.long_recovery import verify_evaluated_checkpoint
    p=tmp_path/"weights.pt";p.write_bytes(b"original")
    student=dict(checkpoint=str(p),checkpoint_files={str(p):digest(p)})
    evaluation=dict(evaluated_checkpoint=str(p),evaluated_checkpoint_sha256=digest(p))
    verify_evaluated_checkpoint(student,evaluation)
    p.write_bytes(b"changed")
    with pytest.raises(ValueError,match="checkpoint"):
        verify_evaluated_checkpoint(student,evaluation)


def test_batched_projection_exact_scalar_and_ties():
    from driversi.contracts.road import Route
    rng=np.random.default_rng(91)
    for points in (np.cumsum(rng.normal(size=(80,2)),axis=0),np.array([[0.,0.],[2.,0.],[2.,2.],[0.,2.]])):
        route=Route(points)
        xy=np.concatenate((rng.normal(size=(170,2))*5,points,[[1.,1.]]))
        np.testing.assert_array_equal(route.project_many(xy),np.asarray([route.project(x) for x in xy]))
    assert route.project_many(np.empty((0,2))).shape==(0,3)
    with pytest.raises(ValueError):route.project_many([0,1])


def test_admission_broadphase_never_excludes_actual_constraint():
    from driversi.sim.execution.traffic_lifecycle import admission_neighbors
    from driversi.agents.traffic_control.traffic import swept_contact
    rng=np.random.default_rng(218)
    for horizon,margin in ((.1,0.),(1.,.5),(5.,2.)):
        for _ in range(10):
            metric=np.r_[rng.normal(size=2)*30,rng.uniform(-np.pi,np.pi),rng.normal(size=2)*20,rng.uniform(.1,12,2)]
            pos=np.zeros((100,7));pos[:,:2]=rng.normal(size=(100,2))*120;pos[:,3]=rng.uniform(-np.pi,np.pi,100);pos[:,4:6]=rng.normal(size=(100,2))*20
            dims=rng.uniform(.1,15,(100,3));ids=set(admission_neighbors(metric,pos,np.ones(100,bool),dims,horizon,margin))
            direction=np.array([np.cos(metric[2]),np.sin(metric[2])])
            for j in range(100):
                other=np.r_[pos[j,:2],pos[j,3],pos[j,4:6],dims[j,:2]]
                delta=other[:2]-metric[:2];ahead=float(delta@direction)
                gap=abs(ahead)-.5*(metric[5]+other[5])
                trailing=max(0.,float((metric[3:5] if ahead>0 else other[3:5])@direction))
                follows=(np.cos(other[2]-metric[2])>.94 and abs(delta@np.array([-direction[1],direction[0]]))<.5*(metric[6]+other[6])+.5 and gap<2.+1.5*trailing)
                if follows or swept_contact(metric,other,metric[3:5],horizon,margin) is not None:
                    assert j in ids


def test_route_support_not_mined_as_driving_failure():
    from driversi.rsi.correction.curriculum import route_support_failure
    flags=np.zeros((600,1),bool)
    assert not route_support_failure(flags)["blocked"]
    flags[490,0]=True
    verdict=route_support_failure(flags)
    assert verdict["blocked"] and verdict["ticks"]==[491]
    assert verdict["reason"]=="route_support_exhausted"
    with pytest.raises(ValueError):route_support_failure([True,False])


def test_support_exhaustion_or_missing_diagnosis_blocks_promotion():
    a,b=evaluation(.96),evaluation(.99)
    b["cases"][0]["branches"]["candidate"]["support_diagnosis"]["blocked"]=True
    assert "source_support_unverified_or_exhausted" in benefit(a,b,dict(exact=True))["reasons"]
    b["cases"][0]["branches"]["candidate"].pop("support_diagnosis")
    assert not benefit(a,b,dict(exact=True))["accepted"]
