import copy
import numpy as np
import pytest
from types import SimpleNamespace
from driversi.contracts.driving_task import bind_driving_tasks
from driversi.contracts.road import Route
from driversi.rsi.correction.perturbation import validate_perturbation,perturbed_action,arrival_constraints


def cfg():return dict(start_tick=20,duration_ticks=20,jerk_delta=1.,lateral_delta=.4)


@pytest.mark.parametrize("key,value",[("start_tick",True),("start_tick",10),("duration_ticks",40),("lateral_delta",float("nan")),("lateral_delta",.9),("jerk_delta",3.)])
def test_invalid_generation_is_rejected(key,value):
    c=cfg();c[key]=value
    with pytest.raises(ValueError):validate_perturbation(c,300)


def test_continuous_prefix_and_correction_horizon_required():
    assert validate_perturbation(cfg(),300)==cfg()
    with pytest.raises(ValueError):validate_perturbation(cfg(),100)
    c=cfg();c.update(jerk_delta=0,lateral_delta=0)
    with pytest.raises(ValueError):validate_perturbation(c,300)


def test_action_delta_respects_native_bounds_without_mutating_model_output():
    raw=np.array([2.9,.9]);original=raw.copy()
    np.testing.assert_allclose(perturbed_action(raw,cfg(),[-5,-1],[3,1]),[3,1])
    np.testing.assert_array_equal(raw,original)


def test_contacts_support_and_heading_reject_arrival(straight_scene):
    s=copy.deepcopy(straight_scene);s.actor_ids=tuple("ego" if x=="a" else x for x in s.actor_ids)
    s.controlled[:]=False;s.controlled[0]=True
    s=bind_driving_tasks(s,{"ego":Route([[0,0],[100,0]])})
    state=np.zeros((len(s.actor_ids),13));state[0,0]=5;state[0,4]=2
    tr=SimpleNamespace(valid=np.array([True]),terminated=np.array([False]),truncated=np.array([False]),events={})
    assert arrival_constraints(s,state,tr,[])["accepted"]
    assert not arrival_constraints(s,state,tr,[("ego","b")])["accepted"]
    state[0,3]=.5
    assert "heading_bound_exceeded" in arrival_constraints(s,state,tr,[])["reasons"]
    state[0,3]=0;tr.events={"route_support_end":[True]}
    assert not arrival_constraints(s,state,tr,[])["accepted"]


def test_changed_generator_cannot_cross_into_training_curriculum(tmp_path):
    from driversi.agents.backends.native import digest
    from driversi.rsi.plugins.recovery import inherited_files
    source=tmp_path/"generator.py";source.write_text("version1")
    plan={"files":{str(source):digest(source)}}
    assert inherited_files(plan,copy.deepcopy(plan))==plan["files"]
    source.write_text("version2")
    with pytest.raises(ValueError):inherited_files(plan,copy.deepcopy(plan))
