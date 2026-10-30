"""Protocol fixtures only: these are not driving models or training results."""
import numpy as np
import pytest
from driversi.training.online_rl.online import Decision, Step, RolloutSession, update_from_batch

class Actor:
    policy_version = "parent-v1"
    likelihood_contract = "test-native-joint-v1"
    def act(self, observation, memory):
        return Decision(np.array([0.1, 0.2]), -0.7, self.policy_version, memory + 1, {"value": 0.2})

class Env:
    source_log, split = "training-log", "train"
    def __init__(self, end=None, kind="terminated", intervene=False):
        self.state = np.array([0.]); self.end = end; self.kind = kind; self.intervene = intervene
    def observation(self): return self.state
    def step(self, action):
        self.state += 1
        done = self.state[0] == self.end
        return Step(self.state, action if not self.intervene else action * 0,
                    1., bool(done and self.kind == "terminated"), bool(done and self.kind == "truncated"),
                    True, "safety_override" if self.intervene else None)

class Learner:
    parent_version = "parent-v1"
    likelihood_contract = "test-native-joint-v1"
    def __init__(self): self.calls = 0
    def update(self, batch, *, max_optimizer_steps):
        self.calls += 1
        assert batch.transitions[0].decision.learner_context == {"value": 0.2}
        return {"candidate": "TEST_ONLY_CANDIDATE", "optimizer_steps": max_optimizer_steps}

def session(env=None, actor=None):
    return RolloutSession(env or Env(), actor or Actor(), memory=0, protected_logs=[])

def test_chunks_preserve_world_memory_and_detach_mutable_buffers():
    s = session(); a = s.collect(max_environment_steps=2, max_wall_seconds=10)
    b = s.collect(max_environment_steps=2, max_wall_seconds=10)
    assert a.boundary_memory == 2 and b.boundary_memory == 4
    assert a.transitions[0].observation[0] == 0 and a.transitions[0].step.observation[0] == 1
    assert b.transitions[0].observation[0] == 2
    assert a.receipt()["chunk_boundary"] and a.receipt()["bootstrap_required"]
    assert a.boundary_observation[0] == 2

@pytest.mark.parametrize("kind,bootstrap", [("terminated", False), ("truncated", True)])
def test_termination_bootstrap_is_not_a_reset(kind, bootstrap):
    s=session(Env(2, kind)); batch=s.collect(max_environment_steps=5,max_wall_seconds=10)
    assert len(batch.transitions)==2 and batch.receipt()["bootstrap_required"] is bootstrap
    assert batch.boundary_observation[0]==2
    with pytest.raises(ValueError,match="Episode ended"):s.collect(max_environment_steps=1,max_wall_seconds=10)

def test_intervention_rejects_entire_batch_without_learning():
    b=session(Env(intervene=True)).collect(max_environment_steps=5,max_wall_seconds=10)
    learner=Learner(); result=update_from_batch(learner,b,max_optimizer_steps=2)
    assert result["status"]=="rejected" and learner.calls==0
    assert "sampled_executed_action_mismatch" in result["reasons"]

def test_parent_and_probability_contract_must_match():
    b=session().collect(max_environment_steps=2,max_wall_seconds=10)
    learner=Learner(); learner.parent_version="different"
    with pytest.raises(ValueError,match="Stale"):update_from_batch(learner,b,max_optimizer_steps=1)
    learner.parent_version=b.policy_version;learner.likelihood_contract="other"
    with pytest.raises(ValueError,match="likelihood"):update_from_batch(learner,b,max_optimizer_steps=1)

def test_update_produces_candidate_without_activating_actor():
    s=session();b=s.collect(max_environment_steps=2,max_wall_seconds=10)
    result=update_from_batch(Learner(),b,max_optimizer_steps=2)
    assert result["optimizer_steps"]==2 and not result["promoted"] and not result["policy_activated"]
    assert s.actor.policy_version=="parent-v1"

def test_mid_rollout_model_change_and_test_source_rejected():
    s=session();s.actor.policy_version="changed"
    with pytest.raises(ValueError,match="Frozen"):s.collect(max_environment_steps=1,max_wall_seconds=10)
    e=Env();e.split="test"
    with pytest.raises(ValueError,match="training source"):session(e)
    e.split="train"
    with pytest.raises(ValueError,match="Protected"):RolloutSession(e,Actor(),memory=0,protected_logs=[e.source_log])

def test_no_fabricated_likelihood_or_probability_conversion():
    class Missing(Actor):
        def act(self,o,m):return Decision([1.],float("nan"),self.policy_version,m)
    with pytest.raises(ValueError,match="sampling likelihood"):session(actor=Missing()).collect(max_environment_steps=1,max_wall_seconds=10)


def test_mutated_or_forged_receipt_cannot_bypass_update_validation():
    from dataclasses import replace
    b=session(Env(intervene=True)).collect(max_environment_steps=1,max_wall_seconds=10)
    forged=replace(b,reasons=())
    with pytest.raises(ValueError,match="provenance"):
        update_from_batch(Learner(),forged,max_optimizer_steps=1)
