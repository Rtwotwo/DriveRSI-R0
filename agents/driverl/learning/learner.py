"""DriveZero appendix A PPO update with the released Beta actor/value network.

This is a DriveRSI implementation, not unpublished upstream training code.
"""

import copy
import hashlib
import importlib.util
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from driversi.training.online_rl.online import Decision

CONTRACT = "driverl-native-physical-beta-v1"


def deterministic_training():
    """Process-local training mode: deterministic SDPA backward for exact recovery."""
    import os

    if torch.cuda.is_initialized() and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (
        ":4096:8",
        ":16:8",
    ):
        raise RuntimeError("Set CUBLAS_WORKSPACE_CONFIG before CUDA initialization")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cudnn.benchmark = False


@dataclass(frozen=True)
class LearningConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    policy_clip: float = 0.2
    value_clip: float = 0.2
    entropy_coefficient: float = 0.01
    value_coefficient: float = 0.5
    gradient_clip: float = 0.5
    learning_rate: float = 3e-4
    epochs: int = 4
    minibatch_size: int = 20480
    schedule_steps: int = 2400 * 44

    def validate(self):
        for key in ("gamma", "gae_lambda"):
            if not 0 <= getattr(self, key) <= 1:
                raise ValueError(key)
        for key in ("epochs", "minibatch_size", "schedule_steps"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(key)
        for key, value in asdict(self).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(key)
        if self.learning_rate == 0 or self.gradient_clip == 0:
            raise ValueError("positive learning settings required")


def decomposed_gae(rewards, values, bootstrap, terminated, truncated, gamma=0.99, lam=0.95):
    """A batch is one continuous world segment; terminal never bootstraps."""
    if rewards.ndim != 2 or rewards.shape != values.shape or bootstrap.shape != values.shape[1:]:
        raise ValueError("component shape mismatch")
    if terminated.shape != rewards.shape[:1] or truncated.shape != terminated.shape:
        raise ValueError("boundary shape mismatch")
    if (terminated[:-1] | truncated[:-1]).any():
        raise ValueError("batch crosses episode reset")
    advantage = torch.zeros_like(rewards)
    carry = torch.zeros_like(bootstrap)
    for i in reversed(range(len(rewards))):
        next_value = bootstrap if i + 1 == len(rewards) else values[i + 1]
        nonterminal = (~terminated[i]).to(values.dtype)
        delta = rewards[i] + gamma * nonterminal * next_value - values[i]
        continuation = (~(terminated[i] | truncated[i])).to(values.dtype)
        carry = delta + gamma * lam * continuation * carry
        advantage[i] = carry
    return advantage, advantage + values


def tensors(observation, device):
    return tuple(
        None if x is None else torch.as_tensor(x, device=device) for x in observation["features"]
    )


def policy(agent, observation, action=None):
    features = tensors(observation, next(agent.parameters()).device)
    params, values = agent.model(*features)
    return agent._policy_outputs(params, values, action, "sample", None)


class NativeActor:
    model_id = "driverl"
    likelihood_contract = CONTRACT

    def __init__(self, agent, version):
        self.agent, self.policy_version = agent, version
        self.agent.eval()  # identical deterministic preprocessing/net mode for collection and update

    @torch.no_grad()
    def act(self, observation, memory):
        action, logp, _, value = policy(self.agent, observation)
        return Decision(
            action[0, 0].cpu().numpy().copy(),
            float(logp[0, 0]),
            self.policy_version,
            memory,
            {"value": value[0, 0].cpu().numpy().copy()},
        )


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(value):
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"].cpu())
    if value["cuda"]:
        torch.cuda.set_rng_state_all([x.cpu() for x in value["cuda"]])


def muon_optimizer(agent, config, root):
    path = Path(root) / "agents/driverl/learning/muon/muon.py"
    from driversi.agents.backends.sources import require_source

    require_source(path.parent, "f98f1cacc0263b04290753e32be8d498c1efc806")
    spec = importlib.util.spec_from_file_location("driversi_pinned_muon", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    groups = {True: [], False: []}
    names = {True: [], False: []}
    # Explicit grouping assumption: hidden matrices Muon; output heads and 1-D terms AdamW.
    for name, param in agent.named_parameters():
        if not param.requires_grad:
            continue
        use = param.ndim == 2 and not any(
            k in name for k in ("value_head", "actor", "policy_head", "action_head")
        )
        groups[use].append(param)
        names[use].append(name)
    settings = []
    for use, params in groups.items():
        if not params:
            continue
        g = dict(params=params, use_muon=use, lr=config.learning_rate, weight_decay=0.0)
        if use:
            g.update(momentum=0.95)
        else:
            g.update(betas=(0.9, 0.95), eps=1e-10)
        settings.append(g)
    return module.SingleDeviceMuonWithAuxAdam(settings), {
        "muon": names[True],
        "adamw": names[False],
        "reference_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


class PaperLearner:
    model_id = "driverl"
    likelihood_contract = CONTRACT
    implementation_version = "driversi-driverl-paper-ppo-v1"

    def __init__(self, agent, parent_version, output, root, config=None):
        self.agent = copy.deepcopy(agent).eval()
        self.parent_version = parent_version
        self.output = Path(output)
        if not self.output.resolve().is_relative_to(Path(root).resolve() / "exp"):
            raise ValueError("learning checkpoints must be under exp")
        self.config = config or LearningConfig()
        self.config.validate()
        self.optimizer, self.optimizer_evidence = muon_optimizer(self.agent, self.config, root)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.config.schedule_steps
        )
        self.updates = 0
        self.last_metrics = []

    def state_dict(self):
        return {
            "schema_version": 1,
            "implementation": self.implementation_version,
            "parent_version": self.parent_version,
            "contract": CONTRACT,
            "config": asdict(self.config),
            "optimizer_evidence": self.optimizer_evidence,
            "model": self.agent.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "updates": self.updates,
            "rng": rng_state(),
        }

    def load_state_dict(self, state):
        if (
            state["implementation"] != self.implementation_version
            or state["contract"] != CONTRACT
            or state["parent_version"] != self.parent_version
            or state["config"] != asdict(self.config)
            or state["optimizer_evidence"] != self.optimizer_evidence
        ):
            raise ValueError("incompatible learner recovery")
        self.agent.load_state_dict(state["model"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.updates = state["updates"]
        restore_rng(state["rng"])

    def update(self, batch, *, max_optimizer_steps):
        if (
            not batch.admissible
            or batch.policy_version != self.parent_version
            or batch.likelihood_contract != CONTRACT
        ):
            raise ValueError("wrong behavior policy or invalid batch")
        if type(max_optimizer_steps) is not int or max_optimizer_steps < 1:
            raise ValueError("positive update budget required")
        device = next(self.agent.parameters()).device
        rows = batch.transitions
        c = self.config
        values = torch.as_tensor(
            np.stack([x.decision.learner_context["value"] for x in rows]), device=device
        )
        rewards = torch.as_tensor(
            np.stack([x.step.observation["reward_components"] for x in rows]), device=device
        )
        scalar = torch.as_tensor([x.step.reward for x in rows], device=device)
        if not torch.allclose(rewards.sum(-1), scalar, atol=1e-6, rtol=1e-5):
            raise ValueError("reward decomposition not conserved")
        terminals = torch.tensor([x.step.terminated for x in rows], device=device)
        truncated = torch.tensor([x.step.truncated for x in rows], device=device)
        with torch.no_grad():
            bootstrap = (
                self.agent.model(*tensors(batch.boundary_observation, device))[1][0, 0]
                if not terminals[-1]
                else torch.zeros_like(values[-1])
            )
        advantages, returns = decomposed_gae(
            rewards, values, bootstrap, terminals, truncated, c.gamma, c.gae_lambda
        )
        old_logp = torch.tensor([x.decision.log_probability for x in rows], device=device)
        if not all(torch.isfinite(x).all() for x in (values, rewards, returns, old_logp)):
            raise ValueError("nonfinite rollout")
        # Re-evaluate actual native physical actions before any update. Never fabricate likelihoods.
        with torch.no_grad():
            errors = []
            for i, row in enumerate(rows):
                action = torch.as_tensor(row.decision.action, device=device).view(1, 1, 2)
                errors.append(
                    (policy(self.agent, row.observation, action)[1][0, 0] - old_logp[i]).abs()
                )
            if torch.stack(errors).max() > 1e-5:
                raise ValueError("behavior-policy likelihood mismatch")
        executed = 0
        metrics = []
        for epoch in range(c.epochs):
            order = torch.randperm(len(rows)).tolist()
            for start in range(0, len(rows), c.minibatch_size):
                if executed >= max_optimizer_steps:
                    break
                ids = order[start : start + c.minibatch_size]
                adv = advantages[ids].sum(-1)
                adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
                self.optimizer.zero_grad(set_to_none=True)
                sums = np.zeros(4)
                # Per-sample accumulation preserves variable native token counts without repadding.
                for j, i in enumerate(ids):
                    action = torch.as_tensor(rows[i].decision.action, device=device).view(1, 1, 2)
                    _, logp, entropy, value = policy(self.agent, rows[i].observation, action)
                    ratio = (logp[0, 0] - old_logp[i]).exp()
                    actor = -torch.minimum(
                        ratio * adv[j], ratio.clamp(1 - c.policy_clip, 1 + c.policy_clip) * adv[j]
                    )
                    v = value[0, 0]
                    clipped = values[i] + (v - values[i]).clamp(-c.value_clip, c.value_clip)
                    critic = (
                        0.5
                        * torch.maximum(
                            (v - returns[i]).square(), (clipped - returns[i]).square()
                        ).sum()
                    )
                    loss = (
                        actor + c.value_coefficient * critic - c.entropy_coefficient * entropy[0, 0]
                    ) / len(ids)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("nonfinite PPO loss")
                    loss.backward()
                    sums += np.array(
                        [
                            float(loss.detach()),
                            float(actor.detach()),
                            float(critic.detach()),
                            float(entropy[0, 0].detach()),
                        ]
                    ) / np.array([1, len(ids), len(ids), len(ids)])
                norm = torch.nn.utils.clip_grad_norm_(
                    self.agent.parameters(), c.gradient_clip, error_if_nonfinite=True
                )
                self.optimizer.step()
                self.scheduler.step()
                self.updates += 1
                executed += 1
                metrics.append(
                    dict(
                        epoch=epoch,
                        loss=sums[0],
                        policy_loss=sums[1],
                        value_loss=sums[2],
                        entropy=sums[3],
                        gradient_norm=float(norm),
                        optimizer_step=self.updates,
                    )
                )
            if executed >= max_optimizer_steps:
                break
        self.last_metrics = metrics
        self.output.mkdir(parents=True, exist_ok=True)
        target = self.output / f"update-{self.updates:06d}.pt"
        if target.exists():
            raise FileExistsError(target)
        temporary = target.with_suffix(".tmp")
        # Native inference artifact is deliberately separate from trusted-pickle recovery state.
        torch.save(
            {
                "format": "driverl_weights_only",
                "update": self.updates,
                "model_state_dict": self.agent.state_dict(),
            },
            temporary,
        )
        temporary.replace(target)
        return {
            "candidate": str(target),
            "optimizer_steps": executed,
            "metrics": metrics,
            "promoted": False,
        }


def advance_training_generation(collector, learner, version):
    """Adopt the updated policy for TRAINING collection, not evaluation/deployment."""
    from driversi.training.online_rl.online import RolloutSession

    if collector.unresolved_step or not version or version == collector.actor.policy_version:
        raise ValueError("invalid generation boundary")
    collector.actor.agent.load_state_dict(learner.agent.state_dict(), strict=True)
    collector.actor.policy_version = version
    learner.parent_version = version
    successor = RolloutSession(
        collector.environment, collector.actor, memory=collector.memory, protected_logs=[]
    )
    successor.ended = collector.ended
    successor.environment_step_calls = collector.environment_step_calls
    successor.confirmed_transitions = collector.confirmed_transitions
    return successor
