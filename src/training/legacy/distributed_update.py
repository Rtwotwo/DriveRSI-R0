"""PPO with complete unequal local batches and globally weighted gradients."""
import math

import torch
from torch import nn

from driversi.training.sampling.batch import learning_mask
from driversi.training.ppo import objective_terms


def update_distributed(context, model, optimizer, rollout, *, epochs=4, minibatch=512):
    """Minibatch is per rank; exhausted ranks join with zero-weight dummy rows."""
    if epochs < 1 or minibatch < 1:
        raise ValueError("epochs/minibatch must be positive")
    keys = ("observations", "actions", "log_probs", "advantages", "returns")
    def prepare():
        valid = learning_mask(rollout)
        tensors = {key: torch.as_tensor(rollout[key][valid], device=context.device)
                   for key in keys}
        if any(not torch.isfinite(value).all() for value in tensors.values()):
            raise ValueError("non-finite PPO inputs")
        if any(len(value) != len(tensors["actions"]) for value in tensors.values()):
            raise ValueError("inconsistent PPO input lengths")
        return tensors
    tensors = context.stage("PPO inputs", prepare)
    counts = context.counts(len(tensors["actions"]))
    total = sum(counts)
    if total < 2:
        raise ValueError("global PPO batch needs at least two valid transitions")
    adv64 = tensors["advantages"].double()
    mean = context.sum([adv64.sum().item()])[0] / total
    # Centered two-pass variance avoids cancellation for nearly constant advantages.
    variance = context.sum([((adv64 - mean) ** 2).sum().item()])[0] / total
    normalized = ((adv64 - mean) / (variance.sqrt() + 1e-8)).to(tensors["advantages"].dtype)
    tensors["advantages"] = normalized
    steps = max(math.ceil(count / minibatch) for count in counts)
    metrics = torch.zeros(6, dtype=torch.float64, device=context.device)
    model.train()
    for _ in range(epochs):
        order = torch.randperm(counts[context.rank], device=context.device)
        for step in range(steps):
            indices = order[step * minibatch:(step + 1) * minibatch]
            local_count = len(indices)
            active = sum(max(0, min(minibatch, count - step * minibatch)) for count in counts)
            if local_count:
                obs = tensors["observations"][indices]
                actions = tensors["actions"][indices]
                old_log_probs = tensors["log_probs"][indices]
                advantages = tensors["advantages"][indices]
                returns = tensors["returns"][indices]
            else:
                obs = torch.zeros((1, *tensors["observations"].shape[1:]),
                                  dtype=tensors["observations"].dtype, device=context.device)
                actions = torch.zeros(1, dtype=torch.long, device=context.device)
                old_log_probs = advantages = returns = torch.zeros(1, device=context.device)
            logits, values = model(obs)
            terms = objective_terms(logits, values, actions, old_log_probs, advantages, returns)
            policy, value, entropy, kl = (term.sum() for term in terms)
            raw_loss = policy + 0.5 * value - 0.01 * entropy
            finite = context.sum([int(torch.isfinite(raw_loss))])
            if finite.item() != context.size:
                raise FloatingPointError("non-finite distributed PPO loss")
            # DDP averages ranks, so multiply by world size to get the global sample mean.
            weight = context.size / active if local_count else 0.0
            loss = raw_loss * weight
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5, error_if_nonfinite=True)
            optimizer.step()
            if local_count:
                metrics += torch.stack((raw_loss, policy, value, entropy, kl,
                                        raw_loss.new_tensor(local_count))).detach().double()
    import torch.distributed as dist
    dist.all_reduce(metrics)
    values = (metrics[:5] / metrics[5]).cpu().tolist()
    return dict(zip(("loss", "policy_loss", "value_loss", "entropy", "approx_kl"), values),
                batch_valid=total, rank_valid=counts, samples_used=total,
                sample_visits=int(metrics[5].item()), dropped_valid=0,
                optimizer_steps=epochs * steps,
                advantage_mean=mean.item(), advantage_std=variance.sqrt().item())
