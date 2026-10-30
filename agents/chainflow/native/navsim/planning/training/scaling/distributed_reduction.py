from __future__ import annotations

import torch
import torch.distributed as dist


def global_masked_mean(
    values: torch.Tensor,
    valid: torch.Tensor,
    *,
    enabled: bool = False,
) -> torch.Tensor:
    """Mean over valid elements with DDP-correct gradients and zero-count safety.

    DDP averages gradients after backward. Scaling each rank's differentiable
    numerator by ``world_size / global_count`` therefore reproduces the gradient
    of a single-process mean over the globally concatenated valid elements.
    Every rank must call this function in the same order, including ranks with no
    valid elements.
    """

    valid_bool = valid.to(device=values.device, dtype=torch.bool)
    if valid_bool.shape != values.shape:
        valid_bool = torch.broadcast_to(valid_bool, values.shape)
    local_sum = torch.where(valid_bool, values, torch.zeros_like(values)).sum()
    # Keep counts in fp32 even under mixed precision; a global element count can
    # exceed fp16's finite integer range for deep multi-stage supervision.
    local_count = valid_bool.sum(dtype=torch.float32).detach()

    distributed = enabled and dist.is_available() and dist.is_initialized()
    if not distributed:
        return local_sum / local_count.clamp_min(1.0)

    global_count = local_count.clone()
    global_sum = local_sum.detach().clone()
    dist.all_reduce(global_count, op=dist.ReduceOp.SUM)
    dist.all_reduce(global_sum, op=dist.ReduceOp.SUM)
    denominator = global_count.clamp_min(1.0)
    loss_for_gradient = local_sum * float(dist.get_world_size()) / denominator
    global_value = global_sum / denominator
    return loss_for_gradient + (global_value - loss_for_gradient.detach())
