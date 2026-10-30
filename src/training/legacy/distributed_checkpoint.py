"""Atomic single-host checkpoint including every rank's sampler and random state."""
import os
import uuid
from pathlib import Path

import torch
import torch.distributed as dist

from driversi.training.legacy.distribution import parameter_digest


def save_distributed(path, context, model, optimizer, collector, index, total, config, *, optimizer_updates=None):
    def snapshot():
        return dict(collector=collector.snapshot(), torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state(context.device)
                    if context.device.type == "cuda" else None,
                    model_digest=parameter_digest(model))
    local = context.stage("checkpoint snapshot", snapshot)
    states = [None] * context.size if context.rank == 0 else None
    dist.gather_object(local, states, dst=0)

    def write():
        if len({state["model_digest"] for state in states}) != 1:
            raise ValueError("rank model states diverged; checkpoint refused")
        payload = dict(schema_version=1, kind="distributed-ppo", config=config,
                       checkpoint_id=uuid.uuid4().hex,
                       model=model.state_dict(), optimizer=optimizer.state_dict(),
                       rank_states=states, update=index, valid_transitions=total)
        if optimizer_updates is not None:
            payload["optimizer_updates"] = optimizer_updates
        target = Path(path)
        temporary = target.with_suffix(".pending")
        with temporary.open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(target)
    context.root("checkpoint commit", write)


def restore_distributed(path, context, model, optimizer, collector, config, *, return_counters=False):
    def restore():
        data = torch.load(path, map_location="cpu", weights_only=False)
        if (data.get("kind") != "distributed-ppo" or data.get("schema_version") != 1
                or data["config"] != config or len(data["rank_states"]) != context.size):
            raise ValueError("distributed checkpoint configuration/world-size mismatch")
        if not isinstance(data.get("checkpoint_id"), str) or not data["checkpoint_id"]:
            raise ValueError("checkpoint has no committed identity")
        updates = data.get("optimizer_updates", data["update"])
        if type(updates) is not int or not 0 <= updates <= data["update"]:
            raise ValueError("Invalid checkpoint optimizer counter")
        state = data["rank_states"][context.rank]
        collector.restore(state["collector"])
        model.load_state_dict(data["model"])
        optimizer.load_state_dict(data["optimizer"])
        if parameter_digest(model) != state["model_digest"]:
            raise ValueError("checkpoint model digest mismatch")
        torch.set_rng_state(state["torch_rng"])
        if context.device.type == "cuda":
            if state["cuda_rng"] is None:
                raise ValueError("checkpoint has no rank CUDA state")
            torch.cuda.set_rng_state(state["cuda_rng"], context.device)
        return (data["update"], data["valid_transitions"], data["checkpoint_id"],
                state["model_digest"], updates)
    identity = context.stage("checkpoint restore", restore)
    identities = [None] * context.size
    dist.all_gather_object(identities, identity)
    if any(item != identity for item in identities):
        raise ValueError("ranks loaded different checkpoint identities or counters")
    return identity[:2] + (identity[4],) if return_counters else identity[:2]
