"""Single-host distributed runtime and coordinated stage failures."""
import hashlib
import os
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist


def partition(total, size):
    """Stable partition of a global resource count (including zero workers)."""
    if type(total) is not int or total < 0 or size < 1:
        raise ValueError("invalid resource partition")
    return [total // size + int(rank < total % size) for rank in range(size)]


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    size: int
    device: torch.device

    def stage(self, name, function):
        """All ranks must enter; do not call around rank-dependent collectives."""
        result, failure = None, None
        try:
            result = function()
        except Exception as error:
            failure = f"rank {self.rank}: {type(error).__name__}: {error}"[:2000]
        failures = [None] * self.size
        dist.all_gather_object(failures, failure)
        if any(failures):
            raise RuntimeError(f"distributed stage {name} failed: {failures}")
        return result

    def root(self, name, function):
        return self.stage(name, function if self.rank == 0 else lambda: None)

    def root_with_progress(self, name, function):
        """Root runs function(progress); peers wait only between progress markers.

        Evaluation calls progress after every scene. The process-group timeout
        still bounds one silent segment, including a hung scene, setup or export.
        There is no background heartbeat that could hide a stalled evaluator.
        Callback results stay on root; only bounded status messages are broadcast.
        """
        sequence = 0
        if self.rank == 0:
            transport_failed = False

            def emit(status, failure=None):
                nonlocal sequence, transport_failed
                packet = [(name, sequence, status, failure)]
                try:
                    dist.broadcast_object_list(packet, src=0, device=self.device)
                except BaseException:
                    transport_failed = True
                    raise
                sequence += 1

            def progress(*_ignored):
                emit("progress")

            try:
                result = function(progress)
            except BaseException as error:
                if transport_failed:
                    raise
                failure = f"rank 0: {type(error).__name__}: {error}"[:2000]
                emit("failed", failure)
                raise RuntimeError(f"distributed stage {name} failed: {failure}") from error
            emit("complete")
            return result
        while True:
            packet = [None]
            dist.broadcast_object_list(packet, src=0, device=self.device)
            message = packet[0]
            if (not isinstance(message, tuple) or len(message) != 4
                    or message[0] != name or message[1] != sequence
                    or message[2] not in {"progress", "complete", "failed"}):
                raise RuntimeError(f"Invalid distributed progress message for {name}")
            sequence += 1
            if message[2] == "complete":
                return None
            if message[2] == "failed":
                raise RuntimeError(f"distributed stage {name} failed: {message[3]}")

    def sum(self, values):
        tensor = torch.as_tensor(values, dtype=torch.float64, device=self.device)
        dist.all_reduce(tensor)
        return tensor

    def counts(self, count):
        values = [None] * self.size
        dist.all_gather_object(values, int(count))
        return values


def initialize(timeout_seconds):
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_world = int(os.environ.get("LOCAL_WORLD_SIZE", "0"))
    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if world < 2 or local_world != world or rank != local_rank:
        raise ValueError("distributed PPO requires single-host torchrun with >= 2 ranks")
    if not torch.cuda.is_available() or local_rank >= torch.cuda.device_count():
        raise ValueError("each distributed rank requires its own visible CUDA device")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=timeout_seconds))
    return DistributedContext(rank, world, torch.device("cuda", local_rank))


def parameter_digest(model):
    digest = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def prepare_ddp(model, device, observation_size):
    """Settle DDP's lazy bucket layout before sampling or restoring RNG state.

    Same fixed graph is required as the real actor/critic forward. These passes
    do not step the optimizer or represent environment transitions.
    """
    from torch.nn.parallel import DistributedDataParallel
    wrapped = DistributedDataParallel(
        model, device_ids=[device.index] if device.type == "cuda" else None)
    saved = {key: tensor.detach().clone() for key, tensor in model.state_dict().items()}
    dummy = torch.zeros((1, observation_size), device=device,
                        dtype=next(model.parameters()).dtype)
    for _ in range(3):
        logits, values = wrapped(dummy)
        ((logits.sum() + values.sum()) * 0).backward()
        wrapped.zero_grad(set_to_none=True)
    model.load_state_dict(saved)
    return wrapped
