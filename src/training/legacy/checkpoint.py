"""Trusted local optimizer, sampler and RNG continuation."""
import torch


def save_checkpoint(path, model, optimizer, collector, update_index, total, config, *, optimizer_updates=None):
    payload = {"schema_version": 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
               "collector": collector.snapshot(), "torch_rng": torch.get_rng_state(),
               "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
               "update": update_index, "valid_transitions": total, "config": config}
    if optimizer_updates is not None:
        payload["optimizer_updates"] = optimizer_updates
    temporary = path.with_suffix(".pending")
    torch.save(payload, temporary)
    temporary.replace(path)


def restore_checkpoint(path, model, optimizer, collector, config, *, return_counters=False):
    # Only locally produced trusted training checkpoints. Public model inference uses NPZ.
    data = torch.load(path, map_location="cpu", weights_only=False)
    if data["schema_version"] != 1 or data["config"] != config:
        raise ValueError("Training checkpoint configuration mismatch")
    updates = data.get("optimizer_updates", data["update"])
    if type(updates) is not int or not 0 <= updates <= data["update"]:
        raise ValueError("Invalid checkpoint optimizer counter")
    collector.restore(data["collector"])
    model.load_state_dict(data["model"])
    optimizer.load_state_dict(data["optimizer"])
    torch.set_rng_state(data["torch_rng"])
    if data["cuda_rng"]:
        torch.cuda.set_rng_state_all(data["cuda_rng"])
    result = (data["update"], data["valid_transitions"])
    return result + (updates,) if return_counters else result


