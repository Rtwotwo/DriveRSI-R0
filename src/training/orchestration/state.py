"""Single-process reproducible input order and explicit training state receipts."""

import hashlib
import random


def deterministic(seed):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(4)


def rng_state():
    import numpy as np
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng(value):
    import numpy as np
    import torch

    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"])
    torch.cuda.set_rng_state_all(value["cuda"])


def fingerprint(value):
    import numpy as np
    import torch

    h = hashlib.sha256()

    def feed(x):
        if torch.is_tensor(x):
            y = x.detach().cpu().contiguous()
            h.update(str((y.dtype, tuple(y.shape))).encode())
            h.update(y.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x, np.ndarray):
            h.update(str((x.dtype, x.shape)).encode())
            h.update(x.tobytes())
        elif isinstance(x, dict):
            for key in sorted(x, key=str):
                feed(key)
                feed(x[key])
        elif isinstance(x, (list, tuple)):
            for item in x:
                feed(item)
        else:
            h.update(repr(x).encode())

    feed(value)
    return h.hexdigest()


class EpochSampler:
    """Epoch-local permutation; independent of the model noise RNG."""

    def __init__(self, dataset, seed):
        self.dataset, self.seed, self.epoch = dataset, seed, 0

    def __len__(self):
        return len(self.dataset)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        import torch

        gen = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(len(self), generator=gen).tolist())
