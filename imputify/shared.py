from __future__ import annotations

import random

import numpy as np
import torch


_current_seed: int | None = None


def global_seed() -> int | None:
    """Get the current global seed set by seed_everything()."""
    return _current_seed


def seed_everything(seed: int | None, deterministic: bool = False) -> None:
    """Set global seeds for reproducibility.

    Parameters
    ----------
    seed : Optional[int]
        Seed value. If None, does nothing (non-deterministic behavior).
    deterministic : bool, default=False
        If True, enables deterministic algorithms globally (may reduce performance).
        Works for CUDA, MPS, and CPU backends.
        Note: MPS may still have non-deterministic operations.
        See https://github.com/pytorch/pytorch/issues/97236
    """
    global _current_seed

    if seed is None:
        return

    _current_seed = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

def cleanup_gpu() -> None:
    import gc

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    if torch.mps.is_available():
        torch.mps.empty_cache()
        torch.mps.synchronize()
