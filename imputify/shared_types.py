"""Device types and resolution for torch backends."""

from typing import Literal, TypeAlias

import torch

Device: TypeAlias = Literal["auto", "cpu", "cuda", "mps"]

def resolve_device(device: Device = "auto") -> torch.device:
    """Resolve device string to torch.device with CUDA -> MPS -> CPU fallback.

    Args:
        device: Target device. 'auto' picks the best available.

    Returns:
        Resolved torch.device.
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    return torch.device(device)
