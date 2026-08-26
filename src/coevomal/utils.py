"""Small shared utilities: RNG seeding and torch device resolution."""

from __future__ import annotations

import random

import numpy as np


def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:  # torch optional at import time
        pass


def resolve_device(device: str = "auto"):
    """Resolve a device string to a torch.device, honouring availability."""
    import torch

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
