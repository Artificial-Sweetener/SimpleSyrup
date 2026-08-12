"""Record deterministic tensor identities without retaining tensor contents."""

from __future__ import annotations

import hashlib

import torch


def snapshot_tensor(tensor: torch.Tensor) -> dict[str, object]:
    """Return shape, source dtype, and canonical float32 content identity."""

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Tensor snapshot input must be a torch.Tensor.")
    canonical = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "float32_sha256": hashlib.sha256(canonical.numpy().tobytes()).hexdigest(),
    }
