"""Publish the installed Anima performance runtime identity."""

from __future__ import annotations

import torch


def performance_environment(device: torch.device) -> dict[str, object]:
    """Return exact runtime and pinned adapter execution properties."""

    properties = torch.cuda.get_device_properties(device)
    return {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": properties.name,
        "gpu_total_memory_bytes": properties.total_memory,
        "device": str(device),
        "model_dtype": "torch.bfloat16",
        "adapter_rank": 32,
        "adapter_target_count": 448,
        "attention_backend": "attention_pytorch",
    }
