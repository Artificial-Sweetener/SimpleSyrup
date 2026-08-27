# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Calculate observational attention logits at model-native CUDA precision."""

from __future__ import annotations

import math

import torch


def scaled_attention_logits(
    query: torch.Tensor,
    key: torch.Tensor,
) -> torch.Tensor:
    """Return FP32 logits using native CUDA tensor-core input precision."""

    if query.ndim != 4 or key.ndim != 4:
        raise ValueError("Attention logit inputs must be BHQD tensors.")
    if (
        int(query.shape[0]) != int(key.shape[0])
        or int(query.shape[1]) != int(key.shape[1])
        or int(query.shape[-1]) != int(key.shape[-1])
    ):
        raise ValueError("Attention logit batch, head, and channel axes must align.")
    native_cuda = (
        query.device.type == "cuda"
        and key.device == query.device
        and query.dtype == key.dtype
        and query.dtype in (torch.float16, torch.bfloat16)
    )
    left = query if native_cuda else query.float()
    right = key if native_cuda else key.float()
    logits = torch.einsum("bhqd,bhkd->bhqk", left, right)
    return logits.float() / math.sqrt(float(query.shape[-1]))
