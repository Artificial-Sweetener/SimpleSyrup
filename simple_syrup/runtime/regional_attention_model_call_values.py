# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Normalize exact Comfy regional model-call sigma and UUID values."""

from __future__ import annotations

import math
from dataclasses import dataclass
from uuid import UUID

import torch


@dataclass(frozen=True, slots=True)
class RegionalAttentionModelCallValues:
    """Retain optional JSON-safe model-call identity without tensor state."""

    sampling_sigma: float | None
    conditioning_uuids: tuple[str, ...]


def regional_attention_model_call_values(
    transformer_options: object | None,
) -> RegionalAttentionModelCallValues:
    """Normalize available sigma and UUID values from transformer options."""

    if transformer_options is None:
        return RegionalAttentionModelCallValues(None, ())
    if not isinstance(transformer_options, dict):
        raise TypeError("Regional model-call transformer_options must be a dictionary.")
    sigma_value = transformer_options.get("sigmas")
    sigma = None if sigma_value is None else uniform_model_call_sigma(sigma_value)
    uuid_values = transformer_options.get("uuids")
    uuids = () if uuid_values is None else conditioning_uuid_strings(uuid_values)
    return RegionalAttentionModelCallValues(sigma, uuids)


def uniform_model_call_sigma(value: object) -> float:
    """Return one finite uniform sigma from Comfy transformer options."""

    if not isinstance(value, torch.Tensor) or value.numel() < 1:
        raise TypeError("Regional model-call sigmas must be a nonempty tensor.")
    if value.numel() == 1 or all(stride == 0 for stride in value.stride()):
        item = float(value[(0,) * value.ndim].item())
        if not math.isfinite(item):
            raise ValueError("Regional model-call sigmas must be finite.")
        return item
    values = tuple(float(item) for item in value.detach().flatten().cpu().tolist())
    if any(not math.isfinite(item) for item in values):
        raise ValueError("Regional model-call sigmas must be finite.")
    if any(item != values[0] for item in values[1:]):
        raise ValueError("Regional model-call sigmas must be uniform.")
    return values[0]


def conditioning_uuid_strings(value: object) -> tuple[str, ...]:
    """Return canonical UUIDv4 strings in supplied Comfy chunk order."""

    if not isinstance(value, list | tuple):
        raise TypeError("Regional model-call UUIDs must be a list or tuple.")
    normalized: list[str] = []
    for item in value:
        try:
            identity = item if isinstance(item, UUID) else UUID(str(item))
        except (ValueError, TypeError, AttributeError) as error:
            raise ValueError(
                "Regional model-call UUIDs must be valid UUIDs."
            ) from error
        if identity.version != 4:
            raise ValueError("Regional model-call UUIDs must be UUIDv4 values.")
        normalized.append(str(identity))
    return tuple(normalized)
