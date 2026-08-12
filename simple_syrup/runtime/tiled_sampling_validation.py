# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Portions of this file incorporate behavior derived from
# multidiffusion-upscaler-for-automatic1111. See third_party/manifest.toml and
# third_party/NOTICE.md.

"""Validate tiled sampling controls, latents, and conditioning admission."""

from __future__ import annotations

from typing import Any, TypeAlias

import torch

from ..domain.regional_features import (
    EMPTY_REGIONAL_CAPABILITY_ADMISSION,
    RegionalCapabilityAdmission,
    RegionalFeature,
)

Latent: TypeAlias = dict[str, Any]

UNSUPPORTED_CONDITIONING_KEYS = frozenset({"area", "control", "gligen"})


def validate_sampling_controls(
    *,
    steps: int,
    denoise: float,
    latent_tile_width: int,
    latent_tile_height: int,
    latent_tile_batch_size: int,
) -> None:
    """Reject invalid KSampler and tile controls before runtime side effects."""

    if steps < 1:
        raise ValueError("steps must be at least 1.")
    if not 0.0 <= denoise <= 1.0:
        raise ValueError("denoise must be between 0 and 1.")
    if latent_tile_width < 4:
        raise ValueError("latent_tile_width must be at least 4.")
    if latent_tile_height < 4:
        raise ValueError("latent_tile_height must be at least 4.")
    if latent_tile_batch_size < 1:
        raise ValueError("latent_tile_batch_size must be at least 1.")


def validate_latent_samples(
    latent_image: Latent,
    *,
    sampler_label: str,
) -> torch.Tensor:
    """Return validated samples from a ComfyUI latent dictionary."""

    samples = latent_image.get("samples")
    if not isinstance(samples, torch.Tensor):
        raise ValueError("latent samples must be a torch tensor.")
    validate_tensor_shape(samples, sampler_label=sampler_label)
    return samples


def validate_tensor_shape(samples: torch.Tensor, *, sampler_label: str) -> None:
    """Reject unsupported latent tensor shapes before spatial tiling."""

    if getattr(samples, "is_nested", False):
        raise ValueError(
            f"{sampler_label} requires non-nested latent samples shaped "
            "[batch, channels, height, width] or "
            "[batch, channels, 1, height, width]."
        )
    if samples.ndim == 4:
        return
    if samples.ndim == 5 and int(samples.shape[2]) == 1:
        return
    if samples.ndim == 5:
        raise ValueError(
            f"{sampler_label} 5D latent support requires a singleton third "
            "axis shaped [batch, channels, 1, height, width]."
        )
    raise ValueError(
        f"{sampler_label} requires latent samples shaped "
        "[batch, channels, height, width] or "
        "[batch, channels, 1, height, width]."
    )


def reject_unsupported_conditioning(
    conditioning: object,
    *,
    sampler_label: str,
    capability_admission: RegionalCapabilityAdmission = (
        EMPTY_REGIONAL_CAPABILITY_ADMISSION
    ),
) -> None:
    """Reject conditioning that the selected tiled path cannot preserve."""

    if contains_unsupported_conditioning_key(
        conditioning,
        capability_admission=capability_admission,
    ):
        raise ValueError(
            f"{sampler_label} does not support regional conditioning or "
            "ControlNet in the first implementation."
        )


def contains_unsupported_conditioning_key(
    value: object,
    *,
    capability_admission: RegionalCapabilityAdmission = (
        EMPTY_REGIONAL_CAPABILITY_ADMISSION
    ),
) -> bool:
    """Return whether nested conditioning exceeds the tiled support policy."""

    if isinstance(value, dict):
        if any(key in UNSUPPORTED_CONDITIONING_KEYS for key in value):
            return True
        if "mask" in value and (
            not capability_admission.supports(
                RegionalFeature.FULL_CONTEXT_MASKED_CONDITIONING
            )
            or value.get("set_area_to_bounds") is not False
        ):
            return True
        return any(
            contains_unsupported_conditioning_key(
                item,
                capability_admission=capability_admission,
            )
            for item in value.values()
        )
    if isinstance(value, list | tuple):
        return any(
            contains_unsupported_conditioning_key(
                item,
                capability_admission=capability_admission,
            )
            for item in value
        )
    return False
