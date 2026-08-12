# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Adapters for ComfyUI differential denoise-mask behavior."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .model_patcher_mutations import ModelDenoiseMaskMutation
from .patcher_lifecycle import PATCHER_LIFECYCLE


def has_denoise_mask_function(model: Any) -> bool:
    """Return whether a model patcher already has denoise-mask behavior."""

    options = getattr(model, "model_options", {})
    return (
        isinstance(options, dict) and options.get("denoise_mask_function") is not None
    )


def clone_with_differential_diffusion(model: Any, strength: float = 1.0) -> Any:
    """Return a lifecycle-owned MODEL with differential denoise masks."""

    if has_denoise_mask_function(model):
        return model
    return PATCHER_LIFECYCLE.derive_model(
        model,
        (differential_diffusion_mutation(strength=strength),),
        operation="SimpleSyrup differential diffusion",
    )


def differential_diffusion_mutation(
    strength: float = 1.0,
) -> ModelDenoiseMaskMutation:
    """Build the ComfyUI differential denoise-mask mutation."""

    differential_diffusion = import_module(
        "comfy_extras.nodes_differential_diffusion"
    ).DifferentialDiffusion
    return ModelDenoiseMaskMutation(
        lambda *args, **kwargs: differential_diffusion.forward(
            *args,
            **kwargs,
            strength=strength,
        )
    )
