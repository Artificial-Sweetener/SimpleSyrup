# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Adapters for ComfyUI differential denoise-mask behavior."""

from __future__ import annotations

from importlib import import_module
from typing import Any


def has_denoise_mask_function(model: Any) -> bool:
    """Return whether a model patcher already has denoise-mask behavior."""

    options = getattr(model, "model_options", {})
    return (
        isinstance(options, dict) and options.get("denoise_mask_function") is not None
    )


def clone_with_differential_diffusion(model: Any, strength: float = 1.0) -> Any:
    """Return a clone patched with ComfyUI differential denoise masks."""

    if has_denoise_mask_function(model):
        return model
    cloned_model = model.clone()
    install_differential_diffusion(cloned_model, strength=strength)
    return cloned_model


def install_differential_diffusion(model: Any, strength: float = 1.0) -> Any:
    """Install ComfyUI differential denoise-mask behavior on a model patcher."""

    if has_denoise_mask_function(model):
        return model
    set_mask_function = getattr(model, "set_model_denoise_mask_function", None)
    if not callable(set_mask_function):
        raise ValueError("Model does not support differential diffusion denoise masks.")
    differential_diffusion = import_module(
        "comfy_extras.nodes_differential_diffusion"
    ).DifferentialDiffusion
    set_mask_function(
        lambda *args, **kwargs: differential_diffusion.forward(
            *args,
            **kwargs,
            strength=strength,
        )
    )
    return model
