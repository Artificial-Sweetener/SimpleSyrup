# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify exact Comfy regional adapter source classification."""

from __future__ import annotations

import torch
from comfy.weight_adapter.lora import LoRAAdapter

from simple_syrup.runtime.regional_lora.comfy_adapter_evidence import (
    classify_raw_sources,
)
from simple_syrup.runtime.regional_lora.comfy_adapter_resolution import (
    ComfyAdapterSourceScope,
)


def test_raw_source_classification_preserves_explicit_fallback_and_unused_order() -> (
    None
):
    """Retain loaded-key evidence, identity fallback, and authored source order."""

    explicit = torch.ones(1)
    fallback = torch.full((1,), 2.0)
    unused = torch.full((1,), 3.0)
    raw = {
        "explicit": explicit,
        "fallback": fallback,
        "unused": unused,
    }
    operation = LoRAAdapter({"explicit"}, (explicit, explicit, None, None, None, None))

    entries = classify_raw_sources(
        raw,
        model_weights={
            "diffusion_model.explicit.weight": operation,
            "diffusion_model.fallback.weight": ("diff", (fallback,)),
        },
        clip_weights={},
        vae_weights={},
    )

    assert tuple((entry.source_key, entry.scope) for entry in entries) == (
        ("explicit", ComfyAdapterSourceScope.MODEL),
        ("fallback", ComfyAdapterSourceScope.MODEL),
        ("unused", ComfyAdapterSourceScope.UNUSED),
    )


def test_raw_source_classification_preserves_scope_precedence() -> None:
    """Keep model, text-encoder, VAE, and unused scope priority exact."""

    model_and_clip = torch.ones(1)
    clip_and_vae = torch.full((1,), 2.0)
    vae = torch.full((1,), 3.0)
    unused = torch.full((1,), 4.0)
    raw = {
        "model-and-clip": model_and_clip,
        "clip-and-vae": clip_and_vae,
        "vae": vae,
        "unused": unused,
    }

    entries = classify_raw_sources(
        raw,
        model_weights={"model": ("diff", (model_and_clip,))},
        clip_weights={
            "model": ("diff", (model_and_clip,)),
            "clip": ("diff", (clip_and_vae,)),
        },
        vae_weights={
            "clip": ("diff", (clip_and_vae,)),
            "vae": ("diff", (vae,)),
        },
    )

    assert tuple(entry.scope for entry in entries) == (
        ComfyAdapterSourceScope.MODEL,
        ComfyAdapterSourceScope.TEXT_ENCODER,
        ComfyAdapterSourceScope.VAE,
        ComfyAdapterSourceScope.UNUSED,
    )
