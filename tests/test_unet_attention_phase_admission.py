# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prove phased self-attention belongs to standard Attention Coupling."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import torch
from comfy.model_patcher import ModelPatcher

from simple_syrup.domain.conditioning_schedule import (
    UNBOUNDED_CONDITIONING_SCHEDULE,
)
from simple_syrup.domain.processed_regional_attention import (
    ProcessedRegionalAttentionBranch,
    ProcessedRegionalAttentionContext,
    ProcessedRegionalAttentionEntry,
    ProcessedRegionalAttentionPlan,
)
from simple_syrup.domain.regional_lora_plan import (
    EMPTY_REGIONAL_LORA_PLAN,
)
from simple_syrup.domain.regional_mask_bank import RegionalMaskBank
from simple_syrup.runtime.attention_coupling.unet import StandardUnetAttentionBackend
from simple_syrup.runtime.attention_coupling.unet_attention_state import (
    StandardUnetAttentionState,
)
from simple_syrup.runtime.regional_attention_diagnostics import (
    RegionalAttentionDiagnosticsBuilder,
)
from simple_syrup.runtime.regional_lora.standard_unet_native_admission import (
    StandardUnetNativeLoraAdmission,
)
from simple_syrup.runtime.regional_lora_plan_adapter import RegionalLoraPlanAdaptation


def test_prompt_only_backend_preserves_reference_self_attention() -> None:
    """Leave attn1 unmodified when no model or regional adapter requires it."""

    built = StandardUnetAttentionBackend().derive(
        model=_patcher(),
        state=_state(),
        admission=StandardUnetNativeLoraAdmission(
            RegionalLoraPlanAdaptation(EMPTY_REGIONAL_LORA_PLAN, ()),
            None,
        ),
    )
    derived: Any = built.model
    patches = derived.model_options["transformer_options"]["patches"]

    assert "attn1_patch" not in patches
    assert "attn1_output_patch" not in patches
    assert len(patches["attn2_patch"]) == 1
    assert len(patches["attn2_output_patch"]) == 1


def _state() -> StandardUnetAttentionState:
    """Return one prompt-only regional plan for backend admission."""

    masks = torch.ones((1, 2, 2))
    plan = ProcessedRegionalAttentionPlan(
        ProcessedRegionalAttentionBranch(
            _context(0, None, 1.0),
            (_context(1, 0, 2.0),),
        ),
        ProcessedRegionalAttentionBranch(
            _context(0, None, -1.0),
            (_context(1, 0, -2.0),),
        ),
        RegionalMaskBank(masks, masks.clone(), 2, 2),
        EMPTY_REGIONAL_LORA_PLAN,
    )
    return StandardUnetAttentionState(
        plan,
        (0.4,),
        RegionalAttentionDiagnosticsBuilder(
            plan.mask_bank,
            backend="comfy.ldm.modules.diffusionmodules.openaimodel.UNetModel",
        ),
    )


def _context(
    conditioning_index: int,
    region_index: int | None,
    value: float,
) -> ProcessedRegionalAttentionContext:
    """Return one finite always-active conditioning context."""

    return ProcessedRegionalAttentionContext(
        conditioning_index,
        region_index,
        (
            ProcessedRegionalAttentionEntry(
                0,
                uuid4(),
                UNBOUNDED_CONDITIONING_SCHEDULE,
                torch.full((1, 2, 4), value),
                1.0,
            ),
        ),
    )


def _patcher() -> ModelPatcher:
    """Return one real CPU ModelPatcher with an empty diffusion module."""

    device = torch.device("cpu")
    base_model = torch.nn.Module()
    base_model.diffusion_model = torch.nn.Linear(1, 1)
    return ModelPatcher(base_model, load_device=device, offload_device=device)
