# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Smoke-test the static regional backend through the installed Anima graph."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, cast

import comfy.ops
import comfy.sampler_helpers
import pytest
import torch
from comfy.ldm.anima.model import Anima
from regional_attention_test_values import single_entry_regions
from torch import nn

from simple_syrup.domain.regional_attention import RegionalAttentionBranch
from simple_syrup.domain.regional_attention_batch import (
    BatchedRegionalAttentionContexts,
    RegionalAttentionChunkBatch,
)
from simple_syrup.domain.regional_lora_plan import (
    RegionalLoraAdapterIdentity,
    RegionalLoraAdapterPlan,
    RegionalLoraBranch,
    RegionalLoraScheduleBoundary,
)
from simple_syrup.domain.regional_mask_bank import RegionalMaskBank
from simple_syrup.runtime.patcher_lifecycle import PATCHER_LIFECYCLE
from simple_syrup.runtime.regional_lora.anima_attention_coupling import (
    anima_attention_coupling_mutations,
)
from simple_syrup.runtime.regional_lora.anima_attention_execution import (
    AnimaRegionalAttentionExecution,
)
from simple_syrup.runtime.regional_lora.anima_composition import (
    AnimaRegionalLoraComposition,
)
from simple_syrup.runtime.regional_lora.anima_execution_scope import (
    AnimaRegionalLoraAdapterExecution,
)
from simple_syrup.runtime.regional_lora.anima_module_surface import (
    ANIMA_MODULE_SURFACE_DISCOVERY,
    AnimaLoraTargetModule,
)
from simple_syrup.runtime.regional_lora.anima_targets import (
    ANIMA_BLOCK_COUNT,
    AnimaLoraAdmission,
    AnimaLoraTarget,
)
from simple_syrup.runtime.regional_lora.execution_cache import (
    ModelCloneLineage,
    RegionalLoraExecutionCache,
)
from simple_syrup.runtime.regional_lora.standard_adapter import StandardLoraTarget


class _AnimaModelRoot(nn.Module):
    """Expose installed Anima under Comfy's diffusion-model patch path."""

    def __init__(self, diffusion_model: Anima) -> None:
        """Retain the exact installed diffusion model."""

        super().__init__()
        self.diffusion_model = diffusion_model


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_static_backend_executes_installed_anima_forward() -> None:
    """Run one complete patched Anima forward on real CUDA tensors."""

    device = torch.device("cuda")
    diffusion_model = _installed_anima(device)
    surface = ANIMA_MODULE_SURFACE_DISCOVERY.discover(diffusion_model)
    source = _patcher(_AnimaModelRoot(diffusion_model))
    composition = _composition(source, surface.lora_targets[0])
    derived = PATCHER_LIFECYCLE.derive_model(
        source,
        anima_attention_coupling_mutations(
            surface,
            composition.attention,
            composition=composition,
        ),
        operation="smoke-test static regional Anima execution",
    )
    derived.patch_model(load_weights=False)
    try:
        model_options = deepcopy(derived.model_options)
        comfy.sampler_helpers.prepare_model_patcher(derived, {}, model_options)
        transformer_options = cast(dict[str, Any], model_options["transformer_options"])
        transformer_options["cond_or_uncond"] = [0]
        transformer_options["sample_sigmas"] = torch.tensor([100.0, 0.0], device=device)
        transformer_options["sigmas"] = torch.tensor([50.0], device=device)
        with torch.no_grad():
            output = diffusion_model(
                torch.zeros((1, 16, 1, 2, 2), device=device, dtype=torch.float16),
                torch.zeros((1,), device=device, dtype=torch.float32),
                torch.zeros((1, 1, 1024), device=device, dtype=torch.float16),
                transformer_options=transformer_options,
            )
    finally:
        derived.unpatch_model(unpatch_weights=False)

    assert output.shape == (1, 16, 1, 2, 2)
    assert output.device.type == "cuda"
    assert output.dtype == torch.float16
    assert composition.executions[0].cache.size == 1
    assert diffusion_model.blocks[0] is surface.blocks[0].block


def _installed_anima(device: torch.device) -> Anima:
    """Construct the exact installed Anima architecture on the selected device."""

    return Anima(
        max_img_h=2,
        max_img_w=2,
        max_frames=1,
        in_channels=16,
        out_channels=16,
        patch_spatial=2,
        patch_temporal=1,
        model_channels=2048,
        num_blocks=ANIMA_BLOCK_COUNT,
        num_heads=16,
        mlp_ratio=4.0,
        crossattn_emb_channels=1024,
        pos_emb_cls="rope3d",
        pos_emb_learnable=False,
        pos_emb_interpolation="crop",
        use_adaln_lora=True,
        adaln_lora_dim=256,
        extra_per_block_abs_pos_emb=False,
        device=device,
        dtype=torch.float16,
        operations=comfy.ops.disable_weight_init,
    )


def _composition(
    model: object,
    descriptor: AnimaLoraTargetModule,
) -> AnimaRegionalLoraComposition:
    """Build one positive regional adapter over the complete one-token canvas."""

    device = next(descriptor.module.parameters()).device
    scalar = torch.ones(())
    adapter = StandardLoraTarget(
        descriptor.target_name,
        scalar.expand(1, descriptor.input_features),
        scalar.expand(descriptor.output_features, 1),
        rank=1,
        input_features=descriptor.input_features,
        output_features=descriptor.output_features,
    )
    attention = AnimaRegionalAttentionExecution(
        BatchedRegionalAttentionContexts(
            1,
            (RegionalAttentionChunkBatch(0, RegionalAttentionBranch.POSITIVE, 0, 1),),
            torch.zeros((1, 1, 1024), device=device, dtype=torch.float16),
            single_entry_regions(
                (torch.zeros((1, 1, 1024), device=device, dtype=torch.float16),)
            ),
        ),
        RegionalMaskBank(torch.ones((1, 2, 2)), torch.ones((1, 2, 2)), 2, 2),
        (1.0,),
    )
    execution = AnimaRegionalLoraAdapterExecution(
        RegionalLoraAdapterPlan(
            RegionalLoraAdapterIdentity("anima-smoke.safetensors"),
            composition_index=0,
            region_index=0,
            branch=RegionalLoraBranch.POSITIVE,
            model_strength=1.0,
            schedule=(RegionalLoraScheduleBoundary(0.0, 100.0, 1.0, 0),),
        ),
        AnimaLoraAdmission(
            (AnimaLoraTarget(descriptor.block_index, descriptor.family, adapter),)
        ),
        attention,
        ModelCloneLineage.from_model(model),
        RegionalLoraExecutionCache(),
    )
    return AnimaRegionalLoraComposition((execution,))


def _patcher(model: nn.Module) -> Any:
    """Create a real Comfy model patcher for clone-local smoke execution."""

    from comfy.model_patcher import ModelPatcher

    device = next(model.parameters()).device
    return ModelPatcher(model, load_device=device, offload_device=device)
