# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify collision-safe MODEL derivation for the standard-UNet backend."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest
import torch

from simple_syrup.domain.conditioning_schedule import ConditioningScheduleRange
from simple_syrup.domain.processed_regional_attention import (
    ProcessedRegionalAttentionBranch,
    ProcessedRegionalAttentionContext,
    ProcessedRegionalAttentionEntry,
    ProcessedRegionalAttentionPlan,
)
from simple_syrup.domain.regional_lora_plan import EMPTY_REGIONAL_LORA_PLAN
from simple_syrup.domain.regional_mask_bank import RegionalMaskBank
from simple_syrup.runtime.attention_coupling.unet import (
    StandardUnetAttentionBackend,
)
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


def test_standard_unet_backend_installs_one_paired_patch_on_a_direct_clone() -> None:
    """Preserve source state while returning one direct collision-safe child."""

    source = _patcher()
    state = _state()

    built = StandardUnetAttentionBackend().derive(
        model=source,
        state=state,
        admission=_empty_admission(),
    )
    derived: Any = built.model

    assert derived.parent is source
    assert built.state is state
    assert source.model_options["transformer_options"].get("patches") is None
    patches = derived.model_options["transformer_options"]["patches"]
    assert tuple(patches) == (
        "attn2_patch",
        "attn2_output_patch",
    )
    assert len(patches["attn2_patch"]) == 1
    assert len(patches["attn2_output_patch"]) == 1
    assert (
        patches["attn2_patch"][0].__self__ is patches["attn2_output_patch"][0].__self__
    )
    wrappers = derived.get_wrappers(
        "diffusion_model",
        "simple_syrup.unet_regional_attention_contexts",
    )
    assert len(wrappers) == 1


def test_standard_unet_backend_preserves_global_self_attention() -> None:
    """Keep SDXL scene formation on Comfy's unmodified attn1 execution."""

    built = StandardUnetAttentionBackend().derive(
        model=_patcher(),
        state=_state(),
        admission=_empty_admission(),
    )
    derived: Any = built.model
    patches = derived.model_options["transformer_options"]["patches"]

    assert "attn1_patch" not in patches
    assert "attn1_output_patch" not in patches
    assert len(patches["attn2_patch"]) == 1
    assert len(patches["attn2_output_patch"]) == 1


def test_standard_unet_backend_rejects_existing_attn2_patch_without_mutation() -> None:
    """Fail before deriving partial state when the source patch surface is owned."""

    source = _patcher()

    def existing(output: torch.Tensor, options: dict[str, Any]) -> torch.Tensor:
        """Return the supplied output through a foreign existing patch."""

        del options
        return output

    source.set_model_attn2_output_patch(existing)
    before = source.model_options["transformer_options"]["patches"].copy()
    state = _state()

    with pytest.raises(ValueError, match="output patch is already installed"):
        StandardUnetAttentionBackend().derive(
            model=source,
            state=state,
            admission=_empty_admission(),
        )

    assert source.model_options["transformer_options"]["patches"] == before


def test_standard_unet_backend_preserves_existing_attn1_patch() -> None:
    """Leave a foreign global self-attention owner intact on the derived child."""

    source = _patcher()

    def existing(
        query: torch.Tensor,
        context: torch.Tensor,
        value: torch.Tensor,
        options: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the supplied tensors through a foreign existing patch."""

        del options
        return query, context, value

    source.set_model_attn1_patch(existing)
    built = StandardUnetAttentionBackend().derive(
        model=source,
        state=_state(),
        admission=_empty_admission(),
    )
    derived: Any = built.model
    source_patches = source.model_options["transformer_options"]["patches"]
    derived_patches = derived.model_options["transformer_options"]["patches"]

    assert source_patches == {"attn1_patch": [existing]}
    assert derived_patches["attn1_patch"] == [existing]
    assert "attn1_output_patch" not in derived_patches
    assert len(derived_patches["attn2_patch"]) == 1
    assert len(derived_patches["attn2_output_patch"]) == 1
    assert source.object_patches == {}


def _empty_admission() -> StandardUnetNativeLoraAdmission:
    """Return one prompt-only standard-family admission."""

    adaptation = RegionalLoraPlanAdaptation(EMPTY_REGIONAL_LORA_PLAN, ())
    return StandardUnetNativeLoraAdmission(adaptation, None)


def _state() -> StandardUnetAttentionState:
    """Return one minimal processed standard-UNet plan and state."""

    masks = torch.ones(1, 1, 1)
    plan = ProcessedRegionalAttentionPlan(
        ProcessedRegionalAttentionBranch(
            _processed_context(0, None, 1.0),
            (_processed_context(1, 0, 2.0),),
        ),
        ProcessedRegionalAttentionBranch(
            _processed_context(0, None, -1.0),
            (_processed_context(1, 0, -2.0),),
        ),
        RegionalMaskBank(masks, masks.clone(), 1, 1),
        EMPTY_REGIONAL_LORA_PLAN,
    )
    return StandardUnetAttentionState(
        plan,
        (1.0,),
        RegionalAttentionDiagnosticsBuilder(
            plan.mask_bank,
            backend="torch.nn.modules.linear.Linear",
        ),
    )


def _processed_context(
    conditioning_index: int,
    region_index: int | None,
    value: float,
) -> ProcessedRegionalAttentionContext:
    """Return one always-active processed context."""

    return ProcessedRegionalAttentionContext(
        conditioning_index,
        region_index,
        (
            ProcessedRegionalAttentionEntry(
                0,
                uuid4(),
                ConditioningScheduleRange(None, None, None, None),
                torch.full((1, 1, 1), value),
                1.0,
            ),
        ),
    )


def _patcher() -> Any:
    """Create one real CPU Comfy ModelPatcher."""

    from comfy.model_patcher import ModelPatcher

    device = torch.device("cpu")
    base_model = torch.nn.Module()
    base_model.diffusion_model = torch.nn.Linear(1, 1)
    return ModelPatcher(
        base_model,
        load_device=device,
        offload_device=device,
    )
