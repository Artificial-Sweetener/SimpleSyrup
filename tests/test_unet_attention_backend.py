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


def test_standard_unet_backend_installs_one_paired_patch_on_a_direct_clone() -> None:
    """Preserve source state while returning one direct collision-safe child."""

    source = _patcher()
    state = _state()

    built = StandardUnetAttentionBackend().derive(
        model=source,
        state=state,
    )
    derived: Any = built.model

    assert derived.parent is source
    assert built.state is state
    assert source.model_options["transformer_options"].get("patches") is None
    patches = derived.model_options["transformer_options"]["patches"]
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
        )

    assert source.model_options["transformer_options"]["patches"] == before


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
