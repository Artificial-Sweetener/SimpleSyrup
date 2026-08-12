"""Verify focused standard-UNet Attention Coupling family policy."""

from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar
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
from simple_syrup.domain.regional_lora_plan import (
    EMPTY_REGIONAL_LORA_PLAN,
    RegionalLoraAdapterIdentity,
    RegionalLoraAdapterPlan,
    RegionalLoraBranch,
    RegionalLoraPlan,
    RegionalLoraScheduleBoundary,
)
from simple_syrup.domain.regional_mask_bank import RegionalMaskBank
from simple_syrup.runtime.attention_coupling.unet_attention_state import (
    StandardUnetAttentionState,
)
from simple_syrup.runtime.attention_coupling.unet_context import (
    STANDARD_UNET_REGIONAL_CONTEXT_VALIDATOR,
)
from simple_syrup.runtime.regional_lora_plan_adapter import RegionalLoraPlanAdaptation
from simple_syrup.services.unet_attention_coupling_model_family import (
    StandardUnetAttentionCouplingModelFamily,
)


class _Backend:
    """Capture the concrete standard-UNet state supplied for derivation."""

    calls: ClassVar[list[dict[str, object]]] = []

    def derive(self, **kwargs: object) -> object:
        """Return a recognizable derived model container."""

        type(self).calls.append(kwargs)
        return SimpleNamespace(model="derived-unet")


def test_unet_family_builds_shared_state_and_derives_paired_backend() -> None:
    """Bind plan, strength, diagnostics, validator, and backend exactly once."""

    family = StandardUnetAttentionCouplingModelFamily()
    plan = _plan()
    adaptation = RegionalLoraPlanAdaptation(EMPTY_REGIONAL_LORA_PLAN, ())
    original_backend = family.backend_class
    type(family).backend_class = _Backend  # type: ignore[assignment]
    _Backend.calls = []
    try:
        family.validate_latent(torch.zeros(2, 4, 8, 8))
        family.validate_adaptation(adaptation)
        derived = family.derive(
            model="model",
            processed_plan=plan,
            adaptation=adaptation,
            region_strengths=(0.75,),
            latent_batch_size=2,
        )
    finally:
        type(family).backend_class = original_backend

    assert family.context_validator is STANDARD_UNET_REGIONAL_CONTEXT_VALIDATOR
    assert derived == "derived-unet"
    assert len(_Backend.calls) == 1
    state = _Backend.calls[0]["state"]
    assert isinstance(state, StandardUnetAttentionState)
    assert state.plan is plan
    assert state.region_strengths == (0.75,)
    assert state.diagnostics.mask_bank is plan.mask_bank
    assert state.diagnostics.backend.endswith(".UNetModel")


def test_unet_family_rejects_regional_model_hooks_before_derivation() -> None:
    """Fail closed instead of silently omitting a regional model-side adapter."""

    family = StandardUnetAttentionCouplingModelFamily()
    adaptation = _regional_lora_adaptation()
    _Backend.calls = []

    with pytest.raises(ValueError, match="1 adapter use"):
        family.validate_adaptation(adaptation)
    with pytest.raises(ValueError, match="1 adapter use"):
        family.derive(
            model="model",
            processed_plan=_plan(),
            adaptation=adaptation,
            region_strengths=(1.0,),
            latent_batch_size=1,
        )

    assert _Backend.calls == []


@pytest.mark.parametrize(
    "samples",
    [torch.zeros(1, 4, 1, 8, 8), torch.zeros(1, 4, 2, 8, 8)],
)
def test_unet_family_rejects_non_bchw_latents(samples: torch.Tensor) -> None:
    """Keep standard image latents distinct from temporal Anima layouts."""

    with pytest.raises(ValueError, match="BxCxHxW"):
        StandardUnetAttentionCouplingModelFamily().validate_latent(samples)


def _regional_lora_adaptation() -> RegionalLoraPlanAdaptation:
    """Return one typed regional model-side adapter use."""

    adapter = RegionalLoraAdapterPlan(
        RegionalLoraAdapterIdentity("private-adapter-token"),
        0,
        0,
        RegionalLoraBranch.POSITIVE,
        1.0,
        (RegionalLoraScheduleBoundary(0.0, 1.0, 1.0, 0),),
    )
    return RegionalLoraPlanAdaptation(RegionalLoraPlan((adapter,)), (object(),))


def _plan() -> ProcessedRegionalAttentionPlan:
    """Return one minimal standard-UNet processed plan."""

    masks = torch.ones(1, 8, 8)
    return ProcessedRegionalAttentionPlan(
        ProcessedRegionalAttentionBranch(
            _context(0, None, 1.0),
            (_context(1, 0, 2.0),),
        ),
        ProcessedRegionalAttentionBranch(
            _context(0, None, -1.0),
            (_context(1, 0, -2.0),),
        ),
        RegionalMaskBank(masks, masks.clone(), 8, 8),
        EMPTY_REGIONAL_LORA_PLAN,
    )


def _context(
    conditioning_index: int,
    region_index: int | None,
    value: float,
) -> ProcessedRegionalAttentionContext:
    """Return one always-active SD-like context."""

    return ProcessedRegionalAttentionContext(
        conditioning_index,
        region_index,
        (
            ProcessedRegionalAttentionEntry(
                0,
                uuid4(),
                ConditioningScheduleRange(None, None, None, None),
                torch.full((1, 77, 768), value),
                1.0,
            ),
        ),
    )
