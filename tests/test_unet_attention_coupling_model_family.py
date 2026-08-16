# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify focused standard-UNet Attention Coupling family policy."""

from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar
from uuid import uuid4

import pytest
import torch

from simple_syrup.domain.conditioning_batch import ConditioningBatch
from simple_syrup.domain.conditioning_schedule import ConditioningScheduleRange
from simple_syrup.domain.processed_regional_attention import (
    ProcessedRegionalAttentionBranch,
    ProcessedRegionalAttentionContext,
    ProcessedRegionalAttentionEntry,
    ProcessedRegionalAttentionPlan,
)
from simple_syrup.domain.raw_regional_attention import (
    build_raw_regional_attention_plan,
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
from simple_syrup.domain.regional_model_capabilities import RegionalModelFamily
from simple_syrup.runtime.attention_coupling.unet import StandardUnetAttentionBackend
from simple_syrup.runtime.attention_coupling.unet_attention_state import (
    StandardUnetAttentionState,
)
from simple_syrup.runtime.attention_coupling.unet_context import (
    STANDARD_UNET_REGIONAL_CONTEXT_VALIDATOR,
)
from simple_syrup.runtime.regional_lora.standard_unet_native_admission import (
    StandardUnetNativeLoraAdmission,
    StandardUnetNativeLoraAdmissionService,
)
from simple_syrup.runtime.regional_lora_host_payload import RegionalLoraHostPayload
from simple_syrup.runtime.regional_lora_plan_adapter import RegionalLoraPlanAdaptation
from simple_syrup.runtime.regional_model_patch_interop import (
    RegionalModelPatchInteropReport,
    RegionalPreservedModelModifier,
)
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


class _NativeAdmission:
    """Capture native adapter admission exactly once."""

    calls: ClassVar[list[tuple[object, RegionalLoraPlanAdaptation]]] = []

    def admit(
        self,
        model: object,
        adaptation: RegionalLoraPlanAdaptation,
    ) -> StandardUnetNativeLoraAdmission:
        """Return a recognizable typed empty native admission."""

        type(self).calls.append((model, adaptation))
        return StandardUnetNativeLoraAdmission(adaptation, None)


def test_unet_family_selects_native_variant_backend_boundaries() -> None:
    """Make the standard family own native admission and variant execution."""

    family = StandardUnetAttentionCouplingModelFamily()

    assert family.backend_class is StandardUnetAttentionBackend
    assert family.native_admission_class is StandardUnetNativeLoraAdmissionService


def test_unet_family_builds_shared_state_and_derives_paired_backend() -> None:
    """Bind native admission, state, diagnostics, and backend exactly once."""

    family = StandardUnetAttentionCouplingModelFamily()
    plan = _plan()
    adaptation = RegionalLoraPlanAdaptation(EMPTY_REGIONAL_LORA_PLAN, ())
    original_backend = family.backend_class
    original_admission = family.native_admission_class
    type(family).backend_class = _Backend  # type: ignore[assignment]
    type(family).native_admission_class = _NativeAdmission  # type: ignore[assignment]
    _Backend.calls = []
    _NativeAdmission.calls = []
    try:
        family.validate_latent(torch.zeros(2, 4, 8, 8))
        admission = family.admit_adaptation("model", adaptation)
        derived = family.derive(
            model="model",
            processed_plan=plan,
            admission=admission,
            interop_report=_interop_report(),
            region_strengths=(0.75,),
            latent_batch_size=2,
        )
    finally:
        type(family).backend_class = original_backend
        type(family).native_admission_class = original_admission

    assert family.context_validator is STANDARD_UNET_REGIONAL_CONTEXT_VALIDATOR
    assert derived == "derived-unet"
    assert admission.adaptation is adaptation
    assert _NativeAdmission.calls == [("model", adaptation)]
    assert len(_Backend.calls) == 1
    assert _Backend.calls[0]["admission"] is admission
    state = _Backend.calls[0]["state"]
    assert isinstance(state, StandardUnetAttentionState)
    assert state.plan is plan
    assert state.region_strengths == (0.75,)
    assert state.diagnostics.mask_bank is plan.mask_bank
    assert state.diagnostics.backend.endswith(".UNetModel")


def test_unet_family_preserves_prompt_only_base_sampler_conditioning() -> None:
    """Avoid native branches when regional conditioning has no model hooks."""

    positive_base = [[torch.ones((1, 2, 3)), {}]]
    negative_base = [[-torch.ones((1, 2, 3)), {}]]
    masks = torch.ones(1, 8, 8)
    raw_plan = build_raw_regional_attention_plan(
        positive=ConditioningBatch((positive_base, [[torch.ones((1, 2, 3)), {}]])),
        negative=ConditioningBatch((negative_base, [[-torch.ones((1, 2, 3)), {}]])),
        mask_bank=RegionalMaskBank(masks, masks.clone(), 8, 8),
    )

    built = StandardUnetAttentionCouplingModelFamily().prepare_sampler_conditioning(
        raw_plan,
        (1.0,),
    )

    assert built.positive is positive_base
    assert built.negative is negative_base


def test_unet_family_derives_variant_capable_backend_for_prompt_coupling() -> None:
    """Keep standard prompt coupling inside the variant-capable backend."""

    family = StandardUnetAttentionCouplingModelFamily()
    adaptation = RegionalLoraPlanAdaptation(EMPTY_REGIONAL_LORA_PLAN, ())
    admission = StandardUnetNativeLoraAdmission(adaptation, None)
    original_backend = family.backend_class
    type(family).backend_class = _Backend  # type: ignore[assignment]
    _Backend.calls = []
    try:
        derived = family.derive(
            model="native-model",
            processed_plan=_plan(adaptation.plan),
            admission=admission,
            interop_report=_interop_report(),
            region_strengths=(1.0,),
            latent_batch_size=1,
        )
    finally:
        type(family).backend_class = original_backend

    assert derived == "derived-unet"
    assert len(_Backend.calls) == 1


def test_unet_family_requires_matching_family_admission() -> None:
    """Fail closed on foreign or plan-divergent family admission."""

    family = StandardUnetAttentionCouplingModelFamily()
    adaptation = _regional_lora_adaptation()
    _Backend.calls = []

    with pytest.raises(TypeError, match="native admission"):
        family.derive(
            model="model",
            processed_plan=_plan(),
            admission=object(),  # type: ignore[arg-type]
            interop_report=_interop_report(),
            region_strengths=(1.0,),
            latent_batch_size=1,
        )
    with pytest.raises(ValueError, match="must share the same regional LoRA plan"):
        family.derive(
            model="model",
            processed_plan=_plan(adaptation.plan),
            admission=StandardUnetNativeLoraAdmission(
                RegionalLoraPlanAdaptation(EMPTY_REGIONAL_LORA_PLAN, ()),
                None,
            ),
            interop_report=_interop_report(),
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
    return RegionalLoraPlanAdaptation(
        RegionalLoraPlan((adapter,)),
        (RegionalLoraHostPayload.unresolved(object()),),
    )


def _interop_report(
    *modifiers: RegionalPreservedModelModifier,
) -> RegionalModelPatchInteropReport:
    """Return generic standard-family modifier evidence."""

    return RegionalModelPatchInteropReport(
        RegionalModelFamily.STANDARD_UNET,
        tuple(modifiers),
    )


def _plan(
    lora_plan: RegionalLoraPlan = EMPTY_REGIONAL_LORA_PLAN,
) -> ProcessedRegionalAttentionPlan:
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
        lora_plan,
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
