# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify central capability admission routes shared model preparation."""

from __future__ import annotations

from typing import Any, ClassVar, TypeVar, cast
from uuid import uuid4

import comfy.latent_formats
import comfy.model_base
import comfy.model_patcher
import pytest
import torch
from comfy.ldm.anima.model import Anima as AnimaDiffusionModel
from comfy.ldm.modules.diffusionmodules.openaimodel import UNetModel

from simple_syrup.domain.conditioning_batch import ConditioningBatch
from simple_syrup.domain.conditioning_schedule import ConditioningScheduleRange
from simple_syrup.domain.processed_regional_attention import (
    ProcessedRegionalAttentionBranch,
    ProcessedRegionalAttentionContext,
    ProcessedRegionalAttentionEntry,
    ProcessedRegionalAttentionPlan,
)
from simple_syrup.domain.regional_attention_execution import (
    RegionalAttentionExecutionMode,
)
from simple_syrup.domain.regional_lora_plan import EMPTY_REGIONAL_LORA_PLAN
from simple_syrup.runtime.attention_coupling.anima_context import (
    ANIMA_REGIONAL_CONTEXT_VALIDATOR,
)
from simple_syrup.runtime.attention_coupling.unet_context import (
    STANDARD_UNET_REGIONAL_CONTEXT_VALIDATOR,
)
from simple_syrup.runtime.regional_lora_plan_adapter import RegionalLoraPlanAdaptation
from simple_syrup.services.anima_attention_coupling_model_family import (
    AnimaAttentionCouplingModelFamily,
)
from simple_syrup.services.attention_coupling_model_preparation_service import (
    AttentionCouplingModelPreparationService,
)
from simple_syrup.services.attention_coupling_preparation_service import (
    AttentionCouplingPreparation,
)
from simple_syrup.services.attention_coupling_sampling_service import (
    AttentionCouplingSamplingService,
)
from simple_syrup.services.contextual_attention_coupling_sampling_service import (
    ContextualAttentionCouplingSamplingService,
)
from simple_syrup.services.tiled_attention_coupling_sampling_service import (
    TiledAttentionCouplingSamplingService,
)

ModuleType = TypeVar("ModuleType", bound=torch.nn.Module)


class _LoraAdapter:
    """Return empty regional model-adapter state for routing tests."""

    calls: ClassVar[list[tuple[object, object]]] = []

    def adapt(self, plan: object, *, model: object) -> RegionalLoraPlanAdaptation:
        """Record one model-family-independent adaptation call."""

        type(self).calls.append((plan, model))
        return RegionalLoraPlanAdaptation(EMPTY_REGIONAL_LORA_PLAN, ())


class _ModelLoader:
    """Record model residency without allocating uninitialized model weights."""

    calls: ClassVar[list[object]] = []

    def load(self, model: object) -> None:
        """Record one official loading boundary invocation."""

        type(self).calls.append(model)


class _ConditioningProcessor:
    """Return model-ready contexts while observing family validator selection."""

    context_dimension: ClassVar[int] = 768
    calls: ClassVar[list[dict[str, object]]] = []

    def process(
        self,
        preparation: AttentionCouplingPreparation,
        **kwargs: object,
    ) -> ProcessedRegionalAttentionPlan:
        """Build a processed plan against the exact prepared mask authority."""

        type(self).calls.append(kwargs)
        return ProcessedRegionalAttentionPlan(
            _branch(1.0, 2.0, self.context_dimension),
            _branch(-1.0, -2.0, self.context_dimension),
            preparation.plan.mask_bank,
            preparation.plan.lora_plan,
        )


class _AnimaBackend:
    """Record selected Anima derivation without discovering a real module graph."""

    calls: ClassVar[list[dict[str, object]]] = []

    def derive(self, **kwargs: object) -> object:
        """Return a recognizable Anima model container."""

        type(self).calls.append(kwargs)
        return type("BuiltAnima", (), {"model": "derived-anima"})()


def test_every_spatial_mode_uses_the_same_central_preparation_owner() -> None:
    """Prevent full, tiled, or Contextual sampling from selecting models itself."""

    assert (
        AttentionCouplingSamplingService.model_preparation_service_class
        is AttentionCouplingModelPreparationService
    )
    assert (
        TiledAttentionCouplingSamplingService.model_preparation_service_class
        is AttentionCouplingModelPreparationService
    )
    assert (
        ContextualAttentionCouplingSamplingService.model_preparation_service_class
        is AttentionCouplingModelPreparationService
    )


@pytest.mark.parametrize(
    ("base_type", "latent_format", "context_dimension"),
    [
        (comfy.model_base.BaseModel, comfy.latent_formats.SD15(), 768),
        (comfy.model_base.SDXL, comfy.latent_formats.SDXL(), 2048),
        (comfy.model_base.SDXLRefiner, comfy.latent_formats.SDXL(), 2048),
    ],
)
def test_central_admission_routes_standard_unet_to_production_backend(
    base_type: type[torch.nn.Module],
    latent_format: object,
    context_dimension: int,
) -> None:
    """Derive real paired-patch children for every admitted SD/SDXL wrapper."""

    source = _patcher(
        base_type=base_type,
        diffusion_type=UNetModel,
        latent_format=latent_format,
    )
    originals = _install_fakes()
    _reset_calls(context_dimension=context_dimension)
    try:
        output = AttentionCouplingModelPreparationService().prepare(
            model=source,
            positive=ConditioningBatch((_conditioning(1.0), _conditioning(2.0))),
            negative=ConditioningBatch((_conditioning(-1.0), _conditioning(-2.0))),
            region_masks=torch.ones(1, 8, 8),
            regional_prompt_weight=0.75,
            region_mask_feather=0,
            latent_image={"samples": torch.zeros(1, 4, 8, 8)},
            execution_mode=RegionalAttentionExecutionMode.FULL,
        )
    finally:
        _restore_fakes(originals)

    derived = cast(Any, output.model)
    assert derived.parent is source
    patches = derived.model_options["transformer_options"]["patches"]
    assert len(patches["attn2_patch"]) == 1
    assert len(patches["attn2_output_patch"]) == 1
    assert source.model_options["transformer_options"].get("patches") is None
    assert _ModelLoader.calls == [source]
    assert (
        _ConditioningProcessor.calls[0]["context_validator"]
        is STANDARD_UNET_REGIONAL_CONTEXT_VALIDATOR
    )
    assert _AnimaBackend.calls == []


def test_central_admission_preserves_anima_family_route() -> None:
    """Retain the existing single-frame validator and full-surface backend."""

    source = _patcher(
        base_type=comfy.model_base.Anima,
        diffusion_type=AnimaDiffusionModel,
        latent_format=comfy.latent_formats.Wan21(),
    )
    originals = _install_fakes()
    _reset_calls(context_dimension=1024)
    try:
        output = AttentionCouplingModelPreparationService().prepare(
            model=source,
            positive=ConditioningBatch((_conditioning(1.0), _conditioning(2.0))),
            negative=ConditioningBatch((_conditioning(-1.0), _conditioning(-2.0))),
            region_masks=torch.ones(1, 8, 8),
            regional_prompt_weight=0.5,
            region_mask_feather=0,
            latent_image={"samples": torch.zeros(1, 16, 1, 8, 8)},
            execution_mode=RegionalAttentionExecutionMode.FULL,
        )
    finally:
        _restore_fakes(originals)

    assert output.model == "derived-anima"
    assert _ModelLoader.calls == [source]
    assert (
        _ConditioningProcessor.calls[0]["context_validator"]
        is ANIMA_REGIONAL_CONTEXT_VALIDATOR
    )
    assert len(_AnimaBackend.calls) == 1
    assert _AnimaBackend.calls[0]["region_strengths"] == (0.5,)


def test_cross_family_model_rejection_precedes_loading_and_mutation() -> None:
    """Reject an SD wrapper with SDXL latent state before lower-layer work."""

    source = _patcher(
        base_type=comfy.model_base.BaseModel,
        diffusion_type=UNetModel,
        latent_format=comfy.latent_formats.SDXL(),
    )
    originals = _install_fakes()
    _reset_calls(context_dimension=768)
    try:
        with pytest.raises(ValueError, match="does not support this model combination"):
            AttentionCouplingModelPreparationService().prepare(
                model=source,
                positive=ConditioningBatch((_conditioning(1.0), _conditioning(2.0))),
                negative=ConditioningBatch((_conditioning(-1.0), _conditioning(-2.0))),
                region_masks=torch.ones(1, 8, 8),
                regional_prompt_weight=1.0,
                region_mask_feather=0,
                latent_image={"samples": torch.zeros(1, 4, 8, 8)},
                execution_mode=RegionalAttentionExecutionMode.FULL,
            )
    finally:
        _restore_fakes(originals)

    assert _LoraAdapter.calls == []
    assert _ModelLoader.calls == []
    assert _ConditioningProcessor.calls == []
    assert source.wrappers == {}
    assert source.model_options["transformer_options"].get("patches") is None


def _branch(
    base_value: float,
    region_value: float,
    context_dimension: int,
) -> ProcessedRegionalAttentionBranch:
    """Return one processed branch with one regional context."""

    return ProcessedRegionalAttentionBranch(
        _context(0, None, base_value, context_dimension),
        (_context(1, 0, region_value, context_dimension),),
    )


def _context(
    conditioning_index: int,
    region_index: int | None,
    value: float,
    context_dimension: int,
) -> ProcessedRegionalAttentionContext:
    """Return one always-active model-consumed context."""

    return ProcessedRegionalAttentionContext(
        conditioning_index,
        region_index,
        (
            ProcessedRegionalAttentionEntry(
                0,
                uuid4(),
                ConditioningScheduleRange(None, None, None, None),
                torch.full((1, 2, context_dimension), value),
                1.0,
            ),
        ),
    )


def _conditioning(value: float) -> list[list[object]]:
    """Return one raw global or regional conditioning entry."""

    return [[torch.full((1, 2, 3), value), {}]]


def _patcher(
    *,
    base_type: type[torch.nn.Module],
    diffusion_type: type[torch.nn.Module],
    latent_format: object,
) -> comfy.model_patcher.ModelPatcher:
    """Build a real patcher around exact allocation-free installed types."""

    base_model = _empty_module(base_type)
    dynamic_base = cast(Any, base_model)
    dynamic_base.diffusion_model = _empty_module(diffusion_type)
    dynamic_base.latent_format = latent_format
    dynamic_base.device = torch.device("cpu")
    return comfy.model_patcher.ModelPatcher(
        base_model,
        load_device=torch.device("cpu"),
        offload_device=torch.device("cpu"),
        size=1,
    )


def _empty_module(module_type: type[ModuleType]) -> ModuleType:
    """Create one exact torch module without allocating its real weights."""

    module = module_type.__new__(module_type)
    torch.nn.Module.__init__(module)
    return module


def _install_fakes() -> tuple[type[Any], ...]:
    """Replace only external loading/processing boundaries and Anima discovery."""

    service = AttentionCouplingModelPreparationService
    originals = (
        service.lora_adapter_class,
        service.model_loader_class,
        service.conditioning_processor_class,
        AnimaAttentionCouplingModelFamily.backend_class,
    )
    service.lora_adapter_class = _LoraAdapter  # type: ignore[assignment]
    service.model_loader_class = _ModelLoader  # type: ignore[assignment]
    service.conditioning_processor_class = _ConditioningProcessor  # type: ignore[assignment]
    AnimaAttentionCouplingModelFamily.backend_class = _AnimaBackend  # type: ignore[assignment]
    return originals


def _restore_fakes(originals: tuple[type[Any], ...]) -> None:
    """Restore every patched external collaborator."""

    service = AttentionCouplingModelPreparationService
    service.lora_adapter_class = originals[0]
    service.model_loader_class = originals[1]
    service.conditioning_processor_class = originals[2]
    AnimaAttentionCouplingModelFamily.backend_class = originals[3]


def _reset_calls(*, context_dimension: int) -> None:
    """Reset ordered observations for one isolated routing case."""

    _LoraAdapter.calls = []
    _ModelLoader.calls = []
    _ConditioningProcessor.calls = []
    _ConditioningProcessor.context_dimension = context_dimension
    _AnimaBackend.calls = []
