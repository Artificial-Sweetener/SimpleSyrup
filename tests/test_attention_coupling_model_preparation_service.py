# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify model-neutral prepare-once Attention Coupling orchestration."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, ClassVar
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
from simple_syrup.domain.regional_attention_execution import (
    RegionalAttentionExecutionMode,
)
from simple_syrup.domain.regional_lora_plan import EMPTY_REGIONAL_LORA_PLAN
from simple_syrup.runtime.regional_lora_plan_adapter import RegionalLoraPlanAdaptation
from simple_syrup.services.attention_coupling_model_preparation_service import (
    AttentionCouplingModelPreparationService,
)
from simple_syrup.services.attention_coupling_preparation_service import (
    AttentionCouplingPreparation,
)


class _CapabilityService:
    """Return one recognizable immutable admission value."""

    capabilities = object()
    calls: ClassVar[list[dict[str, object]]] = []

    def admit(self, **kwargs: object) -> object:
        """Record admission and return the selected capabilities."""

        type(self).calls.append(kwargs)
        return SimpleNamespace(model_capabilities=self.capabilities)


class _InteropValidator:
    """Record centralized modifier admission without requiring a real patcher."""

    report: ClassVar[object] = object()
    calls: ClassVar[list[tuple[object, ...]]] = []

    def validate(self, model: object, capabilities: object) -> object:
        """Record exact orchestration inputs without changing them."""

        type(self).calls.append((model, capabilities))
        return self.report

    def validate_execution(self, report: object, plan: object, mode: object) -> None:
        """Record the canonical plan and explicit execution mode."""

        type(self).calls.append((report, plan, mode))


class _LoraAdapter:
    """Return an empty aligned regional LoRA adaptation."""

    calls: ClassVar[list[tuple[object, object]]] = []

    def adapt(self, plan: object, *, model: object) -> RegionalLoraPlanAdaptation:
        """Record the raw plan and model owner."""

        type(self).calls.append((plan, model))
        return RegionalLoraPlanAdaptation(EMPTY_REGIONAL_LORA_PLAN, ())


class _PreparationService:
    """Preserve the supplied raw plan and base conditionings."""

    calls: ClassVar[list[object]] = []

    def prepare(self, plan: Any) -> AttentionCouplingPreparation:
        """Record and return one base-only sampler preparation."""

        type(self).calls.append(plan)
        return AttentionCouplingPreparation(
            plan,
            plan.positive.base_conditioning,
            plan.negative.base_conditioning,
        )


class _ConditioningProcessor:
    """Return processed contexts while recording the selected validator."""

    calls: ClassVar[list[dict[str, object]]] = []

    def process(
        self,
        preparation: AttentionCouplingPreparation,
        **kwargs: object,
    ) -> ProcessedRegionalAttentionPlan:
        """Build one processed plan from the prepared mask authority."""

        type(self).calls.append(kwargs)
        return ProcessedRegionalAttentionPlan(
            _branch(1.0, 2.0),
            _branch(-1.0, -2.0),
            preparation.plan.mask_bank,
            preparation.plan.lora_plan,
        )


class _ModelLoader:
    """Record the exact model residency request."""

    calls: ClassVar[list[object]] = []

    def load(self, model: object) -> None:
        """Record one official model-load boundary call."""

        type(self).calls.append(model)


class _ModelFamily:
    """Record family-owned validation and derivation boundaries."""

    validator = object()
    latent_calls: ClassVar[list[torch.Tensor]] = []
    adaptation_calls: ClassVar[list[RegionalLoraPlanAdaptation]] = []
    derive_calls: ClassVar[list[dict[str, object]]] = []
    latent_error: ClassVar[Exception | None] = None
    adaptation_error: ClassVar[Exception | None] = None

    @property
    def context_validator(self) -> object:
        """Return the recognizable family context validator."""

        return self.validator

    def validate_latent(self, samples: torch.Tensor) -> None:
        """Record latent validation and optionally fail before plan work."""

        type(self).latent_calls.append(samples)
        if self.latent_error is not None:
            raise self.latent_error

    def validate_adaptation(self, adaptation: RegionalLoraPlanAdaptation) -> None:
        """Record regional model-adapter admission."""

        type(self).adaptation_calls.append(adaptation)
        if self.adaptation_error is not None:
            raise self.adaptation_error

    def derive(self, **kwargs: object) -> object:
        """Record derivation and return a recognizable model."""

        type(self).derive_calls.append(kwargs)
        return "derived-model"


class _ModelFamilySelector:
    """Return one family adapter from the supplied capability value."""

    calls: ClassVar[list[object]] = []

    def select(self, capabilities: object) -> _ModelFamily:
        """Record exact admitted capabilities and return the test family."""

        type(self).calls.append(capabilities)
        return _ModelFamily()


def test_model_preparation_runs_each_shared_owner_once() -> None:
    """Prepare masks, contexts, model residency, and one selected backend once."""

    service = AttentionCouplingModelPreparationService()
    originals = _install_fakes()
    model_owner = object()
    model = SimpleNamespace(model=model_owner, load_device="cpu")
    positive = ConditioningBatch((_conditioning(1.0), _conditioning(2.0)))
    negative = ConditioningBatch((_conditioning(-1.0), _conditioning(-2.0)))
    samples = torch.zeros((2, 4, 8, 8))
    _reset_calls()
    try:
        output = service.prepare(
            model=model,
            positive=positive,
            negative=negative,
            region_masks=torch.ones((1, 8, 8)),
            regional_prompt_weight=0.75,
            region_mask_feather=0,
            latent_image={"samples": samples},
            execution_mode=RegionalAttentionExecutionMode.FULL,
        )
    finally:
        _restore_fakes(originals)

    assert output.model == "derived-model"
    assert output.positive is positive.entries[0]
    assert output.negative is negative.entries[0]
    assert _InteropValidator.calls[0] == (model, _CapabilityService.capabilities)
    assert _InteropValidator.calls[1][0] is _InteropValidator.report
    assert _InteropValidator.calls[1][2] is RegionalAttentionExecutionMode.FULL
    assert (
        _InteropValidator.calls[1][1] is _ModelFamily.derive_calls[0]["processed_plan"]
    )
    assert _ModelFamilySelector.calls == [_CapabilityService.capabilities]
    assert _ModelFamily.latent_calls == [samples]
    assert _LoraAdapter.calls[0][1] is model_owner
    assert len(_ModelFamily.adaptation_calls) == 1
    assert _ModelLoader.calls == [model]
    assert (
        _ConditioningProcessor.calls[0]["context_validator"] is _ModelFamily.validator
    )
    assert _ModelFamily.derive_calls[0]["region_strengths"] == (0.75,)
    assert _ModelFamily.derive_calls[0]["latent_batch_size"] == 2


def test_model_family_latent_rejection_precedes_plan_and_model_work() -> None:
    """Fail before mask, adapter, loading, processing, or backend mutation work."""

    originals = _install_fakes()
    _reset_calls()
    _ModelFamily.latent_error = ValueError("family latent rejected")
    try:
        with pytest.raises(ValueError, match="family latent rejected"):
            AttentionCouplingModelPreparationService().prepare(
                model=SimpleNamespace(model=object(), load_device="cpu"),
                positive=_conditioning(1.0),
                negative=_conditioning(-1.0),
                region_masks=torch.ones((1, 1, 1)),
                regional_prompt_weight=1.0,
                region_mask_feather=0,
                latent_image={"samples": torch.zeros((1, 4, 2, 2))},
                execution_mode=RegionalAttentionExecutionMode.FULL,
            )
    finally:
        _restore_fakes(originals)

    assert _LoraAdapter.calls == []
    assert _PreparationService.calls == []
    assert _ModelLoader.calls == []
    assert _ConditioningProcessor.calls == []
    assert _ModelFamily.derive_calls == []


def test_model_family_adapter_rejection_precedes_loading_and_processing() -> None:
    """Fail unsupported regional model hooks before device or backend work."""

    originals = _install_fakes()
    _reset_calls()
    _ModelFamily.adaptation_error = ValueError("family adapter rejected")
    try:
        with pytest.raises(ValueError, match="family adapter rejected"):
            AttentionCouplingModelPreparationService().prepare(
                model=SimpleNamespace(model=object(), load_device="cpu"),
                positive=ConditioningBatch((_conditioning(1.0), _conditioning(2.0))),
                negative=ConditioningBatch((_conditioning(-1.0), _conditioning(-2.0))),
                region_masks=torch.ones((1, 2, 2)),
                regional_prompt_weight=1.0,
                region_mask_feather=0,
                latent_image={"samples": torch.zeros((1, 4, 2, 2))},
                execution_mode=RegionalAttentionExecutionMode.FULL,
            )
    finally:
        _restore_fakes(originals)

    assert len(_LoraAdapter.calls) == 1
    assert len(_ModelFamily.adaptation_calls) == 1
    assert _PreparationService.calls == []
    assert _ModelLoader.calls == []
    assert _ConditioningProcessor.calls == []
    assert _ModelFamily.derive_calls == []


def _branch(base_value: float, region_value: float) -> ProcessedRegionalAttentionBranch:
    """Return one processed branch with global and regional contexts."""

    return ProcessedRegionalAttentionBranch(
        _processed_context(0, None, base_value),
        (_processed_context(1, 0, region_value),),
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
                torch.full((1, 2, 3), value),
                1.0,
            ),
        ),
    )


def _conditioning(value: float) -> list[list[object]]:
    """Return one standard raw conditioning entry."""

    return [[torch.full((1, 2, 3), value), {}]]


def _install_fakes() -> tuple[type[Any], ...]:
    """Install focused collaborators and return their original classes."""

    service = AttentionCouplingModelPreparationService
    originals = (
        service.capability_service_class,
        service.lora_adapter_class,
        service.preparation_service_class,
        service.model_loader_class,
        service.conditioning_processor_class,
        service.model_family_selector_class,
        service.interop_validator_class,
    )
    service.capability_service_class = _CapabilityService  # type: ignore[assignment]
    service.lora_adapter_class = _LoraAdapter  # type: ignore[assignment]
    service.preparation_service_class = _PreparationService  # type: ignore[assignment]
    service.model_loader_class = _ModelLoader  # type: ignore[assignment]
    service.conditioning_processor_class = _ConditioningProcessor  # type: ignore[assignment]
    service.model_family_selector_class = _ModelFamilySelector  # type: ignore[assignment]
    service.interop_validator_class = _InteropValidator  # type: ignore[assignment]
    return originals


def _restore_fakes(originals: tuple[type[Any], ...]) -> None:
    """Restore every shared preparation collaborator."""

    service = AttentionCouplingModelPreparationService
    service.capability_service_class = originals[0]
    service.lora_adapter_class = originals[1]
    service.preparation_service_class = originals[2]
    service.model_loader_class = originals[3]
    service.conditioning_processor_class = originals[4]
    service.model_family_selector_class = originals[5]
    service.interop_validator_class = originals[6]


def _reset_calls() -> None:
    """Clear every focused collaborator observation."""

    _CapabilityService.calls = []
    _InteropValidator.calls = []
    _LoraAdapter.calls = []
    _PreparationService.calls = []
    _ModelLoader.calls = []
    _ConditioningProcessor.calls = []
    _ModelFamilySelector.calls = []
    _ModelFamily.latent_calls = []
    _ModelFamily.adaptation_calls = []
    _ModelFamily.derive_calls = []
    _ModelFamily.latent_error = None
    _ModelFamily.adaptation_error = None
