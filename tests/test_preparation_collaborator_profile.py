# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify exact benchmark-only preparation collaborator profiling."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from simple_syrup.domain.processed_regional_attention import (
    ProcessedRegionalAttentionPlan,
)
from simple_syrup.domain.raw_regional_attention import RawRegionalAttentionPlan
from simple_syrup.runtime.attention_coupling.context_validation import (
    RegionalContextValidator,
)
from simple_syrup.runtime.comfy_conditioning_model_loader import (
    ComfyConditioningModelLoader,
)
from simple_syrup.runtime.comfy_conditioning_processing import (
    ComfyRegionalConditioningProcessor,
)
from simple_syrup.runtime.regional_lora_conditioning_adapter import (
    RegionalLoraConditioningAdapter,
)
from simple_syrup.runtime.regional_lora_plan_adapter import RegionalLoraPlanAdaptation
from simple_syrup.runtime.regional_model_patch_interop import (
    RegionalModelPatchInteropValidator,
)
from simple_syrup.services.attention_coupling_model_family_selector import (
    AttentionCouplingModelFamilySelector,
)
from simple_syrup.services.attention_coupling_preparation_service import (
    AttentionCouplingPreparation,
    AttentionCouplingPreparationService,
)
from tools.attention_coupling_benchmark.comfy_probe import (
    preparation_collaborator_profile,
)
from tools.attention_coupling_benchmark.comfy_probe.interop_validation_profile import (
    ProfiledRegionalModelPatchInteropValidator,
)
from tools.attention_coupling_benchmark.comfy_probe.model_family_profile import (
    ProfiledAttentionCouplingModelFamilySelector,
)

ProfiledAttentionCouplingModelPreparationService = (
    preparation_collaborator_profile.ProfiledAttentionCouplingModelPreparationService
)
ProfiledAttentionCouplingPreparationService = (
    preparation_collaborator_profile.ProfiledAttentionCouplingPreparationService
)
ProfiledComfyConditioningModelLoader = (
    preparation_collaborator_profile.ProfiledComfyConditioningModelLoader
)
ProfiledComfyRegionalConditioningProcessor = (
    preparation_collaborator_profile.ProfiledComfyRegionalConditioningProcessor
)
ProfiledRegionalLoraConditioningAdapter = (
    preparation_collaborator_profile.ProfiledRegionalLoraConditioningAdapter
)

_LOGGER_NAME = "simple_syrup.runtime.regional_lora.standard_unet_cold_path"


def test_profiled_preparation_service_substitutes_only_exact_collaborators() -> None:
    """Keep production orchestration while replacing its four timed owners."""

    service = ProfiledAttentionCouplingModelPreparationService
    assert service.model_loader_class is ProfiledComfyConditioningModelLoader
    assert service.lora_adapter_class is ProfiledRegionalLoraConditioningAdapter
    assert (
        service.preparation_service_class is ProfiledAttentionCouplingPreparationService
    )
    assert (
        service.conditioning_processor_class
        is ProfiledComfyRegionalConditioningProcessor
    )
    assert service.interop_validator_class is ProfiledRegionalModelPatchInteropValidator
    assert (
        service.model_family_selector_class
        is ProfiledAttentionCouplingModelFamilySelector
    )
    assert issubclass(
        service.interop_validator_class,
        RegionalModelPatchInteropValidator,
    )
    assert issubclass(
        service.model_family_selector_class,
        AttentionCouplingModelFamilySelector,
    )


def test_profiled_collaborators_preserve_arguments_results_and_stage_order(
    monkeypatch: Any,
    caplog: Any,
) -> None:
    """Delegate each collaborator exactly and retain every returned identity."""

    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
    adaptation = cast(RegionalLoraPlanAdaptation, object())
    prepared = cast(AttentionCouplingPreparation, object())
    processed = cast(ProcessedRegionalAttentionPlan, object())

    def load(owner: object, model: object) -> None:
        calls.append(("load", (owner, model), {}))

    def adapt(
        owner: object,
        plan: RawRegionalAttentionPlan,
        *,
        model: object,
    ) -> RegionalLoraPlanAdaptation:
        calls.append(("adapt", (owner, plan), {"model": model}))
        return adaptation

    def prepare(
        owner: object,
        plan: RawRegionalAttentionPlan,
    ) -> AttentionCouplingPreparation:
        calls.append(("prepare", (owner, plan), {}))
        return prepared

    def process(
        owner: object,
        preparation: AttentionCouplingPreparation,
        *,
        model: object,
        noise: torch.Tensor,
        device: torch.device,
        context_validator: RegionalContextValidator,
    ) -> ProcessedRegionalAttentionPlan:
        calls.append(
            (
                "process",
                (owner, preparation),
                {
                    "model": model,
                    "noise": noise,
                    "device": device,
                    "context_validator": context_validator,
                },
            )
        )
        return processed

    monkeypatch.setattr(ComfyConditioningModelLoader, "load", load)
    monkeypatch.setattr(RegionalLoraConditioningAdapter, "adapt", adapt)
    monkeypatch.setattr(AttentionCouplingPreparationService, "prepare", prepare)
    monkeypatch.setattr(ComfyRegionalConditioningProcessor, "process", process)

    model = SimpleNamespace(load_device=torch.device("cpu"))
    base_model = SimpleNamespace()
    plan = cast(RawRegionalAttentionPlan, object())
    noise = torch.zeros((1, 4, 2, 2))
    device = torch.device("cpu")
    validator = cast(RegionalContextValidator, object())
    with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
        ProfiledComfyConditioningModelLoader().load(model)
        actual_adaptation = ProfiledRegionalLoraConditioningAdapter().adapt(
            plan,
            model=base_model,
        )
        actual_preparation = ProfiledAttentionCouplingPreparationService().prepare(plan)
        actual_processed = ProfiledComfyRegionalConditioningProcessor().process(
            prepared,
            model=model,
            noise=noise,
            device=device,
            context_validator=validator,
        )

    assert actual_adaptation is adaptation
    assert actual_preparation is prepared
    assert actual_processed is processed
    assert [call[0] for call in calls] == ["load", "adapt", "prepare", "process"]
    assert calls[0][1][1] is model
    assert calls[1][1][1] is plan
    assert calls[1][2]["model"] is base_model
    assert calls[2][1][1] is plan
    assert calls[3][1][1] is prepared
    assert calls[3][2] == {
        "model": model,
        "noise": noise,
        "device": device,
        "context_validator": validator,
    }
    assert _stages(caplog) == [
        "source_model_load",
        "regional_lora_adaptation",
        "regional_plan_preparation",
        "conditioning_processing",
    ]


def test_profiled_collaborator_preserves_delegate_exception(
    monkeypatch: Any,
    caplog: Any,
) -> None:
    """Propagate the original failure while still closing its timing phase."""

    failure = RuntimeError("source load failed")

    def fail(_owner: object, _model: object) -> None:
        raise failure

    monkeypatch.setattr(ComfyConditioningModelLoader, "load", fail)
    with (
        caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME),
        pytest.raises(RuntimeError) as raised,
    ):
        ProfiledComfyConditioningModelLoader().load(
            SimpleNamespace(load_device=torch.device("cpu"))
        )

    assert raised.value is failure
    assert _stages(caplog) == ["source_model_load"]


def _stages(caplog: Any) -> list[str]:
    """Return ordered focused stage names from captured records."""

    return [
        record.cold_path_diagnostics["stage"]
        for record in caplog.records
        if hasattr(record, "cold_path_diagnostics")
    ]
