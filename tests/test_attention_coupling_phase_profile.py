# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify benchmark-only outer Attention Coupling phase profiling."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, ClassVar

import torch

from simple_syrup.domain.regional_attention_execution import (
    RegionalAttentionExecutionMode,
)
from simple_syrup.domain.regional_mask_bank import RegionalMaskBank
from simple_syrup.services.attention_coupling_model_preparation_service import (
    AttentionCouplingModelPreparationService,
)
from simple_syrup.services.prepared_attention_coupling_model import (
    PreparedAttentionCouplingModel,
)
from tools.attention_coupling_benchmark.comfy_probe import (
    attention_coupling_phase_profile,
)

ProfiledAttentionCouplingSamplingService = (
    attention_coupling_phase_profile.ProfiledAttentionCouplingSamplingService
)

_LOGGER_NAME = "simple_syrup.runtime.regional_lora.standard_unet_cold_path"


class _SamplingService:
    """Record exact delegate inputs and return one recognizable latent."""

    calls: ClassVar[list[dict[str, object]]] = []
    output: ClassVar[dict[str, Any]] = {"samples": torch.ones((1, 4, 2, 2))}

    def sample(self, **arguments: object) -> dict[str, Any]:
        """Retain all delegate arguments."""

        type(self).calls.append(arguments)
        return self.output


def test_profiled_service_preserves_two_call_delegation_and_emits_phases(
    caplog: Any,
) -> None:
    """Add timing evidence without changing preparation or sampling values."""

    preparation_calls: list[dict[str, object]] = []

    class _PreparationService(AttentionCouplingModelPreparationService):
        """Return one fixed prepared value and retain regional arguments."""

        def prepare(
            self,
            **arguments: object,
        ) -> PreparedAttentionCouplingModel:
            """Record exact arguments and return a derived model triple."""

            preparation_calls.append(arguments)
            masks = torch.ones((1, 2, 2))
            return PreparedAttentionCouplingModel(
                "derived",
                "prepared-positive",
                "prepared-negative",
                RegionalMaskBank(masks, masks.clone(), 2, 2),
            )

    original_preparation = (
        ProfiledAttentionCouplingSamplingService.model_preparation_service_class
    )
    original_sampling = ProfiledAttentionCouplingSamplingService.sampling_service_class
    ProfiledAttentionCouplingSamplingService.model_preparation_service_class = (
        _PreparationService
    )
    ProfiledAttentionCouplingSamplingService.sampling_service_class = _SamplingService  # type: ignore[assignment]
    _SamplingService.calls = []
    latent = {"samples": torch.zeros((1, 4, 2, 2))}
    model = SimpleNamespace(load_device=torch.device("cpu"))
    try:
        with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            output = ProfiledAttentionCouplingSamplingService().sample(
                model=model,
                seed=7,
                steps=30,
                cfg=5.0,
                sampler_name="sampler",
                scheduler="scheduler",
                positive="positive",
                negative="negative",
                region_masks="masks",
                regional_prompt_weight=1.0,
                region_mask_feather=0,
                latent_image=latent,
                denoise=1.0,
            )
    finally:
        ProfiledAttentionCouplingSamplingService.model_preparation_service_class = (
            original_preparation
        )
        ProfiledAttentionCouplingSamplingService.sampling_service_class = (
            original_sampling
        )

    assert output is _SamplingService.output
    assert preparation_calls[0]["execution_mode"] is RegionalAttentionExecutionMode.FULL
    assert preparation_calls[0]["model"] is model
    delegate = _SamplingService.calls[0]
    assert delegate["model"] == "derived"
    assert delegate["positive"] == "prepared-positive"
    assert delegate["negative"] == "prepared-negative"
    assert delegate["latent_image"] is latent
    stages = [
        record.cold_path_diagnostics["stage"]
        for record in caplog.records
        if hasattr(record, "cold_path_diagnostics")
    ]
    assert stages == ["model_preparation_total", "ksampler_delegate_total"]
