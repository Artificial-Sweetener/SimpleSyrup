# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify transparent benchmark-only model-family profiling."""

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
from simple_syrup.domain.regional_model_capabilities import RegionalModelCapabilities
from simple_syrup.runtime.attention_coupling.context_validation import (
    RegionalContextValidator,
)
from simple_syrup.runtime.attention_coupling.family_admission import (
    AttentionCouplingFamilyAdmission,
)
from simple_syrup.runtime.regional_lora_plan_adapter import RegionalLoraPlanAdaptation
from simple_syrup.runtime.regional_model_patch_interop import (
    RegionalModelPatchInteropReport,
)
from simple_syrup.services.attention_coupling_model_family import (
    AttentionCouplingModelFamily,
    AttentionCouplingPreparedModelReuse,
    AttentionCouplingSamplerConditioning,
)
from simple_syrup.services.attention_coupling_model_family_selector import (
    AttentionCouplingModelFamilySelector,
)
from tools.attention_coupling_benchmark.comfy_probe.model_family_profile import (
    ProfiledAttentionCouplingModelFamily,
    ProfiledAttentionCouplingModelFamilySelector,
)

_LOGGER_NAME = "simple_syrup.runtime.regional_lora.standard_unet_cold_path"


class _Family:
    """Return recognizable identities while recording exact protocol calls."""

    def __init__(self) -> None:
        """Create immutable property identities and empty call storage."""

        self.validator = cast(RegionalContextValidator, object())
        self.admission = cast(AttentionCouplingFamilyAdmission, object())
        self.conditioning = cast(AttentionCouplingSamplerConditioning, object())
        self.derived = object()
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    @property
    def context_validator(self) -> RegionalContextValidator:
        """Return one fixed validator identity."""

        return self.validator

    @property
    def prepared_model_reuse(self) -> AttentionCouplingPreparedModelReuse:
        """Return one fixed reuse policy."""

        return AttentionCouplingPreparedModelReuse.EXACT_REQUEST

    def validate_latent(self, samples: torch.Tensor) -> None:
        """Record the supplied tensor identity."""

        self.calls.append(("validate", (samples,), {}))

    def admit_adaptation(
        self,
        model: object,
        adaptation: RegionalLoraPlanAdaptation,
    ) -> AttentionCouplingFamilyAdmission:
        """Record and return fixed admission evidence."""

        self.calls.append(("admit", (model, adaptation), {}))
        return self.admission

    def prepare_sampler_conditioning(
        self,
        plan: RawRegionalAttentionPlan,
        region_strengths: tuple[float, ...],
    ) -> AttentionCouplingSamplerConditioning:
        """Record and return fixed sampler conditioning."""

        self.calls.append(("conditioning", (plan, region_strengths), {}))
        return self.conditioning

    def derive(
        self,
        *,
        model: object,
        processed_plan: ProcessedRegionalAttentionPlan,
        admission: AttentionCouplingFamilyAdmission,
        interop_report: RegionalModelPatchInteropReport,
        region_strengths: tuple[float, ...],
        latent_batch_size: int,
    ) -> object:
        """Record every named argument and return one fixed model."""

        self.calls.append(
            (
                "derive",
                (),
                {
                    "model": model,
                    "processed_plan": processed_plan,
                    "admission": admission,
                    "interop_report": interop_report,
                    "region_strengths": region_strengths,
                    "latent_batch_size": latent_batch_size,
                },
            )
        )
        return self.derived


def test_profiled_family_preserves_properties_arguments_results_and_order(
    caplog: Any,
) -> None:
    """Keep the selected family authoritative behind four timed calls."""

    family = _Family()
    profiled = ProfiledAttentionCouplingModelFamily(family)
    samples = torch.zeros((1, 4, 2, 2))
    model = SimpleNamespace(load_device=torch.device("cpu"))
    adaptation = cast(RegionalLoraPlanAdaptation, object())
    plan = cast(RawRegionalAttentionPlan, object())
    processed = cast(ProcessedRegionalAttentionPlan, object())
    report = cast(RegionalModelPatchInteropReport, object())
    strengths = (1.0, 0.5)

    with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
        profiled.validate_latent(samples)
        admission = profiled.admit_adaptation(model, adaptation)
        conditioning = profiled.prepare_sampler_conditioning(plan, strengths)
        derived = profiled.derive(
            model=model,
            processed_plan=processed,
            admission=admission,
            interop_report=report,
            region_strengths=strengths,
            latent_batch_size=1,
        )

    assert profiled.delegate is family
    assert profiled.context_validator is family.validator
    assert (
        profiled.prepared_model_reuse
        is AttentionCouplingPreparedModelReuse.EXACT_REQUEST
    )
    assert admission is family.admission
    assert conditioning is family.conditioning
    assert derived is family.derived
    assert [call[0] for call in family.calls] == [
        "validate",
        "admit",
        "conditioning",
        "derive",
    ]
    assert family.calls[0][1] == (samples,)
    assert family.calls[1][1] == (model, adaptation)
    assert family.calls[2][1] == (plan, strengths)
    assert family.calls[3][2] == {
        "model": model,
        "processed_plan": processed,
        "admission": admission,
        "interop_report": report,
        "region_strengths": strengths,
        "latent_batch_size": 1,
    }
    assert _stages(caplog) == [
        "family_validate_latent",
        "family_admit_adaptation",
        "family_sampler_conditioning",
        "family_derive_total",
    ]


def test_profiled_selector_wraps_exact_production_selected_family(
    monkeypatch: Any,
) -> None:
    """Retain capability routing in the production selector implementation."""

    family = _Family()
    capabilities = cast(RegionalModelCapabilities, object())
    calls: list[RegionalModelCapabilities] = []

    def select(
        _owner: object,
        supplied: RegionalModelCapabilities,
    ) -> AttentionCouplingModelFamily:
        calls.append(supplied)
        return family

    monkeypatch.setattr(AttentionCouplingModelFamilySelector, "select", select)
    selected = ProfiledAttentionCouplingModelFamilySelector().select(capabilities)

    assert calls == [capabilities]
    assert isinstance(selected, ProfiledAttentionCouplingModelFamily)
    assert selected.delegate is family


def test_profiled_family_preserves_delegate_exception(caplog: Any) -> None:
    """Propagate an original family error while closing the timed boundary."""

    failure = RuntimeError("family admission failed")

    class _FailingFamily(_Family):
        """Fail from one exact protocol method."""

        def admit_adaptation(
            self,
            model: object,
            adaptation: RegionalLoraPlanAdaptation,
        ) -> AttentionCouplingFamilyAdmission:
            """Raise the fixed original failure."""

            del model, adaptation
            raise failure

    profiled = ProfiledAttentionCouplingModelFamily(_FailingFamily())
    with (
        caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME),
        pytest.raises(RuntimeError) as raised,
    ):
        profiled.admit_adaptation(
            SimpleNamespace(load_device=torch.device("cpu")),
            cast(RegionalLoraPlanAdaptation, object()),
        )

    assert raised.value is failure
    assert _stages(caplog) == ["family_admit_adaptation"]


def _stages(caplog: Any) -> list[str]:
    """Return ordered focused stage names from captured records."""

    return [
        record.cold_path_diagnostics["stage"]
        for record in caplog.records
        if hasattr(record, "cold_path_diagnostics")
    ]
