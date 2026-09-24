# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify focused Anima Attention Coupling family policy."""

from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar, cast

import pytest
import torch

from simple_syrup.domain.processed_regional_attention import (
    ProcessedRegionalAttentionPlan,
)
from simple_syrup.domain.regional_lora_plan import EMPTY_REGIONAL_LORA_PLAN
from simple_syrup.domain.regional_model_capabilities import RegionalModelFamily
from simple_syrup.runtime.attention_coupling.anima_context import (
    ANIMA_REGIONAL_CONTEXT_VALIDATOR,
)
from simple_syrup.runtime.regional_lora_plan_adapter import RegionalLoraPlanAdaptation
from simple_syrup.runtime.regional_model_patch_interop import (
    RegionalModelPatchInteropReport,
)
from simple_syrup.services.anima_attention_coupling_model_family import (
    AnimaAttentionCouplingModelFamily,
)


class _Backend:
    """Record Anima backend derivation without model execution."""

    calls: ClassVar[list[dict[str, object]]] = []

    def derive(self, **kwargs: object) -> object:
        """Return a recognizable derived model container."""

        type(self).calls.append(kwargs)
        return SimpleNamespace(model="derived-anima")


def test_anima_family_retains_single_frame_context_and_backend_policy() -> None:
    """Forward typed shared state to the existing full-surface Anima backend."""

    family = AnimaAttentionCouplingModelFamily()
    adaptation = RegionalLoraPlanAdaptation(EMPTY_REGIONAL_LORA_PLAN, ())
    processed = cast(ProcessedRegionalAttentionPlan, object())
    original_backend = family.backend_class
    type(family).backend_class = _Backend  # type: ignore[assignment]
    _Backend.calls = []
    try:
        family.validate_latent(torch.zeros(2, 16, 1, 8, 8))
        admission = family.admit_adaptation("model", adaptation)
        derived = family.derive(
            model="model",
            processed_plan=processed,
            admission=admission,
            interop_report=RegionalModelPatchInteropReport(
                RegionalModelFamily.ANIMA,
                (),
            ),
            region_strengths=(0.75,),
            latent_batch_size=2,
        )
    finally:
        type(family).backend_class = original_backend

    assert family.context_validator is ANIMA_REGIONAL_CONTEXT_VALIDATOR
    assert derived == "derived-anima"
    assert _Backend.calls == [
        {
            "model": "model",
            "processed_plan": processed,
            "adaptation": adaptation,
            "region_strengths": (0.75,),
            "latent_batch_size": 2,
            "negpip": None,
        }
    ]


@pytest.mark.parametrize(
    "samples",
    [torch.zeros(1, 4, 8, 8), torch.zeros(1, 4, 2, 8, 8)],
)
def test_anima_family_rejects_non_image_temporal_layout(
    samples: torch.Tensor,
) -> None:
    """Preserve the explicit Anima single-frame image boundary."""

    with pytest.raises(ValueError, match="BxCx1xHxW"):
        AnimaAttentionCouplingModelFamily().validate_latent(samples)
