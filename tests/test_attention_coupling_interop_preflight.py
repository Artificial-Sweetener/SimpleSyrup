"""Prove modifier admission precedes every expensive preparation owner."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from simple_syrup.domain.regional_attention_execution import (
    RegionalAttentionExecutionMode,
)
from simple_syrup.services.attention_coupling_model_preparation_service import (
    AttentionCouplingModelPreparationService,
)


def test_modifier_conflict_preflight_runs_before_family_mask_or_model_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject one MODEL immediately after capability detection."""

    events: list[str] = []
    capabilities = object()
    model = object()

    class CapabilityService:
        """Return one recognized capability sentinel."""

        def admit(self, **kwargs: object) -> object:
            """Record admission without interpreting supplied values."""

            del kwargs
            events.append("capability")
            return SimpleNamespace(model_capabilities=capabilities)

    class InteropValidator:
        """Stop preparation at the required preflight boundary."""

        def validate(
            self, supplied_model: object, supplied_capabilities: object
        ) -> None:
            """Prove exact values and reject before downstream work."""

            assert supplied_model is model
            assert supplied_capabilities is capabilities
            events.append("interop")
            raise ValueError("modifier conflict sentinel")

    class ForbiddenFamilySelector:
        """Fail if model-family selection occurs after a known conflict."""

        def select(self, supplied_capabilities: object) -> object:
            """Reject an invalid ordering call."""

            del supplied_capabilities
            events.append("family")
            raise AssertionError("family selection must not run")

    monkeypatch.setattr(
        AttentionCouplingModelPreparationService,
        "capability_service_class",
        CapabilityService,
    )
    monkeypatch.setattr(
        AttentionCouplingModelPreparationService,
        "interop_validator_class",
        InteropValidator,
    )
    monkeypatch.setattr(
        AttentionCouplingModelPreparationService,
        "model_family_selector_class",
        ForbiddenFamilySelector,
    )

    with pytest.raises(ValueError, match="modifier conflict sentinel"):
        AttentionCouplingModelPreparationService().prepare(
            model=model,
            positive=object(),
            negative=object(),
            region_masks=object(),
            regional_prompt_weight=1.0,
            region_mask_feather=0,
            latent_image={"samples": torch.zeros((1, 16, 1, 8, 8))},
            execution_mode=RegionalAttentionExecutionMode.FULL,
        )

    assert events == ["capability", "interop"]
