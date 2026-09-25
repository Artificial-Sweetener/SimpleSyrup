# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prove every SimpleSyrup sampler exposes one optional negative contract."""

from __future__ import annotations

from inspect import signature
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from simple_syrup.domain.conditioning_batch import ConditioningBatch
from simple_syrup.domain.regional_attention_execution import (
    RegionalAttentionExecutionMode,
)
from simple_syrup.nodes.detail_segs_as_regions import DetailSEGSAsRegions
from simple_syrup.nodes.detail_segs_by_scale_factor import DetailSEGSByScaleFactor
from simple_syrup.nodes.detail_segs_by_scale_factor_tiled_diffusion import (
    DetailSEGSByScaleFactorTiledDiffusion,
)
from simple_syrup.nodes.ksampler_extras import KSamplerExtras
from simple_syrup.nodes_v3.ksampler_attention_coupling import (
    KSamplerAttentionCouplingV3,
)
from simple_syrup.nodes_v3.ksampler_contextual_attention_coupling import (
    KSamplerContextualAttentionCouplingV3,
)
from simple_syrup.nodes_v3.ksampler_contextual_diffusion import (
    KSamplerContextualDiffusionV3,
)
from simple_syrup.nodes_v3.ksampler_prompt_by_region import KSamplerPromptByRegionV3
from simple_syrup.nodes_v3.ksampler_prompt_by_tiled_region import (
    KSamplerPromptByTiledRegionV3,
)
from simple_syrup.nodes_v3.ksampler_tiled_attention_coupling import (
    KSamplerTiledAttentionCouplingV3,
)
from simple_syrup.nodes_v3.ksampler_tiled_diffusion import KSamplerTiledDiffusionV3
from simple_syrup.nodes_v3.legacy_node_wrappers import (
    DetailSEGSAsRegionsV3,
    DetailSEGSByScaleFactorTiledDiffusionV3,
    DetailSEGSByScaleFactorV3,
    KSamplerExtrasV3,
)
from simple_syrup.services.attention_coupling_model_family import (
    AttentionCouplingPreparedModelReuse,
)
from simple_syrup.services.attention_coupling_model_preparation_service import (
    AttentionCouplingModelPreparationService,
)
from simple_syrup.services.regional_conditioning_service import (
    RegionalConditioningService,
)

_V3_SAMPLERS = (
    KSamplerExtrasV3,
    KSamplerPromptByRegionV3,
    KSamplerPromptByTiledRegionV3,
    KSamplerTiledDiffusionV3,
    KSamplerContextualDiffusionV3,
    KSamplerAttentionCouplingV3,
    KSamplerTiledAttentionCouplingV3,
    KSamplerContextualAttentionCouplingV3,
    DetailSEGSByScaleFactorV3,
    DetailSEGSByScaleFactorTiledDiffusionV3,
    DetailSEGSAsRegionsV3,
)

_IMPLEMENTATION_METHODS = (
    KSamplerExtras.sample,
    KSamplerTiledDiffusionV3.execute,
    KSamplerContextualDiffusionV3.execute,
    KSamplerPromptByRegionV3.execute,
    KSamplerPromptByTiledRegionV3.execute,
    KSamplerAttentionCouplingV3.execute,
    KSamplerTiledAttentionCouplingV3.execute,
    KSamplerContextualAttentionCouplingV3.execute,
    DetailSEGSByScaleFactor.detail,
    DetailSEGSByScaleFactorTiledDiffusion.detail,
    DetailSEGSAsRegions.detail,
)


@pytest.mark.parametrize("node_class", _V3_SAMPLERS)
def test_every_sampler_exposes_optional_negative_socket(node_class: Any) -> None:
    """Expose a disconnected negative as the universal positive-only request."""

    negative = next(
        item for item in node_class.define_schema().inputs if item.id == "negative"
    )

    assert negative.optional is True
    assert "positive-only" in negative.tooltip


@pytest.mark.parametrize("method", _IMPLEMENTATION_METHODS)
def test_every_sampler_execution_boundary_defaults_negative_to_none(
    method: Any,
) -> None:
    """Allow Comfy to omit the disconnected socket from execution arguments."""

    assert signature(method).parameters["negative"].default is None


def test_regional_conditioning_preserves_absent_negative() -> None:
    """Build regional positive conditioning without fabricating a CFG branch."""

    global_positive = [[torch.ones((1, 1, 1)), {}]]
    regional_positive = [[torch.full((1, 1, 1), 2.0), {}]]

    positive, negative = RegionalConditioningService().assemble(
        positive=ConditioningBatch((global_positive, regional_positive)),
        negative=None,
        masks=torch.ones((1, 4, 4)),
        regional_prompt_weight=0.5,
        region_mask_feather=0,
    )

    assert positive
    assert negative is None


def test_attention_coupling_prepares_structure_without_sampling_negative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mirror positive for preparation while retaining positive-only sampling."""

    positive = object()
    preparation_calls: list[dict[str, object]] = []
    resolver_calls: list[dict[str, object]] = []
    expected = object()

    class CapabilityService:
        """Admit a recognizable model capability."""

        def admit(self, **kwargs: object) -> object:
            """Return the capability needed to continue orchestration."""

            del kwargs
            return SimpleNamespace(model_capabilities=object())

    class InteropValidator:
        """Return a recognizable interop report."""

        def validate(self, model: object, capabilities: object) -> object:
            """Admit the supplied model without mutation."""

            del model, capabilities
            return object()

    class GlobalHookResolver:
        """Record the conditioning pair used for model resolution."""

        def resolve(self, model: object, **kwargs: object) -> object:
            """Preserve the model after recording structural conditioning."""

            resolver_calls.append(kwargs)
            return model

    class LatentNormalizer:
        """Preserve the test latent tensor."""

        def normalize(self, **kwargs: object) -> torch.Tensor:
            """Return the supplied samples unchanged."""

            samples = kwargs["samples"]
            assert isinstance(samples, torch.Tensor)
            return samples

    class ModelFamily:
        """Disable caching and admit the normalized latent."""

        @property
        def prepared_model_reuse(self) -> AttentionCouplingPreparedModelReuse:
            """Force the directly observable preparation path."""

            return AttentionCouplingPreparedModelReuse.DISABLED

        def validate_latent(self, samples: torch.Tensor) -> None:
            """Accept the floating test latent."""

            assert samples.shape == (1, 4, 2, 2)

    class ModelFamilySelector:
        """Return the no-cache test family."""

        def select(self, capabilities: object) -> ModelFamily:
            """Select the family after capability admission."""

            del capabilities
            return ModelFamily()

    service = AttentionCouplingModelPreparationService()

    def fake_prepare_uncached(**kwargs: object) -> object:
        """Capture the boundary between structural and sampling conditioning."""

        preparation_calls.append(kwargs)
        return expected

    monkeypatch.setattr(service, "capability_service_class", CapabilityService)
    monkeypatch.setattr(service, "interop_validator_class", InteropValidator)
    monkeypatch.setattr(service, "global_hook_model_resolver_class", GlobalHookResolver)
    monkeypatch.setattr(service, "latent_normalizer_class", LatentNormalizer)
    monkeypatch.setattr(service, "model_family_selector_class", ModelFamilySelector)
    monkeypatch.setattr(service, "_prepare_uncached", fake_prepare_uncached)

    result = service.prepare(
        model=object(),
        positive=positive,
        negative=None,
        region_masks=object(),
        regional_prompt_weight=1.0,
        region_mask_feather=0,
        latent_image={"samples": torch.zeros((1, 4, 2, 2))},
        execution_mode=RegionalAttentionExecutionMode.FULL,
    )

    assert result is expected
    assert resolver_calls == [{"positive": positive, "negative": positive}]
    assert preparation_calls[0]["negative"] is positive
    assert preparation_calls[0]["sampling_negative"] is None
