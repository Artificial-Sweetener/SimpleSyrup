# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Resolve one diffusion call to exact active regional attention contexts."""

from __future__ import annotations

import torch

from ..domain.processed_regional_attention import ProcessedRegionalAttentionPlan
from ..domain.regional_attention_batch import BatchedRegionalAttentionContexts
from .regional_attention_batching import (
    REGIONAL_ATTENTION_BATCHING_SERVICE,
    RegionalAttentionBatchingService,
)
from .regional_attention_model_call_values import uniform_model_call_sigma
from .regional_attention_runtime_identity import (
    REGIONAL_ATTENTION_RUNTIME_IDENTITY_RESOLVER,
    RegionalAttentionRuntimeIdentityResolver,
)


class RegionalAttentionModelCallResolver:
    """Validate one model call and align its active schedule and CFG contexts."""

    def __init__(
        self,
        *,
        identity_resolver: RegionalAttentionRuntimeIdentityResolver = (
            REGIONAL_ATTENTION_RUNTIME_IDENTITY_RESOLVER
        ),
        batching: RegionalAttentionBatchingService = (
            REGIONAL_ATTENTION_BATCHING_SERVICE
        ),
    ) -> None:
        """Retain the existing identity and batch-alignment authorities."""

        if not isinstance(
            identity_resolver,
            RegionalAttentionRuntimeIdentityResolver,
        ):
            raise TypeError(
                "Regional model-call identity resolver has an invalid type."
            )
        if not isinstance(batching, RegionalAttentionBatchingService):
            raise TypeError("Regional model-call batching owner has an invalid type.")
        self._identity_resolver = identity_resolver
        self._batching = batching

    def resolve(
        self,
        plan: ProcessedRegionalAttentionPlan,
        *,
        model_input: object,
        base_context: object,
        transformer_options: object,
    ) -> BatchedRegionalAttentionContexts:
        """Return active contexts in the exact supplied model-call batch order."""

        if not isinstance(plan, ProcessedRegionalAttentionPlan):
            raise TypeError("Regional model-call resolution requires a plan.")
        if not isinstance(model_input, torch.Tensor) or model_input.ndim not in (4, 5):
            raise TypeError("Regional model input must be a 4D or 5D tensor.")
        if not isinstance(base_context, torch.Tensor) or base_context.ndim != 3:
            raise TypeError("Regional base context must use BxSxD layout.")
        if not isinstance(transformer_options, dict):
            raise TypeError("Regional transformer_options must be a dictionary.")
        selectors = transformer_options.get("cond_or_uncond")
        if not isinstance(selectors, list | tuple) or not selectors:
            raise TypeError(
                "Regional transformer_options require cond_or_uncond chunks."
            )
        if int(model_input.shape[0]) % len(selectors) != 0:
            raise ValueError(
                "Regional model input batch must divide evenly across CFG chunks."
            )
        latent_batch_size = int(model_input.shape[0]) // len(selectors)
        sigma = uniform_model_call_sigma(transformer_options.get("sigmas"))
        identities = self._identity_resolver.resolve(
            plan,
            cond_or_uncond=selectors,
            base_context=base_context,
            sigma=sigma,
            latent_batch_size=latent_batch_size,
        )
        return self._batching.align(
            plan,
            base_context=base_context,
            cond_or_uncond=selectors,
            conditioning_uuids=identities,
            sigma=sigma,
            latent_batch_size=latent_batch_size,
        )


REGIONAL_ATTENTION_MODEL_CALL_RESOLVER = RegionalAttentionModelCallResolver()
