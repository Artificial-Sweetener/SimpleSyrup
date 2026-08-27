# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Install composable attention observation on a clone-local Comfy MODEL."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch

from .attention_region_capture import AttentionRegionCaptureSession
from .attention_region_open_vocabulary import OpenVocabularyProjectionMutation
from .patcher_lifecycle import PATCHER_LIFECYCLE, ModelMutation

AttentionFunction = Callable[..., torch.Tensor]


class OptimizedAttentionCaptureOverride:
    """Compose one observer around Comfy's configured optimized attention call."""

    def __init__(
        self,
        session: AttentionRegionCaptureSession,
        previous_override: Callable[..., torch.Tensor] | None,
    ) -> None:
        """Store capture state and any upstream optimized-attention owner."""

        self._session = session
        self._previous_override = previous_override

    def __call__(
        self,
        original: AttentionFunction,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        heads: int,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        """Observe projected Q/K, then delegate output computation unchanged."""

        transformer_options = kwargs.get("transformer_options")
        if isinstance(transformer_options, Mapping):
            self._session.observe(
                query,
                key,
                value,
                heads,
                transformer_options,
                skip_reshape=bool(kwargs.get("skip_reshape", False)),
            )
        delegate = self._previous_override or original
        return (
            delegate(original, query, key, value, heads, *args, **kwargs)
            if self._previous_override
            else delegate(query, key, value, heads, *args, **kwargs)
        )


@dataclass(frozen=True, slots=True)
class AttentionCaptureOverrideMutation:
    """Install a composable optimized-attention observer on a cloned MODEL."""

    session: AttentionRegionCaptureSession

    def apply(self, model: object) -> None:
        """Replace only the clone's override option while preserving its predecessor."""

        model_options = getattr(model, "model_options", None)
        if not isinstance(model_options, dict):
            raise TypeError("Attention capture MODEL options must be a dictionary.")
        transformer_options = model_options.get("transformer_options")
        if not isinstance(transformer_options, dict):
            raise TypeError(
                "Attention capture transformer options must be a dictionary."
            )
        previous = transformer_options.get("optimized_attention_override")
        if previous is not None and not callable(previous):
            raise TypeError("Existing optimized attention override must be callable.")
        transformer_options["optimized_attention_override"] = (
            OptimizedAttentionCaptureOverride(self.session, previous)
        )


class AttentionRegionCaptureBackend:
    """Derive one collision-safe observation-only MODEL for a capture session."""

    def derive(
        self,
        model: object,
        session: AttentionRegionCaptureSession,
    ) -> object:
        """Clone MODEL state and install exactly one optimized-attention observer."""

        mutations: tuple[ModelMutation, ...] = (
            AttentionCaptureOverrideMutation(session),
        )
        if session.open_vocabulary_contexts:
            mutations = (
                *mutations,
                OpenVocabularyProjectionMutation(
                    session.open_vocabulary_contexts,
                    session,
                ),
            )
        return PATCHER_LIFECYCLE.derive_model(
            model,
            mutations,
            operation="attention-region capture",
        )


ATTENTION_REGION_CAPTURE_BACKEND = AttentionRegionCaptureBackend()
