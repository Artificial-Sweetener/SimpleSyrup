# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Project absent SDXL query contexts through live cross-attention key layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, cast

import torch
from torch import nn

from ..domain.attention_region_maps import OpenVocabularyContext
from .model_object_patch_batch import (
    ExactModelObjectReplacement,
    ModelObjectPatchBatchMutation,
)


class OpenVocabularyProjectionSink(Protocol):
    """Receive query keys projected for the immediately following attention call."""

    def stage_open_vocabulary_keys(
        self,
        keys: tuple[tuple[str, torch.Tensor], ...],
    ) -> None:
        """Stage projected key tensors under stable query labels."""


class OpenVocabularyKeyProjector(nn.Module):
    """Delegate native key projection and side-project compact query contexts."""

    def __init__(
        self,
        original: nn.Module,
        contexts: tuple[OpenVocabularyContext, ...],
        sink: OpenVocabularyProjectionSink,
    ) -> None:
        """Retain one live key module and immutable CPU query contexts."""

        super().__init__()
        self.original = original
        self._contexts = contexts
        self._sink = sink
        self._projected_cache: dict[
            tuple[torch.device, torch.dtype], tuple[tuple[str, torch.Tensor], ...]
        ] = {}

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Return native keys unchanged after staging selected query keys."""

        native = cast(torch.Tensor, self.original(value))
        cache_key = (value.device, value.dtype)
        projected = self._projected_cache.get(cache_key)
        if projected is None:
            projected = self._project(value)
            self._projected_cache[cache_key] = projected
        self._sink.stage_open_vocabulary_keys(projected)
        return native

    def _project(
        self,
        reference: torch.Tensor,
    ) -> tuple[tuple[str, torch.Tensor], ...]:
        """Project immutable query contexts once for one execution device and dtype."""

        projected: list[tuple[str, torch.Tensor]] = []
        with torch.no_grad():
            for context in self._contexts:
                encoded = context.values.to(
                    device=reference.device,
                    dtype=reference.dtype,
                )
                query_keys = cast(torch.Tensor, self.original(encoded))
                indices = torch.tensor(
                    context.token_indices,
                    device=query_keys.device,
                    dtype=torch.int64,
                )
                projected.append((context.label, query_keys.index_select(1, indices)))
        return tuple(projected)


@dataclass(frozen=True, slots=True)
class OpenVocabularyProjectionMutation:
    """Install clone-local wrappers on every standard-UNet cross-attention key."""

    contexts: tuple[OpenVocabularyContext, ...]
    sink: OpenVocabularyProjectionSink

    def apply(self, model: object) -> None:
        """Discover canonical attn2 key paths and patch them as one batch."""

        from comfy.ldm.modules.attention import BasicTransformerBlock

        base_model = getattr(model, "model", None)
        if not isinstance(base_model, nn.Module):
            raise TypeError("Open-vocabulary MODEL root must be a torch module.")
        replacements: list[ExactModelObjectReplacement] = []
        for path, module in base_model.named_modules():
            if not isinstance(module, BasicTransformerBlock) or module.attn2 is None:
                continue
            to_k = module.attn2.to_k
            if not isinstance(to_k, nn.Module):
                raise TypeError("Open-vocabulary attn2 key projection is invalid.")
            replacements.append(
                ExactModelObjectReplacement(
                    f"{path}.attn2.to_k",
                    to_k,
                    OpenVocabularyKeyProjector(to_k, self.contexts, self.sink),
                )
            )
        if not replacements:
            raise ValueError(
                "Open-vocabulary SDXL cross-attention layers were not found."
            )
        ModelObjectPatchBatchMutation(tuple(replacements)).apply(model)
