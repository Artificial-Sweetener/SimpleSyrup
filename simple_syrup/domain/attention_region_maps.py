# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Model prompt-token spans and compact captured attention observations."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .attention_spatial_transform import AttentionSpatialTransform


@dataclass(frozen=True, slots=True)
class AttentionTokenSpan:
    """Bind a readable prompt occurrence to exact conditioning token positions."""

    label: str
    occurrence: int
    token_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        """Require canonical labels and ordered non-negative token indices."""

        if not self.label or self.label != self.label.strip():
            raise ValueError("Attention token span label must be canonical.")
        if type(self.occurrence) is not int or self.occurrence < 1:
            raise ValueError("Attention token span occurrence must be positive.")
        if (
            not self.token_indices
            or self.token_indices != tuple(sorted(set(self.token_indices)))
            or self.token_indices[0] < 0
        ):
            raise ValueError("Attention token indices must be ordered and unique.")

    @property
    def display_label(self) -> str:
        """Disambiguate repeated concepts while keeping first labels concise."""

        return (
            self.label if self.occurrence == 1 else f"{self.label} #{self.occurrence}"
        )


@dataclass(frozen=True, slots=True)
class AttentionTokenCatalog:
    """Hold readable prompt spans and conditioning sequence length."""

    sequence_length: int
    spans: tuple[AttentionTokenSpan, ...]
    token_ids: tuple[object, ...]

    def __post_init__(self) -> None:
        """Require all spans to fit the captured conditioning sequence."""

        if type(self.sequence_length) is not int or self.sequence_length < 1:
            raise ValueError("Attention token sequence length must be positive.")
        if len(self.token_ids) != self.sequence_length:
            raise ValueError("Attention token ids must match the sequence length.")
        if any(
            index >= self.sequence_length
            for span in self.spans
            for index in span.token_indices
        ):
            raise ValueError("Attention token span exceeds its conditioning sequence.")

    def exact_matches(self, query: str) -> tuple[AttentionTokenSpan, ...]:
        """Return every prompt occurrence whose normalized label equals a query."""

        normalized = _normalized_label(query)
        return tuple(
            span for span in self.spans if _normalized_label(span.label) == normalized
        )


@dataclass(frozen=True, slots=True)
class CapturedAttentionMap:
    """Store one head-aggregated token map and its denoising observation identity."""

    label: str
    values: torch.Tensor
    progress: float
    layer_key: str
    batch_index: int = 0
    confidence: float = 1.0
    spatial_height: int | None = None
    spatial_width: int | None = None
    spatial_transforms: tuple[AttentionSpatialTransform, ...] = ()

    def __post_init__(self) -> None:
        """Require a finite CPU spatial vector and normalized progress."""

        if not self.label:
            raise ValueError("Captured attention map label cannot be empty.")
        if (
            not isinstance(self.values, torch.Tensor)
            or self.values.device.type != "cpu"
            or self.values.ndim != 1
            or self.values.numel() < 1
            or not self.values.is_floating_point()
            or not torch.isfinite(self.values).all().item()
        ):
            raise ValueError("Captured attention values must be a finite CPU vector.")
        if not 0.0 <= self.progress <= 1.0:
            raise ValueError("Captured attention progress must be within 0..1.")
        if not self.layer_key:
            raise ValueError("Captured attention layer key cannot be empty.")
        if type(self.batch_index) is not int or self.batch_index < 0:
            raise ValueError("Captured attention batch index must be non-negative.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Captured attention confidence must be within 0..1.")
        if (self.spatial_height is None) != (self.spatial_width is None):
            raise ValueError("Captured attention geometry must be complete or absent.")
        if self.spatial_height is not None and (
            type(self.spatial_height) is not int
            or self.spatial_height < 1
            or type(self.spatial_width) is not int
            or self.spatial_width < 1
            or self.spatial_height * self.spatial_width != int(self.values.numel())
        ):
            raise ValueError("Captured attention geometry must match its values.")
        if any(
            not isinstance(transform, AttentionSpatialTransform)
            for transform in self.spatial_transforms
        ):
            raise TypeError("Captured attention spatial transforms are invalid.")


@dataclass(frozen=True, slots=True)
class OpenVocabularyContext:
    """Hold one query's encoded SDXL context and semantic token positions."""

    label: str
    values: torch.Tensor
    token_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        """Require one finite CPU context with valid unique token positions."""

        if not self.label.strip() or self.label != self.label.strip():
            raise ValueError("Open-vocabulary labels must be canonical strings.")
        if self.values.ndim != 3 or int(self.values.shape[0]) != 1:
            raise ValueError("Open-vocabulary context must have shape 1xTxC.")
        if self.values.device.type != "cpu" or not torch.isfinite(self.values).all():
            raise ValueError("Open-vocabulary context must be finite CPU storage.")
        if not self.token_indices or self.token_indices != tuple(
            sorted(set(self.token_indices))
        ):
            raise ValueError(
                "Open-vocabulary token positions must be unique and ordered."
            )
        if any(
            index < 0 or index >= int(self.values.shape[1])
            for index in self.token_indices
        ):
            raise ValueError("Open-vocabulary token positions exceed their context.")


def _normalized_label(value: str) -> str:
    """Normalize human prompt labels without changing tokenizer semantics."""

    return " ".join(value.casefold().replace("_", " ").split())
