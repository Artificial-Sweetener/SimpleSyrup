# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Encode aligned standard positive and negative prompt batches."""

from __future__ import annotations

from typing import Any, Protocol

from ..domain.conditioning_batch import ConditioningBatch
from ..domain.prompt_batch_parser import split_prompt_batch
from ..domain.prompt_segment_alignment import build_prompt_segment_alignment


class PromptBatchEncoder(Protocol):
    """Encode ordered prompt text through a Comfy-compatible adapter."""

    def encode_batch(
        self,
        clip: Any,
        chunks: tuple[str, ...],
    ) -> ConditioningBatch:
        """Return one conditioning entry per ordered prompt chunk."""


class PromptBatchEncodingService:
    """Align two SEP prompt sides before delegating Comfy text encoding."""

    def __init__(self, encoder: PromptBatchEncoder) -> None:
        """Store the runtime encoding adapter for one node execution."""

        self._encoder = encoder

    def encode(
        self,
        *,
        clip: Any,
        positive_prompt: str,
        negative_prompt: str,
        separator: str,
    ) -> tuple[ConditioningBatch, ConditioningBatch]:
        """Return positive and negative batches with matched segment counts."""

        positive_chunks = split_prompt_batch(positive_prompt, separator)
        negative_chunks = split_prompt_batch(negative_prompt, separator)
        alignment = build_prompt_segment_alignment(
            positive_count=len(positive_chunks),
            negative_count=len(negative_chunks),
        )
        return (
            self._encoder.encode_batch(
                clip,
                alignment.positive.materialize(positive_chunks),
            ),
            self._encoder.encode_batch(
                clip,
                alignment.negative.materialize(negative_chunks),
            ),
        )
