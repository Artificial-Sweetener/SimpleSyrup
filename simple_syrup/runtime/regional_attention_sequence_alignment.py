# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Align compatible cross-attention sequence lengths without changing attention."""

from __future__ import annotations

import math

import torch

_MAXIMUM_ALIGNED_CONTEXT_ELEMENTS = 67_108_864


class RegionalAttentionSequenceAligner:
    """Own Comfy-compatible whole-sequence repetition for regional contexts."""

    def target_length(self, contexts: tuple[torch.Tensor, ...]) -> int:
        """Return one bounded common token length for every supplied context."""

        if not isinstance(contexts, tuple) or not contexts:
            raise ValueError("Regional sequence alignment requires contexts.")
        lengths: list[int] = []
        feature_width: int | None = None
        for context in contexts:
            if (
                not isinstance(context, torch.Tensor)
                or context.ndim != 3
                or not context.is_floating_point()
                or any(int(size) < 1 for size in context.shape)
            ):
                raise ValueError(
                    "Regional sequence alignment requires non-empty floating "
                    "BxSxD contexts."
                )
            observed_width = int(context.shape[2])
            if feature_width is None:
                feature_width = observed_width
            elif observed_width != feature_width:
                raise ValueError(
                    "Regional sequence alignment feature widths must match."
                )
            lengths.append(int(context.shape[1]))
        target = math.lcm(*lengths)
        aligned_elements = sum(
            int(context.shape[0]) * target * int(context.shape[2])
            for context in contexts
        )
        if aligned_elements > _MAXIMUM_ALIGNED_CONTEXT_ELEMENTS:
            raise ValueError(
                "Regional context alignment exceeds the safe tensor allocation limit."
            )
        return target

    @staticmethod
    def align(context: torch.Tensor, *, target_length: int) -> torch.Tensor:
        """Repeat a complete sequence to the admitted target token count."""

        if (
            not isinstance(context, torch.Tensor)
            or context.ndim != 3
            or isinstance(target_length, bool)
            or not isinstance(target_length, int)
            or target_length < int(context.shape[1])
            or target_length % int(context.shape[1]) != 0
        ):
            raise ValueError("Regional context cannot align to the target sequence.")
        factor = target_length // int(context.shape[1])
        if factor == 1:
            return context
        return context.repeat(1, factor, 1)


REGIONAL_ATTENTION_SEQUENCE_ALIGNER = RegionalAttentionSequenceAligner()
