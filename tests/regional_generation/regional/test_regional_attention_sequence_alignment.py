# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prove attention-neutral regional context sequence alignment."""

from __future__ import annotations

import pytest
import torch

from simple_syrup.runtime.regional_attention_sequence_alignment import (
    RegionalAttentionSequenceAligner,
)


def test_aligner_repeats_whole_sequences_to_the_common_length() -> None:
    """Match Comfy's attention-neutral compatible-length concatenation policy."""

    short = torch.tensor([[[1.0], [2.0]]])
    long = torch.tensor([[[3.0], [4.0], [5.0], [6.0]]])
    aligner = RegionalAttentionSequenceAligner()

    target = aligner.target_length((short, long))
    aligned_short = aligner.align(short, target_length=target)
    aligned_long = aligner.align(long, target_length=target)

    assert target == 4
    assert aligned_short[:, :, 0].tolist() == [[1.0, 2.0, 1.0, 2.0]]
    assert aligned_long is long


def test_aligner_uses_lcm_for_compatible_non_multiple_lengths() -> None:
    """Preserve complete sequences when compatible token chunks differ."""

    first = torch.ones((1, 2, 3))
    second = torch.ones((1, 3, 3))
    aligner = RegionalAttentionSequenceAligner()

    assert aligner.target_length((first, second)) == 6


def test_aligner_accepts_compatible_sequence_expansion_above_factor_four() -> None:
    """Match reference LCM semantics without an arbitrary repetition limit."""

    aligner = RegionalAttentionSequenceAligner()
    short = torch.arange(2, dtype=torch.float32).reshape(1, 2, 1)
    long = torch.arange(10, dtype=torch.float32).reshape(1, 10, 1)

    target = aligner.target_length((short, long))

    assert target == 10
    assert aligner.align(short, target_length=target)[:, :, 0].tolist() == [
        [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    ]


def test_aligner_rejects_pathological_total_allocation() -> None:
    """Fail before allocating an excessive aligned context bank."""

    aligner = RegionalAttentionSequenceAligner()

    with pytest.raises(ValueError, match="safe tensor allocation"):
        aligner.target_length(
            (
                torch.ones((1, 2, 64)).expand(131_072, -1, -1),
                torch.ones((1, 3, 64)).expand(131_072, -1, -1),
            )
        )
