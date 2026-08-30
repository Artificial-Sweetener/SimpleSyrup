# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify complete ordinary and regional Attention Coupling requests."""

from __future__ import annotations

import pytest

from simple_syrup.domain.attention_coupling_request import (
    AttentionCouplingRequestMode,
    classify_attention_coupling_request,
)
from simple_syrup.domain.conditioning_batch import ConditioningBatch


def test_ordinary_conditioning_without_masks_selects_bypass() -> None:
    """Route a complete ordinary request around Attention Coupling."""

    assert (
        classify_attention_coupling_request(
            positive="positive",
            negative="negative",
            region_masks=None,
        )
        is AttentionCouplingRequestMode.BYPASS
    )


@pytest.mark.parametrize(
    ("positive", "negative"),
    [
        (ConditioningBatch(("global", "region")), "negative"),
        ("positive", ConditioningBatch(("global", "region"))),
        (
            ConditioningBatch(("global", "region")),
            ConditioningBatch(("global", "region")),
        ),
    ],
)
def test_conditioning_batch_with_masks_selects_attention_coupling(
    positive: object,
    negative: object,
) -> None:
    """Allow either or both conditioning branches to own regional entries."""

    assert (
        classify_attention_coupling_request(
            positive=positive,
            negative=negative,
            region_masks="masks",
        )
        is AttentionCouplingRequestMode.ACTIVE
    )


def test_conditioning_batch_without_masks_is_rejected() -> None:
    """Reject a regional conditioning batch without its spatial authority."""

    with pytest.raises(ValueError, match="batches require region_masks"):
        classify_attention_coupling_request(
            positive=ConditioningBatch(("global", "region")),
            negative="negative",
            region_masks=None,
        )


def test_masks_without_conditioning_batch_are_rejected() -> None:
    """Reject masks that have no ordered regional conditioning entries."""

    with pytest.raises(ValueError, match="require a CONDITIONING_BATCH"):
        classify_attention_coupling_request(
            positive="positive",
            negative="negative",
            region_masks="masks",
        )
