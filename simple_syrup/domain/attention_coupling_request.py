# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Classify complete Attention Coupling requests before runtime preparation."""

from __future__ import annotations

from enum import StrEnum

from .conditioning_batch import ConditioningBatch


class AttentionCouplingRequestMode(StrEnum):
    """Select ordinary sampling or complete regional Attention Coupling."""

    BYPASS = "bypass"
    ACTIVE = "active"


def classify_attention_coupling_request(
    *,
    positive: object,
    negative: object,
    region_masks: object | None,
) -> AttentionCouplingRequestMode:
    """Return the execution mode or reject a partial regional request."""

    has_conditioning_batch = isinstance(positive, ConditioningBatch) or isinstance(
        negative,
        ConditioningBatch,
    )
    has_region_masks = region_masks is not None
    if not has_conditioning_batch and not has_region_masks:
        return AttentionCouplingRequestMode.BYPASS
    if has_conditioning_batch and has_region_masks:
        return AttentionCouplingRequestMode.ACTIVE
    if has_conditioning_batch:
        raise ValueError(
            "Attention Coupling conditioning batches require region_masks. "
            "Connect ordered masks or use ordinary CONDITIONING on both inputs."
        )
    raise ValueError(
        "Attention Coupling region_masks require a CONDITIONING_BATCH on the "
        "positive or negative input. Disconnect the masks for ordinary sampling."
    )
