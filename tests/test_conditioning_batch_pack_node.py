# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for conditioning batch pack nodes."""

from __future__ import annotations

from simple_syrup.domain.conditioning_batch import ConditioningBatch
from simple_syrup.nodes.conditioning_batch_pack import (
    ConditioningBatchAppend,
    ConditioningBatchStart,
)


def test_conditioning_batch_start_contract() -> None:
    """Start node exposes the internal batch output contract."""

    inputs = ConditioningBatchStart.INPUT_TYPES()

    assert ConditioningBatchStart.RETURN_TYPES == ("CONDITIONING_BATCH",)
    assert ConditioningBatchStart.RETURN_NAMES == ("batch",)
    assert ConditioningBatchStart.CATEGORY == "SimpleSyrup/Conditioning"
    assert list(inputs["required"]) == ["conditioning"]
    assert inputs["required"]["conditioning"][0] == "CONDITIONING"


def test_conditioning_batch_start_packs_one_entry() -> None:
    """Start node wraps one conditioning in a new batch."""

    conditioning = object()

    (batch,) = ConditioningBatchStart().pack(conditioning)

    assert batch == ConditioningBatch((conditioning,))


def test_conditioning_batch_append_contract() -> None:
    """Append node exposes the internal batch extension contract."""

    inputs = ConditioningBatchAppend.INPUT_TYPES()

    assert ConditioningBatchAppend.RETURN_TYPES == ("CONDITIONING_BATCH",)
    assert ConditioningBatchAppend.RETURN_NAMES == ("batch",)
    assert ConditioningBatchAppend.CATEGORY == "SimpleSyrup/Conditioning"
    assert list(inputs["required"]) == ["batch", "conditioning"]
    assert inputs["required"]["batch"][0] == "CONDITIONING_BATCH"
    assert inputs["required"]["conditioning"][0] == "CONDITIONING"


def test_conditioning_batch_append_returns_new_batch() -> None:
    """Append node does not mutate the existing batch."""

    first = object()
    second = object()
    existing = ConditioningBatch((first,))

    (extended,) = ConditioningBatchAppend().append(existing, second)

    assert existing.entries == (first,)
    assert extended.entries == (first, second)
