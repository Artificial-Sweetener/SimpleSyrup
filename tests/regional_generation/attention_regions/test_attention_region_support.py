# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Test topology-aware support selection for isolated attention concepts."""

from __future__ import annotations

import torch

from simple_syrup.services.attention_region_support import (
    ATTENTION_CONCEPT_SUPPORT_SERVICE,
)


def test_weak_extent_survives_only_when_connected_to_a_strong_concept_core() -> None:
    """Keep faint object structure while rejecting detached faint speckles."""

    alpha = torch.tensor([[0.8, 0.04, 0.02, 0.0, 0.04, 0.03, 0.0]])

    support = ATTENTION_CONCEPT_SUPPORT_SERVICE.select(alpha, 0.15)

    assert torch.equal(
        support,
        torch.tensor([[True, True, True, False, False, False, False]]),
    )


def test_support_is_empty_without_a_strong_semantic_seed() -> None:
    """Do not promote an entirely weak diffuse response into a concept region."""

    alpha = torch.tensor([[0.04, 0.03, 0.02]])

    support = ATTENTION_CONCEPT_SUPPORT_SERVICE.select(alpha, 0.15)

    assert not support.any()


def test_zero_strength_preserves_every_positive_isolated_attention_value() -> None:
    """Expose the complete positive evidence field when filtering is disabled."""

    alpha = torch.tensor([[0.0, 0.01, 0.5]])

    support = ATTENTION_CONCEPT_SUPPORT_SERVICE.select(alpha, 0.0)

    assert torch.equal(support, torch.tensor([[False, True, True]]))
