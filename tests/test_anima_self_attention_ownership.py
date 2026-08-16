# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Specify mask-authoritative ownership for Anima image self-attention."""

from __future__ import annotations

import torch

from simple_syrup.runtime.regional_self_attention_ownership import (
    RegionalSelfAttentionOwnershipPolicy,
)


def test_hard_regions_block_cross_region_image_attention() -> None:
    """Allow image tokens to communicate only within their authored owner."""

    masks = torch.tensor(
        [
            [[1.0, 1.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0, 1.0]],
        ]
    )

    relation = RegionalSelfAttentionOwnershipPolicy().allowed_relation(masks)

    assert torch.equal(
        relation,
        torch.tensor(
            [
                [
                    [True, True, False, False],
                    [True, True, False, False],
                    [False, False, True, True],
                    [False, False, True, True],
                ]
            ]
        ),
    )


def test_uncovered_tokens_are_shared_coherence_tokens() -> None:
    """Let uncovered base tokens exchange information with every owner."""

    masks = torch.tensor(
        [
            [[1.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0]],
        ]
    )

    relation = RegionalSelfAttentionOwnershipPolicy().allowed_relation(masks)

    assert bool(relation[0, 0, 2]) is False
    assert bool(relation[0, 0, 1]) is True
    assert bool(relation[0, 1].all()) is True
    assert bool(relation[0, 2, 1]) is True


def test_overlap_uses_maximum_value_then_stable_region_order() -> None:
    """Resolve overlaps deterministically without weakening authored masks."""

    masks = torch.tensor(
        [
            [[0.7, 0.8, 0.0]],
            [[0.9, 0.8, 1.0]],
        ]
    )

    owners = RegionalSelfAttentionOwnershipPolicy().token_owners(masks)

    assert owners.tolist() == [[1, 0, 1]]


def test_cfg_batches_keep_independent_ownership_relations() -> None:
    """Preserve each expanded model batch's projected mask authority."""

    masks = torch.tensor(
        [
            [[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
        ]
    )

    relation = RegionalSelfAttentionOwnershipPolicy().allowed_relation(masks)

    assert relation.shape == (2, 3, 3)
    assert bool(relation[0, 0, 2]) is False
    assert bool(relation[1, 0, 2]) is True
    assert bool(relation[1, 1].all()) is True


def test_invalid_mask_values_fail_closed() -> None:
    """Reject negative and non-finite masks before attention execution."""

    policy = RegionalSelfAttentionOwnershipPolicy()

    for masks in (
        torch.tensor([[[-0.1]]]),
        torch.tensor([[[float("nan")]]]),
    ):
        try:
            policy.token_owners(masks)
        except ValueError as error:
            assert "finite values in the inclusive range" in str(error)
        else:
            raise AssertionError("Invalid ownership masks must be rejected.")
