# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Specify narrow shared coherence bands at Anima region boundaries."""

from __future__ import annotations

import torch

from simple_syrup.runtime.regional_lora.anima_self_attention_coherence import (
    AnimaSelfAttentionCoherencePolicy,
)


def test_vertical_split_builds_continuous_boundary_blend() -> None:
    """Keep regional ownership while tapering global coherence to zero."""

    owners = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]])

    profile = AnimaSelfAttentionCoherencePolicy(radius=1).resolve(
        owners,
        height=1,
        width=8,
    )

    assert profile.query_owners.tolist() == [[0, 0, 0, 0, 1, 1, 1, 1]]
    assert profile.shared_keys.tolist() == [
        [False, False, True, True, True, True, False, False]
    ]
    assert profile.global_blend.tolist() == [
        [0.125, 0.125, 0.5, 1.0, 1.0, 0.5, 0.125, 0.125]
    ]


def test_two_dimensional_corner_does_not_globalize_interiors() -> None:
    """Share the local boundary cross without erasing distant ownership."""

    owners = torch.tensor([[0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]])

    profile = AnimaSelfAttentionCoherencePolicy(
        radius=0,
        interior_global_blend=0.0,
    ).resolve(
        owners,
        height=4,
        width=4,
    )

    shared = profile.shared_keys.reshape(4, 4)
    assert torch.equal(shared[:, 1:3], torch.ones((4, 2), dtype=torch.bool))
    assert torch.equal(shared[:, 0], torch.zeros(4, dtype=torch.bool))
    assert torch.equal(shared[:, 3], torch.zeros(4, dtype=torch.bool))


def test_existing_uncovered_tokens_remain_shared() -> None:
    """Preserve canonical base coherence while adding authored boundaries."""

    owners = torch.tensor([[0, -1, 1, 1]])

    profile = AnimaSelfAttentionCoherencePolicy(
        radius=0,
        interior_global_blend=0.0,
    ).resolve(
        owners,
        height=1,
        width=4,
    )

    assert profile.query_owners.tolist() == [[0, -1, 1, 1]]
    assert profile.shared_keys.tolist() == [[True, True, True, False]]
    assert profile.global_blend.tolist() == [[1.0, 1.0, 1.0, 0.0]]


def test_geometry_mismatch_fails_closed() -> None:
    """Reject flattened ownership that cannot reconstruct the active grid."""

    try:
        AnimaSelfAttentionCoherencePolicy().resolve(
            torch.zeros((1, 3), dtype=torch.long),
            height=2,
            width=2,
        )
    except ValueError as error:
        assert "match active H/W geometry" in str(error)
    else:
        raise AssertionError("Mismatched coherence geometry must fail.")


def test_invalid_global_composition_strength_fails_closed() -> None:
    """Reject invalid channel strengths before model execution."""

    for strength in (-0.1, 1.0, True):
        try:
            AnimaSelfAttentionCoherencePolicy(interior_global_blend=strength)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid global composition strength must fail.")
