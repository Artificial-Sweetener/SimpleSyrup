# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Specify exact compact execution batches for Anima token ownership."""

from __future__ import annotations

import torch

from simple_syrup.runtime.regional_lora import (
    anima_self_attention_partition_execution as partition_execution,
)
from simple_syrup.runtime.regional_lora.anima_self_attention_coherence import (
    AnimaSelfAttentionCoherencePolicy,
    AnimaSelfAttentionCoherenceProfile,
)
from simple_syrup.runtime.regional_lora.anima_self_attention_partition import (
    AnimaSelfAttentionPartitionPlan,
)


def test_symmetric_cfg_split_batches_regions_by_equal_geometry() -> None:
    """Pack both CFG rows and region interiors into two compact call groups."""

    owners = torch.tensor(
        [
            [0, 0, -1, -1, 1, 1],
            [0, 0, -1, -1, 1, 1],
        ]
    )

    plan = AnimaSelfAttentionPartitionPlan.build(_profile(owners))
    groups = plan.regional_groups

    assert len(groups) == 1
    assert (groups[0].query_count, groups[0].key_count, groups[0].call_count) == (
        2,
        4,
        4,
    )
    assert len(plan.global_groups) == 1
    assert (
        plan.global_groups[0].query_count,
        plan.global_groups[0].key_count,
        plan.global_groups[0].call_count,
    ) == (
        2,
        6,
        2,
    )


def test_partition_relation_matches_dense_ownership_relation() -> None:
    """Cover every and only the query/key pairs allowed by ownership policy."""

    owners = torch.tensor([[0, 0, -1, 1, 1]])
    plan = AnimaSelfAttentionPartitionPlan.build(_profile(owners))
    reconstructed = torch.zeros((1, 5, 5), dtype=torch.bool)

    for group in plan.regional_groups:
        for call_index in range(group.call_count):
            batch = int(group.batch_indices[call_index])
            queries = group.query_indices[call_index]
            keys = group.key_indices[call_index]
            reconstructed[batch][queries.unsqueeze(1), keys.unsqueeze(0)] = True

    expected = (
        owners.unsqueeze(2).eq(owners.unsqueeze(1)) | owners.unsqueeze(1).eq(-1)
    ) & owners.unsqueeze(2).ne(-1)
    assert torch.equal(reconstructed, expected)


def test_different_batch_layouts_remain_exact() -> None:
    """Pack compatible shapes without assuming CFG rows share token indices."""

    owners = torch.tensor([[0, -1, 1], [0, 1, 1]])

    plan = AnimaSelfAttentionPartitionPlan.build(_profile(owners))

    assert sum(group.call_count for group in plan.regional_groups) == 4
    assert sum(group.call_count for group in plan.global_groups) == 1
    assert set(plan.query_coverage.flatten().tolist()) == {1}


def test_invalid_owners_fail_closed() -> None:
    """Reject empty, floating, or invalid-negative ownership grids."""

    for owners in (
        torch.zeros((0, 1), dtype=torch.long),
        torch.zeros((1, 1), dtype=torch.float32),
        torch.tensor([[-2]], dtype=torch.long),
    ):
        try:
            AnimaSelfAttentionPartitionPlan.build(_profile(owners))
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("Invalid ownership grids must be rejected.")


def test_compact_execution_matches_dense_masked_sdpa() -> None:
    """Numerically reproduce dense additive-mask attention for allowed pairs."""

    torch.manual_seed(1234)
    owners = torch.tensor([[0, 0, -1, 1, 1]])
    q = torch.randn((1, 5, 2, 4))
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    plan = AnimaSelfAttentionPartitionPlan.build(_profile(owners))

    compact = partition_execution.AnimaSelfAttentionPartitionExecution().execute(
        q,
        k,
        v,
        plan,
        attention=lambda q_value, k_value, v_value: (
            torch.nn.functional.scaled_dot_product_attention(
                q_value.permute(0, 2, 1, 3),
                k_value.permute(0, 2, 1, 3),
                v_value.permute(0, 2, 1, 3),
            )
            .permute(0, 2, 1, 3)
            .reshape(q_value.shape[0], q_value.shape[1], -1)
        ),
    )
    allowed = (
        owners.unsqueeze(2).eq(owners.unsqueeze(1))
        | owners.unsqueeze(2).eq(-1)
        | owners.unsqueeze(1).eq(-1)
    )
    dense = (
        torch.nn.functional.scaled_dot_product_attention(
            q.permute(0, 2, 1, 3),
            k.permute(0, 2, 1, 3),
            v.permute(0, 2, 1, 3),
            attn_mask=allowed.unsqueeze(1),
        )
        .permute(0, 2, 1, 3)
        .reshape(1, 5, -1)
    )

    torch.testing.assert_close(compact, dense, atol=1e-6, rtol=1e-6)


def test_boundary_queries_blend_regional_and_global_attention_continuously() -> None:
    """Interpolate boundary outputs without changing owned interior outputs."""

    owners = torch.tensor([[0, 0, 1, 1]])
    profile = AnimaSelfAttentionCoherenceProfile(
        query_owners=owners,
        shared_keys=torch.tensor([[False, True, True, False]]),
        global_blend=torch.tensor([[0.0, 0.5, 0.5, 0.0]]),
    )
    plan = AnimaSelfAttentionPartitionPlan.build(profile)
    q = torch.arange(4.0).reshape(1, 4, 1, 1)

    output = partition_execution.AnimaSelfAttentionPartitionExecution().execute(
        q,
        q,
        q,
        plan,
        attention=lambda packed_q, packed_k, packed_v: (
            packed_q.reshape(packed_q.shape[0], packed_q.shape[1], -1)
            + packed_k.shape[1]
        ),
    )

    assert output.flatten().tolist() == [3.0, 4.5, 5.5, 6.0]


def test_owned_interiors_retain_the_full_scene_residual() -> None:
    """Give separated subjects one low-strength global composition channel."""

    owners = torch.cat(
        (
            torch.zeros((8, 4), dtype=torch.long),
            torch.ones((8, 4), dtype=torch.long),
        ),
        dim=1,
    ).reshape(1, -1)
    profile = AnimaSelfAttentionCoherencePolicy(radius=0).resolve(
        owners,
        height=8,
        width=8,
    )
    plan = AnimaSelfAttentionPartitionPlan.build(profile)

    interior = 18
    assert not bool(profile.shared_keys[0, interior])
    assert profile.global_blend[0, interior].item() == 0.125
    assert sum(group.query_count for group in plan.global_groups) == 64


def test_global_blend_uses_installed_bfloat16_execution_dtype() -> None:
    """Convert policy weights before the dtype-strict installed lerp operation."""

    owners = torch.tensor([[0, 0, 1, 1]])
    profile = AnimaSelfAttentionCoherenceProfile(
        query_owners=owners,
        shared_keys=torch.tensor([[False, True, True, False]]),
        global_blend=torch.tensor([[0.125, 1.0, 1.0, 0.125]]),
    )
    plan = AnimaSelfAttentionPartitionPlan.build(profile)
    q = torch.arange(4.0, dtype=torch.bfloat16).reshape(1, 4, 1, 1)

    output = partition_execution.AnimaSelfAttentionPartitionExecution().execute(
        q,
        q,
        q,
        plan,
        attention=lambda packed_q, packed_k, packed_v: packed_q.reshape(
            packed_q.shape[0], packed_q.shape[1], -1
        ),
    )

    assert output.dtype is torch.bfloat16
    assert torch.equal(output.flatten(), q.flatten())


def _profile(owners: torch.Tensor) -> AnimaSelfAttentionCoherenceProfile:
    """Build a profile whose existing shared owners use global attention."""

    shared = owners.eq(-1)
    return AnimaSelfAttentionCoherenceProfile(
        query_owners=owners,
        shared_keys=shared,
        global_blend=shared.to(torch.float32),
    )
