# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Characterize native regional and packed global standard-UNet graph inputs."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from simple_syrup.domain.regional_attention import RegionalAttentionBranch
from simple_syrup.domain.regional_attention_batch import (
    BatchedRegionalAttentionContexts,
    BatchedRegionalAttentionEntry,
    BatchedRegionalAttentionRegion,
    RegionalAttentionChunkBatch,
)
from simple_syrup.runtime.attention_coupling.unet_attn2_execution import (
    UnetAttn2Execution,
)
from simple_syrup.runtime.regional_attention_execution_context import (
    RegionalAttentionExecutionContext,
)
from simple_syrup.runtime.regional_lora.standard_unet_variant_base_attention import (
    StandardUnetVariantBaseAttention,
)
from simple_syrup.runtime.regional_lora.standard_unet_variant_conditioning import (
    StandardUnetVariantConditioningResolver,
)
from simple_syrup.runtime.regional_lora.standard_unet_variant_graph_attention import (
    StandardUnetVariantGraphAttentionResolver,
)
from simple_syrup.runtime.regional_lora.standard_unet_variant_invocation import (
    StandardUnetVariantInvocation,
)


class _AttentionResolver:
    """Provide the callback boundary without executing attention."""

    def resolve(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        extra_options: dict[str, Any],
    ) -> UnetAttn2Execution:
        """Reject callback execution in graph-input tests."""

        del query, context, extra_options
        raise AssertionError("Characterization must not execute attention.")


def test_regional_graph_uses_exact_native_context_without_paired_callbacks() -> None:
    """Keep each resident variant on its authored regional conditioning."""

    owner, context_session, contexts = _owner()
    invocation = _invocation()
    source = invocation.transformer_options.copy()

    with context_session.activate(contexts):
        prepared = owner.regional(0, invocation)

    assert prepared.context is contexts.regions[0].entries[0].context
    assert prepared.transformer_options is invocation.transformer_options
    assert "patches" not in prepared.transformer_options
    assert invocation.transformer_options == source


def test_global_base_uses_the_same_packed_graph_inputs() -> None:
    """Keep uncovered canvas on complete Attention Couple semantics."""

    owner, context_session, contexts = _owner()
    invocation = _invocation()

    with context_session.activate(contexts):
        prepared = owner.global_base(invocation)

    assert prepared.context is contexts.base_context
    patches = prepared.transformer_options.get("patches")
    assert isinstance(patches, dict)
    assert len(patches["attn2_patch"]) == 1
    assert len(patches["attn2_output_patch"]) == 1


def test_regional_graph_rejects_invalid_region_before_callbacks() -> None:
    """Fail closed on topology drift before preparing callback state."""

    owner, context_session, contexts = _owner()

    with (
        context_session.activate(contexts),
        pytest.raises(ValueError, match="outside conditioning"),
    ):
        owner.regional(1, _invocation())


def test_regional_graph_preserves_native_callback_options() -> None:
    """Pass a regional graph's existing native option surface through unchanged."""

    owner, context_session, contexts = _owner()
    invocation = _invocation(patches={"attn2_patch": []})

    with context_session.activate(contexts):
        prepared = owner.regional(0, invocation)

    assert prepared.transformer_options is invocation.transformer_options
    assert prepared.transformer_options == {"patches": {"attn2_patch": []}}


def test_global_base_preserves_callback_collision_failure() -> None:
    """Never replace or reorder foreign callbacks on the packed global base."""

    owner, context_session, contexts = _owner()

    with (
        context_session.activate(contexts),
        pytest.raises(ValueError, match="already contains"),
    ):
        owner.global_base(_invocation(patches={"attn2_patch": []}))


def _owner() -> tuple[
    StandardUnetVariantGraphAttentionResolver,
    RegionalAttentionExecutionContext,
    BatchedRegionalAttentionContexts,
]:
    """Return one owner with distinct global and regional contexts."""

    context_session = RegionalAttentionExecutionContext()
    owner = StandardUnetVariantGraphAttentionResolver(
        conditioning=StandardUnetVariantConditioningResolver(context_session),
        base_attention=StandardUnetVariantBaseAttention(_AttentionResolver()),
    )
    contexts = BatchedRegionalAttentionContexts(
        1,
        (RegionalAttentionChunkBatch(0, RegionalAttentionBranch.POSITIVE, 0, 1),),
        torch.tensor([[[10.0]]]),
        (
            BatchedRegionalAttentionRegion(
                0,
                (
                    BatchedRegionalAttentionEntry(
                        0,
                        torch.tensor([[[20.0]]]),
                        (1.0,),
                    ),
                ),
            ),
        ),
    )
    return owner, context_session, contexts


def _invocation(
    *,
    patches: dict[str, object] | None = None,
) -> StandardUnetVariantInvocation:
    """Return one normalized graph invocation."""

    options: dict[str, object] = {}
    if patches is not None:
        options["patches"] = patches
    return StandardUnetVariantInvocation.bind(
        (torch.zeros((1, 4, 1, 1)), torch.ones(1)),
        {
            "context": torch.zeros((1, 1, 1)),
            "control": None,
            "transformer_options": options,
        },
    )
