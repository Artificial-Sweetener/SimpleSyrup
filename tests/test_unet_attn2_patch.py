"""Verify the paired Comfy callback adapter for standard-UNet coupling."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from comfy.ldm.modules.attention import BasicTransformerBlock
from torch import nn

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
from simple_syrup.runtime.attention_coupling.unet_attn2_execution_resolver import (
    StaticUnetAttn2ExecutionResolver,
)
from simple_syrup.runtime.attention_coupling.unet_attn2_patch import (
    UnetAttn2PatchPair,
)


class _ZeroAttention(nn.Module):
    """Return zero while counting the retained self-attention trajectory."""

    def __init__(self) -> None:
        """Initialize the invocation counter."""

        super().__init__()
        self.calls = 0

    def forward(
        self,
        query: torch.Tensor,
        *,
        context: torch.Tensor | None,
        value: torch.Tensor | None,
        transformer_options: dict[str, Any],
    ) -> torch.Tensor:
        """Return a zero tensor after one observed invocation."""

        del context, value, transformer_options
        self.calls += 1
        return torch.zeros_like(query)


class _RegionalAttention(nn.Module):
    """Evaluate recognizable complete branch outputs in one invocation."""

    def __init__(self) -> None:
        """Initialize the invocation counter."""

        super().__init__()
        self.calls = 0

    def forward(
        self,
        query: torch.Tensor,
        *,
        context: torch.Tensor | None,
        value: torch.Tensor | None,
        transformer_options: dict[str, Any],
    ) -> torch.Tensor:
        """Add each branch context mean to its matching query."""

        del value, transformer_options
        self.calls += 1
        if context is None:
            raise AssertionError("Test cross-attention requires context.")
        return query + context.mean(dim=1, keepdim=True)


class _CountingZeroFeedForward(nn.Module):
    """Return zero while counting the retained feed-forward trajectory."""

    def __init__(self) -> None:
        """Initialize the invocation counter."""

        super().__init__()
        self.calls = 0

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Return a zero tensor after one observed invocation."""

        self.calls += 1
        return torch.zeros_like(value)


def test_unet_patch_runs_one_attention_and_restores_before_residual() -> None:
    """Keep one UNet trajectory while batching regional cross-attention rows."""

    contexts = _contexts()
    masks = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    execution = UnetAttn2Execution(contexts, masks, (1.0, 1.0))
    patches = UnetAttn2PatchPair(StaticUnetAttn2ExecutionResolver(execution))
    block = BasicTransformerBlock(
        dim=1,
        n_heads=1,
        d_head=1,
        context_dim=1,
        checkpoint=False,
    )
    self_attention = _ZeroAttention()
    cross_attention = _RegionalAttention()
    feed_forward = _CountingZeroFeedForward()
    block.norm1 = nn.Identity()
    block.attn1 = self_attention
    block.norm2 = nn.Identity()
    block.attn2 = cross_attention
    block.norm3 = nn.Identity()
    block.ff = feed_forward
    query = torch.tensor([[[1.0], [2.0]]])

    actual = block(
        query,
        context=contexts.base_context,
        transformer_options={
            "patches": {
                "attn2_patch": [patches.input_patch],
                "attn2_output_patch": [patches.output_patch],
            }
        },
    )

    expected_attention = torch.tensor([[[31.0], [72.0]]])
    assert torch.equal(actual, query + expected_attention)
    assert actual.shape == query.shape
    assert self_attention.calls == 1
    assert cross_attention.calls == 1
    assert feed_forward.calls == 1


def test_unet_patch_clears_callback_state_after_output_failure() -> None:
    """Remove callback-local state even when restoration rejects native output."""

    execution = UnetAttn2Execution(
        _contexts(),
        torch.ones(2, 1, 2),
        (1.0, 1.0),
    )
    patches = UnetAttn2PatchPair(StaticUnetAttn2ExecutionResolver(execution))
    options: dict[str, Any] = {}
    query = torch.ones(1, 2, 1)
    context = execution.contexts.base_context
    expanded = patches.input_patch(query, context, context, options)

    with pytest.raises(ValueError, match="invalid size"):
        patches.output_patch(expanded[0][:-1], options)
    with pytest.raises(ValueError, match="no matching input state"):
        patches.output_patch(expanded[0], options)

    patches.input_patch(query, context, context, options)


def _contexts() -> BatchedRegionalAttentionContexts:
    """Return one base and two single-entry regional contexts."""

    return BatchedRegionalAttentionContexts(
        latent_batch_size=1,
        chunks=(
            RegionalAttentionChunkBatch(
                0,
                RegionalAttentionBranch.POSITIVE,
                0,
                1,
            ),
        ),
        base_context=torch.tensor([[[10.0]]]),
        regions=(
            BatchedRegionalAttentionRegion(
                0,
                (
                    BatchedRegionalAttentionEntry(
                        0,
                        torch.tensor([[[30.0]]]),
                        (1.0,),
                    ),
                ),
            ),
            BatchedRegionalAttentionRegion(
                1,
                (
                    BatchedRegionalAttentionEntry(
                        0,
                        torch.tensor([[[70.0]]]),
                        (1.0,),
                    ),
                ),
            ),
        ),
    )
