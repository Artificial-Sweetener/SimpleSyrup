# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Characterize installed Comfy's paired UNet attn2 callback contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from comfy.ldm.modules.attention import BasicTransformerBlock
from torch import nn


class _ZeroAttention(nn.Module):
    """Return a zero self-attention contribution while recording no policy."""

    def forward(
        self,
        query: torch.Tensor,
        *,
        context: torch.Tensor | None,
        value: torch.Tensor | None,
        transformer_options: dict[str, Any],
    ) -> torch.Tensor:
        """Return a zero tensor matching the supplied query."""

        del context, value, transformer_options
        return torch.zeros_like(query)


@dataclass
class _Attn2Observations:
    """Retain typed values observed across the installed callback sequence."""

    events: list[str] = field(default_factory=list)
    input_extra: dict[str, Any] | None = None
    attention_query: torch.Tensor | None = None
    attention_context: torch.Tensor | None = None
    attention_value: torch.Tensor | None = None
    attention_options: dict[str, Any] | None = None
    output_value: torch.Tensor | None = None
    output_extra: dict[str, Any] | None = None


class _RecordingCrossAttention(nn.Module):
    """Record the patched tensors received by native cross-attention."""

    def __init__(self, observations: _Attn2Observations) -> None:
        """Retain the ordered callback event sink."""

        super().__init__()
        self._observations = observations

    def forward(
        self,
        query: torch.Tensor,
        *,
        context: torch.Tensor | None,
        value: torch.Tensor | None,
        transformer_options: dict[str, Any],
    ) -> torch.Tensor:
        """Return a recognizable complete-attention output."""

        if context is None or value is None:
            raise AssertionError("Characterized attn2 call requires context and value.")
        self._observations.events.append("attention")
        self._observations.attention_query = query.clone()
        self._observations.attention_context = context
        self._observations.attention_value = value
        self._observations.attention_options = transformer_options
        return torch.full_like(query, 5.0)


class _ZeroFeedForward(nn.Module):
    """Return zero so the installed residual ordering remains observable."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Return a zero tensor matching the feed-forward input."""

        return torch.zeros_like(value)


def test_installed_basic_transformer_block_pairs_attn2_callbacks_around_attention() -> (
    None
):
    """Pin input, native attention, output, residual, and feed-forward ordering."""

    observations = _Attn2Observations()
    block = BasicTransformerBlock(
        dim=2,
        n_heads=1,
        d_head=2,
        context_dim=2,
        checkpoint=False,
    )
    block.norm1 = nn.Identity()
    block.attn1 = _ZeroAttention()
    block.norm2 = nn.Identity()
    block.attn2 = _RecordingCrossAttention(observations)
    block.norm3 = nn.Identity()
    block.ff = _ZeroFeedForward()

    source_query = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    source_context = torch.tensor([[[10.0, 11.0]]])
    input_options: dict[str, Any] = {
        "block": ("input", 2),
        "block_index": 3,
        "original_shape": [1, 4, 2, 2],
    }

    def input_patch(
        query: torch.Tensor,
        context: torch.Tensor,
        value: torch.Tensor,
        extra_options: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Expand one source row to two recognizable attention branches."""

        observations.events.append("input")
        observations.input_extra = extra_options
        assert value is context
        return (
            torch.cat((query, query + 100.0)),
            torch.cat((context, context + 200.0)),
            torch.cat((value, value + 300.0)),
        )

    def output_patch(
        output: torch.Tensor,
        extra_options: dict[str, Any],
    ) -> torch.Tensor:
        """Restore the original batch before Comfy adds the UNet residual."""

        observations.events.append("output")
        observations.output_value = output.clone()
        observations.output_extra = extra_options
        return output[:1] + 7.0

    transformer_options = {
        **input_options,
        "patches": {
            "attn2_patch": [input_patch],
            "attn2_output_patch": [output_patch],
        },
    }
    result = block(
        source_query,
        context=source_context,
        transformer_options=transformer_options,
    )

    assert observations.events == ["input", "attention", "output"]
    input_extra = observations.input_extra
    attention_query = observations.attention_query
    attention_context = observations.attention_context
    attention_value = observations.attention_value
    attention_options = observations.attention_options
    output_value = observations.output_value
    output_extra = observations.output_extra
    assert input_extra is not None
    assert attention_query is not None
    assert attention_context is not None
    assert attention_value is not None
    assert attention_options is not None
    assert output_value is not None
    assert output_extra is not None
    assert input_extra is output_extra
    assert input_extra["block"] == ("input", 2)
    assert input_extra["block_index"] == 3
    assert input_extra["original_shape"] == [1, 4, 2, 2]
    assert input_extra["n_heads"] == 1
    assert input_extra["dim_head"] == 2
    assert attention_options is transformer_options
    assert torch.equal(
        attention_query,
        torch.cat((source_query, source_query + 100.0)),
    )
    assert torch.equal(
        attention_context,
        torch.cat((source_context, source_context + 200.0)),
    )
    assert torch.equal(
        attention_value,
        torch.cat((source_context, source_context + 300.0)),
    )
    assert torch.equal(output_value, torch.full((2, 2, 2), 5.0))
    assert result.shape == source_query.shape
    assert torch.equal(result, source_query + 12.0)
