"""Characterize installed Comfy metadata for UNet attn2 resolution parsing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
import torch
from comfy.ldm.modules.attention import SpatialTransformer


@dataclass(frozen=True, slots=True)
class _CallbackObservation:
    """Retain one paired callback's tensors and exact metadata identity."""

    query_shape: tuple[int, ...]
    context_shape: tuple[int, ...]
    input_options: dict[str, Any]
    output_options: dict[str, Any]


@pytest.mark.parametrize(
    (
        "block",
        "transformer_index",
        "activation_shape",
        "original_shape",
        "context_dimension",
    ),
    [
        (("input", 4), 2, (2, 32, 8, 12), [2, 4, 16, 24], 16),
        (("middle", 0), 7, (2, 32, 4, 6), [2, 4, 16, 24], 16),
        (("output", 5), 10, (2, 32, 2, 3), [2, 4, 16, 24], 32),
    ],
)
def test_spatial_transformer_publishes_exact_rectangular_attn2_geometry(
    block: tuple[str, int],
    transformer_index: int,
    activation_shape: tuple[int, int, int, int],
    original_shape: list[int],
    context_dimension: int,
) -> None:
    """Pin H/W metadata without inferring a grid from token count alone."""

    observations = _run_transformer(
        block=block,
        transformer_index=transformer_index,
        activation_shape=activation_shape,
        original_shape=original_shape,
        context_dimension=context_dimension,
        depth=1,
    )

    assert len(observations) == 1
    observation = observations[0]
    height, width = activation_shape[-2:]
    assert observation.query_shape == (activation_shape[0], height * width, 32)
    assert observation.context_shape == (activation_shape[0], 3, context_dimension)
    assert observation.input_options is observation.output_options
    assert observation.input_options["activations_shape"] == list(activation_shape)
    assert observation.input_options["original_shape"] == original_shape
    assert observation.input_options["block"] == block
    assert observation.input_options["transformer_index"] == transformer_index
    assert observation.input_options["block_index"] == 0
    assert observation.input_options["n_heads"] == 1
    assert observation.input_options["dim_head"] == 32


def test_spatial_transformer_depth_tracks_blocks_at_one_resolution() -> None:
    """Distinguish repeated blocks from spatial-resolution changes."""

    observations = _run_transformer(
        block=("output", 8),
        transformer_index=13,
        activation_shape=(1, 32, 3, 5),
        original_shape=[1, 4, 12, 20],
        context_dimension=32,
        depth=2,
    )

    assert len(observations) == 2
    assert [item.input_options["block_index"] for item in observations] == [0, 1]
    assert all(
        item.input_options["activations_shape"] == [1, 32, 3, 5]
        for item in observations
    )
    assert all(item.query_shape == (1, 15, 32) for item in observations)
    assert observations[0].input_options is not observations[1].input_options


def _run_transformer(
    *,
    block: tuple[str, int],
    transformer_index: int,
    activation_shape: tuple[int, int, int, int],
    original_shape: list[int],
    context_dimension: int,
    depth: int,
) -> tuple[_CallbackObservation, ...]:
    """Run one installed SpatialTransformer with paired recording callbacks."""

    pending: dict[int, tuple[tuple[int, ...], tuple[int, ...], dict[str, Any]]] = {}
    observations: list[_CallbackObservation] = []

    def input_patch(
        query: torch.Tensor,
        context: torch.Tensor,
        value: torch.Tensor,
        options: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Record pre-attention tensors and callback-local metadata."""

        assert value is context
        block_index = options["block_index"]
        assert isinstance(block_index, int)
        pending[block_index] = (
            tuple(query.shape),
            tuple(context.shape),
            options,
        )
        return query, context, value

    def output_patch(
        output: torch.Tensor,
        options: dict[str, Any],
    ) -> torch.Tensor:
        """Pair post-attention output with the exact input callback mapping."""

        block_index = options["block_index"]
        assert isinstance(block_index, int)
        query_shape, context_shape, input_options = pending.pop(block_index)
        observations.append(
            _CallbackObservation(
                query_shape,
                context_shape,
                input_options,
                options,
            )
        )
        return output

    transformer = SpatialTransformer(
        in_channels=32,
        n_heads=1,
        d_head=32,
        depth=depth,
        context_dim=context_dimension,
        use_checkpoint=False,
    )
    model_input = torch.randn(activation_shape)
    context = torch.randn(activation_shape[0], 3, context_dimension)
    options: dict[str, Any] = {
        "block": block,
        "transformer_index": transformer_index,
        "original_shape": original_shape,
        "patches": {
            "attn2_patch": [input_patch],
            "attn2_output_patch": [output_patch],
        },
    }

    result = transformer(model_input, context=context, transformer_options=options)

    assert isinstance(result, torch.Tensor)
    assert result.shape == model_input.shape
    assert not pending
    return tuple(observations)
