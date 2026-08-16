# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# mypy: disable-error-code="no-untyped-def"
# ruff: noqa: ANN001, ANN202

"""Launch ordered fused regional-LoRA accumulation without multiplier packing."""

from __future__ import annotations

from typing import Any, cast

import torch
import triton  # type: ignore[import-untyped]
import triton.language as tl  # type: ignore[import-untyped]

MAX_FUSED_ADAPTERS_PER_LAUNCH = 8
_BLOCK_ROWS = 16
_BLOCK_OUTPUT_FEATURES = 64


@triton.jit  # type: ignore[untyped-decorator]
def _fused_active_accumulation_kernel(
    output,
    rank_values,
    up,
    target_indices,
    multiplier_0,
    multiplier_1,
    multiplier_2,
    multiplier_3,
    multiplier_4,
    multiplier_5,
    multiplier_6,
    multiplier_7,
    indices,
    active_row_count,
    output_feature_count,
    rank,
    rank_row_stride,
    rank_adapter_stride,
    up_adapter_stride,
    adapter_start,
    adapter_count: tl.constexpr,
    use_indices: tl.constexpr,
    use_target_indices: tl.constexpr,
    execution_dtype: tl.constexpr,
    block_rows: tl.constexpr,
    block_output_features: tl.constexpr,
    block_rank: tl.constexpr,
):
    """Project B and add separate multiplier vectors in exact declared order."""

    active_rows = tl.program_id(0) * block_rows + tl.arange(0, block_rows)
    output_features = tl.program_id(1) * block_output_features + tl.arange(
        0, block_output_features
    )
    active_row_mask = active_rows < active_row_count
    output_feature_mask = output_features < output_feature_count
    if use_indices:
        output_rows = tl.load(indices + active_rows, mask=active_row_mask, other=0)
    else:
        output_rows = active_rows
    output_offsets = (
        output_rows[:, None] * output_feature_count + output_features[None, :]
    )
    output_mask = active_row_mask[:, None] & output_feature_mask[None, :]
    value = tl.load(output + output_offsets, mask=output_mask, other=0.0)
    rank_offsets = tl.arange(0, block_rank)
    rank_mask = rank_offsets < rank
    for adapter_index in range(adapter_count):
        group_index = adapter_start + adapter_index
        if use_target_indices:
            source_adapter_index = tl.load(target_indices + group_index)
        else:
            source_adapter_index = group_index
        adapter_rank_offsets = (
            active_rows[:, None] * rank_row_stride
            + source_adapter_index * rank_adapter_stride
            + rank_offsets[None, :]
        )
        projected_rank = tl.load(
            rank_values + adapter_rank_offsets,
            mask=active_row_mask[:, None] & rank_mask[None, :],
            other=0.0,
        )
        if adapter_index == 0:
            multiplier_pointer = multiplier_0
        elif adapter_index == 1:
            multiplier_pointer = multiplier_1
        elif adapter_index == 2:
            multiplier_pointer = multiplier_2
        elif adapter_index == 3:
            multiplier_pointer = multiplier_3
        elif adapter_index == 4:
            multiplier_pointer = multiplier_4
        elif adapter_index == 5:
            multiplier_pointer = multiplier_5
        elif adapter_index == 6:
            multiplier_pointer = multiplier_6
        else:
            multiplier_pointer = multiplier_7
        multiplier = tl.load(
            multiplier_pointer + active_rows,
            mask=active_row_mask,
            other=0.0,
        )
        projected_rank = (projected_rank * multiplier[:, None]).to(execution_dtype)
        up_offsets = (
            source_adapter_index * up_adapter_stride
            + output_features[:, None] * rank
            + rank_offsets[None, :]
        )
        up_values = tl.load(
            up + up_offsets,
            mask=output_feature_mask[:, None] & rank_mask[None, :],
            other=0.0,
        )
        delta = tl.dot(projected_rank, tl.trans(up_values)).to(execution_dtype)
        value = (value + delta).to(execution_dtype)
    tl.store(output + output_offsets, value, mask=output_mask)


class RegionalLoraFusedAccumulationKernel:
    """Own direct multiplier-pointer transport and Triton launch geometry."""

    @classmethod
    def launch(
        cls,
        output: torch.Tensor,
        *,
        rank_values: torch.Tensor,
        up: torch.Tensor,
        multipliers: tuple[torch.Tensor, ...],
        indices: torch.Tensor | None,
        adapter_start: int,
        target_indices: torch.Tensor | None = None,
    ) -> None:
        """Launch one ordered adapter chunk without allocating packed multipliers."""

        if not 1 <= len(multipliers) <= MAX_FUSED_ADAPTERS_PER_LAUNCH:
            raise ValueError("Fused multiplier chunk size is unsupported.")
        padded = multipliers + (multipliers[-1],) * (
            MAX_FUSED_ADAPTERS_PER_LAUNCH - len(multipliers)
        )
        active_row_count = int(rank_values.shape[0])
        rank = int(rank_values.shape[2])
        index_values = rank_values if indices is None else indices
        target_index_values = rank_values if target_indices is None else target_indices
        block_rank = max(16, triton.next_power_of_2(rank))
        grid = (
            triton.cdiv(active_row_count, _BLOCK_ROWS),
            triton.cdiv(int(output.shape[1]), _BLOCK_OUTPUT_FEATURES),
        )
        kernel = cast(Any, _fused_active_accumulation_kernel)
        kernel[grid](
            output,
            rank_values,
            up,
            target_index_values,
            *padded,
            index_values,
            active_row_count,
            int(output.shape[1]),
            rank,
            int(rank_values.stride(0)),
            int(rank_values.stride(1)),
            int(up.stride(0)),
            adapter_start,
            adapter_count=len(multipliers),
            use_indices=indices is not None,
            use_target_indices=target_indices is not None,
            execution_dtype=cls._triton_dtype(output.dtype),
            block_rows=_BLOCK_ROWS,
            block_output_features=_BLOCK_OUTPUT_FEATURES,
            block_rank=block_rank,
            num_warps=4,
        )

    @staticmethod
    def _triton_dtype(dtype: torch.dtype) -> Any:
        """Map the verified execution dtype to its Triton scalar type."""

        if dtype is torch.bfloat16:
            return tl.bfloat16
        if dtype is torch.float16:
            return tl.float16
        raise TypeError(f"Fused active accumulation does not support {dtype}.")


REGIONAL_LORA_FUSED_ACCUMULATION_KERNEL = RegionalLoraFusedAccumulationKernel()
