# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# mypy: disable-error-code="no-untyped-def"
# ruff: noqa: ANN001, ANN202

"""Fuse active regional-LoRA B projections into ordered base accumulation."""

from __future__ import annotations

from typing import Any, cast

import torch
import triton  # type: ignore[import-untyped]
import triton.language as tl  # type: ignore[import-untyped]

_MAX_ADAPTERS_PER_LAUNCH = 8
_BLOCK_ROWS = 16
_BLOCK_OUTPUT_FEATURES = 64


@triton.jit  # type: ignore[untyped-decorator]
def _fused_active_accumulation_kernel(
    output,
    rank_values,
    up,
    multipliers,
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
    execution_dtype: tl.constexpr,
    block_rows: tl.constexpr,
    block_output_features: tl.constexpr,
    block_rank: tl.constexpr,
):
    """Project B and add adapters in exact execution-dtype order."""

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
        source_adapter_index = adapter_start + adapter_index
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
        multiplier = tl.load(
            multipliers + active_rows * adapter_count + adapter_index,
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


class RegionalLoraFusedActiveAccumulator:
    """Own verified CUDA fusion for B projection and ordered output updates."""

    @staticmethod
    def supports(output: torch.Tensor, up: torch.Tensor) -> bool:
        """Admit only verified contiguous CUDA bf16/fp16 projection shapes."""

        return (
            isinstance(output, torch.Tensor)
            and isinstance(up, torch.Tensor)
            and output.device.type == "cuda"
            and up.device == output.device
            and output.dtype in (torch.bfloat16, torch.float16)
            and up.dtype == output.dtype
            and output.ndim == 2
            and up.ndim == 3
            and output.is_contiguous()
            and up.is_contiguous()
        )

    def add(
        self,
        output: torch.Tensor,
        *,
        rank_values: torch.Tensor,
        up: torch.Tensor,
        multipliers: tuple[torch.Tensor, ...],
        indices: torch.Tensor | None,
    ) -> torch.Tensor:
        """Add one or more active low-rank outputs directly into base rows."""

        self._validate_projection(output, rank_values, up, indices)
        active_row_count, adapter_count = (
            int(rank_values.shape[0]),
            int(rank_values.shape[1]),
        )
        self._validate_multipliers(
            output,
            multipliers,
            active_row_count=active_row_count,
            expected_count=adapter_count,
        )
        if active_row_count == 0:
            return output
        for start in range(0, adapter_count, _MAX_ADAPTERS_PER_LAUNCH):
            stop = min(start + _MAX_ADAPTERS_PER_LAUNCH, adapter_count)
            multiplier_chunk = (
                multipliers[start].unsqueeze(1)
                if stop - start == 1
                else torch.stack(multipliers[start:stop], dim=1)
            )
            self._launch(
                output,
                rank_values=rank_values,
                up=up,
                multipliers=multiplier_chunk,
                indices=indices,
                adapter_start=start,
            )
        return output

    def add_target(
        self,
        output: torch.Tensor,
        *,
        rank_values: torch.Tensor,
        up: torch.Tensor,
        multiplier: torch.Tensor,
        indices: torch.Tensor | None,
        target_index: int,
    ) -> torch.Tensor:
        """Add one target from a contiguous shared rank batch without slicing."""

        self._validate_projection(output, rank_values, up, indices)
        if (
            isinstance(target_index, bool)
            or not isinstance(target_index, int)
            or not 0 <= target_index < int(rank_values.shape[1])
        ):
            raise IndexError("Fused compatible target index is out of range.")
        active_row_count = int(rank_values.shape[0])
        self._validate_multipliers(
            output,
            (multiplier,),
            active_row_count=active_row_count,
            expected_count=1,
        )
        if active_row_count == 0:
            return output
        self._launch(
            output,
            rank_values=rank_values,
            up=up,
            multipliers=multiplier.unsqueeze(1),
            indices=indices,
            adapter_start=target_index,
        )
        return output

    @classmethod
    def _launch(
        cls,
        output: torch.Tensor,
        *,
        rank_values: torch.Tensor,
        up: torch.Tensor,
        multipliers: torch.Tensor,
        indices: torch.Tensor | None,
        adapter_start: int,
    ) -> None:
        """Launch one selected contiguous adapter range over shared rank storage."""

        active_row_count = int(rank_values.shape[0])
        adapter_count = int(multipliers.shape[1])
        rank = int(rank_values.shape[2])
        index_values = rank_values if indices is None else indices
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
            multipliers,
            index_values,
            active_row_count,
            int(output.shape[1]),
            rank,
            int(rank_values.stride(0)),
            int(rank_values.stride(1)),
            int(up.stride(0)),
            adapter_start,
            adapter_count=adapter_count,
            use_indices=indices is not None,
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

    @classmethod
    def _validate_projection(
        cls,
        output: torch.Tensor,
        rank_values: torch.Tensor,
        up: torch.Tensor,
        indices: torch.Tensor | None,
    ) -> None:
        """Require one complete aligned CUDA projection contract."""

        if not cls.supports(output, up):
            raise ValueError("Fused active accumulation received an unsupported path.")
        if (
            rank_values.device != output.device
            or rank_values.dtype != output.dtype
            or rank_values.ndim != 3
            or not rank_values.is_contiguous()
        ):
            raise ValueError("Fused active rank values are misaligned.")
        active_rows, adapter_count, rank = rank_values.shape
        if tuple(up.shape) != (adapter_count, output.shape[1], rank):
            raise ValueError("Fused active B weights do not match rank values.")
        if indices is None:
            if active_rows != int(output.shape[0]):
                raise ValueError("Dense fused accumulation requires every output row.")
            return
        if (
            indices.device != output.device
            or indices.dtype is not torch.int64
            or indices.ndim != 1
            or int(indices.shape[0]) != active_rows
        ):
            raise ValueError("Fused active output indices are misaligned.")

    @staticmethod
    def _validate_multipliers(
        output: torch.Tensor,
        multipliers: tuple[torch.Tensor, ...],
        *,
        active_row_count: int,
        expected_count: int,
    ) -> None:
        """Require one exact selected-adapter multiplier matrix contract."""

        if len(multipliers) != expected_count or any(
            multiplier.device != output.device
            or multiplier.dtype != output.dtype
            or multiplier.ndim != 1
            or int(multiplier.shape[0]) != active_row_count
            for multiplier in multipliers
        ):
            raise ValueError("Fused active multipliers are misaligned.")


REGIONAL_LORA_FUSED_ACTIVE_ACCUMULATOR = RegionalLoraFusedActiveAccumulator()
