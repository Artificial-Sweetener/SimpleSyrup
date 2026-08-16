# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fuse active regional-LoRA B projections into ordered base accumulation."""

from __future__ import annotations

import torch

from .fused_active_accumulation_kernel import (
    MAX_FUSED_ADAPTERS_PER_LAUNCH,
    REGIONAL_LORA_FUSED_ACCUMULATION_KERNEL,
)


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
        for start in range(0, adapter_count, MAX_FUSED_ADAPTERS_PER_LAUNCH):
            stop = min(start + MAX_FUSED_ADAPTERS_PER_LAUNCH, adapter_count)
            REGIONAL_LORA_FUSED_ACCUMULATION_KERNEL.launch(
                output,
                rank_values=rank_values,
                up=up,
                multipliers=multipliers[start:stop],
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
        REGIONAL_LORA_FUSED_ACCUMULATION_KERNEL.launch(
            output,
            rank_values=rank_values,
            up=up,
            multipliers=(multiplier,),
            indices=indices,
            adapter_start=target_index,
        )
        return output

    def add_mapped(
        self,
        output: torch.Tensor,
        *,
        rank_values: torch.Tensor,
        up: torch.Tensor,
        multipliers: tuple[torch.Tensor, ...],
        indices: torch.Tensor | None,
        target_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Add declared groups from reusable unique-target rank projections."""

        self._validate_mapped_projection(
            output,
            rank_values,
            up,
            indices,
            target_indices,
        )
        active_row_count = int(rank_values.shape[0])
        self._validate_multipliers(
            output,
            multipliers,
            active_row_count=active_row_count,
            expected_count=int(target_indices.shape[0]),
        )
        if active_row_count == 0:
            return output
        for start in range(0, len(multipliers), MAX_FUSED_ADAPTERS_PER_LAUNCH):
            stop = min(start + MAX_FUSED_ADAPTERS_PER_LAUNCH, len(multipliers))
            REGIONAL_LORA_FUSED_ACCUMULATION_KERNEL.launch(
                output,
                rank_values=rank_values,
                up=up,
                multipliers=multipliers[start:stop],
                indices=indices,
                adapter_start=start,
                target_indices=target_indices,
            )
        return output

    @classmethod
    def _validate_mapped_projection(
        cls,
        output: torch.Tensor,
        rank_values: torch.Tensor,
        up: torch.Tensor,
        indices: torch.Tensor | None,
        target_indices: torch.Tensor,
    ) -> None:
        """Require one unique-target batch and valid declared group mapping."""

        if not cls.supports(output, up):
            raise ValueError("Mapped fused accumulation received an unsupported path.")
        if (
            rank_values.device != output.device
            or rank_values.dtype != output.dtype
            or rank_values.ndim != 3
            or not rank_values.is_contiguous()
        ):
            raise ValueError("Mapped fused rank values are misaligned.")
        active_rows, target_count, rank = rank_values.shape
        if tuple(up.shape) != (target_count, output.shape[1], rank):
            raise ValueError("Mapped fused B weights do not match unique ranks.")
        if (
            target_indices.device != output.device
            or target_indices.dtype is not torch.int64
            or target_indices.ndim != 1
            or int(target_indices.shape[0]) < 1
            or not target_indices.is_contiguous()
        ):
            raise ValueError("Mapped fused target indices are invalid.")
        if indices is None:
            if active_rows != int(output.shape[0]):
                raise ValueError("Dense mapped accumulation requires every output row.")
            return
        if (
            indices.device != output.device
            or indices.dtype is not torch.int64
            or indices.ndim != 1
            or int(indices.shape[0]) != active_rows
        ):
            raise ValueError("Mapped fused output indices are misaligned.")

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
