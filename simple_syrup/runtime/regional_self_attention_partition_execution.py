# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Execute compact regional self-attention partitions."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

from .regional_self_attention_partition import (
    RegionalSelfAttentionPartitionGroup,
    RegionalSelfAttentionPartitionPlan,
)


class RegionalSelfAttentionPartitionExecution:
    """Gather, batch, execute, and scatter ownership-equivalent attention."""

    def execute(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        plan: RegionalSelfAttentionPartitionPlan,
        *,
        attention: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Return flattened B/Q outputs with every query evaluated exactly once."""

        self._validate_inputs(q, k, v, plan, attention)
        output = q.new_zeros((int(q.shape[0]), int(q.shape[1]), math.prod(q.shape[2:])))
        for group in plan.regional_groups:
            self._execute_group(q, k, v, output, group, attention=attention)
        for group in plan.global_groups:
            global_output = self._attend_group(q, k, v, group, attention=attention)
            batches = group.batch_indices.unsqueeze(1)
            regional_output = output[batches, group.query_indices]
            weights = (
                plan.global_blend[batches, group.query_indices]
                .unsqueeze(-1)
                .to(device=regional_output.device, dtype=regional_output.dtype)
            )
            output[batches, group.query_indices] = torch.lerp(
                regional_output, global_output, weights
            )
        return output

    def _execute_group(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        output: torch.Tensor,
        group: RegionalSelfAttentionPartitionGroup,
        *,
        attention: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> None:
        """Scatter one regional group into the accumulated output."""

        batches = group.batch_indices.unsqueeze(1)
        output[batches, group.query_indices] = self._attend_group(
            q, k, v, group, attention=attention
        )

    @staticmethod
    def _attend_group(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        group: RegionalSelfAttentionPartitionGroup,
        *,
        attention: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Gather and execute one compact attention group."""

        batches = group.batch_indices.unsqueeze(1)
        packed_output = attention(
            q[batches, group.query_indices],
            k[batches, group.key_indices],
            v[batches, group.key_indices],
        )
        expected = (
            group.call_count,
            group.query_count,
            math.prod(q.shape[2:]),
        )
        if tuple(packed_output.shape) != expected:
            raise ValueError(
                "Compact regional attention returned an invalid shape; "
                f"expected {expected}, observed {tuple(packed_output.shape)}."
            )
        return packed_output

    @staticmethod
    def _validate_inputs(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        plan: RegionalSelfAttentionPartitionPlan,
        attention: object,
    ) -> None:
        """Require aligned B/Q projection tensors and one compatible plan."""

        if not all(isinstance(tensor, torch.Tensor) for tensor in (q, k, v)):
            raise TypeError("Compact regional attention projections must be tensors.")
        if q.ndim < 3 or q.shape != k.shape or q.shape != v.shape:
            raise ValueError(
                "Compact regional attention requires aligned B/Q projections."
            )
        if not isinstance(plan, RegionalSelfAttentionPartitionPlan):
            raise TypeError("Compact regional attention requires a partition plan.")
        if tuple(plan.query_coverage.shape) != tuple(q.shape[:2]):
            raise ValueError(
                "Compact regional attention plan does not match B/Q geometry."
            )
        if not callable(attention):
            raise TypeError("Compact regional attention backend must be callable.")


REGIONAL_SELF_ATTENTION_PARTITION_EXECUTION = RegionalSelfAttentionPartitionExecution()
