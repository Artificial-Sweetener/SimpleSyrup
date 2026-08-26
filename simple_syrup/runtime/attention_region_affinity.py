# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Calculate and materialize compact selected-token attention affinities."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import torch

from ..domain.attention_region_maps import AttentionTokenSpan, CapturedAttentionMap


@dataclass(frozen=True, slots=True)
class PendingAttentionMap:
    """Retain one compact device-local map until sampling has completed."""

    label: str
    values: torch.Tensor
    progress: float
    layer_key: str
    batch_index: int
    spatial_height: int
    spatial_width: int


class AttentionAffinityCalculator:
    """Own selected-key probability estimation and deferred device transfer."""

    def head_tensors(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        heads: int,
        skip_reshape: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize Comfy's standard and pre-shaped attention tensor layouts."""

        if type(heads) is not int or heads < 1:
            raise ValueError("Attention capture head count must be positive.")
        if skip_reshape and query.ndim == 4 and key.ndim == 4:
            return query, key
        if query.ndim != 3 or key.ndim != 3:
            raise ValueError("Attention capture received unsupported Q/K tensor ranks.")
        if int(query.shape[-1]) % heads or int(key.shape[-1]) % heads:
            raise ValueError("Attention Q/K channels must divide evenly across heads.")
        q = query.view(query.shape[0], query.shape[1], heads, -1).permute(0, 2, 1, 3)
        k = key.view(key.shape[0], key.shape[1], heads, -1).permute(0, 2, 1, 3)
        return q, k

    def positive_rows(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        raw_branches: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select positive CFG rows while retaining ordinary batch members."""

        if not isinstance(raw_branches, list) or not raw_branches:
            return query, key
        if any(type(branch) is not int for branch in raw_branches):
            raise TypeError("Attention cond_or_uncond entries must be integers.")
        if int(query.shape[0]) % len(raw_branches):
            raise ValueError("Attention batch cannot be partitioned by cond_or_uncond.")
        rows_per_branch = int(query.shape[0]) // len(raw_branches)
        row_indices = [
            index
            for branch_index, branch in enumerate(raw_branches)
            if branch == 0
            for index in range(
                branch_index * rows_per_branch,
                (branch_index + 1) * rows_per_branch,
            )
        ]
        if not row_indices:
            return query[:0], key[:0]
        indices = torch.tensor(row_indices, device=query.device, dtype=torch.int64)
        return query.index_select(0, indices), key.index_select(0, indices)

    def log_denominator(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        token_budget: int,
    ) -> torch.Tensor:
        """Estimate softmax normalization from a deterministic bounded key sample."""

        scale = math.sqrt(float(query.shape[-1]))
        context_tokens = int(key.shape[-2])
        sample_count = min(context_tokens, token_budget)
        if sample_count < context_tokens:
            indices = (
                torch.linspace(
                    0,
                    context_tokens - 1,
                    sample_count,
                    device=key.device,
                )
                .round()
                .to(dtype=torch.int64)
            )
            key = key.index_select(-2, indices)
        denominator: torch.Tensor | None = None
        for key_chunk in torch.split(key.float(), 32, dim=-2):
            logits = torch.einsum("bhqd,bhkd->bhqk", query.float(), key_chunk) / scale
            chunk_denominator = torch.logsumexp(logits, dim=-1)
            denominator = (
                chunk_denominator
                if denominator is None
                else torch.logaddexp(denominator, chunk_denominator)
            )
        if denominator is None:
            raise ValueError("Attention context cannot be empty.")
        correction = math.log(float(context_tokens) / float(sample_count))
        return denominator + correction

    def capture_spans(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        spans: tuple[AttentionTokenSpan, ...],
        progress: float,
        layer_key: str,
        denominator: torch.Tensor,
        spatial_height: int,
        spatial_width: int,
    ) -> tuple[PendingAttentionMap, ...]:
        """Compute all native span maps from one unioned selected-key projection."""

        union_indices = tuple(
            sorted({index for span in spans for index in span.token_indices})
        )
        if not union_indices:
            return ()
        indices = torch.tensor(union_indices, device=key.device, dtype=torch.int64)
        selected = key.index_select(-2, indices)
        logits = torch.einsum("bhqd,bhkd->bhqk", query.float(), selected.float())
        logits = logits / math.sqrt(float(query.shape[-1]))
        log_probability = (logits - denominator.unsqueeze(-1)).clamp_max(0.0)
        probability = torch.exp(log_probability)
        union_positions = {
            token_index: index for index, token_index in enumerate(union_indices)
        }
        pending: list[PendingAttentionMap] = []
        for span in spans:
            positions = torch.tensor(
                tuple(union_positions[index] for index in span.token_indices),
                device=probability.device,
                dtype=torch.int64,
            )
            compact_values = (
                probability.index_select(-1, positions)
                .mean(dim=(1, 3))
                .detach()
                .to(dtype=torch.float16)
            )
            pending.extend(
                PendingAttentionMap(
                    span.display_label,
                    values,
                    progress,
                    layer_key,
                    batch_index,
                    spatial_height,
                    spatial_width,
                )
                for batch_index, values in enumerate(compact_values)
            )
        return tuple(pending)

    def capture_open_vocabulary(
        self,
        query: torch.Tensor,
        projected_key: torch.Tensor,
        heads: int,
        label: str,
        progress: float,
        layer_key: str,
        native_key: torch.Tensor,
        denominator_token_budget: int,
        spatial_height: int,
        spatial_width: int,
    ) -> tuple[PendingAttentionMap, ...]:
        """Normalize one layer-specific OVAM key against native projected queries."""

        _unused_query, key_heads = self.head_tensors(
            projected_key,
            projected_key,
            heads,
            False,
        )
        del _unused_query
        if int(key_heads.shape[0]) == 1 and int(query.shape[0]) > 1:
            key_heads = key_heads.expand(int(query.shape[0]), -1, -1, -1)
        combined_key = torch.cat((native_key, key_heads), dim=-2)
        denominator = self.log_denominator(
            query,
            combined_key,
            denominator_token_budget,
        )
        span = AttentionTokenSpan(label, 1, tuple(range(int(key_heads.shape[-2]))))
        return self.capture_spans(
            query,
            key_heads,
            (span,),
            progress,
            layer_key,
            denominator,
            spatial_height,
            spatial_width,
        )

    def materialize(
        self,
        pending: tuple[PendingAttentionMap, ...],
    ) -> tuple[CapturedAttentionMap, ...]:
        """Transfer compatible maps together and calculate concentration on CPU."""

        grouped: dict[
            tuple[torch.device, torch.dtype, tuple[int, ...]],
            list[tuple[int, PendingAttentionMap]],
        ] = defaultdict(list)
        for index, attention_map in enumerate(pending):
            key = (
                attention_map.values.device,
                attention_map.values.dtype,
                tuple(attention_map.values.shape),
            )
            grouped[key].append((index, attention_map))

        ordered: list[CapturedAttentionMap | None] = [None] * len(pending)
        for group in grouped.values():
            cpu_values = torch.stack(tuple(value.values for _index, value in group)).to(
                device="cpu", dtype=torch.float16
            )
            maxima = cpu_values.float().amax(dim=1)
            means = cpu_values.float().mean(dim=1)
            peak_ratios = maxima / means.clamp_min(1e-12)
            confidences = (
                1.0 - torch.exp(-(peak_ratios - 1.0).clamp_min(0.0) / 8.0)
            ).clamp(0.0, 1.0)
            for group_index, (original_index, attention_map) in enumerate(group):
                ordered[original_index] = CapturedAttentionMap(
                    attention_map.label,
                    cpu_values[group_index],
                    attention_map.progress,
                    attention_map.layer_key,
                    attention_map.batch_index,
                    float(confidences[group_index].item()),
                    attention_map.spatial_height,
                    attention_map.spatial_width,
                )
        if any(value is None for value in ordered):
            raise RuntimeError("Attention map materialization lost an observation.")
        return tuple(value for value in ordered if value is not None)


ATTENTION_AFFINITY_CALCULATOR = AttentionAffinityCalculator()
