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
from ..domain.regional_model_capabilities import RegionalModelFamily
from .attention_region_contextual_spans import (
    ATTENTION_CONTEXTUAL_SPAN_SELECTOR,
    MAXIMUM_RELATED_WEIGHT,
)
from .attention_region_logits import scaled_attention_logits
from .attention_region_phrase_evidence import ANIMA_PHRASE_EVIDENCE_SERVICE
from .attention_region_self_completion import (
    ATTENTION_REGION_SELF_COMPLETION,
    SpatialSelfAttention,
)


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
    concept_values: torch.Tensor
    uniform_probability: float


@dataclass(frozen=True, slots=True)
class _ConceptSeeds:
    """Separate exact concept evidence from contextual recall candidates."""

    exact: torch.Tensor
    contextual: torch.Tensor | None


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

        return (
            self.head_tensor(query, heads, skip_reshape),
            self.head_tensor(key, heads, skip_reshape),
        )

    def head_tensor(
        self,
        value: torch.Tensor,
        heads: int,
        skip_reshape: bool,
    ) -> torch.Tensor:
        """Return one attention tensor in batch-head-sequence-channel layout."""

        if type(heads) is not int or heads < 1:
            raise ValueError("Attention capture head count must be positive.")
        if skip_reshape and value.ndim == 4:
            return value
        if value.ndim != 3:
            raise ValueError("Attention capture received an unsupported tensor rank.")
        if int(value.shape[-1]) % heads:
            raise ValueError("Attention channels must divide evenly across heads.")
        return value.view(value.shape[0], value.shape[1], heads, -1).permute(0, 2, 1, 3)

    def positive_rows(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        raw_branches: object,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select positive CFG rows while retaining ordinary batch members."""

        if not isinstance(raw_branches, list) or not raw_branches:
            return query, key, value
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
            return query[:0], key[:0], value[:0]
        indices = torch.tensor(row_indices, device=query.device, dtype=torch.int64)
        return (
            query.index_select(0, indices),
            key.index_select(0, indices),
            value.index_select(0, indices),
        )

    def log_denominator(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        token_budget: int,
    ) -> torch.Tensor:
        """Estimate softmax normalization from a deterministic bounded key sample."""

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
        for key_chunk in torch.split(key, 32, dim=-2):
            logits = scaled_attention_logits(query, key_chunk)
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
        value: torch.Tensor,
        spans: tuple[AttentionTokenSpan, ...],
        progress: float,
        layer_key: str,
        denominator: torch.Tensor,
        spatial_height: int,
        spatial_width: int,
        contextual_targets: tuple[AttentionTokenSpan, ...] = (),
        context_candidates: tuple[AttentionTokenSpan, ...] = (),
        self_attention: SpatialSelfAttention | None = None,
        derive_concept_values: bool = True,
    ) -> tuple[PendingAttentionMap, ...]:
        """Compute all native span maps from one unioned selected-key projection."""

        contextual = ATTENTION_CONTEXTUAL_SPAN_SELECTOR.select(
            key=key,
            targets=contextual_targets,
            candidates=context_candidates,
        )
        union_indices = tuple(
            sorted(
                {
                    index
                    for span in spans
                    for index in (
                        contextual[span].token_indices
                        if span in contextual
                        else span.token_indices
                    )
                }
            )
        )
        if not union_indices:
            return ()
        indices = torch.tensor(union_indices, device=key.device, dtype=torch.int64)
        selected = key.index_select(-2, indices)
        selected_value = value.index_select(-2, indices)
        logits = scaled_attention_logits(query, selected)
        log_probability = (logits - denominator.unsqueeze(-1)).clamp_max(0.0)
        probability = torch.exp(log_probability)
        union_positions = {
            token_index: index for index, token_index in enumerate(union_indices)
        }
        pending: list[PendingAttentionMap] = []
        for span in spans:
            selection = contextual.get(span)
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
            concept_values = ANIMA_PHRASE_EVIDENCE_SERVICE.derive(
                probability=probability,
                span=span,
                union_positions=union_positions,
                contextual_token_indices=(
                    selection.token_indices if selection is not None else ()
                ),
                contextual_token_weights=(
                    selection.token_weights if selection is not None else ()
                ),
            )
            if derive_concept_values:
                concept_indices = (
                    selection.token_indices
                    if selection is not None
                    else span.token_indices
                )
                concept_positions = torch.tensor(
                    tuple(union_positions[index] for index in concept_indices),
                    device=probability.device,
                    dtype=torch.int64,
                )
                token_priors = (
                    torch.tensor(
                        selection.token_weights,
                        device=probability.device,
                        dtype=probability.dtype,
                    )
                    if selection is not None
                    else None
                )
                concept_seeds = _concept_values(
                    probability.index_select(-1, concept_positions),
                    selected_value.index_select(-2, concept_positions),
                    value,
                    token_priors,
                )
                concept_values = ATTENTION_REGION_SELF_COMPLETION.complete(
                    concept_seeds.exact,
                    self_attention,
                    spatial_height,
                    spatial_width,
                    related_seed=concept_seeds.contextual,
                )
            concept_values = concept_values.detach().to(dtype=torch.float16)
            pending.extend(
                PendingAttentionMap(
                    span.display_label,
                    values,
                    progress,
                    layer_key,
                    batch_index,
                    spatial_height,
                    spatial_width,
                    concept_values[batch_index],
                    1.0 / float(key.shape[-2]),
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
        native_value: torch.Tensor,
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
        captured = self.capture_spans(
            query,
            key_heads,
            key_heads,
            (span,),
            progress,
            layer_key,
            denominator,
            spatial_height,
            spatial_width,
        )
        combined_token_count = int(native_key.shape[-2]) + int(key_heads.shape[-2])
        del native_value
        return tuple(
            PendingAttentionMap(
                label=value.label,
                values=value.values,
                progress=value.progress,
                layer_key=value.layer_key,
                batch_index=value.batch_index,
                spatial_height=value.spatial_height,
                spatial_width=value.spatial_width,
                concept_values=value.values,
                uniform_probability=1.0 / float(combined_token_count),
            )
            for value in captured
        )

    def materialize(
        self,
        pending: tuple[PendingAttentionMap, ...],
        model_family: RegionalModelFamily,
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
            cpu_concept_values = torch.stack(
                tuple(value.concept_values for _index, value in group)
            ).to(device="cpu", dtype=torch.float16)
            maxima = cpu_values.float().amax(dim=1)
            means = cpu_values.float().mean(dim=1)
            peak_ratios = maxima / means.clamp_min(1e-12)
            confidences = (
                1.0 - torch.exp(-(peak_ratios - 1.0).clamp_min(0.0) / 8.0)
            ).clamp(0.0, 1.0)
            for group_index, (original_index, attention_map) in enumerate(group):
                ordered[original_index] = CapturedAttentionMap(
                    label=attention_map.label,
                    values=cpu_values[group_index],
                    progress=attention_map.progress,
                    layer_key=attention_map.layer_key,
                    batch_index=attention_map.batch_index,
                    confidence=float(confidences[group_index].item()),
                    spatial_height=attention_map.spatial_height,
                    spatial_width=attention_map.spatial_width,
                    concept_values=cpu_concept_values[group_index],
                    uniform_probability=attention_map.uniform_probability,
                    model_family=model_family,
                )
        if any(value is None for value in ordered):
            raise RuntimeError("Attention map materialization lost an observation.")
        return tuple(value for value in ordered if value is not None)


def _concept_values(
    probability: torch.Tensor,
    selected_value: torch.Tensor,
    full_value: torch.Tensor,
    token_priors: torch.Tensor | None = None,
) -> _ConceptSeeds:
    """Return exact concept evidence and a separately gated contextual candidate."""

    selected_energy = selected_value.float().square().mean(dim=-1).sqrt()
    context_energy = (
        full_value.float().square().mean(dim=-1).sqrt().mean(dim=-1, keepdim=True)
    )
    contribution = (selected_energy / context_energy.clamp_min(1e-12)).clamp(0.25, 4.0)
    token_maps, raw_token_maps = _select_relevant_head_maps(
        probability,
        contribution,
        exact_token_mask=(
            token_priors > MAXIMUM_RELATED_WEIGHT if token_priors is not None else None
        ),
    )
    if token_priors is None:
        return _ConceptSeeds(
            _fuse_concept_tokens(token_maps, raw_token_maps, None),
            None,
        )
    exact_mask = token_priors > MAXIMUM_RELATED_WEIGHT
    exact = _fuse_concept_tokens(
        token_maps[..., exact_mask],
        raw_token_maps[..., exact_mask],
        token_priors[exact_mask],
    )
    if exact_mask.all().item():
        return _ConceptSeeds(exact, None)
    contextual = _fuse_concept_tokens(token_maps, raw_token_maps, token_priors)
    return _ConceptSeeds(exact, contextual)


def _fuse_concept_tokens(
    token_maps: torch.Tensor,
    raw_token_maps: torch.Tensor,
    token_priors: torch.Tensor | None,
) -> torch.Tensor:
    """Fuse one selected set of token maps by specificity and agreement."""

    peak_ratio = raw_token_maps.amax(dim=1) / raw_token_maps.mean(dim=1).clamp_min(
        1e-12
    )
    specificity = (1.0 - torch.exp(-(peak_ratio - 1.0).clamp_min(0.0) / 4.0)).clamp_min(
        0.05
    )
    if token_priors is not None:
        specificity = specificity * token_priors.reshape(1, -1)
    token_weights = specificity / specificity.sum(dim=-1, keepdim=True)
    fused = (token_maps * token_weights.unsqueeze(1)).sum(dim=-1)
    normalized_tokens = raw_token_maps / raw_token_maps.amax(
        dim=1, keepdim=True
    ).clamp_min(1e-12)
    agreement = (normalized_tokens * token_weights.unsqueeze(1)).sum(dim=-1)
    return fused * (0.5 + 0.5 * agreement)


def _select_relevant_head_maps(
    probability: torch.Tensor,
    contribution: torch.Tensor,
    exact_token_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate concept heads and admit related tokens through those same heads."""

    head_count = int(probability.shape[1])
    if head_count == 1:
        return (
            (probability * contribution.unsqueeze(-2))[:, 0],
            probability[:, 0],
        )
    reference = probability.mean(dim=1)
    numerator = (probability * reference.unsqueeze(1)).sum(dim=2)
    denominator = probability.square().sum(dim=2).sqrt() * reference.square().sum(
        dim=1
    ).sqrt().unsqueeze(1)
    agreement = (numerator / denominator.clamp_min(1e-12)).clamp(0.0, 1.0)
    head_means = probability.mean(dim=2)
    peak_ratios = probability.amax(dim=2) / head_means.clamp_min(1e-12)
    specificity = 1.0 - torch.exp(-(peak_ratios - 1.0).clamp_min(0.0) / 4.0)
    relative_response = head_means / head_means.mean(dim=1, keepdim=True).clamp_min(
        1e-12
    )
    scores = (
        agreement.square()
        * (0.25 + 0.75 * specificity)
        * (0.5 + 0.5 * relative_response.clamp(0.0, 2.0))
    )
    if exact_token_mask is not None:
        if (
            exact_token_mask.ndim != 1
            or int(exact_token_mask.shape[0]) != int(scores.shape[-1])
            or not exact_token_mask.any().item()
        ):
            raise ValueError("Exact concept-token mask must select aligned positions.")
        exact = exact_token_mask.to(device=scores.device, dtype=scores.dtype)
        exact_scores = (scores * exact.reshape(1, 1, -1)).sum(dim=-1)
        exact_scores = exact_scores / exact.sum().clamp_min(1.0)
        scores = torch.where(
            exact_token_mask.to(device=scores.device).reshape(1, 1, -1),
            scores,
            exact_scores.unsqueeze(-1),
        )
    selected_count = max(1, math.ceil(head_count * 0.25))
    selected_indices = scores.topk(selected_count, dim=1).indices
    selection = torch.zeros_like(scores).scatter(1, selected_indices, 1.0)
    weights = scores * selection
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
    weighted_probability = probability * weights.unsqueeze(2)
    return (
        (weighted_probability * contribution.unsqueeze(-2)).sum(dim=1),
        weighted_probability.sum(dim=1),
    )


ATTENTION_AFFINITY_CALCULATOR = AttentionAffinityCalculator()
