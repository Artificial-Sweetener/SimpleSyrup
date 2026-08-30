# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Derive phrase-local Anima evidence from exact conditioning-token maps."""

from __future__ import annotations

import math

import torch

from ..domain.attention_region_maps import AttentionTokenSpan

MAXIMUM_MODIFIER_INFLUENCE = 0.85
MINIMUM_MODIFIER_SCORE = 0.05
FULL_MODIFIER_SCORE = 0.4


class AnimaPhraseEvidenceService:
    """Anchor compound phrases to their noun head and specific modifiers."""

    def derive(
        self,
        *,
        probability: torch.Tensor,
        span: AttentionTokenSpan,
        union_positions: dict[int, int],
        contextual_token_indices: tuple[int, ...] = (),
        contextual_token_weights: tuple[float, ...] = (),
    ) -> torch.Tensor:
        """Return noun geometry constrained only by useful phrase agreement."""

        positions = tuple(union_positions[index] for index in span.token_indices)
        indices = torch.tensor(
            positions,
            device=probability.device,
            dtype=torch.int64,
        )
        tokens = probability.index_select(-1, indices).float()
        semantic_heads = set(span.semantic_head_indices)
        head_mask = torch.tensor(
            tuple(index in semantic_heads for index in span.token_indices),
            device=tokens.device,
            dtype=torch.bool,
        )
        head_by_attention = tokens[..., head_mask].mean(dim=-1)
        head_weights = specific_attention_head_weights(head_by_attention)
        head = (head_by_attention * head_weights.unsqueeze(-1)).sum(dim=1)
        exact = head
        if not head_mask.all().item():
            modifier_by_attention = tokens[..., ~head_mask]
            modifier_maps = (
                modifier_by_attention * head_weights.unsqueeze(-1).unsqueeze(-1)
            ).sum(dim=1)
            confirmation, influence = _modifier_confirmation(head, modifier_maps)
            modulation = (1.0 - influence) + (confirmation * influence)
            exact = head * modulation
        contextual = _contextual_recall(
            probability=probability,
            span=span,
            union_positions=union_positions,
            token_indices=contextual_token_indices,
            token_weights=contextual_token_weights,
            head_weights=head_weights,
        )
        if contextual is not None:
            exact = torch.maximum(exact, contextual)
        return exact.detach().to(dtype=torch.float16)


def specific_attention_head_weights(head_values: torch.Tensor) -> torch.Tensor:
    """Select Anima heads whose semantic-object response is spatially specific."""

    head_count = int(head_values.shape[1])
    if head_count == 1:
        return torch.ones_like(head_values[..., 0])
    means = head_values.mean(dim=-1)
    peak_ratios = head_values.amax(dim=-1) / means.clamp_min(1e-12)
    specificity = 1.0 - torch.exp(-(peak_ratios - 1.0).clamp_min(0.0) / 4.0)
    relative_response = means / means.mean(dim=1, keepdim=True).clamp_min(1e-12)
    scores = specificity * (0.5 + 0.5 * relative_response.clamp(0.0, 2.0))
    selected_count = max(1, math.ceil(head_count * 0.25))
    selected = scores.topk(selected_count, dim=1).indices
    selection = torch.zeros_like(scores).scatter(1, selected, 1.0)
    weights = scores * selection
    return weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)


def _modifier_confirmation(
    head: torch.Tensor,
    modifier_maps: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return spatial modifier agreement and its evidence-derived influence."""

    modifier_maxima = modifier_maps.amax(dim=1, keepdim=True)
    normalized = modifier_maps / modifier_maxima.clamp_min(1e-12)
    modifier_means = modifier_maps.mean(dim=1)
    peak_ratios = modifier_maxima.squeeze(1) / modifier_means.clamp_min(1e-12)
    specificity = 1.0 - torch.exp(-(peak_ratios - 1.0).clamp_min(0.0) / 2.0)

    normalized_head = head / head.amax(dim=1, keepdim=True).clamp_min(1e-12)
    numerator = (normalized * normalized_head.unsqueeze(-1)).sum(dim=1)
    denominator = (
        normalized.square().sum(dim=1).sqrt()
        * normalized_head.square()
        .sum(
            dim=1,
            keepdim=True,
        )
        .sqrt()
    )
    overlap = numerator / denominator.clamp_min(1e-12)
    scores = specificity * overlap.clamp(0.0, 1.0)
    weights = scores / scores.sum(dim=1, keepdim=True).clamp_min(1e-12)
    confirmation = (normalized * weights.unsqueeze(1)).sum(dim=-1)
    strongest = scores.amax(dim=1, keepdim=True)
    influence = (
        (strongest - MINIMUM_MODIFIER_SCORE)
        / (FULL_MODIFIER_SCORE - MINIMUM_MODIFIER_SCORE)
    ).clamp(0.0, MAXIMUM_MODIFIER_INFLUENCE)
    return confirmation, influence


def _contextual_recall(
    *,
    probability: torch.Tensor,
    span: AttentionTokenSpan,
    union_positions: dict[int, int],
    token_indices: tuple[int, ...],
    token_weights: tuple[float, ...],
    head_weights: torch.Tensor,
) -> torch.Tensor | None:
    """Return reduced-weight evidence from related prompt-span semantic heads."""

    exact_indices = set(span.token_indices)
    related = tuple(
        (index, weight)
        for index, weight in zip(token_indices, token_weights, strict=True)
        if index not in exact_indices
    )
    if not related:
        return None
    positions = torch.tensor(
        tuple(union_positions[index] for index, _weight in related),
        device=probability.device,
        dtype=torch.int64,
    )
    weights = torch.tensor(
        tuple(weight for _index, weight in related),
        device=probability.device,
        dtype=probability.dtype,
    )
    related_maps = probability.index_select(-1, positions).float()
    per_head = (related_maps * weights.reshape(1, 1, 1, -1)).sum(dim=-1)
    return (per_head * head_weights.unsqueeze(-1)).sum(dim=1)


ANIMA_PHRASE_EVIDENCE_SERVICE = AnimaPhraseEvidenceService()
