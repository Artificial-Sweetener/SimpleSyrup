# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Select related prompt spans from projected model-conditioning keys."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from ..domain.attention_region_maps import AttentionTokenSpan

MAXIMUM_RELATED_SPANS = 8
MINIMUM_RELATED_SIMILARITY = 0.25
RELATED_SIMILARITY_MARGIN = 0.05
RELATED_BEST_MARGIN = 0.12
MAXIMUM_RELATED_WEIGHT = 0.30
MODIFIER_TOKEN_WEIGHT = 0.45


@dataclass(frozen=True, slots=True)
class ContextualSpanSelection:
    """Bind exact and reduced-weight related conditioning token positions."""

    token_indices: tuple[int, ...]
    token_weights: tuple[float, ...]

    def __post_init__(self) -> None:
        """Require aligned unique token positions and normalized weights."""

        if (
            not self.token_indices
            or self.token_indices != tuple(sorted(set(self.token_indices)))
            or len(self.token_indices) != len(self.token_weights)
        ):
            raise ValueError("Contextual span selection must be unique and aligned.")
        if any(not 0.0 < value <= 1.0 for value in self.token_weights):
            raise ValueError("Contextual span weights must be within (0, 1].")


class AttentionContextualSpanSelector:
    """Enrich exact spans through model-native projected-key similarity."""

    def select(
        self,
        *,
        key: torch.Tensor,
        targets: tuple[AttentionTokenSpan, ...],
        candidates: tuple[AttentionTokenSpan, ...],
    ) -> dict[AttentionTokenSpan, ContextualSpanSelection]:
        """Return exact targets plus a bounded set of related prompt positions."""

        if key.ndim != 4:
            raise ValueError("Contextual span selection requires BHSD keys.")
        if not targets:
            return {}
        context_length = int(key.shape[-2])
        if any(
            index >= context_length
            for span in (*targets, *candidates)
            for index in span.token_indices
        ):
            raise ValueError("Contextual span selection exceeds the key sequence.")
        candidate_vectors = tuple(_head_vector(key, span) for span in candidates)
        selections: dict[AttentionTokenSpan, ContextualSpanSelection] = {}
        for target in targets:
            related = _related_candidates(
                target,
                _head_vector(key, target),
                candidates,
                candidate_vectors,
            )
            token_weights = dict.fromkeys(target.token_indices, MODIFIER_TOKEN_WEIGHT)
            token_weights.update(dict.fromkeys(target.semantic_head_indices, 1.0))
            for candidate, score in related:
                weight = min(MAXIMUM_RELATED_WEIGHT, max(0.05, score * 0.35))
                for index in candidate.semantic_head_indices:
                    if index not in token_weights:
                        token_weights[index] = weight
            indices = tuple(sorted(token_weights))
            selections[target] = ContextualSpanSelection(
                indices,
                tuple(token_weights[index] for index in indices),
            )
        return selections


def _related_candidates(
    target: AttentionTokenSpan,
    target_vector: torch.Tensor,
    candidates: tuple[AttentionTokenSpan, ...],
    candidate_vectors: tuple[torch.Tensor, ...],
) -> tuple[tuple[AttentionTokenSpan, float], ...]:
    """Return model-related spans with at least one additional token position."""

    target_indices = set(target.token_indices)
    scored = tuple(
        (candidate, _similarity(target_vector, vector))
        for candidate, vector in zip(candidates, candidate_vectors, strict=True)
        if set(candidate.token_indices) - target_indices
    )
    if not scored:
        return ()
    scores = torch.tensor([score for _candidate, score in scored])
    threshold = max(
        MINIMUM_RELATED_SIMILARITY,
        float(scores.median().item()) + RELATED_SIMILARITY_MARGIN,
        max(score for _candidate, score in scored) - RELATED_BEST_MARGIN,
    )
    admitted = tuple(
        (candidate, score)
        for candidate, score in sorted(scored, key=lambda value: -value[1])
        if score >= threshold
    )
    return admitted[:MAXIMUM_RELATED_SPANS]


def _head_vector(key: torch.Tensor, span: AttentionTokenSpan) -> torch.Tensor:
    """Return one normalized projected-key vector for a span's semantic head."""

    indices = torch.tensor(
        span.semantic_head_indices,
        device=key.device,
        dtype=torch.int64,
    )
    vector = key.float().index_select(-2, indices).mean(dim=-2)
    return functional.normalize(vector, dim=-1, eps=1e-12)


def _similarity(left: torch.Tensor, right: torch.Tensor) -> float:
    """Return mean projected-key cosine similarity across batch and heads."""

    return float((left * right).sum(dim=-1).mean().clamp(-1.0, 1.0).item())


ATTENTION_CONTEXTUAL_SPAN_SELECTOR = AttentionContextualSpanSelector()
