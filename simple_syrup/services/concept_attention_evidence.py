# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Derive stable concept isolation from captured contextual attention."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import torch

from ..domain.attention_region_capture import AttentionRegionControls
from ..domain.attention_region_evidence import AttentionConceptEvidence
from ..domain.attention_region_maps import CapturedAttentionMap
from .attention_region_model_reliability import ATTENTION_REGION_MODEL_RELIABILITY
from .attention_region_observation_projection import (
    ATTENTION_OBSERVATION_PROJECTION_SERVICE,
)

AUTOMATIC_SUPPORT_FLOOR = 0.1
MINIMUM_DETAIL_OBSERVATIONS = 2


@dataclass(frozen=True, slots=True)
class _LayerEvidence:
    """Hold temporally resolved evidence for one attention layer."""

    confidence: torch.Tensor
    persistent: torch.Tensor
    reliability: torch.Tensor
    detail_eligible: bool


class ConceptAttentionEvidencePolicy:
    """Isolate repeatable above-baseline contextual attention evidence."""

    def aggregate(
        self,
        label: str,
        observations: tuple[CapturedAttentionMap, ...],
        controls: AttentionRegionControls,
        height: int,
        width: int,
    ) -> AttentionConceptEvidence:
        """Suppress uniform and transient activation without morphology."""

        family = observations[0].model_family
        if any(value.model_family is not family for value in observations):
            raise ValueError(
                "Concept attention observations cannot mix model families."
            )
        calibration = ATTENTION_REGION_MODEL_RELIABILITY.for_family(family)
        resized = ATTENTION_OBSERVATION_PROJECTION_SERVICE.stack(
            observations,
            lambda value: (
                value.concept_values
                if value.concept_values is not None
                else value.values
            ),
            height,
            width,
        )
        baselines = torch.tensor(
            [float(value.uniform_probability) for value in observations],
            dtype=resized.dtype,
        ).reshape(-1, 1, 1)
        excess = torch.where(
            baselines > 0.0,
            (resized - baselines).clamp_min(0.0),
            resized.clamp_min(0.0),
        )
        maxima = excess.amax(dim=(1, 2))
        normalized = excess / maxima.reshape(-1, 1, 1).clamp_min(1e-12)
        base_weights = _base_weights(
            observations,
            excess,
            maxima,
            baselines.flatten(),
            calibration.lift_scale,
        )
        observation_threshold = max(
            AUTOMATIC_SUPPORT_FLOOR,
            controls.minimum_strength,
        )
        layers = _aggregate_layers(
            observations,
            normalized,
            base_weights,
            observation_threshold,
            controls.minimum_consensus,
            calibration.agreement_power,
            calibration.consensus_influence,
        )
        confidence, persistent = _fuse_layers(
            layers,
            calibration.agreement_power,
            calibration.detail_influence,
        )
        restored = ATTENTION_OBSERVATION_PROJECTION_SERVICE.restore(
            persistent,
            height,
            width,
        )
        maximum = restored.amax()
        alpha = restored if maximum <= 0.0 else restored / maximum
        support = (alpha > 0.0) & (alpha >= controls.minimum_strength)
        restored_confidence = ATTENTION_OBSERVATION_PROJECTION_SERVICE.restore(
            confidence,
            height,
            width,
        )
        return AttentionConceptEvidence(
            label,
            alpha * support,
            support,
            restored_confidence,
        )


def _aggregate_layers(
    observations: tuple[CapturedAttentionMap, ...],
    normalized: torch.Tensor,
    base_weights: torch.Tensor,
    observation_threshold: float,
    minimum_consensus: float,
    agreement_power: float,
    consensus_influence: float,
) -> tuple[_LayerEvidence, ...]:
    """Resolve temporal persistence independently inside each model layer."""

    grouped: dict[str, list[int]] = defaultdict(list)
    for index, observation in enumerate(observations):
        grouped[observation.layer_key].append(index)
    maximum_area = max(_observation_area(observation) for observation in observations)
    layers: list[_LayerEvidence] = []
    for indices in grouped.values():
        selection = torch.tensor(indices, dtype=torch.int64)
        values = normalized.index_select(0, selection)
        weights = base_weights.index_select(0, selection)
        reference = _weighted_mean(values, weights)
        agreements = _agreements(values, reference).pow(agreement_power)
        weights = weights * (0.1 + 0.9 * agreements)
        strength = _weighted_mean(values, weights)
        consensus = _weighted_mean(
            (values >= observation_threshold).to(dtype=values.dtype),
            weights,
        )
        confidence = strength * (
            (1.0 - consensus_influence) + consensus_influence * consensus
        )
        persistent = torch.where(
            consensus >= minimum_consensus,
            confidence,
            torch.zeros_like(confidence),
        )
        area = max(_observation_area(observations[index]) for index in indices)
        resolution = math.sqrt(float(area) / float(maximum_area))
        reliability = weights.mean() * (0.5 + 0.5 * resolution)
        layers.append(
            _LayerEvidence(
                confidence,
                persistent,
                reliability,
                len(indices) >= MINIMUM_DETAIL_OBSERVATIONS and area == maximum_area,
            )
        )
    return tuple(layers)


def _fuse_layers(
    layers: tuple[_LayerEvidence, ...],
    agreement_power: float,
    detail_influence: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse cross-layer cores while retaining repeated layer-local detail."""

    confidence = torch.stack(tuple(layer.confidence for layer in layers))
    persistent = torch.stack(tuple(layer.persistent for layer in layers))
    reliability = torch.stack(tuple(layer.reliability for layer in layers))
    reference = _weighted_mean(confidence, reliability)
    agreements = _agreements(confidence, reference).pow(agreement_power)
    weights = reliability * (0.1 + 0.9 * agreements)
    core_confidence = _weighted_mean(confidence, weights)
    core_persistent = _weighted_mean(persistent, weights)
    detail_weights = weights * torch.tensor(
        [float(layer.detail_eligible) for layer in layers],
        dtype=weights.dtype,
    )
    detail_confidence = _weighted_lehmer_mean(confidence, detail_weights)
    detail_persistent = _weighted_lehmer_mean(persistent, detail_weights)
    return (
        _blend_detail(core_confidence, detail_confidence, detail_influence),
        _blend_detail(core_persistent, detail_persistent, detail_influence),
    )


def _weighted_lehmer_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Favor strong repeated layer evidence without taking a noisy hard maximum."""

    reshaped = weights.reshape(-1, 1, 1)
    numerator = (values.square() * reshaped).sum(dim=0)
    denominator = (values * reshaped).sum(dim=0)
    return numerator / denominator.clamp_min(1e-12)


def _blend_detail(
    core: torch.Tensor,
    detail: torch.Tensor,
    influence: float,
) -> torch.Tensor:
    """Blend an optional layer-local detail estimate into the stable core."""

    if not detail.count_nonzero().item():
        return core
    return core * (1.0 - influence) + detail * influence


def _observation_area(observation: CapturedAttentionMap) -> int:
    """Return one observation's native spatial area."""

    if observation.spatial_height is None or observation.spatial_width is None:
        return int(observation.values.numel())
    return observation.spatial_height * observation.spatial_width


def _base_weights(
    observations: tuple[CapturedAttentionMap, ...],
    excess: torch.Tensor,
    maxima: torch.Tensor,
    baselines: torch.Tensor,
    lift_scale: float,
) -> torch.Tensor:
    """Weight spatial specificity, signal lift, and captured concentration."""

    means = excess.mean(dim=(1, 2))
    peak_ratios = maxima / means.clamp_min(1e-12)
    specificity = 1.0 - torch.exp(-(peak_ratios - 1.0).clamp_min(0.0) / 6.0)
    relative_lift = torch.where(
        baselines > 0.0,
        maxima / baselines.clamp_min(1e-12),
        torch.ones_like(maxima),
    )
    lift = 1.0 - torch.exp(-relative_lift * lift_scale)
    captured = torch.tensor(
        [value.confidence for value in observations],
        dtype=excess.dtype,
    )
    weights = (0.1 + 0.9 * specificity) * lift * (0.25 + 0.75 * captured)
    return torch.where(maxima > 0.0, weights, torch.zeros_like(weights))


def _agreements(values: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Measure each observation's spatial agreement with the shared reference."""

    flat = values.flatten(start_dim=1)
    reference_flat = reference.flatten()
    numerator = (flat * reference_flat).sum(dim=1)
    denominator = flat.square().sum(dim=1).sqrt() * math.sqrt(
        float(reference_flat.square().sum().item())
    )
    return (numerator / denominator.clamp_min(1e-12)).clamp(0.0, 1.0)


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Return a finite weighted spatial mean with zero-evidence handling."""

    reshaped = weights.reshape(-1, 1, 1)
    return (values * reshaped).sum(dim=0) / reshaped.sum().clamp_min(1e-12)


CONCEPT_ATTENTION_EVIDENCE_POLICY = ConceptAttentionEvidencePolicy()
