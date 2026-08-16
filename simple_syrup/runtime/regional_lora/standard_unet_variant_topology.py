# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Collapse paired CFG adapter evidence into branch-neutral UNet variants."""

from __future__ import annotations

from dataclasses import dataclass

from ...domain.regional_lora_plan import (
    RegionalLoraAdapterPlan,
    RegionalLoraBranch,
    RegionalLoraScheduleBoundary,
)
from .comfy_adapter_resolution import (
    ComfyNormalizedAdapterTarget,
    ComfyRegionalAdapterResolution,
    ComfyRegionalLoraResolution,
)


@dataclass(frozen=True, slots=True)
class StandardUnetVariantAdapter:
    """Retain one ordered branch-neutral adapter and its host operations."""

    composition_index: int
    region_index: int
    model_strength: float
    schedule: tuple[RegionalLoraScheduleBoundary, ...]
    targets: tuple[ComfyNormalizedAdapterTarget, ...]


@dataclass(frozen=True, slots=True)
class StandardUnetRegionalVariant:
    """Retain every ordered adapter that owns one authored region."""

    region_index: int
    adapters: tuple[StandardUnetVariantAdapter, ...]

    def __post_init__(self) -> None:
        """Require one nonempty canonical region-local adapter sequence."""

        if self.region_index < 0 or not self.adapters:
            raise ValueError("Standard UNet regional variant must be nonempty.")
        if any(adapter.region_index != self.region_index for adapter in self.adapters):
            raise ValueError("Standard UNet variant adapters must share one region.")
        indices = tuple(adapter.composition_index for adapter in self.adapters)
        if indices != tuple(sorted(indices)):
            raise ValueError("Standard UNet variant adapter order must be canonical.")


@dataclass(frozen=True, slots=True)
class StandardUnetVariantTopology:
    """Retain canonical branch-neutral variants in authored region order."""

    variants: tuple[StandardUnetRegionalVariant, ...]

    def __post_init__(self) -> None:
        """Require unique ascending region ownership."""

        regions = tuple(variant.region_index for variant in self.variants)
        if regions != tuple(sorted(set(regions))):
            raise ValueError("Standard UNet variant regions must be unique and sorted.")


class StandardUnetVariantTopologyBuilder:
    """Validate CFG parity and group one exact variant per regional mask."""

    def build(
        self,
        resolution: ComfyRegionalLoraResolution,
    ) -> StandardUnetVariantTopology:
        """Return branch-neutral variants or fail on any CFG asymmetry."""

        if not isinstance(resolution, ComfyRegionalLoraResolution):
            raise TypeError("Standard UNet variant topology requires resolution.")
        positive = tuple(
            adapter
            for adapter in resolution.adapters
            if adapter.adapter.branch is RegionalLoraBranch.POSITIVE
        )
        negative = tuple(
            adapter
            for adapter in resolution.adapters
            if adapter.adapter.branch is RegionalLoraBranch.NEGATIVE
        )
        if len(positive) != len(negative):
            raise ValueError(
                "Standard UNet regional LoRAs require paired positive and negative "
                "adapter uses."
            )
        grouped: dict[int, list[StandardUnetVariantAdapter]] = {}
        for positive_use, negative_use in zip(positive, negative, strict=True):
            self._require_pair(positive_use, negative_use)
            plan = positive_use.adapter
            grouped.setdefault(plan.region_index, []).append(
                StandardUnetVariantAdapter(
                    plan.composition_index,
                    plan.region_index,
                    plan.model_strength,
                    plan.schedule,
                    positive_use.model_targets,
                )
            )
        return StandardUnetVariantTopology(
            tuple(
                StandardUnetRegionalVariant(region_index, tuple(adapters))
                for region_index, adapters in sorted(grouped.items())
            )
        )

    @classmethod
    def _require_pair(
        cls,
        positive: ComfyRegionalAdapterResolution,
        negative: ComfyRegionalAdapterResolution,
    ) -> None:
        """Require equivalent prompt-branch ownership and model payloads."""

        expected = cls._plan_signature(positive.adapter)
        observed = cls._plan_signature(negative.adapter)
        if expected != observed:
            raise ValueError(
                "Standard UNet regional LoRA CFG pairs must share region, "
                "identity, strength, and schedule."
            )
        if (
            positive.payload.needs_resolution != negative.payload.needs_resolution
            or positive.payload.model_side_weights
            is not negative.payload.model_side_weights
        ):
            raise ValueError(
                "Standard UNet regional LoRA CFG pairs must share one exact "
                "model payload."
            )
        if cls._target_signature(positive) != cls._target_signature(negative):
            raise ValueError(
                "Standard UNet regional LoRA CFG pairs resolved different targets."
            )

    @staticmethod
    def _plan_signature(
        plan: RegionalLoraAdapterPlan,
    ) -> tuple[object, ...]:
        """Describe branch-neutral authored semantics."""

        return (
            plan.adapter_identity,
            plan.region_index,
            plan.model_strength,
            plan.schedule,
        )

    @staticmethod
    def _target_signature(
        result: ComfyRegionalAdapterResolution,
    ) -> tuple[tuple[object, ...], ...]:
        """Describe host-resolved target scope without tensor comparison."""

        return tuple(
            (
                target.path,
                target.operation_type,
                target.source_keys,
                target.ordinary_additive_lora,
            )
            for target in result.model_targets
        )


STANDARD_UNET_VARIANT_TOPOLOGY_BUILDER = StandardUnetVariantTopologyBuilder()
