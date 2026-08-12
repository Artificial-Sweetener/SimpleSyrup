# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Own immutable regional model-side LoRA composition plans."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum


class RegionalLoraBranch(StrEnum):
    """Identify the conditioning branch that owns one regional adapter use."""

    POSITIVE = "positive"
    NEGATIVE = "negative"


@dataclass(frozen=True)
class RegionalLoraAdapterIdentity:
    """Retain the caller-supplied stable identity of one LoRA artifact."""

    value: str

    def __post_init__(self) -> None:
        """Reject identities that cannot distinguish an adapter."""

        if not isinstance(self.value, str) or not self.value.strip():
            raise ValueError(
                "Regional LoRA adapter identity must be a non-empty string."
            )


@dataclass(frozen=True)
class RegionalLoraScheduleBoundary:
    """Retain one ordered Comfy HookKeyframe boundary exactly."""

    start_percent: float
    start_sigma: float
    strength_multiplier: float
    guarantee_steps: int

    def __post_init__(self) -> None:
        """Validate one finite normalized schedule boundary."""

        _require_finite_float(self.start_percent, name="start_percent")
        if not 0.0 <= self.start_percent <= 1.0:
            raise ValueError("Regional LoRA schedule start_percent must be in [0, 1].")
        _require_finite_float(self.start_sigma, name="start_sigma")
        if self.start_sigma < 0.0:
            raise ValueError("Regional LoRA schedule start_sigma must be non-negative.")
        _require_finite_float(
            self.strength_multiplier,
            name="strength_multiplier",
        )
        if isinstance(self.guarantee_steps, bool) or not isinstance(
            self.guarantee_steps, int
        ):
            raise TypeError("Regional LoRA guarantee_steps must be an integer.")
        if self.guarantee_steps < 0:
            raise ValueError("Regional LoRA guarantee_steps must be non-negative.")


@dataclass(frozen=True)
class RegionalLoraAdapterPlan:
    """Describe one ordered regional use of a model-side LoRA adapter."""

    adapter_identity: RegionalLoraAdapterIdentity
    composition_index: int
    region_index: int
    branch: RegionalLoraBranch
    model_strength: float
    schedule: tuple[RegionalLoraScheduleBoundary, ...]

    def __post_init__(self) -> None:
        """Validate adapter ownership, strength, and ordered schedule."""

        if not isinstance(self.adapter_identity, RegionalLoraAdapterIdentity):
            raise TypeError("Regional LoRA adapter_identity has an invalid type.")
        _require_non_negative_index(self.composition_index, name="composition_index")
        _require_non_negative_index(self.region_index, name="region_index")
        if not isinstance(self.branch, RegionalLoraBranch):
            raise TypeError("Regional LoRA branch has an invalid type.")
        _require_finite_float(self.model_strength, name="model_strength")
        if not isinstance(self.schedule, tuple) or not self.schedule:
            raise ValueError(
                "Regional LoRA schedule must contain at least one boundary."
            )
        if any(
            not isinstance(boundary, RegionalLoraScheduleBoundary)
            for boundary in self.schedule
        ):
            raise TypeError("Regional LoRA schedule contains an invalid boundary.")
        starts = tuple(boundary.start_percent for boundary in self.schedule)
        if starts != tuple(sorted(starts)):
            raise ValueError("Regional LoRA schedule boundaries must be ordered.")
        sigmas = tuple(boundary.start_sigma for boundary in self.schedule)
        if sigmas != tuple(sorted(sigmas, reverse=True)):
            raise ValueError(
                "Regional LoRA converted schedule boundaries must be descending."
            )

    @property
    def is_time_invariant(self) -> bool:
        """Report whether every keyframe retains one effective multiplier."""

        first_multiplier = self.schedule[0].strength_multiplier
        return all(
            boundary.strength_multiplier == first_multiplier
            for boundary in self.schedule[1:]
        )


@dataclass(frozen=True)
class RegionalLoraPlan:
    """Store all adapter uses in authoritative global composition order."""

    adapters: tuple[RegionalLoraAdapterPlan, ...]

    def __post_init__(self) -> None:
        """Require immutable entries with contiguous composition indices."""

        if not isinstance(self.adapters, tuple):
            raise TypeError("Regional LoRA plan adapters must be a tuple.")
        if any(
            not isinstance(adapter, RegionalLoraAdapterPlan)
            for adapter in self.adapters
        ):
            raise TypeError("Regional LoRA plan contains an invalid adapter entry.")
        observed_indices = tuple(adapter.composition_index for adapter in self.adapters)
        if observed_indices != tuple(range(len(self.adapters))):
            raise ValueError(
                "Regional LoRA composition indices must be contiguous and ordered."
            )

    @property
    def is_time_invariant(self) -> bool:
        """Report whether every regional adapter retains one effective strength."""

        return all(adapter.is_time_invariant for adapter in self.adapters)


def _require_finite_float(value: object, *, name: str) -> None:
    """Require one exact finite floating-point value."""

    if not isinstance(value, float) or not math.isfinite(value):
        raise TypeError(f"Regional LoRA {name} must be a finite float.")


def _require_non_negative_index(value: object, *, name: str) -> None:
    """Require one non-negative integer index."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Regional LoRA {name} must be an integer.")
    if value < 0:
        raise ValueError(f"Regional LoRA {name} must be non-negative.")


EMPTY_REGIONAL_LORA_PLAN = RegionalLoraPlan(adapters=())
