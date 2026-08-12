# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Resolve immutable regional LoRA schedules with installed Comfy semantics."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..domain.regional_lora_plan import (
    RegionalLoraAdapterPlan,
    RegionalLoraScheduleBoundary,
)


@dataclass(frozen=True, slots=True)
class RegionalLoraScheduleResolution:
    """Retain ordered schedule multipliers and effective model strengths."""

    schedule_multipliers: tuple[float, ...]
    effective_strengths: tuple[float, ...]

    def __post_init__(self) -> None:
        """Validate exact finite aligned composition values."""

        if len(self.schedule_multipliers) != len(self.effective_strengths):
            raise ValueError("Regional LoRA schedule resolution axes must align.")
        if any(
            not math.isfinite(value)
            for value in (*self.schedule_multipliers, *self.effective_strengths)
        ):
            raise ValueError("Regional LoRA schedule resolution must be finite.")

    def effective_strength(self, composition_index: int) -> float:
        """Return one exact ordered adapter strength."""

        if isinstance(composition_index, bool) or not isinstance(
            composition_index, int
        ):
            raise TypeError("Regional LoRA composition index must be an integer.")
        if not 0 <= composition_index < len(self.effective_strengths):
            raise IndexError("Regional LoRA composition index is out of range.")
        return self.effective_strengths[composition_index]

    def scaled(self, multiplier: float) -> RegionalLoraScheduleResolution:
        """Return effective strengths scaled by one validated phase multiplier."""

        if (
            isinstance(multiplier, bool)
            or not isinstance(multiplier, int | float)
            or not math.isfinite(float(multiplier))
            or not 0.0 <= float(multiplier) <= 1.0
        ):
            raise ValueError("Regional LoRA phase multiplier must be in [0, 1].")
        scale = float(multiplier)
        return RegionalLoraScheduleResolution(
            self.schedule_multipliers,
            tuple(strength * scale for strength in self.effective_strengths),
        )

    @property
    def has_active_strength(self) -> bool:
        """Return whether any adapter can contribute to the current model call."""

        return any(strength != 0.0 for strength in self.effective_strengths)


@dataclass(slots=True)
class _AdapterScheduleCursor:
    """Track one adapter's installed-Comfy-equivalent keyframe position."""

    plan: RegionalLoraAdapterPlan
    maximum_sigma: float
    current_index: int = 0
    current_used_steps: int = 0
    current_sigma: float = -1.0

    def resolve(self, sigma: float) -> float:
        """Advance once per distinct sigma and return the current multiplier."""

        if sigma == self.current_sigma:
            return self.current.strength_multiplier
        if self.current_used_steps >= self._effective_guarantee(self.current):
            for index in range(self.current_index + 1, len(self.plan.schedule)):
                candidate = self.plan.schedule[index]
                if candidate.start_sigma < sigma:
                    break
                self.current_index = index
                self.current_used_steps = 0
                if self._effective_guarantee(candidate) > 0:
                    break
        self.current_used_steps += 1
        self.current_sigma = sigma
        return self.current.strength_multiplier

    @property
    def current(self) -> RegionalLoraScheduleBoundary:
        """Return the current immutable boundary."""

        return self.plan.schedule[self.current_index]

    def _effective_guarantee(self, boundary: RegionalLoraScheduleBoundary) -> int:
        """Drop guarantees for boundaries before the sampled sigma range."""

        if boundary.start_sigma > self.maximum_sigma:
            return 0
        return boundary.guarantee_steps


class RegionalLoraScheduleSession:
    """Own independent stateful WeightHook cursors for one sampling run."""

    def __init__(
        self,
        adapters: tuple[RegionalLoraAdapterPlan, ...],
        *,
        maximum_sigma: float,
    ) -> None:
        """Create one cursor per adapter in canonical composition order."""

        if not isinstance(adapters, tuple):
            raise TypeError("Regional LoRA schedule adapters must be a tuple.")
        if tuple(adapter.composition_index for adapter in adapters) != tuple(
            range(len(adapters))
        ):
            raise ValueError("Regional LoRA schedule adapters must be canonical.")
        normalized_maximum = _finite_sigma(maximum_sigma, name="maximum")
        self._cursors = tuple(
            _AdapterScheduleCursor(adapter, normalized_maximum) for adapter in adapters
        )

    def resolve(self, sigma: float) -> RegionalLoraScheduleResolution:
        """Resolve every adapter independently for one current sigma."""

        current_sigma = _finite_sigma(sigma, name="current")
        multipliers = tuple(cursor.resolve(current_sigma) for cursor in self._cursors)
        return RegionalLoraScheduleResolution(
            schedule_multipliers=multipliers,
            effective_strengths=tuple(
                cursor.plan.model_strength * multiplier
                for cursor, multiplier in zip(
                    self._cursors,
                    multipliers,
                    strict=True,
                )
            ),
        )


def _finite_sigma(value: object, *, name: str) -> float:
    """Normalize one finite real schedule sigma."""

    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"Regional LoRA {name} sigma must be a real number.")
    sigma = float(value)
    if not math.isfinite(sigma):
        raise ValueError(f"Regional LoRA {name} sigma must be finite.")
    return sigma
