# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Own one standard-UNet sampling run's regional LoRA schedule state."""

from __future__ import annotations

import math
from contextvars import ContextVar
from dataclasses import dataclass

import torch

from ...domain.regional_lora_plan import RegionalLoraPlan
from ..regional_attention_model_call_values import uniform_model_call_sigma
from ..regional_lora_schedule_resolution import RegionalLoraScheduleSession


@dataclass(frozen=True, slots=True)
class _StandardUnetScheduleSlot:
    """Bind one sampling schedule tensor to its stateful canonical cursor."""

    sample_sigmas: torch.Tensor
    maximum_sigma: float
    session: RegionalLoraScheduleSession
    last_sigma: float | None


class StandardUnetLoraSchedule:
    """Resolve ordered adapter multipliers across one standard-UNet sample."""

    def __init__(self, plan: RegionalLoraPlan) -> None:
        """Retain one immutable nonempty plan and an empty task-local cursor."""

        if not isinstance(plan, RegionalLoraPlan) or not plan.adapters:
            raise ValueError("Standard UNet LoRA schedule requires adapters.")
        self._plan = plan
        self._current: ContextVar[_StandardUnetScheduleSlot | None] = ContextVar(
            "simple_syrup_standard_unet_regional_lora_schedule",
            default=None,
        )

    def resolve(self, transformer_options: dict[str, object]) -> tuple[float, ...]:
        """Advance the canonical cursor for one exact denoiser sigma."""

        if not isinstance(transformer_options, dict):
            raise TypeError("Standard UNet schedule options must be a dictionary.")
        sample_sigmas = _floating_tensor(
            transformer_options.get("sample_sigmas"),
            "sample_sigmas",
        )
        current_sigmas = _floating_tensor(
            transformer_options.get("sigmas"),
            "sigmas",
        )
        current_sigma = uniform_model_call_sigma(current_sigmas)
        slot = self._current.get()
        maximum = (
            _maximum_finite_sigma(sample_sigmas)
            if slot is None or slot.sample_sigmas is not sample_sigmas
            else slot.maximum_sigma
        )
        if (
            slot is None
            or slot.sample_sigmas is not sample_sigmas
            or (slot.last_sigma is not None and current_sigma > slot.last_sigma)
        ):
            slot = _StandardUnetScheduleSlot(
                sample_sigmas,
                maximum,
                RegionalLoraScheduleSession(
                    self._plan.adapters,
                    maximum_sigma=maximum,
                ),
                None,
            )
        resolution = slot.session.resolve(current_sigma)
        self._current.set(
            _StandardUnetScheduleSlot(
                sample_sigmas,
                slot.maximum_sigma,
                slot.session,
                current_sigma,
            )
        )
        return resolution.schedule_multipliers

    def clear(self) -> None:
        """Release the task-local sampling cursor on model detach."""

        self._current.set(None)


def _floating_tensor(value: object, name: str) -> torch.Tensor:
    """Require one nonempty floating Comfy sigma tensor."""

    if (
        not isinstance(value, torch.Tensor)
        or not value.is_floating_point()
        or value.numel() < 1
    ):
        raise TypeError(
            f"Standard UNet regional LoRA {name} must be a floating tensor."
        )
    return value


def _maximum_finite_sigma(sample_sigmas: torch.Tensor) -> float:
    """Validate a new sampling schedule on CPU and return its maximum."""

    values = tuple(
        float(value) for value in sample_sigmas.flatten().detach().cpu().tolist()
    )
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Standard UNet regional LoRA sample_sigmas must be finite.")
    return max(values)
