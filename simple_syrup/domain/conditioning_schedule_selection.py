# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Own installed-Comfy-equivalent conditioning schedule admission."""

from __future__ import annotations

import math

from .conditioning_schedule import ConditioningScheduleRange


class ConditioningScheduleSelectionPolicy:
    """Match installed Comfy's inclusive converted-sigma admission policy."""

    @staticmethod
    def is_active(
        schedule: ConditioningScheduleRange,
        *,
        sigma: float,
    ) -> bool:
        """Return installed Comfy's inclusive start/end decision."""

        if not isinstance(schedule, ConditioningScheduleRange):
            raise TypeError("Conditioning selection requires a schedule range.")
        current_sigma = normalize_conditioning_sigma(sigma)
        if (
            schedule.timestep_start is not None
            and current_sigma > schedule.timestep_start
        ):
            return False
        return not (
            schedule.timestep_end is not None and current_sigma < schedule.timestep_end
        )


def normalize_conditioning_sigma(value: object) -> float:
    """Normalize one finite real current sigma without accepting booleans."""

    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError("Conditioning selection sigma must be a real number.")
    sigma = float(value)
    if not math.isfinite(sigma):
        raise ValueError("Conditioning selection sigma must be finite.")
    return sigma


CONDITIONING_SCHEDULE_SELECTION_POLICY = ConditioningScheduleSelectionPolicy()
