# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Own immutable authored and model-converted conditioning schedule bounds."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ConditioningScheduleRange:
    """Retain optional authored percentages and converted sigma boundaries."""

    start_percent: float | None
    end_percent: float | None
    timestep_start: float | None
    timestep_end: float | None

    def __post_init__(self) -> None:
        """Validate exact optional bounds without inventing absent metadata."""

        start_percent = _optional_finite_float(
            self.start_percent,
            name="start_percent",
        )
        end_percent = _optional_finite_float(
            self.end_percent,
            name="end_percent",
        )
        timestep_start = _optional_finite_float(
            self.timestep_start,
            name="timestep_start",
        )
        timestep_end = _optional_finite_float(
            self.timestep_end,
            name="timestep_end",
        )
        for name, value in (
            ("start_percent", start_percent),
            ("end_percent", end_percent),
        ):
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"Conditioning {name} must be in [0, 1].")
        effective_start = 0.0 if start_percent is None else start_percent
        effective_end = 1.0 if end_percent is None else end_percent
        if effective_start > effective_end:
            raise ValueError("Conditioning start_percent must not exceed end_percent.")
        if (
            timestep_start is not None
            and timestep_end is not None
            and timestep_start < timestep_end
        ):
            raise ValueError(
                "Conditioning timestep_start must not be below timestep_end."
            )
        object.__setattr__(self, "start_percent", start_percent)
        object.__setattr__(self, "end_percent", end_percent)
        object.__setattr__(self, "timestep_start", timestep_start)
        object.__setattr__(self, "timestep_end", timestep_end)

    @property
    def is_time_invariant(self) -> bool:
        """Report whether this entry remains admitted for the whole trajectory."""

        if self.start_percent is None and self.timestep_start is not None:
            return False
        if self.end_percent is None and self.timestep_end is not None:
            return False
        effective_start = 0.0 if self.start_percent is None else self.start_percent
        effective_end = 1.0 if self.end_percent is None else self.end_percent
        return effective_start == 0.0 and effective_end == 1.0


def _optional_finite_float(value: object, *, name: str) -> float | None:
    """Normalize one optional real boundary without accepting booleans."""

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"Conditioning {name} must be a real number or None.")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"Conditioning {name} must be finite.")
    return normalized


UNBOUNDED_CONDITIONING_SCHEDULE = ConditioningScheduleRange(None, None, None, None)
