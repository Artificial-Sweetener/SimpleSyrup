# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Own Comfy-equivalent within-region conditioning-output combination."""

from __future__ import annotations

import math

import torch


class RegionalConditioningOutputCombiner:
    """Combine ordered active-entry outputs with native Comfy strength semantics."""

    def combine(
        self,
        outputs: tuple[torch.Tensor, ...],
        *,
        strengths: tuple[tuple[float, ...], ...],
    ) -> torch.Tensor:
        """Return the ordered strength-weighted output normalized like Comfy."""

        if not isinstance(outputs, tuple) or not outputs:
            raise ValueError("Regional conditioning combination requires outputs.")
        if not isinstance(strengths, tuple) or len(strengths) != len(outputs):
            raise ValueError(
                "Regional conditioning strengths must align with ordered outputs."
            )
        authority = outputs[0]
        if not isinstance(authority, torch.Tensor) or authority.ndim < 1:
            raise TypeError("Regional conditioning output must be a tensor batch.")
        weighted = torch.zeros_like(authority)
        counts = torch.ones_like(authority) * 1e-37
        weight_shape = (int(authority.shape[0]),) + (1,) * (authority.ndim - 1)
        for entry_index, (output, entry_strengths) in enumerate(
            zip(outputs, strengths, strict=True)
        ):
            self._validate_entry(
                output,
                entry_strengths,
                authority=authority,
                entry_index=entry_index,
            )
            weights = authority.new_tensor(entry_strengths).reshape(weight_shape)
            weighted += output * weights
            counts += weights
        return weighted / counts

    @staticmethod
    def _validate_entry(
        output: object,
        strengths: object,
        *,
        authority: torch.Tensor,
        entry_index: int,
    ) -> None:
        """Require exact output structure and one finite strength per sample."""

        if not isinstance(output, torch.Tensor):
            raise TypeError(
                f"Regional conditioning output {entry_index} must be a tensor."
            )
        if (
            output.shape != authority.shape
            or output.device != authority.device
            or output.dtype != authority.dtype
        ):
            raise ValueError(
                f"Regional conditioning output {entry_index} must match the first "
                "output shape, device, and dtype."
            )
        if not isinstance(strengths, tuple) or len(strengths) != int(
            authority.shape[0]
        ):
            raise ValueError(
                f"Regional conditioning output {entry_index} strengths must match "
                "the output batch."
            )
        for strength in strengths:
            if isinstance(strength, bool) or not isinstance(strength, int | float):
                raise TypeError("Regional conditioning strengths must be real numbers.")
            if not math.isfinite(float(strength)):
                raise ValueError("Regional conditioning strengths must be finite.")


REGIONAL_CONDITIONING_OUTPUT_COMBINER = RegionalConditioningOutputCombiner()
