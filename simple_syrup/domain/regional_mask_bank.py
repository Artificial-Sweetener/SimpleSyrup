# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Define the immutable full-canvas authority for regional masks."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class RegionalMaskBank:
    """Hold separate planning and conditioning masks on one latent canvas."""

    planning_masks: torch.Tensor
    conditioning_masks: torch.Tensor
    canvas_width: int
    canvas_height: int

    def __post_init__(self) -> None:
        """Reject malformed, divergent, or aliased canonical mask tensors."""

        if self.canvas_width < 1 or self.canvas_height < 1:
            raise ValueError("Regional mask canvas dimensions must be positive.")
        self._validate_mask_batch(self.planning_masks, name="Planning")
        self._validate_mask_batch(self.conditioning_masks, name="Conditioning")
        if self.planning_masks.shape != self.conditioning_masks.shape:
            raise ValueError(
                "Regional planning and conditioning mask shapes must match."
            )
        if self.planning_masks.dtype != self.conditioning_masks.dtype:
            raise ValueError(
                "Regional planning and conditioning mask dtypes must match."
            )
        if self.planning_masks.device != self.conditioning_masks.device:
            raise ValueError(
                "Regional planning and conditioning mask devices must match."
            )
        if (
            self.planning_masks.untyped_storage().data_ptr()
            == self.conditioning_masks.untyped_storage().data_ptr()
        ):
            raise ValueError(
                "Regional planning and conditioning masks must not share storage."
            )

    @property
    def region_count(self) -> int:
        """Return the number of ordered authored regions."""

        return int(self.planning_masks.shape[0])

    def _validate_mask_batch(self, masks: torch.Tensor, *, name: str) -> None:
        """Validate one normalized floating-point BHW mask batch."""

        if not isinstance(masks, torch.Tensor):
            raise TypeError(f"{name} regional masks must be a torch.Tensor.")
        if masks.ndim != 3:
            raise ValueError(f"{name} regional masks must use BHW layout.")
        if int(masks.shape[0]) < 1:
            raise ValueError(f"{name} regional masks require at least one region.")
        if tuple(masks.shape[1:]) != (self.canvas_height, self.canvas_width):
            raise ValueError(
                f"{name} regional masks must match the full latent canvas "
                f"{self.canvas_width}x{self.canvas_height}."
            )
        if not masks.is_floating_point():
            raise TypeError(f"{name} regional masks must use a floating-point dtype.")
        if not bool(torch.isfinite(masks).all()):
            raise ValueError(f"{name} regional masks must contain only finite values.")
        if not bool(((masks >= 0.0) & (masks <= 1.0)).all()):
            raise ValueError(f"{name} regional masks must stay within [0, 1].")
