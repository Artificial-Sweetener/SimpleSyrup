# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Compose complete persistent-variant denoiser outputs by regional ownership."""

from __future__ import annotations

import torch


class StandardUnetVariantOutputComposer:
    """Blend feathered coverage and normalize overlapping regional outputs."""

    @staticmethod
    def compose(
        variant_outputs: tuple[torch.Tensor, ...],
        masks: torch.Tensor,
        *,
        base_output: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return one shape-preserving normalized model prediction."""

        if not variant_outputs:
            raise ValueError("Standard UNet composition requires variant outputs.")
        reference = variant_outputs[0]
        if not isinstance(reference, torch.Tensor) or reference.ndim != 4:
            raise TypeError("Standard UNet variant outputs must use BCHW layout.")
        if any(
            not isinstance(output, torch.Tensor)
            or output.shape != reference.shape
            or output.device != reference.device
            or output.dtype != reference.dtype
            for output in variant_outputs
        ):
            raise ValueError("Standard UNet variant outputs must be aligned.")
        expected_masks = (
            len(variant_outputs),
            int(reference.shape[0]),
            1,
            int(reference.shape[2]),
            int(reference.shape[3]),
        )
        if not isinstance(masks, torch.Tensor) or tuple(masks.shape) != expected_masks:
            raise ValueError("Standard UNet variant masks must use R/B/1/H/W layout.")
        if masks.device != reference.device or masks.dtype != reference.dtype:
            raise ValueError("Standard UNet variant masks must match output type.")
        stacked = torch.stack(variant_outputs, dim=0)
        coverage = masks.sum(dim=0)
        base_weight = (1.0 - coverage).clamp_min(0.0)
        if base_output is None:
            if bool((coverage <= 0.0).any().item()):
                raise ValueError("Variant-only composition contains uncovered pixels.")
            numerator = (stacked * masks).sum(dim=0)
        else:
            if (
                not isinstance(base_output, torch.Tensor)
                or base_output.shape != reference.shape
                or base_output.device != reference.device
                or base_output.dtype != reference.dtype
            ):
                raise ValueError("Standard UNet base output must match variants.")
            numerator = (stacked * masks).sum(dim=0) + base_output * base_weight
        denominator = (coverage + base_weight).clamp_min(
            torch.finfo(reference.dtype).eps
        )
        return numerator / denominator


STANDARD_UNET_VARIANT_OUTPUT_COMPOSER = StandardUnetVariantOutputComposer()
