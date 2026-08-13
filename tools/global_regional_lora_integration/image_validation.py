# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Validate exact P9.3 same-server transition image relationships."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tools.decoded_image_comparison import (
    DecodedImageComparison,
    compare_decoded_rgb_images,
)


@dataclass(frozen=True, slots=True)
class GlobalRegionalLoraTransitionEvidence:
    """Retain accepted repeatability and global-effect comparisons."""

    distinct_before_after: DecodedImageComparison
    distinct_before_regional_only: DecodedImageComparison

    def as_json(self) -> dict[str, dict[str, int]]:
        """Return deterministic terminal transition evidence."""

        return {
            "distinct_before_after": self.distinct_before_after.as_json(),
            "distinct_before_regional_only": (
                self.distinct_before_regional_only.as_json()
            ),
        }


class GlobalRegionalLoraImageValidator:
    """Require stable distinct-global pixels across one clone transition."""

    def validate(
        self,
        images: dict[str, Path],
    ) -> GlobalRegionalLoraTransitionEvidence:
        """Require repeatability and a visible distinct global-LoRA effect."""

        before = _required_image(images, "global-global_adapter-regional-adapter_a-before")
        after = _required_image(images, "global-global_adapter-regional-adapter_a-after")
        regional = _required_image(images, "regional-adapter_a-only")
        repeated = compare_decoded_rgb_images(before, after)
        if repeated.changed_pixels != 0:
            raise ValueError(
                "P9.3 distinct global/regional output changed across the "
                "same-server regional-only transition."
            )
        global_effect = compare_decoded_rgb_images(before, regional)
        if global_effect.changed_pixels == 0:
            raise ValueError(
                "P9.3 distinct global LoRA has no decoded pixel effect relative "
                "to regional-only execution."
            )
        return GlobalRegionalLoraTransitionEvidence(repeated, global_effect)


def _required_image(images: dict[str, Path], case_id: str) -> Path:
    """Return one existing labeled image path."""

    path = images.get(case_id)
    if not isinstance(path, Path) or not path.is_file():
        raise ValueError(f"P9.3 transition image {case_id!r} is missing.")
    return path


GLOBAL_REGIONAL_LORA_IMAGE_VALIDATOR = GlobalRegionalLoraImageValidator()
