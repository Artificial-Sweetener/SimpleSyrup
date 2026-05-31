# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Application service for WD14 tagging of existing SEGS."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import torch

from ..domain.conditioning_batch import ConditioningBatch
from ..domain.prompt_composition import prefix_prompt
from ..domain.segs import (
    ImpactSegs,
    NativeSegs,
    Segment,
    coerce_segs,
    to_impact_compatible_segs,
)
from ..masking.segs_mask_ops import crop_image, validate_single_image
from ..runtime.conditioning_encoding import ComfyConditioningEncoder
from ..runtime.loaded_models import LoadedWD14Tagger, unwrap_wd14_tagger
from ..runtime.progress import ProgressReporter, create_comfy_progress
from ..runtime.wd14_tagger import WD14TagFormattingControls, WD14Tagger
from ..shared.logging import get_logger

LOGGER = get_logger(__name__)
OPERATION = "Tag SEGS w/ WD14"


class WD14TaggingBoundary(Protocol):
    """Tag ordered image crops."""

    def tag_images(
        self,
        loaded_tagger: LoadedWD14Tagger,
        images: tuple[torch.Tensor, ...],
        controls: WD14TagFormattingControls,
        progress: ProgressReporter | None = None,
    ) -> tuple[str, ...]:
        """Return one tag string per image in input order."""


class ConditioningEncodingBoundary(Protocol):
    """Encode ordered prompts into a conditioning batch."""

    def encode_batch(self, clip: Any, chunks: tuple[str, ...]) -> ConditioningBatch:
        """Return conditioning entries in prompt order."""


@dataclass(frozen=True)
class TagSEGSWithWD14Result:
    """Return unchanged SEGS and aligned positive conditioning."""

    segs: ImpactSegs
    positive: ConditioningBatch


class TagSEGSWithWD14Service:
    """Tag provided SEGS crops and encode aligned regional conditioning."""

    def __init__(
        self,
        tagger: WD14TaggingBoundary | None = None,
        encoder: ConditioningEncodingBoundary | None = None,
        progress_factory: Callable[[int], ProgressReporter] | None = None,
    ) -> None:
        """Create the service with injectable collaborators for tests."""

        self._tagger = tagger or WD14Tagger()
        self._encoder = encoder or ComfyConditioningEncoder()
        self._progress_factory = progress_factory or create_comfy_progress

    def tag(
        self,
        image: object,
        segs: object,
        clip: Any,
        wd14_tagger: object,
        tag_controls: WD14TagFormattingControls,
        universal_positive: str,
    ) -> TagSEGSWithWD14Result:
        """Return original SEGS plus WD14-derived conditioning in segment order."""

        image_tensor = validate_single_image(image, OPERATION)
        native_segs = coerce_segs(segs)
        self._validate_segs_target_image(native_segs, image_tensor)
        _header, segments = native_segs
        if not segments:
            raise ValueError("No SEGS were provided for Tag SEGS w/ WD14.")

        loaded_tagger = unwrap_wd14_tagger(wd14_tagger)
        progress = self._progress_factory(len(segments) + 2)
        progress.update(1)
        crops = tuple(
            crop_image(image_tensor, segment.crop_region) for segment in segments
        )
        tags = self._tagger.tag_images(
            loaded_tagger,
            crops,
            tag_controls,
            progress=progress,
        )
        if len(tags) != len(segments):
            raise ValueError(
                f"WD14 tagger returned {len(tags)} tag(s) for {len(segments)} SEGS."
            )

        prompts = tuple(prefix_prompt(universal_positive, tag) for tag in tags)
        positive = self._encoder.encode_batch(clip, prompts)
        progress.update(1)
        if len(positive.entries) != len(segments):
            raise ValueError(
                "Conditioning encoder returned "
                f"{len(positive.entries)} entries for {len(segments)} SEGS."
            )

        LOGGER.info(
            "Tag SEGS w/ WD14 pass completed",
            extra={
                "operation": "tag_segs_with_wd14",
                "segment_count": len(segments),
                "wd14_model": loaded_tagger.model_id,
                "threshold": tag_controls.threshold,
                "character_threshold": tag_controls.character_threshold,
                "universal_positive_present": bool(universal_positive.strip()),
            },
        )
        return TagSEGSWithWD14Result(
            segs=to_impact_compatible_segs(native_segs),
            positive=positive,
        )

    def _validate_segs_target_image(
        self,
        segs: NativeSegs,
        image: torch.Tensor,
    ) -> None:
        """Reject SEGS that cannot be cropped from the provided image."""

        header, segments = segs
        image_height = int(image.shape[1])
        image_width = int(image.shape[2])
        if header != (image_height, image_width):
            raise ValueError(
                f"{OPERATION} requires SEGS header dimensions to match the image: "
                f"SEGS is {header[0]}x{header[1]}, image is "
                f"{image_height}x{image_width}."
            )
        for index, segment in enumerate(segments):
            _validate_segment_crop(segment, index, image_height, image_width)


def _validate_segment_crop(
    segment: Segment,
    index: int,
    image_height: int,
    image_width: int,
) -> None:
    """Reject a segment crop region that falls outside the image."""

    region = segment.crop_region
    if region.right <= image_width and region.bottom <= image_height:
        return
    raise ValueError(
        f"{OPERATION} SEG {index} ('{segment.label}') crop_region must fit "
        f"inside the image; got ({region.left}, {region.top}, {region.right}, "
        f"{region.bottom}) for image {image_height}x{image_width}."
    )
