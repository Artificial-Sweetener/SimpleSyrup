# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Portions of this file incorporate behavior derived from
# multidiffusion-upscaler-for-automatic1111. See third_party/manifest.toml and
# third_party/NOTICE.md.

"""Domain models for SEGS-driven regional diffusion detailing."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .conditioning_batch import Conditioning, ConditioningBatch
from .segs import CropRegion, NativeSegs, Segment

OPERATION = "Detail SEGS as Regions"


@dataclass(frozen=True)
class SegmentConditioningPair:
    """Bind one SEG to its exact per-region positive conditioning."""

    index: int
    segment: Segment
    positive: Conditioning


@dataclass(frozen=True)
class ImageRegion:
    """Represent one paired region in image-pixel space."""

    index: int
    label: str
    crop_region: CropRegion
    image_mask: torch.Tensor
    positive: Conditioning


@dataclass(frozen=True)
class LatentBox:
    """Represent one rectangular region in latent-space coordinates."""

    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class LatentRegion:
    """Represent one paired region in latent space."""

    index: int
    label: str
    latent_box: LatentBox
    latent_mask: torch.Tensor
    positive: Conditioning


def pair_segments_with_conditioning(
    segs: NativeSegs,
    region_positive: object,
    *,
    image_height: int,
    image_width: int,
) -> tuple[SegmentConditioningPair, ...]:
    """Return SEGS paired exactly with regional positive conditioning."""

    header, segments = segs
    validate_segs_image_dimensions(
        segs_height=header[0],
        segs_width=header[1],
        image_height=image_height,
        image_width=image_width,
    )
    if not segments:
        return ()
    if not isinstance(region_positive, ConditioningBatch):
        raise TypeError(
            f"{OPERATION} requires region_positive to be CONDITIONING_BATCH."
        )
    if len(region_positive.entries) != len(segments):
        raise ValueError(
            f"{OPERATION} requires region_positive to contain exactly one "
            "conditioning entry per SEG: got "
            f"{len(region_positive.entries)} conditioning entries for "
            f"{len(segments)} SEGS."
        )

    pairs: list[SegmentConditioningPair] = []
    for index, (segment, positive) in enumerate(
        zip(segments, region_positive.entries, strict=True)
    ):
        if isinstance(positive, ConditioningBatch):
            raise TypeError(
                f"{OPERATION} region_positive entry {index} must be a normal "
                "CONDITIONING value, not CONDITIONING_BATCH."
            )
        _validate_segment_within_image(
            segment,
            index=index,
            image_height=image_height,
            image_width=image_width,
        )
        pairs.append(
            SegmentConditioningPair(
                index=index,
                segment=segment,
                positive=positive,
            )
        )
    return tuple(pairs)


def validate_segs_image_dimensions(
    *,
    segs_height: int,
    segs_width: int,
    image_height: int,
    image_width: int,
) -> None:
    """Reject SEGS headers that do not describe the current image."""

    if (segs_height, segs_width) == (image_height, image_width):
        return
    raise ValueError(
        f"{OPERATION} requires SEGS header dimensions to match the image: "
        f"SEGS is {segs_height}x{segs_width}, image is "
        f"{image_height}x{image_width}."
    )


def _validate_segment_within_image(
    segment: Segment,
    *,
    index: int,
    image_height: int,
    image_width: int,
) -> None:
    """Reject a segment whose crop region falls outside the image."""

    region = segment.crop_region
    if region.right <= image_width and region.bottom <= image_height:
        return
    raise ValueError(
        f"{OPERATION} SEG {index} ('{segment.label}') crop_region must fit "
        f"inside the image; got ({region.left}, {region.top}, {region.right}, "
        f"{region.bottom}) for image {image_height}x{image_width}."
    )
