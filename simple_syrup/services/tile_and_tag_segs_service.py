# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Application service for Tile & Tag SEGS orchestration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import torch

from ..domain.conditioning_batch import ConditioningBatch
from ..domain.prompt_composition import prefix_prompt
from ..domain.segs import ImpactSegs, NativeSegs, to_impact_compatible_segs
from ..domain.tile_segs import TileSEGSBuilder, TileSEGSControls
from ..masking.segs_mask_ops import crop_image, validate_single_image
from ..runtime.conditioning_encoding import ComfyConditioningEncoder
from ..runtime.loaded_models import LoadedWD14Tagger, unwrap_wd14_tagger
from ..runtime.progress import ProgressReporter, create_comfy_progress
from ..runtime.wd14_tagger import WD14TagFormattingControls, WD14Tagger
from ..shared.logging import get_logger

LOGGER = get_logger(__name__)


class TileSEGSBuildingBoundary(Protocol):
    """Build tile SEGS for an image."""

    def build(
        self,
        image: torch.Tensor,
        controls: TileSEGSControls,
    ) -> NativeSegs:
        """Return native SEGS for ordered tiles."""


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
class TileAndTagSEGSResult:
    """Return tile SEGS and aligned prompt conditioning."""

    segs: ImpactSegs
    positive: ConditioningBatch


class TileAndTagSEGSService:
    """Create tile SEGS and aligned WD14 conditioning for detailing."""

    def __init__(
        self,
        tiler: TileSEGSBuildingBoundary | None = None,
        tagger: WD14TaggingBoundary | None = None,
        encoder: ConditioningEncodingBoundary | None = None,
        progress_factory: Callable[[int], ProgressReporter] | None = None,
    ) -> None:
        """Create the service with injectable collaborators for tests."""

        self._tiler = tiler or TileSEGSBuilder()
        self._tagger = tagger or WD14Tagger()
        self._encoder = encoder or ComfyConditioningEncoder()
        self._progress_factory = progress_factory or create_comfy_progress

    def tile_and_tag(
        self,
        image: object,
        clip: Any,
        wd14_tagger: object,
        tile_controls: TileSEGSControls,
        tag_controls: WD14TagFormattingControls,
        universal_positive: str,
    ) -> TileAndTagSEGSResult:
        """Return ordered tile SEGS and aligned positive conditioning."""

        image_tensor = validate_single_image(image, "Tile & Tag SEGS")
        loaded_tagger = unwrap_wd14_tagger(wd14_tagger)
        native_segs = self._tiler.build(
            image_tensor,
            tile_controls,
        )
        _header, segments = native_segs
        if not segments:
            raise ValueError("No tile SEGS were generated for Tile & Tag SEGS.")
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
                "WD14 tagger returned "
                f"{len(tags)} tag(s) for {len(segments)} tile SEGS."
            )
        prompts = tuple(prefix_prompt(universal_positive, tag) for tag in tags)
        positive = self._encoder.encode_batch(clip, prompts)
        progress.update(1)
        if len(positive.entries) != len(segments):
            raise ValueError(
                "Conditioning encoder returned "
                f"{len(positive.entries)} entries for {len(segments)} tile SEGS."
            )
        LOGGER.info(
            "Tile & Tag SEGS pass completed",
            extra={
                "operation": "tile_and_tag_segs",
                "segment_count": len(segments),
                "wd14_model": loaded_tagger.model_id,
                "threshold": tag_controls.threshold,
                "character_threshold": tag_controls.character_threshold,
                "universal_positive_present": bool(universal_positive.strip()),
            },
        )
        return TileAndTagSEGSResult(
            segs=to_impact_compatible_segs(native_segs),
            positive=positive,
        )
