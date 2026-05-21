# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""ComfyUI node declaration for Tile & Tag SEGS."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

from ..domain.tile_segs import IRREGULAR_MASK_MODES, TileSEGSControls
from ..nodes import tooltips
from ..runtime.wd14_tagger import WD14TagFormattingControls
from ..services.tile_and_tag_segs_service import TileAndTagSEGSService

DEFAULT_EXCLUDE_TAGS = "1girl, solo, long_hair, short_hair, silhouette"


class TileAndTagSEGS:
    """Create tile SEGS and WD14 conditioning for SEGS detailing."""

    RETURN_TYPES = ("SEGS", "CONDITIONING_BATCH")
    RETURN_NAMES = ("segs", "positive")
    OUTPUT_TOOLTIPS = (
        tooltips.TILE_SEGS_OUTPUT,
        tooltips.TILE_POSITIVE_OUTPUT,
    )
    FUNCTION = "tile_and_tag"
    CATEGORY = "SimpleSyrup/Detailing"
    DESCRIPTION = (
        "Creates tile SEGS, tags each tile with a connected WD14 tagger, and "
        "returns aligned conditioning for SEGS detailing."
    )
    SEARCH_ALIASES = ["tile", "tag", "wd14", "segs", "detail"]

    service_class: ClassVar[Callable[[], TileAndTagSEGSService]] = TileAndTagSEGSService

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[Any, ...]]]:
        """Declare ComfyUI inputs for tile creation and WD14 tagging."""

        return {
            "required": {
                "image": ("IMAGE", {"tooltip": tooltips.TILE_IMAGE}),
                "clip": ("CLIP", {"tooltip": tooltips.TILE_CLIP}),
                "wd14_tagger": (
                    "WD14_TAGGER",
                    {"tooltip": tooltips.TILE_WD14_TAGGER},
                ),
                "universal_positive": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": tooltips.TILE_UNIVERSAL_POSITIVE,
                    },
                ),
                "bbox_size": (
                    "INT",
                    {
                        "default": 872,
                        "min": 64,
                        "max": 4096,
                        "step": 8,
                        "tooltip": tooltips.TILE_BBOX_SIZE,
                    },
                ),
                "crop_factor": (
                    "FLOAT",
                    {
                        "default": 1.1,
                        "min": 1.0,
                        "max": 10.0,
                        "step": 0.01,
                        "tooltip": tooltips.TILE_CROP_FACTOR,
                    },
                ),
                "min_overlap": (
                    "INT",
                    {
                        "default": 16,
                        "min": 0,
                        "max": 512,
                        "step": 1,
                        "tooltip": tooltips.TILE_MIN_OVERLAP,
                    },
                ),
                "filter_segs_dilation": (
                    "INT",
                    {
                        "default": 20,
                        "min": -255,
                        "max": 255,
                        "step": 1,
                        "tooltip": tooltips.TILE_FILTER_SEGS_DILATION,
                    },
                ),
                "mask_irregularity": (
                    "FLOAT",
                    {
                        "default": 0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": tooltips.TILE_MASK_IRREGULARITY,
                    },
                ),
                "irregular_mask_mode": (
                    IRREGULAR_MASK_MODES,
                    {
                        "default": "Reuse fast",
                        "tooltip": tooltips.TILE_IRREGULAR_MASK_MODE,
                    },
                ),
                "threshold": (
                    "FLOAT",
                    {
                        "default": 0.35,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": tooltips.TILE_THRESHOLD,
                    },
                ),
                "character_threshold": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": tooltips.TILE_CHARACTER_THRESHOLD,
                    },
                ),
                "replace_underscore": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": tooltips.TILE_REPLACE_UNDERSCORE,
                    },
                ),
                "trailing_comma": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": tooltips.TILE_TRAILING_COMMA,
                    },
                ),
                "exclude_tags": (
                    "STRING",
                    {
                        "default": DEFAULT_EXCLUDE_TAGS,
                        "multiline": False,
                        "tooltip": tooltips.TILE_EXCLUDE_TAGS,
                    },
                ),
            },
        }

    def tile_and_tag(
        self,
        image: object,
        clip: Any,
        wd14_tagger: object,
        universal_positive: object,
        bbox_size: object,
        crop_factor: object,
        min_overlap: object,
        filter_segs_dilation: object,
        mask_irregularity: object,
        irregular_mask_mode: object,
        threshold: object,
        character_threshold: object,
        replace_underscore: object,
        trailing_comma: object,
        exclude_tags: object,
    ) -> tuple[object, object]:
        """Create ordered tile SEGS and WD14 conditioning."""

        tile_controls = TileSEGSControls(
            bbox_size=_int_input(bbox_size, "bbox_size"),
            crop_factor=_float_input(crop_factor, "crop_factor"),
            min_overlap=_int_input(min_overlap, "min_overlap"),
            filter_segs_dilation=_int_input(
                filter_segs_dilation,
                "filter_segs_dilation",
            ),
            mask_irregularity=_float_input(mask_irregularity, "mask_irregularity"),
            irregular_mask_mode=_str_input(
                irregular_mask_mode,
                "irregular_mask_mode",
            ),
        )
        tag_controls = WD14TagFormattingControls(
            threshold=_float_input(threshold, "threshold"),
            character_threshold=_float_input(
                character_threshold,
                "character_threshold",
            ),
            replace_underscore=_bool_input(
                replace_underscore,
                "replace_underscore",
            ),
            trailing_comma=_bool_input(trailing_comma, "trailing_comma"),
            exclude_tags=_str_input(exclude_tags, "exclude_tags"),
        )
        result = (
            type(self)
            .service_class()
            .tile_and_tag(
                image=image,
                clip=clip,
                wd14_tagger=wd14_tagger,
                tile_controls=tile_controls,
                tag_controls=tag_controls,
                universal_positive=_str_input(
                    universal_positive,
                    "universal_positive",
                ),
            )
        )
        return result.segs, result.positive


def _int_input(value: object, name: str) -> int:
    """Return an integer node input."""

    if isinstance(value, (int, float, str)):
        return int(value)
    raise TypeError(f"Tile & Tag SEGS requires '{name}' to be an int.")


def _float_input(value: object, name: str) -> float:
    """Return a float node input."""

    if isinstance(value, (int, float, str)):
        return float(value)
    raise TypeError(f"Tile & Tag SEGS requires '{name}' to be a float.")


def _str_input(value: object, name: str) -> str:
    """Return a string node input."""

    if isinstance(value, str):
        return value
    raise TypeError(f"Tile & Tag SEGS requires '{name}' to be a string.")


def _bool_input(value: object, name: str) -> bool:
    """Return a boolean node input."""

    if isinstance(value, bool):
        return value
    raise TypeError(f"Tile & Tag SEGS requires '{name}' to be a boolean.")
