# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""ComfyUI node declaration for tagging existing SEGS with WD14."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

from ..nodes import tooltips
from ..runtime.wd14_tagger import WD14TagFormattingControls
from ..services.tag_segs_with_wd14_service import TagSEGSWithWD14Service
from .tile_and_tag_segs import DEFAULT_EXCLUDE_TAGS


class TagSEGSWithWD14:
    """Create WD14 conditioning for existing SEGS."""

    RETURN_TYPES = ("SEGS", "CONDITIONING_BATCH")
    RETURN_NAMES = ("segs", "positive")
    OUTPUT_TOOLTIPS = (
        tooltips.TAG_SEGS_SEGS_OUTPUT,
        tooltips.TAG_SEGS_POSITIVE_OUTPUT,
    )
    FUNCTION = "tag"
    CATEGORY = "SimpleSyrup/Detailing"
    DESCRIPTION = (
        "Tags existing SEGS crops with a connected WD14 tagger and returns "
        "aligned conditioning for SEGS detailing."
    )
    SEARCH_ALIASES = ["tag", "wd14", "segs", "detail", "regional"]

    service_class: ClassVar[Callable[[], TagSEGSWithWD14Service]] = (
        TagSEGSWithWD14Service
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[Any, ...]]]:
        """Declare ComfyUI inputs for WD14 tagging of existing SEGS."""

        return {
            "required": {
                "image": ("IMAGE", {"tooltip": tooltips.TAG_SEGS_IMAGE}),
                "segs": ("SEGS", {"tooltip": tooltips.TAG_SEGS_SEGS}),
                "clip": ("CLIP", {"tooltip": tooltips.TAG_SEGS_CLIP}),
                "wd14_tagger": (
                    "WD14_TAGGER",
                    {"tooltip": tooltips.TAG_SEGS_WD14_TAGGER},
                ),
                "universal_positive": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": tooltips.TAG_SEGS_UNIVERSAL_POSITIVE,
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

    def tag(
        self,
        image: object,
        segs: object,
        clip: Any,
        wd14_tagger: object,
        universal_positive: object,
        threshold: object,
        character_threshold: object,
        replace_underscore: object,
        trailing_comma: object,
        exclude_tags: object,
    ) -> tuple[object, object]:
        """Tag existing SEGS and return aligned conditioning."""

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
            .tag(
                image=image,
                segs=segs,
                clip=clip,
                wd14_tagger=wd14_tagger,
                tag_controls=tag_controls,
                universal_positive=_str_input(
                    universal_positive,
                    "universal_positive",
                ),
            )
        )
        return result.segs, result.positive


def _float_input(value: object, name: str) -> float:
    """Return a float node input."""

    if isinstance(value, (int, float, str)):
        return float(value)
    raise TypeError(f"Tag SEGS w/ WD14 requires '{name}' to be a float.")


def _str_input(value: object, name: str) -> str:
    """Return a string node input."""

    if isinstance(value, str):
        return value
    raise TypeError(f"Tag SEGS w/ WD14 requires '{name}' to be a string.")


def _bool_input(value: object, name: str) -> bool:
    """Return a boolean node input."""

    if isinstance(value, bool):
        return value
    raise TypeError(f"Tag SEGS w/ WD14 requires '{name}' to be a boolean.")
