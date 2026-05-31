# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Comfy v3 node wrapper for Tag SEGS w/ WD14."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, ClassVar

from ..nodes import tooltips
from ..nodes.tag_segs_with_wd14 import TagSEGSWithWD14
from ..nodes.tile_and_tag_segs import DEFAULT_EXCLUDE_TAGS

if TYPE_CHECKING:

    class _ComfyNodeBase:
        """Type-checking base for Comfy v3 nodes."""

        RETURN_TYPES: ClassVar[list[str]]
        RETURN_NAMES: ClassVar[list[str]]

else:
    _ComfyNodeBase = import_module("comfy_api.latest").io.ComfyNode

_comfy_io: Any = None if TYPE_CHECKING else import_module("comfy_api.latest").io
ConditioningBatchIO: Any = (
    None if TYPE_CHECKING else _comfy_io.Custom("CONDITIONING_BATCH")
)
WD14TaggerIO: Any = None if TYPE_CHECKING else _comfy_io.Custom("WD14_TAGGER")


class TagSEGSWithWD14V3(_ComfyNodeBase):
    """Expose WD14 tagging for existing SEGS through Comfy's v3 API."""

    @classmethod
    def define_schema(cls) -> Any:
        """Declare the Tag SEGS w/ WD14 v3 schema."""

        return _comfy_io.Schema(
            node_id="SimpleSyrup.TagSEGSWithWD14",
            display_name="Tag SEGS w/ WD14",
            category="SimpleSyrup/Detailing",
            description=(
                "Tags existing SEGS crops with a connected WD14 tagger and "
                "returns aligned conditioning for SEGS detailing."
            ),
            search_aliases=["tag", "wd14", "segs", "detail", "regional"],
            inputs=[
                _comfy_io.Image.Input("image", tooltip=tooltips.TAG_SEGS_IMAGE),
                _comfy_io.SEGS.Input("segs", tooltip=tooltips.TAG_SEGS_SEGS),
                _comfy_io.Clip.Input("clip", tooltip=tooltips.TAG_SEGS_CLIP),
                WD14TaggerIO.Input(
                    "wd14_tagger",
                    tooltip=tooltips.TAG_SEGS_WD14_TAGGER,
                ),
                _comfy_io.String.Input(
                    "universal_positive",
                    multiline=False,
                    default="",
                    tooltip=tooltips.TAG_SEGS_UNIVERSAL_POSITIVE,
                ),
                _comfy_io.Float.Input(
                    "threshold",
                    default=0.35,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    tooltip=tooltips.TILE_THRESHOLD,
                ),
                _comfy_io.Float.Input(
                    "character_threshold",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    tooltip=tooltips.TILE_CHARACTER_THRESHOLD,
                ),
                _comfy_io.Boolean.Input(
                    "replace_underscore",
                    default=True,
                    tooltip=tooltips.TILE_REPLACE_UNDERSCORE,
                ),
                _comfy_io.Boolean.Input(
                    "trailing_comma",
                    default=False,
                    tooltip=tooltips.TILE_TRAILING_COMMA,
                ),
                _comfy_io.String.Input(
                    "exclude_tags",
                    multiline=False,
                    default=DEFAULT_EXCLUDE_TAGS,
                    tooltip=tooltips.TILE_EXCLUDE_TAGS,
                ),
            ],
            outputs=[
                _comfy_io.SEGS.Output(
                    "segs",
                    tooltip=tooltips.TAG_SEGS_SEGS_OUTPUT,
                ),
                ConditioningBatchIO.Output(
                    "positive",
                    tooltip=tooltips.TAG_SEGS_POSITIVE_OUTPUT,
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        image: object,
        segs: object,
        clip: Any,
        wd14_tagger: object,
        universal_positive: str,
        threshold: float,
        character_threshold: float,
        replace_underscore: bool,
        trailing_comma: bool,
        exclude_tags: str,
    ) -> tuple[object, object]:
        """Run the legacy implementation behind the v3 schema."""

        return TagSEGSWithWD14().tag(
            image=image,
            segs=segs,
            clip=clip,
            wd14_tagger=wd14_tagger,
            universal_positive=universal_positive,
            threshold=threshold,
            character_threshold=character_threshold,
            replace_underscore=replace_underscore,
            trailing_comma=trailing_comma,
            exclude_tags=exclude_tags,
        )
