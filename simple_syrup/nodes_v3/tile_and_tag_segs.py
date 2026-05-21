# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Comfy v3 node wrapper for Tile & Tag SEGS."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, ClassVar

from ..domain.tile_segs import IRREGULAR_MASK_MODES
from ..nodes import tooltips
from ..nodes.tile_and_tag_segs import DEFAULT_EXCLUDE_TAGS, TileAndTagSEGS

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


class TileAndTagSEGSV3(_ComfyNodeBase):
    """Expose Tile & Tag SEGS through Comfy's v3 extension API."""

    @classmethod
    def define_schema(cls) -> Any:
        """Declare the Tile & Tag SEGS v3 schema."""

        return _comfy_io.Schema(
            node_id="SimpleSyrup.TileAndTagSEGS",
            display_name="Tile & Tag SEGS",
            category="SimpleSyrup/Detailing",
            description=(
                "Creates tile SEGS, tags each tile with a connected WD14 tagger, "
                "and returns aligned conditioning for SEGS detailing."
            ),
            search_aliases=["tile", "tag", "wd14", "segs", "detail"],
            inputs=[
                _comfy_io.Image.Input("image", tooltip=tooltips.TILE_IMAGE),
                _comfy_io.Clip.Input(
                    "clip",
                    tooltip=tooltips.TILE_CLIP,
                ),
                WD14TaggerIO.Input(
                    "wd14_tagger",
                    tooltip=tooltips.TILE_WD14_TAGGER,
                ),
                _comfy_io.String.Input(
                    "universal_positive",
                    multiline=False,
                    default="",
                    tooltip=tooltips.TILE_UNIVERSAL_POSITIVE,
                ),
                _comfy_io.Int.Input(
                    "bbox_size",
                    default=872,
                    min=64,
                    max=4096,
                    step=8,
                    tooltip=tooltips.TILE_BBOX_SIZE,
                ),
                _comfy_io.Float.Input(
                    "crop_factor",
                    default=1.1,
                    min=1.0,
                    max=10.0,
                    step=0.01,
                    tooltip=tooltips.TILE_CROP_FACTOR,
                ),
                _comfy_io.Int.Input(
                    "min_overlap",
                    default=16,
                    min=0,
                    max=512,
                    step=1,
                    tooltip=tooltips.TILE_MIN_OVERLAP,
                ),
                _comfy_io.Int.Input(
                    "filter_segs_dilation",
                    default=20,
                    min=-255,
                    max=255,
                    step=1,
                    tooltip=tooltips.TILE_FILTER_SEGS_DILATION,
                ),
                _comfy_io.Float.Input(
                    "mask_irregularity",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=tooltips.TILE_MASK_IRREGULARITY,
                ),
                _comfy_io.Combo.Input(
                    "irregular_mask_mode",
                    options=list(IRREGULAR_MASK_MODES),
                    default="Reuse fast",
                    tooltip=tooltips.TILE_IRREGULAR_MASK_MODE,
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
                    tooltip=tooltips.TILE_SEGS_OUTPUT,
                ),
                ConditioningBatchIO.Output(
                    "positive",
                    tooltip=tooltips.TILE_POSITIVE_OUTPUT,
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        image: object,
        clip: Any,
        wd14_tagger: object,
        universal_positive: str,
        bbox_size: int,
        crop_factor: float,
        min_overlap: int,
        filter_segs_dilation: int,
        mask_irregularity: float,
        irregular_mask_mode: str,
        threshold: float,
        character_threshold: float,
        replace_underscore: bool,
        trailing_comma: bool,
        exclude_tags: str,
    ) -> tuple[object, object]:
        """Run the legacy implementation behind the v3 schema."""

        return TileAndTagSEGS().tile_and_tag(
            image=image,
            clip=clip,
            wd14_tagger=wd14_tagger,
            universal_positive=universal_positive,
            bbox_size=bbox_size,
            crop_factor=crop_factor,
            min_overlap=min_overlap,
            filter_segs_dilation=filter_segs_dilation,
            mask_irregularity=mask_irregularity,
            irregular_mask_mode=irregular_mask_mode,
            threshold=threshold,
            character_threshold=character_threshold,
            replace_underscore=replace_underscore,
            trailing_comma=trailing_comma,
            exclude_tags=exclude_tags,
        )
