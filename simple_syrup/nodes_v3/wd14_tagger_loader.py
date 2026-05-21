# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Comfy v3 node wrapper for loading WD14 taggers."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, ClassVar

from ..nodes import tooltips
from ..nodes.wd14_tagger_loader import WD14TaggerLoader
from ..runtime.model_catalog import DEFAULT_WD14_TAGGER_MODEL
from ..runtime.model_choices import ModelChoiceService, default_choice

if TYPE_CHECKING:

    class _ComfyNodeBase:
        """Type-checking base for Comfy v3 nodes."""

        RETURN_TYPES: ClassVar[list[str]]
        RETURN_NAMES: ClassVar[list[str]]

else:
    _ComfyNodeBase = import_module("comfy_api.latest").io.ComfyNode

_comfy_io: Any = None if TYPE_CHECKING else import_module("comfy_api.latest").io
WD14TaggerIO: Any = None if TYPE_CHECKING else _comfy_io.Custom("WD14_TAGGER")


class WD14TaggerLoaderV3(_ComfyNodeBase):
    """Expose Load WD14 Tagger through Comfy's v3 extension API."""

    @classmethod
    def define_schema(cls) -> Any:
        """Declare the Load WD14 Tagger v3 schema."""

        choices = ModelChoiceService().wd14_tagger_choices()
        return _comfy_io.Schema(
            node_id="SimpleSyrup.WD14TaggerLoader",
            display_name="Load WD14 Tagger",
            category="SimpleSyrup/Tagging",
            description=(
                "Loads a WD14 tagger for compatible SimpleSyrup tagging workflows."
            ),
            search_aliases=["wd14", "tagger", "tags"],
            inputs=[
                _comfy_io.Combo.Input(
                    "wd14_model",
                    options=choices,
                    default=default_choice(choices, DEFAULT_WD14_TAGGER_MODEL),
                    tooltip=tooltips.WD14_MODEL_INPUT,
                ),
            ],
            outputs=[
                WD14TaggerIO.Output(
                    "wd14_tagger",
                    tooltip=tooltips.WD14_TAGGER_OUTPUT,
                ),
            ],
        )

    @classmethod
    def execute(cls, wd14_model: str) -> tuple[object]:
        """Run the legacy loader implementation behind the v3 schema."""

        return WD14TaggerLoader().load_model(wd14_model=wd14_model)
