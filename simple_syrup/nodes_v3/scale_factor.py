# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Comfy v3 node wrapper for Scale Factor."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, ClassVar

from ..nodes import tooltips
from ..nodes.scale_factor import (
    SCALE_FACTOR_DEFAULT,
    SCALE_FACTOR_MAX,
    SCALE_FACTOR_MIN,
    SCALE_FACTOR_STEP,
    ScaleFactor,
)

if TYPE_CHECKING:

    class _ComfyNodeBase:
        """Type-checking base for Comfy v3 nodes."""

        RETURN_TYPES: ClassVar[list[str]]
        RETURN_NAMES: ClassVar[list[str]]

else:
    _ComfyNodeBase = import_module("comfy_api.latest").io.ComfyNode

_comfy_io: Any = None if TYPE_CHECKING else import_module("comfy_api.latest").io


class ScaleFactorV3(_ComfyNodeBase):
    """Expose Scale Factor through Comfy's v3 extension API."""

    @classmethod
    def define_schema(cls) -> Any:
        """Declare the Scale Factor v3 schema."""

        return _comfy_io.Schema(
            node_id="SimpleSyrup.ScaleFactor",
            display_name="Scale Factor",
            category="SimpleSyrup/Primitives",
            description="Provides a bounded multiplier for scaling.",
            search_aliases=["scale", "scale factor", "float", "primitive"],
            inputs=[
                _comfy_io.Float.Input(
                    "value",
                    default=SCALE_FACTOR_DEFAULT,
                    min=SCALE_FACTOR_MIN,
                    max=SCALE_FACTOR_MAX,
                    step=SCALE_FACTOR_STEP,
                    tooltip=tooltips.SCALE_FACTOR_VALUE,
                ),
            ],
            outputs=[
                _comfy_io.Float.Output(
                    "scale_factor",
                    tooltip=tooltips.SCALE_FACTOR_OUTPUT,
                ),
            ],
        )

    @classmethod
    def execute(cls, value: float) -> tuple[float]:
        """Return the scale-factor value through the legacy implementation."""

        return ScaleFactor().get_value(value=value)
